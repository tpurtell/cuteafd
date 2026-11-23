from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc.layer import LayerPass, numerics, value_hash
from hy4_flash.storage import Stages


@torch.inference_mode()
def test_core_boundary_installs_serialized_k8_and_recovers_both_widths(tmp_path, monkeypatch):
    module = torch.nn.Linear(4, 4, bias=False)
    engine = LayerPass.__new__(LayerPass)
    engine.device = torch.device("cpu")
    engine.stages = Stages(tmp_path / "stages", {"pin":"test"})
    inputs = torch.arange(12, dtype=torch.float32).view(3, 4)
    called = []
    def encode(name, weight, raw, device):
        called.append(name)
        assert torch.equal(raw["H"], inputs.T @ inputs)
        def payload(bits):
            return dict(trellis=torch.zeros((2, 2), dtype=torch.int16),
                        suh=torch.tensor([bits], dtype=torch.float16),
                        svh=torch.tensor([1.],dtype=torch.float16), mul1=torch.tensor(-1,dtype=torch.int32))
        return SimpleNamespace(packed={8:payload(8),6:payload(6)},metrics={8:{"bits":8},6:{"bits":6}},
                               reconstruction=torch.full_like(weight,8))
    monkeypatch.setattr(numerics, "encode_core", encode)
    monkeypatch.setattr(numerics,"reconstruct",lambda packed,shape,device: torch.full(shape,packed["suh"].item()))
    engine.fit("compact", {"q":module}, lambda: [inputs])
    assert module.weight.eq(8).all()
    module.weight.zero_()
    engine.fit("compact", {"q":module}, lambda: pytest.fail("recaptured completed stage"))
    assert called == ["q"] and module.weight.eq(8).all()
    values = engine.stages.read_batch("compact",0)
    meta = engine.stages.load("compact")["metadata"]
    assert [row["bits"] for row in meta["metrics"]] == [8,6]
    assert meta["hashes"] == {key:value_hash(val) for key,val in values.items()}
