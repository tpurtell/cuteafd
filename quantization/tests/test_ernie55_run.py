from types import SimpleNamespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash import run
from ernie55_flash.storage import Stages


def test_routed_frontier_is_committed_before_its_inputs_retire(tmp_path, monkeypatch):
    job = object.__new__(run.Production)
    job.root = tmp_path
    job.identity = "fixture"
    job.stages = Stages(tmp_path/"stages", job.identity)
    job.stages.write_batches("previous", [(0, {"state":torch.tensor([1.])})])
    job.stages.write_batches("fitted", [(0, {"packed":torch.tensor([2.])})])
    job.count, job.counts, job.tokens = 1, {"calibration":1,"heldout":0}, {"calibration":1,"heldout":0}
    job.model = SimpleNamespace(config=SimpleNamespace(dspark_target_layer_ids=[]),
                                checkpoint=SimpleNamespace(close=lambda: None))
    published = []
    def write(family, index, **bundles):
        assert job.stages.load("frontier-layers-001")
        assert job.stages.load("previous") and job.stages.load("fitted")
        published.append((family,index))
    job.writer = SimpleNamespace(write_block=write, group_complete=lambda label: label=="layers-001" and bool(published))
    result = dict(k4={}, k8={}, k6={}, metrics=[], stages=["fitted"],
                  outputs=lambda start: iter([(0, {"state":torch.tensor([3.])})]))
    monkeypatch.setattr(run, "process_layer", lambda *args: result)
    assert job.block("layers",1,"previous") == "frontier-layers-001"
    assert job.stages.load("previous") is None
    assert job.stages.load("fitted") is None
    monkeypatch.setattr(run,"process_layer",lambda *args: (_ for _ in ()).throw(AssertionError("duplicate encode")))
    assert job.block("layers",1,"previous") == "frontier-layers-001"
    assert published == [("layers",1)]
