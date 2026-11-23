"""The numerical reader accesses final native bytes locally and expert weights at source."""
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc.native_source import prepare_native_source
from inkling2.export import DirectMCWriter, load_source_inventory


class Writer(DirectMCWriter):
    family = "glm5_flash"
    prove_shared_source = True


def test_native_weights_are_the_proved_final_payload_and_resume_uses_them(tmp_path):
    snapshot, output = tmp_path / "snapshot", tmp_path / "output"
    snapshot.mkdir()
    core, expert, norm = "proj.weight", "expert.weight", "norm.weight"
    native = {core: torch.tensor([0., -0.]).bfloat16(), norm: torch.tensor([-1., 4.])}
    weights = {expert: torch.tensor([3., 1.]).bfloat16()}
    mapping = {k: "native.safetensors" for k in native} | {expert: "expert.safetensors"}
    size = 16
    save_file(native, snapshot / mapping[core])
    save_file(weights, snapshot / mapping[expert])
    (snapshot / "config.json").write_text(json.dumps(dict(model_type="glm5_next",
        text_config=dict(qk_rope_head_dim=0, hc_mult=4))))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(dict(
        metadata=dict(total_size=size), weight_map=mapping)))
    inventory = load_source_inventory(snapshot, reviewed_payload_bytes=size, reviewed_tensor_count=3)
    writer = Writer(output, snapshot, "org/model", "revision", {}, inventory)
    def prepare():
        return prepare_native_source(snapshot, writer, core_modules={"proj"}, expert_modules={"expert"})
    source = prepare()
    group = writer.state["groups"]["source-native"]
    assert len(group["shards"]) == 2 and set(source.local_files) == native.keys()
    for row in group["shards"]:
        for name, proof in row["receipt"]["original_tensor_proofs"].items():
            assert proof["source_path"] == str(snapshot / mapping[name])
    source.close()
    # No source-native reads should be required on a recovered producer.
    (snapshot / mapping[core]).rename(snapshot / "retired-native")
    source = prepare()
    for name, reference in (native | weights).items():
        actual = source.tensor(name)
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
        actual.zero_()
        assert torch.equal(source.tensor(name).view(torch.uint8), reference.view(torch.uint8))
    source.close()
