import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_export import packed_specs
from ernie55_flash import export, hca_format
from ernie55_flash.source import FLASHINDEXER_FILE, HeaderSource
import mc_package as mc


def fixture(path):
    path.mkdir()
    fp = "model.layers.7.self_attn.core_attention.indexer.flashindexer_bias"
    bare = "model.layers.0.self_attn.linear_o_group_proj"
    draft = "model.mtp.feature_projection.weight"
    expert = "model.mtp.0.block_sparse_moe.experts.0.w1.weight"
    roles = {fp: "flashindexer", bare: "core", draft: "core", expert: "routed_expert", "lm_head.weight": "native"}
    names = {n: FLASHINDEXER_FILE if n == fp else "model.safetensors" for n in roles}
    tensors = {n: torch.full((128, 128), i, dtype=torch.bfloat16) for i, n in enumerate(roles)}
    tensors[bare] = torch.arange(8, dtype=torch.bfloat16).repeat_interleave(128)[:, None].expand(-1, 128).contiguous()
    (path / "flashindexer").mkdir()
    for shard in set(names.values()):
        save_file({n: t for n, t in tensors.items() if names[n] == shard}, path / shard)
    for asset in export.ASSETS:
        (path / asset).write_text("fixture")
    src = SimpleNamespace(snapshot=path, roles=roles, specs={n: ("BF16", list(t.shape)) for n, t in tensors.items()},
                          weight_map=names, locations={n: dict(shard=s, source_name=n) for n, s in names.items()},
                          sizes={n: t.nbytes for n, t in tensors.items()}, document=dict(model_type="ernie5_5_moe", o_groups=8,
                              o_lora_rank=128, num_attention_heads=8, head_dim=128))
    src.payload_bytes = sum(src.sizes.values())
    src.tensor_count = len(roles)
    src.index = {"metadata": {"total_size": src.payload_bytes - src.sizes[fp]}}
    src.as_inventory = lambda: HeaderSource.as_inventory(src)
    src.source_entries = lambda names: HeaderSource.source_entries(src, names)
    return src


def encode_fixture(path, selection, bits):
    tensors = {}
    for module, geometry in selection.items():
        for name, (dtype, shape) in packed_specs(module, dict(geometry, bits=bits)).items():
            value = mc.MUL1 - 2**32 if dtype == "I32" else 0
            if ".groups." in module and dtype != "I32":
                group = int(module.rsplit(".", 1)[1])
                value = bits * (group + 1)
            tensors[name] = torch.full(shape, value, dtype=getattr(torch, mc.DTYPES[dtype]))
    save_file(tensors, path)
    return {n: (path, n) for n in tensors}


def test_direct_package_with_bare_parameter_and_auxiliary_source(tmp_path):
    src = fixture(tmp_path / "source")
    out = tmp_path / "mc"
    identity = dict(schema="ernie55-causal-mc-v1", inputs=dict(source_revision=export.BASE_REVISION,
        recipe=dict(codebook="mul1", codebook_multiplier=mc.MUL1), corpus=dict(dataset=dict(repo_id="test/panel"),
        components={key: {"tokens": 42} for key in ("agentic", "exl3_standard")})))
    with pytest.raises(ValueError, match="frozen ERNIE"):
        export.DirectMCWriter(out, src, {"status": "candidate", "inputs": identity["inputs"]})
    writer = export.DirectMCWriter(out, src, identity)
    native, core, experts = export.projection_inventory(src)
    core_names = [n for n, role in src.roles.items() if role == "core"]
    packed = {8: encode_fixture(tmp_path/"eight", core, 8), 6: encode_fixture(tmp_path/"six", core, 6),
              4: encode_fixture(tmp_path/"experts", experts, 4)}
    with pytest.raises(ValueError, match="incomplete"):
        writer.write_block("mtp", 0, k4={}, k8={}, k6={})
    for family, layer, prefix in (("layers", 0, "model.layers.0."), ("mtp", 0, "model.mtp.0."),
                                  ("outer", None, "model.mtp.feature_projection.")):
        batches = {f"k{bits}": {n: entry for n, entry in entries.items() if n.startswith(prefix)}
                   for bits, entries in packed.items()}
        writer.write_block(family, layer, **batches)
    errors = tmp_path / "errors.csv"
    errors.write_text("module,bits\n")
    receipt = writer.finalize(card_path=tmp_path / "card.md", error_csv=errors)
    assert receipt["status"] == "validated-headers-and-byte-exactness"
    assert set(receipt["original_core_tensor_sha256"]) == set(native) | set(core_names)
    assert mc.validate(out, verification="full")["status"] == receipt["status"]
    flash = next(k for k in src.roles if src.roles[k] == "flashindexer")
    for view in export.VIEWS:
        index, config_file, qfile, bits = mc.mc_manifest.FILES[view]
        mapping = json.loads((out / index).read_text())["weight_map"]
        assert mapping[flash].startswith("native-flashindexer-")
        assert json.loads((out / config_file).read_text())["quantization_config"]["safetensors_index"] == index
        q = json.loads((out / qfile).read_text())
        assert all(any(re.fullmatch(p, m) for p in q["module_include"]) for m in q["tensor_storage"])
        assert set(q["tensor_storage"]) == (experts.keys() if bits == 16 else experts.keys() | core.keys())
        assert q["meta"]["upstream_activations"] == "k8-compact"
        assert set(core_names).issubset(mapping) == (bits == 16)
        parent = "model.layers.0.self_attn.linear_o_group_proj"
        if bits == 16:
            assert not any(name.startswith(parent + ".groups.") for name in mapping)
        else:
            expected = load_file(str(tmp_path / ("eight" if bits == 8 else "six")))
            for i, bundle in enumerate(hca_format.read_group_bundles(out, mapping, parent, src.document, bits)):
                assert bundle["suh"][0] == bits * (i + 1)
                for suffix, tensor in bundle.items():
                    assert torch.equal(tensor, expected[f"{parent}.groups.{i}.{suffix}"])
            missing = dict(mapping)
            del missing[parent + ".groups.7.suh"]
            with pytest.raises(ValueError, match="incomplete"):
                list(hca_format.read_group_bundles(out, missing, parent, src.document, bits))
            with pytest.raises(ValueError, match="incomplete"):
                list(hca_format.read_group_bundles(out, mapping | {parent: "native.safetensors"},
                                                   parent, src.document, bits))
    assert "Apache License 2.0" in (out / "README.md").read_text()
    assert (out / "encoding_ernie.py").read_bytes() == (src.snapshot / "encoding_ernie.py").read_bytes()


@pytest.mark.parametrize("bad", ["../flashindexer.safetensors", "/tmp/flashindexer.safetensors",
                                 "flashindexer/../model.safetensors", "flashindexer/unreviewed.safetensors"])
def test_reject_unreviewed_source_alias(bad):
    with pytest.raises(ValueError):
        mc.check_source_shard("ernie55_flash", bad)
    with pytest.raises(ValueError):
        mc.check_source_shard("hy4", FLASHINDEXER_FILE)
