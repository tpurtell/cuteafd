"""CPU structure, recovery and native-byte audits for GLM mixed EXL3 export."""
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5.core_k6 import export, recipe
from kimi_k3.uniform_k2.storage import read_header


def geometry(bits=6, inputs=128, outputs=128, group="base_core"):
    return dict(in_features=inputs, out_features=outputs,
        stored_out_features=(outputs + 127)//128*128, bits=bits, group=group)


def packed_fixture(path, module, g):
    tensors = {}
    for name, (dtype, shape) in recipe.packed_specs(module, g).items():
        tensors[name] = torch.ones(shape, dtype={"I16": torch.int16, "I32": torch.int32, "F16": torch.float16}[dtype])
    tensors[module + ".mcg"] = torch.tensor(recipe.MCG_MULTIPLIER, dtype=torch.uint32).view(torch.int32)
    tensors[module + ".svh"][g["out_features"]:] = 0
    save_file(tensors, path)
    return path


@pytest.fixture
def toy_source(tmp_path, monkeypatch):
    directory, output = tmp_path/"source", tmp_path/"export"
    directory.mkdir()
    selection = {
        "model.layers.0.self_attn.q_a_proj": geometry(),
        "model.layers.3.mlp.experts.0.gate_proj": geometry(4, group="routed"),
        "model.layers.78.eh_proj": geometry(4, inputs=256, group="mtp_core"),
        "model.layers.78.self_attn.kv_a_proj_with_mqa": geometry(4, outputs=576, group="mtp_core"),
        "lm_head": geometry(),
    }
    tensors = {name + ".weight": torch.ones(g["out_features"], g["in_features"], dtype=torch.bfloat16)
               for name, g in selection.items()}
    tensors.update({f"model.layers.{layer}.input_layernorm.weight": torch.arange(128, dtype=torch.bfloat16)
                    for layer in range(79)})
    tensors.update({
        "model.layers.3.mlp.gate.e_score_correction_bias": torch.tensor([.125, -.5, 1., 4.], dtype=torch.float32),
        "model.layers.3.self_attn.kv_b_proj.weight": torch.ones(128, 128, dtype=torch.bfloat16),
        "model.layers.3.self_attn.indexer.weights_proj.weight": torch.ones(32, 128, dtype=torch.bfloat16),
        "model.layers.78.enorm.weight": torch.ones(128, dtype=torch.bfloat16),
        "model.layers.78.hnorm.weight": torch.ones(128, dtype=torch.bfloat16),
        "model.layers.78.shared_head.norm.weight": torch.ones(128, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.ones(5, 128, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
    })
    # Deliberately distribute layer data across arbitrary source shards. The
    # output layout cannot infer source names from its own shard ordinals.
    buckets = [{} for _ in range(7)]
    for index, (name, tensor) in enumerate(sorted(tensors.items())):
        buckets[index % len(buckets)][name] = tensor
    weight_map = {}
    for index, values in enumerate(buckets):
        name = f"original-part-{index}.safetensors"
        save_file(values, directory/name)
        weight_map.update({key: name for key in values})
    source = SimpleNamespace(snapshot=directory, weight_map=weight_map,
        config=dict(architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa",
            num_hidden_layers=78, num_nextn_predict_layers=1, tie_word_embeddings=False))
    source_bytes = sum(x.numel()*x.element_size() for x in tensors.values())
    target_bytes = source_bytes - sum(g["out_features"]*g["in_features"]*2 for g in selection.values()) + sum(recipe.packed_bytes(g) for g in selection.values())
    monkeypatch.setattr(export, "reviewed_modules", lambda: selection)
    monkeypatch.setattr(export, "selected_modules", lambda _: selection)
    monkeypatch.setattr(export, "BASE_TENSOR_COUNT", len(tensors))
    monkeypatch.setattr(export, "BASE_PAYLOAD_BYTES", source_bytes)
    monkeypatch.setattr(export, "TARGET_TENSOR_COUNT", len(tensors)+3*len(selection))
    monkeypatch.setattr(export, "TARGET_PAYLOAD_BYTES", target_bytes)
    candidates = {name: packed_fixture(tmp_path/f"candidate-{index}.safetensors", name, g)
                  for index, (name, g) in enumerate(selection.items())}
    evidence = tmp_path/"evidence"
    evidence.mkdir()
    (evidence/"projection.json").write_text('{"metric_status":"ok"}')
    for name in ("LICENSE", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        (directory/name).write_text("original " + name)
    (directory/"README.md").write_text("source model card")
    return source, selection, candidates, output, evidence, target_bytes


def assemble_all(toy_source):
    source, selection, candidates, output, _, _ = toy_source
    for layer in range(79):
        prefix=f"model.layers.{layer}."
        replacements={name: candidates[name] for name in selection if name.startswith(prefix)}
        export.assemble_layer(source, layer, replacements, output, dict(run="toy"))


def test_layer_assembly_preserves_f32_and_native_bf16(toy_source):
    source, selection, candidates, output, _, _ = toy_source
    module="model.layers.3.mlp.experts.0.gate_proj"
    receipt=export.assemble_layer(source, 3, {module:candidates[module]}, output, "identity")
    tensors=load_file(output/receipt["filename"])
    bias="model.layers.3.mlp.gate.e_score_correction_bias"
    assert tensors[bias].dtype == torch.float32
    assert torch.equal(tensors[bias], torch.tensor([.125,-.5,1.,4.]))
    assert module+".weight" not in tensors
    assert tensors[module+".trellis"].shape == (8,8,64)
    assert receipt["preserved_audit"]["tensors"] == 4
    assert receipt["layer"] == 3
    # A committed shard is sufficient after packed candidates have retired.
    candidates[module].unlink()
    assert export.assemble_layer(source,3,{module:candidates[module]},output,"identity") == receipt


def test_complete_80_shard_inventory_and_mixed_metadata(toy_source, monkeypatch):
    source, selection, candidates, output, evidence, target_bytes = toy_source
    assemble_all(toy_source)
    seen=[]
    fsync_tree=export._fsync_tree
    def durable(path):
        fsync_tree(path)
        seen.append(path)
    monkeypatch.setattr(export,"_fsync_tree",durable)
    summary=export.finalize_export(source,selection,{"lm_head":candidates["lm_head"]},output,dict(run="toy"),evidence)
    assert summary["shards"] == 80
    assert summary["payload_bytes"] == target_bytes
    assert summary["quantized_projections"] == len(selection)
    assert summary["base_core_projections"] == 2
    assert summary["mtp_core_projections"] == 2
    assert summary["routed_projections"] == 1
    assert seen[-1] == output
    index=json.loads((output/"model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == target_bytes
    assert index["weight_map"]["model.layers.78.eh_proj.trellis"] == "model-00079-of-00080.safetensors"
    assert index["weight_map"]["model.embed_tokens.weight"] == "model-00080-of-00080.safetensors"
    assert index["weight_map"]["lm_head.svh"] == "model-00080-of-00080.safetensors"
    q=json.loads((output/"quantize_config.json").read_text())
    assert set(q["tensor_storage"]) == set(selection)
    assert q["tensor_storage"]["lm_head"]["bits_per_weight"] == 6
    assert q["tensor_storage"]["model.layers.78.eh_proj"]["bits_per_weight"] == 4
    assert q["tensor_storage"]["model.layers.78.self_attn.kv_a_proj_with_mqa"]["stored_out_features"] == 640
    assert q["codebook"] == "mcg" and q["head_bits"] == 6
    from gptqmodel.quantization.config import EXL3Config, QuantizeConfig
    normalized = QuantizeConfig.from_pretrained(str(output))
    assert isinstance(normalized, EXL3Config)
    assert normalized.tensor_storage["lm_head"]["bits_per_weight"] == 6
    assert normalized.tensor_storage["model.layers.78.eh_proj"]["bits_per_weight"] == 4
    config=json.loads((output/"config.json").read_text())
    assert {k:v for k,v in config.items() if k != "quantization_config"} == source.config
    assert config["quantization_config"]["bits"] == 4
    assert config["quantization_config"]["core_bits"] == 6
    assert not (output/"dense").exists()
    assert (output/"LICENSE").read_text() == "original LICENSE"
    assert (output/"chat_template.jinja").read_text() == "original chat_template.jinja"
    assert (output/"quantization-errors/projection.json").exists()
    assert "Full-model generation" in (output/"README.md").read_text()


@pytest.mark.parametrize("damage",["missing","extra","shape","mcg","missing_buffer","extra_buffer","nan_scale"])
def test_layer_rejects_unreviewed_replacement(toy_source,damage):
    source, _, candidates, output, _, _=toy_source
    name="model.layers.0.self_attn.q_a_proj"
    replacements={name:candidates[name]}
    if damage == "missing":replacements={}
    elif damage == "extra":replacements["other"]=candidates[name]
    else:
        tensors=load_file(candidates[name])
        if damage == "shape":tensors[name+".svh"]=torch.ones(64,dtype=torch.float16)
        elif damage == "mcg":tensors[name+".mcg"]=torch.tensor(0,dtype=torch.int32)
        elif damage == "missing_buffer":del tensors[name+".suh"]
        elif damage == "extra_buffer":tensors[name+".weight"]=torch.ones(128,128,dtype=torch.bfloat16)
        elif damage == "nan_scale":tensors[name+".suh"][0]=float("nan")
        save_file(tensors,candidates[name])
    with pytest.raises(ValueError):
        export.assemble_layer(source,0,replacements,output,"identity")


def test_padded_output_scales_must_be_zero(toy_source):
    _, selection, candidates, _, _, _=toy_source
    name="model.layers.78.self_attn.kv_a_proj_with_mqa"
    expected=export._check_packed(name,selection[name],candidates[name])
    assert expected[name+".trellis"] == ("I16",[8,40,64])
    tensors=load_file(candidates[name]);tensors[name+".svh"][-1]=1
    save_file(tensors,candidates[name])
    with pytest.raises(ValueError,match="padded output"):
        export._check_packed(name,selection[name],candidates[name])


def test_independent_native_byte_audit_rejects_corruption(toy_source,monkeypatch):
    source, _, candidates, output, _, _=toy_source
    name="model.layers.0.self_attn.q_a_proj"
    assemble=export.assemble_shard
    def corrupt(entries,path):
        receipt=assemble(entries,path)
        header,base=read_header(path)
        with path.open("r+b") as f:
            f.seek(base+header["model.layers.0.input_layernorm.weight"]["data_offsets"][0])
            f.write(b"\xff\xff")
        return receipt
    monkeypatch.setattr(export,"assemble_shard",corrupt)
    with pytest.raises(ValueError,match="preserved tensor bytes"):
        export.assemble_layer(source,0,{name:candidates[name]},output,"identity")
    assert not (output/"layer-000-receipt.json").exists()


def test_orphan_shard_regenerated_and_resume_hash_verified(toy_source):
    source, _, candidates, output, _, _=toy_source
    output.mkdir();path=output/"model-00001-of-00080.safetensors"
    path.write_bytes(b"partial")
    name="model.layers.0.self_attn.q_a_proj"
    receipt=export.assemble_layer(source,0,{name:candidates[name]},output,"identity")
    with path.open("r+b") as f:f.seek(-1,2);f.write(b"\xff")
    os.utime(path,ns=(path.stat().st_atime_ns,receipt["file_identity"]["mtime_ns"]))
    with pytest.raises(ValueError,match="checksum"):
        export.assemble_layer(source,0,{name:candidates[name]},output,"identity")


@pytest.mark.parametrize("damage",["orphan","missing","tamper","duplicate","receipt","native_shape"])
def test_finalizer_rejects_incomplete_or_changed_inventory(toy_source,damage):
    source, selection, candidates, output, evidence, _=toy_source
    assemble_all(toy_source)
    path=output/"model-00001-of-00080.safetensors"
    if damage == "orphan":(output/"orphan.safetensors").write_bytes(b"orphan")
    elif damage == "missing":path.unlink()
    elif damage == "receipt":
        rp=output/"layer-000-receipt.json";v=json.loads(rp.read_text());del v["preserved_audit"];rp.write_text(json.dumps(v))
    elif damage == "native_shape":
        values=load_file(path);values["model.layers.0.input_layernorm.weight"]=torch.ones(64,dtype=torch.bfloat16);save_file(values,path)
    elif damage == "duplicate":
        values=load_file(path);values["model.layers.3.mlp.experts.0.gate_proj.trellis"]=torch.ones(8,8,64,dtype=torch.int16);save_file(values,path)
    elif damage == "tamper":
        rp=json.loads((output/"layer-000-receipt.json").read_text())
        with path.open("r+b") as f:f.seek(-1,2);f.write(b"\xff")
        os.utime(path,ns=(path.stat().st_atime_ns,rp["file_identity"]["mtime_ns"]))
    with pytest.raises(ValueError):
        export.finalize_export(source,selection,{"lm_head":candidates["lm_head"]},output,dict(run="toy"),evidence)


def test_durability_flushes_files_before_directories_and_refuses_symlink(tmp_path,monkeypatch):
    target=tmp_path/"target";nested=target/"quantization-errors/layer-078"
    nested.mkdir(parents=True);(nested/"mtp.json").write_text('{}');(target/"README.md").write_text("card")
    seen=[];fsync=os.fsync
    def note(fd):seen.append(Path(os.readlink(f"/proc/self/fd/{fd}")));fsync(fd)
    monkeypatch.setattr(os,"fsync",note)
    export._fsync_tree(target)
    assert seen.index(nested/"mtp.json") < seen.index(nested) < seen.index(nested.parent) < seen.index(target)
    assert seen[-1] == target
    (target/"link").symlink_to(nested/"mtp.json")
    with pytest.raises(ValueError,match="symlink"):export._fsync_tree(target)


def test_compact_module_patterns_cover_exact_full_selection():
    patterns=[re.compile(p) for p in export._module_patterns()]
    assert len(patterns) == 5
    assert all(any(p.fullmatch(name) for p in patterns) for name in recipe.reviewed_modules())
    for kept in ("model.layers.3.self_attn.indexer.wq_b","model.layers.3.self_attn.kv_b_proj",
                 "model.layers.3.mlp.gate","model.embed_tokens","model.layers.78.enorm"):
        assert not any(p.fullmatch(kept) for p in patterns)
