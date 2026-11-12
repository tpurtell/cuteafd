"""MiniMax reference behavior; ported from hughmadden/cuteafd@2cbf6873e, Hugh Madden."""
import importlib
import json
import math
from pathlib import Path
import sys

import pytest
torch = pytest.importorskip("torch")
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
model = importlib.import_module("families.minimax_m3.model")


def tiny_config():
    return {
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "vocab_size": 64, "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
            "use_gemma_norm": True, "attention_output_gate": False,
            "rope_theta": 10000000, "rotary_dim": 16, "partial_rotary_factor": .5,
            "hidden_act": "swigluoai", "use_qk_norm": True, "tie_word_embeddings": False,
            "dense_intermediate_size": 32, "shared_intermediate_size": 32,
            "num_local_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "scoring_func": "sigmoid", "use_routing_bias": True, "moe_layer_freq": [0, 1],
            "qk_norm_type": "per_head", "swiglu_alpha": 1.702, "swiglu_limit": 7.,
            "routed_scaling_factor": 2.,
            "sparse_attention_config": {
                "use_sparse_attention": True, "sparse_index_dim": 32,
                "sparse_num_index_heads": 1, "sparse_topk_blocks": 2,
                "sparse_block_size": 2, "sparse_score_type": "max",
                "sparse_init_block": 0, "sparse_local_block": 1,
                "sparse_attention_freq": [1, 1], "sparse_disable_index_value": [1, 1],
            },
        },
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "mxfp8", "activation_scheme": "dynamic",
            "weight_block_size": [1, 32], "group_size": 16,
            "moe_quant_algo": "NVFP4", "moe_quant_format": "nvfp4-pack-quantized",
        },
    }


@pytest.fixture
def snapshot(tmp_path):
    return make_snapshot(tmp_path / "snapshot")


def make_snapshot(folder, vocab=64):
    folder.mkdir()
    config = tiny_config()
    config["text_config"]["vocab_size"] = vocab
    (folder / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(71)
    tensors = {}

    def bf16(name, shape, scale=.1):
        tensors[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()

    def mxfp8(name, rows, cols):
        tensors[name + ".weight"] = torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
        tensors[name + ".weight_scale_inv"] = torch.full((rows, cols // 32), 123, dtype=torch.uint8)

    def nvfp4(name):
        tensors[name + ".weight_packed"] = torch.randint(0, 256, (32, 16), generator=generator, dtype=torch.uint8)
        tensors[name + ".weight_scale"] = torch.full((32, 2), .125).to(torch.float8_e4m3fn)
        tensors[name + ".weight_global_scale"] = torch.tensor(4.)
        tensors[name + ".input_global_scale"] = torch.tensor(32.)

    bf16("language_model.model.embed_tokens.weight", (vocab, 32), .5)
    bf16("language_model.model.norm.weight", (32,))
    bf16("language_model.lm_head.weight", (vocab, 32))
    for layer in range(2):
        prefix = f"language_model.model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            bf16(prefix + name + ".weight", (32,))
        for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            bf16(prefix + "self_attn." + name + ".weight", (32,))
        for name, rows, cols in (("q_proj", 64, 32), ("k_proj", 32, 32),
                                 ("v_proj", 32, 32), ("o_proj", 32, 64),
                                 ("index_q_proj", 32, 32), ("index_k_proj", 32, 32)):
            mxfp8(prefix + "self_attn." + name, rows, cols)
        if layer == 0:
            for name in ("gate_proj", "up_proj", "down_proj"):
                mxfp8(prefix + "mlp." + name, 32, 32)
        else:
            bf16(prefix + "block_sparse_moe.gate.weight", (4, 32))
            tensors[prefix + "block_sparse_moe.gate.weight"] = tensors[prefix + "block_sparse_moe.gate.weight"].float()
            tensors[prefix + "block_sparse_moe.e_score_correction_bias"] = torch.tensor([.2, -.1, .0, .1])
            for expert in range(4):
                for name in ("w1", "w2", "w3"):
                    nvfp4(prefix + f"block_sparse_moe.experts.{expert}." + name)
            for name in ("gate_proj", "up_proj", "down_proj"):
                mxfp8(prefix + "block_sparse_moe.shared_experts." + name, 32, 32)
    save_file(tensors, folder / "model.safetensors")
    (folder / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))
    return folder


def test_shared_sparse_indexer_vectors_match_independent_msa_reference():
    vectors = ROOT / "rust/crates/cuteafd-loader/src/families/minimax_m3/fixtures/msa-vectors.json"
    for case in json.loads(vectors.read_text())["cases"]:
        scores = torch.tensor(case["block_scores"], dtype=torch.float32).repeat_interleave(128, -1)
        for head, token, value in case["overrides"]:
            scores[head, token] = value
        selected = model.msa_select(scores[:, None, :], torch.tensor([case["position"]]))
        assert selected[:, 0].tolist() == case["selected"], case["name"]
        assert selected.dtype == torch.int32 and selected.shape == (4, 1, 16)


def test_mxfp8_dequantization_row_and_32_column_blocks():
    weights = torch.tensor([[1., -2., .5, 3.] * 16, [-1., 2., -.5, -3.] * 16]).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127, 128], [126, 125]], dtype=torch.uint8)
    expected = weights.float() * torch.tensor([[1.] * 32 + [2.] * 32, [.5] * 32 + [.25] * 32])
    assert torch.equal(model.dequant_mxfp8(weights, scales), expected)


def test_nvfp4_dequantization_nibble_order_signed_values_and_scales():
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8)
    scales = torch.tensor([[2., 4.]]).to(torch.float8_e4m3fn)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    expected = torch.cat((values, values * 2))[None]
    assert torch.equal(model.dequant_nvfp4(packed, scales, torch.tensor(2.)), expected)


def test_activation_quantizers_preserve_representable_values_and_zero():
    mx = torch.tensor([[0., .5, -1., 448.] * 8 + [0., 1., -2., 896.] * 8])
    assert torch.equal(model.quantize_mxfp8(mx), mx)
    fp4 = torch.tensor([[0., .5, -1., 1.5, -2., 3., -4., 6.] * 2])
    assert torch.equal(model.quantize_nvfp4(fp4, torch.tensor(2.)), fp4)
    zero = torch.zeros(2, 32)
    assert torch.equal(model.quantize_mxfp8(zero), zero)
    assert torch.equal(model.quantize_nvfp4(zero, torch.tensor(2.)), zero)


def test_nvfp4_activation_rounding_ties_to_even():
    values = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.] * 2])
    expected = torch.tensor([[0., 1., 1., 2., 2., 4., 4., 6.] * 2])
    assert torch.equal(model.quantize_nvfp4(values, torch.tensor(1.)), expected)


def test_router_rust_vector_bias_is_selection_only():
    ids, weights = model.route(torch.tensor([[0., 2., 1., -1.]]), torch.tensor([5., 0., 0., 0.]), 2, 2.)
    assert ids.tolist() == [[0, 1]]
    s2 = 1 / (1 + math.exp(-2))
    torch.testing.assert_close(weights, torch.tensor([[.5 / (.5 + s2) * 2, s2 / (.5 + s2) * 2]]))
    assert weights.dtype == torch.float32
    assert torch.allclose(weights.sum(-1), torch.tensor([2.]))


def test_router_rust_ties_prefer_lower_id():
    ids, weights = model.route(torch.zeros(1, 4), torch.zeros(4), 2, 2.)
    assert ids.tolist() == [[0, 1]]
    assert torch.equal(weights, torch.ones(1, 2))


def test_swiglu_oai_rust_vectors():
    gate = torch.tensor([0., 9., 7., 2., 2., -9., -7., 2.])
    up = torch.tensor([3., 0., 0., -20., -7., 0., 0., .5])
    out = model.swiglu_oai(gate, up)
    assert out[0] == 0 and out[1] == out[2] and out[3] == out[4]
    assert out[5] != out[6]
    assert out[7].item() == pytest.approx(2 / (1 + math.exp(-1.702 * 2)) * 1.5, abs=1e-6)


def test_gemma_norm_rust_vector_and_bf16_boundary():
    out = model.gemma_norm(torch.tensor([3., 4.]), torch.tensor([0., 1.]), 0.)
    torch.testing.assert_close(out, torch.tensor([3 / math.sqrt(12.5), 8 / math.sqrt(12.5)]))
    x = torch.tensor([[3., 4.]], dtype=torch.bfloat16)
    assert model.gemma_norm(x, torch.tensor([0., 1.]), 0.).dtype == torch.bfloat16


def test_partial_rope_rust_vector_neox_pairs_and_tail():
    x = torch.arange(128, dtype=torch.float32).reshape(1, 1, 128)
    assert torch.equal(model.rope_partial(x, torch.tensor([0]), 64, 1e7), x)
    out = model.rope_partial(x, torch.tensor([5]), 64, 1e7)
    assert torch.equal(out[..., 64:], x[..., 64:])
    assert out[0, 0, 0].item() == pytest.approx(-32 * math.sin(5), abs=1e-4)
    assert out[0, 0, 32].item() == pytest.approx(32 * math.cos(5), abs=1e-4)
    assert torch.equal(x.flatten(), torch.arange(128, dtype=torch.float32))


def test_msa_rust_short_context_vectors_and_padding():
    scores = torch.zeros(1, 2, 2048)
    selected = model.msa_select(scores, torch.tensor([0, 2047]))
    assert selected[0, 0].tolist() == [0] + [-1] * 15
    assert selected[0, 1].tolist() == list(range(16))


def test_msa_rust_long_context_vector_local_forced_and_future_masked():
    position = 128 * 40 + 5
    scores = torch.full((1, 1, position + 128), -1.)
    hot = [3, 7, 11, 13, 17, 19, 23, 29, 31, 33, 35, 36, 37, 38, 39]
    for block in hot:
        scores[0, 0, block * 128 + 77] = 10 + block
    scores[..., position + 1:] = 1e6
    selected = model.msa_select(scores, torch.tensor([position]))
    assert selected[0, 0].tolist() == hot + [40]
    assert torch.equal(selected, model.msa_select(scores[..., :position + 1], torch.tensor([position])))


def test_msa_selections_are_per_kv_head_and_ties_use_lower_id():
    scores = torch.zeros(2, 1, 8)
    scores[0, 0, 2] = 20
    scores[1, 0, 4] = 30
    selected = model.msa_select(scores, torch.tensor([7]), block=2, topk=2, local=1)
    assert selected[:, 0].tolist() == [[1, 3], [2, 3]]
    tied = model.msa_select(torch.zeros_like(scores), torch.tensor([7]), block=2, topk=2, local=1)
    assert tied[:, 0].tolist() == [[0, 3], [0, 3]]


def forward(snapshot, tokens, activation):
    cfg = model.Config.from_dict(json.loads((snapshot / "config.json").read_text()))
    weights = model.Weights(snapshot, device="cpu")
    try:
        weights.validate(cfg)
        hidden = weights.tensor("language_model.model.embed_tokens.weight")[tokens]
        states = [hidden.clone()]
        for index in range(2):
            hidden = model.layer_forward(hidden, weights.load_layer(index, cfg), cfg, index, activation=activation)
            states.append(hidden.clone())
        hidden = model.gemma_norm(hidden, weights.tensor("language_model.model.norm.weight"), 1e-6).bfloat16()
        head = weights.tensor("language_model.lm_head.weight")
        logits = torch.nn.functional.linear(hidden.float(), head.float()) if activation == "served-cache" else torch.nn.functional.linear(hidden, head)
        return states, logits
    finally:
        weights.close()


@pytest.mark.parametrize("activation", ["served", "served-cache", "bf16"])
def test_two_layer_cpu_forward_dense_sparse_moe_and_prefix(snapshot, activation):
    tokens = torch.tensor([1, 7, 3, 9, 11, 5, 2])
    states, logits = forward(snapshot, tokens, activation)
    dtype = torch.float32 if activation == "served-cache" else torch.bfloat16
    assert logits.shape == (7, 64) and logits.dtype == dtype
    assert torch.isfinite(logits).all()
    for before, after in zip(states, states[1:]):
        dtype = torch.float32 if activation == "served-cache" else torch.bfloat16
        assert after.dtype == dtype and torch.isfinite(after).all()
        assert not torch.equal(before, after)
    prefix_states, prefix_logits = forward(snapshot, tokens[:5], activation)
    for whole, prefix in zip(states, prefix_states):
        torch.testing.assert_close(whole[:5], prefix, rtol=0, atol=0)
    torch.testing.assert_close(logits[:5], prefix_logits, rtol=0, atol=0)


def test_served_and_diagnostic_activation_arms_differ(snapshot):
    tokens = torch.tensor([1, 7, 3, 9, 11, 5, 2])
    _, served = forward(snapshot, tokens, "served")
    _, diagnostic = forward(snapshot, tokens, "bf16")
    assert not torch.equal(served, diagnostic)


def panel(tokens):
    from fidelity_windows import set_hash
    result = dict(schema="cuteafd.fidelity.set/1", family="minimax_m3", checkpoint=model.CHECKPOINT,
                  windows=[dict(id="sample", tokens=tokens, roles=["ctx"] * 3 + ["gen"] * (len(tokens) - 3),
                                score_from=3, bucket="0-2K")], quick_windows=["sample"])
    result["set_sha256"] = set_hash(result)
    return result


def test_final_norm_tap_preserves_bf16_words_and_scored_input_offset(tmp_path):
    from families.minimax_m3.golden import write_final_norm
    import numpy as np
    window = panel([1, 7, 3, 9, 11, 5, 2])["windows"][0]
    values = torch.arange(4 * 32).reshape(4, 32).bfloat16() / 3
    write_final_norm(tmp_path, window, values)
    target = tmp_path / "windows/sample"
    np.testing.assert_array_equal(np.fromfile(target / "final_norm.bf16.bin", dtype="<u2"),
                                  values.view(torch.uint16).numpy().ravel())
    assert json.loads((target / "final_norm.json").read_text()) == {
        "dtype": "BF16", "hidden": 32, "first_input_row": 2, "rows": 4, "diagnostic_only": True}
    with pytest.raises(ValueError, match="dtype/rows"):
        write_final_norm(tmp_path, window, values.float())
    with pytest.raises(ValueError, match="dtype/rows"):
        write_final_norm(tmp_path, window, values[:3])


@pytest.mark.parametrize("activation", ["served", "served-cache", "bf16"])
def test_layer_resume_is_byte_exact_and_loads_only_remaining_layer(snapshot, tmp_path, monkeypatch, activation):
    from families.minimax_m3.golden import run_layers
    manifest = panel([1, 7, 3, 9, 11, 5, 2])
    complete, complete_times = run_layers(snapshot, manifest, tmp_path / "complete", activation=activation)
    traced, _ = run_layers(snapshot, manifest, tmp_path / "traced", activation=activation, layers=[0, 1])
    assert traced[0].path.read_bytes() == complete[0].path.read_bytes()
    trace = tmp_path / "traced/windows/sample"
    assert (trace / "layer00.input.bin").stat().st_size == 7 * 32 * 4
    assert (trace / "layer00.attention_projected.bin").stat().st_size == 7 * 32 * 4
    assert (trace / "layer01.router_ids.bin").stat().st_size == 7 * 2 * 4
    assert (trace / "layer01.residual.bin").read_bytes() == (trace / "layer01.bin").read_bytes()
    residuals, _ = run_layers(snapshot, manifest, tmp_path / "residuals", activation=activation,
                              layers=[0, 1], residual_only=True)
    assert residuals[0].path.read_bytes() == complete[0].path.read_bytes()
    residual_trace = tmp_path / "residuals/windows/sample"
    assert sorted(p.name for p in residual_trace.iterdir()) == ["layer00.bin", "layer01.bin"]
    assert (residual_trace / "layer01.bin").read_bytes() == (trace / "layer01.bin").read_bytes()
    stopped, times = run_layers(snapshot, manifest, tmp_path / "partial", activation=activation, stop_after_layer=0)
    assert [t["layer"] for t in times] == [0]
    assert not (tmp_path / "partial/meta.json").exists()
    assert not torch.equal(torch.from_numpy(stopped[0].load()), torch.from_numpy(complete[0].load()))
    loaded = []
    original = model.Weights.load_layer
    def tracked(self, index, cfg):
        loaded.append(index)
        return original(self, index, cfg)
    monkeypatch.setattr(model.Weights, "load_layer", tracked)
    resumed, times = run_layers(snapshot, manifest, tmp_path / "resumed", activation=activation,
                               resume=tmp_path / "partial/state")
    assert loaded == [1]
    assert [t["layer"] for t in times] == [0, 1]
    assert len(complete_times) == 2
    assert complete[0].path.read_bytes() == resumed[0].path.read_bytes()


def test_resume_rejects_arithmetic_token_and_blob_changes(snapshot, tmp_path):
    from families.minimax_m3.golden import run_layers
    manifest = panel([1, 7, 3, 9, 11, 5, 2])
    states, _ = run_layers(snapshot, manifest, tmp_path / "partial", stop_after_layer=0)
    for name, changed, arm in (("arm", manifest, "bf16"),
                               ("tokens", panel([2, 7, 3, 9, 11, 5, 2]), "served")):
        with pytest.raises(ValueError, match="identity differs"):
            run_layers(snapshot, changed, tmp_path / name, activation=arm, resume=tmp_path / "partial/state")
    payload = bytearray(states[0].path.read_bytes())
    payload[0] ^= 1
    states[0].path.write_bytes(payload)
    with pytest.raises(ValueError, match="bytes changed"):
        run_layers(snapshot, manifest, tmp_path / "corrupt", resume=tmp_path / "partial/state")


def test_interrupted_successor_cannot_advance_resume(snapshot, tmp_path, monkeypatch):
    from families.minimax_m3.golden import run_layers
    from fidelity_windows import LayerCheckpoints
    manifest = panel([1, 7, 3, 9, 11, 5, 2])
    complete, _ = run_layers(snapshot, manifest, tmp_path / "complete")
    original = LayerCheckpoints.commit
    def interrupt(self, layer_id, states, times):
        if layer_id == 1:
            raise RuntimeError("synthetic interruption before durable commit")
        return original(self, layer_id, states, times)
    monkeypatch.setattr(LayerCheckpoints, "commit", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_layers(snapshot, manifest, tmp_path / "interrupted")
    assert json.loads((tmp_path / "interrupted/state/latest.json").read_text())["layer"] == "layer00"
    monkeypatch.setattr(LayerCheckpoints, "commit", original)
    resumed, _ = run_layers(snapshot, manifest, tmp_path / "resumed", resume=tmp_path / "interrupted/state")
    assert complete[0].path.read_bytes() == resumed[0].path.read_bytes()


def test_checkpoint_inventory_and_formats_fail_closed(snapshot):
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot)
    weights.validate(cfg)
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    weights.descriptor(name)["shape"] = [32, 32]
    with pytest.raises(ValueError, match="shape/dtype"):
        weights.validate(cfg)
    del weights.index[name]
    with pytest.raises(ValueError, match="inventory"):
        weights.validate(cfg)


@pytest.mark.parametrize("bad", [0., -1., float("inf"), float("nan")])
def test_nvfp4_rejects_invalid_global_scales(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        model.quantize_nvfp4(torch.ones(1, 32), torch.tensor(bad))


def test_projection_gate_up_uses_shared_min_input_scale(snapshot):
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot).load_layer(1, cfg)
    stem = "block_sparse_moe.experts.0"
    weights[stem + ".w1.input_global_scale"] = torch.tensor(8.)
    weights[stem + ".w3.input_global_scale"] = torch.tensor(64.)
    hidden = torch.linspace(-4, 4, 64).reshape(2, 32).bfloat16()
    gate = model.project(hidden, weights, stem + ".w1", "served", torch.tensor(8.))
    up = model.project(hidden, weights, stem + ".w3", "served", torch.tensor(8.))
    assert torch.isfinite(model.swiglu_oai(gate, up)).all()
    weights[stem + ".w3.weight_global_scale"] = torch.tensor(8.)
    # Route every row through expert 0, so a mismatched fused scale cannot hide.
    weights["block_sparse_moe.e_score_correction_bias"][0] = 100
    with pytest.raises(ValueError, match="equal weight global scales"):
        model.feed_forward(hidden, weights, cfg, 1, "served")



def test_publication_metadata_retains_the_run_upstream_source_basis(snapshot, tmp_path):
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from families.minimax_m3 import golden
    from fidelity_windows import canonical, verify_snapshot, write_scored_logits

    Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0")).save(
        str(snapshot / "tokenizer.json"))
    manifest = panel([1, 7, 3, 9, 11, 5, 2])
    out = tmp_path / "golden"
    out.mkdir()
    (out / "windows.json").write_bytes(canonical(manifest))
    entry = write_scored_logits(out, manifest["windows"][0], np.zeros((4, 64), dtype=np.float32))
    identity = golden.source_identity()
    # Backfilling must preserve the run's provenance rather than today's code metadata.
    identity["transformers"]["reason"] = "recorded upstream basis for this run"
    meta = dict(activation="served", snapshot_identity=verify_snapshot(manifest, snapshot),
                windows=[entry], source_identity=identity, platform={"device": "cpu"})
    (out / "meta.json").write_bytes(canonical(meta))
    golden.write_reference_meta(out, snapshot)
    saved = json.loads((out / "meta.json").read_text())
    assert saved["root_arithmetic"]["transformers_source_basis"] == identity["transformers"]
    assert saved["source_pin"]["transformers_revision"] == identity["transformers"]["revision"]
    assert saved["transformers"] == "not used"
    assert saved["scored_tokens"] == 4
    assert saved["nll"] == pytest.approx(math.log(64))


def test_one_token_dense_forward_matches_hand_computed_native_qdq(snapshot):
    cfg = model.Config.from_dict(tiny_config())
    cfg.text["rms_norm_eps"] = 1e-6
    weights = model.Weights(snapshot).load_layer(0, cfg)
    for name, value in weights.items():
        if name.endswith("weight_scale_inv"):
            value.fill_(127)
        else:
            value.zero_()
    # One token, one nonzero head: softmax is exactly one, RoPE is identity.
    for stem in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.index_q_proj",
                 "self_attn.index_k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                 "mlp.up_proj", "mlp.down_proj"):
        weights[stem + ".weight"] = torch.eye(*weights[stem + ".weight"].shape).to(torch.float8_e4m3fn)
    hidden = torch.ones(1, 32, dtype=torch.bfloat16)
    # Norm(1)=1; attention=1; residual=2; Norm(2)=1.
    # SwiGLU(1,1)=2*sigmoid(1.702), BF16=1.6953125, MXFP8 QDQ=1.75.
    actual = model.layer_forward(hidden, weights, cfg, 0, "served")
    torch.testing.assert_close(actual, torch.full_like(hidden, 3.75), rtol=0, atol=0)


def test_isolated_nvfp4_scratch_contract_and_oai_plain_separation():
    import importlib.util
    path = ROOT / 'native/tests/minimax_nvfp4_expert_selftest.py'
    spec = importlib.util.spec_from_file_location('minimax_expert_oracle', path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    for rows, capacity in [(1, 1), (16, 16), (64, 80)]:
        offsets, sizes, extent = oracle.scratch_regions(128, 6144, 3072, rows, capacity)
        assert all(offset % 1024 == 0 for offset in offsets)
        assert all(b >= a + oracle.aligned(size) for a, b, size in zip(offsets, offsets[1:], sizes))
        assert extent == offsets[-1] + oracle.aligned(sizes[-1])
        assert sizes[5:7] == [0, 0]  # fused preactivations are not exported
    gate = torch.tensor([[-10., -1., 0., 1., 10.]], dtype=torch.bfloat16)
    up = torch.tensor([[10., -10., 0., 2., 10.]], dtype=torch.bfloat16)
    clipped_gate, clipped_up = gate.float().clamp(max=7), up.float().clamp(-7, 7)
    expected = (clipped_gate * torch.sigmoid(1.702 * clipped_gate) * (clipped_up + 1)).bfloat16()
    assert torch.equal(model.swiglu_oai(gate, up), expected)
    plain = (torch.nn.functional.silu(clipped_gate) * clipped_up).bfloat16()
    assert not torch.equal(plain, expected)
    assert model.swiglu_oai(gate, up)[0, 0] != model.swiglu_oai(gate.clamp(min=-7), up)[0, 0]
