"""Inkling arithmetic and durable layer-state behavior, without serving imports."""
import copy
import json
import math
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from families.inkling2 import golden, model


def tiny_config():
    return {"model_type": "inkling2", "text_config": {
        "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 4,
        "vocab_size": 40, "unpadded_vocab_size": 34,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 16,
        "d_rel": 4, "rel_extent": 4, "sliding_window_size": 4, "local_layer_ids": [0, 2],
        "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 2,
        "n_routed_experts": 3, "num_experts_per_tok": 2,
        "n_shared_experts": 0, "attn_res_normalize_values": True,
        "use_gate_bias": True, "use_value_norm_no_gamma": True,
        "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 3., "rel_l2_clip_threshold": 2.,
        "mlp_gate_softcap": 4., "mlp_up_softcap": 25., "route_scale": 2.,
        "final_logit_softcapping": 30., "log_scaling_n_floor": 4, "log_scaling_alpha": .1}}


def make_snapshot(folder):
    folder.mkdir()
    config = tiny_config()
    cfg = model.Config.from_dict(config)
    gen = torch.Generator().manual_seed(17)
    tensors, groups = {}, []
    contract = model.tensor_contract(cfg)
    for index in range(cfg.num_hidden_layers):
        contract.update(model.tensor_contract(cfg, index))
    for name, (shape, dtype) in contract.items():
        if dtype == "U8":
            tensors[name] = torch.full(shape, 122, dtype=torch.uint8)
            continue
        target = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}[dtype]
        value = torch.randn(shape, generator=gen) * (.2 if dtype != "F8_E4M3" else 1.)
        if name.endswith("norm.weight") or "_clip.scale" in name:
            value += 1.
        tensors[name] = value.to(target)
        group = {"tensors": [name], "compute_dtype": "float32" if dtype == "F32" else "bfloat16"}
        if dtype == "F8_E4M3":
            group.update(compute_dtype="mxfp8", weight_block_shape=[32, 32], weight_transform="mxfp8_nearest",
                required_activation_quantization={"block_size": 32, "format": "mxfp8_e4m3",
                                                  "rounding": "nearest", "scale_dtype": "ue8m0"})
        groups.append(group)
    safetensors.save_file(tensors, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}}))
    (folder / "hf_quant_config.json").write_text(json.dumps({
        "quant_method": "inkling2", "model": {"parameter_groups": groups}}))
    return cfg, tensors


def hand_attention(q, k, v, relative, gate, sinks, window):
    result = torch.empty_like(q, dtype=torch.float64)
    for row in range(len(q)):
        for head in range(q.shape[1]):
            kvhead = head // (q.shape[1] // k.shape[1])
            chosen = list(range(0 if window is None else max(0, row - window + 1), row + 1))
            scores = [sum(float(a) * float(b) for a, b in zip(q[row, head], k[key, kvhead]))
                      / math.sqrt(q.shape[-1])
                      + (float(relative[row, head, row - key]) if row - key < relative.shape[-1] else 0.)
                      for key in chosen]
            normalizer = math.exp(float(sinks[row, head])) + sum(math.exp(s) for s in scores)
            for channel in range(q.shape[-1]):
                result[row, head, channel] = (sum(math.exp(s) * float(v[key, kvhead, channel])
                    for key, s in zip(chosen, scores)) / normalizer
                    / (1. + math.exp(-float(gate[row, head, channel]))))
    return result.float()


@pytest.mark.parametrize("window", [None, 4])
def test_relative_gqa_attention_sink_gate_and_extent_match_scalar(window):
    torch.manual_seed(27)
    q, k, v = torch.randn(9, 4, 2), torch.randn(9, 2, 2), torch.randn(9, 2, 2)
    relative, gate, sinks = torch.randn(9, 4, 3), torch.randn_like(q), torch.randn(9, 4)
    expected = hand_attention(q, k, v, relative, gate, sinks, window)
    actual = model.relative_attention(q, k, v, relative, gate, sinks, window=window)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    changed = v.clone()
    changed[5:] += 1000.
    torch.testing.assert_close(model.relative_attention(q, k, changed, relative, gate, sinks, window=window)[:5],
                               actual[:5], atol=0, rtol=0)


def test_local_1024_edge_excludes_distance_1024_global_keeps_it_without_bias():
    length = 1026
    q, k, v = torch.zeros(length, 1, 1), torch.zeros(length, 1, 1), torch.zeros(length, 1, 1)
    v[0] = 1.
    relative, gate, sinks = torch.zeros(length, 1, 1024), torch.zeros_like(q), torch.zeros(length, 1)
    relative[-1, 0, -1] = 2.
    local = model.relative_attention(q, k, v, relative, gate, sinks, window=1024)
    full = model.relative_attention(q, k, v, relative, gate, sinks)
    assert float(local[1023]) == pytest.approx(.5 / 1025)
    assert float(local[1024]) == 0.
    assert float(full[1024]) == pytest.approx(.5 / 1026)
    assert float(full[1025]) == pytest.approx(.5 / (1027 + math.exp(2.) - 1.))


@pytest.mark.parametrize("length", [129, 259])
def test_attention_prefix_bytes_survive_partial_tile_and_gqa(length):
    torch.manual_seed(18)
    q, k, v = torch.randn(length + 37, 4, 2), torch.randn(length + 37, 2, 2), torch.randn(length + 37, 2, 2)
    rel, gate, sinks = torch.randn(length + 37, 4, 7), torch.randn_like(q), torch.randn(length + 37, 4)
    args = (q, k, v, rel, gate, sinks)
    for window in (None, 127):
        short = model.relative_attention(*(x[:length] for x in args), window=window)
        long = model.relative_attention(*args, window=window)
        assert torch.equal(short, long[:length])


def test_sink_is_unscaled_denominator_only_and_gate_is_elementwise():
    q, k = torch.zeros(1, 1, 4), torch.zeros(1, 1, 4)
    v = torch.tensor([[[2., 4., 6., 8.]]])
    gate = torch.tensor([[[0., math.log(3.), 0., -math.log(3.)]]])
    result = model.relative_attention(q, k, v, torch.zeros(1, 1, 1), gate, torch.tensor([[math.log(3.)]]))
    torch.testing.assert_close(result, torch.tensor([[[.25, .75, .75, .5]]]))


def test_attention_rejects_incompatible_gqa():
    with pytest.raises(ValueError, match="geometry"):
        model.relative_attention(torch.zeros(2, 3, 4), torch.zeros(2, 2, 4), torch.zeros(2, 2, 4),
                                 torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), torch.zeros(2, 3))


def hand_conv(x, weight, dynamic=None):
    width = weight.shape[0] if dynamic is not None else weight.shape[-1]
    result = torch.empty_like(x)
    for row in range(len(x)):
        for channel in range(x.shape[-1]):
            value = float(x[row, channel])
            for lag in range(min(width, row + 1)):
                tap = (float(weight[lag, channel]) + float(dynamic[row, channel // (x.shape[1] // dynamic.shape[1]), lag])
                       if dynamic is not None else float(weight.reshape(x.shape[1], width)[channel, width - 1 - lag]))
                value += float(x[row - lag, channel]) * tap
            result[row, channel] = value / math.sqrt(1 + width * (2 if dynamic is not None else 1))
    return result


@pytest.mark.parametrize("dynamic", [False, True])
def test_sconv_zero_padding_identity_tap_order_and_dynamic_head_broadcast(dynamic):
    x = torch.arange(1., 25.).reshape(6, 4)
    w = torch.arange(1., 17.).reshape((4, 4) if dynamic else (4, 1, 4)) / 10.
    d = torch.arange(48.).reshape(6, 2, 4) / 20. if dynamic else None
    torch.testing.assert_close(model.short_conv(x, w, d), hand_conv(x, w, d))
    changed = x.clone()
    changed[4:] += 100.
    assert torch.equal(model.short_conv(changed, w, d)[:4], model.short_conv(x, w, d)[:4])


def hand_mix(blocks, partial, query, eps):
    values = torch.stack([*blocks, partial]).double()
    values = values / (values.square().mean(-1, keepdim=True) + eps).sqrt()
    result = []
    for q in query.double().reshape(-1, partial.shape[-1]):
        scores = (values * q / math.sqrt(partial.shape[-1])).sum(-1)
        probs = scores.softmax(0)
        result.append((values * probs[..., None]).sum(0).float())
    return tuple(result)


def test_attn_res_normalizes_both_keys_and_values_and_keeps_three_queries():
    blocks = [torch.tensor([[2., 0., 0., 0.], [1., -2., 0., 1.]])]
    partial = torch.tensor([[0., 3., 0., 0.], [4., 1., 3., -1.]])
    query = torch.tensor([[1., -1., 0., 0.], [-2., 2., 0., 0.], [0., 0., 0., 0.]])
    expected = hand_mix(blocks, partial, query, 1e-6)
    for actual, want in zip(model.attn_res(blocks, partial, query, 1e-6), expected):
        torch.testing.assert_close(actual, want)
    add = torch.ones_like(partial)
    torch.testing.assert_close(model.attn_res(blocks, partial, query[0], 1e-6, add)[0],
                               hand_mix(blocks, partial + add, query[0], 1e-6)[0])


def test_mxfp8_native_scales_and_activation_rounding():
    w = torch.tensor([[1., -2., .5, 3.] * 16, [2., -1., 4., -.5] * 16]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128], [126, 125]], dtype=torch.uint8)
    expected = w.float() * torch.tensor([[1.] * 32 + [2.] * 32, [.5] * 32 + [.25] * 32])
    assert torch.equal(model.dequant_mxfp8(w, scale), expected)
    x = torch.tensor([[449., 2.125, 2.375, 2.625] + [0.] * 28])
    expected = torch.tensor([[448., 2., 2.5, 2.5] + [0.] * 28])
    assert torch.equal(model.quantize_mxfp8(x), expected)
    assert torch.equal(model.quantize_mxfp8(torch.zeros(3, 32)), torch.zeros(3, 32))
    with pytest.raises(ValueError, match="layout"):
        model.dequant_mxfp8(w, torch.full_like(scale, 255))


def test_swiglu_interleaves_gate_up_and_uses_uncapped_sigmoid_before_gamma():
    packed = torch.tensor([[20., 100., -20., -100.]])
    gamma = torch.tensor([.3])
    expected = torch.tensor([[4. * math.tanh(g / 4.) / (1. + math.exp(-g))
                             * 25. * math.tanh(u / 25.) * .3 for g, u in ((20., 100.), (-20., -100.))]])
    torch.testing.assert_close(model.capped_swiglu(packed, gamma, 4., 25.), expected)


def test_router_bias_selects_but_does_not_change_combine_and_ties_use_lower_ids():
    logits, bias = torch.tensor([[0., 1., 2., -1.]]), torch.tensor([2., 0., 0., 3.])
    ids, weights = model.route(logits, bias, 2, 8.)
    assert ids.tolist() == [[3, 0]]
    raw = torch.sigmoid(logits[0, ids[0]])
    torch.testing.assert_close(weights[0], raw / raw.sum() * 8.)
    assert model.route(torch.zeros(1, 4), torch.zeros(4), 2, 1.)[0].tolist() == [[0, 1]]


def test_loader_accepts_native_mxfp8_root_layout_and_preserves_storage(tmp_path):
    cfg, tensors = make_snapshot(tmp_path / model.REVISION)
    weights = model.Weights(tmp_path / model.REVISION)
    weights.validate(cfg)
    assert torch.equal(weights.tensor(model.TEXT + "embed.weight"), tensors[model.TEXT + "embed.weight"])
    layer = weights.load_layer(1, cfg)
    assert layer["mlp.experts.w13_weight"].dtype == torch.float8_e4m3fn
    assert layer["mlp.experts.w13_weight_scale"].dtype == torch.uint8
    assert layer["q_input_norm.weight"].dtype == torch.float32
    assert not weights.maps
    assert layer["attn_res.attn_queries.0"].shape == (3, cfg.hidden_size)


@pytest.mark.parametrize("damage", ["rounding", "w13_shape", "dense_precision", "unknown_tensor"])
def test_loader_rejects_changed_native_policy_or_layout(tmp_path, damage):
    cfg, _ = make_snapshot(tmp_path / "snapshot")
    weights = model.Weights(tmp_path / "snapshot")
    quant_path = tmp_path / "snapshot/hf_quant_config.json"
    quant = json.loads(quant_path.read_text())
    if damage == "rounding":
        for group in quant["model"]["parameter_groups"]:
            if group["compute_dtype"] == "mxfp8":
                group["required_activation_quantization"]["rounding"] = "floor"
    elif damage == "w13_shape":
        name = model.TEXT + "layers.0.mlp.experts.w13_weight"
        weights.descriptor(name)["shape"][-1] += 32
    elif damage == "dense_precision":
        quant["model"]["parameter_groups"][0]["compute_dtype"] = "float16"
    else:
        weights.index[model.TEXT + "unexpected"] = "model.safetensors"
    quant_path.write_text(json.dumps(quant))
    with pytest.raises(ValueError):
        weights.validate(cfg)


def test_head_drops_padding_then_applies_f32_final_softcap():
    cfg = model.Config.from_dict(tiny_config())
    hidden = torch.ones(2, 32)
    head = torch.ones(40, 32).bfloat16()
    result = model.head_forward(hidden, head, cfg)
    assert result.shape == (2, 34) and result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full((2, 34), 30. * math.tanh(32. / 30.)))


def test_layer_boundary_saves_old_partial_and_mlp_mix_uses_attention_not_sum(monkeypatch):
    cfg = model.Config.from_dict(tiny_config())
    partial, block = torch.full((3, 32), 2.), torch.full((3, 32), 3.)
    attention, mlp = torch.full_like(partial, 4.), torch.full_like(partial, 5.)
    observed = []
    def mix(blocks, value, query, eps, addend=None):
        observed.append(([b.clone() for b in blocks], value.clone(), None if addend is None else addend.clone()))
        return (value,) if query.ndim == 1 else (value,) * 3
    monkeypatch.setattr(model, "attn_res", mix)
    monkeypatch.setattr(model, "attention_forward", lambda *args: attention)
    monkeypatch.setattr(model, "moe_forward", lambda *args: mlp)
    monkeypatch.setattr(model, "short_conv", lambda x, *args: x)
    layer = {"attn_sconv.weight": None, "mlp_sconv.weight": None, "mlp_norm.weight": torch.ones(32)}
    for index in (1, 2):
        layer[f"attn_res.attn_queries.{index - 1}"] = torch.zeros(3, 32) if index == 1 else torch.zeros(32)
        layer[f"attn_res.mlp_queries.{index}"] = torch.zeros(32)
        result = model.layer_forward(partial, [block], layer, cfg, index)
        blocks, value, add = observed[-1]
        assert len(blocks) == (2 if index == 2 else 1)
        assert torch.equal(value, attention if index == 2 else partial)
        assert (add is None) if index == 2 else torch.equal(add, attention)
        assert torch.equal(result, attention + mlp if index == 2 else partial + attention + mlp)


def hand_layer(partial, blocks, layer, cfg, index):
    def norm(x, name=None):
        y = x.double() / (x.double().square().mean(-1, keepdim=True) + cfg.rms_norm_eps).sqrt()
        return y.float() if name is None else (y * layer[name + ".weight"].double()).float()
    def project(x, name):
        return (x.double() @ layer["attn." + name + ".weight"].double().T).float()
    def clip(x, threshold, scale=None):
        x = x.double() if scale is None else x.double() * scale.double()
        return (x * (threshold / (x.norm(dim=-1, keepdim=True) + cfg.rms_norm_eps)).clamp(max=1.)).float()
    local = index in cfg.local_layer_ids
    h, kv, d = cfg.geometry(index)
    inputs = hand_mix(blocks, partial, layer[f"attn_res.attn_queries.{index - 1}"], cfg.rms_norm_eps) if blocks else (partial,)
    if local:
        xq = xk = xv = norm(inputs[0], "attn_norm")
    else:
        xq, xk, xv = [norm(x, n) for x, n in zip(inputs, ("q_input_norm", "k_input_norm", "v_input_norm"))]
    q = clip(project(xq, "wq_du").reshape(-1, h, d), cfg.qk_l2_clip_threshold, layer["attn.q_l2_clip.scale"])
    r = clip(project(xq, "wr_du").reshape(-1, h, cfg.d_rel), cfg.rel_l2_clip_threshold, layer["attn.rel_l2_clip.scale"])
    k_and_v = []
    for x, name, proj in ((xk, "k", "wk_dv"), (xv, "v", "wv_dv")):
        stem = "attn." + name + "_sconv."
        dynamic = None if local else (x.double() @ layer[stem + "weight_proj"].double().T).float().reshape(-1, kv, 4)
        k_and_v.append(hand_conv(project(x, proj), layer[stem + ("weight" if local else "static_w")], dynamic).reshape(-1, kv, d))
    k = clip(k_and_v[0], cfg.qk_l2_clip_threshold)
    v = norm(k_and_v[1])
    if not local:
        tau = 1. + cfg.log_scaling_alpha * torch.log(((torch.arange(len(q)) + 1).float() / cfg.log_scaling_n_floor).clamp_min(1.))
        q, r = q * tau[:, None, None], r * tau[:, None, None]
    relative = (r.double() @ layer["attn.rel_logits_proj.proj"].double()).float()
    attn = hand_attention(q, k, v, relative, project(xq, "gate_attn").reshape(-1, h, d),
                          project(xq, "attn_sink_proj"), cfg.sliding_window_size if local else None)
    attn = (attn.reshape(len(q), -1).double() @ layer["attn.wo_ud.weight"].double().T).float()
    attn = hand_conv(attn, layer["attn_sconv.weight"])
    first = index % cfg.n_layers_per_attn_res_block == 0
    inputs = hand_mix([*blocks, partial] if first else blocks, attn if first else partial + attn,
                      layer[f"attn_res.mlp_queries.{index}"], cfg.rms_norm_eps)[0]
    x = norm(inputs, "mlp_norm")
    logits = (x.double() @ layer["mlp.gate.weight"].double().T).float()
    scores = logits.sigmoid() + layer["mlp.gate.bias"]
    result = torch.zeros_like(x)
    def qdq(value):
        values = value.reshape(-1, 32)
        scale = 2. ** torch.ceil(torch.log2(values.abs().amax(-1, keepdim=True).clamp_min(2. ** -120) / 448.))
        return ((values / scale).clamp(-448., 448.).to(torch.float8_e4m3fn).float() * scale).reshape(value.shape)
    def weight(name, expert):
        return (layer[name][expert].float() * torch.exp2(layer[name + "_scale"][expert].float() - 127).repeat_interleave(32, -1)).double()
    for row in range(len(x)):
        ids = sorted(range(cfg.n_routed_experts), key=lambda e: (-float(scores[row, e]), e))[:cfg.num_experts_per_tok]
        gamma = logits[row, ids].double().sigmoid()
        gamma = gamma / gamma.sum() * cfg.route_scale
        for choice, expert in enumerate(ids):
            gu = (qdq(x[row]).double() @ weight("mlp.experts.w13_weight", expert).T).float()
            gate, up = gu[0::2], gu[1::2]
            activation = (4. * torch.tanh(gate / 4.) * gate.sigmoid() * 25. * torch.tanh(up / 25.) * gamma[choice]).float()
            result[row] += (qdq(activation).double() @ weight("mlp.experts.w2_weight", expert).T).float()
    return ((attn if first else partial + attn) + hand_conv(result, layer["mlp_sconv.weight"]))


def test_tiny_native_decoder_matches_hand_layer_for_local_global_and_block_boundaries(tmp_path):
    cfg, tensors = make_snapshot(tmp_path / "snapshot")
    weights = model.Weights(tmp_path / "snapshot")
    hidden = model.rms_norm(tensors[model.TEXT + "embed.weight"][:7], tensors[model.TEXT + "embed_norm.weight"], cfg.rms_norm_eps)
    blocks, manual_blocks, manual = [], [], hidden.clone()
    for index in range(cfg.num_hidden_layers):
        layer = weights.load_layer(index, cfg)
        expected = hand_layer(manual, manual_blocks, layer, cfg, index)
        actual = model.layer_forward(hidden, blocks, layer, cfg, index)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        if index % cfg.n_layers_per_attn_res_block == 0:
            blocks.append(hidden)
            manual_blocks.append(manual)
        hidden, manual = actual, expected


def test_tiny_full_decoder_prefix_is_byte_exact_across_expert_row_populations(tmp_path):
    cfg, tensors = make_snapshot(tmp_path / "snapshot")
    weights = model.Weights(tmp_path / "snapshot")
    ids = torch.arange(166) % cfg.unpadded_vocab_size
    embedding = tensors[model.TEXT + "embed.weight"].float()
    norm = tensors[model.TEXT + "embed_norm.weight"].float()
    short = model.rms_norm(embedding[ids[:129]], norm, cfg.rms_norm_eps)
    extended = model.rms_norm(embedding[ids], norm, cfg.rms_norm_eps)
    left, right = [], []
    for index in range(cfg.num_hidden_layers):
        layer = weights.load_layer(index, cfg)
        a = model.layer_forward(short, left, layer, cfg, index)
        b = model.layer_forward(extended, right, layer, cfg, index)
        assert torch.equal(a, b[:129]), f"layer {index} changed prefix bytes"
        if index % cfg.n_layers_per_attn_res_block == 0:
            left.append(short)
            right.append(extended)
        short, extended = a, b


def test_golden_runs_real_layer_loop_and_resumes_with_aligned_scoring(tmp_path, monkeypatch):
    import argparse
    import numpy as np
    from fidelity_windows import canonical, set_hash
    cfg, _ = make_snapshot(tmp_path / "snapshot")
    windows = [{"id": "tiny", "tokens": list(range(7)), "roles": ["ctx"] * 3 + ["gen"] * 4,
                "score_from": 2, "bucket": "0-2K"}]
    panel = {"schema": "cuteafd.fidelity.set/1", "family": "inkling2", "checkpoint": model.CHECKPOINT,
             "windows": windows, "quick_windows": ["tiny"]}
    panel["set_sha256"] = set_hash(panel)
    path = tmp_path / "windows.json"
    path.write_bytes(canonical(panel))
    monkeypatch.setattr(golden, "source_identity", lambda _: {"test_only": True})
    monkeypatch.setattr(golden, "verify_snapshot", lambda *args: {"snapshot_revision": "test"})
    args = argparse.Namespace(snapshot=tmp_path / "snapshot", windows=path, out=tmp_path / "golden",
                              device="cpu", check_only=False, stop_after_layer=None, _prefix_probe=True)
    golden.execute(args)
    meta = json.loads((args.out / "meta.json").read_bytes())
    assert meta["scored_tokens"] == 5 and meta["roles"]["gen"]["scored_rows"] == 4
    assert meta["roles"]["ctx"]["scored_rows"] == 1
    from importlib.metadata import version
    assert meta["tokenizers"] == version("tokenizers")
    assert meta["transformers"] == "not used; independent text forward"
    entry = meta["windows"][0]
    assert entry["positions"] == [2, 3, 4, 5, 6] and entry["vocab"] == 34
    logits = np.fromfile(args.out / entry["path"] / "logits.bin", dtype="<f4").reshape(5, 34)
    weights = model.Weights(args.snapshot)
    embed = weights.tensor(model.TEXT + "embed.weight").float()
    hidden = model.rms_norm(embed[:7], weights.tensor(model.TEXT + "embed_norm.weight"), cfg.rms_norm_eps)
    blocks = []
    for index in range(cfg.num_hidden_layers):
        result = model.layer_forward(hidden, blocks, weights.load_layer(index, cfg), cfg, index)
        if index % cfg.n_layers_per_attn_res_block == 0:
            blocks.append(hidden)
        hidden = result
    hidden = model.output_norm(hidden, blocks, weights.tensor(model.TEXT + "attn_res.output_query"),
                               weights.tensor(model.TEXT + "norm.weight"), cfg)
    want = model.head_forward(hidden[1:6], weights.tensor(model.TEXT + "unembed.weight"), cfg).numpy()
    assert np.array_equal(logits, want)
    monkeypatch.setattr(golden, "layer_forward", lambda *args: pytest.fail("completed layer replayed"))
    golden.execute(args)
    assert np.array_equal(logits, np.fromfile(args.out / entry["path"] / "logits.bin", dtype="<f4").reshape(5, 34))


def state_binding():
    return {"windows": {"a": "input-hash"}, "state_bytes_per_layer": 128}


def test_state_replays_interrupted_layer_and_keeps_exact_hardlinked_history(tmp_path):
    root, binding = tmp_path / "state", state_binding()
    state = golden.LayerState(root, binding)
    hidden = torch.arange(96.).reshape(3, 32)
    pending = state.begin(-1)
    state.save(pending, "a", hidden)
    state.commit(pending, .1)
    pending = state.begin(0)
    state.save(pending, "a", hidden + 1., append_block=True)
    resumed = golden.LayerState(root, binding)
    value, blocks = resumed.load("a", "cpu")
    assert torch.equal(value, hidden) and not blocks
    pending = resumed.begin(0)
    resumed.save(pending, "a", hidden + 2., append_block=True)
    resumed.commit(pending, .2)
    value, blocks = golden.LayerState(root, binding).load("a", "cpu")
    assert torch.equal(value, hidden + 2.) and len(blocks) == 1 and torch.equal(blocks[0], hidden)
    assert not (root / "layer-01").exists()
    assert (root / "blocks/block000/a.bin").stat().st_nlink == 1
    with pytest.raises(ValueError, match="differs"):
        golden.LayerState(root, {**binding, "torch": "different"})
    (root / "blocks/block000/a.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="blob differs"):
        golden.LayerState(root, binding)


def test_state_rejects_incomplete_manifest_nonfinite_and_nonconsecutive_layers(tmp_path):
    state = golden.LayerState(tmp_path, state_binding())
    with pytest.raises(ValueError, match="consecutive"):
        state.begin(0)
    pending = state.begin(-1)
    with pytest.raises(ValueError, match="incomplete"):
        state.commit(pending, .1)
    with pytest.raises(ValueError, match="finite"):
        state.save(pending, "a", torch.full((1, 32), float("nan")))


@pytest.mark.parametrize("field,value", [("q_bias", True), ("n_shared_experts", 1),
    ("hidden_size", 31), ("local_layer_ids", [0, 0]), ("route_scale", float("nan"))])
def test_config_rejects_unsupported_geometry_and_semantics(field, value):
    config = copy.deepcopy(tiny_config())
    config["text_config"][field] = value
    with pytest.raises(ValueError):
        model.Config.from_dict(config)


@pytest.fixture
def official_inkling_snapshot():
    import os
    value = os.environ.get("INKLING2_REFERENCE_SNAPSHOT")
    if not value:
        pytest.skip("set INKLING2_REFERENCE_SNAPSHOT for pinned release checks")
    snapshot = Path(value)
    golden.source_identity(snapshot)
    return snapshot


@pytest.fixture
def inkling_chat(official_inkling_snapshot):
    import hashlib
    pytest.importorskip("transformers")
    from checkpoint_tokenizer import load_tokenizer
    snapshot = official_inkling_snapshot
    assert hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest() == (
        "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6")
    assert hashlib.sha256((snapshot / "chat_template.jinja").read_bytes()).hexdigest() == (
        "74eb94918c5410bfab8d0af83ce1204fa8af08531f742c18ff4758b92b602d37")
    tokenizer = load_tokenizer(snapshot, chat=True)
    assert tokenizer.get_chat_template() == (snapshot / "chat_template.jinja").read_text()
    return tokenizer


def inkling_request():
    return {"messages": [{"role": "system", "content": "Inspect the repository."},
                         {"role": "user", "content": "Read README.md, then explain it."}],
            "reasoning_effort": "high",
            "tools": [{"type": "function", "function": {"name": "read_file",
                "description": "Read a file", "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]}


def test_official_inkling_tokenizer_keeps_serialized_bpe_and_effort(inkling_chat, official_inkling_snapshot):
    from tokenizers import Tokenizer
    from fidelity_generation import ChatRenderer, token_ids
    native = Tokenizer.from_file(str(official_inkling_snapshot / "tokenizer.json"))
    text = '# A plan\n\n  Keep spaces, tabs\tand punctuation.\n{"z":2,"a":1}\n' * 30
    assert token_ids(inkling_chat, text) == native.encode(text, add_special_tokens=False).ids
    assert inkling_chat.decode(token_ids(inkling_chat, text), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False) == text
    renderer, request = ChatRenderer(inkling_chat), inkling_request()
    assert renderer.template_options(request) == {"reasoning_effort": "high"}
    assert "Thinking effort level: 0.9<|end_message|>" in renderer.render(request)
    default = copy.deepcopy(request)
    default.pop("reasoning_effort")
    assert renderer.prompt(default) == renderer.prompt(request)
    request["reasoning_effort"] = "low"
    assert "Thinking effort level: 0.2<|end_message|>" in renderer.render(request)


@pytest.mark.parametrize("kind", ["plain", "reasoning", "tool"])
def test_official_inkling_native_completion_is_exact_prefix(inkling_chat, kind):
    from fidelity_generation import ChatRenderer, token_ids
    request, renderer = inkling_request(), ChatRenderer(inkling_chat)
    assistant = {"role": "assistant", "content": "A short answer.\n"}
    if kind != "plain":
        assistant["reasoning_content"] = "First inspect the repository.\n"
    if kind == "tool":
        assistant.update(content=None, tool_calls=[{"id": "call0", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"README.md","end":5}'}}])
    before = copy.deepcopy(assistant)
    generated = renderer.completion(request, assistant)
    prompt = renderer.prompt(request)
    full = renderer.render(request, assistant)
    assert prompt + generated == token_ids(inkling_chat, full)
    assert inkling_chat.decode(prompt + generated, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False) == full
    span = inkling_chat.decode(generated, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    assert span.endswith("<|content_model_end_sampling|>")
    if kind != "plain":
        assert "<|content_thinking|>First inspect the repository.\n<|end_message|>" in span
    if kind == "tool":
        assert '<|message_model|>read_file<|content_invoke_tool_json|>' in span
        assert '{"name":"read_file","args":{"end":5,"path":"README.md"}}' in span
    assert assistant == before


def test_official_inkling_tool_reply_and_next_turn_preserve_prior_tokens(inkling_chat):
    from fidelity_generation import ChatRenderer, token_ids
    renderer, request = ChatRenderer(inkling_chat), inkling_request()
    assistant = {"role": "assistant", "content": None, "reasoning_content": "Read the file.",
        "tool_calls": [{"id": "call0", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path":"README.md"}'}}]}
    prior = renderer.prompt(request) + renderer.completion(request, assistant)
    messages = request["messages"] + [assistant,
        {"role": "tool", "tool_call_id": "call0", "content": "# Project\n"},
        {"role": "user", "content": "Explain that title."}]
    ids = renderer.conversation_ids(request, messages)
    assert ids[:len(prior)] == prior
    continued = {**request, "messages": messages}
    text = renderer.render(continued)
    assert ids == renderer.prompt(continued) == token_ids(inkling_chat, text)
    assert "<|message_tool|>read_file<|content_text|># Project\n<|end_message|>" in text
    assert text.count("Thinking effort level:") == 1


def pinned_torch_helpers(root):
    """Compile unchanged helper bodies only after validating every source file."""
    import ast
    import hashlib
    from families.inkling2.sources import VLLM
    selections = {
        "tests/models/inkling2/test_attention.py": ("_ref_attention",),
        "tests/models/inkling2/test_core_ops.py": ("_clip", "_dense_conv"),
        "tests/models/inkling2/test_moe.py": ("dequant_mxfp8", "_torch_select_experts"),
    }
    namespace = {"torch": torch, "REL_EXTENT": 1024}
    sources = {name: (root / name).read_bytes() for name in selections}
    for name, raw in sources.items():
        assert hashlib.sha256(raw).hexdigest() == VLLM["files"][name]
    for name, functions in selections.items():
        path = root / name
        tree = ast.parse(sources[name], filename=str(path))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in functions]
        assert {node.name for node in nodes} == set(functions)
        assert all(not node.decorator_list for node in nodes)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_official_inkling_first_window_matches_pinned_torch_helpers(official_inkling_snapshot, monkeypatch):
    import os
    import torch.nn.functional as F
    from fidelity_windows import load_set
    helpers_root = os.environ.get("INKLING2_UPSTREAM_VLLM")
    windows_path = os.environ.get("INKLING2_REFERENCE_WINDOWS")
    if not helpers_root or not windows_path:
        pytest.skip("set INKLING2_UPSTREAM_VLLM and INKLING2_REFERENCE_WINDOWS for real intermediates")
    helpers = pinned_torch_helpers(Path(helpers_root))
    cfg = model.Config.from_dict(json.loads((official_inkling_snapshot / "config.json").read_bytes()))
    manifest = load_set(Path(windows_path), "inkling2")
    assert manifest["checkpoint"] == model.CHECKPOINT
    window = next(w for w in manifest["windows"] if len(w["tokens"]) >= 640)
    assert not window.get("media") and len(window["tokens"]) <= 4096
    device = torch.device(os.environ.get("INKLING2_REFERENCE_DEVICE", "cuda:0"))
    golden.configure_f32()
    weights = model.Weights(official_inkling_snapshot, device)
    weights.validate(cfg)
    layer_index, measurements, decoded = -1, {}, set()

    def compare(name, actual, expected, *, exact=False):
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0 if exact else 2e-5,
                                   rtol=0 if exact else 2e-5)
        key = f"layer{layer_index}/{name}"
        entry = measurements.setdefault(key, {"calls": 0, "max_abs": 0.})
        entry["calls"] += 1
        entry["max_abs"] = max(entry["max_abs"], (actual - expected).abs().max().item())

    originals = {name: getattr(model, name) for name in
                 ("short_conv", "l2_clip", "relative_attention", "route", "dequant_mxfp8")}

    def conv(x, weight, dynamic=None):
        actual = originals["short_conv"](x, weight, dynamic)
        if layer_index in (0, 5):
            shape = (x.shape[-1], -1) if dynamic is None else weight.shape
            expected = helpers["_dense_conv"](x, weight.reshape(shape), dynamic)
            compare("static_conv" if dynamic is None else "dynamic_conv", actual, expected, exact=True)
        return actual

    def clip(x, threshold, eps, scale=None):
        actual = originals["l2_clip"](x, threshold, eps, scale)
        if layer_index in (0, 5):
            value = x.float() if scale is None else x.float() * scale.float()
            compare("l2_clip", actual, helpers["_clip"](value, threshold, eps), exact=True)
        return actual

    def attention(q, k, v, relative, gate, sinks, *, window=None):
        actual = originals["relative_attention"](q, k, v, relative, gate, sinks, window=window)
        if layer_index in (0, 5):
            block = 64
            blocks = (len(q) + block - 1) // block
            caches = [F.pad(value, (0, 0, 0, 0, 0, blocks * block - len(q))).reshape(
                blocks, block, *value.shape[1:]) for value in (k, v)]
            table = torch.arange(blocks, device=q.device).reshape(1, -1)
            attended, lse = helpers["_ref_attention"](q, *caches, relative, [(len(q), len(q))],
                table, q.shape[-1] ** -.5, None if window is None else window - 1)
            expected = attended * (lse - torch.logaddexp(lse, sinks)).exp()[..., None] * gate.sigmoid()
            compare("relative_attention_sink_gate", actual, expected)
        return actual

    def route(logits, bias, topk, scale):
        ids, gamma = originals["route"](logits, bias, topk, scale)
        if layer_index in (0, 5):
            expected_gamma, expected_ids = helpers["_torch_select_experts"](logits, bias, topk, scale)
            same = (ids == expected_ids).all(-1)
            scores = logits.sigmoid() + bias
            # The upstream Torch helper leaves exact ties to torch.topk; the
            # native kernel and our synthetic oracle choose the lower expert ID.
            compare("route_choice_scores", scores.gather(-1, ids), scores.gather(-1, expected_ids), exact=True)
            assert same.any()
            compare("route_gamma", gamma[same], expected_gamma[same], exact=True)
            measurements[f"layer{layer_index}/route_tie_rows"] = int((~same).sum())
        return ids, gamma

    def dequant(weight, scale):
        actual = originals["dequant_mxfp8"](weight, scale)
        key = (layer_index, tuple(weight.shape))
        if layer_index in (0, 5) and key not in decoded:
            compare("dequant_" + str(weight.shape[-1]), actual,
                    helpers["dequant_mxfp8"](weight, scale), exact=True)
            decoded.add(key)
        return actual

    for name, function in (("short_conv", conv), ("l2_clip", clip),
            ("relative_attention", attention), ("route", route), ("dequant_mxfp8", dequant)):
        monkeypatch.setattr(model, name, function)
    with torch.inference_mode():
        embedding = weights.tensor(model.TEXT + "embed.weight").float()
        norm = weights.tensor(model.TEXT + "embed_norm.weight").float()
        tokens = torch.tensor(window["tokens"], device=device)
        hidden = model.rms_norm(F.embedding(tokens, embedding), norm, cfg.rms_norm_eps)
        del embedding, norm, tokens
        weights.close()
        golden.release(device)
        blocks = []
        for layer_index in range(6):
            if device.type == "cuda":
                from fidelity_windows import admit_reference_memory
                contract = model.tensor_contract(cfg, layer_index)
                native = sum(weights.descriptor(name)["data_offsets"][1]
                             - weights.descriptor(name)["data_offsets"][0] for name in contract)
                dense = sum(2 * math.prod(shape) for shape, dtype in contract.values() if dtype == "BF16")
                admit_reference_memory(torch.cuda, native + dense
                                       + golden.execution_memory(cfg, len(window["tokens"]), layer_index))
            layer = weights.load_layer(layer_index, cfg)
            result = model.layer_forward(hidden, blocks, layer, cfg, layer_index)
            if layer_index % cfg.n_layers_per_attn_res_block == 0:
                blocks.append(hidden)
            hidden = result
            assert torch.isfinite(hidden).all()
            del layer
            golden.release(device)
    for index in (0, 5):
        for name in ("static_conv", "l2_clip", "relative_attention_sink_gate", "route_gamma",
                     "dequant_" + str(cfg.hidden_size), "dequant_" + str(cfg.intermediate_size)):
            assert measurements[f"layer{index}/{name}"]["calls"] > 0
    assert measurements["layer5/dynamic_conv"]["calls"] == 2
    from families.inkling2.sources import VLLM
    print("pinned upstream first-window checks " + json.dumps({"window": window["id"],
        "tokens": len(window["tokens"]), "revision": model.REVISION,
        "panel_diagnostic_only": manifest.get("diagnostic_only", False),
        "upstream": {"revision": VLLM["revision"], "files": {
            name: VLLM["files"][name] for name in (
                "tests/models/inkling2/test_attention.py", "tests/models/inkling2/test_core_ops.py",
                "tests/models/inkling2/test_moe.py")}}, "measurements": measurements}, sort_keys=True))
