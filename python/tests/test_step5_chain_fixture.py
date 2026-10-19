"""Optional pinned, layer-major BF16-root hidden states for the native target chain."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "step5_chain_golden", ROOT / "python/reference/families/step5/golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)
SEED = 274
ROWS = (1, 7, 8, 9, 64)
FP8_REVISION = "364b3b03fed2477aa0953fe855a2c4c8c48ee3cc"


def decode_fp8(weight, scale):
    """Multiply exact E4M3 values by checkpoint F32 128x128 block scales."""
    import torch
    assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert scale.shape == ((weight.shape[0] + 127) // 128,
                           (weight.shape[1] + 127) // 128)
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * expanded[:weight.shape[0], :weight.shape[1]]


def same_quant_ffn(ref, layer, snapshot, captures, save, manifest, layer_id=3):
    """Bind official MoE control flow to checkpoint-decoded TP4/TP6 expert operands.

    The wiring arm reproduces the shared small-row BF16-input arithmetic. The
    exact-decode arm is diagnostic, never a fidelity reference or engine fit.
    """
    import torch
    from safetensors import safe_open
    from unittest.mock import patch
    from shape_invariant import linear

    assert snapshot.name == FP8_REVISION
    tp = int(os.environ.get("CUTEAFD_STEP5_CHAIN_TP", "4"))
    assert tp in (4, 6)
    width = 1536 // tp
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    readers = {}
    decoded = {}
    for projection in ("gate", "up", "down"):
        base = f"model.layers.{layer_id}.moe.{projection}_proj"
        values = []
        bf16 = 88 <= layer_id <= 90
        for suffix in (("weight",) if bf16 else ("weight", "weight_scale_inv")):
            name = base + "." + suffix
            shard = index[name]
            if shard not in readers:
                readers[shard] = safe_open(str(snapshot / shard), framework="pt", device="cpu")
            tensor = readers[shard].get_tensor(name)
            data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            manifest["quant_weights_sha256"][name] = hashlib.sha256(data).hexdigest()
            values.append(tensor)
        # Decode one expert at a time; no BF16 roundtrip in this source operand.
        if bf16:
            assert values[0].dtype == torch.bfloat16
            decoded[projection] = list(values[0].unbind())
        else:
            decoded[projection] = [decode_fp8(w, s) for w, s in zip(*values)]
    readers.clear()
    # This FFN fixture must bind the serving router/shared weights, not assume
    # that the BF16 root carries byte-identical coordinator operands.
    quant_reader = golden.Weights(snapshot)
    for name, parameter in layer.named_parameters():
        if name.startswith(("post_attention_layernorm.", "moe.gate.", "moe.router_bias", "share_expert.")):
            value = quant_reader.get(f"model.layers.{layer_id}." + name)
            assert value.dtype == parameter.dtype and value.shape == parameter.shape
            data = value.contiguous().view(torch.uint8).numpy().tobytes()
            manifest["quant_weights_sha256"][f"model.layers.{layer_id}." + name] = hashlib.sha256(data).hexdigest()
            parameter.copy_(value)
    quant_reader.files.clear()
    moe = layer.moe
    original_activation = ref.clamped_swiglu

    def gather(slots, routes):
        output = torch.zeros(slots.shape[0], slots.shape[-1], device=slots.device,
                             dtype=slots.dtype if slots.dtype == torch.float64 else torch.float32)
        for slot in range(moe.top_k):
            output = output + slots[:, slot].to(output.dtype) * routes[:, slot, None].to(output.dtype)
        return output.to(slots.dtype)

    for rows, residual in captures.items():
        residual = residual.cuda()
        hidden = layer.post_attention_layernorm(residual).to(torch.bfloat16)
        routes, ids = moe.route(hidden)
        logits = linear(hidden.float(), moe.gate.weight.float())
        scores = logits.sigmoid() + moe.router_bias
        ranked = scores.topk(9, dim=-1).values
        margin = (ranked[:, 7] - ranked[:, 8]) / ranked[:, 7].abs().clamp_min(torch.finfo(torch.float32).tiny)
        save(f"q_r{rows}_residual", residual)
        save(f"q_r{rows}_hidden", hidden)
        save(f"q_r{rows}_ids", ids.to(torch.int32))
        save(f"q_r{rows}_routes", routes)
        save(f"q_r{rows}_margin", margin)
        shared = layer.share_expert(hidden)
        save(f"q_r{rows}_shared", shared)
        for arm in ("wiring", "fp64", "exact_decode"):
            parts = []
            for rank in range(tp):
                first, end = rank * width, (rank + 1) * width

                def project(projection, expert, x):
                    weight = decoded[projection][expert]
                    weight = weight[:, first:end] if projection == "down" else weight[first:end]
                    if arm != "exact_decode":
                        weight = weight.to(torch.bfloat16)
                    dtype = torch.float64 if arm == "fp64" else torch.float32
                    result = linear(x.to(dtype), weight.to(device=x.device, dtype=dtype))
                    return result if arm == "fp64" else result.to(torch.bfloat16)

                def activate(gate, up, limit):
                    if arm != "fp64":
                        return original_activation(gate, up, limit)
                    return (torch.nn.functional.silu(gate).clamp(max=limit)
                            * up.clamp(-limit, limit))

                with patch.object(moe, "route", return_value=(routes, ids)), \
                        patch.object(moe.gate_proj, "apply_expert", side_effect=lambda e, x: project("gate", e, x)), \
                        patch.object(moe.up_proj, "apply_expert", side_effect=lambda e, x: project("up", e, x)), \
                        patch.object(moe.down_proj, "apply_expert", side_effect=lambda e, x: project("down", e, x)), \
                        patch.object(ref, "clamped_swiglu", side_effect=activate), \
                        patch.object(ref, "weighted_topk_gather", side_effect=gather):
                    parts.append(moe(hidden.double() if arm == "fp64" else hidden))
            routed = parts[0].to(torch.float64 if arm == "fp64" else torch.float32)
            for part in parts[1:]:
                routed = routed + part.to(routed.dtype)
            if arm != "fp64":
                routed = routed.to(torch.bfloat16)
                combined = routed.float() + shared.float()
                output = residual + combined
            else:
                # Shared branch uses the same admitted BF16 operands, unrounded FP64 math.
                gate = linear(hidden.double(), layer.share_expert.gate_proj.weight.double())
                up = linear(hidden.double(), layer.share_expert.up_proj.weight.double())
                activated = torch.nn.functional.silu(gate).clamp(max=layer.share_expert.limit)
                activated = activated * up.clamp(-layer.share_expert.limit, layer.share_expert.limit)
                shared64 = linear(activated, layer.share_expert.down_proj.weight.double())
                output = residual.double() + (routed + shared64)
            save(f"q_r{rows}_{arm}_output", output)
            if arm == "wiring":
                save(f"q_r{rows}_routed", routed)
                save(f"q_r{rows}_combined", combined)
    manifest["same_quant"] = {
        "checkpoint": "stepfun-ai/Step-5-Preview", "revision": FP8_REVISION,
        "layer": layer_id, "tp": tp, "slice": width, "ingress": "bf16", "maximum_rows": 64,
        "storage": "bf16" if bf16 else "fp8-block128",
        "wiring": "native BF16; F32 accumulate; BF16 per-rank output" if bf16 else
                  "BF16(E4M3 * F32 block128 scale); F32 accumulate; BF16 per-rank output",
        "fp64": "same BF16 operands; unrounded FP64 projections, activation and sums",
        "exact_decode": "report-only E4M3 * F32 scale; F32 accumulate; BF16 boundaries",
        "error_ratio_bound": 1.01, "router_relative_margin": 1e-3,
    }


def test_official_bf16_tail_fixtures(monkeypatch):
    destination = os.environ.get("CUTEAFD_STEP5_TAIL_FIXTURE_DIR")
    if not destination:
        pytest.skip("set CUTEAFD_STEP5_TAIL_FIXTURE_DIR for native BF16 routed tails")
    torch = pytest.importorskip("torch")
    from shape_invariant import install

    root = Path(destination).resolve()
    assert root.is_relative_to(Path(os.environ["CARGO_TARGET_DIR"]).resolve())
    snapshot = Path(os.environ["STEP5_QUANT_SNAPSHOT"])
    ref, _ = golden.load_release(Path(os.environ["STEP5_REFERENCE_SNAPSHOT"]))
    values = json.loads((snapshot / "config.json").read_text())
    config_values = dict(values.get("text_config", values))
    config_values["tp_size"] = 1
    config = ref.Step5Config(**config_values)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.nn.functional, "linear", torch.nn.functional.linear)
    monkeypatch.setattr(torch, "einsum", torch.einsum)
    install()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    residual = torch.randn((max(ROWS), config.hidden_size), generator=generator)
    captures = {rows: residual[:rows].clone() for rows in ROWS}
    reader = golden.Weights(snapshot)
    with torch.inference_mode():
        for layer_id in (88, 89, 90):
            target = root / f"layer{layer_id}"
            target.mkdir(parents=True, exist_ok=True)
            manifest = {"schema": 1, "seed": SEED, "source_pins": golden.SOURCE_PINS,
                        "rows": list(ROWS), "files": {}, "quant_weights_sha256": {}}

            def save(name, tensor):
                value = tensor.detach().contiguous().cpu()
                data = value.view(torch.uint8).numpy().tobytes()
                (target / (name + ".bin")).write_bytes(data)
                manifest["files"][name] = {
                    "dtype": str(value.dtype), "shape": list(value.shape),
                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                }

            with torch.device("meta"):
                layer = ref.Step5DecoderLayer(config, layer_id, dtype=torch.bfloat16)
            layer = layer.to_empty(device="cuda").eval()
            reader.load_layer(layer, layer_id)
            same_quant_ffn(ref, layer, snapshot, captures, save, manifest, layer_id)
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            del layer


def test_official_first_four_layer_fixtures(monkeypatch):
    destination = os.environ.get("CUTEAFD_STEP5_CHAIN_FIXTURE_DIR")
    if not destination:
        pytest.skip("set CUTEAFD_STEP5_CHAIN_FIXTURE_DIR for real-weight CUDA fixtures")
    torch = pytest.importorskip("torch")
    from shape_invariant import install

    root = Path(destination).resolve()
    assert root.is_relative_to(Path(os.environ["CARGO_TARGET_DIR"]).resolve())
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(os.environ["STEP5_REFERENCE_SNAPSHOT"])
    ref, kernel = golden.load_release(snapshot)
    values = json.loads((snapshot / "config.json").read_text())
    config_values = dict(values.get("text_config", values))
    config_values["tp_size"] = 1
    config = ref.Step5Config(**config_values)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    # Restore process-global reference geometry/precision after the optional test.
    monkeypatch.setattr(torch.nn.functional, "linear", torch.nn.functional.linear)
    monkeypatch.setattr(torch, "einsum", torch.einsum)
    install()
    reader = golden.Weights(snapshot)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    residual = torch.randn((max(ROWS), config.hidden_size), generator=generator)
    states = {rows: residual[:rows].clone() for rows in ROWS}
    manifest = {
        "schema": 1, "checkpoint": golden.CHECKPOINT, "revision": golden.REVISION,
        "source_pins": golden.SOURCE_PINS, "seed": SEED, "layers": [0, 1, 2, 3],
        "rows": list(ROWS), "files": {}, "weights_sha256": {}, "quant_weights_sha256": {},
        "arithmetic": "official golden.layer_forward; fixed-M128; BF16-root; no engine calibration",
        "routed_addition": "residual + (float(routed_bf16) + float(shared_bf16))",
        "torch": torch.__version__, "cuda": torch.version.cuda, "allow_tf32": False,
    }

    def save(name, tensor):
        value = tensor.detach().contiguous().cpu()
        data = value.view(torch.uint8).numpy().tobytes()
        (root / (name + ".bin")).write_bytes(data)
        manifest["files"][name] = {
            "dtype": str(value.dtype), "shape": list(value.shape),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        }

    with torch.inference_mode():
        for rows in ROWS:
            save(f"r{rows}_input", states[rows])
        rope = {}
        captures = {}
        for layer_id in range(4):
            with torch.device("meta"):
                layer = ref.Step5DecoderLayer(config, layer_id, dtype=torch.bfloat16)
            layer = layer.to_empty(device="cuda").eval()
            reader.load_layer(layer, layer_id)
            layer.self_attn.build_rope(max_position=128, device=torch.device("cuda"), shared_cache=rope)
            # Hash loaded operands, not all checkpoint shards; fixtures stay independently identifiable.
            for name, parameter in layer.named_parameters():
                data = parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                manifest["weights_sha256"][f"model.layers.{layer_id}.{name}"] = hashlib.sha256(data).hexdigest()
            for rows in ROWS:
                x = states[rows].cuda()
                if layer_id == 3:
                    captured = {}

                    def norm_hook(_module, args, result):
                        captured["hidden"] = result.to(torch.bfloat16)
                        captured["residual"] = args[0].detach().cpu()

                    hook = layer.post_attention_layernorm.register_forward_hook(norm_hook)
                    output = golden.layer_forward(layer, kernel, x, 128)
                    hook.remove()
                    hidden = captured["hidden"]
                    captures[rows] = captured["residual"]
                    routes, ids = layer.moe.route(hidden)
                    routed = layer.moe(hidden)
                    shared = layer.share_expert(hidden)
                    combined = routed.float() + shared.float()
                    assert torch.equal(combined, layer._feed_forward(hidden))
                    # This is exactly the three existing residual-export calls used by the engine.
                    widened = torch.zeros_like(combined) + routed.float()
                    assert torch.equal(widened + shared.float(), combined)
                    for name, tensor in {"ffn_hidden": hidden, "route_ids": ids.to(torch.int32),
                                         "route_weights": routes, "routed": routed, "shared": shared,
                                         "combined": combined}.items():
                        save(f"r{rows}_layer3_{name}", tensor)
                else:
                    output = golden.layer_forward(layer, kernel, x, 128)
                assert torch.isfinite(output).all()
                save(f"r{rows}_layer{layer_id}", output)
                states[rows] = output.cpu()
            if layer_id == 3:
                quant_snapshot = os.environ.get("STEP5_QUANT_SNAPSHOT")
                if quant_snapshot:
                    same_quant_ffn(ref, layer, Path(quant_snapshot), captures, save, manifest)
            del layer
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

def test_fp8_decode_retains_scale_bits_and_block_boundaries():
    torch = pytest.importorskip("torch")
    weight = torch.ones((129, 129), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.001, 2.003], [3.005, 4.007]], dtype=torch.float32)
    decoded = decode_fp8(weight, scale)
    assert decoded.dtype == torch.float32
    assert decoded[0, 0] == scale[0, 0]
    assert decoded[0, 128] == scale[0, 1]
    assert decoded[128, 0] == scale[1, 0]
    assert decoded[128, 128] == scale[1, 1]
    assert not torch.equal(decoded, decoded.bfloat16().float())
    with pytest.raises(AssertionError):
        decode_fp8(weight, scale.double())
    with pytest.raises(AssertionError):
        decode_fp8(weight, scale[:1])
