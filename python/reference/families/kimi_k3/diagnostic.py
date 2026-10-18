"""Bounded native-layer tracing and official-input replays, never a golden path."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path


def tree_map(value, function):
    import torch
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: tree_map(item, function) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_map(item, function) for item in value)
    return value


def differences(left, right, path="output"):
    import torch
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        result = {"path": path, "official_dtype": str(left.dtype), "adapted_dtype": str(right.dtype),
                  "official_shape": list(left.shape), "adapted_shape": list(right.shape)}
        if left.shape != right.shape:
            return [{**result, "shape_mismatch": True}]
        result["different_elements"] = int((left != right).sum())
        if left.is_floating_point() and right.is_floating_point():
            a, b = left.double(), right.double()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("non-finite diagnostic tensor: " + path)
            delta = a - b
            result.update(relative_l2=float(delta.norm() / a.norm().clamp_min(1e-30)),
                          max_abs=float(delta.abs().max()) if delta.numel() else 0.0)
        return [result]
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, f"{path}.{index}")]
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return [item for key in left for item in differences(left[key], right[key], path + "." + key)]
    if left is None and right is None:
        return []
    return [{"path": path, "structure_mismatch": True}]


class Trace:
    """Capture module/function boundaries, restoring hooks even after a failure."""
    def __init__(self):
        self.calls, self.stack, self.counts = {}, [], {}

    def frame(self, name, function, args, kwargs):
        key = name + "#" + str(self.counts.get(name, 0))
        self.counts[name] = self.counts.get(name, 0) + 1
        snapshot = lambda value: value.detach().cpu().clone()
        return {"key": key, "name": name, "function": function,
                "args": tree_map(args, snapshot), "kwargs": tree_map(kwargs, snapshot)}

    def finish(self, frame, output):
        frame["output"] = tree_map(output, lambda value: value.detach().cpu().clone())
        self.calls[frame["key"]] = frame

    def invoke(self, name, function, args, kwargs, replay=None):
        frame = self.frame(name, replay or function, args, kwargs)
        output = function(*args, **kwargs)
        self.finish(frame, output)
        return output

    @contextmanager
    def modules(self, layer):
        handles = []
        def before(name):
            def hook(module, args, kwargs):
                self.stack.append(self.frame(name, module, args, kwargs))
            return hook
        def after(module, args, kwargs, output):
            self.finish(self.stack.pop(), output)
        try:
            for name, module in layer.named_modules():
                handles.append(module.register_forward_pre_hook(before(name or "layer"), with_kwargs=True))
                handles.append(module.register_forward_hook(after, with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.stack.clear()

    @contextmanager
    def functions(self, reference, layer):
        import torch
        import torch.nn.functional as F
        from fla.ops.kda.gate import naive_kda_lowerbound_gate
        linear, chunk, recurrent = F.linear, reference.chunk_kda, reference.fused_recurrent_kda
        combine = layer.block_sparse_moe.moe_infer
        def router(*args, **kwargs):
            if self.stack and self.stack[-1]["name"] == "block_sparse_moe.gate":
                return self.invoke("block_sparse_moe.gate.router_logits", linear, args, kwargs,
                                   replay=lambda *a, **k: F.linear(*a, **k))
            return linear(*args, **kwargs)
        def recurrence(function, name):
            def call(*args, **kwargs):
                if kwargs.get("use_gate_in_kernel") and kwargs.get("lower_bound") is not None:
                    self.invoke("self_attn.effective_gate", naive_kda_lowerbound_gate, (), {
                        "g": kwargs["g"], "A_log": kwargs["A_log"], "dt_bias": kwargs.get("dt_bias"),
                        "lower_bound": kwargs["lower_bound"]})
                if kwargs.get("use_beta_sigmoid_in_kernel"):
                    self.invoke("self_attn.beta_sigmoid", torch.sigmoid, (kwargs["beta"],), {})
                return self.invoke(name, function, args, kwargs)
            return call
        def moe_combine(*args, **kwargs):
            return self.invoke("block_sparse_moe.combine", combine, args, kwargs)
        F.linear = router
        reference.chunk_kda = recurrence(chunk, "self_attn.chunk_kda")
        reference.fused_recurrent_kda = recurrence(recurrent, "self_attn.fused_recurrent_kda")
        layer.block_sparse_moe.moe_infer = moe_combine
        try:
            yield
        finally:
            F.linear, reference.chunk_kda, reference.fused_recurrent_kda = linear, chunk, recurrent
            del layer.block_sparse_moe.moe_infer

    def compare(self, candidate, *, device):
        rows = []
        for key, call in self.calls.items():
            observed = candidate.calls.get(key)
            args = tree_map(call["args"], lambda value: value.to(device))
            kwargs = tree_map(call["kwargs"], lambda value: value.to(device))
            output = call["function"](*args, **kwargs)
            output = tree_map(output, lambda value: value.detach().cpu())
            rows.append({"op": key,
                         "propagated": differences(call["output"], observed["output"]) if observed else [{"missing": True}],
                         "controlled_input": differences(call["output"], output)})
        return rows


def first_difference(rows, field):
    return next(({"op": row["op"], **item} for row in rows for item in row[field]
                 if item.get("relative_l2", 0) > 1e-5 or item.get("shape_mismatch")
                 or item.get("structure_mismatch")), None)


def diagnose(a, *, native_layer, atomic_json, digest, state_type, official_mask):
    import numpy as np
    import torch
    import fla
    from fidelity_windows import load_set, verify_snapshot
    from families.kimi_k3.model import LayerWeights, geometry, load_official, source_identity

    if a.diagnose_layer != 1 or a.probe_state is None:
        raise ValueError("diagnosis currently requires --diagnose-layer 1 and --probe-state")
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(0)
    manifest = load_set(a.windows)
    snapshot = verify_snapshot(manifest, a.snapshot)
    latest = json.loads((a.probe_state / "latest.json").read_bytes())
    binding = latest["binding"]
    if (binding["snapshot_identity"] != snapshot or binding["set_sha256"] != manifest["set_sha256"]
            or binding["torch"] != torch.__version__ or binding["fla"] != fla.__version__
            or binding["image_sha256"] != os.environ.get("CUTEAFD_REFERENCE_IMAGE_SHA256", "")):
        raise ValueError("diagnostic source/set/runtime differs from saved probe")
    state = state_type(a.probe_state, binding)
    blob = state.latest["windows"]["official_short"]["blocks"][0]
    other = state.latest["windows"]["adapted_short"]["blocks"][0]
    if blob["sha256"] != other["sha256"] or "block000" not in blob["path"]:
        raise ValueError("initial official/adapted embedding state differs")
    reference, config, quantization = load_official(a.snapshot)
    weights = LayerWeights(a.snapshot, quantization,
                           kda_num_heads=config.linear_attn_config["num_heads"])
    if not config.linear_attn_config.get("gate_lower_bound") == -5.0:
        raise ValueError("diagnosis requires the pinned bounded-gate checkpoint")
    a.out.mkdir(parents=True, exist_ok=True)
    embedding = torch.from_numpy(np.fromfile(state.verify(blob), dtype="<u2")).view(torch.bfloat16)
    embedding = embedding.reshape(1, -1, config.hidden_size).cuda()
    if embedding.shape[1] != 128:
        raise ValueError("diagnosis requires the saved 128-token short stream")
    input_path = a.out / "layer1-input.bin"
    stamp_path = a.out / "layer1-input.json"
    identity = {"embedding_sha256": blob["sha256"], "snapshot_identity": snapshot,
                "torch": torch.__version__, "image_sha256": binding["image_sha256"],
                "source": source_identity(a.snapshot)}
    with torch.inference_mode():
        if stamp_path.exists():
            stamp = json.loads(stamp_path.read_bytes())
            if stamp["identity"] != identity or digest(input_path) != stamp["sha256"]:
                raise ValueError("persisted diagnostic input differs")
            hidden = torch.from_numpy(np.fromfile(input_path, dtype="<u2")).view(torch.bfloat16)
            hidden = hidden.reshape_as(embedding).cuda()
        else:
            layer = native_layer(reference, config, weights, 0)
            blocks = embedding.new_empty((embedding.shape[1], 0, config.hidden_size))
            mask = None if layer.is_linear_attn else official_mask(embedding)
            hidden, history = layer(embedding, attention_mask=mask, block_residual=blocks,
                                    use_cache=False, output_attentions=False)
            if hidden.dtype != torch.bfloat16 or not torch.equal(history[:, 0], embedding[0]):
                raise ValueError("reconstructed layer-0 state differs in structure")
            pending = input_path.with_suffix(".pending")
            with pending.open("wb") as stream:
                hidden.cpu().contiguous().view(torch.uint16).numpy().tofile(stream)
                stream.flush()
                os.fsync(stream.fileno())
            pending.replace(input_path)
            atomic_json(stamp_path, {"identity": identity, "sha256": digest(input_path)})
            del layer, history, blocks
            torch.cuda.empty_cache()
        print("layer-1 input reconstructed and persisted; loading only layer 1", flush=True)
        layer = native_layer(reference, config, weights, 1)
        if not layer.is_linear_attn:
            raise ValueError("expected KDA at layer index 1")
        traces = []
        for adapted in (False, True):
            trace = Trace()
            def run():
                with trace.modules(layer), trace.functions(reference, layer):
                    layer(hidden.clone(), attention_mask=None, block_residual=embedding[0, :, None].clone(),
                          use_cache=False, output_attentions=False)
            if adapted:
                with geometry(reference):
                    run()
            else:
                run()
            traces.append(trace)
        with geometry(reference):
            rows = traces[0].compare(traces[1], device="cuda")
        native_gate = traces[0].calls["block_sparse_moe.gate#0"]["output"][0]
        adapted_gate = traces[1].calls["block_sparse_moe.gate#0"]["output"][0]
        flips = int((native_gate.sort(-1).values != adapted_gate.sort(-1).values).any(-1).sum())
        a_name = "language_model.model.layers.1.self_attn.A_log"
        raw_a = weights.tensor(a_name, device="cpu")
        exact_heads = torch.equal(layer.self_attn.A_log.detach().cpu(), raw_a[:96])
        recurrence = traces[0].calls["self_attn.chunk_kda#0"]
        flags = {name: value for name, value in recurrence["kwargs"].items()
                 if value is None or isinstance(value, (bool, int, float, str))}
        gate = layer.block_sparse_moe.gate
        report = {"schema": "cuteafd.kimi.layer-diagnostic/1", "layer": 1,
                  "input_sha256": digest(input_path), "source": source_identity(a.snapshot),
                  "diagnostic_sha256": digest(__file__),
                  "adapter_sha256": digest(Path(__file__).with_name("model.py")),
                  "image_sha256": binding["image_sha256"],
                  "torch": torch.__version__, "fla": fla.__version__,
                  "rows": int(hidden.shape[1]), "recurrence": "chunk", "recurrence_flags": flags,
                  "effective_gate_note": "diagnostic-only pinned FLA naive lower-bound formula; not a replacement for fused inference",
                  "a_log_first_96_exact": exact_heads,
                  "a_log_padding": weights.padding_provenance(),
                  "router": {"input_compute_dtype": "torch.float32", "bias_dtype": str(gate.e_score_correction_bias.dtype),
                             "score_function": gate.moe_router_activation_func, "groups": gate.num_expert_group,
                             "topk_groups": gate.topk_group, "topk": gate.top_k,
                             "renormalize": gate.moe_renormalize, "routed_scaling_factor": gate.routed_scaling_factor,
                             "selection": "unchanged torch.topk(sorted=False)", "selection_flip_tokens": flips},
                  "first_propagated_difference": first_difference(rows, "propagated"),
                  "first_controlled_difference": first_difference(rows, "controlled_input"),
                  "subops": rows}
        atomic_json(a.out / "layer-diagnostic.json", report)
        print(json.dumps({key: report[key] for key in ("first_propagated_difference", "first_controlled_difference", "router")}, sort_keys=True), flush=True)
        del layer, traces
        torch.cuda.empty_cache()
