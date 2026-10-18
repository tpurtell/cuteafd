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


def first_difference(rows, field, limit=1e-6):
    return next(({"op": row["op"], **item} for row in rows for item in row[field]
                 if item.get("relative_l2", 0) > limit or item.get("shape_mismatch")
                 or item.get("structure_mismatch") or item.get("missing")), None)


def linear_variants(trace, *, device):
    """Locate GEMV/GEMM rounding changes on identical native expert inputs."""
    import torch
    from shape_invariant import _linear
    strategies = {"native_M1": (1, False), "native_M32": (32, False), "F32_M128_native_output": (128, True)}
    summaries = {name: {"compared": 0, "over_1e-5": 0, "max_relative_l2": 0., "first_difference": None}
                 for name in strategies}
    for key, call in trace.calls.items():
        if ".experts." not in key or not isinstance(call["function"], torch.nn.Linear):
            continue
        x = call["args"][0].to(device)
        module = call["function"]
        for name, (width, widen) in strategies.items():
            dtype = x.dtype
            weight = module.weight.float() if widen else module.weight
            values = x.float() if widen else x
            result = []
            for start in range(0, len(values), width):
                part = values[start:start + width]
                padded = part.new_zeros((width, part.shape[-1]))
                padded[:len(part)].copy_(part)
                result.append(_linear(padded, weight, module.bias)[:len(part)])
            output = torch.cat(result).to(dtype).cpu()
            metric = differences(call["output"], output)[0]
            summary = summaries[name]
            summary["compared"] += 1
            summary["max_relative_l2"] = max(summary["max_relative_l2"], metric["relative_l2"])
            if metric["relative_l2"] > 1e-5:
                summary["over_1e-5"] += 1
                if summary["first_difference"] is None:
                    summary["first_difference"] = {"op": key, **metric}
    return summaries


def duplicate_probe(embedding):
    if embedding.ndim != 3 or embedding.shape[0] != 1 or embedding.shape[1] != 128:
        raise ValueError("official noise diagnosis requires one 128-token probe")
    return embedding.repeat(2, 1, 1)


def logit_metrics(native, candidate):
    import numpy as np
    a, b = np.asarray(native, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        raise ValueError("noise comparison logit geometry differs")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite noise comparison logits")
    logp = a - np.logaddexp.reduce(a, axis=-1, keepdims=True)
    logq = b - np.logaddexp.reduce(b, axis=-1, keepdims=True)
    return {"tokens": len(a), "top1": float(np.mean(a.argmax(-1) == b.argmax(-1))),
            "kl": float((np.exp(logp) * (logp - logq)).sum(-1).mean()),
            "relative_logit_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)),
            "max_abs_logit_difference": float(np.abs(a - b).max())}


def diagnose_official_noise(a, *, native_layer, atomic_json, digest, state_type, official_mask):
    """Change batch geometry only; never turn a diagnostic into qualification."""
    import time
    import numpy as np
    import torch
    import fla
    import transformers
    import compressed_tensors
    from fidelity_windows import load_set, verify_snapshot
    from families.kimi_k3.model import LayerWeights, load_official, package_source_identity, source_identity
    from shape_invariant import linear

    if a.probe_state is None:
        raise ValueError("official noise diagnosis requires --probe-state")
    started = time.monotonic()
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.manual_seed(0)
    manifest = load_set(a.windows)
    snapshot = verify_snapshot(manifest, a.snapshot)
    latest = json.loads((a.probe_state / "latest.json").read_bytes())
    binding = latest["binding"]
    runtime = {"torch": torch.__version__, "fla": fla.__version__,
               "transformers": transformers.__version__, "compressed_tensors": compressed_tensors.__version__,
               "image_sha256": os.environ.get("CUTEAFD_REFERENCE_IMAGE_SHA256", ""),
               "compute_capability": list(torch.cuda.get_device_capability())}
    if (binding["snapshot_identity"] != snapshot or binding["set_sha256"] != manifest["set_sha256"]
            or any(binding[key] != value for key, value in runtime.items())):
        raise ValueError("noise source/set/runtime differs from retained probe")
    pins = source_identity(a.snapshot)
    for package, key in ((fla, "fla"), (transformers, "transformers"), (compressed_tensors, "compressed_tensors")):
        pins[key + "_source_sha256"] = package_source_identity(package)
    pins["reference_adapter_sha256"] = digest(Path(__file__).with_name("model.py"))
    if any(binding["source_pin"][key] != value for key, value in pins.items()):
        raise ValueError("noise native modeling/package sources differ from retained probe")
    reference, config, quantization = load_official(a.snapshot)
    if latest["layer"] != config.num_hidden_layers - 1:
        raise ValueError("noise comparison requires completed native batch-1 decoder states")
    retained = state_type(a.probe_state, binding)
    blob = latest["windows"]["official_short"]["blocks"][0]
    if blob["sha256"] != latest["windows"]["adapted_short"]["blocks"][0]["sha256"]:
        raise ValueError("retained official/candidate initial embeddings differ")
    a.out.mkdir(parents=True, exist_ok=True)
    noise_binding = {"role": "official-batch-noise-diagnostic-only", "source_pin": pins,
                     "parent_probe_manifest_sha256": digest(a.probe_state / "latest.json"),
                     "diagnostic_sha256": digest(__file__), "runtime": runtime,
                     "blas": str(torch.backends.cuda.preferred_blas_library()),
                     "embedding_sha256": blob["sha256"], "batch": 2, "tokens_per_copy": 128,
                     "state_bytes_per_layer": 2 * 128 * config.hidden_size * 2}
    state = state_type(a.out / "layer-state", noise_binding)
    weights = LayerWeights(a.snapshot, quantization,
                           kda_num_heads=config.linear_attn_config["num_heads"])
    with torch.inference_mode():
        if state.latest is None:
            embedding = torch.from_numpy(np.fromfile(retained.verify(blob), dtype="<u2")).view(torch.bfloat16)
            embedding = embedding.reshape(1, -1, config.hidden_size).cuda()
            pending = state.begin(-1)
            state.save(pending, "official_b2", duplicate_probe(embedding))
            state.commit(pending)
            del embedding
        for index in range(state.latest["layer"] + 1, config.num_hidden_layers):
            tick = time.monotonic()
            layer = native_layer(reference, config, weights, index)
            pending = state.begin(index)
            hidden, blocks = state.load("official_b2", config.hidden_size)
            hidden = hidden.reshape(2, 128, config.hidden_size)
            mask = None if layer.is_linear_attn else official_mask(hidden)
            result, history = layer(hidden, attention_mask=mask, block_residual=blocks,
                                    use_cache=False, output_attentions=False)
            append = index % config.attn_res_block_size == 0
            if history.shape[1] != blocks.shape[1] + int(append):
                raise ValueError("noise official block-residual count differs")
            if append and not torch.equal(history[:, -1], hidden.reshape(-1, config.hidden_size)):
                raise ValueError("noise official block-residual state differs")
            state.save(pending, "official_b2", result, append_block=append)
            difference = result[0].float() - result[1].float()
            pending["checks"].append({"layer": index, "duplicate_copy_max_abs": float(difference.abs().max()),
                                      "duplicate_copy_relative_l2": float(difference.norm() / result[0].float().norm().clamp_min(1e-30))})
            state.commit(pending)
            del layer, hidden, blocks, result, history, difference
            torch.cuda.empty_cache()
            print(f"official B2 layer {index + 1}/{config.num_hidden_layers} committed; {time.monotonic() - tick:.1f}s; read {weights.bytes_read / 1e9:.3f} GB", flush=True)
        with torch.device("meta"):
            norm = reference.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            output_norm = reference.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            output_proj = torch.nn.Linear(config.hidden_size, 1, bias=False)
        weights.load_module(norm, "language_model.model.norm")
        weights.load_module(output_norm, "language_model.model.output_attn_res_norm")
        weights.load_module(output_proj, "language_model.model.output_attn_res_proj")
        head = weights.weight("language_model.lm_head.weight").float()
        weights.close()
        arrays, artifacts = {}, {}
        for source, wid, batch in ((retained, "official_short", 1), (retained, "adapted_short", 1),
                                   (state, "official_b2", 2)):
            hidden, blocks = source.load(wid, config.hidden_size)
            hidden = reference._apply_attn_res(hidden[0], blocks, output_proj, output_norm)
            hidden = norm(hidden.reshape(batch, 128, config.hidden_size))
            for copy in range(batch):
                key = wid + "_copy" + str(copy)
                logits = linear(hidden[copy, :127].float(), head).float().cpu().numpy()
                if not np.isfinite(logits).all():
                    raise ValueError("non-finite official noise logits")
                arrays[key] = logits
                path = a.out / (key + "-logits.npy")
                temporary = path.with_suffix(".pending")
                with temporary.open("wb") as stream:
                    np.save(stream, logits, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
                artifacts[key] = {"path": path.name, "sha256": digest(path)}
            del hidden, blocks
        baseline = arrays["official_short_copy0"]
        comparisons = {"official_b1_vs_b2_copy" + str(copy): logit_metrics(baseline, arrays["official_b2_copy" + str(copy)])
                       for copy in range(2)}
        candidate = logit_metrics(baseline, arrays["adapted_short_copy0"])
        comparisons["official_b1_vs_fixed_M128_cublas"] = candidate
        comparisons["official_b2_copy0_vs_copy1"] = logit_metrics(arrays["official_b2_copy0"], arrays["official_b2_copy1"])
        noise = [comparisons["official_b1_vs_b2_copy" + str(copy)] for copy in range(2)]
        envelope = {key: (min if key == "top1" else max)(row[key] for row in noise)
                    for key in ("top1", "kl", "relative_logit_l2", "max_abs_logit_difference")}
        within = {"top1": candidate["top1"] >= envelope["top1"],
                  "kl": candidate["kl"] <= envelope["kl"],
                  "relative_logit_l2": candidate["relative_logit_l2"] <= envelope["relative_logit_l2"]}
        report = {"schema": "cuteafd.kimi.official-noise/1", "binding": noise_binding,
                  "seconds": time.monotonic() - started, "bytes_read": weights.bytes_read,
                  "comparison": "unchanged native 128-token forward, B1 versus two independent duplicate prompts at B2; 127 prediction rows per copy; no sequence split, cache or causal changes",
                  "head": "same widened F32 head and fixed-M128 projection as the retained probe",
                  "metric_precision": "float64 from stored float32 logits",
                  "comparisons": comparisons, "observed_noise_envelope": envelope,
                  "candidate_within_observed_noise": within, "artifacts": artifacts,
                  "qualification": "diagnostic only; original numerical and byte-exact-prefix gates remain unchanged",
                  "layers": state.latest["checks"]}
        atomic_json(a.out / "official-noise.json", report)
        print(json.dumps({"comparisons": comparisons, "observed_noise_envelope": envelope,
                          "candidate_within_observed_noise": within}, sort_keys=True), flush=True)


def diagnose(a, *, native_layer, atomic_json, digest, state_type, official_mask):
    import numpy as np
    import torch
    import fla
    from fidelity_windows import load_set, verify_snapshot
    from families.kimi_k3.model import LayerWeights, configure_f32, geometry, load_official, source_identity

    if a.diagnose_layer != 1 or a.probe_state is None:
        raise ValueError("diagnosis currently requires --diagnose-layer 1 and --probe-state")
    arithmetic = configure_f32()
    torch.manual_seed(0)
    if a.diagnose_blas:
        torch.backends.cuda.preferred_blas_library(a.diagnose_blas)
    blas = str(torch.backends.cuda.preferred_blas_library())
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
    weights = LayerWeights(a.snapshot, quantization, dtype=torch.float32,
                           kda_num_heads=config.linear_attn_config["num_heads"])
    if not config.linear_attn_config.get("gate_lower_bound") == -5.0:
        raise ValueError("diagnosis requires the pinned bounded-gate checkpoint")
    a.out.mkdir(parents=True, exist_ok=True)
    embedding = torch.from_numpy(np.fromfile(state.verify(blob), dtype="<u2")).view(torch.bfloat16)
    embedding = embedding.reshape(1, -1, config.hidden_size).float().cuda()
    if embedding.shape[1] != 128:
        raise ValueError("diagnosis requires the saved 128-token short stream")
    input_path = a.out / "layer1-input.bin"
    stamp_path = a.out / "layer1-input.json"
    identity = {"embedding_sha256": blob["sha256"], "snapshot_identity": snapshot,
                "torch": torch.__version__, "image_sha256": binding["image_sha256"],
                "arithmetic": arithmetic, "adapter_sha256": digest(Path(__file__).with_name("model.py")),
                "source": source_identity(a.snapshot)}
    with torch.inference_mode():
        if stamp_path.exists():
            stamp = json.loads(stamp_path.read_bytes())
            if stamp["identity"] != identity or digest(input_path) != stamp["sha256"]:
                raise ValueError("persisted diagnostic input differs")
            hidden = torch.from_numpy(np.fromfile(input_path, dtype="<f4"))
            hidden = hidden.reshape_as(embedding).cuda()
        else:
            layer = native_layer(reference, config, weights, 0)
            blocks = embedding.new_empty((embedding.shape[1], 0, config.hidden_size))
            mask = None if layer.is_linear_attn else official_mask(embedding)
            hidden, history = layer(embedding, attention_mask=mask, block_residual=blocks,
                                    use_cache=False, output_attentions=False)
            if hidden.dtype != torch.float32 or not torch.equal(history[:, 0], embedding[0]):
                raise ValueError("reconstructed layer-0 state differs in structure")
            pending = input_path.with_suffix(".pending")
            with pending.open("wb") as stream:
                hidden.cpu().contiguous().numpy().tofile(stream)
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
                  "torch": torch.__version__, "fla": fla.__version__, "blas": blas,
                  "rows": int(hidden.shape[1]), "arithmetic": arithmetic, "relative_l2_limit": 1e-6,
                  "recurrence": "chunk", "recurrence_flags": flags,
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
        float_dtypes = {item[key] for row in rows for field in ("propagated", "controlled_input")
                        for item in row[field] for key in ("official_dtype", "adapted_dtype")
                        if key in item and "float" in item[key]}
        report["floating_dtypes"] = sorted(float_dtypes)
        report["passed"] = (float_dtypes == {"torch.float32"} and exact_heads and flips == 0
                            and report["first_propagated_difference"] is None
                            and report["first_controlled_difference"] is None)
        if a.diagnose_linear_variants:
            report["linear_variants"] = linear_variants(traces[0], device="cuda")
            print(json.dumps({"linear_variants": report["linear_variants"]}, sort_keys=True), flush=True)
        atomic_json(a.out / "layer-diagnostic.json", report)
        print(json.dumps({key: report[key] for key in ("first_propagated_difference", "first_controlled_difference", "router")}, sort_keys=True), flush=True)
        del layer, traces
        weights.release_experts()
        torch.cuda.empty_cache()
        if not report["passed"]:
            raise ValueError("F32 layer-1 qualification exceeds 1e-6 or changes dtype/routing")
