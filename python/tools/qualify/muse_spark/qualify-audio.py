#!/usr/bin/env python3
"""Qualify bundled Muse PCM tower stages, processor spans and native admission.

Run inside the matching dev container on an available SM121 or SM120 device.
Requires the official transformers 5.22.1 used by the Muse text goldens.
"""
from __future__ import annotations
import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
BLOCK_FIELDS = "conv conv_bias norm fc1 fc1_bias fc2 fc2_bias".split()

class Block(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in BLOCK_FIELDS]
class Stage(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("projection", "bias", "norm")] + [("blocks", Block * 2)]
class Spec(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("abi_version", "max_samples", "output_width", "reserved")]
    _fields_ += [("weight_bytes", C.c_uint64), ("stages", Stage * 4)]
    _fields_ += [(name, C.c_uint64) for name in ("norm", "projection", "bias", "post_norm")]
class Ledger(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("weights", "scratch", "device_allocations", "encodes")]
    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_}
OBSERVER = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_int32, C.c_uint32, C.c_void_p, C.c_uint32, C.c_uint32)


def scratch_bytes(samples):
    if not 1 <= samples <= 7_200_000:
        raise ValueError("PCM must contain 1..7200000 samples")
    tokens = min((samples + 1919) // 1920, 16)
    rows, largest = tokens * 1920, tokens * 8192
    for width, patch in zip((256, 512, 1024, 2048), (16, 6, 5, 4)):
        rows //= patch
        largest = max(largest, rows * width)
    align = lambda n: (n + 255) // 256 * 256
    return align(largest * 2) * 3 + align(largest * 8) * 2 + sum(align(h * 6 * 2) * 2 for h in (256, 512, 1024, 2048))


def descriptor(weights, max_samples):
    spec = Spec(abi_version=1, max_samples=max_samples, output_width=8192)
    offsets, cursor = {}, 0
    for name, tensor in sorted(weights.items()):
        if str(tensor.dtype) != "torch.bfloat16":
            raise ValueError(f"{name}: checkpoint audio must remain BF16")
        cursor = (cursor + 255) // 256 * 256
        offsets[name] = cursor
        cursor += tensor.numel() * 2
    spec.weight_bytes = (cursor + 255) // 256 * 256
    arena = bytearray(spec.weight_bytes)
    for name, tensor in weights.items():
        raw = tensor.cpu().contiguous().view(__import__('torch').uint16).numpy().tobytes()
        at = offsets[name]
        arena[at:at + len(raw)] = raw
    for i in range(4):
        prefix = f"layers.{i}."
        stage = spec.stages[i]
        stage.projection = offsets[prefix + "patch_embedder.proj.weight"]
        stage.bias = offsets[prefix + "patch_embedder.proj.bias"]
        stage.norm = offsets[prefix + "patch_embedder.norm.weight"]
        for j in range(2):
            prefix_block = prefix + f"conv_blocks.{j}."
            for field, suffix in zip(BLOCK_FIELDS, ("conv.weight", "conv.bias", "norm.weight", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")):
                setattr(stage.blocks[j], field, offsets[prefix_block + suffix])
    for field, suffix in (("norm", "norm.weight"), ("projection", "proj.weight"), ("bias", "proj.bias"), ("post_norm", "post_proj_norm.weight")):
        setattr(spec, field, offsets[suffix])
    return spec, arena


def stage_metrics(actual, bf16, fp32):
    """Official checkpoint BF16-vs-FP32 error is the numerical yardstick."""
    rms = lambda x: float(x.square().mean().sqrt())
    signal = rms(fp32)
    native_error, official_error = rms(actual - fp32), rms(bf16 - fp32)
    floor = 1e-7
    return dict(signal_rms=signal, absolute_floor=floor,
                native_absolute_rms=native_error, official_bf16_absolute_rms=official_error,
                native_relative_rms=native_error / max(signal, floor),
                official_bf16_relative_rms=official_error / max(signal, floor),
                error_ratio=native_error / max(official_error, floor),
                native_vs_bf16_relative_rms=rms(actual - bf16) / max(rms(bf16), floor),
                max_abs=float((actual - fp32).abs().max()),
                passed=native_error <= 1.1 * max(official_error, floor))


def check(code):
    if code:
        raise RuntimeError(f"native PCM status {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--clip", type=Path, action="append", default=[], help="real mono audio decoded by soundfile; exactly 24 kHz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-samples", type=int, default=24000 * 30)
    parser.add_argument("--diagnose-conv", action="store_true", help="report isolated BF16 convolution bias-rounding yardstick")
    parser.add_argument("--diagnose-ops", action="store_true", help="compare each shared primitive on official BF16 inputs")
    parser.add_argument("--strict-bf16-reduction", action="store_true", help="disable PyTorch reduced-precision BF16 GEMM reduction")
    parser.add_argument("--check-owner-lifetime", action="store_true", help="destroy overlapping owners, then unload and recreate the module set")
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkAudioConfig
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkAudioModel
    from transformers.models.muse_spark.feature_extraction_muse_spark import MuseSparkFeatureExtractor
    from transformers.models.muse_spark.processing_muse_spark import MuseSparkProcessor
    ref_path = ROOT / "python/reference/families/muse_spark/audio.py"
    module = importlib.util.spec_from_file_location("muse_audio_reference", ref_path)
    ref = importlib.util.module_from_spec(module)
    module.loader.exec_module(ref)
    sys.path.insert(0, str(ROOT / "python/reference"))
    from families.muse_spark.model import source_identity
    provenance = source_identity()
    import hashlib
    import transformers
    source_root = Path(transformers.__file__).parent / "models/muse_spark"
    processor_pins = {
        "feature_extraction_muse_spark.py": "ea6791c32a840d3bfc62aa61b4193a6865dfa7d05d45068a462580feee76adcf",
        "processing_muse_spark.py": "880a4f2328605018ccfbee7a7aef564d32781c308c34df0d145f8ce939e80d78",
    }
    for name, digest in processor_pins.items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"official processor source differs: {name}")
    provenance["processor_sha256"] = processor_pins
    provenance["snapshot"] = str(args.snapshot)
    provenance["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
    audio_config, processor_config = ref.config(args.snapshot)
    torch.cuda.set_device(0)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.strict_bf16_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    provenance["allow_bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    weights = ref.load_weights(args.snapshot)
    model = MuseSparkAudioModel(MuseSparkAudioConfig(**audio_config)).to(dtype=torch.bfloat16, device="cuda").eval()
    model.load_state_dict(weights, strict=True)
    feature_extractor = MuseSparkFeatureExtractor(**processor_config)
    # replace_audio_token does not depend on the other processor components.
    processor = object.__new__(MuseSparkProcessor)
    processor.feature_extractor = feature_extractor
    processor.audio_start_token, processor.audio_token, processor.audio_end_token = "<|soft_audio_start|>", "<|audio|>", "<|soft_audio_end|>"
    clips = [("very-short", np.array([0.125], dtype=np.float32)), ("silence", np.zeros(24001, dtype=np.float32)),
             ("long-signal", (0.2 * np.sin(np.arange(args.long_samples, dtype=np.float32) * (2 * np.pi * 317 / 24000))).astype(np.float32))]
    for path in args.clip:
        import soundfile as sf
        pcm, sr = sf.read(path, dtype="float32")
        if sr != 24000 or pcm.ndim != 1:
            raise ValueError(f"{path}: qualification requires 24 kHz mono")
        clips.append((path.stem, pcm))
    spec, arena = descriptor(weights, max(len(pcm) for _, pcm in clips))
    lib = C.CDLL(str(args.library))
    cudart = C.CDLL("libcudart.so")
    cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    lib.cuteafd_pcm_required.argtypes = [C.POINTER(Spec), C.POINTER(Ledger)]
    lib.cuteafd_pcm_create.argtypes = [C.POINTER(Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)]
    lib.cuteafd_pcm_upload.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]
    lib.cuteafd_pcm_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint64, OBSERVER, C.c_void_p]
    lib.cuteafd_pcm_get_ledger.argtypes = [C.c_void_p, C.POINTER(Ledger)]
    lib.cuteafd_pcm_destroy.argtypes = [C.c_void_p]
    charge = Ledger()
    check(lib.cuteafd_pcm_required(C.byref(spec), C.byref(charge)))
    expected = dict(weights=spec.weight_bytes, scratch=scratch_bytes(spec.max_samples), device_allocations=2, encodes=0)
    if charge.as_dict() != expected:
        raise AssertionError((charge.as_dict(), expected))
    owner = C.c_void_p()
    budget = charge.weights + charge.scratch
    if lib.cuteafd_pcm_create(C.byref(spec), 0, budget - 1, C.byref(owner)) == 0 or owner.value:
        raise AssertionError("under-admitted owner allocated")
    check(lib.cuteafd_pcm_create(C.byref(spec), 0, budget, C.byref(owner)))
    storage = (C.c_uint8 * len(arena)).from_buffer(arena)
    report = dict(provenance=provenance, ledger=expected, clips=[], passed=True,
                  served_next_token_kl="pending", cumulative_stage_yardstick_pass=True)
    diagnostic_modules = []
    try:
        check(lib.cuteafd_pcm_upload(owner, 0, storage, len(arena)))
        if args.check_owner_lifetime:
            prepared, tokens = ref.prepare(clips[0][1])
            padded = prepared.bfloat16().view(torch.uint16).numpy()
            baseline = np.empty((tokens, 8192), dtype=np.uint16)
            def encode_first(result):
                check(lib.cuteafd_pcm_encode(owner, padded.ctypes.data, len(clips[0][1]),
                    result.ctypes.data, result.nbytes, OBSERVER(), None))
            encode_first(baseline)
            other = C.c_void_p()
            check(lib.cuteafd_pcm_create(C.byref(spec), 0, budget, C.byref(other)))
            check(lib.cuteafd_pcm_destroy(other))
            repeated = np.empty_like(baseline)
            encode_first(repeated)
            if not np.array_equal(baseline, repeated):
                raise AssertionError("overlapping owner destruction changed features")
            check(lib.cuteafd_pcm_destroy(owner))
            owner = C.c_void_p()
            check(lib.cuteafd_pcm_create(C.byref(spec), 0, budget, C.byref(owner)))
            check(lib.cuteafd_pcm_upload(owner, 0, storage, len(arena)))
            encode_first(repeated)
            if not np.array_equal(baseline, repeated):
                raise AssertionError("module unload/recreate changed features")
            report["owner_lifetime_pass"] = True
        baseline_encodes = 1 if args.check_owner_lifetime else 0
        for label, pcm in clips:
            prepared, tokens = ref.prepare(pcm)
            official_inputs = feature_extractor(pcm, sampling_rate=24000, return_tensors="pt")
            if not torch.equal(prepared, official_inputs.input_values[0]):
                raise AssertionError(f"{label}: PCM preparation differs")
            if ref.placeholder(len(pcm)) != processor.replace_audio_token(official_inputs, 0):
                raise AssertionError(f"{label}: placeholder bytes differ")
            primitive_metrics = []
            def official_stages_for(dtype):
                model.to(dtype=dtype)
                stages, hooks = {}, []
                if args.diagnose_ops and dtype == torch.bfloat16:
                    support = C.c_void_p()
                    lib.cuteafd_pcm_support_create.argtypes = [C.POINTER(C.c_void_p)]
                    lib.cuteafd_pcm_support_launch.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p), C.c_int, C.c_void_p]
                    lib.cuteafd_pcm_support_destroy.argtypes = [C.c_void_p]
                    check(lib.cuteafd_pcm_support_create(C.byref(support)))
                    diagnostic_modules.append(support)
                    def primitive_probe(name, module, inputs, output):
                        x = inputs[0].contiguous()
                        out = torch.empty_like(output, memory_format=torch.contiguous_format)
                        width, outputs = x.shape[-1], 0
                        if isinstance(module, torch.nn.Linear):
                            op, width, outputs = 0, module.in_features, module.out_features
                            operands = [x, module.weight, module.bias, out]
                            rows = x.numel() // width
                        elif isinstance(module, torch.nn.Conv1d):
                            op, width = 1, module.in_channels
                            x = x.transpose(1, 2).contiguous()
                            out = torch.empty_like(x)
                            history = torch.zeros((1, width, 6), device="cuda", dtype=torch.bfloat16)
                            operands = [x, module.weight, module.bias, history, out, history]
                            rows = x.shape[1]
                        elif hasattr(module, "with_scale"):
                            op = 2
                            operands = [x] + [module.weight] * 5 + [out]
                            rows = x.numel() // width
                        elif module.__class__.__name__ == "GELUActivation":
                            op = 3
                            operands = [x, out]
                            rows = x.numel() // width
                        else:
                            return
                        pointers = (C.c_void_p * len(operands))(*(t.data_ptr() for t in operands))
                        check(lib.cuteafd_pcm_support_launch(support, op, width, outputs, pointers, rows, torch.cuda.current_stream().cuda_stream))
                        torch.cuda.synchronize()
                        if op == 1:
                            out = out.transpose(1, 2)
                        delta = out.float() - output.float()
                        record = dict(clip=label, primitive=name, rms=float(delta.square().mean().sqrt()),
                            max_abs=float(delta.abs().max()), different=float((out != output).float().mean()))
                        if op == 0:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                                torch.nn.functional.linear(inputs[0], module.weight, module.bias)
                            record["linear_path"] = [event.key for event in profile.key_averages() if event.key.startswith("aten::")]
                            fp32 = torch.nn.functional.linear(inputs[0].float(), module.weight.float(), module.bias.float())
                            native_error = float((out.float()-fp32).square().mean().sqrt())
                            official_error = float((output.float()-fp32).square().mean().sqrt())
                            record.update(native_fp32_rms=native_error, official_fp32_rms=official_error,
                                error_ratio=native_error / max(official_error, 1e-7),
                                native_not_worse=native_error <= official_error,
                                within_bf16_yardstick=native_error <= 1.1 * max(official_error, 1e-7))
                            raw = torch.nn.functional.linear(x.float(), module.weight.float())
                            fused = (raw + module.bias.float()).bfloat16()
                            separate = raw.bfloat16() + module.bias
                            bf16_separate = torch.nn.functional.linear(inputs[0], module.weight) + module.bias
                            contiguous_linear = torch.nn.functional.linear(x, module.weight, module.bias)
                            record.update(input_shape=list(inputs[0].shape), input_stride=list(inputs[0].stride()),
                                output_stride=list(output.stride()),
                                bf16_separate_equal=bool(torch.equal(bf16_separate, output)),
                                contiguous_linear_equal=bool(torch.equal(contiguous_linear, output)),
                                fused_equal=bool(torch.equal(fused, output)),
                                separate_equal=bool(torch.equal(separate, output)),
                                native_vs_fused_rms=float((out.float()-fused.float()).square().mean().sqrt()))
                        if op != 0:
                            if op == 1:
                                fp32 = torch.nn.functional.conv1d(torch.nn.functional.pad(inputs[0].float(), (6, 0)),
                                    module.weight.float(), module.bias.float(), groups=width)
                            elif op == 2:
                                f = inputs[0].float()
                                fp32 = f * torch.pow(f.square().mean(-1, keepdim=True) + module.eps, -0.5) * module.weight.float()
                            else:
                                fp32 = torch.nn.functional.gelu(inputs[0].float(), approximate="none")
                            native_error = float((out.float()-fp32).square().mean().sqrt())
                            official_error = float((output.float()-fp32).square().mean().sqrt())
                            record.update(native_fp32_rms=native_error, official_fp32_rms=official_error,
                                error_ratio=native_error / max(official_error, 1e-7),
                                native_not_worse=native_error <= official_error,
                                within_bf16_yardstick=native_error <= 1.1 * max(official_error, 1e-7))
                        primitive_metrics.append(record)
                        print(json.dumps(record), flush=True)
                    from functools import partial
                    for name, module in model.named_modules():
                        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) or hasattr(module, "with_scale") or module.__class__.__name__ == "GELUActivation":
                            hooks.append(module.register_forward_hook(partial(primitive_probe, name)))
                if args.diagnose_conv and dtype == torch.bfloat16:
                    def conv_probe(module, inputs, output):
                        import torch.nn.functional as F
                        x = F.pad(inputs[0].float(), (6, 0))
                        raw = F.conv1d(x, module.weight.float(), groups=module.groups)
                        fused = (raw + module.bias.float()[None, :, None]).bfloat16().float()
                        separate = (raw.bfloat16() + module.bias[None, :, None]).float()
                        print(json.dumps(dict(clip=label, conv=module.cache_key,
                            fused_rms=float((fused-output.float()).square().mean().sqrt()),
                            separate_rms=float((separate-output.float()).square().mean().sqrt()),
                            fused_equal=bool(torch.equal(fused,output.float())),
                            separate_equal=bool(torch.equal(separate,output.float())))), flush=True)
                    for layer in model.layers:
                        for block in layer.conv_blocks:
                            hooks.append(block.conv.register_forward_hook(conv_probe))
                def capture(stage):
                    return lambda _module, _inputs, output: stages.__setitem__(stage, output.detach().float().cpu().squeeze(0))
                for i, layer in enumerate(model.layers):
                    hooks.append(layer.patch_embedder.register_forward_hook(capture(i * 3)))
                    for j, block in enumerate(layer.conv_blocks):
                        hooks.append(block.register_forward_hook(capture(i * 3 + j + 1)))
                try:
                    with torch.inference_mode():
                        official = model(input_values=prepared[None].to("cuda", dtype)).last_hidden_state
                        stages[12] = official.float().cpu().squeeze(0)
                finally:
                    for hook in hooks:
                        hook.remove()
                return stages
            official_stages = official_stages_for(torch.bfloat16)
            torch.cuda.synchronize()
            while diagnostic_modules:
                check(lib.cuteafd_pcm_support_destroy(diagnostic_modules.pop()))
            independent = ref.forward(pcm, {name: tensor.to("cuda") for name, tensor in weights.items()}).float().cpu()
            if not torch.equal(independent, official_stages[12]):
                raise AssertionError(f"{label}: independent reference differs from official BF16")
            fp32_stages = official_stages_for(torch.float32)
            chunks = {i: [] for i in range(13)}
            errors = []
            @OBSERVER
            def observer(_ctx, stage, offset, ptr, rows, width):
                try:
                    host = torch.empty((rows, width), dtype=torch.bfloat16, pin_memory=True)
                    check(cudart.cudaMemcpy(host.data_ptr(), ptr, rows * width * 2, 2))
                    chunks[stage].append((offset, host.float().clone()))
                    return 0
                except Exception as exc:
                    errors.append(str(exc))
                    return 1
            padded = prepared.bfloat16().view(torch.uint16).numpy()
            output = np.empty((tokens, 8192), dtype=np.uint16)
            status = lib.cuteafd_pcm_encode(owner, padded.ctypes.data, len(pcm), output.ctypes.data, output.nbytes, observer, None)
            if errors:
                raise AssertionError(errors)
            check(status)
            stages = []
            for stage in range(13):
                actual = torch.cat([x for _, x in sorted(chunks[stage])])
                metric = stage_metrics(actual, official_stages[stage], fp32_stages[stage])
                stages.append(dict(stage=stage, **metric))
                report["cumulative_stage_yardstick_pass"] &= metric["passed"]
            final = torch.from_numpy(output).view(torch.bfloat16).float()
            if not torch.equal(final, torch.cat([x for _, x in sorted(chunks[12])])):
                raise AssertionError("returned feature bytes differ from final stage")
            linears = [metric for metric in primitive_metrics if "linear_path" in metric]
            report["clips"].append(dict(name=label, samples=len(pcm), tokens=tokens, spans_exact=True,
                stages=stages, primitives=primitive_metrics,
                per_op_native_not_worse=all(metric["native_not_worse"] for metric in primitive_metrics) if primitive_metrics else None,
                per_op_fp32_yardstick_pass=all(metric["within_bf16_yardstick"] for metric in primitive_metrics) if primitive_metrics else None,
                linear_fp32_yardstick_pass=all(metric["within_bf16_yardstick"] for metric in linears) if linears else None))
        after = Ledger()
        check(lib.cuteafd_pcm_get_ledger(owner, C.byref(after)))
        if (after.weights, after.scratch, after.device_allocations, after.encodes) != (charge.weights, charge.scratch, 2, len(clips) + baseline_encodes):
            raise AssertionError("ledger grew or encode accounting differs")
        report["ledger_after"] = after.as_dict()
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            for support in diagnostic_modules:
                check(lib.cuteafd_pcm_support_destroy(support))
            check(lib.cuteafd_pcm_destroy(owner))
    report["per_op_native_not_worse"] = all(c["per_op_native_not_worse"] for c in report["clips"]) if args.diagnose_ops else None
    report["per_op_fp32_yardstick_pass"] = all(c["per_op_fp32_yardstick_pass"] for c in report["clips"]) if args.diagnose_ops else None
    # Retain the stricter cumulative verdict; per-op diagnostics do not conceal it.
    report["passed"] = report["cumulative_stage_yardstick_pass"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], clips=len(clips), output=str(args.output))))
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
