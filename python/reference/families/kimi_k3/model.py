"""Hash-checked Kimi K3 text decoder; no serving implementation or numeric fallback."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import types

REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
SOURCE_PINS = {
    "configuration_kimi_k3.py": "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
    "modeling_kimi_linear.py": "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    "modeling_kimi_k3.py": "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2",
}


def source_identity(snapshot):
    snapshot = Path(snapshot)
    if snapshot.name != REVISION:
        raise ValueError("Kimi reference requires the pinned official checkpoint")
    actual = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
              for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise ValueError("official Kimi modeling source differs")
    return {"revision": REVISION, **{name.removesuffix(".py") + "_sha256": value
                                    for name, value in actual.items()}}


def load_official(snapshot):
    source_identity(snapshot)
    name = "_fidelity_kimi_k3_checkpoint"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(snapshot).resolve())]
    sys.modules[name] = package
    config_module = importlib.import_module(name + ".configuration_kimi_k3")
    reference = importlib.import_module(name + ".modeling_kimi_linear")
    raw = json.loads((Path(snapshot) / "config.json").read_bytes())
    config = config_module.KimiLinearConfig(**raw["text_config"])
    config._attn_implementation = "eager"
    return reference, config, raw["text_config"]["quantization_config"]


def package_source_identity(module):
    root = Path(module.__file__).resolve().parent
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*.py"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def configure_f32():
    """Apply the same strict IEEE arithmetic to Torch and the official FLA kernels."""
    import torch
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    os.environ["FLA_TRIL_PRECISION"] = "ieee"
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from triton import knobs
    import triton.language as tl
    import fla.ops.kda.chunk_intra as intra
    # This pinned FLA kernel explicitly chooses TF32, independently of Torch.
    knobs.language.fp32_default = "ieee"
    intra.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    return {"dtype": "float32", "matmul_precision": torch.get_float32_matmul_precision(),
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "triton_dot_precision": knobs.language.fp32_default,
            "kda_solve_tril_precision": intra.SOLVE_TRIL_DOT_PRECISION.value}


def decompress_f32(packed, scales):
    """Decode E2M1 and group-32 E8M0 directly in F32, without a BF16 intermediate."""
    import torch
    if (packed.dtype != torch.uint8 or scales.dtype != torch.uint8 or packed.ndim != 2
            or scales.shape != (packed.shape[0], packed.shape[1] * 2 // 32)
            or packed.shape[1] * 2 % 32 or torch.any(scales == 255)):
        raise ValueError("unsupported native MXFP4 bytes or scales")
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         device=packed.device, dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    values = table[codes.long()].reshape(packed.shape[0], -1, 32)
    result = torch.ldexp(values, scales.to(torch.int32)[..., None] - 127).reshape(packed.shape[0], -1)
    if not torch.isfinite(result).all():
        raise ValueError("non-finite checkpoint-native MXFP4 F32 dequantization")
    return result


class LayerWeights:
    """Map one layer's shards; F32 references cache packed experts, never all F32 experts."""
    def __init__(self, snapshot, quantization, *, kda_num_heads, dtype=None):
        from compressed_tensors.quantization import QuantizationScheme
        self.snapshot = Path(snapshot)
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.headers, self.maps = {}, {}
        self.bytes_read = 0
        groups = quantization["config_groups"]
        schemes = [QuantizationScheme.model_validate(value) for value in groups.values()]
        if (quantization["format"] != "mxfp4-pack-quantized" or len(schemes) != 1
                or schemes[0].weights.group_size != 32 or schemes[0].weights.num_bits != 4):
            raise ValueError("unsupported official Kimi expert quantization")
        self.scheme = schemes[0]
        self.native_fp32 = set()
        self.kda_num_heads = kda_num_heads
        self.narrowed_a_log = {}
        self.dtype = dtype
        self.packed_cache = {}

    def padded_a_log(self, name, shape, expected_shape):
        return (self.kda_num_heads == 96 and list(shape) == [128]
                and list(expected_shape) == [96]
                and re.fullmatch(r"language_model\.model\.layers\.\d+\.self_attn\.A_log", name))

    def narrow_a_log(self, name, value, expected_shape):
        import torch
        if (not self.padded_a_log(name, value.shape, expected_shape)
                or value.dtype != torch.float32):
            raise ValueError("unsupported native A_log padding: " + name)
        tail = value[96:].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        # The release pads unused heads with positive F32 zero, not learned values.
        if any(tail):
            raise ValueError("native A_log padding is not bit-exact zero: " + name)
        self.narrowed_a_log[name] = {
            "tensor": name, "stored_shape": [128], "loaded_shape": [96],
            "tail_shape": [32], "tail_dtype": "F32", "tail_bitwise_zero": True,
            "tail_sha256": hashlib.sha256(tail).hexdigest(),
        }
        return value[:96]

    def padding_provenance(self):
        return {"active_heads": self.kda_num_heads,
                "rule": "narrow only native F32 [128] KDA A_log to 96 configured heads after every unused tail byte equals zero",
                "layers": [self.narrowed_a_log[name] for name in sorted(self.narrowed_a_log)]}

    def descriptor(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            with (self.snapshot / shard).open("rb") as stream:
                count = struct.unpack("<Q", stream.read(8))[0]
                self.headers[shard] = json.loads(stream.read(count))
        return shard, self.headers[shard][name]

    def tensor(self, name, device="cuda:0"):
        from safetensors import safe_open
        shard, descriptor = self.descriptor(name)
        if shard not in self.maps:
            self.maps[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        raw = self.maps[shard].get_tensor(name)
        self.bytes_read += descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
        return raw.to(device=device, copy=True)

    def weight(self, name):
        import torch
        if name in self.index:
            value = self.tensor(name)
            if value.dtype == torch.float32:
                self.native_fp32.add(name)
            return value.to(dtype=self.dtype) if self.dtype is not None else value
        if not name.endswith(".weight"):
            raise ValueError("missing native parameter: " + name)
        from compressed_tensors.compressors.mxfp4 import MXFP4PackedCompressor
        prefix = name.removesuffix(".weight")
        if name in self.packed_cache:
            packed, scale = self.packed_cache[name]
        else:
            packed = self.tensor(prefix + ".weight_packed")
            scale = self.tensor(prefix + ".weight_scale")
        if self.dtype == torch.float32:
            return decompress_f32(packed, scale)
        value = MXFP4PackedCompressor.decompress(
            {"weight_packed": packed, "weight_scale": scale}, self.scheme)["weight"]
        if not torch.isfinite(value).all():
            raise ValueError("non-finite checkpoint-native MXFP4 dequantization")
        return value

    def validate_module(self, module, prefix):
        entries = []
        for name, parameter in module.named_parameters():
            full = prefix + "." + name
            packed = full not in self.index
            stored = full.removesuffix(".weight") + ".weight_packed" if packed else full
            if stored not in self.index:
                raise ValueError("missing native parameter: " + full)
            shard, descriptor = self.descriptor(stored)
            shape = list(descriptor["shape"])
            if packed:
                if len(shape) != 2 or descriptor["dtype"] != "U8":
                    raise ValueError("unsupported native packed layout: " + full)
                shape[-1] *= 2
                _, scale = self.descriptor(full.removesuffix(".weight") + ".weight_scale")
                if scale["dtype"] != "U8" or scale["shape"] != [shape[0], shape[1] // 32]:
                    raise ValueError("native MXFP4 scale layout differs: " + full)
            elif descriptor["dtype"] not in ("BF16", "F32"):
                raise ValueError("unsupported native dense dtype: " + full)
            if shape != list(parameter.shape):
                if self.padded_a_log(full, shape, parameter.shape):
                    self.narrow_a_log(full, self.tensor(full, device="cpu"), parameter.shape)
                else:
                    raise ValueError(f"native tensor shape differs: {full}; checkpoint {shape}, official model {list(parameter.shape)}")
            entries.append((shard, descriptor["data_offsets"][0], full, name, parameter))
        return sorted(entries, key=lambda entry: entry[:2])

    def preload_experts(self, entries):
        self.packed_cache.clear()
        for _, _, full, _, _ in entries:
            if ".experts." in full:
                prefix = full.removesuffix(".weight")
                self.packed_cache[full] = (self.tensor(prefix + ".weight_packed"),
                                           self.tensor(prefix + ".weight_scale"))
        self.close()

    def release_experts(self):
        self.packed_cache.clear()

    def stream_parameter(self, module, leaf, full, shape):
        import torch
        def load(owner, args):
            value = self.weight(full)
            if value.dtype != torch.float32 or tuple(value.shape) != tuple(shape):
                raise ValueError("streamed F32 expert parameter differs: " + full)
            owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        def release(owner, args, output):
            owner._parameters[leaf] = torch.nn.Parameter(
                torch.empty(shape, device="meta", dtype=torch.float32), requires_grad=False)
        module.register_forward_pre_hook(load)
        module.register_forward_hook(release, always_call=True)

    def load_module(self, module, prefix, *, stream_experts=False):
        import torch
        entries = self.validate_module(module, prefix)
        if stream_experts:
            self.preload_experts(entries)
        # Expert matrices live only during their own unchanged official Linear call.
        for _, _, full, name, parameter in entries:
            if stream_experts and ".experts." in full:
                parent, _, leaf = name.rpartition(".")
                self.stream_parameter(module.get_submodule(parent), leaf, full, parameter.shape)
                continue
            value = self.weight(full)
            if tuple(value.shape) != tuple(parameter.shape):
                if self.padded_a_log(full, value.shape, parameter.shape):
                    value = self.narrow_a_log(full, value, parameter.shape)
                else:
                    raise ValueError("native tensor shape differs: " + full)
            parent, _, leaf = name.rpartition(".")
            owner = module.get_submodule(parent) if parent else module
            owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        self.close()
        return module.eval()

    def close(self):
        self.maps.clear()


def causal_eager(reference):
    """Unchanged official eager math, with fixed query/key geometry per causal tile."""
    import torch
    import torch.nn.functional as F
    from shape_invariant import ROWS
    official = reference.eager_attention_forward

    def forward(module, query, key, value, attention_mask=None, scaling=1.0, dropout=0.0, **kwargs):
        if module.training or dropout or attention_mask is not None:
            raise ValueError("Kimi bounded eager expects unpadded, cache-free inference")
        length = query.shape[-2]
        outputs = []
        for start in range(0, length, ROWS):
            count = min(ROWS, length - start)
            extent = start + ROWS
            q = F.pad(query[..., start:start + count, :], (0, 0, 0, ROWS - count))
            k = F.pad(key[..., :extent, :], (0, 0, 0, max(0, extent - length)))
            v = F.pad(value[..., :extent, :], (0, 0, 0, max(0, extent - length)))
            qpos = torch.arange(start, extent, device=query.device)[:, None]
            kpos = torch.arange(extent, device=query.device)[None, :]
            mask = torch.zeros((ROWS, extent), dtype=query.dtype, device=query.device)
            mask.masked_fill_((kpos > qpos) | (kpos >= length), float("-inf"))
            result, _ = official(module, q, k, v, mask[None, None], scaling, dropout=0.0, **kwargs)
            outputs.append(result[:, :count])
        return torch.cat(outputs, dim=1), None
    return forward


@contextmanager
def geometry(reference):
    import torch
    import torch.nn.functional as F
    import shape_invariant
    previous = F.linear, torch.einsum, reference.eager_attention_forward
    F.linear = shape_invariant.linear
    torch.einsum = shape_invariant.einsum
    reference.eager_attention_forward = causal_eager(reference)
    try:
        yield
    finally:
        F.linear, torch.einsum, reference.eager_attention_forward = previous
