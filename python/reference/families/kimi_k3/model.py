"""Hash-checked Kimi K3 text decoder; no serving implementation or numeric fallback."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
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


class LayerWeights:
    """Map shards only while loading one layer; preserve each stored dense dtype."""
    def __init__(self, snapshot, quantization):
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
            return value
        if not name.endswith(".weight"):
            raise ValueError("missing native parameter: " + name)
        from compressed_tensors.compressors.mxfp4 import MXFP4PackedCompressor
        prefix = name.removesuffix(".weight")
        packed = self.tensor(prefix + ".weight_packed")
        scale = self.tensor(prefix + ".weight_scale")
        value = MXFP4PackedCompressor.decompress(
            {"weight_packed": packed, "weight_scale": scale}, self.scheme)["weight"]
        if not torch.isfinite(value).all():
            raise ValueError("non-finite checkpoint-native MXFP4 dequantization")
        return value

    def load_module(self, module, prefix):
        import torch
        entries = []
        for name, parameter in module.named_parameters():
            full = prefix + "." + name
            stored = full if full in self.index else full.removesuffix(".weight") + ".weight_packed"
            shard, descriptor = self.descriptor(stored)
            entries.append((shard, descriptor["data_offsets"][0], full, name, parameter))
        # Native parameters are materialized once, not as an extra whole-layer copy.
        for _, _, full, name, parameter in sorted(entries, key=lambda entry: entry[:2]):
            value = self.weight(full)
            if tuple(value.shape) != tuple(parameter.shape):
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
