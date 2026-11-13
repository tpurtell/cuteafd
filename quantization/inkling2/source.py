"""Header-only access to the pinned native Inkling 2 checkpoint, including MTP."""
from __future__ import annotations

import json
from pathlib import Path
import re

from kimi_k3.uniform_k2.storage import file_identity, read_header, sha256_file
from . import recipe


class HeaderSource:
    """Inventory source names without allocating tensors or importing Torch.

    The source's main index deliberately excludes its standalone drafter. The
    published inventory adds only that reviewed extra shard with ``mtp.`` names.
    """

    def __init__(self, snapshot, *, verify=True):
        self.snapshot = Path(snapshot).expanduser().absolute()
        self.document = json.loads((self.snapshot / "config.json").read_text())
        self.config = self.document["text_config"]
        self.quantization_config = json.loads((self.snapshot / "hf_quant_config.json").read_text())
        index_path = self.snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        if verify and sha256_file(index_path) != recipe.SOURCE_INDEX_SHA256:
            raise ValueError("pinned source index changed")
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("source index requires a nonempty weight_map")
        self.index = index
        self.weight_map, self.locations, self.specs, self.sizes = {}, {}, {}, {}
        self.raw_headers, self.shard_offsets, self.source_fingerprints = {}, {}, {}
        by_shard = {}
        for name, shard in mapping.items():
            if (not isinstance(name, str) or not isinstance(shard, str)
                    or not re.fullmatch(r"model-[0-9]{5}-of-00048\.safetensors", shard)):
                raise ValueError("unsafe or unreviewed main source shard")
            by_shard.setdefault(shard, set()).add(name)
        expected = {f"model-{n:05d}-of-00048.safetensors" for n in range(1, 49)}
        if verify and set(by_shard) != expected:
            raise ValueError("main source shard inventory differs")
        for shard, names in sorted(by_shard.items()):
            self._add_shard(shard, "", expected=names)
        declared = index.get("metadata", {}).get("total_size")
        measured = sum(self.sizes.values())
        if declared != measured:
            raise ValueError("main source index payload total differs from headers")
        self._add_shard("mtp.safetensors", "mtp.")
        self.payload_bytes = sum(self.sizes.values())
        self.tensor_count = len(self.specs)
        self.payload_totals = dict(declared=declared, measured=self.payload_bytes,
                                  reviewed=recipe.BASE_PAYLOAD_BYTES)
        if verify:
            actual = {p.name for p in self.snapshot.glob("*.safetensors")}
            if actual != expected | {"mtp.safetensors"}:
                raise ValueError("unreviewed extra source tensor shard")
            self.audit = recipe.inventory(self)

    def _add_shard(self, shard, prefix, *, expected=None):
        path = self.snapshot / shard
        header, offset = read_header(path)
        names = set(header) - {"__metadata__"}
        if expected is not None and names != expected:
            raise ValueError("source shard header and index disagree: " + shard)
        self.shard_offsets[shard] = offset
        self.source_fingerprints[shard] = file_identity(path)
        for raw_name in sorted(names):
            name, spec = prefix + raw_name, header[raw_name]
            if name in self.specs or (prefix and raw_name.startswith("mtp.")):
                raise ValueError("duplicate or already prefixed source tensor: " + name)
            self.weight_map[name] = shard
            self.locations[name] = dict(shard=shard, source_name=raw_name)
            self.specs[name] = (spec["dtype"], spec["shape"])
            self.sizes[name] = spec["data_offsets"][1] - spec["data_offsets"][0]
            self.raw_headers[name] = spec

    def as_inventory(self):
        return dict(mapping=self.weight_map, locations=self.locations, specs=self.specs,
                    sizes=self.sizes, payload_bytes=self.payload_bytes,
                    tensor_count=self.tensor_count, payload_totals=self.payload_totals)

    def close(self):
        """The header reader retains no mappings or open file handles."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


Inkling2Source = HeaderSource


def header_source(snapshot):
    return HeaderSource(snapshot)
