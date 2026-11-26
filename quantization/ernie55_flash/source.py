"""Reviewed ERNIE-5.5-Flash BF16 source and direct-MC tensor roles."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from kimi_k3.uniform_k2.storage import canonical, file_identity, read_header

BASE_REPO = "baidu/ERNIE-5.5-Flash-Preview-BF16"
BASE_REVISION = "33a9925b17506c5ff57accac2b31a6520c951c4e"
FLASHINDEXER_FILE = "flashindexer/flashindexer.safetensors"
DSA_LAYERS = (7, 14, 21, 28, 35, 42)

# Canonical {path: [Git blob or LFS OID, size]} from the pinned Hub tree.
SOURCE_TREE_SHA256 = "6db616f766058f24aaf1f67aa763098b308d1ca9a8e19b8aa8730e80c01ca037"
# Canonical {shard: {tensor: safetensors spec}} includes dtype, shape and offsets.
HEADERS_SHA256 = "8b79a8bfc97f8fbaa0ff394fbfa88ca4bc058d65b8deede56d701ef2f993c803"
SIDECARS = (
    ".gitattributes", "LICENSE", "README.md", "assets/logo.png", "assets/eval-results.png",
    "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "tokenization_ernie.py", "encoding_ernie.py", "decoding_ernie.py",
)


def _sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def reviewed_roles():
    """Enumerate source tensor IDs. New and missing names both fail the audit."""
    roles = {}

    def add(prefix, role, names):
        for name in names.split():
            key = prefix + name
            if key in roles:
                raise ValueError(f"duplicate reviewed tensor: {key}")
            roles[key] = role

    add("", "native", "lm_head.weight model.embed_tokens.weight model.norm.weight "
        "model.lm_head.multimax_ranges model.lm_head.multimax_ts")
    for prefix in ("model.", "model.mtp."):
        add(prefix + "mhc_contract.", "native", "hc_head_base hc_head_fn hc_head_scale")
    add("model.mtp.", "native", "markov_head.markov_w1.weight markov_head.markov_w2.weight "
        "confidence_head.projection.weight feature_norm.weight norm.weight")
    add("model.mtp.", "core", "feature_projection.weight")

    for family, count in (("layers", 43), ("mtp", 3)):
        for layer in range(count):
            p = f"model.{family}.{layer}."
            a = p + "self_attn."
            add(p, "native", "input_layernorm.weight post_attention_layernorm.weight")
            for side in ("self_attention", "mlp"):
                add(p + side + "_hyper_connection.", "native",
                    "alpha_pre alpha_post alpha_res mapping_proj.weight bias")
            add(a, "native", "vha_postmix_U vha_postmix_V")
            add(a, "core", "gate_proj.weight o_proj.weight")
            if family == "layers" and layer in DSA_LAYERS:
                add(a, "core", "q_a_proj.weight q_b_proj.weight kv_a_proj_with_mqa.weight")
                add(a, "native", "kv_b_proj.weight q_a_layernorm.weight kv_a_layernorm.weight "
                    "core_attention.softmax_offset")
                indexer = a + "core_attention.indexer."
                add(indexer, "native", "wq_b.weight wk.weight weights_proj.weight k_norm.bias k_norm.weight")
                add(indexer, "flashindexer", "flashindexer_bias linear_gate.weight")
            else:
                add(a, "core", "linear_q_down_proj.weight linear_q_up_proj.weight linear_kv_proj.weight "
                    "linear_o_group_proj")
                add(a, "native", "q_layernorm.weight kv_layernorm.weight core_attention.attn_sink")
                if family == "layers":
                    cp = a + "core_attention.compressor."
                    add(cp, "native", "ape norm.weight")
                    add(cp, "core", "linear_wkv.weight linear_wgate.weight")
            if family == "layers" and layer == 0:
                add(p + "mlp.", "core", "gate_proj.weight up_proj.weight down_proj.weight")
                continue
            moe = p + "block_sparse_moe."
            add(moe, "core", "fc1_latent_proj.weight fc2_latent_proj.weight shared_experts.w1.weight "
                "shared_experts.w2.weight shared_experts.w3.weight")
            add(moe, "native", "latent_norm.weight e_score_correction_bias gate.routed_scaling_factor_param "
                "gate.weight gate.weight_1")
            for expert in range(512):
                add(moe + f"experts.{expert}.", "routed_expert", "w1.weight w2.weight w3.weight")
    return roles


def _require_names(actual, expected, context):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f"{context} differs: missing {sorted(missing)[:4]}, extra {sorted(extra)[:4]}")


def _pinned_files(snapshot, shards):
    expected = set(SIDECARS) | set(shards)
    blobs = snapshot.parent.parent / "blobs"
    files = {
        p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*")
        if not p.is_dir() and not (p.parent.name == "__pycache__" and p.suffix == ".pyc")
    }
    _require_names(files, expected, "source file manifest")
    manifest = {}
    for name in sorted(expected):
        path = snapshot / name
        alias = path.readlink()
        if alias.is_absolute() or (path.parent / alias).resolve() != (blobs / alias.name).resolve():
            raise ValueError(f"source alias outside pinned cache: {name}")
        manifest[name] = (alias.name, path.stat().st_size)
    if _sha(manifest) != SOURCE_TREE_SHA256:
        raise ValueError("source Git/LFS object names or file sizes differ from pinned Hub tree")
    hashes, documents = {}, {}
    for name in SIDECARS:
        oid, size = manifest[name]
        content = (snapshot / name).read_bytes()
        if len(oid) == 40:
            digest = hashlib.sha1(f"blob {size}\0".encode() + content).hexdigest()
        else:
            digest = hashlib.sha256(content).hexdigest()
        if digest != oid:
            raise ValueError(f"source sidecar content differs: {name}")
        hashes[name] = hashlib.sha256(content).hexdigest()
        if name in ("config.json", "model.safetensors.index.json"):
            documents[name] = json.loads(content)
    return hashes, documents


class HeaderSource:
    """Header-only inventory; the unindexed flashindexer belongs to every MC view."""

    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).expanduser().resolve(strict=True)
        if self.snapshot.name != BASE_REVISION:
            raise ValueError("source snapshot differs from the pinned ERNIE revision")
        main_shards = [f"model-{i:05d}-of-00074.safetensors" for i in range(1, 75)]
        shards = main_shards + [FLASHINDEXER_FILE]
        sidecars, documents = _pinned_files(self.snapshot, shards)
        self.document = self.config = documents["config.json"]
        self.index = documents["model.safetensors.index.json"]
        self.roles = reviewed_roles()
        flash = {name for name, role in self.roles.items() if role == "flashindexer"}
        main_map = self.index["weight_map"]
        _require_names(main_map, self.roles.keys() - flash, "indexed tensor names")
        _require_names(main_map.values(), main_shards, "indexed shards")
        self.weight_map = {**main_map, **dict.fromkeys(flash, FLASHINDEXER_FILE)}
        by_shard = defaultdict(set)
        for name, shard in self.weight_map.items():
            by_shard[shard].add(name)
        self.specs, self.sizes, self.raw_headers, self.locations = {}, {}, {}, {}
        self.shard_offsets, self.source_fingerprints, reviewed_headers = {}, {}, {}
        for shard in shards:
            path = self.snapshot / shard
            identity = file_identity(path)
            header, offset = read_header(path)
            if identity != file_identity(path):
                raise ValueError(f"source changed while reading {shard}")
            header.pop("__metadata__", None)
            _require_names(header, by_shard[shard], shard)
            reviewed_headers[shard] = header
            self.shard_offsets[shard] = offset
            self.source_fingerprints[shard] = identity
            for name, spec in header.items():
                self.locations[name] = dict(shard=shard, source_name=name)
                self.specs[name] = (spec["dtype"], spec["shape"])
                self.raw_headers[name] = spec
                lo, hi = spec["data_offsets"]
                self.sizes[name] = hi - lo
        header_hash = _sha(reviewed_headers)
        if header_hash != HEADERS_SHA256:
            raise ValueError("reviewed source tensor dtype, shape or byte range differs")
        main_bytes = sum(self.sizes[name] for name in main_map)
        if main_bytes != self.index["metadata"]["total_size"]:
            raise ValueError("indexed payload bytes differ from index metadata")
        self.payload_bytes = sum(self.sizes.values())
        self.tensor_count = len(self.specs)
        groups, counts = Counter(), Counter()
        for name, role in self.roles.items():
            group = ("dspark" if name.startswith("model.mtp.") else "target") + "_" + role
            groups[group] += self.sizes[name]
            counts[group] += 1
        self.audit = dict(
            source_repo=BASE_REPO, source_revision=BASE_REVISION,
            source_tensors=self.tensor_count, source_bytes=self.payload_bytes,
            indexed_tensors=len(main_map), indexed_bytes=main_bytes,
            source_files_bytes=sum(stat["bytes"] for stat in self.source_fingerprints.values()),
            source_group_bytes=dict(groups), source_group_tensors=dict(counts),
            source_dtypes=dict(Counter(dtype for dtype, _ in self.specs.values())),
            flashindexer={name: self.raw_headers[name] for name in sorted(flash)},
            sidecars_sha256=sidecars, tree_sha256=SOURCE_TREE_SHA256, headers_sha256=header_hash,
            tensor_roles_sha256=_sha(self.roles),
        )

    def source_entries(self, names):
        return {name: (self.snapshot / self.weight_map[name], name) for name in names}

    def as_inventory(self):
        return dict(mapping=self.weight_map, locations=self.locations, specs=self.specs,
                    sizes=self.sizes, roles=self.roles, payload_bytes=self.payload_bytes,
                    tensor_count=self.tensor_count)
