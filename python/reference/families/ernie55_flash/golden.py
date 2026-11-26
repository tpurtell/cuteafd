#!/usr/bin/env python3
"""Layer-major ERNIE 5.5 Flash F32 golden; --check-only validates all 70,826 + 12 tensors."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
import layer_major
from families import hy4_common
from families.ernie55_flash import model, sources

REL_LIMIT = 1e-4


def expected_source_identity(_runtime=None):
    return {
        "checkpoint": sources.CHECKPOINT,
        "revision": sources.REVISION,
        "vllm_ernie_revision": sources.VLLM_ERNIE_REVISION,
        "vllm_version": sources.VLLM_VERSION,
        "vllm_ernie_files": sources.VLLM_ERNIE_FILES,
        "sglang_revision": sources.SGLANG_REVISION,
        "sglang_files": sources.SGLANG_FILES,
        "config_sha256": sources.CONFIG_SHA256,
        "index_sha256": sources.INDEX_SHA256,
        "flashindexer_sha256": sources.FLASHINDEXER_SHA256,
        "common_sha256": layer_major.digest(hy4_common.__file__),
        "model_sha256": layer_major.digest(model.__file__),
        "golden_sha256": layer_major.digest(__file__),
        "sources_sha256": layer_major.digest(sources.__file__),
        "layer_major_sha256": layer_major.digest(layer_major.__file__),
    }


class Family:
    name = "ernie55_flash"
    compare_official_head = True

    def __init__(self, snapshot, device, *, enable_flashindexer: bool = False):
        snapshot = Path(snapshot)
        if (
            snapshot.name != sources.REVISION
            or layer_major.digest(snapshot / "config.json") != sources.CONFIG_SHA256
            or layer_major.digest(snapshot / "model.safetensors.index.json") != sources.INDEX_SHA256
            or layer_major.digest(snapshot / model.FLASHINDEXER_FILE) != sources.FLASHINDEXER_SHA256
        ):
            raise ValueError("ERNIE 5.5 Flash oracle requires the pinned original BF16 snapshot")
        self.raw_config = json.loads((snapshot / "config.json").read_bytes())
        self.config = model.Config.parse(self.raw_config)
        self.layers = self.config.num_hidden_layers
        self.checkpoint = model.Checkpoint(snapshot, device)
        self.device = self.checkpoint.device
        self.enable_flashindexer = enable_flashindexer
        self.outer = None
        self.upstream_preflight = None

    configure = staticmethod(model.configure_f32)
    provenance = staticmethod(expected_source_identity)
    observe = staticmethod(layer_major.no_observation)

    def tensors(self, index):
        return model.tensor_contract(self.config, index)

    def check_layer(self, index):
        self.checkpoint.check(self.tensors(index))

    def check_all(self):
        outer = model.tensor_contract(self.config)
        known = set(outer)
        self.checkpoint.check(outer)
        for index in range(self.layers):
            self.check_layer(index)
            known.update(self.tensors(index))
        mtp_aux = model.mtp_contract(self.config)
        self.checkpoint.check(mtp_aux)
        known.update(mtp_aux)
        for stage in range(self.config.num_mtp_stages):
            stage_contract = model.mtp_contract(self.config, stage)
            self.checkpoint.check(stage_contract)
            known.update(stage_contract)
        indexed = set(self.checkpoint.index)
        if known != indexed:
            raise ValueError(f"unexpected ERNIE 5.5 tensor inventory: {sorted(indexed ^ known)[:8]}")
        fi_contract = model.flashindexer_contract(self.config)
        self.checkpoint.check_flashindexer(fi_contract)
        return len(known), len(fi_contract)

    def layer_bytes(self, index):
        # Non-expert weights in F32 (4 B) + routed experts packed in BF16 (2 B, streamed 1 expert to F32).
        total = 0
        for name, (shape, _dtype) in self.tensors(index).items():
            is_expert = ".experts." in name
            total += math.prod(shape) * (2 if is_expert else 4)
        return total

    def state_bytes(self, tokens):
        return tokens * self.config.hidden_size * self.config.hc_mult * 4

    def work_bytes(self, tokens):
        c = self.config
        heads = max(c.num_attention_heads, c.hybrid_mla_num_attention_heads)
        width = max(c.head_dim, c.hybrid_mla_qk_nope_head_dim + c.hybrid_mla_qk_rope_head_dim, c.hybrid_mla_v_head_dim)
        return 4 * (
            tokens * (heads * width * 3 + c.hc_mult * c.hidden_size * 4)
            + heads * model.ROWS * tokens * 2
            + c.index_n_heads * model.ROWS * tokens
        )

    def begin_layer_memory(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats(self.device)

    def end_layer_memory(self, stage, index):
        if self.device.type != "cuda":
            return
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        gib = 2 ** 30
        print(
            f"ERNIE55 {stage} layer {index}: CUDA peak allocated={allocated / gib:.3f} GiB "
            f"reserved={reserved / gib:.3f} GiB MemAvailable={available / gib:.3f} GiB",
            flush=True,
        )
        if available < 12 * gib:
            raise RuntimeError(
                f"ERNIE55 MemAvailable guard stopped at {stage} layer {index}; no next layer launched"
            )

    def outer_tensors(self, stage):
        return model.OUTER_NAMES[:1] if stage == "embed" else model.OUTER_NAMES[1:]

    def embed(self, tokens):
        import torch
        import torch.nn.functional as F
        if self.outer is None:
            self.outer = self.checkpoint.load(model.OUTER_NAMES[0])
            self.checkpoint.close()
        # Allocate token index bank on device (never pinned memory for gathers on GB10).
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        embedding = F.embedding(ids, self.outer)
        return embedding[:, None].expand(-1, self.config.hc_mult, -1).unsqueeze(0).contiguous()

    def release_outer(self):
        self.outer = None

    def load_layer(self, index):
        return {
            "weights": self.checkpoint.layer(
                self.config,
                index,
                load_experts=True,
                experts_on_device=(self.device.type == "cuda"),
                load_flashindexer=self.enable_flashindexer,
            )
        }

    def release_layer(self, module):
        module.clear()
        self.checkpoint.close()

    def geometry(self):
        return nullcontext()

    def official_geometry(self):
        import torch.nn.functional as F

        @contextmanager
        def untiled():
            prev = model.linear
            model.linear = F.linear
            try:
                yield
            finally:
                model.linear = prev

        return untiled()

    def forward(self, module, index, hidden, carry):
        result = model.layer_forward(
            hidden[0],
            module["weights"],
            self.config,
            index,
            enable_flashindexer=self.enable_flashindexer,
        ).unsqueeze(0)
        return result, carry, tuple(carry)

    def load_final(self):
        self.outer = {name: self.checkpoint.load(name) for name in self.outer_tensors("final")}
        self.checkpoint.close()

    def release_final(self):
        self.release_outer()

    def final(self, hidden):
        return model.lm_head_logits(hidden, self.outer, self.config, to_cpu=True)

    def official_final(self, hidden):
        with self.official_geometry():
            return model.lm_head_logits(hidden, self.outer, self.config, to_cpu=True)

    def golden_meta(self):
        gates = {"gates": {"upstream_preflight": self.upstream_preflight}} if self.upstream_preflight else {}
        return {
            **gates,
            "reference": "Independent ERNIE 5.5 Flash layer-major IEEE F32 text reference",
            "source_identity_provider": "families.ernie55_flash.golden",
            "root_arithmetic": sources.ROOT_ARITHMETIC,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--enable-flashindexer", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path, help="recorded vLLM-ERNIE preflight JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--pilot-layers", type=int, choices=(2,))
    a = parser.parse_args()

    family = Family(
        a.snapshot,
        "cpu" if a.check_only else a.device,
        enable_flashindexer=a.enable_flashindexer,
    )
    checked, fi_checked = family.check_all()
    print(
        f"Native ERNIE 5.5 Flash headers: {checked} indexed tensors + {fi_checked} flashindexer tensors; "
        f"vLLM-ERNIE {sources.VLLM_ERNIE_REVISION[:12]} / SGLang {sources.SGLANG_REVISION[:12]}",
        flush=True,
    )
    if a.check_only:
        return
    if a.out is None or a.windows is None:
        parser.error("--out and --windows are required")
    if a.preflight_receipt is not None:
        family.upstream_preflight = json.loads(a.preflight_receipt.read_bytes())

    import torch
    a.out.mkdir(parents=True, exist_ok=True)
    family.configure()
    layer_major.DEVICE = a.device
    if a.pilot_layers is not None:
        import resource
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timing = layer_major.run(a, family)
        peak = Path("/sys/fs/cgroup/memory.peak")
        record = {
            "layers": timing,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cgroup_peak_bytes": int(peak.read_text()) if peak.is_file() else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if a.device == "cuda" else 0,
        }
        layer_major.atomic_json(a.out / "pilot.json", record)
        print(json.dumps(record), flush=True)
        return
    layer_major.run(a, family)


if __name__ == "__main__":
    main()
