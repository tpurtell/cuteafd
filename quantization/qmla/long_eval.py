"""Long-context (16K/32K) QMLA evaluation over full-length ``panel_corpus`` v3 records."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import resource
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from .capture import CapturedLayerParams, CapturedSequence, MlaLayerGeometry, ROWS, eval_attention_output
from .fit import DEFAULT_PROTECTION_POLICY, FitConfig
from .measurement import NATIVE_ROPE, _Scorer, _token_relative_squares, combo_key, layer_quantizes

LONG_EVAL_SCHEMA = "cuteafd-qmla-long-eval-v1"
LONG_CONTEXT_EVAL_SPLITS: tuple[str, ...] = ("heldout", "calibration")
DEFAULT_POSITION_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0-2k", 0, 2048), ("2k-4k", 2048, 4096), ("4k-8k", 4096, 8192),
    ("8k-16k", 8192, 16384), ("16k-32k", 16384, 32768), ("32k+", 32768, None),
)


def _panel_corpus():
    quant_root = str(Path(__file__).resolve().parents[1])
    if quant_root not in sys.path:
        sys.path.insert(0, quant_root)
    import panel_corpus
    return panel_corpus


def position_bucket_masks(
    positions: torch.Tensor,
    buckets: Sequence[tuple[str, int, int | None]] = DEFAULT_POSITION_BUCKETS,
) -> dict[str, torch.Tensor]:
    """Boolean [T] mask per query-position bucket (axis 0 for 3D M-RoPE [3, T])."""
    pos = positions[0] if positions.ndim == 2 else positions
    if pos.ndim != 1:
        raise ValueError(f"positions must be [T] or [3, T], got {tuple(positions.shape)}")
    return {name: (pos >= lo) if hi is None else ((pos >= lo) & (pos < hi)) for name, lo, hi in buckets}


@dataclass(frozen=True)
class LongContextRecord:
    seq_id: str
    split: str
    length_class: str
    input_ids: torch.Tensor
    input_ids_sha256: str


def _summarize_records(records: Sequence[LongContextRecord], *, source_kind: str, source_path: Path,
                       dataset_repo: str | None, dataset_revision: str | None) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    by_class: dict[str, dict[str, int]] = {}
    rows = []
    for r in records:
        n = int(r.input_ids.numel())
        for tbl, k in ((by_split, r.split), (by_class, r.length_class)):
            slot = tbl.setdefault(k, {"sequences": 0, "tokens": 0})
            slot["sequences"] += 1
            slot["tokens"] += n
        rows.append({"id": r.seq_id, "split": r.split, "length_class": r.length_class, "tokens": n, "input_ids_sha256": r.input_ids_sha256})
    return {
        "source_kind": source_kind, "source_path": str(source_path), "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision, "sequences": len(records),
        "tokens": sum(int(r.input_ids.numel()) for r in records),
        "splits": by_split, "length_classes": by_class, "records": rows,
    }


def panel_snapshot_dir(panel_revision: str | None = None) -> Path:
    """Local HF-cache snapshot of the pinned panel (default v3); never downloads."""
    from huggingface_hub import snapshot_download

    repo, rev, _ = _panel_corpus().resolve_panel_policy(
        panel_revision=panel_revision, components=("long_context",), full=False)
    return Path(snapshot_download(repo, revision=rev, repo_type="dataset", local_files_only=True))


def load_long_context_records(
    panel_dir: Path | str, *, checkpoint_dir: Path | str | None = None, tokenizer: Any = None,
    chat_kwargs: Mapping[str, Any] | None = None, panel_revision: str | None = None,
    splits: Sequence[str] = LONG_CONTEXT_EVAL_SPLITS,
) -> tuple[list[LongContextRecord], dict[str, Any]]:
    """Render full-length 16K/32K ``long_context`` records from an attested panel snapshot.

    ``panel_corpus.read_panel`` verifies the snapshot against its pin; each record is
    rendered once with the checkpoint's own template (``long_context_mode="full"``).
    """
    pc, root = _panel_corpus(), Path(panel_dir).expanduser().resolve()
    panel = pc.read_panel(root, panel_revision=panel_revision, components=("long_context",), full=False)

    max_ctx = pc.max_context_from_config(Path(checkpoint_dir).resolve()) if checkpoint_dir is not None else None
    if tokenizer is None:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir or tokenizer is required to render long_context records")
        from transformers import PreTrainedTokenizerFast
        ckpt = Path(checkpoint_dir).resolve()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(ckpt, local_files_only=True)
        tpl = ckpt / "chat_template.jinja"
        if tpl.is_file() and (not isinstance(tokenizer.chat_template, str)
                              or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != pc.sha256_file(tpl)):
            raise ValueError(f"native tokenizer at {ckpt} did not load pinned chat_template.jinja")

    records: list[LongContextRecord] = []
    for sp in splits:
        rows = panel["long_context_rows"].get(sp) or []
        if not rows:
            continue
        flat, offsets, report = pc.tokenize_long_context_rows(
            rows, tokenizer, chat_kwargs=chat_kwargs, max_context=max_ctx, mode="full")
        for entry, a, b in zip(report["records"], offsets[:-1], offsets[1:], strict=True):
            records.append(LongContextRecord(str(entry["id"]), sp, str(entry["length_class"]),
                                             torch.tensor(flat[a:b].tolist(), dtype=torch.int64),
                                             str(entry["input_ids_sha256"])))
    if not records:
        raise ValueError(f"panel at {root} has no long_context records in {tuple(splits)}")
    ds = panel["dataset"]
    return records, _summarize_records(records, source_kind="panel_corpus_full", source_path=root,
                                       dataset_repo=ds["repo_id"], dataset_revision=ds["revision"])


@dataclass
class _BucketAcc:
    tokens: int = 0
    shipped: float = 0.0
    rtn: float = 0.0
    hadamard: float = 0.0

    def add(self, tokens: int, shipped: float, rtn: float, hadamard: float) -> None:
        self.tokens += tokens
        self.shipped += shipped
        self.rtn += rtn
        self.hadamard += hadamard

    def summary(self) -> dict[str, Any]:
        s, r, h = self.shipped / self.tokens, self.rtn / self.tokens, self.hadamard / self.tokens
        return {
            "tokens": self.tokens, "attn_rel_err": s, "rtn_attn_rel_err": r, "hadamard_attn_rel_err": h,
            "ratio_to_rtn": s / r if r > 0.0 else None, "hadamard_ratio_to_rtn": h / r if r > 0.0 else None,
        }


class LongContextEvaluator:
    """Streaming per-layer long-context evaluator reusing ``measurement._Scorer``."""

    def __init__(
        self, *, modes: Sequence[tuple[int, int]],
        content_transforms: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        rope_transforms: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        protection_policy: str = DEFAULT_PROTECTION_POLICY, config: FitConfig | None = None,
        dsa_indexer_note: str = "", buckets: Sequence[tuple[str, int, int | None]] = DEFAULT_POSITION_BUCKETS,
    ) -> None:
        self.modes = tuple(modes)
        self.content_arms = {lid: SimpleNamespace(r_c=r, s_c=s) for lid, (r, s) in content_transforms.items()}
        self.rope_arms = {lid: SimpleNamespace(theta=t, s_p=s) for lid, (t, s) in rope_transforms.items()}
        self.config = config if config is not None else FitConfig(protection=protection_policy)
        self.dsa_indexer_note = dsa_indexer_note or (
            "DSA compressed indexer reads only layer input x and q_norm (indexer keys stay native); "
            "quantized MLA latents do not feed it, so per-layer top-k selection drift is structurally zero "
            "(cross-layer residual propagation is out of scope)."
        )
        self.buckets = tuple(buckets)
        self._layer_stats: dict[str, dict[int, dict[str, _BucketAcc]]] = {}
        self._split_stats: dict[str, dict[str, dict[str, _BucketAcc]]] = {}
        self._sparse_layers: set[int] = set()
        self._layers_seen: set[int] = set()
        self._scorer_layer_id: int | None = None
        self._scorer: _Scorer | None = None

    def release_layer_cache(self) -> None:
        self._scorer_layer_id = None
        self._scorer = None

    @torch.no_grad()
    def evaluate_layer_sequence(
        self, seq: CapturedSequence, params: CapturedLayerParams, geometry: MlaLayerGeometry,
        *, split: str, device: torch.device | str,
    ) -> None:
        dev, lid = torch.device(device), geometry.layer_id
        self._layers_seen.add(lid)
        if geometry.kind == "dsa":
            self._sparse_layers.add(lid)
        if self._scorer_layer_id != lid or self._scorer is None:
            self._scorer = _Scorer(None, None, self.config, dev, geom=geometry, params=params, seqs=())
            self._scorer_layer_id = lid

        scorer, seq_dev = self._scorer, seq.to(dev)
        bmasks = position_bucket_masks(seq_dev.positions, self.buckets)
        for mode in self.modes:
            if not layer_quantizes(geometry, mode):
                continue
            cb, rb = mode
            mkey = combo_key(mode)

            def _run(c_arm, r_arm) -> torch.Tensor:
                hooks = {**scorer.content_hooks(cb, c_arm), **scorer.rope_hooks(rb, r_arm), **scorer.protection}
                err = _token_relative_squares(eval_attention_output(seq_dev, scorer.params, geometry, **hooks), seq_dev.ref_output).to(torch.float64)
                if not bool(torch.isfinite(err).all()):
                    raise ValueError(f"non-finite long-eval error at layer {lid} mode {mkey} seq {seq.seq_id}")
                return err

            s_err = _run(("shipped", self.content_arms[lid]) if cb else "rtn",
                         self.rope_arms[lid] if (rb and geometry.has_rope) else NATIVE_ROPE)
            r_err = _run("rtn", NATIVE_ROPE)
            h_err = _run("hadamard", NATIVE_ROPE) if cb else r_err

            lb = self._layer_stats.setdefault(mkey, {}).setdefault(lid, {})
            sb = self._split_stats.setdefault(mkey, {}).setdefault(split, {})
            for bname, mask in bmasks.items():
                n = int(mask.sum().item())
                if n:
                    vals = (n, float(s_err[mask].sum().item()), float(r_err[mask].sum().item()), float(h_err[mask].sum().item()))
                    for tgt in (lb.setdefault(bname, _BucketAcc()), lb.setdefault("__all__", _BucketAcc()),
                                sb.setdefault(bname, _BucketAcc()), sb.setdefault("__all__", _BucketAcc())):
                        tgt.add(*vals)

    def finalize(self, *, corpus_meta: Mapping[str, Any], transform_fingerprint: str,
                 wall_seconds: float, device: torch.device | str) -> dict[str, Any]:
        names = [n for n, _, _ in self.buckets]
        combinations: dict[str, Any] = {}
        for mkey, per_layer_raw in self._layer_stats.items():
            overall, overall_b, per_layer_out = _BucketAcc(), {}, {}
            for lid in sorted(per_layer_raw):
                lb = per_layer_raw[lid]
                a = lb["__all__"]
                overall.add(a.tokens, a.shipped, a.rtn, a.hadamard)
                row = {**a.summary(), "by_position_bucket": {}}
                for bname in names:
                    if bname in lb:
                        b = lb[bname]
                        row["by_position_bucket"][bname] = b.summary()
                        overall_b.setdefault(bname, _BucketAcc()).add(b.tokens, b.shipped, b.rtn, b.hadamard)
                per_layer_out[str(lid)] = row

            by_split_out = {
                sp: {**{k: v for k, v in sb["__all__"].summary().items() if k != "tokens"},
                     "layer_tokens": sb["__all__"].tokens,
                     "by_position_bucket": {b: sb[b].summary() for b in names if b in sb}}
                for sp, sb in self._split_stats.get(mkey, {}).items() if "__all__" in sb
            }
            summary = overall.summary()
            summary["layer_tokens"] = summary.pop("tokens")
            summary.update(
                layers=len(per_layer_raw),
                by_position_bucket={b: overall_b[b].summary() for b in names if b in overall_b},
                by_split=by_split_out, per_layer=per_layer_out,
            )
            combinations[mkey] = summary

        dev = torch.device(device)
        return {
            "schema": LONG_EVAL_SCHEMA, "transform_scope": "shipped_single_transform_per_path",
            "transform_fingerprint": transform_fingerprint, "protection_policy": self.config.protection,
            "query_tile_rows": ROWS,
            "position_buckets": [{"name": n, "start": s, "end": e} for n, s, e in self.buckets],
            "dsa_selection_drift": {
                "structural_zero": True, "overlap_by_construction": 1.0,
                "sparse_layers": sorted(self._sparse_layers), "note": self.dsa_indexer_note,
            },
            "corpus": dict(corpus_meta), "layers_evaluated": sorted(self._layers_seen),
            "combinations": combinations, "wall_seconds": float(wall_seconds),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "process_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
        }
