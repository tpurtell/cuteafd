"""Bounded ERNIE pilot from hash-bound real prompt tails and merged reference parity."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import signal
import time
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from inkling2 import numerical
from kimi_k3.uniform_k2.storage import atomic_json, sha256_file
from . import identity
from .forward import ErnieForward, reference
from .routing import ErnieRoutes, recovered_covariance
from .source import BASE_REVISION, HeaderSource

LAYERS = (0, 1, 7, 20, 21, 42)
BARS = {4: 1e-2, 6: 1e-3, 8: 1e-4}
MOE = "block_sparse_moe."
GROUP = "self_attn.linear_o_group_proj"
SPLITS = ("calibration", "heldout")


def payload_projection(source, groups):
    """Scale-inclusive byte totals; TP2 is an ideal balanced weights-only floor."""
    native = sum(source.sizes[n] for n, r in source.roles.items() if r in ("native", "flashindexer"))
    core_native = sum(source.sizes[n] for n, r in source.roles.items() if r == "core")
    totals = Counter()
    for name, role in source.roles.items():
        if role not in ("core", "routed_expert"):
            continue
        out, width = source.specs[name][1]
        slices = groups if name.endswith(GROUP) else 1
        if out % slices:
            raise ValueError("uneven grouped output")
        padded = math.ceil(out / slices / 128) * 128
        for bits in ((4,) if role == "routed_expert" else (6, 8)):
            totals[bits] += slices * (padded * width * bits // 8 + 2 * (padded + width) + 4)
    views = {}
    for label, core in (("original", core_native), ("k8", totals[8]), ("k6", totals[6])):
        total = native + core + totals[4]
        views[label] = dict(bytes=total, shared_expert_bytes=totals[4], core_and_globals_bytes=native + core,
                            ideal_tp2_all_weights_per_rank_bytes=(total + 1) // 2,
                            ideal_tp2_core_and_globals_per_rank_bytes=(native + core + 1) // 2)
    return dict(views=views, physical_payload_bytes=native + core_native + sum(totals.values()),
                hca_group_encoding="pilot independent input scales and marker per group",
                tp2_assumptions="equal payload halves; excludes duplication, workspaces, KV and runtime reserves")


def recovery_summary(bank):
    fields = ("natural_rows", "augmented_rows", "identity_rows", "effective_rows")
    rows = []
    for index in range(len(bank.bounds) - 1):
        _, _, evidence = bank.calibration(index)
        rows.append({k: value for k, value in evidence.items() if k != "augmented_corpus_rows"})
    return dict(experts=len(rows), natural_only=sum(r["augmented_rows"] == r["identity_rows"] == 0 for r in rows),
                with_candidates=sum(r["augmented_rows"] > 0 for r in rows),
                with_identity=sum(r["identity_rows"] > 0 for r in rows),
                zero_natural=sum(r["natural_rows"] == 0 for r in rows),
                distributions={k: dict(sorted(Counter(r[k] for r in rows).items())) for k in fields},
                per_expert=rows)


def load_fixtures(directory, receipt):
    manifest = directory / "manifest.json"
    document = identity.read(manifest)
    if (document["schema"] != "ernie55-pilot-inputs-v1"
            or receipt["fixtures_manifest_sha256"] != sha256_file(manifest)):
        raise ValueError("pilot inputs do not match maxq forward parity")
    result = {k: [] for k in SPLITS}
    seen = set()
    from .prepare_corpus import panel
    for row in document["records"]:
        path = directory / row["file"]
        if (row["split"] not in result or sha256_file(path) != row["sha256"]
                or row["start"] + row["tokens"] != row["source_tokens"]
                or row["input_ids_sha256"] in seen):
            raise ValueError("invalid or repeated pilot fixture")
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if (sample["ids"].shape != (row["tokens"],)
                or panel.ids_sha256(sample["ids"].tolist()) != row["tail_ids_sha256"]):
            raise ValueError("fixture token IDs differ from attested tail")
        result[row["split"]].append(sample)
        seen.add(row["input_ids_sha256"])
    if any(len(v) < 2 for v in result.values()):
        raise ValueError("pilot requires distinct calibration and heldout prompts")
    return result


def nmse(expected, actual):
    a = expected.double()
    energy = a.square().sum()
    if not bool(energy > 0):
        raise ValueError("heldout projection reference has no signal")
    error = float((actual.double() - a).square().sum() / energy)
    if not math.isfinite(error):
        raise ValueError("nonfinite heldout projection")
    return error


class Pilot:
    def __init__(self, model, source, fixtures, output):
        self.model, self.source, self.fixtures, self.output = model, source, fixtures, output
        self.device = model.checkpoint.device
        self.names = {n for n, role in source.roles.items() if role == "core"}
        self.started = time.monotonic()
        self.report = dict(schema="ernie55-projection-pilot-v1", status="running", source_revision=BASE_REVISION,
                           layers=list(LAYERS), projections=[], recovered={}, completed=[], proposal=None,
                           projection_frontiers="sampled native layer inputs; K8 within target layers",
                           pilot_experts="largest natural calibration count per MoE pilot layer; others native",
                           size_projection=payload_projection(source, model.config.o_groups))

    def checkpoint(self):
        self.report["elapsed_seconds"] = time.monotonic() - self.started
        atomic_json(self.output / "pilot.json", self.report)

    def encode(self, name, weight, train, held, widths, recovery=None):
        """Encode both widths from one raw Hessian. Decode only serialized MUL1 tensors."""
        start = time.monotonic()
        if len(held) == 0:
            raise ValueError(f"no natural heldout rows for {name}")
        acc = numerical.covariance_accumulator(weight.shape[1], device=self.device)
        if len(train):
            acc.add(train)
        raw = (recovered_covariance(acc, recovery) if recovery is not None
               else dict(H=acc.snapshot(), count=acc.count, finalized=False))
        del acc
        expected = reference.linear(held, weight)
        row = dict(module=name, shape=list(weight.shape), observed_rows=len(train), heldout_rows=len(held),
                   effective_rows=raw["count"], identity_rows=raw.get("identity_rows", 0),
                   arms={"native": dict(heldout_nmse=nmse(expected, expected), bpw=16, passed=True)})
        self.report["projections"].append(row)
        best = None
        for bits in widths:
            self.report["active"] = dict(module=name, bits=bits)
            self.checkpoint()
            packed, _, _ = numerical.quantize_projection(weight, raw, bits=bits, seed=787, sigma=.025,
                                                          device=self.device, codebook="mul1")
            decoded = numerical.reconstruct_projection(packed, list(weight.shape), device=self.device,
                                                       dtype=torch.bfloat16, codebook="mul1").float()
            error = nmse(expected, reference.linear(held, decoded))
            nbytes = sum(t.numel() * t.element_size() for t in packed.values())
            row["arms"][str(bits)] = dict(heldout_nmse=error, bpw=8 * nbytes / weight.numel(),
                                          packed_bytes=nbytes, bar=BARS[bits], passed=error <= BARS[bits])
            if best is None:
                best = decoded
            del packed, decoded
            row["seconds"] = time.monotonic() - start
            self.checkpoint()
        return best

    def feature(self):
        model = self.model
        name = "model.mtp.feature_projection.weight"
        aux = model.checkpoint.mtp_aux(model.config)
        def inputs(prompt):
            return torch.cat([prompt[f"tap_{i}"] for i in (40, 41, 42)], dim=-1).to(self.device)
        split_inputs = [[inputs(p) for p in self.fixtures[split]] for split in SPLITS]
        weight = self.encode(name, aux["feature_projection.weight"], *(torch.cat(v) for v in split_inputs),
                             widths=(8, 6))
        self.save("feature", {name: weight})
        for split, rows in zip(SPLITS, split_inputs, strict=True):
            for prompt, x in zip(self.fixtures[split], rows, strict=True):
                ctx = reference.norm(reference.linear(x, weight), aux["feature_norm.weight"], model.config.rms_norm_eps)
                prompt["compact_context"] = ctx[-model.config.sliding_window:].cpu()

    def save(self, label, values):
        dest = self.output / "reconstruct" / (label + ".safetensors")
        tmp = dest.with_suffix(".new")
        save_file({n: v.to(device="cpu", dtype=torch.bfloat16).contiguous() for n, v in values.items()}, str(tmp))
        tmp.replace(dest)

    def layer(self, index, draft=False):
        model, c = self.model, self.model.config
        prefix = f"model.{'mtp' if draft else 'layers'}.{index}."
        weights = model.load_draft_layer(index) if draft else model.load_layer(index)
        manifest = {name[len(prefix):] for name in self.names if name.startswith(prefix)}
        widths = (8,) if draft and index > 0 else (8, 6)
        encoded = set()
        def encode(name, xs):
            if name not in manifest or name in encoded:
                raise ValueError(f"unexpected or duplicate pilot projection {prefix}{name}")
            weights[name] = self.encode(prefix + name, weights[name], *(torch.cat(v) for v in xs), widths)
            encoded.add(name)

        samples = [self.fixtures[s] for s in SPLITS]
        def run_attention(p, observe=None):
            if draft:
                hidden_key = "draft0_hidden" if index == 0 else "compact_hidden"
                hidden = p[hidden_key].to(self.device)
                positions = p[f"draft{index}_positions"].to(self.device)
                ctx = p["compact_context"].to(self.device)
                pos = torch.arange(len(p["ids"]) - len(ctx), len(p["ids"]), device=self.device)
                cos, sin = reference.hca_phase(pos, c, rope_mode="plain")
                kv = reference.norm(reference.linear(ctx, weights["self_attn.linear_kv_proj.weight"]),
                                    weights["self_attn.kv_layernorm.weight"], c.rms_norm_eps)
                kv = reference.rotate_tail(kv.unsqueeze(1), cos, sin).squeeze(1)
            else:
                hidden = p[f"layer_{index}"].to(self.device)
                positions = torch.arange(len(p["ids"]), device=self.device)
                kv = None
            return model.mlp_input(hidden, weights, index, positions, draft_context_kv=kv, observe_hca_group=observe)

        dsa = not draft and c.is_dsa_layer(index)
        if dsa:
            groups = [("q_a_proj", "kv_a_proj_with_mqa"), ("q_b_proj", "gate_proj"), ("o_proj",)]
        else:
            roots = ["linear_q_down_proj", "linear_kv_proj"]
            if not draft:
                roots.extend(["core_attention.compressor.linear_wkv", "core_attention.compressor.linear_wgate"])
            groups = [roots, ("linear_q_up_proj", "gate_proj"), (GROUP,), ("o_proj",)]
        for group in groups:
            if group == (GROUP,):
                captured = [[[] for _ in range(c.o_groups)] for _ in SPLITS]
                for i, prompts in enumerate(samples):
                    for p in prompts:
                        run_attention(p, lambda g, _, x: captured[i][g].append(x.clone()))
                parts = weights[GROUP].chunk(c.o_groups)
                weights[GROUP] = torch.cat([self.encode(f"{prefix}{GROUP}[{g}]", part,
                    *(torch.cat(v[g]) for v in captured), widths) for g, part in enumerate(parts)])
                encoded.add(GROUP)
            else:
                names = ["self_attn." + base + ".weight" for base in group]
                lookup = {weights[name].data_ptr(): name for name in names}
                captured = [defaultdict(list), defaultdict(list)]
                linear = reference.linear
                for i, prompts in enumerate(samples):
                    def capture(x, w):
                        key = lookup.get(w.data_ptr())
                        if key is not None and w.shape == weights[key].shape:
                            captured[i][key].append(x)
                        return linear(x, w)
                    with patch.object(reference, "linear", capture):
                        for p in prompts:
                            run_attention(p)
                for name in names:
                    encode(name, [v[name] for v in captured])
        states = [[run_attention(p) for p in prompts] for prompts in samples]
        mlp = [[v.x for v in batch] for batch in states]
        if not draft and not c.is_moe_layer(index):
            for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                encode(name, mlp)
            active = [[model.activated(x, weights["mlp.gate_proj.weight"], weights["mlp.up_proj.weight"])
                       for x in batch] for batch in mlp]
            encode("mlp.down_proj.weight", active)
        else:
            for name in ("fc1_latent_proj", "shared_experts.w1", "shared_experts.w3"):
                encode(MOE + name + ".weight", mlp)
            shared = MOE + "shared_experts."
            active = [[model.activated(x, weights[shared + "w1.weight"], weights[shared + "w3.weight"])
                       for x in batch] for batch in mlp]
            encode(shared + "w2.weight", active)
            banks = []
            for batch in mlp:
                batches = ((i, {k: t.cpu() for k, t in model.route_batch(x, weights).items()})
                           for i, x in enumerate(batch))
                banks.append(ErnieRoutes(batches, c.n_routed_experts, sum(len(x) for x in batch)))
            recovery = recovery_summary(banks[0])
            evidence_file = f"recovery-{prefix.rstrip('.')}.json"
            atomic_json(self.output / evidence_file, recovery.pop("per_expert"))
            self.report["recovered"][prefix.rstrip(".")] = dict(recovery, file=evidence_file)
            self.checkpoint()
            sums = self.experts(prefix, weights, banks, witness=(not draft or index == 0),
                                use_quantized_witness=not draft)
            fc2 = MOE + "fc2_latent_proj.weight"
            fc2_inputs = [[model.fc2_input(s, weights)] for s in sums]
            encode(fc2, fc2_inputs)
            if draft:
                for prompts, batch, summed in zip(samples, states, sums, strict=True):
                    pieces = summed.split([len(st.x) for st in batch])
                    for p, st, s in zip(prompts, batch, pieces, strict=True):
                        p["compact_hidden"] = model.finish_moe(st, s, weights).cpu()
        if encoded != manifest:
            raise ValueError(f"incomplete projection set for {prefix}: {sorted(manifest ^ encoded)}")
        if draft:
            self.save(f"mtp-{index}", {prefix + n: weights[n] for n in manifest})
        self.report["completed"].append(prefix.rstrip("."))
        self.checkpoint()

    def experts(self, prefix, weights, banks, *, witness, use_quantized_witness):
        calibration, heldout = banks
        counts = [b - a for a, b in zip(calibration.bounds, calibration.bounds[1:])]
        representative = counts.index(max(counts)) if witness else None
        combined = [torch.zeros(bank.latent.shape, device=self.device) for bank in banks]
        for index, _ in enumerate(counts):
            selections = [bank.expert(index) for bank in banks]
            if all(not len(rows) for rows, _ in selections) and index != representative:
                continue
            w1, w3, w2 = weights["experts"].get(index, self.device)
            if index == representative:
                rows, gamma, evidence = calibration.calibration(index)
                held, held_gamma = heldout.expert(index)
                train_x = calibration.latent[rows].to(self.device)
                held_x = heldout.latent[held].to(self.device)
                key = f"{prefix}{MOE}experts.{index}."
                q1, q3 = (self.encode(key + name, w, train_x, held_x, (4,), evidence)
                          for name, w in (("w1.weight", w1), ("w3.weight", w3)))
                train_act = self.model.activated(train_x, q1, q3) * gamma.to(self.device)[:, None]
                held_act = self.model.activated(held_x, q1, q3) * held_gamma.to(self.device)[:, None]
                q2 = self.encode(key + "w2.weight", w2, train_act, held_act, (4,), evidence)
                if use_quantized_witness:
                    w1, w3, w2 = q1, q3, q2
            for bank, summed, (rows, gamma) in zip(banks, combined, selections, strict=True):
                if len(rows):
                    x = bank.latent[rows].to(self.device)
                    active = self.model.activated(x, w1, w3)
                    contribution = reference.linear(active, w2) * gamma.to(self.device)[:, None]
                    summed.index_add_(0, rows.to(self.device), contribution)
        return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "fixtures", "vendor", "preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800, choices=range(1, 1801), metavar="1..1800")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    local = identity.read(args.preflight)
    inputs = identity.candidate(args.source, args.corpus, identity.REPO)
    verified = identity.verified_identity(inputs, identity.read(args.vendor), local)
    fixtures = load_fixtures(args.fixtures, local)
    source = HeaderSource(args.source)
    if not torch.cuda.is_available():
        raise RuntimeError("ERNIE numerical pilot requires maxq CUDA")
    def timeout(signum, frame):
        raise TimeoutError("pilot GPU wall-time budget expired")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    model = ErnieForward.from_snapshot(args.source, "cuda:0")
    pilot = Pilot(model, source, fixtures, args.output)
    pilot.report["identity_sha256"] = identity.digest(verified)
    pilot.checkpoint()
    (args.output / "reconstruct").mkdir()
    try:
        with torch.inference_mode():
            for index in LAYERS:
                pilot.layer(index)
            pilot.feature()
            for stage in range(3):
                pilot.layer(stage, draft=True)
            from .pilot_proposals import evaluate
            pilot.report["proposal"] = evaluate(model, args.fixtures, args.output / "reconstruct")
        passed = all(v["passed"] for r in pilot.report["projections"] for v in r["arms"].values())
        pilot.report["status"] = "passed" if passed and pilot.report["proposal"]["passed"] else "failed"
        pilot.report["active"] = None
    except BaseException as exc:
        pilot.report.update(status="incomplete", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.alarm(0)
        pilot.checkpoint()
    if pilot.report["status"] != "passed":
        raise SystemExit("ERNIE pilot missed its heldout gate")


if __name__ == "__main__":
    main()
