"""Run the qualified GLM Flash K8-causal pass and seal its direct MUL1 package."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import time

from kimi_k3.uniform_k2.storage import atomic_json, canonical, fsync_directory, sha256_file
from . import corpus, source
from .export import DirectMCWriter, REFERENCE_REVISION

REPO = Path(__file__).resolve().parents[3]
PHASES = ("initial", *(f"layer-{i:02}" for i in range(45)), "head", "layer-45")
CODE = (
    "quantization/glm5_flash/mul1_mc", "quantization/glm5_flash/core_k6/recipe.py",
    "quantization/glm5_flash/core_k6/pilot.py", "quantization/glm5/core_k6/recipe.py",
    "quantization/inkling2/numerical.py", "quantization/inkling2/export.py",
    "quantization/kimi_k3/uniform_k2/storage.py", "quantization/kimi_k3/core_k6/storage.py",
    "quantization/kimi_k3/prepare_corpus.py", "quantization/hy4_flash/prepare_corpus.py",
    "quantization/hy4_flash/storage.py", "quantization/hy4_flash/export.py",
    "quantization/hy4/run.py", "quantization/hy4_flash/run.py",
    "quantization/core_export.py", "quantization/mc_package.py", "scripts/lib/mc_manifest.py",
)


def read(path):
    return json.loads(path.read_bytes())


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True, timeout=60).strip()


def frozen_code():
    if os.environ.get("INKLING_GPTQMODEL_ROOT"):
        raise ValueError("production requires the pinned GPTQModel submodule")
    files = git(REPO, "ls-files", "--", *CODE).splitlines()
    for required in ("layer", "pilot", "run", "experts", "numerics", "source", "native_source", "qualification",
                     "corpus", "export", "parity"):
        if f"quantization/glm5_flash/mul1_mc/{required}.py" not in files:
            raise ValueError(f"uncommitted numerical implementation: {required}")
    if git(REPO, "status", "--porcelain", "--", *CODE):
        raise ValueError("uncommitted GLM Flash production source")
    result = {"files": {f: sha256_file(REPO / f) for f in files}}
    tree_hash = runpy.run_path(str(REPO / "scripts/build/verify-sparkinfer-source.py"))["source_tree_sha256"]
    for name in ("gptqmodel", "transformers"):
        submodule = Path("third_party") / name
        lock_path = str(submodule) + ".lock.json"
        revision = git(REPO / submodule, "rev-parse", "HEAD")
        lock = read(REPO / lock_path)
        if (git(REPO / submodule, "status", "--porcelain") or git(REPO, "status", "--porcelain", "--", lock_path)
                or git(REPO, "rev-parse", f"HEAD:{submodule}") != revision
                or lock["revision"] != revision or lock["source_tree_sha256"] != tree_hash(REPO / submodule)):
            raise ValueError(f"{name} differs from its committed source lock")
        result["fork" if name == "gptqmodel" else name] = revision
    if result["transformers"] != REFERENCE_REVISION:
        raise ValueError("GLM Flash reference revision differs")
    return result


def qualification(root, snapshot, code):
    from . import pilot
    if read(root / "source.json") != source.audit(snapshot):
        raise ValueError("pinned source header audit differs")
    manifest = read(root / "corpus/manifest.json")
    corpus_hash = sha256_file(root / "corpus/manifest.json")
    if (manifest["source"].get("files") != corpus.SOURCE_FILES
            or manifest["source"].get("revision") != source.BASE_REVISION
            or manifest["software"].get("transformers_revision") != REFERENCE_REVISION
            or manifest["source"].get("repo_id") != source.BASE_REPO):
        raise ValueError("corpus source identity differs")
    parity = read(root / "parity.json")
    report = read(root / "pilot.json")
    expected_layers = [0, 3, 22, 23, 44, 45]
    sparse = parity["sparse"]
    if (parity["status"] != "passed" or parity["source"] != source.BASE_REVISION
            or parity["layers"] != expected_layers or not sparse["byte_exact"] or not sparse["sparse_selection"]
            or sparse["tokens"] < 2052 or parity["corpus_sha256"] != corpus_hash
            or parity["parity_code_sha256"] != code["files"]["quantization/glm5_flash/mul1_mc/parity.py"]
            or parity["probes"] != ["train_0", "train_1", "train_2", "train_3", "held_0", "held_1"]):
        raise ValueError("invalid reference-forward parity receipt")
    validate_gb10(read(root / "gb10-exactness.json"), code)
    quality_gate(report, pilot.expected_core_arms())
    captures = parity["captures_sha256"]
    if (report["source_revision"] != source.BASE_REVISION or report["fork_revision"] != code["fork"]
            or report["parity_sha256"] != sha256_file(root / "parity.json")
            or report["corpus_sha256"] != corpus_hash or report["capture_sha256"] != captures
            or set(captures) != {str(i) for i in [*expected_layers, 46]}
            or report["pilot_code_sha256"] != code["files"]["quantization/glm5_flash/mul1_mc/pilot.py"]):
        raise ValueError("pilot inputs or executable identity differ")
    return manifest, report


def validate_gb10(receipt, code):
    checks = ("raw_hessians_equal", "packed_bytes_equal", "heldout_sum_equal", "sealed_replay_equal",
              "selection_equal", "gate_evidence_equal", "workspace_reused", "native_bf16_bank_equal")
    sources = {name: code["files"]["quantization/glm5_flash/mul1_mc/" + name]
               for name in ("qualification.py", "experts.py", "numerics.py")}
    if (receipt["status"] != "passed" or receipt["device"] != "NVIDIA GB10"
            or any(receipt.get(k) is not True for k in checks)
            or receipt.get("native_bf16_compared_k4_encodes") != 3 or receipt["compared_k4_encodes"] != 3
            or receipt.get("code_sha256") != sources or receipt["fork_revision"] != code["fork"]):
        raise ValueError("GB10 exactness witness differs from the production input bank or quantizer")


def quality_gate(report, expected_core):
    if (report.get("schema") != "glm5-flash-mul1-pilot-v1" or report["status"] != "passed"
            or report["layers"] != [0, 3, 22, 23, 44, 45, 46] or report["mtp_eh_bits"] != [8, 6]
            or report["memory_plan"]["admitted"] is not True):
        raise ValueError("incomplete GLM Flash pilot qualification")
    expected = set(expected_core)
    winners = report["selected_experts"]
    if set(winners) != {"3", "22", "23", "44", "45"}:
        raise ValueError("pilot MoE layer inventory differs")
    for layer, expert in winners.items():
        if ({r["expert"] for r in report["recovery"][layer]} != set(range(288))
                or len(report["recovery"][layer]) != 288):
            raise ValueError("pilot expert recovery distribution is incomplete")
        if type(expert) is not int or not 0 <= expert < 288:
            raise ValueError("invalid pilot expert")
        for part in ("gate_proj", "up_proj", "down_proj"):
            expected.add((f"model.language_model.layers.{layer}.mlp.experts.{expert}.{part}", 4))
    bars = {8: 1e-4, 6: 1e-3, 4: 1e-2}
    for row in report["modules"]:
        key = (row["module"], row["bits"])
        if (key not in expected or row["side"] != 1 or row["codebook"] != "mul1"
                or row["heldout_rows"] <= 0 or row["multiplier"] != source.MUL1_MULTIPLIER or row["native_nmse"] != 0
                or not 0 <= row["heldout_nmse"] <= bars[key[1]]):
            raise ValueError(f"invalid pilot projection qualification: {key}")
        expected.remove(key)
    if expected:
        raise ValueError("pilot projection inventory is incomplete")


def numerical_identity(root, snapshot):
    import torch
    import transformers
    code = frozen_code()
    manifest, report = qualification(root, snapshot, code)
    if not Path(transformers.__file__).resolve().is_relative_to(REPO / "third_party/transformers"):
        raise ValueError("production must import the pinned Transformers source")
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 1)
            or not all(os.environ.get(key) == "0" for key in
                       ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))):
        raise ValueError("expected one SM121 GPU and both TF32 overrides disabled")
    workspace = report["memory_plan"]["owners"]["workspace"]
    if type(workspace) is not int or workspace <= 0 or workspace != report["expert_workspace_bytes"]:
        raise ValueError("missing pilot expert workspace allowance")
    return dict(schema="cuteafd-glm5-flash-mul1-causal-v1", source_repo=source.BASE_REPO,
                source_revision=source.BASE_REVISION, code=code, dataset=manifest["dataset"],
                components=manifest["components"], diagnostics=manifest["diagnostics"],
                codebook="mul1", codebook_multiplier=source.MUL1_MULTIPLIER, seed=787, sigma=.025,
                core_bits=[8, 6], expert_bits=4, upstream_activations="k8-compact",
                expert_workspace_bytes=workspace, torch=torch.__version__, cuda=torch.version.cuda,
                reference_revision=REFERENCE_REVISION, expert_recovery_ranks=[9,16], minimum_rows=1024,
                proofs={n: sha256_file(root / n) for n in
                        ("corpus/manifest.json", "source.json", "parity.json", "pilot.json", "gb10-exactness.json")})


@contextmanager
def exclusive_run(root):
    with (root / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another producer owns the GLM Flash run") from exc
        yield


def bind(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"frozen run identity differs: {path.name}")
    else:
        atomic_json(path, value)


def write_error_csv(root, expected):
    metrics = root / "metrics"
    rows = [row for path in sorted(metrics.glob("*.json")) for row in read(path)]
    if len(rows) != len(expected) or {(r["module"], r["bits"]) for r in rows} != expected:
        raise ValueError("error metrics must cover every committed encode exactly once")
    for row in rows:
        error = row["hessian_weighted_relative_error"]
        if (row["codebook"] != "mul1" or row["side"] != 1 or not math.isfinite(error) or error < 0
                or (row["bits"] == 6 and error > 1e-3)):
            raise ValueError(f"invalid numerical evidence: {row['module']}")
        if row["bits"] == 4:
            total = row["natural_rows"] + row["augmented_rows"] + row["identity_rows"]
            if (total != row["effective_rows"] or total < 1024 or len(row["augmented_rank_counts"]) != 8
                    or sum(row["augmented_rank_counts"]) != row["augmented_rows"]
                    or not re.fullmatch("[0-9a-f]{32}", row["selection_xxh3_128"])):
                raise ValueError("incomplete routed recovery evidence")
    extra = set().union(*(r.keys() for r in rows)) - {"module", "bits"}
    path = root / "per-module-errors.csv"
    temp = root / "per-module-errors.csv.partial"
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ("module", "bits", *sorted(extra)))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fsync_directory(root)
    return path


def traverse(root, engine, stages, writer, identity, *, attempt, interrupted_phase, status):
    """Commit a complete, durable successor before retiring its only predecessor."""
    key = hashlib.sha256(canonical(identity).encode()).hexdigest()
    path = root / "frontier.json"
    frontier = read(path) if path.exists() else dict(identity=key, history=[])
    history = frontier["history"]
    if frontier["identity"] != key or [row["phase"] for row in history] != list(PHASES[:len(history)]):
        raise ValueError("invalid GLM Flash causal frontier")
    incoming = history[-1]["stage"] if history else None
    def verify(name):
        if not name or stages.load(name) is None:
            raise ValueError(f"missing committed activation frontier: {name}")
    if incoming:
        verify(incoming)
        if len(history) >= 2 and history[-2]["stage"] != incoming:
            stages.retire(history[-2]["stage"])
    for phase in PHASES[len(history):]:
        status("running", phase)
        start = time.monotonic()
        if phase == "initial":
            successor = engine.initial()
        elif phase == "head":
            engine.head(incoming)
            successor = incoming
        elif phase == "layer-45":
            successor = engine.mtp(incoming)
        else:
            successor = engine.target(int(phase.removeprefix("layer-")), incoming)
        verify(successor)
        label = "source-native" if phase == "initial" else phase
        if not writer.group_complete(label) or (phase != "initial" and not (root/"metrics"/f"{label}.json").is_file()):
            raise ValueError(f"cannot advance past unsealed weights or metrics: {phase}")
        row = dict(phase=phase, stage=successor, seconds=time.monotonic()-start, attempt=attempt,
                   interrupted=phase == interrupted_phase, completed_at=utc())
        history.append(row)
        atomic_json(path, frontier)
        print(json.dumps(dict(event="phase_committed", **row)), flush=True)
        if incoming and incoming != successor:
            stages.retire(incoming)
        incoming = successor
    return history


def produce(root, output, snapshot, identity, *, attempt, interrupted_phase, status):
    import torch
    from hy4_flash.storage import Stages
    from .native_source import prepare_native_source
    from .layer import LayerPass
    tokens = [b["input_ids"] for b in corpus.load_token_batches(root / "corpus")]
    expected_count = sum(c["count"] for c in identity["components"].values())
    expected_tokens = sum(c["tokens"] for c in identity["components"].values())
    if len(tokens) != expected_count or sum(t.numel() for t in tokens) != expected_tokens:
        raise ValueError("calibration stream differs from both frozen components")
    stages = Stages(root / "stages", identity)
    writer = DirectMCWriter(output, snapshot, identity)
    status("running", "prepare_native_source")
    source_model = prepare_native_source(snapshot, writer, core_modules=source.core_modules(8),
                                         expert_modules=source.expert_modules())
    try:
        with torch.inference_mode():
            engine = LayerPass(source_model, stages, writer, identity, tokens)
            history = traverse(root, engine, stages, writer, identity, attempt=attempt,
                               interrupted_phase=interrupted_phase, status=status)
    finally:
        source_model.close()
    status("running", "export")
    expected = {(m, bits) for m in source.core_modules(8) for bits in (8, 6)}
    expected.update((m, 4) for m in source.expert_modules())
    csv_path = write_error_csv(root, expected)
    validation = writer.finalize(error_csv=csv_path, card_dir=root/"card")
    complete = dict(status="passed", identity=identity, output=str(output), revision=source.BASE_REVISION,
                    repository=source.TARGET_REPO, completed_target_layers=45, completed_drafter_layers=1,
                    completed_encodes=len(expected), error_csv=str(csv_path), phase_timings=history,
                    task_intermediates=[str(stages.root)], package_validation=str(output/"package-validation.json"),
                    package_status=validation["status"], original_discovery=validation["original_discovery"])
    atomic_json(root / "numerical-complete.json", complete)
    status("passed", "numerical_complete")
    return complete


def run(root, output, snapshot, *, diagnosis=None, after_boot_id=None):
    from torch import set_num_threads, backends
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("publication output and numerical run root must be disjoint")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if after_boot_id == boot:
        raise ValueError("reboot recovery must wait for a changed boot ID")
    with exclusive_run(root):
        completed = root / "numerical-complete.json"
        if completed.exists():
            receipt = read(completed)
            if (receipt["output"] != str(output) or receipt["identity"] != read(root / "identity.json")
                    or receipt["repository"] != source.TARGET_REPO or receipt["status"] != "passed"):
                raise ValueError("inconsistent completion receipt")
            return receipt
        path = root / "status.json"
        prior = read(path) if path.exists() else {}
        if after_boot_id and (not prior or after_boot_id != prior["boot_id"]):
            raise ValueError("reboot recovery does not match the previous producer")
        if prior and not (diagnosis or "").strip():
            raise ValueError("resuming a failed or interrupted producer requires --diagnosis")
        if not prior and ((root / "stages").exists() or (root / "frontier.json").exists()):
            raise ValueError("missing producer attempt receipt for existing numerical work")
        set_num_threads(min(10, len(os.sched_getaffinity(0))))
        backends.cuda.matmul.allow_tf32 = backends.cudnn.allow_tf32 = False
        identity = numerical_identity(root, snapshot)
        bind(root / "identity.json", identity)
        bind(root / "destinations.json", dict(output=str(output), snapshot=str(snapshot)))
        attempt = prior.get("attempt", 0) + 1
        attempt_record = dict(attempt=attempt, boot_id=boot, started_at=utc(), diagnosis=diagnosis,
                              previous_status=prior)
        atomic_json(root / "attempts" / f"{attempt:04}.json", attempt_record)
        current_phase = "preflight"
        def status(state, phase, **details):
            nonlocal current_phase
            current_phase = phase
            atomic_json(path, dict(status=state, phase=phase, updated_at=utc(), attempt=attempt,
                                   boot_id=boot, **details))
        try:
            status("running", current_phase)
            return produce(root, output, snapshot, identity, attempt=attempt,
                           interrupted_phase=prior.get("phase"), status=status)
        except BaseException as exc:
            status("blocked", current_phase, error=f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "snapshot"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--diagnosis", help="diagnosed cause and recovery for the previous producer attempt")
    parser.add_argument("--after-boot-id", help="limit a reboot-only recovery to this previous boot ID")
    args = parser.parse_args()
    result = run(args.root.resolve(strict=True), args.output.resolve(), args.snapshot.resolve(strict=True),
                 diagnosis=args.diagnosis, after_boot_id=args.after_boot_id)
    print(json.dumps(dict(status=result["status"], completed_encodes=result["completed_encodes"],
                          output=result["output"])), flush=True)


if __name__ == "__main__":
    main()
