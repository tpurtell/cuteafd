"""Bind ERNIE source, rendered IDs, producer code and reference before a numerical run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import uuid

from kimi_k3.uniform_k2.storage import atomic_json, canonical, sha256_file
from .source import BASE_REPO, BASE_REVISION, HeaderSource

REPO = Path(__file__).resolve().parents[2]
REFERENCE_HEAD = "9e70bd700e155163ba4ed72dcb1dd5411df59faf"
REFERENCE_FILES = (
    "python/reference/families/ernie55_flash/model.py", "python/reference/families/ernie55_flash/sources.py",
    "python/reference/families/ernie55_flash/__init__.py",
    "python/reference/shape_invariant.py", "python/reference/families/hy4_common.py",
)
SHARED_FILES = (
    "quantization/hy4_flash/storage.py", "quantization/hy4_flash/preflight.py",
    "quantization/hy4_flash/source.py", "quantization/hy4_flash/prepare_corpus.py",
    "quantization/hy4_flash/recipe.py", "quantization/glm5/core_k6/recipe.py",
    "quantization/hy4_family_source.py", "quantization/kimi_k3/core_k6/storage.py",
    "quantization/kimi_k3/uniform_k2/storage.py", "quantization/kimi_k3/prepare_corpus.py",
    "quantization/inkling2/numerical.py", "quantization/inkling2/export.py", "quantization/inkling2/publish.py",
    "quantization/core_export.py", "quantization/mc_package.py", "scripts/lib/mc_manifest.py",
    "quantization/hub_requirements.py", "quantization/hub_lfs.py", "third_party/gptqmodel.lock.json",
    "scripts/build/verify-sparkinfer-source.py",
)
RECIPE = dict(codebook="mul1", codebook_multiplier=0x83DCD12D, seed=787, sigma=.025,
              expert_bits=4, core_bits=[8, 6], upstream_core_bits=8, minimum_expert_rows=1024,
              candidate_ranks=[11, 20], identity_fill="missing-times-I-before-normalization",
              hessian="fp32-products-k128-fp64-sum-fp32-storage-v1", tf32=False,
              replay="serialized-to-bf16-promoted-fp32", flashindexer=False, dsa_index_heads=64)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def committed_files(repo, *paths):
    if git(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths):
        raise ValueError("identity requires clean committed producer files")
    files = git(repo, "ls-files", "--error-unmatch", "--", *paths).splitlines()
    return {path: sha256_file(repo / path) for path in files}


def fork_identity(repo):
    fork_path = "third_party/gptqmodel"
    fork = repo / fork_path
    lock = read(repo / (fork_path + ".lock.json"))
    if (os.environ.get("INKLING_GPTQMODEL_ROOT") or git(fork, "status", "--porcelain")
            or git(repo, "rev-parse", "HEAD:" + fork_path) != git(fork, "rev-parse", "HEAD")
            or lock["revision"] != git(fork, "rev-parse", "HEAD")
            or lock["repository"] != "https://github.com/tpurtell/GPTQModel.git"):
        raise ValueError("GPTQModel override, dirty source or fork pin mismatch")
    tree = runpy.run_path(str(repo / "scripts/build/verify-sparkinfer-source.py"))["source_tree_sha256"]
    if tree(fork) != lock["source_tree_sha256"]:
        raise ValueError("GPTQModel source tree differs from its committed lock")
    return lock


def corpus_identity(root):
    from .prepare_corpus import CONTRACT, DATASET, SCHEMA, SOURCE_FILES
    manifest = read(root / "manifest.json")
    if (manifest["schema"] != SCHEMA or manifest["contract"] != CONTRACT or manifest["dataset"] != DATASET
            or manifest["status"] != "passed" or manifest["source"] != dict(repo_id=BASE_REPO,
                    revision=BASE_REVISION, files=SOURCE_FILES)):
        raise ValueError("prepared corpus is for a different ERNIE source or recipe")
    files = {split + ".safetensors": row for split, row in manifest["splits"].items()}
    files["long-context-witnesses.safetensors"] = manifest["long_context_witnesses"]
    if set(files) != {name + ".safetensors" for name in
                      ("calibration", "screening", "heldout", "long-context-witnesses")}:
        raise ValueError("prepared corpus split inventory differs")
    for name, row in files.items():
        if row["file"] != name or sha256_file(root / name) != row["sha256"]:
            raise ValueError(f"prepared corpus token file differs: {name}")
    standard = manifest["components"]["exl3_standard"]
    packed = read(root / "exl3_standard.json")
    if (standard["input_ids_sha256"] != packed["input_ids_sha256"] or packed["contract"] != standard["contract"]
            or packed["tokenizer_sha256"] != SOURCE_FILES["tokenizer.json"]
            or manifest["long_context_witnesses"]["hessian_eligible"] is not False):
        raise ValueError("standard token pack or long-context eligibility differs")
    return dict(manifest_sha256=sha256_file(root / "manifest.json"),
                standard_receipt_sha256=sha256_file(root / "exl3_standard.json"),
                dataset=DATASET, components=manifest["components"])


def candidate(snapshot, corpus, reference_repo):
    repo, ref = REPO, Path(reference_repo).resolve()
    if ref != repo:
        git(ref, "merge-base", "--is-ancestor", REFERENCE_HEAD, "HEAD")
    reference_files = committed_files(ref, *REFERENCE_FILES)
    code = committed_files(repo, "quantization/ernie55_flash", *SHARED_FILES)
    required = {"source", "prepare_corpus", "forward", "routing", "storage", "export", "identity"}
    if any(f"quantization/ernie55_flash/{name}.py" not in code for name in required):
        raise ValueError("incomplete ERNIE producer checkout")
    source = HeaderSource(snapshot)
    return dict(schema="ernie55-input-identity-v1", source_revision=BASE_REVISION,
                source_headers=source.audit["headers_sha256"], source_tree=source.audit["tree_sha256"],
                source_fingerprints=source.source_fingerprints,
                source_roles=source.audit["tensor_roles_sha256"], corpus=corpus_identity(Path(corpus)),
                code=code, fork=fork_identity(repo), reference_files=reference_files,
                reference_revision=git(ref, "log", "-1", "--format=%H", "--", *REFERENCE_FILES), recipe=RECIPE)


def verified_identity(inputs, vendor, local):
    """Verify preflights after the production worktree has rebased onto the merged reference."""
    from .forward import reference

    if Path(reference.__file__).resolve() != REPO / REFERENCE_FILES[0]:
        raise ValueError("production requires the merged in-tree reference import")
    current = committed_files(REPO, *REFERENCE_FILES)
    if current != inputs["reference_files"]:
        raise ValueError("reference differs from the in-tree producer")
    for path in current:
        if git(REPO, "rev-parse", "origin/dev:" + path) != git(REPO, "rev-parse", "HEAD:" + path):
            raise ValueError("reference differs from merged dev")
    for receipt, schema in ((vendor, "ernie55-vllm-ernie-parity-v1"), (local, "ernie55-quant-forward-parity-v1")):
        if (receipt.get("schema") != schema or receipt.get("status") != "passed"
                or receipt.get("source_revision") != BASE_REVISION
                or receipt.get("reference_files_sha256") != digest(current)):
            raise ValueError("parity receipt differs from the pinned source or merged reference")
        rows = receipt.get("measurements")
        expected = {"hca", "mla_dsa", "mhc", "dense", "latent_moe", "situ", "lm_head", "dspark"}
        if not isinstance(rows, dict) or not expected.issubset(rows):
            raise ValueError("missing measured parity bounds for ERNIE paths")
        for name, row in rows.items():
            value, bound = row["error"], row["bound"]
            if not all(type(x) in (int, float) and math.isfinite(x) for x in (value, bound)) or not 0 <= value <= bound:
                raise ValueError(f"parity bound failed: {name}")
    if (local.get("host") != "maxq" or local.get("input_identity_sha256") != digest(inputs)
            or local.get("vendor_receipt_sha256") != digest(vendor)):
        raise ValueError("maxq forward parity must bind the measured vendor gate and quant inputs")
    return dict(schema="ernie55-causal-mc-v1", inputs=inputs,
                vllm_preflight_sha256=digest(vendor), maxq_preflight_sha256=digest(local))


def check_diagnosis(failure, diagnosis, identity_key):
    if (failure.get("identity_sha256") != identity_key or diagnosis.get("identity_sha256") != identity_key
            or diagnosis.get("failure_sha256") != digest(failure)
            or not all(isinstance(diagnosis.get(key), str) and diagnosis[key].strip() for key in ("cause", "resolution"))):
        raise ValueError("resume requires a diagnosis of this exact producer failure and frozen identity")


@contextmanager
def producer_session(root, snapshot, corpus, vendor, local):
    """Hold across all numerical work. A SIGKILL leaves an identifiable interrupted attempt."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "producer.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "numerical-complete.json").exists():
            raise ValueError("numerical job already completed")
        identity = verified_identity(candidate(snapshot, corpus, REPO), vendor, local)
        frozen, attempt = root / "identity.json", root / "attempt.json"
        if frozen.exists():
            if read(frozen) != identity:
                raise ValueError("frozen numerical identity differs")
        elif any(path.name != "producer.lock" for path in root.iterdir()):
            raise ValueError("production state exists without a frozen identity")
        else:
            atomic_json(frozen, identity)
        key = digest(identity)
        if attempt.exists():
            prior = read(attempt)
            if prior["identity_sha256"] != key or prior["status"] not in ("running", "failed", "closed"):
                raise ValueError("invalid prior producer attempt")
            if prior["status"] != "closed":
                diagnosis = root / "diagnoses" / (digest(prior) + ".json")
                if not diagnosis.is_file():
                    raise ValueError("missing diagnosis for failed or interrupted producer attempt")
                check_diagnosis(prior, read(diagnosis), key)
                atomic_json(root / "failures" / (digest(prior) + ".json"), prior)
        active = dict(status="running", id=uuid.uuid4().hex, identity_sha256=key)
        atomic_json(attempt, active)
        try:
            yield identity
        except BaseException as exc:
            atomic_json(attempt, dict(active, status="failed", error=type(exc).__name__, detail=str(exc)))
            raise
        else:
            atomic_json(attempt, dict(active, status="closed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "reference-checkout", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.name in ("identity.json", "attempt.json", "numerical-complete.json"):
        parser.error("write candidate receipts outside frozen producer state")
    inputs = candidate(args.source, args.corpus, args.reference_checkout)
    atomic_json(args.receipt, dict(status="candidate", identity_sha256=digest(inputs), inputs=inputs,
                                  reference_checkout_head=git(args.reference_checkout, "rev-parse", "HEAD")))
