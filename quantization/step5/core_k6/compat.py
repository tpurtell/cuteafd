"""Exact producer identities and the explicitly audited Step-5 XTX resume lineage.

Only the declared K128 FP32 accumulation repair may cross identities. Tensor
checksums, input identities, bit widths and every other recipe choice remain
strict; this module never rewrites a receipt or changes numerical evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from kimi_k3.uniform_k2.storage import canonical

ACCUMULATION = "step5-native-fp32-XTX-k128-v1"
_ADMIN = frozenset(("schema", "code", "gptqmodel_commit", "predecessor_identities",
                   "resume_lineage_audit_sha256", "resume_lineage_manifest_sha256"))
_REQUIRED = frozenset(("source_repo", "source_revision", "source_config_sha256",
                       "source_index_sha256", "token_manifest_sha256", "calibration", "packages", "recipe"))
_SHA = re.compile(r"[0-9a-f]{64}")


class ProducerLineage(dict):
    """Manifest data with digest caches scoped to this verified read only."""
    pass


def identity_digest(identity):
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def _compatible_semantics(candidate, current):
    if not isinstance(candidate, dict) or not isinstance(current, dict):
        return False
    if not _REQUIRED <= candidate.keys() or not _REQUIRED <= current.keys():
        return False
    prefix = "cuteafd-step5-k4-core-k6-production-"
    if current.get("schema") != prefix + "v2" or candidate.get("schema") not in (prefix + "v1",prefix + "v2"):
        return False
    left = {k: v for k, v in candidate.items() if k not in _ADMIN | {"recipe"}}
    right = {k: v for k, v in current.items() if k not in _ADMIN | {"recipe"}}
    if left != right:
        return False
    old, new = candidate["recipe"], current["recipe"]
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if new.get("hessian_accumulation") != ACCUMULATION:
        return False
    if "hessian_accumulation" in old and old["hessian_accumulation"] != ACCUMULATION:
        return False
    if {k:v for k,v in old.items() if k != "hessian_accumulation"} != {
            k:v for k,v in new.items() if k != "hessian_accumulation"}:
        return False
    return True


def accepts_identity(candidate, current):
    """Accept exact equality or an explicit, semantically identical predecessor.

    Audit/manifest digests bind the declaration; callers at admission and export
    must verify the corresponding files, not merely trust these hash strings.
    """
    if candidate == current:
        return True
    if not isinstance(current, dict) or not isinstance(candidate, dict):
        return False
    if any(not isinstance(current.get(k), str) or not _SHA.fullmatch(current[k]) for k in
           ("resume_lineage_audit_sha256", "resume_lineage_manifest_sha256")):
        return False
    predecessors = current.get("predecessor_identities")
    if not isinstance(predecessors, list) or candidate not in predecessors:
        return False
    return _compatible_semantics(candidate, current)


def load_module_origins(root, current):
    """Read the immutable, hash-bound origin manifest once at final assembly."""
    if not isinstance(current, dict) or not current.get("predecessor_identities"):
        return None
    predecessors = current["predecessor_identities"]
    if not isinstance(predecessors, list) or not all(accepts_identity(p,current) for p in predecessors):
        raise ValueError("unreviewed producer predecessor identity")
    audit = Path(root) / "resume-lineage-audit.json"
    if (audit.is_symlink() or not audit.is_file()
            or hashlib.sha256(audit.read_bytes()).hexdigest() != current.get("resume_lineage_audit_sha256")):
        raise ValueError("resume lineage audit is missing, unsafe or has a different checksum")
    path = Path(root) / "producer-lineage.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("producer lineage manifest is missing or unsafe")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != current.get("resume_lineage_manifest_sha256"):
        raise ValueError("producer lineage manifest checksum differs")
    lineage = json.loads(raw)
    if (lineage.get("predecessor_identities") != predecessors
            or lineage.get("resume_lineage_audit_sha256") != current["resume_lineage_audit_sha256"]):
        raise ValueError("producer lineage manifest identity/audit differs")
    origins = lineage.get("module_origins")
    known = {identity_digest(p) for p in predecessors}
    if not isinstance(origins, dict) or any(not isinstance(n,str) or digest not in known for n,digest in origins.items()):
        raise ValueError("producer lineage module origin is not a declared predecessor")
    verified = ProducerLineage(lineage)
    verified.active_digest = identity_digest(current)
    verified.known_digests = known | {verified.active_digest}
    return verified


def module_producer_digest(module, current, lineage=None):
    """Legacy origins come only from the hash-bound map; absent entries are new."""
    if lineage is None:
        if isinstance(current,dict) and current.get("predecessor_identities"):
            raise ValueError("resumed producer requires its verified module origin manifest")
        return identity_digest(current)
    active = lineage.active_digest if isinstance(lineage,ProducerLineage) else identity_digest(current)
    known = lineage.known_digests if isinstance(lineage,ProducerLineage) else {
        active, *(identity_digest(p) for p in current["predecessor_identities"])}
    digest = lineage["module_origins"].get(module, active)
    if digest not in known:
        raise ValueError("unreviewed module producer origin")
    return digest
