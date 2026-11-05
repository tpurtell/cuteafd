"""CPU-only final checkpoint reload and producer completion audit."""
from __future__ import annotations

import json
from pathlib import Path

from kimi_k3.uniform_k2.storage import atomic_json
from .export import audit_export
from .recipe import header_source, reviewed_modules, MODULE_COUNT


def audit_completed_export(root, source=None):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    complete = json.loads((root / "quantization-complete.json").read_text())
    identity = plan["identity"]
    if complete.get("identity") != identity:
        raise ValueError("completion marker belongs to a different producer")
    if source is None:
        source = header_source(plan.get("source", identity.get("source")))
    export = Path(plan["export"])
    weight_map, total = audit_export(source, reviewed_modules(), export, identity)
    diagnostics = json.loads((export / "quantization-errors/index.json").read_text())
    if diagnostics.get("records") != MODULE_COUNT or sum(diagnostics.get("bundles", {}).values()) != MODULE_COUNT:
        raise ValueError("final diagnostic bundle inventory differs")
    result = dict(status="passed", identity=identity, tensors=len(weight_map), payload_bytes=total,
                  method="independent CPU header reload and committed native-byte audit receipts")
    atomic_json(root / "finalization-audit.json", result)
    return result
