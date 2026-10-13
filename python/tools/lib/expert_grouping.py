"""Pure launch-group metadata for the installed native expert bridges."""
from __future__ import annotations

import ast
from pathlib import Path


def slice_group_rows(source: Path) -> int:
    """Read the pinned route planner and compute tile, without importing CUDA."""
    kernels = source / "b12x/moe/_shared/kernels"
    planner = ast.parse((kernels / "v41_route_plan.py").read_text())
    groups = {
        node.right.value for node in ast.walk(planner)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv)
        and isinstance(node.left, ast.Name) and node.left.id in ("rank", "j")
        and isinstance(node.right, ast.Constant)
    }
    compute = ast.parse((kernels / "w4a8_v41_slice.py").read_text())
    tiles = {
        node.value.args[1].args[0].elts[0].value for node in ast.walk(compute)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "mid"
        and isinstance(node.value, ast.Call) and len(node.value.args) == 2
        and isinstance(node.value.args[1], ast.Call)
        and isinstance(node.value.args[1].args[0], ast.Tuple)
        and isinstance(node.value.args[1].args[0].elts[0], ast.Constant)
    }
    if len(groups) != 1 or groups != tiles or next(iter(groups)) not in (8, 16, 32, 64, 128):
        raise ValueError("pinned slice route/compute grouping is not representable")
    return next(iter(groups))


def native_group_macros(variants: list[dict]) -> list[str]:
    """rows=0 queries the compiled tile; positive rows select the executed path."""
    terms = []
    for variant in variants:
        capacity = variant["capacity_rows"]
        group = variant["group_rows"]
        cutoff = variant.get("compact_live_rows")
        compact = variant.get("compact_group_rows")
        if type(capacity) is not int or capacity < 1 or type(group) is not int or group not in (1, 8, 16, 32, 64, 128):
            raise ValueError("invalid native expert group metadata")
        if (cutoff is None) != (compact is None):
            raise ValueError("incomplete compact expert group metadata")
        value = f"{group}u"
        if cutoff is not None:
            if type(cutoff) is not int or not 1 <= cutoff <= capacity or compact != 1:
                raise ValueError("invalid compact expert group metadata")
            value = f"((rows)>0 && (rows)<={cutoff} ? {compact}u : {group}u)"
        terms.append(f"((capacity)=={capacity}) ? {value} : ")
    return ["#define CUTEAFD_V41_GROUP_ROWS(capacity, rows) (" + "".join(terms) + "0u)"]
