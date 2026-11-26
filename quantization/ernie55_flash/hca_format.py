"""Checkpoint representation of ERNIE's independently scaled HCA output groups."""
from contextlib import ExitStack
from pathlib import Path

from core_export import MUL1, packed_specs

HCA_OUTPUT = ".self_attn.linear_o_group_proj"


def group_modules(parent, config):
    if not parent.endswith(HCA_OUTPUT) or config["o_groups"] != 8:
        raise ValueError("unsupported ERNIE HCA output grouping")
    return tuple(f"{parent}.groups.{i}" for i in range(config["o_groups"]))


def original_selection(compact, config):
    """Reassemble the eight contiguous output-row ranges into one native geometry."""
    result, groups = {}, {}
    for module, geometry in compact.items():
        parent, _, _ = module.rpartition(".groups.")
        if parent.endswith(HCA_OUTPUT):
            groups.setdefault(parent, {})[module] = geometry
        else:
            result[module] = geometry
    for parent, children in groups.items():
        names = group_modules(parent, config)
        if parent in result or children.keys() != set(names):
            raise ValueError(f"incomplete or mixed ERNIE HCA groups: {parent}")
        first = children[names[0]]
        if (first["out_features"] != config["o_lora_rank"]
                or any(geometry != first for geometry in children.values())):
            raise ValueError(f"unequal ERNIE HCA group geometry: {parent}")
        height = len(names) * first["out_features"]
        result[parent] = dict(first, out_features=height, stored_out_features=height)
    return result


def read_group_bundles(snapshot, mapping, parent, config, bits):
    """Read the selected compact index into eight ordinary MUL1 bundles in row order."""
    import torch
    from safetensors import safe_open

    if bits not in (6, 8):
        raise ValueError("HCA compact core requires K6 or K8")
    names = group_modules(parent, config)
    heads_per_group, remainder = divmod(config["num_attention_heads"], len(names))
    if remainder:
        raise ValueError("HCA heads must divide evenly across output groups")
    inputs = heads_per_group * config["head_dim"]
    rows = config["o_lora_rank"]
    geometry = dict(in_features=inputs, out_features=rows, stored_out_features=rows,
                    bits=bits, codebook="mul1")
    dtype_map = {"I16": torch.int16, "I32": torch.int32, "F16": torch.float16}
    prefix = parent + "."
    expected = {key for name in names for key in packed_specs(name, geometry)}
    if parent in mapping or {key for key in mapping if key.startswith(prefix)} != expected:
        raise ValueError(f"incomplete or mixed compact HCA index: {parent}")
    with ExitStack() as stack:
        handles = {}
        for name in names:
            bundle = {}
            for key, (dtype, shape) in packed_specs(name, geometry).items():
                shard = mapping[key]
                if not shard.endswith(".safetensors") or Path(shard).name != shard:
                    raise ValueError(f"invalid HCA index shard: {shard}")
                if shard not in handles:
                    handles[shard] = stack.enter_context(safe_open(Path(snapshot) / shard, framework="pt"))
                tensor = handles[shard].get_tensor(key)
                if tensor.dtype != dtype_map[dtype] or list(tensor.shape) != shape:
                    raise ValueError(f"invalid HCA compact tensor: {key}")
                bundle[key.rsplit(".", 1)[1]] = tensor
            if bundle["mul1"].item() & 0xffffffff != MUL1:
                raise ValueError(f"invalid MUL1 multiplier: {name}")
            yield bundle


def load_grouped_weight(snapshot, mapping, parent, config, bits, *, device):
    """Reconstruct the reference's [groups * rank, width] BF16 matrix from a compact view."""
    import torch
    from inkling2.numerical import reconstruct_projection

    rank = config["o_lora_rank"]
    groups = read_group_bundles(snapshot, mapping, parent, config, bits)
    result = None
    for i, packed in enumerate(groups):
        width = packed["suh"].numel()
        weight = reconstruct_projection(packed, [rank, width], codebook="mul1",
                                        dtype=torch.bfloat16, device=device)
        if result is None:
            result = weight.new_empty((config["o_groups"] * rank, width))
        result.narrow(0, i * rank, rank).copy_(weight)
    return result
