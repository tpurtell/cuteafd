"""ctypes bindings for official TP4 and local dSpark expert qualification."""

import ctypes as C
from spark_packages import native_spark_artifacts
from mxfp4_layouts import stored_width

P = C.c_void_p
I = C.c_int32
U = C.c_uint32
L = C.c_uint64


class Info(C.Structure):
    _fields_ = (
        [
            (n, U)
            for n in [
                "abi_version",
                "role",
                "experts",
                "hidden_size",
                "logical_intermediate",
                "kernel_intermediate",
                "topk",
                "capacity_rows",
            ]
        ]
        + [("scratch_bytes", L)]
        + [
            (n, I)
            for n in [
                "max_rows",
                "rows_padded",
                "max_tasks",
                "max_phys_tiles",
                "max_active_clusters",
            ]
        ]
        + [("input_dtype", U)]
    )


class Launch(C.Structure):
    _fields_ = (
        [("tensors", P * 44)]
        + [
            (n, I)
            for n in [
                "num_tokens",
                "max_rows",
                "scatter_rows",
                "rows_padded",
                "max_tasks",
                "max_phys_tiles",
                "max_active_clusters",
            ]
        ]
        + [("stream", P)]
    )


assert C.sizeof(Info) == 64 and C.sizeof(Launch) == 392


# Only the five per-family launch-ABI entry points are namespaced. The packer
# size query and the packer itself are canonical symbols shared by every family.
FAMILY_SYMBOLS = frozenset({
    "cuteafd_v41_expert_info",
    "cuteafd_v41_expert_initialize",
    "cuteafd_v41_expert_bind_scratch",
    "cuteafd_v41_expert_initialize_scratch_async",
    "cuteafd_v41_expert_launch",
})


def namespaced_symbol(name, prefix):
    """Family symbol for the launch-ABI entry points; canonical for everything else."""
    if prefix != "cuteafd_v41_expert_" and name in FAMILY_SYMBOLS:
        return name.replace("cuteafd_v41_expert_", prefix, 1)
    return name


def expert_symbol_prefix(*, local=False, tp2=False, artifact=None, family="v41"):
    """Select one executable artifact, never a scalar width for a whole world."""
    assert family in ("v41", "dsv4f", "dsv4p")
    assert sum(bool(x) for x in (local, tp2, artifact is not None)) <= 1
    if artifact is not None:
        assert artifact in {a for world in native_spark_artifacts(family).values() for a in world}, artifact
        return artifact.prefix(family)
    prefix = ("cuteafd_v41_tp2_expert_" if tp2 else
              "cuteafd_v41_local_expert_" if local else "cuteafd_v41_expert_")
    return prefix.replace("v41", family, 1)


def library(path, *, local=False, tp2=False, artifact=None, family="v41"):
    prefix = expert_symbol_prefix(local=local, tp2=tp2, artifact=artifact, family=family)
    lib = C.CDLL(path)
    lib._cuteafd_family = family
    for name, args in {
        "cuteafd_v41_expert_info": [I, C.POINTER(Info)],
        "cuteafd_v41_expert_initialize": [I, C.POINTER(P)],
        "cuteafd_v41_expert_bind_scratch": [P, P, L, C.POINTER(P)],
        "cuteafd_v41_expert_initialize_scratch_async": [P, P, L, P],
        "cuteafd_v41_expert_launch": [P, C.POINTER(Launch)],
        "cuteafd_set_expert_hidden": [U],
        "cuteafd_expert_packed_sizes": [U, C.POINTER(L)],
        "cuteafd_pack_expert_async": [C.POINTER(P), C.POINTER(P), U, P],
        "cuteafd_compact_routes_bf16_async": [P, P, U, P],
        "cuteafd_compact_tokens_bf16_async": [P, P, U, P],
    }.items():
        selected = namespaced_symbol(name, prefix)
        fn = getattr(lib, selected)
        if selected != name:
            setattr(lib, name, fn)
        fn.argtypes = args
        fn.restype = I
    if prefix != "cuteafd_v41_expert_":
        lib.cuteafd_v41_expert_output_kind = getattr(lib, prefix + "output_kind")
    return lib


def check(code):
    assert code == 0, code


def pack_reference(weights, scales, *, stored, coded=True):
    """Independent layout oracle for the selected artifact's resident planes."""
    import torch
    from tests.moe.test_v41_exact_slices import pack
    from b12x.moe._shared.scale_codes import pack_scale_codes

    logical = weights["w1"].shape[1]
    exact = stored == logical and logical % 128 != 0
    assert stored == (logical if exact else (logical + 127) // 128 * 128)
    packed = pack(weights, scales, exact)
    if coded:
        gated = torch.cat([torch.nn.functional.pad(scales[name], (0, 0, 0, stored - logical))
                           for name in ("w3", "w1")], 1)
        packed[1] = pack_scale_codes(gated, tensor_name="oracle.w13.scale",
                                     gated_half_rows=stored).view(torch.uint32).flatten().cuda()
        packed[3] = pack_scale_codes(scales["w2"], tensor_name="oracle.w2.scale").view(torch.uint32).flatten().cuda()
    return packed


class Native:
    def __init__(self, lib, capacity, weights, wire, ids, routing, *, coordinator=False, full_backbone=False, tp2=False, artifact=None, storage=None, geometry=None):
        import torch
        assert sum((coordinator, full_backbone, tp2, artifact is not None)) <= 1
        self.lib = lib
        self.info = info = Info()
        self.handle = P()
        check(lib.cuteafd_v41_expert_info(capacity, C.byref(info)))
        family = lib._cuteafd_family
        hidden, experts = {"v41": (5120, 384), "dsv4f": (4096, 256), "dsv4p": (7168, 384)}[family]
        expected = ((artifact.role, experts, hidden, artifact.logical, artifact.stored, 6, capacity, 7) if artifact else
                    (3, 384, 5120, 1152, 1152, 6, capacity, 7) if tp2 else
                    (0, 128, 5120, 2304, 2304, 3, capacity, 1)
                    if coordinator else (2, 384, 5120, 2304, 2304, 6, capacity, 7)
                    if full_backbone else (1, 384, 5120, 576, 640, 6, capacity, 7))
        if geometry is not None:
            assert not any((coordinator, full_backbone, tp2, artifact is not None))
            hidden, intermediate, experts, topk = geometry
            expected = (1, experts, hidden, intermediate, stored_width(hidden, intermediate), topk, capacity, 7)
        assert (
            info.abi_version,
            info.role,
            info.experts,
            info.hidden_size,
            info.logical_intermediate,
            info.kernel_intermediate,
            info.topk,
            info.capacity_rows,
            info.input_dtype,
        ) == (info.abi_version, *expected)
        assert info.abi_version in (2, 3)
        self.token_accumulation = info.abi_version == 3
        if self.token_accumulation:
            query = lib.cuteafd_v41_expert_output_kind
            query.argtypes = [I, C.POINTER(U)]
            query.restype = I
            kind = U(99)
            check(query(capacity, C.byref(kind)))
            assert kind.value == 1
        check(lib.cuteafd_v41_expert_initialize(capacity, C.byref(self.handle)))
        if storage is not None:
            assert storage.dtype == torch.uint8 and storage.is_cuda and storage.is_contiguous()
            assert storage.numel() >= info.scratch_bytes
        self.storage = (torch.empty(info.scratch_bytes, device="cuda", dtype=torch.uint8)
                        if storage is None else storage)
        self.args = Launch()
        slots = self.args.tensors
        check(
            lib.cuteafd_v41_expert_bind_scratch(
                self.handle, self.storage.data_ptr(), self.storage.numel(), slots
            )
        )
        w13, s13, w2, s2 = [x.data_ptr() for x in weights]
        for slot, pointer in [
            (22, w13),
            (23, s13),
            (24, w2),
            (25, s2),
            (26, s13),
            (27, s2),
            (28, slots[34]),
            (29, slots[34]),
            (30, w13),
            (31, s13),
            (32, w2),
            (33, s2),
            (38, slots[37]),
            (39, slots[37]),
            (0, wire.data_ptr()),
            (1, ids.data_ptr()),
            (2, routing.data_ptr()),
        ]:
            slots[slot] = pointer
        check(
            lib.cuteafd_v41_expert_initialize_scratch_async(
                self.handle,
                self.storage.data_ptr(),
                self.storage.numel(),
                torch.cuda.current_stream().cuda_stream,
            )
        )
        for name in [
            "max_rows",
            "rows_padded",
            "max_tasks",
            "max_phys_tiles",
            "max_active_clusters",
        ]:
            setattr(self.args, name, getattr(info, name))
        offset = slots[41] - self.storage.data_ptr()
        output_rows = capacity if self.token_accumulation else capacity * info.topk
        self.output = (
            self.storage[offset : offset + output_rows * info.hidden_size * 4]
            .view(torch.float32)
            .reshape(output_rows, info.hidden_size)
        )

    def run(self, rows):
        import torch
        self.args.num_tokens = rows
        self.args.scatter_rows = rows * self.info.topk
        self.args.stream = torch.cuda.current_stream().cuda_stream
        check(self.lib.cuteafd_v41_expert_launch(self.handle, C.byref(self.args)))
