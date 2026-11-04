"""Native expert compaction with the existing shared EXL3 wire store."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
import _pinned_sparkinfer  # noqa: F401
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd.pointwise import _view, _add
from b12x.moe._shared.kernels.w4a16.reply_store import ReplyStore


class NativeReply:
    def __init__(self, width, format):
        assert width > 0 and width % 128 == 0
        assert format in (0, 1, 2)
        self.hidden_size = width
        self.reply_format = format
        if format:
            self.store = ReplyStore(width, format)
        self.reply_row_bytes = (width * 2 if format == 0 else
                                width + width // 32 if format == 1 else
                                width // 2 + width // 16)

    @cute.jit
    def __call__(self, source: cute.Pointer, out: cute.Pointer, rows: Int32,
                 topk: Int32, source_kind: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(source, out, topk, source_kind).launch(
                grid=(rows, self.hidden_size // 128, 1),
                block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, out: cute.Pointer,
               topk: Int32, source_kind: Int32):
        row, slab, _ = cute.arch.block_idx()
        lane = Int32(cute.arch.thread_idx()[0])
        col = Int32(slab) * Int32(128) + lane * Int32(4)
        h = Int64(self.hidden_size)
        fp32 = _view(cute.make_ptr(Float32, source.toint(), cute.AddressSpace.gmem, assumed_align=16))
        bf16 = _view(cute.make_ptr(BFloat16, source.toint(), cute.AddressSpace.gmem, assumed_align=16))
        values = cute.make_rmem_tensor((4,), Float32)
        for j in cutlass.range_constexpr(4):
            at = Int64(row) * h + Int64(col) + Int64(j)
            value = Float32(0.0)
            if source_kind == Int32(0):
                value = fp32[at]
            elif source_kind == Int32(1):
                # FP32 routes initialize from route zero, preserving signed zero.
                base = Int64(row) * Int64(topk) * h + Int64(col) + Int64(j)
                value = fp32[base]
                for route in cutlass.range(1, topk):
                    value = _add(value, fp32[base + Int64(route) * h])
            else:
                # BF16 routes start at +0, matching the existing native reducer.
                base = Int64(row) * Int64(topk) * h + Int64(col) + Int64(j)
                for route in cutlass.range(topk):
                    value = _add(value, Float32(bf16[base + Int64(route) * h]))
            values[j] = value
        if cutlass.const_expr(self.reply_format == 0):
            dst = _view(cute.make_ptr(BFloat16, out.toint(), cute.AddressSpace.gmem, assumed_align=16))
            for j in cutlass.range_constexpr(4):
                dst[Int64(row) * h + Int64(col) + Int64(j)] = BFloat16(values[j])
        else:
            # This shared store performs the same BF16 rounding as compact+codec.
            self.store.store(_view(out), Int32(row), Int32(slab), lane,
                values[0], values[1], values[2], values[3])


def compile_native_reply_aot(width, format):
    kernel = NativeReply(width, format)
    return compile_program(kernel, name=f"expert_reply_native_h{width}_{format}",
        operands=(Operand("source", torch.uint8, "[rows,source]"),
                  Operand("out", torch.uint8, f"[rows,{kernel.reply_row_bytes}]", "out")),
        scalars=tuple(Scalar(n) for n in ("rows", "topk", "source_kind")),
        key=(width, format), geometry=dict(width=width, format=format,
            row_bytes=kernel.reply_row_bytes,
            source_kinds=["fp32_tokens", "fp32_routes", "bf16_routes"]))
