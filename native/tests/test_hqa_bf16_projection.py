"""Selected Flash BF16 projections and first-use capture (sealed-slot GPU test)."""
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import os

import pytest


@pytest.mark.parametrize("n,k", [
    (16384, 1536), (8192, 1536),  # MLA q_b, whole and TP2 output slice
    (4096, 16384), (4096, 8192),  # MLA o, whole and TP2 input slice
    (4096, 4096),                # KDA o TP2 input slice (whole is 4096 x 8192)
])
@pytest.mark.parametrize("rows", [1, 17, 129])
def test_bf16_projection_first_capture_and_eager_stream(n, k, rows):
    torch = pytest.importorskip("torch")
    path = os.environ.get("CUTEAFD_NATIVE_LIB")
    if not path or not torch.cuda.is_available():
        pytest.skip("requires sealed native CUDA library")
    native = C.CDLL(path)
    for name, args in {
        "cuteafd_cuda_graph_begin_capture": [C.c_void_p],
        "cuteafd_cuda_graph_end_capture": [C.c_void_p, C.POINTER(C.c_void_p)],
        "cuteafd_cuda_graph_launch": [C.c_void_p, C.c_void_p],
        "cuteafd_cuda_graph_exec_destroy": [C.c_void_p],
        "cuteafd_cuda_linear_bf16_cublas_async": [C.c_void_p] * 4 + [C.c_size_t] * 3 + [C.c_void_p],
    }.items():
        function = getattr(native, name)
        function.argtypes = args
        function.restype = C.c_int32
    torch.manual_seed(n + k + rows)
    x = (torch.randn(rows, k, device="cuda") * .1).bfloat16()
    weight = (torch.randn(n, k, device="cuda") * .02).bfloat16()
    output = torch.empty((rows, n), device="cuda", dtype=torch.bfloat16)
    eager = torch.empty_like(output)
    capture_stream, eager_stream = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()

    def run():
        # A fresh host thread has no cached native cuBLAS handle. Its first GEMM
        # must be capturable without any eager native projection warm-up.
        torch.cuda.set_device(x.device)
        graph = C.c_void_p()

        def gemm(out, stream):
            assert native.cuteafd_cuda_linear_bf16_cublas_async(
                x.data_ptr(), weight.data_ptr(), None, out.data_ptr(),
                rows, k, n, stream.cuda_stream) == 0

        assert native.cuteafd_cuda_graph_begin_capture(capture_stream.cuda_stream) == 0
        try:
            gemm(output, capture_stream)
        finally:
            status = native.cuteafd_cuda_graph_end_capture(capture_stream.cuda_stream, C.byref(graph))
        assert status == 0 and graph.value
        try:
            for factor in (1.0, -.5):
                x.mul_(factor)
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                assert native.cuteafd_cuda_graph_launch(graph, capture_stream.cuda_stream) == 0
                capture_stream.synchronize()
                # Capture preparation must not leave the handle on the capture
                # stream when a later noncaptured call requests another stream.
                gemm(eager, eager_stream)
                eager_stream.synchronize()
                assert torch.equal(output.view(torch.int16), eager.view(torch.int16))
                old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                try:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                    reference = torch.mm(x, weight.t())
                finally:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old
                torch.testing.assert_close(output, reference, rtol=1 / 128, atol=1 / 512)
        finally:
            assert native.cuteafd_cuda_graph_exec_destroy(graph) == 0

    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(run).result()
