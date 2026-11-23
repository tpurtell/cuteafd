"""Hy4 MLA singleton vs speculative-width exactness on the pinned AOT kernel."""

import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("heads,window,splits", [(16, None, 16), (32, None, 16), (32, 512, 8), (64, 512, 8)])
@pytest.mark.parametrize("kv_format", ["bf16", "fp8"])
def test_hy4_mla_split_width_exact(heads, window, splits, kv_format):
    import torch

    assert torch.cuda.is_available(), "run this oracle in a coordinator GPU slot"
    fork = Path(__file__).resolve().parents[2] / "third_party" / "sparkinfer"
    sys.path.insert(0, str(fork))
    from b12x.integration.cuteafd.hy4_mla import (
        Hy4MlaGeometry,
        compile_hy4_mla_attention_aot,
        compile_hy4_mla_rank_norm_pack_aot,
        hy4_attention_layout,
    )

    torch.manual_seed(718)
    dev = "cuda"
    bf16 = torch.bfloat16
    geom = Hy4MlaGeometry(heads=heads, window_size=window, kv_cache=kv_format)
    page_size = 64 if window else 1
    cap = window or 2048
    # Include shallow history (empty tail splits) and an unaligned full window.
    width, end = 6, cap + 6 if window else cap
    positions = torch.arange(end, dtype=torch.int64, device=dev)
    normal = lambda *shape: torch.randn(shape, dtype=bf16, device=dev) * 0.25
    cache = torch.empty(
        ((end + 63) // 64, 64, geom.record_bytes if kv_format == "fp8" else 576),
        dtype=torch.uint8 if kv_format == "fp8" else bf16, device=dev,
    )
    inv_freq = 10000 ** (-torch.arange(32, dtype=torch.float32, device=dev) / 32)
    compile_hy4_mla_rank_norm_pack_aot(geom).launch(
        normal(end, 1024), normal(end, 576), torch.ones(1024, dtype=bf16, device=dev),
        torch.ones(512, dtype=bf16, device=dev), positions, positions, inv_freq,
        torch.empty((end, 1024), dtype=bf16, device=dev), cache, scalars=(end,),
    )
    prog = compile_hy4_mla_attention_aot(geom, max_rows=width, max_context=cap, mode="decode")
    scratch = torch.empty(
        hy4_attention_layout(geom, width, max_context=cap, mode="decode")["scratch_bytes"],
        dtype=torch.uint8, device=dev,
    )
    query = normal(width, heads, 576)
    table = torch.arange((end + page_size - 1) // page_size, dtype=torch.int32, device=dev)

    def launch(q, lengths, active_splits):
        rows, batch = len(q), len(lengths)
        tables = table.repeat(batch, 1)
        cu = [0, rows] if batch == 1 else list(range(rows + 1))
        out = torch.empty((rows, heads, 512), dtype=bf16, device=dev)
        lse = torch.empty((rows, heads), dtype=torch.float32, device=dev)
        prog.launch(
            q, cache, tables, torch.tensor(lengths, dtype=torch.int32, device=dev),
            torch.tensor(cu, dtype=torch.int32, device=dev), out, lse, scratch,
            scalars=(rows, batch, len(table), active_splits),
        )
        return out.view(torch.int16), lse.view(torch.int32)

    for active_splits in (1, 2, splits):
        for context in (width, end):
            lengths = list(range(context - width + 1, context + 1))
            batched = launch(query, [context] if window else lengths, active_splits)
            for i, length in enumerate(lengths):
                single = launch(query[i:i + 1], [length], active_splits)
                assert all(torch.equal(many[i], one[0]) for many, one in zip(batched, single)), (
                    heads, window, kv_format, active_splits, context, i
                )
