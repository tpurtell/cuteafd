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
    # SWA queries cross the physical ring wrap. Only the final 640 records
    # are written, so loading the wrapped fixture never races two stores.
    width, end = 6, 646 if window else cap
    ring = 640 if window else cap
    start = end - ring if window else 0
    positions = torch.arange(start, end, dtype=torch.int64, device=dev)
    normal = lambda *shape: torch.randn(shape, dtype=bf16, device=dev) * 0.25
    cache = torch.empty(
        ((ring + 63) // 64, 64, geom.record_bytes if kv_format == "fp8" else 576),
        dtype=torch.uint8 if kv_format == "fp8" else bf16, device=dev,
    )
    inv_freq = 10000 ** (-torch.arange(32, dtype=torch.float32, device=dev) / 32)
    compile_hy4_mla_rank_norm_pack_aot(geom).launch(
        normal(ring, 1024), normal(ring, 576), torch.ones(1024, dtype=bf16, device=dev),
        torch.ones(512, dtype=bf16, device=dev), positions.remainder(ring), positions, inv_freq,
        torch.empty((ring, 1024), dtype=bf16, device=dev), cache, scalars=(ring,),
    )
    prog = compile_hy4_mla_attention_aot(geom, max_rows=width, max_context=cap, mode="decode")
    scratch = torch.empty(
        hy4_attention_layout(geom, width, max_context=cap, mode="decode")["scratch_bytes"],
        dtype=torch.uint8, device=dev,
    )
    query = normal(width, heads, 576)
    table = torch.arange((end + page_size - 1) // page_size, dtype=torch.int32, device=dev).remainder(ring // page_size)

    def launch(q, lengths, active_splits):
        rows, batch = len(q), len(lengths)
        tables = table.repeat(batch, 1)
        cu = [0, rows] if batch == 1 else list(range(rows + 1))
        out = torch.empty((rows, heads, 512), dtype=bf16, device=dev)
        lse = torch.empty((rows, heads), dtype=torch.float32, device=dev)
        # 0xff is NaN in both split planes. Any stale partial read must
        # survive neither a narrower launch nor empty tail splits.
        scratch.fill_(0xff)
        prog.launch(
            q, cache, tables, torch.tensor(lengths, dtype=torch.int32, device=dev),
            torch.tensor(cu, dtype=torch.int32, device=dev), out, lse, scratch,
            scalars=(rows, batch, len(table), active_splits),
        )
        return out.view(torch.int16), lse.view(torch.int32)

    contexts = (64, 65, end - width, end) if window else (width, 64, 65, 513, 1290, end)
    errors = []
    for context in contexts:
        lengths = list(range(context - width + 1, context + 1))
        batch_lengths = [context] if window else lengths
        expected_o, expected_lse = launch(query, batch_lengths, 1)
        expected_o = expected_o.view(bf16).float()
        expected_lse = expected_lse.view(torch.float32)
        for active_splits in (1, 2, splits):
            batched = launch(query, batch_lengths, active_splits)
            for i, length in enumerate(lengths):
                single = launch(query[i:i + 1], [length], active_splits)
                assert all(torch.equal(many[i], one[0]) for many, one in zip(batched, single)), (
                    heads, window, kv_format, active_splits, context, i
                )
            out, lse = batched[0].view(bf16).float(), batched[1].view(torch.float32)
            o_max = (out - expected_o).abs().max().item()
            lse_max = (lse - expected_lse).abs().max().item()
            rel_l2 = ((out - expected_o).norm() / expected_o.norm()).item()
            print(f"heads={heads} window={window} format={kv_format} context={context} "
                  f"splits={active_splits} max_o={o_max:.6g} rel_l2={rel_l2:.6g} max_lse={lse_max:.6g}")
            if not (lse_max <= 2e-4 and rel_l2 <= 0.01 and o_max <= 0.025):
                errors.append((context, active_splits, o_max, rel_l2, lse_max))
    assert not errors, errors
