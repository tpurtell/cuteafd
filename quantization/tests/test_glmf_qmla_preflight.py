from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm5_flash.qmla_preflight import check_selection, select_witnesses
from qmla.capture import SparseSelection


def test_witnesses_use_distinct_panel_records_and_exceed_the_pool_budget():
    records = [{"id": str(i), "input_ids": list(range(n))} for i, n in enumerate((120, 260, 2600))]
    rows = select_witnesses(records, 2048, 4)
    assert [len(ids) for _, ids in rows] == [64, 193, 2080]
    assert [name for name, _ in rows] == ["0", "1", "2"]
    with pytest.raises(ValueError, match="witness"):
        select_witnesses(records[:2], 2048, 4)


def test_sparse_witness_requires_exact_native_mask_including_tail():
    length = 8
    causal = torch.ones((length, length), dtype=torch.bool).tril()
    seq = SimpleNamespace(seq_len=length, allowed_mask=None, sparse_selection=None)
    assert not check_selection(seq, causal, True)["sparse_witness_passed"]
    groups = torch.arange(length).reshape(-1, 1)
    seq.sparse_selection = SparseSelection(groups, compress_ratio=1, candidate_topk=1)
    expected = torch.eye(length, dtype=torch.bool)
    stats = check_selection(seq, expected, True)
    assert stats["mask_matches"] and stats["pruned_rows"] == length - 1
    expected[-1, 0] = True
    assert not check_selection(seq, expected, True)["mask_matches"]


def test_fp64_oracle_uses_native_operands_and_only_selected_keys():
    from glm5_flash.qmla_preflight import fp64_reference
    import math

    generator = torch.Generator().manual_seed(53)
    length, heads, width = 7, 2, 3
    generate = lambda *shape: torch.randn(shape, generator=generator).bfloat16()
    q = generate(length, heads * width)
    kv = generate(length, heads * width * 2)
    projection = generate(5, heads * width)
    mask = torch.eye(length, dtype=torch.bool)
    mask[2:, 0] = True
    geometry = SimpleNamespace(heads=heads, qk_nope_dim=width, v_head_dim=width, effective_scale=width**-0.5)
    result = fp64_reference(q, kv, mask, projection, geometry)
    k, v = kv.double().reshape(length, heads, width * 2).chunk(2, dim=-1)
    expected = []
    for row in range(length):
        visible = mask[row]
        score = torch.einsum("hd,khd->kh", q[row].double().reshape(heads, width), k[visible]) / math.sqrt(width)
        mixed = (score.softmax(0).unsqueeze(-1) * v[visible]).sum(0)
        expected.append(projection.double() @ mixed.flatten())
    torch.testing.assert_close(result, torch.stack(expected), rtol=1e-12, atol=1e-12)
