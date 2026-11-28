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
