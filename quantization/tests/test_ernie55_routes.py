"""Top-10 ties, rank-major selection, learned scales and raw identity witnesses."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash.routing import ErnieRoutes, rank_routes, recovered_covariance


def test_candidate_gamma_substitutes_at_last_rank_with_learned_scale():
    scores = torch.linspace(.1, 1.8, 25).repeat(2, 1)
    bias = torch.linspace(2., 0., 25)
    scales = torch.linspace(-2., 2., 25)
    result = rank_routes(scores, bias, scales)
    order = (scores + bias)[0].argsort(descending=True)
    ids = result["ids"]
    assert set(ids[0].tolist()) == set(order[:10].tolist())
    expected_native = scores.gather(1, ids)
    expected_native /= expected_native.sum(1, keepdim=True)
    torch.testing.assert_close(result["gamma"], expected_native * scales[ids], rtol=0, atol=0)
    for column, candidate in enumerate(order[10:20]):
        proposed = torch.cat((scores[0, order[:9]], scores[0, candidate][None]))
        assert result["candidate_ids"][0, column] == candidate
        torch.testing.assert_close(result["candidate_gamma"][0, column],
                                   proposed[-1] / proposed.sum() * scales[candidate], rtol=0, atol=0)


def test_top10_ties_are_disjoint_and_all_zero_gammas_remain_finite():
    records = rank_routes(torch.zeros(2, 25), torch.zeros(25), torch.ones(25))
    assert torch.cat((records["ids"], records["candidate_ids"]), dim=1)[0].unique().numel() == 20
    assert torch.equal(records["gamma"], torch.zeros(2, 10))
    assert torch.equal(records["candidate_gamma"], torch.zeros(2, 10))


def route_batch(rows, *, offset=0):
    # Expert 10 receives rank 11 on even token ids, rank 12 on odd ids.
    near = torch.arange(10, 20, dtype=torch.int32).repeat(rows, 1)
    swap = (torch.arange(rows) + offset) % 2 == 1
    near[swap] = near[swap][:, [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]]
    return dict(latent=torch.ones(rows, 2), ids=torch.arange(10).repeat(rows, 1),
                gamma=torch.ones(rows, 10), candidate_ids=near, candidate_gamma=-near.float())


def test_cold_hot_identity_only_across_sealed_batches():
    routes = ErnieRoutes([(0, route_batch(31)), (1, route_batch(1000, offset=31))], 21, 1031)
    hot, _, proof = routes.calibration(0)
    assert hot.tolist() == list(range(1031))
    assert (proof["natural_rows"], proof["augmented_rows"], proof["identity_rows"]) == (1031, 0, 0)
    cold, gamma, proof = routes.calibration(10)
    assert cold.tolist() == (list(range(0, 1031, 2)) + list(range(1, 1031, 2)))[:1024]
    assert gamma.tolist() == [-10.] * 1024
    assert proof["augmented_rank_counts"][:2] == [516, 508]
    assert len(routes.expert(10)[0]) == 0
    _, _, identity_only = routes.calibration(20)
    assert identity_only["identity_rows"] == 1024
    assert proof["selection_xxh3_128"] != identity_only["selection_xxh3_128"]


def test_add_identity_to_raw_covariance_before_normalization():
    raw = torch.tensor([[1., 2.], [2., 4.]])
    acc = SimpleNamespace(count=1, snapshot=raw.clone)
    evidence = dict(natural_rows=0, augmented_rows=1, identity_rows=1023, effective_rows=1024)
    result = recovered_covariance(acc, evidence)
    assert torch.equal(result["H"], raw + 1023 * torch.eye(2))
    assert result["count"] == 1024 and result["finalized"] is False
    with pytest.raises(ValueError, match="count"):
        recovered_covariance(acc, evidence | {"natural_rows": 1})


def test_reject_overlapping_ranks_and_misordered_batches():
    sample = route_batch(1)
    sample["candidate_ids"][0, 9] = 0
    with pytest.raises(ValueError, match="duplicate"):
        ErnieRoutes([(0, sample)], 21, 1)
    with pytest.raises(ValueError, match="order"):
        ErnieRoutes([(1, route_batch(1))], 21, 1)
