from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc.experts import ExpertBank, memory_plan, raw_covariance, route_batch


def router_fixture():
    scores = torch.linspace(-0.8, 1.0, 20).repeat(4, 1)
    bias = torch.arange(20, dtype=torch.float32).remainder(4) * 0.2

    class Gate:
        e_score_correction_bias = bias

        def __call__(self, x):
            logits = x.float()
            native = (logits.sigmoid() + bias).topk(8, sorted=False).indices
            gamma = logits.sigmoid().gather(1, native)
            return logits, (gamma / gamma.sum(1, keepdim=True)) * 2.5, native

    moe = SimpleNamespace(gate=Gate(), config=SimpleNamespace(
        n_group=1, topk_group=1, num_experts_per_tok=8, norm_topk_prob=True, routed_scaling_factor=2.5))
    return moe, scores


def test_native_routes_and_hypothetical_gamma():
    moe, x = router_fixture()
    ids, weights, near, gamma = route_batch(moe, x)
    _, expected_weights, expected_ids = moe.gate(x)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)
    ordered = (x.sigmoid() + moe.gate.e_score_correction_bias).argsort(descending=True)
    assert torch.equal(near, ordered[:, 8:16])
    for i in range(8):
        hypothetical = torch.cat((ordered[:, :7], ordered[:, 8+i:9+i]), dim=1)
        scores = x.sigmoid().gather(1, hypothetical)
        assert torch.equal(gamma[:, i], scores[:, -1] / scores.sum(1) * 2.5)


def test_ordered_recovery_and_replay():
    count = 1026
    x = torch.arange(count * 2, dtype=torch.float32).view(count, 2) / 7
    # Expert 8 ranks tenth on even rows and ninth on odd rows. Expert 16
    # has neither natural nor candidate observations.
    native = torch.arange(8).expand(count, -1)
    near = torch.arange(8, 16).repeat(count, 1)
    near[::2, :2] = torch.tensor([9, 8])
    weights = torch.linspace(0.15, 0.85, 8).expand(count, -1)
    data = x, native, weights, near, weights / 2
    bank = ExpertBank([tuple(t[:123] for t in data), tuple(t[123:] for t in data)],
                      tokens=count, width=2, experts=17, device="cpu")
    rows, gamma, natural, identity = bank.selection(8)
    expected = torch.cat((torch.arange(1, count, 2), torch.arange(0, count, 2)))[:1024]
    assert natural == identity == 0 and torch.equal(rows, expected)
    assert torch.equal(gamma[:513], (weights[:513, 0] / 2))
    assert torch.equal(gamma[513:], (weights[:511, 1] / 2))
    workspace = torch.empty(77, 2)
    actual = [value.clone() for value in bank.chunks(rows, workspace)]
    assert torch.equal(torch.cat(actual), x[expected])
    receipts = list(bank.receipts())
    assert receipts[8]["augmented_rank_counts"] == [513, 511, 0, 0, 0, 0, 0, 0]
    assert receipts[16]["identity_rows"] == receipts[16]["effective_rows"] == 1024
    assert receipts[0]["natural_rows"] == count
    assert receipts == list(bank.receipts())
    accumulator = SimpleNamespace(count=0, snapshot=lambda: torch.zeros(2, 2))
    filled = raw_covariance(accumulator, receipts[16]["identity_rows"])
    assert torch.equal(filled["H"], torch.eye(2) * 1024) and filled["count"] == 1024

    def outputs(expert, selected):
        for part in bank.chunks(selected, workspace):
            yield (part * (expert + 1)).bfloat16()

    expected = torch.zeros_like(x)
    for expert in range(8):
        expected += (x * (expert + 1)).bfloat16().float() * weights[:, expert, None]
    assert torch.equal(bank.combine(outputs), expected)
    with pytest.raises(ValueError, match="incomplete"):
        bank.combine(lambda expert, rows: ())


def test_memory_plan_includes_rolling_frontier_and_headroom():
    gib = 2**30
    plan = memory_plan(1_590_000, 4096, frontier_bytes=48*gib, weight_bytes=14*gib,
                       hessian_bytes=gib, workspace_bytes=2*gib,
                       mem_available=117*gib, container_remaining=100*gib)
    assert plan["owners"]["input_bank"] == plan["owners"]["output_sum"] == 1_590_000 * 4096 * 4
    assert plan["available"] == 100*gib and not plan["admitted"]
