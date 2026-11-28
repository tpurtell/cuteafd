from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc.experts import ExpertBank, memory_plan, route_batch


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


def test_fit_and_replay_observe_reconstructed_clamped_weighted_down(monkeypatch):
    from glm5_flash.mul1_mc import experts
    from glm5_flash.mul1_mc import numerics

    width, intermediate, count = 4, 3, 6
    inputs = torch.arange(count * width).reshape(count, width).bfloat16()
    ids = torch.arange(8).repeat(count, 1)
    near = torch.arange(8, 16).repeat(count, 1)
    # Expert zero has natural, candidate and identity rows.
    ids[-2:, 0], near[-2:, 0] = 8, 0
    gamma = torch.arange(count * 8).reshape(count, 8).float() / 16
    bank = ExpertBank([(inputs, ids, gamma, near, gamma / 2)], tokens=count, experts=16,
                      width=width, device="cpu")
    workspace = experts.ExpertWorkspace.allocate(width, intermediate, device="cpu", chunk_rows=3)
    weights = dict(gate_proj=torch.ones(intermediate, width, dtype=torch.bfloat16),
                   up_proj=torch.eye(intermediate, width, dtype=torch.bfloat16),
                   down_proj=torch.eye(width, intermediate, dtype=torch.bfloat16))
    matrices = []

    def quantize(name, weight, covariance, device):
        assert name.startswith("layer.experts.0.")
        assert covariance["count"] == 1024
        matrices.append(covariance["H"].clone())
        return SimpleNamespace(packed={4: {"mock": weight - .25}}, metrics={4: {}}, reconstruction=weight - .25)

    def reconstruct(packed, shape, device):
        return packed["mock"]

    monkeypatch.setattr(numerics, "encode_expert", quantize)
    monkeypatch.setattr(numerics, "reconstruct", reconstruct)
    packed, metrics = experts.fit_expert(bank, 0, weights, prefix="layer.experts.0", workspace=workspace, swiglu_limit=2)
    rows, coefficients, natural, identity = bank.selection(0)
    assert natural == 4 and identity == 1018
    assert set(metrics) == set(weights)
    assert torch.equal(matrices[0], matrices[1])
    assert torch.equal(matrices[0], inputs.float().T @ inputs.float() + torch.eye(width) * identity)
    gate = (inputs[rows] @ (weights["gate_proj"] - .25).T).clamp(max=2)
    up = (inputs[rows] @ (weights["up_proj"] - .25).T).clamp(-2, 2)
    activated = torch.nn.functional.silu(gate) * up
    scaled = activated.float() * coefficients[:, None]
    expected_down = scaled.T @ scaled + torch.eye(intermediate) * identity
    torch.testing.assert_close(matrices[2], expected_down)
    replay = torch.cat([v.clone() for v in experts.replay_expert(
        bank, 0, packed, workspace=workspace, swiglu_limit=2)])
    assert torch.equal(replay, activated[:natural] @ (weights["down_proj"] - .25).T)


def test_native_bf16_bank_preserves_fp32_sum_and_refuses_narrowing():
    kwargs = dict(tokens=3, width=2, experts=17, device="cpu", input_dtype=torch.bfloat16)
    inputs = torch.ones(3, 2)
    ids = torch.arange(8).expand(3, -1)
    gamma = torch.full((3, 8), .1234567)
    routes = (ids, gamma, ids+8, gamma/2)
    with pytest.raises(ValueError, match="cannot narrow"):
        ExpertBank([(inputs, *routes)], **kwargs)
    bank = ExpertBank([(inputs.bfloat16(), *routes)], **kwargs)
    rows = torch.arange(3)
    fp32 = torch.empty(3,2)
    with pytest.raises(ValueError, match="BF16 device gather"):
        list(bank.chunks(rows, fp32))
    narrowed = torch.empty_like(fp32, dtype=torch.bfloat16)
    assert torch.equal(next(bank.chunks(rows, fp32, native_buffer=narrowed)), inputs)
    result = bank.combine(lambda expert, rows: [inputs.bfloat16()] if len(rows) else [])
    exact = torch.zeros_like(inputs)
    for part in gamma.T:
        exact += part[:,None]
    assert torch.equal(result, exact) and result.dtype == torch.float32
    plan = memory_plan(3,2,frontier_bytes=0,weight_bytes=0,hessian_bytes=0,workspace_bytes=0,
                       input_dtype=torch.bfloat16)
    assert plan["owners"]["output_sum"] == plan["owners"]["input_bank"]*2 == 24
