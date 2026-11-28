from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash import production_moe as moe
from ernie55_flash.forward import reference
from ernie55_flash.storage import Stages


def test_recovered_experts_and_sealed_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(moe, "EXPERT_GROUP", 8)
    config = SimpleNamespace(n_routed_experts=24, moe_intermediate_size=2, moe_latent_size=2,
                             activation_situ_beta=4., activation_situ_linear_beta=25.)
    cal = torch.randn((1031, 2), generator=torch.Generator().manual_seed(5))
    held = torch.randn((9, 2), generator=torch.Generator().manual_seed(32))
    native = torch.tensor([[1., -.25], [-.375, 1.125]])
    compact = native * .9375
    seen, xsums = [], []
    def source(i, dev):
        seen.append(i)
        return [native] * 3

    def routes(x, weights):
        ids = torch.arange(10).expand(len(x), -1)
        return dict(latent=x, ids=ids, gamma=torch.full_like(ids, -.25, dtype=torch.float32),
                    candidate_ids=(ids + 10).int(), candidate_gamma=torch.full_like(ids, .125, dtype=torch.float32))

    def active(x, gate, up):
        return reference.situ(reference.linear(x, gate), reference.linear(x, up), 4., 25.)

    model = SimpleNamespace(config=config, checkpoint=SimpleNamespace(device="cpu"),
                            activated=active, route_batch=routes)
    weights = {"experts": SimpleNamespace(get=source)}

    def quantize(w, raw, **kwargs):
        xsums.append((raw["H"].clone(), raw['count']))
        packed = {"trellis": (w * .9375).bfloat16().view(torch.int16)}
        packed.update({k: torch.ones(1) for k in ("mul1", "suh", "svh")})
        return packed, {"hessian_sample_count": raw['count']}, {}

    def reconstruct(packed, shape, **kwargs):
        return packed['trellis'].view(torch.bfloat16)

    monkeypatch.setattr(moe.numerical, "quantize_projection", quantize)
    monkeypatch.setattr(moe.numerical, "reconstruct_projection", reconstruct)
    def run():
        splits = dict(calibration=enumerate(cal.split(256)), heldout=enumerate(held.split(4)))
        return moe.encode_routed(model, weights, "model.layers.1.", Stages(tmp_path, {"test": 1}),
                                 splits, dict(calibration=len(cal), heldout=len(held)),
                                 chunk_rows=128, stage_bytes=128 * 2 * 4)

    first = run()
    assert seen == list(range(config.n_routed_experts))
    assert len(first.metrics) == 72
    assert len(first.packed) == 72 * len(moe.SUFFIXES)
    assert len(first.stages) == 3
    assert all(len({ref[0] for key, ref in first.packed.items() if ref[0].parent.name == stage}) == 1
               for stage in first.stages)
    for split, inputs in (("calibration", cal), ("heldout", held)):
        out = reference.linear(active(inputs, compact, compact), compact) * -.25
        expected = torch.zeros_like(out)
        for _ in range(10):
            expected += out
        assert torch.equal(expected, first.sums[split])
    for e, n, a, ident in ((0, 1031, 0, 0), (10, 0, 1024, 0), (23, 0, 0, 1024)):
        row = first.metrics[3*e]
        assert (row["natural_rows"], row["augmented_rows"], row["identity_rows"]) == (n, a, ident)
        assert row["heldout_nmse"] > 0
        assert row["heldout_natural_rows"] == (9 if e == 0 else 0)
        inputs = cal[:n+a]
        h1 = ident * torch.eye(2)
        h2 = h1.clone()
        if len(inputs):
            h1 += (inputs.T @ inputs)
            a2 = active(inputs, compact, compact) * (-.25 if n else .125)
            h2 += (a2.T @ a2)
        assert torch.allclose(xsums[3*e][0], h1, atol=.001, rtol=1e-5)
        assert torch.allclose(xsums[3*e+2][0], h2, atol=.001, rtol=1e-5)
        assert xsums[3*e+2][1] == n+a+ident
    seen.clear()
    xsums.clear()
    second = run()
    assert not seen and not xsums
    assert first.metrics == second.metrics
    assert first.packed == second.packed
    assert all(torch.equal(first.sums[key], second.sums[key]) for key in first.sums)
