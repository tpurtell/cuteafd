"""CPU witnesses for the pilot's real staged dataflow and bounded recovery."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from test_ernie55_forward import tiny  # shared synthetic pread-backed checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash import pilot
from ernie55_flash.source import reviewed_roles
from safetensors.torch import load_file


def test_projected_bytes_include_every_group_input_scale_and_only_one_expert_copy():
    names = ("model.layers.0." + pilot.GROUP, "routed.w2", "norm")
    source = SimpleNamespace(specs={n: ("BF16", (2048, 256)) for n in names},
        sizes=dict.fromkeys(names, 1 << 20), roles=dict(zip(names, ("core", "routed_expert", "native"))))
    p = pilot.payload_projection(source, groups=8)
    expert = (2048 * 256 // 2) + 2 * (2048 + 256) + 4
    core8 = (2048 * 256) + 2 * (2048 + 8 * 256) + 4 * 8
    assert p["views"]["k8"]["bytes"] == expert + core8 + (1 << 20)
    assert p["views"]["original"]["bytes"] == expert + (2 << 20)
    assert p["physical_payload_bytes"] == (p["views"]["original"]["bytes"] + core8
                                          + core8 - (2048 * 256 // 4))


def test_witness_nmse_errors_preserve_energy_and_reject_nonfinite():
    x = torch.tensor([[1., 2.], [2., 0.]])
    assert pilot.nmse(x, x + 1) == pytest.approx(4 / 9)
    for bad in (torch.zeros_like(x), torch.full_like(x, float("nan"))):
        with pytest.raises(ValueError):
            pilot.nmse(bad, x)


def test_pilot_drives_real_reference_layers_and_distinct_widths(tmp_path, tiny, monkeypatch):
    model = tiny
    cfg = model.config
    with torch.random.fork_rng():
        torch.manual_seed(12)
        fixtures = {s: [dict(ids=torch.arange(131), **{f"layer_{i}":
                    torch.randn(131, cfg.hc_mult, cfg.hidden_size) for i in (0, 1, 2)},
                    **{f"tap_{i}": torch.randn(131, cfg.hidden_size) for i in (40, 41, 42)},
                    draft0_hidden=torch.randn(3, cfg.hc_mult, cfg.hidden_size),
                    **{f"draft{i}_positions": torch.arange(131, 134) for i in range(3)})]
                    for s in pilot.SPLITS}
    # The synthetic model's MLA layer is 2, whereas the real audit calls it 7.
    names = reviewed_roles()
    cores = {n for n, role in names.items() if role == "core" and
             n.startswith(("model.layers.0.", "model.layers.1.", "model.layers.7.", "model.mtp."))}
    source = SimpleNamespace(roles={n.replace("model.layers.7.", "model.layers.2."): "core" for n in cores})
    source.sizes = {n: 128 for n in source.roles}
    source.specs = {n: ("BF16", (128, 128)) for n in source.roles}
    run = pilot.Pilot(model, source, fixtures, tmp_path)
    (tmp_path / "reconstruct").mkdir()
    encoded = {}
    def encode(name, weight, train, held, widths, recovery=None):
        assert name not in encoded
        assert len(train) > 0 and len(held) > 0
        assert train.shape[-1] == held.shape[-1] == weight.shape[-1]
        if ".experts." in name:
            assert recovery["effective_rows"] == 1024
        encoded[name] = (widths, len(train))
        return weight
    monkeypatch.setattr(run, "encode", encode)
    for i in (0, 1, 2):
        run.layer(i)
    run.feature()
    for i in range(3):
        run.layer(i, draft=True)
        assert set(load_file(str(tmp_path / f"reconstruct/mtp-{i}.safetensors"))) == {
            n for n in source.roles if n.startswith(f"model.mtp.{i}.")}
        assert encoded[f"model.mtp.{i}.self_attn.linear_kv_proj.weight"][1] == cfg.sliding_window + 3
    assert encoded["model.layers.2.self_attn.kv_a_proj_with_mqa.weight"][0] == (8, 6)
    assert encoded["model.mtp.1.self_attn.gate_proj.weight"][0] == (8,)
    assert len([n for n in encoded if ".experts." in n]) == 9
    assert sorted(run.report["recovered"]) == ["model.layers.1", "model.layers.2",
                                               "model.mtp.0", "model.mtp.1", "model.mtp.2"]


def test_packed_arms_share_one_owned_raw_hessian_and_decode_mul1(tmp_path, monkeypatch):
    from inkling2 import numerical
    run = pilot.Pilot(SimpleNamespace(checkpoint=SimpleNamespace(device="cpu"),
                                      config=SimpleNamespace(o_groups=8)),
                      SimpleNamespace(roles={}, specs={}, sizes={}), {}, tmp_path)
    torch.manual_seed(5)
    samples, weight = torch.randn(17, 128), torch.randn(12, 128).bfloat16().float()
    hessians = []
    def pack(source, raw, **options):
        assert options["codebook"] == "mul1"
        assert options["seed"] == 787 and options["sigma"] == .025
        hessians.append(raw)
        return {"data": source.to(torch.bfloat16)}, {}, {}
    def decode(packed, shape, **options):
        assert options["codebook"] == "mul1"
        return packed["data"]
    monkeypatch.setattr(numerical, "quantize_projection", pack)
    monkeypatch.setattr(numerical, "reconstruct_projection", decode)
    best = run.encode("feature", weight, samples, samples[:9], (8, 6))
    assert torch.equal(best, weight)
    assert hessians[0] is hessians[1]
    assert hessians[0]["count"] == len(samples)
    assert torch.equal(hessians[0]["H"], samples.T @ samples)
    assert all(row["heldout_nmse"] == 0 for row in run.report["projections"][0]["arms"].values())
