import torch
import pytest
from test_ernie55_forward import tiny
from ernie55_flash import production_core as core
from ernie55_flash.storage import Stages


@pytest.fixture()
def source_codec(monkeypatch):
    """Stand in for GPU LDLQ only. Preserve original matrices for an exact forward oracle."""
    covariances = {}
    def encode(name, weight, cov, widths, device, metrics):
        assert cov["identity_rows"] == 0
        covariances[name] = (cov["H"].clone(), cov["count"])
        packed = {}
        for bits in widths:
            metrics.append(dict(module=name, bits=bits, hessian_rows=cov["count"]))
            for suffix in core.SUFFIXES:
                packed[f"k{bits}.{name}.{suffix}"] = weight if suffix == "trellis" else torch.zeros(1)
        return packed
    monkeypatch.setattr(core, "encode", encode)
    monkeypatch.setattr(core, "reconstruction", lambda values, name, bits, shape, dev: values[f"k{bits}.{name}.trellis"])
    return covariances


def test_dense_successor_and_restart_reuse_sealed_layers(tmp_path, tiny, source_codec):
    w = tiny.load_layer(0)
    rng = torch.Generator().manual_seed(42)
    hidden = [torch.randn(l, tiny.config.hc_mult, tiny.config.hidden_size, generator=rng) for l in (131, 129)]
    inputs = lambda start: ((i, {"state": hidden[i]}) for i in range(start, 2))
    event = lambda *args, **kwargs: None
    stages = Stages(tmp_path/"stages", "fixture")
    args = (tiny, stages, "layers", 0, inputs, {"calibration":1,"heldout":1}, {"calibration":131,"heldout":129}, event)
    r = core.process_layer(*args)
    assert len(r["k8"]) == len(r["k6"]) and not r["k4"]
    assert len(r["metrics"]) == 36 and all(row["heldout_passed"] for row in r["metrics"])
    c = source_codec
    prefix = "model.layers.0.self_attn."
    assert c[prefix+"core_attention.compressor.linear_wkv"][1] == 128
    group_h = [c[prefix+f"linear_o_group_proj.groups.{g}"] for g in range(8)]
    assert all(n == 131 for _, n in group_h)
    assert not torch.equal(group_h[0][0],group_h[1][0])
    for i, row in r["outputs"](0):
        torch.testing.assert_close(row["state"],tiny.native_layer(hidden[i], w, 0), rtol=0, atol=0)
    c.clear()
    restored = core.process_layer(*args)
    assert c == {}
    assert restored['metrics'] == r['metrics']
    assert torch.equal(next(restored['outputs'](1))[1]['state'], next(r['outputs'](1))[1]['state'])


@pytest.mark.parametrize("index,draft",[(2,False),(1,True)])
def test_reference_attention_after_install_including_two_draft_kv_inputs(tmp_path,tiny,source_codec,index,draft):
    c = tiny.config
    weights = tiny.load_draft_layer(index) if draft else tiny.load_layer(index)
    g = torch.Generator().manual_seed(index)
    n = 7
    rows = [dict(state=torch.randn(n,c.hc_mult,c.hidden_size,generator=g), positions=torch.arange(13,13+n),
                 context=torch.randn(c.sliding_window,c.hidden_size,generator=g),
                 context_positions=torch.arange(13-c.sliding_window,13)) for _ in range(2)]
    if not draft:
        rows = [dict(state=r["state"]) for r in rows]
    records = lambda start: ((i,rows[i]) for i in range(start,len(rows)))
    worker = core.LayerCore(tiny,weights,Stages(tmp_path/"stages","fixture"),"test","test.",
                            dict(calibration=1,heldout=1),lambda *a,**k: None)
    worker.fit_attention(index,records,2*n,draft)
    states = worker.prepare_mlp(index,records,draft)
    if draft:
        assert source_codec["test.self_attn.linear_kv_proj"][1] == n+c.sliding_window
    else:
        assert source_codec["test.self_attn.kv_a_proj_with_mqa"][1] == n
    for i, state in states(0):
        r=rows[i]
        kv=None
        if draft:
            cos,sin=core.ref.hca_phase(r['context_positions'],c,rope_mode="plain")
            kv=core.ref.norm(core.ref.linear(r['context'],weights["self_attn.linear_kv_proj.weight"]),
                             weights["self_attn.kv_layernorm.weight"],c.rms_norm_eps)
            kv=core.ref.rotate_tail(kv[:,None],cos,sin).squeeze(1)
        expected=tiny.mlp_input(r["state"],weights,index,r.get("positions",torch.arange(n)),draft_context_kv=kv)
        torch.testing.assert_close(state.x,expected.x,rtol=0,atol=0)
