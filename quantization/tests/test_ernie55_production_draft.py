import torch
import torch.nn.functional as F

from test_ernie55_forward import tiny
from test_ernie55_production_core import source_codec
from ernie55_flash.forward import reference
from ernie55_flash.production_draft import prepare_draft
from ernie55_flash.storage import Stages


def test_features_and_anchor_agree_with_reference_and_resume(tmp_path, tiny, source_codec):
    stages = Stages(tmp_path / "stages", "draft-fixture")
    c = tiny.config
    rng = torch.Generator().manual_seed(790)
    lengths = [128, 17, 2]
    taps = []
    last = [torch.randn(n, c.hc_mult, c.hidden_size, generator=rng) for n in lengths]
    for layer in range(3):
        values = [torch.randn(n, c.hidden_size, generator=rng) for n in lengths]
        stages.write_activations(f"tap-{layer}", lambda start: ((i, {"tap": values[i]})
                                 for i in range(start, len(lengths))), len(lengths))
        taps.append(values)
    stages.write_activations("target-final", lambda start: ((i, {"state": last[i]})
                             for i in range(start, len(lengths))), len(lengths))
    args = (tiny, stages, dict(calibration=2, heldout=1), lengths,
            ["tap-0", "tap-1", "tap-2"], "target-final", lambda *a, **kw: None)
    k8, k6, metrics, name = prepare_draft(*args)
    assert len(k8) == len(k6) == 4
    assert [r["bits"] for r in metrics] == [8, 6]
    assert all(r["heldout_passed"] and r["heldout_rows"] == 2 for r in metrics)
    _, count = source_codec["model.mtp.feature_projection"]
    assert count == sum(lengths[:2])
    outer = {name: tiny.checkpoint.load(name) for name in reference.OUTER_NAMES}
    aux = tiny.checkpoint.mtp_aux(c)
    for i, row in stages.records(name):
        n = lengths[i]
        features = tiny.draft_features([v[i] for v in taps], aux)
        anchor = tiny.logits(last[i][-1:], outer).argmax(-1)[0]
        ids = torch.full((c.dspark_block_size,), c.dspark_noise_token_id)
        ids[0] = anchor
        expected = F.embedding(ids, outer[reference.OUTER_NAMES[0]])[:, None].expand(-1, c.hc_mult, -1)
        assert torch.equal(row["state"], expected)
        assert torch.equal(row["anchor"], anchor)
        assert torch.equal(row["context"], features[-c.sliding_window:])
        assert torch.equal(row["positions"], torch.arange(n, n + c.dspark_block_size))
        assert torch.equal(row["context_positions"], torch.arange(max(n - c.sliding_window, 0), n))
    source_codec.clear()
    assert prepare_draft(*args) == (k8, k6, metrics, name)
    assert not source_codec
