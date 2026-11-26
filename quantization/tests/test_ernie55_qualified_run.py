import pytest
from test_ernie55_identity import parity_inputs
from ernie55_flash import identity, qualified_run as run


def test_qualify_preserves_production_identity_and_binds_numerics(monkeypatch):
    inputs, vendor, local = parity_inputs(monkeypatch)
    forward = "quantization/ernie55_flash/forward.py"
    inputs["code"].update({p: v[1] for p,v in run.TRANSITIONS.items()})
    inputs["code"][forward] = "measured-forward"
    local["input_identity_sha256"] = identity.parity_input_digest(run.measured_inputs(inputs))
    assert run.qualify(inputs, vendor, local)["inputs"]["code"] == inputs["code"]
    inputs["code"][forward] = "changed-forward"
    with pytest.raises(ValueError, match="maxq forward parity"):
        run.qualify(inputs, vendor, local)
    inputs["code"]["quantization/mc_package.py"] = "undocumented-exporter"
    with pytest.raises(ValueError, match="unreviewed"):
        run.qualify(inputs, vendor, local)
