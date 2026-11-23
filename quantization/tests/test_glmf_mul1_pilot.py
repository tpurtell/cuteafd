from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc.pilot import expected_core_arms, nmse, tp2_payload


def test_pilot_core_inventory_and_native_anchor():
    arms = expected_core_arms()
    assert len(arms) == (6 * 7 + 2) * 2
    assert {width for name, width in arms if name.endswith("eh_proj")} == {6,8}
    assert ("model.language_model.layers.45.self_attn.q_b_proj", 8) in arms
    assert ("lm_head", 6) in arms
    x, native = torch.randn(5, 4), torch.randn(3000, 4)
    assert nmse(x, native, native) == 0.
    assert nmse(x, native, 2*native) == pytest.approx(1.)
    with pytest.raises(ValueError, match="energy"):
        nmse(x, native*0, native)


def test_tp2_projection_charges_replicated_scales_and_dedicated_head():
    assert tp2_payload("q_a_proj",(1536,4096),6)[0] == 1536*4096*6//8 + 2*(1536+4096)+4
    assert tp2_payload("lm_head", (154880,4096),8)[1] == 0
    assert tp2_payload("o_proj",(4096,8192),8)[0] == 4096**2+4*4096+4
