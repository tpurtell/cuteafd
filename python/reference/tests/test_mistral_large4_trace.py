"""The engine-golden trace threads layer 35's latent to its readers by the config."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from families.mistral_large4 import trace  # noqa: E402


def test_latent_readers_get_their_source_keys_and_owners_get_none():
    sources = list(range(72))
    for reader in range(39, 72, 4):
        sources[reader] = 35
    config = SimpleNamespace(kv_source_layers=sources, no_rope_layers=[(i % 4) != 3 for i in range(72)])
    latent = torch.randn(1, 5, 512)
    carry = {"kv": {35: latent}}
    for reader in range(39, 72, 4):
        shared = trace.carry_shared(config, reader, carry)
        # Layer 35 is NoPE: its readers see the "nope" slot with an empty rope part.
        assert list(shared) == ["nope"] and shared["nope"][0] is latent and shared["nope"][1].shape[-1] == 0
    for owner in (0, 3, 35, 38):
        assert not trace.carry_shared(config, owner, carry)
