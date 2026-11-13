"""Recovery witnesses for the calibration frontier, independent of CUDA."""
from types import SimpleNamespace
import json

import pytest
import torch
from safetensors.torch import load_file

from inkling2.run import Runner, save_stage, state_from, state_tensors
from kimi_k3.core_k6.storage import StageStore


def test_uncommitted_prompt_phase_can_recover_without_changing_committed_frontier(tmp_path):
    store = StageStore(tmp_path, {'numerics':'frozen'})
    expected = torch.arange(18).reshape(3,6).to(torch.bfloat16)
    def interrupted():
        yield 0, {'partial':expected}
        raise RuntimeError('simulated power loss')
    with pytest.raises(RuntimeError, match='power loss'):
        save_stage(store, 'boundary-000', interrupted())
    assert store.load('boundary-000') is None
    save_stage(store, 'boundary-000', [(0, {'partial':expected}), (1, {'partial':expected+1})])
    manifest = store.load('boundary-000', checksum=True)
    assert len(manifest['entries']) == 2
    assert torch.equal(load_file(str(tmp_path/'boundary-000/00000.safetensors'))['partial'],expected)
    def must_not_run():
        raise AssertionError('committed phase was recomputed')
        yield
    assert save_stage(store, 'boundary-000', must_not_run()) == manifest


def test_uncommitted_phase_rejects_unowned_leaf(tmp_path):
    store = StageStore(tmp_path, {})
    directory = store.directory('attention-000')
    directory.mkdir()
    unrelated = directory/'source-checkpoint.bin'
    unrelated.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='unexpected'):
        save_stage(store, 'attention-000', [])
    assert unrelated.read_bytes() == b'preserve'


def test_initial_boundary_commit_recovers_missing_frontier_without_embedding_again(tmp_path):
    store = StageStore(tmp_path/'activations', {'numerics':'frozen'})
    save_stage(store, 'boundary-000', [(0, {'partial':torch.ones(3,6,dtype=torch.bfloat16)})])
    runner = object.__new__(Runner)
    runner.root, runner.stages = tmp_path, store
    runner.initialize()
    assert json.loads((tmp_path/'frontier.json').read_text()) == {'next_layer':0,'stage':'boundary-000'}


def test_attnres_boundary_roundtrip_preserves_every_bf16_bank_byte():
    class State:
        def __init__(self, partial, blocks):
            self.partial, self.blocks = partial, blocks
    generator = torch.Generator().manual_seed(39)
    state = State(torch.randn(7,12,generator=generator).bfloat16(),
                  [torch.randn(7,12,generator=generator).bfloat16() for _ in range(8)])
    values = state_tensors(state)
    restored = state_from({k:v.clone() for k,v in values.items()}, SimpleNamespace(AttnResState=State))
    for expected, actual in zip([state.partial,*state.blocks],[restored.partial,*restored.blocks]):
        assert actual.dtype == torch.bfloat16
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16))


def test_corpus_rows_stay_unpadded_untruncated():
    ids = torch.arange(1577).reshape(1,-1)
    actual = Runner.ids({'input_ids':ids,'attention_mask':torch.ones_like(ids)})
    assert actual.shape == (1577,)
    assert torch.equal(actual,ids[0])
