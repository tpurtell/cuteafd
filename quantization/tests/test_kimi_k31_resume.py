"""Fault-inject real rolling-frontier/export barriers without GPU numerics."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k31.k2_core_k6 import run
from kimi_k31.k2_core_k6 import export
from kimi_k3.core_k6.storage import StageStore
from kimi_k3.core_k6.run import commit_batch
from kimi_k3.uniform_k2.storage import atomic_json, commit_safetensors


class Progress:
    def emit(self, *args, **kwargs):
        pass


def completed_f(tmp_path, monkeypatch):
    records = [dict(id='a', input_ids=[1, 2]), dict(id='b', input_ids=[3])]
    identity = dict(run='k31-test')
    core = 'language_model.model.layers.1.self_attn.q_proj'
    expert = 'language_model.model.layers.1.block_sparse_moe.experts.0.w1'
    engine = run.Engine(SimpleNamespace(), records, tmp_path/'run', tmp_path/'export', identity,
                        {core: {}, expert: {}}, Progress(), device='cpu', numerical=SimpleNamespace())
    old, successor = [], []
    store = StageStore(engine.root/'layers/001/stages', identity)
    for index, row in enumerate(records):
        hidden = torch.full((1, len(row['input_ids']), 4), index + .25)
        old.append(commit_safetensors(engine.boundary.directory(0)/f'batch-{index}.safetensors',
                   dict(prefix_sum=hidden, block_residual=hidden.new_empty((hidden.shape[1], 0, 4)))))
        old[-1].update(filename=f'batch-{index}.safetensors', id=row['id'], tokens=len(row['input_ids']))
        successor.append(commit_batch(store.directory('f'), index, row,
                         dict(prefix_sum=hidden+1, block_residual=hidden.reshape(-1, 1, 4))))
    # Supply the embedding boundary first; layer 0 then becomes the predecessor.
    embedding = []
    for index, row in enumerate(records):
        hidden = torch.zeros(1, len(row['input_ids']), 4)
        embedding.append(commit_batch(engine.boundary.directory(-1), index, row,
                         dict(prefix_sum=hidden, block_residual=hidden.new_empty((hidden.shape[1], 0, 4)))))
    engine.boundary.commit(-1, embedding)
    current = engine.boundary.commit(0, old)
    store.commit('f', successor)
    packed = engine.root/'layers/001/projections/packed.safetensors'
    packed.parent.mkdir(parents=True)
    packed.write_bytes(b'core-candidate')
    expert_path = engine.root/'layers/001/expert-projections'/f'{expert}.safetensors'
    receipt = commit_safetensors(expert_path, {expert+'.trellis': torch.arange(4, dtype=torch.int16)})
    atomic_json(expert_path.with_suffix('.json'), dict(identity=identity, module=expert, **receipt))
    routes = engine.root/'layers/001/routes'
    routes.mkdir()
    (routes/'latent.bf16').write_bytes(b'owned-route-payload')
    class Checkpoints:
        def replacements(self, modules):
            assert modules == [core] and packed.exists()
            return {core: packed}
        def retire_payloads(self):
            packed.unlink()
    monkeypatch.setattr(engine, '_checkpoints', lambda layer: Checkpoints())
    return engine, current, store, packed, expert_path, routes


def assembly(source, layer, replacements, output, identity):
    assert len(replacements) == 2
    path = Path(output)/'model-00002-of-00094.safetensors'
    receipt = commit_safetensors(path, {'retained': torch.arange(4)})
    receipt.update(identity=identity, filename=path.name, payload_bytes=32)
    atomic_json(Path(output)/'layer-001-receipt.json', receipt)
    return receipt


def test_completed_replay_survives_failed_assembly_and_retires_only_after_commit(tmp_path, monkeypatch):
    engine, current, store, core, expert, routes = completed_f(tmp_path, monkeypatch)
    def failure(*args):
        raise OSError('interrupted shard assembly')
    monkeypatch.setattr(export, 'assemble_layer', failure)
    with pytest.raises(OSError, match='interrupted'):
        engine.layer(1, current)
    assert engine.boundary.current(checksum=True)['layer'] == 0
    assert store.load('f', checksum=True) is not None
    assert core.exists() and expert.exists() and routes.exists()
    monkeypatch.setattr(export, 'assemble_layer', assembly)
    # Source deliberately has no load_block method: F recovery must not reload
    # or recalibrate either the native decoder or the 896 experts.
    successor = engine.layer(1, current)
    assert successor['layer'] == 1
    assert not core.exists() and not expert.exists() and not routes.exists()
    assert store.load('f') is None and not engine.boundary.directory(0).exists()
    hidden = run.boundary_batch(engine.boundary, successor, 0, 'cpu')['prefix_sum']
    assert torch.equal(hidden, torch.full((1, 2, 4), 1.25))


def test_commit_before_retirement_preserves_recoverable_successor(tmp_path, monkeypatch):
    engine, current, store, core, expert, routes = completed_f(tmp_path, monkeypatch)
    monkeypatch.setattr(export, 'assemble_layer', assembly)
    original = StageStore.clear
    def failure(self):
        raise OSError('interrupted after frontier commit')
    monkeypatch.setattr(StageStore, 'clear', failure)
    with pytest.raises(OSError, match='interrupted'):
        engine.layer(1, current)
    successor = engine.boundary.current(checksum=True)
    assert successor['layer'] == 1 and core.exists() and expert.exists() and routes.exists()
    monkeypatch.setattr(StageStore, 'clear', original)
    # Keep this fixture focused on layer 1: the inherited coordinator verifies
    # older committed export receipts independently (already covered for K3).
    monkeypatch.setattr(run.CoreEngine, 'retire_committed',
                        lambda self, current: (store.clear(), core.unlink()))
    engine.retire_committed(successor)
    assert not core.exists() and not expert.exists() and not routes.exists()
    hidden = run.boundary_batch(engine.boundary, successor, 1, 'cpu')['prefix_sum']
    assert torch.equal(hidden, torch.full((1, 1, 4), 2.25))


def test_recovery_from_committed_expert_replay_does_not_requantize_experts(tmp_path):
    events = []
    engine = object.__new__(run.Engine)
    engine._quant_group = lambda block, checkpoints, names, store, phase, hessian: events.append((phase, hessian))
    engine._phase_f = lambda block, store: events.append('finish')
    engine._experts = lambda *args: pytest.fail('committed E must reuse expert replay')
    block = SimpleNamespace(layer_idx=3)
    phases = {name: None for name in ('a', 'b', 'c', 'd', 'e', 'f')}
    phases['e'] = {'committed': True}
    engine._execute_layer(block, None, None, None, phases)
    assert events == [('e', 'shared-down'), ('e', 'routed-up'), 'finish']
