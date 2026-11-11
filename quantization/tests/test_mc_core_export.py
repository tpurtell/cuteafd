"""CPU inventory/recovery guards for isolated MC core payloads."""
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_export import assemble_core_shard, finalize_core_export, packed_specs


def projection(tmp_path, name='layer.q', bits=8):
    geometry = dict(in_features=128, out_features=128, stored_out_features=128, bits=bits)
    path = tmp_path / (name + '.safetensors')
    tensors = {name + '.trellis': torch.zeros((8, 8, bits * 16), dtype=torch.int16),
               name + '.suh': torch.ones(128, dtype=torch.float16),
               name + '.svh': torch.ones(128, dtype=torch.float16),
               name + '.mcg': torch.tensor(-877912083, dtype=torch.int32)}
    # Encode the standard unsigned multiplier without signed conversion overflow.
    tensors[name + '.mcg'] = torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32)
    save_file(tensors, str(path))
    return name, geometry, path


def test_core_only_inventory_bits_and_idempotent_assembly(tmp_path):
    module, g, path = projection(tmp_path)
    export = tmp_path / 'export'
    identity = {'test': 'core-k8'}
    receipt = assemble_core_shard({module: path}, {module: g}, export, 'core.safetensors', identity)
    assert assemble_core_shard({module: path}, {module: g}, export, 'core.safetensors', identity) == receipt
    summary = finalize_core_export({module: g}, export, identity)
    assert summary['modules'] == 1 and summary['tensors'] == 4 and summary['bits'] == [8]
    index = json.loads((export / 'model.safetensors.index.json').read_text())
    assert set(index['weight_map']) == set(packed_specs(module, g))
    storage = json.loads((export / 'quantize_config.json').read_text())['tensor_storage'][module]
    assert storage['bits_per_weight'] == 8
    assert not any(name.endswith('.weight') for name in index['weight_map'])


def test_extra_native_payload_and_wrong_width_rejected(tmp_path):
    module, g, path = projection(tmp_path)
    with pytest.raises(ValueError, match='inventory'):
        assemble_core_shard({module: path}, {module: dict(g, bits=6)}, tmp_path / 'bad', 'core.safetensors', {})
    export = tmp_path / 'export'
    assemble_core_shard({module: path}, {module: g}, export, 'core.safetensors', {})
    save_file({'expert.weight': torch.zeros(1)}, str(export / 'unexpected.safetensors'))
    with pytest.raises((FileNotFoundError, ValueError)):
        finalize_core_export({module: g}, export, {})


def test_mixed_identity_and_missing_core_rejected(tmp_path):
    module, g, path = projection(tmp_path)
    export = tmp_path / 'export'
    assemble_core_shard({module: path}, {module: g}, export, 'core.safetensors', {'arm': 'a'})
    with pytest.raises(ValueError, match='identity'):
        finalize_core_export({module: g}, export, {'arm': 'b'})
    with pytest.raises(ValueError, match='incomplete'):
        finalize_core_export({module: g, 'layer.o': g}, export, {'arm': 'a'})
