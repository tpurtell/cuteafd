"""Bounded, opt-in CPU qualification using actual checkpoint expert bytes.

Set CUTEAFD_SPARK_RANGE_MATRIX to the header-only planner JSON receipt. Each
entry contains model, snapshot and columns (the --spark-range --json output).
Two experts retain the complete intermediate axis and one H128 hidden block;
this tests every rank-map unit without loading an entire layer or using CUDA.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tp = load_module('spark_range_tp_reference', ROOT / 'scripts/bench/tp-placement.py')
exl3 = load_module('spark_range_exl3_reference', ROOT / 'python/tools/lib/exl3_reference.py')
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3(raw):
    raw = np.asarray(raw, dtype=np.uint8)
    exp = (raw >> 3) & 15
    mant = raw & 7
    return ((1 - 2 * (raw >> 7).astype(np.float32))
            * np.where(exp == 0, mant / 8, 1 + mant / 8)
            * np.exp2(np.where(exp == 0, -6, exp.astype(np.int16) - 7))).astype(np.float32)


class Checkpoint:
    def __init__(self, snapshot):
        self.root = Path(snapshot)
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        config = json.loads((self.root / 'config.json').read_text())
        self.config = config.get('text_config', config)
        self.layer = None
        self.headers = {}

    def meta(self, key):
        shard = self.index[key]
        if shard not in self.headers:
            with (self.root / shard).open('rb') as file:
                size = struct.unpack('<Q', file.read(8))[0]
                self.headers[shard] = (8 + size, json.loads(file.read(size)))
        offset, header = self.headers[shard]
        return self.root / shard, offset, header[key]

    def tensor(self, key, rows=None, columns=None, expert=None):
        path, offset, meta = self.meta(key)
        dtype = {'U8': 'u1', 'I8': 'i1', 'F8_E4M3': 'u1', 'F8_E8M0': 'u1',
                 'F16': '<f2', 'BF16': '<u2', 'F32': '<f4', 'I16': '<i2', 'I32': '<i4'}[meta['dtype']]
        data = np.memmap(path, mode='r', dtype=dtype,
                         offset=offset + meta['data_offsets'][0], shape=tuple(meta['shape']) or ())
        if expert is not None:
            data = data[expert]
        if rows is not None:
            data = data[rows]
        if columns is not None:
            data = data[:, columns]
        result = np.array(data)
        if meta['dtype'] == 'BF16':
            result = (result.astype(np.uint32) << 16).view(np.float32)
        return result

    def weight(self, stem, down, expert=None):
        if stem + '.trellis' in self.index:
            marker = np.asarray(self.tensor(stem + '.mcg')).astype(np.uint32)
            if marker.size != 1 or marker.item() != 0xcbac1fed:
                raise ValueError('independent reference requires the checkpoint MCG codebook')
            # H128 rotations do not cross the retained hidden-axis block.
            trellis = self.tensor(stem + '.trellis', None if down else slice(0, 8),
                                  slice(0, 8) if down else None)
            suh = self.tensor(stem + '.suh', None if down else slice(0, 128))
            svh = self.tensor(stem + '.svh', slice(0, 128) if down else None)
            return exl3.dequantize(trellis, suh, svh).astype(np.float32).T
        key = stem + '.weight'
        _, _, meta = self.meta(key)
        packed = meta['dtype'] in ('I8', 'U8')
        rows, columns = (slice(0, 128), None) if down else (None, slice(0, 64 if packed else 128))
        data = self.tensor(key, rows, columns, expert)
        if meta['dtype'] in ('BF16', 'F16', 'F32'):
            return data.astype(np.float32)
        scale_key = next((stem + suffix for suffix in ('.weight_scale', '.weight_scale_inv', '.scale')
                          if stem + suffix in self.index), None)
        if scale_key is None:
            raise ValueError('missing scale for ' + stem)
        _, _, scale_meta = self.meta(scale_key)
        # Slice before materialization, including fused Step-5 expert arrays.
        full_shape = meta['shape'][-2:]
        scale_shape = scale_meta['shape'][-2:]
        nblock = full_shape[0] // scale_shape[0]
        kblock = full_shape[1] * (2 if packed else 1) // scale_shape[1]
        scales = self.tensor(scale_key, slice(0, (128 + nblock - 1) // nblock) if down else None,
                             None if down else slice(0, (128 + kblock - 1) // kblock), expert)
        if scale_meta['dtype'] == 'F8_E4M3':
            scales = e4m3(scales)
        elif scale_meta['dtype'] in ('U8', 'F8_E8M0'):
            scales = np.exp2(scales.astype(np.int16) - 127).astype(np.float32)
        if packed:
            codes = np.stack((data.astype(np.uint8) & 15, data.astype(np.uint8) >> 4), axis=-1)
            data = FP4[codes].reshape(data.shape[0], -1)
        else:
            data = e4m3(data)
        scales = scales.repeat(nblock, 0).repeat(kblock, 1)[:data.shape[0], :data.shape[1]]
        global_scale = self.tensor(stem + '.weight_scale_2') if stem + '.weight_scale_2' in self.index else 1
        return (data * scales * global_scale).astype(np.float32)

    def experts(self):
        stems = sorted({key.rsplit('.', 1)[0] for key in self.index
                        if re.search(r'\.experts\.0\.(gate_proj|w1)\.(weight|trellis)$', key)})
        if stems:
            stem = stems[0]
            self.layer = int(re.search(r'(?:^|\.)layers\.(\d+)\.', stem)[1])
            for expert in (0, 1):
                gate = stem.replace('.experts.0.', f'.experts.{expert}.')
                up = gate.rsplit('.', 1)[0] + ('.up_proj' if gate.endswith('gate_proj') else '.w3')
                down = gate.rsplit('.', 1)[0] + ('.down_proj' if gate.endswith('gate_proj') else '.w2')
                yield expert, tuple(self.weight(s, d) for s, d in ((gate, False), (up, False), (down, True)))
        else:
            gate = sorted(key[:-7] for key in self.index if key.endswith('.moe.gate_proj.weight'))[0]
            self.layer = int(re.search(r'(?:^|\.)layers\.(\d+)\.', gate)[1])
            for expert in (0, 1):
                yield expert, tuple(self.weight(gate.replace('gate_proj', p), d, expert)
                                    for p, d in (('gate_proj', False), ('up_proj', False), ('down_proj', True)))


def rank_sum(weights, intermediate, unit, world, expert, layer, padded=False,
             source_block=128, activation='silu', limit=None, rule=None):
    gate, up, down = weights
    assert gate.shape == up.shape == (intermediate, 128)
    assert down.shape == (128, intermediate)
    assert all(np.isfinite(w).all() and np.linalg.norm(w) > 0 for w in weights)
    x = np.linspace(-.1, .1, 256, dtype=np.float32).reshape(2, 128)
    g, u = x @ gate.T, x @ up.T
    def activate(gate, value):
        if limit is not None and activation != 'post-clamp':
            gate, value = np.minimum(gate, limit), np.clip(value, -limit, limit)
        sigmoid = 1 / (1 + np.exp(-np.clip(gate, -60, 60)))
        if activation == 'situ':
            return 4 * np.tanh(gate / 4) * sigmoid * (25 * np.tanh(value / 25))
        silu = gate * sigmoid
        if activation == 'post-clamp':
            return np.minimum(silu, limit) * np.clip(value, -limit, limit)
        return silu * value

    hidden = activate(g, u)
    whole = hidden @ down.T
    partials = []
    # Integer contributions isolate byte-exact ownership/reduction from the
    # permitted floating-point reordering of separate rank GEMMs.
    contributions = np.rint(np.clip(hidden[:, None, :] * down[None, :, :],
                                     -(1 << 20), 1 << 20) * 256).astype(np.int64)
    exact_whole = contributions.sum(axis=2)
    exact_sum = np.zeros_like(exact_whole)
    covered = np.zeros(intermediate, dtype=np.int32)
    source_unit = source_block if padded else unit
    if rule is None:
        rule = 'contiguous' if intermediate // source_unit % world == 0 else (
            'layer-rotation' if padded or source_unit == 32 else 'rotated')
    for rank in range(world):
        columns = np.array([column for index in range(intermediate // source_unit)
                            if tp.unit_owner(rule, intermediate // source_unit, expert, index, layer, world) == rank
                            for column in range(index * source_unit, (index + 1) * source_unit)])
        assert columns.size > 0
        covered[columns] += 1
        exact_sum += contributions[:, :, columns].sum(axis=2)
        rg, ru = x @ gate[columns].T, x @ up[columns].T
        rh = activate(rg, ru)
        partials.append(rh @ down[:, columns].T)
    assert np.array_equal(covered, np.ones(intermediate, dtype=np.int32))
    assert exact_sum.tobytes() == exact_whole.tobytes(), (world, expert, 'ownership reduction bytes')
    summed = np.zeros_like(whole)
    for partial in partials:
        summed += partial
    error = float(np.linalg.norm(summed - whole) / max(np.linalg.norm(whole), 1e-20))
    assert error < 2e-5, (world, expert, error)
    return error


def test_e4m3_and_fp4_decoding():
    assert e4m3([0, 56, 64, 184]).tolist() == [0, 1, 2, -1]
    assert FP4[[1, 7, 9, 15]].tolist() == [.5, 6, -.5, -6]


@pytest.mark.parametrize('prefix', ['layers.', 'model.layers.', 'language_model.model.layers.'])
def test_checkpoint_layer_names(prefix):
    assert int(re.search(r'(?:^|\.)layers\.(\d+)\.', prefix + '10.mlp.experts.0.w1')[1]) == 10


@pytest.mark.skipif(not os.environ.get('CUTEAFD_SPARK_RANGE_MATRIX'), reason='explicit real-checkpoint CPU qualification')
def test_real_checkpoint_rank_sums(record_property):
    entries = json.loads(Path(os.environ['CUTEAFD_SPARK_RANGE_MATRIX']).read_text())
    results = []
    coverage = []
    for entry in entries:
        if not entry.get('columns'):
            coverage.append(dict(model=entry['model'], qualified=False,
                                 reason=entry.get('error', 'planner returned no range geometry')))
            continue
        checkpoint = Checkpoint(entry['snapshot'])
        columns = entry['columns']
        intermediate, unit = columns[0]['intermediate'], columns[0]['unit_columns']
        # Cover the entire formula/floor ceiling even when today's complete
        # planner cannot resolve N_min because installed packages are absent.
        counts = {count['sparks']: count for column in columns for count in column['counts'] if count['sparks']}
        if not counts:
            coverage.append(dict(model=entry['model'], qualified=False, reason='no positive Spark candidates'))
            continue
        coverage.append(dict(model=entry['model'], qualified=True, worlds=sorted(counts)))
        for expert, weights in checkpoint.experts():
            for world, count in counts.items():
                widths = count.get('resolved_width_columns')
                assert widths is None or all(width > 0 for width in widths)
                family = columns[0]['package'].split(':')[0]
                activation = 'situ' if family == 'kimik3' else ('post-clamp' if family == 'step5' else 'silu')
                limit = checkpoint.config['swiglu_limits'][checkpoint.layer] if family == 'step5' else (
                    10 if family in ('v41', 'dsv4f', 'dsv4p', 'glmf') else None)
                error = rank_sum(weights, intermediate, count['unit_columns'], world, expert, checkpoint.layer,
                                 count['padded'], count['source_block_columns'], activation, limit, count['rule'])
                results.append(dict(model=entry['model'], world=world, expert=expert, relative_l2=error))
    assert results, 'no real tensor cases qualified'
    record_property('rank_sum_cases', json.dumps(results))
    record_property('rank_sum_coverage', json.dumps(coverage))
    output = os.environ.get('CUTEAFD_SPARK_RANGE_SUM_RESULTS')
    if output:
        Path(output).write_text(json.dumps(dict(cases=results, coverage=coverage), indent=2) + '\n')
    assert all(row['qualified'] for row in coverage), [row for row in coverage if not row['qualified']]
