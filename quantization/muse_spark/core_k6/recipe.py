"""Pinned Muse Spark BF16 inventory and split-expert MCG K4/Core-K6 recipe."""
from __future__ import annotations

import hashlib
import math

from glm5.core_k6.recipe import packed_specs, packed_bytes, _geometry as _shared_geometry
from kimi_k3.uniform_k2.storage import canonical, read_header

BASE_REPO = 'meta-models/Muse-Spark-Open-816B-A42B'
BASE_REVISION = '0fb02478b02565b2ce7fc57e6051c91172774ac9'
TARGET_REPO = 'wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-Core-K6'
BASE_PAYLOAD_BYTES = 1_644_333_137_920
BASE_TENSOR_COUNT = 2000
LAYER_COUNT = 62
EXPERT_COUNT = 256
BASE_CORE_MODULE_COUNT = LAYER_COUNT * 6 + 1
EXPERT_MODULE_COUNT = LAYER_COUNT * EXPERT_COUNT * 3
MODULE_COUNT = BASE_CORE_MODULE_COUNT + EXPERT_MODULE_COUNT
PRESERVED_TENSOR_COUNT = BASE_TENSOR_COUNT - LAYER_COUNT * 2 - BASE_CORE_MODULE_COUNT
TARGET_TENSOR_COUNT = PRESERVED_TENSOR_COUNT + MODULE_COUNT * 4
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ('trellis', 'suh', 'svh', 'mcg')
PREFIX = 'model.language_model.'
SOURCE_CONFIG_SHA256 = '557df1ddb9591ccda96c06488e719cd32963ede21fde0aa6bf306fb01e92fe74'
SOURCE_INDEX_SHA256 = '81e4bd2ada0490c1c9e6f10eccd40aed50d9af6b4e6a9b442914ad72da8ff604'
SOURCE_SPECS_SHA256 = '0b09b16a73db5b720279049fff5ced8921c736126850563d7a1fe34729eeb648'
FUSED_EXPERT_LAYOUT = dict(expert_axis=0, output_axis=1, input_axis=2,
    gate_output_rows=[0,4096], up_output_rows=[4096,8192],
    gate_up_shape=[256,8192,4096], down_shape=[256,4096,4096],
    output_names='model.language_model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}')


def _geometry(inputs, outputs, bits, group):
    return dict(_shared_geometry(inputs, outputs, bits, group), sharded_side=1)


def reviewed_core_modules():
    selected = {}
    for layer in range(LAYER_COUNT):
        prefix = f'{PREFIX}layers.{layer}.'
        for name, inputs, outputs in (
            ('self_attn.q_proj',8192,8192),('self_attn.k_proj',8192,1024),
            ('self_attn.v_proj',8192,1024),('self_attn.o_proj',8192,8192),
            ('mlp.pre_expert_proj',8192,4096),('mlp.post_expert_proj',4096,8192)):
            selected[prefix+name] = _geometry(inputs,outputs,6,'base_core')
    selected['lm_head'] = _geometry(8192,202048,6,'base_core')
    return selected


def reviewed_expert_modules():
    return {f'{PREFIX}layers.{layer}.mlp.experts.{expert}.{part}': _geometry(4096,4096,4,'routed')
            for layer in range(LAYER_COUNT) for expert in range(EXPERT_COUNT)
            for part in ('gate_proj','up_proj','down_proj')}


def reviewed_modules():
    return reviewed_core_modules() | reviewed_expert_modules()


def removed_source_names(selection=None):
    selection = reviewed_modules() if selection is None else selection
    names = set()
    for module,g in selection.items():
        if g['group']=='routed':
            layer_prefix = module.split('.mlp.experts.',1)[0]+'.mlp.experts.'
            names.add(layer_prefix+('down_proj' if module.endswith('.down_proj') else 'gate_up_proj'))
        else:
            names.add(module+'.weight')
    return names


def validate_config(document):
    if hashlib.sha256(canonical(document).encode()).hexdigest()!=SOURCE_CONFIG_SHA256:
        raise ValueError('unreviewed Muse Spark source configuration')
    if document.get('quantization_config') or document['text_config'].get('quantization_config'):
        raise ValueError('Muse source must be native BF16')


def inventory(source):
    validate_config(source.document)
    if source.config != source.document['text_config']:
        raise ValueError('Muse text config differs from source document')
    if hashlib.sha256((source.snapshot/'model.safetensors.index.json').read_bytes()).hexdigest()!=SOURCE_INDEX_SHA256:
        raise ValueError('unreviewed Muse source index')
    by_shard={}
    for name,shard in source.weight_map.items():
        if not isinstance(shard,str) or not shard.endswith('.safetensors') or '/' in shard or '\\' in shard:
            raise ValueError('source shard must be a local safetensors basename')
        by_shard.setdefault(shard,set()).add(name)
    specs,total={},0
    for shard,names in by_shard.items():
        header,_=read_header(source.snapshot/shard)
        if set(header)-{'__metadata__'} != names:
            raise ValueError(f'Muse source header/index differs: {shard}')
        for name in names:
            value=header[name]
            specs[name]=[value['dtype'],value['shape']]
            total+=value['data_offsets'][1]-value['data_offsets'][0]
    if (len(specs)!=BASE_TENSOR_COUNT or total!=BASE_PAYLOAD_BYTES or len(by_shard)!=65 or
        hashlib.sha256(canonical(specs).encode()).hexdigest()!=SOURCE_SPECS_SHA256):
        raise ValueError('unreviewed Muse source tensor inventory/geometry')
    selected=reviewed_modules();removed=removed_source_names(selected)
    if not removed<=specs.keys():
        raise ValueError('selected Muse source tensor absent')
    removed_bytes=sum(math.prod(specs[name][1])*{'BF16':2,'F32':4}[specs[name][0]] for name in removed)
    packed=sum(packed_bytes(g) for g in selected.values())
    parts={group:dict(modules=sum(g['group']==group for g in selected.values()),
        source_bytes=sum(g['in_features']*g['out_features']*2 for g in selected.values() if g['group']==group),
        packed_bytes=sum(packed_bytes(g) for g in selected.values() if g['group']==group))
        for group in ('base_core','routed')}
    if (len(specs)-len(removed)!=PRESERVED_TENSOR_COUNT or total-removed_bytes+packed!=TARGET_PAYLOAD_BYTES):
        raise ValueError('Muse output inventory arithmetic differs')
    return dict(source_payload_bytes=total,source_tensors=len(specs),selected_modules=len(selected),
        selected_source_bytes=removed_bytes,selected_packed_bytes=packed,
        preserved_bytes=total-removed_bytes,preserved_tensors=len(specs)-len(removed),
        target_payload_bytes=total-removed_bytes+packed,target_tensors=TARGET_TENSOR_COUNT,
        shards=len(by_shard),parts=parts,fused_expert_layout=FUSED_EXPERT_LAYOUT)


def selected_modules(source):
    inventory(source)
    return reviewed_modules()


_SELECTED_SOURCE_BYTES = sum(g['in_features']*g['out_features']*2 for g in reviewed_modules().values())
TARGET_PAYLOAD_BYTES = BASE_PAYLOAD_BYTES - _SELECTED_SOURCE_BYTES + sum(packed_bytes(g) for g in reviewed_modules().values())
