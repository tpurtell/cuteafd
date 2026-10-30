"""Step-5 source census, tail-expert policy and fused-to-per-expert mapping."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step5.core_k6 import recipe


def test_reviewed_counts_and_independent_payload_arithmetic():
    specs=recipe.reviewed_source_specs();modules=recipe.reviewed_modules()
    assert len(specs)==2449 and len(modules)==93600
    assert sum(math.prod(s)*{'BF16':2,'F32':4}[d] for d,s in specs.values())==1_205_950_680_832
    assert Counter((g['group'],g['bits']) for g in modules.values())=={('base_core',6):645,('mtp_core',4):27,('routed',4):89760,('routed',6):3168}
    source=recipe.selected_source_names(modules)
    assert len(source)==936 and len(specs)-len(source)==1513
    packed=sum(math.prod(shape)*{'I16':2,'F16':2,'I32':4}[d] for m,g in modules.items() for d,shape in recipe.packed_specs(m,g).values())
    kept=sum(math.prod(s)*{'BF16':2,'F32':4}[d] for n,(d,s) in specs.items() if n not in source)
    assert kept+packed==315_357_147_520==recipe.TARGET_PAYLOAD_BYTES
    assert len(specs)-len(source)+4*len(modules)==375913==recipe.TARGET_TENSOR_COUNT


def test_original_expert_index_order_and_tail_policy():
    modules=recipe.reviewed_expert_modules()
    for layer,bits in [(3,4),(87,4),(88,6),(89,6),(90,6)]:
        for e in [0,351]:
            m=f'model.layers.{layer}.moe.experts.{e}.down_proj'
            assert modules[m]['bits']==bits
            assert recipe.source_tensor(m)==(f'model.layers.{layer}.moe.down_proj.weight',e)
            assert recipe.packed_specs(m,modules[m])[m+'.trellis']==('I16',[96,256,bits*16])
    with pytest.raises(ValueError):recipe.source_tensor('model.layers.91.moe.experts.0.up_proj')
    with pytest.raises(ValueError):recipe.source_tensor('model.layers.3.moe.experts.352.up_proj')


def test_mtp_heads_are_real_k4_matrices_and_sensitive_native_tensors_excluded():
    modules=recipe.reviewed_modules();specs=recipe.reviewed_source_specs()
    for layer in [92,93,94]:
        p=f'model.layers.{layer}.'
        g=modules[p+'transformer.shared_head.output']
        assert (g['bits'],g['in_features'],g['out_features'])==(4,4096,128896)
        assert modules[p+'eh_proj']['bits']==4
        assert specs[p+'transformer.shared_head.norm.weight']==('F32',[4096])
    assert modules['lm_head']['bits']==6
    assert not any(n.startswith(('vision_model.','vit_large_projector.')) for n in modules)
    assert sum(n.startswith(('vision_model.','vit_large_projector.')) for n in specs)==667
    assert not any('sparse_indexer' in n or n.endswith('.g_proj') for n in modules)
    assert specs['model.layers.7.self_attn.sparse_indexer_w.weight']==('F32',[16,4096])


def inventory_fixture(tmp_path,monkeypatch):
    document={'text_config':{'hidden_size':4096},'vision_config':{}}
    # Full configuration pin is exercised separately; census tests alter one
    # header/index datum at a time after admitting their deliberately tiny config.
    monkeypatch.setattr(recipe,'SOURCE_CONFIG_SHA256',hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    specs=recipe.reviewed_source_specs()
    header={};offset=0
    for n,(d,s) in specs.items():
        size=math.prod(s)*{'BF16':2,'F32':4}[d]
        header[n]=dict(dtype=d,shape=s,data_offsets=[offset,offset+size]);offset+=size
    source=SimpleNamespace(document=document,config=document['text_config'],snapshot=tmp_path,weight_map={n:'source.safetensors' for n in specs})
    monkeypatch.setattr(recipe,'read_header',lambda p:(header,0))
    return source,header


def test_inventory_accepts_exact_geometry_without_loading_payloads(tmp_path,monkeypatch):
    source,header=inventory_fixture(tmp_path,monkeypatch)
    result=recipe.inventory(source)
    assert result['source_payload_bytes']==recipe.BASE_PAYLOAD_BYTES
    assert result['target_payload_bytes']==recipe.TARGET_PAYLOAD_BYTES
    assert result['parts']['routed_k6']['modules']==3168
    assert result['preserved_tensors']==1513


@pytest.mark.parametrize('damage',['unknown','missing','shape','dtype','offset','shard','text_config','configuration'])
def test_inventory_rejects_drift(tmp_path,monkeypatch,damage):
    source,h=inventory_fixture(tmp_path,monkeypatch);name='model.layers.88.moe.up_proj.weight'
    if damage=='unknown':source.weight_map['unknown']='source.safetensors'
    elif damage=='missing':del source.weight_map[name]
    elif damage=='shape':h[name]['shape']=[351,1536,4096]
    elif damage=='dtype':h[name]['dtype']='F32'
    elif damage=='offset':h[name]['data_offsets'][1]-=2
    elif damage=='shard':source.weight_map[name]='../escape.safetensors'
    elif damage=='text_config':source.config={'hidden_size':12}
    elif damage=='configuration':source.document['unknown']=True
    with pytest.raises(ValueError):recipe.inventory(source)


def test_positive_module_patterns_cover_exactly_reviewed_names():
    import re
    from step5.core_k6.export import _module_patterns
    patterns=[re.compile(p) for p in _module_patterns()]
    assert all(any(p.fullmatch(n) for p in patterns) for n in recipe.reviewed_modules())
    for n in ['model.layers.95.self_attn.q_proj','model.layers.2.moe.experts.0.gate_proj',
              'model.layers.90.moe.experts.352.down_proj','model.layers.7.self_attn.sparse_indexer_q',
              'model.layers.1.self_attn.g_proj','model.layers.93.transformer.shared_head.norm']:
        assert not any(p.fullmatch(n) for p in patterns)


def test_only_reviewed_redundant_headers_are_allowed():
    # The official index chooses the embedding in shard24 and full MTP92 in
    # the MTP file, while older shards still carry redundant copies.
    aliases=recipe.reviewed_header_aliases()
    assert aliases['model-00001.safetensors']=={'model.embed_tokens.weight'}
    assert len(aliases['model-00024.safetensors'])==17
    h={'model.embed_tokens.weight':dict(dtype='BF16',shape=[128896,4096],data_offsets=[0,1_055_916_032])}
    recipe.validate_header_inventory('model-00001.safetensors',h,set())
    with pytest.raises(ValueError):recipe.validate_header_inventory('unknown.safetensors',h,set())
    h['unexpected']={}
    with pytest.raises(ValueError):recipe.validate_header_inventory('model-00001.safetensors',h,set())
    del h['unexpected'];h['model.embed_tokens.weight']['dtype']='F32'
    with pytest.raises(ValueError,match='redundant'):recipe.validate_header_inventory('model-00001.safetensors',h,set())
