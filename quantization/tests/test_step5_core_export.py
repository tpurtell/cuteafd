"""Assembly of fused Step experts, mixed bit widths and native-byte preservation."""
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step5.core_k6 import export,recipe
from kimi_k3.uniform_k2.storage import read_header


def save(path,tensors):
    """Tiny independent safetensors writer; no Torch/GPU dependency."""
    header={};parts=[];offset=0
    for n,(dtype,shape,fill) in sorted(tensors.items()):
        width={'BF16':2,'F32':4,'F16':2,'I16':2,'I32':4}[dtype]
        count=math.prod(shape)
        if dtype=='I32':data=struct.pack('<I',fill)*count
        elif dtype=='F16':data=struct.pack('<e',fill)*count
        else:data=bytes([fill])*count*width
        header[n]=dict(dtype=dtype,shape=shape,data_offsets=[offset,offset+len(data)])
        parts.append(data);offset+=len(data)
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    path.write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(parts))
    return path


def geom(bits=6,group='base_core'):
    return dict(in_features=128,out_features=128,stored_out_features=128,bits=bits,group=group)


def packed(path,module,g,damage=None):
    specs=recipe.packed_specs(module,g)
    ts={n:(d,s,recipe.MCG_MULTIPLIER if n.endswith('.mcg') else 1) for n,(d,s) in specs.items()}
    if damage=='mcg':ts[module+'.mcg']=('I32',[],0)
    elif damage=='scale':ts[module+'.suh']=('F16',[128],float('nan'))
    elif damage=='shape':ts[module+'.svh']=('F16',[64],1)
    return save(path,ts)


@pytest.fixture
def toy(tmp_path,monkeypatch):
    src=tmp_path/'src';src.mkdir();out=tmp_path/'export';evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'observed.json').write_text('{"rows":42}')
    modules={'model.layers.0.self_attn.q_proj':geom(),
             'model.layers.92.transformer.shared_head.output':geom(4,'mtp_core'),'lm_head':geom()}
    modules.update({f'model.layers.88.moe.experts.{e}.gate_proj':geom(6,'routed') for e in range(352)})
    originals={n+'.weight':('BF16',[128,128],1) for n in modules if '.experts.' not in n}
    originals['model.layers.88.moe.gate_proj.weight']=('BF16',[352,128,128],7)
    originals.update({f'model.layers.{l}.input_layernorm.weight':('F32',[128],2) for l in range(95)})
    originals['model.layers.88.moe.router_bias']=('F32',[352],3)
    originals['vision_model.ln_pre.weight']=('BF16',[128],4)
    original=save(src/'source.safetensors',originals)
    for name in ('LICENSE.txt','LICENSE-CN.txt','tokenizer.json','chat_template.jinja'):(src/name).write_text('native '+name)
    source=SimpleNamespace(snapshot=src,weight_map={n:'source.safetensors' for n in originals},document={'model_type':'step5_vl','text_config':{'hidden_size':128}})
    candidates={n:packed(tmp_path/f'candidate-{i}.safetensors',n,g) for i,(n,g) in enumerate(modules.items())}
    source_size=sum(math.prod(s)*{'BF16':2,'F32':4}[d] for d,s,_ in originals.values())
    selected_sources=recipe.selected_source_names(modules)
    kept=sum(math.prod(s)*{'BF16':2,'F32':4}[d] for n,(d,s,_) in originals.items() if n not in selected_sources)
    target_size=kept+sum(recipe.packed_bytes(g) for g in modules.values())
    for name,value in [('BASE_TENSOR_COUNT',len(originals)),('BASE_PAYLOAD_BYTES',source_size),('TARGET_TENSOR_COUNT',len(originals)-len(selected_sources)+4*len(modules)),('TARGET_PAYLOAD_BYTES',target_size)]:monkeypatch.setattr(export,name,value)
    monkeypatch.setattr(export,'reviewed_modules',lambda:modules)
    monkeypatch.setattr(export,'selected_modules',lambda _:modules)
    return source,modules,candidates,out,evidence,target_size


def test_fused_expert_assembly_has_no_remaining_fused_weight_and_recovers_without_candidates(toy):
    source,modules,candidates,out,_,_=toy
    subset={n:p for n,p in candidates.items() if n.startswith('model.layers.88.')}
    receipt=export.assemble_layer(source,88,subset,out,{'run':'toy'})
    header,base=read_header(out/receipt['filename'])
    assert 'model.layers.88.moe.gate_proj.weight' not in header
    assert len([n for n in header if '.experts.' in n])==352*4
    assert header['model.layers.88.moe.experts.351.gate_proj.trellis']['shape']==[8,8,96]
    assert receipt['preserved_audit']['tensors']==2
    for path in subset.values():path.unlink()
    assert export.assemble_layer(source,88,subset,out,{'run':'toy'})==receipt


def test_fused_partial_expert_coverage_is_rejected(toy,monkeypatch):
    source,modules,candidates,out,_,_=toy
    reduced={n:g for n,g in modules.items() if n!='model.layers.88.moe.experts.351.gate_proj'}
    monkeypatch.setattr(export,'reviewed_modules',lambda:reduced)
    subset={n:p for n,p in candidates.items() if n.startswith('model.layers.88.') and n in reduced}
    with pytest.raises(ValueError,match='every expert'):export.assemble_layer(source,88,subset,out,{'run':'toy'})


def test_full_metadata_index_and_native_assets(toy):
    source,modules,candidates,out,evidence,target=toy
    for layer in range(95):
        export.assemble_layer(source,layer,{n:p for n,p in candidates.items() if n.startswith(f'model.layers.{layer}.')},out,{'run':'toy'})
    result=export.finalize_export(source,modules,{'lm_head':candidates['lm_head']},out,{'run':'toy'},evidence)
    assert result['shards']==96 and result['payload_bytes']==target
    q=json.loads((out/'quantize_config.json').read_text());config=json.loads((out/'config.json').read_text())
    assert q['core_bits']==6 and q['mtp_bits']==4 and q['codebook']=='mcg'
    assert q['expert_bits_by_layer']=={'88':6,'89':6,'90':6}
    assert q['tensor_storage']['model.layers.88.moe.experts.1.gate_proj']['bits_per_weight']==6
    assert q['tensor_storage']['model.layers.92.transformer.shared_head.output']['bits_per_weight']==4
    assert q['kept_tensors']['model.layers.88.moe.router_bias']['torch_dtype']=='float32'
    assert 'vision_bits' not in q and 'vision_bits' not in config['quantization_config']
    assert (out/'LICENSE.txt').read_text()=='native LICENSE.txt'
    assert (out/'LICENSE-CN.txt').read_text()=='native LICENSE-CN.txt'
    assert 'license_name: stepfun-model-license-1.0' in (out/'README.md').read_text()
    index=json.loads((out/'model.safetensors.index.json').read_text())
    assert index['weight_map']['model.layers.92.transformer.shared_head.output.trellis']=='model-00093-of-00096.safetensors'
    assert index['weight_map']['vision_model.ln_pre.weight']=='model-00096-of-00096.safetensors'
    assert (out/'quantization-errors/observed.json').is_file()


@pytest.mark.parametrize('damage',['mcg','scale','shape'])
def test_bad_packed_candidate_rejected(toy,damage):
    source,modules,candidates,out,_,_=toy;n='model.layers.0.self_attn.q_proj'
    packed(candidates[n],n,modules[n],damage)
    with pytest.raises(ValueError):export.assemble_layer(source,0,{n:candidates[n]},out,'run')


def test_native_byte_corruption_is_detected_before_receipt(toy,monkeypatch):
    source,_,candidates,out,_,_=toy;n='model.layers.0.self_attn.q_proj';assemble=export.assemble_shard
    def corrupt(entries,path):
        result=assemble(entries,path);h,base=read_header(path)
        with path.open('r+b') as f:
            f.seek(base+h['model.layers.0.input_layernorm.weight']['data_offsets'][0]);f.write(b'\xff')
        return result
    monkeypatch.setattr(export,'assemble_shard',corrupt)
    with pytest.raises(ValueError,match='preserved tensor bytes'):export.assemble_layer(source,0,{n:candidates[n]},out,'run')
    assert not (out/'layer-000-receipt.json').exists()


def test_committed_shard_checksum_rechecked_after_candidate_retirement(toy):
    source,modules,candidates,out,_,_=toy;n='model.layers.0.self_attn.q_proj'
    receipt=export.assemble_layer(source,0,{n:candidates[n]},out,'run')
    candidates[n].unlink();p=out/receipt['filename']
    with p.open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
    with pytest.raises(ValueError):export.assemble_layer(source,0,{n:candidates[n]},out,'run')


def test_export_legacy_receipts_and_truthful_per_module_provenance(toy):
    from test_step5_core_compat import identities,manifest
    from step5.core_k6.compat import identity_digest
    source,modules,candidates,out,evidence,target=toy
    old,current=identities();old_module='model.layers.0.self_attn.q_proj'
    manifest(evidence.parent,old,current,old_module)
    old_receipt=export.assemble_layer(source,0,{old_module:candidates[old_module]},out,old)
    receipt_path=out/'layer-000-receipt.json';before=receipt_path.read_bytes()
    candidates[old_module].unlink()
    assert export.assemble_layer(source,0,{old_module:candidates[old_module]},out,current)==old_receipt
    for layer in range(1,95):
        export.assemble_layer(source,layer,{n:p for n,p in candidates.items() if n.startswith(f'model.layers.{layer}.')},out,current)
    # Legacy evidence remains byte-identical; new evidence explicitly names its
    # active producer. These records are copies, not rewritten old controls.
    (evidence/'observed.json').unlink()
    legacy=evidence/'old.json';legacy.write_text(json.dumps(dict(module=old_module,bits=6)))
    legacy_bytes=legacy.read_bytes()
    for i,module in enumerate(n for n in modules if n!=old_module):
        (evidence/f'new-{i}.json').write_text(json.dumps(dict(module=module,producer_identity_sha256=identity_digest(current))))
    result=export.finalize_export(source,modules,{'lm_head':candidates['lm_head']},out,current,evidence)
    assert result['payload_bytes']==target
    storage=json.loads((out/'quantize_config.json').read_text())['tensor_storage']
    assert storage[old_module]['producer_identity_sha256']==identity_digest(old)
    assert storage['lm_head']['producer_identity_sha256']==identity_digest(current)
    assert json.loads((out/'quantization-provenance.json').read_text())==current
    assert (out/'producer-lineage.json').read_bytes()==(evidence.parent/'producer-lineage.json').read_bytes()
    assert receipt_path.read_bytes()==before and legacy.read_bytes()==legacy_bytes
    assert (out/'quantization-errors/old.json').read_bytes()==legacy_bytes


def test_export_rejects_legacy_shard_without_explicit_origin_map(toy):
    from test_step5_core_compat import identities
    source,_,candidates,out,_,_=toy;old,current=identities();n='model.layers.0.self_attn.q_proj'
    export.assemble_layer(source,0,{n:candidates[n]},out,old)
    changed=json.loads(json.dumps(current));changed['recipe']['core_bits']=5
    with pytest.raises(ValueError,match='identity'):export.assemble_layer(source,0,{n:candidates[n]},out,changed)
    with pytest.raises(ValueError,match='identity'):export.assemble_layer(source,0,{n:candidates[n]},out,dict(run='unlisted'))
