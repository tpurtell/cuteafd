"""Muse fused-span, complete replacement, byte-preservation and layout checks."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import export, recipe, source, publish
from kimi_k3.uniform_k2.storage import read_header


def write_tensors(path, tensors):
    header, blocks, offset = {}, [], 0
    for name, (dtype, shape, payload) in sorted(tensors.items()):
        assert len(payload)==math.prod(shape)*{'BF16':2,'F32':4,'F16':2,'I16':2,'I32':4}[dtype]
        header[name]=dict(dtype=dtype,shape=shape,data_offsets=[offset,offset+len(payload)])
        blocks.append(payload);offset+=len(payload)
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%8)
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(blocks))
    return path


def geometry(bits=4, inputs=128, outputs=128, group='routed'):
    return recipe._geometry(inputs,outputs,bits,group)


def packed(path,module,g):
    tensors={}
    for name,(dtype,shape) in recipe.packed_specs(module,g).items():
        if dtype=='I32':data=struct.pack('<I',recipe.MCG_MULTIPLIER)
        elif dtype=='F16':
            values=np.ones(shape,dtype='<f2')
            if name.endswith('.svh'):values[g['out_features']:]=0
            data=values.tobytes()
        else:data=b'\0'*(math.prod(shape)*2)
        tensors[name]=(dtype,shape,data)
    return write_tensors(path,tensors)


@pytest.fixture
def fused_case(tmp_path):
    prefix='model.language_model.layers.0.mlp.experts.'
    selection={prefix+f'{e}.{part}':geometry() for e in range(2) for part in ('gate_proj','up_proj','down_proj')}
    tensors={prefix+'gate_up_proj':('BF16',[2,256,128],b'\x01\x00'*(2*256*128)),
        prefix+'down_proj':('BF16',[2,128,128],b'\x02\x00'*(2*128*128)),
        prefix+'post_expert_norm.weight':('BF16',[128],b'\x03\x00'*128),
        'model.language_model.layers.0.post_attention_residual_gate.gate':('BF16',[128],b'\x04\x00'*128)}
    path=write_tensors(tmp_path/'source.safetensors',tensors)
    entries={name:(path,name) for name in tensors}
    replacements={name:packed(tmp_path/f'candidate-{i}.safetensors',name,g) for i,(name,g) in enumerate(selection.items())}
    return SimpleNamespace(prefix=prefix,selection=selection,tensors=tensors,path=path,entries=entries,
        replacements=replacements,target=tmp_path/'target.safetensors',receipt=tmp_path/'receipt.json')


def test_fused_source_is_removed_once_and_native_bytes_preserved(fused_case):
    c=fused_case
    receipt=export._assemble(None,c.entries,c.selection,c.replacements,c.target,c.receipt,{'producer':'test'},layer=0)
    header,_=read_header(c.target)
    assert c.prefix+'gate_up_proj' not in header
    assert c.prefix+'down_proj' not in header
    assert len(header)==6*4+2
    assert receipt['payload_bytes']==sum(recipe.packed_bytes(g) for g in c.selection.values())+512
    assert receipt['preserved_audit']['tensors']==2
    assert receipt['preserved_audit']['payload_bytes']==512
    assert export._assemble(None,c.entries,c.selection,c.replacements,c.target,c.receipt,{'producer':'test'},layer=0)==receipt


def test_partial_fused_expert_replacement_is_rejected(fused_case):
    c=fused_case;selection=dict(c.selection);replacements=dict(c.replacements)
    del selection[c.prefix+'1.up_proj'];del replacements[c.prefix+'1.up_proj']
    with pytest.raises(ValueError,match='partial fused'):
        export._assemble(None,c.entries,selection,replacements,c.target,c.receipt,{'producer':'test'})


def test_split_expert_wrong_shape_rejected(fused_case):
    c=fused_case;selection=copy.deepcopy(c.selection)
    selection[c.prefix+'0.gate_proj']['in_features']=256
    with pytest.raises(ValueError,match='split expert'):
        export._assemble(None,c.entries,selection,c.replacements,c.target,c.receipt,{'producer':'test'})


def test_native_byte_audit_detects_corruption(fused_case):
    c=fused_case
    export._assemble(None,c.entries,c.selection,c.replacements,c.target,c.receipt,{'producer':'test'})
    header,base=read_header(c.target);name=c.prefix+'post_expert_norm.weight'
    with c.target.open('r+b') as file:
        file.seek(base+header[name]['data_offsets'][0]);file.write(b'\xff')
    native={name:c.entries[name]}
    with pytest.raises(ValueError,match='preserved tensor bytes'):
        export._preserved_audit(native,c.target)


def test_multiplier_mismatch_rejected(fused_case):
    c=fused_case;module=next(iter(c.selection));path=c.replacements[module]
    header,base=read_header(path)
    with path.open('r+b') as file:
        file.seek(base+header[module+'.mcg']['data_offsets'][0]);file.write(struct.pack('<I',0))
    with pytest.raises(ValueError,match='multiplier'):
        export._assemble(None,c.entries,c.selection,c.replacements,c.target,c.receipt,{'producer':'test'})


def test_padded_head_requires_zero_output_scales(tmp_path):
    g=geometry(6,128,192,'base_core');path=packed(tmp_path/'head.safetensors','lm_head',g)
    export._check_packed('lm_head',g,path)
    header,base=read_header(path)
    with path.open('r+b') as file:
        file.seek(base+header['lm_head.svh']['data_offsets'][0]+192*2);file.write(struct.pack('<e',1))
    with pytest.raises(ValueError,match='padded output'):
        export._check_packed('lm_head',g,path)


def test_nonfinite_scale_rejected(tmp_path):
    g=geometry();path=packed(tmp_path/'packed.safetensors','projection',g)
    header,base=read_header(path)
    with path.open('r+b') as file:
        file.seek(base+header['projection.suh']['data_offsets'][0]);file.write(struct.pack('<e',float('nan')))
    with pytest.raises(ValueError,match='nonfinite'):
        export._check_packed('projection',g,path)


def test_reviewed_head_and_whole_side_contract():
    modules=recipe.reviewed_modules();head=modules['lm_head']
    assert (head['out_features'],head['stored_out_features'])==(202048,202112)
    assert all(g['sharded_side']==1 for g in modules.values())
    assert len(modules)==47989
    assert recipe.TARGET_TENSOR_COUNT==193459
    assert recipe.TARGET_PAYLOAD_BYTES==427855788244
    assert len(recipe.removed_source_names(modules))==497


def test_bounded_expert_offsets_and_gate_up_order(monkeypatch):
    s=object.__new__(source.MuseSource)
    def header(name):
        shape=[256,4096 if name.endswith('down_proj') else 8192,4096]
        return Path('unused'),dict(dtype='BF16',shape=shape),0
    monkeypatch.setattr(s,'header',header)
    monkeypatch.setattr(s,'_read',lambda name,shape,**kw:dict(name=name,shape=shape,**kw))
    matrix=4096*4096*2
    gate=s.expert(0,7,'gate_proj');up=s.expert(0,7,'up_proj');down=s.expert(0,7,'down_proj')
    assert gate['offset']==14*matrix
    assert up['offset']==15*matrix
    assert down['offset']==7*matrix
    assert gate['shape']==up['shape']==down['shape']==[4096,4096]
    with pytest.raises(ValueError):s.expert(0,-1,'gate_proj')
    with pytest.raises(ValueError):s.expert(0,256,'gate_proj')
    with pytest.raises(ValueError):s.expert(62,0,'gate_proj')
    with pytest.raises(ValueError):s.expert(0,0,'unknown')


def test_source_span_rejects_outside_reads(monkeypatch):
    s=object.__new__(source.MuseSource)
    monkeypatch.setattr(s,'header',lambda _: (Path('unused'),dict(data_offsets=[10,30]),100))
    assert s.span('x',offset=4,length=6)==(Path('unused'),114,6)
    for kw in ({'offset':-1},{'offset':21},{'offset':19,'length':2},{'length':-2}):
        with pytest.raises(ValueError):s.span('x',**kw)


def test_publisher_diagnostic_bundles_have_exact_records(tmp_path,monkeypatch):
    monkeypatch.setattr(publish,'MODULE_COUNT',2)
    errors=tmp_path/'quantization-errors';group=errors/'layer-061';group.mkdir(parents=True)
    for index in range(2):(group/f'{index}.json').write_text(json.dumps(dict(module=f'm{index}',bits=4)))
    publish._bundle_diagnostics(tmp_path)
    index=json.loads((errors/'index.json').read_text())
    assert index['records']==2
    assert index['bundles']['layer-061.jsonl']['records']==2
    assert len((errors/'layer-061.jsonl').read_text().splitlines())==2
    publish._bundle_diagnostics(tmp_path,frozen=True)
    (group/'0.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='completed diagnostic'):
        publish._bundle_diagnostics(tmp_path,frozen=True)


def test_final_export_and_independent_reload_agree(tmp_path,monkeypatch):
    """Exercise split fusion through metadata, final native audit and reload."""
    from muse_spark.core_k6 import validate
    src=tmp_path/'original';src.mkdir();out=tmp_path/'output';root=tmp_path/'run';root.mkdir()
    prefix='model.language_model.layers.0.'
    selection={prefix+'mlp.experts.'+f'{e}.{part}':geometry() for e in range(2) for part in ('gate_proj','up_proj','down_proj')}
    selection[prefix+'self_attn.q_proj']=geometry(6,group='base_core')
    selection['lm_head']=geometry(6,outputs=192,group='base_core')
    tensors={prefix+'mlp.experts.gate_up_proj':('BF16',[2,256,128],b'\x01\x00'*(2*256*128)),
        prefix+'mlp.experts.down_proj':('BF16',[2,128,128],b'\x02\x00'*(2*128*128)),
        prefix+'self_attn.q_proj.weight':('BF16',[128,128],b'\x03\x00'*(128*128)),
        'lm_head.weight':('BF16',[192,128],b'\x04\x00'*(192*128)),
        'model.language_model.embed_tokens.weight':('BF16',[3,128],b'\x05\x00'*(3*128))}
    tensors.update({f'model.language_model.layers.{layer}.input_layernorm.weight':('BF16',[128],bytes([layer,0])*128)
        for layer in range(62)})
    shard=write_tensors(src/'original.safetensors',tensors)
    document=dict(model_type='muse_spark',text_config=dict(hidden_size=128))
    source_value=SimpleNamespace(snapshot=src,document=document,config=document['text_config'],
        weight_map={name:shard.name for name in tensors})
    native_names=set(tensors)-recipe.removed_source_names(selection)
    native_bytes=sum(len(tensors[n][2]) for n in native_names)
    target_bytes=native_bytes+sum(recipe.packed_bytes(g) for g in selection.values())
    target_count=len(native_names)+len(selection)*4
    original_bytes=sum(len(v[2]) for v in tensors.values())
    monkeypatch.setattr(export,'reviewed_modules',lambda:selection)
    monkeypatch.setattr(export,'selected_modules',lambda _:selection)
    monkeypatch.setattr(export,'BASE_TENSOR_COUNT',len(tensors))
    monkeypatch.setattr(export,'BASE_PAYLOAD_BYTES',original_bytes)
    monkeypatch.setattr(export,'TARGET_PAYLOAD_BYTES',target_bytes)
    monkeypatch.setattr(export,'TARGET_TENSOR_COUNT',target_count)
    replacements={name:packed(tmp_path/f'packed-{i}.safetensors',name,g) for i,(name,g) in enumerate(selection.items())}
    evidence=tmp_path/'evidence';evidence.mkdir()
    identity=dict(source_repo=recipe.BASE_REPO,source_revision=recipe.BASE_REVISION,calibration=dict(samples=2,tokens=256))
    for name,g in selection.items():
        directory=evidence/('head' if name=='lm_head' else 'layer-000');directory.mkdir(exist_ok=True)
        (directory/(name+'.json')).write_text(json.dumps(dict(module=name,bits=g['bits'],
            logical_shape=[g['out_features'],g['in_features']],metrics=dict(error=.001),evidence=dict(hessian_rows=256))))
    for asset in ('LICENSE.md','USE_POLICY.md','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json','processor_config.json'):
        (src/asset).write_text('original '+asset)
    for layer in range(62):
        names={n:p for n,p in replacements.items() if n.startswith(f'model.language_model.layers.{layer}.')}
        export.assemble_layer(source_value,layer,names,out,identity)
    summary=export.finalize_export(source_value,selection,{'lm_head':replacements['lm_head']},out,identity,evidence)
    assert summary['payload_bytes']==target_bytes
    storage=json.loads((out/'quantize_config.json').read_text())
    assert storage['retained_native_tensors']=={name:dict(dtype=tensors[name][0],shape=tensors[name][1]) for name in native_names}
    assert storage['tensor_storage']['lm_head']['stored_out_features']==256
    assert storage['tensor_storage']['lm_head']['out_features']==192
    assert storage['tensor_storage'][prefix+'self_attn.q_proj']['sharded_side']==1
    assert 'mtp_bits' not in storage
    plan=dict(identity=identity,root=str(root),export=str(out),publication_repo=recipe.TARGET_REPO)
    (root/'plan.json').write_text(json.dumps(plan));(root/'quantization-complete.json').write_text(json.dumps(dict(identity=identity,summary=summary)))
    monkeypatch.setattr(validate,'inventory',lambda _:dict(preserved_bytes=native_bytes))
    monkeypatch.setattr(validate,'reviewed_modules',lambda:selection)
    monkeypatch.setattr(validate,'TARGET_PAYLOAD_BYTES',target_bytes)
    monkeypatch.setattr(validate,'TARGET_TENSOR_COUNT',target_count)
    audit=validate.audit_completed(source_value,root,out)
    assert audit['status']=='passed'
    assert audit['numerical_records']==8
    assert audit['preserved_bytes']==native_bytes
    # Header+receipt agreement is insufficient if the metadata lies about padding.
    storage['tensor_storage']['lm_head']['stored_out_features']=192
    (out/'quantize_config.json').write_text(json.dumps(storage))
    with pytest.raises(ValueError,match='storage declaration'):
        validate.audit_completed(source_value,root,out)


def test_local_source_requires_sealed_nonwriting_union(tmp_path,monkeypatch):
    from muse_spark.core_k6 import source_audit
    snapshot=tmp_path/recipe.BASE_REVISION;snapshot.mkdir();(snapshot/'weight').write_bytes(b'123')
    original_resolve=Path.resolve
    def resolve(path,*args,**kwargs):
        if path==snapshot/'weight':return Path('/mnt/sparknest/hf-home/hub/blobs/xx/digest')
        return original_resolve(path,*args,**kwargs)
    original_stat=Path.stat
    def stat(path,*args,**kwargs):
        if str(path)=='/mnt/sparknest/hf-home/hub/blobs/xx/digest':return SimpleNamespace(st_size=3)
        return original_stat(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',resolve);monkeypatch.setattr(Path,'stat',stat)
    entry=dict(name='digest',kind='regular',hosts=['raptor (D3)'],size=3,sealed=True,writing_on=None)
    def run(*args,**kwargs):return SimpleNamespace(stdout=json.dumps(dict(entries=[entry])))
    assert source_audit.require_local_source(snapshot,runner=run)['status']=='ready'
    entry['hosts']=['ostrich']
    with pytest.raises(ValueError,match='local-ready'):source_audit.require_local_source(snapshot,runner=run)
    entry['hosts']=['raptor (D2)'];entry['writing_on']='raptor (D2)'
    with pytest.raises(ValueError,match='local-ready'):source_audit.require_local_source(snapshot,runner=run)
    entry['writing_on']=None;entry['sealed']=False
    with pytest.raises(ValueError,match='local-ready'):source_audit.require_local_source(snapshot,runner=run)


def test_owned_source_reads_preserve_bf16_and_float32_bits(tmp_path):
    torch=pytest.importorskip('torch')
    tensors={'bf16':('BF16',[2,3],b'\x80\x3f\x00\xc0\x40\x40\x80\xc0\xa0\x40\x00\x00'),
        'scalar':('F32',[],struct.pack('<f',.125))}
    path=write_tensors(tmp_path/'part.safetensors',tensors)
    (tmp_path/'config.json').write_text(json.dumps(dict(text_config=dict(hidden_size=3))))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={n:path.name for n in tensors})))
    src=source.MuseSource(tmp_path)
    original=src.tensor('bf16')
    assert original.dtype==torch.bfloat16
    assert original.reshape(-1).view(torch.uint8).numpy().tobytes()==tensors['bf16'][2]
    sliced=src._read('bf16',[1,3],offset=6)
    assert sliced.reshape(-1).view(torch.uint8).numpy().tobytes()==tensors['bf16'][2][6:]
    assert original.untyped_storage().data_ptr()!=sliced.untyped_storage().data_ptr()
    assert src.tensor('scalar').item()==.125
