"""Fresh covariance capture from native-only teacher banks, without GPU use."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import requalify, pilot
from kimi_k3.uniform_k2.storage import atomic_json, canonical, sha256_file


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    old=tmp_path/'pilot-before-k128';old.mkdir();new=tmp_path/'pilot';new.mkdir();repo=tmp_path/'repo';repo.mkdir()
    for name in set(requalify.NATIVE_CODE+requalify.CAPTURE_CODE):
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('current '+name)
    code={name:sha256_file(repo/name) for name in set(requalify.NATIVE_CODE+requalify.CAPTURE_CODE)}
    records=dict(calibration=[dict(id='cal-0',input_ids=[1,2])],heldout=[dict(id='hold-0',input_ids=[3])])
    rows={split:[dict(id=r['id'],tokens=len(r['input_ids']),ids_sha256=hashlib.sha256(canonical(r['input_ids']).encode()).hexdigest())
        for r in values] for split,values in records.items()}
    identity=dict(schema=requalify.PILOT_SCHEMA,source_repo='meta-models/Muse-Spark-Open-816B-A42B',
        source_revision='source-pin',source_config_sha256='a'*64,source_index_sha256='b'*64,
        token_manifest_sha256='c'*64,native=dict(torch='cpu-test',adapter_sha256=code[requalify.NATIVE_CODE[0]],modelwheel='pinned'),
        records=rows,witness=dict(input_ids=[4]*2049,tokens=2049,use='excluded-from-all-Hessians'),
        code=dict(code),recipe=dict(native_reference=True),gptqmodel_commit='old-fork')
    identity['code']['quantization/muse_spark/core_k6/codec.py']='old-bad-H-code'
    current=copy.deepcopy(identity);current['code']=code;current['gptqmodel_commit']='new-K128-fork'
    report=dict(schema=requalify.PILOT_SCHEMA,status='passed',identity=identity,
        observations=[dict(legacy_quantized_candidate='must-not-be-used')])
    atomic_json(old/'qualification.json',report);atomic_json(old/'identity.json',identity)
    batches=[row for split in ('calibration','heldout') for row in records[split]]
    hidden={}
    for index,row in enumerate(batches):
        base=torch.tensor([[[float(token),float(token+1)] for token in row['input_ids']]],dtype=torch.bfloat16)
        values={'frontier-embedding':base,'frontier-000':base+1,'frontier-001':base+1+2,
            'frontier-060':base+60,'frontier-061':base+60+62}
        for bank,value in values.items():
            path=old/bank/f'{index:03d}.safetensors';path.parent.mkdir(exist_ok=True)
            pilot.save_tensor(path,dict(hidden=value),identity);hidden[(bank,index)]=value
    pilot.save_tensor(old/'frontier-embedding/witness.safetensors',dict(hidden=torch.ones((1,2049,2),dtype=torch.bfloat16)),identity)
    for layer in (0,1,60,61):
        manifest=dict(identity=identity,layer=layer,directory=f'frontier-{layer:03d}',proofs={})
        atomic_json(old/f'frontier-{layer:03d}/manifest.json',manifest)
    atomic_json(old/'capture-current.json',manifest)
    # Deliberately invalid legacy numerical files: opening any of these is a bug.
    (old/'frontier-000/calibration').mkdir()
    forbidden=old/'frontier-000/calibration/self_attn.q_proj.safetensors'
    forbidden.write_bytes(b'OLD-HESSIAN-MUST-NEVER-BE-OPENED')
    (old/'candidates').mkdir();(old/'candidates/packed.safetensors').write_bytes(b'OLD-QUANT-CANDIDATE')
    (old/'head-hessian-calibration.safetensors').write_bytes(b'OLD-HEAD-HESSIAN')
    monkeypatch.setattr(requalify,'NATIVE_WIDTH',2)
    monkeypatch.setattr(requalify,'REPO',repo)
    monkeypatch.setattr(requalify,'_device',lambda:torch.device('cpu'))
    monkeypatch.setattr(pilot,'CORE',('self_attn.q_proj',))
    calls=[];captures=[];proofs=[];fail_layer=[None]
    class Block:
        def __init__(self,layer):self.layer=layer;self.capture=None
    def load_block(source,layer,device,include_experts):
        assert include_experts is True and str(device)=='cpu'
        calls.append(layer)
        return Block(layer)
    def forward(block,state,rotary):
        assert rotary=='native-rotary'
        if fail_layer[0]==block.layer:raise MemoryError('simulated OOM')
        x=state['hidden']
        if block.capture is not None:block.capture.add(x)
        return dict(hidden=x+(block.layer+1))
    class Capture:
        def __init__(self,block):
            assert block.capture is None
            self.block=block;block.capture=self;self.inputs=[];captures.append(self)
        def add(self,hidden):self.inputs.append(hidden.detach().cpu().clone())
        def close(self):self.block.capture=None
        def save(self,directory,identity):
            directory.mkdir(parents=True)
            x=torch.cat([v.reshape(-1,2) for v in self.inputs]);f=x.float();H=f.T@f
            pilot.save_tensor(directory/'self_attn.q_proj.safetensors',dict(H=H),identity)
            pilot.save_tensor(directory/'experts.safetensors',dict(inputs=x,
                selected=torch.tensor([[0,1]]*len(x),dtype=torch.int64)),identity)
            atomic_json(directory/'manifest.json',dict(identity=identity,counts={'self_attn.q_proj':len(x)}))
    def byte_proof(block,hidden,rotary):
        assert block.capture is None
        proofs.append(('byte',block.layer))
        out=forward(block,dict(hidden=hidden),rotary)['hidden']
        return dict(status='passed',byte_exact=True,tokens=hidden.shape[1],output_sha256=tensor_sha(out))
    def window_proof(block,hidden,rotary):
        assert block.capture is None
        proofs.append(('window',block.layer))
        return dict(status='passed',byte_exact=True,tokens=hidden.shape[1],visible_tail_tokens=2048,hessian_rows=0)
    monkeypatch.setattr(requalify.numerical,'load_block',load_block)
    monkeypatch.setattr(requalify.numerical,'rotary',lambda source,device:'native-rotary')
    monkeypatch.setattr(requalify.numerical,'native_forward',forward)
    monkeypatch.setattr(pilot,'Capture',Capture)
    monkeypatch.setattr(pilot,'byte_proof',byte_proof)
    monkeypatch.setattr(pilot,'window_proof',window_proof)
    source=SimpleNamespace(config=dict(hidden_size=2,moe_hidden_size=2,num_experts_per_tok=2,num_local_experts=4))
    return SimpleNamespace(old=old,new=new,repo=repo,old_identity=identity,identity=current,records=records,
        source=source,hidden=hidden,forbidden=forbidden,calls=calls,captures=captures,proofs=proofs,fail_layer=fail_layer)


def qualify_identity(case):
    origin=requalify.native_origin(case.old,case.identity,case.repo)
    case.identity['native_reference_origin']=origin
    atomic_json(case.new/'identity.json',case.identity)
    return origin


def run(case):
    return requalify.capture_from_native(case.source,case.records,case.new,case.identity,case.old)


def test_native_origin_allows_only_native_banks_and_quantizer_change(case):
    origin=qualify_identity(case)
    assert origin['old_report_sha256']==sha256_file(case.old/'qualification.json')
    assert len(origin['native_files'])==5*2+1
    assert all(Path(row['relative']).parts[0] in requalify.BANKS for row in origin['native_files'])
    assert not any('calibration/' in row['relative'] or 'candidate' in row['relative'] for row in origin['native_files'])
    assert 'inherited_numerical_producers' not in case.identity


@pytest.mark.parametrize('field',['token_manifest_sha256','source_config_sha256','source_index_sha256','native','records','witness'])
def test_native_origin_rejects_changed_tokenizer_adapter_or_inputs(case,field):
    changed=copy.deepcopy(case.identity)
    changed[field]='different'
    with pytest.raises(ValueError,match='identity differs'):
        requalify.native_origin(case.old,changed,case.repo)


def test_changed_native_adapter_code_is_rejected(case):
    relative=requalify.NATIVE_CODE[0]
    (case.repo/relative).write_text('changed native adapter')
    case.identity['code'][relative]=sha256_file(case.repo/relative)
    with pytest.raises(ValueError,match='code differs'):
        requalify.native_origin(case.old,case.identity,case.repo)


def test_corrupt_native_teacher_payload_is_rejected(case):
    path=case.old/'frontier-060/000.safetensors'
    with path.open('r+b') as stream:stream.seek(-1,2);stream.write(b'X')
    with pytest.raises(ValueError,match='committed file'):
        requalify.native_origin(case.old,case.identity,case.repo)


def test_non_bf16_teacher_bank_is_rejected_even_with_valid_receipt(case):
    path=case.old/'frontier-060/000.safetensors';path.unlink();path.with_suffix('.receipt.json').unlink()
    pilot.save_tensor(path,dict(hidden=case.hidden[('frontier-060',0)].float()),case.old_identity)
    with pytest.raises(ValueError,match='not a native BF16'):
        requalify.native_origin(case.old,case.identity,case.repo)


def test_requalification_recaptures_h_and_outputs_without_opening_old_quant_state(case,monkeypatch):
    qualify_identity(case)
    original_open=Path.open
    def checked_open(path,*args,**kwargs):
        if path.is_relative_to(case.old) and ('calibration' in path.parts or 'candidates' in path.parts or path.name.startswith('head-hessian')):
            raise AssertionError('old Hessian/candidate accessed: '+str(path))
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked_open)
    current=run(case)
    assert case.calls==[0,1,61]
    assert len(case.captures)==6
    assert case.proofs==[('byte',0),('window',0),('byte',1),('byte',61)]
    assert current['identity']==case.identity and current['directory']=='frontier-061' and current['layer']==61
    for layer in (0,1,61):
        assert len(json.loads((case.new/f'frontier-{layer:03d}/manifest.json').read_text())['native_output_checks'])==2
        for index in range(2):
            value=pilot.read_tensor(case.new/f'frontier-{layer:03d}/{index:03d}.safetensors',case.identity)['hidden']
            assert torch.equal(value.view(torch.uint8),case.hidden[(f'frontier-{layer:03d}',index)].view(torch.uint8))
    raw=pilot.read_tensor(case.new/'frontier-000/calibration/self_attn.q_proj.safetensors',case.identity)['H']
    x=case.hidden[('frontier-embedding',0)].reshape(-1,2).float()
    assert torch.equal(raw,x.T@x)
    assert not (case.new/'candidates').exists()
    assert not (case.new/'head-hessian-calibration.safetensors').exists()


def test_teacher_output_difference_prevents_commit(case,monkeypatch):
    qualify_identity(case)
    original=requalify.numerical.native_forward
    def wrong(block,state,rotary):
        result=original(block,state,rotary)
        result['hidden'][...,0]+=1
        return result
    monkeypatch.setattr(requalify.numerical,'native_forward',wrong)
    with pytest.raises(ValueError,match='byte-for-byte'):run(case)
    assert not (case.new/'frontier-000/manifest.json').exists()
    assert not (case.new/'capture-current.json').exists()


def test_partial_oom_preserves_directory_and_resumes_completed_layers(case):
    qualify_identity(case);case.fail_layer[0]=1
    with pytest.raises(MemoryError):run(case)
    assert (case.new/'frontier-000/manifest.json').exists()
    assert not (case.new/'frontier-001/manifest.json').exists()
    case.fail_layer[0]=None
    current=run(case)
    assert current['layer']==61
    assert case.calls==[0,1,1,61]
    assert list(case.new.glob('frontier-001.incomplete-*'))
    calls=list(case.calls);assert run(case)['layer']==61
    assert case.calls==calls


def test_new_capture_identity_and_hessian_corruption_block_resume(case):
    qualify_identity(case);run(case)
    path=case.new/'frontier-001/calibration/self_attn.q_proj.safetensors'
    with path.open('r+b') as stream:stream.seek(-1,2);stream.write(b'X')
    with pytest.raises(ValueError,match='committed file'):run(case)
    assert case.calls==[0,1,61]


def test_missing_global_native_origin_cannot_reuse_reference(case):
    atomic_json(case.new/'identity.json',case.identity)
    with pytest.raises(ValueError,match='does not bind'):run(case)
    assert case.calls==[]


def test_old_hessian_and_candidate_paths_are_never_accepted(case):
    for path in ('frontier-000/calibration/self_attn.q_proj.safetensors','candidates/packed.safetensors',
        'head-hessian-calibration.safetensors','frontier-060/../000.safetensors'):
        with pytest.raises(ValueError,match='allowlist'):requalify._native_path(case.old,path)
