"""Muse MC core-only geometry, causal native-expert replay and durable oracles."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import recipe, export


def test_k8_selection_and_default_full_recipe():
    original=recipe.reviewed_modules()
    core=recipe.reviewed_modules(core_bits=8,core_only=True)
    assert len(core)==373 and set(core)==set(recipe.reviewed_core_modules())
    assert len(original)==47989
    assert {g['bits'] for g in original.values() if g['group']=='base_core'}=={6}
    assert {g['bits'] for g in original.values() if g['group']=='routed'}=={4}
    assert {g['bits'] for g in core.values()}=={8}
    for name,g in core.items():
        assert recipe.packed_specs(name,g)[name+'.trellis'][1][-1]==128
        assert g['sharded_side']==1
    assert core['lm_head']['out_features']==202048
    assert core['lm_head']['stored_out_features']==202112
    with pytest.raises(ValueError):recipe.reviewed_core_modules(core_bits=True)


def test_core_only_shard_and_storage_exclude_all_native_payload(tmp_path,monkeypatch):
    shared=pytest.importorskip('core_export')
    from test_muse_inventory_export import packed
    module='model.language_model.layers.0.self_attn.q_proj'
    g=recipe._geometry(128,128,8,'base_core')
    path=packed(tmp_path/'replacement.safetensors',module,g)
    monkeypatch.setattr(export,'reviewed_core_modules',lambda **kw:{module:g})
    out=tmp_path/'export'
    # A source that cannot be accessed proves assembly reads only owned packed data.
    receipt=export.assemble_layer(None,0,{module:path},out,{'run':'test'},core_bits=8,core_only=True)
    summary=shared.finalize_core_export({module:g},out,{'run':'test'})
    from kimi_k3.uniform_k2.storage import read_header
    header,_=read_header(next(out.glob('*.safetensors')))
    assert set(header)=={module+'.'+s for s in recipe.SUFFIXES}
    assert summary['tensors']==4 and receipt['payload_bytes']==recipe.packed_bytes(g)
    storage=json.loads((out/'quantize_config.json').read_text())['tensor_storage']
    assert storage[module]['bits_per_weight']==8
    assert not (out/'config.json').exists()


def test_interface_defaults_and_k8_requires_core_only():
    pytest.importorskip('torch')
    from muse_spark.core_k6 import run
    args=run.argument_parser().parse_args(['--source-snapshot','source','--tokens','tokens','--root','root','--export','out'])
    assert args.core_bits==6 and not args.core_only
    assert run.producer_recipe()==run.RECIPE
    with pytest.raises(ValueError,match='requires --core-only'):run.producer_recipe(core_bits=8)
    candidate=run.producer_recipe(core_bits=8,core_only=True)
    assert candidate['head_bits']==candidate['core_bits']==8
    assert candidate['expert_bits']==16 and 'cold_experts' not in candidate


def test_codec_validates_k8_geometry_without_hardware():
    torch=pytest.importorskip('torch')
    from muse_spark.core_k6 import codec
    packed=dict(trellis=torch.zeros(8,8,128,dtype=torch.int16),
        suh=torch.ones(128,dtype=torch.float16),svh=torch.ones(128,dtype=torch.float16),
        mcg=torch.tensor(recipe.MCG_MULTIPLIER-2**32,dtype=torch.int32))
    assert codec.validate_projection(packed,[128,128])['bits']==8
    packed['trellis']=packed['trellis'][:,:,:112]
    with pytest.raises(ValueError,match='geometry'):codec.validate_projection(packed,[128,128])


@pytest.mark.parametrize('phase',[None,'a','b','c','e','f'])
def test_core_causal_replay_and_resume_never_quantizes_experts(tmp_path,monkeypatch,phase):
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    if transformers.__version__!='5.22.1':pytest.skip('official Muse 5.22.1 required')
    from test_muse_core_resume import fixture,Crash,changed_template
    from muse_spark.core_k6 import run
    engine,current,template,hidden,calls,_,progress=fixture(tmp_path,monkeypatch)
    engine.core_only=True;engine.core_bits=8
    # The existing disk-barrier witness uses a tiny deterministic codec, while
    # all official Muse attention, routing, norms and expert execution stay real.
    for name in list(engine.selection):
        if '.experts.' in name:del engine.selection[name]
        else:engine.selection[name]['bits']=8
    monkeypatch.setattr(run.RouteSpool,'create',lambda *a,**k:pytest.fail('expert capture in core-only producer'))
    native=[]
    def materialize(source,block,layer):
        native.append({key:value.clone() for key,value in block.mlp.experts.state_dict().items()})
    monkeypatch.setattr(run.numerical,'materialize_routed_experts',materialize)
    if phase is not None:
        progress.stop=('phase_committed',phase)
        with pytest.raises(Crash):engine.layer(0,current)
    result=engine.layer(0,current)
    assert result['layer']==0 and len(calls)==6
    assert all('.experts.' not in c['module'] and c['identity_rows']==0 for c in calls)
    assert all(c['count']==sum(len(r['input_ids']) for r in engine.records) for c in calls)
    expected=changed_template(engine,template)
    for state in native:
        for key,value in state.items():assert torch.equal(value,expected.mlp.experts.state_dict()[key])
    for index in range(len(engine.records)):
        actual=engine.boundary_batch(result,index)['hidden']
        reference=run.numerical.native_forward(expected,dict(hidden=hidden[index]))['hidden']
        assert torch.equal(actual,reference)
    assert native and not (engine.root/'layers/000/routes').exists()


def test_durable_packed_oracle_runs_on_creation_and_fresh_resume(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    shared=pytest.importorskip('core_export')
    from muse_spark.core_k6.storage import ProjectionStore
    from kimi_k3.uniform_k2.storage import atomic_json
    name='projection';g=recipe._geometry(128,128,8,'base_core');checks=[]
    def oracle(path,module,geometry,**kw):
        assert path.exists() and module==name and geometry['bits']==8
        checks.append(path)
        return dict(passed=True,sampled_sha256='a'*64)
    monkeypatch.setattr(shared,'sampled_packed_oracle',oracle)
    codec=SimpleNamespace(quantize_projection=lambda w,h,**kw:(dict(trellis=w),
        dict(hessian_weighted_relative_error=.0001),dict(hessian_rows=h['count'])))
    def store():return ProjectionStore(tmp_path/'packed',tmp_path/'evidence',{'test':True},{name:g},
        SimpleNamespace(emit=lambda *a,**kw:None),codec,device='cpu',core_only=True)
    first=store();first.ensure(name,torch.ones(128,128),dict(count=1426))
    assert len(checks)==1 and first.load(name)['evidence']['packed_byte_oracle']['passed']
    assert len(checks)==1
    assert store().load(name)['evidence']['packed_byte_oracle']['passed'] and len(checks)==2
    saved=first.load(name);saved['evidence']['packed_byte_oracle']['sampled_sha256']='b'*64
    atomic_json(first.paths(name)[1],saved)
    with pytest.raises(ValueError,match='oracle differs'):store().load(name)


def test_core_plan_requires_complete_rows_without_reusing_k6_pilot(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    shared=pytest.importorskip('core_export')
    from muse_spark.core_k6 import run
    tokens=tmp_path/'tokens';tokens.mkdir()
    (tokens/'manifest.json').write_text(json.dumps(dict(splits=dict(calibration=dict(count=1426,tokens=2852)))))
    source_path=tmp_path/recipe.BASE_REVISION;source_path.mkdir()
    for filename in ('config.json','model.safetensors.index.json'):(source_path/filename).write_text('{}')
    source=SimpleNamespace(snapshot=source_path)
    summary=dict(target_payload_bytes=123,selected_modules=373)
    monkeypatch.setattr(run,'inventory',lambda *a,**kw:summary)
    monkeypatch.setattr(run.numerical,'source_identity',lambda:{'native':'pinned'})
    monkeypatch.setattr(run,'code_identity',lambda:{})
    rows=[dict(input_ids=[1,2]) for _ in range(1426)]
    monkeypatch.setattr(run,'load_token_records',lambda *a:rows)
    plan=run.make_plan(source,tokens,tmp_path/'run',tmp_path/'out',core_bits=8,core_only=True)
    assert plan['inventory']==summary and plan['publication_repo'] is None
    assert plan['identity']['qualification_sha256'] is None
    assert plan['identity']['recipe']['core_bits']==8
    assert plan['identity']['code']['quantization/core_export.py']
    rows.pop()
    with pytest.raises(ValueError,match='all reviewed calibration rows'):
        run.make_plan(source,tokens,tmp_path/'run',tmp_path/'out',core_bits=8,core_only=True)


def test_core_finalizer_requires_oracle_evidence_and_no_other_payload(tmp_path,monkeypatch):
    shared=pytest.importorskip('core_export')
    from test_muse_inventory_export import packed
    from kimi_k3.uniform_k2.storage import atomic_json
    selection={'lm_head':recipe._geometry(128,128,8,'base_core')}
    out=tmp_path/'export';evidence=tmp_path/'evidence';evidence.mkdir()
    candidate=packed(tmp_path/'head.safetensors','lm_head',selection['lm_head'])
    monkeypatch.setattr(export,'SHARD_COUNT',1)
    monkeypatch.setattr(export,'selected_modules',lambda *a,**kw:selection)
    identity={'run':'test'}
    record=dict(module='lm_head',bits=8,logical_shape=[128,128],metrics={'reconstruction':{'mse':.0001}},
        evidence={'packed_byte_oracle':{'passed':True}})
    atomic_json(evidence/'head.json',record)
    summary=export.finalize_core_only(None,selection,{'lm_head':candidate},out,identity,evidence)
    assert summary['quantized_projections']==1 and summary['routed_projections']==0
    assert (out/'quantization-errors/lm_head.json').exists()
    record['evidence']={}
    atomic_json(evidence/'head.json',record)
    with pytest.raises(ValueError,match='reconstruction evidence incomplete'):
        export.finalize_core_only(None,selection,{'lm_head':candidate},out,identity,evidence)
    (out/'foreign').mkdir()
    (out/'foreign'/export._shard_name(1)).write_bytes(b'foreign payload')
    with pytest.raises(ValueError,match='shard inventory differs'):
        export.finalize_core_only(None,selection,{'lm_head':candidate},out,identity,evidence)


def test_k8_codec_forwards_bits_and_requires_reconstruction_diagnostics(monkeypatch):
    torch=pytest.importorskip('torch')
    from muse_spark.core_k6 import codec
    observed=[]
    def quantize(weight,raw,args,**kw):
        observed.append(args['K'])
        args['error_metrics']=dict(quantizer_path='hessian_ldlq',hessian_metric_status='ok',
            hessian_numerical_contract='signed-block-hadamard-congruence-fp64-v1',
            hessian_transform_compute_dtype='torch.float64',hessian_storage_dtype='torch.float32',
            hessian_sample_count=raw['count'],reported_metric_value=.0001,
            hessian_weighted_error_numerator=.01,hessian_weighted_reference_denominator=100,
            hessian_weighted_relative_error=.0001,
            reconstruction=dict(domain='regularized_exl3_search_space',reference_finite=True,
                error_finite=True,mse=.0001,nmse=.0001,error_sum_sq=1.,reference_sum_sq=10000.,max_abs_error=.1))
        packed=dict(trellis=torch.zeros(8,8,16*args['K'],dtype=torch.int16),
            suh=torch.ones(128,dtype=torch.float16),svh=torch.ones(128,dtype=torch.float16),
            mcg=torch.tensor(recipe.MCG_MULTIPLIER-2**32,dtype=torch.int32))
        return None,None,packed
    monkeypatch.setattr(codec,'_quantizer',lambda:SimpleNamespace(quantize_exl3=quantize))
    weight=torch.ones(128,128,dtype=torch.bfloat16)
    raw=dict(H=torch.eye(128),count=1426,finalized=False,identity_rows=0,
        accumulation_contract=codec.ACCUMULATION_CONTRACT)
    for bits in (6,8):
        packed,metrics,evidence=codec.quantize_projection(weight,raw,bits=bits,device='cpu')
        assert evidence['bits']==bits and packed['trellis'].shape[-1]==bits*16
        assert metrics['reconstruction']['domain']=='regularized_exl3_search_space'
    assert observed==[6,8]
    def missing(*args,**kw):
        result=quantize(*args,**kw)
        args[2]['error_metrics']['reconstruction']['mse']=float('nan')
        return result
    monkeypatch.setattr(codec,'_quantizer',lambda:SimpleNamespace(quantize_exl3=missing))
    with pytest.raises(ValueError,match='reconstruction metric'):
        codec.quantize_projection(weight,raw,bits=8,device='cpu')
