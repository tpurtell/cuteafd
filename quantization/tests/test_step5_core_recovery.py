"""Step-5 commit barriers, native-down capture and placement/cleanup admission."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step5.core_k6 import run,storage,handoff,launch

@pytest.mark.parametrize('fails',[False,True])
def test_layer_does_not_advance_frontier_before_durable_assembly(monkeypatch,tmp_path,fails):
    from step5.core_k6 import export
    manifest={'entries':[]}
    phase=SimpleNamespace(load=lambda p,**kw:manifest,clear=Mock(),directory=lambda p:tmp_path/p)
    monkeypatch.setattr(run,'StageStore',lambda *a:phase)
    monkeypatch.setattr(run,'compact',lambda:None)
    monkeypatch.setattr(run,'hardlink_frontier',lambda *a:[])
    error=RuntimeError('assembly audit failed')
    assemble=Mock(side_effect=error if fails else None,return_value={'payload_bytes':1})
    monkeypatch.setattr(export,'assemble_layer',assemble)
    frontier=SimpleNamespace(commit=Mock(return_value={'layer':0}))
    cp=SimpleNamespace(replacements=lambda names:{},retire=Mock())
    engine=object.__new__(run.Engine)
    engine.root=tmp_path;engine.identity={};engine.selection={};engine.source=object();engine.export=tmp_path/'out'
    engine.checkpoints=lambda _:cp;engine.boundary=frontier;engine.progress=SimpleNamespace(emit=Mock())
    if fails:
        with pytest.raises(RuntimeError,match='assembly audit failed'):engine.layer(0,{'layer':-1})
        frontier.commit.assert_not_called();phase.clear.assert_not_called();cp.retire.assert_not_called()
    else:
        assert engine.layer(0,{'layer':-1})=={'layer':0}
        frontier.commit.assert_called_once();phase.clear.assert_called_once();cp.retire.assert_called_once()


def test_route_down_hessian_uses_native_callback_and_only_missing_identity_rows():
    spool=object.__new__(storage.RouteSpool)
    spool.width=2;spool.down_width=1;spool.minimum=4
    spool.route_rows=[np.array([0,1])];spool.recovery_rows=[np.array([2])]
    values=torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.bfloat16)
    spool.gather=lambda rows,device:values[rows]
    callback=Mock(side_effect=lambda x:x[:,:1]*2)
    raw,evidence=spool.hessian(0,down=callback,device='cpu',chunk=2)
    assert callback.call_count==2
    assert raw['count']==4 and raw['identity_rows']==1
    assert torch.equal(raw['H'],torch.tensor([[141.]]))
    assert evidence['natural_rows']==2 and evidence['augmented_rows']==1
    with pytest.raises(ValueError,match='callback'):spool.hessian(0,down=values,device='cpu')


def test_latest_completed_mtp_frontier_does_not_reload_retired_prefix(monkeypatch,tmp_path):
    created=[]
    class Sub:
        def __init__(self,source,records,root,*args,**kwargs):
            created.append((str(root),records))
            self.boundary=SimpleNamespace(current=lambda **kw:{'layer':0})
        def retire_committed(self,layer):assert layer==94
    monkeypatch.setattr(run,'Engine',Sub)
    source=SimpleNamespace(load_block=Mock(side_effect=AssertionError('completed MTP must not reload')))
    parent=SimpleNamespace(root=tmp_path,source=source,records=[{'id':'p','input_ids':[1,2,3,4,5]}],export=tmp_path/'out',identity={},selection={},progress=None,evidence_root=tmp_path/'evidence',device='cpu')
    run.run_mtp(parent,{'layer':91})
    assert len(created)==1 and created[0][1][0]['input_ids']==[4,5]
    source.load_block.assert_not_called()


def test_cleanup_rejects_wrong_parent_and_symlink(tmp_path):
    parent=tmp_path/'owned';parent.mkdir();leaf=parent/'model';leaf.mkdir()
    with pytest.raises(ValueError):handoff.safe_remove(leaf,tmp_path)
    linked=parent/'linked';linked.symlink_to(leaf,target_is_directory=True)
    with pytest.raises(ValueError):handoff.safe_remove(linked,parent)
    assert leaf.is_dir()
    handoff.safe_remove(leaf,parent);assert not leaf.exists()


def test_managed_snapshot_checksum_and_exact_inventory(tmp_path):
    f=tmp_path/'config.json';f.write_bytes(b'{}')
    files={'config.json':{'bytes':2,'sha256':handoff.sha256_file(f)}}
    handoff.verify_files(tmp_path,files)
    f.write_bytes(b'[]')
    with pytest.raises(ValueError,match='checksum'):handoff.verify_files(tmp_path,files)
    f.write_bytes(b'{}');(tmp_path/'unexpected').write_text('x')
    with pytest.raises(ValueError,match='inventory'):handoff.verify_files(tmp_path,files)

@pytest.mark.parametrize('job',[{}, {'finished':False},{'finished':True,'pending':1},{'finished':True,'error':'x'}, {'finished':True,'hosts':{'emu':{'finished':True,'failed':['x']}}}])
def test_handoff_rejects_incomplete_or_failed_sparknest_jobs(job):
    with pytest.raises(ValueError):handoff.successful({'job':job})


def test_completed_handoff_prevents_restarting_published_retired_job(tmp_path):
    (tmp_path/'handoff-complete.json').write_text(json.dumps({'status':'passed'}))
    record=tmp_path/'attempt.json';record.write_text(json.dumps({'root':str(tmp_path)}))
    assert launch.supervise(record)==0


def test_completed_no_copy_import_can_have_unknown_initial_byte_total():
    handoff.successful({'job':{'finished':True,'pending':0,'hosts':{'maxq':{
        'finished':True,'done_files':2,'total_files':2,'done_bytes':100,'total_bytes':0}}}})
    with pytest.raises(ValueError):
        handoff.successful({'job':{'finished':True,'pending':0,'hosts':{'maxq':{
            'finished':True,'done_files':2,'total_files':2,'done_bytes':90,'total_bytes':100}}}})


def test_mtp_recovers_verified_seed_and_latest_stage_after_two_crash_barriers(monkeypatch,tmp_path):
    """Real safetensors receipts/frontiers survive both MTP retirement barriers."""
    class H:
        def __init__(self,width,device):self.H=torch.zeros(width,width);self.count=0
        def add(self,x):x=x.float();self.H.addmm_(x.T,x);self.count+=len(x)
        def raw_dict(self,minimum_rows):
            assert self.count==minimum_rows
            return {'H':self.H,'count':self.count}
    math=SimpleNamespace(DenseRawHessian=H,mtp_envelope=lambda block,e,h:torch.cat((e,h.bfloat16()),-1))
    def block(*args,**kwargs):
        linear=torch.nn.Linear(4,2,bias=False,dtype=torch.bfloat16)
        linear.weight.data.fill_(.25)
        return SimpleNamespace(eh_proj=linear)
    source=SimpleNamespace(config={'hidden_size':2},close=lambda:None,load_block=block,
        tensor=lambda *args:torch.arange(12,dtype=torch.bfloat16).reshape(6,2))
    records=[{'id':'teacher','input_ids':[1,2,3,4,5]}]
    parent=run.Engine(source,records,tmp_path/'run',tmp_path/'export',{'test':'barriers'}, {},
        SimpleNamespace(emit=lambda *args,**kwargs:None),numerical=math,device='cpu')
    target=parent.initial()
    original_engine=run.Engine;original_clear=run.StageStore.clear
    crashes={'seed':True,'stage93':True};ensured=[];decoded=[]
    class Sub(original_engine):
        def __init__(self,*args,**kwargs):kwargs['numerical']=math;super().__init__(*args,**kwargs)
        def checkpoints(self,layer):
            def ensure(name,weight,factory):
                raw=factory();assert raw['count']==self.tokens
                ensured.append((layer,raw['count']))
            return SimpleNamespace(ensure=ensure,install=lambda *a:None)
        def layer(self,layer,current,**kwargs):
            assert current['layer']==-1
            entries=[];directory=self.boundary.directory(0)
            for i,row in enumerate(self.records):
                state=self.boundary_batch(current,i)
                assert state['hidden'].dtype==torch.float32
                entry=run.commit_batch(directory,i,row,{'hidden':state['hidden']+1,'position_ids':state['position_ids']})
                entries.append({k:v for k,v in entry.items() if k!='kind'})
            result=self.boundary.commit(0,entries);decoded.append(layer)
            if layer==93 and crashes['stage93']:
                crashes['stage93']=False;raise RuntimeError('crash after final-stage frontier')
            return result
    def clear(store):
        if store.root.name=='input' and store.root.parent.name=='092' and crashes['seed']:
            seed=run.RollingBoundary(store.root.parent/'frontier',parent.identity).current(checksum=True)
            assert seed['layer']==-1 and store.load('i',checksum=True)
            crashes['seed']=False;raise RuntimeError('crash after EH seed')
        return original_clear(store)
    monkeypatch.setattr(run,'Engine',Sub);monkeypatch.setattr(run.StageStore,'clear',clear)
    monkeypatch.setattr(run,'compact',lambda:None)
    with pytest.raises(RuntimeError,match='after EH seed'):run.run_mtp(parent,target)
    with pytest.raises(RuntimeError,match='final-stage frontier'):run.run_mtp(parent,target)
    # The committed seed consumed its envelope bank even through a restart.
    assert list((parent.root/'mtp/092/input').iterdir())==[]
    run.run_mtp(parent,target)
    assert ensured==[(92,4),(93,3),(94,2)]
    assert decoded==[92,93,94]
    assert not (parent.root/'mtp/092/frontier').exists()
    assert not (parent.root/'mtp/093/frontier').exists()
    latest=run.RollingBoundary(parent.root/'mtp/094/frontier',parent.identity).current(checksum=True)
    assert latest['layer']==0 and latest['manifest']['batches'][0]['tokens']==2
