#!/usr/bin/env python3
"""Causal, layer-resumable Step-5 K4 / Core K6 quantization on MaxQ."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'third_party/gptqmodel'))
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):
    __package__='step5.core_k6'
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
import torch
from torch import nn
from glm5_flash.core_k6.run import Engine as FlashEngine,Progress as FlashProgress,compact,commit_batch
from kimi_k3.uniform_k2.storage import atomic_json,canonical,commit_safetensors,fsync_directory,sha256_file
from kimi_k3.core_k6.run import exclusive_run,hardlink_frontier
from .storage import ProjectionStore,RouteSpool,StageStore,RollingBoundary,load_tensors
from .compat import load_module_origins
from .recipe import BASE_REPO,BASE_REVISION,TARGET_REPO,TARGET_PAYLOAD_BYTES,MODULE_COUNT,reviewed_modules,inventory
GIB=1024**3
RECIPE=dict(hessian_accumulation='step5-native-fp32-XTX-k128-v1',codebook='mcg',multiplier='0xCBAC1FED',core_bits=6,expert_bits=4,late_expert_bits=6,
    late_expert_layers=[88,89,90],mtp_bits=4,seed=787,sigma_reg=.025,out_scales='auto',
    capture='full-native-Step5-calibration-raw-XTX-FP32',core_identity_rows=0,
    hessian='signed-block-hadamard-congruence-fp64-v1',
    cold_experts=dict(minimum=1024,ranks=[9,16],order='rank-major-then-corpus-row',
        floor='raw-XTX-missing-times-I-equivalent-to-normalized-2I'),
    replay='serialized-MCG-FP16-scales-reconstructed-BF16-native-block',
    frontier='native-FP32-residual-and-absolute-positions',
    ordering='interleaved-core-and-experts-before-successor-layer',
    mtp='raw-final-target-residual[t]-and-token[t+1]-position[t+1]-repeat-three-stages')

class Progress(FlashProgress):
    def __init__(self,root):
        super().__init__(root);self.state['total_projections']=MODULE_COUNT

class Engine(FlashEngine):
    def __init__(self,*args,numerical=None,**kwargs):
        if numerical is None:
            from gptqmodel.utils import step5_core as numerical
        super().__init__(*args,numerical=numerical,**kwargs)
        self.boundary=RollingBoundary(self.root/"frontier",self.identity)

    def checkpoints(self,layer):
        return ProjectionStore(self.root/'head/projections' if layer=='head' else self.root/f'layers/{layer:03d}/projections',
            self.evidence_root/('head' if layer=='head' else f'layer-{layer:03d}'),self.identity,
            self.selection,self.progress,self.math,self.device)

    def _context(self,state):
        value=state.get('hidden',state.get('attn_input'))
        return self.math.prepare_context(value.float(),position_ids=state.get('position_ids'))

    @torch.inference_mode()
    def initial(self):
        current=self.boundary.current(checksum=True)
        if current is not None:
            self.boundary.retire_old();self.retire_committed(current['layer']);return current
        directory=self.boundary.directory(-1)
        if directory.is_symlink():raise ValueError('symlink initial frontier')
        if directory.exists():shutil.rmtree(directory)
        embedding=self.source.tensor('model.embed_tokens.weight',self.device)
        entries=[]
        for index,row in enumerate(self.records):
            ids=torch.tensor(row['input_ids'],device=self.device)
            hidden=torch.nn.functional.embedding(ids,embedding).float()
            entry=commit_batch(directory,index,row,dict(hidden=hidden))
            entries.append({k:v for k,v in entry.items() if k!='kind'})
            self._emit('embedding',-1,index)
        del embedding,ids,hidden
        self.source.close();compact();return self.boundary.commit(-1,entries)

    def _group(self,block,layer,checkpoints,names,store,phase,hessian):
        missing=[name for name in names if checkpoints.load(name) is None]
        raw=self._raw(store,phase,hessian) if missing else None
        for name in names:
            linear=self.math.module(block,name.removeprefix(f'model.layers.{layer}.'))
            checkpoints.ensure(name,linear.weight,raw);checkpoints.install(name,linear)
        self.math.invalidate_attention(block)
        del raw;compact()

    def _phase_c(self,block,layer,store):
        self._new_phase(store,'c');h=self._hessian(block.self_attn.o_proj.weight.shape[1]);entries=[]
        for index,row,state in self._read(store,'a'):
            value=self.math.capture_linear_input(block.self_attn.o_proj,self.math.attention_forward,
                block,state['attn_input'],self._context(state))
            h.add(value);out=dict(o_input=value)
            if 'position_ids' in state:out['position_ids']=state['position_ids']
            entries.append(commit_batch(store.directory('c'),index,row,out));self._emit('c',layer,index)
        result=self._commit_phase(store,'c',entries,[('o',h)]);store.retire('a');return result

    @staticmethod
    def _shared(block):
        return block.share_expert if block.moe is not None else block.mlp

    def _phase_e(self,block,layer,store):
        self._new_phase(store,'e');mlp=self._shared(block);h=self._hessian(mlp.down_proj.weight.shape[1])
        for index,_,state in self._read(store,'d'):
            h.add(self.math.shared_down_input(mlp,state['mlp_input']));self._emit('e',layer,index)
        return self._commit_phase(store,'e',[],[('down',h)])

    def _experts(self,block,layer,store,checkpoints):
        root=self.root/f'layers/{layer:03d}'
        names=[n for n in self.selection if n.startswith(f'model.layers.{layer}.moe.experts.')]
        complete=all(checkpoints.load(n) is not None for n in names)
        self.progress.emit('source_experts_load_started',layer=layer,phase='experts')
        self.source.materialize_routed_experts(block,layer=layer,device=self.device)
        spool=None
        try:
            if not complete:
                spool=RouteSpool.create(root/'routes',self.identity,self.tokens,self._read(store,'d'),
                    lambda x:self.math.router_choices(block.moe,x),self.progress,layer,
                    width=self.width,experts=self.source.config['moe_num_experts'])
                spool.down_width=self.source.config['moe_intermediate_size']
            for expert in range(self.source.config['moe_num_experts']):
                cached_raw=cached_evidence=None
                for part in ('gate_proj','up_proj','down_proj'):
                    name=f'model.layers.{layer}.moe.experts.{expert}.{part}'
                    weight=self.math.expert_weight(block,expert,part);raw=evidence=None
                    if checkpoints.load(name) is None:
                        if part=='down_proj':
                            raw,evidence=spool.hessian(expert,down=lambda x:self.math.expert_down_input(block,expert,x),device=self.device)
                        else:
                            if cached_raw is None:cached_raw,cached_evidence=spool.hessian(expert,device=self.device)
                            raw,evidence=cached_raw,cached_evidence
                    checkpoints.ensure(name,weight,raw,evidence)
                    weight.copy_(checkpoints.dense(name));del raw,evidence,weight
                del cached_raw,cached_evidence
                if expert%16==0:
                    compact();self.progress.emit('expert_progress',layer=layer,phase='experts',expert=expert+1,experts=self.source.config['moe_num_experts'])
            atomic_json(root/'experts-complete.json',dict(identity=self.identity,layer=layer,modules=len(names)))
        finally:
            if spool is not None:spool.close()
        compact()

    def _stage_head(self,block,layer,store,checkpoints):
        name=f'model.layers.{layer}.transformer.shared_head.output'
        if checkpoints.load(name) is not None:return
        if store.load('mtp-head',checksum=True) is None:
            self._new_phase(store,'mtp-head');h=self._hessian(self.width)
            for _,_,state in self._read(store,'f'):h.add(self.math.mtp_head_input(block,state['hidden']))
            self._commit_phase(store,'mtp-head',[],[('head',h)])
        checkpoints.ensure(name,block.transformer.shared_head.output.weight,self._raw(store,'mtp-head','head'))

    @torch.inference_mode()
    def layer(self,layer,current,*,block=None,boundary_layer=None):
        from .export import assemble_layer
        start=time.monotonic();boundary_layer=layer if boundary_layer is None else boundary_layer
        store=StageStore(self.root/f'layers/{layer:03d}/stages',self.identity)
        phases={p:store.load(p,checksum=True) for p in ('a','c','d','e','f')}
        checkpoints=self.checkpoints(layer);prefix=f'model.layers.{layer}.'
        names=[n for n in self.selection if n.startswith(prefix)]
        if phases['f'] is None:
            if block is None:block=self.source.load_block(layer,self.device,include_experts=False)
            for name in names:
                if '.experts.' not in name and name!=prefix+'eh_proj':
                    checkpoints.install(name,self.math.module(block,name.removeprefix(prefix)))
            self.math.invalidate_attention(block)
            attn=lambda part:prefix+'self_attn.'+part
            if phases['d'] is None:
                if phases['c'] is None:
                    if phases['a'] is None:self._phase_a(block,layer,current,store)
                    self._group(block,layer,checkpoints,[attn(p) for p in ('q_proj','k_proj','v_proj')],store,'a','input')
                    self._phase_c(block,layer,store)
                self._group(block,layer,checkpoints,[attn('o_proj')],store,'c','o')
                self._phase_d(block,layer,current,store)
            mlp=prefix+('share_expert.' if block.moe is not None else 'mlp.')
            self._group(block,layer,checkpoints,[mlp+'gate_proj',mlp+'up_proj'],store,'d','mlp')
            if phases['e'] is None:self._phase_e(block,layer,store)
            self._group(block,layer,checkpoints,[mlp+'down_proj'],store,'e','down')
            if block.moe is not None:self._experts(block,layer,store,checkpoints)
            self._phase_f(block,layer,store)
        if layer>=92:
            if block is None:block=self.source.load_block(layer,self.device,include_experts=False)
            self._stage_head(block,layer,store,checkpoints)
        receipt=assemble_layer(self.source,layer,checkpoints.replacements(names),self.export,self.identity)
        manifest=store.load('f');links=hardlink_frontier(self.boundary,boundary_layer,store.directory('f'),manifest)
        current=self.boundary.commit(boundary_layer,links)
        store.clear();checkpoints.retire()
        routes=self.root/f'layers/{layer:03d}/routes'
        if routes.exists():shutil.rmtree(routes)
        del block;compact()
        self.progress.emit('layer_committed',layer=layer,seconds=time.monotonic()-start,payload_bytes=receipt['payload_bytes'])
        return current

    @torch.inference_mode()
    def head(self,current):
        store=StageStore(self.root/'head/stages',self.identity);checkpoints=self.checkpoints('head')
        if checkpoints.load('lm_head') is None:
            if store.load('head',checksum=True) is None:
                self._new_phase(store,'head');h=self._hessian(self.width)
                for index,_ in enumerate(self.records):
                    state=self.boundary_batch(current,index);h.add(self.math.head_input(self.source,state['hidden']))
                    self._emit('head',92,index)
                self._commit_phase(store,'head',[],[('head',h)]);del h;compact()
            weight=self.source.tensor('lm_head.weight',self.device)
            checkpoints.ensure('lm_head',weight,self._raw(store,'head','head'));del weight;compact()
        self.source.close();return checkpoints


def run_mtp(engine,target):
    previous_engine,previous=engine,target
    start=92
    # An older MTP boundary is deliberately retired after its successor commits.
    # Recover from the newest surviving stage instead of requantizing its prefix.
    for completed in range(94,91,-1):
        root=engine.root/f'mtp/{completed:03d}'
        records=[dict(id=r['id'],input_ids=r['input_ids'][completed-91:]) for r in engine.records]
        candidate=Engine(engine.source,records,root,engine.export,engine.identity,engine.selection,engine.progress,evidence_root=engine.evidence_root,device=engine.device)
        saved=candidate.boundary.current(checksum=True)
        if saved is not None and saved['layer']==0:
            candidate.retire_committed(completed)
            StageStore(root/'input',engine.identity).clear()
            for older in range(92,completed):
                old_root=engine.root/f'mtp/{older:03d}'
                StageStore(old_root/'input',engine.identity).clear()
                old=old_root/'frontier'
                if old.is_symlink():raise ValueError('symlink retired MTP frontier')
                if old.exists():shutil.rmtree(old)
            previous_engine,previous,start=candidate,saved,completed+1
            break
    for layer in range(start,95):
        root=engine.root/f'mtp/{layer:03d}'
        records=[dict(id=r['id'],input_ids=r['input_ids'][1:]) for r in previous_engine.records]
        sub=Engine(engine.source,records,root,engine.export,engine.identity,engine.selection,engine.progress,evidence_root=engine.evidence_root,device=engine.device)
        current=sub.boundary.current(checksum=True)
        if current is not None and current['layer']==0:
            previous_engine,previous=sub,current;continue
        block=engine.source.load_block(layer,engine.device,include_experts=False)
        checkpoints=sub.checkpoints(layer);store=StageStore(root/'input',engine.identity)
        eh=f'model.layers.{layer}.eh_proj'
        if current is None:
            if store.load('i',checksum=True) is None:
                sub._new_phase(store,'i');h=sub._hessian(engine.width*2);entries=[]
                embedding=engine.source.tensor('model.embed_tokens.weight',engine.device)
                for index,row in enumerate(previous_engine.records):
                    state=previous_engine.boundary_batch(previous,index)
                    ids=torch.tensor(row['input_ids'][1:],device=engine.device)
                    hidden=state['hidden'][:-1]
                    positions=state.get('position_ids',torch.arange(len(state['hidden']),device=engine.device))[1:]
                    envelope=engine.math.mtp_envelope(block,torch.nn.functional.embedding(ids,embedding),hidden)
                    h.add(envelope);entries.append(commit_batch(store.directory('i'),index,records[index],dict(envelope=envelope,position_ids=positions)))
                sub._commit_phase(store,'i',entries,[('eh',h)]);del h,embedding;compact()
            checkpoints.ensure(eh,block.eh_proj.weight,lambda:sub._raw(store,'i','eh'));checkpoints.install(eh,block.eh_proj)
            directory=sub.boundary.directory(-1)
            if directory.exists():shutil.rmtree(directory)
            entries=[]
            for index,row,state in sub._read(store,'i'):
                entry=commit_batch(directory,index,row,dict(hidden=block.eh_proj(state['envelope']).float(),position_ids=state['position_ids']))
                entries.append({k:v for k,v in entry.items() if k!='kind'})
            current=sub.boundary.commit(-1,entries);store.clear()
        else:
            store.clear()  # The verified seed owns its hidden rows; EH inputs are obsolete.
        current=sub.layer(layer,current,block=block,boundary_layer=0)
        previous_engine,previous=sub,current
        if layer>92:
            old=engine.root/f'mtp/{layer-1:03d}/frontier'
            if old.exists():shutil.rmtree(old)


def make_plan(source,tokens,root,export):
    summary=inventory(source);manifest=json.loads((tokens/'manifest.json').read_text())
    if source.snapshot.name!=BASE_REVISION or manifest['splits']['calibration']['count']!=1426:
        raise ValueError('pinned full BF16 source/full calibration panel required')
    code=[*(Path(__file__).with_name(n) for n in ('run.py','storage.py','recipe.py','export.py','compat.py','migrate.py')),
        REPO/'quantization/step5/prepare_corpus.py',REPO/'quantization/glm5_flash/core_k6/run.py',
        REPO/'quantization/glm5/core_k6/storage.py',REPO/'quantization/kimi_k3/core_k6/run.py',
        REPO/'quantization/kimi_k3/core_k6/storage.py',REPO/'quantization/kimi_k3/uniform_k2/storage.py',
        REPO/'python/reference/families/step5/golden.py',REPO/'python/reference/families/step5/mtp.py',
        REPO/'python/reference/shape_invariant.py',REPO/'third_party/transformers.lock.json']
    fork=REPO/'third_party/gptqmodel';code += [fork/'gptqmodel/utils/step5_core.py',fork/'gptqmodel/utils/glm5_core.py']
    identity=dict(runtime=source.reference.verify_runtime_source(),schema='cuteafd-step5-k4-core-k6-production-v2',source_repo=BASE_REPO,source_revision=BASE_REVISION,
        source_config_sha256=sha256_file(source.snapshot/'config.json'),source_index_sha256=sha256_file(source.snapshot/'model.safetensors.index.json'),
        token_manifest_sha256=sha256_file(tokens/'manifest.json'),recipe=RECIPE,
        calibration=dict(samples=manifest['splits']['calibration']['count'],tokens=manifest['splits']['calibration']['tokens']),
        code={str(p.relative_to(REPO)):sha256_file(p) for p in code},
        gptqmodel_commit=subprocess.check_output(['git','-C',str(fork),'rev-parse','HEAD'],text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','safetensors','huggingface-hub','tokenizers')})
    lineage_path=root/'producer-lineage.json'
    if lineage_path.exists():
        if lineage_path.is_symlink():raise ValueError('symlink producer lineage')
        lineage=json.loads(lineage_path.read_text())
        identity.update(predecessor_identities=lineage['predecessor_identities'],
            resume_lineage_audit_sha256=lineage['resume_lineage_audit_sha256'],
            resume_lineage_manifest_sha256=sha256_file(lineage_path))
        load_module_origins(root,identity)
    temporary=192*GIB
    return dict(identity=identity,source=str(source.snapshot),tokens=str(tokens),root=str(root),export=str(export),inventory=summary,
        temporary_reserve_bytes=temporary,required_free_bytes=TARGET_PAYLOAD_BYTES+temporary+64*GIB,publication_repo=TARGET_REPO)


def cleanup_complete(root):
    root=Path(root)
    for path in [root/'frontier',root/'head/stages',*(root/'mtp').glob('*/frontier'),*(root/'mtp').glob('*/input')]:
        if path.is_symlink():raise ValueError('symlink recovery cleanup')
        if path.exists():shutil.rmtree(path)
    for base in [root/'head/projections',*(root/'mtp').glob('*/layers/*/projections')]:
        if base.exists():
            for path in base.glob('*.safetensors'):path.unlink()
    fsync_directory(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot','tokens','root','export'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--publish-only',action='store_true')
    args=p.parse_args()
    from gptqmodel.utils.step5_core import Step5BF16Source
    source=Step5BF16Source(args.source_snapshot,reference_root=REPO/'python/reference')
    args.tokens,args.root,args.export=(v.expanduser().resolve() for v in (args.tokens,args.root,args.export))
    args.root.mkdir(parents=True,exist_ok=True);plan=make_plan(source,args.tokens,args.root,args.export);prior=args.root/'plan.json'
    if prior.exists() and json.loads(prior.read_text())!=plan:raise ValueError('immutable production plan differs')
    if args.preflight:
        if not prior.exists() and shutil.disk_usage(args.root).free<plan['required_free_bytes']:raise OSError('insufficient bounded NVMe reserve')
        print(canonical(plan));return
    if prior.exists() and not args.resume and not args.publish_only:raise ValueError('existing run requires resume')
    if not prior.exists() and (args.resume or args.publish_only):raise ValueError('resume requires committed plan')
    if args.publish_only:
        from .publish import publish
        with exclusive_run(args.root):publish(args.export,args.root,TARGET_REPO)
        return
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(86*GIB/torch.cuda.get_device_properties(0).total_memory)
    progress=Progress(args.root)
    with torch.inference_mode(),exclusive_run(args.root):
        if not prior.exists():atomic_json(prior,plan)
        if (args.root/'quantization-complete.json').exists():cleanup_complete(args.root);return
        from step5.prepare_corpus import load_token_records
        records=load_token_records(args.tokens,'calibration');engine=Engine(source,records,args.root,args.export,plan['identity'],reviewed_modules(),progress)
        current=engine.initial()
        for layer in range(current['layer']+1,92):current=engine.layer(layer,current)
        head=engine.head(current);run_mtp(engine,current)
        from .export import finalize_export
        summary=finalize_export(source,engine.selection,head.replacements(['lm_head']),args.export,engine.identity,args.root/'evidence')
        atomic_json(args.root/'quantization-complete.json',dict(identity=engine.identity,summary=summary))
        cleanup_complete(args.root);progress.state['status']='quantized';progress.emit('quantization_complete',**summary)
    source.close()

if __name__=='__main__':main()
