#!/usr/bin/env python3
"""One causal, layer-resumable Muse Spark K4 experts / Core K6 production pass."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):__package__='muse_spark.core_k6'
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
import psutil
import torch
from torch import nn
from muse_spark.prepare_corpus import load_token_records
from kimi_k3.uniform_k2.storage import (RollingBoundary,atomic_json,canonical,commit_safetensors,
    fsync_directory,sha256_file)
from kimi_k3.core_k6.run import exclusive_run,hardlink_frontier
from . import numerical,codec
from .source import MuseSource
from .recipe import BASE_REPO,BASE_REVISION,TARGET_REPO,MODULE_COUNT,reviewed_modules,inventory
from .storage import ProjectionStore,StageStore,RouteSpool,load_tensors

GIB=1024**3
PREFIX='model.language_model.'
RECIPE=dict(codebook='mcg',multiplier='0xCBAC1FED',core_bits=6,head_bits=6,expert_bits=4,
    seed=787,sigma_reg=.025,out_scales='auto',capture='causal-all-native-prompt-rows-raw-FP32-XTX',
    hessian='signed-block-hadamard-congruence-fp64-v1',hessian_accumulation=codec.ACCUMULATION_CONTRACT,core_identity_rows=0,
    cold_experts=dict(minimum=1024,ranks=[9,16],order='rank-major-then-corpus-row',
        floor='raw-XTX-missing-times-I-equivalent-to-normalized-2I'),
    replay='serialized-MCG-FP16-scales-reconstructed-BF16-official-Muse-block',
    ordering='qkv-then-o-then-pre-expert-then-experts-then-post-expert-before-successor-layer',
    expert_split='gate_up_proj[E,0:4096,:]=gate;[E,4096:8192,:]=up;down_proj[E,:,:]=down',
    expert_post_norm='official-input-scaled-RMSNorm-before-routing-weight',head_logical_rows=202048,
    head_stored_rows=202112,whole_sharded_side=1)


def compact():
    gc.collect()
    if torch.cuda.is_initialized():torch.cuda.empty_cache()


class Progress:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic()
        self.state=dict(status='running',started=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(),projections=sum(1 for _ in (self.root/'evidence').rglob('*.json')),
            total_projections=MODULE_COUNT)
    def emit(self,event,**fields):
        if event=='projection_committed':self.state['projections']+=1
        row=dict(event=event,time=datetime.now(timezone.utc).isoformat(),
            rss_gib=psutil.Process().memory_info().rss/GIB,
            cuda_gib=torch.cuda.memory_allocated()/GIB if torch.cuda.is_initialized() else 0,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/GIB if torch.cuda.is_initialized() else 0,
            free_gib=shutil.disk_usage(self.root).free/GIB,**fields)
        self.state.update(row)
        with (self.root/'events.jsonl').open('a') as f:f.write(canonical(row)+'\n')
        atomic_json(self.root/'status.json',self.state)
        print(canonical(row),flush=True)
        if row['rss_gib']>76 or row['cuda_gib']>86 or row['free_gib']<64:
            raise MemoryError('Muse producer exceeded bounded host/GPU/disk admission')


def commit_batch(directory,index,row,state):
    name=f'batch-{index:05d}.safetensors'
    return dict(filename=name,kind='batch',id=row['id'],tokens=len(row['input_ids']),
        **commit_safetensors(Path(directory)/name,state))


class Engine:
    def __init__(self,source,records,root,export,identity,progress):
        self.source,self.records,self.root,self.export=source,records,Path(root),Path(export)
        self.identity,self.progress=identity,progress
        self.selection=reviewed_modules()
        self.tokens=sum(len(r['input_ids']) for r in records)
        self.boundary=RollingBoundary(self.root/'frontier',identity)
        self.rotary=None
    def store(self,layer):
        location='head' if layer=='head' else f'layers/{layer:03d}'
        evidence='head' if layer=='head' else f'layer-{layer:03d}'
        return ProjectionStore(self.root/location/'projections',self.root/'evidence'/evidence,
            self.identity,self.selection,self.progress,codec)
    def emit_batch(self,phase,layer,index):
        if index%64==0 or index+1==len(self.records):
            self.progress.emit('phase_progress',phase=phase,layer=layer,batches=index+1,total_batches=len(self.records))
    def boundary_batch(self,current,index):
        entry=current['manifest']['batches'][index];row=self.records[index]
        if entry['id']!=row['id'] or entry['tokens']!=len(row['input_ids']):
            raise ValueError('frontier batch differs from immutable corpus order')
        return load_tensors(self.boundary.root/current['directory']/entry['filename'])
    def phase_batches(self,stages,phase):
        manifest=stages.load(phase)
        entries=[e for e in manifest['entries'] if e.get('kind')=='batch']
        if len(entries)!=len(self.records):raise ValueError('phase batch count differs')
        for index,(entry,row) in enumerate(zip(entries,self.records,strict=True)):
            if entry['id']!=row['id'] or entry['tokens']!=len(row['input_ids']):
                raise ValueError('phase batch identity differs')
            yield index,load_tensors(stages.directory(phase)/entry['filename'])
    def new_phase(self,stages,phase):
        stages.retire(phase);stages.directory(phase).mkdir(parents=True)
    def commit_phase(self,stages,phase,entries,hessian=None):
        if hessian is not None:
            raw=hessian.raw_dict(minimum_rows=self.tokens)
            if raw['count']!=self.tokens:raise ValueError('dense Hessian loses/duplicates native rows')
            entries.append(dict(filename='hessian.safetensors',kind='hessian',count=self.tokens,
                **commit_safetensors(stages.directory(phase)/'hessian.safetensors',dict(H=raw['H']))))
        result=stages.commit(phase,entries,dict(tokens=self.tokens,batches=len(self.records),
            hessian_accumulation=codec.ACCUMULATION_CONTRACT))
        self.progress.emit('phase_committed',phase=phase)
        return result
    def raw(self,stages,phase):
        manifest=stages.load(phase)
        entries=[] if manifest is None else [e for e in manifest['entries'] if e.get('kind')=='hessian']
        if (len(entries)!=1 or entries[0]['count']!=self.tokens
                or manifest['metadata'].get('hessian_accumulation')!=codec.ACCUMULATION_CONTRACT):raise ValueError('raw Hessian unavailable or stale')
        return dict(H=load_tensors(stages.directory(phase)/entries[0]['filename'])['H'],
            count=self.tokens,finalized=False,identity_rows=0,accumulation_contract=codec.ACCUMULATION_CONTRACT)
    def group(self,block,layer,checkpoints,names,stages,phase):
        prefix=f'{PREFIX}layers.{layer}.'
        for relative in names:
            name=prefix+relative;linear=block.get_submodule(relative)
            checkpoints.ensure(name,linear.weight,lambda:self.raw(stages,phase))
            checkpoints.install(name,linear)
            compact()
    def retire_committed(self,layer):
        for directory in (self.root/'layers').glob('[0-9][0-9][0-9]'):
            if int(directory.name)>layer:continue
            for name in ('stages','routes'):
                path=directory/name
                if path.is_symlink():raise ValueError('refuse symlink recovery cleanup')
                if path.exists():shutil.rmtree(path)
            for path in (directory/'projections').glob('*.safetensors'):path.unlink()
    @torch.inference_mode()
    def initial(self):
        current=self.boundary.current(checksum=True)
        if current is not None:
            self.boundary.retire_old();self.retire_committed(current['layer']);return current
        directory=self.boundary.directory(-1)
        if directory.is_symlink():raise ValueError('refuse symlink embedding frontier')
        if directory.exists():shutil.rmtree(directory)
        embed=numerical.embedding(self.source);entries=[]
        for index,row in enumerate(self.records):
            ids=torch.tensor([row['input_ids']],device='cuda:0')
            hidden=embed(ids)
            if not torch.isfinite(hidden).all():raise ValueError('nonfinite normalized embedding')
            entry=commit_batch(directory,index,row,dict(hidden=hidden));entry.pop('kind');entries.append(entry)
            self.emit_batch('embedding',-1,index)
        del embed,hidden,ids;compact()
        return self.boundary.commit(-1,entries)
    @torch.inference_mode()
    def layer(self,layer,current):
        start=time.monotonic();prefix=f'{PREFIX}layers.{layer}.'
        checkpoints=self.store(layer);stages=StageStore(self.root/f'layers/{layer:03d}/stages',self.identity)
        names=[n for n in self.selection if n.startswith(prefix)]
        for phase in ('a','b','c','e','f'):stages.load(phase,checksum=True)
        if stages.load('f',checksum=True) is None:
            block=numerical.load_block(self.source,layer)
            if self.rotary is None:self.rotary=numerical.rotary(self.source)
            if stages.load('b') is None and stages.load('c') is None and stages.load('e') is None:
                if stages.load('a',checksum=True) is None:
                    self.new_phase(stages,'a');h=codec.DenseRawHessian(8192)
                    for index,row in enumerate(self.records):
                        state=self.boundary_batch(current,index)
                        h.add(numerical.attention_inputs(block,state['hidden']))
                        self.emit_batch('a',layer,index)
                    self.commit_phase(stages,'a',[],h);del h;compact()
                self.group(block,layer,checkpoints,['self_attn.'+p for p in ('q_proj','k_proj','v_proj')],stages,'a')
                self.new_phase(stages,'b');h=codec.DenseRawHessian(8192);entries=[]
                for index,row in enumerate(self.records):
                    state=self.boundary_batch(current,index)
                    normalized=numerical.attention_inputs(block,state['hidden'])
                    context=numerical.prepare_context(block,state['hidden'],self.rotary)
                    operand=numerical.attention_output_input(block,normalized,context)
                    h.add(operand)
                    entries.append(commit_batch(stages.directory('b'),index,row,dict(o_input=operand)))
                    self.emit_batch('b',layer,index)
                self.commit_phase(stages,'b',entries,h);del h,operand,normalized,context,state;stages.retire('a');compact()
            if stages.load('c') is None and stages.load('e') is None:
                self.group(block,layer,checkpoints,['self_attn.o_proj'],stages,'b')
                self.new_phase(stages,'c');h=codec.DenseRawHessian(8192);entries=[]
                for index,state in self.phase_batches(stages,'b'):
                    original=self.boundary_batch(current,index)['hidden']
                    output=block.self_attn.o_proj(state['o_input'])
                    residual,normalized=numerical.attention_finish(block,original,output)
                    h.add(normalized)
                    entries.append(commit_batch(stages.directory('c'),index,self.records[index],
                        dict(residual=residual,normalized=normalized)))
                    self.emit_batch('c',layer,index)
                self.commit_phase(stages,'c',entries,h);del h,state,original,output,residual,normalized
                stages.retire('b');compact()
            if stages.load('e') is None:
                self.group(block,layer,checkpoints,['mlp.pre_expert_proj'],stages,'c')
                def observations():
                    for index,state in self.phase_batches(stages,'c'):
                        selected,weights,near=numerical.router_choices(block.mlp,state['normalized'])
                        yield index,dict(input=numerical.expert_input(block,state['normalized']),selected=selected,near=near)
                spool=RouteSpool.create(self.root/f'layers/{layer:03d}/routes',self.identity,self.tokens,
                    observations(),self.progress,layer)
                # Allocate once, read one expert at a time. Source values are
                # replaced by committed serialized reconstructions before any
                # downstream capture; resume uses exactly the same receipts.
                numerical.materialize_routed_experts(self.source,block,layer)
                try:
                    for expert in range(256):
                        ep=prefix+f'mlp.experts.{expert}.'
                        input_covariance,input_evidence=spool.hessian(expert)
                        for part in ('gate_proj','up_proj'):
                            target=numerical.expert_weight(block,expert,part)
                            checkpoints.ensure(ep+part,target,input_covariance,extra_evidence=input_evidence)
                            target.copy_(checkpoints.dense(ep+part))
                        del input_covariance
                        down_covariance,down_evidence=spool.hessian(expert,down=block.mlp.experts.gate_up_proj[expert])
                        # Keep actual route counts and the exact recovery rule
                        # beside every candidate, including resumed records.
                        extra=dict(natural_rows=len(spool.route_rows[expert]),
                            augmented_rows=len(spool.recovery_rows[expert]),ranks=[9,16],
                            post_expert_norm='official-before-routing-weight')
                        down=numerical.expert_weight(block,expert,'down_proj')
                        checkpoints.ensure(ep+'down_proj',down,down_covariance,extra_evidence=dict(down_evidence,**extra))
                        down.copy_(checkpoints.dense(ep+'down_proj'))
                        del down_covariance
                        if expert%16==0:self.progress.emit('expert_progress',layer=layer,expert=expert)
                        compact()
                finally:spool.close()
                self.new_phase(stages,'e');h=codec.DenseRawHessian(4096);entries=[]
                for index,state in self.phase_batches(stages,'c'):
                    selected,weights,near=numerical.router_choices(block.mlp,state['normalized'])
                    projected=numerical.expert_input(block,state['normalized'])
                    expert_output=block.mlp.experts(projected,selected,weights)
                    h.add(expert_output)
                    entries.append(commit_batch(stages.directory('e'),index,self.records[index],
                        dict(residual=state['residual'],expert_output=expert_output)))
                    self.emit_batch('e',layer,index)
                self.commit_phase(stages,'e',entries,h)
                del h,state,projected,expert_output,selected,weights,near;stages.retire('c');compact()
            self.group(block,layer,checkpoints,['mlp.post_expert_proj'],stages,'e')
            self.new_phase(stages,'f');entries=[]
            for index,state in self.phase_batches(stages,'e'):
                output=block.mlp.post_expert_proj(state['expert_output']).view_as(state['residual'])
                hidden=numerical.mlp_finish(block,state['residual'],output)
                if not torch.isfinite(hidden).all():raise ValueError('nonfinite successor layer frontier')
                entries.append(commit_batch(stages.directory('f'),index,self.records[index],dict(hidden=hidden)))
                self.emit_batch('f',layer,index)
            self.commit_phase(stages,'f',entries)
            del block,state,hidden,output;compact()
        from .export import assemble_layer
        receipt=assemble_layer(self.source,layer,checkpoints.replacements(names),self.export,self.identity)
        manifest=stages.load('f',checksum=True)
        links=hardlink_frontier(self.boundary,layer,stages.directory('f'),manifest)
        result=self.boundary.commit(layer,links)
        stages.clear();checkpoints.retire()
        routes=self.root/f'layers/{layer:03d}/routes'
        if routes.exists():shutil.rmtree(routes)
        elapsed=time.monotonic()-start
        self.progress.emit('layer_committed',layer=layer,seconds=elapsed,
            remaining_layers=61-layer,eta_hours=(61-layer)*elapsed/3600,payload_bytes=receipt['payload_bytes'])
        return result
    @torch.inference_mode()
    def head(self,current):
        checkpoints=self.store('head');stages=StageStore(self.root/'head/stages',self.identity)
        if checkpoints.load('lm_head') is None:
            if stages.load('head',checksum=True) is None:
                self.new_phase(stages,'head');h=codec.DenseRawHessian(8192)
                for index,row in enumerate(self.records):
                    h.add(numerical.final_norm(self.source,self.boundary_batch(current,index)['hidden']))
                    self.emit_batch('head',62,index)
                self.commit_phase(stages,'head',[],h);del h;compact()
            weight=self.source.tensor('lm_head.weight','cpu')
            checkpoints.ensure('lm_head',weight,lambda:self.raw(stages,'head'))
            del weight;compact()
        return checkpoints


def code_identity():
    paths=[Path(__file__).with_name(name) for name in ('run.py','codec.py','numerical.py','storage.py','recipe.py','source.py','export.py')]+[REPO/'quantization/muse_spark/prepare_corpus.py',
        REPO/'quantization/glm5/core_k6/storage.py',REPO/'quantization/kimi_k3/core_k6/storage.py',
        REPO/'quantization/kimi_k3/uniform_k2/storage.py',REPO/'quantization/kimi_k3/core_k6/run.py',
        REPO/'quantization/kimi_k3/prepare_corpus.py']
    paths+=list((REPO/'third_party/gptqmodel/gptqmodel/exllamav3').rglob('*.py'))
    paths+=list((REPO/'third_party/gptqmodel/gptqmodel_ext/exllamav3').rglob('*'))
    paths+=[REPO/'third_party/gptqmodel/gptqmodel/utils'/n for n in ('cpp.py','env.py','logger.py','jit_compile_baselines.py')]
    paths+=[REPO/'third_party/gptqmodel/gptqmodel/extension.py']
    paths+=[REPO/'third_party/gptqmodel/gptqmodel/utils'/n for n in
        ('exl3_hessian.py',)]
    paths=[p for p in paths if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(REPO)):sha256_file(p) for p in sorted(paths)}


def make_plan(source,tokens,root,export):
    summary=inventory(source)
    manifest=json.loads((tokens/'manifest.json').read_text())
    if source.snapshot.name!=BASE_REVISION or manifest['splits']['calibration']['count']!=1426:
        raise ValueError('production requires pinned source and full1426-prompt calibration stream')
    from .pilot import validate_report
    qualification_path=root.parent/'pilot/qualification.json'
    qualification=validate_report(qualification_path)
    if qualification['identity']['native']!=numerical.source_identity():
        raise ValueError('pilot runtime/source identity differs from production')
    if qualification['identity']['token_manifest_sha256']!=sha256_file(tokens/'manifest.json'):
        raise ValueError('pilot token stream differs from production')
    for relative,digest in qualification['identity']['code'].items():
        if sha256_file(REPO/relative)!=digest:
            raise ValueError('pilot numerical source changed: '+relative)
    identity=dict(schema='cuteafd-muse-spark-k4-core-k6-production-k128-v2',source_repo=BASE_REPO,
        source_revision=BASE_REVISION,qualification_sha256=sha256_file(qualification_path),source_config_sha256=sha256_file(source.snapshot/'config.json'),
        source_index_sha256=sha256_file(source.snapshot/'model.safetensors.index.json'),
        token_manifest_sha256=sha256_file(tokens/'manifest.json'),recipe=RECIPE,
        runtime=dict(numerical=numerical.source_identity(),cuda=torch.version.cuda,
            image=os.environ.get('MUSE_QUANT_IMAGE_ID'),gpu=os.environ.get('MUSE_AUTHORIZED_GPU'),
            tf32=False,matmul_precision='highest',attention_backend='official-sdpa',
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
        calibration=dict(samples=1426,tokens=manifest['splits']['calibration']['tokens']),code=code_identity(),
        gptqmodel_commit=subprocess.check_output(['git','-C',str(REPO/'third_party/gptqmodel'),'rev-parse','HEAD'],text=True).strip())
    return dict(identity=identity,source=str(source.snapshot),tokens=str(tokens),root=str(root),export=str(export),
        inventory=summary,publication_repo=TARGET_REPO,temporary_reserve_bytes=256*GIB,
        required_free_bytes=summary['target_payload_bytes']+320*GIB)


def cleanup_complete(root):
    for path in (root/'frontier',root/'head/stages'):
        if path.is_symlink():raise ValueError('refuse symlink completed state cleanup')
        if path.exists():shutil.rmtree(path)
    for path in (root/'head/projections').glob('*.safetensors'):path.unlink()
    fsync_directory(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot','tokens','root','export'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('resume','preflight','publish-only'):parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    args.tokens,args.root,args.export=(p.expanduser().resolve() for p in (args.tokens,args.root,args.export))
    source=MuseSource(args.source_snapshot);args.root.mkdir(parents=True,exist_ok=True)
    plan=make_plan(source,args.tokens,args.root,args.export)
    prior=args.root/'plan.json'
    if prior.exists() and json.loads(prior.read_text())!=plan:raise ValueError('immutable production plan differs')
    if args.preflight:
        if not prior.exists() and shutil.disk_usage(args.root).free<plan['required_free_bytes']:
            raise OSError('insufficient space for final export and bounded rolling work')
        print(canonical(plan));return
    if prior.exists() and not(args.resume or args.publish_only):raise ValueError('existing production requires --resume')
    if not prior.exists() and(args.resume or args.publish_only):raise ValueError('resume requires committed production plan')
    if args.publish_only:
        from .publish import publish
        with exclusive_run(args.root):
            done=json.loads((args.root/'quantization-complete.json').read_text())
            if done['identity']!=plan['identity']:raise ValueError('publication producer differs')
            publish(args.export,args.root,TARGET_REPO)
        return
    torch.set_num_threads(8);numerical.configure_math()
    torch.cuda.set_per_process_memory_fraction(86*GIB/torch.cuda.get_device_properties(0).total_memory)
    progress=Progress(args.root)
    with torch.inference_mode(),exclusive_run(args.root):
        if not prior.exists():atomic_json(prior,plan)
        if (args.root/'quantization-complete.json').exists():
            cleanup_complete(args.root);progress.emit('quantization_complete');return
        records=load_token_records(args.tokens,'calibration')
        engine=Engine(source,records,args.root,args.export,plan['identity'],progress)
        current=engine.initial()
        for layer in range(current['layer']+1,62):current=engine.layer(layer,current)
        head=engine.head(current)
        from .export import finalize_export
        summary=finalize_export(source,engine.selection,head.replacements(['lm_head']),args.export,
            engine.identity,args.root/'evidence')
        atomic_json(args.root/'quantization-complete.json',dict(identity=engine.identity,summary=summary))
        from .validate import audit_completed
        atomic_json(args.root/'completed-export-audit.json',audit_completed(source,args.root,args.export))
        cleanup_complete(args.root);progress.state['status']='quantized';progress.emit('quantization_complete',**summary)
    source.close()

if __name__=='__main__':main()
