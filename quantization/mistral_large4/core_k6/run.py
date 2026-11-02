#!/usr/bin/env python3
"""One interleaved, crash-resumable native Mistral FP8 -> K4/K6 causal pass."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'quantization'))
if __package__ in (None, ''):
    __package__ = 'mistral_large4.core_k6'
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import psutil
import torch
from torch import nn

from kimi_k3.uniform_k2.storage import (RollingBoundary, atomic_json, canonical,
    commit_safetensors, fsync_directory, sha256_file)
from glm5.core_k6.run import Engine as StageEngine, commit_batch, batches
from kimi_k3.core_k6.run import exclusive_run, hardlink_frontier
from mistral_large4.prepare_corpus import load_token_records
from . import native
from .recipe import (BASE_REPO, BASE_REVISION, TARGET_REPO, TARGET_PAYLOAD_BYTES,
                     MODULE_COUNT, inventory, reviewed_modules)
from .source import MistralSource, official_identity
from .storage import ProjectionStore, RouteSpool, StageStore

GIB = 1024 ** 3
RECIPE = dict(codebook='mcg', multiplier='0xCBAC1FED', core_bits=6, expert_bits=4,
    seed=787, sigma_reg=.025, out_scales='auto', source='exact-FP8-times-block128-qscale-to-BF16',
    dense_hessian='actual-native-Tekken-high-full-calibration-raw-XTX-FP32-no-synthetic-rows',
    hessian='signed-block-hadamard-congruence-fp64-v1',
    cold_experts=dict(minimum=1024, ranks=[5,8], order='rank-major-then-corpus-row',
        floor='raw-XTX-missing-times-I-equivalent-to-normalized-2I'),
    replay='serialized-MCG-FP16-scales-reconstructed-BF16-native-official-block',
    frontier='hidden-exact-DSA-topk-and-layer35-compressed-nope-KV',
    ordering='interleaved-core-and-experts-before-successor-layer',
    mtp='retained-native-byte-for-byte')


def compact():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class Progress:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()
        self.state = dict(status='running', started=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(), host=os.uname().nodename,
            projections=sum(1 for _ in (self.root/'evidence').rglob('*.json')),
            total_projections=MODULE_COUNT)

    def emit(self, event, **fields):
        if event == 'projection_committed':
            self.state['projections'] += 1
        row = dict(event=event, time=datetime.now(timezone.utc).isoformat(),
            rss_gib=self.process.memory_info().rss/GIB,
            cuda_gib=torch.cuda.memory_allocated()/GIB,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/GIB,
            free_gib=shutil.disk_usage(self.root).free/GIB, **fields)
        self.state.update(row)
        with (self.root/'events.jsonl').open('a') as stream:
            stream.write(canonical(row)+'\n')
        atomic_json(self.root/'status.json', self.state)
        print(canonical(row), flush=True)
        if row['free_gib'] < 64:
            raise OSError('producer exceeded disk admission')
        if row['rss_gib'] > 74 or row['cuda_gib'] > 86:
            raise MemoryError('producer exceeded host/GPU admission')


class Engine(StageEngine):
    def __init__(self, source, records, root, export, identity, selection, progress,
                 *, numerical=native, device='cuda:0'):
        self.source, self.records = source, records
        self.root, self.export = Path(root), Path(export)
        self.identity, self.selection, self.progress = identity, selection, progress
        self.math, self.device = numerical, device
        self.tokens = sum(len(row['input_ids']) for row in records)
        self.width = source.text_config['hidden_size']
        self.evidence_root = self.root/'evidence'
        self.boundary = RollingBoundary(self.root/'frontier', identity)
        self.rotary = None

    def checkpoints(self, layer):
        return ProjectionStore(self.root/('head/projections' if layer == 'head' else f'layers/{layer:03d}/projections'),
            self.evidence_root/('head' if layer == 'head' else f'layer-{layer:03d}'),
            self.identity, self.selection, self.progress, self.math, self.device)

    @torch.inference_mode()
    def initial(self):
        current = self.boundary.current(checksum=True)
        if current is not None:
            self.boundary.retire_old()
            self.retire_committed(current['layer'])
            return current
        directory = self.boundary.directory(-1)
        if directory.is_symlink():
            raise ValueError('refuse symlink initial frontier')
        if directory.exists():
            shutil.rmtree(directory)
        embedding = self.source.tensor('tok_embeddings.weight', self.device)
        entries = []
        for index, row in enumerate(self.records):
            ids = torch.tensor(row['input_ids'], device=self.device)
            hidden = torch.nn.functional.embedding(ids, embedding).unsqueeze(0)
            entry = commit_batch(directory, index, row, dict(hidden=hidden))
            entries.append({k:v for k,v in entry.items() if k != 'kind'})
            self._emit('embedding', -1, index)
        del embedding, hidden, ids
        self.source.close()
        compact()
        return self.boundary.commit(-1, entries)

    def _group(self, block, layer, checkpoints, names, store, phase, hessian):
        missing = [name for name in names if checkpoints.load(name) is None]
        raw = self._raw(store, phase, hessian) if missing else None
        for name in names:
            linear = native.linear_module(block, name)
            checkpoints.ensure(name, linear.weight, raw)
            checkpoints.install(name, linear)
        del raw
        compact()

    def _phase_a(self, block, layer, current, store):
        self._new_phase(store, 'a')
        h = self._hessian(self.width)
        entries = []
        for index, row in enumerate(self.records):
            state = self.boundary_batch(current, index)
            value = self.math.attention_inputs(block, state['hidden'])
            h.add(value)
            out = dict(attn_input=value, **{k:state[k] for k in native.STATE_KEYS if k in state})
            entries.append(commit_batch(store.directory('a'), index, row, out))
            self._emit('a', layer, index)
        return self._commit_phase(store, 'a', entries, [('input',h)])

    def _phase_c(self, block, layer, store):
        self._new_phase(store, 'c')
        h = self._hessian(block.self_attn.o_proj.in_features)
        entries = []
        if self.rotary is None:
            self.rotary = self.source.rotary(self.device)
        for index, row, state in self._read(store, 'a'):
            value, persistent = self.math.attention_capture(block, state['attn_input'], self.rotary, state)
            h.add(value)
            entries.append(commit_batch(store.directory('c'), index, row, dict(o_input=value, **persistent)))
            self._emit('c', layer, index)
        result = self._commit_phase(store, 'c', entries, [('o',h)])
        store.retire('a')
        return result

    def _phase_d(self, block, layer, current, store):
        self._new_phase(store, 'd')
        h = self._hessian(self.width)
        entries = []
        for index, row, state in self._read(store, 'c'):
            original = self.boundary_batch(current, index)
            residual, normalized = self.math.attention_finish(block, original['hidden'], block.self_attn.o_proj(state['o_input']))
            h.add(normalized)
            out = dict(residual=residual, mlp_input=normalized, **{k:state[k] for k in native.STATE_KEYS if k in state})
            entries.append(commit_batch(store.directory('d'), index, row, out))
            self._emit('d', layer, index)
        result = self._commit_phase(store, 'd', entries, [('mlp',h)])
        store.retire('c')
        return result

    def _phase_e(self, block, layer, store):
        self._new_phase(store, 'e')
        mlp = block.mlp if layer == 0 else block.mlp.shared_experts
        h = self._hessian(mlp.down_proj.in_features)
        for index, _, state in self._read(store, 'd'):
            h.add(self.math.shared_down_input(mlp, state['mlp_input']))
            self._emit('e', layer, index)
        return self._commit_phase(store, 'e', [], [('down',h)])

    def _experts(self, block, layer, store, checkpoints):
        root = self.root/f'layers/{layer:03d}'
        names = [name for name in self.selection if name.startswith(f'layers.{layer}.experts.')]
        complete = all(checkpoints.load(name) is not None for name in names)
        if complete:
            for name, parameter in list(block.mlp.experts.named_parameters()):
                block.mlp.experts._parameters[name] = nn.Parameter(torch.empty(parameter.shape,
                    dtype=torch.bfloat16, device=self.device), requires_grad=False)
        else:
            self.progress.emit('source_experts_load_started', layer=layer)
            self.source.materialize_routed_experts(block, layer=layer, device=self.device)
            self.progress.emit('source_experts_loaded', layer=layer)
        spool = None
        try:
            if not complete:
                spool = RouteSpool.create(root/'routes', self.identity, self.tokens, self._read(store,'d'),
                    lambda x:self.math.router_choices(block.mlp,x), self.progress, layer,
                    width=self.width, experts=128)
            for expert in range(128):
                cached_raw = cached_evidence = None
                for part in ('w1','w3','w2'):
                    name = f'layers.{layer}.experts.{expert}.{part}'
                    weight = self.math.expert_weight(block,expert,part)
                    raw = evidence = None
                    if checkpoints.load(name) is None:
                        if part == 'w2':
                            raw,evidence = spool.hessian(expert, down=(self.math.expert_weight(block,expert,'w1'),
                                self.math.expert_weight(block,expert,'w3')),device=self.device)
                        else:
                            if cached_raw is None:
                                cached_raw,cached_evidence = spool.hessian(expert,device=self.device)
                            raw,evidence = cached_raw,cached_evidence
                    checkpoints.ensure(name,weight,raw,evidence)
                    weight.copy_(checkpoints.dense(name))
                    del raw,evidence,weight
                del cached_raw,cached_evidence
                if expert % 16 == 0:
                    compact()
                    self.progress.emit('expert_progress', layer=layer, expert=expert+1, experts=128)
            atomic_json(root/'experts-complete.json',dict(identity=self.identity,layer=layer,modules=len(names)))
        finally:
            if spool is not None:
                spool.close()
        compact()

    def _phase_f(self, block, layer, store):
        self._new_phase(store,'f')
        entries = []
        for index,row,state in self._read(store,'d'):
            hidden = (state['residual']+block.mlp(state['mlp_input']) if layer == 0 else
                      self.math.complete_moe(state['residual'],state['mlp_input'],block.mlp))
            if not bool(torch.isfinite(hidden).all()):
                raise ValueError('nonfinite quantized Mistral successor')
            out = dict(hidden=hidden, **{k:state[k] for k in native.STATE_KEYS if k in state})
            entries.append(commit_batch(store.directory('f'),index,row,out))
            self._emit('f',layer,index)
        result = self._commit_phase(store,'f',entries)
        store.retire('a','c','d','e')
        return result

    @torch.inference_mode()
    def layer(self, layer, current):
        from .export import assemble_layer
        start = time.monotonic()
        store = StageStore(self.root/f'layers/{layer:03d}/stages',self.identity)
        phases = {phase:store.load(phase,checksum=True) for phase in ('a','c','d','e','f')}
        checkpoints = self.checkpoints(layer)
        prefix = f'layers.{layer}.'
        names = [name for name in self.selection if name.startswith(prefix)]
        if phases['f'] is None:
            block = self.source.load_block(layer,self.device,include_experts=False)
            for name in names:
                if '.experts.' not in name:
                    checkpoints.install(name,native.linear_module(block,name))
            if phases['d'] is None:
                if phases['c'] is None:
                    if phases['a'] is None:
                        self._phase_a(block,layer,current,store)
                    entry = [prefix+'attention.wq']
                    kv = prefix+'attention.wkv_a_with_mqa'
                    if kv in self.selection:
                        entry.append(kv)
                    self._group(block,layer,checkpoints,entry,store,'a','input')
                    self._phase_c(block,layer,store)
                self._group(block,layer,checkpoints,[prefix+'attention.wo'],store,'c','o')
                self._phase_d(block,layer,current,store)
            mlp = prefix+('feed_forward.' if layer == 0 else 'shared_experts.')
            self._group(block,layer,checkpoints,[mlp+'w1',mlp+'w3'],store,'d','mlp')
            if phases['e'] is None:
                self._phase_e(block,layer,store)
            self._group(block,layer,checkpoints,[mlp+'w2'],store,'e','down')
            if layer:
                self._experts(block,layer,store,checkpoints)
            self._phase_f(block,layer,store)
            del block
            compact()
        receipt = assemble_layer(self.source,layer,checkpoints.replacements(names),self.export,self.identity)
        manifest = store.load('f')
        links = hardlink_frontier(self.boundary,layer,store.directory('f'),manifest)
        current = self.boundary.commit(layer,links)
        store.clear()
        checkpoints.retire()
        routes = self.root/f'layers/{layer:03d}/routes'
        if routes.exists():
            shutil.rmtree(routes)
        self.progress.emit('layer_committed',layer=layer,seconds=time.monotonic()-start,payload_bytes=receipt['payload_bytes'])
        return current

    @torch.inference_mode()
    def head(self,current):
        store = StageStore(self.root/'head/stages',self.identity)
        checkpoints = self.checkpoints('head')
        if checkpoints.load('output') is None:
            if store.load('head',checksum=True) is None:
                self._new_phase(store,'head')
                norm = self.source.final_norm(self.device)
                h = self._hessian(self.width)
                for index,row in enumerate(self.records):
                    state = self.boundary_batch(current,index)
                    h.add(norm(state['hidden']))
                    self._emit('head',72,index)
                self._commit_phase(store,'head',[],[('head',h)])
                del norm,h
                compact()
            weight = self.source.tensor('output.weight',self.device)
            checkpoints.ensure('output',weight,self._raw(store,'head','head'))
            del weight
            compact()
        self.source.close()
        return checkpoints


def make_plan(source,tokens,root,export):
    summary = inventory(source)
    manifest = json.loads((tokens/'manifest.json').read_text())
    calibration = manifest['splits']['calibration']
    if source.snapshot.name != BASE_REVISION or calibration['count'] != 1426 or calibration['tokens'] != 1_126_268:
        raise ValueError('production requires pinned source and full native Mistral calibration stream')
    code = [Path(__file__).with_name(name) for name in
            ('run.py','storage.py','recipe.py','export.py','source.py','native.py','numerical.py')]
    code += [REPO/'quantization/mistral_large4/prepare_corpus.py',
        REPO/'quantization/glm5/core_k6/run.py',REPO/'quantization/glm5/core_k6/storage.py',
        REPO/'quantization/glm5/prepare_corpus.py',
        REPO/'quantization/kimi_k3/core_k6/run.py',REPO/'quantization/kimi_k3/core_k6/storage.py',
        REPO/'quantization/kimi_k3/uniform_k2/storage.py']
    identity = dict(schema='cuteafd-mistral-large4-k4-core-k6-production-v1', source_repo=BASE_REPO,
        source_revision=BASE_REVISION, source_config_sha256=sha256_file(source.snapshot/'config.json'),
        source_index_sha256=sha256_file(source.snapshot/source.index_name),
        token_manifest_sha256=sha256_file(tokens/'manifest.json'),recipe=RECIPE,
        code={str(p.relative_to(REPO)):sha256_file(p) for p in sorted(code)},
        official=official_identity(), calibration=dict(samples=calibration['count'],tokens=calibration['tokens']),
        gptqmodel_commit=subprocess.check_output(['git','-C',str(REPO/'third_party/gptqmodel'),'rev-parse','HEAD'],text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','safetensors','mistral-common')})
    return dict(identity=identity,source=str(source.snapshot),tokens=str(tokens),root=str(root),export=str(export),
        inventory=summary,temporary_reserve_bytes=192*GIB,
        required_free_bytes=TARGET_PAYLOAD_BYTES+256*GIB,publication_repo=TARGET_REPO)


def cleanup_complete(root):
    for path in (root/'frontier',root/'head/stages',root/'head/projections'):
        if path.is_symlink():
            raise ValueError('refuse symlink recovery cleanup')
        if path.exists():
            shutil.rmtree(path)
    fsync_directory(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('source-snapshot','tokens','root','export'):
        parser.add_argument('--'+arg,type=Path,required=True)
    for flag in ('resume','preflight','publish-only'):
        parser.add_argument('--'+flag,action='store_true')
    args = parser.parse_args()
    source = MistralSource(args.source_snapshot)
    args.tokens,args.root,args.export = (p.expanduser().resolve() for p in (args.tokens,args.root,args.export))
    args.root.mkdir(parents=True,exist_ok=True)
    plan = make_plan(source,args.tokens,args.root,args.export)
    prior = args.root/'plan.json'
    if prior.exists() and json.loads(prior.read_text()) != plan:
        raise ValueError('immutable numerical production plan differs: investigate cache validity before recovery')
    if args.preflight:
        if not prior.exists() and shutil.disk_usage(args.root).free < plan['required_free_bytes']:
            raise OSError('insufficient work disk for output plus bounded frontier reserve')
        print(canonical(plan),flush=True)
        return
    if prior.exists() and not args.resume and not args.publish_only:
        raise ValueError('existing production requires explicit --resume')
    if not prior.exists() and (args.resume or args.publish_only):
        raise ValueError('recovery requires committed plan')
    if args.publish_only:
        from .publish import publish
        with exclusive_run(args.root):
            completed = json.loads((args.root/'quantization-complete.json').read_text())
            if completed['identity'] != plan['identity']:
                raise ValueError('publication requires matching completed numerical producer')
            publish(args.export,args.root,plan['publication_repo'])
        return
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86*GIB/torch.cuda.get_device_properties(0).total_memory)
    progress = Progress(args.root)
    with torch.inference_mode(),exclusive_run(args.root):
        if not prior.exists():
            atomic_json(prior,plan)
        if (args.root/'quantization-complete.json').exists():
            cleanup_complete(args.root)
            progress.state['status'] = 'quantized'
            progress.emit('quantization_complete')
            return
        records = load_token_records(args.tokens,'calibration')
        engine = Engine(source,records,args.root,args.export,plan['identity'],reviewed_modules(),progress)
        current = engine.initial()
        for layer in range(current['layer']+1,72):
            current = engine.layer(layer,current)
        head = engine.head(current)
        from .export import finalize_export, audit_completed_export
        summary = finalize_export(source,engine.selection,head.replacements(['output']),args.export,engine.identity,args.root/'evidence')
        audit = audit_completed_export(source,args.export,engine.identity)
        atomic_json(args.root/'completed-export-audit.json',audit)
        atomic_json(args.root/'quantization-complete.json',dict(identity=engine.identity,summary=summary))
        cleanup_complete(args.root)
        progress.state['status'] = 'quantized'
        progress.emit('quantization_complete',**summary)
    source.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]).expanduser()
            category='oom' if isinstance(error,(MemoryError,torch.OutOfMemoryError)) else 'other'
            message=str(error).lower()
            if any(token in message for token in ('nonfinite','non-finite','nan','cholesky','positive-definite','hessian')):
                category='numerical'
            atomic_json(root/'failure.json',dict(category=category,error_type=type(error).__name__,
                error=str(error),attempt=os.environ.get('MISTRAL_QUANT_ATTEMPT'),time=time.time_ns()))
        raise
