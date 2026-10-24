#!/usr/bin/env python3
"""Interleaved, resumable GLM K4 experts / K6 core on one local RTX."""
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

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,os.environ.get("GLM_GPTQMODEL_SOURCE",str(REPO/"third_party/gptqmodel")))
sys.path.insert(0,str(REPO/"quantization"))
if __package__ in (None,""):
    __package__="glm5.core_k6"
os.environ.setdefault("PYTORCH_ALLOC_CONF","expandable_segments:True")

import psutil
import torch
from torch import nn
from safetensors.torch import load_file

from glm5.prepare_corpus import load_token_records
from kimi_k3.uniform_k2.storage import (RollingBoundary,atomic_json,canonical,
    commit_safetensors,file_identity,fsync_directory,sha256_file,verify_receipt)
from kimi_k3.core_k6.run import exclusive_run,hardlink_frontier
from .recipe import (BASE_REPO,BASE_REVISION,TARGET_REPO,TARGET_PAYLOAD_BYTES,
    MODULE_COUNT,reviewed_modules,inventory)
from .storage import ProjectionStore,RouteSpool,StageStore,load_tensors

GIB=1024**3
RECIPE=dict(codebook="mcg",multiplier="0xCBAC1FED",core_bits=6,expert_bits=4,mtp_bits=4,
    seed=787,sigma_reg=.025,out_scales="auto",capture="all-native-GLM-calibration-raw-XTX-FP32",
    hessian="signed-block-hadamard-congruence-fp64-v1",core_identity_rows=0,
    cold_experts=dict(minimum=1024,ranks=[9,16],order="rank-major-then-corpus-row",
        floor="raw-XTX-missing-times-I-equivalent-to-normalized-2I"),
    replay="serialized-MCG-FP16-scales-reconstructed-BF16-native-block",
    frontier="hidden-and-exact-shared-DSA-topk-and-absolute-positions",
    ordering="interleaved-core-and-experts-before-successor-layer",
    mtp="post-final-target-norm-target[t]-and-token[t+1]-target-position[t]-zero-pos0-embedding")


def compact():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class Progress:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.process=psutil.Process()
        self.state=dict(status="running",started=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            host=os.uname().nodename,projections=sum(1 for _ in (self.root/"evidence").rglob("*.json")),
            total_projections=MODULE_COUNT)

    def emit(self,event,**fields):
        if event == "projection_committed":
            self.state["projections"]+=1
        row=dict(event=event,time=datetime.now(timezone.utc).isoformat(),
            rss_gib=self.process.memory_info().rss/GIB,open_files=self.process.num_fds(),
            cuda_gib=torch.cuda.memory_allocated()/GIB,gpu_peak_gib=torch.cuda.max_memory_allocated()/GIB,
            free_gib=shutil.disk_usage(self.root).free/GIB,**fields)
        self.state.update(row)
        with (self.root/"events.jsonl").open("a") as f:
            f.write(canonical(row)+"\n")
        atomic_json(self.root/"status.json",self.state)
        print(canonical(row),flush=True)
        if row["rss_gib"] > 58 or row["cuda_gib"] > 86 or row["free_gib"] < 64:
            raise MemoryError("GLM producer exceeded host/GPU/disk admission")


def commit_batch(directory,index,row,state):
    name=f"batch-{index:05d}.safetensors"
    return dict(filename=name,kind="batch",id=row["id"],tokens=len(row["input_ids"]),
        **commit_safetensors(Path(directory)/name,state))


def batches(manifest):
    return [entry for entry in manifest["entries"] if entry.get("kind")=="batch"]


class Engine:
    def __init__(self,source,records,root,export,identity,selection,progress,*,numerical=None,
                 evidence_root=None,device="cuda:0"):
        if numerical is None:
            from gptqmodel.utils import glm5_core as numerical
        self.source,self.records,self.root,self.export=source,records,Path(root),Path(export)
        self.identity,self.selection,self.progress=identity,selection,progress
        self.math,self.device=numerical,device
        self.tokens=sum(len(r["input_ids"]) for r in records)
        self.width=source.config["hidden_size"]
        self.evidence_root=Path(evidence_root or self.root/"evidence")
        self.boundary=RollingBoundary(self.root/"frontier",identity)
        self.rotary=None

    def checkpoints(self,layer):
        return ProjectionStore(self.root/"head/projections" if layer=="head" else self.root/f"layers/{layer:03d}/projections",
            self.evidence_root/("head" if layer=="head" else f"layer-{layer:03d}"),self.identity,
            self.selection,self.progress,self.math,self.device)

    def _emit(self,phase,layer,index):
        if index % 64 == 0 or index+1==len(self.records):
            self.progress.emit("phase_progress",phase=phase,layer=layer,batches=index+1,total_batches=len(self.records))

    def _context(self,state):
        if self.rotary is None:
            self.rotary=self.source.rotary(self.device)
        return self.math.prepare_context(state.get("attn_input",state.get("hidden")),self.rotary,
            state.get("prev_topk_indices"),position_ids=state.get("position_ids"))

    def boundary_batch(self,current,index):
        entry=current["manifest"]["batches"][index]
        row=self.records[index]
        if entry["id"] != row["id"] or entry["tokens"] != len(row["input_ids"]):
            raise ValueError("frontier batch differs from exact corpus order")
        return load_tensors(self.boundary.root/current["directory"]/entry["filename"],self.device)

    def retire_committed(self,last_layer):
        for directory in (self.root/"layers").glob("[0-9][0-9][0-9]"):
            if int(directory.name)>last_layer:
                continue
            for name in ("stages","routes"):
                path=directory/name
                if path.is_symlink():
                    raise ValueError("refuse symlink committed recovery state")
                if path.exists():
                    shutil.rmtree(path)
            payloads=directory/"projections"
            if payloads.exists():
                for path in payloads.glob("*.safetensors"):
                    path.unlink()

    @torch.inference_mode()
    def initial(self):
        current=self.boundary.current(checksum=True)
        if current is not None:
            self.boundary.retire_old()
            self.retire_committed(current["layer"])
            return current
        directory=self.boundary.directory(-1)
        if directory.is_symlink():
            raise ValueError("refuse symlink initial frontier")
        if directory.exists():
            shutil.rmtree(directory)
        embedding=self.source.tensor("model.embed_tokens.weight",self.device)
        entries=[]
        for index,row in enumerate(self.records):
            ids=torch.tensor(row["input_ids"],device=self.device)
            hidden=torch.nn.functional.embedding(ids,embedding).unsqueeze(0)
            entry=commit_batch(directory,index,row,dict(hidden=hidden))
            entries.append({k:v for k,v in entry.items() if k!="kind"})
            self._emit("embedding",-1,index)
        del embedding,hidden,ids
        self.source.close();compact()
        return self.boundary.commit(-1,entries)

    def _new_phase(self,store,phase):
        store.retire(phase)
        store.directory(phase).mkdir(parents=True)

    def _hessian(self,width):
        return self.math.DenseRawHessian(width,device=self.device)

    def _commit_phase(self,store,phase,entries,hessians=()):
        for name,h in hessians:
            raw=h.raw_dict(minimum_rows=self.tokens)
            if raw["count"]!=self.tokens:
                raise ValueError("dense Hessian must contain only the full actual stream")
            filename=f"hessian-{name}.safetensors"
            entries.append(dict(filename=filename,kind="hessian",name=name,count=self.tokens,
                **commit_safetensors(store.directory(phase)/filename,dict(H=raw["H"])) ))
        result=store.commit(phase,entries,dict(tokens=self.tokens,batches=len(self.records)))
        self.progress.emit("phase_committed",phase=phase)
        return result

    def _raw(self,store,phase,name):
        manifest=store.load(phase)
        entries=[] if manifest is None else [x for x in manifest["entries"] if x.get("kind")=="hessian" and x.get("name")==name]
        if len(entries)!=1 or entries[0]["count"]!=self.tokens:
            raise ValueError("committed raw Hessian is absent or has wrong sample count")
        return dict(H=load_tensors(store.directory(phase)/entries[0]["filename"],self.device)["H"],
            count=self.tokens,finalized=False,identity_rows=0)

    def _read(self,store,phase):
        manifest=store.load(phase)
        if manifest is None or len(batches(manifest))!=len(self.records):
            raise ValueError("committed phase batch inventory differs")
        for index,(entry,row) in enumerate(zip(batches(manifest),self.records,strict=True)):
            if entry["id"]!=row["id"] or entry["tokens"]!=len(row["input_ids"]):
                raise ValueError("committed phase corpus order differs")
            yield index,row,load_tensors(store.directory(phase)/entry["filename"],self.device)

    def _group(self,block,layer,checkpoints,names,store,phase,hessian):
        missing=[name for name in names if checkpoints.load(name) is None]
        raw=self._raw(store,phase,hessian) if missing else None
        for name in names:
            linear=block.get_submodule(name.removeprefix(f"model.layers.{layer}."))
            checkpoints.ensure(name,linear.weight,raw)
            checkpoints.install(name,linear)
        del raw;compact()

    def _phase_a(self,block,layer,current,store):
        self._new_phase(store,"a");h=self._hessian(self.width);entries=[]
        for index,row in enumerate(self.records):
            state=self.boundary_batch(current,index)
            value=self.math.attention_inputs(block,state["hidden"])
            h.add(value)
            out=dict(attn_input=value)
            out.update({k:state[k] for k in ("prev_topk_indices","position_ids") if k in state})
            entries.append(commit_batch(store.directory("a"),index,row,out))
            self._emit("a",layer,index)
        return self._commit_phase(store,"a",entries,[("input",h)])

    def _phase_b(self,block,layer,store):
        self._new_phase(store,"b");h=self._hessian(self.source.config["q_lora_rank"])
        for index,_,state in self._read(store,"a"):
            value=block.self_attn.q_a_layernorm(block.self_attn.q_a_proj(state["attn_input"]))
            h.add(value);self._emit("b",layer,index)
        return self._commit_phase(store,"b",[],[("q-b",h)])

    def _phase_c(self,block,layer,store):
        self._new_phase(store,"c");h=self._hessian(block.self_attn.o_proj.in_features);entries=[]
        for index,row,state in self._read(store,"a"):
            captured={}
            def retain_topk(module,args,result):
                captured["topk"]=result.detach().clone()
            handle=None
            if block.self_attn.indexer is not None:
                handle=block.self_attn.indexer.register_forward_hook(retain_topk)
            try:
                value=self.math.capture_linear_input(block.self_attn.o_proj,block.self_attn,
                    hidden_states=state["attn_input"],past_key_values=None,**self._context(state))
            finally:
                if handle is not None:
                    handle.remove()
            topk=captured.get("topk",state.get("prev_topk_indices"))
            if topk is None:
                raise ValueError("attention capture lost exact DSA top-k state")
            h.add(value)
            out=dict(o_input=value,prev_topk_indices=topk)
            if "position_ids" in state:
                out["position_ids"]=state["position_ids"]
            entries.append(commit_batch(store.directory("c"),index,row,out))
            self._emit("c",layer,index)
        result=self._commit_phase(store,"c",entries,[("o",h)])
        store.retire("a","b")
        return result

    def _phase_d(self,block,layer,current,store):
        self._new_phase(store,"d");h=self._hessian(self.width);entries=[]
        for index,row,state in self._read(store,"c"):
            original=self.boundary_batch(current,index)
            residual,normalized=self.math.attention_finish(block,original["hidden"],block.self_attn.o_proj(state["o_input"]))
            h.add(normalized)
            out=dict(residual=residual,mlp_input=normalized,prev_topk_indices=state["prev_topk_indices"])
            if "position_ids" in state:
                out["position_ids"]=state["position_ids"]
            entries.append(commit_batch(store.directory("d"),index,row,out))
            self._emit("d",layer,index)
        result=self._commit_phase(store,"d",entries,[("mlp",h)])
        store.retire("c")
        return result

    def _phase_e(self,block,layer,store):
        self._new_phase(store,"e")
        mlp=block.mlp.shared_experts if layer>=3 else block.mlp
        h=self._hessian(mlp.down_proj.in_features)
        for index,_,state in self._read(store,"d"):
            h.add(self.math.shared_down_input(mlp,state["mlp_input"]))
            self._emit("e",layer,index)
        return self._commit_phase(store,"e",[],[("down",h)])

    @staticmethod
    def _expert_weight(block,expert,part):
        module=block.mlp.experts
        if part=="down_proj":
            return module.down_proj[expert]
        inter=module.gate_up_proj.shape[1]//2
        return module.gate_up_proj[expert,0:inter] if part=="gate_proj" else module.gate_up_proj[expert,inter:]

    def _experts(self,block,layer,store,checkpoints):
        root=self.root/f"layers/{layer:03d}"
        names=[name for name in self.selection if name.startswith(f"model.layers.{layer}.mlp.experts.")]
        complete=all(checkpoints.load(name) is not None for name in names)
        if complete:
            for name,parameter in list(block.mlp.experts.named_parameters()):
                block.mlp.experts._parameters[name]=nn.Parameter(torch.empty(parameter.shape,
                    dtype=torch.bfloat16,device=self.device),requires_grad=False)
        else:
            self.progress.emit("source_experts_load_started",layer=layer,phase="experts")
            self.source.materialize_routed_experts(block,layer=layer,device=self.device)
            self.progress.emit("source_experts_loaded",layer=layer)
        spool=None
        try:
            if not complete:
                spool=RouteSpool.create(root/"routes",self.identity,self.tokens,self._read(store,"d"),
                    lambda x:self.math.router_choices(block.mlp,x),self.progress,layer,
                    width=self.width,experts=self.source.config["n_routed_experts"])
            for expert in range(self.source.config["n_routed_experts"]):
                cached_raw=cached_evidence=None
                for part in ("gate_proj","up_proj","down_proj"):
                    name=f"model.layers.{layer}.mlp.experts.{expert}.{part}"
                    weight=self._expert_weight(block,expert,part)
                    raw=evidence=None
                    if checkpoints.load(name) is None:
                        if part=="down_proj":
                            raw,evidence=spool.hessian(expert,down=block.mlp.experts.gate_up_proj[expert],device=self.device)
                        else:
                            if cached_raw is None:
                                cached_raw,cached_evidence=spool.hessian(expert,device=self.device)
                            raw,evidence=cached_raw,cached_evidence
                    checkpoints.ensure(name,weight,raw,evidence)
                    weight.copy_(checkpoints.dense(name))
                    del raw,evidence,weight
                del cached_raw,cached_evidence
                if expert % 16==0:
                    compact();self.progress.emit("expert_progress",layer=layer,phase="experts",expert=expert+1,experts=256)
            atomic_json(root/"experts-complete.json",dict(identity=self.identity,layer=layer,modules=len(names)))
        finally:
            if spool is not None:
                spool.close()
        compact()

    def _phase_f(self,block,layer,store):
        self._new_phase(store,"f");entries=[]
        for index,row,state in self._read(store,"d"):
            hidden=(self.math.complete_moe(state["residual"],state["mlp_input"],block.mlp) if layer>=3
                    else state["residual"]+block.mlp(state["mlp_input"]))
            if not bool(torch.isfinite(hidden).all()):
                raise ValueError("nonfinite final quantized successor")
            out=dict(hidden=hidden,prev_topk_indices=state["prev_topk_indices"])
            if "position_ids" in state:
                out["position_ids"]=state["position_ids"]
            entries.append(commit_batch(store.directory("f"),index,row,out))
            self._emit("f",layer,index)
        result=self._commit_phase(store,"f",entries)
        store.retire("a","b","c","d","e")
        return result

    @torch.inference_mode()
    def layer(self,layer,current,*,block=None,boundary_layer=None):
        from .export import assemble_layer
        start=time.monotonic()
        boundary_layer=layer if boundary_layer is None else boundary_layer
        store=StageStore(self.root/f"layers/{layer:03d}/stages",self.identity)
        phases={phase:store.load(phase,checksum=True) for phase in ("a","b","c","d","e","f")}
        checkpoints=self.checkpoints(layer)
        prefix=f"model.layers.{layer}."
        names=[name for name in self.selection if name.startswith(prefix)]
        if phases["f"] is None:
            if block is None:
                block=self.source.load_block(layer,self.device,include_experts=False)
            for name in names:
                if ".experts." not in name and name!=prefix+"eh_proj":
                    checkpoints.install(name,block.get_submodule(name.removeprefix(prefix)))
            attention=lambda part:prefix+"self_attn."+part
            if phases["d"] is None:
                if phases["c"] is None:
                    if phases["a"] is None:
                        self._phase_a(block,layer,current,store)
                    self._group(block,layer,checkpoints,[attention("q_a_proj"),attention("kv_a_proj_with_mqa")],store,"a","input")
                    if phases["b"] is None:
                        self._phase_b(block,layer,store)
                    self._group(block,layer,checkpoints,[attention("q_b_proj")],store,"b","q-b")
                    self._phase_c(block,layer,store)
                self._group(block,layer,checkpoints,[attention("o_proj")],store,"c","o")
                self._phase_d(block,layer,current,store)
            mlp=prefix+("mlp.shared_experts." if layer>=3 else "mlp.")
            self._group(block,layer,checkpoints,[mlp+"gate_proj",mlp+"up_proj"],store,"d","mlp")
            if phases["e"] is None:
                self._phase_e(block,layer,store)
            self._group(block,layer,checkpoints,[mlp+"down_proj"],store,"e","down")
            if layer>=3:
                self._experts(block,layer,store,checkpoints)
            self._phase_f(block,layer,store)
            del block;compact()
        # Do not advance/retire the frontier until the complete layer shard
        # and its independent preserved-byte audit are durable.
        receipt=assemble_layer(self.source,layer,checkpoints.replacements(names),self.export,self.identity)
        manifest=store.load("f")
        links=hardlink_frontier(self.boundary,boundary_layer,store.directory("f"),manifest)
        current=self.boundary.commit(boundary_layer,links)
        store.clear();checkpoints.retire()
        routes=self.root/f"layers/{layer:03d}/routes"
        if routes.exists():
            shutil.rmtree(routes)
        self.progress.emit("layer_committed",layer=layer,seconds=time.monotonic()-start,
            payload_bytes=receipt["payload_bytes"])
        return current

    @torch.inference_mode()
    def head(self,current):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm
        store=StageStore(self.root/"head/stages",self.identity)
        checkpoints=self.checkpoints("head")
        if checkpoints.load("lm_head") is None:
            if store.load("head",checksum=True) is None:
                self._new_phase(store,"head")
                norm=GlmMoeDsaRMSNorm(self.width,self.source.config["rms_norm_eps"]).to(self.device)
                norm.weight=nn.Parameter(self.source.tensor("model.norm.weight",self.device),requires_grad=False)
                h=self._hessian(self.width)
                for index,row in enumerate(self.records):
                    state=self.boundary_batch(current,index)
                    h.add(norm(state["hidden"]))
                    self._emit("head",79,index)
                self._commit_phase(store,"head",[],[("head",h)])
                del norm,h;compact()
            weight=self.source.tensor("lm_head.weight",self.device)
            checkpoints.ensure("lm_head",weight,self._raw(store,"head","head"))
            del weight;compact()
        self.source.close()
        return checkpoints


def make_plan(source,tokens,root,export):
    summary=inventory(source)
    manifest=json.loads((tokens/"manifest.json").read_text())
    if (source.snapshot.name!=BASE_REVISION or manifest["splits"]["calibration"]["tokens"]!=1_097_737
            or manifest["splits"]["calibration"]["count"]!=1426):
        raise ValueError("production requires the pinned full BF16 source and full GLM calibration stream")
    code=[*(Path(__file__).with_name(name) for name in ("run.py","storage.py","recipe.py","export.py")),
        REPO/"quantization/glm5/prepare_corpus.py",REPO/"quantization/kimi_k3/prepare_corpus.py",
        REPO/"quantization/kimi_k3/core_k6/run.py",
        REPO/"quantization/kimi_k3/core_k6/storage.py",REPO/"quantization/kimi_k3/uniform_k2/storage.py"]
    fork=REPO/"third_party/gptqmodel"
    code += [fork/"gptqmodel/utils/glm5_core.py",fork/"gptqmodel/utils/glm5_mtp.py"]
    identity=dict(schema="cuteafd-glm53-k4-core-k6-production-v1",source_repo=BASE_REPO,source_revision=BASE_REVISION,
        source_config_sha256=sha256_file(source.snapshot/"config.json"),source_index_sha256=sha256_file(source.snapshot/"model.safetensors.index.json"),
        token_manifest_sha256=sha256_file(tokens/"manifest.json"),recipe=RECIPE,
        code={str(p.relative_to(REPO)):sha256_file(p) for p in code},
        gptqmodel_commit=subprocess.check_output(["git","-C",str(fork),"rev-parse","HEAD"],text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ("torch","transformers","numpy","safetensors")})
    # Native hidden+topk predecessor/successor, o-input and post-attention
    # stages, route spool, and one layer's packed/assembled shard; no full
    # model or growing history of activation banks is ever reserved.
    temporary=192*GIB
    return dict(identity=identity,source=str(source.snapshot),tokens=str(tokens),root=str(root),export=str(export),
        inventory=summary,temporary_reserve_bytes=temporary,required_free_bytes=TARGET_PAYLOAD_BYTES+temporary+64*GIB,
        publication_repo=TARGET_REPO)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("source-snapshot","tokens","root","export"):
        parser.add_argument("--"+arg,type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--publish-only",action="store_true")
    args=parser.parse_args()
    from gptqmodel.utils.glm5_core import GlmBF16Source
    source=GlmBF16Source(args.source_snapshot)
    args.tokens,args.root,args.export=(p.expanduser().resolve() for p in (args.tokens,args.root,args.export))
    args.root.mkdir(parents=True,exist_ok=True)
    plan=make_plan(source,args.tokens,args.root,args.export)
    prior=args.root/"plan.json"
    if prior.exists() and json.loads(prior.read_text())!=plan:
        raise ValueError("immutable production plan differs")
    if args.preflight:
        if shutil.disk_usage(args.root).free < plan["required_free_bytes"] and not (args.root/"plan.json").exists():
            raise OSError("not enough root NVMe space for output and bounded recovery reserve")
        print(canonical(plan),flush=True)
        return
    if prior.exists():
        if not args.resume and not args.publish_only:
            raise ValueError("existing production run requires explicit --resume")
    elif args.resume or args.publish_only:
        raise ValueError("recovery requires a committed plan")
    progress=Progress(args.root)
    if args.publish_only:
        from .publish import publish
        with exclusive_run(args.root):
            completed=json.loads((args.root/"quantization-complete.json").read_text())
            if completed["identity"]!=plan["identity"]:
                raise ValueError("publication requires completed matching producer")
            publish(args.export,args.root,plan["publication_repo"])
        return
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(86*GIB/torch.cuda.get_device_properties(0).total_memory)
    with torch.inference_mode(),exclusive_run(args.root):
        if not prior.exists():
            atomic_json(prior,plan)
        if (args.root/"quantization-complete.json").exists():
            cleanup_complete(args.root)
            progress.state["status"]="quantized"
            progress.emit("quantization_complete")
            return
        records=load_token_records(args.tokens,"calibration")
        engine=Engine(source,records,args.root,args.export,plan["identity"],reviewed_modules(),progress)
        current=engine.initial()
        for layer in range(current["layer"]+1,78):
            current=engine.layer(layer,current)
        head=engine.head(current)
        run_mtp(engine,current)
        from .export import finalize_export
        summary=finalize_export(source,engine.selection,head.replacements(["lm_head"]),args.export,
            engine.identity,args.root/"evidence")
        atomic_json(args.root/"quantization-complete.json",dict(identity=engine.identity,summary=summary))
        cleanup_complete(args.root)
        progress.state["status"]="quantized"
        progress.emit("quantization_complete",**summary)
    source.close()


def cleanup_complete(root):
    for path in (root/"frontier",root/"mtp/frontier",root/"mtp/input",root/"head/stages"):
        if path.is_symlink():
            raise ValueError("refuse symlink recovery cleanup")
        if path.exists():
            shutil.rmtree(path)
    for base in (root/"head/projections",root/"mtp/layers/078/projections"):
        if base.exists():
            for path in base.glob("*.safetensors"):
                path.unlink()
    fsync_directory(root)


def run_mtp(engine,target_current):
    # The MTP envelope helper and its independent native alignment oracle
    # are supplied by the fork, separate from the Flash/mHC adapter.
    from gptqmodel.utils import glm5_mtp as mtp_math
    source=engine.source
    root=engine.root/"mtp"
    records=[dict(id=row["id"],input_ids=row["input_ids"][1:]) for row in engine.records]
    sub=Engine(source,records,root,engine.export,engine.identity,engine.selection,engine.progress,
        numerical=engine.math,evidence_root=engine.evidence_root,device=engine.device)
    current=sub.boundary.current(checksum=True)
    if current is not None and current["layer"]==0:
        sub.retire_committed(78)
        return
    adapter=mtp_math.load_mtp(source,engine.device,include_experts=False)
    checkpoints=sub.checkpoints(78)
    eh_name="model.layers.78.eh_proj"
    store=StageStore(root/"input",engine.identity)
    if current is None:
        if store.load("i",checksum=True) is None:
            sub._new_phase(store,"i")
            embedding=source.tensor("model.embed_tokens.weight",engine.device)
            from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm
            norm=GlmMoeDsaRMSNorm(engine.width,source.config["rms_norm_eps"]).to(engine.device)
            norm.weight=nn.Parameter(source.tensor("model.norm.weight",engine.device),requires_grad=False)
            h=sub._hessian(engine.width*2);entries=[]
            for index,row in enumerate(engine.records):
                state=engine.boundary_batch(target_current,index)
                ids=torch.tensor(row["input_ids"],device=engine.device).unsqueeze(0)
                aligned=mtp_math.aligned_inputs(state["hidden"],ids,embedding,norm)
                envelope=adapter.envelope(aligned)
                h.add(envelope)
                entries.append(commit_batch(store.directory("i"),index,records[index],
                    dict(envelope=envelope,position_ids=aligned["position_ids"])))
                sub._emit("mtp-envelope",78,index)
            sub._commit_phase(store,"i",entries,[("eh",h)])
            del embedding,norm,h,state,aligned,envelope,ids;compact()
        checkpoints.ensure(eh_name,adapter.eh_proj.weight,lambda:sub._raw(store,"i","eh"))
        checkpoints.install(eh_name,adapter.eh_proj)
        directory=sub.boundary.directory(-1)
        if directory.exists():
            shutil.rmtree(directory)
        entries=[]
        for index,row,state in sub._read(store,"i"):
            hidden=adapter.eh_proj(state["envelope"])
            entry=commit_batch(directory,index,row,dict(hidden=hidden,position_ids=state["position_ids"]))
            entries.append({k:v for k,v in entry.items() if k!="kind"})
        current=sub.boundary.commit(-1,entries)
        store.clear()
    sub.layer(78,current,block=adapter.decoder,boundary_layer=0)
    del adapter,sub;compact()


if __name__=="__main__":
    main()
