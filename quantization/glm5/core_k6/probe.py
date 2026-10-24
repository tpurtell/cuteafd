#!/usr/bin/env python3
"""Bounded real-weight causal qualification; its partial export is not published."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/"quantization"))
sys.path.insert(0,str(REPO/"third_party/gptqmodel"))
if __package__ in (None,""):
    __package__="glm5.core_k6"

import torch
from torch import nn
from glm5.prepare_corpus import load_token_records
from gptqmodel.utils import glm5_core as math,glm5_mtp as mtp_math
from . import run
from .recipe import inventory,reviewed_modules
from .storage import load_tensors


class CheckedEngine(run.Engine):
    def _phase_f(self,block,layer,store):
        manifest=super()._phase_f(block,layer,store)
        previous=self.boundary.current()
        for index,row,actual in self._read(store,"f"):
            state=self.boundary_batch(previous,index)
            expected,topk=block(state["hidden"],past_key_values=None,use_cache=False,
                **math.prepare_context(state["hidden"],self.rotary,state.get("prev_topk_indices"),
                    position_ids=state.get("position_ids")))
            if not torch.equal(expected,actual["hidden"]) or not torch.equal(topk,actual["prev_topk_indices"]):
                raise ValueError("full native and independently staged quantized GLM boundary differ")
        self.progress.emit("native_quantized_boundary_exact",layer=layer,tokens=self.tokens)
        return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot","tokens","root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    source=math.GlmBF16Source(args.source_snapshot)
    summary=inventory(source)
    rows=load_token_records(args.tokens,"calibration")
    row=next(row for row in rows if row["id"]=="cal-1304")
    if len(row["input_ids"])!=2205:
        raise ValueError("native qualification requires the unchanged sparse-attention witness")
    identity=dict(schema="glm53-native-qualification-v1",source=args.source_snapshot.name,
        record=row["id"],tokens=len(row["input_ids"]),fork_commit=subprocess.check_output(
            ["git","-C",str(REPO/"third_party/gptqmodel"),"rev-parse","HEAD"],text=True).strip(),
        token_manifest_sha256=run.sha256_file(args.tokens/"manifest.json"),
        code={name:run.sha256_file(Path(__file__).with_name(name)) for name in
              ("run.py","storage.py","probe.py","recipe.py","export.py")})
    plan=args.root/"plan.json"
    if plan.exists() and json.loads(plan.read_text())!=identity:
        raise ValueError("native probe source changed; use a new root")
    run.atomic_json(plan,identity)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(86*run.GIB/torch.cuda.get_device_properties(0).total_memory)
    progress=run.Progress(args.root)
    progress.state["total_projections"]=798
    with torch.inference_mode(),run.exclusive_run(args.root):
        engine=CheckedEngine(source,[row],args.root,args.root/"partial-export",identity,
            reviewed_modules(),progress,numerical=math)
        current=engine.initial()
        for layer in range(current["layer"]+1,4):
            current=engine.layer(layer,current)
        state=engine.boundary_batch(current,0)
        adapter=mtp_math.load_mtp(source,"cuda:0",include_experts=True)
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm
        norm=GlmMoeDsaRMSNorm(6144,source.config["rms_norm_eps"]).cuda()
        norm.weight=nn.Parameter(source.tensor("model.norm.weight","cuda:0"),requires_grad=False)
        embedding=source.tensor("model.embed_tokens.weight","cuda:0")
        ids=torch.tensor(row["input_ids"],device="cuda:0").unsqueeze(0)
        aligned=mtp_math.aligned_inputs(state["hidden"],ids,embedding,norm)
        envelope=adapter.envelope(aligned)
        accumulator=math.DenseRawHessian(12288,device="cuda:0")
        accumulator.add(envelope)
        checkpoints=engine.checkpoints(78)
        name="model.layers.78.eh_proj"
        checkpoints.ensure(name,adapter.eh_proj.weight,accumulator.raw_dict())
        checkpoints.install(name,adapter.eh_proj)
        hidden=adapter.projected_inputs(aligned)
        normalized=adapter.decoder.input_layernorm(hidden)
        accumulator=math.DenseRawHessian(6144,device="cuda:0")
        accumulator.add(normalized)
        name="model.layers.78.self_attn.kv_a_proj_with_mqa"
        checkpoints.ensure(name,adapter.decoder.self_attn.kv_a_proj_with_mqa.weight,accumulator.raw_dict())
        checkpoints.install(name,adapter.decoder.self_attn.kv_a_proj_with_mqa)
        context=math.prepare_context(hidden,source.rotary("cuda:0"),position_ids=aligned["position_ids"],
                                     attention_mask=aligned["attention_mask"])
        expected,topk=adapter.decoder(hidden,past_key_values=None,use_cache=False,**context)
        attention,staged_topk=math.attention_forward(adapter.decoder,math.attention_inputs(adapter.decoder,hidden),context)
        residual,mlp=math.attention_finish(adapter.decoder,hidden,attention)
        actual=math.complete_moe(residual,mlp,adapter.decoder.mlp)
        if (not torch.equal(expected,actual) or not torch.equal(topk,staged_topk)
                or not torch.equal(adapter.shared_head_norm(expected),adapter.shared_head_norm(actual))):
            raise ValueError("real MTP native/staged boundary differs")
        if not torch.equal(aligned["position_ids"],torch.arange(2204,device="cuda:0").unsqueeze(0)):
            raise ValueError("MTP qualification lost the target-position alignment")
        source.close()
        report=dict(status="passed",identity=identity,source_inventory=summary,
            quantized_target_layers=[0,1,2,3],native_target_boundaries_byte_exact=True,
            natural_sparse_witness_tokens=2205,mtp_valid_rows=2204,mtp_native_boundary_byte_exact=True,
            mtp_shared_head_norm_byte_exact=True,gpu_peak_gib=torch.cuda.max_memory_allocated()/run.GIB,
            rss_gib=progress.process.memory_info().rss/run.GIB,
            note="bounded source-native numerical qualification; no full-model quality or serving measurement")
        run.atomic_json(args.output,report)
        progress.emit("qualification_passed",**{k:v for k,v in report.items()
            if k not in ("identity","gpu_peak_gib","rss_gib")})


if __name__=="__main__":
    main()
