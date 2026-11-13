"""Ten-minute, bounded target/core/expert correctness smoke (no drafter).

The parent process enforces a hard wall-clock timeout, including Python startup,
CPU oracles, source IO, compilation, encoding and serialized reconstruction.
No full-model parity or quality result is inferred from bounded layer states.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BARS = {8: 1e-4, 6: 1e-3, 4: 1e-2}
SOURCE_REVISION = '3febc62259ee92ea36e2fb6e8303371c4eb3922c'
REFERENCE_REVISION = 'ec5e7ff575beb33e54509fe01934d6a4483b5830'
REQUIRED = ('fork_cpu_oracles', 'gpu_mxfp8_bytes', 'gpu_attnres_fp64', 'gpu_sconv_fp64',
            'source_local_attention', 'source_global_attention', 'head_fp32_logits',
            'wr_du_k8', 'wr_du_k6', 'gate_attn_k8', 'gate_attn_k6',
            'unembed_k8', 'unembed_k6', 'expert_gate_k4', 'expert_up_k4', 'expert_down_k4')
PROMPTS = (
    'Explain why a causal convolution uses only the current and earlier tokens.',
    'A router selects eight experts. Describe how their weighted outputs combine.',
    'Compute the next term of the sequence 2, 4, 8, 16 and explain the rule.',
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def successful(report):
    checks = {row['name']: row for row in report.get('checks', [])}
    return all(checks.get(name, {}).get('passed') is True for name in REQUIRED)


def run_bounded(command, output, timeout_seconds, *, run=subprocess.run):
    """Hard bound; timeout/incomplete/worker failure always produces failed JSON."""
    if not 0 < timeout_seconds <= 600:
        raise ValueError('GPU pilot wall-clock budget must be in (0,600] seconds')
    output = Path(output)
    if output.exists():
        prior = json.loads(output.read_text())
        if prior.get('passed') is True and successful(prior):
            return prior  # Never restart a finished smoke.
        raise ValueError('pilot output already exists; diagnose it before choosing a fresh receipt')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log = output.with_suffix('.log')
    with log.open('wb') as stream:
        try:
            result = run(command, timeout=timeout_seconds, stdout=stream, stderr=subprocess.STDOUT, check=False)
            reason = None if result.returncode == 0 else f'worker exit {result.returncode}'
        except subprocess.TimeoutExpired:
            reason = f'wall-clock cap {timeout_seconds:g}s exceeded'
    report = json.loads(output.read_text()) if output.exists() else dict(checks=[])
    complete = successful(report)
    report.update(schema='cuteafd-inkling2-quick-pilot-v1', passed=reason is None and complete,
                  elapsed_seconds=time.monotonic() - started, cap_seconds=timeout_seconds,
                  worker_log=str(log), required_checks=list(REQUIRED), bars={str(k): v for k,v in BARS.items()})
    if reason is not None:
        report['failure'] = reason
    elif not complete:
        report['failure'] = 'worker omitted or failed a required check'
    atomic_json(output, report)
    return report


class Checks:
    def __init__(self, output, report, synchronize=lambda: None):
        self.output, self.report, self.synchronize = output, report, synchronize
    def run(self, name, function):
        started = time.monotonic()
        try:
            evidence = function() or {}
            self.synchronize()
            row = dict(name=name, passed=True, elapsed_seconds=time.monotonic() - started, **evidence)
        except Exception as error:
            row = dict(name=name, passed=False, elapsed_seconds=time.monotonic() - started,
                       error=f'{type(error).__name__}: {error}')
            self.report['checks'].append(row)
            atomic_json(self.output, self.report)
            raise
        self.report['checks'].append(row)
        atomic_json(self.output, self.report)
        return evidence


def _load_test_oracles(path):
    spec = importlib.util.spec_from_file_location('inkling2_pilot_independent_oracles', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _nmse(weight, restored, x):
    # Small heldout row count; use independent FP64 CPU evaluation rather than
    # the quantizer's Hessian metric or a hidden BF16 output rounding tolerance.
    import torch
    x, weight, restored = (v.detach().cpu().double() for v in (x,weight,restored))
    target = x @ weight.T
    error = x @ (restored-weight).T
    denominator = target.square().sum()
    if not torch.isfinite(denominator) or float(denominator) <= 0:
        raise ValueError('nonfinite or zero-energy heldout projection')
    value = float(error.square().sum()/denominator)
    if not math.isfinite(value):
        raise ValueError('nonfinite heldout NMSE')
    return value


def worker(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    from inkling2 import numerical

    if args.output.exists():
        raise ValueError('worker refuses an existing pilot receipt')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available() or torch.version.hip:
        raise ValueError('this pilot is scoped to the leased maxq NVIDIA GPU')
    torch.set_float32_matmul_precision('highest')
    report = dict(schema='cuteafd-inkling2-quick-pilot-v1', passed=False, checks=[],
                  source_revision=SOURCE_REVISION, reference_revision=REFERENCE_REVISION,
                  scope='bounded source paths and projection tiles; no full-model or drafter quality claim',
                  max_output_rows=args.output_rows, cap_seconds=args.timeout_seconds,
                  ceiling='GPU correctness smoke, at most600seconds; stop when all required bars pass',
                  state_provenance='embedding, local attention and one selected expert; not full causal target replay',
                  expert_down_provenance='native full gate/up with encoded128-row gate/up prefixes substituted; partial K4 upstream smoke')
    checks = Checks(args.output, report, torch.cuda.synchronize)
    core = numerical.fork_module('utils.inkling2_core')
    if core.REFERENCE_REVISION != REFERENCE_REVISION:
        raise ValueError('unreviewed native reference revision')
    fork_root = Path(core.__file__).resolve().parents[2]
    test_path = fork_root / 'tests/utils/test_inkling2_core.py'
    def cpu_oracles():
        result = subprocess.run([sys.executable,'-m','pytest','-q',str(test_path)],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,check=False)
        if result.returncode:
            raise ValueError('fork CPU oracles failed: ' + result.stdout.decode(errors='replace')[-4000:])
        return dict(test_file=str(test_path), summary=result.stdout.decode(errors='replace').splitlines()[-1])
    checks.run('fork_cpu_oracles',cpu_oracles)
    oracle = _load_test_oracles(test_path)
    gen = torch.Generator().manual_seed(787)
    def gpu_bytes():
        x=torch.randn(7,64,generator=gen)*40
        x[0].zero_(); x[0,1]=-0.
        expected,scales=oracle.oracle_encode(x)
        actual,actual_scales=core.mxfp8_encode(x.to(device))
        if not torch.equal(actual.cpu().view(torch.uint8),expected.view(torch.uint8)) or not torch.equal(actual_scales.cpu(),scales):
            raise ValueError('actual-device MXFP8 complete bytes differ from scalar oracle')
        return dict(shape=list(x.shape),payload_mismatches=0,scale_mismatches=0)
    checks.run('gpu_mxfp8_bytes',gpu_bytes)
    def gpu_attnres():
        values=[torch.randn(4,32,generator=gen).bfloat16() for _ in range(3)]
        query=torch.randn(3,32,generator=gen)
        actual=torch.stack(core.block_attention_residual([v.to(device) for v in values[:2]],values[2].to(device),query.to(device))).cpu().double()
        stacked=torch.stack(values).double()
        normalized=stacked/(stacked.square().mean(-1,keepdim=True)+1e-6).sqrt()
        expected=torch.stack([((normalized*q.double()/math.sqrt(32)).sum(-1).softmax(0)[...,None]*normalized).sum(0) for q in query])
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        return dict(max_abs=float((actual-expected).abs().max()))
    checks.run('gpu_attnres_fp64',gpu_attnres)
    def gpu_sconv():
        x=torch.randn(7,32,generator=gen)
        weight=torch.randn(32,1,4,generator=gen)
        actual=core.short_convolution(x.to(device),weight.to(device)).cpu().double()
        expected=x.double().clone()
        for t in range(len(x)):
            for lag in range(min(t+1,4)):
                expected[t]+=x[t-lag].double()*weight[:,0,3-lag].double()
        expected/=math.sqrt(5)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        return dict(max_abs=float((actual-expected).abs().max()))
    checks.run('gpu_sconv_fp64',gpu_sconv)

    reader=core.SourceReader(args.source)
    model=core.Inkling2Reference(reader,device=device)
    tokenizer=Tokenizer.from_file(str(args.source/'tokenizer.json'))
    tokens=[torch.tensor(tokenizer.encode(p,add_special_tokens=False).ids[:args.max_tokens],device=device) for p in PROMPTS]
    if any(len(t)<2 for t in tokens):
        raise ValueError('pilot prompts tokenize to fewer than2tokens')
    report['prompt_token_counts']=[len(t) for t in tokens]
    captures={'gate_attn':[], 'wr_du':[], 'unembed':[], 'expert_x':[], 'expert_gamma':[]}
    global_states=[]
    def source_local():
        for ids in tokens:
            state=model.start(ids)
            inputs=model.attention_inputs(state,0)
            captures['gate_attn'].append(inputs['model.llm.layers.0.attn.gate_attn.weight'].detach())
            attention,_=model.attention(state,0)
            pending,x=model.attention_finish(state,attention,0)
            routed,gamma,_=model.route(0,x)
            global_states.append((pending,x,routed,gamma))
        return dict(layer=0,local=True,prompt_count=len(tokens))
    checks.run('source_local_attention',source_local)
    common = set.intersection(*(set(r.flatten().tolist()) for _,_,r,_ in global_states))
    if not common:
        raise ValueError('no naturally routed expert occurs in both calibration and heldout prompts')
    routed_all=torch.cat([r.flatten() for _,_,r,_ in global_states])
    expert=max(sorted(common),key=lambda value:int((routed_all==value).sum()))
    report['expert_index']=expert
    report['expert_rows_per_prompt']=[]
    for _,x,routed,gamma in global_states:
        token,rank=torch.where(routed==expert)
        captures['expert_x'].append(x[token].detach())
        captures['expert_gamma'].append(gamma[token,rank].detach())
        report['expert_rows_per_prompt'].append(len(token))
    expert_prefix='model.llm.layers.0.mlp.experts.'
    gate_weight=reader.expert(expert_prefix+'w13_weight',expert,projection='gate',device=device).bfloat16()
    up_weight=reader.expert(expert_prefix+'w13_weight',expert,projection='up',device=device).bfloat16()
    down_weight=reader.expert(expert_prefix+'w2_weight',expert,projection='down',device=device).bfloat16()
    def source_global():
        for pending,x,routed,gamma in global_states:
            # Single chosen expert is a bounded path stimulus, not a full MoE.
            qx=core.mxfp8_qdq(x)
            gate=F.linear(qx,gate_weight); up=F.linear(qx,up_weight)
            fused=torch.stack((gate,up),-1).flatten(-2)
            selected_gamma=torch.where(routed==expert,gamma,0).sum(-1)
            hidden=core.mxfp8_qdq(core.capped_swiglu(fused,selected_gamma))
            selected=F.linear(hidden,down_weight)
            state=model.mlp_finish(pending,selected)
            inputs=model.attention_inputs(state,5)
            captures['wr_du'].append(inputs['model.llm.layers.5.attn.wr_du.weight'].detach())
            attention,_=model.attention(state,5)
            # Head smoke uses normalized target output mixture over this bounded
            # state; no omitted intermediate layers are represented as executed.
            captures['unembed'].append(model.finish(core.AttnResState(attention.to(state.partial.dtype),state.blocks)).detach())
        return dict(layer=5,local=False,state_provenance=report['state_provenance'])
    checks.run('source_global_attention',source_global)
    def head_fp32():
        weight=reader.tensor('model.llm.unembed.weight',rows=slice(0,args.output_rows),device=device)
        x=captures['unembed'][-1]
        actual=model.project('model.llm.unembed.weight',x,weight=weight,output_dtype=torch.float32)
        expected=(x.detach().cpu().double() @ weight.detach().cpu().double().T)
        torch.testing.assert_close(actual.cpu().double(),expected,rtol=2e-3,atol=2e-3)
        if actual.dtype!=torch.float32:
            raise ValueError('native unembed output did not preserve FP32')
        capped=(actual/30).tanh()*30
        if not torch.isfinite(capped).all():
            raise ValueError('nonfinite FP32 head softcap')
        return dict(shape=list(actual.shape),dtype=str(actual.dtype),max_abs=float((actual.cpu().double()-expected).abs().max()))
    checks.run('head_fp32_logits',head_fp32)

    def projection(label,weight,xs,bits,directory):
        train=torch.cat(xs[:2]).contiguous()
        heldout=xs[2].contiguous()
        matrix=weight[:args.output_rows].contiguous()
        raw=dict(H=train.float().T @ train.float(),count=len(train),finalized=False,identity_rows=0)
        packed,metrics,evidence=numerical.quantize_projection(matrix,raw,bits=bits,seed=787,sigma=.025,device=str(device))
        filename=directory/f'{label}-k{bits}.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in packed.items()},filename)
        serialized=load_file(filename)
        restored=numerical.reconstruct_projection(serialized,list(matrix.shape),device=str(device),dtype=torch.bfloat16)
        error=_nmse(matrix,restored,heldout)
        if error>BARS[bits]:
            raise ValueError(f'{label} K{bits} heldout NMSE {error:.9g} exceeds {BARS[bits]:.9g}')
        return dict(heldout_nmse=error,bar=BARS[bits],shape=list(matrix.shape),train_rows=len(train),heldout_rows=len(heldout),
                    serialized_bytes=filename.stat().st_size,identity_rows=0,
                    quantizer_path=metrics.get('quantizer_path'),hessian_rows=evidence.get('hessian_rows')),restored

    # The task owns this staging. It is removed when the bounded worker exits;
    # no source checkpoint or production numerical state is modified/reused.
    with tempfile.TemporaryDirectory(prefix='inkling2-pilot-',dir=args.output.parent) as temporary:
        directory=Path(temporary)
        for label,key in (('wr_du','model.llm.layers.5.attn.wr_du.weight'),('gate_attn','model.llm.layers.0.attn.gate_attn.weight'),('unembed','model.llm.unembed.weight')):
            weight=reader.tensor(key,rows=slice(0,args.output_rows),device=device)
            for bits in (8,6):
                checks.run(f'{label}_k{bits}',lambda label=label,weight=weight,bits=bits: projection(label,weight,captures[label],bits,directory)[0])
        xs=[core.mxfp8_qdq(x) for x in captures['expert_x']]
        recon={}
        for label,weight in (('expert_gate',gate_weight),('expert_up',up_weight)):
            def encode(label=label,weight=weight):
                evidence,restored=projection(label,weight,xs,4,directory)
                recon[label]=restored
                return evidence
            checks.run(label+'_k4',encode)
        down_x=[]
        for x,gamma in zip(xs,captures['expert_gamma']):
            gate=F.linear(x,gate_weight);up=F.linear(x,up_weight)
            gate[:,:args.output_rows]=F.linear(x,recon['expert_gate'])
            up[:,:args.output_rows]=F.linear(x,recon['expert_up'])
            fused=torch.stack((gate,up),-1).flatten(-2)
            down_x.append(core.mxfp8_qdq(core.capped_swiglu(fused,gamma)))
        checks.run('expert_down_k4',lambda: projection('expert_down',down_weight,down_x,4,directory)[0])
    report['passed']=successful(report)
    atomic_json(args.output,report)
    return 0 if report['passed'] else 1


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--timeout-seconds',type=float,default=600)
    p.add_argument('--output-rows',type=int,default=128)
    p.add_argument('--max-tokens',type=int,default=32)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.output_rows<128 or args.output_rows%128 or not 2<=args.max_tokens<=64:
        raise ValueError('pilot requires >=128 output rows (multiple128), and2..64tokens/prompt')
    if not 0<args.timeout_seconds<=600:
        raise ValueError('GPU pilot cap must not exceed600seconds')
    args.source=args.source.resolve();args.output=args.output.resolve()
    if args.worker:
        try:
            return worker(args)
        except Exception as error:
            report=json.loads(args.output.read_text()) if args.output.exists() else dict(checks=[])
            report.update(passed=False,failure=f'{type(error).__name__}: {error}')
            atomic_json(args.output,report)
            return 1
    command=[sys.executable,'-m','inkling2.pilot','--source',str(args.source),'--output',str(args.output),
             '--device',args.device,'--timeout-seconds',str(args.timeout_seconds),'--output-rows',str(args.output_rows),
             '--max-tokens',str(args.max_tokens),'--worker']
    report=run_bounded(command,args.output,args.timeout_seconds)
    print(json.dumps(dict(passed=report['passed'],receipt=str(args.output),failure=report.get('failure'))))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
