#!/usr/bin/env python3
"""One layer-major independent F32/vendor-expert-boundary reference pass.

#1915 F32 arithmetic remains unchanged. Vendor mode changes only the explicit
BF16 expert boundaries documented in serving_boundary.py. Multiple prefixes
share each native layer load. Synthetic token panels are diagnostics, never
fidelity publications. All states remain independent across modes.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python/reference'))
import torch
import torch.nn.functional as F
from families.inkling2 import model as f32
from families.inkling2 import serving_boundary as vendor
from families.inkling2 import native_boundary as native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--last-layer', type=int, default=47)
    parser.add_argument('--prefixes', default='')
    parser.add_argument('--modes', default='f32,vendor,native')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tap-layers', default='')
    args = parser.parse_args()
    if args.snapshot.name != f32.REVISION:
        parser.error('reference requires pinned official MXFP8 snapshot')
    cfg = f32.Config.from_dict(json.loads((args.snapshot/'config.json').read_bytes()))
    tokens = json.loads(args.tokens.read_bytes())
    modes = args.modes.split(',')
    tap_layers = {int(x) for x in args.tap_layers.split(',') if x}
    prefixes = sorted(set([args.last_layer] + [int(x)-1 for x in args.prefixes.split(',') if x]))
    if (not modes or len(modes)!=len(set(modes)) or any(m not in ('f32','vendor','native') for m in modes)
            or any(not 0<=p<=args.last_layer<cfg.num_hidden_layers for p in prefixes)
            or not 2<=len(tokens)<=4096 or any(type(t) is not int or not 0<=t<cfg.unpadded_vocab_size for t in tokens)):
        parser.error('invalid modes/prefix/tokens')
    if args.out.exists():
        parser.error('output directory must be new')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    weights = f32.Weights(args.snapshot, 'cuda')
    weights.validate(cfg)
    args.out.mkdir(parents=True)
    with torch.inference_mode():
        from safetensors import safe_open
        with safe_open(str(args.snapshot/weights.index[f32.TEXT+'embed.weight']), framework='pt', device='cpu') as stream:
            table = stream.get_slice(f32.TEXT+'embed.weight')
            embedded = torch.cat([table[t:t+1] for t in tokens[:-1]]).cuda().float()
        initial = f32.rms_norm(embedded, weights.tensor(f32.TEXT+'embed_norm.weight'), cfg.rms_norm_eps)
        del embedded
        states = {mode:(initial.clone(),[]) for mode in modes}
        del initial
        # Retained globals belong to one reference-only source, never the engine.
        query = weights.tensor(f32.TEXT+'attn_res.output_query')
        norm = weights.tensor(f32.TEXT+'norm.weight')
        head = weights.tensor(f32.TEXT+'unembed.weight')
        weights.close()
        for index in range(args.last_layer+1):
            layer = weights.load_layer(index,cfg)
            for mode in modes:
                hidden,blocks = states[mode]
                taps = {} if mode=='native' and index in tap_layers else None
                if mode=='native':
                    result=native.layer_forward(hidden,blocks,layer,cfg,index,taps=taps)
                else:
                    result={'f32':f32.layer_forward,'vendor':vendor.layer_forward}[mode](hidden,blocks,layer,cfg,index)
                if taps is not None:
                    folder=args.out/mode/f'layer-{index}'
                    folder.mkdir(parents=True)
                    for name,value in taps.items():
                        value.cpu().view(torch.uint16).numpy().tofile(folder/f'{name}.bin') if value.dtype==torch.bfloat16 else value.cpu().numpy().tofile(folder/f'{name}.bin')
                    (folder/'meta.json').write_text(json.dumps(dict(rows=len(tokens)-1,hidden=cfg.hidden_size,
                        experts=cfg.n_routed_experts,topk=cfg.num_experts_per_tok,local=index in cfg.local_layer_ids)))
                    del taps
                if index%cfg.n_layers_per_attn_res_block==0:
                    blocks.append(hidden)
                states[mode]=(result,blocks)
                if index in prefixes:
                    normalized=f32.output_norm(result,blocks,query,norm,cfg)
                    logits=f32.head_forward(native.narrow(normalized) if mode=='native' else normalized,head,cfg)
                    nll=F.cross_entropy(logits,torch.tensor(tokens[1:],device='cuda')).item()
                    out=args.out/mode/f'prefix-{index+1}'
                    out.mkdir(parents=True)
                    logits.float().cpu().numpy().tofile(out/'logits.bin')
                    (out/'tokens.bin').write_bytes(b''.join(int(t).to_bytes(4,'little') for t in tokens))
                    (out/'result.json').write_text(json.dumps(dict(partial=vendor.is_partial(index,cfg.num_hidden_layers),
                        last_layer=index,mean_nll=nll,vocab=cfg.unpadded_vocab_size,rows=len(tokens)-1,
                        mode=mode,oracle={'f32':'independent #1915','vendor':'pinned vLLM BF16 expert boundaries',
                                          'native':'independent audited native BF16/F32 boundaries'}[mode]),indent=2)+'\n')
                    print(f'Inkling {mode} prefix{index+1} mean NLL={nll:.8f}',flush=True)
                    del normalized,logits
            del layer
            torch.cuda.empty_cache()
            print(f'Inkling reference layer{index} complete modes={modes}',flush=True)


if __name__=='__main__':
    main()
