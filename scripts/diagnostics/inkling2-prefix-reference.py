#!/usr/bin/env python3
"""Matching #1915 partial-prefix logits for the native Inkling golden.

Not a full-model reference publication. Layer0..last run the pinned independent
F32 oracle unchanged; the final reference head is applied to that prefix using
its completed six-layer blocks. Intended for cheap one-GPU stage2 qualification.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python/reference'))

import torch
import torch.nn.functional as F
from families.inkling2.model import Config, Weights, TEXT, REVISION, rms_norm, layer_forward, output_norm, head_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--last-layer', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error('reference requires pinned official MXFP8 snapshot')
    cfg = Config.from_dict(json.loads((args.snapshot/'config.json').read_bytes()))
    tokens = json.loads(args.tokens.read_bytes())
    if (not 0 <= args.last_layer < cfg.num_hidden_layers or not 2 <= len(tokens) <= 4096
            or any(type(t) is not int or not 0 <= t < cfg.unpadded_vocab_size for t in tokens)):
        parser.error('invalid prefix/tokens')
    if args.out.exists():
        parser.error('output directory must be new')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    weights = Weights(args.snapshot, 'cuda')
    weights.validate(cfg)
    with torch.inference_mode():
        from safetensors import safe_open
        with safe_open(str(args.snapshot/weights.index[TEXT+'embed.weight']), framework='pt', device='cpu') as stream:
            table = stream.get_slice(TEXT+'embed.weight')
            embedded = torch.cat([table[t:t+1] for t in tokens[:-1]]).cuda().float()
        hidden = rms_norm(embedded, weights.tensor(TEXT+'embed_norm.weight'), cfg.rms_norm_eps)
        del embedded
        blocks = []
        for index in range(args.last_layer+1):
            layer = weights.load_layer(index, cfg)
            result = layer_forward(hidden, blocks, layer, cfg, index)
            if index % cfg.n_layers_per_attn_res_block == 0:
                blocks.append(hidden)
            hidden = result
            del layer
            torch.cuda.empty_cache()
            print(f'Inkling partial reference layer{index} complete', flush=True)
        hidden = output_norm(hidden, blocks, weights.tensor(TEXT+'attn_res.output_query'),
                             weights.tensor(TEXT+'norm.weight'), cfg)
        head = weights.tensor(TEXT+'unembed.weight')
        logits = head_forward(hidden, head, cfg)
        nll = F.cross_entropy(logits, torch.tensor(tokens[1:], device='cuda')).item()
        args.out.mkdir(parents=True)
        logits.float().cpu().numpy().tofile(args.out/'logits.bin')
        (args.out/'tokens.bin').write_bytes(b''.join(int(t).to_bytes(4,'little') for t in tokens))
        (args.out/'result.json').write_text(json.dumps(dict(partial=True, last_layer=args.last_layer,
            mean_nll=nll, vocab=cfg.unpadded_vocab_size, rows=len(tokens)-1,
            oracle='python/reference/families/inkling2/model.py #1915'), indent=2)+'\n')
        print(f'Inkling partial reference mean NLL={nll:.8f}', flush=True)


if __name__=='__main__':
    main()
