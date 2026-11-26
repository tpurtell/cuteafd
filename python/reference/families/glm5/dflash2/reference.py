#!/usr/bin/env python3
"""DFlash2 drafters for GLM 5.3 (incoai/GLM-5.3-DFlash2) and GLM 5.3 Flash
(incoai/GLM-5.3-Flash-DFlash2) in plain PyTorch.

The oracle the engine's drafter is compared with. Semantics are those of the
upstream DFlash2DraftModel as glmrt reproduced them bit for bit
(python/reference/glmrt_reference/{dspark_update,dspark_body,dflash_head}_capture.py
in ../glmrt-release), written as BF16 torch statements:

  context  fused = RMSNorm(fc(concat(target layer outputs, target_layer_ids)))
           (GLM 5.3 Flash: each output is the BF16 mean of the four mHC
           streams, the golden's layerNN.bin [T, 4, hidden])
           per draft layer: K = rope(k_norm(k_proj(fused))), V = v_proj(fused)
  block    [anchor, mask x 7] embeddings at positions p..p+7; per layer:
           n = RMSNorm(h); dyn = attention_conv.kernel_projection(n)
           qkv(conv0(n)) -> q/k norm, rope; non-causal attention over the
           last <= 2048 context entries and the whole block (with
           --row-window, the Flash default, block row r at p + r sees only
           context positions q with p + r - q < 2048, upstream's sliding
           window); o_proj;
           h += conv1(o); n = RMSNorm(h); the MLP likewise with mlp_conv.
           conv_s(x)[r] = base[s,0]*x[r] + dyn[r,s,0]*x[r] + base[s,1]*x[r-1]
           + dyn[r,s,1]*x[r-1] (per 16-channel group, x[-1] = 0 in the block),
           each term accumulated in BF16 as upstream does.
  head     logits = lm_head(norm(h))[1:8]; top-16 per row; the selector walks
           the rows greedily: score_k = unary_k + ((pred_cb[prev] * proj(h)) .
           succ_cb[cand_k]) in BF16, prev = anchor, then each chosen token.

Writes, for every anchor position p in --positions (context = golden taps of
rows 0..p-1, anchor = tokens[p]):
  drafts.bin    u32 [N, 7]
  features.bin  f32 [N, 7, 4]  margin, best probability, entropy, rank among the 16
  hidden.bin    bf16 [N, 8, hidden] final-norm output
  meta.json

  reference.py --draft SNAP --target SNAP --golden DIR --positions 64:1500:16 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-5


def rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Qwen3RMSNorm: the unit-RMS value is rounded to BF16 before the weight."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """rotate_half RoPE on [n, heads, dim] with BF16 cos/sin, as transformers."""
    dim = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.int64).float() / dim))
    freqs = positions.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]
    half = dim // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin


def conv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, side: int, group: int) -> torch.Tensor:
    """Two-tap grouped dynamic convolution over block rows ([B, H], row 0 has no predecessor)."""
    groups = x.shape[-1] // group
    d0 = dyn[:, side * 2 * groups:side * 2 * groups + groups].repeat_interleave(group, -1)
    d1 = dyn[:, side * 2 * groups + groups:(side + 1) * 2 * groups].repeat_interleave(group, -1)
    prev = torch.zeros_like(x)
    prev[1:] = x[:-1]
    out = base[side, 0] * x
    out = torch.addcmul(out, d0, x)
    out = out + base[side, 1] * prev
    return torch.addcmul(out, d1, prev)


class Drafter:
    def __init__(self, draft: Path, target: Path, device: str):
        self.cfg = json.loads((draft / "config.json").read_text())
        dc = self.cfg["dflash_config"]
        self.block, self.mask, self.taps = dc["block_size"], dc["mask_token_id"], dc["target_layer_ids"]
        self.group, self.topk = dc["conv_group_size"], dc["selector_top_k"]
        self.heads, self.kv_heads = self.cfg["num_attention_heads"], self.cfg["num_key_value_heads"]
        self.head_dim, self.theta = self.cfg["head_dim"], float(self.cfg["rope_parameters"]["rope_theta"])
        self.window, self.layers = self.cfg["sliding_window"], self.cfg["num_hidden_layers"]
        self.device = device
        self.eps = self.cfg["rms_norm_eps"]
        self.post_softcap = dc.get("final_logit_softcapping") if dc.get("selector_softcap_mode") == "after_correction" else None
        f = safe_open(str(draft / "model.safetensors"), framework="pt", device=device)
        self.w = {k: f.get_tensor(k) for k in f.keys()}
        index = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
        self.target_files = {}

        def target_tensor(name: str) -> torch.Tensor:
            shard = index[name]
            if shard not in self.target_files:
                self.target_files[shard] = safe_open(str(target / shard), framework="pt", device="cpu")
            return self.target_files[shard].get_tensor(name)

        target_cfg = json.loads((target / "config.json").read_text())
        self.mhc = target_cfg.get("model_type") == "glm5_next"
        prefix = "model.language_model." if self.mhc else "model."
        self.inkling = target_cfg.get("model_type") == "inkling2_loop"
        if self.inkling:
            self.embed = target_tensor("model.llm.embed.weight")
            self.lm_head = target_tensor("model.llm.unembed.weight").to(device)
            self.valid_vocab = target_cfg["text_config"]["unpadded_vocab_size"]
        else:
            self.embed = target_tensor(prefix + "embed_tokens.weight")
            self.lm_head = target_tensor("lm_head.weight").to(device)
            self.valid_vocab = self.cfg["vocab_size"]

    def dump(self, name, value):
        if getattr(self, "dump_dir", None) is not None:
            value = value.detach().contiguous().cpu()
            if value.dtype == torch.bfloat16: value = value.view(torch.int16)
            value.numpy().tofile(self.dump_dir / (name + ".bin"))

    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer context K (roped) and V for tap rows [n, 6 * hidden]."""
        fused = rms(F.linear(taps, self.w["fc.weight"]), self.w["hidden_norm.weight"], self.eps)
        self.dump("context-norm", fused)
        out = []
        for l in range(self.layers):
            p = f"layers.{l}.self_attn."
            k = F.linear(fused, self.w[p + "k_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            v = F.linear(fused, self.w[p + "v_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            self.dump(f"context-{l}-kv", torch.cat([k.flatten(1), v.flatten(1)], dim=-1))
            out.append((rope(rms(k, self.w[p + "k_norm.weight"], self.eps), positions, self.theta), v))
        return out

    def body(self, anchor: int, position: int, ctx: list[tuple[torch.Tensor, torch.Tensor]],
             row_window: bool = False) -> torch.Tensor:
        """Final-norm output [block, hidden] of one block at `position` (the
        context holds positions position - len(ctx) .. position - 1)."""
        mask = None
        if row_window:
            n = ctx[0][0].shape[0]
            q = torch.arange(position, position + self.block, device=self.device)[:, None]
            k = torch.cat([torch.arange(position - n, position, device=self.device),
                           torch.arange(position, position + self.block, device=self.device)])[None]
            mask = (q - k).abs() < self.window
        tokens = torch.tensor([anchor] + [self.mask] * (self.block - 1))
        h = self.embed[tokens].to(self.device)
        positions = torch.arange(position, position + self.block, device=self.device)
        self.dump("input", h)
        n = rms(h, self.w["layers.0.input_layernorm.weight"], self.eps)
        self.dump("input-norm", n)
        for l in range(self.layers):
            p = f"layers.{l}."
            dyn = F.linear(n, self.w[p + "attention_conv.kernel_projection.weight"])
            base = self.w[p + "attention_conv.base_kernel"]
            x = conv(n, dyn, base, 0, self.group)
            self.dump(f"layer-{l}-attn-dyn", dyn)
            self.dump(f"layer-{l}-attn-conv", x)
            a = p + "self_attn."
            q = F.linear(x, self.w[a + "q_proj.weight"]).view(self.block, self.heads, self.head_dim)
            k = F.linear(x, self.w[a + "k_proj.weight"]).view(self.block, self.kv_heads, self.head_dim)
            v = F.linear(x, self.w[a + "v_proj.weight"]).view(self.block, self.kv_heads, self.head_dim)
            self.dump(f"layer-{l}-qkv", torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1))
            q = rope(rms(q, self.w[a + "q_norm.weight"], self.eps), positions, self.theta)
            k = rope(rms(k, self.w[a + "k_norm.weight"], self.eps), positions, self.theta)
            self.dump(f"layer-{l}-q-rope", q)
            self.dump(f"layer-{l}-k-rope", k)
            kc, vc = ctx[l]
            keys, values = torch.cat([kc, k]), torch.cat([vc, v])
            attn = F.scaled_dot_product_attention(q.transpose(0, 1)[None], keys.transpose(0, 1)[None],
                values.transpose(0, 1)[None], attn_mask=mask, is_causal=False, enable_gqa=True,
                scale=1.0 / math.sqrt(self.head_dim))
            attn = attn[0].transpose(0, 1).reshape(self.block, -1)
            self.dump(f"layer-{l}-attn", attn)
            o = F.linear(attn, self.w[a + "o_proj.weight"])
            self.dump(f"layer-{l}-attn-out", o)
            h = h + conv(o, dyn, base, 1, self.group)
            n = rms(h, self.w[p + "post_attention_layernorm.weight"], self.eps)
            self.dump(f"layer-{l}-post-attn", h)
            self.dump(f"layer-{l}-post-attn-norm", n)
            dyn = F.linear(n, self.w[p + "mlp_conv.kernel_projection.weight"])
            base = self.w[p + "mlp_conv.base_kernel"]
            x = conv(n, dyn, base, 0, self.group)
            m = p + "mlp."
            gate, up = F.linear(x, self.w[m + "gate_proj.weight"]), F.linear(x, self.w[m + "up_proj.weight"])
            self.dump(f"layer-{l}-gate-up", torch.cat([gate, up], dim=-1))
            act = F.silu(gate) * up
            self.dump(f"layer-{l}-act", act)
            d = F.linear(act, self.w[m + "down_proj.weight"])
            h = h + conv(d, dyn, base, 1, self.group)
            last = l + 1 == self.layers
            n = rms(h, self.w["norm.weight" if last else f"layers.{l + 1}.input_layernorm.weight"], self.eps)
            self.dump(f"layer-{l}", h)
            self.dump(f"layer-{l}-norm", n)
        return n

    def select(self, anchor: int, hidden: torch.Tensor) -> tuple[list[int], list[list[float]]]:
        """Greedy candidate path over rows 1.. of the block."""
        rows = hidden[1:]
        logits = F.linear(rows, self.lm_head[:self.valid_vocab])
        self.dump("logits", F.linear(hidden, self.lm_head).float())
        unary, candidates = torch.topk(logits, self.topk, dim=-1)
        projected = F.linear(rows, self.w["candidate_selector.hidden_projection.weight"])
        self.dump("unary", unary.float())
        self.dump("candidates", candidates.int())
        self.dump("projected", F.linear(hidden, self.w["candidate_selector.hidden_projection.weight"]))
        pred_cb = self.w["candidate_selector.predecessor_codebook"]
        succ_cb = self.w["candidate_selector.successor_codebook"]
        previous, tokens, features = anchor, [], []
        for i in range(rows.shape[0]):
            conditioned = pred_cb[previous] * projected[i]
            transition = torch.einsum("r,kr->k", conditioned, succ_cb[candidates[i]])
            scores = unary[i] + transition
            if self.post_softcap is not None:
                scores = self.post_softcap * torch.tanh(scores / self.post_softcap)
            scores = scores.float()
            best = scores.max()
            index = int((scores == best).nonzero()[0])
            runner = torch.where(torch.arange(self.topk, device=scores.device) == index,
                torch.tensor(-math.inf, device=scores.device), scores).max()
            shifted = scores - best
            mass = shifted.exp()
            total = mass.sum()
            entropy = total.log() - (mass * shifted).sum() / total
            features.append([float(best - runner), float(1.0 / total), float(entropy), float(index)])
            previous = int(candidates[i, index])
            tokens.append(previous)
        return tokens, features


def mini_fixture(draft: Path, target: Path, out: Path, device: str):
    """Real checkpoint oracle on deterministic, fixed BF16 context inputs."""
    model = Drafter(draft, target, device)
    assert model.inkling
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    with torch.inference_mode():
        rows = 1031
        width = len(model.taps) * model.cfg["hidden_size"]
        taps = ((torch.arange(rows * width, device=device, dtype=torch.int64) * 17 % 251).float() / 128 - 1).reshape(rows, width).bfloat16()
        taps.cpu().view(torch.int16).numpy().tofile(out / "taps.bin")
        model.dump_dir = out / "ref"
        model.dump_dir.mkdir(exist_ok=True)
        ctx = model.context(taps, torch.arange(rows, device=device))
        result = []
        for anchor in (101, 501, 1013, 200006):
            h = model.body(anchor, rows, ctx, True)
            tokens, features = model.select(anchor, h)
            result.append(dict(anchor=anchor, tokens=tokens, features=features))
            model.dump("features", torch.tensor(features))
            model.dump_dir = None
        (out / "meta.json").write_text(json.dumps(dict(rows=rows, results=result)))


def parse_positions(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b, *s = (int(x) for x in part.split(":"))
            out.extend(range(a, b, s[0] if s else 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--mini-fixture", action="store_true")
    parser.add_argument("--positions", help="anchor positions, e.g. 64:1500:16,1510")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-window", type=int, choices=(0, 1), help="per-row sliding window (default: 1 for a "
                        "glm5_next target, else 0)")
    args = parser.parse_args()
    if args.mini_fixture:
        return mini_fixture(args.draft, args.target, args.out, args.device)
    if args.golden is None or args.positions is None:
        parser.error("--golden and --positions are required without --mini-fixture")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    drafter = Drafter(args.draft, args.target, args.device)
    tokens = torch.frombuffer(bytearray((args.golden / "tokens.bin").read_bytes()), dtype=torch.int32)
    hidden_size = drafter.cfg["hidden_size"]
    taps = []
    for layer in drafter.taps:
        raw = torch.frombuffer(bytearray((args.golden / f"layer{layer:02d}.bin").read_bytes()), dtype=torch.bfloat16)
        if drafter.mhc:
            # FP32 sum of the four streams in order, then the division, rounded to BF16.
            streams = raw.view(-1, 4, hidden_size).float()
            raw = (((streams[:, 0] + streams[:, 1]) + streams[:, 2]) + streams[:, 3]) / 4
            raw = raw.to(torch.bfloat16)
        taps.append(raw.view(-1, hidden_size))
    taps = torch.cat(taps, -1).to(args.device)
    positions = parse_positions(args.positions)
    rows = torch.arange(taps.shape[0], device=args.device)
    row_window = bool((drafter.mhc or drafter.inkling) if args.row_window is None else args.row_window)
    with torch.no_grad():
        ctx = drafter.context(taps, rows)
        drafts, features, hidden = [], [], []
        for p in positions:
            first = max(0, p - drafter.window)
            window = [(k[first:p], v[first:p]) for k, v in ctx]
            h = drafter.body(int(tokens[p]), p, window, row_window)
            t, f = drafter.select(int(tokens[p]), h)
            drafts.append(t)
            features.append(f)
            hidden.append(h.cpu())
    args.out.mkdir(parents=True, exist_ok=True)
    torch.tensor(drafts, dtype=torch.int32).numpy().tofile(args.out / "drafts.bin")
    torch.tensor(features, dtype=torch.float32).numpy().tofile(args.out / "features.bin")
    torch.stack(hidden).view(torch.int16).numpy().tofile(args.out / "hidden.bin")
    truth = [[int(tokens[p + 1 + i]) if p + 1 + i < len(tokens) else -1 for i in range(drafter.block - 1)]
             for p in positions]
    accepted = [next((i for i, (d, t) in enumerate(zip(ds, ts)) if d != t), len(ds)) for ds, ts in zip(drafts, truth)]
    (args.out / "meta.json").write_text(json.dumps({"positions": positions, "accepted_vs_tokens": accepted,
        "mean_accepted": sum(accepted) / len(accepted)}, indent=1))
    print(f"{len(positions)} anchors, mean accepted prefix vs golden tokens {sum(accepted) / len(accepted):.2f}")


if __name__ == "__main__":
    main()
