#!/usr/bin/env python3
"""Mistral Large 4's bundled ``mtp_dspark`` drafter in plain PyTorch: the oracle
the engine's drafter is compared with.

Semantics are vLLM's MistralLarge4DSparkModel and DSparkSpeculator (bd4fa263,
``vllm/models/mistral_large4/nvidia/dspark.py``, ``.../spec_decode/dspark``),
whose config comes from ``params.json:mtp`` (``get_mistral_large_4_dspark_config``),
written as BF16 torch statements over the shared reference's operations
(``ops.py``):

  taps     ``target_activation_layers`` (1, 18, 36, 54, 71) are 0-based target
           layer outputs (vLLM aux ids + 1, each the residual stream entering
           the next layer; 71's is the stream entering the final norm), the
           engine golden's ``layerNN.bin``
  context  ``ctx = target_norm(target_projection(cat(taps)))``; each draft layer
           keeps the latent of every context row, ``kv_a_norm(wkv_a(ctx)[:512])``
           and ``rope(wkv_a(ctx)[512:])`` (no input norm, as in training); a
           block at anchor position p sees context positions p - 2048 .. p - 1
  block    ``[anchor, mask_1, mask_2]`` (position-specific masks: slot k reads
           ``mask_embedding[k - 1]``) at positions p .. p + 2; three dense MLA
           layers (the target's geometry: q 64 x (64 nope + 64 rope), latent
           512, v 128; no q-LoRA, no SSMax, no indexer), non-causal over the
           context window and the whole block, interleaved RoPE with the
           target's YaRN but without LeRoPE; SwiGLU FFN (24576); final norm
  heads    ``sample_from_anchor``: block row k predicts token p + k + 1.
           base = tanh(head(h_k) / 30) * 30 (the target's head and cap);
           RNN Markov: state 0; per row ``joint = joint_proj(cat(state,
           embedding[prev], h_k))`` -> gate, candidate, output; ``state = g *
           state + (1 - g) * tanh(candidate)``; logits = base + output(tanh(o))
           (the bias is not soft-capped); greedy argmax (lowest index on ties);
           prev is the anchor for k = 0, else draft k - 1.

  dspark.py --snapshot SNAP --golden DIR --positions 64:1500:16 --out DIR

writes, for every anchor position p (context = golden taps of rows 0..p-1,
anchor = tokens[p]):  drafts.bin u32 [N, 3], hidden.bin bf16 [N, 3, hidden],
meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from families.mistral_large4 import ops  # noqa: E402

PREFIX = "mtp_dspark."
TAPS = (1, 18, 36, 54, 71)


class Drafter:
    """The drafter over a weight getter ``w(name) -> tensor`` (checkpoint names
    without the ``mtp_dspark.`` prefix, plus ``embed`` / ``head`` for the
    target's embedding and vocabulary head)."""

    def __init__(self, w, *, layers=3, depth=3, window=2048, heads=64, nope=64, rope=64, latent=512,
                 v_head=128, eps=1e-5, softcap=30.0, inv_freq=None):
        self.w, self.layers, self.depth, self.window = w, layers, depth, window
        self.heads, self.nope, self.rope, self.latent, self.v_head = heads, nope, rope, latent, v_head
        self.eps, self.softcap = eps, softcap
        self.scale = (nope + rope) ** -0.5
        self.inv_freq = ops.yarn_lerope_inv_freq(alpha=None) if inv_freq is None else inv_freq

    def _cos_sin(self, positions):
        return ops.cos_sin(positions, self.inv_freq)

    def context(self, taps: torch.Tensor, positions: torch.Tensor):
        """Per draft layer (normed latent [n, 512], rotated key [n, 64]) of the
        context rows; ``taps`` [n, 5 * hidden] BF16."""
        ctx = ops.rms_norm(F.linear(taps, self.w("backbone.target_projection.weight")),
                           self.w("backbone.target_norm.weight"), self.eps)
        return [self._latent(ctx, positions, i) for i in range(self.layers)]

    def _latent(self, x, positions, i):
        a = f"backbone.layers.{i}.attention."
        cos, sin = self._cos_sin(positions)
        kv = F.linear(x, self.w(a + "wkv_a_with_mqa.weight"))
        # ops.mla_latent at this drafter's latent width (512 in the checkpoint).
        latent = ops.rms_norm(kv[..., :self.latent], self.w(a + "kv_a_norm.weight"), self.eps)
        return latent, ops.rope_interleave(kv[..., self.latent:][:, None, :], cos, sin)[:, 0]

    def _attention(self, x, positions, i, ctx):
        a = f"backbone.layers.{i}.attention."
        rows = x.shape[0]
        cos, sin = self._cos_sin(positions)
        q = F.linear(x, self.w(a + "wq.weight")).view(rows, self.heads, self.nope + self.rope)
        q = torch.cat([q[..., :self.nope], ops.rope_interleave(q[..., self.nope:], cos, sin)], -1)
        latent, k_rot = self._latent(x, positions, i)
        keys = torch.cat([ctx[0], latent]), torch.cat([ctx[1], k_rot])
        kv = F.linear(keys[0], self.w(a + "wkv_b.weight")).view(-1, self.heads, self.nope + self.v_head)
        k = torch.cat([kv[..., :self.nope], keys[1][:, None, :].expand(-1, self.heads, -1)], -1)
        v = kv[..., self.nope:]
        # Official eager attention: BF16 matmuls, F32 softmax cast back; non-causal.
        scores = torch.matmul(q.transpose(0, 1), k.permute(1, 2, 0)) * self.scale
        probs = scores.softmax(-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(rows, -1)
        return F.linear(out, self.w(a + "wo.weight"))

    def _ffn(self, x, i):
        f = f"backbone.layers.{i}.feed_forward."
        return F.linear(F.silu(F.linear(x, self.w(f + "w1.weight"))) * F.linear(x, self.w(f + "w3.weight")),
                        self.w(f + "w2.weight"))

    def window_of(self, ctx, position: int):
        """The context rows a block at ``position`` sees (rows indexed by position)."""
        start = max(position - self.window, 0)
        return [(latent[start:position], k_rot[start:position]) for latent, k_rot in ctx]

    def body(self, anchor: int, position: int, ctx) -> torch.Tensor:
        """Final-norm output [depth, hidden] of one block at ``position``; ``ctx``
        already windowed (``window_of``)."""
        emb = self.w("embed")
        masks = self.w("backbone.mask_embedding").to(emb.dtype)
        x = torch.cat([emb[anchor][None], masks[: self.depth - 1]])
        positions = torch.arange(position, position + self.depth, device=x.device)
        for i in range(self.layers):
            l = f"backbone.layers.{i}."
            x = x + self._attention(ops.rms_norm(x, self.w(l + "attention_norm.weight"), self.eps), positions, i, ctx[i])
            x = x + self._ffn(ops.rms_norm(x, self.w(l + "ffn_norm.weight"), self.eps), i)
        return ops.rms_norm(x, self.w("backbone.final_norm.weight"), self.eps)

    def heads_(self, anchor: int, hidden: torch.Tensor) -> list[int]:
        """Greedy drafts of one block (sequential RNN Markov)."""
        base = ops.softcap(F.linear(hidden, self.w("head")).float(), self.softcap)
        embedding = self.w("autoregressive_head.embedding.weight")
        joint_w, joint_b = self.w("autoregressive_head.joint_proj.weight"), self.w("autoregressive_head.joint_proj.bias")
        output = self.w("autoregressive_head.output.weight")
        state, prev, drafts = None, anchor, []
        for k in range(self.depth):
            e = embedding[prev]
            state = torch.zeros_like(e) if state is None else state
            gate, candidate, out = F.linear(torch.cat([state, e, hidden[k]]), joint_w, joint_b).chunk(3)
            gate = torch.sigmoid(gate)
            state = gate * state + (1.0 - gate) * torch.tanh(candidate)
            logits = base[k] + F.linear(torch.tanh(out), output).float()
            prev = int(logits.argmax())
            drafts.append(prev)
        return drafts


def checkpoint_getter(snapshot: Path, device: str):
    """BF16 drafter weights and the target's BF16 embedding and head, read from
    the consolidated shards (the shared model.py's Checkpoint)."""
    from families.mistral_large4 import model
    checkpoint = model.Checkpoint(snapshot)
    names = {"embed": model.OUTER["embed_tokens.weight"], "head": model.OUTER["lm_head.weight"]}
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = checkpoint.get(names.get(name, PREFIX + name), device, torch.bfloat16)
        return cache[name]
    return get


def parse_positions(spec: str) -> list[int]:
    start, stop, *step = (int(v) for v in spec.split(":"))
    return list(range(start, stop, step[0] if step else 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--golden", type=Path, required=True, help="trace.py output (tokens.bin, layerNN.bin)")
    p.add_argument("--positions", required=True, help="START:STOP[:STEP] anchor positions")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    tokens = [int.from_bytes(b, "little") for b in
              (lambda d: [d[i:i + 4] for i in range(0, len(d), 4)])((args.golden / "tokens.bin").read_bytes())]
    taps = torch.cat([torch.frombuffer(bytearray((args.golden / f"layer{l:02d}.bin").read_bytes()), dtype=torch.bfloat16)
                      .view(len(tokens), -1) for l in TAPS], -1).to(args.device)
    drafter = Drafter(checkpoint_getter(args.snapshot, args.device))
    positions = parse_positions(args.positions)
    with torch.no_grad():
        ctx = drafter.context(taps, torch.arange(len(tokens), device=args.device))
        drafts, hidden = [], []
        for position in positions:
            h = drafter.body(tokens[position], position, drafter.window_of(ctx, position))
            drafts.append(drafter.heads_(tokens[position], h))
            hidden.append(h)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "drafts.bin").write_bytes(b"".join(t.to_bytes(4, "little") for row in drafts for t in row))
    (args.out / "hidden.bin").write_bytes(torch.stack(hidden).contiguous().view(torch.uint16).cpu().numpy().tobytes())
    (args.out / "meta.json").write_text(json.dumps({"positions": positions, "depth": drafter.depth,
                                                    "taps": TAPS, "window": drafter.window}, indent=1))


if __name__ == "__main__":
    main()
