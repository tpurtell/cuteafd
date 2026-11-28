"""Shared IEEE-F32 arithmetic primitives for Hy4 and Hy4-flash references."""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
from pathlib import Path
import struct

import torch
import torch.nn.functional as F

from shape_invariant import ROWS, linear

OUTER_NAMES = ("model.embed_tokens.weight", "model.hc_head.hc_head_fn", "model.hc_head.hc_head_base",
               "model.hc_head.hc_head_scale", "model.norm.weight", "lm_head.weight")
EXPERTS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
SPARSE = "deepseek_sparse_attention"
_TORCH_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I32": torch.int32}


class Checkpoint:
    def __init__(self, snapshot, device="cpu"):
        self.snapshot = Path(snapshot)
        self.device = torch.device(device)
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.headers, self.header_offsets, self.maps = {}, {}, {}
        self.read_bytes, self.read_seconds = 0, 0.

    def descriptor(self, name):
        shard = self.index.get(name)
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"missing tensor or invalid Hy4 shard path: {name}")
        if shard not in self.headers:
            path = self.snapshot / shard
            with path.open("rb") as stream:
                prefix = stream.read(8)
                size, = struct.unpack("<Q", prefix)
                if not 2 <= size <= 64 * 2**20:
                    raise ValueError(f"invalid Hy4 safetensors header size: {shard}")
                header = json.loads(stream.read(size))
            if max(entry["data_offsets"][1] for key, entry in header.items() if key != "__metadata__") + 8 + size != path.stat().st_size:
                raise ValueError(f"incomplete Hy4 shard: {shard}")
            self.headers[shard] = header
            self.header_offsets[shard] = 8 + size
        return self.headers[shard][name]

    def check(self, contract):
        for name, (shape, dtype) in contract.items():
            d = self.descriptor(name)
            first, last = d["data_offsets"]
            if d["shape"] != shape or d["dtype"] != dtype or first < 0 or last - first != math.prod(shape) * (2 if dtype == "BF16" else 4):
                raise ValueError(f"wrong Hy4 tensor shape, dtype or extent: {name}: {d}")

    def raw(self, name):
        from safetensors import safe_open
        self.descriptor(name)
        shard = self.index[name]
        if shard not in self.maps:
            self.maps[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.maps[shard].get_tensor(name)

    def read_tensor(self, name):
        """Read a tensor directly via pread into a freshly allocated CPU tensor (avoids FUSE mmap stalls)."""
        d = self.descriptor(name)
        shard = self.index[name]
        dtype = _TORCH_DTYPES[d["dtype"]]
        first, last = d["data_offsets"]
        nbytes = last - first
        base = self.header_offsets[shard] + first
        tensor = torch.empty(d["shape"], dtype=dtype)
        mv = memoryview(tensor.view(torch.uint8).numpy().ravel())
        fd = os.open(self.snapshot / shard, os.O_RDONLY)
        try:
            if nbytes < 64 * 2**20:
                pos = 0
                while pos < nbytes:
                    chunk = os.pread(fd, min(16 * 2**20, nbytes - pos), base + pos)
                    if not chunk:
                        raise ValueError(f"short read on {name}")
                    mv[pos:pos + len(chunk)] = chunk
                    pos += len(chunk)
            else:
                stripe = 64 * 2**20

                def read_stripe(start):
                    end = min(start + stripe, nbytes)
                    pos = start
                    while pos < end:
                        chunk = os.pread(fd, min(16 * 2**20, end - pos), base + pos)
                        if not chunk:
                            raise ValueError(f"short read on {name}")
                        mv[pos:pos + len(chunk)] = chunk
                        pos += len(chunk)

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(read_stripe, range(0, nbytes, stripe)))
        finally:
            os.close(fd)
        return tensor

    def load(self, name, *, native=False):
        import time
        tick = time.monotonic()
        value = self.read_tensor(name)
        self.read_bytes += value.numel() * value.element_size()
        target_dtype = value.dtype if native else torch.float32
        if self.device.type == "cpu" and value.dtype == target_dtype:
            pass
        else:
            value = value.to(self.device, dtype=target_dtype, copy=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.read_seconds += time.monotonic() - tick
        return value

    def close(self):
        self.maps.clear()


def inverse_rms(x, eps):
    return torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def norm(x, w, eps):
    return (x * inverse_rms(x, eps)) * w


def project(x, w, name, bias=False):
    return linear(x, w[name + ".weight"], w[name + ".bias"] if bias else None)


def hyper(x, weights, name, c, matmul=linear):
    p = name + ".hc_pre.hc_"
    flat = x.flatten(-2)
    pre, post = (matmul(flat, weights[p + "fn"]) * inverse_rms(flat, c.rms_norm_eps)).chunk(2, dim=-1)
    bp, bo = weights[p + "base"].chunk(2, dim=-1)
    sp, so = weights[p + "scale"].unbind(0)
    pre = (pre * sp + bp).sigmoid() + c.hc_eps
    post = c.hc_magnitude * (post * so + bo).sigmoid() + c.hc_eps
    return (x * pre.unsqueeze(-1)).sum(-2), post.unsqueeze(-1)


def head_features(hidden, w, c, matmul=linear):
    flat = hidden.flatten(-2)
    mix = matmul(flat, w[OUTER_NAMES[1]]) * inverse_rms(flat, c.rms_norm_eps)
    pre = (mix * w[OUTER_NAMES[3]] + w[OUTER_NAMES[2]]).sigmoid() + c.hc_eps
    return norm((hidden * pre.unsqueeze(-1)).sum(-2), w[OUTER_NAMES[4]], c.rms_norm_eps)


def phase(positions, width, base):
    inv = 1. / (base ** (torch.arange(0, width, 2, dtype=torch.float32, device=positions.device) / width))
    freq = positions.float().unsqueeze(-1) * inv
    return tuple(f(freq).repeat_interleave(2, dim=-1).unsqueeze(1) for f in (torch.cos, torch.sin))


def rotate(x, cos, sin):
    alternating = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + alternating * sin


def rotate_tail(x, cos, sin, *, rotate_fn=rotate):
    prefix, tail = x.split([x.shape[-1] - cos.shape[-1], cos.shape[-1]], dim=-1)
    return torch.cat((prefix, rotate_fn(tail, cos, sin)), dim=-1)


def phase_split_half(positions, width, base):
    inv = 1. / (base ** (torch.arange(0, width, 2, dtype=torch.float32, device=positions.device) / width))
    freq = positions.float().unsqueeze(-1) * inv
    emb = torch.cat((freq, freq), dim=-1)
    return tuple(f(emb).unsqueeze(1) for f in (torch.cos, torch.sin))


def rotate_split_half(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def mla_projections(x, w, c, h, nope, rope, vdim, cos, sin, *, qa_kva_eps=None, rotate_fn=rotate):
    p = "self_attn."
    eps = c.rms_norm_eps if qa_kva_eps is None else qa_kva_eps
    q0 = norm(project(x, w, p + "q_a_proj"), w[p + "q_a_layernorm.weight"], eps)
    q = rotate_tail(project(q0, w, p + "q_b_proj").reshape(len(x), h, nope + rope), cos, sin, rotate_fn=rotate_fn)
    k0, kr = project(x, w, p + "kv_a_proj_with_mqa").split([c.kv_lora_rank, rope], dim=-1)
    k0 = norm(k0, w[p + "kv_a_layernorm.weight"], eps)
    k, v = project(k0, w, p + "kv_b_proj").reshape(len(x), h, nope + vdim).split([nope, vdim], dim=-1)
    k = torch.cat((k, rotate_fn(kr.unsqueeze(1), cos, sin).expand(-1, h, -1)), dim=-1)
    gate = project(x, w, p + "linear_gate").reshape(len(x), h, vdim).sigmoid()
    sink = w[p + "learnable_sink_param"][:, None, None]
    return q0, q, k, v, gate, sink


def tiled_sink_attention(q, k, v, gate, sink, scale, *, sliding_window=None, tile_mask=None,
                         exact_keys=None, exact_values=None, sink_tokens=0, recent_tokens=0):
    seq_len = len(q)
    device = q.device
    no_grad = not torch.is_grad_enabled()
    k_ht = k.transpose(0, 1).contiguous() if no_grad else None
    v_ht = v.transpose(0, 1).contiguous() if no_grad else None
    result = []
    for start in range(0, seq_len, ROWS):
        stop = min(start + ROWS, seq_len)
        first = max(0, start + 1 - sliding_window) if sliding_window is not None else 0
        width = start + ROWS - first
        num_valid = stop - first
        # Pad the tail to preserve causal row and key geometry across prefix extensions.
        q_pad = ROWS - (stop - start)
        kv_pad = width - num_valid
        q_slice = q[start:stop].transpose(0, 1)
        query = F.pad(q_slice, (0, 0, 0, q_pad)) if q_pad else (q_slice.contiguous() if no_grad else q_slice)
        if no_grad:
            k_slice = k_ht[:, first:stop]
            v_slice = v_ht[:, first:stop]
            keys = F.pad(k_slice, (0, 0, 0, kv_pad)) if kv_pad else k_slice
            vals = F.pad(v_slice, (0, 0, 0, kv_pad)) if kv_pad else v_slice
        else:
            keys = F.pad(k[first:stop].transpose(0, 1), (0, 0, 0, kv_pad))
            vals = F.pad(v[first:stop].transpose(0, 1), (0, 0, 0, kv_pad))
        key_id = torch.arange(first, start + ROWS, device=device)
        query_id = torch.arange(start, start + ROWS, device=device)
        distance = query_id[:, None] - key_id
        allowed = (distance >= 0) & (key_id < stop)
        if sliding_window is not None:
            allowed &= distance < sliding_window
        if tile_mask is not None:
            allowed &= tile_mask(start, stop, first, width, query_id)
        # Queries in this tile can only protect the sink and a band near the
        # tile. Keep the second dot product bounded as the context grows.
        score = query @ keys.transpose(-1, -2)
        protected_regions = []
        if exact_keys is not None:
            sink_stop = min(stop, max(first, sink_tokens))
            recent_start = max(sink_stop, start + 1 - recent_tokens)
            regions = [(first, sink_stop), (recent_start, stop)] if recent_tokens else [(first, sink_stop)]
            if no_grad:
                for begin, end in regions:
                    if begin >= end:
                        continue
                    span = slice(begin - first, end - first)
                    protected = (key_id[span] < sink_tokens) | (distance[:, span] < recent_tokens)
                    protected_regions.append((begin, end, span, protected))
                    exact_score = query @ exact_keys[begin:end].permute(1, 2, 0)
                    score[..., span] = torch.where(protected, exact_score, score[..., span])
            else:
                pieces = []
                cursor = 0
                for begin, end in regions:
                    if begin >= end:
                        continue
                    span = slice(begin - first, end - first)
                    protected = (key_id[span] < sink_tokens) | (distance[:, span] < recent_tokens)
                    protected_regions.append((begin, end, span, protected))
                    exact_score = query @ exact_keys[begin:end].permute(1, 2, 0)
                    pieces.extend([score[..., cursor:span.start], torch.where(protected, exact_score, score[..., span])])
                    cursor = span.stop
                pieces.append(score[..., cursor:])
                score = torch.cat(pieces, dim=-1)
        if no_grad:
            score.mul_(scale)
            score.masked_fill_(~allowed, torch.finfo(score.dtype).min)
        else:
            score = score * scale
            score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
        if sink is not None:
            score = torch.cat((score, sink.expand(-1, ROWS, -1)), dim=-1)
        probabilities = (score - score.max(dim=-1, keepdim=True).values).softmax(-1)
        del score
        if sink is not None:
            probabilities = probabilities[..., :-1]
        if exact_values is None:
            attended = probabilities @ vals
        elif no_grad:
            exact_terms = []
            for begin, end, span, protected in protected_regions:
                weights = probabilities[..., span].clone()
                probabilities[..., span].masked_fill_(protected, 0.0)
                exact_terms.append(weights.masked_fill(~protected, 0.0) @ exact_values[begin:end].transpose(0, 1))
            attended = probabilities @ vals
            for term in exact_terms:
                attended.add_(term)
        else:
            # Each key contributes its own exact or quantized value with the
            # probability from the single joint softmax above.
            regular = probabilities.clone()
            exact_terms = []
            for begin, end, span, protected in protected_regions:
                weights = probabilities[..., span]
                regular[..., span] = weights.masked_fill(protected, 0.0)
                exact_terms.append(weights.masked_fill(~protected, 0.0) @ exact_values[begin:end].transpose(0, 1))
            attended = regular @ vals
            for term in exact_terms:
                attended = attended + term
        del probabilities
        output = attended.transpose(0, 1)[:stop - start]
        result.append(output * gate[start:stop] if gate is not None else output)
    return torch.cat(result).reshape(seq_len, -1)


def dense(x, w, prefix):
    return project(F.silu(project(x, w, prefix + "gate_proj")) * project(x, w, prefix + "up_proj"),
                   w, prefix + "down_proj")


def routing_scores(x, w, c, matmul=linear):
    scores = matmul(x, w["mlp.gate.weight"]).sigmoid()
    choice = scores + w["mlp.gate.e_score_correction_bias"]
    group_scores = choice.reshape(len(x), c.n_group, -1).topk(2, dim=-1).values.sum(dim=-1)
    groups = group_scores.topk(c.topk_group, dim=-1, sorted=False).indices
    valid = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
    return scores, choice.masked_fill(~valid.repeat_interleave(scores.shape[-1] // c.n_group, dim=1), -torch.inf)


def native_routes(scores, choice, c):
    ids = choice.topk(c.num_experts_per_tok, dim=-1, sorted=False).indices
    probs = scores.gather(1, ids)
    if c.norm_topk_prob:
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    return ids, probs * c.routed_scaling_factor


def routing(x, w, c):
    return native_routes(*routing_scores(x, w, c), c)


def routed_experts(x, ids, probs, w, c):
    routed = torch.zeros_like(x)
    # Expert order follows the official eager accumulation. Within each expert,
    # sort by token, so extending a prefix never changes its GEMM tile membership.
    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=c.n_routed_experts).tolist()
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        slots = order[start:start + count]
        tokens = slots // c.num_experts_per_tok
        # Stream one expert matrix to F32 at a time (supports device BF16 or host mmap).
        gate_up_w = w[EXPERTS[0]][expert].to(x.device, dtype=torch.float32)
        gate, up = linear(x[tokens], gate_up_w).chunk(2, dim=-1)
        active = F.silu(gate.clamp_max(c.swiglu_limit)) * up.clamp(-c.swiglu_limit, c.swiglu_limit)
        down_w = w[EXPERTS[1]][expert].to(x.device, dtype=torch.float32)
        down = linear(active, down_w)
        routed[tokens] = routed[tokens] + down * probs.flatten()[slots, None]
        start += count
    return routed


def configure_f32():
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"dtype": "float32", "matmul_precision": torch.get_float32_matmul_precision(),
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}
