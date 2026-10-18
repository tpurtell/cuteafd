#!/usr/bin/env python3
"""Scoped router parity: official Kimi gate, synthetic ties, and old-family ABI."""
import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def oracle(logits, correction, topk=16, scale=1.0, sigmoid=True):
    """Official ungrouped gate math, with an explicit lower-ID tie order."""
    logits = np.asarray(logits, dtype=np.float32)
    if sigmoid:
        with np.errstate(over="ignore"):
            scores = np.float32(1) / (np.float32(1) + np.exp(-logits))
    else:
        scores = np.sqrt(np.where(logits > 20, logits, np.logaddexp(0, logits)))
    ids = np.argsort(-(scores + correction), axis=-1, kind="stable")[:, :topk]
    selected = np.take_along_axis(scores, ids, axis=1)
    total = np.zeros((len(logits), 1), dtype=np.float32)
    for rank in range(topk):
        total += selected[:, rank:rank + 1]
    return ids, selected / (total + np.float32(1e-20)) * np.float32(scale)


def range_tensor(snapshot, index, name):
    """Read only one indexed tensor's bytes, never the containing weight shard."""
    with (snapshot / index[name]).open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
        tensor = header[name]
        start, end = tensor["data_offsets"]
        f.seek(8 + length + start)
        data = f.read(end - start)
    if len(data) != end - start:
        raise ValueError("short tensor read")
    return tensor, data


def official_gate(path, config, torch):
    # Execute only the reviewed gate class, not the checkpoint's import hooks.
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KimiMoEGate")
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional,
                     KimiLinearConfig=object, math=__import__("math"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "official_gate", "exec"), namespace)
    return namespace["KimiMoEGate"](config).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--baseline-lib", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--official-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from types import SimpleNamespace

    torch.manual_seed(273)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    config = json.loads((args.snapshot / "config.json").read_text())["text_config"]
    assert (config["num_experts"], config["num_experts_per_token"],
            config["num_expert_group"], config["topk_group"], config["moe_renormalize"],
            config["moe_router_activation_func"], config["routed_scaling_factor"]) == (
                896, 16, 1, 1, True, "sigmoid", 1.0)
    gate = official_gate(args.official_model, SimpleNamespace(**config), torch)
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = next(name.removesuffix(".weight") for name in sorted(index)
                  if name.endswith(".block_sparse_moe.gate.weight") and ".layers.1." in name)
    tensors, hashes = {}, {}
    for suffix in ("weight", "e_score_correction_bias"):
        name = prefix + "." + suffix
        info, data = range_tensor(args.snapshot, index, name)
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[info["dtype"]]
        tensors[suffix] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(info["shape"]).float()
        hashes[name] = hashlib.sha256(data).hexdigest()
    gate.load_state_dict(tensors)
    bias = tensors["e_score_correction_bias"].cuda()

    def bind(path):
        lib = ctypes.CDLL(str(path.resolve()))
        fn = lib.cuteafd_router_select
        fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int32] * 3 + [ctypes.c_float,
                      ctypes.c_int32, ctypes.c_void_p]
        fn.restype = ctypes.c_int32
        return fn

    candidate = bind(args.lib)
    baseline = bind(args.baseline_lib) if args.baseline_lib else None
    cases = []

    def launch(fn, logits, correction, topk, scale, sigmoid):
        work = logits.clone()
        n = logits.shape[0] * topk
        # Guard output stores and initialize to values no valid result can have.
        guarded_ids = torch.full((n + 32,), -123, dtype=torch.int32, device="cuda")
        guarded_weights = torch.full((n + 32,), -123.0, device="cuda")
        ids = guarded_ids[16:-16].view(-1, topk)
        weights = guarded_weights[16:-16].view(-1, topk)
        status = fn(work.data_ptr(), correction.data_ptr(), None, None,
                    ids.data_ptr(), weights.data_ptr(), *logits.shape, topk, scale,
                    int(sigmoid), torch.cuda.current_stream().cuda_stream)
        assert status == 0, status
        torch.cuda.synchronize()
        for guarded in (guarded_ids, guarded_weights):
            assert (guarded[:16] == -123).all() and (guarded[-16:] == -123).all()
        return ids.long(), weights, work

    def check(logits, correction, topk=16, scale=1.0, sigmoid=True, label="synthetic"):
        ids, weights, scores = launch(candidate, logits.cuda(), correction, topk, scale, sigmoid)
        ref_ids, ref_weights = oracle(logits.cpu().numpy(), correction.cpu().numpy(), topk, scale, sigmoid)
        np.testing.assert_array_equal(ids.cpu().numpy(), ref_ids)
        np.testing.assert_allclose(weights.cpu().numpy(), ref_weights, atol=2e-7, rtol=3e-6)
        if baseline and logits.shape[1] <= 512 and topk <= 8:
            old = launch(baseline, logits.cuda(), correction, topk, scale, sigmoid)
            for new, previous in zip((ids, weights, scores), old):
                assert torch.equal(new, previous), "old-family output changed"
        cases.append(dict(label=label, rows=len(logits), experts=logits.shape[1], topk=topk,
                          sigmoid=sigmoid, max_weight_error=float(np.max(np.abs(weights.cpu().numpy() - ref_weights)))))
        return ids.cpu(), weights.cpu()

    for rows in (1, 6, 16, 80):
        hidden = torch.randn(rows, 1, 7168).bfloat16().float()
        logits = hidden.reshape(rows, -1) @ tensors["weight"].T
        ids, weights = check(logits, bias, label="real-weight-f32-projection")
        with torch.no_grad():
            official_ids, official_weights = gate(hidden)
        # Official torch.topk(sorted=False) does not specify output order.
        order = official_ids.argsort(dim=-1)
        ours = ids.argsort(dim=-1)
        assert torch.equal(ids.gather(1, ours), official_ids.gather(1, order))
        torch.testing.assert_close(weights.gather(1, ours), official_weights.gather(1, order),
                                   atol=2e-7, rtol=3e-6)
    for experts, topk in ((1, 1), (128, 3), (256, 6), (384, 6), (512, 8),
                          (512, 16), (513, 16), (896, 16), (1024, 16)):
        correction = torch.randn(experts, device="cuda") * 0.01
        for sigmoid in (True, False):
            check(torch.randn(6, experts), correction, topk, 1.5, sigmoid)
        zero = torch.zeros(experts, device="cuda")
        check(torch.zeros(2, experts), zero, topk, label="all-ties")
        check(torch.full((2, experts), -1000.0), zero, topk, label="zero-scores")
        high_bias = zero.clone()
        high_bias[-topk:] = 2
        check(torch.zeros(2, experts), high_bias, topk, label="high-expert-ids")
    # Invalid bounds and unsupported hash extensions must fail before touching memory.
    for experts, topk, sigmoid, hashed in ((1025, 16, 1, False), (896, 17, 1, False),
                                         (896, 16, 0, True), (512, 16, 0, True)):
        assert candidate(4, None if hashed else 8, 8 if hashed else None,
                         12 if hashed else None, 16, 20, 1, experts, topk, 1.0,
                         sigmoid, None) == 1
    result = dict(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  native_sha256=hashlib.sha256(args.lib.read_bytes()).hexdigest(),
                  weights=hashes, cases=cases, verdict="pass")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
