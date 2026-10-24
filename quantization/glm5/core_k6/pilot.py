#!/usr/bin/env python3
"""Bounded BF16-reference GLM core-format pilot; cannot export or publish."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "third_party/gptqmodel"))
sys.path.insert(0, str(REPO / "quantization"))

from kimi_k3.uniform_k2.storage import (
    RollingBoundary, atomic_json, canonical, commit_safetensors, sha256_file, verify_receipt,
)
from kimi_k3.core_k6.pilot import payload_bytes, quantize_candidate, reconstruct, subset
from .source import GlmBF16Source

ARMS = ("mcg-k6", "mcg-k5", "mcg-k4", "fp8-block128", "bf16")
LAYERS = (1, 6)
SCHEMA = "cuteafd-glm53-core-pilot-v1"
GIB = 1024**3


def modules(layer):
    prefix = f"model.layers.{layer}."
    names = ["self_attn." + part for part in (
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj", "kv_b_proj",
        "indexer.wq_b", "indexer.wk", "indexer.weights_proj")]
    mlp = "mlp." if layer < 3 else "mlp.shared_experts."
    return [prefix + name for name in names + [mlp + part for part in ("gate_proj", "up_proj", "down_proj")]]


def tp2_projection_bytes(name, shape, arm):
    """Rank-local payloads, respecting replicated and split GLM projections."""
    n, k = shape
    size = n*k*2 if arm == "bf16" else payload_bytes(arm, shape)
    if name == "lm_head":
        return [size, 0]
    if any(name.endswith(part) for part in ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wq_b",
                                            "indexer.wk", "indexer.weights_proj")):
        return [size, size]
    if name.endswith(("o_proj", "down_proj")):
        split = (n, k//2)
    else:
        split = (n//2, k)
    local = split[0]*split[1]*2 if arm == "bf16" else payload_bytes(arm, split)
    return [local, local]


def weighted_nmse(weight, candidate, hessian):
    """Same trace metric as Kimi's pilot, with FP64 final accumulation."""
    import torch
    weight, candidate, hessian = weight.float(), candidate.float(), hessian.float()
    if weight.shape != candidate.shape or hessian.shape != (weight.shape[1],)*2:
        raise ValueError("incompatible projection metric geometry")
    error = candidate - weight
    denominator = float(((weight @ hessian)*weight).sum(dtype=torch.float64))
    numerator = float(((error @ hessian)*error).sum(dtype=torch.float64))
    if not 0 < denominator or not torch.isfinite(torch.tensor([numerator, denominator])).all():
        raise ValueError("nonfinite or zero-energy heldout metric")
    if numerator < -1e-6*denominator:
        raise ValueError("heldout quadratic error is negative")
    return max(0., numerator)/denominator


def mcg_candidate(weight, raw, arm, codec, quantizer):
    # The pinned MCG codec predates the optional codebook keyword used by
    # the newer Kimi E4M3 pilot. Its default is already standard MCG.
    if arm == "mcg-k6":
        return codec.quantize_core_projection(weight, raw)[0]
    return quantize_candidate(weight, raw, arm, codec, quantizer)


def validate_capture_basis(prior, current, old_code, new_code):
    """Permit only a reviewed scoring change over unchanged captured inputs."""
    relative="quantization/glm5/core_k6/pilot.py"
    if hashlib.sha256(old_code.encode()).hexdigest() != prior["code"][relative]:
        raise ValueError("original capture producer is not the recorded Git source")
    def capture_nodes(code):
        names={"modules", "Capture", "forward", "staged_forward", "capture_split", "compact", "commit_batch"}
        return [ast.dump(node, include_attributes=False) for node in ast.parse(code).body
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    if capture_nodes(old_code) != capture_nodes(new_code):
        raise ValueError("capture implementation changed; recapture is required")
    def inputs(record):
        record=json.loads(json.dumps(record))
        del record["code"][relative]
        return record
    if inputs(prior) != inputs(current):
        raise ValueError("capture inputs or dependencies changed")


class Capture:
    def __init__(self, block, layer):
        self.raw, self.handles = {}, []
        prefix = f"model.layers.{layer}."
        for name in modules(layer):
            module = block.get_submodule(name.removeprefix(prefix))
            self.handles.append(module.register_forward_pre_hook(self.hook(name)))

    def hook(self, name):
        def accumulate(module, args):
            import torch
            x = args[0].detach().reshape(-1, module.in_features).float()
            if name not in self.raw:
                self.raw[name] = dict(H=torch.zeros((x.shape[1], x.shape[1]), device=x.device), count=0,
                                      sample=x[:128].clone())
            raw = self.raw[name]
            raw["H"].addmm_(x.T, x)
            raw["count"] += len(x)
        return accumulate

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compact():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def commit_batch(path, tensors):
    return dict(filename=Path(path).name, **commit_safetensors(path, tensors))


def forward(block, rotary, state):
    import torch
    hidden = state["hidden"].to("cuda:0")
    seq = hidden.shape[1]
    positions = torch.arange(seq, device=hidden.device).unsqueeze(0)
    kwargs = dict(attention_mask=torch.ones((1, 1, seq, seq), device=hidden.device,
        dtype=torch.bool).tril(), position_ids=positions, position_embeddings=rotary(hidden, positions),
        prev_topk_indices=state.get("prev_topk_indices", None), use_cache=False)
    if kwargs["prev_topk_indices"] is not None:
        kwargs["prev_topk_indices"] = kwargs["prev_topk_indices"].to(hidden.device)
    out, topk = block(hidden, **kwargs)
    return dict(hidden=out, prev_topk_indices=topk), kwargs


def staged_forward(block, hidden, kwargs):
    attention, _, topk = block.self_attn(block.input_layernorm(hidden),
        **{key: value for key, value in kwargs.items() if key != "use_cache"})
    residual = hidden + attention
    return residual + block.mlp(block.post_attention_layernorm(residual)), topk


def progress(root, event, **fields):
    import psutil
    import torch
    record = dict(time=datetime.now(timezone.utc).isoformat(), event=event,
        rss_gib=psutil.Process().memory_info().rss/GIB,
        cuda_gib=torch.cuda.memory_allocated()/GIB, gpu_peak_gib=torch.cuda.max_memory_allocated()/GIB,
        free_gib=shutil.disk_usage(root).free/GIB, **fields)
    with (root/"events.jsonl").open("a") as f:
        f.write(json.dumps(record)+"\n")
    atomic_json(root/"status.json", record)
    print(json.dumps(record), flush=True)
    if record["rss_gib"] > 58 or record["cuda_gib"] > 86 or record["free_gib"] < 64:
        raise MemoryError("pilot exceeded host/GPU/disk admission")


def load_hessian(root, split, layer, name, identity, device):
    from safetensors.torch import load_file
    directory=root/split/"hessians"
    manifest=json.loads((directory/f"layer-{layer}.json").read_text())
    if manifest["identity"] != identity or set(manifest["files"]) != set(modules(layer)):
        raise ValueError("Hessian manifest differs from reviewed pilot inventory")
    path=directory/(name+".safetensors")
    verify_receipt(path, manifest["files"][name], checksum=True)
    raw=load_file(str(path), device=device)
    if int(raw["count"]) != identity[split+"_tokens"]:
        raise ValueError("Hessian observation count differs")
    return raw


def capture_split(source, records, root, identity):
    import torch
    from safetensors.torch import load_file
    boundary = RollingBoundary(root/"frontier", identity)
    current = boundary.current()
    if current is None:
        initial=boundary.directory(-1)
        if initial.is_symlink():
            raise ValueError("refuse symlink pilot embedding boundary")
        if initial.exists():
            shutil.rmtree(initial)
        embedding = source.tensor("model.embed_tokens.weight", "cuda:0")
        entries = []
        for batch, row in enumerate(records):
            ids = torch.tensor(row["input_ids"], device="cuda:0")
            hidden = torch.nn.functional.embedding(ids, embedding).unsqueeze(0)
            entries.append(commit_batch(boundary.directory(-1)/f"{batch:05d}.safetensors",
                                             dict(hidden=hidden)))
        current = boundary.commit(-1, entries)
        del embedding, hidden, ids
        source.close()
        compact()
    rotary = source.rotary()
    for layer in range(current["layer"]+1, max(LAYERS)+1):
        # A partial successor is not a recovery point. Recompute it from
        # the last committed boundary without overwriting committed files.
        partial=boundary.directory(layer)
        if partial.is_symlink():
            raise ValueError("refuse symlink pilot successor")
        if partial.exists():
            shutil.rmtree(partial)
        for name in modules(layer) if layer in LAYERS else ():
            (root/"hessians"/(name+".safetensors")).unlink(missing_ok=True)
        (root/"hessians"/f"layer-{layer}.json").unlink(missing_ok=True)
        progress(root, "loading_layer", layer=layer)
        block = source.load_block(layer)
        capture = Capture(block, layer) if layer in LAYERS else None
        entries = []
        try:
            for batch, entry in enumerate(current["manifest"]["batches"]):
                state = load_file(str(boundary.root/current["directory"]/entry["filename"]))
                successor, kwargs = forward(block, rotary, state)
                if batch == 0 or records[batch]["id"] == identity["long_context_witness"]:
                    # The hooks must not alter the native forward. Check an
                    # independently staged attention + residual + MLP boundary.
                    handles = [] if capture is None else capture.handles
                    for handle in handles:
                        handle.remove()
                    if capture is not None:
                        capture.handles = []
                    staged, topk = staged_forward(block, state["hidden"].cuda(), kwargs)
                    if not torch.equal(staged, successor["hidden"]) or not torch.equal(topk, successor["prev_topk_indices"]):
                        raise ValueError("staged/native GLM replay differs")
                    if capture is not None:
                        prefix=f"model.layers.{layer}."
                        for name in modules(layer):
                            capture.handles.append(block.get_submodule(name.removeprefix(prefix)).register_forward_pre_hook(capture.hook(name)))
                    del staged, topk
                    progress(root, "native_boundary_exact", layer=layer, record=records[batch]["id"],
                             tokens=len(records[batch]["input_ids"]))
                entries.append(commit_batch(boundary.directory(layer)/f"{batch:05d}.safetensors", successor))
                del state, successor, kwargs
                if batch % 8 == 0:
                    progress(root, "capture_progress", layer=layer, batches=batch+1, total=len(records))
            if capture is not None:
                receipts = {}
                for name, raw in capture.raw.items():
                    if raw["count"] != sum(len(row["input_ids"]) for row in records):
                        raise ValueError("Hessian observations differ from unmodified stream")
                    receipts[name] = commit_safetensors(root/"hessians"/(name+".safetensors"),
                        dict(H=raw["H"], count=torch.tensor(raw["count"]), sample=raw["sample"]))
                atomic_json(root/"hessians"/f"layer-{layer}.json", dict(identity=identity, files=receipts))
                del raw, receipts
            # Hessian and full successor are durable before retiring old state.
            current = boundary.commit(layer, entries)
        finally:
            if capture is not None:
                capture.close()
            del capture, block
            compact()
        progress(root, "layer_committed", layer=layer)
    del rotary
    compact()


def score(source, root, identity):
    import torch
    from safetensors.torch import load_file
    from gptqmodel.utils import kimi_k3_core_format as codec
    from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as quantizer
    rows=[]
    basis=identity.get("capture_basis")
    capture_root=Path(basis["root"]) if basis else root
    capture_identity=json.loads((capture_root/"plan.json").read_text()) if basis else identity
    for layer in LAYERS:
        for name in modules(layer):
            raw = load_hessian(capture_root, "calibration", layer, name, capture_identity, "cuda:0")
            weight = source.tensor(name+".weight", "cuda:0")
            source.close()
            shape = tuple(weight.shape)
            for arm in ARMS:
                path = root/"candidates"/(name+"."+arm+".safetensors")
                receipt = path.with_suffix(".json")
                if arm == "bf16":
                    continue
                if receipt.exists():
                    prior=json.loads(receipt.read_text())
                    if prior["identity"] != identity:
                        raise ValueError("candidate provenance differs")
                    verify_receipt(path, prior["receipt"], checksum=True)
                else:
                    # A payload without its immutable receipt is an orphan.
                    path.unlink(missing_ok=True)
                    packed=mcg_candidate(weight, dict(H=raw["H"], count=int(raw["count"]), finalized=False),
                                              arm, codec, quantizer)
                    size=sum(x.numel()*x.element_size() for x in packed.values())
                    if size != payload_bytes(arm, shape):
                        raise ValueError("serialized candidate accounting differs")
                    rec=commit_safetensors(path, packed)
                    atomic_json(receipt, dict(identity=identity, receipt=rec))
                    del packed
                progress(root, "candidate_committed", tensor=name, arm=arm)
            del raw
            heldout=load_hessian(capture_root, "heldout", layer, name, capture_identity, "cuda:0")
            for arm in ARMS:
                if arm == "bf16":
                    dense=weight.float()
                    size=weight.numel()*weight.element_size()
                else:
                    packed=load_file(str(root/"candidates"/(name+"."+arm+".safetensors")))
                    dense=reconstruct(packed, arm, shape, codec, "cuda:0")
                    size=sum(x.numel()*x.element_size() for x in packed.values())
                    del packed
                nmse=weighted_nmse(weight, dense, heldout["H"])
                row=dict(tensor=name, arm=arm, nmse=nmse, payload_bytes=size,
                    bpw=size*8/weight.numel(), tp2_bytes=tp2_projection_bytes(name, shape, arm),
                    calibration_tokens=identity["calibration_tokens"], heldout_tokens=identity["heldout_tokens"])
                rows.append(row)
                atomic_json(root/"results.json", dict(identity=identity, results=rows))
                progress(root, "pilot_result", **row)
                del dense
            del heldout, weight
            compact()
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("source-snapshot", "tokens", "root"):
        parser.add_argument("--"+arg, type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=int, default=32768)
    parser.add_argument("--heldout-tokens", type=int, default=4096)
    parser.add_argument("--capture-root", type=Path, help="reuse a complete, verified compatible pilot capture")
    parser.add_argument("--preflight", action="store_true")
    args=parser.parse_args(argv)
    if not 32768 <= args.calibration_tokens <= 65536 or not 4096 <= args.heldout_tokens <= 16384:
        parser.error("bounded pilot token admission differs")
    from glm5.prepare_corpus import load_token_records
    from .recipe import BASE_REVISION, inventory
    source=GlmBF16Source(args.source_snapshot)
    if source.snapshot.name != BASE_REVISION:
        raise ValueError("pilot source is not the pinned BF16 revision")
    reviewed=inventory(source)
    all_calibration=load_token_records(args.tokens, "calibration")
    calibration=subset(all_calibration, args.calibration_tokens)
    # An unchanged natural calibration prompt exercises DSA's sparse branch;
    # heldout prompts are all below topk2048, so this is a correctness witness,
    # not independent long-context quality evidence.
    long_row=next((r for r in all_calibration if len(r["input_ids"]) > 2048), None)
    if long_row is not None and long_row["id"] not in {r["id"] for r in calibration}:
        calibration.append(long_row)
    heldout=subset(load_token_records(args.tokens, "heldout"), args.heldout_tokens)
    if {r["id"] for r in calibration} & {r["id"] for r in heldout}:
        raise ValueError("pilot splits overlap")
    code_files=[Path(__file__), Path(__file__).with_name("source.py"),
        Path(__file__).with_name("recipe.py"),
        REPO/"quantization/glm5/prepare_corpus.py", REPO/"quantization/kimi_k3/prepare_corpus.py",
        REPO/"quantization/kimi_k3/core_k6/pilot.py", REPO/"quantization/kimi_k3/uniform_k2/storage.py",
        REPO/"third_party/gptqmodel/gptqmodel/utils/kimi_k3_core_format.py"]
    identity=dict(schema=SCHEMA, source=str(source.snapshot), arms=ARMS, layers=LAYERS,
        source_inventory=reviewed,
        source_config_sha256=sha256_file(source.snapshot/"config.json"),
        source_index_sha256=sha256_file(source.snapshot/"model.safetensors.index.json"),
        token_manifest_sha256=sha256_file(args.tokens/"manifest.json"),
        calibration_sha256=hashlib.sha256(canonical(calibration).encode()).hexdigest(),
        heldout_sha256=hashlib.sha256(canonical(heldout).encode()).hexdigest(),
        calibration_tokens=sum(len(r["input_ids"]) for r in calibration),
        heldout_tokens=sum(len(r["input_ids"]) for r in heldout),
        replay="fixed-native-bf16-reference-no-candidate-affects-replay",
        selection="original-order-prefix-plus-first-natural-calibration-prompt-over2048",
        long_context_witness=None if long_row is None else long_row["id"],
        head="not-measured-final-normalized-input-requires-full-78-layer-replay",
        code={str(p.relative_to(REPO)):sha256_file(p) for p in code_files},
        gptqmodel_commit=subprocess.check_output(["git", "-C", str(REPO/"third_party/gptqmodel"), "rev-parse", "HEAD"], text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")})
    # JSON round trip normalizes tuples for exact immutable comparisons.
    identity=json.loads(json.dumps(identity))
    if args.capture_root:
        args.capture_root=args.capture_root.resolve()
        if args.capture_root == args.root.resolve():
            raise ValueError("capture basis must be an independent immutable run")
        prior=json.loads((args.capture_root/"plan.json").read_text())
        old_code=subprocess.check_output(["git", "-C", str(REPO), "show",
            "a6b1b22f7:quantization/glm5/core_k6/pilot.py"],text=True)
        validate_capture_basis(prior,identity,old_code,Path(__file__).read_text())
        for split in ("calibration", "heldout"):
            for layer in LAYERS:
                for name in modules(layer):
                    raw=load_hessian(args.capture_root,split,layer,name,prior,"cpu")
                    del raw
        identity["capture_basis"]=dict(root=str(args.capture_root),
            plan_sha256=sha256_file(args.capture_root/"plan.json"),
            contract="unchanged-capture-AST-and-identical-input-dependencies-verified")
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root/"plan.json").exists() and json.loads((args.root/"plan.json").read_text()) != identity:
        raise ValueError("pilot identity differs; select a fresh run root")
    if args.preflight:
        for directory, paths in ((REPO, ["quantization/glm5", "quantization/kimi_k3/prepare_corpus.py"]),
                                 (REPO/"third_party/gptqmodel", [])):
            if subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--", *paths], text=True).strip():
                raise ValueError("pilot requires committed producer code")
        print(json.dumps(dict(admission="passed", identity=identity)), flush=True)
        return
    if (args.root/"summary.json").exists():
        completed=json.loads((args.root/"summary.json").read_text())
        expected={(name,arm) for layer in LAYERS for name in modules(layer) for arm in ARMS}
        if (completed["identity"] != identity or len(completed["results"]) != len(expected)
                or {(row["tensor"],row["arm"]) for row in completed["results"]} != expected):
            raise ValueError("pilot completion inventory differs")
        for split in ("calibration", "heldout"):
            frontier=(args.capture_root or args.root)/split/"frontier"
            if frontier.is_symlink():
                raise ValueError("refuse symlink pilot frontier")
            if frontier.exists():
                shutil.rmtree(frontier)
        print("Pilot already complete; no further GPU work.")
        return
    import torch
    from kimi_k3.core_k6.run import exclusive_run
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(86*GIB/torch.cuda.get_device_properties(0).total_memory)
    with torch.inference_mode(), exclusive_run(args.root):
        atomic_json(args.root/"plan.json", identity)
        if not args.capture_root:
            for split, records in (("calibration", calibration), ("heldout", heldout)):
                (args.root/split).mkdir(exist_ok=True)
                capture_split(source, records, args.root/split, identity)
        rows=score(source, args.root, identity)
        atomic_json(args.root/"summary.json", dict(identity=identity, results=rows,
            stop="pilot-only-await-recipe-decision", head_measured=False))
        # Keep candidates, Hessians, receipts and numerical evidence, but no
        # replay activations once every pilot result is durable.
        for split in ("calibration", "heldout"):
            frontier=(args.capture_root or args.root)/split/"frontier"
            if frontier.exists():
                shutil.rmtree(frontier)
        progress(args.root, "complete", results=len(rows))
    source.close()


if __name__ == "__main__":
    main()
