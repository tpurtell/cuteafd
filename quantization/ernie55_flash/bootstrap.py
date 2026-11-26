"""Capture the first production HCA inputs once on the complete frozen corpus."""
from pathlib import Path
import json
import socket
import time

import torch
from safetensors.torch import load_file

from inkling2.numerical import covariance_accumulator
from kimi_k3.uniform_k2.storage import atomic_json, commit_safetensors, verify_receipt
from . import identity
from .container import TASK
from .forward import ErnieForward, reference
from .forward_preflight import embed
from .prepare_corpus import load_token_batches
from .source import BASE_REVISION

SOURCE = Path("/mnt/sparknest/hf-home/hub/models--baidu--ERNIE-5.5-Flash-Preview-BF16/snapshots") / BASE_REVISION
PREFIX = "model.layers.0.self_attn."
INPUTS = {"linear_q_down_proj": "all", "linear_kv_proj": "all",
          "core_attention.compressor.linear_wkv": "compressor",
          "core_attention.compressor.linear_wgate": "compressor"}


def inputs_key(inputs):
    return identity.digest(dict(parity=identity.parity_input_digest(inputs), fork=inputs["fork"],
                                bootstrap=inputs["code"]["quantization/ernie55_flash/bootstrap.py"],
                                schema="ernie55-full-corpus-layer0-hca-inputs-v1"))


def load_bootstrap(run, key):
    root = run / "bootstrap"
    receipt = identity.read(root / "complete.json")
    if receipt["identity"] != key or receipt["status"] != "passed":
        raise ValueError("initial Hessian capture identity differs")
    path = root / "hessians.safetensors"
    verify_receipt(path, receipt["payload"], checksum=True)
    matrices = load_file(str(path))
    return {PREFIX + name + ".weight": dict(H=matrices[k], count=receipt["counts"][k], finalized=False)
            for name, k in INPUTS.items()}


@torch.inference_mode()
def main():
    run = TASK
    if socket.gethostname() != "maxq" or not torch.cuda.is_available():
        raise RuntimeError("full ERNIE Hessian capture requires the maxq CUDA worker")
    root = run / "bootstrap"
    root.mkdir(exist_ok=True)
    if (root / "complete.json").exists():
        raise RuntimeError("production HCA Hessians already committed")
    local = identity.read(run / "preflight/forward-parity.json")
    vendor = identity.read(run / "preflight/vendor-merged.json")
    pilot_path = run / "pilot/pilot.json"
    pilot = identity.read(pilot_path)
    if pilot["status"] != "passed" or identity.sha256_file(pilot_path) != "1b32bd6586530fbb1e656ba1cb565a8b17c64c1724e9c4d58c20b1955dbe4341":
        raise ValueError("expected the passed full ERNIE §11 pilot receipt")
    inputs = identity.candidate(SOURCE, run/"corpus", identity.REPO)
    identity.verified_identity(inputs, vendor, local)
    key = inputs_key(inputs)
    model = ErnieForward.from_snapshot(SOURCE, "cuda")
    c = model.config
    embedding = model.checkpoint.load(reference.OUTER_NAMES[0])
    w = model.load_layer(0)
    acc = {k: covariance_accumulator(c.hidden_size, device="cuda") for k in ("all","compressor")}
    manifest = identity.read(run / "corpus/manifest.json")["splits"]["calibration"]
    start = time.monotonic()
    for i, batch in enumerate(load_token_batches(run / "corpus")):
        ids = batch["input_ids"][0]
        h = embed(ids, embedding, c)
        _, _, x = reference.mhc_pre(h, w, "self_attention_hyper_connection", w["input_layernorm.weight"], c)
        acc["all"].add(x)
        complete = (len(x) // reference.COMPRESS_RATIO_HCA) * reference.COMPRESS_RATIO_HCA
        if complete:
            acc["compressor"].add(x[:complete])
        if i % 32 == 0:
            progress = dict(phase="production_hessian", layer=0, projections=list(INPUTS), prompts=i+1,
                            total=len(manifest["records"]), tokens=acc["all"].count, elapsed=time.monotonic()-start)
            print(json.dumps(progress),flush=True)
    counts = {k: v.count for k,v in acc.items()}
    if i+1 != len(manifest["records"]) or counts["all"] != manifest["tokens"]:
        raise ValueError("full calibration coverage differs")
    payload = commit_safetensors(root/"hessians.safetensors", {k:v.snapshot() for k,v in acc.items()})
    atomic_json(root/"complete.json", dict(status="passed", identity=key, counts=counts, payload=payload,
                                           prompts=i+1, corpus=inputs["corpus"], elapsed=time.monotonic()-start))
    print("Completed full-corpus layer 0 input Hessians",counts,flush=True)


if __name__ == "__main__":
    main()
