"""Compare native GLM Flash decoder boundaries and capture real prompt tails for the pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead, Glm5NextTextRMSNorm

from inkling2.numerical import fork_module
from kimi_k3.uniform_k2.storage import atomic_json
from glm5_flash.core_k6.pilot import native_staged_boundary
from . import corpus, source as recipe

LAYERS = (0, 3, 22, 23, 44)
PREFIX = "model.language_model."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(directory, layer, states):
    path = directory / f"layer-{layer:02}.safetensors"
    save_file({key: value.contiguous().cpu() for key, value in states.items()}, str(path))
    return digest(path)


@torch.inference_mode()
def run(snapshot, root):
    import torch.nn.functional as F
    math = fork_module("utils.glm5_flash_core")
    mtp = fork_module("utils.glm5_flash_mtp")
    audit = json.loads((root / "source.json").read_text())
    if audit["status"] != "passed" or audit["source"] != str(snapshot.resolve()) or snapshot.name != recipe.BASE_REVISION:
        raise ValueError("source audit does not cover the pilot snapshot")
    out = root / "pilot-inputs"
    out.mkdir(exist_ok=True)
    if (root / "parity.json").exists():
        raise ValueError("parity receipt exists; verify it before scheduling another run")
    source = math.GlmFlashBF16Source(snapshot)
    embedding = source.tensor(PREFIX + "embed_tokens.weight", "cuda")
    ids = {}
    for split, label, count, length in (("calibration", "train", 4, 256), ("heldout", "held", 2, 128)):
        batches = corpus.load_token_batches(root / "corpus", split)
        for i in range(count):
            ids[f"{label}_{i}"] = next(batches)["input_ids"][:, -length:].cuda()
    def initial(tokens):
        return F.embedding(tokens, embedding).unsqueeze(2).expand(-1, -1, 4, -1).contiguous()
    hidden = {key: initial(tokens) for key, tokens in ids.items()}
    assets, checks = {}, []
    start = time.monotonic()
    for layer in range(45):
        block = source.load_block(layer, include_experts=True)
        if layer in LAYERS:
            assets[layer] = capture(out, layer, hidden)
        for key, value in hidden.items():
            state = dict(hidden=value)
            if layer in LAYERS:
                state, _ = native_staged_boundary(math, block, state)
            else:
                state = math.native_forward(block, state)
            hidden[key] = state["hidden"]
        if layer == 3:
            # Use the long witness only for decoder parity. The first 513 complete
            # pools already exceed the 512-pool budget and exercise real selection.
            witness = json.loads((root / "corpus/long-context-smoke.json").read_text())
            tokens = torch.tensor([witness["input_ids"][-2052:]], device="cuda")
            if witness["hessian_eligible"] or tokens.shape[1] != 2052:
                raise ValueError("missing sparse DSA parity witness")
            _, sparse = native_staged_boundary(math, block, dict(hidden=initial(tokens)))
            if not sparse or not sparse["sparse_selection"]:
                raise ValueError("DSA did not select a sparse subset")
        if layer in LAYERS:
            checks.append(layer)
        print(json.dumps(dict(layer=layer, elapsed=time.monotonic()-start)), flush=True)
        del block
    cfg = source.config
    head_norm = Glm5NextTextRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"]).cuda()
    head_norm.weight = torch.nn.Parameter(source.tensor(PREFIX+"norm.weight", "cuda"), requires_grad=False)
    head_input = {key: head_norm(Glm5NextTextHyperHead()(value)) for key, value in hidden.items()}
    assets[46] = capture(out, 46, head_input)
    adapter = mtp.load_mtp(source, include_experts=True)
    envelopes = {}
    for key, value in hidden.items():
        aligned = mtp.aligned_inputs(value, ids[key], embedding, head_norm)
        envelope = adapter.envelope(aligned)
        positions = torch.arange(1, ids[key].shape[1], device="cuda").unsqueeze(0)
        expected = torch.cat((adapter.enorm(F.embedding(ids[key][:, 1:], embedding)),
                              adapter.hnorm(head_input[key][:, :-1])), dim=-1)
        if not torch.equal(expected, envelope) or not torch.equal(positions, aligned["position_ids"]):
            raise ValueError("MTP alignment differs from shifted-token native envelope")
        native_staged_boundary(math, adapter.decoder, dict(hidden=adapter.eh_proj(envelope), position_ids=positions))
        envelopes[key] = envelope
    assets[45] = capture(out, 45, envelopes)
    source.close()
    atomic_json(root / "parity.json", dict(status="passed", source=snapshot.name, layers=checks+[45],
        probes=list(ids), sparse=sparse, corpus_sha256=digest(root/"corpus/manifest.json"),
        captures_sha256=assets, parity_code_sha256=digest(Path(__file__)), elapsed=time.monotonic()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(10)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    run(args.snapshot, args.root)
