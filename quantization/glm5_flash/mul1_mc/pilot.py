"""Gate representative GLM Flash MUL1 projections using attested real prompt tails."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from glm5_flash.core_k6.recipe import reviewed_core_modules
from inkling2.numerical import fork_module
from kimi_k3.uniform_k2.storage import atomic_json
from . import experts, numerics

LAYERS = (0, 3, 22, 23, 44, 45, 46)
PREFIX = "model.language_model.layers."
REPO = Path(__file__).resolve().parents[3]
BUDGET_SECONDS = 1800
EH_PROJ = PREFIX + "45.eh_proj"
PREVIOUS_CODE = "1283bc376eee3fa287a79145991cd50f4782bd13f8f0a7603e53b8f9c435cae4"
PREVIOUS_HASHES = {
    "pilot-events.jsonl": "b55636f843563d1242e770e2175f54491c7048f280036165580b904f105b6b0b",
    "pilot-stop.json": "0f896a2908047cfb01d78c9bbf6f8adf16741463d947d1e4862f13793b176f73",
    "parity.json": "7d326b6f6e3a3f8bb83a862027fa1a7ed2dce0b216d37d36a14224b5dce7eb28",
    "corpus/manifest.json": "d2d9924c343d660d758298c583caf57004b40d5990bac901523ca3154f4917d2",
    "source-original-pilot.json": "1f614a590e9ee309a6d17ba05d64df0e7dd82797ce5e8f1008abd15707f241a4",
}


def restore_previous(root):
    """Replay only measurements sealed before the reviewed MTP quality stop."""
    for filename, expected in PREVIOUS_HASHES.items():
        if digest(root / filename) != expected:
            raise ValueError(f"stopped pilot receipt differs: {filename}")
    stop = json.loads((root / "pilot-stop.json").read_text())
    old_source = json.loads((root / "source-original-pilot.json").read_text())
    source = json.loads((root / "source.json").read_text())
    for key in ("files", "config_sha256", "manifest_sha256", "revision", "shards", "source_payload_bytes"):
        if source[key] != old_source[key]:
            raise ValueError(f"stopped pilot source differs: {key}")
    rows = [json.loads(line) for line in (root / "pilot-events.jsonl").read_text().splitlines()]
    if (stop["failed_module"] != EH_PROJ or stop["completed"] != len(rows) or len(rows) != 82
            or len({(row["module"], row["bits"]) for row in rows}) != len(rows)):
        raise ValueError("stopped pilot frontier differs")
    return rows


def expected_core_arms():
    prefixes = tuple(f"{PREFIX}{layer}." for layer in LAYERS)
    return {(name, bits) for name in reviewed_core_modules()
            if name != EH_PROJ and (name == "lm_head" or name.startswith(prefixes)) for bits in (6, 8)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tp2_payload(name, shape, bits):
    n, k = shape
    def size(n, k):
        padded = math.ceil(n / 128) * 128
        return padded*k*bits//8 + 2*(padded + k) + 4
    if name == "lm_head":
        return [size(n, k), 0]
    if name.endswith(("q_a_proj", "kv_a_proj_with_mqa", "eh_proj")):
        return [size(n, k)] * 2
    if name.endswith(("o_proj", "down_proj")):
        return [size(n, k//2)] * 2
    return [size(n//2, k)] * 2


def nmse(inputs, native, candidate):
    """FP32 reference projection on disjoint held-out activations, bounded over outputs."""
    x = inputs.reshape(-1, inputs.size(-1)).float()
    sums = torch.zeros(2, dtype=torch.float64, device=x.device)
    for w, q in zip(native.split(2048), candidate.split(2048), strict=True):
        expected = x @ w.float().T
        diff = x @ q.float().T - expected
        sums[0].add_(diff.square().sum(dtype=torch.float64))
        sums[1].add_(expected.square().sum(dtype=torch.float64))
    a, b = sums.tolist()
    if not (math.isfinite(a) and math.isfinite(b) and b > 0):
        raise ValueError("invalid held-out projection energy")
    return a/b


def as_rows(samples):
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in samples])


class Pilot:
    def __init__(self, snapshot, root):
        self.root, self.snapshot = root, snapshot
        self.source_report = json.loads((root/"source.json").read_text())
        self.parity = json.loads((root/"parity.json").read_text())
        self.corpus = json.loads((root/"corpus/manifest.json").read_text())
        if (self.source_report["status"] != "passed" or self.source_report["source"] != str(snapshot.resolve())
                or self.parity["status"] != "passed" or self.parity["source"] != snapshot.name
                or self.parity["layers"] != list(LAYERS[:-1])
                or not self.parity["sparse"]["sparse_selection"]
                or self.parity["corpus_sha256"] != digest(root/"corpus/manifest.json")):
            raise ValueError("pilot requires passed native parity for this source and corpus")
        self.math = fork_module("utils.glm5_flash_core")
        self.source = self.math.GlmFlashBF16Source(snapshot)
        self.device = torch.device("cuda:0")
        self.start = time.monotonic()
        self.modules = []
        self.recovery, self.selected = {}, {}
        self.expert_workspace_bytes = 0
        self.events = self.root / "pilot-events.jsonl"
        self.recovery_scope = {}
        self.continuation = None

    def record(self, metrics, shape, held_rows):
        metrics.update(native_nmse=0., native_bytes=math.prod(shape)*2,
                       heldout_rows=held_rows, tp2_payload_bytes=tp2_payload(
                           metrics["module"], shape, metrics["bits"]))
        self.modules.append(metrics)
        print(json.dumps({k:metrics[k] for k in ("module", "bits", "heldout_nmse", "heldout_rows")}), flush=True)
        with self.events.open("a") as events:
            events.write(json.dumps(metrics, allow_nan=False)+"\n")
        if time.monotonic()-self.start >= BUDGET_SECONDS:
            raise TimeoutError("GLM Flash pilot exceeded 30 minute budget")

    def fit(self, name, weight, values, *, install=True):
        train = [v for key, v in values.items() if key.startswith("train_")]
        held = as_rows([v for key, v in values.items() if key.startswith("held_")])
        acc = numerics.covariance_accumulator(weight.shape[1], self.device)
        for value in train:
            acc.add(value)
        raw = numerics.raw_hessian(acc)
        del acc
        res = numerics.encode_core(name, weight, raw, self.device,
            evaluate=lambda bits, reconstructed: nmse(held, weight, reconstructed))
        for metrics in res.metrics.values():
            self.record(metrics, weight.shape, len(held))
        if install:
            weight.copy_(res.reconstruction)

    def expert(self, layer, block, values):
        def routes(kind):
            for key, hidden in values.items():
                if key.startswith(kind):
                    hidden = hidden.reshape(-1, hidden.size(-1))
                    yield hidden, *experts.route_batch(block.mlp, hidden)
        def bank(kind):
            count = sum(v.shape[1] for k,v in values.items() if k.startswith(kind))
            return experts.ExpertBank(routes(kind), tokens=count, device=self.device)
        train, held = bank("train_"), bank("held_")
        receipts = list(train.receipts())
        self.recovery[str(layer)] = receipts
        self.recovery_scope[str(layer)] = "all-experts"
        # Evaluate a routed expert with real held-out rows. Break ties by ID.
        expert = max(range(train.experts), key=lambda e: (
            min(held.native.bounds[e+1]-held.native.bounds[e], receipts[e]["natural_rows"]), -e))
        selected = held.native.rows[held.native.part(expert)]
        if not len(selected):
            raise ValueError("no naturally routed held-out expert witness")
        self.selected[str(layer)] = expert
        prefix = f"{PREFIX}{layer}.mlp.experts.{expert}"
        parts = experts._PARTS
        weights = {part:self.source.tensor(f"{prefix}.{part}.weight", self.device) for part in parts}
        limit = self.source.config["swiglu_limit"]
        workspace = experts.ExpertWorkspace.allocate(4096, 2048, device=self.device)
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        packed, metrics = experts.fit_expert(train, expert, weights, prefix=prefix, workspace=workspace,
                                             swiglu_limit=limit)
        delta = torch.cuda.max_memory_allocated() - baseline
        self.expert_workspace_bytes = max(self.expert_workspace_bytes, delta + sum(
            v.numel()*v.element_size() for v in vars(workspace).values()))
        inputs = held.inputs.index_select(0, selected)
        fused = experts._gate_up(packed, workspace)
        act = torch.cat([workspace.activate(chunk, fused, limit).clone()
                         for chunk in inputs.split(len(workspace.gather))]).float()
        # The down Hessian and its held-out metric measure weighted down outputs.
        act *= held.native.gamma[held.native.part(expert)][:, None]
        for part in parts:
            w = weights[part]
            restored = numerics.reconstruct(packed[part], w.shape, self.device)
            error = nmse(act if part=="down_proj" else inputs, w, restored)
            metrics[part].update(heldout_nmse=error, heldout_bar=numerics.ERROR_BARS[4],
                                 **{k:v for k,v in receipts[expert].items() if k!="identity_rows"})
            self.record(metrics[part], w.shape, len(selected))
            if error > numerics.ERROR_BARS[4]:
                raise ValueError(f"{prefix}.{part} K4 held-out NMSE {error} exceeds 0.01")
        self.source.close()

    def layer(self, layer):
        path = self.root/f"pilot-inputs/layer-{layer:02}.safetensors"
        if digest(path) != self.parity["captures_sha256"][str(layer)]:
            raise ValueError(f"pilot input capture changed for layer {layer}")
        values = load_file(str(path), device=str(self.device))
        if set(values) != set(self.parity["probes"]):
            raise ValueError("pilot capture omitted a prompt")
        if layer == 46:
            self.fit("lm_head", self.source.tensor("lm_head.weight", self.device), values, install=False)
            return
        prefix = f"{PREFIX}{layer}."
        if layer == 45:
            adapter = fork_module("utils.glm5_flash_mtp").load_mtp(self.source, self.device)
            if adapter.eh_proj.weight.dtype != torch.bfloat16:
                raise ValueError("native MTP fusion must retain source BF16")
            values = {k:adapter.eh_proj(v) for k,v in values.items()}
            block = adapter.decoder
        else:
            block = self.source.load_block(layer, self.device)
        att = block.self_attn
        def fit(path, inputs):
            self.fit(prefix+path, self.math.module(block, path).weight, inputs)
        x = {k:self.math.attention_inputs(block,v) for k,v in values.items()}
        kda = block.block_type == "linear_attention"
        for part in (("q_proj","k_proj","v_proj") if kda else ("q_a_proj","kv_a_proj_with_mqa")):
            fit("self_attn."+part, x)
        if not kda:
            fit("self_attn.q_b_proj", {k:att.q_a_layernorm(att.q_a_proj(v)) for k,v in x.items()})
        o = {}
        for k, v in x.items():
            pos = torch.arange(1, v.shape[1]+1, device=self.device)[None,:] if layer==45 else None
            ctx = self.math.prepare_context(v, position_ids=pos)
            o[k] = self.math.capture_linear_input(att.o_proj, self.math.attention_forward, block, v, ctx)
        fit("self_attn.o_proj", o)
        mlp_inputs = {k:self.math.attention_finish(block, values[k], att.o_proj(v))[1] for k,v in o.items()}
        shared = block.mlp if layer == 0 else block.mlp.shared_experts
        base = "mlp." if layer == 0 else "mlp.shared_experts."
        for part in ("gate_proj", "up_proj"):
            fit(base+part, mlp_inputs)
        fit(base+"down_proj", {k:self.math.shared_down_input(shared,v) for k,v in mlp_inputs.items()})
        if layer >= 3:
            self.expert(layer, block, mlp_inputs)

    @torch.inference_mode()
    def run(self, *, resume_mtp=False):
        out = self.root/"pilot.json"
        if out.exists():
            raise ValueError("pilot receipt already exists")
        layers = LAYERS
        if resume_mtp:
            self.modules = restore_previous(self.root)
            layers = (45, 46)
            self.continuation = dict(previous_pilot_code_sha256=PREVIOUS_CODE,
                                     hashes=PREVIOUS_HASHES, reused_measurements=len(self.modules),
                                     layers=list(layers))
            for row in self.modules:
                if row["bits"] != 4 or not row["module"].endswith(".gate_proj"):
                    continue
                label = row["module"].removeprefix(PREFIX).split(".", 1)[0]
                self.selected[label] = row["expert"]
                self.recovery[label] = [row]
                self.recovery_scope[label] = "selected-expert-only"
            self.events = self.root / "pilot-continuation-events.jsonl"
        if self.events.exists():
            raise ValueError(f"refuse to append a second attempt to {self.events.name}")
        for layer in layers:
            self.layer(layer)
            torch.cuda.empty_cache()
        actual = {(row["module"], row["bits"]) for row in self.modules if row["bits"] in (8,6)}
        if actual != expected_core_arms():
            raise ValueError("pilot core inventory differs")
        tokens = sum(c["tokens"] for c in self.corpus["components"].values())
        width, inter = 4096, 2048
        ws = self.expert_workspace_bytes
        plan = experts.memory_plan(tokens, width, frontier_bytes=tokens*4*width*2, input_dtype=torch.bfloat16,
            weight_bytes=3*width*inter*2, hessian_bytes=width**2*16, workspace_bytes=ws,
            container_remaining=100<<30)
        if not plan["admitted"]:
            raise MemoryError(f"full routed working set exceeds unified memory allowance: {plan}")
        fork_revision = json.loads((REPO/"third_party/gptqmodel.lock.json").read_text())["revision"]
        eh_policy = dict(module=EH_PROJ, stored_dtype="BF16", views=["original","k8-compact","k6-compact"])
        if resume_mtp:
            stop = json.loads((self.root / "pilot-stop.json").read_text())
            eh_policy.update(prior_k8_nmse=stop["heldout_nmse"], prior_k8_bar=stop["limit"])
        atomic_json(out, dict(schema="glm5-flash-mul1-pilot-v1", status="passed", layers=list(LAYERS),
            source_revision=self.snapshot.name, fork_revision=fork_revision, corpus_sha256=self.parity["corpus_sha256"],
            parity_sha256=digest(self.root/"parity.json"), pilot_code_sha256=digest(Path(__file__)),
            capture_sha256=self.parity["captures_sha256"], elapsed_seconds=time.monotonic()-self.start,
            modules=self.modules, mtp_eh_bits=[16,16], mtp_eh_policy=eh_policy,
            continuation=self.continuation, recovery_scope=self.recovery_scope,
            recovery=self.recovery, selected_experts=self.selected,
            memory_plan=plan, expert_workspace_bytes=ws, view_payload_bytes=self.source_report["view_payload_bytes"],
            predecessor="native BF16 target frontier; within-layer serialized K8 propagation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--resume-mtp", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(10)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    Pilot(args.snapshot, args.root).run(resume_mtp=args.resume_mtp)


if __name__ == "__main__":
    main()
