"""Causal GLM Flash layer traversal for the direct three-view writer."""
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import xxhash

import sys
from inkling2.numerical import fork_module
from kimi_k3.uniform_k2.storage import atomic_json
from . import experts, numerics

PREFIX = "model.language_model."
PARTS = ("gate_proj", "up_proj", "down_proj")
SUFFIXES = ("trellis", "suh", "svh", "mul1")


def value_hash(tensor):
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return xxhash.xxh3_128_hexdigest(memoryview(raw))


class LayerPass:
    def __init__(self, source, stages, writer, identity, tokens, device="cuda:0"):
        self.source, self.stages, self.writer = source, stages, writer
        self.identity, self.device = identity, torch.device(device)
        self.tokens = [ids.reshape(1, -1) for ids in tokens]
        self.math = fork_module("utils.glm5_flash_core")
        self.metrics_dir = stages.root.parent / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

    def rows(self, stage, *keys, start=0):
        return self.stages.records(stage, device=self.device, keys=keys or None, start=start)

    def save(self, stage, generate):
        def checked(start):
            for index, values in generate(start):
                finite = [torch.isfinite(t).all() for t in values.values() if t.is_floating_point()]
                if finite and not bool(torch.stack(finite).all()):
                    raise ValueError(f"{stage} batch {index} has nonfinite activations")
                yield index, values
        if self.stages.load(stage) is None:
            self.stages.write_activations(stage, checked, len(self.tokens))
        return stage

    def initial(self):
        def batches(start):
            embedding = self.source.tensor(PREFIX+"embed_tokens.weight", self.device)
            for i in range(start, len(self.tokens)):
                hidden = F.embedding(self.tokens[i].to(self.device), embedding)
                yield i, dict(hidden=hidden.unsqueeze(2).expand(-1, -1, 4, -1))
        return self.save("initial", batches)

    def fit(self, stage, modules, inputs):
        """Capture once per input boundary, seal both widths, then install K8."""
        metrics = []
        hashes = {}
        def records():
            first = next(iter(modules.values()))
            accumulator = numerics.covariance_accumulator(first.in_features, self.device)
            for value in inputs():
                accumulator.add(value)
            raw = numerics.raw_hessian(accumulator)
            del accumulator
            for index, (name, module) in enumerate(modules.items()):
                result = numerics.encode_core(name, module.weight, raw, self.device)
                packed = {f"k{bits}.{name}.{suffix}": tensor for bits, tensors in result.packed.items()
                          for suffix, tensor in tensors.items()}
                hashes.update({key: value_hash(tensor) for key, tensor in packed.items()})
                metrics.extend(result.metrics.values())
                module.weight.copy_(result.reconstruction)
                del result
                yield index, packed
        meta = dict(names=list(modules), metrics=metrics, hashes=hashes)
        manifest = self.stages.load(stage)
        if manifest is None:
            self.stages.write_batches(stage, records(), meta)
            return stage
        if manifest["metadata"]["names"] != list(modules):
            raise ValueError(f"compact checkpoint modules differ in {stage}")
        for index, (name, module) in enumerate(modules.items()):
            keys = [f"k8.{name}.{suffix}" for suffix in SUFFIXES]
            values = self.stages.fields(stage, index, keys)
            packed = dict(zip(SUFFIXES, (values[key] for key in keys), strict=True))
            module.weight.copy_(numerics.reconstruct(packed, module.weight.shape, self.device))
        return stage

    def publish_layer(self, label, fit_stages):
        packed = {4: {}, 6: {}, 8: {}}
        hashes = {4: {}, 6: {}, 8: {}}
        rows = []
        for stage in fit_stages:
            manifest = self.stages.load(stage)
            meta = manifest["metadata"]
            rows.extend(meta["metrics"])
            for i, group in enumerate(meta["names"]):
                for name in ([group] if isinstance(group, str) else group):
                    for bits in ((4,) if ".mlp.experts." in name else (8, 6)):
                        for suffix in SUFFIXES:
                            leaf = f"{name}.{suffix}"
                            key = f"k{bits}.{leaf}"
                            packed[bits][leaf] = self.stages.reference(stage, i, key)
                            hashes[bits][leaf] = meta["hashes"][key]
        self.writer.write_group(label, shared=packed[4], k8=packed[8], k6=packed[6],
            packed_hashes={"shared": hashes[4], "k6-compact": hashes[6], "k8-compact": hashes[8]})
        atomic_json(self.metrics_dir / f"{label}.json", rows)

    def routed(self, block, label, prefix, mlp_stage, count):
        width = self.source.config["hidden_size"]
        streams = () if block.layer_idx == 45 else (self.source.config["hc_mult"],)
        residual_bytes = count * width * 2
        if streams:
            residual_bytes *= streams[0]
        intermediate = self.source.config["moe_intermediate_size"]
        # Return unused core-fit buffers to unified memory before admitting the routed phase.
        torch.cuda.empty_cache()
        plan = experts.memory_plan(count, width, frontier_bytes=residual_bytes, input_dtype=torch.bfloat16,
            weight_bytes=3*width*intermediate*2, hessian_bytes=width**2*16,
            workspace_bytes=self.identity["expert_workspace_bytes"], container_remaining=self.container_remaining())
        if not plan["admitted"]:
            raise MemoryError(f"unified RAM cannot admit {label}: {plan}")
        atomic_json(self.metrics_dir.parent / f"{label}-memory.json", plan)
        residuals = torch.empty((count, *streams, width), dtype=torch.bfloat16, device=self.device)
        def routes():
            offset = 0
            for _, record in self.rows(mlp_stage):
                hidden = record["normalized"].reshape(-1, width)
                residual = record["residual"].squeeze(0)
                if residual.dtype != residuals.dtype or residual.shape != (len(hidden), *streams, width):
                    raise ValueError("GLM residual storage would change shape or precision")
                residuals[offset:offset+len(hidden)].copy_(residual)
                offset += len(hidden)
                yield hidden, *experts.route_batch(block.mlp, hidden)
        bank = experts.ExpertBank(routes(), tokens=count, device=self.device, input_dtype=torch.bfloat16)
        evidence = list(bank.receipts())
        work = experts.ExpertWorkspace.allocate(width, intermediate, device=self.device)
        stages = []
        def replay(expert, _rows):
            base = expert // 16 * 16
            stage = f"{label}-experts-{base:03d}"
            if expert == base:
                stages.append(stage)
                names, metrics, hashes = [], [], {}
                def fit_group():
                    for idx in range(base, min(base+16, bank.experts)):
                        name = prefix+f"mlp.experts.{idx}"
                        weights = {part: self.source.tensor(f"{name}.{part}.weight") for part in PARTS}
                        outputs, rows = experts.fit_expert(bank, idx, weights, prefix=name, workspace=work,
                            swiglu_limit=block.mlp.experts.swiglu_limit)
                        names.append([f"{name}.{part}" for part in PARTS])
                        metrics.extend(rows[part] | evidence[idx] for part in PARTS)
                        packed = {f"k4.{name}.{part}.{suffix}": tensor for part in PARTS
                                  for suffix, tensor in outputs[part].items()}
                        hashes.update({key: value_hash(tensor) for key, tensor in packed.items()})
                        yield idx-base, packed
                self.stages.write_batches(stage, fit_group(), dict(names=names, metrics=metrics, hashes=hashes))
                encoder = sys.modules.get("gptqmodel.exllamav3.modules.quant.exl3_lib.quantize")
                print(json.dumps(dict(event="expert_group_complete", layer=label, first=base,
                    trellis_workspace_count=len(encoder._workspace_buffers) if encoder else 0,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(self.device))), flush=True)
            values = self.stages.read_batch(stage, expert-base)
            name = prefix+f"mlp.experts.{expert}"
            tensors = {part: {s: values[f"k4.{name}.{part}.{s}"] for s in SUFFIXES} for part in PARTS}
            return experts.replay_expert(bank, expert, tensors, workspace=work,
                                         swiglu_limit=block.mlp.experts.swiglu_limit)
        combined = bank.combine(replay, weighted_buffer=work.gather)
        return combined, bank.inputs, residuals, stages

    @staticmethod
    def container_remaining():
        cgroup = Path("/sys/fs/cgroup")
        limit = (cgroup / "memory.max").read_text().strip()
        if limit == "max":
            return None
        stats = dict(line.split() for line in (cgroup/"memory.stat").read_text().splitlines())
        reclaimable = max(0, int(stats["file"])-int(stats["file_dirty"])-int(stats.get("shmem", 0)))
        return int(limit)-int((cgroup/"memory.current").read_text())+reclaimable

    @torch.inference_mode()
    def target(self, index, incoming):
        block = self.source.load_block(index, self.device)
        return self.block(index, incoming, block)

    def block(self, index, incoming, block, *, extra_fits=()):
        label = f"layer-{index:02d}"
        successor = label+"-next"
        if (self.writer.group_complete(label) and self.stages.load(successor)
                and (self.metrics_dir / f"{label}.json").is_file()):
            return successor
        prefix = PREFIX+f"layers.{index}."
        fit_stages = list(extra_fits)
        def fit(stage, leaves, inputs):
            modules = {prefix+name: self.math.module(block, name) for name in leaves}
            fit_stages.append(self.fit(label+"-"+stage, modules, inputs))
        attn = block.self_attn
        def a_batches(start):
            for i, state in self.rows(incoming, start=start):
                yield i, dict(x=self.math.attention_inputs(block, state["hidden"]),
                              **{k:v for k,v in state.items() if k=="position_ids"})
        a = self.save(label+"-attention", a_batches)
        def samples(stage, key):
            return (record[key] for _, record in self.rows(stage, key))
        kda = block.block_type == "linear_attention"
        first = ("q_proj","k_proj","v_proj") if kda else ("q_a_proj","kv_a_proj_with_mqa")
        fit("qkv-fit", ["self_attn."+part for part in first], lambda: samples(a, "x"))
        if not kda:
            fit("qb-fit", ["self_attn.q_b_proj"],
                lambda: (attn.q_a_layernorm(attn.q_a_proj(x)) for x in samples(a, "x")))
        def o_batches(start):
            for i, state in self.rows(a, start=start):
                ctx = self.math.prepare_context(state["x"], position_ids=state.get("position_ids"))
                value = self.math.capture_linear_input(attn.o_proj, self.math.attention_forward, block, state["x"], ctx)
                yield i, dict(x=value)
        o = self.save(label+"-o-input", o_batches)
        fit("o-fit", ["self_attn.o_proj"], lambda: samples(o, "x"))
        def m_batches(start):
            for (i, state), (_, inp) in zip(self.rows(incoming, "hidden", start=start),
                                            self.rows(o, "x", start=start), strict=True):
                residual, normalized = self.math.attention_finish(block, state["hidden"], attn.o_proj(inp["x"]))
                yield i, dict(residual=residual, normalized=normalized)
        m = self.save(label+"-mlp", m_batches)
        shared = block.mlp if index < 3 else block.mlp.shared_experts
        mlp = "mlp." if index < 3 else "mlp.shared_experts."
        fit("gate-up-fit", [mlp+"gate_proj", mlp+"up_proj"], lambda: samples(m, "normalized"))
        fit("down-fit", [mlp+"down_proj"],
            lambda: (self.math.shared_down_input(shared,x) for x in samples(m,"normalized")))
        lengths = [ids.numel() - int(index==45) for ids in self.tokens]
        combined = normalized_bank = residuals = None
        if index >= 3:
            combined, normalized_bank, residuals, expert_stages = self.routed(block, label, prefix, m, sum(lengths))
            fit_stages.extend(expert_stages)
        def final_batches(start):
            offset = sum(lengths[:start])
            def states():
                if combined is None:
                    yield from self.rows(m, start=start)
                else:
                    cursor = offset
                    for i in range(start, len(lengths)):
                        part = slice(cursor, cursor+lengths[i])
                        yield i, dict(normalized=normalized_bank[part].unsqueeze(0),
                                      residual=residuals[part].unsqueeze(0))
                        cursor += lengths[i]
            for i, state in states():
                x = state["normalized"]
                output = shared(x)
                if combined is not None:
                    length = lengths[i]
                    routed = combined.narrow(0, offset, length).view_as(x).to(output.dtype)
                    output = routed + output
                    offset += length
                residual = state["residual"]
                hidden = (self.math._expand(block.ffn_hc, residual, output)
                          if hasattr(block, "ffn_hc") else residual + output)
                yield i, dict(hidden=hidden)
        self.save(successor, final_batches)
        self.publish_layer(label, fit_stages)
        self.stages.retire(a, o, m, *fit_stages)
        return successor

    def norm(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
        cfg = self.source.config
        norm = Glm5NextTextRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"]).to(self.device)
        norm.weight = torch.nn.Parameter(self.source.tensor(PREFIX+"norm.weight", self.device),requires_grad=False)
        return norm

    @torch.inference_mode()
    def head(self, incoming):
        if self.writer.group_complete("head") and (self.metrics_dir/"head.json").exists():
            return
        weight = self.source.tensor("lm_head.weight", self.device)
        with torch.device("meta"):
            head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        head.weight = torch.nn.Parameter(weight, requires_grad=False)
        norm = self.norm()
        fitted = self.fit("head-fit", {"lm_head":head},
                          lambda: (norm(r["hidden"].mean(2)) for _,r in self.rows(incoming, "hidden")))
        self.publish_layer("head", [fitted])
        self.stages.retire(fitted)

    @torch.inference_mode()
    def mtp(self, incoming):
        if (self.writer.group_complete("layer-45") and self.stages.load("layer-45-next")
                and (self.metrics_dir / "layer-45.json").is_file()):
            return "layer-45-next"
        mtp = fork_module("utils.glm5_flash_mtp")
        adapter = mtp.load_mtp(self.source, self.device)
        norm = self.norm()
        embedding = self.source.tensor(PREFIX+"embed_tokens.weight", self.device)
        def envelope(start):
            for i, row in self.rows(incoming,"hidden",start=start):
                aligned = mtp.aligned_inputs(row["hidden"], self.tokens[i].to(self.device), embedding, norm)
                yield i, dict(x=adapter.envelope(aligned), position_ids=aligned["position_ids"])
        stage = self.save("mtp-envelope", envelope)
        fitted = self.fit("mtp-envelope-fit", {PREFIX+"layers.45.eh_proj": adapter.eh_proj},
                          lambda: (r["x"] for _,r in self.rows(stage,"x")))
        def fused(start):
            for i, row in self.rows(stage,start=start):
                yield i, dict(hidden=adapter.eh_proj(row["x"]), position_ids=row["position_ids"])
        start = self.save("mtp-fused", fused)
        result = self.block(45, start, adapter.decoder, extra_fits=[fitted])
        self.stages.retire(stage,start,fitted)
        return result
