"""Full-corpus K8-causal ERNIE target and dSpark layers."""
from unittest.mock import patch

import torch

from hy4_flash import gates
from hy4_flash.run import encode, raw, reconstruction
from inkling2 import numerical
from .forward import MlpInput, reference as ref

ATTN = "self_attn."
GROUP = ATTN + "linear_o_group_proj"
MOE = "block_sparse_moe."
SUFFIXES = ("trellis", "suh", "svh", "mul1")


def project(_, x, weight):
    return ref.linear(x, weight)


class LayerCore:
    """Seal each dependency group, then install just its serialized K8 reconstruction."""

    def __init__(self, model, weights, stages, label, prefix, counts, event):
        self.model, self.weights, self.stages = model, weights, stages
        self.config, self.device = model.config, model.checkpoint.device
        self.label, self.prefix, self.counts, self.event = label, prefix, counts, event
        self.train = counts["calibration"]
        self.total = self.train + counts["heldout"]
        self.stage_names = []
        self.packed = {8: {}, 6: {}}
        self.metrics = []

    def weight(self, name):
        if name.startswith(GROUP + ".groups."):
            group = int(name.rsplit(".", 1)[1])
            return self.weights[GROUP].view(self.config.o_groups, self.config.o_lora_rank, -1)[group]
        return self.weights[name + ".weight"]

    def fit(self, tag, groups, samples, bootstrap=None):
        """Each tuple of module names shares exactly one raw input covariance."""
        groups = [tuple(g) for g in groups]
        names = [name for group in groups for name in group]
        full = [self.prefix + name for name in names]
        stage = self.label + "-" + tag
        metrics = []
        def produce():
            if bootstrap is not None and all("packed" in bootstrap[module + ".weight"] for module in full):
                for i, module in enumerate(full):
                    entry = bootstrap[module + ".weight"]
                    pair = entry["metrics"]
                    if ([(r["module"], r["bits"]) for r in pair] != [(module, 8), (module, 6)]
                            or not all("heldout_nmse" in r and r["hessian_rows"] == entry["count"]
                                       for r in pair)):
                        raise ValueError(f"{module}: invalid bootstrap metrics")
                    packed = entry["packed"][8] | entry["packed"][6]
                    if set(packed) != {f"k{b}.{module}.{s}" for b in (8, 6) for s in SUFFIXES}:
                        raise ValueError(f"{module}: incomplete bootstrap encode")
                    metrics.extend(pair)
                    yield i, packed
                return
            accumulators = [None if bootstrap is not None else numerical.covariance_accumulator(
                self.weight(group[0]).shape[1], device=self.device) for group in groups]
            tails = [[] for _ in groups]
            start = self.train if bootstrap is not None else 0
            seen = start
            self.event("core_covariance", stage=stage, projections=full)
            for i, inputs in samples(start):
                if i != seen:
                    raise ValueError(f"{stage}: prompt order mismatch")
                seen += 1
                for n, (x, acc) in enumerate(zip(inputs, accumulators, strict=True)):
                    if len(x):
                        if i < self.train:
                            acc.add(x)
                        else:
                            tails[n].append(x[-gates.TAIL:].clone())
            if seen != self.total:
                raise ValueError(f"{stage}: missing calibration or held-out prompts")
            ordinal = 0
            for n, group in enumerate(groups):
                covariance = (bootstrap[self.prefix + group[0] + ".weight"]
                              if bootstrap is not None else raw(accumulators[n]))
                accumulators[n] = None
                if not tails[n]:
                    raise ValueError(f"{stage}: no held-out rows")
                witness = torch.cat(tails[n])
                tails[n].clear()
                for name in group:
                    module = self.prefix + name
                    weight = self.weight(name)
                    self.event("core_projection", stage=stage, projection=module)
                    packed = encode(module, weight, covariance, (8, 6), self.device, metrics)
                    for row in metrics[-2:]:
                        decoded = reconstruction(packed, module, row["bits"], weight.shape, self.device)
                        gates.evaluated(row, witness, weight, decoded, "last-16-rows-each-heldout-prompt",
                                        projection=project)
                    yield ordinal, packed
                    ordinal += 1
        manifest = self.stages.write_batches(stage, produce(), {"names": full, "metrics": metrics})
        if manifest["metadata"]["names"] != full:
            raise ValueError(f"{stage}: compact inventory mismatch")
        self.metrics.extend(manifest["metadata"]["metrics"])
        self.stage_names.append(stage)
        grouped = []
        for i, name in enumerate(names):
            module = self.prefix + name
            matrix = reconstruction(self.stages.read_batch(stage, i), module, 8, self.weight(name).shape, self.device)
            if name.startswith(GROUP + ".groups."):
                grouped.append(matrix)
            else:
                self.weights[name + ".weight"] = matrix
            for bits in (8, 6):
                for suffix in SUFFIXES:
                    key = module + "." + suffix
                    self.packed[bits][key] = self.stages.reference(stage, i, f"k{bits}.{key}")
        if grouped:
            if len(grouped) != self.config.o_groups:
                raise ValueError("incomplete ERNIE HCA group set")
            self.weights[GROUP] = torch.cat(grouped)

    def normalized_bank(self, records, total_tokens, draft):
        """Keep the post-mHC normalized input once. The 4-stream residual stays on NVMe."""
        bank = torch.empty((total_tokens, self.config.hidden_size), device=self.device, dtype=torch.float32)
        batches = []
        offset = 0
        for i, row in records(0):
            hidden = row["state"]
            _, _, x = ref.mhc_pre(hidden, self.weights, "self_attention_hyper_connection",
                                  self.weights["input_layernorm.weight"], self.config)
            target = bank[offset:offset+len(x)]
            target.copy_(x)
            offset += len(x)
            if i != len(batches):
                raise ValueError("attention input order differs from token stream")
            positions = row["positions"] if draft else torch.arange(len(x), device=self.device)
            context = row["context"] if draft else None
            context_pos = row["context_positions"] if draft else None
            batches.append((target, positions, context, context_pos))
        if offset != total_tokens or len(batches) != self.total:
            raise ValueError("attention input token or prompt count differs")
        return batches

    def attend(self, index, x, pos, ctx, ctx_pos):
        w, c = self.weights, self.config
        if ctx is None and c.is_dsa_layer(index):
            return ref.mla_dsa_attention(x, w, c, pos, enable_flashindexer=False)
        kv = None
        if ctx is not None:
            kv = ref.norm(ref.linear(ctx, w[ATTN+"linear_kv_proj.weight"]),
                          w[ATTN+"kv_layernorm.weight"], c.rms_norm_eps)
            cos, sin = ref.hca_phase(ctx_pos, c, rope_mode="plain")
            kv = ref.rotate_tail(kv[:, None], cos, sin).squeeze(1)
        return ref.hca_attention(x, w, c, pos, context_kv=kv, non_causal=ctx is not None,
                                 rope_mode="yarn" if ctx is None else "plain")

    def fit_attention(self, index, records, total_tokens, draft=False, bootstrap=None):
        tag = "attn-root"
        batches = self.normalized_bank(records, total_tokens, draft)
        dsa = not draft and self.config.is_dsa_layer(index)
        q = "q_a_proj" if dsa else "linear_q_down_proj"
        kv = "kv_a_proj_with_mqa" if dsa else "linear_kv_proj"
        root = [(ATTN + q, ATTN + kv)]
        if draft:
            root = [(ATTN + q,), (ATTN + kv,)]
        elif not dsa:
            root.append(tuple(ATTN + "core_attention.compressor.linear_" + n for n in ("wkv", "wgate")))
        def first(start):
            for i in range(start, len(batches)):
                x, _, ctx, _ = batches[i]
                if draft:
                    yield i, (x, torch.cat((ctx, x)))
                elif dsa:
                    yield i, (x,)
                else:
                    yield i, (x, x[:len(x)//ref.COMPRESS_RATIO_HCA*ref.COMPRESS_RATIO_HCA])
        self.fit(tag, root, first, bootstrap)
        norm = self.weights[ATTN + ("q_a_layernorm.weight" if dsa else "q_layernorm.weight")]
        def expand(start):
            for i in range(start, len(batches)):
                q_rows = ref.norm(ref.linear(batches[i][0], self.weight(ATTN + q)), norm, self.config.rms_norm_eps)
                yield i, (q_rows,)
        self.fit("attn-expand", [(ATTN + ("q_b_proj" if dsa else "linear_q_up_proj"), ATTN + "gate_proj")], expand)

        def capture(names):
            # Observe entire, unpadded inputs at the reference's linear boundary.
            # In grouped HCA each distinct slice has a distinct pointer and Hessian.
            ptrs = {self.weight(name).data_ptr(): i for i, name in enumerate(names)}
            linear = ref.linear
            def records_at_boundary(start):
                for i in range(start, len(batches)):
                    observed = [None] * len(names)
                    def observe(x, weight):
                        j = ptrs.get(weight.data_ptr())
                        if j is not None:
                            if observed[j] is not None:
                                raise ValueError("repeated attention input in the same prompt")
                            observed[j] = x
                        return linear(x, weight)
                    with patch.object(ref, "linear", observe):
                        self.attend(index, *batches[i])
                    if any(x is None for x in observed):
                        raise ValueError("missing attention input at the reference boundary")
                    yield i, observed
            return records_at_boundary
        if not dsa:
            grouped = [f"{GROUP}.groups.{g}" for g in range(self.config.o_groups)]
            self.fit("attn-grouped", [(n,) for n in grouped], capture(grouped))
        out = ATTN + "o_proj"
        self.fit("attn-out", [(out,)], capture([out]))

    def prepare_mlp(self, index, records, draft):
        name = self.label + "-attention-residual"
        def residual_batches(start):
            for i, row in records(start):
                h = row["state"]
                w = self.weights
                post, comb, x = ref.mhc_pre(h, w, "self_attention_hyper_connection",
                                            w["input_layernorm.weight"], self.config)
                pos = row["positions"] if draft else torch.arange(len(x), device=self.device)
                output = self.attend(index, x, pos, row.get("context"), row.get("context_positions"))
                residual = ref.mhc_post(output, h, post, comb)
                if not bool(torch.isfinite(residual).all()):
                    raise ValueError(f"{name}: nonfinite attention output")
                yield i, {"residual": residual}
        self.stages.write_activations(name, residual_batches, self.total)
        self.stage_names.append(name)
        def states(start):
            for i, row in self.stages.records(name, start=start, device=str(self.device)):
                hidden = row["residual"]
                post, comb, x = ref.mhc_pre(hidden, self.weights, "mlp_hyper_connection",
                                            self.weights["post_attention_layernorm.weight"], self.config)
                yield i, MlpInput(hidden, post, comb, x)
        return states

    def mlp_inputs(self, states, token_count):
        # One normalized GPU row per token (~25 GiB), independent of the four
        # mHC streams. Release it after the route banks consume their rows.
        bank = torch.empty((token_count, self.config.hidden_size), dtype=torch.float32, device=self.device)
        rows = []
        offset = 0
        for i, state in states(0):
            if i != len(rows):
                raise ValueError("MLP prompt order differs")
            end = offset + len(state.x)
            row = bank[offset:end]
            row.copy_(state.x)
            rows.append(row)
            offset = end
        if offset != token_count or len(rows) != self.total:
            raise ValueError("MLP row bank differs from calibration stream")
        return rows

    def fit_mlp(self, rows, dense):
        first = (("mlp.gate_proj", "mlp.up_proj") if dense else
                 (MOE+"shared_experts.w1", MOE+"shared_experts.w3", MOE+"fc1_latent_proj"))
        self.fit("mlp-root", [first], lambda start: ((i, (rows[i],)) for i in range(start, self.total)))
        def active(start):
            for i in range(start, self.total):
                yield i, (self.model.activated(rows[i], self.weight(first[0]), self.weight(first[1])),)
        down = "mlp.down_proj" if dense else MOE + "shared_experts.w2"
        self.fit("mlp-down", [(down,)], active)


@torch.inference_mode()
def process_layer(model, stages, family, index, records, counts, tokens, event, bootstrap=None):
    """Advance one full layer. Caller seals its successor before retiring returned stages."""
    if family not in ("layers", "mtp"):
        raise ValueError(f"unknown ERNIE block family {family}")
    draft = family == "mtp"
    weights = model.load_draft_layer(index) if draft else model.load_layer(index)
    label = f"{family}-{index:03d}"
    prefix = f"model.{family}.{index}."
    worker = LayerCore(model, weights, stages, label, prefix, counts, event)
    worker.fit_attention(index, records, sum(tokens.values()), draft, bootstrap)
    states = worker.prepare_mlp(index, records, draft)
    dense = not draft and not model.config.is_moe_layer(index)
    rows = worker.mlp_inputs(states, sum(tokens.values()))
    worker.fit_mlp(rows, dense)
    k4 = {}
    if not dense:
        from .production_moe import encode_routed
        bounds = {"calibration": [0], "heldout": [0]}
        def bank(start, split):
            for j in range(counts[split]):
                x = rows[start+j]
                bounds[split].append(bounds[split][-1] + len(x))
                yield j, x
                rows[start+j] = None
        routed = encode_routed(model, weights, prefix, stages,
                               {"calibration": bank(0, "calibration"), "heldout": bank(worker.train, "heldout")},
                               tokens, event=event)
        worker.stage_names.extend(routed.stages)
        worker.metrics.extend(routed.metrics)
        k4 = routed.packed
        if any(len(bounds[s]) != counts[s]+1 or bounds[s][-1] != tokens[s] for s in bounds):
            raise ValueError("routed sums differ from prompt token boundaries")
        def summed(i):
            s = "calibration" if i < worker.train else "heldout"
            j = i if i < worker.train else i-worker.train
            return routed.sums[s][bounds[s][j]:bounds[s][j+1]]
        def normalized(start):
            for i in range(start, worker.total):
                yield i, (model.fc2_input(summed(i), weights),)
        worker.fit("mlp-fc2", [(MOE+"fc2_latent_proj",)], normalized)

    del rows

    @torch.inference_mode()
    def outputs(start):
        for i, state in states(start):
            h = model.finish_dense(state, weights) if dense else model.finish_moe(state, summed(i), weights)
            if not bool(torch.isfinite(h).all()):
                raise ValueError(f"{label}: nonfinite successor at prompt {i}")
            yield i, {"state": h}
    return dict(k4=k4, k8=worker.packed[8], k6=worker.packed[6],
                metrics=worker.metrics, stages=worker.stage_names, outputs=outputs)
