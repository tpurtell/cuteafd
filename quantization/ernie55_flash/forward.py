"""Layer-major calibration boundaries backed by the ERNIE 5.5 Flash F32 reference.

A production caller holds one layer of core weights. It can replace entries in
that dictionary with the decoded K8 weights as projections are sealed. Routed
weights come from the reader's lazy expert store or the caller's sealed K4
store. Natural expert outputs are reduced across the corpus before fc2.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch

from .routing import rank_routes

# Before the independent reference lands, put its checkout's python/reference
# on PYTHONPATH. Once landed, the in-tree reference is sufficient.
REFERENCE = Path(__file__).resolve().parents[2] / "python/reference"
if str(REFERENCE) not in sys.path:
    sys.path.append(str(REFERENCE))
from families.ernie55_flash import model as reference


_MOE = "block_sparse_moe."


@dataclass
class MlpInput:
    """The post-attention residual, its mHC mix and normalized MLP input."""
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    x: torch.Tensor

    def finish(self, output):
        return reference.mhc_post(output, self.residual, self.post, self.comb)


class ErnieForward:
    def __init__(self, checkpoint, config):
        self.checkpoint = checkpoint
        self.config = config
        if (config.num_experts_per_tok != 10 or not config.norm_topk_prob
                or config.scoring_func != "sigmoid" or config.topk_method != "noaux_tc"
                or not config.routed_scaling_factor_learnable or config.rope_theta != 8e6):
            raise ValueError("unsupported ERNIE calibration routing or RoPE contract")

    @classmethod
    def from_snapshot(cls, snapshot, device="cpu"):
        reference.configure_f32()
        config = reference.Config.parse(json.loads((Path(snapshot) / "config.json").read_bytes()))
        return cls(reference.Checkpoint(snapshot, device), config)

    def load_layer(self, index):
        if not 0 <= index < self.config.num_hidden_layers:
            raise IndexError("ERNIE target layer outside checkpoint")
        return self.checkpoint.layer(self.config, index, load_experts=False, load_flashindexer=False)

    def load_draft_layer(self, stage):
        if not 0 <= stage < self.config.num_mtp_stages:
            raise IndexError("ERNIE dSpark stage outside checkpoint")
        return self.checkpoint.mtp_stage(self.config, stage, load_experts=False)

    def native_layer(self, hidden, weights, index, positions=None):
        """Reference oracle for parity checks on a single prompt."""
        return reference.layer_forward(hidden, weights, self.config, index, positions,
                                       enable_flashindexer=False)

    def mlp_input(self, hidden, weights, index, positions, *, draft_context_kv=None):
        """Enter the dense or latent MLP after native/K8 attention and reference mHC."""
        c = self.config
        post, comb, inputs = reference.mhc_pre(
            hidden, weights, "self_attention_hyper_connection", weights["input_layernorm.weight"], c)
        if draft_context_kv is not None:
            attended = reference.hca_attention(
                inputs, weights, c, positions, context_kv=draft_context_kv, non_causal=True, rope_mode="plain")
        elif c.is_dsa_layer(index):
            attended = reference.mla_dsa_attention(inputs, weights, c, positions, enable_flashindexer=False)
        else:
            attended = reference.hca_attention(inputs, weights, c, positions, rope_mode="yarn")
        residual = reference.mhc_post(attended, hidden, post, comb)
        post, comb, x = reference.mhc_pre(
            residual, weights, "mlp_hyper_connection", weights["post_attention_layernorm.weight"], c)
        return MlpInput(residual, post, comb, x)

    def route_batch(self, inputs, weights):
        """Evaluate both F32 gates once; only fc1-projected rows enter the routed bank."""
        l0 = reference.linear(inputs, weights[_MOE + "gate.weight"])
        l1 = reference.linear(inputs, weights[_MOE + "gate.weight_1"])
        scores = l0.sigmoid() + l1.sigmoid()
        routes = rank_routes(scores, weights[_MOE + "e_score_correction_bias"],
                             weights[_MOE + "gate.routed_scaling_factor_param"])
        routes["latent"] = reference.linear(inputs, weights[_MOE + "fc1_latent_proj.weight"])
        return routes

    def activated(self, inputs, gate_weight, up_weight):
        c = self.config
        return reference.situ(reference.linear(inputs, gate_weight), reference.linear(inputs, up_weight),
                              c.activation_situ_beta, c.activation_situ_linear_beta)

    def fc2_input(self, summed, weights):
        """The fc2 input follows the ordered, gamma-weighted routed sum."""
        return reference.norm(summed, weights[_MOE + "latent_norm.weight"], self.config.rms_norm_eps)

    def finish_moe(self, state, summed, weights):
        routed = reference.linear(self.fc2_input(summed, weights), weights[_MOE + "fc2_latent_proj.weight"])
        shared = _MOE + "shared_experts."
        active = self.activated(state.x, weights[shared + "w1.weight"], weights[shared + "w3.weight"])
        return state.finish(routed + reference.linear(active, weights[shared + "w2.weight"]))

    def finish_dense(self, state, weights):
        return state.finish(reference.dense_ffn(state.x, weights, self.config))

    def logits(self, hidden, outer):
        return reference.lm_head_logits(hidden, outer, self.config, eps_outside_sqrt=False)

    def draft_features(self, taps, auxiliary):
        return reference.dspark_project_target_taps(taps, auxiliary, self.config)

    def native_draft_stage(self, hidden, weights, positions, context_kv):
        return reference.dspark_stage_forward(hidden, weights, self.config, positions, context_kv,
                                              draft_rope_mode="plain")

    def native_proposal(self, taps, anchor_token, auxiliary, stages, outer):
        return reference.dspark_propose(taps, anchor_token, auxiliary, stages, outer, self.config,
                                        draft_rope_mode="plain", multimax_after_markov=True,
                                        eps_outside_sqrt=False)
