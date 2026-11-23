"""Bitwise witness for resident GLM Flash calibration, K4 encoding and sealed replay."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import subprocess
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from kimi_k3.uniform_k2.storage import atomic_json
from . import numerics
from .experts import ExpertBank, ExpertWorkspace, fit_expert, replay_expert

WIDTH = 128
CHUNK = 192
PARTS = ("gate_proj", "up_proj", "down_proj")
LIMIT = 10.0


def exact(actual, expected, label):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8))):
        raise ValueError(f"unified-memory exactness witness failed: {label}")


def fixture(n, generator, device):
    hidden = torch.randn(n, WIDTH, generator=generator).bfloat16().to(device)
    # Rotating expert ranks exercise distinct corpus and candidate-rank orders.
    positions = torch.arange(n, device=device)[:, None]
    ranks = (torch.arange(16, device=device) + positions * 3) % 16
    scores = torch.randn(n, 16, generator=generator).to(device).sigmoid()
    ids, near = ranks[:, :8], ranks[:, 8:]
    selected = scores.gather(1, ids)
    natural_gamma = selected / selected.sum(1, keepdim=True) * 2.5
    near_gamma = torch.empty_like(natural_gamma)
    for slot in range(8):
        alternative = torch.cat((selected[:, :7], scores.gather(1, near[:, slot:slot+1])), dim=1)
        near_gamma[:, slot] = alternative[:, -1] / alternative.sum(1) * 2.5
    data = (hidden, ids, natural_gamma, near, near_gamma)
    batches = (tuple(field[start:start+137] for field in data) for start in range(0, n, 137))
    bank = ExpertBank(batches, tokens=n, width=WIDTH, experts=17, device=device)
    return data, bank


def reference_selection(data, expert):
    _, native, weights, near, aug_weights = data
    rows, slots = torch.where(native == expert)
    gamma = weights[rows, slots]
    augmented_rank, augmented_rows = torch.where(near.T == expert)
    needed = max(0, 1024 - len(rows))
    augmented_rows, augmented_rank = augmented_rows[:needed], augmented_rank[:needed]
    return (torch.cat((rows, augmented_rows)), torch.cat((gamma, aug_weights[augmented_rows, augmented_rank])),
            len(rows), max(0, needed - len(augmented_rows)))


def reference_raw(inputs, missing):
    width = inputs.shape[1]
    total = torch.zeros((width, width), dtype=torch.float64, device=inputs.device)
    for gathered in inputs.split(CHUNK):
        for block in gathered.split(128):
            total.add_(block.float().T @ block.float())
    matrix = total.float()
    matrix.diagonal().add_(missing)
    return dict(H=matrix, count=len(inputs) + missing, identity_rows=missing, finalized=False)


def activate(inputs, fused):
    gate, up = (inputs.bfloat16() @ fused.T).chunk(2, dim=-1)
    return F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)


def reference_sum(data, fused, down):
    hidden, ids, gamma, _, _ = data
    total = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(16):
        rows, slots = torch.where(ids == expert)
        for selected, selected_slots in zip(rows.split(CHUNK), slots.split(CHUNK), strict=True):
            activation = activate(hidden[selected], fused)
            output = activation @ down.T
            total[selected] = total[selected] + output.float() * gamma[selected, selected_slots, None]
    return total


@torch.inference_mode()
def qualify(device, scratch):
    generator = torch.Generator().manual_seed(787)
    # Calibration exceeds the 128-row product boundary and has identity rows.
    data, bank = fixture(385, generator, device)
    heldout, other_bank = fixture(37, generator, device)
    workspace = ExpertWorkspace.allocate(WIDTH, WIDTH, device=device, chunk_rows=CHUNK)
    pointers = {name: tensor.data_ptr() for name, tensor in vars(workspace).items()}
    for expert in range(17):
        rows, gamma, natural, missing = bank.selection(expert)
        ref_rows, ref_gamma, ref_natural, ref_missing = reference_selection(data, expert)
        exact(rows, ref_rows, "ordered selected row IDs")
        exact(gamma, ref_gamma, "ordered coefficients")
        if (natural, missing) != (ref_natural, ref_missing):
            raise ValueError("natural or synthetic row count differs")
        if len(rows):
            captured = torch.cat([chunk.clone() for chunk in bank.chunks(rows, workspace.gather)])
            exact(captured, data[0][ref_rows].float(), "device gather")

    rows, gamma, _, missing = reference_selection(data, 0)
    inputs = data[0][rows]
    matrices = {name: (torch.randn(WIDTH, WIDTH, generator=generator) / WIDTH**.5).bfloat16().to(device)
                for name in PARTS}
    input_h = reference_raw(inputs, missing)
    reference, covariances, replay = {}, {}, {}
    for name in PARTS[:2]:
        covariances[name] = input_h
        encoded = numerics.encode_expert(name, matrices[name], input_h, device)
        reference[name], replay[name] = encoded.packed[4], encoded.reconstruction
    fused = torch.cat([replay[name] for name in PARTS[:2]])
    activations = torch.cat([activate(x, fused) for x in inputs.split(CHUNK)])
    down_h = reference_raw(activations.float() * gamma[:, None], missing)
    covariances["down_proj"] = down_h
    down = numerics.encode_expert("down_proj", matrices["down_proj"], down_h, device)
    reference["down_proj"] = down.packed[4]

    encode = numerics.encode_expert
    def compare_covariance(name, weight, raw, device):
        expected = covariances[name.rsplit(".", 1)[1]]
        exact(raw["H"], expected["H"], f"{name} raw Hessian")
        if any(raw[key] != expected[key] for key in ("count", "identity_rows", "finalized")):
            raise ValueError(f"{name} Hessian provenance mismatch")
        return encode(name, weight, raw, device)

    with patch.object(numerics, "encode_expert", compare_covariance):
        packed, _ = fit_expert(bank, 0, matrices, prefix="fixture.experts.0",
                               workspace=workspace, swiglu_limit=LIMIT)
    for part in PARTS:
        for suffix, expected in reference[part].items():
            exact(packed[part][suffix], expected, f"{part}.{suffix} K4 serialized bytes")

    # Use the same sealed expert for each different natural selection, including
    # overlapping rows. Reopening this file is independent of live packed state.
    shard = scratch / "fixture.safetensors"
    save_file({f"{p}.{k}": v for p in PARTS for k, v in packed[p].items()}, str(shard))
    sealed = load_file(str(shard))
    recovered = {p: {k: sealed[f"{p}.{k}"] for k in packed[p]} for p in PARTS}
    weighted = torch.empty_like(workspace.gather)
    def combine(target, weights):
        return target.combine(lambda expert, _: replay_expert(
            target, expert, weights, workspace=workspace, swiglu_limit=LIMIT), weighted_buffer=weighted)
    for label, target, fixture_data in (("calibration", bank, data), ("heldout", other_bank, heldout)):
        expected = reference_sum(fixture_data, fused, down.reconstruction)
        exact(combine(target, packed), expected, label + " live ordered reduction")
        exact(combine(target, recovered), expected, label + " sealed recovery replay")
    # Compare held-out error numerator and denominator as the gate records them.
    rows, _, natural, _ = reference_selection(heldout, 0)
    h = heldout[0][rows[:natural]]
    base = activate(h, torch.cat([matrices[n] for n in PARTS[:2]])) @ matrices["down_proj"].T
    reference_out = activate(h, fused) @ down.reconstruction.T
    actual_out = torch.cat([out.clone() for out in replay_expert(
        other_bank, 0, recovered, workspace=workspace, swiglu_limit=LIMIT)])
    def error_sums(output):
        return torch.stack(((output.float() - base.float()).square().sum(dtype=torch.float64),
                            base.float().square().sum(dtype=torch.float64)))
    exact(error_sums(actual_out), error_sums(reference_out), "heldout gate evidence")
    if pointers != {name: tensor.data_ptr() for name, tensor in vars(workspace).items()}:
        raise ValueError("resident workspace replaced during encode or heldout replay")
    return dict(status="passed", fixture="deterministic-128-channel-glm-flash-v1", calibration_rows=385,
                heldout_rows=37, shape=[WIDTH, WIDTH], compared_k4_encodes=3, raw_hessians_equal=True,
                packed_bytes_equal=True, heldout_sum_equal=True, sealed_replay_equal=True,
                selection_equal=True, gate_evidence_equal=True, workspace_reused=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (12, 1):
        raise ValueError("GB10 SM121 qualification requires cassowary's GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        receipt = qualify(device, Path(temp))
    repo = Path(__file__).resolve().parents[3]
    def revision(path):
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    atomic_json(args.output, dict(receipt, device=torch.cuda.get_device_name(device),
                                  engine_revision=revision(repo), fork_revision=revision(repo / "third_party/gptqmodel")))


if __name__ == "__main__":
    main()
