"""Mistral Large 4 mtp_dspark RNN Markov head (native) vs the PyTorch reference.

Usage: python native/tests/ml4_dspark_selftest.py LIBRARY
The library can contain only ml4_dspark.cu. Synthetic weights at the
checkpoint geometry (vocab 204800, rank 512, hidden 6144), random base logits
with near-ties; the drafts must equal the reference (heads_ in
python/reference/families/mistral_large4/dspark.py) for every sequence.
"""
import argparse
import ctypes as C
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

P, I = C.c_void_p, C.c_int32
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))


def bind(path):
    lib = C.CDLL(path)
    lib.cuteafd_ml4_dspark_markov_workspace.argtypes = [I, I]
    lib.cuteafd_ml4_dspark_markov_workspace.restype = C.c_uint64
    lib.cuteafd_ml4_dspark_markov_norms.argtypes = [P, P, I, I, P]
    lib.cuteafd_ml4_dspark_markov_norms.restype = I
    lib.cuteafd_ml4_dspark_markov.argtypes = [P] * 10 + [I] * 5 + [P]
    lib.cuteafd_ml4_dspark_markov.restype = I
    return lib


def reference(base, hidden, anchors, w):
    """dspark.Drafter.heads_ on the given base logits (the RNN steps verbatim)."""
    drafts = []
    for s in range(hidden.shape[0]):
        state, prev, out = None, int(anchors[s]), []
        for k in range(hidden.shape[1]):
            e = w["embedding"][prev]
            state = torch.zeros_like(e) if state is None else state
            gate, candidate, o = F.linear(torch.cat([state, e, hidden[s, k]]), w["joint_w"], w["joint_b"]).chunk(3)
            gate = torch.sigmoid(gate)
            state = gate * state + (1.0 - gate) * torch.tanh(candidate)
            logits = base[s, k] + F.linear(torch.tanh(o), w["output"]).float()
            prev = int(logits.argmax())
            out.append(prev)
        drafts.append(out)
    return drafts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("library")
    p.add_argument("--vocab", type=int, default=204800)
    args = p.parse_args()
    lib = bind(args.library)
    torch.manual_seed(0)
    vocab, rank, hidden_width, block = args.vocab, 512, 6144, 3
    w = {"embedding": (torch.randn(vocab, rank) * 0.05).bfloat16().cuda(),
         "output": (torch.randn(vocab, rank) * 0.05).bfloat16().cuda(),
         "joint_w": (torch.randn(3 * rank, 2 * rank + hidden_width) * 0.02).bfloat16().cuda(),
         "joint_b": (torch.randn(3 * rank) * 0.1).bfloat16().cuda()}
    stream = torch.cuda.current_stream().cuda_stream
    norms = torch.empty(vocab, dtype=torch.float32, device="cuda")
    assert lib.cuteafd_ml4_dspark_markov_norms(w["output"].data_ptr(), norms.data_ptr(), vocab, rank, stream) == 0
    mismatches = 0
    for sequences in (1, 7, 32):
        hidden = torch.randn(sequences, block, hidden_width).bfloat16().cuda()
        # Soft-capped base logits with a few strong candidates per row (near-ties included).
        base = torch.tanh(torch.randn(sequences, block, vocab, device="cuda") * 0.3) * 30
        for s in range(sequences):
            for k in range(block):
                hot = torch.randint(vocab, (4,))
                base[s, k, hot] = 29.0 + torch.rand(4, device="cuda") * 0.05
        anchors = torch.randint(vocab, (sequences,), dtype=torch.int32).cuda()
        drafts = torch.zeros(sequences, block, dtype=torch.int32, device="cuda")
        workspace = torch.empty(lib.cuteafd_ml4_dspark_markov_workspace(sequences, block), dtype=torch.uint8,
                                device="cuda")
        rc = lib.cuteafd_ml4_dspark_markov(base.data_ptr(), hidden.data_ptr(), w["embedding"].data_ptr(),
                                           w["joint_w"].data_ptr(), w["joint_b"].data_ptr(), w["output"].data_ptr(),
                                           norms.data_ptr(), anchors.data_ptr(), drafts.data_ptr(),
                                           workspace.data_ptr(), sequences, block, vocab, rank, hidden_width, stream)
        torch.cuda.synchronize()
        assert rc == 0, rc
        expected = reference(base, hidden, anchors.cpu(), w)
        got = drafts.cpu().tolist()
        bad = sum(a != b for a, b in zip(got, expected))
        mismatches += bad
        print(f"sequences {sequences}: {sequences - bad}/{sequences} proposals equal the reference")
    assert mismatches == 0, f"{mismatches} proposals differ"
    print("ml4 dspark markov: pass")


if __name__ == "__main__":
    main()
