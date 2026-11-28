"""Replay the frozen GLM Flash MUL1 test token IDs through the actual K8 core and K4 experts."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from qmla.transfer import VIEW, run_view_transfer
from .qmla_mc import GlmFlashMCSource


def run(checkpoint, corpus_dir, output, device):
    from .qmla_capture import Glm5FlashCaptureAdapter
    from .qmla_run import TARGET_REPO, TARGET_REVISION, load_pinned_qmla_corpus, run_evaluation

    if checkpoint.name != TARGET_REVISION:
        raise ValueError(f"GLM Flash K8 replay requires the immutable MC revision {TARGET_REVISION}")
    reader = GlmFlashMCSource(checkpoint, view=VIEW)
    adapter = Glm5FlashCaptureAdapter(reader.native_config())
    try:
        return run_view_transfer(
            checkpoint, corpus_dir, output, device, repo=TARGET_REPO, reader=reader,
            load_corpus=load_pinned_qmla_corpus, evaluate=partial(run_evaluation, reader, adapter))
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.checkpoint.resolve(), args.corpus_dir, args.output, args.device)
    print(json.dumps(dict(phase="k8_transfer", status="passed", modes=result["modes"],
                          transform_fingerprint=result["transform_fingerprint"])), flush=True)


if __name__ == "__main__":
    main()
