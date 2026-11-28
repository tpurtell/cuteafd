"""Score the frozen Hy4-flash QMLA transforms on the MC K8 core and shared K4 experts.

Run after MC original-view calibration with --checkpoint MC_SNAPSHOT --corpus-dir PREPARED
--output CALIBRATION_OUTPUT. Only the recorded test sequences run through the model.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from qmla.transfer import VIEW, run_view_transfer, summarize_transfer, transfer_records

from . import recipe
from .qmla_mc import MCModel, MCReader
from .qmla_capture import Hy4FlashCaptureAdapter
from .qmla_run import load_pinned_qmla_corpus, run_hy4_flash_long_eval


def run(checkpoint, corpus_dir, output, device):
    reader = MCReader(checkpoint, device, view=VIEW)
    adapter = Hy4FlashCaptureAdapter(MCModel(reader))
    return run_view_transfer(
        checkpoint, corpus_dir, output, device, repo=recipe.TARGET_REPO, reader=reader,
        expected_attention_arithmetic=MCModel.attention_arithmetic,
        load_corpus=load_pinned_qmla_corpus, evaluate=partial(run_hy4_flash_long_eval, adapter),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.checkpoint.resolve(), args.corpus_dir, args.output, args.device)
    print(json.dumps({"phase": "k8_transfer", "status": "passed", "modes": result["modes"],
                      "transform_fingerprint": result["transform_fingerprint"]}), flush=True)


if __name__ == "__main__":
    main()
