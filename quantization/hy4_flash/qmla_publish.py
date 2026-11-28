"""Publish a validated Hy4-flash QMLA sidecar onto a new revision of the same HF repo.

Thin family CLI wrapper over ``quantization/qmla/publish.py``:
- Single sidecar file: ``mla_calibration.safetensors`` (no ``mla_calibration.json``).
- MC checkpoints (``--update-mc-manifest``): rewrites ``cuteafd-mc.json`` with the added
  ``"qmla"`` section and adds ``mla_calibration.safetensors``.
- Non-MC checkpoints (omit ``--update-mc-manifest``): adds ONLY
  ``mla_calibration.safetensors`` (self-describing via ``__metadata__["cuteafd_qmla"]``).
- No new repository; zero weight shard re-uploads; HTTP LFS only (``HF_HUB_DISABLE_XET=1``).
- Reads HF write token from stdin only when not ``--dry-run``.
"""
from qmla.publish_cli import build_parser as _parser, publish_main
from .recipe import TARGET_REPO


def build_parser():
    return _parser(__doc__, TARGET_REPO)


def main(argv=None):
    return publish_main(build_parser(), "hy4_flash", argv)


if __name__ == "__main__":
    raise SystemExit(main())
