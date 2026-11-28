"""Publish the audited ERNIE package over receipt-backed, largest-first HTTP LFS."""
from functools import partial

from hy4_flash.publish import run_cli, validate_mc
from inkling2.publish import PublicationTarget
from .export import PUBLISHED
from .source import BASE_REPO, BASE_REVISION

TARGET = PublicationTarget(PUBLISHED, "ernie55_flash", "Publish calibrated ERNIE 5.5 Flash K4 MC checkpoint",
                           partial(validate_mc, repo=BASE_REPO, revision=BASE_REVISION))


if __name__ == "__main__":
    run_cli(TARGET, BASE_REVISION)
