"""Audit the pinned ERNIE snapshot, including all 74 indexed shards and the flashindexer."""
from hy4_flash.preflight import run_cli
from .source import HeaderSource

if __name__ == "__main__":
    run_cli(HeaderSource, __doc__)
