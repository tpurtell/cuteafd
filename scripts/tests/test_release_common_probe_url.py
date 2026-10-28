import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def probe_url(addr):
    script = f'source "{ROOT}/scripts/lib/release-common.sh"; release_probe_url "{addr}"'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("addr, url", [
    ("0.0.0.0:8000", "http://127.0.0.1:8000"),
    (":8000", "http://127.0.0.1:8000"),
    ("[::]:8000", "http://127.0.0.1:8000"),
    ("127.0.0.1:18850", "http://127.0.0.1:18850"),
    ("192.0.2.10:18860", "http://192.0.2.10:18860"),
])
def test_readiness_probes_reach_the_bound_address(addr, url):
    assert probe_url(addr) == url
