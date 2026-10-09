import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECRET = "ab" * 32


def print_link(tmp_path, env_extra=None):
    secret = tmp_path / "secret"
    secret.write_text(SECRET + "\n")
    script = f'source "{ROOT}/scripts/lib/release-common.sh"; CONSOLE_SECRET_FILE="{secret}"; release_print_console_link http://raptor:8000'
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    env.update(env_extra or {})
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, check=True).stdout


def test_redirected_output_never_carries_the_token(tmp_path):
    out = print_link(tmp_path)
    assert SECRET not in out
    assert "console unlock: http://raptor:8000/console/unlock?token=<contents of" in out


def test_explicit_opt_in_prints_the_token(tmp_path):
    out = print_link(tmp_path, {"CUTEAFD_PRINT_CONSOLE_LINK": "1"})
    assert out == f"console unlock: http://raptor:8000/console/unlock?token={SECRET}\n"
