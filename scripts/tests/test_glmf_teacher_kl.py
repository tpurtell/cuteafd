"""scripts/bench/glmf-teacher-kl.py against a stand-in server that writes probe row dumps."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench" / "glmf-teacher-kl.py"
VOCAB = 5


def _module():
    spec = importlib.util.spec_from_file_location("glmf_teacher_kl", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(position: int) -> list[float]:
    """The stand-in engine's log-probabilities of the row predicting ``position``."""
    return [-float(position) - v / 8 for v in range(VOCAB)]


def _write_dump(leaf: Path, positions: list[int]) -> None:
    """A dump as the engine writes it (cuteafd-api probe/rows.rs): one safetensors per row and a manifest."""
    mod = _module()
    leaf.mkdir()
    with open(leaf / "manifest.jsonl", "w") as manifest:
        for index, position in enumerate(positions):
            name = f"row-{index:08}.safetensors"
            mod.write_safetensors(str(leaf / name), [("log_probs", "F32", [VOCAB],
                                                      struct.pack(f"<{VOCAB}f", *_row(position)))], {})
            manifest.write(json.dumps({"position": position, "vocab_size": VOCAB, "file": name,
                                       "tensor": "log_probs", "dtype": "F32", "byte_order": "little"}) + "\n")


def _server(tmp_path, prefill_scoring="streamed"):
    root = tmp_path / "dumps"
    root.mkdir()
    seen, headers = [], []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            headers.append(self.headers.get("authorization"))
            self._reply({} if prefill_scoring == "absent" else {"prefill_scoring": prefill_scoring})

        def do_POST(self):  # noqa: N802
            headers.append(self.headers.get("authorization"))
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            spec = payload["spec"]
            seen.append(spec)
            wanted = [p for p in spec["score_positions"] if p >= spec["score_from"]]
            _write_dump(root / spec["dump_rows"], wanted)
            self._reply({"probe": {"scored": len(wanted), "score_path": spec["score_path"], "error": None},
                         "server": {"build": "test", "settings": {"kda_state": "bf16"}}})

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}", root, seen, headers


@pytest.fixture
def server(tmp_path):
    httpd, url, root, seen, headers = _server(tmp_path)
    yield url, root, seen, headers
    httpd.shutdown()


def _plan(path: Path) -> dict:
    windows = []
    for wid, tokens, rows in (("final-0000", [3, 1, 4, 1, 5, 9, 2], [0, 1, 4, 5]), ("final-0001", [2, 7, 1, 8], [2])):
        windows.append({"window_id": wid, "tokens": tokens, "positions": rows, "role": "final",
                        "tokens_sha256": hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()})
    plan = {"schema": "test", "vocab": VOCAB, "windows": windows}
    path.write_text(json.dumps(plan))
    return plan


def _run(*args, env=None):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=60,
                          env=env)


@pytest.mark.parametrize("path", ["decode", "prefill"])
def test_rows_become_one_klgate_window_file_each(tmp_path, server, path):
    url, root, seen, _ = server
    plan = _plan(tmp_path / "plan.json")
    out = tmp_path / "engine"
    run = _run("--url", url, "--plan", tmp_path / "plan.json", "--dump-root", root, "--out", out, "--path", path,
               "--verify-rows", "8")
    assert run.returncode == 0, run.stderr
    mod = _module()
    plan_sha = hashlib.sha256((tmp_path / "plan.json").read_bytes()).hexdigest()
    for window, spec in zip(plan["windows"], seen):
        # The plan's row r is the probe position r + 1, cold, from position 1, on the asked path.
        assert spec["score_positions"] == [r + 1 for r in window["positions"]]
        assert (spec["score_from"], spec["score_path"], spec["cold"], spec["no_speculation"]) == (1, path, True, True)
        assert spec.get("verify_rows") == (8 if path == "decode" else None)
        meta, header, data = mod.read_safetensors(str(out / f"{window['window_id']}.safetensors"))
        assert meta["window_id"] == window["window_id"] and meta["tokens_sha256"] == window["tokens_sha256"]
        assert meta["plan_sha256"] == plan_sha and meta["score_path"] == path
        assert json.loads(meta["engine"])["settings"] == {"kda_state": "bf16"}
        k = len(window["positions"])
        begin, end = header["positions"]["data_offsets"]
        assert header["positions"]["dtype"] == "I32" and list(struct.unpack(f"<{k}i", data[begin:end])) == window["positions"]
        begin, end = header["logits"]["data_offsets"]
        assert header["logits"]["shape"] == [k, VOCAB]
        values = struct.unpack(f"<{k * VOCAB}f", data[begin:end])
        expected = [v for r in window["positions"] for v in _row(r + 1)]
        assert list(values) == pytest.approx(expected, abs=0)
    assert not any(root.iterdir()), "every dump leaf is removed once converted"
    record = json.loads((out / "run.json").read_text())
    assert record["complete"] and [w["window_id"] for w in record["windows"]] == ["final-0000", "final-0001"]
    # A rerun skips the finished windows; another path's run into the same directory is refused.
    again = _run("--url", url, "--plan", tmp_path / "plan.json", "--dump-root", root, "--out", out, "--path", path)
    assert again.returncode == 0 and len(seen) == 2
    other = "prefill" if path == "decode" else "decode"
    refused = _run("--url", url, "--plan", tmp_path / "plan.json", "--dump-root", root, "--out", out, "--path", other)
    assert refused.returncode != 0 and "another plan's or path's run" in refused.stderr


@pytest.mark.parametrize("capability", [None, "absent", "unknown", True])
def test_prefill_path_needs_explicit_capability_before_any_window(tmp_path, capability):
    httpd, url, root, seen, _ = _server(tmp_path, prefill_scoring=capability)
    try:
        _plan(tmp_path / "plan.json")
        refused = _run("--url", url, "--plan", tmp_path / "plan.json", "--dump-root", root, "--out", tmp_path / "out",
                       "--path", "prefill")
        assert refused.returncode != 0 and "prefill_scoring=streamed" in refused.stderr
        assert not seen and not (tmp_path / "out").exists()
        # The decode path needs no admission.
        decode = _run("--url", url, "--plan", tmp_path / "plan.json", "--dump-root", root, "--out", tmp_path / "out",
                      "--path", "decode")
        assert decode.returncode == 0, decode.stderr
        assert len(seen) == 2
    finally:
        httpd.shutdown()


def test_api_key_is_sent_as_a_bearer_token(tmp_path, server):
    url, root, seen, headers = server
    _plan(tmp_path / "plan.json")
    env = {"PATH": "/usr/bin:/bin", "CUTEAFD_API_KEY": "test-key"}
    run = _run("--url", url, "--plan", tmp_path / "plan.json", "--dump-root", root, "--out", tmp_path / "out",
               "--path", "prefill", "--windows", "final-0001", env=env)
    assert run.returncode == 0, run.stderr
    assert headers == ["Bearer test-key", "Bearer test-key"] and len(seen) == 1


def test_missing_rows_and_vocabulary_changes_are_refused(tmp_path):
    mod = _module()
    _write_dump(tmp_path / "dump", [1, 3])
    with pytest.raises(ValueError, match="no row for 1 plan rows"):
        mod.collect(str(tmp_path / "dump"), [0, 1], VOCAB)
    assert len(mod.collect(str(tmp_path / "dump"), [0, 2], VOCAB)) == 2 * VOCAB * 4
    with pytest.raises(ValueError, match="vocabulary"):
        mod.collect(str(tmp_path / "dump"), [0], VOCAB + 1)
