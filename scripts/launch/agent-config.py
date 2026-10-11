#!/usr/bin/env python3
"""Provision private sidecar credentials and default gateway route without logging secrets."""
import json
import os
from pathlib import Path
import secrets
import sys
import urllib.request


def write_private(path, text):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def provision(home, emit_key=False):
    for directory in [home, home / "home", home / "dsh", home / "agents"]:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
    legacy = Path(os.environ["API_KEY_FILE"])
    named = legacy.with_name("api-keys")
    legacy.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if legacy.exists():
        default = legacy.read_text().rstrip("\r\n")
    else:
        raise ValueError("coordinator API key is missing")
    if not default or not all(33 <= ord(c) <= 126 for c in default):
        raise ValueError("invalid legacy API key")
    keys = json.loads(named.read_text()) if named.exists() else {}
    if named.exists() and named.stat().st_mode & 0o777 != 0o600:
        raise ValueError("named API key file must be 0600")
    if keys.get("default", default) != default:
        raise ValueError("named default disagrees with legacy API key")
    keys["default"] = default
    keys.setdefault("agent", secrets.token_urlsafe(32))
    if not isinstance(keys["agent"], str) or not keys["agent"] or not all(33 <= ord(c) <= 126 for c in keys["agent"]):
        raise ValueError("invalid agent API key")
    write_private(named, json.dumps(keys, indent=2) + "\n")
    if emit_key:
        sys.stdout.write(keys["agent"] + "\n")
        return
    gateway = os.environ.get("CUTEAFD_AGENT_GATEWAY_URL", "http://host.docker.internal:8000")
    model = os.environ.get("CUTEAFD_AGENT_MODEL")
    if not model:
        discovery = os.environ.get("CUTEAFD_AGENT_DISCOVERY_URL", "http://127.0.0.1:8000")
        request = urllib.request.Request(discovery.rstrip("/") + "/v1/models", headers={"Authorization": "Bearer " + default})
        with urllib.request.urlopen(request, timeout=5) as response:
            model = json.load(response)["data"][0]["id"]
    profile = home / "dsh/profiles/cuteafd/cordis.patch.yml"
    patch = [
        {"id": "llm-pi-ai", "config": {"providers": {"cuteafd": {"apiKeyEnv": "CUTEAFD_AGENT_API_KEY",
            "api": "openai-responses", "baseURL": gateway.rstrip("/") + "/v1", "models": [{"id": model,
            "contextWindow": 262144, "maxTokens": 32768}]}}}},
        {"id": "agent-default-model", "config": {"provider": "cuteafd", "model": model}},
    ]
    # JSON is valid YAML; preserve operator-added rows and provider configuration.
    if profile.exists():
        try:
            previous = json.loads(profile.read_text())
        except json.JSONDecodeError:
            import yaml
            previous = yaml.safe_load(profile.read_text())
        if not isinstance(previous, list):
            raise ValueError("profile patch must be a list")
        rows = {row.get("id"): row for row in previous if isinstance(row, dict)}
        for row in patch:
            existing = rows.get(row["id"])
            if existing is None:
                previous.append(row)
            elif row["id"] == "llm-pi-ai":
                existing.setdefault("config", {}).setdefault("providers", {})["cuteafd"] = row["config"]["providers"]["cuteafd"]
            else:
                existing["config"] = row["config"]
        patch = previous
    desired = json.dumps(patch, indent=2) + "\n"
    if not profile.exists() or profile.read_text() != desired:
        if profile.exists():
            write_private(profile.with_suffix(".yml.bak"), profile.read_text())
        write_private(profile, desired)


if __name__ == "__main__":
    try:
        provision(Path(sys.argv[1]), emit_key="--emit-key" in sys.argv[2:])
    except Exception:
        print("Agent provisioning failed: check gateway/model and private key configuration", file=sys.stderr)
        sys.exit(1)
