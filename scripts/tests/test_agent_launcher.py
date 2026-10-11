"""CPU-only sidecar lifecycle and persistent private route contracts."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def setup(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("""#!/usr/bin/env python3
import json,os,sys,pathlib
with open(os.environ['DOCKER_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='inspect': sys.exit(1)
if 'set-credential' in sys.argv:
    mount=sys.argv[sys.argv.index('--mount')+1]
    home=pathlib.Path(mount.split('src=',1)[1].split(',dst=',1)[0])
    path=home/'dsh/.credentials.yaml'
    document=json.loads(path.read_text()) if path.exists() else {'version':1,'refs':{},'records':{}}
    document['refs']['CUTEAFD_AGENT_API_KEY']=sys.stdin.readline().rstrip('\\n')
    path.write_text(json.dumps(document));path.chmod(0o600)
""")
    docker.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    key = home / ".config/cuteafd/api-key"
    key.parent.mkdir(parents=True)
    key.write_text("coordinator-key\n")
    key.chmod(0o600)
    env = {**os.environ, "HOME": str(home), "PATH": str(bin_dir) + ":" + os.environ["PATH"],
           "DOCKER_LOG": str(tmp_path / "docker.log"), "CUTEAFD_AGENT_MODEL": "served-model",
           "API_KEY_FILE": str(key), "CUTEAFD_AGENT_IMAGE": "stub-agent:test", "CUTEAFD_AGENT_PORT": "3311"}
    return env, home


def run(env, *args):
    return subprocess.run([str(ROOT / "scripts/launch/agent.sh"), *args], env=env, text=True, capture_output=True)


def test_launcher_private_mount_route_and_no_secrets(tmp_path):
    env, home = setup(tmp_path)
    first = run(env, "start")
    assert first.returncode == 0, first.stderr
    base = home / ".local/share/cuteafd/agent"
    keys = json.loads((home / ".config/cuteafd/api-keys").read_text())
    for path, mode in [(base, 0o700), (base / "dsh/.credentials.yaml", 0o600),
                       (home / ".config/cuteafd/api-keys", 0o600)]:
        assert path.stat().st_mode & 0o777 == mode
    assert keys["agent"] in (base / "dsh/.credentials.yaml").read_text()
    patch = base / "dsh/profiles/cuteafd/cordis.patch.yml"
    rows = json.loads(patch.read_text())
    assert rows[0]["config"]["providers"]["cuteafd"]["baseURL"] == "http://host.docker.internal:8000/v1"
    assert rows[1]["config"] == {"provider": "cuteafd", "model": "served-model"}
    calls = [json.loads(line) for line in Path(env["DOCKER_LOG"]).read_text().splitlines()]
    command = next(c for c in calls if c[0] == "run" and "-d" in c)
    assert "127.0.0.1:3311:3010" in command
    assert command.count("--mount") == 1
    for item in ["HOME=/agent/home", "DSH_HOME=/agent/dsh", "DSH_AGENTS_HOME=/agent/agents", "DSH_TELEMETRY_DISABLED=1", "DSH_REMOTE_ONLY=1", "stub-agent:test"]:
        assert item in command
    assert "--host" not in command
    output = first.stdout + first.stderr + Path(env["DOCKER_LOG"]).read_text()
    assert keys["agent"] not in output and keys["default"] not in output
    credentials = base / "dsh/.credentials.yaml"
    saved = json.loads(credentials.read_text())
    saved["refs"]["OTHER_PROVIDER_KEY"] = "operator-secret"
    saved["records"] = {"client-connection/browser-session": {"kind": "grant", "payload": {"secret": "persistent-signing-secret"}}}
    credentials.write_text(json.dumps(saved))
    old_mtime = patch.stat().st_mtime_ns
    assert run(env, "start").returncode == 0
    assert patch.stat().st_mtime_ns == old_mtime
    assert json.loads(credentials.read_text()) == saved
    assert json.loads((home / ".config/cuteafd/api-keys").read_text()) == keys
    env["CUTEAFD_AGENT_MODEL"] = "new-model"
    assert run(env, "start").returncode == 0
    assert (patch.with_suffix(".yml.bak")).exists()
    assert run(env, "stop").returncode == 0
    assert run(env, "status").stdout == "stopped\n"


def test_agent_delegation_and_invalid_port(tmp_path):
    env, _ = setup(tmp_path)
    delegated = subprocess.run([str(ROOT / "run.sh"), "--agent", "status"], env=env, text=True, capture_output=True)
    assert delegated.stdout == "stopped\n"
    env["CUTEAFD_AGENT_PORT"] = "0"
    assert run(env, "start").returncode == 2


def test_configured_key_and_listen_defaults_and_public_override(tmp_path):
    env, home = setup(tmp_path)
    configured = home / "alternate-key"
    configured.write_text("configured-secret\n")
    config = tmp_path / "coordinator.config"
    config.write_text(f"API_KEY_FILE={configured}\nADDR=0.0.0.0:8123\n")
    env["CUTEAFD_AGENT_CONFIG"] = str(config)
    assert run(env, "start").returncode == 0
    calls = [json.loads(line) for line in Path(env["DOCKER_LOG"]).read_text().splitlines()]
    command = next(c for c in calls if c[0] == "run" and "-d" in c)
    assert "http://localhost:8123/agent/app/" in command
    assert "localhost:8123" in command
    assert json.loads(configured.with_name("api-keys").read_text())["default"] == "configured-secret"
    env["CUTEAFD_AGENT_PUBLIC_URL"] = "https://dashboard.test:9443/agent/app/"
    assert run(env, "start").returncode == 0
    command = [json.loads(line) for line in Path(env["DOCKER_LOG"]).read_text().splitlines() if json.loads(line)[0] == "run" and "-d" in json.loads(line)][-1]
    assert "dashboard.test:9443" in command
    env["CUTEAFD_AGENT_PUBLIC_URL"] = "http://dashboard.test:443/agent/app/"
    assert run(env, "start").returncode == 0
    command = [json.loads(line) for line in Path(env["DOCKER_LOG"]).read_text().splitlines() if json.loads(line)[0] == "run" and "-d" in json.loads(line)][-1]
    assert "dashboard.test:443" in command


def test_unknown_coordinator_key_refuses_instead_of_guessing(tmp_path):
    env, home = setup(tmp_path)
    Path(env.pop("API_KEY_FILE")).unlink()
    result = run(env, "start")
    assert result.returncode == 2
    assert "Cannot determine coordinator API key" in result.stderr
    assert not (home / ".config/cuteafd/api-keys").exists()


def test_nonwildcard_listen_reachability_and_loopback_refusal(tmp_path):
    env, home = setup(tmp_path)
    config = tmp_path / "coordinator.config"
    config.write_text(f"API_KEY_FILE={env['API_KEY_FILE']}\nADDR=10.55.0.22:8123\n")
    env["CUTEAFD_AGENT_CONFIG"] = str(config)
    assert run(env, "start").returncode == 0
    profile = home / ".local/share/cuteafd/agent/dsh/profiles/cuteafd/cordis.patch.yml"
    assert json.loads(profile.read_text())[0]["config"]["providers"]["cuteafd"]["baseURL"] == "http://10.55.0.22:8123/v1"
    config.write_text(f"API_KEY_FILE={env['API_KEY_FILE']}\nADDR=127.0.0.1:8123\n")
    result = run(env, "start")
    assert result.returncode != 0
    assert "Loopback-only coordinator" in result.stderr
