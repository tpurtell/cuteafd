"""Exercise the hub mounts in a remote shell; no Docker or cluster required."""
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts/lib/release-common.sh"


def snapshot(hub, model):
    repo = hub / f"models--{model.replace('/', '--')}"
    (repo / "blobs").mkdir(parents=True)
    snap = repo / "snapshots/rev"
    snap.mkdir(parents=True)
    (repo / "blobs/config").write_text(json.dumps({"model_type": "glm5_next"}))
    (snap / "config.json").symlink_to("../../blobs/config")
    return snap


def run_mounts(hf, namespace, relative, *, site=1):
    return subprocess.run(["bash", "-euo", "pipefail", "-c", '''
source "$1"
release_hub_mounts "$2" "$3" "$4" "$5"
printf '%s\\n' "$RELEASE_HUB_SNAPSHOT" "${RELEASE_HUB_MOUNTS[@]}"
''', "test", str(COMMON), str(namespace), str(site), str(hf), relative],
        text=True, capture_output=True, timeout=10)


@pytest.mark.parametrize("symlink_hub", [False, True])
def test_site_hub_mounts_cover_blob_links_companions_and_hf_home(tmp_path, symlink_hub):
    hub = tmp_path / "real-hub"
    main = snapshot(hub, "org/main")
    draft = snapshot(hub, "org/draft")
    template = snapshot(hub, "org/template")
    hf = tmp_path / "hf"
    hf.mkdir()
    if symlink_hub:
        (hf / "hub").symlink_to(hub)
    else:
        hub.rename(hf / "hub")
        hub = hf / "hub"
        main, draft, template = [hub / p.relative_to(tmp_path / "real-hub") for p in (main, draft, template)]
    result = run_mounts(hf, tmp_path / "absent", "hub/models--org--main/snapshots/rev")
    assert result.returncode == 0, result.stderr
    values = result.stdout.splitlines()
    assert values[0] == str(main)
    mounts = [v.split(":") for v in values[2::2]]
    # Emulate bind lookup, then resolve links on the source. Every file must be
    # accessible through both the resolved snapshot and HF_HOME cache paths.
    for path in [main / "config.json", draft / "config.json", template / "config.json",
                 hf / "hub/models--org--template/snapshots/rev/config.json"]:
        mapped = [Path(src) / path.relative_to(dst) for src, dst, mode in mounts
                  if path.is_relative_to(dst)]
        assert mapped and any(p.read_text() for p in mapped)


def test_remote_resolves_its_own_hub_and_refuses_missing_snapshot_before_docker(tmp_path):
    hf = tmp_path / "remote-hf"
    hf.mkdir()
    (hf / "hub").symlink_to(tmp_path / "remote-storage")
    main = snapshot(tmp_path / "remote-storage", "org/main")
    relative = "hub/models--org--main/snapshots/rev"
    result = run_mounts(hf, tmp_path / "absent", relative)
    assert result.returncode == 0 and result.stdout.splitlines()[0] == str(main)
    main.rename(main.with_name("moved"))
    result = run_mounts(hf, tmp_path / "absent", relative)
    assert result.returncode == 2
    assert "missing snapshot" in result.stderr
    assert not main.exists()


def test_remote_refuses_dangling_snapshot_files(tmp_path):
    hf = tmp_path / "hf"
    main = snapshot(hf / "hub", "org/main")
    (main / "config.json").resolve().unlink()
    result = run_mounts(hf, tmp_path / "absent", "hub/models--org--main/snapshots/rev")
    assert result.returncode == 2
    assert "dangling links" in result.stderr
