"""Release-build site fixes: empty EXL3 scope, ssh image distribution, staging-dir
measurement, the free-space override and the single-host build.

A deepseek_v41-scoped build ships no V4.1 EXL3 family, so the artifact directory
gained no exl3/ and Dockerfile.release's unconditional `COPY .../exl3` failed the
Spark image ("/.cuteafd-release-image/exl3: not found"). Like fp8/, exl3/ always
exists and the bake-time verify loop skips an unmatched glob.

The Spark staging directory may resolve off a small $HOME (a symlink onto the
Spark /scratch), so the free-space check measures the filesystem that really
holds it; CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB overrides the 60 GiB floor. A
single build host has nobody to copy to, and hosts without rdmapipe fall back to
`ssh docker save | load` (rdmapipe stays first when present).
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_artifacts_script_always_creates_exl3_dir_before_the_family_loop():
    s = (REPO / "scripts/build/build-release-artifacts.sh").read_text()
    mk = s.index('install -d "$output_dir/exl3"')
    loop = s.index('[[ "$release_family" == *:exl3-k* ]] || continue')
    assert mk < loop


def test_dockerfile_copies_exl3_and_skips_an_empty_family_glob():
    s = (REPO / "docker/Dockerfile.release").read_text()
    assert "COPY .cuteafd-release-image/exl3 /opt/cuteafd/lib/exl3" in s
    loop = s.index("for exl3_family in /opt/cuteafd/lib/exl3/exl3-*/; do")
    guard = s.index('[ -d "$exl3_family" ] || continue;', loop)
    verify = s.index("package_exl3_aot.py verify", loop)
    assert guard < verify


def test_build_verifies_dist_exl3_only_when_it_holds_a_package():
    s = (REPO / "build.sh").read_text()
    guard = s.index('if compgen -G "$repo_root/dist/$role/exl3/*" >/dev/null; then')
    verify = s.index('package_exl3_aot.py" verify --package "$repo_root/dist/$role/exl3"')
    assert guard < verify < s.index("fi", verify)


def test_spark_image_distribution_falls_back_to_ssh_without_rdmapipe():
    s = (REPO / "build.sh").read_text()
    assert 'release_die "rdmapipe is required on every Spark build host' not in s
    fb = s.index("rdmapipe missing on a build host: distributing Spark image over ssh")
    assert s.index('docker image save $spark_image_quoted"', fb) < s.index('"set -o pipefail; docker image load"', fb)
    # the identity gates that follow distribution still run for every build host
    assert fb < s.index('release_die "$host Spark image revision mismatch: $revision"')


def test_rdmapipe_stays_first_when_present():
    s = (REPO / "build.sh").read_text()
    rdma = s.index("concurrently distributing Spark image over RDMA")
    fallback = s.index("rdmapipe missing on a build host: distributing Spark image over ssh")
    assert rdma < fallback


def test_single_build_host_needs_no_rdmapipe():
    s = (REPO / "build.sh").read_text()
    guard = s.index('((${#RELEASE_BUILD_HOSTS[@]} > 1)) || rdmapipe_ready=2')
    probe = s.index('command -v rdmapipe >/dev/null 2>&1', guard)
    assert guard < probe
    assert 'echo "== single Spark build host: nothing to distribute =="' in s


def test_remote_free_space_measures_the_staging_directory_filesystem():
    s = (REPO / "build.sh").read_text()
    block = s.index('release_ssh "$seed_host" bash -s -- "$remote_dir"')
    resolve = s.index('while [ ! -e "$p" ]; do p="$(dirname "$p")"; done', block)
    df = s.index('df -Pk "$(readlink -f "$p")"', resolve)
    assert block < resolve < df
    # the old $HOME-only probe is gone
    assert 'df -Pk "$HOME" | awk \'NR == 2 { print $4 }\'' not in s


def test_remote_free_space_floor_is_overridable():
    s = (REPO / "build.sh").read_text()
    assert s.count("${CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB:-60}") >= 2
    assert 'release_die "$seed_host build needs at least ${CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB:-60} GiB free"' in s
    assert ('release_die "$seed_host release build root $release_build_root needs at least '
            '${CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB:-60} GiB free"') in s
