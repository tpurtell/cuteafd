"""submodule-init-local: local mirror store, URL restore, and the opt-out fallback.

Every fixture is local: the "upstream" of each submodule is a bare repo under the
test's temporary directory, so nothing here touches the network.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "dev/submodule-init-local"
IDENTITY = {
    "GIT_AUTHOR_NAME": "script tests",
    "GIT_AUTHOR_EMAIL": "tests@example.invalid",
    "GIT_COMMITTER_NAME": "script tests",
    "GIT_COMMITTER_EMAIL": "tests@example.invalid",
}
# Git's own defaults must not decide whether a submodule clone is allowed, or the
# fallback would pass with the script's conditional file-transport flag removed.
HERMETIC = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
FILE_ALLOWED = ("-c", "protocol.file.allow=always")


def git(cwd, *args, env=None, check=True):
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True,
        env={**os.environ, **IDENTITY, **HERMETIC, **(env or {})})
    if check and result.returncode:
        raise AssertionError(f"git {' '.join(args)}: {result.stderr}")
    return result


def run_script(mirrors, worktree):
    env = {**os.environ, **HERMETIC, "CUTEAFD_GIT_MIRRORS": str(mirrors)}
    return subprocess.run(
        [str(SCRIPT), str(worktree)], text=True, capture_output=True, env=env)


@dataclass
class Fixture:
    """super --modules/sub--> middle --inner--> inner, with local bare upstreams."""

    root: Path
    super_repo: Path
    mirrors: Path
    middle_url: str
    inner_url: str
    middle_pin: str
    inner_pin: str
    stale_url: str  # middle's history up to its first commit, without the pin


def seed_bare(root, name, filename):
    """A bare repo whose master holds one commit; returns that commit."""
    repo = root / name
    git(root, "init", "-q", "--bare", "-b", "master", str(repo))
    work = root / f"{name}.work"
    git(root, "clone", "-q", str(repo), str(work))
    (work / filename).write_text(f"{name}\n")
    git(work, "add", filename)
    git(work, "commit", "-qm", f"add {filename}")
    git(work, "push", "-q", "origin", "master")
    return git(work, "rev-parse", "HEAD").stdout.strip()


class SubmoduleInitLocalTest(unittest.TestCase):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        inner_url = root / "up-inner.git"
        inner_pin = seed_bare(root, inner_url.name, "inner.txt")

        middle_work = root / "middle-work"
        git(root, "init", "-q", "-b", "master", str(middle_work))
        (middle_work / "middle.txt").write_text("middle\n")
        git(middle_work, "add", "middle.txt")
        git(middle_work, "commit", "-qm", "base")
        git(middle_work, *FILE_ALLOWED, "submodule", "add", "-q", str(inner_url), "inner")
        git(middle_work, "commit", "-qm", "add inner")
        middle_url = root / "middle.git"
        git(root, "init", "-q", "--bare", "-b", "master", str(middle_url))
        git(middle_work, "push", "-q", str(middle_url), "master")
        middle_pin = git(middle_work, "rev-parse", "HEAD").stdout.strip()
        stale_url = root / "stale.git"
        git(root, "init", "-q", "--bare", "-b", "master", str(stale_url))
        git(middle_work, "push", "-q", str(stale_url), "HEAD~1:refs/heads/master")

        super_repo = root / "super"
        git(root, "init", "-q", "-b", "master", str(super_repo))
        git(super_repo, *FILE_ALLOWED, "submodule", "add", "-q",
            str(middle_url), "modules/sub")
        git(super_repo, "commit", "-qm", "add submodule")
        return Fixture(root, super_repo, root / "mirrors", str(middle_url), str(inner_url),
                       middle_pin, inner_pin, str(stale_url))

    def test_mirror_store_restores_pins_and_upstream_urls(self):
        fixture = self.fixture()
        sub = fixture.super_repo / "modules/sub"

        result = run_script(fixture.mirrors, fixture.super_repo)
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertEqual(git(sub, "rev-parse", "HEAD").stdout.strip(), fixture.middle_pin)
        self.assertEqual(git(sub / "inner", "rev-parse", "HEAD").stdout.strip(),
                         fixture.inner_pin)
        # Slashes in a submodule path become '-' in the mirror name, and a nested
        # submodule gets a mirror of its own.
        self.assertTrue((fixture.mirrors / "modules-sub.git").is_dir())
        self.assertTrue((fixture.mirrors / "modules-sub-inner.git").is_dir())
        # Both the recorded URL and the submodule's own remote are the upstream.
        self.assertEqual(git(fixture.super_repo, "config",
                             "submodule.modules/sub.url").stdout.strip(), fixture.middle_url)
        self.assertEqual(git(sub, "config", "submodule.inner.url").stdout.strip(),
                         fixture.inner_url)
        self.assertEqual(git(sub, "remote", "get-url", "origin").stdout.strip(),
                         fixture.middle_url)
        self.assertEqual(git(sub / "inner", "remote", "get-url", "origin").stdout.strip(),
                         fixture.inner_url)
        self.assertIn("at pins", result.stdout)

    def test_default_worktree_is_the_current_repo_root(self):
        fixture = self.fixture()

        env = {**os.environ, **HERMETIC, "CUTEAFD_GIT_MIRRORS": str(fixture.mirrors)}
        result = subprocess.run([str(SCRIPT)], cwd=fixture.super_repo, text=True,
                                capture_output=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            git(fixture.super_repo / "modules/sub", "rev-parse", "HEAD").stdout.strip(),
            fixture.middle_pin)

    def test_initialized_submodule_fetches_the_pin_from_the_mirror(self):
        """A submodule left behind by an older clone must switch to the mirror.

        `git submodule update` fetches the pinned commit from the submodule's own
        origin, not from the URL recorded in the superproject, so a submodule
        initialized from a repo that lacks the pin has to be repointed first.
        """
        fixture = self.fixture()
        sub = fixture.super_repo / "modules/sub"
        # The store already holds the pin; the upstream it came from goes away.
        git(fixture.mirrors.parent, "clone", "-q", "--mirror", fixture.middle_url,
            str(fixture.mirrors / "modules-sub.git"))
        git(fixture.super_repo, "submodule", "deinit", "-f", "modules/sub")
        # deinit keeps the module's own clone behind; drop it so the next init
        # really starts from the stale repo.
        shutil.rmtree(fixture.super_repo / ".git/modules/modules/sub")
        git(fixture.super_repo, "config", "submodule.modules/sub.url", fixture.stale_url)
        stale_init = git(fixture.super_repo, *FILE_ALLOWED, "submodule", "update",
                         "--init", "modules/sub", check=False)
        self.assertNotEqual(stale_init.returncode, 0)
        self.assertEqual(git(sub, "remote", "get-url", "origin").stdout.strip(),
                         fixture.stale_url)
        Path(fixture.middle_url).rename(fixture.middle_url + ".moved")

        result = run_script(fixture.mirrors, fixture.super_repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git(sub, "rev-parse", "HEAD").stdout.strip(), fixture.middle_pin)
        self.assertEqual(git(sub / "inner", "rev-parse", "HEAD").stdout.strip(),
                         fixture.inner_pin)
        self.assertEqual(git(sub, "remote", "get-url", "origin").stdout.strip(),
                         fixture.middle_url)

    def test_mirror_is_fetched_when_it_lacks_the_pin(self):
        """A stored mirror older than the pin is fetched, not re-cloned or skipped."""
        fixture = self.fixture()
        mirror = fixture.mirrors / "modules-sub.git"
        # The store holds a mirror of the stale repo; its origin learns the pin
        # only afterwards, and the original upstream disappears.
        git(fixture.mirrors.parent, "clone", "-q", "--mirror", fixture.stale_url, str(mirror))
        git(fixture.super_repo, "submodule", "deinit", "-f", "modules/sub")
        shutil.rmtree(fixture.super_repo / ".git/modules/modules/sub")
        git(Path(fixture.middle_url).parent / "middle-work", "push", "-q",
            fixture.stale_url, "master")
        Path(fixture.middle_url).rename(fixture.middle_url + ".moved")

        result = run_script(fixture.mirrors, fixture.super_repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git(mirror, "rev-parse", "refs/heads/master").stdout.strip(),
                         fixture.middle_pin)
        sub = fixture.super_repo / "modules/sub"
        self.assertEqual(git(sub, "rev-parse", "HEAD").stdout.strip(), fixture.middle_pin)
        self.assertEqual(git(sub / "inner", "rev-parse", "HEAD").stdout.strip(),
                         fixture.inner_pin)

    def test_a_failed_run_still_restores_upstream_urls(self):
        """A run that stops partway must not leave a submodule pointed at the mirror.

        The second submodule's upstream is gone and has no mirror, so the run ends
        with the upstream clone failing; the first submodule and its nested
        submodule have already been cloned from the store by then.
        """
        fixture = self.fixture()
        seed_bare(fixture.root, "gone.git", "gone.txt")
        gone_url = fixture.root / "gone.git"
        git(fixture.super_repo, *FILE_ALLOWED, "submodule", "add", "-q",
            str(gone_url), "modules/gone")
        git(fixture.super_repo, "commit", "-qm", "add an unreachable submodule")
        git(fixture.super_repo, "submodule", "deinit", "-f", "modules/gone")
        shutil.rmtree(fixture.super_repo / ".git/modules/modules/gone")
        shutil.rmtree(gone_url)

        result = run_script(fixture.mirrors, fixture.super_repo)
        self.assertNotEqual(result.returncode, 0)

        sub = fixture.super_repo / "modules/sub"
        self.assertEqual(git(fixture.super_repo, "config",
                             "submodule.modules/sub.url").stdout.strip(), fixture.middle_url)
        self.assertEqual(git(sub, "remote", "get-url", "origin").stdout.strip(),
                         fixture.middle_url)
        self.assertEqual(git(sub / "inner", "remote", "get-url", "origin").stdout.strip(),
                         fixture.inner_url)

    def test_second_run_reuses_the_mirror(self):
        fixture = self.fixture()
        self.assertEqual(run_script(fixture.mirrors, fixture.super_repo).returncode, 0)

        marker = fixture.mirrors / "modules-sub.git" / "REUSE-MARKER"
        marker.write_text("a re-clone would have removed me\n")
        second = run_script(fixture.mirrors, fixture.super_repo)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue(marker.exists())

    def test_off_falls_back_to_upstream(self):
        fixture = self.fixture()

        result = run_script("off", fixture.super_repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUTEAFD_GIT_MIRRORS=off", result.stderr)
        self.assertEqual(
            git(fixture.super_repo / "modules/sub", "rev-parse", "HEAD").stdout.strip(),
            fixture.middle_pin)
        self.assertEqual(
            git(fixture.super_repo / "modules/sub", "remote", "get-url",
                "origin").stdout.strip(), fixture.middle_url)
        self.assertFalse(fixture.mirrors.exists())

    def test_unusable_store_falls_back_to_upstream(self):
        fixture = self.fixture()
        blocker = fixture.mirrors.parent / "blocker"
        blocker.write_text("a file where the store's parent should be a directory\n")

        result = run_script(blocker / "mirrors", fixture.super_repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cannot create the mirror store", result.stderr)
        self.assertEqual(
            git(fixture.super_repo / "modules/sub", "rev-parse", "HEAD").stdout.strip(),
            fixture.middle_pin)
        self.assertEqual(
            git(fixture.super_repo / "modules/sub", "remote", "get-url",
                "origin").stdout.strip(), fixture.middle_url)


if __name__ == "__main__":
    unittest.main()
