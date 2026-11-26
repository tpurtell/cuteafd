import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.hf_api import RepoFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upload_model import upload_artifact
from materialize_cache import materialize_cache


class FakeApi:
    """Hub fake for parent-guarded HTTP LFS batches; records every transfer."""

    def __init__(self):
        self.head = "a" * 40
        self.files = {}
        self.trees = {self.head: {}}
        self.uploaded = []
        self.batches = []
        self.commits = 0
        self.fail_upload = False
        self.lose_commit_response = False

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(sha=self.head)

    def list_repo_tree(self, repo, revision=None, **kwargs):
        return list(self.trees[revision].values())

    def create_commit(self, repo, operations, **kwargs):
        assert kwargs["parent_commit"] == self.head
        names = [op.path_in_repo for op in operations]
        assert not set(names) & set(self.files), "committed file sent again"
        files = dict(self.files)
        for op in operations:
            assert op.upload_info.is_hashed and not op._is_uploaded
            if self.fail_upload and self.uploaded:
                raise RuntimeError("transport failed")
            self.uploaded.append(op.path_in_repo)
            with op.as_file() as stream:
                payload = stream.read()
            assert op.upload_info.sha256 == hashlib.sha256(payload).digest()
            if op.path_in_repo.endswith(".safetensors"):
                blob = op.upload_info.sha256.hex()
                lfs = dict(oid=blob, size=op.upload_info.size, pointerSize=130)
            else:
                blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                lfs = None
            files[op.path_in_repo] = RepoFile(path=op.path_in_repo, size=op.upload_info.size,
                                              oid=blob, lfs=lfs)
        self.files = files
        self.batches.append(names)
        self.commits += 1
        self.head = f"{self.commits + 10:040x}"
        self.trees[self.head] = dict(files)
        if self.lose_commit_response:
            self.lose_commit_response = False
            raise RuntimeError("lost commit response")
        return SimpleNamespace(oid=self.head)


class UploadTest(unittest.TestCase):
    def fixture(self, root):
        output = root / "export"
        output.mkdir()
        for name in (".gitattributes", "README.md", "config.json", "model.safetensors.index.json", "model.safetensors"):
            (output / name).write_bytes(name.encode())
        return output, root / "state", FakeApi()

    def test_upload_receipt_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, state, api = self.fixture(root)
            receipt = upload_artifact(output, state, "test/model", api=api)
            self.assertEqual(receipt["status"], "uploaded")
            self.assertEqual(upload_artifact(output, state, "test/model", api=api, resume=True), receipt)
            self.assertEqual(api.commits, 1)
            self.assertEqual(len(api.uploaded), 5)
            self.assertEqual(materialize_cache(output, root / "hub", receipt)["status"], "materialized")

    def test_batches_use_options_and_resume_skips_committed_files(self):
        import hub_lfs
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            options = hub_lfs.UploadOptions(workers=7, batch_files=2)
            api.fail_upload = True
            create = api.create_commit
            calls = []

            def counted(repo, operations, **kwargs):
                calls.append(kwargs["num_threads"])
                api.fail_upload = len(calls) == 2
                return create(repo, operations, **kwargs)

            api.create_commit = counted
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                upload_artifact(output, state, "test/model", api=api, options=options)
            committed = set(api.files)
            self.assertEqual(len(committed), 2)
            api.fail_upload = False
            upload_artifact(output, state, "test/model", api=api, resume=True, options=options)
            sent = [name for batch in api.batches for name in batch]
            self.assertEqual(len(sent), len(set(sent)))
            self.assertEqual(set(sent), set(api.files))
            self.assertEqual(set(calls), {7})
            self.assertTrue(all(len(batch) <= 2 for batch in api.batches))
            self.assertFalse(committed & {n for batch in api.batches[1:] for n in batch})

    def test_resume_reuses_hash_cache_without_rereading(self):
        import hub_lfs
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            api.lose_commit_response = True
            with self.assertRaisesRegex(RuntimeError, "lost commit response"):
                upload_artifact(output, state, "test/model", api=api)
            with patch.object(hub_lfs.hashlib, "file_digest", side_effect=AssertionError("rehash")):
                receipt = upload_artifact(output, state, "test/model", api=api, resume=True)
            self.assertEqual(receipt["commit"], api.head)
            self.assertEqual(api.commits, 1)

    def test_lost_commit_response_recovers_without_reupload_or_recommit(self):
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            api.lose_commit_response = True
            with self.assertRaisesRegex(RuntimeError, "lost commit response"):
                upload_artifact(output, state, "test/model", api=api)
            with patch.object(api, "create_commit", side_effect=AssertionError("must not reupload")):
                receipt = upload_artifact(output, state, "test/model", api=api, resume=True)
            self.assertEqual(receipt["commit"], api.head)
            self.assertEqual(api.commits, 1)

    def test_source_change_mid_batch_is_refused(self):
        import hub_lfs
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            create = api.create_commit

            def mutate(repo, operations, **kwargs):
                result = create(repo, operations, **kwargs)
                (output / operations[0].path_in_repo).write_bytes(b"changed during transfer")
                return result

            api.create_commit = mutate
            with self.assertRaisesRegex(ValueError, "changed"):
                upload_artifact(output, state, "test/model", api=api,
                                options=hub_lfs.UploadOptions(batch_files=2))
            self.assertEqual(api.commits, 1)
            self.assertFalse((state / "upload-complete.json").exists())

    def test_populated_remote_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            api.trees[api.head]["existing"] = RepoFile(path="existing", size=3, oid="c" * 40)
            with self.assertRaisesRegex(ValueError, "populated"):
                upload_artifact(output, state, "test/model", api=api)
            self.assertEqual(api.uploaded, [])
            self.assertEqual(api.commits, 0)
