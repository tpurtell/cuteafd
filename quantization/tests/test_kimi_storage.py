"""Fault-injected rolling-boundary and byte-preserving export contracts."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.uniform_k2 import storage


class SimulatedCrash(RuntimeError):
    pass


class KimiStorageTest(unittest.TestCase):
    def make_batch(self, boundary, layer, value):
        path = boundary.directory(layer) / "batch-000.safetensors"
        receipt = storage.commit_safetensors(path, dict(hidden=torch.tensor([[value]], dtype=torch.bfloat16)))
        return dict(filename=path.name, **receipt)

    def make_initial(self, root):
        boundary = storage.RollingBoundary(root, dict(plan="immutable-run"))
        batch = self.make_batch(boundary, -1, 1)
        initial = boundary.commit(-1, [batch])
        return boundary, initial

    def test_failed_successor_tensor_publication_preserves_old_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary, initial = self.make_initial(Path(directory) / "boundaries")
            old_path = boundary.directory(-1) / "batch-000.safetensors"
            old_bytes = old_path.read_bytes()
            replace = storage.os.replace
            def crash_on_tensor(source, target):
                if Path(target).name == "batch-000.safetensors":
                    raise SimulatedCrash("tensor rename")
                return replace(source, target)
            with mock.patch.object(storage.os, "replace", side_effect=crash_on_tensor):
                with self.assertRaisesRegex(SimulatedCrash, "tensor rename"):
                    self.make_batch(boundary, 0, 2)
            restarted = storage.RollingBoundary(boundary.root, boundary.identity)
            self.assertEqual(restarted.current(), initial)
            restarted.retire_old()
            self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertEqual(list(boundary.directory(0).iterdir()), [])
            # An aborted write is retried without replacing the committed bank.
            batch = self.make_batch(restarted, 0, 2)
            restarted.commit(0, [batch])
            self.assertFalse(old_path.exists())

    def test_crash_before_pointer_commit_preserves_old_boundary_and_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary, initial = self.make_initial(Path(directory) / "boundaries")
            old_bytes = (boundary.directory(-1) / "batch-000.safetensors").read_bytes()
            batch = self.make_batch(boundary, 0, 2)
            atomic_json = storage.atomic_json
            def crash_before_current(path, value):
                if Path(path).name == "current.json":
                    raise SimulatedCrash("before pointer")
                return atomic_json(path, value)
            with mock.patch.object(storage, "atomic_json", side_effect=crash_before_current):
                with self.assertRaisesRegex(SimulatedCrash, "before pointer"):
                    boundary.commit(0, [batch])
            restarted = storage.RollingBoundary(boundary.root, boundary.identity)
            self.assertEqual(restarted.current(), initial)
            self.assertTrue((boundary.directory(0) / "manifest.json").exists())
            restarted.retire_old()
            self.assertEqual((boundary.directory(-1) / "batch-000.safetensors").read_bytes(), old_bytes)
            self.assertTrue(boundary.directory(0).exists())
            restarted.commit(0, [batch])
            self.assertEqual(restarted.current()["layer"], 0)
            self.assertFalse(boundary.directory(-1).exists())

    def test_crash_after_pointer_commit_recovers_successor_and_retires_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary, _ = self.make_initial(Path(directory) / "boundaries")
            batch = self.make_batch(boundary, 0, 2)
            with mock.patch.object(boundary, "retire_old", side_effect=SimulatedCrash("after pointer")):
                with self.assertRaisesRegex(SimulatedCrash, "after pointer"):
                    boundary.commit(0, [batch])
            restarted = storage.RollingBoundary(boundary.root, boundary.identity)
            self.assertEqual(restarted.current()["layer"], 0)
            self.assertTrue(boundary.directory(-1).exists())
            restarted.retire_old()
            self.assertFalse(boundary.directory(-1).exists())
            self.assertEqual(load_file(str(boundary.directory(0) / batch["filename"]))["hidden"].item(), 2)
            restarted.retire_old()
            self.assertEqual(restarted.current()["layer"], 0)

    def test_retirement_keeps_current_successor_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary, _ = self.make_initial(Path(directory) / "boundaries")
            batch = self.make_batch(boundary, 0, 2)
            self.make_batch(boundary, 1, 3)
            self.make_batch(boundary, -2, 9)
            unrelated = boundary.root / "layer-not-a-number"
            unrelated.mkdir()
            (unrelated / "keep").write_bytes(b"unrelated")
            marker = boundary.root / "keep.log"
            marker.write_bytes(b"log")
            boundary.commit(0, [batch])
            self.assertFalse(boundary.directory(-1).exists())
            self.assertFalse(boundary.directory(-2).exists())
            self.assertTrue(boundary.directory(0).exists())
            self.assertTrue(boundary.directory(1).exists())
            self.assertEqual((unrelated / "keep").read_bytes(), b"unrelated")
            self.assertEqual(marker.read_bytes(), b"log")

    def test_retirement_refuses_old_symlink_without_deleting_its_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary, initial = self.make_initial(root / "boundaries")
            outside = root / "outside"
            outside.mkdir()
            (outside / "keep").write_bytes(b"external")
            boundary.directory(-2).symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                boundary.retire_old()
            self.assertEqual((outside / "keep").read_bytes(), b"external")
            self.assertEqual(boundary.current(), initial)

    def test_missing_or_changed_committed_batch_cannot_be_retired(self):
        for mutation in ("missing", "changed"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                boundary, _ = self.make_initial(Path(directory) / "boundaries")
                path = boundary.directory(-1) / "batch-000.safetensors"
                if mutation == "missing":
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes() + b"corrupt")
                self.make_batch(boundary, 0, 2)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    boundary.current()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    boundary.retire_old()
                self.assertTrue(boundary.directory(-1).exists())
                self.assertTrue(boundary.directory(0).exists())

    def test_changed_manifest_or_run_identity_fails_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary, _ = self.make_initial(Path(directory) / "boundaries")
            other = storage.RollingBoundary(boundary.root, dict(plan="different-run"))
            with self.assertRaisesRegex(ValueError, "run identity"):
                other.current()
            manifest = boundary.directory(-1) / "manifest.json"
            content = json.loads(manifest.read_text())
            content["layer"] = 123
            manifest.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                boundary.current()

    def test_boundary_cannot_skip_layers_or_publish_duplicate_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary, initial = self.make_initial(Path(directory) / "boundaries")
            batch = self.make_batch(boundary, 1, 2)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                boundary.commit(1, [batch])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                boundary.commit(0, [batch, batch])
            self.assertEqual(boundary.current(), initial)

    def raw_payload(self, path, name):
        header, base = storage.read_header(path)
        start, end = header[name]["data_offsets"]
        with Path(path).open("rb") as source:
            source.seek(base + start)
            return source.read(end - start)

    def test_assembly_copies_exact_selected_bytes_and_valid_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, target = root / "first.safetensors", root / "second.safetensors", root / "out.safetensors"
            # Arbitrary BF16 bit patterns, including NaN payloads, must survive
            # assembly without decoding or floating-point canonicalization.
            bf16 = torch.tensor([0x7FC1, 0x8000, 0x3F80, 0xFFFF], dtype=torch.uint16).view(torch.bfloat16).reshape(2, 2)
            save_file(dict(weight=bf16, unused=torch.ones(7)), str(first))
            save_file(dict(counter=torch.tensor(2147483647, dtype=torch.int32),
                           mask=torch.tensor([True, False, True])), str(second))
            original_hashes = [storage.sha256_file(first), storage.sha256_file(second)]
            entries = {"z.weight": (first, "weight"), "a.counter": (second, "counter"), "m.mask": (second, "mask")}
            receipt = storage.assemble_shard(entries, target, chunk_bytes=3)
            header, base = storage.read_header(target)
            expected_payload = b"".join(self.raw_payload(source, name) for _, (source, name) in sorted(entries.items()))
            self.assertEqual(target.read_bytes()[base:], expected_payload)
            self.assertEqual(receipt["payload_bytes"], 15)
            self.assertEqual(receipt["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertEqual(receipt["file_identity"]["bytes"], target.stat().st_size)
            self.assertEqual(header["a.counter"], dict(dtype="I32", shape=[], data_offsets=[0, 4]))
            self.assertEqual(header["m.mask"], dict(dtype="BOOL", shape=[3], data_offsets=[4, 7]))
            self.assertEqual(header["z.weight"], dict(dtype="BF16", shape=[2, 2], data_offsets=[7, 15]))
            loaded = load_file(str(target))
            self.assertTrue(torch.equal(loaded["z.weight"].view(torch.uint16), bf16.view(torch.uint16)))
            self.assertEqual([storage.sha256_file(first), storage.sha256_file(second)], original_hashes)
            with self.assertRaises(FileExistsError):
                storage.assemble_shard(entries, target)
            self.assertEqual(receipt["sha256"], storage.sha256_file(target))

    def test_truncated_source_during_copy_never_publishes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.safetensors", root / "out.safetensors"
            save_file(dict(weight=torch.arange(10, dtype=torch.int64)), str(source))
            read_header = storage.read_header
            def truncate_after_header(path, *, offset=0, length=None):
                result = read_header(path, offset=offset, length=length)
                with Path(path).open("r+b") as file:
                    file.truncate(result[1] + 1)
                return result
            with mock.patch.object(storage, "read_header", side_effect=truncate_after_header):
                with self.assertRaisesRegex(ValueError, "truncated source tensor"):
                    storage.assemble_shard(dict(weight=(source, "weight")), target, chunk_bytes=3)
            self.assertFalse(target.exists())

    def test_noncontiguous_source_payload_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.safetensors", root / "out.safetensors"
            header = json.dumps(dict(weight=dict(dtype="U8", shape=[1], data_offsets=[1, 2]))).encode()
            source.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\1")
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                storage.assemble_shard(dict(weight=(source, "weight")), target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
