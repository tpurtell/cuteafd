import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5 import prepare_corpus as glm
from kimi_k3 import prepare_corpus as kimi


class NativeGlmTokenizer:
    def __init__(self, *, changed=False):
        self.calls, self.changed = [], changed

    def __len__(self):
        return 200000

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        text = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>"
        text += messages[0]["content"] + "<|assistant|><think>"
        if self.changed:
            text += "changed"
        return self.encode(text) if kwargs["tokenize"] else text


class GlmCorpusTest(unittest.TestCase):
    def saved(self, root):
        flat, offsets, report = glm.tokenize_rows(
            [dict(id="z", prompt="你好"), dict(id="a", prompt="x" * 2100)], NativeGlmTokenizer())
        path = root / "calibration.safetensors"
        save_file(dict(input_ids=torch.tensor(list(flat), dtype=torch.int64),
                       offsets=torch.tensor(offsets, dtype=torch.int64)), str(path))
        report.update(file=path.name, sha256=glm.sha256_file(path))
        manifest = dict(schema=glm.SCHEMA, status="passed", contract=glm.CONTRACT,
            dataset=dict(repo_id=glm.REPO_ID, revision=glm.REVISION, files=glm.PAYLOAD_SHA256),
            tokenizer=dict(vocab_size=200000), splits=dict(calibration=report))
        (root / "manifest.json").write_bytes(glm.canonical(manifest))
        return manifest

    def test_native_glm_kwargs_preserve_whole_prompts_and_order(self):
        t = NativeGlmTokenizer()
        rows = [dict(id="z", prompt="你好[gMASK]"), dict(id="a", prompt="x" * 2100)]
        flat, offsets, report = glm.tokenize_rows(rows, t)
        self.assertEqual([r["id"] for r in report["records"]], ["z", "a"])
        self.assertGreater(report["longest"], 2048)
        self.assertEqual(offsets[-1], len(flat))
        self.assertEqual(t.calls[0], ([dict(role="user", content=rows[0]["prompt"])], glm.CHAT_KWARGS))
        self.assertEqual(t.calls[2][1], glm.CHAT_KWARGS)
        self.assertNotIn("thinking_effort", t.calls[0][1])

    def test_saved_glm_tokens_use_shared_validated_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.saved(root)
            rows = glm.load_token_records(root)
            self.assertEqual([r["id"] for r in rows], ["z", "a"])
            batches = list(glm.load_token_batches(root))
            self.assertEqual([b["input_ids"].shape[1] for b in batches], [len(r["input_ids"]) for r in rows])
            for b in batches:
                self.assertEqual(b["input_ids"].dtype, torch.int64)
                self.assertTrue(torch.equal(b["attention_mask"], torch.ones_like(b["input_ids"])))
            with self.assertRaisesRegex(ValueError, "contract differs"):
                kimi.load_token_records(root)

    def test_native_probes_require_exact_text_and_id_equality(self):
        probes = glm.tokenizer_probes(NativeGlmTokenizer(), NativeGlmTokenizer())
        self.assertEqual(len(probes), 5)
        with self.assertRaisesRegex(ValueError, "differs from the native"):
            glm.tokenizer_probes(NativeGlmTokenizer(changed=True), NativeGlmTokenizer())

    def test_wrong_thinking_boundary_is_rejected_even_if_both_agree(self):
        with self.assertRaisesRegex(ValueError, "thinking/max-effort"):
            glm.tokenizer_probes(NativeGlmTokenizer(changed=True), NativeGlmTokenizer(changed=True))

    def test_wrong_glm_contract_cannot_load(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self.saved(root)
            manifest["contract"] = {**glm.CONTRACT, "reasoning_effort": "low"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "contract differs"):
                glm.load_token_records(root)


if __name__ == "__main__":
    unittest.main()
