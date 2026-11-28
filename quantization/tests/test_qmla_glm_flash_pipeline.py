"""Exercise GLM NoPE calibration, frozen forward provenance and release scope."""
import hashlib
import os
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash import qmla_publish, qmla_run as run
from qmla.capture import CapturedLayerBank, CapturedLayerParams, CapturedSequence, MlaLayerGeometry, eval_attention_output
from qmla.fit import FitConfig
from qmla.pipeline import calibrate_from_banks, corpus_split_description, check_reused_captures
import panel_corpus


def mc_snapshot():
    hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    root = hub / ("models--" + run.TARGET_REPO.replace("/", "--")) / "snapshots" / run.TARGET_REVISION
    if not root.is_dir():
        pytest.skip("pinned GLM Flash MUL1 snapshot unavailable")
    return root


def test_glm_flash_release_scope_and_presets(tmp_path):
    argv = ["--checkpoint", str(tmp_path), "--source-revision", run.TARGET_REVISION, "--output", str(tmp_path)]
    p = run.build_parser()
    eligible = tuple(range(3, 44, 4))
    for flags, layers, counts in ((["--pilot"], run.PILOT_LAYERS, (16, 4, 4, 100)),
                                 # Full runs: counts resolve to 60/30/10 of the loaded base-component rows.
                                 ([], eligible, (None, None, None, 300))):
        settings = run.resolve_run_settings(p.parse_args(argv + flags), eligible)
        assert settings["layers"] == layers
        assert settings["modes"] == ((8, 0), (4, 0), (2, 0))
        assert tuple(settings[k] for k in ("train_sequences", "val_sequences", "test_sequences", "steps")) == counts
    with pytest.raises(ValueError, match="RoPE"):
        run.resolve_run_settings(p.parse_args(argv + ["--modes", "4:4"]), eligible)
    publish_parser = qmla_publish.build_parser()
    safe = ["--source-revision", run.TARGET_REVISION, "--sidecar-dir", str(tmp_path), "--state-root", str(tmp_path)]
    assert publish_parser.parse_args(safe).update_mc_manifest
    for flag in ("--readme", "--validation-report"):
        with pytest.raises(SystemExit):
            publish_parser.parse_args(safe + [flag, "file"])


def test_glm_real_tokenizer_shared_fit_and_capture_identity(tmp_path):
    from transformers import PreTrainedTokenizerFast

    root = mc_snapshot()
    args = run.build_parser().parse_args(["--checkpoint", str(root), "--source-revision", run.TARGET_REVISION,
                                          "--output", str(tmp_path)])
    source, capture_id = run.read_identities(root, args)
    assert source["forward"]["routed_experts"] == "exl3-k4-mul1"
    assert source["forward"]["index_sha256"] == hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="shared K4 experts"):
        run.read_identities(root, type(args)(**{**vars(args), "teacher_checkpoint": root}))
    # The real tokenizer provides unique sequence IDs and token hashes; seeded synthetic
    # layer weights exercise the same fitter and sidecar validator without a GPU model load.
    tok = PreTrainedTokenizerFast.from_pretrained(root, local_files_only=True)
    gen = torch.Generator().manual_seed(403)
    records = []
    for i in range(3):
        text = tok.decode(torch.randint(20, 2000, (20,), generator=gen).tolist())
        ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False)
        records.append((f"s{i}", torch.tensor(ids)))
    splits = corpus_split_description(records, train_count=1, val_count=1, test_count=1, seed=0)
    geom = MlaLayerGeometry(layer_id=3, kv_source_layer=3, kind="dsa", heads=2, kv_lora_rank=64,
                            qk_nope_dim=8, qk_rope_dim=0, v_head_dim=8, rope_kind="none")
    def rand(*shape):
        return torch.randn(*shape, generator=gen)
    params = CapturedLayerParams(kv_norm_weight=torch.ones(64), kv_b_weight=rand(32, 64), o_proj_weight=rand(12, 16))
    inputs = dict(records)

    def sequence(name):
        sid, = splits[name]["seq_ids"]
        count = len(inputs[sid])
        seq = CapturedSequence(seq_id=sid, positions=torch.arange(count), u=rand(count, 64), q_nope=rand(count, 2, 8),
                               q_rope=None, k_rope=None, allowed_mask=None, gate=None, ref_output=torch.zeros(count, 12))
        return replace(seq, ref_output=eval_attention_output(seq, params, geom))

    hashes = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in records}
    components = {key: {"sequences": 1, "tokens": len(record[1]), "input_ids_sha256": hashes[record[0]]}
                  for key, record in zip(panel_corpus.V3_PANEL.components, records, strict=True)}
    prov = {**capture_id, "dataset_repo": panel_corpus.V3_PANEL.repo_id, "dataset_revision": panel_corpus.V3_PANEL.revision,
            "components": components, "input_ids_sha256": hashlib.sha256("".join(hashes.values()).encode()).hexdigest(),
            "splits": splits}
    bank = CapturedLayerBank(geom, params, (sequence("train"),), (sequence("val"),), (sequence("test"),), prov)
    with pytest.raises(ValueError, match="identity"):
        check_reused_captures({3: replace(bank, provenance={k: v for k, v in prov.items() if k not in capture_id})},
                              {}, capture_id, seed=0)
    measured = calibrate_from_banks({3: bank}, output_dir=tmp_path, source=source, modes=((8, 0), (4, 0), (2, 0)),
                                    corpus_provenance=prov, fit_config=FitConfig(steps=1, val_every=1,
                                    sink_tokens=2, recent_tokens=4), device="cpu")
    m = measured["manifest"]
    assert measured["validation"]["status"] == "passed"
    assert m["profiles"]["rope"] is None
    assert set(m["supported_modes"]) == set(run.MODES)
    assert m["source"]["forward"] == capture_id["capture_forward"]
    with pytest.raises(ValueError, match="all 11 MLA"):
        qmla_publish.verify_release(type(args)(sidecar_dir=tmp_path / "sidecar", repo_id=run.TARGET_REPO,
                                                source_revision=run.TARGET_REVISION))
