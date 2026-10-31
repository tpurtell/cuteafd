"""Muse Spark / Mistral Large 4 reference adapters against the official modules.

Tiny random configs, written in each checkpoint's own tensor naming and storage
formats, run through (a) the official full text model and (b) the layer-major
adapter path; logits must agree, and fixed-geometry prefixes must be bit-exact.
"""
import contextlib
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
if not hasattr(transformers, "MistralLarge4TextConfig") or not hasattr(transformers, "MuseSparkTextConfig"):
    pytest.skip("transformers lacks Muse Spark / Mistral Large 4", allow_module_level=True)
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from families.muse_spark import model as muse
from families.mistral_large4 import model as mistral


@pytest.fixture(autouse=True)
def f32():
    # CPU elementwise kernels split vector loops by length across threads; the
    # golden's prefix proof runs on CUDA, where elementwise ops are per element.
    previous, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous)
    torch.set_num_threads(threads)


def tiny_muse_config():
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
    layers = 4
    config = MuseSparkTextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=64, moe_hidden_size=32, moe_intermediate_size=48,
        num_hidden_layers=layers, num_attention_heads=8, num_key_value_heads=2, head_dim=8,
        num_local_experts=6, num_experts_per_tok=2, sliding_window=5,
        layer_types=["sliding_attention", "full_attention"] * 2, mlp_layer_types=["sparse"] * layers,
        layer_rope_theta=[500000.0, 0, 500000.0, 0], rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        pad_token_id=0)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def write_checkpoint(tmp_path, tensors, index_name):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(snapshot / "shard.safetensors"))
    (snapshot / index_name).write_text(json.dumps({"weight_map": {k: "shard.safetensors" for k in tensors}}))
    return snapshot


def randomize(module):
    with torch.no_grad():
        for name, value in list(module.named_parameters()) + list(module.named_buffers()):
            if value.dtype.is_floating_point and "inv_freq" not in name:
                value.copy_(torch.randn_like(value) * (0.3 if value.ndim > 1 else 0.5))


def test_muse_layer_major_matches_official_model(tmp_path):
    from transformers.models.muse_spark import modeling_muse_spark as reference
    config = tiny_muse_config()
    official = reference.MuseSparkTextModel(config).eval()
    randomize(official)
    head = torch.randn(config.vocab_size, config.hidden_size) * 0.3
    tensors = {muse.PREFIX + k: v.detach().clone() for k, v in official.state_dict().items()
               if "rotary" not in k}
    tensors[muse.HEAD] = head.bfloat16().float()
    # The checkpoint ships BF16 weights with the F32 router and bias.
    for key, value in tensors.items():
        if not key.endswith(("mlp.gate.weight", "e_score_correction_bias")):
            tensors[key] = value.bfloat16()
    official.load_state_dict({k[len(muse.PREFIX):]: v.float() for k, v in tensors.items() if k.startswith(muse.PREFIX)},
                             strict=False)
    snapshot = write_checkpoint(tmp_path, tensors, "model.safetensors.index.json")
    ids = torch.randint(1, config.vocab_size, (1, 23))
    with torch.inference_mode():
        expected = muse.logits(official(input_ids=ids).last_hidden_state, head.bfloat16().float(), config)
        checkpoint = muse.Checkpoint(snapshot)
        outer = muse.load_outer(checkpoint, config, device="cpu", dtype=torch.float32)
        hidden = outer["embed"](ids)
        routes = []
        for index in range(config.num_hidden_layers):
            layer = muse.load_layer(checkpoint, config, index, device="cpu", dtype=torch.float32)
            assert isinstance(layer.mlp.experts.gate_up_proj, muse.LazyExperts)
            with muse.capture_routes(layer) as records:
                hidden = muse.layer_forward(layer, config, index, hidden, outer["rotary"])
            routes.append(records[0][0])
        actual = muse.logits(outer["norm"](hidden), outer["head"], config)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert all(r.shape == (ids.shape[1], config.num_experts_per_tok) for r in routes)


def test_muse_fixed_geometry_prefix_is_bit_exact(tmp_path):
    from transformers.models.muse_spark import modeling_muse_spark as reference
    config = tiny_muse_config()
    layer = reference.MuseSparkTextDecoderLayer(config, 0).eval()
    randomize(layer)
    rotary = reference.MuseSparkTextRotaryEmbedding(config)
    hidden = torch.randn(1, 300, config.hidden_size)
    with torch.inference_mode(), muse.geometry(reference):
        long = muse.layer_forward(layer, config, 0, hidden, rotary)
        short = muse.layer_forward(layer, config, 0, hidden[:, :170], rotary)
    plain = muse.layer_forward(layer, config, 0, hidden, rotary)
    assert torch.equal(long[:, :170], short)
    assert torch.allclose(long, plain, atol=1e-5, rtol=1e-5)


def test_muse_source_pins_fail_closed(monkeypatch):
    monkeypatch.setattr(muse, "SOURCE_PINS", {**muse.SOURCE_PINS, "masking_utils.py": "0" * 64})
    with pytest.raises(ValueError, match="source differs"):
        muse.source_identity()


def tiny_mistral_config():
    from transformers.models.mistral_large4.configuration_mistral_large4 import MistralLarge4TextConfig
    layers = 8
    config = MistralLarge4TextConfig(
        vocab_size=89, hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        kv_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32, head_dim=16,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
        index_n_heads=4, index_head_dim=32, index_topk=128,
        mlp_layer_types=["dense"] + ["sparse"] * (layers - 1),
        indexer_types=["full", "full", "shared", "full", "full", "shared", "shared", "full"],
        kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3], no_rope_layers=[1, 1, 1, 0, 1, 1, 1, 0],
        layer_types=["indexed_attention"] * layers, final_logit_softcapping=30.0)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def power_of_two_fp8(weight):
    rows, cols = weight.shape
    sr, sc = (rows + 127) // 128, (cols + 127) // 128
    scale = torch.exp2(torch.randint(-9, -5, (sr, sc)).float())
    grown = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    quantized = (weight / grown).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale, quantized.float() * grown


def mistral_checkpoint(official, config, head):
    """Official text state -> consolidated names; FP8 for the converted modules."""
    inverse = {v: k for k, v in mistral._ATTENTION.items()}
    fp8 = ("q_proj", "o_proj", "indexer.wq_b", "experts", "shared_experts", "gate_proj", "up_proj", "down_proj")
    keep = ("kv_b_proj", "indexer.wk", "indexer.weights_proj", "kv_a_proj_with_mqa", "mlp.gate.weight")
    tensors, exact = {}, {}

    def put(name, value, official_name):
        if value.ndim == 2 and any(t in official_name for t in fp8) and not any(t in official_name for t in keep):
            q, s, v = power_of_two_fp8(value)
            tensors[name], tensors[name.removesuffix("weight") + "qscale_weight"] = q, s
            exact[official_name] = v
        else:
            tensors[name] = value.bfloat16()
            exact[official_name] = value.bfloat16().float()

    for index, layer in enumerate(official.layers):
        mapping = mistral.layer_map(config, index, layer)
        params = dict(layer.named_parameters())
        for name, source in mapping.items():
            if isinstance(source, str):
                put(source, params[name].detach(), f"layers.{index}.{name}")
                continue
            stack = params[name].detach()
            values = []
            for e, item in enumerate(source):
                if isinstance(item, str):
                    put(item, stack[e], f"layers.{index}.{name}.{e}")
                    values.append(exact[f"layers.{index}.{name}.{e}"])
                else:
                    half = stack.shape[1] // 2
                    put(item[0], stack[e, :half], f"layers.{index}.{name}.{e}.g")
                    put(item[1], stack[e, half:], f"layers.{index}.{name}.{e}.u")
                    values.append(torch.cat([exact[f"layers.{index}.{name}.{e}.g"],
                                             exact[f"layers.{index}.{name}.{e}.u"]]))
            exact[f"layers.{index}.{name}"] = torch.stack(values)
    tensors["tok_embeddings.weight"] = official.embed_tokens.weight.detach().bfloat16()
    tensors["norm.weight"] = official.norm.weight.detach().bfloat16()
    tensors["lerope.alpha"] = (torch.randn(official.rotary_emb.alpha.shape) * 0.1).bfloat16()
    tensors["output.weight"] = head.bfloat16()
    return tensors, exact


def test_mistral_layer_major_matches_official_model(tmp_path):
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    config = tiny_mistral_config()
    official = reference.MistralLarge4TextModel(config).eval()
    randomize(official)
    head = torch.randn(config.vocab_size, config.hidden_size) * 0.3
    tensors, exact = mistral_checkpoint(official, config, head)
    state = {}
    for index, layer in enumerate(official.layers):
        for name, _ in layer.named_parameters():
            state[f"layers.{index}.{name}"] = exact[f"layers.{index}.{name}"]
    state["embed_tokens.weight"] = tensors["tok_embeddings.weight"].float()
    state["norm.weight"] = tensors["norm.weight"].float()
    official.load_state_dict(state, strict=False)
    official.rotary_emb.alpha.copy_(tensors["lerope.alpha"].float())
    snapshot = write_checkpoint(tmp_path, tensors, "consolidated.safetensors.index.json")
    ids = torch.randint(1, config.vocab_size, (1, 19))
    with torch.inference_mode():
        expected = mistral.logits(official(input_ids=ids).last_hidden_state, head.bfloat16().float(), config)
        checkpoint = mistral.Checkpoint(snapshot)
        outer = mistral.load_outer(checkpoint, config, device="cpu", dtype=torch.float32)
        hidden = torch.nn.functional.embedding(ids, outer["embed"])
        shared, topk = {}, None
        for index in range(config.num_hidden_layers):
            layer = mistral.load_layer(checkpoint, config, index, device="cpu", dtype=torch.float32)
            hidden, topk = mistral.layer_forward(layer, config, hidden, outer["rotary"], shared, topk)
        actual = mistral.logits(outer["norm"](hidden), outer["head"], config)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mistral_fixed_geometry_prefix_is_bit_exact_and_matches_untiled():
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    config = tiny_mistral_config()
    official = reference.MistralLarge4TextModel(config).eval()
    randomize(official)
    hidden = torch.randn(1, 400, config.hidden_size) * 0.5

    def run(x, tiled):
        shared, topk, out = {}, None, x
        context = mistral.geometry(reference) if tiled else stable_topk(reference)
        with torch.inference_mode(), context:
            for layer in official.layers:
                out, topk = mistral.layer_forward(layer, config, out, official.rotary_emb, shared, topk)
        return out
    long, short = run(hidden, True), run(hidden[:, :290], True)
    assert torch.equal(long[:, :290], short)
    # Untiled official indexer with the same deterministic tie rule (relu makes exact
    # 0.0 ties common, and official topk leaves their order undefined).
    untiled = run(hidden, False)
    assert ((long - untiled).norm() / untiled.norm()).item() < 1e-5


@contextlib.contextmanager
def stable_topk(reference):
    """Untiled official indexer whose only change is the stable tie rule."""
    previous = torch.Tensor.topk

    def topk(self, k, dim=-1, largest=True, sorted=True):
        if self.dtype == torch.float32 and dim == -1 and largest:
            return mistral.fixed_index_topk(self, k)
        return previous(self, k, dim=dim, largest=largest, sorted=sorted)
    original = reference.MistralLarge4TextIndexer.forward

    def forward(self, *args, **kwargs):
        torch.Tensor.topk = topk
        try:
            return original(self, *args, **kwargs)
        finally:
            torch.Tensor.topk = previous
    reference.MistralLarge4TextIndexer.forward = forward
    try:
        yield
    finally:
        reference.MistralLarge4TextIndexer.forward = original
