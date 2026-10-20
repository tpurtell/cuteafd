"""Pinned Step-5 native predictor wiring; CPU-only, optional Torch math."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "python/reference/families/step5/mtp.py"
spec = importlib.util.spec_from_file_location("step5_mtp_reference", PATH)
mtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtp)


def test_mtp_semantics_provenance_is_complete_and_fail_closed(tmp_path):
    assert mtp.VLLM_COMMIT == "dde9aca41cd3ccceb1ffa902933b6678a09c83ce"
    assert set(mtp.VLLM_SOURCES) == {"vllm/models/step5/nvidia/mtp.py",
        "vllm/models/step5/nvidia/model.py", "vllm/models/step5/nvidia/layernorm.py",
        "vllm/v1/spec_decode/step3p5.py", "vllm/model_executor/layers/logits_processor.py",
        "vllm/models/step5/nvidia/ops/attention_gate.py"}
    for name in mtp.VLLM_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("wrong source")
    with pytest.raises(ValueError, match="differs"):
        mtp.provenance(tmp_path)


def test_mtp_teacher_shift_and_geometry():
    assert mtp.teacher_inputs([1, 2, 3], [10, 20, 30], [4, 5, 6]) == (
        [2, 3], [10, 20], [5, 6])
    with pytest.raises(ValueError, match="aligned"):
        mtp.teacher_inputs([1], [10], [4])
    for geometry in [dict(heads=3, groups=2), dict(dim=3), dict(window=0)]:
        with pytest.raises(ValueError, match="geometry"):
            mtp.Stage({}, **geometry)


def _fixture():
    torch = pytest.importorskip("torch")
    torch.manual_seed(274)
    h, heads, groups, dim, intermediate = 8, 4, 2, 4, 12
    shapes = {"enorm.weight": (h,), "hnorm.weight": (h,),
        "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
        "transformer.shared_head.norm.weight": (h,),
        "eh_proj.weight": (h, 2*h), "self_attn.q_proj.weight": (heads*dim, h),
        "self_attn.k_proj.weight": (groups*dim, h),
        "self_attn.v_proj.weight": (groups*dim, h),
        "self_attn.q_norm.weight": (dim,), "self_attn.k_norm.weight": (dim,),
        "self_attn.g_proj.weight": (heads, h), "self_attn.o_proj.weight": (h, heads*dim),
        "mlp.gate_proj.weight": (intermediate, h), "mlp.up_proj.weight": (intermediate, h),
        "mlp.down_proj.weight": (h, intermediate),
        "transformer.shared_head.output.weight": (17, h)}
    weights = {name: torch.randn(shape)*.2 for name, shape in shapes.items()}
    weights = {name: value if len(value.shape)==1 else value.bfloat16().float()
               for name, value in weights.items()}
    embedding, hidden = torch.randn(7,h), torch.randn(7,h)
    positions = torch.arange(3, 10)
    angles = torch.randn(32,dim//2)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    kwargs = dict(heads=heads, groups=groups, dim=dim, window=3)
    return torch, weights, embedding, hidden, positions, cos, sin, kwargs


def test_mtp_checkpoint_binding_all_three_stages_and_native_precision():
    torch, weights, *_ = _fixture()
    native = {name: value if value.ndim == 1 else value.bfloat16()
              for name, value in weights.items()}
    config = {"model_type": "step5", "num_hidden_layers": 92,
        "num_nextn_predict_layers": 3, "norm_dtype": "float32",
        "fp32_residual_connection": True, "layer_types": ["sliding_attention"]*95,
        "hidden_size": 8, "num_attention_heads": 4, "num_attention_groups": 2,
        "head_dim": 4, "intermediate_size": 12, "vocab_size": 17,
        "sliding_window": 3, "rms_norm_eps": 1e-5, "swiglu_limits_shared": [7.0]*95}
    class Reader:
        def __init__(self, layer):
            self.prefix, self.seen = f"model.layers.{layer}.", []
        def get(self, name):
            assert name.startswith(self.prefix)
            self.seen.append(name)
            return native[name.removeprefix(self.prefix)]
    for stage in range(3):
        reader = Reader(92+stage)
        result = mtp.Stage.from_checkpoint(reader, {"text_config": config}, stage)
        assert len(reader.seen) == 17
        assert set(result.weights) == set(native)
        assert all(value.dtype == torch.float32 for value in result.weights.values())
        assert result.heads == 4 and result.groups == 2 and result.window == 3
    with pytest.raises(ValueError, match="configuration"):
        mtp.Stage.from_checkpoint(Reader(95), config, 3)
    native["enorm.weight"] = native["enorm.weight"].bfloat16()
    with pytest.raises(ValueError, match="shape/dtype"):
        mtp.Stage.from_checkpoint(Reader(92), config, 0)


def test_mtp_concat_narrows_before_zero_centered_norm_embedding_first():
    torch, weights, embedding, hidden, *_ = _fixture()
    actual = mtp.norm_concat(embedding, hidden, weights, 1e-5)
    def independent(x, weight):
        x = x.bfloat16().float()
        return (x * torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)
                * (1+weight)).bfloat16().float()
    expected = torch.cat([independent(embedding,weights["enorm.weight"]),
                          independent(hidden,weights["hnorm.weight"])],-1)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, expected.flip(-1))


def test_mtp_stage_chunking_rollback_window_and_own_head():
    torch, weights, embedding, hidden, positions, cos, sin, kwargs = _fixture()
    full = mtp.Stage(weights, **kwargs)
    expected_hidden, expected_logits = full.forward(embedding, hidden, positions, cos, sin)
    chunks = mtp.Stage(weights, **kwargs)
    parts = [chunks.forward(embedding[i:i+1], hidden[i:i+1], positions[i:i+1], cos, sin)
             for i in range(len(positions))]
    # CPU F32 GEMMs may select a different reduction for one vs multiple rows.
    torch.testing.assert_close(torch.cat([x[0] for x in parts]), expected_hidden, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([x[1] for x in parts]), expected_logits, atol=2e-6, rtol=2e-6)
    assert chunks.cache[0].tolist() == [7,8,9]
    for x,y in zip(chunks.cache, full.cache):
        assert torch.equal(x,y)
    with pytest.raises(ValueError,match="rollback"):
        chunks.forward(embedding[-1:], hidden[-1:], positions[-1:], cos, sin)
    chunks.cache = None
    chunks.forward(embedding[:2],hidden[:2],positions[:2],cos,sin)
    snapshot = tuple(value.clone() for value in chunks.cache)
    result = chunks.forward(embedding[2:3],hidden[2:3],positions[2:3],cos,sin)
    chunks.cache = snapshot
    replay = chunks.forward(embedding[2:3],hidden[2:3],positions[2:3],cos,sin)
    assert all(torch.equal(x,y) for x,y in zip(result,replay))
    own = dict(weights)
    own["transformer.shared_head.output.weight"] = -weights["transformer.shared_head.output.weight"]
    stage = mtp.Stage(own,**kwargs)
    other_hidden, other_logits = stage.forward(embedding,hidden,positions,cos,sin)
    assert torch.equal(other_hidden,expected_hidden)
    assert torch.equal(other_logits,-expected_logits)
    assert stage.cache is not full.cache
    bf16_head = mtp.Stage(weights, head_dtype="bfloat16", **kwargs)
    rounded_hidden, rounded_logits = bf16_head.forward(embedding, hidden, positions, cos, sin)
    assert torch.equal(rounded_hidden, expected_hidden)
    assert torch.equal(rounded_logits, expected_logits.bfloat16().float())


def test_mtp_stage_positions_fail_before_cache_publication():
    torch, weights, embedding, hidden, positions, cos, sin, kwargs = _fixture()
    stage = mtp.Stage(weights,**kwargs)
    for bad in [positions.flip(0), positions-10, positions[None]]:
        with pytest.raises(ValueError,match="positions"):
            stage.forward(embedding,hidden,bad,cos,sin)
        assert stage.cache is None
