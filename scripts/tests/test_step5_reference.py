"""Step-5 golden contracts; optional pinned-release CPU checks, never GPU."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "python/reference/families/step5/golden.py"
spec = importlib.util.spec_from_file_location("step5_golden", PATH)
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)


def test_step5_source_pin_fails_closed(tmp_path):
    snapshot = tmp_path / golden.REVISION
    folder = snapshot / "inference"
    folder.mkdir(parents=True)
    for name in golden.SOURCE_PINS:
        (folder / name).write_text("not the pinned source")
    with pytest.raises(ValueError, match="differs"):
        golden.verify_source(snapshot)
    with pytest.raises(ValueError, match="revision"):
        golden.verify_source(tmp_path)


def test_reference_uses_official_kernels_and_target_layers_only():
    tree = ast.parse(PATH.read_text())
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert {"Step5DecoderLayer", "indexer_logits", "prefill_sparse_meta", "sparse_attention_prefill",
            "update_summaries_prefill", "_feed_forward", "_project", "_gate_and_project"} <= names
    assert "Step5ForConditionalGeneration" not in names
    assert "num_nextn_predict_layers" not in names


def test_sparse_launch_override_changes_stages_only():
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))
                return "compiled"
            return launch

    wrapped = golden.SparsePrefillLaunch(Kernel())
    assert wrapped[(128, 4)]("q", "k", BLOCK_H=16, BLOCK_D=256, BLOCK_R=16) == "compiled"
    assert calls == [((128, 4), ("q", "k"),
                      dict(BLOCK_H=16, BLOCK_D=256, BLOCK_R=16, num_stages=1))]
    assert golden.SPARSE_LAUNCH["official_launch_configuration"] is False
    with pytest.raises(ValueError, match="explicitly"):
        wrapped[(128, 4)](num_stages=3)
    assert len(calls) == 1


@pytest.fixture
def release():
    snapshot = os.environ.get("STEP5_REFERENCE_SNAPSHOT")
    if not snapshot:
        pytest.skip("set STEP5_REFERENCE_SNAPSHOT for pinned official CPU arithmetic checks")
    pytest.importorskip("torch")
    return golden.load_release(Path(snapshot))[0]


def test_official_zero_centered_norm_and_clamp_byte_exact(release):
    import torch
    x = torch.tensor([[0.125, -9, 8, 0.5], [1, -1, 0.0625, 3]], dtype=torch.bfloat16)
    norm = release.Step5RMSNorm(4)
    norm.weight.data.copy_(torch.tensor([0.1, -0.25, 0.002, -0.125]))
    expected = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
                * (norm.weight + 1)).bfloat16()
    assert torch.equal(norm(x).view(torch.int16), expected.view(torch.int16))
    up = torch.tensor([[8, -8, 0.5, 0.25], [-9, 6, -7.5, 2]], dtype=torch.bfloat16)
    expected = (torch.nn.functional.silu(x.float()).clamp(max=7) * up.float().clamp(-7, 7)).bfloat16()
    assert torch.equal(release.clamped_swiglu(x, up, 7).view(torch.int16), expected.view(torch.int16))
    wrong = (torch.nn.functional.silu(x.clamp(-7, 7)).bfloat16() * up.clamp(-7, 7)).bfloat16()
    assert not torch.equal(wrong, expected)


def test_official_cuda_qkv_dispatch_uses_deployed_norm_rotary(release, monkeypatch):
    import torch
    calls = []
    packed = SimpleNamespace(is_cuda=True, shape=(2, 768))
    q, k, v = (torch.zeros(2, heads * 192, dtype=torch.bfloat16) for heads in (2, 1, 1))

    def fused(*args, **kwargs):
        calls.append((args, kwargs))
        return q, k, v

    def fallback(*args):
        pytest.fail("deployed CUDA projection must not use CPU norm/rotary fallback")

    monkeypatch.setattr(release, "fused_qknorm_rope", fused)
    weight = torch.zeros(192)
    for span in (192, 64):
        attention = SimpleNamespace(
            _project_packed_qkv=lambda hidden: packed, _split_packed_qkv=fallback, _apply_rope=fallback,
            head_dim=192, num_heads=2, num_kv_groups=1, rotary_span=span,
            q_norm=SimpleNamespace(weight=weight, variance_epsilon=1e-5),
            k_norm=SimpleNamespace(weight=weight), rope_cos=torch.ones(16, span // 2),
            rope_sin=torch.zeros(16, span // 2))
        positions = torch.tensor([0, 1], dtype=torch.int32)
        result = release.Step5Attention._project(attention, torch.zeros(2, 4), positions)
        assert [tuple(t.shape) for t in result] == [(2, 2, 192), (2, 1, 192), (2, 1, 192)]
        assert calls[-1][0][0] is packed
        assert calls[-1][1] == dict(head_dim=192, num_q_heads=2, num_kv_heads=1,
                                    rotary_pairs=span // 2, eps=1e-5, norm_weight_bias=1.0)
    assert len(calls) == 2


def test_official_router_bias_selection_only_and_fp32_slot_sum(release):
    import torch
    from shape_invariant import linear
    x = torch.tensor([[1., -2., 0.25, 0.5]], dtype=torch.bfloat16)
    gate = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=torch.bfloat16)
    bias = torch.tensor([0., 2., 0.1])
    router = SimpleNamespace(gate=SimpleNamespace(weight=gate), router_bias=bias,
                             top_k=2, renormalize=True, scaling_factor=3.)
    weights, ids = release.Step5MoE.route(router, x)
    probabilities = torch.sigmoid(torch.nn.functional.linear(x.float(), gate.float()))
    selected = (probabilities + bias).topk(2)
    expected = selected.values - bias[selected.indices]
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * 3
    assert torch.equal(ids, selected.indices)
    assert torch.equal(weights.view(torch.int32), expected.view(torch.int32))
    assert ids[0, 0] == 1  # The biased winner is not the highest raw probability.
    # Fixed expert GEMM rows preserve prefix arithmetic when routed counts grow.
    torch.manual_seed(7)
    h = torch.randn(141, 4).bfloat16()
    assert torch.equal(linear(h[:61], gate), linear(h, gate)[:61])
    # Expert iteration order differs from top-k slot order in this cancellation.
    constants = [-1e8, 1., 1e8]
    projection = SimpleNamespace(apply_expert=lambda expert, tokens: torch.ones_like(tokens))
    down = SimpleNamespace(apply_expert=lambda expert, tokens: torch.full_like(tokens, constants[expert]))
    slots = SimpleNamespace(route=lambda hidden: (torch.ones(1, 3), torch.tensor([[2, 0, 1]])),
                            top_k=3, n_local_experts=3, experts_start_idx=0, limit=7,
                            gate_proj=projection, up_proj=projection, down_proj=down)
    result = release.Step5MoE.forward(slots, x)
    assert torch.equal(result.view(torch.int16), torch.ones_like(x).view(torch.int16))


def test_bounded_sliding_cpu_prefix_exact_and_matches_dense():
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    q = torch.randn(141, 4, 8).bfloat16()
    k, v = torch.randn(141, 2, 8).bfloat16(), torch.randn(141, 2, 8).bfloat16()
    output = golden.sliding_attention(q, k, v, 16)
    prefix = golden.sliding_attention(q[:133], k[:133], v[:133], 16)
    assert torch.equal(prefix, output[:133])
    p = torch.arange(141)
    keep = (p[:, None] >= p[None, :]) & (p[:, None] - p[None, :] < 16)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
        attn_mask=keep[None, None], enable_gqa=True)[0].transpose(0, 1)
    torch.testing.assert_close(output, expected, atol=0.002, rtol=0.01)

def test_official_tokenizer_preserves_serialized_bpe_and_chat_template(release):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    snapshot = Path(os.environ["STEP5_REFERENCE_SNAPSHOT"])
    fast = PreTrainedTokenizerFast.from_pretrained(snapshot, local_files_only=True)
    native = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    text = "# A plan\n\nKeep spaces, newlines and punctuation intact.\n" * 30
    ids = fast(text, add_special_tokens=False)["input_ids"]
    assert ids == native.encode(text, add_special_tokens=False).ids
    assert fast.decode(ids, skip_special_tokens=False) == text
    assert fast.chat_template == (snapshot / "chat_template.jinja").read_text()


def test_native_template_keeps_unfinished_reasoning_open(release):
    from transformers import PreTrainedTokenizerFast
    spec = importlib.util.spec_from_file_location("step5_panel", ROOT / "scripts/bench/fidelity-set.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    fast = PreTrainedTokenizerFast.from_pretrained(os.environ["STEP5_REFERENCE_SNAPSHOT"], local_files_only=True)
    text = "<|im_start|>user\n" + "context " * 600 + "<|im_end|>\n<|im_start|>assistant\n<think>\n" + "reason " * 600
    class Characters:
        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)
    window = dict(id="b00", block="B", tokens=list(map(ord, text)), roles=["gen"] * len(text),
                  score_from=len(text) - 512, bucket="0-2K")
    result = builder.render_retokenized_window(window, Characters(), fast)
    rendered = fast.decode(result["tokens"], skip_special_tokens=False)
    assert rendered.count("<think>") == 1
    assert "</think>" not in rendered
    assert rendered.endswith("reason " * 600)
    assert result["roles"][-1] == "gen"
