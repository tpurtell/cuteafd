"""K3.1 DSA selection, gated MLA and isolated root loader behavior."""
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from families.kimi_k3 import k31, model


def config(topk=2):
    return types.SimpleNamespace(hidden_size=4, num_attention_heads=1, num_key_value_heads=1,
        q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=2, qk_rope_head_dim=2,
        v_head_dim=2, mla_use_nope=True, mla_use_output_gate=True, rms_norm_eps=1e-5,
        dsa_config=dict(dsa_layers=[2], index_topk=topk, index_head_dim=4,
            index_num_heads=2, index_rope_dim=0, index_k_norm_type="layernorm",
            index_topk_freq=1, index_skip_topk_offset=1),
        linear_attn_config=dict(full_attn_layers=[2]), _attn_implementation="eager")


def memory_config():
    return types.SimpleNamespace(hidden_size=7168, num_attention_heads=96,
        num_hidden_layers=93, attn_res_block_size=12, q_lora_rank=1536,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        dsa_config=dict(index_num_heads=32, index_head_dim=128, index_topk=2048),
        linear_attn_config=dict(num_heads=96, head_dim=128),
        num_experts=896, num_experts_per_token=16, routed_expert_hidden_size=3584,
        moe_intermediate_size=3072, intermediate_size=33792)


def test_k31_live_memory_bounds_tiled_scores_history_and_routed_combine():
    c = memory_config()
    small = k31.execution_memory(c, 128, 0)
    large = k31.execution_memory(c, 15915, 92)
    assert large['margin'] == 4 * 2**30
    assert large['residual'] == 4 * 15915 * 7168 * (5 * 9 + 6)
    assert large['peak'] == large['residual'] + max(large[k] for k in ('attention', 'kda', 'moe', 'dense'))
    assert large['attention'] > 4 * 128 * 16000 * (3 * 96 + 3 * 32)
    assert large['moe'] > 4 * 15915 * 16 * 3584 * 5
    assert small['peak'] < large['peak'] < 64 * 2**30
    assert k31.execution_memory(c, 15915, 0)['residual'] < large['residual']
    for length in (0, -1):
        with pytest.raises(ValueError, match='memory geometry'):
            k31.execution_memory(c, length, 0)


def test_layer_admission_reclaims_gb10_cache_without_cuda_free_gate(tmp_path, monkeypatch):
    from families.kimi_k3 import golden
    import allocatable_memory
    import fidelity_windows
    info = tmp_path / 'meminfo'
    info.write_text('MemFree: 1048576 kB\nMemAvailable: 117440512 kB\n')
    cuda = types.SimpleNamespace(synchronize=lambda: None, current_device=lambda: 0,
        get_device_properties=lambda _: types.SimpleNamespace(is_integrated=True, total_memory=119 * 2**30),
        mem_get_info=lambda *a: (_ for _ in ()).throw(AssertionError('UMA must not gate on CUDA free')))
    monkeypatch.setattr(fidelity_windows, 'cuda_allocatable_memory_info',
        lambda actual: allocatable_memory.cuda_allocatable_memory_info(actual, meminfo=info))
    golden.admit_layer_memory(cuda, memory_config(), 92, 17 * 2**30, 2**26, 15915)
    # The unchanged K3 64-GiB reserve uses the same reclaim-aware admission.
    golden.admit_layer_memory(cuda, types.SimpleNamespace(), 0, 17 * 2**30, 2**26, 15915)
    info.write_text('MemFree: 1048576 kB\nMemAvailable: 2097152 kB\n')
    with pytest.raises(RuntimeError, match='admission failed'):
        golden.admit_layer_memory(cuda, memory_config(), 92, 17 * 2**30, 2**26, 15915)


def tiny_reference(torch):
    class RMSNorm(torch.nn.Module):
        def __init__(self, size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(size))
            self.variance_epsilon = eps

        def forward(self, x):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight

    class MLA(torch.nn.Module):
        def __init__(self, c, layer_idx):
            super().__init__()
            self.num_heads, self.q_head_dim = 1, 4
            self.qk_nope_head_dim, self.qk_rope_head_dim = 2, 2
            self.kv_lora_rank, self.v_head_dim, self.scaling = 4, 2, .5
            self.q_a_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_a_layernorm = RMSNorm(4)
            self.q_b_proj = torch.nn.Linear(4, 4, bias=False)
            self.kv_a_proj_with_mqa = torch.nn.Linear(4, 6, bias=False)
            self.kv_a_layernorm = RMSNorm(4)
            self.kv_b_proj = torch.nn.Linear(4, 4, bias=False)
            self.g_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(2, 4, bias=False)

    class Decoder(torch.nn.Module):
        def __init__(self, c, index):
            super().__init__()
            self.self_attn = MLA(c, index)

    def eager(module, q, k, v, mask, scaling=1., dropout=0.):
        probs = ((q @ k.transpose(-1, -2)) * scaling + mask).softmax(-1)
        return (probs @ v).transpose(1, 2), None

    return types.SimpleNamespace(KimiMLAAttention=MLA, KimiDecoderLayer=Decoder,
        KimiRMSNorm=RMSNorm, eager_attention_forward=eager, __name__="tiny_kimi")


def hand_forward(torch, attention, hidden):
    """Scalar causal selection and sum, independent of the tiled oracle helpers."""
    import math
    a = attention
    x = hidden[0].double()
    norm = lambda z, n: z / (z.square().mean(-1, keepdim=True) + n.variance_epsilon).sqrt() * n.weight.double()
    qa = norm(x @ a.q_a_proj.weight.double().T, a.q_a_layernorm)
    q = qa @ a.q_b_proj.weight.double().T
    compressed = x @ a.kv_a_proj_with_mqa.weight.double().T
    kv = norm(compressed[:, :4], a.kv_a_layernorm) @ a.kv_b_proj.weight.double().T
    keys = torch.cat((kv[:, :2], compressed[:, 4:]), -1)
    values = kv[:, 2:]
    iq = (qa @ a.indexer.wq_b.weight.double().T).view(-1, a.indexer.heads, a.indexer.dim)
    raw = x @ a.indexer.wk.weight.double().T
    ik = (raw - raw.mean(-1, keepdim=True)) / (raw.var(-1, keepdim=True, unbiased=False) + a.indexer.k_norm.eps).sqrt()
    ik = ik * a.indexer.k_norm.weight.double() + a.indexer.k_norm.bias.double()
    iw = x @ a.indexer.weights_proj.weight.double().T
    result, selections = [], []
    for row in range(len(x)):
        scores = [sum(float(iw[row, head].detach()) * max(float((iq[row, head] @ ik[key]).detach()), 0.)
                      / math.sqrt(a.indexer.heads * a.indexer.dim)
                      for head in range(a.indexer.heads)) for key in range(row + 1)]
        chosen = sorted(sorted(range(row + 1), key=lambda key: (-scores[key], key))[:a.indexer.topk])
        selections.append(chosen)
        logits = torch.stack([q[row] @ keys[key] * .5 for key in chosen])
        output = logits.softmax(0) @ values[chosen]
        output *= (x[row] @ a.g_proj.weight.double().T).sigmoid()
        result.append(output @ a.o_proj.weight.double().T)
    return torch.stack(result)[None], selections


@pytest.mark.parametrize("topk,length", [(8, 5), (2, 5), (2, 131)])
def test_gated_dsa_matches_hand_causal_selection(topk, length):
    torch = pytest.importorskip("torch")
    torch.manual_seed(73)
    c = config(topk)
    reference = tiny_reference(torch)
    extended = k31.extend_reference(reference, c)
    a = extended.KimiDSAAttention(c, 1).eval().float()
    hidden = torch.randn(1, length, 4)
    expected, selections = hand_forward(torch, a, hidden)
    actual = a(hidden)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    assert selections[0] == [0]  # offset=1 is an index-sharing offset, NOT key removal.
    torch.testing.assert_close(a(hidden[:, :3]), actual[:, :3], rtol=0, atol=0)
    # Future rows can never enter an earlier row's selected keys or MLA sum.
    changed = hidden.clone()
    changed[:, 3:] += 100
    torch.testing.assert_close(a(changed)[:, :3], actual[:, :3], rtol=0, atol=0)


def test_topk_ties_choose_lower_id_and_invalid_slots_stay_masked():
    torch = pytest.importorskip("torch")
    from shape_invariant import fixed_index_topk
    scores = torch.tensor([[0., float("-inf"), float("-inf")], [2., 2., 2.]])
    chosen = fixed_index_topk(scores, 2)
    assert chosen.indices.tolist() == [[0, 1], [0, 1]]
    assert torch.isneginf(chosen.values[0, 1])
    c = config(2)
    a = k31.extend_reference(tiny_reference(torch), c).KimiDSAAttention(c, 1).eval().float()
    with torch.no_grad():
        a.indexer.weights_proj.weight.zero_()
    hidden = torch.arange(20).reshape(1, 5, 4).float() / 10
    expected, selections = hand_forward(torch, a, hidden)
    assert selections == [[0], [0, 1], [0, 1], [0, 1], [0, 1]]
    torch.testing.assert_close(a(hidden).double(), expected, rtol=2e-5, atol=2e-6)


def test_dsa_config_eps_without_mutating_k3_or_non_dsa_layers():
    torch = pytest.importorskip("torch")
    reference, c = tiny_reference(torch), config()
    original = reference.KimiMLAAttention
    torch.manual_seed(7)
    before = reference.KimiDecoderLayer(c, 0)
    extended = k31.extend_reference(reference, c)
    torch.manual_seed(7)
    after = extended.KimiDecoderLayer(c, 0)
    assert reference.KimiMLAAttention is original
    assert type(after.self_attn) is original
    for name, value in before.state_dict().items():
        assert torch.equal(value, after.state_dict()[name])
    dsa = extended.KimiDecoderLayer(c, 1).self_attn
    for name in ("q_a_layernorm", "kv_a_layernorm"):
        assert getattr(before.self_attn, name).variance_epsilon == 1e-6
        assert getattr(dsa, name).variance_epsilon == c.rms_norm_eps


@pytest.mark.parametrize("key,value", [("index_rope_dim", 2), ("index_topk_freq", 2),
    ("index_skip_topk_offset", 0), ("index_skip_topk_offset", 2), ("index_k_norm_type", "rmsnorm")])
def test_unsupported_dsa_rules_fail_closed(key, value):
    c = config()
    c.dsa_config[key] = value
    with pytest.raises(ValueError, match="unsupported K3.1 DSA"):
        k31.validate_config(c)


def test_new_root_loader_dispatches_without_changing_k3(monkeypatch, tmp_path):
    snapshot = tmp_path / k31.REVISION
    result = object()
    monkeypatch.setattr(k31, "load_official", lambda path: result)
    assert model.load_official(snapshot) is result
    monkeypatch.setattr(k31, "source_identity", lambda path: {"revision": k31.REVISION})
    assert model.source_identity(snapshot)["revision"] == k31.REVISION
    with pytest.raises(ValueError, match="pinned official checkpoint"):
        model.source_identity(tmp_path / ("a" * 40))


def test_ml4_shared_score_reduction_preserves_math():
    torch = pytest.importorskip("torch")
    from families.mistral_large4.ops import indexer_score_reduction
    torch.manual_seed(8)
    q, k, w = torch.randn(7, 3, 4), torch.randn(9, 4), torch.randn(7, 3)
    expected = torch.einsum("qh,qhk->qk", w.float() * 3 ** -.5,
        torch.relu(torch.einsum("qhd,kd->qhk", q.float(), k.float()) * 4 ** -.5))
    assert torch.equal(indexer_score_reduction(q, k, w), expected)
