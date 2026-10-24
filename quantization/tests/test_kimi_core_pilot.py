"""CPU contracts for the bounded shared codebook pilot (no model imports)."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.core_k6 import pilot
from kimi_k3.core_k6.container import command


def arguments(*extra):
    return ['--source-snapshot', '/base', '--replay-snapshot', '/core',
            '--tokens', '/tokens', '--root', '/results', *extra]


def result(arm='mcg'):
    return dict(schema=pilot.SCHEMA, tensor='layer.q', arm=arm, nmse=.01,
                bf16_output_proxy_nmse=.011, weight_nmse=.009,
                calibration_tokens=33000, heldout_tokens=4300)


def test_default_pilot_is_bounded_and_has_no_full_quant_or_head_switch():
    args = pilot.configuration(arguments())
    assert args.layers == (1, 3) and args.arms == pilot.ARMS
    assert args.calibration_tokens == 32768 and args.heldout_tokens == 4096
    assert not hasattr(args, 'export') and not hasattr(args, 'publish')
    with pytest.raises(SystemExit):
        pilot.configuration(arguments('--head'))


@pytest.mark.parametrize('extra', [('--layers', '3,1'), ('--layers', '1,1'),
    ('--layers', '1,92'), ('--arms', 'mcg,e4m3'), ('--calibration-tokens', '8'),
    ('--heldout-tokens', '0')])
def test_reject_unbounded_or_ambiguous_selection(extra):
    with pytest.raises(ValueError):
        pilot.configuration(arguments(*extra))


def test_storage_identity_survives_json_round_trip():
    identity = pilot.selection_identity(pilot.configuration(arguments()))
    assert json.loads(json.dumps(identity)) == identity


def test_subset_preserves_whole_original_prompts_and_order():
    records = [dict(id='a', input_ids=[1, 2]), dict(id='b', input_ids=[3, 4, 5])]
    assert pilot.subset(records, 3) == records
    assert pilot.subset(records, 2) == records[:1]
    with pytest.raises(ValueError):
        pilot.subset(records, 6)


@pytest.mark.parametrize('field,value', [('nmse', float('nan')), ('nmse', -1),
    ('arm', 'other'), ('heldout_tokens', 0), ('calibration_tokens', True)])
def test_result_schema_fails_closed(field, value):
    row = result()
    row[field] = value
    with pytest.raises(ValueError):
        pilot.validate_result(row)


def test_three_arm_table_requires_complete_finite_results():
    rows = [result(arm) for arm in pilot.ARMS]
    table = pilot.markdown(rows)
    assert 'e4m3 NMSE (ratio)' in table and '1.0000x' in table
    with pytest.raises(ValueError):
        pilot.markdown(rows[:-1])
    row = result()
    row['unexpected'] = 1
    with pytest.raises(ValueError):
        pilot.validate_result(row)


def test_format_inventory_and_expert_admission():
    args = pilot.configuration(arguments('--arms', ','.join(pilot.FORMAT_ARMS),
        '--expert-count', '32', '--expert-source-snapshot', '/original'))
    assert args.arms == pilot.FORMAT_ARMS and args.expert_count == 32
    with pytest.raises(ValueError):
        pilot.configuration(arguments('--expert-count', '32'))


def test_expert_sample_is_deterministic_stratified_and_includes_rarest():
    counts = [index // 3 for index in reversed(range(896))]
    chosen = pilot.stratified_experts(counts)
    assert chosen == pilot.stratified_experts(counts)
    assert len({expert for expert, _ in chosen}) == 32
    assert chosen[0][0] == min(range(896), key=lambda index: (counts[index], index))
    assert [sum(q == quartile for _, q in chosen) for quartile in range(1, 5)] == [8] * 4
    with pytest.raises(ValueError):
        pilot.stratified_experts([1, 2])


def test_exact_payload_accounting_includes_scales_and_mla_padding():
    shape = (576, 7168)
    assert pilot.payload_bytes('fp8-block128', shape) == 576 * 7168 + 5 * 56 * 4
    assert pilot.payload_bytes('fp8-mx32', shape) == 576 * 7168 + 576 * 224
    assert pilot.payload_bytes('mcg-k4', shape) == 640 * 7168 // 2 + 2 * (640 + 7168) + 4
    assert pilot.payload_bytes('e4m3-k2', (3072, 3584)) == 3072 * 3584 // 4 + 2 * (3072 + 3584) + 4


@pytest.mark.parametrize('arm', pilot.FORMAT_ARMS[-2:])
@pytest.mark.parametrize('shape', [(129, 256), (128, 128), (576, 128)])
def test_fp8_cpu_serialized_reconstruction_and_storage(arm, shape):
    torch = pytest.importorskip('torch')
    torch.manual_seed(787)
    weight = torch.randn(shape).to(torch.bfloat16)
    weight[0].zero_()
    packed = pilot.fp8_quantize(weight, arm)
    assert sum(value.numel() * value.element_size() for value in packed.values()) == pilot.payload_bytes(arm, shape)
    dense = pilot.fp8_reconstruct(packed, arm, 'cpu')
    assert dense.shape == weight.shape and bool(torch.isfinite(dense).all())
    assert not bool(dense[0].any())
    assert float((dense - weight.float()).square().mean()) < .005
    if arm == 'fp8-mx32':
        assert packed['scales'].dtype == torch.uint8
        assert not bool((packed['scales'] == 255).any())
    else:
        assert packed['scales'].dtype == torch.float32


def test_format_table_ratios_and_duplicates():
    rows = [dict(result(), arm=arm) for arm in pilot.FORMAT_ARMS]
    assert 'mcg-k6' in pilot.markdown(rows)
    with pytest.raises(ValueError):
        pilot.markdown(rows + rows[:1])


@pytest.mark.parametrize('present', [False, True])
def test_container_mounts_sparknest_readonly_when_present(monkeypatch, present):
    original = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda path: present if str(path) == '/mnt/sparknest' else original(path))
    argv = command(['-m', 'pilot'], name='pilot')
    mount = 'type=bind,src=/mnt/sparknest,dst=/mnt/sparknest,readonly'
    assert (mount in argv) == present
    if present:
        assert argv[argv.index(mount) - 1] == '--mount'


def test_container_cap_and_private_compiler_directory():
    argv = command(['-m', 'quantization.kimi_k3.core_k6.pilot'], name='pilot',
                   memory_gib=50, build_root=Path('/build/pilot'))
    assert argv[argv.index('--memory') + 1] == '50g'
    assert argv[argv.index('--memory-swap') + 1] == '50g'
    assert 'GPTQMODEL_EXLLAMAV3_BUILD_ROOT=/build/pilot' in argv
    with pytest.raises(ValueError):
        command(['-m', 'pilot'], name='pilot', memory_gib=0)
