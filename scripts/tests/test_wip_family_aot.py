"""Direct WIP builds derive the same family scopes as single gate cards."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import wip_aot_scopes as aot

SPEC = importlib.util.spec_from_file_location('wip_family_cards', ROOT / 'scripts/bench/wip-cards.py')
cards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cards)

CASES = [
    ('deepseek_v41', 'DeepSeek-V4.1-Flash', 4, None, ''),
    ('deepseek_v4', 'DeepSeek-V4-Flash', 2, 'DSV4', 'dsv4f:rtx_backbone'),
    ('deepseek_v4', 'DeepSeek-V4-Pro-EXL3-K2', 4, 'DSV4', 'dsv4p:exl3-k23'),
    ('glm5', 'GLM-5.3-EXL3-K4', 4, 'GLM', 'glm:exl3-k45'),
    ('glm5_flash', 'GLM-5.3-Flash-tr3', 2, 'GLMF', 'glmf:exl3-k34'),
    ('glm5_flash', 'GLM-5.3-Flash-NVFP4', 4, 'GLMF', 'glmfdense:nvfp4'),
    ('mimo_v2', 'MiMo-V2.6-Flash', 4, 'MIMO', 'mimof:fp8'),
    ('mimo_v2', 'MiMo-V2.6-Pro', 6, 'MIMO', 'mimop:fp8'),
    ('qwen4', 'Qwen3.8-Flash-Next-EXL3-K4.25', 0, 'QWEN4', 'qwen4:exl3-k45'),
    ('step5', 'Step-5-Preview-NVFP4', 4, 'STEP5', 'step5:nvfp4'),
    ('step5', 'Step-5-Preview', 6, 'STEP5', 'step5:fp8'),
    ('kimi_k3', 'Kimi-K3-EXL3-K2', 6, 'KIMIK3', 'kimik3:exl3-k23'),
]


def config_file(tmp_path, family, model, count):
    path = tmp_path / 'model.config'
    hosts = '\n'.join(f'SPARK_{i}_HOST=spark{i}\nSPARK_{i}_LANE_A=192.0.2.{i + 1}\n'
                      f'SPARK_{i}_LANE_B=198.51.100.{i + 1}' for i in range(count))
    path.write_text(f'FAMILY={family}\nMODEL_ID=test/{model}\nSPARK_COUNT={count}\n'
                    f'RTX_GPUS=2\nCOORDINATOR_GPU=1\n{hosts}\n')
    return path


@pytest.mark.parametrize('family,model,count,switch,scope', CASES)
def test_config_scopes_match_equivalent_single_card(tmp_path, family, model, count, switch, scope):
    path = config_file(tmp_path, family, model, count)
    detected, derived = aot.config_scopes(aot.read_config(path))
    gate_card = dict(family=family, name=model.lower(), config=str(path))
    expected = cards.build_scopes([gate_card])
    assert detected == family
    assert {key: derived[key] for key in expected if key in derived} == {
        key: value for key, value in expected.items() if key in derived}
    if switch:
        assert derived[f'CUTEAFD_WIP_{switch}_AOT'] == 'ON'
    if scope:
        assert scope in derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    if family in ('glm5', 'glm5_flash'):
        assert derived['CUTEAFD_WIP_CONTEXT_SPLIT_AOT'] == 'ON'
    if count == 0:
        assert derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == ''
    if family == 'step5':
        assert set(derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')) == {scope, 'step5:bf16'}
        assert set(derived['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')) == {scope, 'step5:bf16'}
        assert set(derived['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'].split(';')) == {f'{scope}={count}', f'step5:bf16={count}'}


def dry_run(tmp_path, path, overrides=None):
    stub = tmp_path / 'bin'
    stub.mkdir(exist_ok=True)
    marker = tmp_path / 'hardware-called'
    for tool in ('docker', 'ssh', 'rsync', 'nvidia-smi'):
        script = stub / tool
        script.write_text(f'#!/bin/sh\necho {tool} >> "{marker}"\nexit 1\n')
        script.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('CUTEAFD_WIP_') and key not in ('SPARK_COUNT', 'SPARK_HOSTS', 'WIP_ROOT', 'WIP_INSTANCE')}
    env.update(PATH=str(stub) + ':' + os.environ['PATH'], **(overrides or {}))
    result = subprocess.run([str(ROOT / 'wip.sh'), '--config', str(path), '--role', 'coordinator', '--dry-run'],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()
    return result.stdout


@pytest.mark.parametrize('family,model,count,switch,scope', CASES)
def test_direct_wip_derives_defaults_and_env_override_wins(tmp_path, family, model, count, switch, scope):
    path = config_file(tmp_path, family, model, count)
    output = dry_run(tmp_path, path)
    assert f'WIP AOT scopes ({family})' in output
    if scope:
        assert scope in output
    if switch:
        key = f'CUTEAFD_WIP_{switch}_AOT'
        assert key + '=ON' in output
        output = dry_run(tmp_path, path, {key: 'OFF', 'CUTEAFD_WIP_EXPERT_FAMILIES': '',
                                          'CUTEAFD_WIP_CONTEXT_SPLIT_AOT': 'OFF',
                                          'CUTEAFD_WIP_AUDIO_AOT': 'OFF'})
        assert key + '=OFF' in output and key + '=ON' not in output
        assert 'CUTEAFD_WIP_EXPERT_FAMILIES= ' in output
        assert 'Audio AOT: OFF' in output
        if family in ('glm5', 'glm5_flash'):
            assert 'CUTEAFD_WIP_CONTEXT_SPLIT_AOT=OFF' in output


@pytest.mark.parametrize('family,model,count,switch,scope', CASES)
def test_seal_refuses_empty_programs_before_touching_slot(tmp_path, family, model, count, switch, scope):
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'cuteafd').write_text('#!/bin/sh\n')
    (output / 'cuteafd').chmod(0o755)
    (output / 'libcuteafd_native.so').write_text('stub native output\n')
    (output / 'PROGRAMS.json').write_text('{"schema":1,"programs":[]}\n')
    if switch is None:
        aot.check_programs(family, output / 'PROGRAMS.json')
        return
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/finalize-wip-slot.sh'), str(ROOT),
                             'coordinator', 'test', str(output), 'image', 'image-id'],
                            env={**os.environ, 'CUTEAFD_WIP_PROGRAM_FAMILY': family},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'cannot seal coordinator slot for ' + family in result.stderr
    assert f'CUTEAFD_WIP_{switch}_AOT=ON' in result.stderr
    assert 'ARTIFACT_SHA256SUMS' not in result.stderr  # refused before checksums/copy/seal
    (output / 'PROGRAMS.json').write_text('{"schema":1,"programs":[{"name":"stub"}]}\n')
    aot.check_programs(family, output / 'PROGRAMS.json')


@pytest.mark.parametrize('family', ['deepseek_v41', 'deepseek_v4', 'glm5', 'glm5_flash', 'mimo_v2', 'qwen4'])
def test_resolves_model_metadata_with_existing_launcher_table(tmp_path, monkeypatch, family):
    configs = [case['config'] for case in json.loads((ROOT / 'rust/crates/cuteafd-loader/tests/fixtures/launch-families.json').read_text())
               if case.get('expected', {}).get('family') == family]
    if not configs:
        model_type = next(kind for kind, value in aot.checkpoint_family.FAMILIES.items() if value == family)
        config = dict(model_type=model_type, num_hidden_layers=2, first_k_dense_replace=0,
                      layer_types=['linear_attention', 'full_attention'] if family == 'qwen4'
                      else ['linear_attention', 'deepseek_sparse_attention'])
    else:
        config = configs[0]
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    snapshot = tmp_path / 'hub/models--test--unlabelled/snapshots/revision'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text(json.dumps(config))
    values = dict(MODEL_ID='test/unlabelled', MODEL_REVISION='revision')
    assert aot.config_family(values) == family
    values['FAMILY'] = 'qwen4' if family != 'qwen4' else 'glm5'
    with pytest.raises(ValueError, match='does not match checkpoint'):
        aot.config_family(values)


def test_missing_metadata_fails_closed_unless_family_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    with pytest.raises(ValueError, match='set FAMILY'):
        aot.config_family(dict(MODEL_ID='test/unavailable'))
    assert aot.config_family(dict(MODEL_ID='test/unavailable', FAMILY='glm5')) == 'glm5'
    with pytest.raises(ValueError, match='no WIP AOT scope mapping for unknown'):
        aot.config_family(dict(FAMILY='unknown'))


def test_seal_guard_is_forwarded_after_native_build():
    wip = (ROOT / 'wip.sh').read_text()
    coordinator = wip.split('build_coordinator() {', 1)[1].split('build_expert() (', 1)[0]
    assert coordinator.index('"${coordinator_build[@]}" native') < coordinator.index('CUTEAFD_WIP_PROGRAM_FAMILY=')
    assert coordinator.index('CUTEAFD_WIP_PROGRAM_FAMILY=') < coordinator.index('finalize-wip-slot.sh')


@pytest.mark.parametrize('contents', ['{}', '{"programs":null}', '{"programs":{}}', 'broken'])
def test_missing_or_malformed_programs_are_refused(tmp_path, contents):
    path = tmp_path / 'PROGRAMS.json'
    path.write_text(contents)
    with pytest.raises(ValueError, match='cannot seal'):
        aot.check_programs('qwen4', path)
    with pytest.raises(ValueError, match='cannot seal'):
        aot.check_programs('qwen4', tmp_path / 'missing.json')


@pytest.mark.parametrize('count,tp,role', [('6', '2', 'tp2'), ('6', '6', 'tp6'), ('3', '3', 'tp3'), ('6', '', 'tp6')])
def test_v41_direct_config_uses_explicit_spark_tp_role(count, tp, role):
    values = {'FAMILY': 'deepseek_v41', 'SPARK_COUNT': count, 'SPARK_TP': tp}
    _, scopes = aot.config_scopes(values)
    assert scopes['CUTEAFD_WIP_SPARK_TP_ROLES'] == role


@pytest.mark.parametrize('hidden,intermediate,layers,heads,value_heads,switch', [
    (4096, 12288, 32, 16, 32, 'QWEN35'),
    (5120, 17408, 64, 24, 48, 'QWEN38'),
])
def test_dense_cards_and_direct_configs_use_checkpoint_geometry(tmp_path, monkeypatch,
        hidden, intermediate, layers, heads, value_heads, switch):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    snapshot = tmp_path / 'hub/models--test--unlabelled/snapshots/revision'
    snapshot.mkdir(parents=True)
    config = dict(model_type='qwen3_5', text_config=dict(hidden_size=hidden,
        intermediate_size=intermediate, num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=4, linear_num_key_heads=16, linear_num_value_heads=value_heads,
        layer_types=['linear_attention', 'full_attention'] * (layers // 2)))
    (snapshot / 'config.json').write_text(json.dumps(config))
    values = dict(MODEL_ID='test/unlabelled', MODEL_REVISION='revision', SPARK_COUNT='0', RTX_GPUS='1')
    expected = {'CUTEAFD_WIP_QWEN_DENSE_AOT': 'ON',
        'CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES': '', 'CUTEAFD_WIP_EXPERT_FAMILIES': '',
        'CUTEAFD_WIP_GENERIC_SPARK_COUNTS': '', 'CUTEAFD_WIP_SPARK_TP_ROLES': ''}
    # Misleading names must never choose the native geometry.
    card = dict(family='qwen3_5', name='qwen38-27b' if switch == 'QWEN35' else 'qwen35-9b', config=None)
    assert aot.family_scopes([card], lambda _: values) == expected
    family, scopes = aot.config_scopes(values)
    assert family == 'qwen3_5' and scopes == expected
    values['SPARK_COUNT'] = '2'
    with pytest.raises(ValueError, match='no Spark layout'):
        aot.family_scopes([card], lambda _: values)
    values['SPARK_COUNT'] = '0'
    config['text_config']['hidden_size'] = 1
    (snapshot / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unsupported dense Qwen WIP AOT checkpoint geometry'):
        aot.config_scopes(values)


@pytest.mark.parametrize('count', [0, 4])
def test_kimi_k3_refuses_every_layout_but_tp6(count):
    values = dict(MODEL_ID='test/Kimi-K3-EXL3-K2', SPARK_COUNT=str(count), RTX_GPUS='2')
    with pytest.raises(ValueError, match='TP6 only'):
        aot.family_scopes([dict(family='kimi_k3', name='kimi-k3-exl3-k2', config=None)], lambda _: values)


def test_kimi_k3_routed_range_from_its_own_config():
    config = dict(model_type='kimi_k3', text_config=dict(num_hidden_layers=93, first_k_dense_replace=1, moe_layer_freq=1))
    assert aot.checkpoint_family.describe(config) == ('kimi_k3', 'kimi_k3', 1, 92)
    assert aot.checkpoint_family.routed_layers(config) == 92


@pytest.mark.parametrize('operation', ['card', 'config', 'seal'])
def test_unknown_family_has_no_silent_empty_scope(operation, tmp_path):
    with pytest.raises(ValueError, match='no WIP AOT scope mapping for unknown'):
        if operation == 'card':
            aot.family_scopes([dict(family='unknown', config=None)], lambda _: {})
        elif operation == 'config':
            aot.config_scopes(dict(FAMILY='unknown'))
        else:
            aot.check_programs('unknown', tmp_path / 'PROGRAMS.json')


@pytest.mark.parametrize('family,model,count,program,coordinator,spark,counts,roles,extra', [
    ('deepseek_v41', 'DeepSeek-V4.1-Flash', 4, None, '', '', '', '', {}),
    ('deepseek_v4', 'DeepSeek-V4-Flash', 2, 'DSV4', 'dsv4f:rtx_backbone;dsv4f:spark_tp2',
     'dsv4f:rtx_backbone;dsv4f:spark_tp2', '', '', {}),
    ('glm5', 'GLM-5.3-EXL3-K4', 4, 'GLM', 'glm:exl3-k45;glm:fp8', 'glm:exl3-k45;glm:fp8', 'glm=4', '', {}),
    ('glm5_flash', 'GLM-5.3-Flash-NVFP4', 4, 'GLMF', 'glmf:nvfp4;glmfdense:nvfp4', 'glmf:nvfp4', 'glmf=4', '', {}),
    ('mimo_v2', 'MiMo-V2.6-Pro', 6, 'MIMO', 'mimop:fp8', 'mimop:fp8', 'mimop=6', '',
     {'CUTEAFD_WIP_AUDIO_AOT': 'ON', 'CUTEAFD_WIP_MIMO_GEOMETRIES': 'mimop'}),
    ('qwen4', 'Qwen3.8-Flash-Next-EXL3', 0, 'QWEN4', 'qwen4:exl3-k45;qwen4:nvfp4', '', '', '', {}),
    ('kimi_k3', 'Kimi-K3-EXL3-K2', 6, 'KIMIK3', 'kimik3:exl3-k23', 'kimik3:exl3-k23', 'kimik3=6', '', {}),
])
def test_existing_family_scope_outputs_are_unchanged(family, model, count, program,
        coordinator, spark, counts, roles, extra):
    values = dict(MODEL_ID='test/' + model, SPARK_COUNT=str(count), RTX_GPUS='1')
    expected = dict(CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES=coordinator,
        CUTEAFD_WIP_EXPERT_FAMILIES=spark, CUTEAFD_WIP_GENERIC_SPARK_COUNTS=counts,
        CUTEAFD_WIP_SPARK_TP_ROLES=roles, **extra)
    if program:
        expected[f'CUTEAFD_WIP_{program}_AOT'] = 'ON'
    assert aot.family_scopes([dict(family=family, name=model.lower(), config=None)], lambda _: values) == expected


@pytest.mark.parametrize('kind', ['step5_vl', 'step5'])
def test_step5_metadata_build_scope_and_routed_extent(tmp_path, monkeypatch, kind):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    text = dict(model_type='step5', num_hidden_layers=92, moe_layer_list=list(range(3, 91)))
    config = dict(model_type=kind, text_config=text) if kind == 'step5_vl' else text
    snapshot = tmp_path / 'hub/models--test--step5/snapshots/revision'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text(json.dumps(config))
    family, scopes = aot.config_scopes(dict(MODEL_ID='test/step5', MODEL_REVISION='revision', SPARK_COUNT='6'))
    assert family == 'step5'
    assert scopes['CUTEAFD_WIP_STEP5_AOT'] == 'ON'
    assert aot.checkpoint_family.describe(config) == ('step5', kind, 3, 90)
    assert aot.checkpoint_family.routed_layers(config) == 88
    text['moe_layer_list'][-1] = 91
    with pytest.raises(SystemExit, match='routed layer map'):
        aot.checkpoint_family.describe(config)


def test_step5_bf16_input_sibling_is_allowed_in_wip(tmp_path):
    config = config_file(tmp_path, 'step5', 'Step-5-Preview', 6)
    output = dry_run(tmp_path, config, {'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES': 'step5'})
    assert 'WIP AOT scopes (step5)' in output
    # Derived without the environment, and the derived value reaches the build (#516).
    derived = dry_run(tmp_path, config)
    assert 'FP8 BF16-input siblings: step5' in derived
    wip = (ROOT / 'wip.sh').read_text()
    pattern = wip.split("bf16_family_pattern='^(", 1)[1].split(')', 1)[0]
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    allowed = build.split('case "$bf16_family" in', 1)[1].split(') ;;', 1)[0].strip()
    assert set(pattern.split('|')) == set(allowed.split('|'))
    assert 'step5' in allowed.split('|')
    with pytest.raises(AssertionError, match='must be a semicolon list'):
        dry_run(tmp_path, config, {'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES': 'step5;unknown'})


def test_step5_card_union_keeps_native_format_tp_counts():
    cards = [dict(family='step5', name='step5-nvfp4', config='min'),
             dict(family='step5', name='step5-fp8', config='max')]
    configs = {'min': dict(MODEL_ID='stepfun-ai/Step-5-Preview-NVFP4', SPARK_COUNT='4', RTX_GPUS='1'),
               'max': dict(MODEL_ID='stepfun-ai/Step-5-Preview', SPARK_COUNT='6', RTX_GPUS='2')}
    scopes = aot.family_scopes(cards, configs.__getitem__)
    assert scopes['CUTEAFD_WIP_STEP5_AOT'] == 'ON'
    assert set(scopes['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'].split(';')) == {
        'step5:nvfp4=4', 'step5:fp8=6', 'step5:bf16=4', 'step5:bf16=6'}
    # The FP8 card's BF16 rows need the BF16-input sibling; it is derived, not hand-set (#516).
    assert scopes['CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES'] == 'step5'
    nvfp4_only = aot.family_scopes(cards[:1], configs.__getitem__)
    assert 'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES' not in nvfp4_only


def test_non_kimi_scope_does_not_enable_kimi_programs(tmp_path):
    path = config_file(tmp_path, 'glm5', 'GLM-5.3-EXL3-K4', 4)
    _, scopes = aot.config_scopes(aot.read_config(path))
    assert 'CUTEAFD_WIP_KIMIK3_AOT' not in scopes
    output = dry_run(tmp_path, path)
    assert 'CUTEAFD_WIP_KIMIK3_AOT' not in output
    assert scopes['CUTEAFD_WIP_GLM_AOT'] == 'ON'
