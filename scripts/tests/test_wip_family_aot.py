"""Direct WIP builds derive the same family scopes as single gate cards."""
import ast
import importlib.util
import json
import os
import re
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
    ('step5', 'Step-5-Preview-EXL3-K4-Core-K6', 6, 'STEP5', 'step5:exl3-k46'),
    ('kimi_k3', 'Kimi-K3-EXL3-K2', 6, 'KIMIK3', 'kimik3:exl3-k23'),
    ('muse_spark', 'Muse-Spark-Open-EXL3-K4-Core-K6', 6, 'MUSE', 'muse:exl3-k45'),
    ('mistral_large4', 'Mistral-Large-4-EXL3-K4-Core-K6', 6, 'ML4', 'ml4:exl3-k45'),
]


def test_local_kimi_scope_does_not_request_zero_rank_spark_exports():
    _, scopes = aot.config_scopes({'FAMILY': 'kimi_k3', 'MODEL_ID': 'test/Kimi-K3-EXL3-K2', 'SPARK_COUNT': '0'})
    assert scopes['CUTEAFD_WIP_EXPERT_FAMILIES'] == ''
    assert scopes['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == ''
    assert scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == 'kimik3:exl3-k23'


def config_file(tmp_path, family, model, count):
    path = tmp_path / 'model.config'
    hosts = '\n'.join(f'SPARK_{i}_HOST=spark{i}\nSPARK_{i}_LANE_A=192.0.2.{i + 1}\n'
                      f'SPARK_{i}_LANE_B=198.51.100.{i + 1}' for i in range(count))
    path.write_text(f'FAMILY={family}\nMODEL_ID=test/{model}\nSPARK_COUNT={count}\n'
                    f'RTX_GPUS=2\nCOORDINATOR_GPUS=1,0\n{hosts}\n')
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
    if family == 'step5' and 'exl3' in scope:
        # EXL3 K4 experts carry K6 tails in the same package: no BF16 tail scope.
        assert set(derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')) == {scope}
        assert set(derived['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')) == {scope}
        assert set(derived['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'].split(';')) == {f'step5:exl3={count}'}
    elif family == 'step5':
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


def test_every_build_aot_switch_is_forwarded_to_coordinator_container():
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    required = set(re.findall(r'CUTEAFD_WIP_[A-Z0-9_]+_AOT\b', build))
    wip = (ROOT / 'wip.sh').read_text()
    command = wip.split('local -a coordinator_build=(docker exec', 1)[1].split(
        '"$coordinator_container"', 1)[0]
    # EXLR8 is passed via an optional -e array; inspect its definition too.
    for array in set(re.findall(r'\$\{([a-z0-9_]+)\[@\]', command)):
        command += '\n' + '\n'.join(re.findall(r'\b' + array + r'=\([^\n]*', wip))
    forwarded = set(re.findall(r'-e\s+[\"\']?(CUTEAFD_WIP_[A-Z0-9_]+_AOT)=', command))
    assert required and required <= forwarded, sorted(required - forwarded)


@pytest.mark.parametrize('mode', ['empty', 'muse', 'spark-partial'])
def test_program_geometry_is_validated_at_configure_time(tmp_path, mode):
    cmake = ROOT / 'native/cmake/shared/dsv4_programs.cmake'
    (tmp_path / 'CMakeLists.txt').write_text('\n'.join([
        'cmake_minimum_required(VERSION 3.18)', 'project(muse_scope NONE)',
        f'set(CUTEAFD_CUDA_ARCHITECTURES {121 if mode == "spark-partial" else 120})',
        'set(CUTEAFD_CONTEXT_SPLIT_AOT OFF)',
        f'set(CUTEAFD_ENABLE_MUSE_AOT {"ON" if mode == "muse" else "OFF"})',
        'set(CUTEAFD_PARTIAL_REPLY_PROGRAMS ON)',
        'add_custom_target(cuteafd_verify_sparkinfer_source)', f'include("{cmake}")',
        'file(WRITE "${CMAKE_BINARY_DIR}/geometry.txt" "${CUTEAFD_PROGRAM_GEOMETRY}")',
    ]))
    build = tmp_path / 'build'
    result = subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build)],
                            capture_output=True, text=True, timeout=30)
    if mode == 'empty':
        assert result.returncode != 0
        assert 'Coordinator program geometry is empty' in result.stdout + result.stderr
        assert not (build / 'geometry.txt').exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        expected = 'muse,muse2,expert_partial,dense_trellis_k6' if mode == 'muse' else 'expert_partial'
        assert (build / 'geometry.txt').read_text() == expected


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


@pytest.mark.parametrize('count,tp,role', [('6', '2', 'tp2'), ('6', '6', 'tp6'), ('3', '3', 'tp3'), ('6', '', 'tp6'), ('8', '8', 'tp8')])
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


@pytest.mark.parametrize('count', [4, 5, 6, 7, 8])
def test_kimi_k3_scopes_use_the_requested_world(count):
    values = dict(MODEL_ID='test/Kimi-K3-EXL3-K2', SPARK_COUNT=str(count), RTX_GPUS='2')
    result = aot.family_scopes([dict(family='kimi_k3', name='kimi-k3-exl3-k2', config=None)], lambda _: values)
    assert result['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == f'kimik3={count}'


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
    ('muse_spark', 'Muse-Spark-Open-EXL3-K4-Core-K6', 6, 'MUSE', 'muse:exl3-k45', 'muse:exl3-k45', 'muse=6', '', {}),
    ('mistral_large4', 'Mistral-Large-4-EXL3-K4-Core-K6', 6, 'ML4', 'ml4:exl3-k45', 'ml4:exl3-k45', 'ml4=6', '', {}),
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


def test_optional_bf16_input_sibling_is_allowed_in_wip(tmp_path):
    config = config_file(tmp_path, 'mimo_v2', 'MiMo-V2.6-Pro', 6)
    output = dry_run(tmp_path, config, {'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES': 'mimop'})
    assert 'WIP AOT scopes (mimo_v2)' in output
    assert 'FP8 BF16-input siblings: mimop' in output
    wip = (ROOT / 'wip.sh').read_text()
    pattern = wip.split("bf16_family_pattern='^(", 1)[1].split(')', 1)[0]
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    allowed = build.split('case "$bf16_family" in', 1)[1].split(') ;;', 1)[0].strip()
    assert set(pattern.split('|')) == set(allowed.split('|'))
    assert 'mimop' in allowed.split('|')
    with pytest.raises(AssertionError, match='must be a semicolon list'):
        dry_run(tmp_path, config, {'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES': 'mimop;unknown'})


def test_step5_card_union_keeps_native_format_tp_counts():
    cards = [dict(family='step5', name='step5-nvfp4', config='min'),
             dict(family='step5', name='step5-fp8', config='max')]
    configs = {'min': dict(MODEL_ID='stepfun-ai/Step-5-Preview-NVFP4', SPARK_COUNT='4', RTX_GPUS='1'),
               'max': dict(MODEL_ID='stepfun-ai/Step-5-Preview', SPARK_COUNT='6', RTX_GPUS='2')}
    scopes = aot.family_scopes(cards, configs.__getitem__)
    assert scopes['CUTEAFD_WIP_STEP5_AOT'] == 'ON'
    assert set(scopes['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'].split(';')) == {
        'step5:nvfp4=4', 'step5:fp8=6', 'step5:bf16=4', 'step5:bf16=6'}
    # FP8 uses K32; BF16 tail layers use their native weight package, not an FP8 sibling.
    assert 'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES' not in scopes
    for card in cards:
        single = aot.family_scopes([card], configs.__getitem__)
        assert 'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES' not in single


def test_non_kimi_scope_does_not_enable_kimi_programs(tmp_path):
    path = config_file(tmp_path, 'glm5', 'GLM-5.3-EXL3-K4', 4)
    _, scopes = aot.config_scopes(aot.read_config(path))
    assert 'CUTEAFD_WIP_KIMIK3_AOT' not in scopes
    output = dry_run(tmp_path, path)
    assert 'CUTEAFD_WIP_KIMIK3_AOT' not in output
    assert scopes['CUTEAFD_WIP_GLM_AOT'] == 'ON'


@pytest.mark.parametrize('kimi,glm,glmf,exlr8,families', [
    (True, False, False, False, 'kimik3'),
    (False, True, False, False, 'glm'),
    (True, True, False, False, 'kimik3,glm'),
    (True, True, False, True, 'kimik3,glm'),
    (False, False, True, False, 'glmf'),
    (False, False, True, True, 'glmf'),
    (False, True, True, False, 'glm,glmf'),
    (True, False, True, False, 'kimik3,glmf'),
    (True, True, True, True, 'kimik3,glm,glmf'),
])
def test_native_k6_architecture_tables_follow_existing_cmake_scope(tmp_path, kimi, glm, glmf, exlr8, families):
    cmake = ROOT / 'native/cmake/shared/dsv4_programs.cmake'
    source = tmp_path / 'CMakeLists.txt'
    exlr8_families = ';'.join(family for family in families.split(',') if family in {'glm', 'glmf'})
    source.write_text('\n'.join([
        'cmake_minimum_required(VERSION 3.18)',
        'project(native_scope NONE)',
        f'set(CUTEAFD_ENABLE_GLMF_AOT {"ON" if glmf else "OFF"})',
        'set(CUTEAFD_CUDA_ARCHITECTURES 120)',
        'set(CUTEAFD_CONTEXT_SPLIT_AOT OFF)',
        'set(CUTEAFD_GLM_GEOMETRIES "glm;glm2")',
        f'set(CUTEAFD_ENABLE_KIMIK3_AOT {"ON" if kimi else "OFF"})',
        f'set(CUTEAFD_ENABLE_GLM_AOT {"ON" if glm else "OFF"})',
        f'set(CUTEAFD_ENABLE_EXLR8_AOT {"ON" if exlr8 else "OFF"})',
        f'set(CUTEAFD_EXLR8_FAMILIES "{exlr8_families}")',
        'add_custom_target(cuteafd_verify_sparkinfer_source)',
        f'include("{cmake}")',
        'file(WRITE "${CMAKE_BINARY_DIR}/scope.txt" "${CUTEAFD_PROGRAM_GEOMETRY}\\n${CUTEAFD_DSV4_EXPORT_ARGS}\\n")',
    ]))
    build = tmp_path / 'build'
    result = subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    geometries, args = (build / 'scope.txt').read_text().splitlines()
    geometries = geometries.split(',')
    assert len(geometries) == len(set(geometries))
    assert geometries.count('dense_trellis_k6') == 1
    assert ('glm_stage' in geometries) == (glm or exlr8)
    assert ('glm2_stage' in geometries) == (glm or exlr8)
    assert ('glmf_stage' in geometries) == glmf
    assert ('glmf2_stage' in geometries) == glmf
    assert f'--dense-trellis-families;{families}' in args
    assert 'snapshot' not in args and 'checkpoint' not in args


@pytest.mark.parametrize('mode', ['OFF', 'ON', 'ONLY'])
@pytest.mark.parametrize('kimi,exlr8', [(False, False), (True, False), (False, True), (True, True)])
def test_glm_context_only_passes_only_supported_base_geometries(tmp_path, mode, kimi, exlr8):
    cmake = ROOT / 'native/cmake/shared/dsv4_programs.cmake'
    (tmp_path / 'CMakeLists.txt').write_text('\n'.join([
        'cmake_minimum_required(VERSION 3.18)', 'project(context_scope NONE)',
        'set(CUTEAFD_CUDA_ARCHITECTURES 120)',
        f'set(CUTEAFD_CONTEXT_SPLIT_AOT {mode})',
        'set(CUTEAFD_GLM_GEOMETRIES "glm;glm2")', 'set(CUTEAFD_ENABLE_GLM_AOT ON)',
        f'set(CUTEAFD_ENABLE_KIMIK3_AOT {"ON" if kimi else "OFF"})',
        f'set(CUTEAFD_ENABLE_EXLR8_AOT {"ON" if exlr8 else "OFF"})',
        'set(CUTEAFD_EXLR8_FAMILIES glm)',
        'add_custom_target(cuteafd_verify_sparkinfer_source)', f'include("{cmake}")',
        'file(WRITE "${CMAKE_BINARY_DIR}/args.txt" "${CUTEAFD_DSV4_EXPORT_ARGS}")',
    ]))
    build = tmp_path / 'build'
    result = subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    args = (build / 'args.txt').read_text().split(';')
    geometries = args[args.index('--geometry') + 1].split(',')
    assert args[args.index('--context-split') + 1] == mode.lower()
    if mode == 'ONLY':
        assert geometries == ['glm']
        assert set(geometries) <= {'flash', 'pro', 'glm', 'glmf'}
    else:
        expected = ['glm', 'glm2']
        if kimi:
            expected += ['kimik3', 'dense_trellis_k6', 'kimi_split']
        else:
            expected += ['dense_trellis_k6']
        expected += ['glm_stage', 'glm2_stage']
        if exlr8:
            expected += ['exlr8', 'exlr8_dense']
        assert geometries == expected


@pytest.mark.parametrize('ranks', [1, 2])
def test_muse_scoped_exports_cover_every_startup_program(ranks, monkeypatch):
    from functools import partial
    from types import ModuleType

    # Run the real program-list builders on CPU; only GPU compiler entrypoints
    # are stubbed. No static list duplicates the exporter's scope decisions.
    source = ROOT / 'python/tools/aot/export_b12x_muse_aot.py'
    module = ast.parse(source.read_text())
    body = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    scope = {'partial': partial, 'EXPERT_WIDTH': 4096, 'WINDOW': 2048, 'SWA_RING_STEP': 64}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), scope)

    muse_source = (ROOT / 'third_party/sparkinfer/b12x/integration/cuteafd/muse.py').read_text()
    norms = next(node.value for node in ast.parse(muse_source).body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'NORMS' for target in node.targets))
    compiler_calls = []

    def compiler(**kwargs):
        compiler_calls.append(kwargs)

    packages = {}
    for name in ('b12x', 'b12x.integration', 'b12x.integration.cuteafd',
                 'b12x.integration.cuteafd.muse', 'b12x.integration.cuteafd.rope',
                 'b12x.integration.cuteafd.muse_tp', 'b12x.integration.cuteafd.pointwise',
                 'b12x.integration.cuteafd._muse_attention'):
        packages[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, packages[name])
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            monkeypatch.setattr(packages[parent], child, packages[name], raising=False)
    packages['b12x.integration.cuteafd.muse'].NORMS = ast.literal_eval(norms)
    for name, functions in {
        'muse': ('compile_norm_aot', 'compile_residual_gate_aot', 'compile_route_select_aot',
                 'compile_softcap_aot', 'compile_qk_norm_rope_aot', 'compile_attention_output_aot',
                 'compile_router_aot'),
        'rope': ('compile_rope_tables_aot',), 'muse_tp': ('compile_tp_combine_aot',),
        'pointwise': ('compile_bf16_projection_aot',),
        '_muse_attention': ('compile_attention_aot', 'compile_swa_commit_aot'),
    }.items():
        for function in functions:
            setattr(packages[f'b12x.integration.cuteafd.{name}'], function, compiler)

    exported = {f'{prefix}_{name}' for heads, prefix in ((128, 'muse'), (64, 'muse2'))
                for name, _, _, _ in scope['programs'](heads)}
    rust = ROOT / 'rust/crates/cuteafd-daemon/src/families/muse_spark'
    bindings = (rust / 'programs.rs').read_text()
    required = set()
    for section, prefix in (('POINTWISE', 'muse'), ('PER_HEADS', 'muse' if ranks == 1 else 'muse2')):
        table = bindings.split(f'const {section}:', 1)[1].split('\n];', 1)[0]
        required.update(f'{prefix}_{name}' for name in re.findall(r'\("([a-z0-9_]+)",', table))
    rope = re.search(r'const ROPE_TABLES: &str = "([^"]+)";', bindings).group(1)
    required.add(rope)
    assert len(required) == 25
    assert required <= exported, sorted(required - exported)
    assert 'required.push(programs::ROPE_TABLES.into())' in (rust / 'startup.rs').read_text()
    assert 'RotaryTables::load(device, programs, plan, "muse_",' in (rust / 'engine.rs').read_text()
    scope['builders']()['rope_s64_plain']()
    assert compiler_calls == [{'rotary_span': 64, 'llama3': False}]


def test_kimi_builds_always_package_k6_programs(tmp_path):
    path = config_file(tmp_path, 'kimi_k3', 'Kimi-K3-EXL3-K2', 6)
    _, scopes = aot.config_scopes(aot.read_config(path))
    assert scopes['CUTEAFD_WIP_KIMIK3_AOT'] == 'ON'
    assert not any('KIMIK6' in key for key in scopes)
    for name in ('wip.sh', 'scripts/build/build-wip-artifacts.sh', 'scripts/build/build-release-artifacts.sh',
                 'native/CMakeLists.txt', 'native/cmake/shared/dsv4_programs.cmake'):
        assert 'KIMIK6' not in (ROOT / name).read_text(), name
    table = (ROOT / 'native/cmake/shared/dsv4_programs.cmake').read_text()
    assert '"kimik3,dense_trellis_k6,kimi_split"' in table
    for exporter in ['export_b12x_dense_trellis_k6_aot.py', 'export_b12x_kimi_split_aot.py']:
        assert exporter in table


@pytest.mark.parametrize('glm,kimi', [(False, False), (True, False), (False, True)])
def test_ml4_exports_its_programs_and_k6_shapes_beside_other_families(tmp_path, glm, kimi):
    cmake = ROOT / 'native/cmake/shared/dsv4_programs.cmake'
    (tmp_path / 'CMakeLists.txt').write_text('\n'.join([
        'cmake_minimum_required(VERSION 3.18)', 'project(ml4_scope NONE)',
        'set(CUTEAFD_CUDA_ARCHITECTURES 120)', 'set(CUTEAFD_CONTEXT_SPLIT_AOT OFF)',
        'set(CUTEAFD_GLM_GEOMETRIES "glm;glm2")', 'set(CUTEAFD_ENABLE_ML4_AOT ON)',
        f'set(CUTEAFD_ENABLE_GLM_AOT {"ON" if glm else "OFF"})',
        f'set(CUTEAFD_ENABLE_KIMIK3_AOT {"ON" if kimi else "OFF"})',
        'add_custom_target(cuteafd_verify_sparkinfer_source)', f'include("{cmake}")',
        'file(WRITE "${CMAKE_BINARY_DIR}/args.txt" "${CUTEAFD_DSV4_EXPORT_ARGS}")',
    ]))
    build = tmp_path / 'build'
    result = subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    args = (build / 'args.txt').read_text().split(';')
    geometries = args[args.index('--geometry') + 1].split(',')
    assert geometries.count('ml4') == 1 and geometries.count('dense_trellis_k6') == 1
    families = args[args.index('--dense-trellis-families') + 1].split(',')
    assert families[-1] == 'ml4' and ('glm' in families) == glm and ('kimik3' in families) == kimi
    assert ('glm_stage' in geometries) == glm


def test_ml4_engine_program_names_are_exported_stems():
    """Every ml4_* / ml42_* program name the GLM engine builds for Mistral Large 4 is a stem the
    exporter emits (ml4_programs for the whole model, ml4_head_split_programs for a head split's
    share; RoPE stems carry no tag, NoPE ones ``_nope``)."""
    import re
    exporter = (ROOT / 'python/tools/aot/export_b12x_dsv4_aot.py').read_text()
    exported = set()
    for function, family in (('ml4_programs', 'ml4'), ('ml4_head_split_programs', 'ml42')):
        body = exporter.split(f'def {function}(')[1].split('\ndef ')[0]
        for stem in re.findall(r'\(f?"([a-z0-9_{}]+)", "', body):
            names = {stem}
            for field, values in (('{tag}', ('', '_nope')), ('{mode}', ('decode', 'prefill')),
                                  ('{rows}', ('64', '4096')), ('{inter}', ('6144', '12288', '3072'))):
                names = {n.replace(field, v) for n in names for v in values} if field in stem else names
            exported |= {f'{family}_{n}' for n in names}
    engine = ''.join((ROOT / p).read_text() for p in (
        'rust/crates/cuteafd-daemon/src/families/glm5/engine.rs',
        'rust/crates/cuteafd-daemon/src/families/glm5/engine/native_dense.rs'))
    used = re.findall(r'"((?:ml4|\{family\})_[a-z0-9_{}]+)"', engine)
    assert any(u.startswith('{family}') for u in used) and any(u.startswith('ml4_') for u in used)
    for pattern in used:
        assert '_rope' not in pattern, pattern
        # Positional `{}` hides which tag a call passes (it once asked for `_rope`); name it.
        assert '{}' not in pattern, f'{pattern}: use named fields'
        names = {pattern}
        for field, values in (('{family}', ('ml4', 'ml42')), ('{tag}', ('', '_nope')),
                              ('{mode}', ('decode', 'prefill')), ('{cap}', ('m64', 'm4096')),
                              ('{intermediate}', ('6144', '12288', '3072')),
                              ('{placed}', ('', '_context')), ('{scale_placed}', ('', '_context'))):
            names = {n.replace(field, v) for n in names for v in values}
        assert all('{' not in n for n in names), pattern
        missing = sorted(n for n in names if n not in exported and not n.startswith('ml4_draft')
                         # Whole-model programs have no _context form; the engine only places a
                         # head split's share (ml42) into the combined query.
                         and not (n.startswith('ml4_') and '_context' in n)
                         # NoPE query scales write no tail: they have no _context form.
                         and not ('query_scale_nope_context' in n))
        if pattern.startswith('{family}'):
            missing = [n for n in missing if not (n.startswith('ml4_') and '_context' in n)]
        assert not missing, f'{pattern}: not exported: {missing}'


@pytest.mark.parametrize('family,exl3_model,fp8_model,exl3_scope,extra', [
    ('glm5', 'wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1', 'zai-org/GLM-5.3', 'glm:exl3-k45', {}),
    ('glm5_flash', 'brandonmusic/GLM-5.3-Flash-tr3-4bpw', 'zai-org/GLM-5.3-Flash', 'glmf:exl3-k34', {}),
    ('mimo_v2', 'test/MiMo-V2.6-Flash-EXL3-K4', 'XiaomiMiMo/MiMo-V2.6-Flash-MOPD', 'mimof:exl3-k45', {}),
    ('qwen4', 'wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1', 'nvidia/Qwen3.8-Flash-Next-NVFP4', 'qwen4:exl3-k45', {}),
    ('step5', 'wrldsuksgo2mars/Step-5-Preview-EXL3-K4-Core-K6', 'stepfun-ai/Step-5-Preview', 'step5:exl3-k46', {}),
    ('deepseek_v4', 'wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1', 'deepseek-ai/DeepSeek-V4-Pro', 'dsv4p:exl3-k23', {'MODEL_VARIANT': 'pro'}),
])
def test_expert_scope_derived_from_checkpoint_not_card_name(tmp_path, monkeypatch, family, exl3_model, fp8_model, exl3_scope, extra):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    base_values = dict(SPARK_COUNT='4', RTX_GPUS='1', **extra)

    # 1. Identical EXL3 card named without 'exl3' or 'tr3' still gets the exl3 scope
    #    (both from MODEL_ID and from local config.json quantization_config).
    exl3_values = dict(base_values, MODEL_ID=exl3_model)
    named_card = dict(family=family, name=exl3_model.lower(), config=None)
    renamed_card = dict(family=family, name='renamed-plain-card', config=None)
    expected = aot.family_scopes([named_card], lambda _: exl3_values)
    assert aot.family_scopes([renamed_card], lambda _: exl3_values) == expected
    assert exl3_scope in expected['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert exl3_scope in expected['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')

    snap = tmp_path / 'hub/models--test--custom-checkpoint/snapshots/rev'
    snap.mkdir(parents=True, exist_ok=True)
    (snap / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
    metadata_values = dict(base_values, MODEL_ID='test/custom-checkpoint', MODEL_REVISION='rev')
    assert aot.family_scopes([renamed_card], lambda _: metadata_values) == expected

    # 2. A card whose name contains 'exl3'/'tr3' but whose checkpoint is not EXL3 does not get exl3_scope.
    misleading_card = dict(family=family, name='misleading-exl3-tr3-card', config=None)
    fp8_values = dict(base_values, MODEL_ID=fp8_model)
    fp8_scopes = aot.family_scopes([misleading_card], lambda _: fp8_values)
    assert exl3_scope not in fp8_scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert exl3_scope not in fp8_scopes['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')

    (snap / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'fp8'}}))
    fp8_meta_scopes = aot.family_scopes([misleading_card], lambda _: metadata_values)
    assert exl3_scope not in fp8_meta_scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert exl3_scope not in fp8_meta_scopes['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')

    # 3. Unknown checkpoint without quantization metadata fails closed instead of guessing from card name.
    (snap / 'config.json').write_text(json.dumps({'model_type': family}))
    with pytest.raises(ValueError, match='cannot derive WIP AOT expert format'):
        aot.family_scopes([misleading_card], lambda _: metadata_values)
    with pytest.raises(ValueError, match='cannot derive WIP AOT expert format'):
        aot.family_scopes([misleading_card], lambda _: dict(base_values))


@pytest.mark.parametrize('family,exl3_model,fp8_model,exl3_scope,extra', [
    ('glm5', 'wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1', 'zai-org/GLM-5.3', 'glm:exl3-k45', {}),
    ('glm5_flash', 'brandonmusic/GLM-5.3-Flash-tr3-4bpw', 'zai-org/GLM-5.3-Flash', 'glmf:exl3-k34', {}),
    ('mimo_v2', 'test/MiMo-V2.6-Flash-EXL3-K4', 'XiaomiMiMo/MiMo-V2.6-Flash-MOPD', 'mimof:exl3-k45', {}),
    ('qwen4', 'wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1', 'nvidia/Qwen3.8-Flash-Next-NVFP4', 'qwen4:exl3-k45', {}),
    ('step5', 'wrldsuksgo2mars/Step-5-Preview-EXL3-K4-Core-K6', 'stepfun-ai/Step-5-Preview', 'step5:exl3-k46', {}),
    ('deepseek_v4', 'wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1', 'deepseek-ai/DeepSeek-V4-Pro', 'dsv4p:exl3-k23', {'MODEL_VARIANT': 'pro'}),
])
def test_expert_format_config_key_overrides_checkpoint_and_card_name(tmp_path, monkeypatch, family, exl3_model, fp8_model, exl3_scope, extra):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    base_values = dict(SPARK_COUNT='4', RTX_GPUS='1', **extra)

    # EXPERT_FORMAT=fp8 overrides an EXL3 checkpoint (both MODEL_ID and local config.json) and 'exl3' card name.
    exl3_snap = tmp_path / ('hub/models--' + exl3_model.replace('/', '--') + '/snapshots/rev')
    exl3_snap.mkdir(parents=True, exist_ok=True)
    (exl3_snap / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
    overridden_to_fp8 = aot.family_scopes(
        [dict(family=family, name='card-with-exl3-tr3', config=None)],
        lambda _: dict(base_values, MODEL_ID=exl3_model, MODEL_REVISION='rev', EXPERT_FORMAT='fp8'),
    )
    assert exl3_scope not in overridden_to_fp8['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert exl3_scope not in overridden_to_fp8['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')

    # EXPERT_FORMAT=exl3 overrides an FP8 checkpoint (both MODEL_ID and local config.json) and plain card name.
    fp8_snap = tmp_path / ('hub/models--' + fp8_model.replace('/', '--') + '/snapshots/rev')
    fp8_snap.mkdir(parents=True, exist_ok=True)
    (fp8_snap / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'fp8'}}))
    overridden_to_exl3 = aot.family_scopes(
        [dict(family=family, name='plain-fp8-card', config=None)],
        lambda _: dict(base_values, MODEL_ID=fp8_model, MODEL_REVISION='rev', EXPERT_FORMAT='exl3'),
    )
    assert exl3_scope in overridden_to_exl3['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert exl3_scope in overridden_to_exl3['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')

    with pytest.raises(ValueError, match='unsupported EXPERT_FORMAT'):
        aot.family_scopes(
            [dict(family=family, name='plain-card', config=None)],
            lambda _: dict(base_values, MODEL_ID=fp8_model, EXPERT_FORMAT='bogus'),
        )
