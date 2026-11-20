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

from launcher_sandbox import seed_exl3_index

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
    ('mistral_large4', 'Mistral-Large-4-EXL3-K4', 6, 'ML4', 'ml4:exl3-k45'),
]


def test_local_kimi_scope_does_not_request_zero_rank_spark_exports():
    seed_exl3_index(os.environ['HF_HOME'], 'test/Kimi-K3-EXL3-K2')
    _, scopes = aot.config_scopes({'FAMILY': 'kimi_k3', 'MODEL_ID': 'test/Kimi-K3-EXL3-K2', 'SPARK_COUNT': '0'})
    assert scopes['CUTEAFD_WIP_EXPERT_FAMILIES'] == ''
    assert scopes['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == ''
    assert scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == 'kimik3:exl3-k23'


def config_file(tmp_path, family, model, count):
    if 'EXL3' in model or '-tr3' in model or family in ('kimi_k3', 'muse_spark', 'mistral_large4'):
        seed_exl3_index(os.environ['HF_HOME'], f'test/{model}')
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
        assert (build / 'geometry.txt').read_text() == expected + ',expert_reply'


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


@pytest.mark.parametrize('count,role', [('6', 'tp6'), ('3', 'tp3'), ('8', 'tp8'), ('5', 'tp5')])
def test_v41_direct_config_uses_the_spark_count_role(count, role):
    values = {'FAMILY': 'deepseek_v41', 'SPARK_COUNT': count}
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
    seed_exl3_index(os.environ['HF_HOME'], 'test/Kimi-K3-EXL3-K2')
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
    ('deepseek_v41', 'DeepSeek-V4.1-Flash', 4, None, '', '', '', '', {'CUTEAFD_WIP_V41_AOT': 'ON'}),
    ('deepseek_v4', 'DeepSeek-V4-Flash', 2, 'DSV4', 'dsv4f:rtx_backbone;dsv4f:spark_tp2',
     'dsv4f:rtx_backbone;dsv4f:spark_tp2', '', '', {}),
    ('glm5', 'GLM-5.3-EXL3-K4', 4, 'GLM', 'glm:exl3-k45;glm:fp8', 'glm:exl3-k45;glm:fp8', 'glm=4', '', {}),
    ('glm5_flash', 'GLM-5.3-Flash-NVFP4', 4, 'GLMF', 'glmf:nvfp4;glmfdense:nvfp4', 'glmf:nvfp4', 'glmf=4', '',
     {'CUTEAFD_WIP_VISION_ATTENTION_AOT': 'ON'}),
    ('mimo_v2', 'MiMo-V2.6-Pro', 6, 'MIMO', 'mimop:fp8', 'mimop:fp8', 'mimop=6', '',
     {'CUTEAFD_WIP_AUDIO_AOT': 'ON', 'CUTEAFD_WIP_MIMO_GEOMETRIES': 'mimop'}),
    ('qwen4', 'Qwen3.8-Flash-Next-EXL3', 0, 'QWEN4', 'qwen4:exl3-k45;qwen4:nvfp4', '', '', '',
     {'CUTEAFD_WIP_VISION_ATTENTION_AOT': 'ON'}),
    ('kimi_k3', 'Kimi-K3-EXL3-K2', 6, 'KIMIK3', 'kimik3:exl3-k23', 'kimik3:exl3-k23', 'kimik3=6', '', {'CUTEAFD_WIP_VISION_ATTENTION_AOT': 'ON'}),
    ('muse_spark', 'Muse-Spark-Open-EXL3-K4-Core-K6', 6, 'MUSE', 'muse:exl3-k45', 'muse:exl3-k45', 'muse=6', '',
     {'CUTEAFD_WIP_AUDIO_AOT': 'ON', 'CUTEAFD_WIP_VISION_ATTENTION_AOT': 'ON'}),
    ('mistral_large4', 'Mistral-Large-4-EXL3-K4', 6, 'ML4', 'ml4:exl3-k45', 'ml4:exl3-k45', 'ml4=6', '',
     {'CUTEAFD_WIP_VISION_ATTENTION_AOT': 'ON'}),
])
def test_existing_family_scope_outputs_are_unchanged(family, model, count, program,
        coordinator, spark, counts, roles, extra):
    if ':exl3-' in coordinator:
        seed_exl3_index(os.environ['HF_HOME'], 'test/' + model)
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
            expected += ['kimik3', 'kimik31', 'dense_trellis_k6', 'kimi_split']
        else:
            expected += ['dense_trellis_k6']
        expected += ['glm_stage', 'glm2_stage']
        if exlr8:
            expected += ['exlr8', 'exlr8_dense']
        assert geometries == expected + ['expert_reply']


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
    assert '"kimik3,kimik31,dense_trellis_k6,kimi_split"' in table
    for exporter in ['export_b12x_dense_trellis_k6_aot.py', 'export_b12x_kimi_split_aot.py']:
        assert exporter in table


def test_kimi_shipping_k8_inventory_preserves_every_existing_geometry():
    import ast

    source = (ROOT / 'python/tools/aot/export_b12x_dense_trellis_k6_aot.py').read_text()
    tree = ast.parse(source)
    scope = {}
    definitions = {'_bits', '_suffix', '_key', '_codebooks', 'shapes_for', 'row_policies', 'grouped_policy', 'output_dtype'}
    for node in tree.body:
        if ((isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets))
                or isinstance(node, ast.FunctionDef) and node.name in definitions):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<trellis inventory>', 'exec'), scope)
    old_scope = dict(scope)
    old_tree = ast.parse(source)
    generator = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'shapes_for')
    width_families = next(node.comparators[0] for node in ast.walk(generator) if isinstance(node, ast.Compare)
                          and isinstance(node.left, ast.Name) and node.left.id == 'family'
                          and isinstance(node.ops[0], ast.In) and isinstance(node.comparators[0], ast.Tuple))
    assert 'kimik3' in ast.literal_eval(width_families)
    width_families.elts = [name for name in width_families.elts if ast.literal_eval(name) != 'kimik3']
    for node in old_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('shapes_for', 'row_policies'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<old trellis inventory>', 'exec'), old_scope)
    for family in ('glm', 'glmf', 'muse', 'step5', 'ml4'):
        assert scope['shapes_for']((family,)) == old_scope['shapes_for']((family,))
    shapes = scope['shapes_for'](('kimik3',))
    existing = scope['KIMI_SHAPES']
    assert tuple(shape for shape in shapes if scope['_bits'](shape) == 6) == existing
    assert tuple(shape for shape in shapes if scope['_bits'](shape) == 8) == tuple(
        (*shape, 8) for shape in existing)
    policies = scope['row_policies'](('kimik3',))
    old_policies = old_scope['row_policies'](('kimik3',))
    assert [row for row in policies if row.get('bits', 6) == 6] == old_policies
    k8 = [row for row in policies if row.get('bits') == 8]
    assert len(k8) == 2 * len(existing)
    assert len({row['native_program'] for row in k8}) == len(k8)
    assert any(row['input_features'] == 7168 and row['output_features'] == 163840
               and row['native_program'] == 'trellis_lm_head_mcg_k8' for row in k8)
    assert {row['codebook'] for row in k8} == {'mcg', 'mul1'}


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
    ('glm5_flash', 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC', 'zai-org/GLM-5.3-Flash', 'glmf:exl3-k34', {}),
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
    seed_exl3_index(tmp_path, exl3_model)
    named_card = dict(family=family, name=exl3_model.lower(), config=None)
    renamed_card = dict(family=family, name='renamed-plain-card', config=None)
    expected = aot.family_scopes([named_card], lambda _: exl3_values)
    assert aot.family_scopes([renamed_card], lambda _: exl3_values) == expected
    assert exl3_scope in expected['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert exl3_scope in expected['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')

    snap = seed_exl3_index(tmp_path, 'test/custom-checkpoint', revision='rev')
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
    ('glm5_flash', 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC', 'zai-org/GLM-5.3-Flash', 'glmf:exl3-k34', {}),
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
    fp8_snap = seed_exl3_index(tmp_path, fp8_model, revision='rev')
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


@pytest.mark.parametrize('family,model,base_scope,extra', [
    ('deepseek_v4', 'test/DeepSeek-V4-Pro-EXL3-K2', 'dsv4p:exl3-k23', {'MODEL_VARIANT': 'pro'}),
    ('glm5', 'test/GLM-5.3-EXL3-K4', 'glm:exl3-k45', {}),
    ('glm5_flash', 'test/GLM-5.3-Flash-EXL3-K3', 'glmf:exl3-k34', {}),
    ('qwen4', 'test/Qwen3.8-Flash-Next-EXL3-K4.25', 'qwen4:exl3-k45', {}),
    ('step5', 'test/Step-5-Preview-EXL3-K4', 'step5:exl3-k46', {}),
    ('kimi_k3', 'test/Kimi-K3-EXL3-K2', 'kimik3:exl3-k23', {}),
    ('muse_spark', 'test/Muse-Spark-Open-EXL3-K4', 'muse:exl3-k45', {}),
    ('mistral_large4', 'test/Mistral-Large-4-EXL3-K4', 'ml4:exl3-k45', {}),
    ('inkling2', 'test/Inkling-2-Small-EXL3-K4', 'inkling2:exl3-k45', {}),
])
def test_exl3_codebook_scope_uses_only_routed_index_markers_and_fails_closed(
    tmp_path, monkeypatch, family, model, base_scope, extra,
):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    values = dict(MODEL_ID=model, MODEL_REVISION='rev', SPARK_COUNT='4', RTX_GPUS='1', **extra)
    card = [dict(family=family, name='card', config=None)]
    snap = tmp_path / ('hub/models--' + model.replace('/', '--') + '/snapshots/rev')
    snap.mkdir(parents=True, exist_ok=True)

    # Routed .mul1 markers select -mul1 on both coordinator and Spark scopes;
    # dense .mcg markers and conflicting config.json codebook fields do not mask routed .mul1.
    (snap / 'config.json').write_text(json.dumps({
        'quantization_config': {'quant_method': 'exl3', 'codebook': 'mcg'},
    }))
    (snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'model.layers.0.self_attn.q_proj.trellis': 's0.safetensors',
        'model.layers.0.self_attn.q_proj.mcg': 's0.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 's1.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mul1': 's1.safetensors',
    }}))
    mul1_scopes = aot.family_scopes(card, lambda _: values)
    assert base_scope + '-mul1' in mul1_scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert base_scope + '-mul1' in mul1_scopes['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')
    assert base_scope not in mul1_scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')

    # Routed .mcg markers keep MCG scope unchanged even when dense/head tensors use .mul1
    # or config.json declares codebook='mul1'.
    (snap / 'config.json').write_text(json.dumps({
        'quantization_config': {'quant_method': 'exl3', 'codebook': 'mul1'},
    }))
    (snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'language_model.lm_head.trellis': 's0.safetensors',
        'language_model.lm_head.mul1': 's0.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 's1.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mcg': 's1.safetensors',
    }}))
    mcg_scopes = aot.family_scopes(card, lambda _: values)
    assert base_scope in mcg_scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert base_scope in mcg_scopes['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')
    assert base_scope + '-mul1' not in mcg_scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')

    # EXPERT_CODEBOOK override is refused.
    with pytest.raises(ValueError, match='EXPERT_CODEBOOK override is not supported'):
        aot.family_scopes(card, lambda _: dict(values, EXPERT_CODEBOOK='mul1'))

    # Mixed routed .mcg + .mul1, dense-only index, incomplete markers, and .3inst all fail closed.
    for bad_map, pattern in [
        ({'model.layers.1.mlp.experts.0.gate_proj.mcg': 's.safetensors',
          'model.layers.1.mlp.experts.0.up_proj.mul1': 's.safetensors'}, 'mixed routed EXL3 codebooks'),
        ({'language_model.lm_head.mul1': 's.safetensors'}, 'no routed .mcg or .mul1 markers'),
        ({'model.layers.1.mlp.experts.0.gate_proj.trellis': 's.safetensors',
          'model.layers.1.mlp.experts.0.gate_proj.mul1': 's.safetensors',
          'model.layers.1.mlp.experts.0.up_proj.trellis': 's.safetensors'}, 'incomplete routed EXL3 codebook markers'),
        ({'model.layers.1.mlp.experts.0.gate_proj.3inst': 's.safetensors'}, 'unsupported routed EXL3 codebook marker'),
    ]:
        (snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': bad_map}))
        with pytest.raises(ValueError, match=pattern):
            aot.family_scopes(card, lambda _: values)


def _serve_mock_hub(monkeypatch, routes: dict[str, tuple[int, dict[str, str], bytes]]):
    from contextlib import contextmanager
    import http.server
    import threading

    requests: list[tuple[str, str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _dispatch(self, send_body: bool):
            requests.append((self.command, self.path))
            status, headers, body = routes.get(self.path, (404, {'Content-Type': 'application/json'}, b'{"error":"not found"}'))
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def do_HEAD(self):
            self._dispatch(False)

        def do_GET(self):
            self._dispatch(True)

        def log_message(self, *_args):
            pass

    @contextmanager
    def running():
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}'
            url_template = f'{url}/{{repo_id}}/resolve/{{revision}}/{{filename}}'
            monkeypatch.setenv('HF_ENDPOINT', url)
            monkeypatch.setenv('HF_HUB_OFFLINE', '0')
            monkeypatch.setenv('HF_HUB_DISABLE_TELEMETRY', '1')
            monkeypatch.setenv('HF_HUB_DISABLE_XET', '1')
            # Patch huggingface_hub constants cached at import time so HF_ENDPOINT/offline updates apply.
            import huggingface_hub.constants as hf_const
            monkeypatch.setattr(hf_const, 'ENDPOINT', url)
            monkeypatch.setattr(hf_const, 'HUGGINGFACE_CO_URL_TEMPLATE', url_template)
            monkeypatch.setattr(hf_const, 'HF_HUB_OFFLINE', False)
            yield requests
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    return running()


def test_cold_cache_fetches_only_exact_revision_index_for_wip_and_release_callers(tmp_path, monkeypatch):
    import hashlib

    release_spec = importlib.util.spec_from_file_location('release_scope_cold', ROOT / 'scripts/build/release-scope.py')
    release_scope = importlib.util.module_from_spec(release_spec)
    release_spec.loader.exec_module(release_scope)

    glmf_cfg = aot.read_config(ROOT / 'examples/configs/glm53-flash-exl3-k4-mc-min.config')
    glmf_repo = glmf_cfg['MODEL_ID']
    glmf_rev = glmf_cfg['MODEL_REVISION']
    mcg_index = json.dumps({'weight_map': {
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 'model-00001-of-00010.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mcg': 'model-00001-of-00010.safetensors',
    }}).encode()
    mcg_etag = f'"{hashlib.sha256(mcg_index).hexdigest()}"'

    mul1_repo = 'org/GLM-5.3-Flash-EXL3-K4-MUL1'
    mul1_rev = 'a' * 40
    mul1_index = json.dumps({'weight_map': {
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 'model-00001-of-00010.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mul1': 'model-00001-of-00010.safetensors',
    }}).encode()
    mul1_etag = f'"{hashlib.sha256(mul1_index).hexdigest()}"'

    routes = {
        f'/{glmf_repo}/resolve/{glmf_rev}/model.safetensors.index.json': (
            200, {'Content-Type': 'application/json', 'ETag': mcg_etag, 'X-Repo-Commit': glmf_rev}, mcg_index,
        ),
        f'/api/models/{mul1_repo}/revision/main': (
            200, {'Content-Type': 'application/json'}, json.dumps({'id': mul1_repo, 'sha': mul1_rev}).encode(),
        ),
        f'/{mul1_repo}/resolve/{mul1_rev}/model.safetensors.index.json': (
            200, {'Content-Type': 'application/json', 'ETag': mul1_etag, 'X-Repo-Commit': mul1_rev}, mul1_index,
        ),
    }

    cold_hf = tmp_path / 'cold-hf'
    monkeypatch.setenv('HF_HOME', str(cold_hf))
    with _serve_mock_hub(monkeypatch, routes) as requests:
        # 1. Cold-cache WIP card on GLM Flash MCG fetches ONLY exact-revision model.safetensors.index.json.
        wip_mcg = aot.family_scopes(
            [dict(family='glm5_flash', name='glmf-mc', config=str(ROOT / 'examples/configs/glm53-flash-exl3-k4-mc-min.config'))],
            aot.read_config,
        )
        assert wip_mcg['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == 'glmf:exl3-k34;glmf:fp8'
        assert wip_mcg['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glmf:exl3-k34;glmf:fp8'
        requested_paths = {path for _, path in requests}
        assert requested_paths == {f'/{glmf_repo}/resolve/{glmf_rev}/model.safetensors.index.json'}
        cached_snap = cold_hf / f'hub/models--{glmf_repo.replace("/", "--")}/snapshots/{glmf_rev}'
        assert [p.name for p in cached_snap.iterdir()] == ['model.safetensors.index.json']

        # Warm repeat through release-scope.py hits the cached index with zero new HTTP requests.
        requests.clear()
        matrix_path = tmp_path / 'matrix.json'
        matrix_path.write_text(json.dumps({
            'entries': [{
                'name': 'glmf-mc-min',
                'family': 'glm5_flash',
                'config': 'examples/configs/glm53-flash-exl3-k4-mc-min.config',
                'set': {},
            }],
        }))
        rel_mcg, _ = release_scope.derive(ROOT, matrix_path, 'glmf-mc-min', {'HF_HOME': str(cold_hf)})
        assert rel_mcg['CUTEAFD_RELEASE_EXPERT_FAMILIES'] == 'glmf:exl3-k34;glmf:fp8'
        assert requests == []

        # 2. Cold-cache release-scope.py on unpinned MUL1 (no local refs/main) resolves remote main SHA and caches ONLY index.
        cold_hf_mul1 = tmp_path / 'cold-hf-mul1'
        matrix_mul1 = tmp_path / 'matrix-mul1.json'
        matrix_mul1.write_text(json.dumps({
            'entries': [{
                'name': 'glmf-mul1',
                'family': 'glm5_flash',
                'config': 'examples/configs/glm53-flash-exl3-k4-mc-min.config',
                'set': {'MODEL_ID': mul1_repo, 'MODEL_REVISION': ''},
            }],
        }))
        rel_mul1, _ = release_scope.derive(ROOT, matrix_mul1, 'glmf-mul1', {'HF_HOME': str(cold_hf_mul1)})
        assert rel_mul1['CUTEAFD_RELEASE_EXPERT_FAMILIES'] == 'glmf:exl3-k34-mul1;glmf:fp8'
        assert {path for _, path in requests} == {
            f'/api/models/{mul1_repo}/revision/main',
            f'/{mul1_repo}/resolve/{mul1_rev}/model.safetensors.index.json',
        }
        mul1_snap = cold_hf_mul1 / f'hub/models--{mul1_repo.replace("/", "--")}/snapshots/{mul1_rev}'
        assert [p.name for p in mul1_snap.iterdir()] == ['model.safetensors.index.json']


def test_unresolvable_exact_index_fails_closed_without_config_fallback(tmp_path, monkeypatch):
    release_spec = importlib.util.spec_from_file_location('release_scope_refuse', ROOT / 'scripts/build/release-scope.py')
    release_scope = importlib.util.module_from_spec(release_spec)
    release_spec.loader.exec_module(release_scope)

    repo = 'org/Unresolvable-GLM-Flash-EXL3-K4'
    rev = 'b' * 40
    cfg_bytes = json.dumps({
        'model_type': 'glm5_next',
        'quantization_config': {'quant_method': 'exl3', 'codebook': 'mcg'},
    }).encode()
    routes = {
        f'/{repo}/resolve/{rev}/config.json': (200, {'Content-Type': 'application/json'}, cfg_bytes),
    }

    cold_hf = tmp_path / 'hf'
    snap = cold_hf / f'hub/models--{repo.replace("/", "--")}/snapshots/{rev}'
    snap.mkdir(parents=True)
    # Even with config.json locally cached AND served remotely declaring codebook='mcg' or 'mul1',
    # a 404 on exact-revision model.safetensors.index.json MUST fail closed (never infer MCG).
    for cfg_book in ('mcg', 'mul1'):
        (snap / 'config.json').write_text(json.dumps({
            'model_type': 'glm5_next',
            'num_hidden_layers': 2,
            'layer_types': ['linear_attention', 'deepseek_sparse_attention'],
            'quantization_config': {'quant_method': 'exl3', 'codebook': cfg_book},
        }))
        monkeypatch.setenv('HF_HOME', str(cold_hf))
        with _serve_mock_hub(monkeypatch, routes) as requests:
            with pytest.raises(ValueError, match='could not fetch exact checkpoint model.safetensors.index.json'):
                aot.family_scopes(
                    [dict(family='glm5_flash', name='card', config=None)],
                    lambda _: dict(MODEL_ID=repo, MODEL_REVISION=rev, SPARK_COUNT='2', RTX_GPUS='1'),
                )
            assert not any('config.json' in path for _, path in requests)

            matrix_path = tmp_path / 'bad-matrix.json'
            matrix_path.write_text(json.dumps({
                'entries': [{
                    'name': 'bad-glmf',
                    'family': 'glm5_flash',
                    'config': 'examples/configs/glm53-flash-exl3-k4-mc-min.config',
                    'set': {'MODEL_ID': repo, 'MODEL_REVISION': rev},
                }],
            }))
            with pytest.raises(ValueError, match='could not fetch exact checkpoint model.safetensors.index.json'):
                release_scope.derive(ROOT, matrix_path, 'bad-glmf', {'HF_HOME': str(cold_hf)})

    # Non-directory HF_HOME=/dev/null cannot hold a cache and fails closed without inferring MCG.
    monkeypatch.setenv('HF_HOME', '/dev/null')
    with pytest.raises(ValueError, match='not a writable cache directory'):
        aot.family_scopes(
            [dict(family='glm5_flash', name='card', config=str(ROOT / 'examples/configs/glm53-flash-exl3-k4-mc-min.config'))],
            aot.read_config,
        )


def test_stale_local_refs_main_mcg_wins_over_remote_main_mul1_and_fetches_only_old_index(tmp_path, monkeypatch):
    import hashlib

    release_spec = importlib.util.spec_from_file_location('release_scope_stale_ref', ROOT / 'scripts/build/release-scope.py')
    release_scope = importlib.util.module_from_spec(release_spec)
    release_spec.loader.exec_module(release_scope)

    repo = 'org/GLM-5.3-Flash-EXL3-K4-MovingMain'
    old_mcg_sha = '1' * 40
    new_mul1_sha = '2' * 40
    explicit_mul1_sha = '3' * 40

    mcg_index = json.dumps({'weight_map': {
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 'model-00001-of-00010.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mcg': 'model-00001-of-00010.safetensors',
    }}).encode()
    mul1_index = json.dumps({'weight_map': {
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 'model-00001-of-00010.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mul1': 'model-00001-of-00010.safetensors',
    }}).encode()

    routes = {
        f'/api/models/{repo}/revision/main': (
            200, {'Content-Type': 'application/json'}, json.dumps({'id': repo, 'sha': new_mul1_sha}).encode(),
        ),
        f'/{repo}/resolve/{old_mcg_sha}/model.safetensors.index.json': (
            200,
            {
                'Content-Type': 'application/json',
                'ETag': f'"{hashlib.sha256(mcg_index).hexdigest()}"',
                'X-Repo-Commit': old_mcg_sha,
            },
            mcg_index,
        ),
        f'/{repo}/resolve/{new_mul1_sha}/model.safetensors.index.json': (
            200,
            {
                'Content-Type': 'application/json',
                'ETag': f'"{hashlib.sha256(mul1_index).hexdigest()}"',
                'X-Repo-Commit': new_mul1_sha,
            },
            mul1_index,
        ),
        f'/{repo}/resolve/{explicit_mul1_sha}/model.safetensors.index.json': (
            200,
            {
                'Content-Type': 'application/json',
                'ETag': f'"{hashlib.sha256(mul1_index).hexdigest()}"',
                'X-Repo-Commit': explicit_mul1_sha,
            },
            mul1_index,
        ),
    }

    # Local cache has old refs/main (MCG SHA) WITHOUT local model.safetensors.index.json,
    # while remote main has advanced to new_mul1_sha (MUL1).
    hf_home = tmp_path / 'hf'
    model_root = hf_home / f'hub/models--{repo.replace("/", "--")}'
    (model_root / 'refs').mkdir(parents=True)
    (model_root / 'refs/main').write_text(old_mcg_sha + '\n')
    (model_root / 'snapshots' / old_mcg_sha).mkdir(parents=True)
    (model_root / 'snapshots' / old_mcg_sha / 'config.json').write_text(json.dumps({
        'model_type': 'glm5_next',
        'num_hidden_layers': 2,
        'layer_types': ['linear_attention', 'deepseek_sparse_attention'],
        'quantization_config': {'quant_method': 'exl3'},
    }))

    monkeypatch.setenv('HF_HOME', str(hf_home))
    monkeypatch.delenv('CUTEAFD_NAMESPACE_ROOT', raising=False)
    with _serve_mock_hub(monkeypatch, routes) as requests:
        family, derived = aot.config_scopes({'MODEL_ID': repo, 'SPARK_COUNT': '2', 'RTX_GPUS': '1'})
        assert family == 'glm5_flash'
        assert derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == 'glmf:exl3-k34;glmf:fp8'
        assert derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glmf:exl3-k34;glmf:fp8'
        assert {path for _, path in requests} == {f'/{repo}/resolve/{old_mcg_sha}/model.safetensors.index.json'}
        assert not (model_root / 'snapshots' / new_mul1_sha).exists()
        assert sorted(p.name for p in (model_root / 'snapshots' / old_mcg_sha).iterdir()) == [
            'config.json', 'model.safetensors.index.json',
        ]

        # Launcher revision resolver returns the exact same old_mcg_sha.
        launcher_rev = subprocess.check_output(
            ['bash', '-euo', 'pipefail', '-c',
             'source "$1"; release_resolve_model_revision "$2" "" "$3"',
             'test', str(ROOT / 'scripts/lib/release-common.sh'), repo, str(hf_home)],
            text=True, timeout=30,
        ).strip()
        assert launcher_rev == old_mcg_sha

        # Selected CUTEAFD_NAMESPACE_ROOT/hf-home local refs/main also wins over empty HF_HOME and remote main.
        requests.clear()
        ns_root = tmp_path / 'namespace'
        ns_model = ns_root / f'hf-home/hub/models--{repo.replace("/", "--")}'
        (ns_model / 'refs').mkdir(parents=True)
        (ns_model / 'refs/main').write_text(old_mcg_sha + '\n')
        empty_hf = tmp_path / 'empty-hf'
        empty_hf.mkdir()
        monkeypatch.setenv('HF_HOME', str(empty_hf))
        monkeypatch.setenv('CUTEAFD_NAMESPACE_ROOT', str(ns_root))
        _, ns_derived = aot.config_scopes({'FAMILY': 'glm5_flash', 'MODEL_ID': repo, 'SPARK_COUNT': '2', 'RTX_GPUS': '1'})
        assert ns_derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glmf:exl3-k34;glmf:fp8'
        assert {path for _, path in requests} == {f'/{repo}/resolve/{old_mcg_sha}/model.safetensors.index.json'}
        assert not (empty_hf / 'hub').exists()

        # Explicit MODEL_REVISION wins first over local refs/main and remote main.
        requests.clear()
        monkeypatch.delenv('CUTEAFD_NAMESPACE_ROOT', raising=False)
        monkeypatch.setenv('HF_HOME', str(hf_home))
        _, explicit_derived = aot.config_scopes({
            'FAMILY': 'glm5_flash', 'MODEL_ID': repo, 'MODEL_REVISION': explicit_mul1_sha,
            'SPARK_COUNT': '2', 'RTX_GPUS': '1',
        })
        assert explicit_derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glmf:exl3-k34-mul1;glmf:fp8'
        assert {path for _, path in requests} == {f'/{repo}/resolve/{explicit_mul1_sha}/model.safetensors.index.json'}

        # Malformed local refs/main fails closed without falling back to remote main.
        requests.clear()
        (model_root / 'refs/main').write_text('   \n')
        with pytest.raises(ValueError, match='invalid revision'):
            aot.config_scopes({'FAMILY': 'glm5_flash', 'MODEL_ID': repo, 'SPARK_COUNT': '2', 'RTX_GPUS': '1'})
        assert requests == []


def test_no_ref_revision_parity_between_scope_and_launcher(tmp_path, monkeypatch):
    import hashlib

    repo = 'org/GLM-5.3-Flash-EXL3-K4-NoRef'
    unreferenced_mcg_sha = 'a' * 40
    remote_mul1_sha = 'f' * 40
    mul1_index = json.dumps({'weight_map': {
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 'model-00001-of-00010.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mul1': 'model-00001-of-00010.safetensors',
    }}).encode()
    routes = {
        f'/api/models/{repo}/revision/main': (
            200, {'Content-Type': 'application/json'}, json.dumps({'id': repo, 'sha': remote_mul1_sha}).encode(),
        ),
        f'/{repo}/resolve/{remote_mul1_sha}/model.safetensors.index.json': (
            200,
            {
                'Content-Type': 'application/json',
                'ETag': f'"{hashlib.sha256(mul1_index).hexdigest()}"',
                'X-Repo-Commit': remote_mul1_sha,
            },
            mul1_index,
        ),
    }

    hf_home = tmp_path / 'hf'
    model_root = hf_home / f'hub/models--{repo.replace("/", "--")}'
    # Seed an unreferenced stale MCG snapshot directory WITHOUT refs/main: neither scope nor launcher may pick it.
    stale_snap = model_root / 'snapshots' / unreferenced_mcg_sha
    stale_snap.mkdir(parents=True)
    (stale_snap / 'config.json').write_text(json.dumps({
        'model_type': 'glm5_next',
        'num_hidden_layers': 2,
        'layer_types': ['linear_attention', 'deepseek_sparse_attention'],
        'quantization_config': {'quant_method': 'exl3', 'codebook': 'mcg'},
    }))
    (stale_snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'model.layers.1.mlp.experts.0.gate_proj.trellis': 'model-00001-of-00010.safetensors',
        'model.layers.1.mlp.experts.0.gate_proj.mcg': 'model-00001-of-00010.safetensors',
    }}))

    monkeypatch.setenv('HF_HOME', str(hf_home))
    monkeypatch.delenv('CUTEAFD_NAMESPACE_ROOT', raising=False)

    # 1. Offline with no refs/main: both scope and launcher fail closed (never guess from snapshots/ or config.json).
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    with pytest.raises(ValueError, match='HF_HUB_OFFLINE'):
        aot.config_scopes({'FAMILY': 'glm5_flash', 'MODEL_ID': repo, 'SPARK_COUNT': '2', 'RTX_GPUS': '1'})
    offline_launcher = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c',
         'source "$1"; RELEASE_MODEL_ID="$2"; RELEASE_MODEL_REVISION=""; AUDIO=off; release_resolve_local_model_revision "$3"',
         'test', str(ROOT / 'scripts/lib/release-common.sh'), repo, str(hf_home)],
        text=True, capture_output=True, timeout=30,
    )
    assert offline_launcher.returncode == 2
    assert 'has no local refs/main' in offline_launcher.stderr

    # 2. Online with no refs/main: both scope and launcher resolve the exact same remote main SHA.
    (model_root / 'snapshots' / remote_mul1_sha).mkdir(parents=True)
    (model_root / 'snapshots' / remote_mul1_sha / 'config.json').write_text(json.dumps({
        'model_type': 'glm5_next',
        'num_hidden_layers': 2,
        'layer_types': ['linear_attention', 'deepseek_sparse_attention'],
        'quantization_config': {'quant_method': 'exl3'},
    }))
    cfg_file = tmp_path / 'noref.config'
    cfg_file.write_text(f'MODEL_ID={repo}\n')
    with _serve_mock_hub(monkeypatch, routes) as requests:
        family, derived = aot.config_scopes({'MODEL_ID': repo, 'SPARK_COUNT': '2', 'RTX_GPUS': '1'})
        assert family == 'glm5_flash'
        assert derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glmf:exl3-k34-mul1;glmf:fp8'
        assert {path for _, path in requests} == {
            f'/api/models/{repo}/revision/main',
            f'/{repo}/resolve/{remote_mul1_sha}/model.safetensors.index.json',
        }

        launcher_probe = subprocess.check_output(
            ['bash', '-euo', 'pipefail', '-c',
             'source "$1"; '
             'fam="$(release_config_family "$2")"; '
             'RELEASE_MODEL_ID="$3"; RELEASE_MODEL_REVISION=""; AUDIO=off; '
             'release_resolve_local_model_revision "$4"; '
             'printf "%s %s\\n" "$fam" "$RELEASE_MODEL_REVISION"',
             'test', str(ROOT / 'scripts/lib/release-common.sh'), str(cfg_file), repo, str(hf_home)],
            text=True, timeout=30,
        ).strip()
        assert launcher_probe == f'glm5_flash {remote_mul1_sha}'


def _write_single_file_safetensors(path: Path, tensor_names: list[str], payload_poison: bytes = b'\xff\xfe\xfd\xfc' * 16) -> int:
    header = {
        '__metadata__': {'format': 'pt'},
        **{
            name: {'dtype': 'I32' if name.endswith(('.mcg', '.mul1')) else 'I16',
                   'shape': [] if name.endswith(('.mcg', '.mul1')) else [16, 16, 64],
                   'data_offsets': [0, 4]}
            for name in tensor_names
        },
    }
    raw_header = json.dumps(header, separators=(',', ':')).encode('utf-8')
    path.write_bytes(len(raw_header).to_bytes(8, 'little') + raw_header + payload_poison)
    return 8 + len(raw_header)


def test_single_file_exl3_snapshot_without_index_classifies_from_header_without_reading_payload(tmp_path, monkeypatch):
    import builtins

    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    real_open = builtins.open
    read_log: list[tuple[Path, int]] = []

    class _HeaderOnlySpy:
        def __init__(self, inner, path: Path, max_header_bytes: int):
            self._inner = inner
            self._path = path
            self._max_header_bytes = max_header_bytes
            self._bytes_read = 0

        def read(self, size: int = -1):
            if size < 0 or self._bytes_read + size > self._max_header_bytes:
                raise AssertionError(
                    f'attempted to read weight payload from {self._path} '
                    f'(read({size}) at offset {self._bytes_read}, header ends at {self._max_header_bytes})'
                )
            chunk = self._inner.read(size)
            self._bytes_read += len(chunk)
            return chunk

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            read_log.append((self._path, self._bytes_read))
            return self._inner.__exit__(exc_type, exc, tb)

    expected_header_end: dict[Path, int] = {}

    def guarded_open(file, mode='r', *args, **kwargs):
        path = Path(file)
        inner = real_open(file, mode, *args, **kwargs)
        if path in expected_header_end and 'b' in mode:
            return _HeaderOnlySpy(inner, path, expected_header_end[path])
        return inner

    monkeypatch.setattr(builtins, 'open', guarded_open)

    # 1. Direct local single-file snapshot directory (no index) for both MCG and MUL1.
    for marker, expected_scope in (('mcg', 'glmf:exl3-k34;glmf:fp8'), ('mul1', 'glmf:exl3-k34-mul1;glmf:fp8')):
        direct_snap = tmp_path / f'direct-single-{marker}'
        direct_snap.mkdir()
        (direct_snap / 'config.json').write_text(json.dumps({
            'quantization_config': {'quant_method': 'exl3'},
        }))
        st_path = direct_snap / 'model.safetensors'
        expected_header_end[st_path] = _write_single_file_safetensors(st_path, [
            'model.layers.0.self_attn.q_proj.trellis',
            'model.layers.0.self_attn.q_proj.mcg',
            'model.layers.1.mlp.experts.0.gate_proj.trellis',
            f'model.layers.1.mlp.experts.0.gate_proj.{marker}',
        ])
        read_log.clear()
        derived = aot.family_scopes(
            [dict(family='glm5_flash', name='single', config=None)],
            lambda _: {'MODEL_ID': str(direct_snap), 'SPARK_COUNT': '2', 'RTX_GPUS': '1'},
        )
        assert derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == expected_scope
        assert derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == expected_scope
        assert read_log == [(st_path, expected_header_end[st_path])]

    # 2. Cached HF snapshot via refs/main (no index, offline) reads only the safetensors header.
    hf_home = tmp_path / 'hf-single'
    monkeypatch.setenv('HF_HOME', str(hf_home))
    repo = 'org/GLM-5.3-Flash-Single-EXL3-MUL1'
    sha = 'c' * 40
    model_root = hf_home / f'hub/models--{repo.replace("/", "--")}'
    (model_root / 'refs').mkdir(parents=True)
    (model_root / 'refs/main').write_text(sha + '\n')
    cached_snap = model_root / 'snapshots' / sha
    cached_snap.mkdir(parents=True)
    (cached_snap / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
    cached_st = cached_snap / 'model.safetensors'
    expected_header_end[cached_st] = _write_single_file_safetensors(cached_st, [
        'model.layers.1.mlp.experts.0.gate_proj.trellis',
        'model.layers.1.mlp.experts.0.gate_proj.mul1',
    ])
    read_log.clear()
    cached_derived = aot.family_scopes(
        [dict(family='glm5_flash', name='cached-single', config=None)],
        lambda _: {'MODEL_ID': repo, 'SPARK_COUNT': '2', 'RTX_GPUS': '1'},
    )
    assert cached_derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glmf:exl3-k34-mul1;glmf:fp8'
    assert read_log == [(cached_st, expected_header_end[cached_st])]


def test_index_with_routed_marker_and_zero_trellis_entries_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    repo = 'test/Orphan-MUL1-No-Trellis-EXL3'
    snap = tmp_path / f'hub/models--{repo.replace("/", "--")}/snapshots/rev'
    snap.mkdir(parents=True)
    (snap / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))

    for marker in ('mul1', 'mcg'):
        (snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'model.embed_tokens.weight': 's0.safetensors',
            f'model.layers.1.mlp.experts.0.gate_proj.{marker}': 's1.safetensors',
            'model.layers.1.mlp.experts.0.gate_proj.suh': 's1.safetensors',
            'model.layers.1.mlp.experts.0.gate_proj.svh': 's1.safetensors',
        }}))
        with pytest.raises(ValueError, match='incomplete routed EXL3 codebook markers'):
            aot.family_scopes(
                [dict(family='glm5_flash', name='orphan', config=None)],
                lambda _: {'MODEL_ID': repo, 'MODEL_REVISION': 'rev', 'SPARK_COUNT': '2', 'RTX_GPUS': '1'},
            )

    # Same rejection for a single-file safetensors header with .mul1 and zero .trellis entries.
    single_dir = tmp_path / 'orphan-single'
    single_dir.mkdir()
    (single_dir / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
    _write_single_file_safetensors(single_dir / 'model.safetensors', [
        'model.embed_tokens.weight',
        'model.layers.1.mlp.experts.0.gate_proj.mul1',
    ])
    with pytest.raises(ValueError, match='incomplete routed EXL3 codebook markers'):
        aot.family_scopes(
            [dict(family='glm5_flash', name='orphan-single', config=None)],
            lambda _: {'MODEL_ID': str(single_dir), 'SPARK_COUNT': '2', 'RTX_GPUS': '1'},
        )


@pytest.mark.parametrize('extra,expected_fp8', [
    ({}, 'mimof:fp8'),
    ({'MODEL_VARIANT': 'pro'}, 'mimop:fp8'),
])
def test_mimo_routed_exl3_scopes_fail_early_as_unsupported(tmp_path, monkeypatch, extra, expected_fp8):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    model = 'test/MiMo-V2.6-EXL3-K4'
    snap = seed_exl3_index(tmp_path, model, revision='rev')
    (snap / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
    card = [dict(family='mimo_v2', name='mimo-exl3', config=None)]
    base = dict(MODEL_ID=model, MODEL_REVISION='rev', SPARK_COUNT='4', RTX_GPUS='1', **extra)

    with pytest.raises(ValueError, match='MiMo V2 .* routed EXL3 AOT scopes are not supported'):
        aot.family_scopes(card, lambda _: base)

    with pytest.raises(ValueError, match='MiMo V2 .* routed EXL3 AOT scopes are not supported'):
        aot.family_scopes(card, lambda _: dict(base, MODEL_ID='XiaomiMiMo/MiMo-V2.6-Flash-MOPD', EXPERT_FORMAT='exl3'))

    # Explicit EXPERT_FORMAT=fp8 still selects the supported MiMo FP8/MXFP4 scope.
    fp8_env = aot.family_scopes(card, lambda _: dict(base, EXPERT_FORMAT='fp8'))
    assert fp8_env['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == expected_fp8
    assert fp8_env['CUTEAFD_WIP_EXPERT_FAMILIES'] == expected_fp8



