#!/usr/bin/env python3
"""Shared WIP family scopes for cards and direct config builds."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

SPEC = importlib.util.spec_from_file_location('checkpoint_family', Path(__file__).with_name('checkpoint-family.py'))
checkpoint_family = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_family)
# The launcher's default checkpoint when a config names none (release-common.sh).
SPARK_SPEC = importlib.util.spec_from_file_location(
    'wip_spark_packages', Path(__file__).resolve().parents[2] / 'python/tools/lib/spark_packages.py')
spark_packages = importlib.util.module_from_spec(SPARK_SPEC)
SPARK_SPEC.loader.exec_module(spark_packages)
native_spark_artifacts = spark_packages.native_spark_artifacts

DEFAULT_MODEL_ID = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
# Official V4.1 releases, for builds whose checkpoint metadata is not local.
V41_RELEASES = ('deepseek-ai/DeepSeek-V4.1-Flash', 'deepseek-ai/DeepSeek-V4.1-Flash-1017')
PROGRAM_SWITCHES = dict(deepseek_v4='DSV4', glm5='GLM', glm5_flash='GLMF', mimo_v2='MIMO', qwen4='QWEN4',
                        qwen3_5='QWEN_DENSE', step5='STEP5', kimi_k3='KIMIK3', muse_spark='MUSE', mistral_large4='ML4')
FORMAT_ALIASES = dict(exl3='exl3', tr3='exl3', nvfp4='nvfp4', nvfp4a4='nvfp4',
                      fp8='fp8', mxfp4='fp8', bf16='fp8', native='fp8')
NATIVE_FAMILY_TOKENS = {'deepseek', 'glm', 'glmf', 'mimo', 'mimof', 'mimop', 'qwen', 'qwen4', 'step', 'step5'}
NATIVE_FP8_SLUGS = {
    'deepseek-v4-flash', 'deepseek-v4-flash-0731', 'deepseek-v4-pro',
    'deepseek-v4.1-flash', 'deepseek-v4.1-flash-1017',
    'glm-5.3', 'glm-5.3-flash',
    'mimo-v2.6-flash', 'mimo-v2.6-flash-mopd', 'mimo-v2.6-pro', 'mimo-v2.6-pro-mopd',
    'step-5-preview',
}


def require_mapping(family):
    if family != 'deepseek_v41' and family not in PROGRAM_SWITCHES:
        raise ValueError(f'no WIP AOT scope mapping for {family}')


def checkpoint_config(values, default_model=DEFAULT_MODEL_ID):
    model = values.get('MODEL_ID', values.get('MODEL', default_model))
    if not model:
        return None
    hub = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'
    root = hub / ('models--' + model.replace('/', '--'))
    revision = values.get('MODEL_REVISION', '')
    if not revision and (root / 'refs/main').is_file():
        revision = (root / 'refs/main').read_text().strip()
    path = root / 'snapshots' / revision / 'config.json'
    return json.loads(path.read_text()) if revision and path.is_file() else None


def config_quant_format(config):
    quant = config.get('quantization_config') or (config.get('text_config') or {}).get('quantization_config')
    if not isinstance(quant, dict) or not quant:
        return None
    for key in ('quant_method', 'method', 'format', 'checkpoint_format'):
        value = str(quant.get(key, '')).strip().lower()
        if value in ('exl3', 'tr3'):
            return 'exl3'
    method = str(quant.get('quant_method') or quant.get('method') or '').strip().lower()
    algo = str(quant.get('moe_quant_algo') or quant.get('quant_algo') or '').strip().lower()
    store = str(quant.get('store_dtype') or '').strip().lower()
    if method in ('nvfp4', 'modelopt') or algo == 'nvfp4':
        return 'nvfp4'
    if method in ('fp8', 'mxfp8', 'mxfp4', 'bf16', 'compressed-tensors') or algo in ('fp8', 'mxfp4') or store in ('fp8', 'mxfp4', 'bf16'):
        return 'fp8'
    raise ValueError(f'unsupported checkpoint quantization_config: {quant}')


def model_id_format(model):
    found = spark_packages.checkpoint_expert_format(model)
    if found is not None:
        return found
    slug = model.strip().rsplit('/', 1)[-1].lower()
    tokens = set(re.split(r'[^a-z0-9.]+', slug)) - {''}
    if 'exl3' in tokens or 'tr3' in tokens or 'exl3' in slug:
        return 'exl3'
    if 'nvfp4' in tokens or 'nvfp4' in slug:
        return 'nvfp4'
    if tokens & ({'fp8', 'mxfp4', 'bf16'} | NATIVE_FAMILY_TOKENS) or slug in NATIVE_FP8_SLUGS:
        return 'fp8'
    return None


def expert_format(family, values):
    declared = str(values.get('EXPERT_FORMAT', '')).strip().lower()
    if declared:
        if declared not in FORMAT_ALIASES:
            raise ValueError(f'unsupported EXPERT_FORMAT={values["EXPERT_FORMAT"]!r} for {family}')
        return FORMAT_ALIASES[declared]
    model = str(values.get('MODEL_ID') or values.get('MODEL') or '').strip()
    config = checkpoint_config(values, default_model='')
    if config is not None:
        quant_fmt = config_quant_format(config)
        if quant_fmt is not None:
            return quant_fmt
    if model:
        from_model = model_id_format(model)
        if from_model is not None:
            return from_model
    raise ValueError(f'cannot derive WIP AOT expert format for {family} ({model or "missing MODEL_ID"}): '
                     'checkpoint config.json is missing; download its metadata or set EXPERT_FORMAT')


def is_pro_variant(values):
    variant = str(values.get('MODEL_VARIANT', '')).strip().lower()
    if variant in ('pro', 'flash'):
        return variant == 'pro'
    model = str(values.get('MODEL_ID') or values.get('MODEL') or '').lower()
    return 'pro' in model


def dense_switch(values):
    config = checkpoint_config(values)
    if config is None:
        raise ValueError('dense Qwen WIP AOT scopes require checkpoint config.json')
    if checkpoint_family.describe(config)[0] != 'qwen3_5':
        raise ValueError('dense Qwen WIP AOT scopes require qwen3_5 checkpoint geometry')
    text = config.get('text_config', config)
    geometry = tuple(text.get(key) for key in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                                              'num_attention_heads', 'num_key_value_heads',
                                              'linear_num_key_heads', 'linear_num_value_heads'))
    switches = {(4096, 12288, 32, 16, 4, 16, 32): 'QWEN_DENSE',
                (5120, 17408, 64, 24, 4, 16, 48): 'QWEN_DENSE'}
    if geometry not in switches:
        raise ValueError(f'unsupported dense Qwen WIP AOT checkpoint geometry {geometry}')
    return switches[geometry]


def family_scopes(cards, config_reader):
    # Derived per card: a card with no Sparks (local experts) contributes only
    # coordinator scopes, and a Spark card's TP count is keyed to its own family
    # (FAMILY=COUNT, read by native/cmake/shared/{exl3,fp8_moe}.cmake), so one
    # card's geometry is never exported at another card's TP.
    scopes, spark_scopes, roles, counts, env = set(), set(), set(), set(), {}
    for card in cards:
        family = card.get('family')
        require_mapping(family)
        values = {**config_reader(card['config']), **card.get('set', {})}
        count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
        if family == 'qwen3_5':
            if count or values.get('SPARK_HOSTS') or values.get('EXPERT_BACKEND') == 'spark':
                raise ValueError('dense Qwen has no Spark layout or dense-FFN offboarding')
            env['CUTEAFD_WIP_' + dense_switch(values) + '_AOT'] = 'ON'
        scopes_before = set(scopes)
        scopes = set()
        if family == 'deepseek_v41' and count != 4 and count in native_spark_artifacts('v41'):
            roles.add('tp' + str(count))
        if family == 'deepseek_v4':
            tag = 'dsv4p' if is_pro_variant(values) else 'dsv4f'
            scopes.update({tag + ':rtx_backbone', tag + (f':spark_tp{count}' if count != 4 and count in native_spark_artifacts(tag) else ':spark')})
            if int(values.get('RTX_GPUS', len(card.get('gpus', [0])))) == 2:
                scopes.add(tag + ':rtx_tp2')
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            if tag == 'dsv4p' and expert_format(family, values) == 'exl3':
                scopes.add(tag + ':exl3-k23')
        if family == 'kimi_k3':
            tp = int(values.get('SPARK_TP', count))
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            scopes.add('kimik3:exl3-k23')
            if count:
                counts.add(f'kimik3={tp}')
        if family == 'muse_spark':
            # EXL3 K4 experts, intermediate-sliced on the Sparks (partial replies) and
            # whole or TP2 on onboarded RTX layers: one package on both roles.
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            scopes.add('muse:exl3-k45')
            if count:
                counts.add(f"muse={int(values.get('SPARK_TP', count))}")
        if family == 'mistral_large4':
            # The GLM engine's ml4 variant: its own coordinator programs and K6
            # core (ML4_AOT), EXL3 K4/K5 experts at the card's Spark TP.
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            scopes.add('ml4:exl3-k45')
            if count:
                counts.add(f"ml4={int(values.get('SPARK_TP', count))}")
        # EXLR8 checkpoints retain family coordinator programs without expert packages.
        exlr8 = bool(values.get('EXLR8_MANIFEST'))
        if exlr8:
            env['CUTEAFD_WIP_EXLR8_AOT'] = 'ON'
        for fam, tag, switch in [('glm5', 'glm', 'GLM'), ('glm5_flash', 'glmf', 'GLMF'), ('mimo_v2', 'mimop' if is_pro_variant(values) else 'mimof', 'MIMO'), ('qwen4', 'qwen4', 'QWEN4'), ('step5', 'step5', 'STEP5')]:
            if family == fam:
                env['CUTEAFD_WIP_' + switch + '_AOT'] = 'ON'
                if exlr8:
                    families = set(filter(None, env.get('CUTEAFD_WIP_EXLR8_FAMILIES', '').split(';')))
                    families.add(tag)
                    env['CUTEAFD_WIP_EXLR8_FAMILIES'] = ';'.join(sorted(families))
                    continue
                fmt = expert_format(family, values)
                if count and fam != 'step5':
                    counts.add(f"{tag}={int(values.get('SPARK_TP', count))}")
                # Qwen's FP8 package has no Spark TP layout (intermediate 640): request the NVFP4 family,
                # which CMake expands to nvfp4, nvfp4a4 and, on the coordinator only, the FP8 MTP sibling
                # (fp8_moe.cmake), as the release image ships them.
                nvfp4 = fmt == 'nvfp4'
                if fam == 'step5' and fmt == 'exl3':
                    # EXL3 K4 experts with K6 tails (layers 88-90) as one two-tier package;
                    # the K6 core and K4 MTP run the shared dense trellis programs.
                    scopes.add('step5:exl3-k46')
                    if count:
                        counts.add(f"step5:exl3={int(values.get('SPARK_TP', count))}")
                    card_scopes, scopes = scopes, scopes_before | scopes
                    if count:
                        spark_scopes |= card_scopes
                    continue
                scopes.add(tag + (':nvfp4' if tag == 'qwen4' or nvfp4 else ':fp8'))
                if fam == 'step5':
                    # Both official quants retain native BF16 routed layers.
                    # Format keys keep the min/max card TP layouts independent.
                    scopes.add('step5:bf16')
                    if count:
                        counts.update(f"{scope}={int(values.get('SPARK_TP', count))}" for scope in scopes)
                if fam == 'glm5_flash' and nvfp4:
                    scopes.add('glmfdense:nvfp4')
                if fmt == 'exl3':
                    scopes.add(tag + (':exl3-k34' if tag == 'glmf' else ':exl3-k45'))
                if fam == 'mimo_v2':
                    # MiMo places its audio tower unless the card turns audio off.
                    if values.get('AUDIO', 'auto') != 'off':
                        env['CUTEAFD_WIP_AUDIO_AOT'] = 'ON'
                    # Two RTX cards run the head-split programs (mimof2/mimop2).
                    wanted = {tag, tag + '2'} if int(values.get('RTX_GPUS', len(card.get('gpus', [0])))) == 2 else {tag}
                    env['CUTEAFD_WIP_MIMO_GEOMETRIES'] = ';'.join(sorted(set(env.get('CUTEAFD_WIP_MIMO_GEOMETRIES', '').split(';')) - {''} | wanted))
        card_scopes, scopes = scopes, scopes_before | scopes
        if count:
            spark_scopes |= {scope for scope in card_scopes if not scope.startswith('glmfdense:')}
    env.update(CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES=';'.join(sorted(scopes)),
               CUTEAFD_WIP_GENERIC_SPARK_COUNTS=';'.join(sorted(counts)),
               CUTEAFD_WIP_EXPERT_FAMILIES=';'.join(sorted(spark_scopes)),
               CUTEAFD_WIP_SPARK_TP_ROLES=';'.join(sorted(roles)))
    return env


def read_config(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('\"\'')
    return values


def config_family(values):
    declared = values.get('FAMILY', '')
    if declared:
        require_mapping(declared)
    model = values.get('MODEL_ID', DEFAULT_MODEL_ID)
    config = checkpoint_config(values)
    if config is not None:
        detected = checkpoint_family.describe(config)[0]
        require_mapping(detected)
        if declared and declared != detected:
            raise ValueError(f'FAMILY={declared} does not match checkpoint family {detected}')
        return detected
    if declared:
        return declared
    if model in V41_RELEASES:
        return 'deepseek_v41'
    raise ValueError(f'cannot derive WIP AOT family for {model}: checkpoint config.json is missing; '
                     'download its metadata or set FAMILY in the build config')


def config_scopes(values):
    family = config_family(values)
    resolved = dict(values)
    # Auto exports both coordinator shapes without probing a GPU. Explicit 1/2
    # stays narrow, as on cards; the launcher chooses placement at run time.
    if resolved.get('RTX_GPUS') == 'auto':
        resolved['RTX_GPUS'] = '2'
    if not resolved.get('SPARK_TP'):
        resolved.pop('SPARK_TP', None)
    if 'SPARK_COUNT' not in resolved:
        resolved['SPARK_COUNT'] = str(sum(bool(values.get(f'SPARK_{i}_HOST')) for i in range(8)))
    card = dict(family=family, name='', config=None)
    scopes = family_scopes([card], lambda _: resolved)
    # Direct configs name their Spark TP explicitly (TP2xEP3 on six Sparks is
    # the tp2 role, not tp6); cards keep the Spark-count role.
    spark_tp = resolved.get('SPARK_TP', '')
    if family == 'deepseek_v41' and scopes['CUTEAFD_WIP_SPARK_TP_ROLES'] and spark_tp != '4' and spark_tp in {str(tp) for tp in native_spark_artifacts('v41')}:
        scopes['CUTEAFD_WIP_SPARK_TP_ROLES'] = 'tp' + spark_tp
    if family in ('glm5', 'glm5_flash'):
        scopes['CUTEAFD_WIP_CONTEXT_SPLIT_AOT'] = 'ON'
    return family, scopes


def check_programs(family, path):
    require_mapping(family)
    switch = PROGRAM_SWITCHES.get(family)
    if switch is None:
        return  # V4.1 uses separate native manifests, not PROGRAMS.json.
    try:
        programs = json.loads(Path(path).read_text()).get('programs')
    except (OSError, ValueError, AttributeError):
        programs = None
    if not isinstance(programs, list) or not programs:
        raise ValueError(f'cannot seal coordinator slot for {family}: {path} has no programs; '
                         f'enable CUTEAFD_WIP_{switch}_AOT=ON (or remove its OFF override) and rebuild')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--family', choices=sorted(set(checkpoint_family.FAMILIES.values())))
    parser.add_argument('--check-programs', type=Path)
    args = parser.parse_args()
    try:
        if args.check_programs:
            if not args.family:
                parser.error('--check-programs requires --family')
            check_programs(args.family, args.check_programs)
        else:
            if not args.config:
                parser.error('--config is required')
            values = read_config(args.config)
            values.update(setting.split('=', 1) for setting in args.set)
            family, scopes = config_scopes(values)
            print('CUTEAFD_WIP_PROGRAM_FAMILY=' + family)
            for key, value in sorted(scopes.items()):
                print(key + '=' + value)
    except (OSError, ValueError) as error:
        parser.exit(2, f'WIP AOT: {error}\n')


if __name__ == '__main__':
    main()
