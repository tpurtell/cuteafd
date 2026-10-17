#!/usr/bin/env python3
"""Shared WIP family scopes for cards and direct config builds."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

SPEC = importlib.util.spec_from_file_location('checkpoint_family', Path(__file__).with_name('checkpoint-family.py'))
checkpoint_family = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_family)
# The launcher's default checkpoint when a config names none (release-common.sh).
DEFAULT_MODEL_ID = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
# Official V4.1 releases, for builds whose checkpoint metadata is not local.
V41_RELEASES = ('deepseek-ai/DeepSeek-V4.1-Flash', 'deepseek-ai/DeepSeek-V4.1-Flash-1017')
PROGRAM_SWITCHES = dict(deepseek_v4='DSV4', glm5='GLM', glm5_flash='GLMF', mimo_v2='MIMO', qwen4='QWEN4',
                        qwen3_5='QWEN_DENSE')


def require_mapping(family):
    if family != 'deepseek_v41' and family not in PROGRAM_SWITCHES:
        raise ValueError(f'no WIP AOT scope mapping for {family}')


def checkpoint_config(values):
    model = values.get('MODEL_ID', DEFAULT_MODEL_ID)
    hub = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'
    root = hub / ('models--' + model.replace('/', '--'))
    revision = values.get('MODEL_REVISION', '')
    if not revision and (root / 'refs/main').is_file():
        revision = (root / 'refs/main').read_text().strip()
    path = root / 'snapshots' / revision / 'config.json'
    return json.loads(path.read_text()) if revision and path.is_file() else None


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
        if family == 'deepseek_v41' and count in (2, 3, 6):
            roles.add('tp' + str(count))
        if family == 'deepseek_v4':
            tag = 'dsv4p' if 'pro' in card['name'] else 'dsv4f'
            scopes.update({tag + ':rtx_backbone', tag + (':spark_tp2' if count == 2 else ':spark')})
            if int(values.get('RTX_GPUS', len(card.get('gpus', [0])))) == 2:
                scopes.add(tag + ':rtx_tp2')
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            if 'exl3' in card['name']:
                scopes.add(tag + ':exl3-k23')
        for fam, tag, switch in [('glm5', 'glm', 'GLM'), ('glm5_flash', 'glmf', 'GLMF'), ('mimo_v2', 'mimop' if 'pro' in card['name'] else 'mimof', 'MIMO'), ('qwen4', 'qwen4', 'QWEN4')]:
            if family == fam:
                env['CUTEAFD_WIP_' + switch + '_AOT'] = 'ON'
                if count:
                    counts.add(f"{tag}={int(values.get('SPARK_TP', count))}")
                # Qwen's FP8 package has no Spark TP layout (intermediate 640): request the NVFP4 family,
                # which CMake expands to nvfp4, nvfp4a4 and, on the coordinator only, the FP8 MTP sibling
                # (fp8_moe.cmake), as the release image ships them.
                checkpoint_id = values.get('MODEL_ID', values.get('MODEL', ''))
                nvfp4 = any('nvfp4' in str(value).lower() for value in
                            (card['name'], checkpoint_id, values.get('EXPERT_FORMAT', '')))
                scopes.add(tag + (':nvfp4' if tag == 'qwen4' or nvfp4 else ':fp8'))
                if fam == 'glm5_flash' and nvfp4:
                    scopes.add('glmfdense:nvfp4')
                if 'exl3' in card['name'] or 'tr3' in card['name']:
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
    hint = ' '.join(str(values.get(key, '')) for key in ('MODEL_ID', 'MODEL_VARIANT', 'EXPERT_FORMAT')).lower()
    card = dict(family=family, name=hint, config=None)
    scopes = family_scopes([card], lambda _: resolved)
    # Direct configs name their Spark TP explicitly (TP2xEP3 on six Sparks is
    # the tp2 role, not tp6); cards keep the Spark-count role.
    spark_tp = resolved.get('SPARK_TP', '')
    if family == 'deepseek_v41' and scopes['CUTEAFD_WIP_SPARK_TP_ROLES'] and spark_tp in ('2', '3', '6'):
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
