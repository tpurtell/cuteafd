#!/usr/bin/env python3
"""Shared WIP family scopes for cards and direct config builds."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SPEC = importlib.util.spec_from_file_location('checkpoint_family', Path(__file__).with_name('checkpoint-family.py'))
checkpoint_family = importlib.util.module_from_spec(SPEC)
if SPEC is not None and SPEC.loader is not None and Path(SPEC.origin).is_file():
    SPEC.loader.exec_module(checkpoint_family)
# The launcher's default checkpoint when a config names none (release-common.sh).
_SPARK_PACKAGES_PATH = Path(__file__).resolve().parents[2] / 'python/tools/lib/spark_packages.py'
if _SPARK_PACKAGES_PATH.is_file():
    SPARK_SPEC = importlib.util.spec_from_file_location('wip_spark_packages', _SPARK_PACKAGES_PATH)
    spark_packages = importlib.util.module_from_spec(SPARK_SPEC)
    SPARK_SPEC.loader.exec_module(spark_packages)
    native_spark_artifacts = spark_packages.native_spark_artifacts
else:
    spark_packages = None
    native_spark_artifacts = None

DEFAULT_MODEL_ID = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
# Official V4.1 releases, for builds whose checkpoint metadata is not local.
V41_RELEASES = ('deepseek-ai/DeepSeek-V4.1-Flash', 'deepseek-ai/DeepSeek-V4.1-Flash-1017')
PROGRAM_SWITCHES = dict(deepseek_v4='DSV4', glm5='GLM', glm5_flash='GLMF', mimo_v2='MIMO', qwen4='QWEN4',
                        qwen3_5='QWEN_DENSE', step5='STEP5', kimi_k3='KIMIK3', muse_spark='MUSE', mistral_large4='ML4', minimax_m3='MINIMAX',
                        inkling2='INKLING2')
FORMAT_ALIASES = dict(exl3='exl3', tr3='exl3', nvfp4='nvfp4', nvfp4a4='nvfp4',
                      fp8='fp8', mxfp4='fp8', bf16='fp8', native='fp8')
# Families whose bundled vision tower runs the shared BF16 attention exports
# (native/shared/cuda/vision_runtime.cu shared-runtime towers, including PE and
# independent still towers). MiMo's ABI-1 tower and V4.1's own tower need none.
VISION_ATTENTION_FAMILIES = {'qwen4', 'glm5_flash', 'muse_spark', 'mistral_large4',
                             'minimax_m3', 'kimi_k3', 'inkling2', 'step5'}
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


_REPO_ID_RE = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z')
_HEX_REV_RE = re.compile(r'[0-9a-f]{40,64}\Z')
_HUB_REMOTE_MAIN_CODE = """
import os, re, sys
from huggingface_hub import HfApi
model = sys.argv[1]
endpoint = os.environ.get("HF_ENDPOINT") or None
info = HfApi(endpoint=endpoint).model_info(model, revision="main")
exact = str(getattr(info, "sha", "") or "").strip()
if not re.fullmatch(r"[0-9a-f]{40,64}", exact):
    raise SystemExit(f"unresolvable commit SHA for {model}@main")
print(exact)
"""
_HUB_INDEX_DOWNLOAD_CODE = """
import os, pathlib, sys
from huggingface_hub import hf_hub_download
model, exact, hub = sys.argv[1], sys.argv[2], sys.argv[3]
endpoint = os.environ.get("HF_ENDPOINT") or None
print(pathlib.Path(hf_hub_download(
    repo_id=model, filename="model.safetensors.index.json",
    repo_type="model", revision=exact, cache_dir=hub,
    local_files_only=False, endpoint=endpoint,
)))
"""


def _is_hf_offline() -> bool:
    return str(os.environ.get('HF_HUB_OFFLINE', '')).strip().lower() in ('1', 'true', 'yes', 'on')


def resolve_hf_home(explicit_hf_home: str | Path | None = None) -> Path:
    if explicit_hf_home is not None and str(explicit_hf_home).strip():
        return Path(str(explicit_hf_home).strip()).expanduser()
    ns_raw = os.environ.get('CUTEAFD_NAMESPACE_ROOT')
    if ns_raw is not None and ns_raw.strip():
        ns_hf = Path(ns_raw.strip()).expanduser().resolve() / 'hf-home'
        if ns_hf.is_dir():
            return ns_hf
    hf_raw = os.environ.get('HF_HOME')
    if hf_raw is not None and hf_raw.strip():
        return Path(hf_raw.strip()).expanduser()
    default_ns = Path(ns_raw.strip() if ns_raw is not None and ns_raw.strip() else '/mnt/sparknest').expanduser().resolve() / 'hf-home'
    if default_ns.is_dir():
        return default_ns
    return (Path.home() / '.cache/huggingface').expanduser()


def _validate_revision_token(revision: str, source: str) -> str:
    rev = revision.strip()
    if (not rev or '/' in rev or '\\' in rev or rev in ('.', '..')
            or Path(rev).parts != (rev,)):
        raise ValueError(f'{source} contains invalid revision {revision!r}')
    return rev


def _hub_helper_python() -> Path | None:
    repo = Path(__file__).resolve().parents[2]
    candidates = []
    if os.environ.get('CUTEAFD_TEST_PYTHON'):
        candidates.append(Path(os.environ['CUTEAFD_TEST_PYTHON']))
    candidates.append(repo / '.venv/bin/python')
    gate_venv = Path(os.environ.get('CUTEAFD_GATE_VENV', Path.home() / '.cache/cuteafd/gate-venv'))
    candidates.append(gate_venv / 'bin/python')
    git_common = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        text=True, capture_output=True,
    )
    if git_common.returncode == 0 and git_common.stdout.strip():
        candidates.append(Path(git_common.stdout.strip()).parent / '.venv/bin/python')
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_remote_main_sha(model: str) -> str:
    if not _REPO_ID_RE.fullmatch(model) or '..' in model:
        raise ValueError(f'invalid Hugging Face model id {model!r}')
    if _is_hf_offline():
        raise ValueError(f'no local refs/main for {model} and HF_HUB_OFFLINE is enabled')
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as import_err:
        helper = _hub_helper_python()
        if helper is None:
            raise ValueError(f'huggingface_hub is unavailable in {sys.executable}') from import_err
        proc = subprocess.run(
            [str(helper), '-c', _HUB_REMOTE_MAIN_CODE, model],
            text=True, capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ['hub main resolution failed']
            raise ValueError(detail[0])
        return proc.stdout.strip()
    try:
        endpoint = os.environ.get('HF_ENDPOINT') or None
        info = HfApi(endpoint=endpoint).model_info(model, revision='main')
        exact = str(getattr(info, 'sha', '') or '').strip()
        if not _HEX_REV_RE.fullmatch(exact):
            raise ValueError(f'could not resolve exact 40..64-hex commit revision for {model}@main')
        return exact
    except Exception as error:
        raise ValueError(str(error)) from error


def resolve_model_revision(
    model: str,
    explicit_revision: str = '',
    *,
    hf_home: str | Path | None = None,
    allow_remote: bool = True,
) -> tuple[str, str]:
    """Shared launcher/scope revision resolver: explicit MODEL_REVISION -> local refs/main -> remote main."""
    explicit = str(explicit_revision or '').strip()
    if explicit:
        return _validate_revision_token(explicit, 'MODEL_REVISION'), 'explicit'
    resolved_hf = resolve_hf_home(hf_home)
    if resolved_hf.exists() and not resolved_hf.is_dir():
        raise ValueError(f'HF_HOME={resolved_hf} is not a writable cache directory')
    root = resolved_hf / 'hub' / ('models--' + model.replace('/', '--'))
    ref_path = root / 'refs/main'
    if os.path.lexists(ref_path):
        if os.path.islink(ref_path) and not ref_path.exists():
            raise ValueError(f'Hugging Face main ref {ref_path} is a dangling symlink')
        if not ref_path.is_file():
            raise ValueError(f'Hugging Face main ref {ref_path} is not a regular file')
        return _validate_revision_token(ref_path.read_text(), f'Hugging Face main ref {ref_path}'), 'local_ref'
    if not allow_remote:
        raise ValueError(f'{model} has no local refs/main under {root}')
    return _resolve_remote_main_sha(model), 'remote_main'


def checkpoint_snapshot(values, default_model=DEFAULT_MODEL_ID):
    model = values.get('MODEL_ID', values.get('MODEL', default_model))
    if not model:
        return None
    direct = Path(model).expanduser()
    if direct.is_dir():
        return direct
    hf_home = resolve_hf_home()
    if hf_home.exists() and not hf_home.is_dir():
        return None
    root = hf_home / 'hub' / ('models--' + model.replace('/', '--'))
    try:
        revision, _ = resolve_model_revision(
            model, values.get('MODEL_REVISION', ''), hf_home=hf_home, allow_remote=not _is_hf_offline(),
        )
    except (OSError, ValueError):
        return None
    path = root / 'snapshots' / revision
    return path if revision and path.is_dir() else None


def checkpoint_config(values, default_model=DEFAULT_MODEL_ID):
    snapshot = checkpoint_snapshot(values, default_model=default_model)
    path = snapshot / 'config.json' if snapshot is not None else None
    return json.loads(path.read_text()) if path is not None and path.is_file() else None


def is_routed_tensor(name):
    if '.experts.' not in name:
        return False
    expert, sep, rest = name.split('.experts.', 1)[1].partition('.')
    return bool(sep and rest) and expert.isdigit()


def _download_checkpoint_index_at_sha(model: str, exact_sha: str, hub: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as import_err:
        helper = _hub_helper_python()
        if helper is None:
            raise ValueError(f'huggingface_hub is unavailable in {sys.executable}') from import_err
        proc = subprocess.run(
            [str(helper), '-c', _HUB_INDEX_DOWNLOAD_CODE, model, exact_sha, str(hub)],
            text=True, capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ['hub index download failed']
            raise ValueError(detail[0])
        return Path(proc.stdout.strip())

    try:
        endpoint = os.environ.get('HF_ENDPOINT') or None
        return Path(hf_hub_download(
            repo_id=model,
            filename='model.safetensors.index.json',
            repo_type='model',
            revision=exact_sha,
            cache_dir=str(hub),
            local_files_only=False,
            endpoint=endpoint,
        ))
    except Exception as error:
        raise ValueError(str(error)) from error


_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _read_safetensors_header_names(path: Path) -> list[str]:
    try:
        with open(path, 'rb') as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f'truncated safetensors header prefix in {path}')
            header_len = int.from_bytes(prefix, 'little')
            if header_len <= 0 or header_len > _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f'invalid safetensors header length {header_len} in {path}')
            raw_header = handle.read(header_len)
            if len(raw_header) != header_len:
                raise ValueError(f'truncated safetensors header body in {path}')
        header = json.loads(raw_header.decode('utf-8'))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f'invalid model.safetensors header in {path}: {error}') from error
    if not isinstance(header, dict):
        raise ValueError(f'safetensors header in {path} is not a JSON object')
    names = []
    for key, meta in header.items():
        if key == '__metadata__':
            continue
        if not isinstance(key, str) or not isinstance(meta, dict):
            raise ValueError(f'invalid safetensors tensor header {key!r} in {path}')
        names.append(key)
    if not names:
        raise ValueError(f'safetensors header in {path} has no tensors')
    return names


def _single_file_safetensors_in_snapshot(snapshot: Path) -> Path | None:
    single = snapshot / 'model.safetensors'
    if not single.is_file():
        return None
    if any(snapshot.glob('model-*-of-*.safetensors')):
        return None
    return single


def resolve_checkpoint_index(family, values) -> Path:
    model = str(values.get('MODEL_ID') or values.get('MODEL') or '').strip()
    if not model:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} (missing MODEL_ID): '
                         'checkpoint model.safetensors.index.json is missing')
    direct = Path(model).expanduser()
    if direct.is_dir():
        index_path = direct / 'model.safetensors.index.json'
        if not index_path.is_file():
            raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                             'checkpoint model.safetensors.index.json is missing')
        return index_path
    hf_home = resolve_hf_home()
    if hf_home.exists() and not hf_home.is_dir():
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         f'HF_HOME={hf_home} is not a writable cache directory')
    try:
        revision, rev_source = resolve_model_revision(
            model, values.get('MODEL_REVISION', ''), hf_home=hf_home, allow_remote=True,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         f'checkpoint model.safetensors.index.json is missing: {error}') from error
    hub = hf_home / 'hub'
    root = hub / ('models--' + model.replace('/', '--'))
    local_index = root / 'snapshots' / revision / 'model.safetensors.index.json'
    if local_index.is_file():
        return local_index
    if not _REPO_ID_RE.fullmatch(model) or '..' in model:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         'checkpoint model.safetensors.index.json is missing')
    if not _HEX_REV_RE.fullmatch(revision):
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         f'exact 40..64-hex commit revision required to fetch missing index, '
                         f'got {revision!r} from {rev_source}')
    if _is_hf_offline():
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         'checkpoint model.safetensors.index.json is missing in offline cache')
    try:
        fetched = _download_checkpoint_index_at_sha(model, revision, hub)
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         f'could not fetch exact checkpoint model.safetensors.index.json: {error}') from error
    if not fetched.is_file():
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         'checkpoint model.safetensors.index.json is missing')
    return fetched


def _read_index_tensor_names(family, model, index_path: Path) -> list[str]:
    try:
        weight_map = json.loads(index_path.read_text()).get('weight_map')
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         f'invalid model.safetensors.index.json: {error}') from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         'checkpoint model.safetensors.index.json has no weight_map')
    return [name for name in weight_map if isinstance(name, str)]


def resolve_checkpoint_tensor_names(family, values) -> tuple[Path, list[str]]:
    model = str(values.get('MODEL_ID') or values.get('MODEL') or '').strip()
    if not model:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} (missing MODEL_ID): '
                         'checkpoint model.safetensors.index.json is missing')
    direct = Path(model).expanduser()
    if direct.is_dir():
        index_path = direct / 'model.safetensors.index.json'
        if index_path.is_file():
            return index_path, _read_index_tensor_names(family, model, index_path)
        single = _single_file_safetensors_in_snapshot(direct)
        if single is not None:
            try:
                return single, _read_safetensors_header_names(single)
            except ValueError as error:
                raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): {error}') from error
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         'checkpoint model.safetensors.index.json is missing')
    hf_home = resolve_hf_home()
    if hf_home.exists() and not hf_home.is_dir():
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         f'HF_HOME={hf_home} is not a writable cache directory')
    try:
        revision, _ = resolve_model_revision(
            model, values.get('MODEL_REVISION', ''), hf_home=hf_home, allow_remote=False,
        )
    except (OSError, ValueError):
        revision = ''
    if revision:
        snap = hf_home / 'hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
        local_index = snap / 'model.safetensors.index.json'
        if local_index.is_file():
            return local_index, _read_index_tensor_names(family, model, local_index)
        single = _single_file_safetensors_in_snapshot(snap)
        if single is not None:
            try:
                return single, _read_safetensors_header_names(single)
            except ValueError as error:
                raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): {error}') from error
    index_path = resolve_checkpoint_index(family, values)
    return index_path, _read_index_tensor_names(family, model, index_path)


def exl3_codebook_suffix(family, values):
    if 'EXPERT_CODEBOOK' in values:
        raise ValueError(f'EXPERT_CODEBOOK override is not supported for {family}; '
                         'EXL3 codebook is derived from checkpoint tensor inventory')
    model = str(values.get('MODEL_ID') or values.get('MODEL') or '').strip()
    source_path, tensor_names = resolve_checkpoint_tensor_names(family, values)
    trellis_stems, mcg_stems, mul1_stems, unknown_markers = set(), set(), set(), set()
    for name in tensor_names:
        if not isinstance(name, str) or not is_routed_tensor(name):
            continue
        stem, sep, suffix = name.rpartition('.')
        if not sep:
            continue
        if suffix == 'trellis':
            trellis_stems.add(stem)
        elif suffix == 'mcg':
            mcg_stems.add(stem)
        elif suffix == 'mul1':
            mul1_stems.add(stem)
        elif suffix == '3inst':
            unknown_markers.add(name)
    if unknown_markers:
        raise ValueError(f'unsupported routed EXL3 codebook marker in {source_path}: {sorted(unknown_markers)[0]}')
    if mcg_stems and mul1_stems:
        raise ValueError(f'mixed routed EXL3 codebooks (.mcg and .mul1) in {source_path}')
    if not mcg_stems and not mul1_stems:
        raise ValueError(f'cannot derive WIP AOT EXL3 codebook for {family} ({model}): '
                         f'{source_path} has no routed .mcg or .mul1 markers')
    if not trellis_stems or trellis_stems != (mcg_stems | mul1_stems):
        raise ValueError(f'incomplete routed EXL3 codebook markers in {source_path}')
    return '-mul1' if mul1_stems else ''


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


def exl3_spark_layouts(scope, tp):
    if not 1 <= tp <= 8:
        raise ValueError(f'invalid Spark TP {tp}')
    family, _ = scope.split(':', 1)
    intermediate = spark_packages.package_entry(family, 'exl3')['intermediate']
    compact = '-compact' if intermediate // 128 % tp else ''
    return [f'tp{tp}{compact}-rank{rank}/m1' for rank in range(tp)]


def family_scopes(cards, config_reader):
    # Derived per card: a card with no Sparks (local experts) contributes only
    # coordinator scopes, and a Spark card's TP count is keyed to its own family
    # (FAMILY=COUNT, read by native/cmake/shared/{exl3,fp8_moe}.cmake), so one
    # card's geometry is never exported at another card's TP.
    scopes, spark_scopes, roles, counts, env = set(), set(), set(), set(), {}
    for card in cards:
        family = card.get('family')
        require_mapping(family)
        values = card['resolved_values'] if 'resolved_values' in card else {**config_reader(card['config']), **card.get('set', {})}
        count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
        if family == 'qwen3_5':
            if count or values.get('SPARK_HOSTS') or values.get('EXPERT_BACKEND') == 'spark':
                raise ValueError('dense Qwen has no Spark layout or dense-FFN offboarding')
            env['CUTEAFD_WIP_' + dense_switch(values) + '_AOT'] = 'ON'
        scopes_before = set(scopes)
        scopes = set()
        if family == 'deepseek_v41':
            # The V4.1 coordinator and expert exports (native, NVFP4, FP8, attention).
            env['CUTEAFD_WIP_V41_AOT'] = 'ON'
            if count != 4 and count in native_spark_artifacts('v41'):
                roles.add('tp' + str(count))
        if family in VISION_ATTENTION_FAMILIES and values.get('VISION', 'auto') != 'off':
            env['CUTEAFD_WIP_VISION_ATTENTION_AOT'] = 'ON'
        if family == 'deepseek_v4':
            tag = 'dsv4p' if is_pro_variant(values) else 'dsv4f'
            scopes.update({tag + ':rtx_backbone', tag + (f':spark_tp{count}' if count != 4 and count in native_spark_artifacts(tag) else ':spark')})
            if int(values.get('RTX_GPUS', len(card.get('gpus', [0])))) == 2:
                scopes.add(tag + ':rtx_tp2')
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            if tag == 'dsv4p' and expert_format(family, values) == 'exl3':
                scopes.add(tag + ':exl3-k23' + exl3_codebook_suffix(family, values))
        if family == 'kimi_k3':
            tp = count
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            scopes.add('kimik3:exl3-k23' + exl3_codebook_suffix(family, values))
            if count:
                counts.add(f'kimik3={tp}')
        if family == 'muse_spark':
            # EXL3 K4 experts, intermediate-sliced on the Sparks (partial replies) and
            # whole or TP2 on onboarded RTX layers: one package on both roles.
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            scopes.add('muse:exl3-k45' + exl3_codebook_suffix(family, values))
            if values.get('AUDIO', 'auto') != 'off':
                env['CUTEAFD_WIP_AUDIO_AOT'] = 'ON'
            if count:
                counts.add(f"muse={count}")
        if family == 'minimax_m3':
            env['CUTEAFD_WIP_MINIMAX_AOT'] = 'ON'
            scopes.add('minimax:nvfp4a4')
            if count:
                counts.add(f'minimax={count}')
        if family == 'inkling2':
            # Biased-sink GQA attention programs, and the MXFP8 W8A8 expert package every
            # non-EXL3 checkpoint needs (all layers of the MXFP8 release, 40-47 of the
            # official one, whose layers 0-39 are native dynamic-amax NVFP4 W4A4).
            env['CUTEAFD_WIP_INKLING2_AOT'] = 'ON'
            model = str(values.get('MODEL_ID') or values.get('MODEL') or '')
            if model_id_format(model) == 'exl3' or config_quant_format(checkpoint_config(values, default_model='') or {}) == 'exl3':
                scopes.add('inkling2:exl3-k45' + exl3_codebook_suffix(family, values))
            else:
                scopes.add('inkling2:mxfp8')
                if 'mxfp8' not in model.lower():
                    scopes.add('inkling2:nvfp4a4')
            if count:
                counts.add(f'inkling2={count}')
        if family == 'mistral_large4':
            # The GLM engine's ml4 variant: its own coordinator programs and K6
            # core (ML4_AOT), EXL3 K4/K5 experts at the card's Spark TP.
            env['CUTEAFD_WIP_' + PROGRAM_SWITCHES[family] + '_AOT'] = 'ON'
            scopes.add('ml4:exl3-k45' + exl3_codebook_suffix(family, values))
            if count:
                counts.add(f"ml4={count}")
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
                if fam == 'mimo_v2' and fmt == 'exl3':
                    raise ValueError(f'MiMo V2 ({tag}) routed EXL3 AOT scopes are not supported')
                if count and fam != 'step5':
                    counts.add(f"{tag}={count}")
                # Qwen's FP8 package has no Spark TP layout (intermediate 640): request the NVFP4 family,
                # which CMake expands to nvfp4, nvfp4a4 and, on the coordinator only, the FP8 MTP sibling
                # (fp8_moe.cmake), as the release image ships them.
                nvfp4 = fmt == 'nvfp4'
                if fam == 'step5' and fmt == 'exl3':
                    # EXL3 K4 experts with K6 tails (layers 88-90) as one two-tier package;
                    # the K6 core and K4 MTP run the shared dense trellis programs.
                    scopes.add('step5:exl3-k46' + exl3_codebook_suffix(family, values))
                    if count:
                        counts.add(f"step5:exl3={count}")
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
                        counts.update(f"{scope}={count}" for scope in scopes)
                if fam == 'glm5_flash' and nvfp4:
                    scopes.add('glmfdense:nvfp4')
                if fmt == 'exl3':
                    scopes.add(tag + (':exl3-k34' if tag == 'glmf' else ':exl3-k45') + exl3_codebook_suffix(family, values))
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
    if 'SPARK_COUNT' not in resolved:
        resolved['SPARK_COUNT'] = str(sum(bool(values.get(f'SPARK_{i}_HOST')) for i in range(8)))
    card = dict(family=family, name='', config=None)
    scopes = family_scopes([card], lambda _: resolved)
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
    family_choices = sorted(set(getattr(checkpoint_family, 'FAMILIES', {}).values())) or None
    parser.add_argument('--family', choices=family_choices)
    parser.add_argument('--check-programs', type=Path)
    parser.add_argument('--print-hf-home', action='store_true')
    parser.add_argument('--resolve-revision', action='store_true')
    parser.add_argument('--model', default='')
    parser.add_argument('--revision', default='')
    parser.add_argument('--hf-home', default='')
    args = parser.parse_args()
    try:
        if args.print_hf_home:
            print(resolve_hf_home(args.hf_home or None))
        elif args.resolve_revision:
            if not args.model:
                parser.error('--resolve-revision requires --model')
            resolved, _ = resolve_model_revision(
                args.model, args.revision, hf_home=args.hf_home or None, allow_remote=True,
            )
            print(resolved)
        elif args.check_programs:
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
