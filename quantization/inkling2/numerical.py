"""Pinned numerical primitives; operational code never changes their identity."""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types

ROOT = Path(os.environ.get('INKLING_GPTQMODEL_ROOT',
    str(Path(__file__).resolve().parents[2] / 'third_party/gptqmodel'))) / 'gptqmodel'


def fork_module(name):
    for package, path in (('gptqmodel', ROOT), ('gptqmodel.utils', ROOT / 'utils')):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(path)]
            module.__spec__ = importlib.util.spec_from_loader(package, loader=None, is_package=True)
            sys.modules[package] = module
        elif Path(next(iter(sys.modules[package].__path__))).resolve() != path.resolve():
            raise ValueError('a different GPTQModel source is already imported')
    return importlib.import_module('gptqmodel.' + name)


def quantize_projection(weight, raw_hessian, *, bits=6, seed=787, sigma=.025, device='cuda:0'):
    if type(bits) is not int or bits not in (4, 6, 8):
        raise ValueError('Inkling production encodes are K4, K6 and K8')
    return fork_module('utils.glm5_core').quantize_projection(
        weight, raw_hessian, bits=bits, seed=seed, sigma=sigma, device=device)


def covariance_accumulator(width, device='cuda:0'):
    return fork_module('utils.exl3_hessian').BoundedHessianAccumulator(width, device=device)


def __getattr__(name):
    if name == 'reconstruct_projection':
        return fork_module('utils.kimi_k3_core_format').reconstruct_core_projection
    if name == 'validate_projection':
        return fork_module('utils.kimi_k3_core_format').validate_core_tensors
    if name == 'ACCUMULATION_CONTRACT':
        return fork_module('utils.exl3_hessian').HESSIAN_ACCUMULATION_CONTRACT
    if name == 'mxfp8_qdq':
        return fork_module('utils.inkling2_core').mxfp8_qdq
    raise AttributeError(name)
