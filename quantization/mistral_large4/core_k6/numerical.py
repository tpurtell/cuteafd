"""Use the pinned trellis producer without importing unrelated model wrappers."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3] / 'third_party/gptqmodel/gptqmodel'
PREFIX = 'gptqmodel'


def fork_module(name):
    for package, path in ((PREFIX, ROOT), (PREFIX + '.utils', ROOT / 'utils')):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(path)]
            module.__spec__ = importlib.util.spec_from_loader(package, loader=None, is_package=True)
            sys.modules[package] = module
    return importlib.import_module(PREFIX + '.' + name)


def __getattr__(name):
    if name in ('DenseRawHessian', 'capture_linear_input'):
        return getattr(fork_module('utils.kimi_k3_core_stream'), name)
    if name in ('reconstruct_projection', 'validate_projection'):
        key = {'reconstruct_projection': 'reconstruct_core_projection', 'validate_projection': 'validate_core_tensors'}[name]
        return getattr(fork_module('utils.kimi_k3_core_format'), key)
    raise AttributeError(name)


def quantize_projection(weight, raw_hessian, *, bits=6, seed=787, sigma=.025, device='cuda:0'):
    """Native [output,input] matrix to the unchanged standard MCG format.

    K5 is used only by the bounded pilot. The production K4/K6 implementation
    is exactly the pinned fork's validated producer and numerical contract.
    """
    if bits in (4, 6):
        return fork_module('utils.glm5_core').quantize_projection(
            weight, raw_hessian, bits=bits, seed=seed, sigma=sigma, device=device)
    if bits != 5:
        raise ValueError('only K4/K5/K6 are reviewed')
    import math
    import torch
    if weight.ndim != 2 or weight.shape[1] % 128 or not bool(torch.isfinite(weight).all()):
        raise ValueError('invalid K5 pilot weight')
    H = raw_hessian['H']
    count = raw_hessian['count']
    if (raw_hessian.get('finalized') is not False or count <= 0 or H.dtype != torch.float32
            or tuple(H.shape) != (weight.shape[1], weight.shape[1])
            or not bool(torch.isfinite(H).all()) or float(H.diagonal().sum()) <= 0):
        raise ValueError('invalid K5 pilot raw Hessian')
    outputs, inputs = weight.shape
    stored = math.ceil(outputs / 128) * 128
    working = torch.zeros((inputs, stored), device=device, dtype=torch.float32)
    working[:, :outputs].copy_(weight.T)
    args = dict(K=bits, seed=seed, sigma_reg=sigma, devices=[torch.device(device)], mcg=True, apply_out_scales=None)
    quant = fork_module('exllamav3.modules.quant.exl3_lib.quantize')
    _, _, packed = quant.quantize_exl3(working, dict(H=H.to(device, copy=True), count=count, finalized=False),
                                      args, return_weight_q=False)
    metrics = args['error_metrics']
    if (metrics.get('quantizer_path') != 'hessian_ldlq' or metrics.get('hessian_metric_status') != 'ok'
            or metrics.get('hessian_numerical_contract') != 'signed-block-hadamard-congruence-fp64-v1'):
        raise ValueError('K5 did not use the qualified calibrated numerical producer')
    for key in ('reported_metric_value', 'hessian_weighted_error_numerator',
                'hessian_weighted_reference_denominator', 'hessian_weighted_relative_error'):
        if not math.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError('nonfinite K5 metric')
    packed = {k: v.detach().cpu() for k, v in packed.items()}
    evidence = __getattr__('validate_projection')(packed, list(weight.shape))
    evidence.update(hessian_rows=count, identity_rows=raw_hessian.get('identity_rows', 0),
                    replay='serialized-exl3-fp16-scales-reconstructed-then-native-output-crop')
    return packed, metrics, evidence
