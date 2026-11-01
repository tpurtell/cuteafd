"""Call the pinned EXL3 implementation without loading GPTQModel's model registry."""
from __future__ import annotations
import importlib
import math
from pathlib import Path
import sys
import types
import torch

FORK = Path(__file__).resolve().parents[3] / 'third_party/gptqmodel/gptqmodel'
ACCUMULATION_CONTRACT = 'fp32-products-k128-fp64-sum-fp32-storage-v1'


def _namespaces():
    # These namespaces expose unmodified pinned implementation files. The top
    # level package initializer installs unrelated model/Transformers patches;
    # this isolated runner needs only the EXL3 codec and its JIT loader.
    for name, path in [('gptqmodel', FORK), ('gptqmodel.utils', FORK/'utils'),
                       ('gptqmodel.exllamav3', FORK/'exllamav3')]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__file__ = str(path/'__init__.py')
            sys.modules[name] = package


def _quantizer():
    _namespaces()
    return importlib.import_module('gptqmodel.exllamav3.modules.quant.exl3_lib.quantize')


def covariance_accumulator(width, device='cuda:0'):
    # Preserve a single FP64 sum across the complete native observation stream.
    # Each product uses at most 128 real rows; exported covariance stays FP32.
    _namespaces()
    from gptqmodel.utils.exl3_hessian import BoundedHessianAccumulator, HESSIAN_ACCUMULATION_CONTRACT
    if HESSIAN_ACCUMULATION_CONTRACT != ACCUMULATION_CONTRACT:
        raise ValueError('fork covariance contract differs')
    return BoundedHessianAccumulator(width, device=device, max_rows=128)


def validate_projection(packed, shape):
    if not isinstance(shape,(list,tuple)) or len(shape)!=2 or any(type(x) is not int or x<=0 for x in shape):
        raise ValueError("invalid logical projection shape")
    outputs, inputs = shape
    stored = (outputs+127)//128*128
    if (inputs%128 or set(packed) != {'trellis','suh','svh','mcg'}
            or packed['trellis'].dtype != torch.int16 or packed['trellis'].ndim != 3
            or tuple(packed['trellis'].shape[:2]) != (inputs//16,stored//16)
            or packed['trellis'].shape[-1] not in (64,80,96)
            or packed['suh'].dtype != torch.float16 or tuple(packed['suh'].shape) != (inputs,)
            or packed['svh'].dtype != torch.float16 or tuple(packed['svh'].shape) != (stored,)
            or packed['mcg'].dtype != torch.int32 or tuple(packed['mcg'].shape) not in ((),(1,))
            or int(packed['mcg'].reshape(()).view(torch.uint32).item()) != 0xCBAC1FED):
        raise ValueError('EXL3 codec returned invalid MCG geometry')
    if (not torch.isfinite(packed['suh']).all() or not torch.isfinite(packed['svh']).all()
            or torch.count_nonzero(packed['svh'][outputs:])):
        raise ValueError('nonfinite scales or nonzero padded output rows')
    return dict(bits=packed['trellis'].shape[-1]//16,native_shape=list(shape),
                stored_shape=[stored,inputs],output_padding=stored-outputs)


@torch.inference_mode()
def quantize_projection(weight, raw_hessian, *, bits=6, seed=787, sigma=.025, device='cuda:0'):
    if bits not in (4,5,6) or type(bits) is not int or seed!=787 or sigma!=.025:
        raise ValueError('unreviewed Muse codec parameters')
    if (weight.ndim!=2 or weight.dtype!=torch.bfloat16 or weight.shape[1]%128
            or raw_hessian.get('finalized') is not False or type(raw_hessian.get('count')) is not int
            or raw_hessian['count']<=0):
        raise ValueError('invalid projection or raw calibrated Hessian')
    if raw_hessian.get('accumulation_contract') != ACCUMULATION_CONTRACT:
        raise ValueError('Muse quantization requires verified K128 covariance capture')
    identity_rows=raw_hessian.get('identity_rows',0)
    if type(identity_rows) is not int or not 0<=identity_rows<=raw_hessian['count']:
        raise ValueError('invalid declared synthetic row count')
    outputs,inputs=weight.shape
    H=raw_hessian['H']
    if (H.dtype!=torch.float32 or tuple(H.shape)!=(inputs,inputs)
            or not torch.isfinite(H).all() or H.diagonal().sum()<=0):
        raise ValueError('nonfinite or empty raw calibration covariance')
    for chunk in weight.split(4096):
        if not torch.isfinite(chunk).all():
            raise ValueError('nonfinite source projection')
    # Keep large heads on the CPU: the pinned codec tiles CPU-resident weights.
    working=torch.zeros((inputs,(outputs+127)//128*128),device=weight.device,dtype=torch.float32)
    working[:,:outputs].copy_(weight.T)
    raw=dict(H=H.to(device=device,copy=True),count=raw_hessian['count'],finalized=False)
    args=dict(K=bits,seed=seed,sigma_reg=sigma,devices=[torch.device(device)],mcg=True,apply_out_scales=None)
    _,_,packed=_quantizer().quantize_exl3(working,raw,args,return_weight_q=False)
    metrics=args.get('error_metrics',{})
    required=dict(quantizer_path='hessian_ldlq',hessian_metric_status='ok',
        hessian_numerical_contract='signed-block-hadamard-congruence-fp64-v1',
        hessian_transform_compute_dtype='torch.float64',hessian_storage_dtype='torch.float32',
        hessian_sample_count=raw_hessian['count'])
    if any(metrics.get(k)!=v for k,v in required.items()):
        raise ValueError('codec failed calibrated FP64-congruence LDLQ contract')
    for key in ('reported_metric_value','hessian_weighted_error_numerator',
                'hessian_weighted_reference_denominator','hessian_weighted_relative_error'):
        if type(metrics.get(key)) not in (float,int) or not math.isfinite(metrics[key]) or metrics[key]<0:
            raise ValueError('nonfinite/negative calibrated quantization metric: '+key)
    if metrics['hessian_weighted_reference_denominator']<=0:
        raise ValueError('calibrated reference energy is zero')
    packed={k:v.detach().cpu() for k,v in packed.items()}
    evidence=validate_projection(packed,list(weight.shape))
    if evidence['bits']!=bits:
        raise ValueError('codec bitrate differs')
    evidence.update(hessian_rows=raw_hessian['count'],identity_rows=raw_hessian.get('identity_rows',0),
        accumulation_contract=ACCUMULATION_CONTRACT,
        replay='serialized-exl3-fp16-scales-reconstructed-then-native-output-crop')
    return packed,metrics,evidence


@torch.inference_mode()
def reconstruct_projection(packed, logical_shape, *, device='cuda:0',dtype=torch.bfloat16):
    validate_projection(packed,logical_shape)
    result=_quantizer().reconstruct_exl3_tensors(packed,device=device,dtype=dtype)
    return result.T[:logical_shape[0],:logical_shape[1]].contiguous()


class DenseRawHessian:
    def __init__(self,width,device='cuda:0'):
        self._state=covariance_accumulator(width,device)
    @property
    def H(self):
        return self._state.snapshot()
    @property
    def count(self):
        return self._state.count
    def add(self,x):
        self._state.add(x)
    def raw_dict(self,minimum_rows=1):
        H=self.H
        if self.count<minimum_rows or H.diagonal().sum()<=0:
            raise ValueError('invalid or insufficient Hessian observations')
        return dict(H=H,count=self.count,finalized=False,identity_rows=0,
            accumulation_contract=ACCUMULATION_CONTRACT)
