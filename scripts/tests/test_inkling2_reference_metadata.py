"""Reference full-depth metadata is independent of CUDA/Torch availability."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_partial_flag_tracks_actual_layer_depth():
    path=ROOT/'python/reference/families/inkling2/serving_boundary.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='is_partial')
    scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    assert scope['is_partial'](0,48)
    assert scope['is_partial'](5,48)
    assert scope['is_partial'](46,48)
    assert not scope['is_partial'](47,48)
    source=(ROOT/'scripts/diagnostics/inkling2-prefix-reference.py').read_text()
    assert 'partial=vendor.is_partial(index,cfg.num_hidden_layers)' in source


def test_vendor_mode_retains_both_qdqs_and_bf16_materializations():
    source=(ROOT/'python/reference/families/inkling2/serving_boundary.py').read_text()
    assert 'inputs = f32.quantize_mxfp8(x)' in source
    assert 'linear(inputs[rows], w13).bfloat16().float()' in source
    assert 'cfg.mlp_up_softcap).bfloat16().float()' in source
    assert 'linear(f32.quantize_mxfp8(activation), down).bfloat16().float()' in source
    assert 'combined += slots[:, choice]' in source
    assert 'return combined.bfloat16().float()' in source
