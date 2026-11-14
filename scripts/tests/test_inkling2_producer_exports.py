"""The stage2 producer inventory cannot silently widen an unrelated family."""
import ast
from pathlib import Path
import types
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_predeclared_inkling_producer_stems(monkeypatch):
    class Dtype:
        def __init__(self, name): self.name = name
        def __str__(self): return 'torch.'+self.name
    torch = types.ModuleType('torch')
    torch.dtype = Dtype
    torch.bfloat16, torch.float32 = Dtype('bfloat16'), Dtype('float32')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    modules = {
        'inkling2_convolution': ['compile_inkling2_convolution_aot'],
        'inkling2_clip': ['compile_inkling2_clip_aot'],
        'inkling2_residual': ['compile_inkling2_residual_aot'],
        'inkling2_router': ['compile_inkling2_router_aot', 'compile_inkling2_gate_aot'],
        'inkling2_glue': ['compile_inkling2_norm_aot', 'compile_inkling2_kv_pack_aot'],
    }
    def compiler_stub(**kwargs): return kwargs
    for name, functions in modules.items():
        module = types.ModuleType('b12x.integration.cuteafd.'+name)
        for function in functions: setattr(module, function, compiler_stub)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    tree = ast.parse((ROOT/'python/tools/aot/export_b12x_dsv4_aot.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='inkling2_producer_programs')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<producer>', 'exec'), scope)
    programs = scope['inkling2_producer_programs'](4096)
    stems = [p[0] for p in programs]
    expected = []
    for suffix in ['bf16', 'f32']:
        expected += [f'conv_static_h4608_{suffix}', f'conv_static_h1024_{suffix}', f'conv_dynamic_h1024_{suffix}']
        expected += [f'clip_{kind}_{name}_{suffix}' for kind, names in [('local', ['q', 'k', 'r']), ('global', ['q', 'r'])] for name in names]
        expected += [f'gate_h6144_{suffix}', f'situ_h2304_{suffix}', f'norm_h4608_{suffix}',
                     f'norm_v_h8_d128_{suffix}', f'kv_pack_{suffix}']
    expected += ['residual_q1_keep', 'residual_q1_add', 'residual_q3_keep', 'residual_q3_add', 'router_e384_k8']
    assert stems == expected
    assert len(stems) == len(set(stems)) == 31
    for stem, _, params, compiler in programs:
        import json
        json.dumps(params)
        runtime = compiler()
        assert runtime['max_rows'] == 4096
        if stem.startswith('clip_global'):
            assert runtime['log_floor'] == 128000 and runtime['log_alpha'] == .1
        if stem == 'router_e384_k8':
            assert runtime['route_scale'] == 8.


def test_inkling_projection_and_wire_quantization_inventory():
    tree = ast.parse((ROOT/'python/tools/aot/export_b12x_dsv4_aot.py').read_text())
    shape = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='inkling2_projection_shapes')
    scope = {}
    exec(compile(ast.Module(body=[shape], type_ignores=[]), '<projection>', 'exec'), scope)
    assert scope['inkling2_projection_shapes']() == {
        (4608, 6144), (4608, 768), (4608, 1024), (4608, 48), (4608, 32), (4608, 384),
        (6144, 4608), (64, 1024)}
    quant = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='inkling2_input_quant_program')
    calls = [n for n in ast.walk(quant) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id=='compile_mxfp8_rows_quant_aot']
    assert len(calls)==1
    params = {key.arg: ast.literal_eval(key.value) for key in calls[0].keywords}
    assert params == dict(size_k=4608, expected_m=80, amax_floor=0., wire_rows=True)
    source = ast.unparse(tree)
    assert "('inkling2', 'expert_input_quant', 'expert_input_quant'" in source
