"""32-column MXFP4 expert slices: export gate and exact layouts (CPU only, no torch/GPU)."""
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'fp8_moe_package_exact32', Path(__file__).resolve().parents[1] / 'tools/aot/package_fp8_moe_aot.py')
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


@dataclass(frozen=True)
class _Geometry:
    """Minimal stand-in for b12x's GEOMETRIES entry (the real one needs torch)."""

    weights: str
    intermediate: int
    hidden: int = 4096
    experts: int = 256
    top_k: int = 6
    tp: int = 1
    swiglu_limit: float = 0.0
    activation: int = 1
    projection: int = 1
    kind: str = 'mxfp4'

    @property
    def slice(self):
        return self.intermediate // self.tp

    def with_tp(self, tp):
        return _Geometry(weights=self.weights, intermediate=self.intermediate,
                         hidden=self.hidden, experts=self.experts, top_k=self.top_k,
                         tp=tp, swiglu_limit=self.swiglu_limit,
                         activation=self.activation, projection=self.projection,
                         kind=self.kind)


class Exact32ExportTest(unittest.TestCase):
    def test_exact_widths_tp4_unit32(self):
        self.assertEqual(tool.exact_widths(2304, 4, 32), [576] * 4)

    def test_exact_widths_tp8_unit32(self):
        self.assertEqual(tool.exact_widths(2304, 8, 32), [288] * 8)

    def test_exact_widths_tp6_unit32_uneven_split(self):
        self.assertEqual(tool.exact_widths(2048, 6, 32), [352, 352, 352, 352, 320, 320])

    def test_exact_widths_unit128_unchanged_for_tp1_to_6(self):
        self.assertEqual(tool.exact_widths(2048, 6, 128), [384, 384, 384, 384, 256, 256])
        self.assertEqual(tool.exact_widths(2304, 4, 128), [640, 640, 512, 512])
        self.assertEqual(tool.exact_widths(2304, 1, 128), [2304])
        self.assertEqual(tool.exact_widths(2304, 6, 128), [384] * 6)

    def test_exact_widths_sum_to_intermediate_for_every_tp(self):
        # The export arithmetic must partition the logical intermediate exactly.
        for tp in range(1, 9):
            widths = tool.exact_widths(2304, tp, 32)
            self.assertEqual(len(widths), tp, f'TP{tp}')
            self.assertEqual(sum(widths), 2304, f'TP{tp}: {widths}')
            self.assertTrue(all(width % 32 == 0 for width in widths), f'TP{tp}: {widths}')

    def test_layout_geometry_accepts_exact32_widths(self):
        base = _Geometry(weights='mxfp4', intermediate=2304)
        tp, g = tool.layout_geometry(base, 'tp4-w576')
        self.assertEqual(tp, 4)
        self.assertEqual(g.slice, 576)
        tp, g = tool.layout_geometry(base, 'tp8-w288')
        self.assertEqual(tp, 8)
        self.assertEqual(g.slice, 288)

    def test_layout_geometry_refuses_non_exact_width_and_tp9(self):
        base = _Geometry(weights='mxfp4', intermediate=2304)
        with self.assertRaises(SystemExit):
            tool.layout_geometry(base, 'tp4-w600')
        with self.assertRaises(SystemExit):
            tool.layout_geometry(base, 'tp9')

    def test_spark_layouts_emits_tp8_for_mxfp4(self):
        layouts = tool.spark_layouts(2304, unit=32)
        self.assertIn('tp8', layouts)
        self.assertIn('tp4', layouts)
        # Unequal 32-unit splits still carry their exact widths.
        self.assertIn('tp5-w480', layouts)
        self.assertIn('tp5-w448', layouts)
        self.assertIn('tp7-w352', layouts)
        self.assertIn('tp7-w320', layouts)


if __name__ == '__main__':
    unittest.main()
