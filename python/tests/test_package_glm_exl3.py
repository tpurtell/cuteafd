"""GLM 5.3 EXL3 package geometry: shard profiles, SwiGLU clamp and route policy (no CUDA)."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'exl3_package_glm', Path(__file__).resolve().parents[1] / 'tools/aot/package_exl3_aot.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class GlmExl3PackageTest(unittest.TestCase):
    def test_geometry_and_clamp(self):
        self.assertEqual(tool.GEOMETRIES['glm'], (6144, 2048, 256, 8))
        self.assertIsNone(tool.swiglu_limit('glm'))
        self.assertEqual(tool.swiglu_limit('dsv4p'), 10.0)
        self.assertEqual(tool.package_name('glm', [4, 5]), 'exl3-glm-k45')

    def test_kimi_geometry_activation_and_profiles(self):
        self.assertEqual(tool.GEOMETRIES["kimik3"], (3584, 3072, 896, 16))
        self.assertEqual(tool.activation("kimik3"), "situ")
        self.assertIsNone(tool.swiglu_limit("kimik3"))
        self.assertEqual(tool.package_name("kimik3", [2, 3]), "exl3-kimik3-k23")
        spark = {p[0]: p[1:] for p in tool.profiles_for_role("spark", "kimik3")}
        self.assertEqual(set(spark), {"tp6-width512"})
        self.assertEqual(spark["tp6-width512"],
                         (512, 896, 16, "bf16", [f"tp6-rank{i}" for i in range(6)]))
        coordinator = {p[0]: p[1:] for p in tool.profiles_for_role("coordinator", "kimik3")}
        self.assertEqual(set(coordinator), {"rtx-tp1"})
        self.assertEqual(coordinator["rtx-tp1"], (3072, 896, 16, "fp32", ["rtx-tp1"]))
        for geometry in set(tool.GEOMETRIES) - {"kimik3"}:
            self.assertEqual(tool.activation(geometry), "silu")

    def test_kimi_capacity_set(self):
        requested = '1,16,80,256,1024,4096'
        self.assertEqual(tool.package_capacities('kimik3', requested), [1, 80, 256, 1024, 4096])
        self.assertEqual(tool.package_capacities('glmf', requested), [1, 16, 80, 256, 1024, 4096])
        for invalid in ('16', '2', '32', '0', '4097'):
            with self.assertRaises(ValueError):
                tool.package_capacities('kimik3', invalid)
        for capacity in (1, 80, 256, 1024, 4096):
            self.assertEqual(tool.route_block('kimik3', capacity), 8)
            for width in (512, 3072):
                self.assertIsNone(tool.decode_tile('kimik3', width, capacity))
                self.assertFalse(tool.warp_specialized('kimik3', 'spark', width, capacity))

    def test_profiles(self):
        coordinator = {p[0]: p[1:] for p in tool.profiles_for_role('coordinator', 'glm')}
        self.assertEqual(coordinator['rtx-tp1'], (2048, 256, 8, 'fp32', ['rtx-tp1']))
        spark = {p[0]: (p[1], p[5]) for p in tool.profiles_for_role('spark', 'glm')}
        self.assertEqual(spark['tp4-width512'], (512, ['tp4-rank0', 'tp4-rank1', 'tp4-rank2', 'tp4-rank3']))
        self.assertEqual(spark['tp2-width1024'][0], 1024)
        self.assertEqual(spark['tp3-width768'][1], ['tp3-rank0'])

    def test_route_policy(self):
        self.assertEqual([tool.route_block('glm', c) for c in (1, 16, 80, 256, 1024, 4096)],
                         [8, 8, 8, 8, 64, 64])
        self.assertEqual([tool.token_major_rotation('glm', c) for c in (80, 256, 1024, 4096)],
                         [False, False, True, True])
        # Other geometries keep their policies.
        self.assertEqual([tool.route_block('dsv4p', c) for c in (256, 1024, 4096)], [8, 16, 64])
        self.assertEqual(tool.route_block('v41', 4096), 8)


if __name__ == '__main__':
    unittest.main()
