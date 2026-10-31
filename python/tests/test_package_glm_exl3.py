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
        spark = {p[0]: p[1:] for p in tool.build_profiles("spark", "kimik3")}
        self.assertEqual(set(spark), {"tp6-width512", "tp7-width512", "tp7-width384", "tp8-width384"})
        self.assertEqual(spark["tp6-width512"],
                         (512, 896, 16, "bf16", [f"tp6-rank{i}" for i in range(6)]))
        coordinator = {p[0]: p[1:] for p in tool.profiles_for_role("coordinator", "kimik3")}
        self.assertEqual(set(coordinator), {"rtx-tp1", "rtx-tp2"})
        self.assertEqual(coordinator["rtx-tp1"], (3072, 896, 16, "fp32", ["rtx-tp1"]))
        self.assertEqual(coordinator["rtx-tp2"], (1536, 896, 16, "fp32", ["rtx-tp2"]))
        for geometry in set(tool.GEOMETRIES) - {"kimik3"}:
            self.assertEqual(tool.activation(geometry), "silu")

    def test_kimi_capacity_set(self):
        # Kimi drops only m16 (K128 tiles drift against M1 decode; rows 2..16
        # take m80); the prefill wave units 2048/2752 package like any other.
        # Wave merging pads one <=2752-row unit to two or three merged units,
        # so the packed extents reach 8256.
        requested = '1,16,80,256,1024,2048,2752,4096,5504,8256'
        self.assertEqual(tool.package_capacities('kimik3', requested),
                         [1, 80, 256, 1024, 2048, 2752, 4096, 5504, 8256])
        self.assertEqual(tool.package_capacities('glmf', '1,16,80,256,1024,2048,2752,4096'),
                         [1, 16, 80, 256, 1024, 2048, 2752, 4096])
        for invalid in ('16', '0', '8257'):
            with self.assertRaises(ValueError):
                tool.package_capacities('kimik3', invalid)
        with self.assertRaises(ValueError):
            tool.package_capacities('glmf', '4097')
        for capacity in (1, 80, 256, 1024, 2048, 2752, 4096, 5504, 8256):
            self.assertEqual(tool.route_block('kimik3', capacity, 'coordinator'), 8)
            self.assertEqual(tool.route_block('kimik3', capacity, 'spark'), 8)
            self.assertFalse(tool.token_major_rotation('kimik3', capacity, 'coordinator'))
            self.assertFalse(tool.token_major_rotation('kimik3', capacity, 'spark'))
            for width in (512, 3072, 1536):
                if width == 512:
                    # Spark: width 512 is warp-specialized from m1024 up and
                    # keeps its cooperative 8-row export below that.
                    self.assertEqual(tool.warp_specialized('kimik3', 'spark', width, capacity),
                                     capacity > 80)
                    continue
                # Every Kimi K3 coordinator prefill capacity, both RTX splits,
                # takes the prerotated 256-wide FC1 arm (TJ 2026-10-29); the
                # decode capacities m1/m80 keep the cooperative export whose
                # geometry the decode reduction plan freezes.
                if capacity <= 80:
                    self.assertFalse(tool.warp_specialized('kimik3', 'coordinator', width, capacity))
                    self.assertIsNone(tool.ws_tile('kimik3', 'coordinator', width, capacity))
                    continue
                self.assertTrue(tool.warp_specialized('kimik3', 'coordinator', width, capacity))
                self.assertEqual(tool.ws_route_block('kimik3', 'coordinator', width, capacity), 64)
                self.assertEqual(tool.ws_tile('kimik3', 'coordinator', width, capacity),
                                 (64, 256, 64, 256))

    def test_profiles(self):
        coordinator = {p[0]: p[1:] for p in tool.profiles_for_role('coordinator', 'glm')}
        self.assertEqual(coordinator['rtx-tp1'], (2048, 256, 8, 'fp32', ['rtx-tp1']))
        spark = {p[0]: (p[1], p[5]) for p in tool.profiles_for_role('spark', 'glm')}
        self.assertEqual(spark['tp4-width512'], (512, ['tp4-rank0', 'tp4-rank1', 'tp4-rank2', 'tp4-rank3']))
        self.assertEqual(spark['tp2-width1024'][0], 1024)
        self.assertEqual(spark['tp3-width768'][1], ['tp3-rank0'])

    def test_route_policy(self):
        self.assertEqual([tool.route_block('glm', c, 'coordinator') for c in (1, 16, 80, 256, 1024, 4096)],
                         [8, 8, 8, 8, 8, 8])
        self.assertEqual([tool.token_major_rotation('glm', c, 'coordinator') for c in (80, 256, 1024, 4096)],
                         [False, False, False, False])
        # Other geometries keep their policies.
        self.assertEqual([tool.route_block('dsv4p', c, 'coordinator') for c in (256, 1024, 2048, 4096)], [8, 8, 64, 8])
        self.assertEqual([tool.route_block('dsv4p', c, 'spark') for c in (256, 1024, 4096)], [8, 8, 8])
        self.assertNotIn('v41', tool.GEOMETRIES)
        self.assertNotIn('dsv4f', tool.GEOMETRIES)

    def test_policy_functions_require_an_explicit_role(self):
        for call in (lambda: tool.route_block('dsv4p', 1024),
                     lambda: tool.token_major_rotation('dsv4p', 1024)):
            with self.assertRaises(TypeError):
                call()


if __name__ == '__main__':
    unittest.main()
