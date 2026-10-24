"""Exercise package contents with a fake compiler; no CUDA, no device context.

The tile assertions ask the pinned B12x planner what its own policy resolves, so
this file does not restate numbers it cannot own. That import is a pure arithmetic
resolver and creates no CUDA context; it is not fully isolated, because unloading
torch's C extensions corrupts later imports in the same process (see
`_load_planner_tile`), so the pin may stay cached for the session. Where the pin is
absent the dependent tests skip rather than error.
"""
import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]

def _load_planner_tile(source=None):
    """Import the pinned planner's tile resolver without leaking into this process.

    `_projection_mixed_tile_config` is pure arithmetic, so no CUDA context is created.
    On success only `sys.path` is restored: the import drags in torch's C extensions,
    and deleting those from `sys.modules` afterwards leaves them half-registered, so a
    later `import torch` in the same process fails with
    `SystemError: ... bad call flags`. Anything unavailable yields None so the
    dependent tests skip instead of erroring. `source` is a parameter so the failure
    path is testable.
    """
    source = str(source or ROOT / 'third_party' / 'sparkinfer')
    path_snapshot, modules_before = list(sys.path), set(sys.modules)
    try:
        sys.path.insert(0, source)
        from b12x.moe.fused_moe._impl import _projection_mixed_tile_config
        return _projection_mixed_tile_config
    except Exception:
        # A failed import may leave half-initialized b12x modules behind. Drop only
        # the newly added b12x ones: torch and the rest of the dependency graph are
        # shared with the rest of the session and unloading them corrupts later
        # imports (SystemError: bad call flags).
        for name in set(sys.modules) - modules_before:
            if name == 'b12x' or name.startswith('b12x.'):
                sys.modules.pop(name, None)
        return None
    finally:
        sys.path[:] = path_snapshot


POLICY_TILE = _load_planner_tile()


def _load_direct_route_policy():
    """The same helper production uses to choose direct vs packed routing.

    Imported the way the tile policy is, so the fixture follows the pin's rule
    instead of freezing a constant that happens to be right today.
    """
    path_snapshot = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT / 'third_party' / 'sparkinfer'))
        from b12x.moe.fused_moe._impl import _projection_mixed_direct_topk_routes
        return _projection_mixed_direct_topk_routes
    except Exception:  # pragma: no cover - absence only relaxes the fixture
        return None
    finally:
        sys.path[:] = path_snapshot


DIRECT_ROUTES = _load_direct_route_policy()
# Positive placeholder used when the pinned planner is unavailable. Never compared
# against the planner; it exists so non-policy tests still carry a valid tile shape.
NEUTRAL_POLICY_TILE = [16, 32, 16, 32]

spec = importlib.util.spec_from_file_location('exl3_package_profiles',
    Path(__file__).resolve().parents[1] / 'tools/aot/package_exl3_aot.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageProfileTests(unittest.TestCase):
    @staticmethod
    def expected_policy_tile(variant):
        """Ask the pinned planner; never restate its numbers in a test."""
        # Two-tier fixture exports are direct-capacity-bound exactly as production;
        # with this pin the resolver says packed for the capacities used here.
        direct = (False if DIRECT_ROUTES is None else bool(DIRECT_ROUTES(
            variant['capacity'], variant['top_k'], direct_exl3=True)))
        return list(POLICY_TILE(None, hidden_size=5120, intermediate_size=variant['intermediate'],
                                token_count=variant['capacity'], direct_topk_routes=direct))

    @staticmethod
    def pinned_tile(variant):
        """The package's own answer for a variant: the pin, or the planner's value."""
        return package.decode_tile('v41', variant['intermediate'], variant['capacity'])

    # How many ranks share one export per width, from the layout contract.
    expected_rank_count = {'tp2-rank': 2, 'tp3-rank': 3}

    def build_fixture(self, root, role, bits, require=(), tiles=(), geometry='v41',
                      capacities=(1, 80), compact=False, profiles=()):
        args = argparse.Namespace(
            output=root / 'package', build_dir=root / 'build', role=role, geometry=geometry,
            bits=list(bits), capacities=','.join(map(str, capacities)),
            require_layout=list(require), tile=list(tiles),
            compact_companions=compact, profile=list(profiles),
            cxx='unused-cxx', cuda_include=root, cuda_libdir=root,
            cuda_driver=root / 'libcuda.so', runtime=root / 'libcute_dsl_runtime.so')
        args.runtime.write_bytes(b'runtime fixture')
        pinned = ModuleType('_pinned_sparkinfer')
        pinned.REVISION = 'fixture-revision'
        exporter = ModuleType('export_b12x_exl3_aot')
        torch = ModuleType('torch')
        compute = [12, 1 if role == 'spark' else 0]
        torch.cuda = SimpleNamespace(
            get_device_properties=Mock(return_value=SimpleNamespace(
                major=compute[0], minor=compute[1], multi_processor_count=48)),
            get_device_capability=Mock(return_value=tuple(compute)),
            empty_cache=Mock())
        jobs = ModuleType('_export_jobs')
        jobs.export_jobs = lambda: 1
        jobs.gpu_workers = lambda requested: requested
        jobs.export_slot = nullcontext
        jobs.park_context_lease = lambda: None
        image = bytearray(64)
        image[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<H', image, 18, 190)
        struct.pack_into('<I', image, 48, (10 * compute[0] + compute[1]) << 8)

        def export(raw, width, experts, capacity, tiers, routing, topk, dtype, **options):
            self.assertEqual(routing, 'auto')
            raw.mkdir(parents=True)
            meta = dict(capacity=capacity, intermediate=width, experts=experts,
                        top_k=topk, output_dtype=dtype, bits=list(tiers),
                        blocks_per_sm=options.get('blocks_per_sm', 1),
                        compute=compute, sparkinfer_revision=pinned.REVISION)
            for label in ('v41_exl3_core', 'v41_exl3_sum'):
                (raw / f'{label}.o').write_bytes(image)
            # Preserve the policy's options in the fake export record.
            self.assertLessEqual(set(options), {'tile', 'blocks_per_sm', 'route_block',
                                                'hidden', 'swiglu_limit', 'decode_schedule',
                                                'compact_weights', 'token_major_rotation', 'warp_specialized',
                                                'wire_input', 'ws_input_stages', 'ws_dynamic_tiles', 'activation'},
                                 'unexpected option consumed by the fake export')
            if options.get('compact_weights'):
                self.assertEqual(len(tiers), 2)
                meta.update(descriptor_rows=7, native_info_version=3)
            if 'token_major_rotation' in options:
                meta['token_major_rotation'] = options['token_major_rotation']
            if 'route_block' in options:
                meta['route_block'] = options['route_block']
            if 'decode_schedule' in options:
                # Pure schedules are already canonical; retain the alias sentinel.
                schedule = options['decode_schedule']
                meta['decode_schedule'] = schedule if schedule.startswith('l2=') else f'canonical:{schedule}'
            if geometry != 'v41':
                meta['hidden'] = options['hidden']
                self.assertEqual(options['hidden'], package.GEOMETRIES[geometry][0])
                self.assertEqual(options.get('swiglu_limit', 10.0), package.swiglu_limit(geometry))
            # Same derivation as the exporter: direct vs packed routing decides the
            # route bridge, so m1 needs it too whenever routing is packed.
            direct = (False if DIRECT_ROUTES is None else bool(DIRECT_ROUTES(
                capacity, topk, direct_exl3=len(tiers) == 2)))
            meta['direct'] = direct
            meta['requires_route_preparation'] = not direct
            # The real exporter always records the tile it resolved, and the pinned policy differs per width *and* per capacity,
            # so the fixture asks the planner itself rather than restating numbers.
            if 'tile' in options:
                meta['tile'] = list(options['tile'])
            elif POLICY_TILE is None:
                # A positive placeholder that is clearly not a policy value: the
                # ungated tests only need "a tile was recorded and survives
                # verification", and verify() rejects zeros outright.
                meta['tile'] = list(NEUTRAL_POLICY_TILE)
            else:
                meta['tile'] = list(POLICY_TILE(None, hidden_size=package.GEOMETRIES[geometry][0],
                    intermediate_size=width, token_count=capacity,
                    direct_topk_routes=direct))
            if meta['requires_route_preparation']:
                routes = raw / 'routes'
                routes.mkdir()
                route_manifest = routes / 'v41_exl3_routes.json'
                route_manifest.write_text('{}')
                meta['route_preparation'] = dict(manifest='routes/v41_exl3_routes.json',
                                                  sha256=package.digest(route_manifest))
            (raw / 'v41_exl3.json').write_text(json.dumps(meta))
            (raw / 'trellis_lut.bin').write_bytes(b'lut fixture')
            return meta

        exporter.export = Mock(side_effect=export)

        def link(command, *, check):
            self.assertTrue(check)
            self.assertEqual(command[0], 'unused-cxx')
            Path(command[-1]).write_bytes(image)

        # Keep fake compiler modules in-process; exercise the real worker and
        # package assembly, not multiprocessing's serialization of local mocks.
        def compile_inline(payloads, _jobs):
            return [package.compile_variant(payload) for payload in payloads]

        with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned, '_export_jobs': jobs,
                                     'export_b12x_exl3_aot': exporter, 'torch': torch}), \
                patch.object(package, 'compile_variants', side_effect=compile_inline), \
                patch.object(package.os, 'setsid'), \
                patch.object(package.subprocess, 'run', side_effect=link):
            package.build(args)
        manifest = package.verify(args.output, pinned.REVISION, args.runtime, role)
        self.assertEqual(torch.cuda.empty_cache.call_count, exporter.export.call_count)
        return args.output, manifest, exporter.export.call_args_list, capacities

    def test_compact_companions_only_follow_selected_uneven_worlds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, manifest, calls, capacities = self.build_fixture(
                root, 'spark', (4, 5), geometry='glm', capacities=(1, 16),
                compact=True, profiles=('tp6-width256',))
            self.assertEqual(len(calls), 4)  # one fixed and one compact per capacity
            compact_calls = [call for call in calls if call.kwargs.get('compact_weights')]
            self.assertEqual(len(compact_calls), 2)
            for call in compact_calls:
                self.assertEqual(call.args[1], 384)
                self.assertEqual(call.kwargs['tile'], (64, 128, 64, 128))
                self.assertNotIn('warp_specialized', call.kwargs)
                expected = package.default_decode_schedule('glm', 'spark', 'tp6-compact-width384',
                                                           call.args[3], [4, 5])
                self.assertEqual(call.kwargs.get('decode_schedule'), expected)
                if expected:
                    self.assertIn('+' + expected, str(call.args[0]))
            self.assertEqual(manifest['requested_layouts'],
                             [f'tp6-compact-rank{rank}' for rank in range(6)])
            self.assertEqual(manifest['requested_capacities'], [1, 16])
            self.assertEqual(len(manifest['variants']), 16)
            for variant in manifest['variants']:
                compact = '-compact-' in variant['directory']
                self.assertEqual(variant.get('descriptor_rows'), 7 if compact else None)
                self.assertEqual(variant.get('native_info_version'), 3 if compact else None)
            # Re-sign the record: rejection must be semantic, not only a checksum.
            variant = next(v for v in manifest['variants'] if '-compact-' in v['directory'])
            path = output / variant['directory'] / 'v41_exl3.json'
            meta = json.loads(path.read_text())
            meta['descriptor_rows'] = 3
            path.write_text(json.dumps(meta))
            name = str(path.relative_to(output))
            manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': package.digest(path)}
            (output / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'descriptor contract mismatch'):
                package.verify(output)

    def test_compact_prefill_keeps_cooperative_route_and_legacy_cache_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _output, manifest, calls, _ = self.build_fixture(
                root, 'spark', (3, 4), capacities=(1, 16, 4096), compact=True,
                profiles=('tp4-width512', 'tp4-width640'))
            self.assertEqual(len(calls), 9)
            self.assertEqual(len(manifest['variants']), 24)
            for call in calls:
                path = call.args[0].relative_to(root / 'build')
                if call.kwargs.get('compact_weights'):
                    self.assertEqual(path.parts[0], 'tp4-compact-width640+tile64-128-64-128')
                    self.assertNotIn('warp_specialized', call.kwargs)
                elif call.args[3] == 16:
                    self.assertIn('+tile', path.parts[0])
                else:
                    self.assertIn(path.parts[0], ('tp4-width512', 'tp4-width640'))
                    self.assertNotIn('descriptor_rows', json.loads((call.args[0] / 'v41_exl3.json').read_text()))

    def test_compact_even_control_preserves_legacy_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            _output, manifest, calls, _ = self.build_fixture(
                Path(temporary), 'spark', (4, 5), geometry='glm', capacities=(1, 16),
                compact=True, profiles=('tp2-width1024',))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(not call.kwargs.get('compact_weights') for call in calls))
            self.assertNotIn('requested_layouts', manifest)
            self.assertEqual({v['directory'].split('/')[0] for v in manifest['variants']},
                             {'tp2-rank0', 'tp2-rank1'})

    def test_compact_rejects_coordinator_and_non_two_tier_before_cuda(self):
        for role, bits in [('coordinator', (4, 5)), ('spark', (2, 3, 4))]:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, 'two-tier Spark exports'):
                    self.build_fixture(Path(temporary), role, bits, compact=True)

    def test_compact_profile_inventory_uses_h128_ceil_only_when_uneven(self):
        expected = {'v41': {4: 640}, 'glm': {3: 768, 6: 384},
                    'qwen4': {2: 384, 3: 256, 4: 256}, 'dsv4p': {}}
        for geometry, widths in expected.items():
            profiles = package.build_profiles('spark', geometry)
            companions = package.compact_profiles(geometry, profiles)
            self.assertEqual({int(name.split('-')[0][2:]): width
                              for name, width, *_ in companions}, widths)

    def assert_layouts(self, manifest, capacities, expected, bits):
        variants = {v['directory']: v for v in manifest['variants']}
        self.assertEqual(set(variants),
                         {f'{layout}/m{capacity}' for layout in expected for capacity in capacities})
        for directory, variant in variants.items():
            width, experts, topk, dtype = expected[directory.split('/')[0]]
            self.assertEqual((variant['intermediate'], variant['experts'], variant['top_k'],
                              variant['output_dtype']), (width, experts, topk, dtype))
            self.assertEqual(variant['bits'], list(bits))

    def test_spark_disjoint_families_include_tp2_tp3_and_preserve_tp4(self):
        """One Spark package serves TP4, exact TP2 and exact TP3 ownership.

        TP3 is the 6+6+6 split of the 18 whole 128-wide Trellis blocks at I=2304, so
        all three ranks share one width-768 export per capacity - like TP2, and
        unlike the padded 640/512 TP4 pair-split.
        """
        expected = {f'tp4-rank{rank}': (640 if rank < 2 else 512, 384, 6, 'bf16')
                    for rank in range(4)}
        expected.update({f'tp2-rank{rank}': (1152, 384, 6, 'bf16') for rank in range(2)})
        expected.update({f'tp3-rank{rank}': (768, 384, 6, 'bf16') for rank in range(3)})
        expected.update({f'tp6-rank{rank}': (384, 384, 6, 'bf16') for rank in range(6)})
        for bits in ((2, 3), (3, 4)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                output, manifest, calls, capacities = self.build_fixture(Path(temporary), 'spark', bits)
                self.assert_layouts(manifest, capacities, expected, bits)
                self.assertEqual(len(calls), 5 * len(capacities))
                for group, width in (('tp2-rank', 1152), ('tp3-rank', 768)):
                    for capacity in capacities:
                        shared = [call for call in calls
                                  if call.args[1] == width and call.args[3] == capacity]
                        self.assertEqual(
                            len(shared), 1,
                            f'compile width {width} once per capacity, not per rank')
                        ranks = sorted(layout for layout in expected if layout.startswith(group))
                        self.assertEqual(len(ranks), self.expected_rank_count[group])
                        first = output / f'{ranks[0]}/m{capacity}'
                        for twin_layout in ranks[1:]:
                            twin = output / f'{twin_layout}/m{capacity}'
                            for path in first.rglob('*'):
                                if path.is_file():
                                    self.assertEqual(
                                        path.read_bytes(),
                                        (twin / path.relative_to(first)).read_bytes())

    def test_qwen_spark_tp1_packages_whole_width_bf16_experts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, calls, capacities = self.build_fixture(
                Path(temporary), 'spark', (4, 5), require=['tp1-rank0'], geometry='qwen4')
            self.assertEqual(manifest['geometry'], 'qwen4')
            self.assertEqual(manifest['requested_layouts'], ['tp1-rank0'])
            variants = [v for v in manifest['variants'] if v['directory'].startswith('tp1-rank0/')]
            self.assertEqual(len(variants), len(capacities))
            for variant in variants:
                self.assertEqual((variant['intermediate'], variant['experts'], variant['top_k'],
                                  variant['output_dtype'], variant['bits']), (640, 512, 10, 'bf16', [4, 5]))
                meta = json.loads((output / variant['directory'] / 'v41_exl3.json').read_text())
                self.assertEqual(meta['hidden'], 2560)
            whole = [call for call in calls if call.args[1] == 640]
            self.assertEqual(len(whole), len(capacities))
            for call in whole:
                self.assertEqual(call.kwargs['hidden'], 2560)
                self.assertIsNone(call.kwargs['swiglu_limit'])

    def test_glmf_spark_tp4_decode_capacities_carry_the_gb10_schedule(self):
        """GLM 5.3 Flash's TP4 decode exports (m1, m80) get an m<capacity>-gb10 sibling with
        the gb10 decode schedule, compiled once per capacity and shared by the four ranks;
        default m1/m80 instead carry the measured pure evict-first identity."""
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, calls, capacities = self.build_fixture(
                Path(temporary), 'spark', (3, 4), geometry='glmf')
            scheduled = {v['directory']: v for v in manifest['variants'] if 'schedule' in v}
            self.assertEqual(set(scheduled), {f'tp4-rank{rank}/m{capacity}-gb10'
                                              for rank in range(4) for capacity in capacities})
            for directory, variant in scheduled.items():
                self.assertEqual((variant['schedule'], variant['decode_schedule']),
                                 ('gb10', 'canonical:gb10'))
                meta = json.loads((output / directory / 'v41_exl3.json').read_text())
                self.assertEqual(meta['decode_schedule'], 'canonical:gb10')
                default = json.loads((output / directory.removesuffix('-gb10') / 'v41_exl3.json').read_text())
                self.assertEqual(default['decode_schedule'], package.PURE_EVICT_FIRST)
                default_variant = next(v for v in manifest['variants']
                                       if v['directory'] == directory.removesuffix('-gb10'))
                self.assertNotIn('schedule', default_variant)
                self.assertEqual(default_variant['decode_schedule'], default['decode_schedule'])
                # m80-gb10 runs 64x128 tiles at two CTAs per SM; m1-gb10 keeps the default tile.
                wide = directory.endswith('/m80-gb10')
                self.assertEqual(variant['blocks_per_sm'], 2 if wide else 1)
                if wide:
                    self.assertEqual(variant['tile'], [64, 128, 64, 128])
                    self.assertNotIn('tile_requested', variant)
            gb10 = [call for call in calls if call.kwargs.get('decode_schedule') == 'gb10']
            self.assertEqual(sorted(call.args[3] for call in gb10), sorted(capacities))
            self.assertTrue(all(call.args[1] == 512 for call in gb10))
            for call in gb10:
                expected = ({'tile': (64, 128, 64, 128), 'blocks_per_sm': 2} if call.args[3] == 80 else {})
                self.assertEqual({key: call.kwargs[key] for key in ('tile', 'blocks_per_sm')
                                  if key in call.kwargs}, expected)
            profiles = package.build_profiles('spark', 'glmf')
            self.assertEqual(len(calls) - len(gb10), len(profiles) * len(capacities))
            for directory in (v['directory'] for v in manifest['variants'] if 'schedule' not in v):
                self.assertRegex(directory, r'^tp[2346]-rank[0-5]/m(1|80)$')

    def test_verify_rejects_a_relabelled_decode_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, _, _ = self.build_fixture(Path(temporary), 'spark', (3, 4), geometry='glmf')
            for mutate in (lambda v: v.pop('schedule'), lambda v: v.update(decode_schedule='l2=2'),
                           lambda v: v.update(schedule='gb11')):
                broken = json.loads(json.dumps(manifest))
                mutate(next(v for v in broken['variants'] if v.get('schedule') == 'gb10'))
                (output / 'manifest.json').write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, 'decode schedule mismatch'):
                    package.verify(output)

    def test_decode_schedules_leave_every_other_family_untouched(self):
        for geometry in package.GEOMETRIES:
            for role in ('spark', 'coordinator'):
                for profile, *_ in package.profiles_for_role(role, geometry):
                    for capacity in (1, 16, 80, 256, 1024, 4096):
                        variants = package.decode_schedule_variants(geometry, role, profile, capacity)
                        options = {1: {'decode_schedule': 'gb10'},
                                   80: {'decode_schedule': 'gb10', 'tile': (64, 128, 64, 128),
                                        'blocks_per_sm': 2}}
                        expected = ([('gb10', options[capacity])]
                                    if (geometry, role) == ('glmf', 'spark')
                                    and profile.startswith('tp4-') and capacity in (1, 80) else [])
                        self.assertEqual(variants, expected, (geometry, role, profile, capacity))

    def test_requested_layouts_are_recorded_and_reverified(self):
        """A declared contract survives into verify(); its absence does not (v9)."""
        require = ['tp3-rank0', 'tp3-rank1', 'tp3-rank2']
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, _, _ = self.build_fixture(Path(temporary), 'spark', (2, 3),
                                                        require=require)
            self.assertEqual(manifest['requested_layouts'], require)
            self.assertEqual(package.verify(output, 'fixture-revision'), manifest)

    def test_requesting_a_layout_the_role_cannot_build_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError) as caught:
                self.build_fixture(Path(temporary), 'coordinator', (2, 3), require=['tp3-rank0'])
        self.assertIn('unknown EXL3 layout requested: tp3-rank0', str(caught.exception))

    def test_verify_rejects_a_package_missing_a_requested_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, _, _ = self.build_fixture(Path(temporary), 'spark', (2, 3),
                                                        require=['tp3-rank0'])
            tampered = dict(manifest, requested_layouts=['tp3-rank0', 'tp4-rank0', 'tp9-rank0'])
            (output / 'manifest.json').write_text(json.dumps(tampered))
            with self.assertRaises(ValueError) as caught:
                package.verify(output)
        self.assertIn('missing requested layout', str(caught.exception))

    def test_a_spark_only_tile_knob_cannot_be_requested_on_the_coordinator(self):
        """No silent knobs: an override for a profile this role never builds fails.

        Accepting it and ignoring it would ship looking effective, so the profile
        set is scoped to the role being built, exactly like --require-layout.
        """
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError) as caught:
                self.build_fixture(Path(temporary), 'coordinator', (2, 3),
                                   tiles=['tp3-width768=1:64,256,64,256'])
        message = str(caught.exception)
        self.assertIn('must name a coordinator profile', message)
        self.assertIn('rtx-tp1', message, 'the error lists what the role can build')

    def test_the_tile_grammar_documented_in_help_is_the_grammar_parsed(self):
        """The help example has to parse; a typo in it is an invalid shipped knob.

        Reads the example straight out of the argparse help text so the two cannot
        drift apart, and keeps the two historical malformed spellings rejected.
        """
        import argparse
        tool = argparse.ArgumentParser()
        tool.add_argument('--tile', action='append', default=[])
        source = Path(package.__file__).read_text()
        examples = [token for token in source.split() if 'tp3-width768=' in token]
        self.assertTrue(examples, 'the package tool documents no tile example')
        for example in examples:
            cleaned = example.strip("'(),;")
            with self.subTest(example=cleaned):
                parsed = package.tile_overrides([cleaned], [1, 16, 80], 'spark')
                self.assertEqual(list(parsed), ['tp3-width768'])
        for bad in ('tp3-width768=64,256,64,256', 'tp3-width768=16=64,256,64,256'):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as caught:
                package.tile_overrides([bad], [1, 16, 80], 'spark')
            self.assertIn('must name a spark profile as PROFILE=CAPACITIES', str(caught.exception))

    @unittest.skipIf(POLICY_TILE is None, 'requires the pinned b12x planner')
    def test_recorded_tiles_are_the_planner_values_for_their_own_shape(self):
        """Policy equality alone: what the pin resolved per width and capacity.

        Split from the tamper negatives so those always run; only this comparison
        needs the planner importable.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, manifest, _calls, _ = self.build_fixture(root, 'spark', (2, 3), capacities=(1, 16, 80))
            pinned = [v for v in manifest['variants'] if self.pinned_tile(v)]
            self.assertTrue(pinned, 'the fixture must carry a capacity the planner widens')
            for variant in manifest['variants']:
                with self.subTest(directory=variant['directory']):
                    pin = self.pinned_tile(variant)
                    self.assertEqual(variant['tile'],
                                     list(pin) if pin else self.expected_policy_tile(variant))
                    self.assertNotIn('tile_requested', variant)
                    if pin:
                        self.assertNotEqual(variant['tile'], self.expected_policy_tile(variant),
                                            'the pin exists because the planner widens this capacity')
                    self.assertEqual(package.verify(output, 'fixture-revision')['variants'],
                                     manifest['variants'])

    def test_verify_cross_checks_the_recorded_tile_against_the_export(self):
        """A tile in the manifest is a claim about compiled geometry.

        The export's own record is the authority: an edited manifest must not be
        able to rename the geometry a variant was compiled for, and a requested
        override that the compiler resolved differently must fail closed.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self.build_fixture(root, 'spark', (2, 3), tiles=['tp3-width768=80:64,256,64,256'])
            manifest = json.loads((root / 'package/manifest.json').read_text())
            variant = next(v for v in manifest['variants'] if v['directory'] == 'tp3-rank0/m80')
            self.assertEqual(variant['tile'], [64, 256, 64, 256], 'the override is the compiled tile')
            self.assertEqual(variant['tile_requested'], variant['tile'])
            policy = next(v for v in manifest['variants'] if v['directory'] == 'tp3-rank0/m1')
            self.assertIn('tile', policy, 'a build with no override still records a tile')
            self.assertNotIn('tile_requested', policy)

            for key, expected in (('tile', 'variant tile mismatch'),
                                  ('tile_requested', 'override was not applied')):
                with self.subTest(field=key):
                    tampered = json.loads((root / 'package/manifest.json').read_text())
                    target = next(v for v in tampered['variants'] if v['directory'] == 'tp3-rank0/m80')
                    target[key] = [128, 128, 128, 128]
                    with tempfile.TemporaryDirectory() as other:
                        copy = Path(other) / 'package'
                        shutil.copytree(root / 'package', copy)
                        (copy / 'manifest.json').write_text(json.dumps(tampered))
                        with self.assertRaises(ValueError) as caught:
                            package.verify(copy, 'fixture-revision')
                    self.assertIn(expected, str(caught.exception))

    def test_six_rank_profiles_split_whole_blocks(self):
        """V4 Pro's 24 H128 blocks split four apiece; a 2048 intermediate's 16
        split 3, 3, 3, 3, 2, 2 (one export per width). V4.1's exporter
        carries three blocks per rank; Qwen's five blocks cannot span TP6."""
        pro = {name: (width, dest) for name, width, _, _, _, dest in package.shard_profiles('dsv4p', 'spark')}
        self.assertEqual(pro['tp6-width512'], (512, [f'tp6-rank{rank}' for rank in range(6)]))
        for geometry in ('glm', 'glmf', 'dsv4f'):
            six = {name: (width, dest) for name, width, _, _, _, dest in package.shard_profiles(geometry, 'spark')
                   if name.startswith('tp6')}
            self.assertEqual(six, {'tp6-width384': (384, [f'tp6-rank{rank}' for rank in range(4)]),
                                   'tp6-width256': (256, ['tp6-rank4', 'tp6-rank5'])})
        qwen = [name for name, *_ in package.shard_profiles('qwen4', 'spark')]
        self.assertFalse([name for name in qwen if name.startswith('tp6')])
        v41 = [name for name, *_ in package.profiles_for_role('spark')]
        self.assertEqual([name for name in v41 if name.startswith('tp6')], ['tp6-width384'])
        coordinator = [name for name, *_ in package.shard_profiles('dsv4p', 'coordinator')]
        self.assertFalse([name for name in coordinator if name.startswith('tp6')])

    def test_fused_input_rotation_only_where_few_tiles_share_a_half(self):
        """Spark V4 Pro <= 768 at m4096, GLM <= 1024 above m256, GLM Flash
        <= 768 above m256 and 1024 at m1024; never V4.1, the coordinator, or
        widths that are not whole 256 tiles."""
        f = package.fused_input_rotation
        self.assertTrue(f('dsv4p', 'spark', 768, 4096) and f('dsv4p', 'spark', 512, 4096))
        self.assertFalse(f('dsv4p', 'spark', 768, 1024) or f('dsv4p', 'spark', 1024, 4096))
        self.assertTrue(f('glm', 'spark', 512, 1024) and f('glm', 'spark', 1024, 4096))
        self.assertFalse(f('glm', 'spark', 640, 4096) or f('glm', 'spark', 512, 256))
        self.assertTrue(all(f('glmf', 'spark', w, c) for w in (512, 768) for c in (1024, 4096)))
        self.assertTrue(f('glmf', 'spark', 1024, 1024))
        self.assertFalse(f('glmf', 'spark', 1024, 4096) or f('glmf', 'spark', 640, 4096)
                         or f('glmf', 'spark', 512, 256) or f('glmf', 'coordinator', 1024, 1024))
        for capacity in (1, 16, 80, 256, 1024, 4096):
            self.assertFalse(f('v41', 'spark', 512, capacity) or f('dsv4p', 'coordinator', 768, capacity))

    def test_warp_specialized_spark_prefill_policy_is_unchanged(self):
        """Existing Spark widths retain WS; unrelated coordinator families do not."""
        w = package.warp_specialized
        for geometry in ('dsv4p', 'glm', 'glmf'):
            for width in (512, 640, 768, 1024, 1536):
                self.assertEqual([w(geometry, 'spark', width, c) for c in (1, 16, 80, 256, 1024, 4096)],
                                 [False, False, False, True, True, True])
                if geometry != 'glm' or width != 1024:
                    self.assertFalse(any(w(geometry, 'coordinator', width, c) for c in (256, 1024, 4096)))
        for geometry in ('v41', 'dsv4f', 'qwen4'):
            self.assertFalse(any(w(geometry, 'spark', 512, c) for c in (256, 1024, 4096)))

    def test_glm_coordinator_ws_only_for_measured_tiers_and_capacities(self):
        for bits in ((4, 5), (3, 4), (3, 4, 5)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                _, _, calls, _ = self.build_fixture(
                    Path(temporary), 'coordinator', bits, geometry='glm',
                    capacities=(80, 256, 1024, 2752, 4096))
                for call in calls:
                    width, capacity = call.args[1], call.args[3]
                    ws = bits == (4, 5) and (
                        (width == 1024 and capacity in (256, 1024, 2752))
                        or (width == 2048 and capacity == 256))
                    wire = ws and width == 1024 and capacity == 256
                    self.assertEqual(call.kwargs.get('warp_specialized', False), ws)
                    self.assertEqual(call.kwargs.get('wire_input', False), wire)
                    self.assertEqual(call.kwargs.get('ws_input_stages'), 3 if wire else None)
                    self.assertEqual(call.kwargs.get('ws_dynamic_tiles', False), wire)
                    self.assertEqual('+ws-' in str(call.args[0]), ws)
                    if ws:
                        self.assertEqual(call.kwargs['route_block'], 64)
                        self.assertNotIn('token_major_rotation', call.kwargs)
                        self.assertIn('-wire-s3-dynamic' if wire else '-bf16-static', str(call.args[0]))
        for width in (256, 384, 512, 640, 768, 1536):
            self.assertFalse(package.warp_specialized('glm', 'coordinator', width, 256))
        for capacity in (1, 16, 80, 128, 512, 2048, 8192):
            self.assertFalse(package.warp_specialized('glm', 'coordinator', 1024, capacity))

    def test_kimi_ws64_only_for_qualified_ordinary_tp6_prefill(self):
        capacities = (1, 16, 80, 256, 1024, 2048, 4096)
        self.assertEqual([package.warp_specialized('kimik3', 'spark', 512, c)
                          for c in capacities], [False] * 4 + [True] * 3)
        for width in (384, 768, 1024, 1536, 3072):
            self.assertFalse(any(package.warp_specialized('kimik3', 'spark', width, c)
                                 for c in capacities))
        self.assertFalse(any(package.warp_specialized('kimik3', 'coordinator', 512, c)
                             for c in capacities))
        self.assertFalse(any(package.compact_warp_specialized('kimik3', 'spark', 512, c)
                             for c in capacities))
        with tempfile.TemporaryDirectory() as temporary:
            _, _, calls, _ = self.build_fixture(
                Path(temporary), 'spark', (2, 3), geometry='kimik3',
                capacities=(80, 256, 1024, 4096), profiles=('tp6-width512',))
            for call in calls:
                capacity = call.args[3]
                self.assertEqual(call.kwargs['activation'], 'situ')
                if capacity >= 1024:
                    self.assertTrue(call.kwargs['warp_specialized'])
                    self.assertEqual(call.kwargs['route_block'], 64)
                    self.assertIn('+ws-m64/', str(call.args[0]))
                else:
                    self.assertNotIn('warp_specialized', call.kwargs)
                    self.assertNotIn('+ws-', str(call.args[0]))

    def test_decode_capacities_ship_the_canonical_tile(self):
        """A decode capacity the B12x planner widens is pinned to m1's tile (#309).

        The widened (128,128,128,128) export is a different K partition, so the
        decode plan cannot admit it and rows in its range fall through to the next
        capacity: V4.1's m16 is the capacity in that window. The pin is per width
        (the tile is a whole 256-wide rank slice or 128) and only where the
        geometry is enabled.
        """
        d = package.decode_tile
        for width, expected in ((2304, (64, 256, 64, 256)), (1152, (64, 128, 64, 128)),
                                (768, (64, 256, 64, 256)), (640, (64, 128, 64, 128))):
            with self.subTest(width=width):
                self.assertEqual(d('v41', width, 16), expected)
        for capacity in (1, 9, 33, 80, 256, 4096):
            self.assertIsNone(d('v41', 2304, capacity))
        for geometry in ('dsv4f', 'dsv4p', 'glm', 'glmf', 'qwen4'):
            self.assertIsNone(d(geometry, 512, 16))

    def test_explicit_tile_overrides_the_decode_pin_without_leaking_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, manifest, calls, _ = self.build_fixture(
                root, 'spark', (2, 3), capacities=(1, 16, 80),
                tiles=['tp3-width768=16:64,128,64,128'])
            requested = {v['directory'] for v in manifest['variants'] if 'tile_requested' in v}
            self.assertEqual(requested, {f'tp3-rank{rank}/m16' for rank in range(3)})
            for variant in manifest['variants']:
                if variant['directory'] in requested:
                    self.assertEqual(variant['tile_requested'], [64, 128, 64, 128])
                    self.assertEqual(variant['tile'], variant['tile_requested'])
                elif variant['capacity'] == 16:
                    self.assertEqual(variant['tile'], list(self.pinned_tile(variant)))
            for call in calls:
                if call.args[3] == 16:
                    width = call.args[1]
                    tile = ((64, 128, 64, 128) if width == 768 else
                            package.decode_tile('v41', width, 16))
                    self.assertEqual(call.kwargs['tile'], tile)
                    self.assertIn('+tile' + '-'.join(map(str, tile)), str(call.args[0]))
            self.assertEqual(package.verify(output, 'fixture-revision'), manifest)

    def test_route_blocks_widen_only_for_dsv4p_prefill(self):
        """V4.1 and V4 Flash keep 8-row blocks; V4 Pro prefill widens (Spark and RTX)."""
        for capacity in (1, 16, 80, 256, 1024, 4096):
            self.assertEqual(package.route_block('v41', capacity), 8)
            self.assertEqual(package.route_block('dsv4f', capacity), 8)
        self.assertEqual([package.route_block('dsv4p', c) for c in (1, 16, 80, 256, 1024, 4096)],
                         [8, 8, 8, 8, 16, 64])
        self.assertEqual([package.token_major_rotation(g, c) for g in ('v41', 'dsv4f', 'dsv4p')
                          for c in (1024, 4096)], [False, False, False, False, False, True])

    def test_qwen4_uneven_blocks_and_route_policy(self):
        """Qwen's five H128 blocks use native unequal TP2 and Spark shards."""
        coordinator = {name: (width, dest) for name, width, _, _, _, dest
                       in package.shard_profiles('qwen4', 'coordinator')}
        self.assertEqual(coordinator, {'rtx-tp1': (640, ['rtx-tp1']),
                                      'rtx-tp2-rank0': (384, ['rtx-tp2-rank0']),
                                      'rtx-tp2-rank1': (256, ['rtx-tp2-rank1'])})
        self.assertEqual(package.parse_requested_layouts(['rtx-tp2-rank0', 'rtx-tp2-rank1'],
                                                        'coordinator', 'qwen4'),
                         ['rtx-tp2-rank0', 'rtx-tp2-rank1'])
        spark = {name: (width, dest) for name, width, _, _, _, dest in package.shard_profiles('qwen4', 'spark')}
        self.assertEqual(spark['tp1-width640'], (640, ['tp1-rank0']))
        tp1 = next(p for p in package.shard_profiles('qwen4', 'spark') if p[0] == 'tp1-width640')
        self.assertEqual(tp1[2:5], (512, 10, 'bf16'))
        self.assertEqual(package.parse_requested_layouts(['tp1-rank0'], 'spark', 'qwen4'), ['tp1-rank0'])
        self.assertEqual(spark['tp4-width256'], (256, ['tp4-rank0']))
        self.assertEqual(spark['tp4-width128'], (128, ['tp4-rank1', 'tp4-rank2', 'tp4-rank3']))
        self.assertEqual(spark['tp3-width256'], (256, ['tp3-rank0', 'tp3-rank1']))
        self.assertEqual(spark['tp2-width384'], (384, ['tp2-rank0']))
        self.assertTrue(all(topk == 10 and experts == 512
                            for _, _, experts, topk, _, _ in package.shard_profiles('qwen4', 'spark')))
        self.assertEqual([package.route_block('qwen4', c) for c in (1, 16, 80, 256, 1024, 4096)],
                         [8, 8, 8, 32, 32, 32])
        self.assertIsNone(package.swiglu_limit('qwen4'))

    def test_qwen4_coordinator_requires_both_native_tp2_widths(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, calls, capacities = self.build_fixture(
                Path(temporary), 'coordinator', (4, 5),
                require=['rtx-tp2-rank0', 'rtx-tp2-rank1'], geometry='qwen4')
            for rank, width in enumerate((384, 256)):
                variants = [v for v in manifest['variants']
                            if v['directory'].startswith(f'rtx-tp2-rank{rank}/')]
                self.assertEqual({v['capacity'] for v in variants}, set(capacities))
                self.assertTrue(all(v['intermediate'] == width and v['output_dtype'] == 'fp32'
                                    for v in variants))
            package.verify(output)

    def test_verify_cross_checks_the_recorded_route_block(self):
        """A variant's route block must be the one its export compiled."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_fixture(root, 'spark', (2, 3))
            manifest = json.loads((root / 'package/manifest.json').read_text())
            self.assertFalse(any('route_block' in v for v in manifest['variants']),
                             'V4.1 manifests keep their 8-row records unchanged')
            manifest['variants'][0]['route_block'] = 32
            (root / 'package/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError) as caught:
                package.verify(root / 'package', 'fixture-revision')
            self.assertIn('route block mismatch', str(caught.exception))

    def test_verify_rejects_malformed_tile_fields_even_when_a_planner_is_absent(self):
        """A recorded tile is data, so it has to be shaped like one.

        Runs without the pinned planner: the check protects every package, and v9
        manifests are exempt only because they carry neither key.
        """
        broken = {
            'zeros': [0, 0, 0, 0],
            'negative': [-1, 64, 64, 64],
            'short': [64, 256, 64],
            'long': [64, 256, 64, 256, 64],
            'string': '64,256,64,256',
            'floats': [64.0, 256.0, 64.0, 256.0],
            'booleans': [True, True, True, True],
        }
        for field in ('tile', 'tile_requested'):
            for label, value in broken.items():
                if field == 'tile_requested' and label in ('zeros', 'short'):
                    continue  # covered by the tile column; same validation path
                with self.subTest(field=field, value=label):
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        self.build_fixture(root, 'spark', (2, 3),
                                           tiles=['tp3-width768=80:64,256,64,256'])
                        manifest_path = root / 'package/manifest.json'
                        manifest = json.loads(manifest_path.read_text())
                        target = next(v for v in manifest['variants']
                                      if v['directory'] == 'tp3-rank0/m80')
                        target[field] = value
                        manifest_path.write_text(json.dumps(manifest))
                        with self.assertRaises(ValueError) as caught:
                            package.verify(root / 'package', 'fixture-revision')
                        self.assertIn(f'{field} must be four positive integers',
                                      str(caught.exception))

    def test_planner_import_failure_cleans_only_b12x(self):
        """A failed planner import must not unload the shared dependency graph.

        Deleting `b12x` from `sys.modules` is not enough to force the failure path
        in a full-suite run, since the pin is importable from elsewhere; a meta-path
        blocker makes it deterministic. Unloading torch alongside a half-imported
        pin is exactly what corrupted later torch imports, so that is asserted too.
        """
        class BlockB12x:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'b12x' or fullname.startswith('b12x.'):
                    raise ImportError(f'blocked for {self.name}')
                return None
            name = 'b12x-blocker'

        cached = {name: module for name, module in list(sys.modules.items())
                  if name == 'b12x' or name.startswith('b12x.')}
        torch_before = sys.modules.get('torch')
        path_snapshot = list(sys.path)
        blocker = BlockB12x()
        sys.meta_path.insert(0, blocker)
        for name in cached:
            del sys.modules[name]
        try:
            with tempfile.TemporaryDirectory() as temporary:
                self.assertIsNone(_load_planner_tile(source=Path(temporary) / 'absent'))
            leaked = {name for name in sys.modules if name == 'b12x' or name.startswith('b12x.')}
            self.assertEqual(leaked - set(cached), set(), 'half-imported b12x modules leaked')
            self.assertIs(sys.modules.get('torch'), torch_before,
                          'torch and other shared deps must survive a failed pin import')
            self.assertEqual(sys.path, path_snapshot, 'sys.path must be restored verbatim')
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(cached)

    def test_a_v9_package_without_tile_records_still_verifies(self):
        """Backward compatibility: the tile check is conditional on the key existing."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_fixture(root, 'spark', (3, 4))
            manifest_path = root / 'package/manifest.json'
            manifest = json.loads(manifest_path.read_text())
            for variant in manifest['variants']:
                variant.pop('tile', None)
                variant.pop('tile_requested', None)
            manifest_path.write_text(json.dumps(manifest))
            verified = package.verify(root / 'package', 'fixture-revision')
            self.assertEqual(len(verified['variants']), len(manifest['variants']))

    def test_verify_catches_a_contract_that_outruns_the_package(self):
        """A declared capacity with no variant for a required rank is an error.

        Deriving the capacity set from the variants alone would let a build that
        silently lost one capacity for every rank verify cleanly, so the recorded
        capacities drive the cross-product check.
        """
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, _, capacities = self.build_fixture(
                Path(temporary), 'spark', (2, 3), require=['tp3-rank0'])
            self.assertEqual(manifest['requested_capacities'], list(capacities))
            inflated = dict(manifest, requested_capacities=[*capacities, 999])
            (output / 'manifest.json').write_text(json.dumps(inflated))
            with self.assertRaises(ValueError) as caught:
                package.verify(output)
        self.assertIn('tp3-rank0/m999', str(caught.exception))

    def test_tile_override_is_capacity_scoped_and_directory_keyed(self):
        """An A/B retargets one capacity without flattening the policy at others.

        The m16 geometry is where the pinned policy already differs, so scoping is
        what keeps an override honest; the override also must not reuse the
        policy-keyed export directory.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, manifest, calls, _ = self.build_fixture(
                root, 'spark', (2, 3), tiles=['tp3-width768=80:64,256,64,256'])
            overridden = [call for call in calls if call.args[1] == 768 and 'tile' in call.kwargs]
            self.assertEqual([call.args[3] for call in overridden], [80])
            self.assertEqual(overridden[0].kwargs['tile'], (64, 256, 64, 256))
            by_directory = {v['directory']: v for v in manifest['variants']}
            self.assertEqual(by_directory['tp3-rank0/m80']['tile'], [64, 256, 64, 256])
            self.assertEqual(by_directory['tp3-rank0/m80']['tile_requested'], [64, 256, 64, 256])
            # Every variant records the tile the compiler resolved, so an override
            # is distinguished by the *request*, not by a missing key elsewhere.
            self.assertNotIn('tile_requested', by_directory['tp3-rank0/m1'])
            requested = {v['directory'] for v in manifest['variants'] if 'tile_requested' in v}
            self.assertEqual(requested, {f'tp3-rank{rank}/m80' for rank in range(3)})
            # Every capacity keeps its own export directory; the override does not
            # flatten the policy at the others.
            self.assertEqual({v['directory'] for v in manifest['variants']
                              if v['directory'].startswith('tp3')},
                             {f'tp3-rank{rank}/m{cap}' for rank in range(3) for cap in (1, 80)})
            staged = root / 'build'
            self.assertTrue((staged / 'tp3-width768+tile64-256-64-256/m80').is_dir())
            self.assertFalse((staged / 'tp3-width768/m80').exists())
            self.assertTrue((staged / 'tp3-width768/m1').is_dir(), 'policy path unchanged')
            self.assertEqual(package.verify(output, 'fixture-revision'), manifest)

    def test_tile_override_rejects_unscoped_requests(self):
        for value, reason in (
                ('tp9-width1=1:64,256,64,256', 'must name a spark profile'),
                ('tp3-width768=99:64,256,64,256', 'not packaged'),
                ('tp3-width768=1:64,256,64', 'four positive tiles'),
                ('tp3-width768:64,256,64,256', 'must name a spark profile')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError) as caught:
                    self.build_fixture(Path(temporary), 'spark', (2, 3), tiles=[value])
            self.assertIn(reason, str(caught.exception))
    def test_coordinator_families_unchanged(self):
        expected = {'rtx-tp1': (2304, 384, 6, 'fp32'),
                    'rtx-tp2': (1152, 384, 6, 'fp32'), 'dspark': (2304, 128, 3, 'bf16')}
        for bits in ((2, 3), (3, 4)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                _, manifest, calls, capacities = self.build_fixture(Path(temporary), 'coordinator', bits)
                self.assert_layouts(manifest, capacities, expected, bits)
                self.assertEqual(len(calls), 3 * len(capacities))


if __name__ == '__main__':
    unittest.main()
