"""Exercise the real launcher without contacting Docker, SSH or a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MOCK = r'''
import json, os, subprocess, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['MOCK_LOG'], 'a') as log:
    log.write(json.dumps([tool, os.environ.get('MOCK_HOST'), args]) + '\n')
if tool == 'ssh':
    while args[0] == '-o': args = args[2:]
    host, *command = args
    env = dict(os.environ, MOCK_HOST=host)
    sys.exit(subprocess.run(command, input=sys.stdin.read(), text=True, env=env).returncode)
if tool == 'nvidia-smi':
    gpu = '0, GPU-00000000-0000-0000-0000-000000000000, 00000000:11:00.0'
    memory = os.environ.get('MOCK_GPU_MIB', '97887')
    if '--query-gpu=memory.total' in args:
        print(memory)
    else:
        print(gpu + (f', {memory}, {memory}' if 'memory.total' in ' '.join(args) else ''))
    sys.exit(0)
if tool == 'docker':
    if args == ['info']: sys.exit(0)
    if args[:2] == ['container', 'inspect']: sys.exit(1)
    if args[:2] == ['image', 'inspect']:
        if '-f' in args:
            print('fixture-engine' if 'org.opencontainers.image.revision' in ' '.join(args)
                  else os.environ['MOCK_REVISION'])
        sys.exit(0)
    if args[:1] == ['run'] and 'plan' in args:
        assert '--gpus' not in args and '--network' in args
        rows = int(args[args.index('--prefill-rows') + 1])
        def value(flag, default=None):
            return args[args.index(flag) + 1] if flag in args else default
        ranks = int(value('--spark-ranks'))
        tp, ep = int(value('--spark-tp', str(ranks))), int(value('--spark-ep', '1'))
        explicit = '--spark-tp' in args or ranks == 3
        gpus = int(value('--rtx', '1'))
        print(json.dumps(dict(spark_ranks=ranks, prefill_unit_rows=rows or 1024,
            prefill_waves=1, prefill_span_rows=rows or 1024, decode_lanes=2,
            version=1, family='deepseek_v41', executable=True, wip_launchable=True,
            launch=dict(family='deepseek_v41'),
            topology=dict(tp=tp, ep=ep,
                  wire="native_ownership" if explicit else "canonical"),
            artifact_requirements=dict(spark_tp_role=None),
            speculator=value('--speculator', 'off'),
            admission=dict(kv_pool_size=value('--kv-pool-size'), memory_reservation=value('--memory-reservation')),
            peers=value('--deployment-peers', '').split(',')[:ranks],
            spark_admission='coordinator_authoritative')))
        sys.exit(0)
    if args[:1] == ['run'] and '--entrypoint' in args:
        entry = args[args.index('--entrypoint') + 1]
        assert '--gpus' not in args and '--network' in args
        if entry == '/bin/cat':
            # Legacy single-family image: read the package manifest directly.
            assert args[-1] == '/opt/cuteafd/lib/exl3/manifest.json'
        else:
            # Multi-family image: a shell probes the tier family first and
            # falls back to the legacy package manifest.
            assert entry == '/bin/sh'
            script = args[args.index('-c') + 1]
            assert 'exl3' in script and 'manifest.json' in script and 'cat' in script
        identity = 'different' if os.environ.get('MOCK_MISMATCH') == os.environ['MOCK_HOST'] else 'matching'
        tp = int(os.environ.get('MOCK_TP', '2'))
        variants = [dict(directory=f'tp{tp}-rank{rank}/m{capacity}', capacity=capacity,
                         intermediate=2304//tp, experts=384, top_k=6, output_dtype='bf16', bits=[2,3])
                    for rank in range(tp) for capacity in [1,16,80,256,1024,4096]]
        if os.environ.get('MOCK_LEGACY') == '1': variants = []
        print(json.dumps(dict(schema='cuteafd.exl3-package.v1', role='spark', variants=variants, compute=[12,1],
                              sparkinfer_revision=os.environ['MOCK_REVISION'], files={identity: {}})))
        sys.exit(0)
raise SystemExit('Unexpected external action: ' + tool + ' ' + repr(args))
'''


class Exl3ReleasePreflightTest(unittest.TestCase):
    def launch(self, *, exl3=True, mismatch=False, spark_count=4,
               reservation='', rtx_gpus='1', extra=(), legacy=False, gpu_mib=97887,
               instance=''):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / 'bin'
            binary.mkdir()
            for tool in ('docker', 'ssh', 'nvidia-smi', 'curl', 'ss'):
                path = binary / tool
                path.write_text(f'#!{sys.executable}\n' + MOCK)
                path.chmod(0o755)
            from launcher_sandbox import stub_cuda_python
            stub_cuda_python(binary, gpu_mib * 2**20)
            snapshot = directory / 'hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277'
            snapshot.mkdir(parents=True)
            (snapshot / 'config.json').write_text(json.dumps(
                {'quantization_config': {'quant_method': 'exl3' if exl3 else 'fp4', 'bits': 2.0}}))
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       HF_HOME=str(directory / 'hf'), MOCK_LOG=str(directory / 'calls.jsonl'),
                       MOCK_REVISION=json.loads((ROOT / 'third_party/sparkinfer.lock.json').read_text())['revision'],
                       MOCK_LEGACY='1' if legacy else '', MOCK_GPU_MIB=str(gpu_mib),
                       MOCK_TP=str(spark_count if spark_count in (2, 3, 6) else 2),
                       MOCK_MISMATCH='dodo' if mismatch else '')
            config = directory / 'release.config'
            config.write_text((ROOT / 'cuteafd.config').read_text()
                              + f'\nSPARK_COUNT={spark_count}\n'
                              + ('EXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto\n' if spark_count in (2, 3, 6) else '')
                              + (f'MEMORY_RESERVATION={reservation}\n' if reservation else '')
                              + (f'INSTANCE={instance}\n' if instance else ''))
            result = subprocess.run(['bash', 'run.sh', '--config', str(config),
                                     '--set', f'RTX_GPUS={rtx_gpus}', *extra,
                                     '--restart' if mismatch else '--dry-run'],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            log = directory / 'calls.jsonl'
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            # Every Docker operation must be read-only inspection or the
            # disposable manifest reader. Never stop or launch a serving process.
            for tool, host, args in calls:
                if tool == 'docker':
                    self.assertTrue(args == ['info'] or args[:2] in
                                    (['image', 'inspect'], ['container', 'inspect']) or
                                    (args[0] == 'run' and ('--entrypoint' in args or 'plan' in args)), (host, args))
            return result, calls

    def test_instance_namespaces_the_coordinator_container(self):
        """INSTANCE must reach the dry-run plan: two agents' cleanups must not
        be able to target each other's coordinator."""
        result, _ = self.launch(instance='own')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('coordinator container: cuteafd-coordinator-own\n', result.stdout)
        plain, _ = self.launch()
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertIn('coordinator container: cuteafd-coordinator\n', plain.stdout)

    def test_matching_packages_pass_dry_run(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Spark EXL3 package: disjoint:', result.stdout)
        self.assertEqual(sum(tool == 'docker' and args[0] == 'run' and 'plan' not in args for tool, _, args in calls), 4)

    def test_compact_two_rank_dry_run_selects_only_two_hosts(self):
        result, calls = self.launch(spark_count=2, rtx_gpus='auto')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Spark ranks: 2; hosts: ostrich,dodo', result.stdout)
        self.assertIn('Spark peers: 10.55.0.1:19441,10.55.0.2:19441\n', result.stdout)
        self.assertNotIn('memory reservation: 32GiB', result.stdout)
        self.assertEqual({host for _, host, _ in calls if host}, {'ostrich', 'dodo'})
        self.assertEqual(sum(tool == 'docker' and args[0] == 'run' and 'plan' not in args for tool, _, args in calls), 2)
        four, _ = self.launch()
        self.assertNotEqual(result.stdout.split('release identity: ')[1],
                            four.stdout.split('release identity: ')[1])

    def test_compact_tp3_normalizes_topology_and_keeps_compressed_proofs(self):
        for explicit in (False, True):
            extra = ('--set', 'SPARK_TP=3', '--set', 'SPARK_EP=1') if explicit else ()
            result, calls = self.launch(spark_count=3, extra=extra)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Spark ranks: 3;', result.stdout)
            self.assertIn('Spark EXL3 package: disjoint:', result.stdout)
            self.assertNotIn('Spark image must advertise V41 expert role:', result.stdout)
            fixed, _ = self.launch(spark_count=3,
                                   extra=(*extra, '--set', 'RTX_EXPERT_LAYERS=20'))
            self.assertEqual(fixed.returncode, 0, fixed.stderr)
            self.assertNotIn('unsupported Spark weight format', fixed.stderr)
            for kwargs, message in (({'legacy': True}, 'lacks required TP3'),):
                failed, _ = self.launch(spark_count=3, **kwargs, extra=extra)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn(message, failed.stderr)

    def test_six_rank_disjoint_package_admission_is_exact_and_fail_fast(self):
        result, calls = self.launch(spark_count=6)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Spark ranks: 6;', result.stdout)
        self.assertEqual(sum(tool == 'docker' and args[0] == 'run' and 'plan' not in args for tool, _, args in calls), 6)
        for kwargs, message in [({'legacy': True}, 'lacks required TP6')]:
            result, _ = self.launch(spark_count=6, **kwargs)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def test_spark_count_preserves_prefill_cli_override(self):
        for requested, expected, capacity in [('4096', '4096', '4096'), ('80', '80', '256')]:
            result, _ = self.launch(spark_count=2,
                                    extra=('--set', f'PREFILL_BATCH_TOKENS={requested}'))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f'prefill batch tokens: {expected}; expert capacity: {capacity}', result.stdout)
        four, calls = self.launch()
        self.assertIn('prefill batch tokens: 1024; expert capacity: 1024', four.stdout)
        plan = next(args for tool, _, args in calls if tool == 'docker' and 'plan' in args)
        self.assertEqual(plan[plan.index('--prefill-rows') + 1], '0')

    def test_compact_accepts_nominal_32gib_board_with_smaller_usable_total(self):
        result, _ = self.launch(spark_count=2, gpu_mib=32607)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('memory reservation: 32GiB', result.stdout)

    def test_compact_legacy_disjoint_package_rejected_before_restart(self):
        result, _ = self.launch(spark_count=2, legacy=True, mismatch=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('lacks required TP2', result.stderr)

    def test_two_rank_packages_allow_memory_reservations(self):
        for reservation in ('33GiB', '90%'):
            result, _ = self.launch(spark_count=2, reservation=reservation)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_mismatch_rejects_restart_before_stopping_services(self):
        result, _ = self.launch(mismatch=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('Spark EXL3 packages differ across hosts', result.stderr)

    def test_full_model_skips_exl3_package_reads(self):
        result, calls = self.launch(exl3=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(tool == 'docker' and args[0] == 'run' and 'plan' not in args for tool, _, args in calls))


if __name__ == '__main__':
    unittest.main()
