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
            prefill_waves=1, prefill_span_rows=rows or 1024, prefill_reserve_rows=8192, decode_lanes=2,
            version=1, family='deepseek_v41', executable=True, wip_launchable=True,
            launch=dict(family='deepseek_v41'),
            topology=dict(tp=tp, ep=ep,
                  wire="native_ownership" if explicit else "canonical"),
            artifact_requirements=dict(spark_tp_role=None),
            speculator=value('--speculator', 'off'),
            admission=dict(kv_pool_size=value('--kv-pool-size'), memory_reservation=value('--memory-reservation')),
            peers=value('--deployment-peers', '').split(',')[:ranks],
            spark_admission='coordinator_authoritative',
            memory_layout=dict(devices=[dict(kind='rtx', index=i, items=[dict(category='weights', bytes=8 << 30)])
                                        for i in range(gpus)]))))
        sys.exit(0)
raise SystemExit('Unexpected external action: ' + tool + ' ' + repr(args))
'''


class Exl3ReleasePreflightTest(unittest.TestCase):
    def launch(self, *, exl3=False, mismatch=False, spark_count=4,
               reservation='', rtx_gpus='1', extra=(), gpu_mib=97887,
               instance=''):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / 'bin'
            binary.mkdir()
            for tool in ('docker', 'ssh', 'curl', 'ss'):
                path = binary / tool
                path.write_text(f'#!{sys.executable}\n' + MOCK)
                path.chmod(0o755)
            from launcher_sandbox import fake_gpus, fake_nvidia_smi, stub_cuda_python
            fake_nvidia_smi(binary)
            stub_cuda_python(binary, gpu_mib * 2**20)
            snapshot = directory / 'hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277'
            snapshot.mkdir(parents=True)
            (snapshot / 'config.json').write_text(json.dumps(
                {'quantization_config': {'quant_method': 'exl3' if exl3 else 'fp4', 'bits': 2.0}}))
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       HF_HOME=str(directory / 'hf'), MOCK_LOG=str(directory / 'calls.jsonl'),
                       MOCK_REVISION=json.loads((ROOT / 'third_party/sparkinfer.lock.json').read_text())['revision'],
                       FAKE_GPUS=json.dumps(fake_gpus(1, gpu_mib, gpu_mib)))
            config = directory / 'release.config'
            config.write_text((ROOT / 'cuteafd.config').read_text()
                              + f'\nSPARK_COUNT={spark_count}\n'
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

    def test_flash_exl3_refused_before_remote_preflight_or_restart(self):
        for ranks in (2, 3, 4, 6):
            result, calls = self.launch(exl3=True, spark_count=ranks, mismatch=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('unsupported: EXL3 for DeepSeek V4/V4.1 Flash', result.stderr)
            self.assertFalse(any(tool == 'ssh' for tool, _, _ in calls))

    def test_native_two_rank_dry_run_selects_only_two_hosts(self):
        result, calls = self.launch(spark_count=2, rtx_gpus='auto')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Spark ranks: 2; hosts: ostrich,dodo', result.stdout)
        self.assertIn('Spark peers: 10.55.0.1:19441,10.55.0.2:19441\n', result.stdout)
        self.assertNotIn('memory reservation: 32GiB', result.stdout)
        self.assertEqual({host for _, host, _ in calls if host}, {'ostrich', 'dodo'})
        four, _ = self.launch()
        self.assertNotEqual(result.stdout.split('release identity: ')[1],
                            four.stdout.split('release identity: ')[1])

    def test_native_tp3_preserves_implicit_and_explicit_topology(self):
        for explicit in (False, True):
            extra = ('--set', 'SPARK_TP=3', '--set', 'SPARK_EP=1') if explicit else ()
            result, _ = self.launch(spark_count=3, extra=extra)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Spark ranks: 3;', result.stdout)
            fixed, _ = self.launch(spark_count=3,
                                   extra=(*extra, '--set', 'RTX_EXPERT_LAYERS=20'))
            self.assertEqual(fixed.returncode, 0, fixed.stderr)

    def test_native_six_rank_preflight_reaches_every_host(self):
        result, calls = self.launch(spark_count=6)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Spark ranks: 6;', result.stdout)
        self.assertEqual({host for _, host, _ in calls if host},
                         {'ostrich', 'dodo', 'emu', 'kiwi', 'rhea', 'moa'})

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

    def test_native_accepts_nominal_32gib_board_with_smaller_usable_total(self):
        result, _ = self.launch(spark_count=2, gpu_mib=32607)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('memory reservation: 32GiB', result.stdout)

    def test_native_two_rank_allows_memory_reservations(self):
        for reservation in ('33GiB', '90%'):
            result, _ = self.launch(spark_count=2, reservation=reservation)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_model_skips_exl3_package_reads(self):
        result, calls = self.launch(exl3=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(tool == 'docker' and args[0] == 'run' and 'plan' not in args for tool, _, args in calls))


if __name__ == '__main__':
    unittest.main()
