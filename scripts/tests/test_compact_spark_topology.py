"""CPU-only coverage: no real Docker, SSH, GPU, or serving operations."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'runs/v7q-a1/serve-diffbot.sh'


class CompactSparkTopologyTest(unittest.TestCase):
    def config(self, settings, command='printf "%s\\n" "$MEMORY_RESERVATION" "$KV_POOL_SIZE"'):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'config'
            config.write_text((ROOT / 'cuteafd.config').read_text() + '\n' + settings + '\n')
            return subprocess.run(['bash', '-euc',
                'source scripts/lib/release-common.sh; release_load_config "$1"; ' + command,
                'test', str(config)], cwd=ROOT, text=True, capture_output=True)

    def test_planner_wire_is_mandatory(self):
        source = (ROOT / 'scripts/launch/run-family.sh').read_text()
        body = source.split('spark_wire="$(jq -er', 1)[1].split('\nrelease_validate_spark_topology', 1)[0]
        body = 'spark_wire="$(jq -er' + body
        for wire, status, expected in [
            ('canonical', 0, '0::'),
            ('native_ownership', 0, '1:3:1'),
            ('unqualified', 2, None),
            (None, 1, None),
        ]:
            with self.subTest(wire=wire):
                topology = dict(tp=3, ep=1)
                if wire is not None:
                    topology['wire'] = wire
                script = ('source scripts/lib/release-common.sh; count_plan="$1"; '
                          'spark_tp=3; spark_ep=1; SPARK_TP=6; SPARK_EP=1; '
                          + body + '\nprintf "%s:%s:%s" "$topology_explicit" "$SPARK_TP" "$SPARK_EP"')
                result = subprocess.run(['bash', '-euc', script, 'test',
                                         json.dumps(dict(topology=topology))], cwd=ROOT,
                                        text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, status, result.stderr)
                if expected is not None:
                    self.assertEqual(result.stdout, expected)

    def test_blank_count_remains_implicit_across_config_reloads(self):
        result = self.config('SPARK_COUNT=\nSPARK_HOSTS=',
            'release_load_config "$1"; printf "%s %s\\n" "$SPARK_COUNT" "$RELEASE_SPARK_COUNT_EXPLICIT"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '6 0')

    def test_prefill_explicitness_survives_config_reload(self):
        command = ('release_load_config "$1"; printf "%s %s\\n" '
                   '"$PREFILL_BATCH_TOKENS" "$RELEASE_PREFILL_EXPLICIT"')
        implicit = self.config('', command)
        self.assertEqual(implicit.returncode, 0, implicit.stderr)
        self.assertEqual(implicit.stdout.strip(), '2048 0')
        explicit = self.config('PREFILL_BATCH_TOKENS=1024', command)
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(explicit.stdout.strip(), '1024 1')

    def test_count_does_not_set_memory_or_prefill_profiles(self):
        for count in (2, 3, 4, 6):
            for reservation in ('', '31GiB', '80GiB', '50%'):
                result = self.config(f'SPARK_COUNT={count}\nMEMORY_RESERVATION={reservation}',
                    'printf "%s\\n" "$MEMORY_RESERVATION" "$KV_POOL_SIZE" "$PREFILL_BATCH_TOKENS"')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [reservation, '', '2048'])

    def test_prefill_requests_are_preserved_at_every_count(self):
        for count in (2, 3, 4, 6):
            for requested in (80, 128, 256, 2048, 4096):
                result = self.config(f'SPARK_COUNT={count}\nPREFILL_BATCH_TOKENS={requested}',
                                     'printf "%s" "$PREFILL_BATCH_TOKENS"')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, str(requested))
                self.assertNotIn('caps PREFILL_BATCH_TOKENS', result.stderr)

    def test_only_active_hosts_and_rails_are_required(self):
        result = self.config('SPARK_COUNT=2\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto\n'
                             'SPARK_2_HOST=\nSPARK_3_HOST=\nSPARK_2_LANE_A=\nSPARK_3_LANE_A=\n'
                             'SPARK_2_LANE_B=\nSPARK_3_LANE_B=',
                             'release_hosts_csv; release_lane_a_csv; release_lane_b_csv; release_expert_hosts_csv')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ostrich,dodo\n', result.stdout)
        self.assertNotIn('spark-2', result.stdout)

    def test_stop_helpers_scope_to_active_hosts(self):
        # Replace every external operation used by the stop helpers before calling.
        command = '''
release_stop_local_container() { :; }
release_stop_host_api() { :; }
release_stop_remote_containers() { echo "release:$1"; }
release_stop_persistent_local_container() { :; }
release_stop_persistent_remote_container() { echo "persistent:$1"; }
release_stop_services coord spark
release_stop_wip_containers
'''
        result = self.config('SPARK_COUNT=2\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto', command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()),
                         {'release:ostrich', 'release:dodo', 'persistent:ostrich', 'persistent:dodo'})

    def helper(self, *args, **env):
        return subprocess.run(['bash', str(HELPER), *args], cwd=ROOT, text=True,
                              capture_output=True,
                              env=dict(os.environ, DRY_RUN='1', SPARK_COUNT='2', **env))

    @unittest.skipUnless(HELPER.exists(), 'manual WIP helper lives in ignored runs/')
    def test_wip_commands_have_world_two_peers_and_ceiling(self):
        result = self.helper('start-experts', '3')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line[len('DRY RUN:'):]) for line in result.stdout.splitlines()
                    if line.startswith('DRY RUN:')]
        self.assertEqual(len(commands), 2)
        for rank, command in enumerate(commands):
            self.assertIn(f'--rank {rank} --world 2', command[-1])
            self.assertIn('--capacity 256', command[-1])
        result = self.helper('start-coordinator', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.splitlines()[0][len('DRY RUN:'):])[-1]
        self.assertIn('--memory-reservation 32GiB', command)
        self.assertIn('--kv-pool-size 2GiB', command)
        self.assertIn('--prefill-batch-tokens 256', command)
        self.assertIn('--peers 10.55.0.1:19441,10.55.0.2:19441 --rtx-gpus 1', command)
        for args in [('start-coordinator', '2'),
                     ('start-coordinator', '1', '--memory-reservation', '80GiB')]:
            self.assertEqual(self.helper(*args).returncode, 2)
        self.assertEqual(self.helper('start-coordinator', '1', MEMORY_RESERVATION='33GiB').returncode, 2)


    # --- v10 three-rank profiles: implicit compact TP3, explicit native bypass --

    COMPACT3 = 'SPARK_COUNT=3\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto'
    NATIVE3 = ('SPARK_COUNT=3\nEXPERT_FORMAT=native\nSPARKINFER_EXL3=disable\n'
               'SPARK_TP=3\nSPARK_EP=1')

    def test_implicit_counts_are_admitted(self):
        for count in (2, 3):
            for settings in (f'SPARK_COUNT={count}', f'SPARK_COUNT={count}\nRTX_GPUS=2'):
                self.assertEqual(self.config(settings).returncode, 0)
    def test_explicit_native_tp3ep1_bypasses_every_compact_rule(self):
        # The bypass must hold for MEMORY_RESERVATION, KV, prefill and the
        # single-RTX rule: nothing in the compact block may touch this launch.
        result = self.config(self.NATIVE3,
                             'printf "%s\\n" "$MEMORY_RESERVATION" "$KV_POOL_SIZE" '
                             '"$PREFILL_BATCH_TOKENS" "$EXPERT_FORMAT"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['', '', '2048', 'native'])
        result = self.config(self.NATIVE3 + '\nMEMORY_RESERVATION=90%',
                             'printf "%s" "$MEMORY_RESERVATION"')
        self.assertEqual(result.returncode, 0, result.stderr)   # percentages stay legal
        self.assertEqual(result.stdout, '90%')
        result = self.config(self.NATIVE3 + '\nPREFILL_BATCH_TOKENS=4096',
                             'printf "%s" "$PREFILL_BATCH_TOKENS"')
        self.assertEqual(result.returncode, 0, result.stderr)   # no 256 cap
        self.assertEqual(result.stdout, '4096')
        self.assertNotIn('caps PREFILL_BATCH_TOKENS', result.stderr)
    def test_explicit_flash_exl3_topology_is_refused(self):
        result = self.config('SPARK_COUNT=3\nSPARK_TP=3\nSPARK_EP=1\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires EXPERT_FORMAT=native', result.stderr)

    def test_malformed_partial_three_rank_topologies_are_rejected(self):
        for settings, message in [
            ('SPARK_COUNT=3\nSPARK_TP=3', 'set together or omitted together'),
            ('SPARK_COUNT=3\nSPARK_EP=1', 'set together or omitted together'),
            ('SPARK_COUNT=3\nSPARK_TP=3\nSPARK_EP=2', 'must equal SPARK_COUNT'),
            ('SPARK_COUNT=3\nSPARK_TP=1\nSPARK_EP=3', 'unsupported native Spark topology TP1EP3'),
            ('SPARK_COUNT=3\nSPARK_TP=2\nSPARK_EP=1', 'must equal SPARK_COUNT'),
            ('SPARK_COUNT=9', 'SPARK_COUNT must be in 0..8'),
        ]:
            with self.subTest(settings=settings):
                result = self.config(settings)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)

    def test_out_of_range_worlds_are_refused_before_launch(self):
        for count in (9, 10):
            for topology in ('', f'\nSPARK_TP={count}\nSPARK_EP=1'):
                with self.subTest(count=count, topology=topology):
                    result = self.config(f'SPARK_COUNT={count}{topology}')
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('must be in', result.stderr)
        # TP1 and NVFP4 TP5 remain canonical package contracts, not native
        # ownership layouts.
        for count in (1, 5, 7):
            result = self.config(f'SPARK_COUNT={count}\nSPARK_TP={count}\nSPARK_EP=1')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('unsupported native Spark topology', result.stderr)


    def test_daemon_remote_arguments_preserve_world(self):
        source = (ROOT / 'scripts/launch/run-family.sh').read_text()
        invocation, remote = source.split('echo "== starting native Spark experts =="', 1)[1].split("<<'REMOTE' &", 1)
        remote = remote.split('\nREMOTE', 1)[0]
        self.assertIn('"$SPARK_COUNT"', invocation)
        # The legacy ten-positional invocation must survive for both compact
        # worlds: the coordinator keeps the worker at the world it was given.
        for world in (2, 3):
            with self.subTest(world=world):
                result = subprocess.run(['bash', '-euc', 'source scripts/lib/release-common.sh; source scripts/lib/cdi-gpus.sh; HF_HOME="$(mktemp -d)"; trap \'rm -rf "$HF_HOME"\' EXIT; mkdir -p "$HF_HOME/hub" "$HF_HOME/snapshot"; CUTEAFD_SITE_HUB=1; docker() { printf "%s\\n" "$@"; }\n' +
                    remote.replace(' >/dev/null', ''), 'test', 'image', 'name', '1',
                    '107374182400', '19441', 'snapshot', 'fingerprint', str(world), '0', str(world), '1'],
                    cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = result.stdout.splitlines()
                self.assertEqual(arguments[arguments.index('--world') + 1], str(world))
                self.assertNotIn('--first-layer', arguments)
                # A legacy world-2/3 launch carries no explicit topology flags.
                self.assertNotIn('--spark-tp', arguments)
                self.assertNotIn('--spark-ep', arguments)


if __name__ == '__main__':
    unittest.main()
