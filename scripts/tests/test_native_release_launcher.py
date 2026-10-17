from __future__ import annotations

import os
import subprocess
import json
import shlex
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / 'scripts' / 'launch' / 'run-family.sh'


def v41_source() -> str:
    """DeepSeek V4.1's launch: run-family.sh's launch_deepseek_v41 section."""
    text = FAMILY.read_text()
    start = text.index('launch_deepseek_v41() {\n')
    return text[start:text.index('\n}\n', text.index('echo "  context/output:', start)) + 3]


class NativeReleaseLauncherTest(unittest.TestCase):
    def test_coordinator_capacity_registry_boundary_and_invalid_inputs(self) -> None:
        for total, budget, expected in [('98304', '', 16), ('32768', '', 8), ('98304', '32', 8),
                                        ('98304', '32.001', 16), ('32768', '96', 8)]:
            result = subprocess.run(['bash', '-c',
                'source scripts/lib/release-common.sh; repo_root=$PWD; release_coordinator_profile "$1" "$2"',
                'test', total, budget], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            gib, concurrency = result.stdout.split()
            self.assertEqual(float(gib), min(float(total) / 1024, float(budget) if budget else float('inf')))
            self.assertEqual(int(concurrency), expected)
        for total, budget in [('0', ''), ('NaN', ''), ('98304', 'inf'), ('98304', '0'), ('bad', '')]:
            result = subprocess.run(['bash', '-c',
                'source scripts/lib/release-common.sh; repo_root=$PWD; release_coordinator_profile "$1" "$2"',
                'test', total, budget], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('capacity registry', result.stderr)

    def test_coordinator_capacity_registry_rejects_literal_drift(self) -> None:
        registry = Path('rust/crates/cuteafd-core/src/serving_capacity.rs')
        source = (ROOT / registry).read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / registry).parent.mkdir(parents=True)
            for changed in [source.replace('SMALL_CARD_DEFAULT_CONCURRENCY: u64 = 8;',
                                            'SMALL_CARD_DEFAULT_CONCURRENCY: u64 = 4 * 2;'),
                            source.replace('pub const LARGE_CARD_DEFAULT_CONCURRENCY: u64 = 16;', ''),
                            source + '\npub const SMALL_CARD_DEFAULT_CONCURRENCY: u64 = 8;\n']:
                (root / registry).write_text(changed)
                result = subprocess.run(['bash', '-c',
                    'source scripts/lib/release-common.sh; repo_root="$1"; release_coordinator_profile 98304',
                    'test', str(root)], cwd=ROOT, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('capacity registry', result.stderr)

    def test_concurrency_explicitness_resets_on_every_config_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary) / 'absent.config'
            present = Path(temporary) / 'present.config'
            source = (ROOT / 'cuteafd.config').read_text()
            absent.write_text('\n'.join(line for line in source.splitlines() if not line.startswith('CONCURRENCY=')) + '\n')
            present.write_text(absent.read_text() + 'CONCURRENCY=4\n')
            result = subprocess.run(['bash', '-c',
                'source scripts/lib/release-common.sh; for cfg in "$1" "$2" "$1"; do '
                'release_load_config "$cfg"; printf "%s:%s\\n" "$RELEASE_CONCURRENCY_EXPLICIT" "$CONCURRENCY"; done',
                'test', str(absent), str(present)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ['0:', '1:4', '0:'])

    def test_serving_state_refuses_unwritable_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / 'profile'
            nested.mkdir()
            trace = nested / 'trace.jsonl'
            trace.write_text('{}\n')
            command = ['bash', '-c',
                'source scripts/lib/release-common.sh; release_require_writable_state "$1"',
                'test', str(root)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for path, mode in [(trace, 0o400), (nested, 0o500)]:
                original = path.stat().st_mode & 0o777
                path.chmod(mode)
                try:
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('serving state is not writable by uid', result.stderr)
                    self.assertIn(str(path), result.stderr)
                    self.assertIn('migrate legacy ownership before launching', result.stderr)
                    self.assertEqual(path.stat().st_mode & 0o777, mode)
                finally:
                    path.chmod(original)

    def test_trace_admission_checks_target_not_unrelated_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sibling = root / 'unrelated'
            sibling.mkdir()
            sibling.chmod(0o500)
            trace = root / 'trace.jsonl'
            command = ['bash', '-c',
                'source scripts/lib/release-common.sh; release_require_writable_state --file "$1"',
                'test', str(trace)]
            try:
                for existing in [False, True]:
                    if existing:
                        trace.write_text('{}\n')
                        trace.chmod(0o600)
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                trace.chmod(0o400)
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(trace), result.stderr)
                self.assertEqual(trace.stat().st_mode & 0o777, 0o400)
                trace.chmod(0o600)
                root.chmod(0o500)
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(root), result.stderr)
            finally:
                root.chmod(0o700)
                sibling.chmod(0o700)
        # One shared trace admission for every family (V4.1 included).
        self.assertEqual(FAMILY.read_text().count('release_require_writable_state --file "$trace"'), 1)
        self.assertIn('shared_trace_args', v41_source())

    def test_v41_placement_uses_writable_serving_home(self) -> None:
        source = v41_source()
        self.assertNotIn('--placement-directory', source)
        self.assertNotIn('placement_directory=/run/cuteafd-placement', source)

    def test_serving_identity_is_evaluated_on_each_host(self) -> None:
        for uid, gid, user in [('1000', '1000', 'tj'), ('1001', '1002', 'spark user')]:
            result = subprocess.run(['bash', '-c', '''
set -euo pipefail
source scripts/lib/release-common.sh
id() { case "$1" in -u) printf '%s' "$TEST_UID" ;; -g) printf '%s' "$TEST_GID" ;; -G) printf '%s 44 107' "$TEST_GID" ;; -un) printf '%s' "$TEST_USER" ;; esac; }
export -f id
export -f release_serving_docker_args
bash -c 'release_serving_docker_args; printf "%s\\0" "${RELEASE_SERVING_ARGS[@]}"'
'''], cwd=ROOT, capture_output=True,
                env={**os.environ, 'TEST_UID': uid, 'TEST_GID': gid, 'TEST_USER': user})
            self.assertEqual(result.returncode, 0, result.stderr)
            args = result.stdout.decode().rstrip('\0').split('\0')
            self.assertEqual(args[:2], ['--user', f'{uid}:{gid}'])
            pairs = list(zip(args[2::2], args[3::2]))
            self.assertEqual([value for flag, value in pairs if flag == '--group-add'], ['44', '107'])
            env = dict(value.split('=', 1) for flag, value in pairs if flag == '-e')
            self.assertEqual(env['USER'], user)
            self.assertEqual(env['LOGNAME'], user)
            self.assertEqual(env['HF_HOME'], '/hf')
            self.assertEqual(env['CUTEAFD_BENCH_DIR'], '/run/cuteafd/bench')
            for key in ['HOME', 'TORCH_EXTENSIONS_DIR', 'XDG_CACHE_HOME',
                        'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR']:
                self.assertTrue(env[key].startswith('/tmp/cuteafd-home'))

    def test_storage_run_mount_is_optional_readonly_and_host_local(self) -> None:
        for present in (True, False):
            result = subprocess.run(['bash', '-c', r"""
set -euo pipefail
source scripts/lib/release-common.sh
# Simulate the dedicated directory, even while its socket is absent on restart.
body=$(declare -f release_serving_docker_args)
body=${body/\[\[ -d \/srv\/sparknest\/run \]\]/$TEST_RUN_DIR}
eval "$body"
release_serving_docker_args
printf '%s\0' "${RELEASE_SERVING_ARGS[@]}"
"""], cwd=ROOT, capture_output=True, env={**os.environ,'TEST_RUN_DIR': 'true' if present else 'false'})
            self.assertEqual(result.returncode,0,result.stderr)
            args=result.stdout.decode().rstrip('\0').split('\0')
            mount='type=bind,src=/srv/sparknest/run,dst=/srv/sparknest/run,readonly'
            self.assertEqual(mount in args,present)
            self.assertNotIn('type=bind,src=/srv/sparknest,dst=/srv/sparknest,readonly',args)
            self.assertFalse(any('rdma-counters.sock' in arg for arg in args))

    def test_all_serving_launchers_use_shared_identity_and_nonroot_mounts(self) -> None:
        for source in [v41_source(), FAMILY.read_text()]:
            self.assertIn('$(declare -f release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args)', source)
            self.assertIn('docker run -d "${RELEASE_SERVING_ARGS[@]}"', source)
            self.assertNotIn('/root/', source)
            self.assertNotIn('-e USER=tj', source)
            self.assertIn(':/run/cuteafd/bench', source)
        # Usage state comes from the one shared API/console helper.
        self.assertEqual(FAMILY.read_text().count('-v "$USAGE_DIR:/run/cuteafd/usage"'), 1)
        self.assertIn('shared_api_state', v41_source())
        v41 = v41_source().split("<<'REMOTE' &", 1)[1].split('\nREMOTE', 1)[0]
        self.assertIn('release_serving_docker_args\ncuteafd_cdi_gpu_args all || exit 2\ndocker run -d "${RELEASE_SERVING_ARGS[@]}"', v41)

    def test_v41_worker_capacity_covers_every_coordinator_chunk_and_replay(self) -> None:
        chunks = list(range(80, 4097))
        result = subprocess.run(['bash', '-c',
            'source scripts/lib/release-common.sh; repo_root=$PWD; release_v41_expert_capacity "$@"',
            'test', *map(str, chunks)], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        capacities = list(map(int, result.stdout.splitlines()))
        self.assertEqual(len(capacities), len(chunks))
        for chunk, capacity in zip(chunks, capacities):
            self.assertEqual(capacity, next(rows for rows in [256, 1024, 4096] if rows >= max(chunk, 256, 128)))
        source = v41_source()
        self.assertIn('expert_capacity="$(release_v41_expert_capacity "$PREFILL_BATCH_TOKENS")"', source)
        self.assertIn('--prefill-rows "$count_prefill"', source)
        self.assertIn('encoder_plan="$count_plan"', source)
        self.assertNotIn('--prefill-rows "$expert_capacity"', source)

    def test_count_fallback_precedes_bf16_package_preflight(self) -> None:
        source = (ROOT / 'scripts/launch/run-family.sh').read_text()
        selection = source.index('selected_ranks="')
        reexec = source.index('exec bash "$repo_root/scripts/launch/run-family.sh"', selection)
        package = source.index('printf -v preflight_command')
        self.assertLess(reexec, package)
        self.assertLess(package, source.index('if [[ "$restart" == 1 ]];'))

    def test_v41_capacity_registry_rejects_invalid_chunks_and_syntax_drift(self) -> None:
        registry = Path('rust/crates/cuteafd-core/src/coordinator_programs.rs')
        for value in ['0', '79', '4097', 'x']:
            result = subprocess.run(['bash', '-c',
                'source scripts/lib/release-common.sh; repo_root=$PWD; release_v41_expert_capacity "$1"',
                'test', value], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / registry).parent.mkdir(parents=True)
            source = (ROOT / registry).read_text()
            for changed in [source.replace('V41_LIVE_MIN_ROWS: u32 = 256;', 'V41_LIVE_MIN_ROWS: u32 = 128 * 2;'),
                            source.replace('pub const V41_DECODER_REPLAY_ROWS: u32 = 128;', ''),
                            source + '\npub const V41_PREFILL_MIN_ROWS: u32 = 80;\n']:
                (root / registry).write_text(changed)
                result = subprocess.run(['bash', '-c',
                    'source scripts/lib/release-common.sh; repo_root="$1"; release_v41_expert_capacity 80',
                    'test', str(root)], cwd=ROOT, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('row registry', result.stderr)

    def test_attention_placement_is_strict_before_starting_containers(self) -> None:
        source = (ROOT / 'scripts/launch/run-family.sh').read_text()
        generic = source[source.index('# Validate the name before it is used to identify allocations'):]
        block = generic.split('attention_placement="$(get ATTENTION_PLACEMENT auto)"', 1)[1].split('\ncase "$family" in', 1)[0]
        self.assertLess(generic.index('case "$attention_placement:$family" in'), generic.index('docker run'))
        self.assertIn('--attention-placement "$attention_placement"', source)
        self.assertIn('CUTEAFD_ATTENTION_PLACEMENT=$attention_placement', source)
        for family in ['glm5', 'glm5_flash', 'deepseek_v4', 'mimo_v2', 'qwen4']:
            for mode in ['auto', 'heads', 'context', 'layers', 'bad']:
                for wip_slot in ['', 'slot-test']:
                    result = subprocess.run(['bash', '-c',
                        'source scripts/lib/release-common.sh; release_known_key ATTENTION_PLACEMENT; '
                        'family="$1"; attention_placement="$2"; wip_slot="$3"; ' + block,
                        'test', family, mode, wip_slot], cwd=ROOT, text=True, capture_output=True)
                    # GLM and V4 run context; Qwen layers requires a Spark-free pair (checked later).
                    runs = mode in ['auto', 'heads'] or (family, mode) in [
                        ('glm5', 'context'), ('glm5_flash', 'context'), ('deepseek_v4', 'context'), ('qwen4', 'layers')]
                    # K4/P7 qualified explicit GLM layers in release and WIP images.
                    runs = runs or (mode == 'layers' and family in ['glm5', 'glm5_flash'])
                    self.assertEqual(result.returncode == 0, runs, result.stderr)
                    if mode == 'layers' and family in ['glm5', 'glm5_flash']:
                        self.assertIn('layers explicitly selected; auto remains context on two GPUs', result.stderr)
                    elif mode in ['context', 'layers'] and not runs:
                        self.assertIn(f'{family} cannot run attention placement {mode}', result.stderr)

    def test_v41_layers_refuses_before_runtime_consumes_solver_placement(self) -> None:
        source = v41_source()
        block = source.split('attention_placement="${ATTENTION_PLACEMENT:-auto}"', 1)[1].split(
            'case "$EMBEDDING" in', 1)[0]
        self.assertLess(source.index('case "$attention_placement" in'), source.index('docker run'))
        for mode in ['auto', 'heads', 'context', 'layers', 'bad']:
            result = subprocess.run(['bash', '-c',
                'source scripts/lib/release-common.sh; attention_placement="$1"; ' + block,
                'test', mode], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode == 0, mode in ['auto', 'heads'], result.stderr)
            if mode in ['context', 'layers']:
                self.assertIn(f'deepseek_v41 cannot run attention placement {mode}', result.stderr)
                self.assertIn('existing ranges stay unchanged', result.stderr)

    def test_table_backend_selectors_are_removed_and_seccomp_is_narrow(self) -> None:
        result = subprocess.run(['bash', '-c',
            'source scripts/lib/release-common.sh; release_known_key TABLE_BACKEND'],
            cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        profile = json.loads((ROOT / 'docker/seccomp-code-bench.json').read_text())
        self.assertEqual(profile['defaultAction'], 'SCMP_ACT_ERRNO')
        allowed = {name for rule in profile['syscalls'] if rule['action'] == 'SCMP_ACT_ALLOW'
                   for name in rule['names']}
        self.assertTrue({'io_uring_setup', 'io_uring_enter', 'io_uring_register'} <= allowed)
        for launcher in ['run.sh', 'scripts/launch/run-family.sh']:
            source = (ROOT / launcher).read_text()
            if launcher == 'run.sh':
                self.assertNotIn('TABLE_BACKEND', source)
                continue
            self.assertNotIn('--table-backend', source)
            self.assertNotIn('TABLE_BACKEND', source)
            self.assertIn('seccomp=$repo_root/docker/seccomp-code-bench.json', source)

    def test_table_accounting_cannot_disable_hybrid_residency(self) -> None:
        for launcher in ['run.sh', 'scripts/launch/run-family.sh']:
            self.assertNotIn('CUTEAFD_TABLE_ACCOUNTING', (ROOT / launcher).read_text())

    def test_matched_nonce_is_opt_in_and_default_argv_is_identical(self) -> None:
        # One shared helper; V4.1's coordinator expands the same array.
        self.assertIn('"${bench_nonce_env_args[@]}"', v41_source())
        self.assertIn('shared_bench_nonce_args', v41_source())
        for source in [FAMILY.read_text()]:
            block = 'bench_nonce_env_args=()\n' + source.split('bench_nonce_env_args=()\n', 1)[1].split('\n', 1)[0]
            self.assertIn('"${bench_nonce_env_args[@]}"', source)
            for seed in [None, '', 'pair-123']:
                environment = dict(os.environ)
                environment.pop('CUTEAFD_BENCH_NONCE_SEED', None)
                if seed is not None:
                    environment['CUTEAFD_BENCH_NONCE_SEED'] = seed
                result = subprocess.run(['bash', '-c', block + '\nprintf "%s\\0" before "${bench_nonce_env_args[@]}" after'], env=environment, capture_output=True)
                self.assertEqual(result.returncode, 0)
                expected = b'before\0after\0' if not seed else b'before\0-e\0CUTEAFD_BENCH_NONCE_SEED=pair-123\0after\0'
                self.assertEqual(result.stdout, expected)

    def test_private_diag_env_reaches_coordinator_only_when_set(self) -> None:
        source = FAMILY.read_text()
        block = 'shared_bench_nonce_args() {\n' + source.split('shared_bench_nonce_args() {\n', 1)[1].split('\n}\n', 1)[0] \
            + '\n}\nshared_bench_nonce_args\nprintf "%s\\0" before "${bench_nonce_env_args[@]}" after'
        environment = {k: v for k, v in os.environ.items() if not k.startswith(('CUTEAFD_DIAG_', 'CUTEAFD_BENCH_NONCE'))}
        result = subprocess.run(['bash', '-c', block], env=environment, capture_output=True)
        self.assertEqual(result.stdout, b'before\0after\0')
        environment.update(CUTEAFD_DIAG_V4_VERIFY='/tmp/x', CUTEAFD_DIAG_A='1', CUTEAFD_OTHER='no')
        result = subprocess.run(['bash', '-c', block], env=environment, capture_output=True)
        self.assertEqual(result.stdout, b'before\0-e\0CUTEAFD_DIAG_A=1\0-e\0CUTEAFD_DIAG_V4_VERIFY=/tmp/x\0after\0')

    def test_v41_speculation_trace_reaches_coordinator_with_host_mount(self) -> None:
        source = FAMILY.read_text()
        # The shared helper body, run with V4.1's family name.
        block = 'family=deepseek_v41\ndeclare -A cfg=()\n[[ -z "${SPECULATION_TRACE:-}" ]] || cfg[SPECULATION_TRACE]="$SPECULATION_TRACE"\n' \
            + 'get() { printf "%s" "${cfg[$1]:-${2:-}}"; }\nkey() { printf "%s" "${cfg[$1]:-${3:-}}"; }\n' \
            + source.split('shared_trace_args() {\n', 1)[1].split('\n}\n', 1)[0] + '\n'
        coordinator = v41_source().split('start_coordinator() {', 1)[1].split('\n}', 1)[0]
        self.assertIn('"${trace_args[@]}"', coordinator)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'trace with spaces' / 'rounds.jsonl')
            for value, valid in [(None, True), ('', True), (path, True), ('relative.jsonl', False)]:
                with self.subTest(value=value):
                    environment = dict(os.environ)
                    environment.pop('SPECULATION_TRACE', None)
                    if value is not None:
                        environment['SPECULATION_TRACE'] = value
                    harness = f'set -euo pipefail\nsource {shlex.quote(str(ROOT / "scripts/lib/release-common.sh"))}\n'
                    result = subprocess.run(['bash', '-c', harness + block +
                        '\nprintf "%s\\0" before "${trace_args[@]}" after'], env=environment, capture_output=True)
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    if valid:
                        # The shared draft policy switches (default off) precede the trace.
                        expected = ['before', '-e', 'CUTEAFD_DRAFT_SURVIVAL_LENGTHS=0']
                        if value:
                            parent = str(Path(value).parent)
                            self.assertTrue(Path(parent).is_dir())
                            expected += ['-v', f'{parent}:{parent}', '-e', f'CUTEAFD_SPECULATION_TRACE={value}']
                        expected += ['after']
                        self.assertEqual(result.stdout, ('\0'.join(expected) + '\0').encode())

    def test_graph_census_is_wip_only_and_default_argv_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / 'bank.txt'
            control.write_text('baseline\n')
            self.assertIn('shared_graph_census_args', v41_source())
            for launcher in ['scripts/launch/run-family.sh']:
                source = (ROOT / launcher).read_text()
                suffix = source.split('graph_census_args=()\n', 1)[1]
                block = 'graph_census_args=()\n' + suffix.split('\nfi', 1)[0] + '\nfi\n'
                self.assertIn('"${graph_census_args[@]}"', source)
                for value, wip, path, valid in [
                    (None, '', str(control), True), ('0', '', str(control), True),
                    ('1', '/slot', str(control), True), ('1', '/slot', '', True),
                    ('1', '', str(control), False), ('1', '/slot', 'relative', False),
                    ('1', '/slot', str(control) + '.missing', False),
                ]:
                    with self.subTest(launcher=launcher, value=value, wip=wip, path=path):
                        env = dict(os.environ)
                        env.pop('CUTEAFD_GRAPH_CENSUS', None)
                        env.pop('CUTEAFD_GRAPH_EAGER', None)
                        if value is not None:
                            env['CUTEAFD_GRAPH_CENSUS'] = value
                        env['CUTEAFD_GRAPH_CENSUS_CONTROL'] = path
                        harness = 'set -euo pipefail\nrelease_die() { exit 2; }\nwip_layout=$1\n'
                        result = subprocess.run(['bash', '-c', harness + block +
                            'printf "%s\\0" before "${graph_census_args[@]}" after', 'test', wip],
                            env=env, capture_output=True)
                        self.assertEqual(result.returncode == 0, valid, result.stderr)
                        if not valid:
                            continue
                        expected = ['before']
                        if value == '1':
                            expected += ['-e', 'CUTEAFD_GRAPH_CENSUS=1', '-e',
                                         'RUST_LOG=info,cuteafd::graph_capture=debug']
                            if path:
                                expected += ['-v', f'{directory}:/run/cuteafd-graph-census:ro', '-e',
                                    'CUTEAFD_GRAPH_CENSUS_CONTROL=/run/cuteafd-graph-census/bank.txt']
                        expected += ['after']
                        self.assertEqual(result.stdout, ('\0'.join(expected) + '\0').encode())

    def test_startup_graph_eager_override_is_wip_only_and_untimed(self) -> None:
        for launcher in ['scripts/launch/run-family.sh']:
            source = (ROOT / launcher).read_text()
            suffix = source.split('if [[ -n "${CUTEAFD_GRAPH_EAGER:-}" ]]; then\n', 1)[1]
            block = 'if [[ -n "${CUTEAFD_GRAPH_EAGER:-}" ]]; then\n' + suffix.split('\nfi', 1)[0] + '\nfi\n'
            for eager, census, wip, valid in [
                (None, None, '', True), ('', None, '', True),
                ('layer_row', None, '/slot', True), ('layer_row', '0', '/slot', True),
                ('layer_row', None, '', False), ('sparse', None, '/slot', False),
                ('layer_row', '1', '/slot', False),
            ]:
                with self.subTest(launcher=launcher, eager=eager, census=census, wip=wip):
                    env = dict(os.environ)
                    for name, value in [('CUTEAFD_GRAPH_EAGER', eager), ('CUTEAFD_GRAPH_CENSUS', census)]:
                        env.pop(name, None)
                        if value is not None:
                            env[name] = value
                    harness = 'set -euo pipefail\nrelease_die() { exit 2; }\nwip_layout=$1\ngraph_census_args=()\n'
                    result = subprocess.run(['bash', '-c', harness + block +
                        'printf "%s\\0" before "${graph_census_args[@]}" after', 'test', wip],
                        env=env, capture_output=True)
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    if valid:
                        expected = ['before'] + (['-e', 'CUTEAFD_GRAPH_EAGER=layer_row'] if eager else []) + ['after']
                        self.assertEqual(result.stdout, ('\0'.join(expected) + '\0').encode())

    def test_embedding_placement_registered_and_gpu_default(self) -> None:
        result = subprocess.run(['bash', '-c',
            'source scripts/lib/release-common.sh; release_known_key EMBEDDING; '
            'release_load_config scripts/fixtures/cuteafd.build-v11.config; printf "%s" "$EMBEDDING"'], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'gpu')
        native = v41_source()
        generic = (ROOT / 'scripts/launch/run-family.sh').read_text()
        self.assertIn('--embedding-placement "$EMBEDDING"', native)
        self.assertIn('family_args=(--embedding-placement "$embedding")', generic)
        self.assertIn('cfg_overrides[EMBEDDING]="$embedding_override"', generic)

    def test_small_card_embedding_default_and_explicit_overrides(self) -> None:
        source = v41_source()
        block = source.split('if python3 -c', 1)[1].split('\nsnapshot_rel=', 1)[0]
        block = 'if python3 -c' + block
        for gib, config_value, override, expected in (
            ('31.8', '', '', 'host'), ('32', '', '', 'host'),
            ('96', '', '', 'gpu'), ('31.8', 'gpu', '', 'gpu'),
            ('31.8', '', 'gpu', 'gpu'), ('96', 'host', '', 'host'),
        ):
            with self.subTest(gib=gib, config=config_value, override=override):
                with tempfile.NamedTemporaryFile(mode='w') as config:
                    if config_value:
                        config.write(f'EMBEDDING={config_value}\n')
                    config.flush()
                    harness = '''set -euo pipefail
release_spark_compact_active() { return 1; }
release_die() { exit 1; }
declare -A cfg_overrides
profile_gib=$1
config=$2
EMBEDDING=${3:-gpu}
cfg_overrides[EMBEDDING]=$4
[[ -z "$4" ]] || EMBEDDING=$4
PREFILL_BATCH_TOKENS=2048
MEMORY_RESERVATION=
RTX_EXPERT_LAYERS=auto
'''
                    result = subprocess.run(['bash', '-c', harness + block + '\nprintf "%s" "$EMBEDDING"',
                        'test', gib, config.name, config_value, override], cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, expected)

    def test_spark_build_arguments_survive_ssh_empty_argument_elision(self) -> None:
        source = (ROOT / 'build.sh').read_text()
        block = source.split('echo "== building Spark development and inference images natively on $seed_host =="', 1)[1]
        invocation, remote = block.split("<<'REMOTE'", 1)
        invocation = invocation.split('  local phase="$1"\n', 1)[1]
        preamble = remote.split('# release-spark-process-group:start', 1)[0].replace('cd "$remote_dir"', ':')
        # The optional source manifest and the optional V41 expert roles are
        # both carried behind non-empty sentinels. An empty earlier value must
        # not shift a later one, because OpenSSH joins argv into one command
        # string and does not preserve an empty argument.
        for digest, roles, jobs, expected in (
            ('', '', '', ['', '', '']),
            ('a' * 64, '', '1', ['a' * 64, '', '1']),
            ('', 'tp2', '2', ['', 'tp2', '2']),
            ('a' * 64, 'tp2;tp3', '4', ['a' * 64, 'tp2;tp3', '4']),
        ):
            with self.subTest(digest=digest, roles=roles, jobs=jobs):
                harness = f'''set -euo pipefail
# The timed SSH leg joins arguments just as OpenSSH does. Preserve the
# empty-argument elision regression while testing the new export phase fields.
timeout() {{ shift 1; [[ $1 != --foreground ]] || shift; "$@"; }}
ssh() {{ shift 1; bash -c "$*"; }}
release_ssh_opts=()
export_timeout=60
export_container=fixture-export
phase=export
seed_host=fixture
remote_dir=/fixture
SPARK_EXPERT_DOCKER_DEV=dev
SPARK_EXPERT_DOCKER_INFERENCE=inference
engine_commit=engine
sparkinfer_commit=fork
release_version=v5
bf16_families=
audio_aot=OFF
spark_export_min_free_gib=90
release_idle_wait_seconds=123
release_export_gpu_poll_seconds=7
release_scope_exports="export CUTEAFD_RELEASE_SCOPE=all; export CUTEAFD_RELEASE_MATRIX_SHA256=fixture"
source_manifest_sha256={shlex.quote(digest)}
spark_tp_roles={shlex.quote(roles)}
native_build_jobs={shlex.quote(jobs)}
'''
                harness += invocation + "<<'REMOTE'" + preamble
                harness += 'printf "%s\\n" "$source_manifest_sha256" "$spark_tp_roles" "$native_build_jobs" "$CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB" "$CUTEAFD_RELEASE_IDLE_WAIT_SECONDS" "$CUTEAFD_RELEASE_EXPORT_GPU_POLL_SECONDS"\nREMOTE\n'
                result = subprocess.run(['bash', '-c', harness],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected + ['90', '123', '7'])

    def test_ssh_config_file_escaped_onto_child_command_line_is_never_reparsed(self) -> None:
        source = (ROOT / 'build.sh').read_text()
        block = source.split('echo "== building Spark development and inference images natively on $seed_host =="', 1)[1]
        invocation, remote = block.split("<<'REMOTE'", 1)
        invocation = invocation.split('  local phase="$1"\n', 1)[1]
        preamble = remote.split('cd "$remote_dir"', 1)[0]
        # An ssh config path chosen for the build must reach ssh as its own argv
        # element and must never be re-spelled inside the command string the remote
        # shell executes: OpenSSH flattens that string, so a value that landed there
        # would be re-parsed by a second shell. The leg is a quoted heredoc, so the
        # only thing the remote shell receives is the positional argument vector.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'cuteafd-release.config'
            config.write_text("Include ~/.ssh/config\n")
            log = root / 'ssh.argv'
            stub_dir = root / 'bin'
            stub_dir.mkdir()
            (stub_dir / 'ssh').write_text(
                '#!/usr/bin/env bash\n'
                'set -euo pipefail\n'
                'opts=(); rest=()\n'
                'while [[ $# -gt 0 ]]; do\n'
                '  case "$1" in\n'
                '    -o|-F|-i|-l|-p) opts+=("$1" "$2"); shift 2 ;;\n'
                '    -*) opts+=("$1"); shift ;;\n'
                '    *) host="$1"; shift; rest+=("$@"); break ;;\n'
                '  esac\n'
                'done\n'
                '{ printf "H\\t%s\\n" "$host"\n'
                '  for token in ${opts[@]+"${opts[@]}"}; do printf "O\\t%s\\n" "$token"; done\n'
                '  for token in ${rest[@]+"${rest[@]}"}; do printf "C\\t%s\\n" "$token"; done; } '
                f'>>"{log}"\n'
                'exec bash -c "${rest[*]}"\n'
            )
            (stub_dir / 'ssh').chmod(0o755)
            harness = f'''set -euo pipefail
source scripts/lib/release-common.sh
export CUTEAFD_RELEASE_SSH_CONFIG={shlex.quote(str(config))}
release_configure_ssh_transport
export_timeout=60
export_container=fixture-export
phase=export
seed_host=fixture
remote_dir=/fixture
SPARK_EXPERT_DOCKER_DEV=dev
SPARK_EXPERT_DOCKER_INFERENCE=inference
engine_commit=engine
sparkinfer_commit=fork
release_version=v5
bf16_families=
audio_aot=OFF
spark_export_min_free_gib=90
release_idle_wait_seconds=123
release_export_gpu_poll_seconds=7
release_scope_exports="export CUTEAFD_RELEASE_SCOPE=all; export CUTEAFD_RELEASE_MATRIX_SHA256=fixture"
source_manifest_sha256=
spark_tp_roles=
'''
            environment = dict(os.environ)
            environment['PATH'] = f"{stub_dir}:{environment['PATH']}"
            result = subprocess.run(
                ['bash', '-c', harness + invocation + "<<'REMOTE'" + preamble
                 + 'printf "%s\\n" "$source_manifest_sha256" "$spark_tp_roles"\nREMOTE\n'],
                cwd=ROOT, text=True, capture_output=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = {'H': [], 'O': [], 'C': []}
            for line in log.read_text().splitlines():
                kind, _, token = line.partition('\t')
                records[kind].append(token)
            self.assertEqual(records['O'], ['-o', 'BatchMode=yes', '-F', str(config)])
            self.assertEqual(records['H'], ['fixture'])
            self.assertEqual(records['C'][:5], ['setsid', '--wait', 'bash', '-s', '--'],
                             'the remote command is what OpenSSH joins into one string')
            self.assertFalse([t for t in records['C'] if str(config) in t],
                             'the config path must not appear in the remote command line')
            self.assertEqual(result.stdout.splitlines(), ['', ''])

    def test_native_api_identity_is_the_launched_checkpoint(self) -> None:
        cases = [('deepseek-ai/DeepSeek-V4.1-Flash', '', 'deepseek-ai/DeepSeek-V4.1-Flash'),
                 ('deepseek-ai/DeepSeek-V4.1-Flash-1017', '', 'deepseek-ai/DeepSeek-V4.1-Flash-1017'),
                 ('wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1', '', 'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1'),
                 ('deepseek-ai/DeepSeek-V4.1-Flash-1017', 'deepseek-ai/DeepSeek-V4.1-Flash', 'deepseek-ai/DeepSeek-V4.1-Flash')]
        for model, served, expected in cases:
            with self.subTest(model=model, served=served), tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / 'cuteafd.config'
                config.write_text(f'MODEL_ID={model}\nMODEL_REVISION={"a" * 40}\n' + (f'SERVED_MODEL_ID={served}\n' if served else ''))
                result = subprocess.run(
                    ['bash', '-c', 'source scripts/lib/release-common.sh; release_load_config "$1" stop >/dev/null; '
                     'printf %s "$RELEASE_NATIVE_API_MODEL_ID"', '_', str(config)],
                    cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def package_identity(self, manifest: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            ['bash', '-c', 'source scripts/lib/release-common.sh; release_exl3_package_identity test-revision'],
            cwd=ROOT, input=json.dumps(manifest), text=True, capture_output=True,
        )

    def test_exl3_preflight_identity_binds_layout_and_package(self) -> None:
        manifest = dict(schema='cuteafd.exl3-package.v1', role='spark',
                        sparkinfer_revision='test-revision', files={'kernel': 'first'})
        disjoint = self.package_identity(manifest)
        self.assertEqual(disjoint.returncode, 0, disjoint.stderr)
        self.assertTrue(disjoint.stdout.startswith('disjoint:'))
        manifest['files']['kernel'] = 'second'
        self.assertNotEqual(disjoint.stdout, self.package_identity(manifest).stdout)
        for field, value in [('role', 'coordinator'), ('schema', 'wrong'),
                             ('sparkinfer_revision', 'another-build')]:
            with self.subTest(field=field, value=value):
                result = self.package_identity({**manifest, field: value})
                self.assertEqual(result.returncode, 2)
                self.assertIn('invalid Spark EXL3 package identity', result.stderr)

    def test_shell_is_valid_and_help_exposes_native_controls(self) -> None:
        subprocess.run(
            ["bash", "-n", "run.sh", "scripts/launch/run-family.sh", "scripts/lib/release-common.sh"],
            cwd=ROOT,
            check=True,
        )
        help_text = subprocess.run(
            ["./run.sh", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        # The V4.1 controls are configuration keys, settable per launch with --set.
        for option in ("--set KEY=VALUE", "--dry-run", "--wip", "--restart", "--embedding-placement"):
            self.assertIn(option, help_text)
        for key in ("RTX_GPUS", "CONCURRENCY", "KV_POOL_SIZE", "MEMORY_RESERVATION", "PREFIX_CACHE_ENTRIES",
                    "MAX_CONTEXT_TOKENS", "MAX_OUTPUT_TOKENS", "PREFILL_BATCH_TOKENS", "SPECULATOR",
                    "DSPARK_DRAFT_LIMIT", "TP2_ATTENTION"):
            self.assertIn(key, help_text)

    def test_standard_release_defaults_are_native(self) -> None:
        script = r'''
source scripts/lib/release-common.sh
release_load_config cuteafd.config
printf '%s\n' "$MODEL_ID" "$MODEL_REVISION" "$EXPERT_FORMAT" "$SPARKINFER_EXL3" \
  "$CONCURRENCY" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" \
  "$MAX_OUTPUT_TOKENS" "$ADDR" "$EXPERT_PORT" "$RTX_GPUS"
'''
        values = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(
            values,
            [
                "deepseek-ai/DeepSeek-V4.1-Flash",
                "dba1be0a40aa45a94ad051997016db3960a90277",
                "native",
                "disable",
                "16",
                "20",
                "1048576",
                "393216",
                "0.0.0.0:8000",
                "19441",
                "auto",
            ],
        )


    def test_launchers_use_cuteafd_container_names(self) -> None:
        combined = (ROOT / "build.sh").read_text() + (ROOT / "run.sh").read_text() + FAMILY.read_text()
        common = (ROOT / "scripts/lib/release-common.sh").read_text()
        self.assertNotIn("ds4rt", combined.lower())
        self.assertIn("cuteafd-coordinator", common)
        self.assertIn("cuteafd-spark-expert", common)
        self.assertIn("expertd-native", combined)
        self.assertIn("serve-native", combined)

    def test_standard_launch_records_runtime_placement(self) -> None:
        script = v41_source()
        self.assertIn('RUST_LOG=${RUST_LOG:-info}', script)
        self.assertIn('--rtx-gpus "$RELEASE_RTX_GPUS"', script)
        self.assertNotIn('--first-layer', script)
        self.assertIn('CUDA_VISIBLE_DEVICES=$gpu_uuid_csv', script)
        self.assertIn("trap cleanup EXIT", script)

    def test_coordinator_release_contains_both_rtx_expert_interfaces(self) -> None:
        script = (ROOT / "scripts/build/build-release-artifacts.sh").read_text()
        self.assertIn('-DCUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT="$coordinator_aot"', script)
        self.assertIn('-DCUTEAFD_ENABLE_V41_TP2_EXPERT_AOT="$coordinator_aot"', script)
        self.assertIn('"cuteafd_v41_local_expert_info"', script)
        self.assertIn('"cuteafd_v41_tp2_expert_info"', script)

    def test_invalid_direct_overrides_fail_before_external_checks(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            env = {**os.environ, "HOME": home, "PATH": "/usr/bin:/bin"}
            for args, message in (
                (["--set", "CONCURRENCY=17"], "CONCURRENCY must be in 1..16"),
                (["--set", "PREFIX_CACHE_ENTRIES=129"], "PREFIX_CACHE_ENTRIES must be in 0..128"),
                (["--set", "MAX_CONTEXT_TOKENS=1048577"], "MAX_CONTEXT_TOKENS must be in 1..1048576"),
                (["--set", "MAX_OUTPUT_TOKENS=393217"], "MAX_OUTPUT_TOKENS must be in 1..393216"),
                (["--set", "RTX_GPUS=3"], "RTX_GPUS must be auto, 1, or 2"),
                (["--set", "DSPARK_DRAFT_LIMIT=8"], "DSPARK_DRAFT_LIMIT must be in 1..7"),
                (["--set", "NOT_A_KEY=1"], "unknown configuration key: NOT_A_KEY"),
                # The former options name the key that replaced them.
                (["--concurrency", "4"], "--set CONCURRENCY=VALUE"),
                (["--dspark-draft-limit", "7"], "--set DSPARK_DRAFT_LIMIT=VALUE"),
                (["--tp2-attention"], "--set TP2_<OPTION>=on|off"),
            ):
                with self.subTest(args=args):
                    # docker/ssh/nvidia-smi are absent from PATH: a launch reaching an
                    # external check would fail on the missing tool, not this message.
                    result = subprocess.run(["./run.sh", *args], cwd=ROOT, capture_output=True, text=True, env=env)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)


class V10BuildTargetTest(unittest.TestCase):
    """The retained v10 build target and the promoted runtime default.

    `cuteafd.build-v10.config` is retained as an explicit historical release
    BUILD target (build.sh derives `release_version` from its coordinator tag).
    The runtime default is now promoted to `v11`, so the retained target
    deliberately differs from `cuteafd.config`; the promoted pair's own equality
    assertion lives in `test_promoted_build_config_matches_the_runtime_default`.

    The default-pair assertions are deliberately derived from `cuteafd.config`
    rather than hardcoded, so promoting the runtime default to a later release
    does not require rewriting this class: only the retained `v10` target below
    and the example-config expectation remain version-pinned.
    """

    BUILD_CONFIG = ROOT / "scripts" / "fixtures" / "cuteafd.build-v10.config"

    def dry_run(self, config: Path | None) -> str:
        args = ["bash", "build.sh"]
        if config is not None:
            args += ["--config", str(config)]
        args += ["--dry-run"]
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def assignments(self, path: Path) -> list[str]:
        return [line for line in path.read_text().splitlines()
                if "=" in line and not line.lstrip().startswith("#")]

    def config_value(self, path: Path, key: str) -> str:
        for line in self.assignments(path):
            name, _, value = line.partition("=")
            if name.strip() == key:
                return value.strip()
        self.fail(f"{key} not found in {path}")


    def test_default_build_reports_the_runtime_default_pair(self) -> None:
        """The default derives its tag from cuteafd.config, whatever it names."""
        default = self.dry_run(None)
        config = ROOT / "cuteafd.config"
        coordinator = self.config_value(config, "COORDINATOR_DOCKER_INFERENCE")
        spark = self.config_value(config, "SPARK_EXPERT_DOCKER_INFERENCE")
        tag = coordinator.rsplit(":", 1)[1]
        self.assertIn(f"release tag: {tag}", default)
        self.assertIn(f"coordinator image: {coordinator}", default)
        self.assertIn(f"spark image: {spark}", default)
        # The promoted default carries the universal role set.
        self.assertIn("tp2;tp3;tp6", default)


    def test_all_examples_use_the_promoted_pair(self) -> None:
        config = ROOT / "cuteafd.config"
        coordinator = self.config_value(config, "COORDINATOR_DOCKER_INFERENCE")
        spark = self.config_value(config, "SPARK_EXPERT_DOCKER_INFERENCE")
        examples = sorted((ROOT / "examples" / "configs").glob("*.config"))
        self.assertTrue(examples, "the example directory must not be empty")
        for path in examples:
            with self.subTest(example=path.name):
                text = path.read_text()
                self.assertIn(f"COORDINATOR_DOCKER_INFERENCE={coordinator}", text)
                self.assertIn(f"SPARK_EXPERT_DOCKER_INFERENCE={spark}", text)


class V11ReleaseBuildTargetTest(unittest.TestCase):
    """The promoted v11 release build target.

    `cuteafd.build-v11.config` selects the `v11` tag for the release build and,
    after promotion, is identical to the runtime default; `run.sh` therefore
    derives the same `v11` pair with or without `--config
    cuteafd.build-v11.config`. That equality is asserted by the promoted
    build-target test above and by `scripts/tests/test_release_helpers.py`,
    owned by the release executor.
    """

    BUILD_CONFIG = ROOT / "scripts" / "fixtures" / "cuteafd.build-v11.config"

    def dry_run(self, config: Path) -> str:
        result = subprocess.run(
            ["bash", "build.sh", "--config", str(config), "--dry-run"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_the_v11_target_reports_v11_and_the_universal_roles(self) -> None:
        if not self.BUILD_CONFIG.is_file():
            self.skipTest("cuteafd.build-v11.config is not present in this checkout")
        v11 = self.dry_run(self.BUILD_CONFIG)
        self.assertIn("release tag: v11", v11)
        self.assertIn("coordinator image: ghcr.io/tpurtell/cuteafd-coordinator:v11", v11)
        self.assertIn("spark image: ghcr.io/tpurtell/cuteafd-spark-expert:v11", v11)
        self.assertIn("tp2;tp3;tp6", v11)


if __name__ == "__main__":
    unittest.main()
