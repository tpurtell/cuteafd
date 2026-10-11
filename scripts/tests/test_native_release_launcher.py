from __future__ import annotations

import os
import subprocess
import json
import shlex
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class NativeReleaseLauncherTest(unittest.TestCase):
    def test_attention_placement_is_strict_before_starting_containers(self) -> None:
        source = (ROOT / 'scripts/launch/run-family.sh').read_text()
        block = source.split('attention_placement="$(get ATTENTION_PLACEMENT auto)"', 1)[1].split('\ncase "$family" in', 1)[0]
        self.assertLess(source.index('case "$attention_placement" in'), source.index('docker run'))
        self.assertIn('--attention-placement "$attention_placement"', source)
        self.assertIn('CUTEAFD_ATTENTION_PLACEMENT=$attention_placement', source)
        for family in ['glm5', 'glm5_flash', 'deepseek_v4', 'mimo_v2', 'qwen4']:
            for mode in ['auto', 'heads', 'context', 'layers', 'bad']:
                result = subprocess.run(['bash', '-c',
                    'source scripts/lib/release-common.sh; release_known_key ATTENTION_PLACEMENT; '
                    'family="$1"; attention_placement="$2"; ' + block,
                    'test', family, mode], cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, mode in ['auto', 'heads'], result.stderr)
                if mode in ['context', 'layers']:
                    self.assertIn(f'{family} cannot run attention placement {mode}', result.stderr)

    def test_table_backend_preflight_and_seccomp_are_narrow(self) -> None:
        for backend, valid in [('mmap', True), ('uring', True), ('mincore-routed', True), ('direct', False)]:
            result = subprocess.run(['bash', '-c',
                'source scripts/lib/release-common.sh; release_known_key TABLE_BACKEND; '
                'release_validate_table_backend "$1"', 'test', backend],
                cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode == 0, valid, result.stderr)
        profile = json.loads((ROOT / 'docker/seccomp-code-bench.json').read_text())
        self.assertEqual(profile['defaultAction'], 'SCMP_ACT_ERRNO')
        allowed = {name for rule in profile['syscalls'] if rule['action'] == 'SCMP_ACT_ALLOW'
                   for name in rule['names']}
        self.assertTrue({'io_uring_setup', 'io_uring_enter', 'io_uring_register'} <= allowed)
        for launcher in ['run.sh', 'scripts/launch/run-family.sh']:
            source = (ROOT / launcher).read_text()
            self.assertIn('--table-backend', source)
            self.assertLess(source.index('release_validate_table_backend'), source.index('docker run'))
            self.assertIn('seccomp=$repo_root/docker/seccomp-code-bench.json', source)

    def test_table_accounting_reaches_coordinator_without_changing_default(self) -> None:
        for launcher in ['run.sh', 'scripts/launch/run-family.sh']:
            source = (ROOT / launcher).read_text()
            block = 'table_env_args=()\n' + source.split('table_env_args=()\n', 1)[1].split('\n', 1)[0]
            self.assertIn('"${table_env_args[@]}"', source)
            for value in [None, '', 'full', 'off']:
                with self.subTest(launcher=launcher, value=value):
                    environment = dict(os.environ)
                    environment.pop('CUTEAFD_TABLE_ACCOUNTING', None)
                    if value is not None:
                        environment['CUTEAFD_TABLE_ACCOUNTING'] = value
                    result = subprocess.run(['bash', '-c', block + '\nprintf "%s\\n" "${table_env_args[@]}"'],
                                            cwd=ROOT, env=environment, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = ['-e', f'CUTEAFD_TABLE_ACCOUNTING={value}'] if value else ['']
                    self.assertEqual(result.stdout.splitlines(), expected)

    def test_matched_nonce_is_opt_in_and_default_argv_is_identical(self) -> None:
        for launcher in ['run.sh', 'scripts/launch/run-family.sh']:
            source = (ROOT / launcher).read_text()
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

    def test_embedding_placement_registered_and_gpu_default(self) -> None:
        result = subprocess.run(['bash', '-c',
            'source scripts/lib/release-common.sh; release_known_key EMBEDDING; '
            'release_load_config scripts/fixtures/cuteafd.build-v11.config; printf "%s" "$EMBEDDING"'], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'gpu')
        native = (ROOT / 'run.sh').read_text()
        generic = (ROOT / 'scripts/launch/run-family.sh').read_text()
        self.assertIn('--embedding-placement "$EMBEDDING"', native)
        self.assertIn('family_args=(--embedding-placement "$embedding")', generic)
        self.assertIn('cfg[EMBEDDING]="$embedding_override"', generic)

    def test_small_card_embedding_default_and_explicit_overrides(self) -> None:
        source = (ROOT / 'run.sh').read_text()
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
declare -A overrides
profile_gib=$1
config=$2
EMBEDDING=${3:-gpu}
overrides[EMBEDDING]=$4
[[ -z "$4" ]] || EMBEDDING=$4
PREFILL_BATCH_TOKENS=2048
MEMORY_RESERVATION=
RTX_EXPERT_LAYERS=auto
'''
                    result = subprocess.run(['bash', '-c', harness + block + '\nprintf "%s" "$EMBEDDING"',
                        'test', gib, config.name, config_value, override], cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, expected)

    def test_explicit_dual_layer_boundary_covers_delegated_experts(self) -> None:
        for layout, layers, expected in [('2', 'auto', '20'), ('2', '17', '17'),
                                         ('2', '1', '1'), ('2', '40', '39'),
                                         ('1', '17', '0'), ('1', '0', '0')]:
            with self.subTest(layout=layout, layers=layers):
                result = subprocess.run(['bash', '-c',
                    'source scripts/lib/release-common.sh; release_spark_first_layer "$1" "$2"',
                    'test', layout, layers], cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)
        result = subprocess.run(['bash', '-c',
            'source scripts/lib/release-common.sh; release_spark_first_layer 2 0'],
            cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

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
            ('', '', '', ['on', '', '', '']),
            ('a' * 64, '', '1', ['on', 'a' * 64, '', '1']),
            ('', 'tp2', '2', ['on', '', 'tp2', '2']),
            ('a' * 64, 'tp2;tp3', '4', ['on', 'a' * 64, 'tp2;tp3', '4']),
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
EXL3_PAIRED_TP4=on
bf16_families=
audio_aot=OFF
source_manifest_sha256={shlex.quote(digest)}
spark_tp_roles={shlex.quote(roles)}
native_build_jobs={shlex.quote(jobs)}
'''
                harness += invocation + "<<'REMOTE'" + preamble
                harness += 'printf "%s\\n" "$exl3_paired_tp4" "$source_manifest_sha256" "$spark_tp_roles" "$native_build_jobs"\nREMOTE\n'
                result = subprocess.run(['bash', '-c', harness],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected)

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
EXL3_PAIRED_TP4=on
bf16_families=
audio_aot=OFF
source_manifest_sha256=
spark_tp_roles=
'''
            environment = dict(os.environ)
            environment['PATH'] = f"{stub_dir}:{environment['PATH']}"
            result = subprocess.run(
                ['bash', '-c', harness + invocation + "<<'REMOTE'" + preamble
                 + 'printf "%s\\n" "$exl3_paired_tp4" "$source_manifest_sha256" "$spark_tp_roles"\nREMOTE\n'],
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
            self.assertEqual(result.stdout.splitlines(), ['on', '', ''])

    def test_native_api_identity_is_independent_of_checkpoint_repository(self) -> None:
        for model, expected in [('deepseek-ai/DeepSeek-V4.1-Flash', 0),
                                ('wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1', 1)]:
            with self.subTest(model=model):
                result = subprocess.run(
                    ['bash', '-c', 'source scripts/lib/release-common.sh; release_native_model_list_matches "$RELEASE_NATIVE_API_MODEL_ID"'],
                    cwd=ROOT, input=json.dumps({'object': 'list', 'data': [{'id': model}]}),
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_paired_build_setting_is_explicit_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'release.config'
            for setting in ('on', 'off', 'invalid'):
                with self.subTest(setting=setting):
                    config.write_text((ROOT / 'cuteafd.config').read_text()
                                      + f'\nEXL3_PAIRED_TP4={setting}\n')
                    result = subprocess.run(
                        ['bash', '-c', 'source scripts/lib/release-common.sh; release_load_config "$1"; printf "%s" "$EXL3_PAIRED_TP4"',
                         'test', str(config)], cwd=ROOT, capture_output=True, text=True)
                    if setting == 'invalid':
                        self.assertEqual(result.returncode, 2)
                        self.assertIn('EXL3_PAIRED_TP4 must be on or off', result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout, setting)

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
        manifest['paired_tp4'] = True
        paired = self.package_identity(manifest)
        self.assertEqual(paired.returncode, 0, paired.stderr)
        self.assertTrue(paired.stdout.startswith('paired:'))
        self.assertNotEqual(disjoint.stdout, paired.stdout)
        manifest['files']['kernel'] = 'second'
        self.assertNotEqual(paired.stdout, self.package_identity(manifest).stdout)
        for field, value in [('paired_tp4', 'true'), ('paired_tp4', None),
                             ('role', 'coordinator'), ('schema', 'wrong'),
                             ('sparkinfer_revision', 'another-build')]:
            with self.subTest(field=field, value=value):
                result = self.package_identity({**manifest, field: value})
                self.assertEqual(result.returncode, 2)
                self.assertIn('invalid Spark EXL3 package identity', result.stderr)

    def test_shell_is_valid_and_help_exposes_native_controls(self) -> None:
        subprocess.run(
            ["bash", "-n", "run.sh", "scripts/lib/release-common.sh"],
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
        for option in (
            "--listen",
            "--rtx-gpus",
            "--concurrency",
            "--kv-pool-size",
            "--memory-reservation",
            "--prefix-cache-entries",
            "--max-context-tokens",
            "--max-output-tokens",
            "--prefill-batch-tokens",
            "--dspark",
            "--no-dspark",
        ):
            self.assertIn(option, help_text)

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
        combined = (ROOT / "build.sh").read_text() + (ROOT / "run.sh").read_text()
        common = (ROOT / "scripts/lib/release-common.sh").read_text()
        self.assertNotIn("ds4rt", combined.lower())
        self.assertIn("cuteafd-coordinator", common)
        self.assertIn("cuteafd-spark-expert", common)
        self.assertIn("expertd-native", combined)
        self.assertIn("serve-native", combined)

    def test_standard_launch_records_runtime_placement(self) -> None:
        script = (ROOT / "run.sh").read_text()
        self.assertIn('RUST_LOG=${RUST_LOG:-info}', script)
        self.assertIn('--rtx-gpus "$RELEASE_RTX_GPUS"', script)
        self.assertIn('--first-layer "$first_layer"', script)
        self.assertIn('CUDA_VISIBLE_DEVICES=$gpu_uuid_csv', script)
        self.assertIn("trap cleanup EXIT", script)

    def test_coordinator_release_contains_both_rtx_expert_interfaces(self) -> None:
        script = (ROOT / "scripts/build/build-release-artifacts.sh").read_text()
        self.assertIn('-DCUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT="$coordinator_aot"', script)
        self.assertIn('-DCUTEAFD_ENABLE_V41_TP2_EXPERT_AOT="$coordinator_aot"', script)
        self.assertIn('"cuteafd_v41_local_expert_info"', script)
        self.assertIn('"cuteafd_v41_tp2_expert_info"', script)

    def test_invalid_direct_overrides_fail_before_external_checks(self) -> None:
        for args, message in (
            (["--concurrency", "17"], "CONCURRENCY must be in 1..16"),
            (["--prefix-cache-entries", "129"], "PREFIX_CACHE_ENTRIES must be in 0..128"),
            (["--max-context-tokens", "1048577"], "MAX_CONTEXT_TOKENS must be in 1..1048576"),
            (["--max-output-tokens", "393217"], "MAX_OUTPUT_TOKENS must be in 1..393216"),
            (["--rtx-gpus", "3"], "RTX_GPUS must be auto, 1, or 2"),
        ):
            result = subprocess.run(
                ["./run.sh", *args], cwd=ROOT, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2)
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
