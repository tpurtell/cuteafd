"""PROFILE_SPARK wraps Spark expert workers in nsys or ncu (rank selection, argv,
file naming, window control, graceful stop), and PROFILE_NCU_SKIP / its Spark
twin skip readiness launches before ncu replays any."""
import importlib.util
import json
from pathlib import Path
import shlex
import socket
import subprocess

import pytest

from test_run_family_dispatch import GLMF_TWO_LAYER, _family_launch_result

ROOT = Path(__file__).resolve().parents[2]
HOST = socket.gethostname().split('.')[0]
SPEC = importlib.util.spec_from_file_location('wip_cards_spark_profiling', ROOT / 'scripts/bench/wip-cards.py')
cards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cards)


def _spark(*args):
    # NUL-separated: the container script is one argv element and contains newlines.
    script = ('source scripts/lib/release-common.sh; release_profile_spark_wrapper "$@"; '
              'printf "W:%s\\0" ${RELEASE_PROFILE_WRAPPER[@]+"${RELEASE_PROFILE_WRAPPER[@]}"}; '
              'printf "D:%s\\0" ${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]+"${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]}"}')
    out = subprocess.run(['bash', '-c', script, 'test', *args], cwd=ROOT, capture_output=True, text=True)

    def take(prefix):
        return [item[len(prefix):] for item in out.stdout.split('\0') if item.startswith(prefix) and len(item) > len(prefix)]

    return out, take('W:'), take('D:')


def _coordinator(*args):
    script = ('source scripts/lib/release-common.sh; release_profile_wrapper "$@"; '
              'printf "W:%s\\0" ${RELEASE_PROFILE_WRAPPER[@]+"${RELEASE_PROFILE_WRAPPER[@]}"}')
    out = subprocess.run(['bash', '-c', script, 'test', *args], cwd=ROOT, capture_output=True, text=True)
    return out, [i[2:] for i in out.stdout.split('\0') if i.startswith('W:') and len(i) > 2]


def _docker_args(*args):
    script = 'source scripts/lib/release-common.sh; release_profile_docker_args "$@"; printf "%s\\n" ${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"}'
    return subprocess.run(['bash', '-c', script, 'test', *args], cwd=ROOT, capture_output=True, text=True)


GRACEFUL = ['--stop-signal', 'SIGINT', '--stop-timeout', '300', '--label', 'cuteafd.graceful-stop=1']


# Arguments: PROFILE_SPARK RANKS PROFILE_DIR HOST RANK WORLD NCU_KERNEL NCU_COUNT NCU_SET NCU_SKIP NSYS_GPU_METRICS.
def test_spark_nsys_wraps_rank_zero_by_default_paused_into_host_rank_file():
    out, wrapper, docker = _spark('nsys', '', '/data/prof', 'ostrich', '0', '6')
    assert out.returncode == 0, out.stderr
    assert wrapper[:2] == ['/bin/bash', '-c'] and wrapper[3:6] == ['release-profile-resolve', 'nsys', 'profile']
    assert 'release_profile_resolve_tool "$tool"' in wrapper[2]
    for flag in ('--start-later=true', '--session-new=cuteafd-profile', '--cuda-graph-trace=node',
                 '--output=/data/prof/nsys-ostrich-rank0', '--force-overwrite=true'):
        assert flag in wrapper
    assert not [f for f in wrapper if f.startswith('--gpu-metrics')]
    assert docker == GRACEFUL
    # Any other rank is not wrapped by default.
    out, wrapper, docker = _spark('nsys', '', '/data/prof', 'dodo', '1', '6')
    assert out.returncode == 0 and wrapper == [] and docker == []


@pytest.mark.parametrize('rank,host,wrapped', [('0', 'ostrich', True), ('1', 'dodo', False), ('3', 'kiwi', True), ('5', 'moa', False)])
def test_spark_ranks_select_exactly_the_listed_workers(rank, host, wrapped):
    out, wrapper, docker = _spark('ncu', '0,3', '/data/prof', host, rank, '6', 'grouped')
    assert out.returncode == 0, out.stderr
    assert bool(wrapper) == wrapped and bool(docker) == wrapped
    if wrapped:
        assert f'/data/prof/ncu-{host}-rank{rank}' in wrapper


def test_spark_ncu_argv_carries_filter_skip_count_and_set():
    out, wrapper, docker = _spark('ncu', '', '/data/prof', 'ostrich', '0', '4', 'exlr8.*grouped', '', '', '')
    assert out.returncode == 0, out.stderr
    assert wrapper[3:5] == ['release-profile-resolve', 'ncu']
    joined = ' '.join(wrapper[4:])
    for part in ('--kernel-name regex:exlr8.*grouped --launch-count 20 --set full', '--graph-profiling node',
                 '--target-processes all', '--export /data/prof/ncu-ostrich-rank0 --force-overwrite'):
        assert part in joined, joined
    assert '--launch-skip' not in wrapper
    assert docker == GRACEFUL
    out, wrapper, _ = _spark('ncu', '0', '/data/prof', 'ostrich', '0', '4', 'gate_up', '5', 'basic', '400')
    assert out.returncode == 0, out.stderr
    assert '--kernel-name regex:gate_up --launch-skip 400 --launch-count 5 --set basic' in ' '.join(wrapper[4:])


def test_spark_nsys_gpu_metrics_reach_the_worker():
    out, wrapper, _ = _spark('nsys', '0', '/data/prof', 'ostrich', '0', '4', '', '', '', '', 'all')
    assert out.returncode == 0, out.stderr
    assert '--gpu-metrics-devices=all' in wrapper


@pytest.mark.parametrize('args,message', [
    (('', '0', '/data/prof', 'ostrich', '0', '4'), 'need PROFILE_SPARK'),
    (('', '', '/data/prof', 'ostrich', '0', '4', 'gemm'), 'need PROFILE_SPARK'),
    (('perf', '', '/data/prof', 'ostrich', '0', '4'), 'PROFILE_SPARK must be nsys or ncu'),
    (('nsys', '', '', 'ostrich', '0', '4'), 'needs an absolute PROFILE_DIR'),
    (('nsys', '', 'rel', 'ostrich', '0', '4'), 'needs an absolute PROFILE_DIR'),
    (('nsys', '', '/data/prof', 'ostrich', '0', '0'), 'needs Spark expert workers'),
    (('nsys', '4', '/data/prof', 'ostrich', '0', '4'), 'names rank 4; this launch has ranks 0..3'),
    (('nsys', '0,0', '/data/prof', 'ostrich', '0', '4'), 'lists rank 0 twice'),
    (('nsys', '0;1', '/data/prof', 'ostrich', '0', '4'), 'comma list of Spark ranks'),
    (('nsys', '01', '/data/prof', 'ostrich', '0', '4'), 'comma list of Spark ranks'),
    (('nsys', '', '/data/prof', 'ostrich', '0', '4', 'gemm'), 'PROFILE_SPARK_NCU_KERNEL/COUNT/SET/SKIP need PROFILE_SPARK=ncu'),
    (('ncu', '', '/data/prof', 'ostrich', '0', '4'), 'PROFILE_SPARK=ncu needs PROFILE_SPARK_NCU_KERNEL'),
    (('ncu', '', '/data/prof', 'ostrich', '0', '4', 'a b'), 'PROFILE_SPARK_NCU_KERNEL must not contain whitespace'),
    (('ncu', '', '/data/prof', 'ostrich', '0', '4', 'g', '0'), 'PROFILE_SPARK_NCU_COUNT must be'),
    (('ncu', '', '/data/prof', 'ostrich', '0', '4', 'g', '', 'Full'), 'PROFILE_SPARK_NCU_SET must be'),
    (('ncu', '', '/data/prof', 'ostrich', '0', '4', 'g', '', '', '-1'), 'PROFILE_SPARK_NCU_SKIP must be'),
    (('ncu', '', '/data/prof', 'ostrich', '0', '4', 'g', '', '', '07'), 'PROFILE_SPARK_NCU_SKIP must be'),
    (('ncu', '', '/data/prof', 'ostrich', '0', '4', 'g', '', '', '', 'all'), 'PROFILE_SPARK_NSYS_GPU_METRICS needs PROFILE_SPARK=nsys'),
    (('nsys', '', '/data/prof', 'ostrich', '0', '4', '', '', '', '', 'all;x'), 'PROFILE_SPARK_NSYS_GPU_METRICS must be'),
])
def test_spark_wrapper_fails_closed(args, message):
    out, _, _ = _spark(*args)
    assert out.returncode != 0 and message in out.stderr, out.stderr


def test_unselected_rank_still_validates_every_setting():
    # Rank 2 is not wrapped, but a bad ncu value still stops the launch on it.
    out, _, _ = _spark('ncu', '0', '/data/prof', 'emu', '2', '4', '')
    assert out.returncode != 0 and 'needs PROFILE_SPARK_NCU_KERNEL' in out.stderr


def test_coordinator_ncu_skip_reaches_launch_skip():
    out, wrapper = _coordinator('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm', '6', 'full', '', '1200')
    assert out.returncode == 0, out.stderr
    assert '--kernel-name regex:gemm --launch-skip 1200 --launch-count 6 --set full' in ' '.join(wrapper[4:])
    assert '--export /data/prof/ncu-raptor --force-overwrite' in ' '.join(wrapper[4:])
    out, wrapper = _coordinator('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm', '', '', '', '0')
    assert out.returncode == 0 and '--launch-skip' not in wrapper
    for profile, skip, message in [('ncu', 'x', 'PROFILE_NCU_SKIP must be'), ('ncu', '1234567890', 'PROFILE_NCU_SKIP must be'),
                                   ('nsys', '5', 'NCU_KERNEL/COUNT/SET/SKIP need PROFILE=ncu'),
                                   ('', '5', 'NCU_KERNEL/COUNT/SET/SKIP need PROFILE=ncu')]:
        out, _ = _coordinator(profile, '/data/prof', 'raptor', 'coordinator', 'gemm' if profile == 'ncu' else '', '', '', '', skip)
        assert out.returncode != 0 and message in out.stderr, (profile, skip, out.stderr)


def test_profile_spark_alone_mounts_the_dir_and_names_the_host():
    out = _docker_args('', '/data/prof', 'kiwi', 'ncu')
    assert out.returncode == 0, out.stderr
    assert out.stdout.split('\n')[:-1] == ['-e', 'CUTEAFD_PROFILE_HOST=kiwi', '-v', '/data/prof:/data/prof',
                                           '-e', 'CUTEAFD_PROFILE_DIR=/data/prof']
    out = _docker_args('', '', 'kiwi', 'nsys')
    assert out.returncode != 0 and 'PROFILE_SPARK=nsys needs an absolute PROFILE_DIR' in out.stderr
    # Unchanged when neither is set.
    assert _docker_args('', '', 'kiwi', '').stdout == '\n'


def test_keys_are_known_and_v41_defaults_reset_them():
    text = (ROOT / 'scripts/lib/release-common.sh').read_text()
    for key in ('PROFILE_NCU_SKIP', 'PROFILE_SPARK', 'PROFILE_SPARK_RANKS', 'PROFILE_SPARK_NCU_KERNEL', 'PROFILE_SPARK_NCU_COUNT',
                'PROFILE_SPARK_NCU_SET', 'PROFILE_SPARK_NCU_SKIP', 'PROFILE_SPARK_NSYS_GPU_METRICS'):
        out = subprocess.run(['bash', '-c', 'source scripts/lib/release-common.sh; release_known_key "$1"', 'test', key], cwd=ROOT)
        assert out.returncode == 0, key
        assert f'\n  {key}=\n' in text, key
        assert key in cards.BUILD_DROP, key


def _launch_lines(result):
    lines = result.stderr.splitlines()
    return next(line for line in lines if 'cuteafd serve-glmf' in line), [line for line in lines if 'cuteafd expertd-native' in line]


def test_family_launch_wraps_only_the_selected_spark_ranks(tmp_path):
    prof = tmp_path / 'prof'
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf',
                                   f'SPARK_COUNT=2\nSPARK_1_HOST=h1\nSPARK_1_LANE_A=10.0.0.2\n'
                                   f'PROFILE_SPARK=nsys\nPROFILE_SPARK_RANKS=1\nPROFILE_DIR={prof}\n')
    assert result.returncode == 0, result.stderr
    coordinator, workers = _launch_lines(result)
    assert len(workers) == 2
    # The coordinator is not wrapped; it still mounts the shared directory.
    assert 'release_profile' not in coordinator and 'graceful-stop' not in coordinator
    assert f'CUTEAFD_PROFILE_DIR={prof}' in coordinator
    rank0, rank1 = (next(w for w in workers if f'--rank {r} ' in w) for r in (0, 1))
    assert 'release_profile' not in rank0 and 'graceful-stop' not in rank0
    assert 'cuteafd.graceful-stop=1' in rank1 and '--stop-signal SIGINT' in rank1
    assert 'release_profile_exec' in rank1 and ' nsys profile ' in rank1
    assert f'--output={prof}/nsys-h1-rank1' in rank1
    # The wrapper sits between the image and the served command.
    assert rank1.index('--force-overwrite=true') < rank1.index('cuteafd expertd-native')
    # Spark push binds the whole SparkNest namespace (#587), not the old /hf/hub alias.
    assert rank1.index(':ro ') < rank1.index('/bin/bash -c ') < rank1.index('cuteafd expertd-native')
    for worker in workers:
        assert f'mkdir -p {prof} && release_require_writable_state {prof} && docker run' in worker


def test_family_launch_profiles_coordinator_and_spark_together(tmp_path):
    prof = tmp_path / 'prof'
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf',
                                   f'PROFILE=ncu,expert-timing\nPROFILE_NCU_KERNEL=gemm\n'
                                   f'PROFILE_NCU_SKIP=300\nPROFILE_SPARK=ncu\nPROFILE_SPARK_NCU_KERNEL=grouped\n'
                                   f'PROFILE_SPARK_NCU_SKIP=900\nPROFILE_SPARK_NCU_COUNT=4\nPROFILE_DIR={prof}\n')
    assert result.returncode == 0, result.stderr
    text = result.stderr
    marker = 'cuteafd serve-glmf'
    start = text.rfind('docker ', 0, text.index(marker))
    coordinator = text[start:text.index('\n', text.index(marker))]
    assert '--launch-skip 300' in coordinator and f'--export {prof}/ncu-{HOST} ' in coordinator
    _, workers = _launch_lines(result)
    rank0 = next(w for w in workers if '--rank 0 ' in w)
    assert '--kernel-name regex:grouped --launch-skip 900 --launch-count 4' in rank0
    assert f'--export {prof}/ncu-h0-rank0' in rank0 and 'CUTEAFD_EXPERT_TIMING=1' in rank0


@pytest.mark.parametrize('keys,message', [
    ('PROFILE_SPARK=ncu\n', 'PROFILE_SPARK=ncu needs an absolute PROFILE_DIR'),
    ('PROFILE_SPARK=ncu\nPROFILE_DIR=/x/prof\n', 'PROFILE_SPARK=ncu needs PROFILE_SPARK_NCU_KERNEL'),
    ('PROFILE_SPARK=nsys\nPROFILE_SPARK_RANKS=5\nPROFILE_DIR=/x/prof\n', 'names rank 5'),
    ('PROFILE_SPARK_RANKS=0\n', 'need PROFILE_SPARK'),
])
def test_family_launch_spark_profile_fails_before_any_worker(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf', keys)
    assert result.returncode != 0 and message in result.stderr, result.stderr
    assert 'cuteafd expertd-native' not in result.stderr


def test_v41_worker_puts_the_spark_wrapper_between_image_and_command():
    source = (ROOT / 'scripts/launch/run-family.sh').read_text()
    invocation, remote = source.split('echo "== starting native Spark experts =="', 1)[1].split("<<'REMOTE' &", 1)
    remote = remote.split('\nREMOTE', 1)[0]
    assert 'release_profile_spark_wrapper "${PROFILE_SPARK:-}" "${PROFILE_SPARK_RANKS:-}" "${PROFILE_DIR:-}" "$host" "$i" "$SPARK_COUNT"' in invocation
    # Unprofiled workers keep the exact pre-existing argv (no wrapper sentinel).
    assert 'if ((${#RELEASE_PROFILE_WRAPPER[@]})); then' in invocation
    positional = ['image', 'name', '0', '107374182400', '19441', 'snapshot', 'fingerprint', '4',
                  '0', '', '', '', '', '', 'info', '__none__']
    tail = ['--profile-args', '-e', 'CUTEAFD_PROFILE_HOST=emu', *GRACEFUL, '--profile-wrapper', '/bin/bash', '-c', 'script', 'x', 'ncu', '--set', 'full']
    result = subprocess.run(['bash', '-euc', 'source scripts/lib/release-common.sh; source scripts/lib/cdi-gpus.sh; docker() { printf "%s\\n" "$@"; }\n'
                             + remote.replace(' >/dev/null', ''), 'test', *positional, *tail], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    image = arguments.index('image')
    assert arguments[image + 1:image + 8] == ['/bin/bash', '-c', 'script', 'x', 'ncu', '--set', 'full']
    assert arguments[image + 8:image + 10] == ['cuteafd', 'expertd-native']
    assert arguments.index('cuteafd.graceful-stop=1') < image and '--profile-wrapper' not in arguments
    # Without a wrapper the worker command is unchanged.
    result = subprocess.run(['bash', '-euc', 'source scripts/lib/release-common.sh; source scripts/lib/cdi-gpus.sh; docker() { printf "%s\\n" "$@"; }\n'
                             + remote.replace(' >/dev/null', ''), 'test', *positional, '--profile-args', '--profile-wrapper'],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index('image') + 1:arguments.index('image') + 3] == ['cuteafd', 'expertd-native']


def test_remote_stop_honours_the_graceful_label(tmp_path):
    fake = tmp_path / 'docker'
    # Only the release container exists; the legacy name is absent.
    fake.write_text('#!/bin/sh\necho "$*" >>"$LOG"\ncase "$1" in container) [ "$3" = cuteafd-spark-expert-kiwi-19441 ];; inspect)\n'
                    '  case "$3" in *Running*) echo true;; *) echo "$LABEL";; esac;; esac\n')
    fake.chmod(0o755)
    for label, expected in [('1', 'stop cuteafd-spark-expert-kiwi-19441'), ('', 'stop -t 30 cuteafd-spark-expert-kiwi-19441')]:
        log = tmp_path / f'log{label}'
        script = ('source scripts/lib/release-common.sh; release_ssh() { shift; "$@"; }; '
                  'release_stop_remote_containers kiwi cuteafd-spark-expert-kiwi-19441 cuteafd-phase0-tcp-expertd-kiwi-19441')
        out = subprocess.run(['bash', '-c', script], cwd=ROOT, capture_output=True, text=True,
                             env={'PATH': f'{tmp_path}:/usr/bin:/bin', 'LOG': str(log), 'LABEL': label})
        assert out.returncode == 0, out.stderr
        calls = log.read_text().splitlines()
        assert expected in calls and 'rm -f cuteafd-spark-expert-kiwi-19441' in calls, calls
        assert calls.index(expected) < calls.index('rm -f cuteafd-spark-expert-kiwi-19441')


@pytest.fixture
def entry(tmp_path):
    base = tmp_path / 'model.config'
    base.write_text('MODEL_ID=test/model\n')
    return dict(name='glm-max', family='glm5', config=str(base), gpus=[0, 1], sparks=['ostrich', 'dodo', 'emu', 'kiwi'], levels=[1],
                set={'SPARK_COUNT': '4', 'ADDR': '0.0.0.0:8500', 'EXPERT_PORT': '19600'}, source_matrix='kit')


def _generate(entry, tmp_path, overrides):
    return cards.generate(entry, 'a', dict(instance='p', slot='s'), tmp_path, tmp_path / 'key', overrides, set(), None, 1)


def test_wip_cards_spark_nsys_window_opens_and_closes_each_selected_rank(entry, tmp_path):
    job = _generate(entry, tmp_path, {'PROFILE_SPARK': 'nsys', 'PROFILE_SPARK_RANKS': '0,3'})
    values = job['entry']['set']
    assert values['PROFILE_DIR'] == str(Path(job['state']) / 'profile') and Path(values['PROFILE_DIR']).is_dir()
    window = job['entry']['window']
    assert window['open']['argv'][:2] == ['sh', '-c'] and window['open']['required'] and window['close']['required']
    for kind, verb in (('open', 'start'), ('close', 'stop')):
        script = window[kind]['argv'][2]
        for host in ('ostrich', 'kiwi'):
            command = shlex.join(['ssh', '-o', 'BatchMode=yes', host, 'docker', 'exec', f'cuteafd-spark-expert-{host}-19600',
                                  '/usr/local/cuda/bin/nsys', verb, '--session=cuteafd-profile'])
            assert command + ' || rc=1;' in script
        assert 'dodo' not in script and 'cuteafd-coordinator' not in script and script.endswith('exit $rc')


def test_wip_cards_window_covers_coordinator_and_spark_together(entry, tmp_path):
    job = _generate(entry, tmp_path, {'PROFILE': 'nsys', 'PROFILE_SPARK': 'nsys'})
    script = job['entry']['window']['open']['argv'][2]
    coordinator = 'cuteafd-coordinator-' + job['entry']['set']['INSTANCE']
    assert script.index(coordinator) < script.index('cuteafd-spark-expert-ostrich-19600')
    # Coordinator alone keeps its single-command window.
    alone = cards.generate(entry, 'b', dict(instance='q', slot='s'), tmp_path, tmp_path / 'key', {'PROFILE': 'nsys'}, set(), None, 1)
    assert alone['entry']['window']['open']['argv'][:2] == ['docker', 'exec']
    # Spark ncu needs no window.
    ncu = cards.generate(entry, 'c', dict(instance='r', slot='s'), tmp_path, tmp_path / 'key',
                         {'PROFILE_SPARK': 'ncu', 'PROFILE_SPARK_NCU_KERNEL': 'grouped'}, set(), None, 1)
    assert 'window' not in ncu['entry']


def test_window_script_attempts_every_session_and_fails_if_any_failed(tmp_path):
    marks = tmp_path / 'marks'
    argv = cards.window_argv([['sh', '-c', f'echo a >>{marks}; exit 1'], ['sh', '-c', f'echo b >>{marks}']])
    assert subprocess.run(argv).returncode == 1
    assert marks.read_text().split() == ['a', 'b']
    marks.unlink()
    argv = cards.window_argv([['sh', '-c', f'echo a >>{marks}'], ['sh', '-c', f'echo b >>{marks}']])
    assert subprocess.run(argv).returncode == 0


@pytest.mark.parametrize('overrides,message', [
    ({'PROFILE_SPARK': 'nsys', 'PROFILE_SPARK_RANKS': '4'}, 'names rank 4; this card has ranks 0..3'),
    ({'PROFILE_SPARK': 'nsys', 'PROFILE_SPARK_RANKS': '1,1'}, 'lists a rank twice'),
    ({'PROFILE_SPARK': 'nsys', 'PROFILE_SPARK_RANKS': 'a'}, 'comma list of Spark ranks'),
])
def test_wip_cards_refuses_bad_spark_ranks(entry, tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _generate(entry, tmp_path, overrides)


def test_wip_cards_spark_nsys_needs_a_sweep(entry, tmp_path):
    with pytest.raises(ValueError, match='capture the timed sweep'):
        _generate({**entry, 'levels': None}, tmp_path, {'PROFILE_SPARK': 'nsys'})


def _sweep_only(job):
    sweep = Path(job['state']) / 'reports' / 'sweep-only' / job['entry']['name'] / 'sweep.json'
    sweep.parent.mkdir(parents=True)
    sweep.write_text(json.dumps({'points': [{'c': 1, 'phase': 'levels', 'aggregate_tok_s': 50.0}]}))


def test_wip_cards_checks_each_rank_report_by_host_and_rank(entry, tmp_path, monkeypatch):
    entry = {**entry, 'profile': 'panels:'}
    job = _generate(entry, tmp_path, {'PROFILE_SPARK': 'ncu', 'PROFILE_SPARK_NCU_KERNEL': 'grouped', 'PROFILE_SPARK_RANKS': '0,2'})
    profile = Path(job['entry']['set']['PROFILE_DIR'])
    _sweep_only(job)
    pulled = []
    monkeypatch.setattr(cards.subprocess, 'run', lambda cmd, **kw: pulled.append(cmd[-2].split(':')[0]))
    row = cards.summarize_job(job, 0)
    assert pulled == ['ostrich', 'dodo', 'emu', 'kiwi']
    assert row['status'] == 'failed' and 'PROFILE_SPARK=ncu produced no ncu-ostrich-rank0.ncu-rep, ncu-emu-rank2.ncu-rep' in row['error']
    (profile / 'ncu-ostrich-rank0.ncu-rep').write_bytes(b'x' * 2**20)
    (profile / 'ncu-emu-rank2.ncu-rep').write_bytes(b'x')
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'pass', row
    assert row['profile']['spark'] == {'tool': 'ncu', 'expected': ['ncu-ostrich-rank0.ncu-rep', 'ncu-emu-rank2.ncu-rep']}
    assert {r['name'] for r in row['profile']['reports']} == {'ncu-ostrich-rank0.ncu-rep', 'ncu-emu-rank2.ncu-rep'}


def test_wip_cards_spark_report_does_not_satisfy_the_coordinator(entry, tmp_path, monkeypatch):
    entry = {**entry, 'profile': 'panels:'}
    job = _generate(entry, tmp_path, {'PROFILE': 'nsys', 'PROFILE_SPARK': 'nsys'})
    profile = Path(job['entry']['set']['PROFILE_DIR'])
    _sweep_only(job)
    monkeypatch.setattr(cards.subprocess, 'run', lambda cmd, **kw: None)
    (profile / 'nsys-ostrich-rank0.nsys-rep').write_bytes(b'x')
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'failed' and 'PROFILE=nsys produced no .nsys-rep report' in row['error']
    (profile / f'nsys-raptor.nsys-rep').write_bytes(b'x')
    assert cards.summarize_job(job, 0)['status'] == 'pass'


def test_wip_cards_requires_a_smoke_binary_with_graceful_worker_stop():
    source = (ROOT / 'scripts/bench/wip-cards.py').read_text()
    assert "(['worker_stop'] if any(job['entry']['set'].get('PROFILE_SPARK') for job in jobs) else [])" in source
    smoke = (ROOT / 'rust/crates/cuteafd-bench/src/smoke.rs').read_text()
    assert 'worker_stop={}' in smoke and 'const WORKER_STOP: &str = "graceful";' in smoke
    assert 'worker_teardown_script(&worker, stop_timeout)' in smoke
