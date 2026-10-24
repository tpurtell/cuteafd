"""Exercise run.sh's actual startup sequence with process-boundary stubs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

STUB = r'''#!/usr/bin/env python3
import json,os,sys,shlex
from pathlib import Path
args=sys.argv[1:]; tool=Path(sys.argv[0]).name
# SSH receives a shell command string; decode its quoted worker argv.
if tool=='ssh' and args and 'bash -s -- ' in args[-1]:
 args=args[:-1]+shlex.split(args[-1][args[-1].rindex('bash -s -- '):])
with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps([tool,args])+'\n')
if tool=='docker':
 if args[0]=='inspect':print('running')
 elif args[0]=='exec' and 'cat' in args:
  print(Path(os.environ['PLAN']).read_text())
 elif args[0]=='run' and 'plan' in args:
  print(os.environ['ENCODER_PLAN'])
 elif args[0]=='run':print('container-id')
else:
 sys.stdin.read() if '-s' in args else None
'''

class PlacementHandoffTest(unittest.TestCase):
    def run_startup(self, gpus, plan, options=(), spark_count=4, local_layers="auto", encoder=None):
        source=(ROOT/'scripts/launch/run-family.sh').read_text()
        block=source[source.index('# Workers bootstrap first;'):source.index('api_url=')]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'plan').write_text(json.dumps(plan))
            for name in ['docker','ssh','nest']:
                path=root/name;path.write_text(STUB);path.chmod(0o755)
            env=dict(os.environ,HOME=str(root/"home"),PATH=str(root)+os.pathsep+os.environ['PATH'],EVENTS=str(root/'events'),PLAN=str(root/'plan'),
                ENCODER_PLAN=json.dumps(encoder))
            setup=r'''
set -euo pipefail
source scripts/lib/release-common.sh
release_die() { echo "$*" >&2; exit 1; }
RELEASE_RTX_GPUS="$1"
repo_root="$PWD"
coordinator=coordinator
snapshot_rel=model
console_supported=0
CONSOLE_SECRET_FILE=/fake-console-secret
USAGE_DIR=/fake-usage
peers=peer-list
ADDR=127.0.0.1:8000
PREFILL_BATCH_TOKENS=2048
CONCURRENCY=16
PREFIX_CACHE_ENTRIES=20
MAX_CONTEXT_TOKENS=1048576
MAX_OUTPUT_TOKENS=393216
HTTP_QUEUE_WAIT_MS=25000
RTX_EXPERT_LAYERS=auto
HOST_CACHE_BYTES=auto
KV_POOL_SIZE=
MEMORY_RESERVATION=
SPECULATOR=dspark
EMBEDDING=gpu
VISION=off
AUDIO=off
RELEASE_MODEL_REVISION=0123456789012345678901234567890123456789
lanes=(10.55.0.1 10.55.0.2 10.55.0.3 10.55.0.4)
# Mirror the run.sh scope the extracted block relies on: release_load_config
# defaults DSPARK_DRAFT_POLICY, and run.sh's argument parsing always defines
# dspark_draft_limit (empty unless --dspark-draft-limit was passed). Omitting
# either makes the block die on `set -u` before any stub is invoked.
DSPARK_DRAFT_POLICY=adaptive
dspark_draft_limit=
TP2_ATTENTION=off
TP2_QUERY_PROJECTION=off
TP2_OUTPUT_PROJECTION=off
TP2_DSPARK_EXPERTS=off
topology_explicit=0
spark_ep=1
spark_admission=not-applicable
spark_exl3_identity=
gpu_request=device=uuid0,uuid1
gpu_uuid_csv=uuid0,uuid1
fingerprint=test
hf_home=/models
namespace_root=/mnt/sparknest
COORDINATOR_DOCKER_INFERENCE=coordinator-image
SPARK_EXPERT_DOCKER_INFERENCE=spark-image
SPARK_DEVICE_BUDGET_BYTES=1000000
spark_prefix=worker
hosts=(a b c d)
EXPERT_PORT=19441
expert_capacity=4096
wip_layout=
wip_slot=
# Shared launch features (run-family.sh helpers) at their defaults.
bench_dir="$HOME/.cache/cuteafd/bench"
fidelity_cache="$HOME/.cache/cuteafd/fidelity"
api_args=() api_mount_args=() probe_args=() trace_args=() bench_nonce_env_args=() graph_census_args=()
profile_args=() profile_wrapper=() profile_wrapper_args=()
shared_env_args=(-e CUTEAFD_SPARK_INTAKE=auto -e CUTEAFD_CONSOLE_TEXT=true -e CUTEAFD_ATTENTION_PLACEMENT=auto -e CUTEAFD_FP8_EXPERT_PREFILL=auto)
shared_worker_env_args=(-e CUTEAFD_FP8_EXPERT_PREFILL=auto -e CUTEAFD_THERMAL=on)
thermal_args=(-e CUTEAFD_THERMAL=on)
v41_fp8_head_args=()
v41_l2_prefetch=
v41_decode_lanes=
resolved_decode_lanes=2
'''
            if encoder is not None:
                setup+='\nVISION=auto\ncount_plan="$ENCODER_PLAN"\n'
            setup+=f'\nRTX_EXPERT_LAYERS={local_layers}\nSPARK_COUNT={spark_count}\nhosts=("${{hosts[@]:0:SPARK_COUNT}}")\n'
            # Legacy geometry: release_spark_tp defaults to SPARK_COUNT and
            # release_spark_ep to 1 when no explicit topology is configured.
            setup+=f'spark_tp={spark_count}\n'
            if spark_count == 2:
                setup+='MEMORY_RESERVATION=32GiB\nKV_POOL_SIZE=2GiB\npeers=10.55.0.1:19441,10.55.0.2:19441\n'
            setup+=''.join(f'\nTP2_{option}=on\n' for option in options)
            count_plan = {**(encoder or {}), "spark_admission": "coordinator_authoritative"}
            setup += "\ncount_plan=" + __import__("shlex").quote(json.dumps(count_plan)) + "\n"
            result=subprocess.run(['bash','-c',setup+block,'test',str(gpus)],env=env,cwd=ROOT,capture_output=True,text=True,timeout=10)
            events=[json.loads(line) for line in (root/'events').read_text().splitlines()]
            return result,events

    @staticmethod
    def worker_tail(args):
        """Worker values after `--` in a worker-start ssh invocation.

        run.sh passes image, remote, rank, budget, port, snapshot,
        fingerprint, world, explicit_topology, tp, ep and then
        optional RDMA environment values. Index from the `--` separator so the
        optional tail does not shift the assertions.
        """
        tail = args[args.index('--') + 1:]
        # Worker docker options (the shared worker environment, PROFILE switches)
        # follow the positional contract behind a --profile-args sentinel.
        if '--profile-args' in tail:
            options = tail[tail.index('--profile-args') + 1:]
            assert 'CUTEAFD_THERMAL=on' in options, options
            tail = tail[:tail.index('--profile-args')]
        return tail

    def test_every_layout_starts_bootstrap_workers_before_coordinator(self):
        for gpus in (1, 2):
            for local_layers in ("auto", "0", "5", "40"):
                with self.subTest(gpus=gpus, local_layers=local_layers):
                    result, events = self.run_startup(gpus, {}, local_layers=local_layers)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    starts = [(i, args) for i, (tool, args) in enumerate(events)
                              if tool == 'ssh' and '-s' in args]
                    self.assertEqual(len(starts), 4)
                    tails = [self.worker_tail(args) for _, args in starts]
                    self.assertTrue(all(len(tail) == 16 for tail in tails), tails)
                    self.assertTrue(all(tail[7:11] == ['4', '0', '4', '1'] for tail in tails))
                    ready = [i for i, (tool, args) in enumerate(events)
                             if tool == 'ssh' and any('native worker bootstrap ready' in arg for arg in args)]
                    self.assertFalse(any('/dev/tcp' in arg for tool, args in events
                                         if tool == 'ssh' for arg in args))
                    coordinator = next(i for i, (tool, args) in enumerate(events)
                                       if tool == 'docker' and 'serve-native' in args)
                    self.assertEqual(len(ready), 4)
                    self.assertLess(max(i for i, _ in starts), min(ready))
                    self.assertLess(max(ready), coordinator)
                    argv = events[coordinator][1]
                    self.assertIn(f'seccomp={ROOT}/docker/seccomp-code-bench.json', argv)
                    self.assertNotIn('--placement-directory', argv)
                    self.assertFalse(any('plan.json' in ' '.join(args) or 'ready.json' in ' '.join(args)
                                         for _, args in events))

    def test_spark_vision_plan_reaches_only_selected_encoder_rank(self):
        encoder=dict(encoder=dict(kind=dict(kind='spark',rank=2)),encoder_plan_hash='ab'*32)
        result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',
            rtx_expert_layers=20,spark_first_layer=20),encoder=encoder)
        self.assertEqual(result.returncode,0,result.stderr)
        plans=[args for tool,args in events if tool=='docker' and 'plan' in args]
        self.assertEqual(len(plans),0)
        coordinator=next(args for tool,args in events if tool=='docker' and 'serve-native' in args)
        self.assertEqual(coordinator[coordinator.index('--vision-peers')+1],'10.55.0.3:19541')
        self.assertEqual(coordinator[coordinator.index('--encoder-plan-hash')+1],'ab'*32)
        tails=[self.worker_tail(args) for tool,args in events if tool=='ssh' and '-s' in args]
        self.assertEqual(len(tails),4)
        self.assertTrue(all(tail[16:]==['2','19541','ab'*32,
            '0123456789012345678901234567890123456789'] for tail in tails))
        self.assertIn('if [[ "$rank" == "${17:--1}" ]]', (ROOT/'scripts/launch/run-family.sh').read_text())

    def test_compact_starts_exactly_two_workers_and_passes_ceiling(self):
        result,events=self.run_startup(1,dict(version=1,rtx_gpus=1,nonce="fresh",rtx_expert_layers=0,spark_first_layer=0),spark_count=2)
        self.assertEqual(result.returncode,0,result.stderr)
        starts=[args for tool,args in events if tool=='ssh' and '-s' in args]
        self.assertEqual(len(starts),2)
        tails=[self.worker_tail(args) for args in starts]
        self.assertTrue(all(tail[7]=='2' for tail in tails))
        # Legacy geometry tail: explicit flag off, TP=SPARK_COUNT, EP=1.
        self.assertTrue(all(tail[8:11]==['0','2','1'] for tail in tails))
        coordinator=next(args for tool,args in events if tool=='docker' and args[0]=='run')
        self.assertEqual(coordinator[coordinator.index('--peers')+1], '10.55.0.1:19441,10.55.0.2:19441')
        self.assertEqual(coordinator[coordinator.index('--memory-reservation')+1], '32GiB')
        self.assertEqual(coordinator[coordinator.index('--kv-pool-size')+1], '2GiB')

    def test_zero_spark_launch_has_no_remote_workers(self):
        result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',rtx_expert_layers=40,spark_first_layer=39),spark_count=0)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(any(tool=='ssh' for tool,_ in events))
        # No hosts: never `nest drop-caches` without --host (that drops every host's cache).
        self.assertFalse(any(tool=='nest' for tool,_ in events))

    def test_spark_launch_preserves_page_caches(self):
        result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',rtx_expert_layers=20,spark_first_layer=20))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(any(tool=='nest' for tool,_ in events),events)

    def test_cli_tp2_overrides_and_invalid_config(self):
        """The former --tp2-* flags are configuration keys: run.sh --set reaches the
        loader as the file would, and the loader rejects a bad value."""
        source=(ROOT/'scripts/launch/run-family.sh').read_text()
        block=source[source.index('# The release loader validates every key'):source.index('release_validate_tp2_options\n')]
        finish='\nprintf "%s\\n" "$TP2_ATTENTION" "$TP2_QUERY_PROJECTION" "$TP2_OUTPUT_PROJECTION" "$TP2_DSPARK_EXPERTS"\n'
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'recipe.config'
            config.write_text((ROOT/'cuteafd.config').read_text()+'\nTP2_ATTENTION=on\nTP2_DSPARK_EXPERTS=on\n'
                              '\n# run.sh --set\nTP2_ATTENTION=off\nTP2_DSPARK_EXPERTS=off\nTP2_QUERY_PROJECTION=on\n')
            setup='set -euo pipefail; source scripts/lib/release-common.sh; config="$1"; wip_slot=\n'
            result=subprocess.run(['bash','-c',setup+block+finish,'test',str(config)],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.splitlines(),['off','on','off','off'])
            config.write_text((ROOT/'cuteafd.config').read_text()+'\nTP2_OUTPUT_PROJECTION=invalid\n')
            result=subprocess.run(['bash','-c',setup+block+finish,'test',str(config)],capture_output=True,text=True,cwd=ROOT)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('TP2_OUTPUT_PROJECTION must be on or off',result.stderr)
        # The removed flags name the key that replaced them.
        for flag in ['--tp2-attention','--no-tp2-query-projection','--dspark']:
            result=subprocess.run(['./run.sh',flag],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,2)
            self.assertIn('--set',result.stderr)

    def test_tp2_options_reach_coordinator_independently(self):
        options=['ATTENTION','QUERY_PROJECTION','OUTPUT_PROJECTION','DSPARK_EXPERTS']
        flags=['--tp2-'+option.lower().replace('_','-') for option in options]
        for enabled in [[], *[[option] for option in options], options]:
            with self.subTest(enabled=enabled):
                result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',rtx_expert_layers=20,spark_first_layer=20),enabled)
                self.assertEqual(result.returncode,0,result.stderr)
                args=next(args for tool,args in events if tool=='docker' and 'serve-native' in args)
                for option,flag in zip(options,flags):
                    self.assertEqual(flag in args,option in enabled)

    def test_filesystem_plan_is_not_a_startup_input(self):
        # A stale/invalid filesystem payload is ignored; the daemon validates
        # checkpoint identity, ownership and model geometry over TCP instead.
        result, events = self.run_startup(1, {"version": -1}, local_layers='5')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len([args for tool, args in events if tool == 'ssh' and '-s' in args]), 4)
        self.assertFalse(any(tool == 'docker' and args[0] == 'exec' for tool, args in events))

if __name__=='__main__':unittest.main()
