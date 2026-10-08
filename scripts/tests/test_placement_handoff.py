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
if tool=='ssh' and args and args[-1].startswith('bash -s -- '):
 args=args[:-1]+shlex.split(args[-1])
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
        source=(ROOT/'run.sh').read_text()
        block=source[source.index('placement_directory='):source.index('api_url=')]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'plan').write_text(json.dumps(plan))
            for name in ['docker','ssh','nest']:
                path=root/name;path.write_text(STUB);path.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],EVENTS=str(root/'events'),PLAN=str(root/'plan'),
                ENCODER_PLAN=json.dumps(encoder))
            setup=r'''
set -euo pipefail
source scripts/lib/release-common.sh
release_die() { echo "$*" >&2; exit 1; }
RELEASE_RTX_GPUS="$1"
repo_root="$PWD"
coordinator=coordinator
snapshot_rel=model
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
TABLE_BACKEND=mmap
KV_POOL_SIZE=
MEMORY_RESERVATION=
DSPARK=on
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
COORDINATOR_DOCKER_INFERENCE=coordinator-image
SPARK_EXPERT_DOCKER_INFERENCE=spark-image
SPARK_DEVICE_BUDGET_BYTES=1000000
spark_prefix=worker
hosts=(a b c d)
EXPERT_PORT=19441
expert_capacity=4096
spark_first_layer=0
wip_layout=
wip_slot=
'''
            if encoder is not None:
                setup+='\nVISION=auto\n'
            setup+=f'\nRTX_EXPERT_LAYERS={local_layers}\nSPARK_COUNT={spark_count}\nhosts=("${{hosts[@]:0:SPARK_COUNT}}")\n'
            # Legacy geometry: release_spark_tp defaults to SPARK_COUNT and
            # release_spark_ep to 1 when no explicit topology is configured.
            setup+=f'spark_tp={spark_count}\n'
            if spark_count == 2:
                setup+='MEMORY_RESERVATION=32GiB\nKV_POOL_SIZE=2GiB\npeers=10.55.0.1:19441,10.55.0.2:19441\n'
            setup+=''.join(f'\nTP2_{option}=on\n' for option in options)
            result=subprocess.run(['bash','-c',setup+block,'test',str(gpus)],env=env,cwd=ROOT,capture_output=True,text=True,timeout=10)
            events=[json.loads(line) for line in (root/'events').read_text().splitlines()]
            return result,events

    @staticmethod
    def worker_tail(args):
        """Worker values after `--` in a worker-start ssh invocation.

        run.sh passes image, remote, rank, capacity, budget, port, snapshot,
        fingerprint, first_layer, world, explicit_topology, tp, ep and then
        three optional RDMA env values. Index from the `--` separator so the
        optional tail does not shift the assertions.
        """
        return args[args.index('--') + 1:]

    def test_dual_starts_coordinator_then_correct_workers_then_acknowledges(self):
        for layers in [1,17,20,40]:
            with self.subTest(layers=layers):
                result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',rtx_expert_layers=layers,spark_first_layer=min(layers,39)))
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(events[0][0],'docker')
                self.assertEqual(events[0][1][0],'run')
                self.assertIn(f'seccomp={ROOT}/docker/seccomp-code-bench.json',events[0][1])
                self.assertIn('--placement-directory',events[0][1])
                starts=[args for tool,args in events if tool=='ssh' and '-s' in args]
                self.assertEqual(len(starts),4)
                tails=[self.worker_tail(args) for args in starts]
                self.assertTrue(all(len(tail)==18 for tail in tails),tails)
                # first_layer, world, then the legacy topology tail.
                self.assertTrue(all(tail[8]==str(min(layers,39)) and tail[9]=='4' for tail in tails))
                self.assertTrue(all(tail[10:13]==['0','4','1'] for tail in tails))
                ack=[i for i,(tool,args) in enumerate(events) if tool=='docker' and args[:3]==['exec','coordinator','sh']]
                self.assertEqual(len(ack),1)
                ready=[i for i,(tool,args) in enumerate(events) if tool=='ssh' and any('timeout 1' in a for a in args)]
                self.assertEqual(len(ready),4)
                self.assertGreater(ack[0],max(ready))

    def test_unavailable_p2p_accepts_single_gpu_fallback_before_worker_handoff(self):
        for layers in [0, 5]:
            with self.subTest(layers=layers):
                result, events = self.run_startup(2, dict(version=1, rtx_gpus=1,
                    nonce='fresh', rtx_expert_layers=layers, spark_first_layer=layers))
                self.assertEqual(result.returncode, 0, result.stderr)
                starts = [args for tool, args in events if tool == 'ssh' and '-s' in args]
                self.assertEqual(len(starts), 4)
                self.assertTrue(all(self.worker_tail(args)[8] == str(layers) for args in starts))

    def test_spark_vision_plan_reaches_only_selected_encoder_rank(self):
        encoder=dict(encoder=dict(kind=dict(kind='spark',rank=2)),encoder_plan_hash='ab'*32)
        result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',
            rtx_expert_layers=20,spark_first_layer=20),encoder=encoder)
        self.assertEqual(result.returncode,0,result.stderr)
        plans=[args for tool,args in events if tool=='docker' and 'plan' in args]
        self.assertEqual(len(plans),1)
        coordinator=next(args for tool,args in events if tool=='docker' and 'serve-native' in args)
        self.assertEqual(coordinator[coordinator.index('--vision-peers')+1],'10.55.0.3:19541')
        self.assertEqual(coordinator[coordinator.index('--encoder-plan-hash')+1],'ab'*32)
        tails=[self.worker_tail(args) for tool,args in events if tool=='ssh' and '-s' in args]
        self.assertEqual(len(tails),4)
        self.assertTrue(all(tail[18:]==['2','19541','ab'*32,
            '0123456789012345678901234567890123456789'] for tail in tails))
        self.assertIn('if [[ "$rank" == "${19:--1}" ]]', (ROOT/'run.sh').read_text())

    def test_compact_starts_exactly_two_workers_and_passes_ceiling(self):
        result,events=self.run_startup(1,dict(version=1,rtx_gpus=1,nonce="fresh",rtx_expert_layers=0,spark_first_layer=0),spark_count=2)
        self.assertEqual(result.returncode,0,result.stderr)
        starts=[args for tool,args in events if tool=='ssh' and '-s' in args]
        self.assertEqual(len(starts),2)
        tails=[self.worker_tail(args) for args in starts]
        self.assertTrue(all(tail[8]=='0' and tail[9]=='2' for tail in tails))
        # Legacy geometry tail: explicit flag off, TP=SPARK_COUNT, EP=1.
        self.assertTrue(all(tail[10:13]==['0','2','1'] for tail in tails))
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

    def test_spark_launch_drops_only_its_hosts_page_caches(self):
        result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',rtx_expert_layers=20,spark_first_layer=20))
        self.assertEqual(result.returncode,0,result.stderr)
        drops=[args for tool,args in events if tool=='nest']
        self.assertEqual(len(drops),1,events)
        self.assertEqual(drops[0][0],'drop-caches')
        self.assertTrue(all(a=='--host' for a in drops[0][1::2]),drops)

    def test_cli_tp2_overrides_and_invalid_config(self):
        source=(ROOT/'run.sh').read_text()
        block=source[source.index('config="$repo_root/cuteafd.config"'):source.index('for tool in docker ssh')]
        setup='repo_root="$1"; shift; source "$repo_root/scripts/lib/release-common.sh"\n'
        finish='\nprintf "%s\\n" "$TP2_ATTENTION" "$TP2_QUERY_PROJECTION" "$TP2_OUTPUT_PROJECTION" "$TP2_DSPARK_EXPERTS"\n'
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'recipe.config'
            config.write_text((ROOT/'cuteafd.config').read_text()+'\nTP2_ATTENTION=on\nTP2_DSPARK_EXPERTS=on\n')
            args=['bash','-c',setup+block+finish,'test',str(ROOT),'--config',str(config)]
            result=subprocess.run(args+['--no-tp2-attention','--tp2-query-projection','--no-tp2-dspark-experts'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.splitlines(),['off','on','off','off'])
            config.write_text((ROOT/'cuteafd.config').read_text()+'\nTP2_OUTPUT_PROJECTION=invalid\n')
            result=subprocess.run(args,capture_output=True,text=True,cwd=ROOT)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('TP2_OUTPUT_PROJECTION must be on or off',result.stderr)

    def test_tp2_options_reach_coordinator_independently(self):
        options=['ATTENTION','QUERY_PROJECTION','OUTPUT_PROJECTION','DSPARK_EXPERTS']
        flags=['--tp2-'+option.lower().replace('_','-') for option in options]
        for enabled in [[], *[[option] for option in options], options]:
            with self.subTest(enabled=enabled):
                result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',rtx_expert_layers=20,spark_first_layer=20),enabled)
                self.assertEqual(result.returncode,0,result.stderr)
                args=events[0][1]
                for option,flag in zip(options,flags):
                    self.assertEqual(flag in args,option in enabled)

    def test_invalid_boundary_never_starts_workers(self):
        result,events=self.run_startup(2,dict(version=1,rtx_gpus=2,nonce='fresh',rtx_expert_layers=17,spark_first_layer=20))
        self.assertNotEqual(result.returncode,0)
        # Pin the intended rejection so a fixture/scope error cannot satisfy the
        # nonzero exit by accident.
        self.assertIn('invalid coordinator placement plan',result.stderr)
        self.assertFalse(any(tool=='ssh' for tool,_ in events))

    def test_single_all_remote_keeps_worker_first_startup_without_handoff(self):
        result,events=self.run_startup(1,{},local_layers="0")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(events[0][0],'ssh')
        runs=[args for tool,args in events if tool=='docker' and args[0]=='run']
        self.assertEqual(len(runs),1)
        self.assertNotIn('--placement-directory',runs[0])
        self.assertFalse(any(tool=='docker' and args[0]=='exec' for tool,args in events))

    def test_single_auto_publishes_live_boundary_before_workers(self):
        # Auto may resolve to no local experts under a small memory ceiling,
        # or keep any number of layers. Workers must use the actual plan.
        for layers in [0,5,39,40]:
            with self.subTest(layers=layers):
                result,events=self.run_startup(1,dict(version=1,rtx_gpus=1,nonce='fresh',
                    rtx_expert_layers=layers,spark_first_layer=min(layers,39)))
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(events[0][0],'docker')
                self.assertEqual(events[0][1][0],'run')
                self.assertIn('--placement-directory',events[0][1])
                starts=[args for tool,args in events if tool=='ssh' and '-s' in args]
                self.assertEqual(len(starts),4)
                self.assertTrue(all(self.worker_tail(args)[8]==str(min(layers,39)) for args in starts))
                ack=[i for i,(tool,args) in enumerate(events) if tool=='docker'
                    and args[:3]==['exec','coordinator','sh']]
                ready=[i for i,(tool,args) in enumerate(events) if tool=='ssh'
                    and any('timeout 1' in a for a in args)]
                self.assertEqual(len(ack),1)
                self.assertEqual(len(ready),4)
                self.assertGreater(ack[0],max(ready))

    def test_single_explicit_boundary_is_handed_off_without_topology_flags(self):
        for layers in [1,5,40]:
            with self.subTest(layers=layers):
                result,events=self.run_startup(1,dict(version=1,rtx_gpus=1,nonce='fresh',
                    rtx_expert_layers=layers,spark_first_layer=min(layers,39)),local_layers=str(layers))
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(events[0][0],'docker')
                starts=[args for tool,args in events if tool=='ssh' and '-s' in args]
                self.assertTrue(all(self.worker_tail(args)[8]==str(min(layers,39)) for args in starts))

    def test_explicit_boundary_must_match_the_requested_count(self):
        result,events=self.run_startup(1,dict(version=1,rtx_gpus=1,nonce='fresh',
            rtx_expert_layers=4,spark_first_layer=4),local_layers='5')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('invalid coordinator placement plan',result.stderr)
        self.assertFalse(any(tool=='ssh' for tool,_ in events))

if __name__=='__main__':unittest.main()
