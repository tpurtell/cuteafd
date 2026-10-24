"""CPU-only coverage for the opt-in Spark TP x EP replicated topology.

No Docker, SSH, GPU, build or serving operation is performed. Every case drives
the launcher through `scripts/lib/release-common.sh` or the process-boundary stubs
used by `test_placement_handoff.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "cuteafd.config"
EXAMPLES = ROOT / "examples" / "configs"


def source_common(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-euc", "source scripts/lib/release-common.sh; " + script, "test", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def load(extra: str = "", command: str = 'printf "%s\\n" "$SPARK_COUNT" "$SPARK_TP" "$SPARK_EP"') -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as temporary:
        config = Path(temporary) / "topology.config"
        config.write_text(CONFIG.read_text() + "\n" + extra + "\n")
        return source_common('release_load_config "$1"; ' + command, str(config))


def load_file(path: Path, command: str = 'printf "%s\\n" "$SPARK_COUNT" "$SPARK_TP" "$SPARK_EP"') -> subprocess.CompletedProcess[str]:
    return source_common('release_load_config "$1"; ' + command, str(path))


# Six placeholders are deliberate: REPLACE-ME-* is not resolvable and
# 192.0.2.0/24 is the RFC 5737 documentation range. The base config's secondary
# rail is cleared for every active rank so the six-rank rail stays all-or-none
# even though the shipped default now names all six Spark hosts.
SIX_HOSTS = """\
SPARK_4_HOST=REPLACE-ME-spark-4
SPARK_5_HOST=REPLACE-ME-spark-5
SPARK_4_LANE_A=192.0.2.5
SPARK_5_LANE_A=192.0.2.6
SPARK_0_LANE_B=
SPARK_1_LANE_B=
SPARK_2_LANE_B=
SPARK_3_LANE_B=
SPARK_4_LANE_B=
SPARK_5_LANE_B=
"""


class TopologyConfigTest(unittest.TestCase):
    def test_retained_canonical_and_owned_tp_counts_use_configured_hosts(self) -> None:
        for explicit, counts in [(False, range(1, 7)), (True, (2, 3, 4, 6))]:
            for count in counts:
                overlay = [f"SPARK_COUNT={count}",
                           f"SPARK_TP={count}" if explicit else "SPARK_TP=",
                           "SPARK_EP=1" if explicit else "SPARK_EP="]
                for rank in range(6):
                    overlay += [f"SPARK_{rank}_HOST=worker-{rank}",
                                f"SPARK_{rank}_LANE_A=192.0.2.{rank + 1}",
                                f"SPARK_{rank}_LANE_B="]
                with self.subTest(explicit=explicit, count=count):
                    result = load("\n".join(overlay), "release_spark_rank_map; release_hosts_csv")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines(),
                                     [f"{rank} 0 {rank}" for rank in range(count)] +
                                     [",".join(f"worker-{rank}" for rank in range(count))])

    def test_deleted_worlds_and_unqualified_ownership_refuse(self) -> None:
        for count in (1, 5, 7, 8):
            with self.subTest(count=count):
                result = load(f"SPARK_COUNT={count}\nSPARK_TP={count}\nSPARK_EP=1")
                self.assertEqual(result.returncode, 2)
                self.assertIn("unsupported native Spark topology" if count <= 6
                              else "SPARK_TP must be in 1..6", result.stderr)

    def test_deleted_rank_inventory_keys_refuse(self) -> None:
        for rank in (6, 7):
            for suffix in ("HOST", "LANE_A", "LANE_B"):
                key = f"SPARK_{rank}_{suffix}"
                with self.subTest(key=key):
                    result = load(f"{key}=unused")
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(f"unknown configuration key: {key}", result.stderr)

    def test_v41_accepts_explicit_vision_placement(self) -> None:
        for vision in ("auto", "off", "rtx", "rtx:0", "spark", "spark:3"):
            with self.subTest(vision=vision):
                result = load(
                    f"MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash\nVISION={vision}\n",
                    'printf "%s\\n" "$VISION"',
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), vision)

    def test_v41_rejects_malformed_vision_placement(self) -> None:
        for vision in ("auto:0", "off:0", "rtx:-1", "spark:rank", "cpu"):
            with self.subTest(vision=vision):
                result = load(f"MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash\nVISION={vision}\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("VISION must be", result.stderr)

    def test_default_config_keeps_legacy_topology_and_is_not_mutated(self) -> None:
        before = CONFIG.read_bytes()
        result = load(
            "",
            'printf "%s\\n" "$SPARK_COUNT" "$SPARK_TP" "$SPARK_EP" '
            '"$(release_spark_tp)" "$(release_spark_ep)"; '
            "if release_spark_topology_explicit; then echo explicit; else echo legacy; fi",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["6", "", "", "6", "1", "legacy"])
        self.assertEqual(CONFIG.read_bytes(), before)
        self.assertNotIn("SPARK_TP=", CONFIG.read_text())
        self.assertNotIn("SPARK_EP=", CONFIG.read_text())

    def test_approved_topologies_resolve_rank_map(self) -> None:
        cases = {
            "SPARK_COUNT=4\nSPARK_TP=2\nSPARK_EP=2\n": ["0 0 0", "1 0 1", "2 1 0", "3 1 1"],
            "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS: [
                "0 0 0", "1 0 1", "2 0 2", "3 1 0", "4 1 1", "5 1 2",
            ],
            "SPARK_COUNT=6\nSPARK_TP=2\nSPARK_EP=3\n" + SIX_HOSTS: [
                "0 0 0", "1 0 1", "2 1 0", "3 1 1", "4 2 0", "5 2 1",
            ],
            "SPARK_COUNT=4\nSPARK_TP=4\nSPARK_EP=1\n": ["0 0 0", "1 0 1", "2 0 2", "3 0 3"],
            # Pure unreplicated TP6: one group, six disjoint intermediate shards.
            "SPARK_COUNT=6\nSPARK_TP=6\nSPARK_EP=1\n" + SIX_HOSTS: [
                "0 0 0", "1 0 1", "2 0 2", "3 0 3", "4 0 4", "5 0 5",
            ],
        }
        for overlay, expected in cases.items():
            with self.subTest(overlay=overlay):
                result = load(overlay, "release_spark_rank_map")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected)

    def test_six_host_bounds_generate_every_rank_and_peer(self) -> None:
        result = load(
            "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS,
            'release_hosts_csv; release_lane_a_csv; release_expert_hosts_csv',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        hosts, lanes, experts = result.stdout.splitlines()
        self.assertEqual(
            hosts,
            "ostrich,dodo,emu,kiwi,REPLACE-ME-spark-4,REPLACE-ME-spark-5",
        )
        self.assertEqual(lanes, "10.55.0.1,10.55.0.2,10.55.0.3,10.55.0.4,192.0.2.5,192.0.2.6")
        self.assertTrue(experts.endswith("spark-4=192.0.2.5:19441,spark-5=192.0.2.6:19441"))

    def test_optional_secondary_rail_is_all_or_none_for_six_hosts(self) -> None:
        base = "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS
        partial = base + "\n".join(
            f"SPARK_{index}_LANE_B=10.55.0.{index + 5}" for index in range(5)
        )
        result = load(partial, "true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("secondary Spark rail", result.stderr)

        complete = base + "\n".join(
            f"SPARK_{index}_LANE_B=10.55.0.{index + 5}" for index in range(6)
        )
        result = load(complete, "release_lane_b_csv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.strip().split(",")), 6)

        result = load(base, "release_lane_b_csv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_invalid_combinations_are_rejected(self) -> None:
        cases = {
            "SPARK_TP=2\nSPARK_EP=1\n": "must equal SPARK_COUNT",
            "SPARK_TP=5\nSPARK_EP=2\n": "must equal SPARK_COUNT",
            "SPARK_TP=9\nSPARK_EP=1\n": "SPARK_TP must be in 1..6",
            "SPARK_EP=4\n": "SPARK_EP must be 1, 2, or 3",
            "SPARK_TP=2\n": "set together or omitted together",
            "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=3\n" + SIX_HOSTS: "must equal SPARK_COUNT",
            "SPARK_COUNT=4\nSPARK_TP=3\nSPARK_EP=2\n": "must equal SPARK_COUNT",
        }
        for overlay, message in cases.items():
            with self.subTest(overlay=overlay):
                result = load(overlay, "true")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(message, result.stderr)

    def test_config_accepts_any_rtx_gpus_and_defers_to_the_budget(self) -> None:
        # The Spark TP/EP degree must not hardcode the RTX layout. Both RTX
        # counts parse; the launcher rejects an infeasible combination with the
        # resolved weight budget instead.
        for overlay in ("SPARK_COUNT=4\nSPARK_TP=2\nSPARK_EP=2\nRTX_GPUS=1\n",
                        "SPARK_COUNT=4\nSPARK_TP=2\nSPARK_EP=2\nRTX_GPUS=2\n",
                        "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=2\n" + SIX_HOSTS,
                        "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\nRTX_GPUS=1\n" + SIX_HOSTS):
            with self.subTest(overlay=overlay):
                result = load(overlay, "true")
                self.assertEqual(result.returncode, 0, result.stderr)
    def test_explicit_topology_rejects_non_native_formats(self) -> None:
        for overlay, message in (
            ("EXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto\n", "requires EXPERT_FORMAT=native"),
            ("SPARKINFER_EXL3=auto\n", "requires SPARKINFER_EXL3=disable"),
        ):
            with self.subTest(overlay=overlay):
                result = load("SPARK_COUNT=4\nSPARK_TP=2\nSPARK_EP=2\n" + overlay, "true")
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_compact_exl3_accepts_only_explicit_tp3_ep1(self) -> None:
        for mode in ("auto", "force"):
            result = load(f"SPARK_COUNT=3\nSPARK_TP=3\nSPARK_EP=1\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3={mode}\n",
                          "release_spark_rank_map")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["0 0 0", "1 0 1", "2 0 2"])
        for tp, ep, count, extra in ((2, 1, 2, ""), (3, 2, 6, ""), (4, 1, 4, ""),
                                      (3, 1, 3, "SPARKINFER_EXL3=disable\n")):
            with self.subTest(tp=tp, ep=ep, extra=extra):
                result = load(f"SPARK_COUNT={count}\nSPARK_TP={tp}\nSPARK_EP={ep}\n"
                              "EXPERT_FORMAT=exl3\nSPARKINFER_EXL3=force\n" + extra)
                self.assertEqual(result.returncode, 2)

    def test_unknown_topology_alias_is_not_accepted(self) -> None:
        result = load("SPARK_GROUPS=2\n", "true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown configuration key: SPARK_GROUPS", result.stderr)


class ExampleConfigTest(unittest.TestCase):
    def test_every_example_parses_with_its_approved_geometry(self) -> None:
        expected = {
            "tp4ep1-explicit-native.config": ("4", "4", "1"),
            "tp2ep2-native.config": ("4", "2", "2"),
            "tp3ep2-native.config": ("6", "3", "2"),
            "tp2ep3-native.config": ("6", "2", "3"),
            "tp6ep1-native.config": ("6", "6", "1"),
            # V10 candidate profiles. The native one names its explicit keys;
            # the compact EXL3 one is IMPLICIT (no SPARK_TP/SPARK_EP keys),
            # so its resolved keys stay empty while SPARK_COUNT=3 activates the
            # compact TP3 rules.
            "tp3ep1-native.config": ("3", "3", "1"),
            "exl3-compact-tp3.config": ("3", "", ""),
        }
        # The per-family starting configs (<model>-<quant>-<min|max>.config) are
        # covered by test_example_configs.py; this set is the V4.1 TP x EP layouts.
        self.assertEqual(
            sorted(path.name for path in EXAMPLES.glob("*.config")
                   if not path.name.endswith(("-min.config", "-max.config"))),
            sorted(expected),
        )
        for name, geometry in expected.items():
            with self.subTest(name=name):
                result = load_file(EXAMPLES / name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(tuple(result.stdout.splitlines()), geometry)

    def test_v10_native_tp3_profile_reserves_five_local_layers(self) -> None:
        # The v10 native TP3xEP1 profile pins an explicit single-RTX local count,
        # so the launcher takes the placement handoff instead of the all-remote
        # no-handoff shape. Pin the shape and its weight-only arithmetic.
        result = load_file(
            EXAMPLES / "tp3ep1-native.config",
            'printf "%s\\n" "$RTX_GPUS" "$RTX_EXPERT_LAYERS" "$SPARK_COUNT" '
            '"$SPARK_TP" "$SPARK_EP" "$SPARK_DEVICE_BUDGET_BYTES" '
            '"$(release_spark_layer_bytes "$SPARK_TP")"',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        gpus, layers, count, tp, ep, budget, layer_bytes = result.stdout.splitlines()
        self.assertEqual((gpus, layers, count, tp, ep), ("1", "5", "3", "3", "1"))
        self.assertEqual(int(layer_bytes), 2406481920)
        self.assertEqual(40 - int(layers), 35)
        self.assertEqual(40 * int(layer_bytes), 96259276800)
        self.assertEqual(35 * int(layer_bytes), 84226867200)
        self.assertEqual(int(budget) - 35 * int(layer_bytes), 24892452864)

    def test_six_host_examples_use_the_connected_fifth_and_sixth_sparks(self) -> None:
        for name in ("tp3ep2-native.config", "tp2ep3-native.config", "tp6ep1-native.config"):
            text = (EXAMPLES / name).read_text()
            self.assertIn("SPARK_4_HOST=rhea", text)
            self.assertIn("SPARK_5_HOST=moa", text)
            # Both rails use matching host numbers on separate subnets;
            # the candidate launcher still selects fabric A.
            self.assertIn("SPARK_4_LANE_A=10.55.0.5", text)
            self.assertIn("SPARK_5_LANE_A=10.55.0.6", text)
            self.assertIn("NOT RELEASE-QUALIFIED", text)
            self.assertIn("SPARK_4_LANE_B=10.55.1.5", text)
            self.assertIn("SPARK_5_LANE_B=10.55.1.6", text)
            self.assertNotIn("REPLACE-ME", text)
            self.assertNotIn("192.0.2.", text)
        # Unspecified count selects every configured host.
        default = CONFIG.read_text()
        self.assertNotRegex(default, r"(?m)^SPARK_COUNT=")
        self.assertNotIn("SPARK_TP=", default)

    def test_two_rtx_tp2ep2_example_is_opt_in_and_bound_to_real_hosts(self) -> None:
        text = (EXAMPLES / "tp2ep2-native.config").read_text()
        self.assertIn("SPARK_TP=2", text)
        self.assertIn("SPARK_EP=2", text)
        self.assertIn("RTX_GPUS=2", text)
        self.assertIn("SPARK_0_HOST=ostrich", text)
        self.assertIn("SPARK_3_HOST=kiwi", text)


STUB = r'''#!/usr/bin/env python3
import json,os,sys,shlex
from pathlib import Path
args=sys.argv[1:]; tool=Path(sys.argv[0]).name
if tool=='ssh' and args and 'bash -s -- ' in args[-1]:
 args=args[:-1]+shlex.split(args[-1][args[-1].rindex('bash -s -- '):])
with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps([tool,args])+'\n')
if tool=='docker':
 if args[0]=='inspect':print('running')
 elif args[0]=='exec' and 'cat' in args:
  print(Path(os.environ['PLAN']).read_text())
 elif args[0]=='run':print('container-id')
else:
 sys.stdin.read() if '-s' in args else None
'''


class LauncherTopologyTest(unittest.TestCase):
    """Run run.sh's real startup block with process-boundary stubs."""

    def run_startup(self, *, gpus, plan, topology_explicit, spark_tp, spark_ep, spark_count, hosts,
                    extra_env=None, extra_setup=""):
        source = (ROOT / "scripts/launch/run-family.sh").read_text()
        block = source[source.index("# Workers bootstrap first;"):source.index("api_url=")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan").write_text(json.dumps(plan))
            for name in ["docker", "ssh"]:
                path = root / name
                path.write_text(STUB)
                path.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                HOME=str(root / "home"),
                EVENTS=str(root / "events"),
                PLAN=str(root / "plan"),
                **(extra_env or {}),
            )
            setup = f'''set -euo pipefail
source scripts/lib/release-common.sh
release_die() {{ echo "$*" >&2; exit 1; }}
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
# These topology tests exercise the text-only legacy startup contract.
VISION=off
AUDIO=off
DSPARK_DRAFT_POLICY=adaptive
dspark_draft_limit=
TP2_ATTENTION=off
TP2_QUERY_PROJECTION=off
TP2_OUTPUT_PROJECTION=off
TP2_DSPARK_EXPERTS=off
topology_explicit={topology_explicit}
spark_tp={spark_tp}
spark_ep={spark_ep}
spark_admission=not-applicable
expert_format=native
model_is_exl3=false
spark_exl3_identity=
gpu_request=device=uuid0,uuid1
gpu_uuid_csv=uuid0,uuid1
fingerprint=test
hf_home=/models
namespace_root=/mnt/sparknest
COORDINATOR_DOCKER_INFERENCE=coordinator-image
SPARK_EXPERT_DOCKER_INFERENCE=spark-image
SPARK_DEVICE_BUDGET_BYTES=107374182400
spark_prefix=worker
hosts=({' '.join(hosts)})
EXPERT_PORT=19441
expert_capacity=4096
wip_layout=
wip_slot=
SPARK_COUNT={spark_count}
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
{extra_setup}'''
            setup += '\ncount_plan=\'{"spark_admission":"coordinator_authoritative"}\'\n'
            result = subprocess.run(
                ["bash", "-c", setup + block, "test", str(gpus)],
                env=env,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            events_path = root / "events"
            events = (
                [json.loads(line) for line in events_path.read_text().splitlines()]
                if events_path.exists()
                else []
            )
            # Worker docker options follow the positional contract behind a
            # --profile-args sentinel; these tests index the positionals.
            trimmed = []
            for tool, args in events:
                if tool == "ssh" and "-s" in args and "--profile-args" in args:
                    tail = args[args.index("--profile-args") + 1:]
                    assert "-e" in tail and "CUTEAFD_THERMAL=on" in tail, tail
                    args = args[:args.index("--profile-args")]
                trimmed.append([tool, args])
            return result, trimmed

    def test_default_launch_passes_no_topology_flags_or_dummy_peers(self) -> None:
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
            topology_explicit=0,
            spark_tp=4,
            spark_ep=1,
            spark_count=4,
            hosts=["a", "b", "c", "d"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertNotIn("--spark-tp", coordinator)
        self.assertNotIn("--spark-ep", coordinator)
        # Worker positional tail ends with topology(explicit,tp,ep) then the
        # three optional RDMA env values.
        self.assertTrue(all(args[-8] == "0" for tool, args in events if tool == "ssh" and "-s" in args))

    def test_planner_pool_is_forwarded_only_when_selected(self) -> None:
        for setting, expected in [("", None), ("auto", "0"), ("123456", "123456")]:
            with self.subTest(setting=setting):
                result, events = self.run_startup(
                    gpus=2,
                    plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
                    topology_explicit=0, spark_tp=4, spark_ep=1, spark_count=4,
                    hosts=["a", "b", "c", "d"], extra_setup=f"POOL_TOKENS={setting}\n",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
                if expected is None:
                    self.assertNotIn("--pool-tokens", coordinator)
                else:
                    self.assertEqual(coordinator[coordinator.index("--pool-tokens") + 1], expected)

    def test_explicit_topology_reaches_coordinator_and_every_worker(self) -> None:
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
            topology_explicit=1,
            spark_tp=2,
            spark_ep=2,
            spark_count=4,
            hosts=["a", "b", "c", "d"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertEqual(coordinator[coordinator.index("--spark-tp") + 1], "2")
        self.assertEqual(coordinator[coordinator.index("--spark-ep") + 1], "2")
        # The launcher peer list is assembled from the real lane addresses for
        # count 4/6; the zero-Spark dummy list is unreachable for an explicit
        # topology because SPARK_COUNT must be 4 or 6.
        self.assertEqual(coordinator[coordinator.index("--peers") + 1], "peer-list")
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 4)
        for args in starts:
            self.assertEqual(args[-8:-5], ["1", "2", "2"])
            self.assertEqual(args[-9], "4")

    def test_compact_tp3_startup_keeps_compressed_admission_and_topology(self) -> None:
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
            topology_explicit=1, spark_tp=3, spark_ep=1, spark_count=3,
            hosts=["a", "b", "c"],
            extra_setup='model_is_exl3=true\nexpert_format=exl3-k34\n',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertEqual(coordinator[coordinator.index("--spark-tp") + 1], "3")
        self.assertEqual(coordinator[coordinator.index("--spark-ep") + 1], "1")
        self.assertNotIn("--exl3-paired-tp4", coordinator)
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 3)
        for args in starts:
            self.assertEqual(args[-8:-5], ["1", "3", "1"])
            self.assertEqual(args[-9], "3")

    def test_six_rank_explicit_topology_starts_six_workers(self) -> None:
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=12, spark_first_layer=12),
            topology_explicit=1,
            spark_tp=2,
            spark_ep=3,
            spark_count=6,
            hosts=["a", "b", "c", "d", "e", "f"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 6)
        self.assertTrue(all("--first-layer" not in args for args in starts))
        self.assertEqual([args[-9] for args in starts], ["6"] * 6)
        self.assertEqual([args[-8:-5] for args in starts], [["1", "2", "3"]] * 6)

    def test_single_rtx_local_count_publishes_and_hands_off_five(self) -> None:
        # The current official 1-RTX shape on an explicit topology: 5 local
        # routed layers, so the coordinator publishes boundary 5 and every worker
        # is started at --first-layer 5 instead of 0.
        result, events = self.run_startup(
            gpus=1,
            plan=dict(version=1, rtx_gpus=1, nonce="fresh", rtx_expert_layers=5, spark_first_layer=5),
            topology_explicit=1,
            spark_tp=6,
            spark_ep=1,
            spark_count=6,
            hosts=["a", "b", "c", "d", "e", "f"],
            extra_setup="RTX_EXPERT_LAYERS=5\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertNotIn("--placement-directory", coordinator)
        self.assertEqual(coordinator[coordinator.index("--rtx-gpus") + 1], "1")
        self.assertEqual(coordinator[coordinator.index("--rtx-expert-layers") + 1], "5")
        self.assertEqual(coordinator[coordinator.index("--spark-tp") + 1], "6")
        self.assertEqual(coordinator[coordinator.index("--spark-ep") + 1], "1")
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 6)
        # Worker positional tail: ... first_layer world topology explicit,tp,ep
        for args in starts:
            self.assertEqual(args[args.index("--") + 1 + 7], str(len(starts)), args)
            self.assertEqual(args[-9], "6")
            self.assertEqual(args[-8:-5], ["1", "6", "1"])
        # The boundary is acknowledged after the workers start.
        acks = [args for tool, args in events if tool == "docker" and args[0] == "exec"]
        self.assertFalse(any("ready.json" in " ".join(a) for a in acks), acks)

    def test_single_rtx_tp3_local_five_publishes_and_hands_off(self) -> None:
        # The v10 native TP3xEP1 profile: one RTX with 5 local routed layers and
        # three Spark ranks. The handoff is topology-independent, so the same
        # publication/acknowledgement path must carry TP=3.
        result, events = self.run_startup(
            gpus=1,
            plan=dict(version=1, rtx_gpus=1, nonce="fresh", rtx_expert_layers=5, spark_first_layer=5),
            topology_explicit=1,
            spark_tp=3,
            spark_ep=1,
            spark_count=3,
            hosts=["ostrich", "dodo", "emu"],
            extra_setup="RTX_EXPERT_LAYERS=5\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertNotIn("--placement-directory", coordinator)
        self.assertEqual(coordinator[coordinator.index("--rtx-expert-layers") + 1], "5")
        self.assertEqual(coordinator[coordinator.index("--spark-tp") + 1], "3")
        self.assertEqual(coordinator[coordinator.index("--spark-ep") + 1], "1")
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 3)
        for args in starts:
            self.assertEqual(args[args.index("--") + 1 + 7], str(len(starts)), args)
            self.assertEqual(args[-9], "3")
            self.assertEqual(args[-8:-5], ["1", "3", "1"])
        acks = [args for tool, args in events if tool == "docker" and args[0] == "exec"]
        self.assertFalse(any("ready.json" in " ".join(a) for a in acks), acks)

    def test_single_rtx_all_remote_keeps_the_legacy_no_handoff_launch(self) -> None:
        # RTX_EXPERT_LAYERS=0 has no local boundary to publish on one RTX.
        result, events = self.run_startup(
            gpus=1,
            plan=dict(version=1, rtx_gpus=1, nonce="fresh", rtx_expert_layers=40, spark_first_layer=39),
            topology_explicit=1,
            spark_tp=6,
            spark_ep=1,
            spark_count=6,
            hosts=["a", "b", "c", "d", "e", "f"],
            extra_setup="RTX_EXPERT_LAYERS=0\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertNotIn("--placement-directory", coordinator)
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 6)
        for args in starts:
            self.assertEqual(args[-9], "6", args)
        acks = [args for tool, args in events if tool == "docker" and args[0] == "exec"]
        self.assertFalse(any("ready.json" in " ".join(a) for a in acks), acks)

    def test_filesystem_boundary_is_not_used_by_launcher(self) -> None:
        result, events = self.run_startup(
            gpus=1, plan={"version": -1}, topology_explicit=1,
            spark_tp=6, spark_ep=1, spark_count=6, hosts=["a", "b", "c", "d", "e", "f"],
            extra_setup="RTX_EXPERT_LAYERS=5\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("plan.json" in " ".join(args) for _, args in events))

    def test_worker_remote_block_emits_flags_only_when_explicit(self) -> None:
        source = (ROOT / "scripts/launch/run-family.sh").read_text()
        _, remote = source.split("echo \"== starting native Spark experts ==\"", 1)[1].split("<<'REMOTE' &", 1)
        remote = remote.split("\nREMOTE", 1)[0].replace(" >/dev/null", "")
        harness = 'source scripts/lib/release-common.sh; source scripts/lib/cdi-gpus.sh; docker() { printf "%s\\n" "$@"; }\n' + remote
        explicit = subprocess.run(
            ["bash", "-euc", harness, "test", "image", "name", "1",
             "107374182400", "19441", "snapshot", "fingerprint", "4", "1", "2", "2"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        arguments = explicit.stdout.splitlines()
        self.assertEqual(arguments[arguments.index("--spark-tp") + 1], "2")
        self.assertEqual(arguments[arguments.index("--spark-ep") + 1], "2")
        self.assertEqual(arguments[arguments.index("--rank") + 1], "1")
        self.assertEqual(arguments[arguments.index("--world") + 1], "4")
        self.assertNotIn("--first-layer", arguments)
        self.assertNotIn("--capacity", arguments)
        self.assertNotIn("--rdma-endpoints", arguments)

        # Legacy invocation: ten positional arguments, no topology flags.
        legacy = subprocess.run(
            ["bash", "-euc", harness, "test", "image", "name", "1",
             "107374182400", "19441", "snapshot", "fingerprint", "4", "0", "4", "1"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        legacy_arguments = legacy.stdout.splitlines()
        self.assertNotIn("--spark-tp", legacy_arguments)
        self.assertEqual(legacy_arguments[legacy_arguments.index("--world") + 1], "4")

    def test_six_rank_launch_forwards_rdma_env_and_draft_controls(self) -> None:
        device_map = "10.55.0.1=rocep1s0f0,10.55.0.6=roceP2p1s0f0"
        result, events = self.run_startup(
            gpus=2,
            plan=dict(version=1, rtx_gpus=2, nonce="fresh", rtx_expert_layers=20, spark_first_layer=20),
            topology_explicit=1,
            spark_tp=2,
            spark_ep=3,
            spark_count=6,
            hosts=["a", "b", "c", "d", "e", "f"],
            extra_env={
                "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": device_map,
                "CUTEAFD_VERBS_APP_IB_PORT_NUM": "1",
                "CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES": "2",
            },
            extra_setup="DSPARK_DRAFT_POLICY=full\ndspark_draft_limit=7\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coordinator = next(args for tool, args in events if tool == "docker" and args[0] == "run")
        self.assertIn("--dspark-fixed", coordinator)
        self.assertEqual(coordinator[coordinator.index("--dspark-draft-limit") + 1], "7")
        self.assertIn(f"CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP={device_map}", coordinator)
        starts = [args for tool, args in events if tool == "ssh" and "-s" in args]
        self.assertEqual(len(starts), 6)
        for args in starts:
            self.assertEqual(args[-5:-2], [device_map, "1", "2"])


class VerbsDeviceMapTest(unittest.TestCase):
    """The shared device-map validator is what both launchers depend on."""

    def validate(self, value):
        return subprocess.run(
            ["bash", "-euc", "source scripts/lib/release-common.sh; release_validate_verbs_device_map", "test"],
            cwd=ROOT, text=True, capture_output=True,
            env=dict(os.environ, CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=value),
        )

    def test_accepts_unique_ipv4_device_entries(self) -> None:
        result = self.validate("10.55.0.1=rocep1s0f0,10.55.0.6=roceP2p1s0f0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_malformed_or_duplicate_entries(self) -> None:
        for bad in ("10.55.0.1", "10.55.0.1=", "=rocep1s0f0", "10.55.0.1=a,10.55.0.1=b"):
            with self.subTest(bad=bad):
                result = self.validate(bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP", result.stderr)


class StopScriptTest(unittest.TestCase):
    """Stop must cover every configured rank, not a fixed four-host list."""

    def test_usage_names_configured_ranks(self) -> None:
        text = (ROOT / "stop.sh").read_text()
        self.assertNotIn("coordinator and four", text)
        self.assertIn("configured Spark rank", text)
        # Cleanup must not be advertised as bounded by the active rank count.
        self.assertIn("regardless of SPARK_COUNT", text)
        self.assertIn("superset of the", text)

    def test_release_stop_services_covers_six_configured_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "stops"
            stub = f'''#!/usr/bin/env bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> {log}
exit 0
'''
            for name in ("docker", "ssh"):
                path = root / name
                path.write_text(stub)
                path.chmod(0o755)
            setup = f'''set -euo pipefail
source scripts/lib/release-common.sh
SPARK_COUNT=6
SPARK_0_HOST=h0
SPARK_1_HOST=h1
SPARK_2_HOST=h2
SPARK_3_HOST=h3
SPARK_4_HOST=h4
SPARK_5_HOST=h5
EXPERT_PORT=19441
ADDR=127.0.0.1:18000
release_stop_services coord worker
'''
            result = subprocess.run(
                ["bash", "-euc", setup, "test"], cwd=ROOT, text=True, capture_output=True,
                env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"]),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log.read_text() if log.exists() else ""
            for host in ("h0", "h1", "h2", "h3", "h4", "h5"):
                self.assertIn(host, log_text)


class ManifestWriterTest(unittest.TestCase):
    """The built-role manifest must be derived from validated build outputs."""

    def role_export(self, root: Path, role: str, *, intermediate: int = 1152, schema: int = 1,
                    capability=(12, 1), capacities=(1, 16, 80, 256, 1024, 4096),
                    object_bytes: bytes = b"object-bytes",
                    declared_bytes: bytes | None = None) -> None:
        export = root / f"v41_spark_{role}_experts"
        export.mkdir(parents=True, exist_ok=True)
        object_name = f"v41_{role}_m1.o"
        (export / object_name).write_bytes(object_bytes)
        manifest = {
            "schema": schema,
            "role": f"spark_{role}",
            "spark_tp_degree": {"tp2": 2, "tp3": 3}[role],
            "capability": list(capability),
            "geometry": {
                "experts": 384,
                "hidden": 5120,
                "intermediate": intermediate,
                "kernel_intermediate": intermediate,
                "topk": 6,
            },
            "variants": [{"capacity_rows": capacity} for capacity in capacities],
            "artifact_sha256": {
                object_name: hashlib.sha256(
                    object_bytes if declared_bytes is None else declared_bytes
                ).hexdigest(),
            },
            "sparkinfer_revision": "test-revision",
        }
        (export / "v41_experts.json").write_text(json.dumps(manifest))

    def write(self, root: Path, requested: str, *, role: str = "expert",
              library: bool = True, output: str = "V41_EXPERT_TP_AOT.json") -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable, str(ROOT / "scripts/build/write-v41-expert-tp-manifest.py"),
            "--role", role, "--requested", requested,
            "--native-build-dir", str(root),
            "--output", str(root / output),
        ]
        if library:
            (root / "libcuteafd_native.so").write_bytes(b"fake-library")
            command += ["--native-library", str(root / "libcuteafd_native.so")]
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    def test_valid_role_manifest_binds_geometry_and_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.role_export(root, "tp2", intermediate=1152)
            result = self.write(root, "tp2")
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads((root / "V41_EXPERT_TP_AOT.json").read_text())
            self.assertEqual(document["spark_tp_roles"], ["tp2"])
            self.assertEqual(document["manifests"]["tp2"]["geometry"]["intermediate"], 1152)
            self.assertEqual(len(document["native_library_sha256"]), 64)
            self.assertIn(document["symbols_verified"], (True, False))

    def test_default_empty_role_manifest_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.write(root, "")
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads((root / "V41_EXPERT_TP_AOT.json").read_text())
            self.assertEqual(document["spark_tp_roles"], [])
            self.assertEqual(document["manifests"], {})

    def test_stale_or_wrong_exports_cannot_advertise_a_role(self) -> None:
        cases = (
            ("schema", dict(schema=2), "schema"),
            ("capability", dict(capability=(12, 0)), "capability"),
            ("tp2_geometry", dict(intermediate=576), "geometry.intermediate"),
            ("missing_capacity", dict(capacities=(1, 16, 80)), "capacities"),
            ("wrong_object_hash", dict(object_bytes=b"file-bytes", declared_bytes=b"declared"), "hashes to"),
        )
        for name, kwargs, message in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self.role_export(root, "tp2", **kwargs)
                    result = self.write(root, "tp2")
                    self.assertEqual(result.returncode, 2, result.stdout)
                    if message:
                        self.assertIn(message, result.stderr)

    def test_role_requires_native_library_and_expert_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.role_export(root, "tp2", intermediate=1152)
            result = self.write(root, "tp2", library=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--native-library is required", result.stderr)
            result = self.write(root, "tp2", role="coordinator")
            self.assertEqual(result.returncode, 2)
            self.assertIn("only the expert role", result.stderr)

    def test_unknown_role_is_a_parse_error_with_status_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.write(root, "tp4")
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsupported Spark TP role", result.stderr)


class BuildScopeTest(unittest.TestCase):
    """Dry-run build/distribution paths must cover six hosts without touching
    Docker, SSH or any image."""

    def stub_path(self, root: Path) -> tuple[dict, Path]:
        stub_bin = root / "bin"
        stub_bin.mkdir()
        marker = root / "called"
        for name in ("docker", "ssh", "rsync", "nvidia-smi"):
            path = stub_bin / name
            path.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "{name} $*" >>"{marker}"\n'
                "exit 1\n"
            )
            path.chmod(0o755)
        env = dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ['PATH']}")
        return env, marker

    def test_build_dry_run_six_host_advertises_universal_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, marker = self.stub_path(root)
            result = subprocess.run(
                ["./build.sh", "--config", str(EXAMPLES / "tp3ep2-native.config"), "--dry-run"],
                cwd=ROOT, text=True, capture_output=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("build hosts (6)", result.stdout)
            # The published Spark image is universal: one build serves every
            # approved topology, so an explicit TP3 config must not narrow the
            # built role set to tp3 alone.
            self.assertIn("V41 Spark expert roles: tp2;tp3;tp6", result.stdout)
            self.assertFalse(marker.exists(), marker.read_text() if marker.exists() else "")

    def test_wip_dry_run_six_host_official_only_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, marker = self.stub_path(root)
            env["CUTEAFD_WIP_EXL3_AOT"] = "OFF"
            env["CUTEAFD_WIP_NVFP4_AOT"] = "OFF"
            result = subprocess.run(
                ["./wip.sh", "--config", str(EXAMPLES / "tp2ep3-native.config"), "--dry-run"],
                cwd=ROOT, text=True, capture_output=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Spark hosts (6)", result.stdout)
            self.assertIn("V41 Spark expert roles: tp2", result.stdout)
            self.assertIn("EXL3 AOT: OFF; NVFP4 AOT: OFF", result.stdout)
            self.assertFalse(marker.exists(), marker.read_text() if marker.exists() else "")

    def test_helpers_plumb_roles_and_official_only_scope(self) -> None:
        release_helper = (ROOT / "scripts/build/build-release-artifacts.sh").read_text()
        wip_helper = (ROOT / "scripts/build/build-wip-artifacts.sh").read_text()
        dockerfile = (ROOT / "docker/Dockerfile.release").read_text()
        self.assertIn('-DCUTEAFD_SPARK_TP_ROLES="$spark_tp_roles"', release_helper)
        self.assertIn('-DCUTEAFD_SPARK_TP_ROLES="$spark_tp_roles"', wip_helper)
        self.assertIn("exl3_aot=\"${CUTEAFD_WIP_EXL3_AOT:-ON}\"", wip_helper)
        self.assertIn('-DCUTEAFD_ENABLE_EXL3_PACKAGES="$exl3_aot"', wip_helper)
        self.assertIn('if [[ "$exl3_aot" == ON ]]; then', wip_helper)
        self.assertIn("io.cuteafd.spark_tp_roles", dockerfile)
        self.assertIn("V41_EXPERT_TP_AOT.json", dockerfile)

    def test_run_sh_uses_network_admission_without_filesystem_handoff(self) -> None:
        release = (ROOT / "scripts/launch/run-family.sh").read_text()
        self.assertNotIn("--placement-directory", release)
        self.assertNotIn("ready.json", release)
        self.assertNotIn("plan.json", release)
        native = release[release.index("# Workers bootstrap first;"):release.index("api_url=")]
        self.assertLess(native.index('echo "== starting native Spark experts =="'),
                        native.rindex("start_coordinator"))
        readiness = native[native.index('for pid in "${pids[@]}"'):native.rindex("start_coordinator")]
        self.assertIn("native worker bootstrap ready", readiness)
        self.assertIn("docker logs '$remote'", readiness)
        self.assertNotIn("/dev/tcp", readiness)

    def test_native_pool_policy_changes_fingerprint_and_preserves_omitted_policy(self) -> None:
        expression = next(line for line in (ROOT / "scripts/launch/run-family.sh").read_text().splitlines()
                          if line.startswith('fingerprint="'))
        hashes = []
        for setting in (None, "", "auto", "1024"):
            environment = os.environ.copy()
            environment.pop("POOL_TOKENS", None)
            if setting is not None:
                environment["POOL_TOKENS"] = setting
            result = subprocess.run(
                ["bash", "-c", 'release_hosts_csv() { echo a,b,c,d; }\n' + expression
                 + '\nprintf "%s\n" "$fingerprint"'], text=True, capture_output=True, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            hashes.append(result.stdout.strip())
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(len(set(hashes)), 3)

    def test_run_sh_image_diagnostics_name_the_reference_and_host(self) -> None:
        release = (ROOT / "scripts/launch/run-family.sh").read_text()
        # A missing coordinator image is a naming problem, not a build problem:
        # the message must carry the configured reference and both remedies.
        self.assertIn("coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE", release)
        self.assertIn("pull the published pair", release)
        # A missing Spark image on one rank must identify that rank instead of
        # surfacing as a bare SSH failure, and the role refusal must show what
        # the image actually advertises.
        self.assertIn("inference image is missing: $image", release)
        self.assertIn("spark preflight on $host", release)
        self.assertIn("Spark host preflight failed on $host", release)
        self.assertIn("(advertised: ${advertised_roles:-<none>})", release)
        self.assertIn("is the Spark image present on that host?", release)
        # The published images are universal, so the remedy is the published
        # pair, not only a narrower local rebuild.
        self.assertIn("use the published universal release pair", release)
        self.assertIn("selected rank advertises", release)

    def test_shared_config_has_no_topology_to_rtx_hardcode(self) -> None:
        common = (ROOT / "scripts/lib/release-common.sh").read_text()
        self.assertNotIn("one-RTX layout", common)
        self.assertNotIn("two-RTX layout", common)

    def test_removed_adaptive_cost_mode_is_not_forwarded(self) -> None:
        # The online length policy has no offline cost profile or mode switch.
        self.assertNotIn("CUTEAFD_ADAPTIVE_COST_MODE", (ROOT / "scripts/launch/run-family.sh").read_text())



CANDIDATE = ROOT / "scripts" / "launch" / "run-tp-ep-native-candidate.sh"

CANDIDATE_STUB = r'''#!/usr/bin/env python3
import os, re, sys
name = os.path.basename(sys.argv[0]); args = sys.argv[1:]
scenario = os.environ.get("STUB_SCENARIO", "normal")
worker_status = os.environ.get("STUB_WORKER_STATUS", "running")
log_offset = os.environ.get("STUB_LOG_OFFSET", "0")
log_full = os.environ.get("STUB_LOG_FULL", "")
plan = os.environ.get("STUB_PLAN", "")
def status_for(joined):
    if "expert" in joined:
        print(f"{worker_status} 123")
    else:
        print("stopped" if scenario == "child-exit" else "running 123")
def tail_from(joined):
    match = re.search(r"\+(\d+)", joined)
    start = int(match.group(1)) if match else 1
    sys.stdout.write(log_full[start-1:])
if name == "ssh":
    if scenario == "ssh-fail":
        sys.exit(255)
    remote = " ".join(args)
    if "docker inspect" in remote:
        print("true")
    elif " stat -c" in remote:
        if os.environ.get("STUB_SSH_STAT_EXIT") == "1":
            sys.exit(255)
        print(log_offset)
    elif " tail -c" in remote:
        tail_from(remote)
    elif "plan.json" in remote and " cat " in remote:
        sys.stdout.write(plan)
    elif "wip-process.sh" in remote and " status " in remote:
        status_for(remote)
    sys.exit(0)
if name == "curl":
    # Health and model-list probes both succeed; the model id matches the
    # example config's checkpoint (the native API advertises the launched id).
    print('{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash-1017"}]}')
    sys.exit(0)
if args[:1] == ["inspect"]:
    print("true"); sys.exit(0)
if args[:1] == ["exec"]:
    tokens = list(args[1:]); cmd = []; i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("-d", "-i", "-t"):
            i += 1; continue
        if t in ("-e", "--env", "-w", "--workdir"):
            i += 2; continue
        if t.startswith("-"):
            i += 1; continue
        cmd = tokens[i:]; break
    if not cmd:
        sys.exit(0)
    rest = cmd[1:]
    if rest[:1] == ["test"]:
        # `test -e <dir>` is the stale-placement probe (absent on a clean run);
        # `test -s <file>` is the per-role wip-process existence preflight.
        if rest[1:2] == ["-e"]:
            sys.exit(0 if scenario == "stale" else 1)
        sys.exit(0)
    if rest[:1] == ["cat"]:
        if scenario == "child-exit":
            sys.exit(1)
        sys.stdout.write(plan); sys.exit(0)
    if rest[:1] == ["stat"]:
        print(log_offset); sys.exit(0)
    if rest[:1] == ["sh"] or rest[:1] == ["bash"]:
        joined = " ".join(rest)
        if "stat -c" in joined:
            print(log_offset); sys.exit(0)
        sys.exit(0)
    if rest[:1] == ["tail"]:
        tail_from(" ".join(rest)); sys.exit(0)
    if rest and rest[0].endswith("wip-process.sh") and len(rest) >= 2:
        if rest[1] == "status":
            status_for(" ".join(rest))
        sys.exit(0)
    sys.exit(0)
sys.exit(0)
'''

COORDINATOR_CONTROL_FLAGS = (
    "--prefill-batch-tokens",
    "--concurrency",
    "--prefix-cache-entries",
    "--max-context-tokens",
    "--max-output-tokens",
    "--http-queue-depth",
    "--http-queue-wait-ms",
    "--dspark",
    "--host-cache-bytes",
)


OPS_GPU_UUIDS = (
    "GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0,"
    "GPU-95f8f212-9131-df99-fd53-7535965197d7"
)

# Exact shape of the real G1 worker log line
# (runs/tp-ep-preflight/g1-live/raw/dodo-expert.log): the tracing formatter
# inserts ANSI dim/italic codes between a field name and its `=`.
ANSI_READY_LINE = (
    "\x1b[2m2026-09-20T07:13:46.765350Z\x1b[0m \x1b[32m INFO\x1b[0m "
    "\x1b[2mcuteafd::v41_experts::service::local\x1b[0m\x1b[2m:\x1b[0m "
    "native worker bootstrap ready "
    "\x1b[3mrank\x1b[0m\x1b[2m=\x1b[0m{rank} "
    "\x1b[3mworld\x1b[0m\x1b[2m=\x1b[0m4 "
    "\x1b[3mcapacity\x1b[0m\x1b[2m=\x1b[0m4096 "
    "\x1b[3mlayer_ids\x1b[0m\x1b[2m=\x1b[0m[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] "
    "\x1b[3mlayers\x1b[0m\x1b[2m=\x1b[0m20\n"
)
ANSI_READY_LINES = "".join(
    ANSI_READY_LINE.format(rank=rank) + ANSI_READY_LINE.format(rank=rank).replace(
        "native worker bootstrap ready", "coordinator-selected worker weights ready")
    for rank in range(4))


class CandidateLauncherTest(unittest.TestCase):
    def plan(self, config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CANDIDATE), "plan", "--config", str(config), *extra],
            cwd=ROOT, text=True, capture_output=True,
        )

    def test_plan_forwards_every_matched_control_and_orders_the_handshake(self) -> None:
        result = self.plan(
            EXAMPLES / "tp2ep2-native.config",
            "--coordinator-cuda-visible-devices", OPS_GPU_UUIDS,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        coordinator = next(line for line in lines if line.startswith("CMD coordinator:"))
        for flag in COORDINATOR_CONTROL_FLAGS:
            self.assertIn(flag, coordinator, flag)
        self.assertIn("--spark-tp 2", coordinator)
        self.assertIn("--spark-ep 2", coordinator)
        self.assertNotIn("--placement-directory", coordinator)
        # CUTEAFD_NATIVE_LIB must be present before the container and match --native-lib.
        self.assertIn("-e CUTEAFD_NATIVE_LIB=/scratch/coord-native/libcuteafd_native.so", coordinator)
        self.assertIn("-e CUTEAFD_WIP_RUNTIME_ROOT=/scratch/candidate/run", coordinator)
        self.assertIn("-e RUST_LOG=info", coordinator)
        # The plan render escapes the CSV comma; both UUIDs and the CSV order
        # still appear in the coordinator command.
        self.assertIn(
            "-e CUDA_VISIBLE_DEVICES=GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0",
            coordinator,
        )
        self.assertIn("GPU-95f8f212-9131-df99-fd53-7535965197d7", coordinator)
        self.assertLess(
            coordinator.index("CUTEAFD_NATIVE_LIB=/scratch/coord-native/libcuteafd_native.so"),
            coordinator.index("cuteafd-tpep-nvme-dev"),
        )
        self.assertLess(
            coordinator.index("CUDA_VISIBLE_DEVICES="),
            coordinator.index("cuteafd-tpep-nvme-dev"),
        )
        self.assertIn(f"coordinator_cuda_visible_devices={OPS_GPU_UUIDS}", result.stdout)
        self.assertIn("--native-lib /scratch/coord-native/libcuteafd_native.so", coordinator)
        self.assertIn("worker_geometry=coordinator-selected typed model admission", result.stdout)
        # Plan handshake before any worker, worker readiness before ack, gate last.
        order = [line for line in lines if line.startswith("STEP")]
        self.assertLess(order.index("STEP worker-bootstrap-ready"), order.index("STEP coordinator-start"))
        self.assertLess(order.index("STEP expert-start rank=0"), order.index("STEP worker-bootstrap-ready"))
        self.assertLess(order.index("STEP coordinator-start"), order.index("STEP worker-admission-ready"))
        self.assertLess(order.index("STEP worker-admission-ready"), order.index("STEP readiness-gate"))
        experts = [line for line in lines if line.startswith("CMD expert-")]
        self.assertEqual(len(experts), 4)
        for rank, line in enumerate(experts):
            self.assertIn(f"--rank {rank}", line)
            self.assertIn("--world 4", line)
            self.assertIn(f"--rank {rank} --world 4 --spark-tp 2 --spark-ep 2", line)
            self.assertIn("-e CUTEAFD_NATIVE_LIB=/scratch/candidate/libcuteafd_native.so", line)
            self.assertIn("-e RUST_LOG=info", line)
            self.assertIn("--native-lib /scratch/candidate/libcuteafd_native.so", line)
            self.assertLess(line.index("CUTEAFD_NATIVE_LIB="), line.index("cuteafd-tpep-dev"))
            # The GPU pin is coordinator-only.
            self.assertNotIn("CUDA_VISIBLE_DEVICES", line)

    def test_coordinator_gpu_pin_is_validated(self) -> None:
        config = EXAMPLES / "tp2ep2-native.config"
        u1 = "GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0"
        for pin, message in (
            (f"{u1},{u1}", "duplicate"),
            (u1, "exactly 2 UUID"),
            ("0,1", "not a physical GPU UUID"),
        ):
            with self.subTest(pin=pin):
                result = self.plan(config, "--coordinator-cuda-visible-devices", pin)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_qualified_requires_the_coordinator_gpu_pin(self) -> None:
        digest = "a" * 64
        common = (
            "--qualified",
            "--expect-coordinator-lib-sha256", digest,
            "--expect-spark-lib-sha256", digest,
            "--expect-coordinator-daemon-sha256", digest,
            "--expect-spark-daemon-sha256", digest,
        )
        result = self.plan(EXAMPLES / "tp2ep2-native.config", *common)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--qualified requires --coordinator-cuda-visible-devices", result.stderr)
        result = self.plan(
            EXAMPLES / "tp2ep2-native.config", *common,
            "--coordinator-cuda-visible-devices", OPS_GPU_UUIDS,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_six_rank_single_rtx_loads_from_layer_zero(self) -> None:
        result = self.plan(EXAMPLES / "tp3ep2-native.config")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn("topology tp=3 ep=2 world=6", lines)
        self.assertIn("ownership=coordinator-selected exact layer IDs", lines)
        self.assertNotIn("STEP plan-read", lines)
        experts = [line for line in lines if line.startswith("CMD expert-")]
        self.assertEqual(len(experts), 6)
        self.assertIn("rhea", [line for line in experts if "--rank 4" in line][0])
        self.assertIn("moa", [line for line in experts if "--rank 5" in line][0])
        self.assertTrue(all("--world 6 --spark-tp 3 --spark-ep 2" in line for line in experts))
        order = [line for line in lines if line.startswith("STEP")]
        self.assertLess(order.index("STEP expert-start rank=5"), order.index("STEP worker-bootstrap-ready"))
        self.assertLess(order.index("STEP worker-bootstrap-ready"), order.index("STEP coordinator-start"))

    def test_plan_supports_the_tp4_reference_arm(self) -> None:
        result = self.plan(EXAMPLES / "tp4ep1-explicit-native.config")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("topology tp=4 ep=1 world=4", result.stdout.splitlines())
        # No tp4 extra role: the reference arm runs without a role manifest.
        provider = next(line for line in result.stdout.splitlines() if line.startswith("CMD expert-0:"))
        self.assertIn("--world 4 --spark-tp 4 --spark-ep 1", provider)

    def test_default_config_without_explicit_topology_is_rejected(self) -> None:
        result = self.plan(CONFIG)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires explicit SPARK_TP", result.stderr)

    def test_examples_never_use_the_stale_secondary_rail(self) -> None:
        # The shipped default now carries the corrected six-host rail, so no
        # address names rhea/moa fabric A by accident.
        default = CONFIG.read_text()
        for stale in ("SPARK_0_LANE_B=10.55.0.5", "SPARK_1_LANE_B=10.55.0.6"):
            self.assertNotIn(stale, default)
        for corrected in (
            "SPARK_0_LANE_B=10.55.1.1", "SPARK_1_LANE_B=10.55.1.2",
            "SPARK_2_LANE_B=10.55.1.3", "SPARK_3_LANE_B=10.55.1.4",
            "SPARK_4_LANE_B=10.55.1.5", "SPARK_5_LANE_B=10.55.1.6",
        ):
            self.assertIn(corrected, default)
        for name in ("tp2ep2-native.config", "tp4ep1-explicit-native.config"):
            text = (EXAMPLES / name).read_text()
            self.assertNotIn("SPARK_0_LANE_B=10.55.0.5", text)
            self.assertNotIn("SPARK_1_LANE_B=10.55.0.6", text)
            self.assertIn("SPARK_0_LANE_B=10.55.1.1", text)
            self.assertIn("SPARK_1_LANE_B=10.55.1.2", text)
            self.assertIn("SPARK_2_LANE_B=10.55.1.3", text)
            self.assertIn("SPARK_3_LANE_B=10.55.1.4", text)

    def candidate(self, root: Path, *args: str, scenario: str = "normal",
                  env_extra: dict | None = None) -> subprocess.CompletedProcess[str]:
        stub_bin = root / "bin"
        stub_bin.mkdir(exist_ok=True)
        for name in ("docker", "ssh", "curl"):
            path = stub_bin / name
            path.write_text(CANDIDATE_STUB)
            path.chmod(0o755)
        env = dict(
            os.environ,
            PATH=f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
            STUB_SCENARIO=scenario,
            CUTEAFD_TPEP_L3_GRANT="1",
        )
        if env_extra:
            env.update(env_extra)
        return subprocess.run([str(CANDIDATE), *args], cwd=ROOT, text=True, capture_output=True, env=env)

    READY_PLAN = '{"version":1,"rtx_gpus":2,"nonce":"n","rtx_expert_layers":20,"spark_first_layer":20}'
    READY_PREFIX = "stale-prefix-line\n"

    def test_start_waits_for_worker_ready_before_reporting_ready(self) -> None:
        # Nonzero offset before the real ANSI line: the launcher must search only
        # the current epoch and strip ANSI before matching rank/world/layer_ids.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "ready-run",
                env_extra={"STUB_PLAN": self.READY_PLAN,
                           "STUB_LOG_OFFSET": str(len(self.READY_PREFIX)),
                           "STUB_LOG_FULL": self.READY_PREFIX + ANSI_READY_LINES},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("worker bootstrap ready", result.stdout)
            self.assertIn("candidate ready", result.stdout)
            self.assertLess(result.stdout.index("worker installed ready"), result.stdout.index("candidate ready"))

    def test_stale_worker_log_before_offset_does_not_satisfy_ready(self) -> None:
        # The readiness bytes precede the pre-start offset, so the current-epoch
        # slice is empty and the wait must time out rather than accept stale text.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            offset = str(len(self.READY_PREFIX))
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "stale-log-run",
                env_extra={"STUB_PLAN": self.READY_PLAN, "STUB_LOG_OFFSET": offset,
                           "STUB_LOG_FULL": self.READY_PREFIX,
                           "CUTEAFD_TPEP_WORKER_READY_TIMEOUT_SECONDS": "1"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("worker readiness timed out", result.stderr)

    def test_wrong_rank_and_split_line_do_not_satisfy_ready(self) -> None:
        wrong_rank = ANSI_READY_LINE.format(rank=10)  # rank=10 must not match rank=1
        split_line = (
            "\x1b[2m cuteafd: native worker bootstrap ready\x1b[0m\n"
            "\x1b[3mrank\x1b[0m\x1b[2m=\x1b[0m1 \x1b[3mworld\x1b[0m\x1b[2m=\x1b[0m4 "
            "\x1b[3mlayer_ids\x1b[0m\x1b[2m=\x1b[0m[20, 21]\n"
        )
        full = ANSI_READY_LINES
        no_bootstrap = full.replace("native worker bootstrap ready", "old worker ready")
        no_install = full.replace("coordinator-selected worker weights ready", "old worker ready")
        for name, body in (("wrong-rank", wrong_rank), ("split-line", split_line),
                           ("no-bootstrap", no_bootstrap), ("no-installed-admission", no_install)):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    result = self.candidate(
                        root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                        "--host-artifact-root", str(root), "--run-id", f"{name}-run",
                        env_extra={"STUB_PLAN": self.READY_PLAN,
                                   "STUB_LOG_OFFSET": str(len(self.READY_PREFIX)),
                                   "STUB_LOG_FULL": self.READY_PREFIX + body,
                                   "CUTEAFD_TPEP_WORKER_READY_TIMEOUT_SECONDS": "1"},
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("worker readiness timed out", result.stderr)

    def test_ssh_stat_failure_is_fatal_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "stat-fail-run",
                env_extra={"STUB_PLAN": self.READY_PLAN, "STUB_SSH_STAT_EXIT": "1"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed to read worker log size", result.stderr)

    def test_worker_death_before_ready_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "dead-worker-run",
                env_extra={"STUB_PLAN": self.READY_PLAN, "STUB_LOG_OFFSET": "0",
                           "STUB_LOG_FULL": "", "STUB_WORKER_STATUS": "stopped"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exited before reporting ready", result.stderr)

    def test_obsolete_worker_and_filesystem_flags_are_rejected(self) -> None:
        for flag, value in (("--first-layer", "20"), ("--capacity", "4096"),
                            ("--placement-root", "/unused"), ("--restart", None)):
            with self.subTest(flag=flag):
                args = (flag,) if value is None else (flag, value)
                result = self.plan(EXAMPLES / "tp2ep2-native.config", *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unknown candidate launcher argument", result.stderr)

    def test_coordinator_child_exit_fails_during_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "start", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), "--run-id", "child-run", scenario="child-exit", env_extra={"STUB_LOG_FULL": ANSI_READY_LINES},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exited during typed admission", result.stderr)

    def test_stop_reports_unreachable_hosts_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.candidate(
                root, "stop", "--config", str(EXAMPLES / "tp2ep2-native.config"),
                "--host-artifact-root", str(root), scenario="ssh-fail",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unreachable", result.stderr)
            self.assertIn("had one or more failures", result.stderr)



def run_sh_block(start: str, end: str) -> str:
    """Extract one literal run.sh block so a test can execute the real code."""
    source = (ROOT / "scripts/launch/run-family.sh").read_text()
    return source[source.index(start):source.index(end, source.index(start))]


class PublishedImageReferenceTest(unittest.TestCase):
    """Every example must name the pair that actually exists.

    The launcher resolves COORDINATOR_DOCKER_INFERENCE with a plain
    `docker image inspect`, so an example that pins an unpublished per-topology
    tag fails before any topology work, even though the published universal
    images serve that topology.
    """

    # The v10 promotion retargeted every example onto the promoted pair, so no
    # exemption remains: all of them are checked against `cuteafd.config`.
    def published_pair(self) -> tuple[str, str]:
        values = {}
        for line in CONFIG.read_text().splitlines():
            for key in ("COORDINATOR_DOCKER_INFERENCE", "SPARK_EXPERT_DOCKER_INFERENCE"):
                if line.startswith(f"{key}="):
                    values[key] = line.split("=", 1)[1]
        return values["COORDINATOR_DOCKER_INFERENCE"], values["SPARK_EXPERT_DOCKER_INFERENCE"]

    def test_every_example_names_the_published_release_pair(self) -> None:
        coordinator, spark = self.published_pair()
        self.assertTrue(coordinator.startswith("ghcr.io/"), coordinator)
        known = {"tp4ep1-explicit-native.config", "tp2ep2-native.config",
                 "tp3ep2-native.config", "tp2ep3-native.config", "tp6ep1-native.config"}
        paths = sorted(EXAMPLES.glob("*.config"))
        self.assertTrue(known <= {p.name for p in paths},
                        f"missing examples: {sorted(known - {p.name for p in paths})}")
        # Every example in the directory, present or later added, must be aligned.
        for path in paths:
            with self.subTest(example=path.name):
                values = dict(
                    line.split("=", 1)
                    for line in path.read_text().splitlines()
                    if line.startswith(("COORDINATOR_DOCKER_INFERENCE=",
                                        "SPARK_EXPERT_DOCKER_INFERENCE="))
                )
                self.assertEqual(values.get("COORDINATOR_DOCKER_INFERENCE"), coordinator)
                if "SPARK_EXPERT_DOCKER_INFERENCE" not in values:
                    # RTX-only launchers need no expert image and may reject its key.
                    self.assertIn("SPARK_COUNT=0", path.read_text().splitlines())
                else:
                    self.assertEqual(values["SPARK_EXPERT_DOCKER_INFERENCE"], spark)
                    # One release, one tag: a mixed pair fails the launcher's engine
                    # identity check at startup.
                    self.assertEqual(
                        values["COORDINATOR_DOCKER_INFERENCE"].rsplit(":", 1)[1],
                        values["SPARK_EXPERT_DOCKER_INFERENCE"].rsplit(":", 1)[1],
                    )

    def test_no_example_pins_an_unpublished_candidate_tag(self) -> None:
        for path in sorted(EXAMPLES.glob("*.config")):
            with self.subTest(example=path.name):
                for line in path.read_text().splitlines():
                    if not line.startswith(("COORDINATOR_DOCKER_INFERENCE=",
                                            "SPARK_EXPERT_DOCKER_INFERENCE=")):
                        continue
                    reference = line.split("=", 1)[1]
                    self.assertIn("/", reference, f"{path.name}: {reference} is a local-only tag")
                    self.assertNotIn("-candidate", reference, path.name)


class SparkHostPreflightTest(unittest.TestCase):
    """Run run.sh's real per-host SSH preflight through an OpenSSH-style shim.

    OpenSSH joins its command arguments into one remote shell line, so an empty
    argument is elided and every later positional shifts. The host name therefore
    has to survive an empty EXL3 family tag, and a failing check has to name the
    host it ran on. `ssh` is replaced by a shim and `docker` by a PATH stub,
    because the remote half runs in a child shell that inherits the environment
    but not shell functions.
    """

    DOCKER_STUB = """#!/usr/bin/env bash
if [[ "$1" == info ]]; then exit 0; fi
if [[ "$1" == image && "$2" == inspect ]]; then
  if [[ "$*" == *"-f "* ]]; then
    case "$*" in
      *image.revision*) printf '%s\\n' "$CUTEAFD_STUB_ENGINE"; exit 0 ;;
      *sparkinfer.revision*) printf '%s\\n' "$CUTEAFD_STUB_SPARKINFER"; exit 0 ;;
      *) printf '\\n'; exit 0 ;;
    esac
  fi
  if [[ "${CUTEAFD_STUB_MISSING:-0}" == 1 ]]; then exit 1; fi
  echo present
  exit 0
fi
echo "docker $*"
"""

    HARNESS = """set -euo pipefail
source scripts/lib/release-common.sh
ssh() {{ shift 5; bash -c "$*"; }}
release_die() {{ echo "die: $*" >&2; exit 1; }}
SPARK_EXPERT_DOCKER_INFERENCE=registry.example/spark:v9
engine_commit=engine-revision
sparkinfer_commit=sparkinfer-revision
snapshot_rel=hub/models--x--M/snapshots/rev
namespace_root=/mnt/sparknest
host=dodo
model_is_exl3={exl3}
exl3_family_tag={family}
wip_slot={wip_slot}
HOME={home}
unset HF_HOME
"""

    STATEMENT = None

    @classmethod
    def statement(cls) -> str:
        if cls.STATEMENT is None:
            source = (ROOT / "scripts/launch/run-family.sh").read_text()
            start = source.index('spark_manifest="$(release_ssh')
            end = source.index(' (see the messages above)"', start) + len(' (see the messages above)"')
            cls.STATEMENT = source[start:end]
        return cls.STATEMENT

    def preflight(self, *, exl3: str = "false", family: str = '""',
                  missing: int = 0, sparkinfer: str = "sparkinfer-revision",
                  wip_slot: str = '""') -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            # HF_HOME is unset below, so the remote resolves the snapshot under
            # $HOME/.cache/huggingface.
            (home / ".cache/huggingface/hub/models--x--M/snapshots/rev").mkdir(parents=True)
            stub = home / "bin"
            stub.mkdir()
            docker = stub / "docker"
            docker.write_text(self.DOCKER_STUB)
            docker.chmod(0o755)
            script = (self.HARNESS.format(exl3=exl3, family=family, wip_slot=wip_slot,
                                          home=str(home))
                      + self.statement()
                      + '\nprintf "manifest=[%s]\n" "$spark_manifest"\n')
            return subprocess.run(
                ["bash", "-c", script], cwd=ROOT, text=True, capture_output=True, timeout=20,
                env=dict(os.environ, PATH=f"{stub}{os.pathsep}{os.environ['PATH']}",
                         CUTEAFD_STUB_ENGINE="engine-revision",
                         CUTEAFD_STUB_SPARKINFER=sparkinfer,
                         CUTEAFD_STUB_MISSING=str(missing)),
            )

    def test_native_empty_family_tag_completes_without_shifting_arguments(self) -> None:
        result = self.preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "manifest=[]\n")

    def test_failing_check_names_the_host_it_ran_on(self) -> None:
        # The host now travels ahead of the optional family tag, so an empty
        # family value cannot elide it: this diagnostic has to name the host the
        # check ran on, not an empty or shifted value.
        result = self.preflight(missing=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("spark preflight on dodo: inference image is missing", result.stderr)
        self.assertIn("Spark host preflight failed on dodo", result.stderr)

    def test_sparkinfer_mismatch_names_both_revisions_and_the_rebuild(self) -> None:
        # A pin bump leaves every image behind the lock; the refusal must say
        # which revision each side has and which rebuild fixes that launch kind.
        release = self.preflight(sparkinfer="old-revision")
        self.assertNotEqual(release.returncode, 0)
        self.assertIn("carries SparkInfer old-revision but this checkout pins sparkinfer-revision",
                      release.stderr)
        self.assertIn("./build.sh", release.stderr)
        wip = self.preflight(sparkinfer="old-revision", wip_slot='"slot"')
        self.assertNotEqual(wip.returncode, 0)
        self.assertNotIn("carries SparkInfer", wip.stderr)
        self.assertIn("Spark host preflight failed on dodo", wip.stderr)

    def test_exl3_family_tag_reaches_the_manifest_path(self) -> None:
        result = self.preflight(exl3="true", family='"k23"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("exl3-k23/manifest.json", result.stdout)
        self.assertNotIn("exl3-/manifest.json", result.stdout)

    def test_empty_family_exl3_launch_uses_the_legacy_manifest(self) -> None:
        result = self.preflight(exl3="true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("exl3-/manifest.json", result.stdout)


class SparkRoleGateTest(unittest.TestCase):
    """The published universal image must satisfy every explicit topology."""

    def gate(self, *, required: str, advertised: str | None, hosts: int = 3,
             ssh_rc: int = 0) -> subprocess.CompletedProcess[str]:
        block = run_sh_block('spark_advertised_roles=""', "\n# Zero-Spark deployments")
        script = f'''set -euo pipefail
source scripts/lib/release-common.sh
release_die() {{ echo "die: $*" >&2; exit 1; }}
ssh() {{ [[ {ssh_rc} -eq 0 ]] || {{ echo "No such image" >&2; return 255; }}; printf '%s\\n' '{advertised}'; }}
hosts=(h0 h1 h2 h3 h4 h5)
hosts=("${{hosts[@]:0:{hosts}}}")
spark_tp=2
spark_tp_roles_required={required!r}
SPARK_EXPERT_DOCKER_INFERENCE=registry.example/spark:v9
''' + block + '\nprintf "advertised=%s\\n" "$spark_advertised_roles"\n'
        return subprocess.run(["bash", "-c", script], cwd=ROOT,
                              text=True, capture_output=True, timeout=20)

    def wip_gate(self, *, roles: list[str], verified: bool = True,
                 corrupt_hash: bool = False, corrupt_host: bool = False,
                 expert_format: str = 'native', nvfp4_roles: list[str] | None = None) -> subprocess.CompletedProcess[str]:
        block = run_sh_block('spark_advertised_roles=""', "\n# Zero-Spark deployments")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for host in ('h0', 'h1', 'h2'):
                slot = root / host / 'slot' / 'spark-expert'
                artifacts = slot / 'workspace/.cuteafd-wip'
                artifacts.mkdir(parents=True)
                (artifacts / 'libcuteafd_native.so').write_bytes(b'test library')
                manifest = json.dumps({'schema': 1, 'spark_tp_roles': roles,
                                      'nvfp4_spark_tp_roles': nvfp4_roles or [],
                                      'symbols_verified': verified,
                                      'native_library_sha256': hashlib.sha256(b'test library').hexdigest()}).encode()
                (artifacts / 'V41_EXPERT_TP_AOT.json').write_bytes(manifest)
                meta = json.dumps({'schema': 1, 'slot': 'slot', 'role': 'spark-expert',
                                   'spark_tp_roles': roles,
                                   'v41_expert_tp_manifest_sha256': 'bad' if corrupt_hash or (corrupt_host and host == 'h2') else hashlib.sha256(manifest).hexdigest()}).encode()
                (slot / 'META.json').write_bytes(meta)
                (slot / 'FINGERPRINT').write_text(hashlib.sha256(meta).hexdigest())
            # Execute the actual per-host Python guard; Docker and SSH alone are mocked.
            script = '''set -euo pipefail
source scripts/lib/release-common.sh
release_die() { echo "die: $*" >&2; exit 1; }
release_ssh() {
  [[ "$1" == -o ]]; shift 2
  host="$1"; shift
  # The remote command runs the guard in the builder container when it exists, else on the WIP root.
  [[ "$1" == *"docker exec -i 'wip-test' python3 - "* && "$1" == *"WIP_SLOTS="* ]]
  read -r -a w <<<"$(tr -d "'" <<<"$1")"
  python3 -c 'import sys; code=sys.stdin.read().replace("/wip/slots",sys.argv[1]); sys.argv=["guard",*sys.argv[2:]]; exec(compile(code,"guard","exec"))' "$MOCK_SLOTS/$host" "${w[-4]}" "${w[-3]}" "${w[-2]%;}"
}
hosts=(h0 h1 h2)
spark_tp=3
spark_tp_roles_required=tp3
wip_slot=slot
wip_spark_container=wip-test
SPARK_EXPERT_DOCKER_INFERENCE=unused
''' + f'expert_format={expert_format}\n' + block + '\nprintf "advertised=%s\\n" "$spark_advertised_roles"\n'
            return subprocess.run(['bash', '-c', script], cwd=ROOT, text=True,
                                  capture_output=True, timeout=20,
                                  env={**os.environ, 'MOCK_SLOTS': str(root)})

    def test_wip_accepts_verified_tp3_slot(self) -> None:
        result = self.wip_gate(roles=['tp3', 'tp6'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('advertised=tp3;tp6', result.stdout)

    def test_wip_nvfp4_requires_its_own_verified_role(self) -> None:
        result = self.wip_gate(roles=['tp3'], expert_format='nvfp4')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('NVFP4 TP role is not built', result.stderr)
        result = self.wip_gate(roles=['tp3'], expert_format='nvfp4', nvfp4_roles=['tp3'])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wip_rejects_missing_role_or_unverified_symbols(self) -> None:
        for roles, verified in ((['tp6'], True), (['tp3'], False)):
            with self.subTest(roles=roles, verified=verified):
                result = self.wip_gate(roles=roles, verified=verified)
                self.assertNotEqual(result.returncode, 0)

    def test_wip_rejects_bad_manifest_hash_on_each_host(self) -> None:
        result = self.wip_gate(roles=['tp3'], corrupt_hash=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('TP manifest hash mismatch', result.stderr)
        result = self.wip_gate(roles=['tp3'], corrupt_host=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('h2 cannot verify', result.stderr)

    def test_universal_label_satisfies_every_required_role(self) -> None:
        for required in ("tp2", "tp3", "tp6"):
            with self.subTest(required=required):
                result = self.gate(required=required, advertised="tp2;tp3;tp6")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("advertised=tp2;tp3;tp6", result.stdout)

    def test_subset_and_legacy_labels_are_refused(self) -> None:
        for required, advertised in (("tp6", "tp2;tp3"), ("tp3", ""), ("tp2", "tp4")):
            with self.subTest(required=required, advertised=advertised):
                result = self.gate(required=required, advertised=advertised)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"does not advertise required expert role {required}", result.stderr)
                self.assertIn("published universal release pair", result.stderr)

    def test_legacy_tp4_launch_never_probes_the_role_label(self) -> None:
        result = self.gate(required="", advertised=None, ssh_rc=255)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unreadable_label_is_reported_per_host(self) -> None:
        result = self.gate(required="tp2", advertised=None, hosts=2, ssh_rc=255)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("h0 cannot report the role label", result.stderr)


if __name__ == "__main__":
    unittest.main()
