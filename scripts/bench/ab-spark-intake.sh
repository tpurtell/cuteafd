#!/usr/bin/env bash
# Spark A/B for how routed-expert partials reach the coordinator GPU
# (rust/crates/cuteafd-daemon/src/shared/spark_intake.rs): the base engine's host
# staging against this checkout's intakes (host, pinned, gpu, auto), with the
# real Spark ranks. Run the steps in order once raptor GPU0 and the Sparks are
# free (one model served at a time; this script stops what it launched):
#
#   scripts/bench/ab-spark-intake.sh build          # coordinator images intake-base / intake-new
#   scripts/bench/ab-spark-intake.sh glmf 3         # GLM 5.3 Flash TP4 (ostrich..kiwi): 8K prefill, C1 decode
#   scripts/bench/ab-spark-intake.sh pro 3          # V4 Pro EXL3 K2 TP6 (all six Sparks): 8K prefill, C1 decode
#   scripts/bench/ab-spark-intake.sh v41 4          # V4.1 Flash decode parity (C1, C16) via bench-ab.py
#   scripts/bench/ab-spark-intake.sh summary        # per-arm medians of every results file
#
# build: clones BASE_REV (default: merge-base with origin/work/p0) and NEW_REV
# (default: HEAD) into $OUT/src-{base,new}, compiles the coordinator release
# artifacts of each in the development image exactly as build.sh does (p7's
# family set), and layers them over OVERLAY_FROM (the p7 coordinator image) as
# cuteafd-coordinator:intake-{base,new}. Spark images stay SPARK_IMAGE (p7):
# the workers are unchanged. glmf/pro arms: base (intake-base image) and
# host/pinned/gpu/auto (intake-new with SPARK_INTAKE=...); rounds alternate
# the arm order. v41 runs scripts/bench/bench-ab.py between the two clones (its
# decode path, V41Tp4RocePending::receive_owned, is untouched by this change).
#
# Environment: OUT (default ~/.cache/cuteafd/builds/intake-ab), BASE_REV,
# PRO_RANKS (6: TP6 over all six Sparks; 4 for TP4 on ostrich..kiwi),
# NEW_REV, OVERLAY_FROM, SPARK_IMAGE, BUILD_GPU (0), ARMS (glmf/pro arm list),
# V41_CONFIG (a working V4.1 cuteafd.config; its images are replaced),
# PROMPT_FILE (text for the 8K prompt, default this repo's PLAN.md).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$HOME/.cache/cuteafd/builds/intake-ab}"
OVERLAY_FROM="${OVERLAY_FROM:-ghcr.io/tpurtell/cuteafd-coordinator:p7}"
SPARK_IMAGE="${SPARK_IMAGE:-ghcr.io/tpurtell/cuteafd-spark-expert:p7}"
ARMS="${ARMS:-base host pinned gpu auto}"
PROMPT_FILE="${PROMPT_FILE:-$repo/PLAN.md}"
mkdir -p "$OUT"
python3 "$repo/scripts/build/assert-build-filesystem.py" "$OUT"

# Release coordinator families (p7's set; see build.sh --help).
export CUTEAFD_RELEASE_DSV4_AOT=ON CUTEAFD_RELEASE_GLM_AOT=ON CUTEAFD_RELEASE_GLMF_AOT=ON
export CUTEAFD_RELEASE_MIMO_AOT=ON CUTEAFD_RELEASE_QWEN4_AOT=ON CUTEAFD_RELEASE_MIMO_GEOMETRIES=mimo,mimo2,mimop,mimop2
export CUTEAFD_RELEASE_EXPERT_FAMILIES="${CUTEAFD_RELEASE_EXPERT_FAMILIES:-dsv4f:spark;dsv4f:rtx_backbone;dsv4f:rtx_tp2;dsv4p:rtx_tp2;dsv4p:exl3-k23;glm:exl3-k45;glm:fp8;glmf:exl3-k34;mimo:fp8;mimop:fp8;qwen4:exl3-k45}"

clone() { # clone NAME REV: standalone clone (build containers cannot see worktree gitdirs)
  local dir="$OUT/src-$1" rev="$2" sub name
  if [[ ! -d "$dir/.git" ]]; then
    git clone -q "$repo" "$dir"
  fi
  git -C "$dir" fetch -q "$repo" "$rev"
  git -C "$dir" checkout -q --detach "$rev"
  # Submodule commits from local stores when they have them (the fork checkout
  # next to this repo, then the main clone's module store), else upstream.
  local main_git; main_git="$(cd "$repo" && realpath "$(git rev-parse --git-common-dir)")"
  for sub in sparkinfer xgrammar; do
    local want; want="$(git -C "$dir" ls-tree HEAD "third_party/$sub" | awk '{print $3}')"
    local url=""
    for candidate in "${SPARKINFER_SRC:-$repo/../sparkinfer-glmrt}" "$main_git/modules/third_party/$sub"; do
      [[ "$sub" == xgrammar && "$candidate" != "$main_git"* ]] && continue
      if git -C "$candidate" cat-file -e "$want^{commit}" 2>/dev/null; then url="$(realpath "$candidate")"; break; fi
    done
    if [[ -n "$url" ]]; then
      git -C "$dir" config "submodule.third_party/$sub.url" "$url"
      [[ -e "$dir/third_party/$sub/.git" ]] && git -C "$dir/third_party/$sub" remote set-url origin "$url"
      git -C "$dir" -c protocol.file.allow=always submodule update -q --init "third_party/$sub"
    else
      git -C "$dir" submodule update -q --init "third_party/$sub"
    fi
  done
  # XGrammar's own submodules (dlpack, ...) from the main clone's store.
  local nested="$main_git/modules/third_party/xgrammar/modules/3rdparty"
  for name in $(git -C "$dir/third_party/xgrammar" config -f .gitmodules --name-only --get-regexp 'submodule\..*\.url' |
      sed 's/^submodule\.//; s/\.url$//'); do
    [[ -d "$nested/${name#3rdparty/}" ]] &&
      git -C "$dir/third_party/xgrammar" config "submodule.$name.url" "$nested/${name#3rdparty/}"
  done
  git -C "$dir/third_party/xgrammar" -c protocol.file.allow=always submodule update -q --init --recursive
  git -C "$dir/third_party/sparkinfer" remote set-url origin https://github.com/tpurtell/sparkinfer-glmrt.git
  git -C "$dir/third_party/xgrammar" remote set-url origin https://github.com/mlc-ai/xgrammar.git
}

build_arm() { # build_arm NAME REV
  local name="$1" rev="$2" src="$OUT/src-$1" art="$OUT/artifacts-$1" root="$OUT/build-root-$1"
  docker image inspect cuteafd-coordinator-dev:latest >/dev/null 2>&1 ||
    { echo "cuteafd-coordinator-dev:latest is missing; build it as build.sh does (docker/Dockerfile.dev)" >&2; exit 1; }
  clone "$name" "$rev"
  rm -rf "$art" "$root"; mkdir -p "$art" "$root"
  python3 "$repo/scripts/build/assert-build-filesystem.py" "$art" "$root"
  echo "== $name ($rev): coordinator release artifacts"
  docker run --rm --gpus "device=${BUILD_GPU:-0}" --ipc=host --ulimit memlock=-1:-1 \
    -e CUTEAFD_RELEASE_BUILD_ROOT="$root" -v "$root:$root" \
    -e CUTEAFD_RELEASE_EXPERT_FAMILIES -e CUTEAFD_RELEASE_DSV4_AOT -e CUTEAFD_RELEASE_GLM_AOT \
    -e CUTEAFD_RELEASE_MIMO_AOT -e CUTEAFD_RELEASE_MIMO_GEOMETRIES -e CUTEAFD_RELEASE_GLMF_AOT \
    -e CUTEAFD_RELEASE_QWEN4_AOT -v "$src:/source:ro" -v "$art:/output" \
    cuteafd-coordinator-dev:latest /source/scripts/build/build-release-artifacts.sh /source coordinator 120 /output \
    > "$OUT/build-$name.log" 2>&1 || { tail -20 "$OUT/build-$name.log"; exit 1; }
  cat > "$art/Dockerfile" <<EOF
FROM $OVERLAY_FROM
RUN rm -rf /opt/cuteafd/lib/exl3 /opt/cuteafd/lib/fp8
COPY cuteafd /opt/cuteafd/bin/cuteafd
COPY libcuteafd_native.so /opt/cuteafd/lib/libcuteafd_native.so
COPY exl3 /opt/cuteafd/lib/exl3
COPY fp8 /opt/cuteafd/lib/fp8
COPY *.json /opt/cuteafd/share/
LABEL io.cuteafd.intake-ab.revision=$rev org.opencontainers.image.revision=$rev \
  io.cuteafd.sparkinfer.revision=$(git -C "$src/third_party/sparkinfer" rev-parse HEAD)
EOF
  docker build -q -t "cuteafd-coordinator:intake-$name" "$art" >/dev/null
  echo "== cuteafd-coordinator:intake-$name ready"
}

stop_all() { # stop_all SPARK_COUNT PORT
  docker rm -f cuteafd-coordinator >/dev/null 2>&1 || true
  local hosts=(ostrich dodo emu kiwi rhea moa)
  for ((r = 0; r < $1; r++)); do
    ssh "${hosts[$r]}" "docker rm -f cuteafd-spark-expert-${hosts[$r]}-$2 >/dev/null 2>&1 || true" &
  done
  wait
}

model_config() { # model_config glmf|pro ARM FILE
  local model="$1" arm="$2" file="$3" count port image intake=auto
  case "$model" in
    glmf) count=4; port=19495
      printf '%s\n' MODEL_ID=wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 ADDR=0.0.0.0:8600 > "$file" ;;
    pro) count="${PRO_RANKS:-6}"; port=19441
      printf '%s\n' MODEL_ID=wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 ADDR=0.0.0.0:8600 > "$file" ;;
  esac
  image=cuteafd-coordinator:intake-new
  [[ "$arm" == base ]] && image=cuteafd-coordinator:intake-base || intake="$arm"
  local hosts=(ostrich dodo emu kiwi rhea moa)
  {
    echo "COORDINATOR_GPU=0"; echo "EXPERT_PORT=$port"; echo "SPARK_COUNT=$count"
    echo "SPARK_DEVICE_BUDGET_BYTES=110000000000"; echo "CONCURRENCY=4"
    echo "MAX_CONTEXT_TOKENS=32768"; echo "MAX_OUTPUT_TOKENS=4096"; echo "SPARK_INTAKE=$intake"
    for ((r = 0; r < count; r++)); do echo "SPARK_${r}_HOST=${hosts[$r]}"; echo "SPARK_${r}_LANE_A=10.55.0.$((r + 1))"; done
    echo "COORDINATOR_DOCKER_INFERENCE=$image"; echo "SPARK_EXPERT_DOCKER_INFERENCE=$SPARK_IMAGE"
  } >> "$file"
  echo "$count $port"
}

bench() { # bench URL LABEL ROUND: 3x ~8K-token prefill (TTFT, 1 output token), 3x C1 256-token decode
  python3 - "$1" "$2" "$3" "$PROMPT_FILE" <<'PY'
import json, sys, time, urllib.request
url, label, rnd, prompt_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
def stream(content, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "enable_thinking": False}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for c in chunk.get("choices", []):
                d = c.get("delta", {})
                if (d.get("content") or d.get("reasoning_content")):
                    now = time.time(); first = first or now; last = now
    n = usage.get("completion_tokens", 0)
    return usage.get("prompt_tokens", 0), (first or time.time()) - t0, (n - 1) / (last - first) if first and last and last > first and n > 1 else float("nan")
base = open(prompt_file).read()
stream("warm up", 8)
for i in range(3):
    text = (f"[{label} {i} {time.time()}] " + base * 8)[:30000] + "\n\nSummarize the text above in one word."
    tokens, ttft, _ = stream(text, 1)
    print(json.dumps({"label": label, "round": rnd, "kind": "prefill", "tokens": tokens, "ttft_s": round(ttft, 3),
                      "tok_s": round(tokens / ttft)}))
for i, topic in enumerate(["a lighthouse keeper who finds a message in a bottle", "how a CPU executes an instruction",
                           "growing tomatoes at home"]):
    _, ttft, rate = stream(f"Write a detailed 600-word piece about {topic}.", 256)
    print(json.dumps({"label": label, "round": rnd, "kind": "c1", "decode_tok_s": round(rate, 1),
                      "ttft_s": round(ttft, 3)}))
PY
}

run_model() { # run_model glmf|pro ROUNDS
  local model="$1" rounds="$2" dir="$OUT/$1"
  mkdir -p "$dir"
  read -ra arms <<< "$ARMS"
  for ((round = 1; round <= rounds; round++)); do
    local order=("${arms[@]}")
    (( round % 2 == 0 )) && order=($(printf '%s\n' "${arms[@]}" | tac))
    for arm in "${order[@]}"; do
      local cfg="$dir/$arm.config" count port
      read -r count port < <(model_config "$model" "$arm" "$cfg")
      stop_all "$count" "$port"
      "$repo/scripts/launch/run-family.sh" --config "$cfg" > "$dir/launch-$arm-$round.log" 2>&1 ||
        { tail -20 "$dir/launch-$arm-$round.log"; exit 1; }
      docker logs cuteafd-coordinator 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -i "intake" | head -5 |
        sed "s/^/$arm r$round: /" | tee -a "$dir/results.txt" || true
      bench http://127.0.0.1:8600 "$arm" "$round" | tee -a "$dir/results.jsonl"
      docker logs cuteafd-coordinator 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$dir/serve-$arm-$round.log"
    done
  done
  stop_all "$count" "$port"
}

summary() {
  python3 - "$OUT" <<'PY'
import json, statistics, sys, pathlib, collections
for path in sorted(pathlib.Path(sys.argv[1]).glob("*/results.jsonl")):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.startswith("{")]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["kind"] == "prefill": by[r["label"]]["prefill tok/s"].append(r["tok_s"])
        else: by[r["label"]]["C1 decode tok/s"].append(r["decode_tok_s"])
    print(f"== {path.parent.name}")
    base = by.get("base", {})
    for label, metrics in by.items():
        parts = []
        for key, values in metrics.items():
            m = statistics.median(values)
            ratio = f" ({m / statistics.median(base[key]):.3f}x base)" if key in base and label != "base" else ""
            parts.append(f"{key} {m:.1f}{ratio} n={len(values)}")
        print(f"{label:7} " + " | ".join(parts))
PY
}

case "${1:-}" in
  build)
    BASE_REV="${BASE_REV:-$(git -C "$repo" merge-base HEAD origin/work/p0)}"
    NEW_REV="${NEW_REV:-$(git -C "$repo" rev-parse HEAD)}"
    build_arm base "$BASE_REV"
    build_arm new "$NEW_REV"
    ;;
  glmf|pro) run_model "$1" "${2:-3}" ;;
  v41)
    : "${V41_CONFIG:?set V41_CONFIG to a working V4.1 cuteafd.config}"
    # run.sh requires the Spark image to carry the coordinator's engine and
    # SparkInfer revisions. The Spark side is SPARK_IMAGE unchanged for both
    # arms; a per-arm local tag relabels it (its V4.1 kernels are not rebuilt).
    for arm in base new; do
      labels="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}} {{index .Config.Labels "io.cuteafd.sparkinfer.revision"}}' "cuteafd-coordinator:intake-$arm")"
      for host in $(sed -n 's/^SPARK_[0-3]_HOST=//p' "$V41_CONFIG"); do
        printf 'FROM %s\nLABEL org.opencontainers.image.revision=%s io.cuteafd.sparkinfer.revision=%s io.cuteafd.intake-ab.spark-from=%s\n' \
          "$SPARK_IMAGE" ${labels} "$SPARK_IMAGE" | ssh "$host" "docker build -q -t cuteafd-spark-expert:intake-$arm -" >/dev/null &
      done
      wait
    done
    for arm in base new; do
      sed -e "s#^COORDINATOR_DOCKER_INFERENCE=.*#COORDINATOR_DOCKER_INFERENCE=cuteafd-coordinator:intake-$arm#" \
          -e "s#^SPARK_EXPERT_DOCKER_INFERENCE=.*#SPARK_EXPERT_DOCKER_INFERENCE=cuteafd-spark-expert:intake-$arm#" \
          "$V41_CONFIG" > "$OUT/src-$arm/cuteafd.config"
    done
    "$repo/.venv/bin/python" "$repo/scripts/bench/bench-ab.py" --label intake-v41 \
      --arm base="$OUT/src-base" --arm new="$OUT/src-new" --layouts 1 --sessions "${2:-4}" --concurrency 1 16
    ;;
  summary) summary ;;
  *) sed -n '2,32p' "$0"; exit 2 ;;
esac
