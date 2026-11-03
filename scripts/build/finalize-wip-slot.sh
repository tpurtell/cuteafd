#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: finalize-wip-slot.sh SOURCE_DIR ROLE SLOT BUILD_OUTPUT IMAGE IMAGE_ID [IMAGE_LAYERS]" >&2
  exit 2
fi

source_dir="$(realpath "$1")"
role="$2"
slot="$3"
build_output="$(realpath "$4")"
image="$5"
image_id="$6"
# The image's RootFS layer list (JSON), its identity across Docker's classic and
# containerd stores, where save/load gives the same image a different .Id.
image_layers="${7:-}"
wip_instance="${WIP_INSTANCE:-}"
wip_root="${WIP_ROOT:-}"
wip_build_key="${CUTEAFD_WIP_BUILD_KEY:-}"
[[ "$role" == coordinator || "$role" == spark-expert ]] || {
  echo "invalid WIP role: $role" >&2
  exit 2
}
[[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || {
  echo "invalid WIP slot: $slot" >&2
  exit 2
}
[[ -x "$build_output/cuteafd" && -s "$build_output/libcuteafd_native.so" ]] || {
  echo "WIP build output is incomplete: $build_output" >&2
  exit 2
}
# Direct WIP builds must not seal a coordinator that cannot serve its config.
if [[ "$role" == coordinator && -n "${CUTEAFD_WIP_PROGRAM_FAMILY:-}" ]]; then
  python3 "$source_dir/scripts/lib/wip_aot_scopes.py" \
    --family "$CUTEAFD_WIP_PROGRAM_FAMILY" --check-programs "$build_output/PROGRAMS.json"
fi
(
  cd "$build_output"
  sha256sum -c ARTIFACT_SHA256SUMS
)
# EXLR8 slots: the engine's own fail-closed loader opens the built table with its
# manifest (pointer/scalar ABI, scratch, geometry) and requires the role's programs.
if [[ "${CUTEAFD_WIP_EXLR8_AOT:-OFF}" == ON ]]; then
  "$build_output/cuteafd" programs-check --native-lib "$build_output/libcuteafd_native.so" \
    --manifest "$build_output/PROGRAMS.json" \
    --exlr8-role "$([[ "$role" == coordinator ]] && echo coordinator || echo spark)" \
    --exlr8-family "${CUTEAFD_WIP_EXLR8_FAMILIES:-}"
fi

slot_root="/wip/slots/$slot/$role"
incoming="/wip/incoming/${slot}.${role}.$$"
rm -rf "$incoming"
mkdir -p "$incoming/workspace/.cuteafd-wip"
cp -a "$source_dir/." "$incoming/workspace/"
install -m 0755 "$build_output/cuteafd" "$incoming/workspace/.cuteafd-wip/cuteafd"
if [[ -f "$build_output/COMPILER_PROVENANCE.json" ]]; then
  install -m 0644 "$build_output/COMPILER_PROVENANCE.json" "$incoming/workspace/.cuteafd-wip/COMPILER_PROVENANCE.json"
fi
install -m 0755 \
  "$build_output/libcuteafd_native.so" \
  "$incoming/workspace/.cuteafd-wip/libcuteafd_native.so"
install -m 0644 \
  "$build_output/V41_EXPERT_AOT.json" \
  "$incoming/workspace/.cuteafd-wip/V41_EXPERT_AOT.json"
install -m 0644 \
  "$build_output/V41_EXPERT_TP_AOT.json" \
  "$incoming/workspace/.cuteafd-wip/V41_EXPERT_TP_AOT.json"
install -m 0644 "$build_output/V41_FP8_AOT.json" "$incoming/workspace/.cuteafd-wip/V41_FP8_AOT.json"
cp -a "$build_output/V41_EXPERT_WORKSPACES" "$incoming/workspace/.cuteafd-wip/V41_EXPERT_WORKSPACES"
# Host-ISA variants of the daemon and library (scripts/build/host-isa.sh), in
# ARTIFACT_SHA256SUMS like the base files.
for variants in bin-variants lib-variants; do
  [[ ! -d "$build_output/$variants" ]] || cp -a "$build_output/$variants" "$incoming/workspace/.cuteafd-wip/$variants"
done
install -m 0644 "$build_output/PROGRAMS.json" "$incoming/workspace/.cuteafd-wip/PROGRAMS.json"
if [[ -d "$build_output/exl3" ]]; then
  mkdir -p "$incoming/workspace/.cuteafd-wip"
  cp -a "$build_output/exl3" "$incoming/workspace/.cuteafd-wip/exl3"
fi
if [[ -d "$build_output/fp8" ]]; then
  mkdir -p "$incoming/workspace/.cuteafd-wip"
  cp -a "$build_output/fp8" "$incoming/workspace/.cuteafd-wip/fp8"
fi
install -m 0644 \
  "$build_output/ARTIFACT_SHA256SUMS" \
  "$incoming/workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS"

python3 "$source_dir/scripts/build/verify-release-source-manifest.py" \
  --source "$incoming/workspace" \
  --write "$incoming/SOURCE_SHA256SUMS"
source_manifest_sha256="$(sha256sum "$incoming/SOURCE_SHA256SUMS" | awk '{print $1}')"
artifact_sha256="$(sha256sum "$incoming/workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS" | awk '{print $1}')"
sparkinfer_revision="$(
  python3 "$source_dir/scripts/build/verify-sparkinfer-source.py" \
    --source "$source_dir/third_party/sparkinfer" \
    --lock "$source_dir/third_party/sparkinfer.lock.json" \
    --print-revision
)"

transformers_revision="$(python3 "$source_dir/scripts/build/verify-transformers-source.py" \
  --source "$incoming/workspace/third_party/transformers" \
  --lock "$incoming/workspace/third_party/transformers.lock.json" --print-revision)"
transformers_source_digest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_tree_sha256"])' \
  "$incoming/workspace/third_party/transformers.lock.json")"

python3 - "$incoming/META.json" \
  "$incoming/workspace/.cuteafd-wip/V41_EXPERT_TP_AOT.json" <<PY
import hashlib
import json
import pathlib
import time

path = pathlib.Path(__import__("sys").argv[1])
tp_manifest_path = pathlib.Path(__import__("sys").argv[2])
# The role identity is read from the artifact that was just hash-verified by
# ARTIFACT_SHA256SUMS; it is never passed in as an independent claim.
tp_manifest_bytes = tp_manifest_path.read_bytes()
tp_manifest = json.loads(tp_manifest_bytes)
assert tp_manifest.get("schema") == 1, tp_manifest
roles = tp_manifest.get("spark_tp_roles")
assert isinstance(roles, list) and all(isinstance(role, str) for role in roles), roles
metadata = {
    "schema": 1,
    "wip_instance": ${wip_instance@Q},
    "wip_root": ${wip_root@Q},
    "slot": ${slot@Q},
    "role": ${role@Q},
    "base_image": ${image@Q},
    "base_image_id": ${image_id@Q},
    "source_manifest_sha256": ${source_manifest_sha256@Q},
    "artifact_manifest_sha256": ${artifact_sha256@Q},
    "sparkinfer_revision": ${sparkinfer_revision@Q},
    "transformers_revision": ${transformers_revision@Q},
    "transformers_source_sha256": ${transformers_source_digest@Q},
    "spark_tp_roles": roles,
    "v41_expert_tp_manifest_sha256": hashlib.sha256(tp_manifest_bytes).hexdigest(),
    "built_unix": int(time.time()),
}
# Build key (wip.sh: source tree + scopes + toolchain) and per-phase timings,
# when the caller recorded them; older callers leave both absent.
build_key = ${wip_build_key@Q}
if build_key:
    metadata["build_key"] = build_key
# Sealed where IMAGE_ID is this host's store-local ID: the layer list is the
# store-independent identity remote Sparks must match (golden-run).
image_layers = ${image_layers@Q}
if image_layers:
    metadata["base_image_layers"] = json.loads(image_layers)
phases_path = pathlib.Path(${build_output@Q}) / "BUILD_PHASES.tsv"
if phases_path.is_file():
    phases = []
    for line in phases_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].isdigit():
            phases.append({"role": parts[0], "phase": parts[1], "seconds": int(parts[2]), "detail": parts[3] if len(parts) > 3 else ""})
    metadata["build_phases"] = phases
path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$incoming/META.json" | awk '{print $1}' >"$incoming/FINGERPRINT"

mkdir -p "$(dirname "$slot_root")"
rm -rf "$slot_root"
mv "$incoming" "$slot_root"
echo "WIP slot finalized: slot=$slot role=$role fingerprint=$(<"$slot_root/FINGERPRINT")"
