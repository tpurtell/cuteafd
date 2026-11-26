"""docs/options.md must match the launcher's real per-family key acceptance.

The family x key table is parsed from the doc and compared, for every family
and every key, against the launcher's own validation function
(`release_reject_unused_keys`, the function `run-family.sh` calls before a
launch): accepted keys must exit 0, and refused keys must die with
`unused configuration key: KEY` and status 2. No source-text grepping.
"""

from __future__ import annotations

import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "options.md"

# Document column label -> family id the launcher uses.
FAMILIES = {
    "V4.1": "deepseek_v41",
    "V4": "deepseek_v4",
    "GLM 5.3 / ML4": "glm5",
    "GLM Flash": "glm5_flash",
    "MiMo": "mimo_v2",
    "Qwen4": "qwen4",
    "Qwen dense": "qwen3_5",
    "Kimi K3": "kimi_k3",
    "Step-5": "step5",
    "Muse": "muse_spark",
    "Inkling": "inkling2",
    "Inkling Mini": "inkling2_mini",
    "Hy4-flash": "hy4_flash",
    "MiniMax": "minimax_m3",
}
COLUMNS = list(FAMILIES)

BASH_PRELUDE = """
set -euo pipefail
source scripts/lib/release-common.sh
source scripts/lib/family-keys.sh
source scripts/lib/attention-placement.sh
allowed_keys() {
  if [[ "$1" == deepseek_v41 ]]; then _v41_allowed_keys; else _family_allowed_keys "$1"; fi
}
"""


def run_bash(script: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", BASH_PRELUDE + script],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
    )


def family_keys(family: str) -> set[str]:
    """The keys the family's launch path passes to release_reject_unused_keys."""
    result = subprocess.run(
        ["bash", "-c", BASH_PRELUDE + 'allowed_keys "$1"', "_", family],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # release_reject_unused_keys always allows the shared site/build keys.
    return set(result.stdout.split()) | {"WIP_BUILD_SEED"}


def tables(lines: list[str]):
    """Yield (section, header_cells, body_rows) for each markdown table."""
    index = 0
    section = ""
    while index < len(lines):
        if lines[index].startswith("## "):
            section = lines[index][3:].strip()
        if not lines[index].startswith("|"):
            index += 1
            continue
        block = []
        while index < len(lines) and lines[index].startswith("|"):
            block.append(lines[index])
            index += 1
        rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block]
        yield section, rows[0], rows[2:]


def parse_doc() -> tuple[dict[str, dict[str, bool]], set[str]]:
    """Return the documented matrix and every key named in the area tables."""
    lines = DOC.read_text().splitlines()
    matrix: dict[str, dict[str, bool]] = {}
    documented_keys: set[str] = set()
    found_matrix = False
    for section, header, rows in tables(lines):
        is_matrix = section == "Family support: launch keys" and header == ["Key", *COLUMNS]
        if is_matrix:
            found_matrix = True
            for row in rows:
                assert len(row) == len(header), f"short matrix row: {row}"
                key = row[0].strip("`")
                if key.startswith("**"):
                    continue
                cells = row[1:]
                assert all(re.fullmatch(r"[✓✗](?: \(.+\))?", cell) for cell in cells), row
                assert key not in matrix, f"duplicate matrix row: {key}"
                matrix[key] = {column: cell.startswith("✓") for column, cell in zip(COLUMNS, cells)}
            continue
        if header and header[0] == "Key":
            for row in rows:
                key_cell = row[0]
                if key_cell.startswith("**"):
                    continue
                # Grouped rows name several keys: `A`, `B (alias `C`)`.
                documented_keys.update(re.findall(r"`([^`]+)`", key_cell))
    assert found_matrix, "docs/options.md has no family x key table with the documented columns"
    return matrix, documented_keys


def expand(key: str) -> list[str]:
    match = re.fullmatch(r"SPARK_<rank>_(\w+)", key)
    if match:
        return [f"SPARK_{rank}_{match.group(1)}" for rank in range(8)]
    return [key]


MATRIX, DOCUMENTED = parse_doc()
UNION = set().union(*(family_keys(family) for family in FAMILIES.values()))

ATTENTION_VALUES = {"auto", "heads", "context", "layers"}


def label(cell: str) -> str:
    """`auto` (default) -> auto."""
    match = re.search(r"`([^`]+)`", cell)
    return match.group(1) if match else cell


def parse_attention() -> dict[str, dict[str, str]]:
    """The value table whose rows are the ATTENTION_PLACEMENT values."""
    for section, header, rows in tables(DOC.read_text().splitlines()):
        if section != "Family support: option values" or header[1:] != COLUMNS:
            continue
        if {label(row[0]) for row in rows} != ATTENTION_VALUES:
            continue
        table: dict[str, dict[str, str]] = {}
        for row in rows:
            cells = row[1:]
            assert len(cells) == len(COLUMNS), f"short attention row: {row}"
            assert all(cell[:1] in {"✓", "✗"} for cell in cells), row
            table[label(row[0])] = {
                column: cell[:1] for column, cell in zip(COLUMNS, cells)
            }
        return table
    raise AssertionError("docs/options.md has no ATTENTION_PLACEMENT value table")


ATTENTION = parse_attention()

LEVERS_SECTION = "Optional performance levers"
LEVER_COLUMNS = ["Lever", "Families", "Enable", "Effect", "Cost"]


def parse_levers() -> list[dict[str, object]]:
    """The optional, default-off performance-lever rows."""
    for section, header, rows in tables(DOC.read_text().splitlines()):
        if section != LEVERS_SECTION:
            continue
        assert header == LEVER_COLUMNS, f"{LEVERS_SECTION} header is {header}"
        levers: list[dict[str, object]] = []
        for row in rows:
            assert len(row) == len(LEVER_COLUMNS), f"short lever row: {row}"
            keys = re.findall(r"`([A-Z_0-9]+)`", row[0])
            assert keys, f"lever row names no key: {row}"
            families = [family.strip() for family in row[1].split(",") if family.strip()]
            assert families, f"lever row {keys} names no family"
            assert all(cell for cell in row[2:]), f"lever row {keys} has an empty cell"
            levers.append({"keys": keys, "families": families, "enable": row[2]})
        assert levers, f"{LEVERS_SECTION} lists no levers"
        return levers
    raise AssertionError(f"docs/options.md has no {LEVERS_SECTION} section")


def parse_defaults() -> dict[str, str]:
    """Every key's documented default from the Key | Default area tables."""
    defaults: dict[str, str] = {}
    for _, header, rows in tables(DOC.read_text().splitlines()):
        if header[:2] != ["Key", "Default"]:
            continue
        for row in rows:
            for key in re.findall(r"`([A-Z_0-9]+)`", row[0]):
                value = "unset" if row[1].startswith("unset") else label(row[1])
                assert defaults.setdefault(key, value) == value, \
                    f"{key} has two documented defaults"
    return defaults


LEVERS = parse_levers()
DEFAULTS = parse_defaults()


def test_unset_defaults_preserve_resolved_value_annotations(monkeypatch, tmp_path):
    doc = tmp_path / "options.md"
    doc.write_text(
        "## Test\n| Key | Default |\n| --- | --- |\n"
        "| `CORE_VIEW` | unset (= `original`) |\n"
        "| `FP8_PREFILL` | unset (= `auto`) |\n"
        "| `GLM5_FLASH_KDA_FP8` | unset (`auto` = `off`) |\n"
        "| `EXPERT_USAGE` | `on` |\n")
    monkeypatch.setitem(globals(), "DOC", doc)
    assert parse_defaults() == {
        "CORE_VIEW": "unset", "FP8_PREFILL": "unset",
        "GLM5_FLASH_KDA_FP8": "unset", "EXPERT_USAGE": "on",
    }


def test_tp2_dflash_default_table_matches_supported_families():
    header, rows = next((header, rows) for section, header, rows in tables(DOC.read_text().splitlines())
                        if section == "Family support: option values" and header[0] == "Layout")
    assert header[1:] == [family for family in COLUMNS if MATRIX["TP2_DFLASH"][family]]
    assert rows == [
        ["Two RTX with DFlash2 / DFly2", "`on`", "`on`", "`on`", "checkpoint geometry", "`on`", "`on`"],
        ["One RTX or no DFlash2 / DFly2", "`off`", "`off`", "`off`", "`off`", "`off`", "`off`"],
    ]
    assert DEFAULTS["TP2_DFLASH"] == "on"


def test_graph_reserve_keys_are_separate_and_bounded_cache_families_only():
    # Families on the bounded graph cache (#1627) take the reserve keys.
    bounded = {"mimo_v2", "kimi_k3", "glm5", "glm5_flash", "qwen4", "deepseek_v4", "deepseek_v41", "qwen3_5", "muse_spark", "step5", "inkling2", "inkling2_mini", "hy4_flash", "minimax_m3"}
    for family in FAMILIES.values():
        keys = family_keys(family)
        assert "GRAPH_RESERVE" not in keys
        for key in ("GPU_GRAPH_RESERVE", "GPU_GRAPH_RESERVE_EXTRA", "GPU_GRAPH_RESERVE_EXTRA_CAP",
                    "GRAPH_TRACE"):
            assert (key in keys) == (family in bounded), (family, key)
        # Accepted worker keys do not imply an active worker graph owner.
        assert ("SPARK_GRAPH_RESERVE" in keys) == (family in {"mimo_v2", "glm5", "glm5_flash", "qwen4", "deepseek_v41"}), family
    assert "GRAPH_RESERVE" not in MATRIX


def test_area_tables_cover_every_matrix_key():
    for key in MATRIX:
        # A grouped area row may abbreviate a family of keys with a trailing *.
        covered = any(
            fnmatch(expanded, pattern)
            for documented in DOCUMENTED
            for pattern in expand(documented)
            for expanded in expand(key)
        )
        assert covered, f"{key} is in the family table but no option section documents it"
    for documented in DOCUMENTED:
        if "*" in documented:
            continue
        assert documented in MATRIX, f"option section documents undocumented key {documented}"


@pytest.mark.parametrize("column", COLUMNS)
def test_family_table_matches_launcher(column: str):
    family = FAMILIES[column]
    accepted = family_keys(family)
    documented = set()
    for key, cells in MATRIX.items():
        if not cells[column]:
            continue
        documented.update(expand(key))
    assert documented == accepted, (
        f"{column}: docs say {sorted(documented - accepted)} are accepted but the launcher "
        f"refuses them; launcher accepts {sorted(accepted - documented)} the docs mark refused"
    )


@pytest.mark.parametrize("column", COLUMNS)
def test_reject_matches_doc_for_every_key(column: str):
    """Drive release_reject_unused_keys with each key and compare with the doc."""
    family = FAMILIES[column]
    expected = {
        key: cells[column]
        for row, cells in MATRIX.items()
        for key in expand(row)
    }
    keys = "\n".join(sorted(expected)) + "\n"
    script = r"""
family="$1"
allow=$(allowed_keys "$family")
declare -A cfg_file=()
while IFS= read -r key; do
  [[ -n "$key" ]] || continue
  cfg_file=([$key]=dummy)
  if output=$( ( release_reject_unused_keys "$family" $allow ) 2>&1 ); then
    printf 'accept %s\n' "$key"
  else
    status=$?
    printf 'refuse %s %s %s\n' "$key" "$status" "$output"
  fi
done
"""
    result = subprocess.run(
        ["bash", "-c", BASH_PRELUDE + script, "_", family],
        cwd=ROOT,
        input=keys,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    seen = set()
    for line in result.stdout.splitlines():
        words = line.split(" ", 3)
        verdict, key = words[0], words[1]
        seen.add(key)
        want = expected[key]
        if want:
            assert verdict == "accept", f"{family} {key}: doc says accepted, launcher refused: {line}"
        else:
            assert verdict == "refuse", f"{family} {key}: doc says refused, launcher accepted"
            status, message = words[2], words[3]
            assert status == "2", line
            assert f"unused configuration key: {key}" in message, line
    assert seen == set(expected), f"{family}: {sorted(set(expected) - seen)} not exercised"


@pytest.mark.parametrize("column", COLUMNS)
def test_attention_value_table_matches_launcher(column: str):
    """Drive the launcher's ATTENTION_PLACEMENT validation for every cell."""
    family = FAMILIES[column]
    expected = {value: cells[column] for value, cells in ATTENTION.items()}
    script = r"""
family="$1"
shift
for placement in "$@"; do
  if output=$( ( release_check_attention_placement "$placement" "$family" ) 2>&1 ); then
    printf 'accept %s\n' "$placement"
  else
    status=$?
    printf 'refuse %s %s %s\n' "$placement" "$status" "$output"
  fi
done
"""
    result = subprocess.run(
        ["bash", "-c", BASH_PRELUDE + script, "_", family, *sorted(expected)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    seen = set()
    for line in result.stdout.splitlines():
        words = line.split(" ", 3)
        verdict, value = words[0], words[1]
        seen.add(value)
        if expected[value] == "✓":
            assert verdict == "accept", f"{family} {value}: doc says accepted, launcher refused: {line}"
        else:
            assert verdict == "refuse", f"{family} {value}: doc says refused, launcher accepted"
            assert words[2] == "2", line
            assert "attention" in words[3].lower(), line
    assert seen == set(expected), f"{family}: {sorted(set(expected) - seen)} not exercised"


@pytest.mark.parametrize("placement", sorted(ATTENTION_VALUES))
def test_kimi_accepts_explicit_mla_placements(placement: str):
    result = run_bash(f"release_check_attention_placement {placement} kimi_k3")
    assert result.returncode == 0, result.stderr
    assert ATTENTION[placement]["Kimi K3"] == "✓"


def test_doc_is_linked():
    readme = (ROOT / "README.md").read_text()
    examples = (ROOT / "examples" / "configs" / "README.md").read_text()
    assert "docs/options.md" in readme
    assert "docs/options.md" in examples


def test_every_key_is_a_known_config_key():
    keys = "\n".join(sorted(UNION)) + "\n"
    script = r"""
while IFS= read -r key; do
  [[ -n "$key" ]] || continue
  release_known_key "$key" || { echo "unknown $key"; exit 1; }
done
"""
    result = run_bash(script, keys)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("path", sorted((ROOT / "examples/configs").glob("*.config")), ids=lambda p: p.name)
def test_example_media_values_pass_the_release_config_validator(path: Path, tmp_path: Path):
    # Exercise release_load_config itself, without checkpoint or image lookups:
    # stop mode still validates values but needs no serving/model inventory.
    media = [line for line in path.read_text().splitlines()
             if line.startswith(("VISION=", "AUDIO="))]
    config = tmp_path / "media.config"
    config.write_text("\n".join(media) + "\n")
    result = run_bash(f'release_load_config "{config}" stop')
    assert result.returncode == 0, f"{path.name}: {result.stderr}"


def test_optional_levers_are_default_off_and_documented():
    """Every lever is a real launch key, off by default and accepted where listed."""
    seen = set()
    for lever in LEVERS:
        for key in lever["keys"]:
            for family in lever["families"]:
                assert (key, family) not in seen, f"duplicate lever row for {key} on {family}"
                seen.add((key, family))
            assert key in MATRIX, f"lever {key} is absent from the family support table"
            assert key in DEFAULTS, f"lever {key} has no documented default"
            default = DEFAULTS[key]
            if key == "GRAPH_PREWARM":
                assert lever["families"] == COLUMNS
                assert default == "on"
                assert "`GRAPH_PREWARM=off`" in DOC.read_text()
                assert 'case "$(get GRAPH_PREWARM on)"' in (ROOT / "scripts/launch/run-family.sh").read_text()
            elif key == "GRAPH_DONOR_ALWAYS":
                assert default == "off"
                assert lever["families"] == COLUMNS
            elif key == "GRAPH_DONOR_COLD_LAG":
                assert default == "0"
                assert lever["families"] == COLUMNS
                assert '`GRAPH_DONOR_ALWAYS=on GRAPH_DONOR_COLD_LAG=16`' in DOC.read_text()
                assert 'graph_donor_cold_lag="$(get GRAPH_DONOR_COLD_LAG 0)"' in (ROOT / "scripts/launch/run-family.sh").read_text()
            elif key == "DECODE_LANES":
                # V4.1 defaults by coordinator RTX count (1 on 1× RTX, 2 on 2× RTX); explicit 1/2 overrides.
                assert lever["families"] == ["V4.1"]
                assert default == "unset"
                assert "`DECODE_LANES=1`" in lever["enable"] and "`DECODE_LANES=2`" in lever["enable"]
            elif key == "DRAFT_POLICY":
                # The lever is the opt-in `clamped` policy; `history` stays the default.
                assert default == "history"
                assert lever["families"] == COLUMNS
                assert "`DRAFT_POLICY=clamped`" in lever["enable"]
                assert "clamped) draft_policy_args=(--draft-policy clamped) ;;" in (ROOT / "scripts/launch/run-family.sh").read_text()
            elif key == "DRAFT_CONTEXT_WINDOW":
                assert default == "0"
                assert lever["families"] == ["Kimi K3"]
                assert 'draft_context_window="$(get DRAFT_CONTEXT_WINDOW 0)"' in (ROOT / "scripts/launch/run-family.sh").read_text()
            elif key == "KV_CACHE":
                assert lever["families"] == ["Muse"]
                assert default == "family default"
                assert 'muse_kv_cache="$(get KV_CACHE bf16)"' in (ROOT / "scripts/launch/run-family.sh").read_text()
            elif key == "ATTENTION_PRECISION":
                assert default == "bf16"
                assert "`ATTENTION_PRECISION=fp8`" in lever["enable"]
            elif key == "SPECULATOR_FP8":
                assert lever["families"] == ["Kimi K3"]
                assert default == "auto"
                source = (ROOT / "rust/crates/cuteafd-core/src/launch_settings.rs").read_text()
                assert 'selected(explicit, None, dflash,' in source
                assert 'kimi_draft_fp8(args.draft_fp8, args.speculator == "dflash2")' in (
                    ROOT / "rust/crates/cuteafd-daemon/src/commands/plan.rs").read_text()
            elif key == "DECODE_ROWS":
                assert lever["families"] == ["V4.1", "GLM Flash", "MiniMax", "Inkling", "Inkling Mini", "Kimi K3", "Qwen4", "Muse"]
                assert default == "auto"
                assert "`DECODE_ROWS=128`" in lever["enable"]
            else:
                assert default.startswith(("off", "unset")), \
                    f"lever {key} default is {default!r}, not off/unset"
            for family in lever["families"]:
                assert family in COLUMNS, f"lever {key} names unknown family column {family}"
                assert MATRIX[key][family], f"lever {key} is not accepted by {family}"
    assert seen


def test_glmf_documents_lazy_exact_rows_and_working_set_reserve():
    text = DOC.read_text()
    assert "GLM Flash uses exact live-row keys" in text
    assert "`max(1.2 × W1, W1 + 100 MB, W1 + 2 × step)`" in text
    assert "twice the largest measured single-capture growth" in text


def test_minimax_bundled_dspark_is_a_family_default_not_an_optional_lever():
    assert all("MiniMax" not in lever["families"] for lever in LEVERS if "SPECULATOR" in lever["keys"])
    assert "SPECULATOR" in family_keys("minimax_m3")
    text = DOC.read_text()
    assert "`SPECULATOR=dspark` is the default" in text
    assert "`SPECULATOR=off`" in text and "`--dspark=false`" in text
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    selection = launcher.split('speculator="$(get SPECULATOR)"', 1)[1].split(
        'case "$family:$speculator" in', 1)[0]
    for value in ("", "auto", "off", "dspark"):
        script = 'speculator="$1"; family=minimax_m3; ' + selection
        result = subprocess.run(["bash", "-c", script + '\nprintf "%s" "$speculator"', "_", value],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ("off" if value == "off" else "dspark")


def test_optional_levers_section_is_near_the_top_and_linked():
    text = DOC.read_text()
    assert text.index(f"## {LEVERS_SECTION}") < text.index("## How options are set")
    assert "[Optional performance levers](#optional-performance-levers)" in text
    agents = (ROOT / "AGENTS.md").read_text()
    assert "docs/options.md#optional-performance-levers" in agents


def test_embedding_value_support_and_expert_usage_adapters():
    text = DOC.read_text()
    section = text.split("Token embedding placement (`EMBEDDING`", 1)[1].split("Speculator", 1)[0]
    host = next(line for line in section.splitlines() if line.startswith("| `host` |"))
    assert [cell.strip() for cell in host.split("|")[2:-1]] == ["✓"] * 7 + ["✗"] * 7
    assert {family for family in COLUMNS if MATRIX["EXPERT_USAGE"][family]} == {
        "V4.1", "V4", "GLM 5.3 / ML4", "MiMo", "Kimi K3", "Hy4-flash", "MiniMax"}


def test_qwen_pool_first_onboarding_documents_whole_and_hybrid_tp2():
    text = DOC.read_text()
    assert "`RTX_EXPERT_LAYERS=auto` is pool-first on one or two RTX devices" in text
    assert "Spark-backed layouts: whole routed layers on one RTX, TP2 halves on two" in text
    assert "`--rtx-expert-layers` does not require `--split-device`" in text


def test_muse_target_kv_precision_is_opt_in_and_draft_is_independent():
    for _, header, rows in tables(DOC.read_text().splitlines()):
        if header == ["Value", *COLUMNS] and any(label(row[0]) == "bf16" for row in rows):
            precision = {label(row[0]): dict(zip(COLUMNS, row[1:])) for row in rows}
            assert precision["bf16"]["Muse"] == "✓ (default)"
            assert precision["fp8"]["Muse"] == "✓ (opt-in)"
            break
    else:
        raise AssertionError("missing KV precision table")
    assert "KV_CACHE" in family_keys("muse_spark")
    assert "DRAFT_KV_CACHE" in family_keys("muse_spark")
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    assert 'family_args+=(--kv-cache "$muse_kv_cache")' in launcher
    assert 'plan_draft_args+=(--kv-cache "$muse_kv_cache")' in launcher
    assert 'muse_draft_kv="$(get DRAFT_KV_CACHE fp8)"' in launcher



def test_inkling_global_kv_precision_defaults_fp8_and_accepts_bf16():
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    assert "KV_CACHE" in family_keys("inkling2")
    assert "KV_CACHE" in family_keys("inkling2_mini")
    assert 'inkling_kv="$(get KV_CACHE fp8)"' in launcher
    assert 'family_args+=(--kv-cache "$inkling_kv")' in launcher
    assert 'plan_draft_args+=(--kv-cache "$inkling_kv")' in launcher
    assert "2,112 bytes per global layer" in DOC.read_text()

def test_mc_mtp_docs_distinguish_original_core_from_unchanged_experts():
    text = DOC.read_text()
    assert "projections stay K4 in every view" not in text
    assert "restore native BF16 MTP core matrices in `original`" in text
    assert "MTP routed experts are unchanged" in text
    assert "neither loaded nor charged" in text
    assert "not converted by `fp8-compact`" in text
    for family in ["glm5", "glm5_flash", "step5"]:
        doc = (ROOT / "docs" / "models" / f"{family}.md").read_text()
        core = doc.split("## Core views", 1)[1].split("\n## ", 1)[0]
        assert "MTP core" in core and "BF16" in core and "K4 MTP" in core


def test_mc_core_view_owns_glm_target_conversion_without_global_switch():
    text = DOC.read_text()
    assert "CUTEAFD_GLM_BF16" not in text
    assert "CUTEAFD_GLM_BF16" not in (ROOT / "scripts/lib/family-keys.sh").read_text()
    assert "CUTEAFD_GLM_BF16" not in (ROOT / "scripts/launch/run-family.sh").read_text()
    assert "`CORE_VIEW`" in text and "`fp8-compact`" in text


def test_mc_fp8_lever_reports_measured_glm_and_step_quality_costs():
    text = DOC.read_text()
    assert "49.04/180.68/3392 to 53.52/191.81/3122" in text
    assert "59.87/216.73/2740 to 68.98/266.95/2562" in text
    assert "0.04530/0.03880" in text and "0.01597/0.01429" in text
    assert "no paired non-inferiority claim" in text


def test_every_family_accepts_the_shared_draft_policy_key():
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    assert "DRAFT_POLICY must be history, clamped or off" in launcher
    assert "CUTEAFD_DRAFT_POLICY" not in launcher
    for removed in ("DRAFT_SURVIVAL_LENGTHS", "DRAFT_ROW_COST"):
        assert removed not in MATRIX and removed not in launcher
    assert all(MATRIX["DRAFT_POLICY"].values())
    for family in ("deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4", "qwen3_5", "kimi_k3", "step5",
                   "muse_spark", "minimax_m3", "inkling2", "inkling2_mini", "hy4_flash"):
        assert "DRAFT_POLICY" in family_keys(family), family



def test_attention_precision_is_shared_and_has_no_legacy_alias():
    assert "ATTENTION_PRECISION" in MATRIX
    for family in FAMILIES.values():
        keys = family_keys(family)
        assert "ATTENTION_PRECISION" in keys
        assert "ATTENTION_PREFILL_PRECISION" not in keys
    rows = [row for _, header, rows in tables(DOC.read_text().splitlines())
            if header[:3] == ["Key", "Default", "Values"] for row in rows
            if row[0] == "`ATTENTION_PRECISION`"]
    assert len(rows) == 1
    assert rows[0][1:3] == ["`bf16`", "`bf16`, `fp8`"]


@pytest.mark.parametrize("checkpoint_family", ["glm5", "mistral_large4"])
def test_serve_glm_standard_option_contract(checkpoint_family):
    result = run_bash(f"_family_allowed_keys glm5 {checkpoint_family}")
    assert result.returncode == 0, result.stderr
    keys = set(result.stdout.split())
    assert {"FP8_PREFILL", "DECODE_GRAPHS", "HTTP_QUEUE_DEPTH", "HTTP_QUEUE_WAIT_MS", "MEDIA_CACHE_BYTES",
            "PREFILL_ROWS", "PREFILL_WAVES", "PREFILL_SPAN_ROWS", "TP2_DFLASH"} <= keys
    assert not {"CHAT_TEMPLATE_FROM", "EXPERT_INPUT", "RTX_EXPERT_PEER",
                "PREFILL_BATCH_TOKENS"} & keys



def test_v41_public_id_and_option_limits_are_documented():
    assert MATRIX["SERVED_MODEL_ID"]["V4.1"]
    assert MATRIX["AUDIO"]["V4.1"]
    assert MATRIX["ATTENTION_PRECISION"]["V4.1"]
    text = DOC.read_text()
    assert "| `AUDIO` | `off` (default; no tower) |" in text
    assert "| `ATTENTION_PRECISION` | ✓ (`fp8` resolves to `bf16`) |" in text


def test_retired_v41_keys_are_not_known_or_documented():
    for key in ("MODEL_VARIANT", "EXPERT_FORMAT", "SPARKINFER_EXL3", "SPARK_REDUCTION_MIN_ROWS", "KV_POOL_TOKENS"):
        assert key not in MATRIX and key not in DOCUMENTED and key not in UNION
        assert run_bash(f"release_known_key {key}").returncode != 0
