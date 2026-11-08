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
    return set(result.stdout.split()) | {"CUTEAFD_EXPORT_HOST", "WIP_BUILD_SEED"}


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
                assert all(cell in {"✓", "✗"} for cell in cells), row
                assert key not in matrix, f"duplicate matrix row: {key}"
                matrix[key] = {column: cell == "✓" for column, cell in zip(COLUMNS, cells)}
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
