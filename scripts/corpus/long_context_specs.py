"""Declarative case specifications for the long-context calibration bank (16K & 32K).

Component ``long_context`` extends ``wrldsuksgo2mars/agentic-quant-panel-v2`` with
16 long sequences (~352K tokens total):
- 16K class (10 sequences, ~160K tokens): 8 calibration (~128K) + 2 heldout (~32K)
- 32K class (6 sequences, ~192K tokens): 4 calibration (~128K) + 2 heldout (~64K)

Every record is stored in model-neutral chat format (``messages`` plus optional
``tools``) with per-record provenance, source hashes, and license notes. Token
IDs are never stored; token counts are estimated using local reference
tokenizers for length targeting and metadata reporting only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRAINING_DATA_ROOT = Path.home() / "Developer/training-data"
V2_PANEL_SNAPSHOT = Path(
    "/mnt/sparknest/hf-home/hub/datasets--wrldsuksgo2mars--agentic-quant-panel-v2"
    "/snapshots/54683731eb01c7c2fa5b801a0888e029a0e1b1c7"
)
FIDELITY_CACHE_ROOT = Path.home() / ".cache/cuteafd/fidelity/wrldsuksgo2mars--cuteafd-fidelity"
OPENROUTER_KEY_PATH = Path.home() / ".config/cuteafd/secrets/openrouter-key"

REFERENCE_TOKENIZERS: dict[str, str] = {
    "deepseek_v41": (
        "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
        "/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
    ),
    "glm_53": (
        "/mnt/sparknest/hf-home/hub/models--zai-org--GLM-5.3"
        "/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42"
    ),
    "qwen_38": (
        "/mnt/sparknest/hf-home/hub/models--Qwen--Qwen3.8-27B"
        "/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    ),
}

CHAT_TEMPLATE_VERIFY_TOKENIZERS: dict[str, str] = {
    "glm_53": REFERENCE_TOKENIZERS["glm_53"],
    "qwen_38": REFERENCE_TOKENIZERS["qwen_38"],
    "hy4_flash": (
        "/mnt/sparknest/hf-home/hub/models--tencent--Hy4-flash"
        "/snapshots/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
    ),
}

# Documented OpenRouter first-party max_completion_tokens (capped at 64K per call so
# frontier reasoning models never truncate mid-thought while staying within every model's ceiling).
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "moonshotai/kimi-k3.1-pro": 65536,
    "z-ai/glm-5.3": 65536,
    "z-ai/glm-5.3-flash": 65536,
    "qwen/qwen3.8-max-0902": 65536,
    "qwen/qwen3.8-flash": 65536,
    "mistralai/mistral-large-4-0": 65536,
    "tencent/hy4": 65536,
    "tencent/hy4-flash": 65536,
    "xiaomi/mimo-v2.6-pro": 65536,
    "xiaomi/mimo-v2.6-flash": 65536,
    "stepfun/step-5-preview": 64000,
    "openai/gpt-6-astra": 65536,
}


@dataclass(frozen=True)
class CaseSpec:
    id: str
    length_class: str  # "16k" or "32k"
    split: str  # "calibration" or "heldout"
    group: str
    source_kind: str  # "openrouter_agentic", "claude_cli", "codex_cli", "wiki_qa", "oss_repo_qa"
    archetype: str
    task_name: str
    languages: tuple[str, ...]
    model: str
    provider_only: tuple[str, ...] = ()
    expected_provider_names: tuple[str, ...] = ()
    expect_reasoning: bool = True
    reasoning_effort: str = "high"
    target_tokens: int = 16384
    min_tokens: int = 14800
    max_tokens: int = 18000
    system_prompt: str = ""
    phase_prompts: tuple[str, ...] = ()
    local_sources: tuple[str, ...] = ()
    oss_repo_dir: str = ""
    oss_repo_slug: str = ""
    oss_repo_commit: str = ""
    oss_include_globs: tuple[str, ...] = ()
    license_notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


AGENTIC_SYSTEM_PROMPT = (
    "You are an expert systems and software engineer working inside a sandboxed "
    "workspace. Use the provided file and shell tools to inspect, write, edit, "
    "compile, and thoroughly test your code. Rely only on the language's standard "
    "library (no external network downloads or third-party packages). Keep your "
    "reasoning concise and action-oriented: prefer calling file/shell tools "
    "immediately rather than writing long essays before acting. Always run real "
    "tests and verify every requirement before finishing each phase."
)


def _agentic_phases(
    *,
    lang: str,
    build_cmd: str,
    project_desc: str,
    phase1: str,
    phase2: str,
    phase3: str,
    phase4: str,
    phase5: str = "",
) -> tuple[str, ...]:
    phases = [
        f"Phase 1 — Test-driven initial implementation ({lang}, stdlib only, test via `{build_cmd}`):\n"
        f"{project_desc}\n\n{phase1}\n\n"
        "Use strict TDD: FIRST write the minimal project skeleton and unit test suite with `write_file` "
        f"and run `{build_cmd}` with `run_shell` to confirm the tests fail (red). THEN implement the "
        f"modules with `write_file`/`edit_file`, run `{build_cmd}` to verify all tests pass (green), "
        "fix any warnings or failures, and provide a concise summary.",
        f"Phase 2 — Test-driven feature extension & edge cases:\n{phase2}\n\n"
        "Follow test-driven development: FIRST read the existing files, add failing unit tests "
        f"for the new features, and run `{build_cmd}` to verify the new tests fail as expected. "
        "THEN implement the features with `edit_file`/`write_file` and re-run "
        f"`{build_cmd}` until every test passes cleanly.",
        f"Phase 3 — Bug investigation, stress testing & invariants:\n{phase3}\n\n"
        "Add deterministic stress/property tests that exercise boundary sizes, corrupted/malformed "
        f"inputs, and invariants; run `{build_cmd}`, fix any defects or warnings you uncover, "
        f"and re-run `{build_cmd}` until all tests pass cleanly.",
        f"Phase 4 — Refactoring, diagnostics & CLI/export tooling:\n{phase4}\n\n"
        "Refactor any duplicated helpers, add structured diagnostic/stats reporting and a small "
        "self-contained CLI or benchmark harness, add integration tests covering the end-to-end "
        f"workflow, and run `{build_cmd}` to verify zero warnings and 100% passing tests.",
    ]
    if phase5:
        phases.append(
            f"Phase 5 — Deep audit & persistence/recovery hardening:\n{phase5}\n\n"
            "Inspect all files you wrote from Phase 1 onward, check every invariant against the "
            "earliest data structures, add crash-recovery / round-trip persistence tests, and "
            f"confirm `{build_cmd}` passes completely."
        )
    return tuple(phases)


CASES: tuple[CaseSpec, ...] = (
    # =========================================================================
    # 32K LENGTH CLASS (4 calibration ~128K + 2 heldout ~64K)
    # =========================================================================
    CaseSpec(
        id="lc-32k-cal-01",
        length_class="32k",
        split="calibration",
        group="agentic:rust-lsm-wal-engine",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="rust_lsm_wal_storage_engine",
        languages=("Rust",),
        model="moonshotai/kimi-k3.1-pro",
        provider_only=("moonshotai",),
        expected_provider_names=("Moonshot AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=32768,
        min_tokens=27800,
        max_tokens=37800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="Rust (Cargo, stdlib only, no external crates)",
            build_cmd="cargo test",
            project_desc=(
                "Build a self-contained embedded LSM-tree key-value storage engine crate `minilsm` "
                "with Cargo.toml (edition 2021, zero dependencies)."
            ),
            phase1=(
                "Implement: (1) a checksummed Write-Ahead Log (`src/wal.rs`) with custom CRC32, "
                "length-prefixed frames, and torn-tail recovery; (2) an in-memory sorted MemTable "
                "(`src/memtable.rs`) backed by BTreeMap with MVCC sequence numbers and tombstones; "
                "(3) immutable sorted-string tables (`src/sstable.rs`) with 4KB data blocks, "
                "sparse index block, Bloom filter (pure Rust FNV/Murmur double hashing), and footer; "
                "(4) `Db` engine API (`src/db.rs`) supporting `put`, `delete`, `get`, `scan(start..end)`, "
                "and automatic memtable flush to L0 SSTables."
            ),
            phase2=(
                "Extend `minilsm` with: (1) Snapshot isolation (`db.snapshot()`) supporting consistent "
                "point reads and range iterators at historical sequence numbers; (2) Leveled compaction "
                "(`src/compaction.rs`) merging overlapping L0 tables into sorted non-overlapping L1/L2 "
                "runs while dropping shadowed versions older than the oldest active snapshot; "
                "(3) Atomic `WriteBatch` commits that write a single atomic batch record to the WAL."
            ),
            phase3=(
                "Write a comprehensive fault-injection and property test suite (`tests/recovery_and_mvcc.rs`): "
                "simulate truncated WAL records, bit-flipped SSTable block checksums, interleaved snapshots "
                "during compaction, delete-then-reinsert tombstones across 3 levels, and a randomized "
                "model-checking test comparing `Db` against a reference BTreeMap over 2,000 deterministic "
                "operations with periodic reopens and compactions."
            ),
            phase4=(
                "Add block-level prefix compression for keys in SSTables, a `src/stats.rs` metrics "
                "collector (write amplification, bloom false-positive count, level byte sizes, live vs "
                "tombstone entries), and a `src/bin/lsm_inspect.rs` CLI that dumps SSTable/WAL headers "
                "and runs a micro-benchmark workload. Update and run `cargo test`."
            ),
            phase5=(
                "Add a MANIFEST log (`src/manifest.rs`) tracking live SSTable sets and sequence watermarks "
                "across restarts so orphaned partial compaction outputs are garbage-collected on open. "
                "Verify full crash-and-reopen consistency across all earlier modules."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via Moonshot AI (moonshotai/kimi-k3.1-pro) in a clean stdlib-only Rust sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-32k-cal-02",
        length_class="32k",
        split="calibration",
        group="agentic:python-minisql-vm",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="python_minisql_planner_and_vm",
        languages=("Python",),
        model="qwen/qwen3.8-max-0902",
        provider_only=("alibaba",),
        expected_provider_names=("Alibaba",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=32768,
        min_tokens=27800,
        max_tokens=37800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="Python 3.12 (stdlib only, unittest)",
            build_cmd="python3 -m unittest discover -s tests -v",
            project_desc=(
                "Build a self-contained relational mini-SQL engine `minisql` in Python (stdlib only)."
            ),
            phase1=(
                "Implement: (1) SQL lexer & recursive-descent parser (`minisql/parser.py`, `minisql/ast_nodes.py`) "
                "supporting `CREATE TABLE`, `INSERT`, `SELECT` with `WHERE`, `AND/OR/NOT`, comparisons, "
                "`ORDER BY`, `LIMIT/OFFSET`, `INNER JOIN`, `LEFT JOIN`, `GROUP BY`, `HAVING`, and aggregates "
                "(`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`); (2) Catalog & columnar/row table storage "
                "(`minisql/storage.py`) with secondary B-tree/sorted indexes; (3) Volcano iterator "
                "execution engine (`minisql/executor.py`)."
            ),
            phase2=(
                "Add a cost-based logical/physical query optimizer (`minisql/optimizer.py`): "
                "predicate pushdown through joins, constant folding, projection pruning, index-scan "
                "selection, hash-join vs nested-loop selection, and `EXPLAIN` / `EXPLAIN ANALYZE` output "
                "showing operator row counts and timings. Add subqueries (`IN (SELECT ...)` and scalar "
                "subqueries) and `CREATE INDEX`."
            ),
            phase3=(
                "Add transactional savepoints (`BEGIN`, `SAVEPOINT`, `ROLLBACK TO`, `COMMIT`), "
                "NULL three-valued logic (`IS NULL`, `IS NOT NULL`, `COALESCE`, proper SQL NULL semantics "
                "in comparisons, `GROUP BY`, and `LEFT JOIN`), and a differential fuzz test suite "
                "comparing optimized vs unoptimized plans on 50 randomized multi-join queries."
            ),
            phase4=(
                "Implement a stack-based expression bytecode VM (`minisql/expr_vm.py`) that compiles "
                "`WHERE` / `SELECT` expressions into compact opcodes and executes them faster than tree "
                "walking, plus a JSON dump/load catalog persistence layer (`minisql/persist.py`) and "
                "CLI runner (`minisql/cli.py`). Verify all tests pass."
            ),
            phase5=(
                "Add window functions (`ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`, `RANK()`, "
                "running `SUM() OVER (...)`) and common table expressions (`WITH cte AS (...)`), "
                "re-checking all parser/optimizer/VM invariants from Phase 1."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via Alibaba (qwen/qwen3.8-max-0902) in a clean stdlib-only Python sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-32k-cal-03",
        length_class="32k",
        split="calibration",
        group="agentic:go-raft-scheduler-claude",
        source_kind="claude_cli",
        archetype="agentic_coding_cli_no_thinking",
        task_name="go_raft_replicated_scheduler_claude_cli",
        languages=("Go",),
        model="anthropic/claude-opus-5-5-cli",
        provider_only=(),
        expected_provider_names=("Claude Code CLI",),
        expect_reasoning=False,
        target_tokens=32768,
        min_tokens=27800,
        max_tokens=37800,
        phase_prompts=(
            "Build a Go module `raftsched` (Go stdlib only, `go mod init raftsched`) implementing a "
            "deterministic, in-memory-transport Raft consensus core (`raft/node.go`, `raft/log.go`, "
            "`raft/rpc.go`, `raft/transport.go`) with leader election, randomized election timeouts, "
            "log replication, commit index advancement, and conflict backtracking. Write comprehensive "
            "tests in `raft/raft_test.go` covering 3-node and 5-node elections, leader crash failover, "
            "network partitions, and log reconciliation, and verify `go test -race ./...` passes.",
            "Now extend `raftsched` with: (1) log snapshotting & `InstallSnapshot` RPC (`raft/snapshot.go`) "
            "so lagging followers catch up after log compaction; (2) a replicated state machine "
            "(`scheduler/sm.go`) implementing a leased distributed job queue (submit job, claim with lease "
            "TTL, renew lease, complete/fail with retry backoff, deterministic logical clock); "
            "(3) thorough unit & integration tests verifying linearizable job claims across leader failovers "
            "and snapshot restores. Run `go test -race -v ./...`.",
            "Now write a deterministic fault-injection simulator (`sim/chaos_test.go`) that runs a 5-node "
            "cluster under packet drops, reorderings, asymmetric partitions, node crashes/restarts with "
            "persisted Raft state (`raft/storage.go`), and concurrent client job submissions. Assert "
            "Election Safety, Log Matching, Leader Completeness, and State Machine Safety (no job "
            "completed twice) at every step. Fix any subtle races or bugs exposed by `-race`.",
            "Finally, refactor common RPC/codec helpers, add a cluster diagnostics inspector "
            "(`scheduler/inspect.go` producing per-node lag, term history, and lease summaries), add "
            "benchmarks (`raft/bench_test.go`), and run `go vet ./... && go test -race -count=1 ./...`.",
        ),
        license_notes="Synthetic agentic coding trace recorded from Claude Code CLI (claude-opus-5-5) in a clean stdlib-only Go sandbox; no raw thinking tokens.",
    ),
    CaseSpec(
        id="lc-32k-cal-04",
        length_class="32k",
        split="calibration",
        group="wiki:cluster-en-zh-fr-philosophy-history-ai",
        source_kind="wiki_qa",
        archetype="long_doc_multilingual_qa_reasoning",
        task_name="multilingual_wiki_synthesis_and_early_recall",
        languages=("en", "zh", "fr"),
        model="qwen/qwen3.8-max-0902",
        provider_only=("alibaba",),
        expected_provider_names=("Alibaba",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=32768,
        min_tokens=27800,
        max_tokens=37800,
        local_sources=(
            "processed/wiki/enwiki/phil/friedrich-nietzsche.md",
            "processed/wiki/enwiki/phil/gottfried-wilhelm-leibniz.md",
            "processed/wiki/zhwiki/worldhist/tang-taizong.md",
            "processed/wiki/frwiki/ai/intelligence-artificielle.md",
        ),
        extra={
            "doc_budget_tokens": 17500,
            "qa_turns": (
                "You have been given a four-document multilingual reference dossier (English, Chinese, and French). "
                "First, provide a structured cross-document index listing each document's title, language, "
                "approximate historical period or scope, and 5 specific factual anchors drawn specifically from "
                "the opening sections of Document 1 (Friedrich Nietzsche) and Document 2 (Gottfried Wilhelm Leibniz). "
                "Cite exact dates, places, and names from the text without inventing facts.",
                "Now perform a deep cross-linguistic comparison in English and Chinese: compare how Document 2 "
                "(Leibniz's formal logic / calculus / characteristica universalis) connects conceptually to "
                "Document 4 (French article on *Intelligence artificielle*, specifically its early symbolic AI "
                "and historical foundations), and contrast how institutional governance and patronage shaped "
                "intellectual life in Document 2 (17th-century Europe) versus Document 3 (唐太宗 / Tang Taizong's "
                "court and 贞观之治 in Chinese). Include exact terms from all three source texts.",
                "Verify five subtle cross-section details across the entire dossier:\n"
                "1. From early Document 1 (Nietzsche): what specific philology professorship did he obtain, at what age, and what health/military events interrupted his early career?\n"
                "2. From middle/late Document 1: how were his notebooks edited after his collapse, and what controversy surrounds them?\n"
                "3. From Document 2 (Leibniz): trace the specific dispute mentioned with Newton and the role of the Royal Society.\n"
                "4. From Document 3 (唐太宗, answer in Chinese): 列出文中提到的玄武门之变的关键人物、时间以及贞观时期的重要制度或大臣。\n"
                "5. From Document 4 (*Intelligence artificielle*, answer in French): citez les étapes historiques majeures (Conférence de Dartmouth, hivers de l'IA, apprentissage profond) mentionnées dans le texte.",
                "Finally, construct a unified chronological cross-reference table (12-16 dated events spanning "
                "all four documents from the 7th century to the 21st century), explicitly marking whether each "
                "fact appears in the early, middle, or late part of its source document, and resolve any "
                "terminology contrasts across English, Chinese, and French.",
            ),
        },
        license_notes="Wikipedia excerpts from training-data 2026-07-01 dumps (CC BY-SA 4.0) plus multi-turn synthesis QA generated via Alibaba (qwen/qwen3.8-max-0902).",
    ),
    CaseSpec(
        id="lc-32k-hold-01",
        length_class="32k",
        split="heldout",
        group="agentic:ts-reactive-spreadsheet-dag",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="typescript_reactive_spreadsheet_engine",
        languages=("TypeScript",),
        model="z-ai/glm-5.3-flash",
        provider_only=("z-ai",),
        expected_provider_names=("Z.AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=32768,
        min_tokens=27800,
        max_tokens=37800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="TypeScript executed directly via Node.js built-in type stripping (`node --test`) — zero npm packages",
            build_cmd="node --test tests/*.test.ts",
            project_desc=(
                "Build a self-contained reactive spreadsheet & incremental DAG evaluation engine "
                "`cellgrid` in TypeScript (ESM, standard Node 22/24 built-ins only, `node:test` and "
                "`node:assert/strict`, zero third-party packages)."
            ),
            phase1=(
                "Implement: (1) A1-notation cell address & range parser (`src/address.ts`) supporting "
                "absolute/relative refs (`$A$1`, `B2:D10`, multi-sheet `Sheet2!A1:B5`); (2) Formula "
                "tokenizer & Pratt/recursive-descent parser (`src/parser.ts`) supporting arithmetic, "
                "comparisons, string concat `&`, unary ops, nested function calls, and cell/range refs; "
                "(3) Dependency DAG & topological evaluator (`src/engine.ts`) with cycle detection "
                "(`#CYCLE!`), error propagation (`#DIV/0!`, `#REF!`, `#NAME!`, `#VALUE!`), and core "
                "functions (`SUM`, `AVERAGE`, `MIN`, `MAX`, `IF`, `COUNT`, `ROUND`, `ABS`, `CONCAT`)."
            ),
            phase2=(
                "Extend `cellgrid` with: (1) Incremental dirty-subgraph recomputation with memoized "
                "evaluation counters so unrelated subgraphs are never re-evaluated when a cell changes; "
                "(2) Structural mutations (`insertRow`, `deleteRow`, `insertCol`, `deleteCol`) that "
                "automatically rewrite relative references across all formulas and turn deleted targets "
                "into `#REF!`; (3) Lookup & conditional functions (`VLOOKUP`, `INDEX`, `MATCH`, `SUMIF`, "
                "`COUNTIF`, `IFERROR`, `LET` local bindings)."
            ),
            phase3=(
                "Add transactional undo/redo command history (`src/history.ts`), named ranges & "
                "cross-sheet references (`src/workbook.ts`), and a deterministic property test suite "
                "(`tests/incremental_equivalence.test.ts`) verifying that 500 random edits + row/col "
                "insertions/deletions + undo/redo operations produce cell values bit-identical to a "
                "full from-scratch recomputation."
            ),
            phase4=(
                "Add CSV/TSV and JSON workbook serializer/deserializer (`src/codec.ts`) with RFC-4180 "
                "quoting, a formula pretty-printer and AST dependency graph exporter (DOT + JSON), "
                "and an end-to-end financial model integration test (`tests/integration.test.ts`)."
            ),
            phase5=(
                "Add dynamic array spilling (e.g. `SEQUENCE`, `FILTER`, `SORT`, `UNIQUE` with `#SPILL!` "
                "collision detection) and volatile-function tracking, then re-verify every invariant "
                "across all earlier modules."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via Z.AI (z-ai/glm-5.3-flash) in a clean stdlib-only TypeScript/Node sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-32k-hold-02",
        length_class="32k",
        split="heldout",
        group="code:sgl-project/sgl-attn",
        source_kind="oss_repo_qa",
        archetype="multi_file_oss_repo_audit_reasoning",
        task_name="sgl_attn_cuda_cpp_architecture_audit",
        languages=("C++", "CUDA"),
        model="xiaomi/mimo-v2.6-flash",
        provider_only=("xiaomi",),
        expected_provider_names=("Xiaomi",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=32768,
        min_tokens=27800,
        max_tokens=37800,
        oss_repo_dir="data/github-oss-snapshots/raw/20260802-inference-deps/sgl-project/sgl-attn",
        oss_repo_slug="sgl-project/sgl-attn",
        oss_repo_commit="f89bc2306632d1ec5f97b014dded4254f5b4a907",
        oss_include_globs=(
            "csrc/flash_attn/flash_api.cpp",
            "csrc/flash_attn/src/flash.h",
            "csrc/flash_attn/src/block_info.h",
            "csrc/flash_attn/src/softmax.h",
            "csrc/flash_attn/src/mask.h",
            "csrc/flash_attn/src/static_switch.h",
        ),
        extra={
            "doc_budget_tokens": 17500,
            "qa_turns": (
                "You are given a multi-file excerpt from the C++/CUDA attention library `sgl-project/sgl-attn` "
                "(with exact file paths and line numbers). Trace the complete end-to-end parameter validation "
                "and dispatch path from `flash_api.cpp` into the kernel parameter struct (`Flash_fwd_params` "
                "in `flash.h`). Specifically list: (1) how paged KV-cache (`block_table`), sliding window "
                "(`window_size_left/right`), and softcap (`softcap`) are validated and normalized; "
                "(2) which tensor strides and alignments are asserted; and (3) exact line references in both "
                "the early header files and the later dispatch functions.",
                "Now analyze numerical precision and masking interactions across `softmax.h`, `mask.h`, "
                "and `block_info.h`: (1) how does online softmax track row max (`row_max`) and sum "
                "(`row_sum`) across tiles, and how does it handle completely masked rows (e.g., causal + "
                "sliding window or local window) without producing NaNs? (2) How do `BlockInfo` sequence "
                "length calculations differ between varlen (`cu_seqlens_q`/`cu_seqlens_k`), `seqused_k`, "
                "and fixed-length padded batches? Cite exact symbol names and file/line locations.",
                "Audit all edge cases around split-KV / num_splits selection and GQA/MQA head-group ratios "
                "in `flash_api.cpp` and `flash.h`: how are `h_h_k_ratio`, `num_splits`, `softmax_lseaccum`, "
                "and `oaccum` sized and dispatched, and what happens for zero-length sequences or non-multiple "
                "head dimensions? Cite exact line numbers.",
                "Finally, design a minimal, concrete code-review checklist and patch plan for adding a "
                "new per-head attention sink logit or sparse top-k block index table to this codebase: "
                "identify every struct field in `flash.h`, validation check in `flash_api.cpp`, template "
                "switch in `static_switch.h`, and inner-loop hook in `mask.h`/`softmax.h` that would need "
                "to change, highlighting potential ABI or numerical pitfalls.",
            ),
        },
        license_notes="Source excerpts from sgl-project/sgl-attn @ f89bc2306632d1ec5f97b014dded4254f5b4a907 (BSD-3-Clause / Apache-2.0 per upstream LICENSE); multi-turn audit QA generated via Xiaomi (xiaomi/mimo-v2.6-flash). Group matches v2 heldout-only repo (zero overlap with calibration).",
    ),
    # =========================================================================
    # 16K LENGTH CLASS (8 calibration ~128K + 2 heldout ~32K)
    # =========================================================================
    CaseSpec(
        id="lc-16k-cal-01",
        length_class="16k",
        split="calibration",
        group="agentic:cpp20-ringbuf-btree",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="cpp20_lockfree_ringbuf_and_btree",
        languages=("C++",),
        model="z-ai/glm-5.3",
        provider_only=("z-ai",),
        expected_provider_names=("Z.AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="C++20 (`g++ -std=c++20 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined -pthread`)",
            build_cmd="g++ -std=c++20 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined -pthread tests/test_main.cpp -o build/test_main && ./build/test_main",
            project_desc=(
                "Build a header-only C++20 systems library in `include/` with: "
                "(1) a cache-line-padded SPSC lock-free ring buffer (`include/spsc_ring.hpp`) using "
                "`std::atomic` acquire/release semantics; and (2) an in-memory B+ tree (`include/btree.hpp`) "
                "supporting insert, lookup, erase, and range scan."
            ),
            phase1=(
                "Implement `include/spsc_ring.hpp` (power-of-two capacity, `try_push`, `try_pop`, "
                "`try_push_bulk`, `try_pop_bulk`, `size_approx`, proper move-only type support) and "
                "`include/btree.hpp` (configurable branching order, leaf-linked range iteration, "
                "node split on insert, and node borrow/merge on erase). Write unit tests in "
                "`tests/test_main.cpp` and compile/run under ASan+UBSan."
            ),
            phase2=(
                "Add multi-threaded producer-consumer stress tests for `SpscRing` (100K items with "
                "checksums and move-only payloads verifying zero leaks/races) and a randomized "
                "differential test for `BTree` comparing against `std::map` across 5,000 inserts, "
                "updates, erases, and range queries, plus structural B+ tree invariant checks "
                "(height balance, sorted keys, sibling links, occupancy bounds)."
            ),
            phase3=(
                "Add an LRU page-buffer cache (`include/lru_cache.hpp`) with intrusive doubly-linked "
                "list + hash index and pin/unpin eviction protection, integrate it in a small benchmark "
                "and verification test, and confirm zero ASan/UBSan findings."
            ),
            phase4=(
                "Refactor shared diagnostics and add a tiny CLI benchmark driver validating latency "
                "histograms and throughput stats; ensure clean zero-warning compilation and passing tests."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via Z.AI (z-ai/glm-5.3) in a clean stdlib-only C++20 sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-16k-cal-02",
        length_class="16k",
        split="calibration",
        group="agentic:ts-jsonrpc-lsp-framing",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="typescript_jsonrpc_lsp_buffer",
        languages=("TypeScript",),
        model="z-ai/glm-5.3",
        provider_only=("z-ai",),
        expected_provider_names=("Z.AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="TypeScript via Node.js (`node --test`, zero external packages)",
            build_cmd="node --test tests/*.test.ts",
            project_desc=(
                "Build a self-contained JSON-RPC 2.0 / LSP framing & incremental text-document engine "
                "in TypeScript (Node built-ins only)."
            ),
            phase1=(
                "Implement: (1) a streaming `Content-Length` byte-accurate frame decoder/encoder "
                "(`src/framing.ts`) handling arbitrary chunk splits, multi-byte UTF-8 payloads, "
                "case-insensitive headers, and malformed header recovery; (2) a JSON-RPC 2.0 "
                "dispatcher (`src/rpc.ts`) supporting requests, notifications, cancellation, "
                "structured error codes, and out-of-order async replies; (3) a UTF-16 LSP position "
                "to UTF-8/line-index incremental document buffer (`src/document.ts`)."
            ),
            phase2=(
                "Extend `src/document.ts` with a line-offset index supporting incremental range edits "
                "(including emoji/surrogate pairs, CRLF vs LF line endings, multi-line insertions and "
                "deletions) and document version validation. Add a mock in-memory bidirectional LSP "
                "server/client loop (`src/session.ts`) supporting `didOpen`, `didChange`, `hover`, "
                "and `diagnostics` notifications."
            ),
            phase3=(
                "Write randomized fuzz tests (`tests/fuzz.test.ts`) slicing encoded JSON-RPC streams "
                "at every 1-byte and random-byte boundary, injecting malformed frames, and comparing "
                "incremental document edits against naive string reconstruction across Unicode/CRLF "
                "inputs."
            ),
            phase4=(
                "Add request timeout handling, backpressure watermarks, and protocol trace logging "
                "with full integration tests."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via Z.AI (z-ai/glm-5.3) in a clean stdlib-only TypeScript sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-16k-cal-03",
        length_class="16k",
        split="calibration",
        group="agentic:rust-thompson-regex-vm",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="rust_thompson_regex_vm",
        languages=("Rust",),
        model="z-ai/glm-5.3-flash",
        provider_only=("z-ai",),
        expected_provider_names=("Z.AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="Rust (Cargo, stdlib only, no external crates)",
            build_cmd="cargo test",
            project_desc=(
                "Build a linear-time regular expression compiler and PikeVM matcher crate `pikere` "
                "(edition 2021, zero external crates)."
            ),
            phase1=(
                "Implement: (1) regex parser (`src/parser.rs`) supporting literals, `.`, character "
                "classes `[a-z0-9_]`, negated classes `[^...]`, escapes (`\\d`, `\\w`, `\\s`), "
                "alternation `|`, grouping `(...)`, non-capturing `(?:...)`, quantifiers `*`, `+`, `?`, "
                "`{m,n}`, and anchors `^`, `$`; (2) bytecode compiler (`src/compiler.rs`) lowering AST "
                "to PikeVM instructions (`Char`, `Class`, `Any`, `Split`, `Jmp`, `Save`, `Assert`, `Match`); "
                "(3) linear-time PikeVM executor (`src/vm.rs`) returning leftmost-first capture spans."
            ),
            phase2=(
                "Add non-greedy quantifiers (`*?`, `+?`, `??`), word boundaries (`\\b`, `\\B`), "
                "case-insensitive flag `(?i)`, `find_iter`, `replace_all` with `$1`/`$0` templates, "
                "and a pathological backtracking test (`a?^30 a^30`) proving strictly linear execution time."
            ),
            phase3=(
                "Add a lazy DFA fast-path (`src/dfa.rs`) for unanchored/non-capture matching with "
                "state-cache eviction, plus differential tests verifying PikeVM and DFA agree on "
                "hundreds of patterns and UTF-8 test strings."
            ),
            phase4=(
                "Add bytecode disassembler, error span diagnostics for invalid regex syntax, and "
                "comprehensive integration tests."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via Z.AI (z-ai/glm-5.3-flash) in a clean stdlib-only Rust sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-16k-cal-04",
        length_class="16k",
        split="calibration",
        group="agentic:python-netsim-codex",
        source_kind="codex_cli",
        archetype="agentic_coding_cli_no_thinking",
        task_name="python_discrete_event_netsim_codex_cli",
        languages=("Python",),
        model="openai/gpt-6.2-sol-codex-cli",
        provider_only=(),
        expected_provider_names=("Codex CLI",),
        expect_reasoning=False,
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        phase_prompts=(
            "Build a deterministic discrete-event network simulator package `netsim` in Python 3 "
            "(stdlib only, `unittest`). Implement: (1) a heap-based event scheduler (`netsim/clock.py`) "
            "with tie-breaking sequence IDs; (2) links with bandwidth, propagation delay, tail-drop "
            "queues, and RED (Random Early Detection) active queue management (`netsim/link.py`); "
            "(3) a simplified AIMD/Reno congestion-controlled flow (`netsim/tcp.py`) with slow start, "
            "congestion avoidance, duplicate-ACK fast retransmit, and RTO timeouts. Write unit tests "
            "in `tests/test_netsim.py` and run `python3 -m unittest discover -s tests -v`.",
            "Follow TDD to extend `netsim`: FIRST add failing tests in `tests/test_shaper_routing.py` "
            "for (1) Token-bucket shapers & WFQ schedulers (`netsim/shaper.py`), (2) multi-hop Dijkstra "
            "routing with link failover (`netsim/topology.py`), and (3) Jain's fairness index & latency "
            "percentiles (`netsim/metrics.py`), and run `python3 -m unittest discover -s tests -v` to "
            "confirm the new tests fail. THEN implement those modules and re-run the tests until all pass.",
            "Add a seeded deterministic trace-replay and golden-invariant test suite (`tests/test_invariants.py`) "
            "verifying packet conservation (sent == delivered + dropped + in_flight), monotonic delivery "
            "timestamps, and reproducibility across identical seeds. Run all tests to verify 100% pass.",
        ),
        license_notes="Synthetic agentic coding trace recorded from Codex CLI (gpt-6.2-sol) in a clean stdlib-only Python sandbox; no raw thinking tokens.",
    ),
    CaseSpec(
        id="lc-16k-cal-05",
        length_class="16k",
        split="calibration",
        group="agentic:go-dag-workflow-engine",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="go_dag_workflow_orchestrator",
        languages=("Go",),
        model="stepfun/step-5-preview",
        provider_only=("stepfun",),
        expected_provider_names=("StepFun",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="Go (stdlib only, `go mod init dagflow`)",
            build_cmd="go test -race -v ./...",
            project_desc=(
                "Build a concurrent DAG workflow orchestrator `dagflow` in Go (stdlib only)."
            ),
            phase1=(
                "Write small, modular files one at a time: first `go.mod`, `dag/graph.go` (stubs) and "
                "`dag/graph_test.go`, and run `go test -race -v ./...` to observe initial red failures. "
                "Then implement: (1) DAG validation & Kahn topological sort (`dag/graph.go`) with detailed "
                "cycle path reporting; (2) bounded worker-pool executor (`dag/executor.go`) respecting "
                "task dependencies, per-task timeouts, exponential retry backoff with jitter, and "
                "`context.Context` cancellation; (3) typed artifact passing (`dag/artifacts.go`) and "
                "unit tests (`dag/executor_test.go`). Run `go test -race -v ./...` after each module."
            ),
            phase2=(
                "Add: (1) resource-pool semaphore constraints (e.g., max 2 GPU tasks concurrently while "
                "CPU tasks continue); (2) checkpointed execution state (`dag/checkpoint.go`) serialized "
                "to JSON so an interrupted DAG can resume skipping already-succeeded nodes; "
                "(3) conditional skip edges (`OnFailure`, `OnSuccess`, `Always`)."
            ),
            phase3=(
                "Write concurrency race and fault-injection tests (`dag/stress_test.go`): 50-node "
                "diamond/mesh DAGs with random transient failures, mid-flight cancellation, checkpoint "
                "resume, and strict verification that no node executes before all its dependencies "
                "complete."
            ),
            phase4=(
                "Add critical-path timing analysis, execution Gantt summary formatting, and end-to-end "
                "tests under `go vet ./... && go test -race ./...`."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via StepFun (stepfun/step-5-preview) in a clean stdlib-only Go sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-16k-cal-06",
        length_class="16k",
        split="calibration",
        group="wiki:en-audi-mississippi-river",
        source_kind="wiki_qa",
        archetype="long_doc_en_qa_reasoning",
        task_name="enwiki_long_document_early_recall",
        languages=("en",),
        model="moonshotai/kimi-k3.1-pro",
        provider_only=("moonshotai",),
        expected_provider_names=("Moonshot AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        local_sources=(
            "processed/wiki/enwiki/eng/audi.md",
            "processed/wiki/enwiki/earth/mississippi-river.md",
        ),
        extra={
            "doc_budget_tokens": 8500,
            "qa_turns": (
                "You have been given two long English reference articles (Document 1: Audi; Document 2: "
                "Mississippi River). Answer strictly from the provided text: "
                "(1) From the opening sections of Document 1, explain the exact etymology of the name "
                "'Audi', who founded the company, why he couldn't use his own surname, and which four "
                "companies merged into Auto Union in 1932. "
                "(2) From the opening sections of Document 2, state the river's source lake, length, "
                "drainage basin scope, and the major tributaries named early in the article.",
                "Now trace the middle sections of both documents: "
                "(1) In Document 1, summarize Audi's post-war revival under Volkswagen ownership and its "
                "major engineering milestones (front-wheel drive, Quattro all-wheel drive, aluminum space "
                "frame, TDI/motorsport), citing exact model names and years from the text. "
                "(2) In Document 2, explain the geographical division between the Upper and Lower "
                "Mississippi and the specific hydrological/geological features described in the middle sections.",
                "Finally, cross-check early vs late details across both articles: "
                "(1) Compare the early 20th-century industrial structure of Auto Union (Document 1 opening) "
                "with the modern global manufacturing/electrification details near the end of Document 1. "
                "(2) Summarize the major historical floods, levees/lock-and-dam engineering projects, and "
                "ecological or economic notes in the later sections of Document 2, linking them back to "
                "the river's basin characteristics introduced at the start.",
            ),
        },
        license_notes="Wikipedia excerpts from training-data 2026-07-01 enwiki dump (CC BY-SA 4.0) plus multi-turn QA generated via Moonshot AI (moonshotai/kimi-k3.1-pro).",
    ),
    CaseSpec(
        id="lc-16k-cal-07",
        length_class="16k",
        split="calibration",
        group="wiki:ja-de-tezuka-aristoteles",
        source_kind="wiki_qa",
        archetype="long_doc_multilingual_qa_reasoning",
        task_name="ja_de_wiki_cross_section_qa",
        languages=("ja", "de"),
        model="qwen/qwen3.8-max-0902",
        provider_only=("alibaba",),
        expected_provider_names=("Alibaba",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        local_sources=(
            "processed/wiki/jawiki/art/osamu-tezuka.md",
            "processed/wiki/dewiki/phil/aristoteles.md",
        ),
        extra={
            "doc_budget_tokens": 8500,
            "qa_turns": (
                "以下の2つの長文記事（日本語：手塚治虫、ドイツ語：Aristoteles）を参照して回答してください。\n"
                "1. 【日本語で回答】文書1（手塚治虫）の冒頭〜前半セクションに基づき、生没年・出身地・医師免許に関する経歴、および初期の代表作とマンガ表現への革新を具体的な作品名とともに整理してください。\n"
                "2. 【ドイツ語で回答】Fassen Sie anhand der ersten Abschnitte von Dokument 2 (Aristoteles) seine Herkunft (Stageira), seine Zeit an Platons Akademie, seine Tätigkeit als Lehrer Alexanders des Großen und die Gründung des Lykeions präzise zusammen.",
                "次に両文書の中盤セクションに注目して詳細を確認してください：\n"
                "1. 【日本語】手塚治虫のアニメーション制作（虫プロダクション等）と1960年代〜1970年代の作風の変化について本文中の具体例を挙げて説明してください。\n"
                "2. 【ドイツ語】Erläutern Sie die im mittleren Teil von Dokument 2 dargestellten Kerngebiete von Aristoteles' Philosophie (Logik/Organon, Metaphysik, Naturphilosophie) mit genauen Fachbegriffen aus dem Text.",
                "最後に両文書の冒頭と後半を横断して比較・要約してください：\n"
                "1. 【日本語】初期の経歴（医学・戦後漫画）が後年の生命観や社会的影響とどう結びついているか、本文の記述に基づいて整理してください。\n"
                "2. 【ドイツ語】Wie beschreibt der späte Abschnitt von Dokument 2 die Ethik, Politik und die historische Rezeption des Aristoteles im Vergleich zu seiner im ersten Abschnitt genannten Schulgründung?",
            ),
        },
        license_notes="Wikipedia excerpts from training-data 2026-07-01 jawiki & dewiki dumps (CC BY-SA 4.0) plus bilingual QA generated via Alibaba (qwen/qwen3.8-max-0902).",
    ),
    CaseSpec(
        id="lc-16k-cal-08",
        length_class="16k",
        split="calibration",
        group="code:huggingface/tokenizers",
        source_kind="oss_repo_qa",
        archetype="multi_file_oss_repo_audit_reasoning",
        task_name="hf_tokenizers_rust_bpe_audit",
        languages=("Rust",),
        model="openai/gpt-6-astra",
        provider_only=("openai",),
        expected_provider_names=("OpenAI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        oss_repo_dir="data/github-oss-snapshots/raw/20260802-inference-deps/huggingface/tokenizers",
        oss_repo_slug="huggingface/tokenizers",
        oss_repo_commit="7f1623b90b5adfb9bc327d4c3468d2f70bbce262",
        oss_include_globs=(
            "tokenizers/src/models/bpe/model.rs",
            "tokenizers/src/models/bpe/word.rs",
            "tokenizers/src/models/bpe/serialization.rs",
            "tokenizers/src/tokenizer/encoding.rs",
        ),
        extra={
            "doc_budget_tokens": 8500,
            "qa_turns": (
                "You are given multiple Rust source files from `huggingface/tokenizers` (BPE model, "
                "word merge mechanics, serialization, and `Encoding` representation). Trace how a "
                "pre-tokenized string flows through `BPE::tokenize` into `Word::merge_all` and finally "
                "into `Encoding`. Cite exact struct names, method names, and line numbers across the "
                "early and late files.",
                "Analyze edge cases and algorithmic details in `model.rs` and `word.rs`: "
                "(1) how does `BPE` handle unknown tokens (`unk_token`), `fuse_unk`, `byte_fallback`, "
                "and `dropout` (including cache bypass)? (2) How does `Word::merge_all` maintain linked "
                "symbol indices and a priority queue (`BinaryHeap`) without quadratic scans, and what "
                "happens when stale merge pairs are popped?",
                "Now cross-check `serialization.rs` and `encoding.rs` against `model.rs`: "
                "(1) Which BPE configuration fields are persisted during JSON serialization/deserialization "
                "and how are legacy/missing fields defaulted? (2) How does `Encoding::merge` combine "
                "token IDs, byte offsets, `word_ids`, and `sequence_ids`, and what concrete borrow-checker-safe "
                "optimization or test case would guard against offset corruption?",
            ),
        },
        license_notes="Source excerpts from huggingface/tokenizers @ 7f1623b90b5adfb9bc327d4c3468d2f70bbce262 (Apache-2.0); multi-turn audit QA generated via OpenAI (openai/gpt-6-astra). Group is v2 calibration-only (zero overlap with heldout).",
    ),
    CaseSpec(
        id="lc-16k-hold-01",
        length_class="16k",
        split="heldout",
        group="agentic:c11-http-chunked-arena",
        source_kind="openrouter_agentic",
        archetype="agentic_coding_reasoning",
        task_name="c11_http_chunked_parser_and_arena",
        languages=("C",),
        model="moonshotai/kimi-k3.1-pro",
        provider_only=("moonshotai",),
        expected_provider_names=("Moonshot AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        phase_prompts=_agentic_phases(
            lang="C11 (`gcc -std=c11 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined`)",
            build_cmd="gcc -std=c11 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined src/arena.c src/http_parser.c tests/test_http.c -o build/test_http && ./build/test_http",
            project_desc=(
                "Build a zero-dependency C11 library implementing: (1) a region/bump arena allocator "
                "with alignment, save/restore checkpoints, and overflow checks (`src/arena.h`, `src/arena.c`); "
                "and (2) a streaming incremental HTTP/1.1 request/response & chunked-transfer-encoding "
                "parser (`src/http_parser.h`, `src/http_parser.c`)."
            ),
            phase1=(
                "Implement the arena allocator (chained blocks, `arena_alloc_aligned`, `arena_strdup`, "
                "`arena_mark`/`arena_rewind`, `arena_destroy`) and the incremental HTTP/1.1 parser "
                "(state machine supporting request line, headers with case-insensitive lookup, "
                "`Content-Length`, `Transfer-Encoding: chunked` with hex chunk sizes, chunk extensions, "
                "and trailing headers). Reject smuggling ambiguities (both CL and TE, negative/overflow "
                "lengths, bare CR/LF in headers). Write thorough tests in `tests/test_http.c`."
            ),
            phase2=(
                "Add byte-by-byte streaming fuzz tests feeding valid and malicious HTTP messages in "
                "1-byte, 3-byte, and random chunk sizes, verifying identical parsed headers/bodies and "
                "proper rejection of smuggling/overflow/truncated inputs under ASan+UBSan."
            ),
            phase3=(
                "Add persistent keep-alive pipelined message parsing (parsing multiple back-to-back "
                "requests in one buffer, returning exact consumed byte offset and resetting parser state "
                "via arena checkpoint rewind), plus an HTTP response serializer and round-trip tests."
            ),
            phase4=(
                "Add query-string percent-decoder with strict UTF-8 / malformed-escape validation, "
                "arena utilization stats, and comprehensive integration tests passing cleanly with "
                "ASan+UBSan and `-Wall -Wextra -Werror`."
            ),
        ),
        license_notes="Synthetic agentic coding trace generated via Moonshot AI (moonshotai/kimi-k3.1-pro) in a clean stdlib-only C11 sandbox; CC0-1.0 / permissive calibration trace.",
    ),
    CaseSpec(
        id="lc-16k-hold-02",
        length_class="16k",
        split="heldout",
        group="wiki:zh-prc-wikipedia",
        source_kind="wiki_qa",
        archetype="long_doc_zh_qa_reasoning",
        task_name="zhwiki_long_document_early_recall",
        languages=("zh",),
        model="z-ai/glm-5.3",
        provider_only=("z-ai",),
        expected_provider_names=("Z.AI",),
        expect_reasoning=True,
        reasoning_effort="medium",
        target_tokens=16384,
        min_tokens=13900,
        max_tokens=18800,
        local_sources=(
            "processed/wiki/zhwiki/regionhist/people-s-republic-of-china.md",
            "processed/wiki/zhwiki/cs/wikipedia.md",
        ),
        extra={
            "doc_budget_tokens": 8500,
            "qa_turns": (
                "请严格依据上面提供的两篇中文维基百科长文（文档1：中华人民共和国；文档2：维基百科）回答以下问题，不要引入文档之外的臆测信息：\n"
                "1. 根据文档1开头与早期章节，列出中华人民共和国的成立时间、首都、面积与人口概况、以及文中提到的主要历史阶段划分。\n"
                "2. 根据文档2开头与早期章节，说明维基百科的创立时间、创始人、核心编辑原则（如中立观点等）以及早期技术平台演变。",
                "请继续分析两篇文档的中部章节：\n"
                "1. 归纳文档1中关于政治体制、行政区划与地理气候章节的关键制度与核心事实。\n"
                "2. 归纳文档2中关于多语言版本发展、编辑协作流程与版权协议（如GFDL向CC BY-SA迁移）的具体描述。",
                "最后请将两篇文档的开头内容与后部章节对照综合：\n"
                "1. 结合文档1中后部的经济、科技与文化/社会部分，说明其如何呼应开篇所述的发展阶段与国家概况。\n"
                "2. 结合文档2后部关于质量评估、可靠性争议与社区治理机制的描述，对照开篇确立的编辑原则分析其面临的挑战与应对措施。",
            ),
        },
        license_notes="Wikipedia excerpts from training-data 2026-07-01 zhwiki dump (CC BY-SA 4.0) plus multi-turn Chinese QA generated via Z.AI (z-ai/glm-5.3).",
    ),
)

CASES_BY_ID: dict[str, CaseSpec] = {c.id: c for c in CASES}
