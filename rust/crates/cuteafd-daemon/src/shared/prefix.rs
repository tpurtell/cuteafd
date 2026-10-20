//! The prefix cache's serving knobs and host tier setup, shared by every generic family
//! (`cuteafd_engine::prefix` does the work; each family implements `PrefixFamily`).
use anyhow::Result;
use cuteafd_engine::prefix::{FamilyLayout, PointPolicy};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};

#[path = "prefix/budget.rs"]
pub(crate) mod budget;
pub(crate) use budget::HostBudget;
pub(crate) mod cuda_copy;
pub(crate) use cuda_copy::CudaCopyEngine;

/// The prefix cache's knobs.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct PrefixArgs {
    /// Experimental MiMo DFlash context in positional prefix marks (default off; qualification: https://github.com/tpurtell/cuteafd/issues/41).
    #[arg(long, env = "CUTEAFD_MIMO_PREFIX_DRAFT", default_value_t = false)]
    pub mimo_prefix_draft: bool,
    #[arg(skip)]
    pub scoring_host_bytes: u64,
    /// Device snapshots per bank; the host tier is limited only by --host-cache-bytes.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_ENTRIES", default_value_t = 20)]
    pub prefix_cache_entries: usize,
    /// Device mark budget in MiB. With host retention, two staging marks are reserved;
    /// host-off retains the legacy entry/budget count with two per decoding sequence plus two.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_MARK_MIB", default_value_t = 2048)]
    pub prefix_cache_mark_mib: usize,
    /// Shortest prompt or turn worth a snapshot.
    #[arg(long, default_value_t = 64)]
    pub prefix_cache_min_tokens: usize,
    /// Pinned retained-prefix memory: auto sizes full-context KV for concurrency + 4,
    /// capped at min(32GiB, 25% of host RAM);
    /// a byte count fixes the quota (e.g. 64GiB), and 0 disables the host tier.
    #[arg(long, env = "CUTEAFD_HOST_CACHE_BYTES", default_value = "0")]
    pub host_cache_bytes: HostBudget,
    /// KV tokens to keep free after host pressure eviction (0 disables batching).
    #[arg(long, default_value_t = cuteafd_core::prefix::DEFAULT_HOST_CACHE_EVICT_TOKENS)]
    pub host_cache_evict_tokens: u64,
    /// RAM kept outside an automatic pinned prefix pool, at least 10% of total RAM.
    #[arg(long, default_value = "8GiB", value_parser = parse_bytes)]
    pub host_cache_headroom_bytes: u64,
    /// Shortest snapshot the host tier keeps.
    #[arg(long, default_value_t = 512)]
    pub host_cache_min_tokens: u32,
    /// Intermediate snapshot points (off by default; agentic sessions hit prompt-end and
    /// turn-end snapshots): one every N prefilled tokens at a chunk end (0 = none, e.g. 8192).
    #[arg(long, env = "CUTEAFD_PREFIX_POINT_GAP", default_value_t = 0)]
    pub prefix_point_gap: usize,
    /// Intermediate snapshot points at the last N message boundaries of the rendered prompt
    /// (before the generation prompt, before the last message, ...; 0 = none, e.g. 2).
    #[arg(long, env = "CUTEAFD_PREFIX_POINT_BOUNDARIES", default_value_t = 0)]
    pub prefix_point_boundaries: usize,
    /// Most intermediate points per prompt (the deepest are kept).
    #[arg(long, env = "CUTEAFD_PREFIX_POINTS_PER_REQUEST", default_value_t = 4)]
    pub prefix_points_per_request: usize,
    /// Partial reuse: MiMo replays the SWA window before the aligned common prefix
    /// (approximate); GLM 5.3 resumes at the last common page (exact: its pages are its whole
    /// state). Off: exact snapshot frontiers only. GLM 5.3 Flash has none (recurrent state).
    #[arg(long, env = "CUTEAFD_PREFIX_PARTIAL", value_enum, default_value_t = Toggle::Off)]
    pub prefix_partial: Toggle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Toggle {
    On,
    Off,
}

/// The device mark arena's knobs for `lanes` decoding sequences ([`PrefixArgs::mark_rule`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkRule {
    pub lanes: usize,
    pub entries: usize,
    /// `--prefix-cache-mark-mib`, in bytes.
    pub budget_bytes: u64,
    pub host: bool,
}

impl MarkRule {
    /// Slots of the arena for marks of `mark_bytes` (every rank's part): what the runtime
    /// allocates (`MarkArena::slots_for`) and what its planner reserves
    /// ([`cuteafd_core::prefix::mark_slots_with_host`]). The budget holds whole marks, so the count
    /// depends on the exact mark bytes: size both sides from the same mark.
    pub fn slots(self, mark_bytes: usize) -> usize {
        cuteafd_core::prefix::mark_slots_with_host(self.lanes as u64, self.entries as u64, mark_bytes as u64,
            self.budget_bytes, self.host) as usize
    }
}

impl PrefixArgs {
    pub fn with_scoring(&self, rows: usize, lane_rows: usize, vocab: usize) -> Result<Self> {
        let mut args = self.clone();
        args.scoring_host_bytes = cuteafd_core::memory_layout::host_logits_bytes(rows as u64, lane_rows as u64, vocab as u64)
            .ok_or_else(|| anyhow::anyhow!("host logits reservation overflow"))?;
        Ok(args)
    }

    /// The device mark arena's knobs for `lanes` decoding sequences.
    pub fn mark_rule(&self, lanes: usize) -> MarkRule {
        MarkRule { lanes, entries: self.prefix_cache_entries,
            budget_bytes: (self.prefix_cache_mark_mib as u64).saturating_mul(1 << 20), host: self.host_cache_bytes.enabled() }
    }

    pub fn points(&self) -> PointPolicy {
        PointPolicy { gap: self.prefix_point_gap, boundaries: self.prefix_point_boundaries,
            per_request: self.prefix_points_per_request }
    }

    /// Carve embedding-cache bytes in addition to the usual host-prefix headroom.
    pub fn with_media_headroom(&self, requested: Option<u64>) -> Result<(Self, usize)> {
        self.media_headroom(requested, budget::host_memory()?)
    }

    fn media_headroom(&self, requested: Option<u64>, memory: budget::HostMemory) -> Result<(Self, usize)> {
        let bytes = requested.unwrap_or((8u64 << 30).min(memory.total / 20));
        let base = self.host_cache_headroom_bytes.max(memory.total / 10);
        let headroom = base.checked_add(bytes).ok_or_else(|| anyhow::anyhow!("media host headroom overflow"))?;
        let fixed = match self.host_cache_bytes { HostBudget::Bytes(bytes) => bytes, HostBudget::Auto => 0 };
        anyhow::ensure!(memory.available >= headroom.checked_add(fixed).ok_or_else(|| anyhow::anyhow!("media and prefix host quota overflow"))?,
            "host RAM cannot admit embedding cache {bytes} bytes plus prefix quota {fixed} and headroom {base}");
        let mut prefix = self.clone();
        prefix.host_cache_headroom_bytes = headroom;
        Ok((prefix, usize::try_from(bytes)?))
    }

    /// Resolve the host quota once, before pinned allocation. Call only when
    /// this family's copy engine can restore every rank's snapshot exactly.
    /// Host prefix bytes are not active device KV capacity.
    pub fn host_config(&self, layout: FamilyLayout, max_context: usize, concurrency: usize)
        -> Result<Option<cuteafd_hostcache::config::Config>> {
        if self.prefix_cache_entries == 0 || !self.host_cache_bytes.enabled() {
            return Ok(None);
        }
        let chunk = (256u64 << 20).max(layout.page_bytes as u64)
            .max(layout.mark_bytes as u64).max(layout.draft_bytes as u64);
        let bytes = match self.host_cache_bytes {
            HostBudget::Bytes(bytes) => bytes,
            HostBudget::Auto => {
                let memory = budget::host_memory()?;
                let required = budget::retained_bytes(layout, concurrency, max_context, chunk)?;
                let bytes = budget::resolve_bytes(layout, None, concurrency, max_context, chunk, memory)?;
                tracing::info!(required_bytes = required, resolved_bytes = bytes, available_bytes = memory.available,
                    total_bytes = memory.total, headroom_bytes = self.host_cache_headroom_bytes,
                    "automatic retained-prefix host budget (does not add active KV capacity)");
                bytes
            }
        };
        anyhow::ensure!(bytes > 0,
            "host prefix quota resolves to zero; provide a positive quota or disable it explicitly for device retention");
        budget::admit(bytes, self.scoring_host_bytes, self.host_cache_headroom_bytes)?;
        let config = cuteafd_hostcache::config::Config {
            bytes,
            eviction_headroom: budget::eviction_headroom(layout, self.host_cache_evict_tokens)?,
            chunk_bytes: chunk.min(bytes),
            min_tokens: self.host_cache_min_tokens,
            max_tokens: u32::try_from(max_context)?,
            ..Default::default()
        };
        config.validate()?;
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: layout.page_rows as u64,
            page_bytes: layout.page_bytes as u64, mark_bytes: layout.mark_bytes as u64,
            draft_bytes: layout.draft_bytes as u64 };
        let conversations_max_context = geometry.conversations(bytes, max_context as u64, config.chunk_bytes);
        let conversations_50k = geometry.conversations(bytes, 50_000, config.chunk_bytes);
        tracing::info!(mark_bytes = layout.mark_bytes, host_bytes = bytes, conversations_max_context,
            conversations_50k, "retained-prefix conversation capacity (two marks per conversation)");
        if conversations_50k < concurrency as u64 + 4 {
            tracing::warn!(conversations_50k, desired = concurrency + 4,
                "host prefix quota holds fewer 50K conversations than concurrency plus four");
        }
        Ok(Some(config))
    }

    /// Single-device copy engine for the resolved retained-prefix budget.
    pub fn host_tier<'a>(&self, library: &'a NativeLibrary, template: CuteafdDeviceBuffer,
        layout: FamilyLayout, max_context: usize, concurrency: usize)
        -> Result<Option<(cuteafd_hostcache::config::Config, CudaCopyEngine<'a>)>> {
        self.host_config(layout, max_context, concurrency)?.map(|config|
            CudaCopyEngine::new(library, template).map(|engine| (config, engine))).transpose()
    }
}

/// Ids of the `markers` (message-start tokens such as `<|user|>`) that `tokenizer.json`'s added
/// tokens define, for intermediate snapshot points at message boundaries.
pub(crate) fn marker_ids(snapshot: &std::path::Path, markers: &[&str]) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(snapshot.join("tokenizer.json"))?;
    let tokenizer: serde_json::Value = serde_json::from_str(&text)?;
    let added = tokenizer["added_tokens"].as_array().map_or(&[][..], Vec::as_slice);
    Ok(markers.iter().filter_map(|marker| added.iter().find(|t| t["content"] == *marker)
        .and_then(|t| t["id"].as_u64()).map(|id| id as u32)).collect())
}

/// Positions of any of `markers` in `tokens` (message boundaries, in order).
pub(crate) fn boundaries(tokens: &[u32], markers: &[u32]) -> Vec<usize> {
    tokens.iter().enumerate().filter(|(_, t)| markers.contains(t)).map(|(i, _)| i).collect()
}

/// `123`, `512MiB`, `64GiB`, `1.5GB`.
pub(crate) fn parse_bytes(text: &str) -> std::result::Result<u64, String> {
    let text = text.trim();
    let split = text.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let scale: f64 = match unit {
        "" | "B" => 1.0,
        "KB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown byte unit {other:?}")),
    };
    let value: f64 = number.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if !value.is_finite() || value < 0.0 || value * scale >= u64::MAX as f64 {
        return Err(format!("{text:?} is not a finite nonnegative byte count that fits u64"));
    }
    Ok((value * scale) as u64)
}

/// A byte range inside `buffer` (bounds-checked; pointer arithmetic only).
pub(crate) fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    anyhow::ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "view {offset}+{bytes} past a {}-byte buffer", buffer.bytes);
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        prefix: PrefixArgs,
    }

    #[test]
    fn runtime_mark_rule_matches_shared_planner_and_arena_with_or_without_host() {
        for host in ["0", "auto", "32GiB"] {
            for entries in [0, 1, 20] {
                let cli = Cli::parse_from(["serve", "--host-cache-bytes", host,
                    "--prefix-cache-entries", &entries.to_string()]);
                for lanes in [1, 16, 64] {
                    // Paged MLA (no positional mark), dense/recurrent state, and SWA marks.
                    for bytes in [0, 37_500_000, 147_619_840, 256 << 20] {
                        let rule = cli.prefix.mark_rule(lanes);
                        let planned = cuteafd_core::prefix::mark_slots_with_host(lanes as u64, entries,
                            bytes as u64, rule.budget_bytes, host != "0") as usize;
                        let arena = cuteafd_engine::prefix::MarkArena::slots_for(lanes, entries as usize, bytes,
                            rule.budget_bytes as usize, host != "0");
                        assert_eq!((rule.slots(bytes), arena), (planned, planned));
                    }
                }
            }
        }
    }

    #[test]
    fn mimo_prefix_marks_help_links_experimental_qualification() {
        use clap::CommandFactory;
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("Experimental MiMo DFlash"), "{help}");
        assert!(help.contains("https://github.com/tpurtell/cuteafd/issues/41"), "{help}");
        assert!(!Cli::parse_from(["serve"]).prefix.mimo_prefix_draft);
    }

    #[test]
    fn media_quota_is_carved_before_auto_prefix_sizing() {
        let prefix = Cli::parse_from(["serve", "--host-cache-bytes", "auto"]).prefix;
        let memory = budget::HostMemory { total: 100 << 30, available: 80 << 30 };
        let (carved, bytes) = prefix.media_headroom(None, memory).unwrap();
        assert_eq!(bytes, 5usize << 30);
        assert_eq!(carved.host_cache_headroom_bytes, 15 << 30);
        assert_eq!(prefix.host_cache_headroom_bytes, 8 << 30);
        assert_eq!(carved.host_cache_bytes, HostBudget::Auto);
        let fixed = Cli::parse_from(["serve", "--host-cache-bytes", "64GiB"]).prefix;
        assert!(fixed.media_headroom(Some(8 << 30), memory).is_err());
        let (carved, _) = prefix.media_headroom(Some(8 << 30), memory).unwrap();
        assert_eq!(carved.host_cache_headroom_bytes, 18 << 30);
        assert!(prefix.media_headroom(Some(u64::MAX), memory).is_err());
    }

    #[test]
    fn prefix_knobs_default_on_with_the_host_tier_off() {
        let cli = Cli::parse_from(["serve"]);
        assert_eq!((cli.prefix.prefix_cache_entries, cli.prefix.host_cache_bytes), (20, HostBudget::Bytes(0)));
        assert_eq!(cli.prefix.host_cache_evict_tokens, 16 * 1024);
        assert_eq!(Cli::parse_from(["serve", "--host-cache-evict-tokens", "0"]).prefix.host_cache_evict_tokens, 0);
        assert_eq!(cli.prefix.prefix_partial, Toggle::Off);
        assert_eq!(cli.prefix.points(), PointPolicy { gap: 0, boundaries: 0, per_request: 4 });
        let cli = Cli::parse_from(["serve", "--prefix-partial", "on", "--prefix-point-gap", "8192",
            "--prefix-point-boundaries", "2"]);
        assert_eq!(cli.prefix.points(), PointPolicy { gap: 8192, boundaries: 2, per_request: 4 });
        assert_eq!(cli.prefix.prefix_partial, Toggle::On);
        let cli = Cli::parse_from(["serve", "--prefix-cache-entries", "0", "--host-cache-bytes", "64GiB"]);
        assert_eq!((cli.prefix.prefix_cache_entries, cli.prefix.host_cache_bytes), (0, HostBudget::Bytes(64 << 30)));
        assert_eq!(parse_bytes("512MiB"), Ok(512 << 20));
        assert_eq!(parse_bytes("1.5GB"), Ok(1_500_000_000));
        assert_eq!(parse_bytes("123"), Ok(123));
        assert!(parse_bytes("12 parsecs").is_err() && parse_bytes("-1").is_err());
        assert!(parse_bytes("NaN").is_err() && parse_bytes("inf").is_err());
    }

    #[test]
    fn auto_parses_and_disabled_retention_does_not_resolve_or_allocate() {
        let cli = Cli::try_parse_from(["serve", "--host-cache-bytes", "auto", "--prefix-cache-entries", "0"])
            .unwrap();
        assert_eq!(cli.prefix.host_cache_bytes, HostBudget::Auto);
        let invalid = FamilyLayout { page_rows: 0, pages: 0, page_bytes: 0, mark_bytes: 0,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT, mark_store: Default::default(), page_owners: Default::default() };
        assert!(cli.prefix.host_config(invalid, 0, 1).unwrap().is_none());
        let cli = Cli::parse_from(["serve", "--host-cache-bytes", "1MiB", "--host-cache-headroom-bytes", "0"]);
        let layout = FamilyLayout { page_rows: 64, pages: 1024, page_bytes: 65536, mark_bytes: 4096,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT, mark_store: Default::default(), page_owners: Default::default() };
        let config = cli.prefix.host_config(layout, 32768, 1).unwrap().unwrap();
        assert_eq!((config.bytes, config.max_tokens), (1 << 20, 32768));
    }
}
