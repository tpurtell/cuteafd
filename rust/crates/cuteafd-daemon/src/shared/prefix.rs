//! The prefix cache's serving knobs and host tier setup, shared by every generic family
//! (`cuteafd_engine::prefix` does the work; each family implements `PrefixFamily`).
use crate::families::deepseek_v41::v41_native_serve::prefix::CudaCopyEngine;
use anyhow::Result;
use cuteafd_engine::prefix::{FamilyLayout, PointPolicy};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};

#[path = "prefix/budget.rs"]
mod budget;
pub(crate) use budget::HostBudget;

/// The prefix cache's knobs.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct PrefixArgs {
    /// MiMo only: retain the DFlash context in positional prefix marks (experimental).
    #[arg(long, env = "CUTEAFD_MIMO_PREFIX_DRAFT", default_value_t = false)]
    pub mimo_prefix_draft: bool,
    #[arg(skip)]
    pub mimo_host_cap: bool,
    /// Retained snapshots per bank (prompts, completed turns); 0 turns the prefix cache off.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_ENTRIES", default_value_t = 20)]
    pub prefix_cache_entries: usize,
    /// Device memory for retained positional marks (SWA rows, MTP hidden rows, recurrent
    /// state), MiB; the arena holds two marks per entry pair while they fit, and never fewer
    /// than two per decoding sequence plus two.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_MARK_MIB", default_value_t = 2048)]
    pub prefix_cache_mark_mib: usize,
    /// Shortest prompt or turn worth a snapshot.
    #[arg(long, default_value_t = 64)]
    pub prefix_cache_min_tokens: usize,
    /// Pinned retained-prefix memory: auto sizes both banks within available host RAM,
    /// a byte count fixes the quota (e.g. 64GiB), and 0 disables the host tier.
    #[arg(long, env = "CUTEAFD_HOST_CACHE_BYTES", default_value = "0")]
    pub host_cache_bytes: HostBudget,
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

impl PrefixArgs {
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
    pub fn host_config(&self, layout: FamilyLayout, max_context: usize)
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
                let required = budget::retained_bytes(layout, self.prefix_cache_entries, max_context, chunk)?;
                let bytes = budget::automatic_bytes(required, chunk, memory, self.host_cache_headroom_bytes);
                let bytes = if self.mimo_host_cap { bytes.min(mimo_host_ceiling(memory.available, chunk)) } else { bytes };
                tracing::info!(required_bytes = required, resolved_bytes = bytes, available_bytes = memory.available,
                    total_bytes = memory.total, headroom_bytes = self.host_cache_headroom_bytes,
                    "automatic retained-prefix host budget (does not add active KV capacity)");
                bytes
            }
        };
        if bytes == 0 {
            return Ok(None);
        }
        let config = cuteafd_hostcache::config::Config {
            bytes,
            chunk_bytes: chunk.min(bytes),
            min_tokens: self.host_cache_min_tokens,
            max_tokens: u32::try_from(max_context)?,
            ..Default::default()
        };
        config.validate()?;
        Ok(Some(config))
    }

    /// Single-device copy engine for the resolved retained-prefix budget.
    pub fn host_tier<'a>(&self, library: &'a NativeLibrary, template: CuteafdDeviceBuffer,
        layout: FamilyLayout, max_context: usize)
        -> Result<Option<(cuteafd_hostcache::config::Config, CudaCopyEngine<'a>)>> {
        self.host_config(layout, max_context)?.map(|config|
            CudaCopyEngine::new(library, template).map(|engine| (config, engine))).transpose()
    }
}

fn mimo_host_ceiling(available: u64, chunk: u64) -> u64 {
    ((32u64 << 30).min(available / 5 * 2)) / chunk * chunk
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
    fn mimo_auto_ceiling_is_bounded_and_chunk_aligned() {
        let gib = 1u64 << 30;
        assert_eq!(mimo_host_ceiling(100 * gib, gib), 32 * gib);
        assert_eq!(mimo_host_ceiling(20 * gib, gib), 8 * gib);
        assert_eq!(mimo_host_ceiling(gib, gib), 0);
    }

    #[test]
    fn mimo_cap_preserves_explicit_disabled_and_fixed_budgets() {
        let layout = FamilyLayout { page_rows: 64, pages: 1024, page_bytes: 65536, mark_bytes: 4096,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT };
        let mut disabled = Cli::parse_from(["serve", "--host-cache-bytes", "0"]).prefix;
        disabled.mimo_host_cap = true;
        assert!(disabled.host_config(layout, 32768).unwrap().is_none());
        let mut fixed = Cli::parse_from(["serve", "--host-cache-bytes", "64GiB"]).prefix;
        fixed.mimo_host_cap = true;
        assert_eq!(fixed.host_config(layout, 32768).unwrap().unwrap().bytes, 64 << 30);
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
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT };
        assert!(cli.prefix.host_config(invalid, 0).unwrap().is_none());
        let cli = Cli::parse_from(["serve", "--host-cache-bytes", "64GiB"]);
        let layout = FamilyLayout { page_rows: 64, pages: 1024, page_bytes: 65536, mark_bytes: 4096,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT };
        let config = cli.prefix.host_config(layout, 32768).unwrap().unwrap();
        assert_eq!((config.bytes, config.max_tokens), (64 << 30, 32768));
    }
}
