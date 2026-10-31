//! Allocation-owner geometry for the default native V4.1 executor.
//! Projection scratch is read from the same AOT manifest as the linked native
//! library. Lanes own separate arenas; immutable weights and cache banks do not.
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category, ItemLifetime};
use serde_json::Value;
use crate::placement::{Demand, ProgramSet};
use super::resident::Fp8Head;

const BLAS: u64 = 4 << 20;
const VOCAB: u64 = 129280;

/// Sealed WIP artifacts sit beside the library; installed images keep their
/// manifests in share/, including when the library is a host-ISA variant.
pub fn native_manifest(library: &std::path::Path, name: &str) -> Result<Value> {
    let directory = library.parent().context("V4.1 native library has no parent")?;
    let candidates = [Some(directory.join(name)),
        directory.parent().map(|root| root.join("share").join(name)),
        directory.parent().and_then(std::path::Path::parent)
            .map(|root| root.join("share").join(name))];
    let path = candidates.into_iter().flatten().find(|path| path.is_file())
        .with_context(|| format!("missing V4.1 native manifest {name} beside the library or in its image share directory"))?;
    serde_json::from_slice(&std::fs::read(&path)
        .with_context(|| format!("reading V4.1 native manifest {}", path.display()))?)
        .with_context(|| format!("parsing V4.1 native manifest {}", path.display()))
}

const COMPRESSOR_WORKSPACE: u64 = 4 << 20;

/// Rows of a decode lane's private owners: the compiled cover of the widest
/// decode/verify step (`V41_VERIFY_ROWS`).
pub const DECODE_LANE_ROWS: u64 = 80;
const _: () = assert!(cuteafd_core::V41_VERIFY_ROWS as u64 <= DECODE_LANE_ROWS);

/// The attention GPU of `layer` on two RTX (the encoder/decoder map: layers
/// 0..20 on GPU0, 20..40 on GPU1).
fn v41_attention_gpu(layer: usize) -> usize { usize::from(layer >= 20) }

/// Full-head sparse attention storage (64 heads) for units of `unit` rows.
fn sparse64(unit: u64) -> u64 {
    let batch = unit.min(cuteafd_core::V41_VERIFY_ROWS as u64);
    2 * unit * 64 * 1024 + batch * 10 * 64 * 514 * 4 / 64 * 64 + batch * 120 + unit * 8 + unit * 80
}

/// Index selection's unit scratch (`IndexSelectionWave::sizes`).
fn index_selection(unit: u64) -> Result<u64> {
    const WIDTH: u64 = 16384;
    const BLOCKS: u64 = WIDTH / 8;
    let topk = |candidates: u64, k: u64| candidates.div_ceil(2 * k) * k * 8 * 2;
    Ok(unit * (48 + 8 + 8 + WIDTH * 8 + WIDTH * 4 + BLOCKS * 4 + BLOCKS * 8 + 4096
        + topk(WIDTH, 512) + 2048 + 16384 + topk(BLOCKS, 2048) + 8192))
}

/// Single-RTX cache execution retains a per-layer descriptor/destination table.
/// Placed two-RTX producers use their own publication storage instead.
pub fn window_batch_bytes(rows: u64) -> Result<u64> {
    ensure!((1..=4096).contains(&rows), "invalid V4.1 window batch rows");
    Ok(40 * (64 + rows * 8 + 16 * 16))
}

/// Coordinator-owned Spark wave planes. Pinned intake belongs to its own domain.
pub fn transport_bytes(rows: u64, ranks: usize) -> Result<u64> {
    ensure!((1..=4096).contains(&rows) && (1..=cuteafd_core::MAX_SPARK_WORLD).contains(&ranks),
        "invalid V4.1 transport geometry");
    Ok(rows * (ranks as u64 * 5120 * 2 + 2 * 5120 * 2))
}

/// Every Spark transport (PLAN "Prefill contract"): the two decode lanes'
/// links at the decode width ([`transport_bytes`] of `decode_rows`), each of
/// the `slots` prefill slots' reduce output at the unit `rows` (slot 0's FFN
/// owner also holds the link), and the one prefill link's intake: every
/// rank's partial-sum plane over its `slots` return slots.
pub fn transports_bytes(decode_rows: u64, rows: u64, ranks: usize, slots: usize) -> Result<u64> {
    ensure!((1..=4).contains(&slots), "invalid V4.1 prefill slots");
    let reserve = slots as u64 * rows;
    let lanes = 2 * transport_bytes(decode_rows, ranks)?;
    let outputs = slots as u64 * rows * 5120 * 2;
    Ok(lanes + outputs + reserve * ranks as u64 * 5120 * 2)
}

/// Full vocabulary packing peak above the pre-weight sample: the resident
/// weights (but the head being packed) and the packing staging. Every lane
/// and slot owner, transport, draft, vision and mark allocates after it.
pub fn vocabulary_load_phase(weights: &[Demand], target: &[Demand], packing: u64) -> Result<Demand> {
    ensure!(weights.iter().chain(target).all(|d| d.gpu == 0),
        "vocabulary startup phase requires the single-RTX owners");
    let bytes = weights.iter().filter(|d| !matches!(d.group.as_str(), "lm_head" | "norm")).try_fold(packing, |sum, d|
        sum.checked_add(d.bytes).context("V4.1 vocabulary phase overflow"))?;
    Ok(Demand::new(0, Category::Staging, "vocabulary packing", bytes, Basis::Formula))
}

/// Serving lanes allocated after the dual-RTX draft loader drains. Setup is
/// admission slack, not an allocation live at ready; transports are resident.
pub fn deferred_owners(ranks: usize, transport: u64) -> Result<Vec<Demand>> {
    ensure!(matches!(ranks, 1 | 2), "invalid V4.1 coordinator ranks");
    let mut demands = Vec::new();
    if transport > 0 {
        demands.push(Demand::new((ranks - 1) as u8, Category::Transport,
            "target/prefill transports", transport, Basis::Formula));
    }
    if ranks == 2 {
        for gpu in 0..2 {
            demands.push(Demand::new(gpu, Category::Reserved, "expert setup headroom",
                64 << 20, Basis::Formula).with_lifetime(ItemLifetime::PostReadyReserve));
        }
    }
    Ok(demands)
}

/// Shared pre-weight admission inventory. Only `before_draft` coexists with the
/// transient draft load: one RTX has transports but no snapshots/vision yet;
/// two RTX have snapshots/vision but no transports or setup allocation yet.
pub struct TargetOwners {
    pub fixed: Vec<Demand>,
    before_draft: Vec<Demand>,
    ranks: usize,
}

impl TargetOwners {
    pub fn new(weights: Vec<Demand>, target: Vec<Demand>, ranks: usize,
        transport: u64, snapshot_slots: u64) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2), "invalid V4.1 coordinator ranks");
        let mut fixed = weights;
        fixed.extend(target);
        ensure!(fixed.iter().all(|d| usize::from(d.gpu) < ranks
            && d.lifetime == ItemLifetime::ReadyResident), "invalid V4.1 target owners");
        let mut before_draft = fixed.clone();
        let deferred = deferred_owners(ranks, transport)?;
        let snapshots = super::resident::prefix_arena_bytes(snapshot_slots, false, ranks)
            .into_iter().enumerate().map(|(gpu, bytes)| Demand::new(gpu as u8,
                Category::Prefix, "target snapshot arenas", bytes, Basis::Formula))
            .collect::<Vec<_>>();
        if ranks == 1 { before_draft.extend(deferred.iter().cloned()); }
        else { before_draft.extend(snapshots.iter().cloned()); }
        fixed.extend(deferred);
        fixed.extend(snapshots);
        Ok(Self { fixed, before_draft, ranks })
    }

    pub fn add_vision(&mut self, demand: Demand) -> Result<()> {
        ensure!(usize::from(demand.gpu) < self.ranks
            && demand.lifetime == ItemLifetime::ReadyResident, "invalid V4.1 vision owner");
        if self.ranks == 2 { self.before_draft.push(demand.clone()); }
        self.fixed.push(demand);
        Ok(())
    }

    pub fn draft_load_phases(&self, draft: &super::DraftInventory) -> Result<Vec<Demand>> {
        let mut phases = draft.load_phase(self.ranks)?;
        for phase in &mut phases {
            phase.bytes = self.before_draft.iter().filter(|d| d.gpu == phase.gpu)
                .try_fold(phase.bytes, |sum, d| sum.checked_add(d.bytes)
                    .context("V4.1 draft loading coexistence overflow"))?;
        }
        Ok(phases)
    }
}

pub struct NativePrograms<'a> {
    manifest: &'a Value,
    sms: u32,
}

impl<'a> NativePrograms<'a> {
    pub fn new(manifest: &'a Value, sms: u32) -> Result<Self> {
        ensure!(sms > 0, "V4.1 native plans require physical SM count");
        Ok(Self { manifest, sms })
    }

    pub fn scratch(&self, rows: u64, k: u64, n: u64) -> Result<u64> {
        ProgramSet::v41_fp8(self.manifest, &[(k, n)], rows, self.sms)?.scratch
            .into_iter().try_fold(0u64, |sum, (_, bytes)| sum.checked_add(bytes)
                .context("V4.1 projection arena overflow"))
    }

    fn shared_ffn(&self, rows: u64) -> Result<u64> {
        Ok(self.scratch(rows, 5120, 2304)? + self.scratch(rows, 2304, 5120)?
            + rows * 2304 * 6 + 4)
    }

    fn projection(&self, rows: u64, k: u64, n: u64) -> Result<u64> {
        Ok(self.scratch(rows, k, n)? + rows * (k + n) * 2 + 4)
    }

    fn main_context(&self, rows: u64) -> Result<u64> {
        Ok(self.projection(rows, 15360, 5120)? + 1152
            + rows * (10240 + 256 + 8 + 3 * 1024)
            + 3 * (self.projection(rows, 5120, 512)? - rows * 10240))
    }

    /// The vocabulary GEMVs' split-K over 128-wide K blocks (gemv_tiles.cuh).
    fn vocabulary_splits(&self, rows: u64, n: u64) -> Result<u64> {
        ensure!((1..=4096).contains(&rows) && matches!(n, 64640 | VOCAB),
            "invalid V4.1 vocabulary shape");
        let tiles = n / 16;
        let blocks = 5120 / 128;
        let wanted = (u64::from(self.sms) * 16).div_ceil(tiles).clamp(1, blocks);
        Ok(blocks.div_ceil(blocks.div_ceil(wanted)))
    }

    /// `cuteafd_fp8_w8a16_workspace`: the row-64 base plus runtime-SM split-K.
    pub fn vocabulary_scratch(&self, rows: u64, n: u64) -> Result<u64> {
        let splits = self.vocabulary_splits(rows, n)?;
        let k = 5120u64;
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        Ok(align(2 * 64 * 4) + align(k.div_ceil(256) * 64 * 4) + align(k * 64 * 2)
            + if splits > 1 { align(splits * rows.min(64) * n * 4) } else { 0 })
    }

    /// `cuteafd_packed_head_workspace`: the row-64 B fragments plus runtime-SM split-K.
    pub fn packed_vocabulary_scratch(&self, rows: u64, n: u64) -> Result<u64> {
        let splits = self.vocabulary_splits(rows, n)?;
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        Ok(align(5120 * 64 * 2) + if splits > 1 { align(splits * rows.min(64) * n * 4) } else { 0 })
    }

    /// Scratch of one projection through `mode`'s resident copy of the vocabulary.
    pub fn head_scratch(&self, mode: Fp8Head, rows: u64, n: u64) -> Result<u64> {
        match mode {
            Fp8Head::Packed => self.packed_vocabulary_scratch(rows, n),
            _ => self.vocabulary_scratch(rows, n),
        }
    }

    /// The prefill slots' one storage set per GPU (`row_window` contract,
    /// #744 WP5/WP6): row planes at the `reserve` rows, scratch at the `unit`
    /// cap, every component as the runtime's `storage_bytes` (asserted equal
    /// by the daemon's plan == runtime test), plus each of the `slots` slot
    /// passes' private device owners (its verify-width target head and window
    /// batch table). Excludes local expert / TP2 FFN arenas.
    pub fn prefill_slots(&self, reserve: u64, unit: u64, ranks: usize, slots: usize, head_mode: Fp8Head,
        window_batch: bool) -> Result<Vec<Demand>> {
        ensure!((1..=4096).contains(&unit) && unit <= reserve && reserve <= 8256 && matches!(ranks, 1 | 2)
            && (1..=4).contains(&slots) && (!window_batch || ranks == 1), "invalid V4.1 prefill slot geometry");
        let (r, u) = (reserve, unit);
        let mut out = Vec::new();
        for rank in 0..ranks {
            let add = |out: &mut Vec<Demand>, group: &str, bytes| out.push(Demand::new(
                rank as u8, Category::Workspace, format!("prefill slots {group}"), bytes, Basis::Formula));
            let placed = ranks == 2;
            // LaneStorage (WP7): HC boundaries, query, output, shared, sparse, router.
            let query = 4 + u * 81160 + self.scratch(u, 5120, 1280)? + self.scratch(u, 1280, 32768)?;
            let output = 4 + r * 92424 + self.scratch(u, 32768, 8192)? + self.scratch(u, 8192, 5120)?;
            let shared = if placed { 0 } else { self.shared_ffn(u)? + r * 20480 };
            let sparse = sparse64(u);
            add(&mut out, "lane", 2 * r * 112752 + query + output + shared + sparse + r * 17105);
            // Cache producers: window storage and the source compressor storage(s).
            let windows = if placed { (0..40).filter(|&l| v41_attention_gpu(l) == rank).count() as u64 } else { 40 };
            let window = if windows == 0 { 0 } else {
                self.scratch(u, 5120, 512)? + 4 + u * (10240 + 1024 + 1024 + 8 + 256) + windows * r * (512 + 16 + 8)
            };
            let sources: Vec<u64> = [2u64, 8, 14, 20].into_iter()
                .filter(|&l| !placed || v41_attention_gpu(l as usize) == rank).collect();
            // Compression ratio two at sources 2, 8 and 14; one at 20.
            let paired = |l: u64| matches!(l, 2 | 8 | 14);
            let compressor = if sources.is_empty() { 0 } else {
                COMPRESSOR_WORKSPACE + u * (10240 + 8 + 256 + 256 + 256 + 1024
                    + if sources.iter().any(|&l| paired(l)) { 8 } else { 0 })
                    + sources.iter().map(|&l| r * (288 + 64 + 4 + 8)
                        + u * if paired(l) { 2048 + 2048 } else { 1024 }).sum::<u64>()
            };
            add(&mut out, "cache producers", window + compressor
                + if window_batch { slots as u64 * window_batch_bytes(u)? } else { 0 });
            // Index lane (query, source selection, reindex follower on source 20's GPU).
            let owns_index = !placed || [2usize, 8, 14, 20, 24, 28, 32, 36].iter().any(|&l| v41_attention_gpu(l) == rank);
            if owns_index {
                let index_query = COMPRESSOR_WORKSPACE + self.scratch(u, 1280, 4096)? + 4 + u * 23560;
                let selection = index_selection(u)? + (r - u) * 8192;
                let follower = if !placed || v41_attention_gpu(20) == rank { u * (2048 + 8192) } else { 0 };
                add(&mut out, "index", index_query + selection + follower);
            }
            if !placed || rank == 0 {
                add(&mut out, "encoder suffix", unit.min(16 * 128) * (40960 + 16));
                add(&mut out, "embedding", u * (4 + 40960 + 16 + 10240 + 4));
            }
            // Engram: reserve row planes on each gate GPU (layers 1, 14), gates' scratch at the unit.
            let gates = [1usize, 14].iter().filter(|&&l| !placed || v41_attention_gpu(l) == rank).count() as u64;
            if gates > 0 {
                add(&mut out, "engram rows", r * crate::ENGRAM_DEVICE_ROW_BYTES as u64);
                add(&mut out, "engram gates", gates * (u * (40960 * 2 + 51200 + 24 * 512 + 1)
                    + self.scratch(u, 6144, 25600)? + 4));
            }
            if rank == ranks - 1 { add(&mut out, "target taps", r * 30720); }
            add(&mut out, "target heads", slots as u64 * self.head(ranks, rank, head_mode)?);
        }
        Ok(out)
    }

    /// One slot pass's verify-width target head on `rank` (every pass has one),
    /// with the GEMV scratch of `head_mode`'s resident copy when target logits
    /// project through it.
    fn head(&self, ranks: usize, rank: usize, head_mode: Fp8Head) -> Result<u64> {
        let head_rows = cuteafd_core::V41_VERIFY_ROWS as u64;
        let normalization = head_rows * (40960 + 16 + 10240 + 10240);
        let sampler = head_rows * (64 + 64 + 2 * VOCAB.div_ceil(32) * 4 + 2048 * 4 + 256 * 12 + 32);
        let head = if ranks == 1 {
            BLAS + normalization + head_rows * (VOCAB * 4 + 8) + sampler
        } else {
            BLAS + head_rows * (10240 + VOCAB / 2 * 4 + if rank == 0 { 8 } else { 24 })
                + if rank == 1 { normalization + head_rows * VOCAB * 4 + sampler } else { 0 }
        };
        Ok(head + if head_mode.target() {
            self.head_scratch(head_mode, head_rows, VOCAB / ranks as u64)?
        } else { 0 })
    }

    /// Every target owner but the conditional local expert / TP2 FFN arenas,
    /// weights, cache/prefix banks, vision and Spark transports: the two
    /// decode lanes, private at `decode_rows` (`DECODE_LANE_ROWS`), and the
    /// prefill slots' storage ([`Self::prefill_slots`]).
    pub fn target(&self, decode_rows: u64, reserve: u64, unit: u64, ranks: usize, slots: usize, head_mode: Fp8Head,
        window_batch: bool) -> Result<Vec<Demand>> {
        let mut out = self.decode_lanes(decode_rows, ranks, head_mode, window_batch)?;
        out.extend(self.prefill_slots(reserve, unit, ranks, slots, head_mode, window_batch)?);
        Ok(out)
    }

    /// The two decode lanes, every owner private at `rows`.
    pub fn decode_lanes(&self, rows: u64, ranks: usize, head_mode: Fp8Head,
        window_batch: bool) -> Result<Vec<Demand>> {
        ensure!((1..=4096).contains(&rows) && matches!(ranks, 1 | 2)
            && (!window_batch || ranks == 1), "invalid V4.1 target owner geometry");
        let mut out = Vec::new();
        for rank in 0..ranks {
            for lane in 0..2 {
                let add = |out: &mut Vec<Demand>, group: &str, bytes| out.push(Demand::new(
                    rank as u8, Category::Workspace, format!("lane {lane} {group}"), bytes, Basis::Formula));
                let query = 4 + rows * (10240 + 2560 + 2560 + 65536 + 8 + 256)
                    + self.scratch(rows, 5120, 1280)? + self.scratch(rows, 1280, 32768)?;
                let output = 4 + rows * (65536 + 16384 + 10240 + 8 + 256)
                    + self.scratch(rows, 32768, 8192)? + self.scratch(rows, 8192, 5120)?;
                let sparse = rows * (64 * 2048 + 88)
                    + rows.min(cuteafd_core::V41_VERIFY_ROWS as u64) * (10 * 64 * 514 * 4 + 120);
                let shared = if ranks == 1 { self.shared_ffn(rows)? + rows * 20480 } else { 0 };
                add(&mut out, "backbone", 2 * rows * 112752 + query + output + sparse
                    + rows * (11825 + 5280) + shared);
                // One shared window producer scratch per pass and GPU; each of
                // the GPU's window layers keeps only its proposal and commit
                // destinations until the accepted rows commit.
                let window_scratch = self.scratch(rows, 5120, 512)? + 4 + rows * (10240 + 1024 + 1024 + 8 + 256);
                let window_proposal = rows * (512 + 16 + 8);
                let compressor = |ratio2| BLAS + rows * if ratio2 {
                    10240 + 2048 + 2048 + 8 + 1024 + 264 + 512 + 68 + 8 + 288
                } else { 10240 + 1024 + 1024 + 264 + 512 + 68 + 8 + 288 };
                add(&mut out, "cache producers", window_scratch + window_proposal * if ranks == 1 { 40 } else { 20 }
                    + if rank == 0 { 3 * compressor(true) } else { 0 }
                    + if rank == ranks - 1 { compressor(false) } else { 0 });
                if window_batch {
                    add(&mut out, "window batch table", window_batch_bytes(rows)?);
                }
                let index_rows = rows;
                let index_query = BLAS + self.scratch(index_rows, 1280, 4096)? + 4 + index_rows * 23560;
                let topk = 16384u64.div_ceil(2 * 512) * 512 * 8 * 2;
                let blocks = 2048u64.div_ceil(2 * 2048) * 2048 * 8 * 2;
                let selection = index_rows * (48 + 8 + 8 + 16384 * 8 + 16384 * 4
                    + 2048 * 4 + 2048 * 8 + 4096 + topk + 2048 + 16384 + blocks + 8192);
                let follower = if rank == ranks - 1 { index_rows * (2048 + 8192) } else { 0 };
                add(&mut out, "index", index_query + selection + follower);
                if rank == 0 {
                    add(&mut out, "embedding", rows * (4 + 40960 + 16 + 10240 + 4));
                    // Each live unit owns one upload region, reused by L1/L14.
                    // Keep it distinct from gate scratch so admission can assert
                    // k * unit_rows exactly, without charging the upload twice.
                    add(&mut out, "engram rows", rows * crate::ENGRAM_DEVICE_ROW_BYTES as u64);
                    add(&mut out, "engram gates", 2 * (rows * (40960 * 2 + 51200 + 24 * 512 + 1)
                        + self.scratch(rows, 6144, 25600)? + 4));
                }
                if rank == ranks - 1 { add(&mut out, "target taps", index_rows * 30720); }
                add(&mut out, "target head", self.head(ranks, rank, head_mode)?);
            }
        }
        Ok(out)
    }

    /// Decoder-local default dSpark: request chains are independent by lane;
    /// main-context producers and committed windows are shared across them.
    /// `experts` is the published coordinator (not RTX backbone) manifest.
    pub fn dspark(&self, experts: &Value, rows: u64, requests: u64, width: u64,
        ranks: usize, head_mode: Fp8Head) -> Result<Vec<Demand>> {
        ensure!((1..=4096).contains(&rows) && (1..=16).contains(&requests)
            && matches!(width, 5 | 7) && matches!(ranks, 1 | 2), "invalid V4.1 draft owners");
        ensure!(experts["role"].as_str() == Some("coordinator"), "dSpark expert manifest role mismatch");
        ensure!(experts["input_format"].as_str() == Some("bf16"),
            "dSpark owner inventory requires the native BF16 coordinator input");
        for (key, expected) in [("hidden", 5120), ("intermediate", 2304),
            ("kernel_intermediate", 2304), ("experts", 128), ("topk", 3)] {
            ensure!(experts["geometry"][key].as_u64() == Some(expected), "dSpark expert geometry mismatch: {key}");
        }
        let lanes = if requests == 1 { 1 } else { 2 };
        let members = requests.div_ceil(lanes);
        let live = members * width;
        let capacity = if live <= 16 { 16 } else if live <= 40 { 40 } else if live <= 80 { 80 } else { 256 };
        let variants = experts["variants"].as_array().context("missing dSpark expert variants")?;
        let matching: Vec<_> = variants.iter().filter(|v|
            v.get("requested_rows").unwrap_or(&v["capacity_rows"]).as_u64() == Some(capacity)).collect();
        ensure!(matching.len() == 1, "missing or duplicate dSpark expert capacity {capacity}");
        let expert = matching[0]["core_scratch_nbytes"].as_u64().context("missing dSpark expert scratch")?
            + capacity * (10240 + 3 * 8 + 2 * 10240);
        let output = self.projection(capacity, 8192, 5120)?
            + self.scratch(capacity, 32768, 8192)? + 4 + capacity * (65536 + 256);
        let attention = live * (1024 + 8) + 128 + output
            + self.projection(capacity, 5120, 1280)?
            + self.projection(capacity, 1280, 32768)?
            + self.projection(capacity, 5120, 512)?;
        let chain = 3 * (expert + self.shared_ffn(capacity)? + capacity * 128 * 4
            + 2 * capacity * 112752 + attention) + members * 4;
        let terminal = members * (width * VOCAB * 8 + (width + 2) * 4 + 16
            + width * 10240 + width * (40960 + 16)) + BLAS
            + members * (4 + 512 + VOCAB * 4) + BLAS
            + live * (10240 + 512 + 4);
        let decoder = ranks - 1;
        let mut out = Vec::new();
        let mut add = |rank, group, bytes| out.push(Demand::new(rank as u8, Category::Drafter,
            group, bytes, Basis::Formula));
        for rank in 0..ranks {
            let terminal = if ranks == 1 { terminal } else {
                BLAS + live * (10240 + VOCAB / 2 * 4 + if rank == 0 { 8 } else { 24 })
                    + if rank == decoder { terminal - BLAS } else { 0 }
            };
            add(rank, "dSpark request lanes", lanes * (terminal
                + if rank == decoder { chain } else { 0 }
                + if head_mode.draft() { self.head_scratch(head_mode, live, VOCAB / ranks as u64)? } else { 0 }));
        }
        add(decoder, "dSpark main context", self.main_context(rows)?
            + if lanes > 1 { self.main_context(80)? } else { 0 });
        add(decoder, "dSpark windows", 3 * (requests * 128 * 528 + rows * 1024 + 384));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest() -> Value {
        let matrices = [(5120, 1280), (1280, 32768), (32768, 8192), (8192, 5120),
            (5120, 2304), (2304, 5120), (5120, 512), (1280, 4096),
            (6144, 25600), (15360, 5120)];
        let variants = matrices.into_iter().flat_map(|(k, n)| {
            [1, 16, 80, 256, 1024, 4096].map(move |rows| json!({
                "capacity": rows, "input_dim": k, "output_dim": n,
                "label": format!("matrix_{k}_{n}_{rows}"), "scratch_bytes": rows + 1,
                "expected_m": if rows <= 80 { 16 } else { rows },
                "split_k_slices": 1, "all_rows": rows <= 80,
            }))
        }).collect::<Vec<_>>();
        json!({"variants": variants})
    }

    fn experts() -> Value {
        json!({"role":"coordinator", "input_format":"bf16", "geometry":{
            "hidden":5120, "intermediate":2304, "kernel_intermediate":2304,
            "experts":128, "topk":3}, "variants": ([16, 40, 80, 256].map(|rows|
                json!({"capacity_rows":rows, "core_scratch_nbytes":rows * 7})))})
    }

    fn owner(demands: &[Demand], rank: u8, group: &str) -> u64 {
        demands.iter().find(|d| d.gpu == rank && d.group == group).unwrap().bytes
    }

    #[test]
    fn vocabulary_phase_charges_only_the_resident_weights_before_any_lane() -> Result<()> {
        let demand = |group: &str, bytes| Demand::new(0, Category::Workspace, group, bytes, Basis::Formula);
        let weights = vec![demand("attention", 100), demand("engram-projections", 20),
            demand("embedding", 50), demand("lm_head", 300), demand("norm", 10)];
        let target = vec![demand("lane 0 backbone", 40), demand("lane 0 engram", 30),
            demand("lane 0 window batch table", 5), demand("lane 0 target head", 60),
            demand("lane 0 target taps", 70), demand("lane 1 backbone", 80)];
        // Lanes and slots allocate after the head is packed.
        let phase = vocabulary_load_phase(&weights, &target, 400)?;
        assert_eq!(phase.bytes, 400 + 100 + 20 + 50);
        assert_eq!(phase.category, Category::Staging);
        assert!(vocabulary_load_phase(&[demand("attention", u64::MAX)], &target, 400).is_err());
        assert!(vocabulary_load_phase(&[demand("attention", 1).on(1)], &target, 400).is_err());
        for ranks in 1..=cuteafd_core::MAX_SPARK_WORLD {
            assert_eq!(transport_bytes(1024, ranks)?, 1024 * (ranks as u64 + 2) * 10240);
        }
        assert!(transport_bytes(0, 4).is_err());
        assert!(transport_bytes(4097, 4).is_err());
        assert!(transport_bytes(1024, 0).is_err());
        assert!(transport_bytes(1024, cuteafd_core::MAX_SPARK_WORLD + 1).is_err());
        Ok(())
    }

    #[test]
    fn startup_and_ready_fixture_pins_single_and_dual_allocation_order() -> Result<()> {
        use super::super::DraftInventory;
        use super::super::resident::Fp8Head;
        let demand = |gpu, group: &str, bytes| Demand::new(gpu, Category::Workspace,
            group, bytes, Basis::Formula);
        // A synthetic allocation trace, independent of the composition helper:
        // weights=170 (then the head packs), lanes and slots after it.
        let weights = vec![demand(0, "attention", 100), demand(0, "engram-projections", 20),
            demand(0, "embedding", 50), demand(0, "lm_head", 682_598_400), demand(0, "norm", 10)];
        let target = vec![demand(0, "lane 0 backbone", 40), demand(0, "lane 0 engram", 30),
            demand(0, "lane 0 window batch table", 5), demand(0, "lane 0 target head", 60),
            demand(0, "lane 0 target taps", 70), demand(0, "lane 1 backbone", 80)];
        let vocabulary = vocabulary_load_phase(&weights, &target,
            Fp8Head::All.load_bytes(1_323_827_200))?;
        assert_eq!(vocabulary.bytes, 2_006_425_845 - 40 - 30 - 5);
        let draft = DraftInventory { resident: 600, loading_experts: [500, 0],
            staging: [200, 0], loader_peak: [600, 0], marks: 50,
            runtime: vec![demand(0, "draft runtime", 70)], ..Default::default() };
        let mut single = TargetOwners::new(weights, target, 1, 200, 2)?;
        single.add_vision(demand(0, "vision", 300))?;
        // Transport already exists. Vision, target snapshots, draft runtime and
        // marks allocate after draft weights, and cannot enter this phase.
        assert_eq!(single.draft_load_phases(&draft)?[0].bytes, 682_599_765);
        assert_eq!(single.fixed.iter().map(|d| d.bytes).sum::<u64>(), 688_039_877);
        assert!(single.fixed.iter().all(|d| d.lifetime == ItemLifetime::ReadyResident));
        let draft_owner = super::super::drafter(&draft, 1)?;
        assert_eq!(draft_owner.conditional.iter().flat_map(|home| &home.demands).map(|d| d.bytes).sum::<u64>(), 720);

        let mut dual = TargetOwners::new(vec![demand(0, "weights", 100), demand(1, "weights", 200)],
            vec![demand(0, "target", 30), demand(1, "target", 40)], 2, 200, 2)?;
        dual.add_vision(demand(0, "vision", 300))?;
        let split = DraftInventory { resident: 600, peer_half: 400,
            loading_experts: [500, 300], staging: [200, 150], loader_peak: [600, 400],
            ..Default::default() };
        let phases = dual.draft_load_phases(&split)?;
        // Decoder weights/lanes + draft only: later transport (200) and both
        // 64 MiB setup reserves are absent. GPU0 also has vision and snapshots.
        assert_eq!(phases.iter().map(|d| (d.gpu, d.bytes)).collect::<Vec<_>>(),
            vec![(1, 940), (0, 5_441_392)]);
        let totals = |lifetime| (0..2).map(|gpu| dual.fixed.iter()
            .filter(|d| d.gpu == gpu && d.lifetime == lifetime).map(|d| d.bytes).sum::<u64>())
            .collect::<Vec<_>>();
        assert_eq!(totals(ItemLifetime::ReadyResident), vec![5_440_942, 440]);
        assert_eq!(totals(ItemLifetime::PostReadyReserve), vec![67_108_864; 2]);
        // Renaming owners must not change their structural phase membership.
        for owner in &mut dual.fixed { owner.group = "renamed".into(); }
        assert_eq!(dual.draft_load_phases(&split)?, phases);
        assert!(TargetOwners::new(vec![demand(2, "bad", 1)], vec![], 2, 0, 0).is_err());
        let overflow = TargetOwners::new(vec![demand(0, "overflow", u64::MAX)], vec![], 1, 0, 0)?;
        assert!(overflow.draft_load_phases(&draft).is_err());
        Ok(())
    }

    #[test]
    fn native_manifest_uses_the_matching_artifact_layout_and_refuses_bad_json() -> Result<()> {
        let root = tempfile::tempdir()?;
        let lib = root.path().join("lib");
        let share = root.path().join("share");
        let variant = lib.join("x86-64-v4");
        std::fs::create_dir_all(&variant)?;
        std::fs::create_dir(&share)?;
        let name = "V41_FP8_AOT.json";
        assert!(native_manifest(&lib.join("native.so"), name).is_err());
        std::fs::write(share.join(name), br#"{"source":"installed"}"#)?;
        for directory in [&lib, &variant] {
            assert_eq!(native_manifest(&directory.join("native.so"), name)?["source"], "installed");
        }
        std::fs::write(lib.join(name), br#"{"source":"sealed"}"#)?;
        assert_eq!(native_manifest(&lib.join("native.so"), name)?["source"], "sealed");
        std::fs::write(lib.join(name), b"invalid")?;
        assert!(native_manifest(&lib.join("native.so"), name).is_err());
        Ok(())
    }

    #[test]
    fn target_is_two_decode_lanes_and_one_prefill_slot_set() -> Result<()> {
        let manifest = manifest();
        let programs = NativePrograms::new(&manifest, 188)?;
        for ranks in [1, 2] {
            let target = programs.target(80, 8256, 4096, ranks, 4, Fp8Head::Off, ranks == 1)?;
            for rank in 0..ranks as u8 {
                // Decode lanes are identical, private, at decode width.
                for group in ["backbone", "index", "target head"] {
                    assert_eq!(owner(&target, rank, &format!("lane 0 {group}")),
                        owner(&target, rank, &format!("lane 1 {group}")), "{group}");
                }
                // One slot set per GPU; four slot passes' heads in it.
                assert_eq!(owner(&target, rank, "prefill slots target heads"),
                    4 * owner(&target, rank, "lane 0 target head"));
                assert!(owner(&target, rank, "prefill slots lane") > owner(&target, rank, "lane 0 backbone"));
            }
            assert_eq!(owner(&target, 0, "prefill slots encoder suffix"), 2048 * (40960 + 16));
            assert!(!target.iter().any(|d| d.gpu != 0 && d.group.ends_with("encoder suffix")));
            let small = programs.prefill_slots(256, 64, ranks, 2, Fp8Head::Off, ranks == 1)?;
            assert_eq!(owner(&small, 0, "prefill slots encoder suffix"), 64 * (40960 + 16));
            assert!(target.iter().all(|d| d.category == Category::Workspace));
            assert!(!target.iter().any(|d| d.group.starts_with("lane 2")), "decode lanes only");
            // Engram planes live with the gates (layers 1 and 14, GPU0 on two RTX).
            assert!(!target.iter().any(|d| d.gpu != 0 && d.group.contains("engram")));
            // The slot set's rows scale with the reserve, its scratch with the unit.
            let narrow = programs.prefill_slots(8256, 2048, ranks, 4, Fp8Head::Off, ranks == 1)?;
            let short = programs.prefill_slots(4224, 2048, ranks, 4, Fp8Head::Off, ranks == 1)?;
            for rank in 0..ranks as u8 {
                assert!(owner(&narrow, rank, "prefill slots lane") > owner(&short, rank, "prefill slots lane"));
            }
            let fp8 = programs.target(80, 8256, 4096, ranks, 4, Fp8Head::All, ranks == 1)?;
            let packed = programs.target(80, 8256, 4096, ranks, 4, Fp8Head::Packed, ranks == 1)?;
            let draft = programs.target(80, 8256, 4096, ranks, 4, Fp8Head::Draft, ranks == 1)?;
            for rank in 0..ranks as u8 {
                for group in ["lane 0 target head", "lane 1 target head"] {
                    assert_eq!(owner(&fp8, rank, group) - owner(&target, rank, group), 660992);
                    // The exact copy's GEMV packs BF16 activations unscaled: no row scales or maxima.
                    assert_eq!(owner(&packed, rank, group) - owner(&target, rank, group), 655360);
                    assert_eq!(owner(&draft, rank, group), owner(&target, rank, group));
                }
                // Every slot pass's private head carries its own copy scratch.
                assert_eq!(owner(&packed, rank, "prefill slots target heads")
                    - owner(&target, rank, "prefill slots target heads"), 4 * 655360);
            }
        }
        assert!(programs.target(0, 8256, 4096, 1, 2, Fp8Head::All, false).is_err());
        assert!(programs.target(80, 8256, 4097, 2, 2, Fp8Head::All, false).is_err());
        assert!(programs.target(80, 8256, 4096, 3, 2, Fp8Head::All, false).is_err());
        assert!(programs.target(80, 8256, 4096, 2, 2, Fp8Head::All, true).is_err());
        assert!(programs.prefill_slots(2048, 4096, 1, 2, Fp8Head::Off, false).is_err(), "unit wider than the reserve");
        let off = programs.target(80, 8256, 4096, 1, 4, Fp8Head::Off, false)?;
        let on = programs.target(80, 8256, 4096, 1, 4, Fp8Head::Off, true)?;
        assert_eq!(on.iter().map(|d| d.bytes).sum::<u64>() - off.iter().map(|d| d.bytes).sum::<u64>(),
            2 * window_batch_bytes(80)? + 4 * window_batch_bytes(4096)?);
        for rows in [1, 80, 256, 1024, 4096] {
            assert_eq!(window_batch_bytes(rows)?, 40 * (320 + 8 * rows));
        }
        assert!(window_batch_bytes(0).is_err());
        assert!(window_batch_bytes(4097).is_err());
        Ok(())
    }

    #[test]
    fn draft_chains_are_lane_local_but_windows_and_producers_are_shared() -> Result<()> {
        let manifest = manifest();
        let programs = NativePrograms::new(&manifest, 188)?;
        let experts = experts();
        for ranks in [1, 2] {
            let one = programs.dspark(&experts, 1024, 1, 5, ranks, Fp8Head::Off)?;
            let two = programs.dspark(&experts, 1024, 2, 5, ranks, Fp8Head::Off)?;
            let decoder = (ranks - 1) as u8;
            for rank in 0..ranks as u8 {
                assert_eq!(owner(&two, rank, "dSpark request lanes"),
                    2 * owner(&one, rank, "dSpark request lanes"));
            }
            assert_eq!(owner(&two, decoder, "dSpark main context")
                - owner(&one, decoder, "dSpark main context"), programs.main_context(80)?);
            assert_eq!(owner(&two, decoder, "dSpark windows")
                - owner(&one, decoder, "dSpark windows"), 3 * 128 * 528);
            let lanes_packed = programs.dspark(&experts, 1024, 2, 5, ranks, Fp8Head::Packed)?;
            for rank in 0..ranks as u8 {
                assert_eq!(owner(&lanes_packed, rank, "dSpark request lanes") - owner(&two, rank, "dSpark request lanes"),
                    2 * programs.packed_vocabulary_scratch(5, VOCAB / ranks as u64)?);
            }
            for requests in 1..=16 {
                for width in [5, 7] {
                    let demands = programs.dspark(&experts, 1024, requests, width, ranks, Fp8Head::All)?;
                    assert!(demands.iter().all(|d| d.category == Category::Drafter));
                    assert_eq!(owner(&demands, decoder, "dSpark windows"),
                        3 * (requests * 128 * 528 + 1024 * 1024 + 384));
                }
            }
        }
        for (key, value) in [("input_format", json!("fp8")), ("role", json!("rtx_backbone"))] {
            let mut bad = experts.clone();
            bad[key] = value;
            assert!(programs.dspark(&bad, 256, 1, 5, 1, Fp8Head::Off).is_err());
        }
        let mut bad = experts.clone();
        bad["geometry"]["kernel_intermediate"] = 2560.into();
        assert!(programs.dspark(&bad, 256, 1, 5, 1, Fp8Head::Off).is_err());
        let mut duplicate = experts.clone();
        let variant = duplicate["variants"][0].clone();
        duplicate["variants"].as_array_mut().unwrap().push(variant);
        assert!(programs.dspark(&duplicate, 256, 1, 5, 1, Fp8Head::Off).is_err());
        Ok(())
    }

    #[test]
    fn vocabulary_workspace_matches_native_alignment_and_physical_sm_split() -> Result<()> {
        let manifest = manifest();
        for sms in [170, 188] {
            let programs = NativePrograms::new(&manifest, sms)?;
            for rows in [1, 16, 64, 80, 4096] {
                for n in [64640, VOCAB] {
                    assert_eq!(programs.vocabulary_scratch(rows, n)?, 512 + 5120 + 655360);
                    assert_eq!(programs.packed_vocabulary_scratch(rows, n)?, 655360);
                    assert_eq!(programs.head_scratch(Fp8Head::Packed, rows, n)?, 655360);
                    assert_eq!(programs.head_scratch(Fp8Head::All, rows, n)?, 512 + 5120 + 655360);
                }
            }
            assert!(programs.vocabulary_scratch(0, VOCAB).is_err());
            assert!(programs.vocabulary_scratch(4097, VOCAB).is_err());
            assert!(programs.vocabulary_scratch(64, u64::MAX).is_err());
            assert!(programs.packed_vocabulary_scratch(0, VOCAB).is_err());
            assert!(programs.packed_vocabulary_scratch(64, 4096).is_err());
        }
        let programs = NativePrograms::new(&manifest, 1024)?;
        assert_eq!(programs.vocabulary_scratch(64, 64640)?, 660992 + 5 * 64 * 64640 * 4);
        assert_eq!(programs.packed_vocabulary_scratch(64, 64640)?, 655360 + 5 * 64 * 64640 * 4);
        assert_eq!(programs.packed_vocabulary_scratch(3, 64640)?, 655360 + 5 * 3 * 64640 * 4);
        assert!(NativePrograms::new(&manifest, 0).is_err());
        Ok(())
    }
}
