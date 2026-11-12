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

/// One retained encoder-suffix row: the layer-19 residual (40960 B) and its
/// pre-norm scales (16 B). The runtime's `EncoderSuffix` sizes rows with it.
pub const ENCODER_SUFFIX_ROW_BYTES: u64 = 40960 + 16;
/// Rows of one prompt's retained suffix: the decoder replay's final window.
pub const ENCODER_SUFFIX_ROWS: u64 = 128;
/// The packed arena's rows, bounded by the resolved boot request count.
pub fn packed_suffix_rows(unit: u64, concurrency: u64) -> u64 {
    unit.min(concurrency * ENCODER_SUFFIX_ROWS)
}
/// The prefill slots' encoder suffix storage: the packed arena plus one
/// prompt suffix per request slot (`concurrency`), every one allocated at
/// startup. Prompts mid-encoder (several interleave under the decode share)
/// each hold one; a request slot holds at most one prompt.
pub fn encoder_suffix_bytes(unit: u64, concurrency: u64) -> u64 {
    (packed_suffix_rows(unit, concurrency) + concurrency * ENCODER_SUFFIX_ROWS) * ENCODER_SUFFIX_ROW_BYTES
}

/// Rows of a decode lane's private owners: the compiled cover of the widest
/// decode/verify step (`V41_VERIFY_ROWS`).
pub const DECODE_LANE_ROWS: u64 = cuteafd_core::V41_VERIFY_ROWS as u64;
const _: () = assert!(cuteafd_core::V41_VERIFY_ROWS as u64 == DECODE_LANE_ROWS);

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

/// Source and follower index-selection owners. Selected IDs survive later
/// attention hooks in both owners; only source candidate blocks span the reserve.
/// Shared with the runtime so admission and allocation use one retention formula.
pub fn index_selection_storage_bytes(reserve: u64, unit: u64) -> Result<[u64; 2]> {
    ensure!((1..=4096).contains(&unit) && unit <= reserve && reserve <= 8256,
        "invalid V4.1 index selection geometry");
    Ok([index_selection(unit)? + (reserve - unit) * (2048 + 8192),
        reserve * 2048 + unit * 8192])
}

/// Single-RTX cache execution retains a per-layer descriptor/destination table.
/// Placed two-RTX producers use their own publication storage instead.
pub fn window_batch_bytes(rows: u64) -> Result<u64> {
    ensure!((1..=4096).contains(&rows), "invalid V4.1 window batch rows");
    Ok(40 * (64 + rows * 8 + cuteafd_core::V41_MAX_REQUESTS as u64 * 16))
}

/// Coordinator-owned Spark wave planes. Pinned intake belongs to its own domain.
pub fn transport_bytes(rows: u64, ranks: usize) -> Result<u64> {
    ensure!((1..=4096).contains(&rows) && (1..=cuteafd_core::MAX_SPARK_WORLD).contains(&ranks),
        "invalid V4.1 transport geometry");
    Ok(crate::serving_capacity::spark_intake_bytes(1, ranks as u64, rows, 5120) + rows * 2 * 5120 * 2)
}

/// Every Spark transport (PLAN "Prefill contract"): the two decode lanes'
/// links at the decode width ([`transport_bytes`] of `decode_rows`), each of
/// the `slots` prefill slots' reduce output at the unit `rows` (slot 0's FFN
/// owner also holds the link), and the one prefill link's intake: every
/// rank's partial-sum plane over its `slots` return slots.
pub fn transports_bytes(decode_rows: u64, rows: u64, ranks: usize, slots: usize,
    format: cuteafd_core::SparkReplyFormat) -> Result<u64> {
    ensure!((1..=4).contains(&slots), "invalid V4.1 prefill slots");
    let lanes = 2 * transport_bytes(decode_rows, ranks)?;
    let outputs = slots as u64 * rows * 5120 * 2;
    Ok(lanes + outputs + crate::serving_capacity::prefill_return_bytes_with_format(slots as u64, rows, ranks as u64, 5120, format)?)
}

/// Single-RTX vocabulary packing peak above the pre-weight sample. The whole
/// target arena (final head copy included) is allocated before any payload, so
/// the peak is its extent B plus the head conversion's transient source, the
/// same bound `TargetArenas::new` checks before allocating. Every lane and slot
/// owner, transport, draft, vision and mark allocates after it.
pub fn vocabulary_load_phase(arena: &crate::serving_capacity::WeightArenaLayout,
    head: super::resident::Fp8Head) -> Result<Demand> {
    let transient = super::weight_layout::TargetLayout::vocabulary_transient_bytes(head, 1)? as u64;
    let bytes = (arena.bytes() as u64).checked_add(transient).context("V4.1 vocabulary phase overflow")?;
    Ok(Demand::new(0, Category::Staging, "vocabulary packing", bytes, Basis::Formula))
}

/// BF16 sources coexist with all push destinations and workspaces until the
/// group commits. Charge them during Load, never as retained target weights.
pub fn packed_head_load_owners(layouts: &[crate::serving_capacity::WeightArenaLayout])
    -> Result<Vec<super::admission::Owner>> {
    use crate::placement::planner::{Lifetime, Phase};
    if !layouts.iter().any(|layout| layout.view("head.packed12").is_some()) { return Ok(Vec::new()); }
    let heads = layouts.iter().filter(|layout| layout.view("head.packed12").is_some()).collect::<Vec<_>>();
    let bytes = super::weight_layout::TargetLayout::vocabulary_transient_bytes(Fp8Head::Packed, heads.len())? as u64;
    Ok(heads.iter().enumerate()
        .map(|(rank, _)| super::admission::Owner::charged(rank as u8, Category::Staging,
            "vocabulary BF16 push source", bytes)
            .with_lifetime(Lifetime { first: Phase::Load, last: Phase::Load })).collect())
}

/// Retained device target weights as one demand per arena (extent B, resident
/// B), replacing the per-tensor `resident_headers` device items. Load phases,
/// the pre-draft sample and admission all see the arena exactly where the
/// payload used to be; `ProductionInput::weight_arenas` carries the layouts.
pub fn target_arena_demands(layout: &super::weight_layout::TargetLayout) -> Vec<Demand> {
    layout.ranks.iter().enumerate().map(|(rank, arena)| arena_demand(rank as u8, arena)).collect()
}

/// One arena's placeholder demand (extent B, resident B): every load phase and
/// sum sees it where the payload was. `ProductionInput` binds it by arena ID to
/// its layout and charges the shared root, logical claim and tail.
pub fn arena_demand(gpu: u8, arena: &crate::serving_capacity::WeightArenaLayout) -> Demand {
    Demand::new(gpu, Category::Weights, format!("{WEIGHT_ARENA_PREFIX}{}", arena.id()),
        arena.bytes() as u64, Basis::Exact)
        .with_resident_weights(crate::placement::ResidentWeights(arena.bytes() as u64))
}

/// Arena demands are named `weight arena:<layout id>`.
pub const WEIGHT_ARENA_PREFIX: &str = "weight arena:";

/// The device weight items the arenas replace: everything `resident_headers`
/// places on a coordinator GPU (a host-placed embedding stays a host owner).
pub fn is_arena_weight(item: &cuteafd_core::memory_layout::Item) -> bool {
    matches!(item.category, Category::Weights | Category::Embedding | Category::Tables)
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

/// All compressor/index plans borrow the serving thread's Lt pool. dSpark's
/// Markov GEMM lives only beside the terminal; vocabulary plans join that pool.
pub fn blas_owners(ranks: usize, head: crate::placement::planner::serving::BlasHead,
    dspark: bool) -> Result<Vec<crate::placement::planner::serving::BlasOwners>> {
    use crate::placement::planner::{Activation, Lifetime, Location, Phase, serving::BlasOwners};
    ensure!(matches!(ranks, 1 | 2), "invalid V4.1 BLAS owners");
    let mut owners = (0..ranks).map(|gpu| BlasOwners {
        name: format!("V4.1 compressor, index and vocabulary owner{gpu}"),
        process: "coordinator".into(), thread: "v41/blas/serve".into(),
        location: Location::coordinator(gpu as u8), gemm: head.gemm(), lt: true,
        lifetime: Lifetime::RESIDENT, activation: Activation::always(),
    }).collect::<Vec<_>>();
    if dspark {
        owners.push(BlasOwners { name: "V4.1 dSpark Markov and terminal".into(),
            process: "coordinator".into(), thread: "v41/blas/serve".into(),
            location: Location::coordinator((ranks - 1) as u8), gemm: true, lt: false,
            lifetime: Lifetime { first: Phase::DraftLoad, last: Phase::Growth },
            activation: Activation::always() });
    }
    Ok(owners)
}

/// Draft-row bucket a concurrency and draft width resolve to: the smallest
/// compiled draft shape the lanes' live rows fit. Offline layout and serving
/// admission both select modules with it, so the predicted module set and the
/// one the drafter loads are the same.
pub fn draft_rows(concurrency: u64, width: u8) -> u64 {
    let lanes = if concurrency == 1 { 1 } else { 2 };
    let live = concurrency.div_ceil(lanes) * if width > 5 { 7u64 } else { 5u64 };
    [16, 40, cuteafd_core::V41_DRAFT_LEGACY_ROWS as u64,
        cuteafd_core::V41_DRAFT_WIDE_ROWS as u64].into_iter()
        .find(|&rows| rows >= live)
        .unwrap_or(cuteafd_core::V41_DRAFT_WIDE_ROWS as u64)
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

    /// Selected native projection modules, not the all-family PROGRAMS.json.
    /// Each selected variant loads quantization and GEMM; grouped queries also
    /// load their RoPE quantizer. The shared module formula includes fixed CUDA
    /// kernels, but no BLAS residency or graph executables.
    pub fn modules(&self, decode_rows: u64, unit: u64, ranks: usize,
        draft_rows: Option<u64>) -> Result<Vec<ProgramSet>> {
        ensure!(matches!(ranks, 1 | 2), "invalid V4.1 module owners");
        let target = [(5120, 1280), (1280, 32768), (32768, 8192),
            (8192, 5120), (5120, 512), (1280, 4096)];
        (0..ranks).map(|rank| {
            let mut selected = std::collections::BTreeSet::new();
            let mut add = |rows, matrices: &[(u64, u64)]| -> Result<()> {
                let plans = ProgramSet::v41_fp8(self.manifest, matrices, rows, self.sms)?;
                for label in plans.names {
                    selected.insert(format!("{label}/quant"));
                    selected.insert(format!("{label}/gemm"));
                    let variant = self.manifest["variants"].as_array().context("missing native module variants")?.iter()
                        .find(|variant| variant["label"].as_str() == Some(label.as_str()))
                        .context("selected native module label disappeared")?;
                    if variant["groups"].as_u64().unwrap_or(1) > 1 {
                        selected.insert(format!("{label}/quant_rope"));
                    }
                }
                Ok(())
            };
            for rows in [decode_rows, unit] {
                add(rows, &target)?;
                if rank == 0 { add(rows, &[(6144, 25600)])?; }
                add(rows, if ranks == 1 { &[(5120, 2304), (2304, 5120)] }
                    else { &[(5120, 1152), (1152, 5120)] })?;
            }
            if rank == ranks - 1 {
                if let Some(rows) = draft_rows {
                    add(rows, &target)?;
                    add(rows, &[(15360, 5120), (5120, 2304), (2304, 5120)])?;
                }
            }
            Ok(ProgramSet { families: vec!["v41".into()],
                names: selected.into_iter().collect(), scratch: Vec::new() })
        }).collect()
    }

    fn shared_ffn(&self, rows: u64) -> Result<u64> {
        Ok(self.scratch(rows, 5120, 2304)? + self.scratch(rows, 2304, 5120)?
            + rows * 2304 * 6 + 4)
    }

    fn projection(&self, rows: u64, k: u64, n: u64) -> Result<u64> {
        let backings = crate::serving_capacity::v41_workspace::projection_backings(rows as usize,
            k as usize, n as usize, self.scratch(rows, k, n)? as usize, true)?;
        Ok(crate::serving_capacity::v41_workspace::projection_backing_bytes(&backings)? as u64)
    }

    fn external_projection(&self, rows: u64, k: u64, n: u64) -> Result<u64> {
        let backings = crate::serving_capacity::v41_workspace::projection_backings(rows as usize,
            k as usize, n as usize, self.scratch(rows, k, n)? as usize, false)?;
        Ok(crate::serving_capacity::v41_workspace::projection_backing_bytes(&backings)? as u64)
    }

    fn main_context(&self, rows: u64) -> Result<u64> {
        Ok(self.projection(rows, 15360, 5120)? + 3 * cuteafd_core::V41_WRITE_DESCRIPTOR_BYTES as u64
            + rows * (10240 + 256 + 8 + 3 * 1024)
            + 3 * self.external_projection(rows, 5120, 512)?)
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
        self.vocabulary_splits(rows, n)?;
        Ok(cuteafd_core::packed_head::workspace_bytes(rows, 5120, n, u64::from(self.sms)))
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
    pub fn prefill_slots(&self, reserve: u64, unit: u64, ranks: usize, slots: usize, concurrency: u64,
        head_mode: Fp8Head, window_batch: bool) -> Result<Vec<Demand>> {
        cuteafd_core::concurrency::validate("deepseek_v41", concurrency)?;
        ensure!((1..=4096).contains(&unit) && unit <= reserve && reserve <= 8256 && matches!(ranks, 1 | 2)
            && (1..=4).contains(&slots) && (!window_batch || ranks == 1),
            "invalid V4.1 prefill slot geometry");
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
            add(&mut out, "lane", 2 * crate::serving_capacity::v41_workspace::hc(r as usize)?.bytes() as u64
                + query + output + shared + sparse + r * 17105);
            // Cache producers: window storage and the source compressor storage(s).
            let windows = if placed { (0..40).filter(|&l| v41_attention_gpu(l) == rank).count() as u64 } else { 40 };
            let window = if windows == 0 { 0 } else {
                crate::serving_capacity::v41_workspace::window(r as usize, u as usize,
                    &(0..windows as usize).collect::<Vec<_>>(), self.scratch(u, 5120, 512)? as usize)?.bytes() as u64
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
                let [selection, retained_follower] = index_selection_storage_bytes(r, u)?;
                let follower = if !placed || v41_attention_gpu(20) == rank { retained_follower } else { 0 };
                add(&mut out, "index", index_query + selection + follower);
            }
            if !placed || rank == 0 {
                add(&mut out, "encoder suffix", encoder_suffix_bytes(unit, concurrency));
                add(&mut out, "embedding", u * (4 + 40960 + 16 + 10240 + 4));
            }
            // Engram: reserve row planes on each gate GPU (layers 1, 14), gates' scratch at the unit.
            let gates = [1usize, 14].iter().filter(|&&l| !placed || v41_attention_gpu(l) == rank).count() as u64;
            if gates > 0 {
                add(&mut out, "engram rows", r * crate::ENGRAM_DEVICE_ROW_BYTES as u64);
                add(&mut out, "engram gates", gates * crate::serving_capacity::v41_workspace::engram(
                    u as usize, self.scratch(u, 6144, 25600)? as usize)?.bytes() as u64);
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
    pub fn target(&self, decode_rows: u64, reserve: u64, unit: u64, ranks: usize, slots: usize, concurrency: u64,
        head_mode: Fp8Head, window_batch: bool) -> Result<Vec<Demand>> {
        let mut out = self.decode_lanes(decode_rows, ranks, head_mode, window_batch)?;
        out.extend(self.prefill_slots(reserve, unit, ranks, slots, concurrency, head_mode, window_batch)?);
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
                add(&mut out, "backbone", 2 * crate::serving_capacity::v41_workspace::hc(rows as usize)?.bytes() as u64
                    + query + output + sparse
                    + rows * (11825 + 5280) + shared);
                // One shared window producer scratch per pass and GPU; each of
                // the GPU's window layers keeps only its proposal and commit
                // destinations until the accepted rows commit.
                let window = crate::serving_capacity::v41_workspace::window(rows as usize, rows as usize,
                    &(0..if ranks == 1 { 40 } else { 20 }).collect::<Vec<_>>(),
                    self.scratch(rows, 5120, 512)? as usize)?.bytes() as u64;
                let compressor = |ratio2| BLAS + rows * if ratio2 {
                    10240 + 2048 + 2048 + 8 + 1024 + 264 + 512 + 68 + 8 + 288
                } else { 10240 + 1024 + 1024 + 264 + 512 + 68 + 8 + 288 };
                add(&mut out, "cache producers", window
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
                    add(&mut out, "engram gates", 2 * crate::serving_capacity::v41_workspace::engram(
                        rows as usize, self.scratch(rows, 6144, 25600)? as usize)?.bytes() as u64);
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
        ensure!((1..=4096).contains(&rows) && (1..=cuteafd_core::V41_MAX_REQUESTS as u64).contains(&requests)
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
        let capacity = if live <= 16 { 16 } else if live <= 40 { 40 } else if live <= cuteafd_core::V41_DRAFT_LEGACY_ROWS as u64 { cuteafd_core::V41_DRAFT_LEGACY_ROWS as u64 } else { cuteafd_core::V41_DRAFT_WIDE_ROWS as u64 };
        let variants = experts["variants"].as_array().context("missing dSpark expert variants")?;
        let matching: Vec<_> = variants.iter().filter(|v|
            v.get("requested_rows").unwrap_or(&v["capacity_rows"]).as_u64() == Some(capacity)).collect();
        ensure!(matching.len() == 1, "missing or duplicate dSpark expert capacity {capacity}");
        let expert = matching[0]["core_scratch_nbytes"].as_u64().context("missing dSpark expert scratch")?
            + capacity * (10240 + 3 * 8 + 2 * 10240);
        let output = self.projection(capacity, 8192, 5120)?
            + self.scratch(capacity, 32768, 8192)? + 4 + capacity * (65536 + 256);
        let attention = live * (1024 + 8) + cuteafd_core::V41_READ_DESCRIPTOR_BYTES as u64 + output
            + self.projection(capacity, 5120, 1280)?
            + self.projection(capacity, 1280, 32768)?
            + self.projection(capacity, 5120, 512)?;
        let chain = 3 * (expert + self.shared_ffn(capacity)? + capacity * 128 * 4
            + 2 * crate::serving_capacity::v41_workspace::hc(capacity as usize)?.bytes() as u64
            + attention) + members * 4;
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
            + if lanes > 1 { self.main_context(DECODE_LANE_ROWS)? } else { 0 });
        add(decoder, "dSpark windows", 3 * (requests * 128 * 528 + rows * 1024 + cuteafd_core::V41_WRITE_DESCRIPTOR_BYTES as u64));
        Ok(out)
    }

    /// Replace each full draft FFN by its native TP2 fork/join owners. The
    /// scratch maximum covers every installed variant the runtime retains.
    pub fn dspark_tp2(&self, experts: &Value, halves: &Value, rows: u64, requests: u64,
        width: u64, head_mode: Fp8Head) -> Result<Vec<Demand>> {
        let mut out = self.dspark(experts, rows, requests, width, 2, head_mode)?;
        ensure!(halves["role"] == "dspark_tp2" && halves["input_format"] == "fp8_k32",
            "invalid V4.1 TP2 draft expert manifest role/input");
        for (key, expected) in [("hidden", 5120), ("intermediate", 1152),
            ("kernel_intermediate", 1152), ("experts", 128), ("topk", 3)] {
            ensure!(halves["geometry"][key].as_u64() == Some(expected),
                "V4.1 TP2 draft expert geometry mismatch: {key}");
        }
        let lanes = if requests == 1 { 1 } else { 2 };
        let live = requests.div_ceil(lanes) * width;
        let capacity = if live <= 16 { 16 } else if live <= 40 { 40 }
            else if live <= cuteafd_core::V41_DRAFT_LEGACY_ROWS as u64 { cuteafd_core::V41_DRAFT_LEGACY_ROWS as u64 }
            else { cuteafd_core::V41_DRAFT_WIDE_ROWS as u64 };
        let cover = [1u64, 16, 80, 256, 1024, 4096].into_iter().find(|&c| c >= capacity)
            .context("V4.1 TP2 draft capacity out of range")?;
        let variants = halves["variants"].as_array().context("missing V4.1 TP2 draft variants")?;
        let mut scratch = 0;
        for cap in [1, 16, 80, 256, 1024, 4096].into_iter().filter(|&c| c <= cover) {
            let matching: Vec<_> = variants.iter().filter(|v| v["capacity_rows"].as_u64() == Some(cap)).collect();
            ensure!(matching.len() == 1, "missing or duplicate V4.1 TP2 draft capacity {cap}");
            scratch = scratch.max(matching[0]["core_scratch_nbytes"].as_u64()
                .context("missing V4.1 TP2 draft scratch")?);
        }
        let variants = experts["variants"].as_array().context("missing dSpark expert variants")?;
        let full = variants.iter().find(|v| v.get("requested_rows").unwrap_or(&v["capacity_rows"]).as_u64() == Some(capacity))
            .context("missing full dSpark capacity")?["core_scratch_nbytes"].as_u64()
            .context("missing full dSpark scratch")? + capacity * (10240 + 3 * 8 + 2 * 10240);
        let peer = scratch + capacity * (5280 + 12 + 12);
        let home = scratch + capacity * (5280 + 3 * 5120 * 4 + 10240 + 2 * 10240 + 24);
        for demand in out.iter_mut().filter(|d| d.group == "dSpark request lanes") {
            demand.bytes = if demand.gpu == 0 { demand.bytes + lanes * 3 * peer }
                else { demand.bytes.checked_sub(lanes * 3 * full).context("V4.1 full draft owner underflow")?
                    + lanes * 3 * home };
        }
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
            (6144, 25600), (15360, 5120), (5120, 1152), (1152, 5120)];
        let variants = matrices.into_iter().flat_map(|(k, n)| {
            [1, 16, 64, 80, 256, 1024, 4096].map(move |rows| json!({
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
    fn selected_native_modules_follow_devices_capacities_and_draft_home() -> Result<()> {
        let manifest = manifest();
        let programs = NativePrograms::new(&manifest, 188)?;
        let target = programs.modules(DECODE_LANE_ROWS, 256, 2, None)?;
        assert!(target[0].names.iter().any(|name| name.contains("6144_25600")));
        assert!(!target[1].names.iter().any(|name| name.contains("6144_25600")));
        assert!(target.iter().all(|set| set.names.iter().any(|name| name.contains("5120_1152"))));
        assert!(target.iter().all(|set| !set.names.iter().any(|name| name.contains("_4096/"))));
        let draft = programs.modules(DECODE_LANE_ROWS, 256, 2, Some(80))?;
        assert_eq!(target[0], draft[0]);
        assert!(draft[1].names.iter().any(|name| name.contains("15360_5120")));
        assert!(draft[1].names.len() > target[1].names.len());
        for set in draft {
            assert_eq!(set.names.len(), set.names.iter().collect::<std::collections::BTreeSet<_>>().len());
            assert!(set.scratch.is_empty());
        }
        Ok(())
    }

    #[test]
    fn blas_pools_follow_compressors_and_decoder_markov_not_lanes() -> Result<()> {
        use crate::placement::planner::serving::BlasHead;
        for ranks in [1, 2] {
            for head in [BlasHead::Native, BlasHead::Gemm, BlasHead::TensorLt] {
                let target = blas_owners(ranks, head, false)?;
                assert_eq!(target.len(), ranks);
                assert!(target.iter().all(|owner| owner.lt && owner.gemm == head.gemm()));
                let draft = blas_owners(ranks, head, true)?;
                assert_eq!(draft.len(), ranks + 1);
                assert_eq!(draft.last().unwrap().location, crate::placement::planner::Location::coordinator((ranks - 1) as u8));
                assert!(draft.last().unwrap().gemm);
            }
        }
        Ok(())
    }

    #[test]
    fn packed_push_source_coexists_with_all_load_owners_on_each_gpu() -> Result<()> {
        use crate::placement::planner::{Lifetime, Phase};
        use crate::serving_capacity::WeightArenaLayout;
        assert!(packed_head_load_owners(&[])?.is_empty());
        for ranks in [1, 2] {
            let layouts = (0..ranks).map(|rank| WeightArenaLayout::new(format!("v41/target/gpu{rank}"),
                super::super::weight_layout::head_requests(129280 / ranks, Fp8Head::Packed)?))
                .collect::<Result<Vec<_>>>()?;
            let owners = packed_head_load_owners(&layouts)?;
            assert_eq!(owners.len(), ranks);
            for (rank, owner) in owners.iter().enumerate() {
                assert_eq!(owner.location, crate::placement::planner::Location::coordinator(rank as u8));
                assert_eq!(owner.bytes, (129280 / ranks * 10240) as u64);
                assert_eq!(owner.resident, 0);
                assert_eq!(owner.category, Category::Staging);
                assert_eq!(owner.lifetime, Lifetime { first: Phase::Load, last: Phase::Load });
                assert!(layouts[rank].view("head.bf16").is_none());
            }
        }
        Ok(())
    }

    #[test]
    fn vocabulary_phase_charges_only_the_resident_weights_before_any_lane() -> Result<()> {
        let demand = |group: &str, bytes| Demand::new(0, Category::Workspace, group, bytes, Basis::Formula);
        let weights = vec![demand("attention", 100), demand("engram-projections", 20),
            demand("embedding", 50), demand("lm_head", 300), demand("norm", 10)];
        let target = vec![demand("lane 0 backbone", 40), demand("lane 0 engram", 30),
            demand("lane 0 window batch table", 5), demand("lane 0 target head", 60),
            demand("lane 0 target taps", 70), demand("lane 1 backbone", 80)];
        let _ = (&weights, &target);
        // The whole target arena exists before the head converts; only the
        // mode's transient source sits beside it. Lanes and slots come after.
        let arena = crate::serving_capacity::WeightArenaLayout::new("v41/target/gpu0",
            [("head.fp8.payload".to_string(), 3 << 20), ("norm.weight".to_string(), 10)])?;
        for (head, transient) in [(Fp8Head::All, 16 * 10240), (Fp8Head::Off, 0), (Fp8Head::Draft, 0),
            (Fp8Head::Packed, 129280 * 10240)] {
            let phase = vocabulary_load_phase(&arena, head)?;
            assert_eq!(phase.bytes, arena.bytes() as u64 + transient, "{head:?}");
            assert_eq!(phase.category, Category::Staging);
        }
        for ranks in 1..=cuteafd_core::MAX_SPARK_WORLD {
            assert_eq!(transport_bytes(1024, ranks)?, ranks as u64 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(1024 * 10240)?
                + 2 * 1024 * 10240);
        }
        assert!(transport_bytes(0, 4).is_err());
        assert!(transport_bytes(4097, 4).is_err());
        assert!(transport_bytes(1024, 0).is_err());
        assert!(transport_bytes(1024, cuteafd_core::MAX_SPARK_WORLD + 1).is_err());
        Ok(())
    }

    #[test]
    fn transport_backing_pads_decode_ranges_and_the_whole_prefill_reserve() -> Result<()> {
        for ranks in 1..=cuteafd_core::MAX_SPARK_WORLD {
            for decode_rows in [64, 80] {
                let decode = ranks as u64 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(decode_rows * 10240)?
                    + decode_rows * 20480;
                assert_eq!(transport_bytes(decode_rows, ranks)?, decode);
                let outputs = 3 * 2752 * 10240;
                assert_eq!(transports_bytes(decode_rows, 2752, ranks, 3, cuteafd_core::SparkReplyFormat::Bf16)?,
                    2 * decode + outputs + ranks as u64 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(3 * 2752 * 10240)?);
            }
        }
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
        let _ = (&weights, &target);
        let arena = crate::serving_capacity::WeightArenaLayout::new("v41/target/gpu0",
            [("weights".to_string(), 170), ("head.fp8.payload".to_string(), 682_598_400)])?;
        let vocabulary = vocabulary_load_phase(&arena, Fp8Head::All)?;
        assert_eq!(vocabulary.bytes, arena.bytes() as u64 + 16 * 10240);
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
        assert_eq!(super::super::admission::draft_owners(&draft, 1)?.iter().map(|o| o.bytes).sum::<u64>(), 720);

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
    fn encoder_suffix_admission_scales_with_boot_c_not_global_ceiling() -> Result<()> {
        let manifest = manifest();
        let programs = NativePrograms::new(&manifest, 188)?;
        for ranks in [1, 2] {
            for concurrency in [16, 33, 64] {
                let owners = programs.prefill_slots(8256, 2752, ranks, 3, concurrency,
                    Fp8Head::Off, ranks == 1)?;
                let expected = (2752u64.min(concurrency * 128) + concurrency * 128) * 40976;
                assert_eq!(owner(&owners, 0, "prefill slots encoder suffix"), expected);
                assert!(!owners.iter().any(|d| d.gpu != 0 && d.group.ends_with("encoder suffix")));
                if concurrency == 16 { assert_eq!(expected, 167837696); }
                if concurrency == 64 { assert_eq!(expected, 448441344); }
            }
        }
        for unit in [64, 256, 2048, 2752, 4096] {
            assert_eq!(encoder_suffix_bytes(unit, 16), (unit.min(2048) + 2048) * 40976);
            assert_eq!(packed_suffix_rows(unit, 33), unit.min(33 * 128));
            assert_eq!(packed_suffix_rows(unit, 64), unit.min(64 * 128));
        }
        Ok(())
    }

    #[test]
    fn target_is_two_decode_lanes_and_one_prefill_slot_set() -> Result<()> {
        let manifest = manifest();
        let programs = NativePrograms::new(&manifest, 188)?;
        for ranks in [1, 2] {
            let target = programs.target(80, 8256, 4096, ranks, 4, 16, Fp8Head::Off, ranks == 1)?;
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
            // The packed arena plus one prompt suffix per request slot.
            assert_eq!(owner(&target, 0, "prefill slots encoder suffix"), (2048 + 16 * 128) * (40960 + 16));
            assert!(!target.iter().any(|d| d.gpu != 0 && d.group.ends_with("encoder suffix")));
            let small = programs.prefill_slots(256, 64, ranks, 2, 16, Fp8Head::Off, ranks == 1)?;
            assert_eq!(owner(&small, 0, "prefill slots encoder suffix"), (64 + 16 * 128) * (40960 + 16));
            let one = programs.prefill_slots(256, 64, ranks, 2, 1, Fp8Head::Off, ranks == 1)?;
            assert_eq!(owner(&one, 0, "prefill slots encoder suffix"), (64 + 128) * (40960 + 16));
            assert!(programs.prefill_slots(256, 64, ranks, 2, 0, Fp8Head::Off, ranks == 1).is_err());
            assert!(target.iter().all(|d| d.category == Category::Workspace));
            assert!(!target.iter().any(|d| d.group.starts_with("lane 2")), "decode lanes only");
            // Engram planes live with the gates (layers 1 and 14, GPU0 on two RTX).
            assert!(!target.iter().any(|d| d.gpu != 0 && d.group.contains("engram")));
            // The slot set's rows scale with the reserve, its scratch with the unit.
            let narrow = programs.prefill_slots(8256, 2048, ranks, 4, 16, Fp8Head::Off, ranks == 1)?;
            let short = programs.prefill_slots(4224, 2048, ranks, 4, 16, Fp8Head::Off, ranks == 1)?;
            for rank in 0..ranks as u8 {
                assert!(owner(&narrow, rank, "prefill slots lane") > owner(&short, rank, "prefill slots lane"));
            }
            let fp8 = programs.target(80, 8256, 4096, ranks, 4, 16, Fp8Head::All, ranks == 1)?;
            let packed = programs.target(80, 8256, 4096, ranks, 4, 16, Fp8Head::Packed, ranks == 1)?;
            let draft = programs.target(80, 8256, 4096, ranks, 4, 16, Fp8Head::Draft, ranks == 1)?;
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
        assert!(programs.target(0, 8256, 4096, 1, 2, 16, Fp8Head::All, false).is_err());
        assert!(programs.target(80, 8256, 4097, 2, 2, 16, Fp8Head::All, false).is_err());
        assert!(programs.target(80, 8256, 4096, 3, 2, 16, Fp8Head::All, false).is_err());
        assert!(programs.target(80, 8256, 4096, 2, 2, 16, Fp8Head::All, true).is_err());
        assert!(programs.prefill_slots(2048, 4096, 1, 2, 16, Fp8Head::Off, false).is_err(), "unit wider than the reserve");
        let off = programs.target(80, 8256, 4096, 1, 4, 16, Fp8Head::Off, false)?;
        let on = programs.target(80, 8256, 4096, 1, 4, 16, Fp8Head::Off, true)?;
        assert_eq!(on.iter().map(|d| d.bytes).sum::<u64>() - off.iter().map(|d| d.bytes).sum::<u64>(),
            2 * window_batch_bytes(80)? + 4 * window_batch_bytes(4096)?);
        for rows in [1, 80, 256, 1024, 4096] {
            assert_eq!(window_batch_bytes(rows)?, 40 * (64 + cuteafd_core::V41_MAX_REQUESTS as u64 * 16 + 8 * rows));
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
                - owner(&one, decoder, "dSpark main context"), programs.main_context(DECODE_LANE_ROWS)?);
            assert_eq!(owner(&two, decoder, "dSpark windows")
                - owner(&one, decoder, "dSpark windows"), 3 * 128 * 528);
            let lanes_packed = programs.dspark(&experts, 1024, 2, 5, ranks, Fp8Head::Packed)?;
            for rank in 0..ranks as u8 {
                assert_eq!(owner(&lanes_packed, rank, "dSpark request lanes") - owner(&two, rank, "dSpark request lanes"),
                    2 * programs.packed_vocabulary_scratch(5, VOCAB / ranks as u64)?);
            }
            for requests in [1, 4, 16, 17, 32, 64] {
                for width in [5, 7] {
                    let demands = programs.dspark(&experts, 1024, requests, width, ranks, Fp8Head::All)?;
                    assert!(demands.iter().all(|d| d.category == Category::Drafter));
                    assert_eq!(owner(&demands, decoder, "dSpark windows"),
                        3 * (requests * 128 * 528 + 1024 * 1024 + cuteafd_core::V41_WRITE_DESCRIPTOR_BYTES as u64));
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
    fn tp2_draft_owners_charge_each_stage_fork_join_at_the_covering_capacity() -> Result<()> {
        let fp8 = manifest();
        let programs = NativePrograms::new(&fp8, 188)?;
        let experts = experts();
        let mut halves = json!({"role":"dspark_tp2", "input_format":"fp8_k32", "geometry":{
            "hidden":5120, "intermediate":1152, "kernel_intermediate":1152, "experts":128, "topk":3},
            "variants": ([1, 16, 80, 256].map(|rows| json!({"capacity_rows":rows, "core_scratch_nbytes":rows * 11}))) });
        for requests in [1u64, 2, 16, 32, 64] {
            for width in [5, 7] {
                let lanes = if requests == 1 { 1 } else { 2 };
                let live = requests.div_ceil(lanes) * width;
                let capacity = if live <= 16 { 16 } else if live <= 40 { 40 } else if live <= 80 { 80 } else { 256 };
                let cover = if capacity <= 16 { 16 } else if capacity <= 80 { 80 } else { 256 };
                let full = programs.dspark(&experts, 1024, requests, width, 2, Fp8Head::All)?;
                let split = programs.dspark_tp2(&experts, &halves, 1024, requests, width, Fp8Head::All)?;
                let scratch = cover * 11;
                assert_eq!(owner(&split, 0, "dSpark request lanes") - owner(&full, 0, "dSpark request lanes"),
                    lanes * 3 * (scratch + capacity * 5304));
                assert_eq!(owner(&split, 1, "dSpark request lanes"),
                    owner(&full, 1, "dSpark request lanes") - lanes * 3 * capacity * (7 + 30744)
                        + lanes * 3 * (scratch + capacity * 97464));
                for group in ["dSpark main context", "dSpark windows"] {
                    assert_eq!(owner(&split, 1, group), owner(&full, 1, group));
                }
            }
        }
        halves["variants"].as_array_mut().unwrap().remove(1);
        assert!(programs.dspark_tp2(&experts, &halves, 1024, 1, 5, Fp8Head::All).is_err());
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
