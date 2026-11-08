//! V4's coordinator host tier as contract owners: the retained prefix quota,
//! the pinned step tables, deferred host logits, dense startup staging, a
//! host-placed embedding and the Spark links' pinned storage. `plan --layout`
//! and serve-dsv4 declare the same owners from the same [`V4Host`] facts, so a
//! host shortfall refuses in the one solve, before any weight or KV allocation.
use super::{host_marks_enabled, V4Inputs};
use crate::placement::planner::serving::HostPrefixOwners;
use crate::placement::planner::*;
use cuteafd_core::memory_layout::Category;

/// Host facts both admissions resolve before the solve.
#[derive(Debug, Clone)]
pub struct V4Host {
    /// Host RAM total and available (`MemAvailable`), sampled before admission.
    pub total: u64,
    pub available: u64,
    /// RAM kept free beside every host owner.
    pub headroom: u64,
    /// `--host-cache-bytes`: `None` is automatic, `Some(0)` disables the tier.
    pub cache_bytes: Option<u64>,
    /// Retained prefix entries; zero disables the tier.
    pub prefix_entries: u64,
    /// Rows one prefill call spans: the scoring logits gather.
    pub span_rows: u64,
    pub full_prefill_logits: bool,
    /// A host-placed token embedding's pinned mapped copy; zero on the GPU.
    pub embedding_bytes: u64,
    /// The compiled C128 index width (`compiled_c128_width`).
    pub c128_width: u64,
}

/// Ring geometry both V4 Spark links register (as GLM 5.3's links).
pub const RINGS: cuteafd_core::SparkRings = cuteafd_core::SparkRings {
    max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };

/// Future pinned owners in the coordinator process, summed across its RTX ranks.
/// `units` includes sequence-tail padding and context parity rounding from PoolShape.
#[derive(Debug, Clone, Copy)]
pub struct PinnedTables {
    pub units: u64,
    pub prefill_rows: u64,
    pub decode_rows: u64,
    pub waves: u64,
    pub ranks: u64,
    pub c128_width: u64,
    pub hidden: u64,
    pub topk: u64,
    pub host_embedding: bool,
    /// Compiled context extent before the pool-limited user context is clamped.
    pub context_extent: Option<u64>,
}

impl PinnedTables {
    pub fn named_bytes(&self) -> Result<Vec<(&'static str, u64)>, Refusal> {
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(Refusal::Overflow("V4 pinned table"));
        let add = |a: u64, b: u64| a.checked_add(b).ok_or(Refusal::Overflow("V4 pinned table"));
        if !(self.units > 0 && self.prefill_rows > 0 && self.decode_rows > 0
            && (1..=crate::serving_capacity::prefill_waves::MAX_WAVES).contains(&self.waves)
            && (1..=2).contains(&self.ranks) && self.c128_width > 0 && self.hidden > 0 && self.topk > 0)
        {
            return Err(Refusal::InvalidFact("invalid V4 pinned table geometry".into()));
        }
        let mut tables = Vec::new();
        for (prefill, rows, slots) in [(false, self.decode_rows, 1), (true, self.prefill_rows, self.waves)] {
            let copies = mul(slots, self.ranks)?;
            let ints = mul(rows, 4)?;
            // Match reserve_slot_uploads/step_buffers, with a floor per destination.
            let mut bounded = [ints, mul(rows, 8)?, mul(rows, 8)?, mul(ints, 128)?, ints,
                ints, ints, mul(ints, self.c128_width)?, ints].into_iter()
                .try_fold(0, |sum, bytes| add(sum, bytes.max(256)))?;
            bounded = add(bounded, mul(mul(add(rows, 2)?, 4)?.max(256), 18)?)?;
            tables.push((if prefill { "deepseek_v4/ordered-h2d/prefill-tables" }
                else { "deepseek_v4/ordered-h2d/decode-tables" }, mul(bounded, copies)?));
            let lists = if prefill { 1 } else { rows };
            tables.push((if prefill { "deepseek_v4/c4-page-table/prefill" }
                else { "deepseek_v4/c4-page-table/decode" },
                mul(mul(mul(self.units, lists)?, 4)?.max(256), copies)?));
            if self.host_embedding {
                tables.push((if prefill { "deepseek_v4/ordered-h2d/prefill-embedding" }
                    else { "deepseek_v4/ordered-h2d/decode-embedding" },
                    mul(mul(mul(rows, self.hidden)?, 8)?.max(256), slots)?));
            }
            // One HostAllocation per workspace, not per prefill slot; peer is a 256-byte dummy.
            let router_stride = add(mul(self.topk, 8)?, add(self.hidden, self.hidden / 32)?)?;
            tables.push((if prefill { "deepseek_v4/prefill-router-host" }
                else { "deepseek_v4/decode-router-host" },
                add(mul(rows, router_stride)?, mul(self.ranks - 1, 256)?)?));
        }
        if let Some(extent) = self.context_extent {
            if self.ranks != 2 || extent == 0 {
                return Err(Refusal::InvalidFact("V4 context tables need two RTX ranks and nonzero extent".into()));
            }
            let half = extent.div_ceil(256).div_ceil(2);
            let copies = mul(self.ranks, self.waves)?;
            // Each owner subview is reserved at its exact extent, without the device floor.
            tables.push(("deepseek_v4/ordered-h2d/c4-context-gather", mul(mul(half, 8)?, copies)?));
            let commits = add(self.prefill_rows.div_ceil(256), 1)?.div_ceil(2);
            tables.push(("deepseek_v4/ordered-h2d/c4-context-commit", mul(mul(commits, 96)?.max(256), copies)?));
        }
        tables.iter().try_fold(0, |sum, (_, bytes)| add(sum, *bytes))?;
        Ok(tables)
    }
}

/// The units serve-dsv4's `PoolShape` allocates for `pages` pool pages of
/// `page_rows` tokens: one partial unit per sequence, even under context.
pub fn pool_units(pages: u64, page_rows: u64, sequences: u64, context: bool) -> Result<u64, Refusal> {
    let units = pages.checked_mul(page_rows).map(|tokens| tokens.div_ceil(256))
        .and_then(|units| units.checked_add(sequences)).ok_or(Refusal::Overflow("V4 pool units"))?;
    Ok(if context { units.next_multiple_of(2) } else { units })
}

/// The retained prefix quota and the snapshot geometry it holds.
#[derive(Debug, Clone, Copy)]
pub struct HostPrefix {
    pub owners: HostPrefixOwners,
    pub geometry: Option<cuteafd_core::prefix::HostCacheGeometry>,
}

impl HostPrefix {
    /// Operator notes: mark size and how many conversations the quota holds.
    pub fn notes(&self, context: u64, concurrency: u64) -> Vec<String> {
        let (Some(geometry), quota) = (self.geometry, self.owners.quota_bytes) else { return Vec::new() };
        let chunk = self.owners.chunk_bytes;
        let max = geometry.conversations(quota, context, chunk);
        let short = geometry.conversations(quota, 50_000, chunk);
        let mut notes = vec![format!("host prefix: {} B per mark; {max} conversations at max context, {short} at 50K (two marks each)",
            geometry.mark_bytes)];
        if short < concurrency + 4 {
            notes.push(format!("WARNING: host prefix 50K capacity {short} is below concurrency+4 ({})", concurrency + 4));
        }
        notes
    }
}

/// The retained prefix quota, resolved from the same facts on both sides and
/// before admission: the compiled context, not the solved pool.
pub fn prefix_owners(inputs: &V4Inputs, host: &V4Host, partitioned: bool)
    -> Result<HostPrefix, Refusal> {
    let off = HostPrefix { owners: HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }, geometry: None };
    if host.prefix_entries == 0 || host.cache_bytes == Some(0)
        || !host_marks_enabled(inputs.gpus.len(), partitioned, true) {
        return Ok(off);
    }
    let cache = crate::serving_capacity::deepseek_v4_cache_geometry(&inputs.cfg, inputs.gpus.len(),
        inputs.prefill_rows, inputs.cache_stages)
        .map_err(|error| Refusal::InvalidFact(format!("V4 host prefix geometry: {error}")))?;
    // V4 stores the lead's complete mark once and publishes the backbone prefix
    // to the peer; every rank holds every page.
    let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: cache.logical_unit_rows,
        page_bytes: cache.ranks[0].persistent_unit_bytes, mark_bytes: cache.ranks[0].retained_mark_bytes,
        draft_bytes: 0 };
    let chunk = geometry.chunk_bytes();
    let quota = geometry.resolve(host.cache_bytes, inputs.max_context, inputs.sequences, host.total, chunk)
        .ok_or(Refusal::Overflow("V4 host prefix quota"))?;
    if quota == 0 {
        return Err(Refusal::InvalidFact("host prefix quota resolves to zero; use a positive quota or \
            explicitly disable host retention".into()));
    }
    Ok(HostPrefix { owners: HostPrefixOwners { quota_bytes: quota, chunk_bytes: chunk.min(quota) },
        geometry: Some(geometry) })
}

/// The coordinator host budget: host and pinned memory share physical RAM.
pub fn budget(host: &V4Host) -> Result<Budget, Refusal> {
    if host.total == 0 || host.available > host.total {
        return Err(Refusal::InvalidFact("invalid V4 host memory sample".into()));
    }
    Ok(Budget { id: "coordinator host RAM".into(), locations: vec![Location::HOST, Location::PINNED],
        physical_bytes: host.total, logical_bytes: host.total,
        baseline: Baseline::Sampled { free: host.available, phase: Phase::Load,
            source: "host MemAvailable sampled before admission".into() },
        headroom: host.headroom })
}

/// One owner per page, then one linear piece once every table floor is past.
fn piecewise(maximum: u64, first_linear: u64, value: impl Fn(u64) -> Result<u64, Refusal>)
    -> Result<ByteCurve, Refusal> {
    let mut pieces = Vec::new();
    let linear = first_linear.clamp(1, maximum);
    for pages in 1..linear {
        let bytes = value(pages)?;
        pieces.push(BytePiece { granularity: 1, first: pages, last: pages, bytes_at_first: bytes, bytes_per_page: 0, backings: 1, });
    }
    let at_first = value(linear)?;
    let per_page = if linear < maximum { value(linear + 1)?.checked_sub(at_first)
        .ok_or_else(|| Refusal::InvalidFact("V4 pinned tables shrink with the pool".into()))? } else { 0 };
    pieces.push(BytePiece { granularity: 1, first: linear, last: maximum, bytes_at_first: at_first, bytes_per_page: per_page, backings: 1, });
    Ok(ByteCurve(pieces))
}

fn host_component(id: String, location: Location, bytes: ByteCurve, lifetime: Lifetime,
    category: Category, group: &str, maximum: u64) -> Result<Component, Refusal> {
    let first = bytes.at(1)?;
    let mut component = Component::allocation(Allocation::buffer(id, location, first, lifetime, category, group),
        bytes.clone(), false, maximum);
    component.byte_extent(bytes)?;
    Ok(component)
}

/// Every host and pinned owner of one prepared V4 contract.
#[allow(clippy::too_many_arguments)]
pub fn components(inputs: &V4Inputs, host: &V4Host, prefix: HostPrefixOwners, page_rows: u64,
    partitioned: bool, maximum: u64) -> Result<Vec<Component>, Refusal> {
    let mut components = prefix.allocations("v4.host.prefix", Lifetime { first: Phase::Warmup, last: Phase::Growth })?
        .into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, maximum);
            Component::allocation(owner, bytes, false, maximum)
        }).collect::<Vec<_>>();
    let tables = |pages: u64| -> Result<Vec<(&'static str, u64)>, Refusal> {
        tables(inputs, host, pool_units(pages, page_rows, inputs.sequences, partitioned)?, partitioned)
    };
    // Every floor is past once a prefill page table holds 64 units.
    let first_linear = (1..=maximum.min(64)).find(|&pages|
        pool_units(pages, page_rows, inputs.sequences, partitioned).is_ok_and(|units| units >= 64))
        .unwrap_or(maximum.min(64));
    let names = tables(1)?;
    for (index, (name, _)) in names.iter().enumerate() {
        let curve = piecewise(maximum, first_linear, |pages| Ok(tables(pages)?[index].1))?;
        components.push(host_component(format!("v4.host.{name}"), Location::PINNED, curve,
            Lifetime::TARGET, Category::Staging, name, maximum)?);
    }
    let rows = if host.full_prefill_logits { host.span_rows } else { 64 };
    let logits = cuteafd_core::memory_layout::host_logits_bytes(rows, inputs.prefill_rows,
        inputs.cfg.vocab_size as u64).ok_or(Refusal::Overflow("V4 host logits"))?;
    components.push(host_component("v4.host.logits".into(), Location::HOST, ByteCurve::fixed(logits, maximum),
        Lifetime::TARGET, Category::Workspace, "scoring host logits", maximum)?);
    if host.embedding_bytes > 0 {
        let bytes = ByteCurve::fixed(host.embedding_bytes, maximum);
        components.push(host_component("v4.host.embedding".into(), Location::PINNED, bytes.clone(),
            Lifetime::RESIDENT, Category::Embedding, "host embedding", maximum)?);
    }
    // Dense startup staging and push rings exist only while weights load.
    for (index, item) in inputs.weights.startup.host.iter().enumerate().filter(|(_, item)| item.bytes > 0) {
        components.push(host_component(format!("v4.host.startup.{index}"), Location::PINNED,
            ByteCurve::fixed(item.bytes, maximum), Lifetime { first: Phase::Load, last: Phase::Load },
            Category::Staging, item.name, maximum)?);
    }
    // Both links' pinned and host storage; their GPU intake is a fixed demand.
    if inputs.spark_ranks > 0 {
        for link in links(inputs)? {
            components.extend(link.components(maximum)?.into_iter().filter(|component|
                matches!(&component.kind, ComponentKind::Allowance(allowance)
                    if allowance.location.device == Device::Host)));
        }
    }
    Ok(components)
}

/// The pinned tables serve allocates for `units` pool units.
pub fn tables(inputs: &V4Inputs, host: &V4Host, units: u64, partitioned: bool)
    -> Result<Vec<(&'static str, u64)>, Refusal> {
    PinnedTables { units, prefill_rows: inputs.prefill_rows, decode_rows: inputs.decode_rows,
        waves: inputs.prefill_waves, ranks: inputs.gpus.len() as u64, c128_width: host.c128_width,
        hidden: inputs.cfg.dim as u64, topk: inputs.cfg.n_activated_experts as u64,
        // TokenEmbedding always gathers rows on the device, even from a host table.
        host_embedding: false, context_extent: partitioned.then_some(inputs.max_context) }.named_bytes()
}

/// The decode link and the prefill link serve-dsv4 constructs.
pub fn links(inputs: &V4Inputs) -> Result<Vec<crate::placement::PreparedSparkLink>, Refusal> {
    use crate::placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates};
    let cfg = &inputs.cfg;
    let narrow = |value: usize| u32::try_from(value).map_err(|_| Refusal::Overflow("V4 Spark geometry"));
    let geometry = cuteafd_core::ExpertGeometry { hidden: narrow(cfg.dim)?, experts: narrow(cfg.n_routed_experts)?,
        topk: narrow(cfg.n_activated_experts)?, intermediate: narrow(cfg.moe_inter_dim)?, layers: narrow(cfg.n_layers)? };
    let endpoint = |name: &str, rows: u64, waves: u64, prefill: bool, reply| {
        PreparedSparkLink::endpoint(name, 0, SparkEndpointOwners { ranks: inputs.spark_ranks, geometry,
            endpoint: cuteafd_core::SparkEndpoint { capacity: u32::try_from(rows)
                .map_err(|_| Refusal::Overflow("V4 Spark rows"))?, ingress: cuteafd_core::SparkIngress::Fp8K32,
                ring_depth: None, partial_reply_bytes: None }, rings: RINGS },
            waves as usize, prefill, true, SparkIntakeCandidates::Probed, reply, Lifetime::TARGET)
    };
    Ok(vec![endpoint("v4.decode", inputs.decode_rows, 1, false, cuteafd_core::SparkReplyFormat::Bf16)?,
        endpoint("v4.prefill", inputs.prefill_rows.max(inputs.decode_rows), inputs.prefill_waves, true,
            inputs.spark_reply_format)?])
}

#[cfg(test)]
mod tests {
    use super::PinnedTables;

    fn geometry() -> PinnedTables {
        PinnedTables { units: 8200, prefill_rows: 4096, decode_rows: 64, waves: 2,
            ranks: 2, c128_width: 8192, hidden: 4096, topk: 8,
            host_embedding: false, context_extent: Some(1_048_576) }
    }

    #[test]
    fn named_pinned_tables_match_startup_destinations() {
        for ranks in [1, 2] {
            for waves in [1, 2, 3] {
                for prefill_rows in [512, 2048, 4096] {
                    for decode_rows in [1, 64, 80] {
                        for host_embedding in [false, true] {
                            let g = PinnedTables { ranks, waves, prefill_rows, decode_rows, host_embedding,
                                context_extent: (ranks == 2).then_some(1_048_576), ..geometry() };
                            let named = g.named_bytes().unwrap();
                            let sum: u64 = named.iter().map(|(_, bytes)| bytes).sum();
                            let mut expected = 0;
                            for (rows, slots, lists) in [(decode_rows, 1, decode_rows), (prefill_rows, waves, 1)] {
                                let buffers = [rows*4, rows*8, rows*8, rows*128*4, rows*4,
                                    rows*4, rows*4, rows*g.c128_width*4, rows*4, lists*g.units*4];
                                expected += ranks*slots*(buffers.into_iter().map(|b| b.max(256)).sum::<u64>()
                                    + 18*((rows+2)*4).max(256));
                                expected += rows*(g.topk*8+g.hidden+g.hidden/32)+(ranks-1)*256;
                                if host_embedding { expected += slots*(rows*g.hidden*8).max(256); }
                            }
                            if ranks == 2 {
                                expected += ranks*waves*(2*2048*4 + (96*(prefill_rows.div_ceil(256)+1).div_ceil(2)).max(256));
                            }
                            assert_eq!(sum, expected);
                            assert_eq!(named.iter().find(|(n, _)| *n == "deepseek_v4/c4-page-table/prefill").unwrap().1,
                                ranks*waves*8200*4);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn named_pinned_tables_preserve_small_destination_floors() {
        let g = PinnedTables { units: 1, decode_rows: 1, prefill_rows: 1, c128_width: 1,
            context_extent: Some(1), ..geometry() };
        let tables = g.named_bytes().unwrap();
        let get = |name| tables.iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(get("deepseek_v4/c4-page-table/decode"), 2*256);
        assert_eq!(get("deepseek_v4/c4-page-table/prefill"), 4*256);
        assert_eq!(get("deepseek_v4/ordered-h2d/c4-context-gather"), 4*2*4);
        assert_eq!(get("deepseek_v4/ordered-h2d/c4-context-commit"), 4*256);
    }

    #[test]
    fn named_pinned_tables_refuse_overflow_and_invalid_geometry() {
        for g in [PinnedTables { units: u64::MAX, ..geometry() },
            PinnedTables { hidden: u64::MAX, ..geometry() },
            PinnedTables { c128_width: u64::MAX, ..geometry() },
            PinnedTables { prefill_rows: u64::MAX, ..geometry() },
            PinnedTables { ranks: 0, ..geometry() }, PinnedTables { waves: 0, ..geometry() },
            PinnedTables { ranks: 1, ..geometry() }] {
            assert!(g.named_bytes().is_err());
        }
    }
}
