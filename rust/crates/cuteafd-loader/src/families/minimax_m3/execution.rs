//! Exact MSA program and backing inventory shared by admission and execution.
//! Whole-layer and TP2 heads geometry; neither chooses a placement policy.
use super::MiniMaxM3Config;
use crate::placement::{
    inventory::ProgramEntry,
    planner::{Allocation, ByteCurve, Component, Lifetime, Location, Refusal},
};
use cuteafd_core::memory_layout::Category;
use cuteafd_ffi::{
    minimax_m3::{Layout, Mode},
    programs::ProgramSpec,
};

/// Producer scratch is separate from dense activations and MSA scoring scratch.
/// Every concurrently queued execution owner gets its own instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerPrograms {
    pub layouts: [cuteafd_ffi::minimax_m3::producers::ProducerLayout; 5],
}
impl ProducerPrograms {
    pub fn resolve(manifest: &serde_json::Value, rows: u32) -> Result<(Self, Vec<ProgramSpec>), Refusal> {
        use cuteafd_ffi::minimax_m3::producers::{Kind, validate_producer_entry};
        let entries = manifest["programs"].as_array()
            .ok_or_else(|| Refusal::InvalidFact("MiniMax PROGRAMS.json has no program list".into()))?;
        let mut specs = Vec::new();
        let mut layouts = Vec::new();
        for kind in Kind::ALL {
            let missing = |detail| Refusal::MissingExport { export: kind.name().into(),
                geometry: "MiniMax native norm/partial-NeoX/index/OAI producers".into(), context: 0, detail };
            let mut found = entries.iter().filter(|entry| entry["name"] == kind.name());
            let entry = found.next().ok_or_else(|| missing("absent from PROGRAMS.json".into()))?;
            if found.next().is_some() { return Err(missing("duplicate program".into())); }
            let parsed = ProgramEntry::from_manifest(entry).map_err(|e| missing(e.to_string()))?;
            let layout = validate_producer_entry(&parsed.spec, entry).map_err(|e| missing(e.to_string()))?;
            if rows == 0 || rows > layout.rows {
                return Err(missing(format!("compiled rows={}, requested rows={rows}", layout.rows)));
            }
            layouts.push(layout);
            specs.push(parsed.spec);
        }
        Ok((Self { layouts: layouts.try_into().expect("five producers") }, specs))
    }
    pub fn workspace(&self) -> Vec<(&'static str, u64)> {
        self.layouts.iter().filter(|layout| layout.scratch() != 0)
            .map(|layout| (layout.kind.name(), layout.scratch())).collect()
    }
    pub fn workspace_components(&self, prefix: &str, rank: u8, lifetime: Lifetime,
        maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        components(self.workspace(), prefix, rank, lifetime, maximum_pages, Category::Workspace)
    }
}

/// Stateless forward kernels add no private scratch owner. Their outputs live
/// in the lane/unit activation owners; gather reuses the MSA prefill staging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardPrograms {
    pub layouts: [cuteafd_ffi::minimax_m3::forward::ForwardLayout; 3],
    pub gather: cuteafd_ffi::index_cache::Layout,
}
impl ForwardPrograms {
    pub fn resolve(manifest: &serde_json::Value, rows: u32) -> Result<(Self, Vec<ProgramSpec>), Refusal> {
        use cuteafd_ffi::minimax_m3::forward::{Kind, validate_entry};
        let entries = manifest["programs"].as_array()
            .ok_or_else(|| Refusal::InvalidFact("MiniMax PROGRAMS.json has no program list".into()))?;
        let mut specs = Vec::new();
        let mut layouts = Vec::new();
        let mut gather = None;
        for name in Kind::ALL.map(Kind::name).into_iter().chain(["minimax_index_cache_gather"]) {
            let missing = |detail| Refusal::MissingExport { export: name.into(),
                geometry: "MiniMax residual norm/FP32 router/paged KV and bounded index staging".into(), context: 0, detail };
            let mut found = entries.iter().filter(|entry| entry["name"] == name);
            let entry = found.next().ok_or_else(|| missing("absent from PROGRAMS.json".into()))?;
            if found.next().is_some() { return Err(missing("duplicate program".into())); }
            let parsed = ProgramEntry::from_manifest(entry).map_err(|e| missing(e.to_string()))?;
            let capacity = if name == "minimax_index_cache_gather" {
                let layout = cuteafd_ffi::index_cache::validate_entry(&parsed.spec, entry)
                    .map_err(|e| missing(e.to_string()))?;
                gather = Some(layout);
                layout.rows as u32
            } else {
                let layout = validate_entry(&parsed.spec, entry).map_err(|e| missing(e.to_string()))?;
                layouts.push(layout);
                layout.rows
            };
            if rows == 0 || rows > capacity {
                return Err(missing(format!("compiled rows={capacity}, requested rows={rows}")));
            }
            specs.push(parsed.spec);
        }
        Ok((Self { layouts: layouts.try_into().expect("three forward kernels"),
            gather: gather.expect("bounded gather") }, specs))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MsaPrograms {
    pub layouts: [Layout; 4],
}
impl MsaPrograms {
    pub fn resolve(
        cfg: &MiniMaxM3Config,
        manifest: &serde_json::Value,
        decode_rows: u32,
        prefill_rows: u32,
        max_context: u32,
    ) -> Result<(Self, Vec<ProgramSpec>), Refusal> {
        let bad = |message: &str| Refusal::InvalidFact(message.into());
        if (
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.msa.index_heads,
            cfg.msa.index_dim,
            cfg.msa.block,
            cfg.msa.topk_blocks,
            cfg.msa.local_blocks,
        ) != (64, 4, 128, 4, 128, 128, 16, 1)
            || cfg.sparse.iter().any(|enabled| !enabled)
        {
            return Err(bad("MiniMax native MSA requires 64Q/4KV/D128, independent 4x128 index heads, top16/block128/local1 on every layer"));
        }
        if max_context == 0
            || max_context as usize > cfg.max_position
            || decode_rows == 0
            || prefill_rows == 0
            || prefill_rows > 4096
        {
            return Err(bad("MiniMax MSA requested row or context capacity"));
        }
        let expected = [
            (Mode::IndexDecode, decode_rows, max_context.div_ceil(64)),
            (Mode::IndexPrefill, prefill_rows, max_context.div_ceil(64)),
            (Mode::AttentionDecode, decode_rows, decode_rows),
            (Mode::AttentionExtend, prefill_rows, decode_rows),
        ];
        let entries = manifest["programs"]
            .as_array()
            .ok_or_else(|| bad("MiniMax PROGRAMS.json has no program list"))?;
        let mut specs = Vec::new();
        let mut layouts = Vec::new();
        for (mode, rows, extent) in expected {
            let refusal = |detail| Refusal::MissingExport {
                export: mode.name().into(),
                geometry: "MiniMax native MSA 64Q/4KV/D128, top16/block128".into(),
                context: u64::from(max_context),
                detail,
            };
            let mut found = entries.iter().filter(|entry| entry["name"] == mode.name());
            let entry = found
                .next()
                .ok_or_else(|| refusal("absent from PROGRAMS.json".into()))?;
            if found.next().is_some() {
                return Err(refusal("duplicate program".into()));
            }
            let parsed = ProgramEntry::from_manifest(entry).map_err(|e| refusal(e.to_string()))?;
            let layout = cuteafd_ffi::minimax_m3::validate_entry(&parsed.spec, entry)
                .map_err(|e| refusal(e.to_string()))?;
            if layout.rows < rows || layout.extent < extent {
                return Err(refusal(format!(
                    "compiled capacities rows={} extent={}, requested rows={rows} extent={extent}",
                    layout.rows, layout.extent
                )));
            }
            // The compiled selection head stride is layout.rows, even for a
            // smaller live row count. Owners use that stride, not the request.
            layouts.push(layout);
            specs.push(parsed.spec);
        }
        if layouts[0].rows != layouts[2].rows || layouts[1].rows != layouts[3].rows {
            return Err(bad(
                "MiniMax index/attention compiled selection strides differ",
            ));
        }
        Ok((
            Self {
                layouts: layouts.try_into().expect("four MSA modes"),
            },
            specs,
        ))
    }

    /// One serial execution owner. Independent lanes/prefill slots call this
    /// separately; scratch is never shared between simultaneously queued owners.
    pub fn workspace(&self, prefill: bool) -> Vec<(&'static str, u64)> {
        let [index, attention] = if prefill {
            [self.layouts[1], self.layouts[3]]
        } else {
            [self.layouts[0], self.layouts[2]]
        };
        let rows = u64::from(index.rows);
        let batch = u64::from(attention.extent);
        let mut buffers = vec![
            ("index.query", rows * 512),
            ("index.scales", rows * 16),
            ("index.produced_keys", rows * 128),
            ("index.produced_scales", rows * 4),
            ("selections", rows * 256),
            ("index.scratch", index.scratch),
            ("attention.query", rows * 16384),
            ("attention.key", rows * 1024),
            ("attention.output", rows * 16384),
            ("attention.lse", rows * 256),
            ("attention.scratch", attention.scratch),
            ("attention.k_descale", batch * 16),
            ("attention.v_descale", batch * 16),
            ("attention.query_offsets", (batch + 1) * 4),
        ];
        if prefill {
            // Contiguous scorer reads a 256-row padded K extent, including
            // request-alignment gaps. This is separate from the paged KV pool.
            let tokens = (u64::from(index.extent) * 64).div_ceil(256) * 256;
            buffers.extend([
                ("index.keys", tokens * 128),
                ("index.key_scales", tokens * 4),
                ("index.starts", rows * 4),
                ("index.ends", rows * 4),
            ]);
        }
        buffers
    }

    pub fn workspace_components(
        &self,
        prefix: &str,
        rank: u8,
        prefill: bool,
        lifetime: Lifetime,
        maximum_pages: u64,
    ) -> Result<Vec<Component>, Refusal> {
        components(
            self.workspace(prefill),
            prefix,
            rank,
            lifetime,
            maximum_pages,
            Category::Workspace,
        )
    }
}

/// Retained arithmetic outputs for one in-flight shared prefill unit. Dense
/// quantization and MSA scratch are separate owners, not repeated charges here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitLayout { pub rows: u32 }
impl UnitLayout {
    pub fn new(cfg: &MiniMaxM3Config, rows: u32) -> Result<Self, Refusal> {
        if rows == 0 || rows > 4096 || (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim,
            cfg.dense_intermediate, cfg.shared_intermediate, cfg.experts, cfg.topk)
                != (6144, 64, 4, 128, 12288, 3072, 128, 4)
            || cfg.routed_scale != 2.0 || cfg.rms_norm_eps != 1e-6
            || cfg.swiglu_alpha != 1.702 || cfg.swiglu_limit != 7.0 {
            return Err(Refusal::InvalidFact("MiniMax native unit arithmetic geometry".into()));
        }
        Ok(Self { rows })
    }
    pub fn buffers(self) -> Vec<(&'static str, u64)> {
        let rows = u64::from(self.rows);
        [("residual", 6144 * 4), ("query", 8192 * 2), ("key", 512 * 2),
            ("value", 512 * 2), ("index_query", 512 * 2), ("index_key", 128 * 2),
            ("gate", 12288 * 2), ("up", 12288 * 2), ("ffn", 6144 * 2),
            ("routed", 6144 * 2), ("combined", 6144 * 2), ("logits", 128 * 4), ("route_ids", 4 * 4),
            ("route_weights", 4 * 4)].into_iter().map(|(name, stride)| (name, rows * stride)).collect()
    }
    pub fn components(self, prefix: &str, rank: u8, lifetime: Lifetime,
        maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if self.rows == 0 || self.rows > 4096 {
            return Err(Refusal::InvalidFact("MiniMax native unit row capacity".into()));
        }
        components(self.buffers(), prefix, rank, lifetime, maximum_pages, Category::Workspace)
    }
}

/// One TP2 rank executes every layer, owning two paired KV/index groups and
/// their 32 query heads. Shared index keys and the residual remain replicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadsLayout {
    rank: u8,
    rows: u32,
    layers: u32,
}
impl HeadsLayout {
    pub fn new(cfg: &MiniMaxM3Config, rank: u8, rows: u32) -> Result<Self, Refusal> {
        cfg.release_geometry().map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        UnitLayout::new(cfg, rows)?;
        if rank > 1 {
            return Err(Refusal::InvalidFact("MiniMax heads requires TP2 rank 0 or 1".into()));
        }
        Ok(Self { rank, rows, layers: cfg.layers as u32 })
    }
    pub fn rank(self) -> u8 { self.rank }
    pub fn rows(self) -> u32 { self.rows }
    pub fn layers(self) -> u32 { self.layers }
    pub fn query_head_offset(self) -> u32 { u32::from(self.rank) * self.query_heads() }
    pub fn query_heads(self) -> u32 { 32 }
    pub fn kv_head_offset(self) -> u32 { u32::from(self.rank) * self.kv_heads() }
    pub fn kv_heads(self) -> u32 { 2 }
    // Checkpoint index head h selects the blocks consumed by KV group h.
    pub fn index_head_offset(self) -> u32 { self.kv_head_offset() }
    pub fn index_heads(self) -> u32 { self.kv_heads() }
    pub fn dense_neuron_offset(self) -> u32 { u32::from(self.rank) * self.dense_neurons() }
    pub fn dense_neurons(self) -> u32 { 6144 }
    pub fn shared_neuron_offset(self) -> u32 { u32::from(self.rank) * self.shared_neurons() }
    pub fn shared_neurons(self) -> u32 { 1536 }

    /// Unit arithmetic only; compiled MSA/producer/dense scratch is separate.
    /// One FP32 local partial serves both reductions. Native BF16 routed outputs
    /// are promoted before the fixed-order FFN fold.
    pub fn buffers(self) -> Vec<(&'static str, u64)> {
        let rows = u64::from(self.rows);
        [("residual", 6144 * 4), ("query", 4096 * 2), ("key", 256 * 2),
            ("value", 256 * 2), ("index_query", 256 * 2), ("index_key", 128 * 2),
            ("gate", 6144 * 2), ("up", 6144 * 2), ("ffn", 6144 * 2),
            ("routed", 6144 * 2), ("reduction", 6144 * 4), ("logits", 128 * 4),
            ("route_ids", 4 * 4), ("route_weights", 4 * 4)]
            .into_iter().map(|(name, stride)| (name, rows * stride)).collect()
    }
    pub fn components(self, prefix: &str, lifetime: Lifetime,
        maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        components(self.buffers(), prefix, self.rank, lifetime, maximum_pages, Category::Workspace)
    }

    /// One exchange per rank, not per layer or reduction. Independent lanes and
    /// units supply separate receive slots, each reused for O and FFN. Includes
    /// the abort word allocated by PeerExchange::new_abortable_sized.
    pub fn peer_buffers(slot_rows: &[u32], diagnostics: bool) -> Result<Vec<(String, u64)>, Refusal> {
        let mut out = Vec::new();
        for (slot, &rows) in slot_rows.iter().enumerate() {
            UnitMetadata::new(rows)?;
            out.push((format!("recv.{slot}"), u64::from(rows) * 6144 * 4));
        }
        let largest = out.iter().map(|(_, bytes)| *bytes).max()
            .ok_or_else(|| Refusal::InvalidFact("MiniMax heads peer exchange has no slots".into()))?;
        let storage = crate::serving_capacity::PeerStorageLayout::new(
            slot_rows.len(), largest as usize, diagnostics)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        out.push(("control".into(), storage.control_bytes as u64));
        out.push(("abort".into(), 256));
        if storage.progress_bytes != 0 {
            out.push(("progress".into(), storage.progress_bytes as u64));
        }
        Ok(out)
    }
    pub fn peer_components(rank: u8, slot_rows: &[u32], diagnostics: bool,
        prefix: &str, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        components(Self::peer_buffers(slot_rows, diagnostics)?, prefix, rank,
            Lifetime::TARGET, maximum_pages, Category::Transport)
    }
}

/// Embedding/final-norm and shared vocabulary-head storage for bounded scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadLayout { pub rows: u32, pub vocab: u32 }
impl HeadLayout {
    pub fn new(cfg: &MiniMaxM3Config, rows: u32) -> Result<Self, Refusal> {
        if rows == 0 || rows > 4096 || cfg.hidden != 6144 || cfg.vocab_size != 200064 {
            return Err(Refusal::InvalidFact("MiniMax target head geometry".into()));
        }
        Ok(Self { rows, vocab: cfg.vocab_size as u32 })
    }
    pub fn buffers(self) -> Vec<(&'static str, u64)> {
        let rows = u64::from(self.rows);
        vec![("tokens", rows * 4), ("normalized", rows * 6144 * 2),
            ("logits", rows * u64::from(self.vocab) * 4),
            ("workspace", cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64)]
    }
    pub fn components(self, prefix: &str, rank: u8, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        components(self.buffers(), prefix, rank, Lifetime::TARGET, maximum_pages, Category::Workspace)
    }
}

/// One immutable unit's request metadata. Shared page tables belong to MsaCache;
/// positions/causal extents differ by unit and must not alias a sibling slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitMetadata { pub rows: u32, pub requests: u32 }
impl UnitMetadata {
    pub fn new(rows: u32) -> Result<Self, Refusal> {
        Self::for_requests(rows, 1)
    }
    pub fn for_requests(rows: u32, requests: u32) -> Result<Self, Refusal> {
        if rows == 0 || rows > 4096 || requests == 0 || requests > rows {
            return Err(Refusal::InvalidFact("MiniMax metadata row/request capacity".into()));
        }
        Ok(Self { rows, requests })
    }
    /// Row cursor shared by the four rowwise metadata planes. Each segment's
    /// i32 and i64 views retain the AOT ABI's 16-byte base alignment.
    pub fn segment_row(self, cursor: u32) -> Result<u32, Refusal> {
        let row = cursor.checked_add(3).ok_or(Refusal::Overflow("MiniMax metadata row"))? / 4 * 4;
        if row >= self.metadata_rows() {
            return Err(Refusal::InvalidFact("MiniMax metadata segment row exceeds owner".into()));
        }
        Ok(row)
    }
    pub fn metadata_rows(self) -> u32 {
        if self.requests == 1 { self.rows }
        else { (self.rows + 3 * (self.requests - 1)).div_ceil(4) * 4 }
    }
    /// Per-segment length, local [0, rows] offsets and descale block base.
    pub fn segment_offset(self, segment: u32) -> Result<u64, Refusal> {
        if segment >= self.requests {
            return Err(Refusal::InvalidFact("MiniMax metadata segment exceeds owner".into()));
        }
        Ok(u64::from(segment) * 16)
    }
    pub fn buffers(self) -> Vec<(&'static str, u64)> {
        let rows = u64::from(self.metadata_rows());
        let requests = u64::from(self.requests);
        let (lengths, offsets) = if self.requests == 1 { (4, 8) }
            else { (requests * 16, requests * 16) };
        vec![("positions", rows * 8), ("slots", rows * 8), ("batch_ids", rows * 4),
            ("causal_lengths", rows * 4), ("cache_lengths", lengths),
            ("query_offsets", offsets),
            ("key_descale", requests * 16), ("value_descale", requests * 16)]
    }
    pub fn components(self, prefix: &str, rank: u8, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        Self::for_requests(self.rows, self.requests)?;
        components(self.buffers(), prefix, rank, Lifetime::TARGET, maximum_pages, Category::Workspace)
    }
}

/// Native producers read one interleaved [positions, cos32 + sin32] FP32 table.
/// This is intentionally not the shared two-plane BF16 rotary representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsaRope { pub positions: u32 }
impl MsaRope {
    pub fn new(cfg: &MiniMaxM3Config, positions: u32) -> Result<Self, Refusal> {
        if positions == 0 || u64::from(positions) > cfg.max_position as u64
            || cfg.rotary_dim != 64 || cfg.rope_theta != 10_000_000.0 {
            return Err(Refusal::InvalidFact("MiniMax FP32 partial rotary geometry".into()));
        }
        Ok(Self { positions })
    }
    pub fn buffers(self) -> Vec<(&'static str, u64)> {
        vec![("cos_sin", u64::from(self.positions) * 64 * 4)]
    }
    pub fn components(self, prefix: &str, rank: u8, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        components(self.buffers(), prefix, rank, Lifetime::TARGET, maximum_pages, Category::Workspace)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsaCache {
    pub layers: u32,
    pub pool_pages: u32,
    pub requests: u32,
    pub table_width: u32,
    kv_heads: u32,
}
impl MsaCache {
    pub fn new(
        layers: u32,
        pool_pages: u32,
        requests: u32,
        max_context: u32,
    ) -> Result<Self, Refusal> {
        if [layers, pool_pages, requests, max_context].contains(&0)
            || [pool_pages, requests, max_context]
                .iter()
                .any(|&n| n > i32::MAX as u32)
        {
            return Err(Refusal::InvalidFact(
                "MiniMax MSA cache needs positive signed-int32 extents".into(),
            ));
        }
        Ok(Self {
            layers,
            pool_pages,
            requests,
            table_width: max_context.div_ceil(64).div_ceil(4) * 4,
            kv_heads: 4,
        })
    }
    pub fn for_heads(layout: &HeadsLayout, pool_pages: u32, requests: u32,
        max_context: u32) -> Result<Self, Refusal> {
        let mut cache = Self::new(layout.layers(), pool_pages, requests, max_context)?;
        cache.kv_heads = layout.kv_heads();
        Ok(cache)
    }
    pub fn kv_heads(self) -> u32 { self.kv_heads }
    pub fn buffers(self) -> Result<Vec<(String, u64)>, Refusal> {
        let mut out = Vec::new();
        for layer in 0..self.layers {
            let kv_stride = u64::from(self.kv_heads) * 64 * 128;
            for (name, stride) in [("keys", kv_stride), ("values", kv_stride), ("index", 8448)] {
                let bytes = u64::from(self.pool_pages)
                    .checked_mul(stride)
                    .filter(|&n| n <= isize::MAX as u64)
                    .ok_or(Refusal::Overflow("MiniMax MSA cache"))?;
                out.push((format!("layer.{layer}.{name}"), bytes));
            }
        }
        for name in ["page_table", "decode_table"] {
            out.push((name.into(), u64::from(self.requests) * u64::from(self.table_width) * 4));
        }
        out.push(("lengths".into(), u64::from(self.requests) * 4));
        Ok(out)
    }
    pub fn components(
        self,
        prefix: &str,
        rank: u8,
        maximum_pages: u64,
    ) -> Result<Vec<Component>, Refusal> {
        components(
            self.buffers()?,
            prefix,
            rank,
            Lifetime::TARGET,
            maximum_pages,
            Category::Kv,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    fn manifest() -> Value {
        serde_json::from_str(include_str!(
            "../../../../cuteafd-ffi/tests/fixtures/minimax_msa_programs.json"
        ))
        .unwrap()
    }
    fn cfg() -> MiniMaxM3Config {
        MiniMaxM3Config::from_hf(
            &serde_json::from_str(include_str!("fixtures/config.json")).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn heads_layout_partitions_paired_queries_index_groups_and_ffn_neurons() {
        let cfg = cfg();
        let ranks = [HeadsLayout::new(&cfg, 0, 17).unwrap(), HeadsLayout::new(&cfg, 1, 17).unwrap()];
        for (rank, layout) in ranks.into_iter().enumerate() {
            assert_eq!((layout.rank(), layout.rows(), layout.layers()), (rank as u8, 17, 60));
            assert_eq!((layout.query_heads(), layout.kv_heads(), layout.index_heads()), (32, 2, 2));
            assert_eq!(layout.index_head_offset(), layout.kv_head_offset());
            for local_kv in 0..layout.kv_heads() {
                let global_kv = layout.kv_head_offset() + local_kv;
                for local_query in local_kv * 16..(local_kv + 1) * 16 {
                    assert_eq!((layout.query_head_offset() + local_query) / 16, global_kv);
                }
            }
        }
        let ranges = |start: fn(HeadsLayout) -> u32, count: fn(HeadsLayout) -> u32| {
            ranks.into_iter().flat_map(|layout| start(layout)..start(layout) + count(layout)).collect::<Vec<_>>()
        };
        assert_eq!(ranges(HeadsLayout::query_head_offset, HeadsLayout::query_heads), (0..64).collect::<Vec<_>>());
        assert_eq!(ranges(HeadsLayout::kv_head_offset, HeadsLayout::kv_heads), (0..4).collect::<Vec<_>>());
        assert_eq!(ranges(HeadsLayout::index_head_offset, HeadsLayout::index_heads), (0..4).collect::<Vec<_>>());
        assert_eq!(ranges(HeadsLayout::dense_neuron_offset, HeadsLayout::dense_neurons), (0..12288).collect::<Vec<_>>());
        assert_eq!(ranges(HeadsLayout::shared_neuron_offset, HeadsLayout::shared_neurons), (0..3072).collect::<Vec<_>>());
        for (rank, rows) in [(2, 1), (0, 0), (1, 4097)] {
            assert!(HeadsLayout::new(&cfg, rank, rows).is_err());
        }
        let mut wrong = cfg.clone(); wrong.msa.index_heads = 2;
        assert!(HeadsLayout::new(&wrong, 0, 1).is_err());
    }

    #[test]
    fn heads_cache_halves_only_kv_and_retains_all_layers_and_one_shared_index() {
        for rank in 0..2 {
            let layout = HeadsLayout::new(&cfg(), rank, 64).unwrap();
            for context in [63, 64, 65, 270, 1048576] {
                let whole = MsaCache::new(60, 17, 16, context).unwrap();
                let local = MsaCache::for_heads(&layout, 17, 16, context).unwrap();
                assert_eq!((whole.kv_heads(), local.kv_heads()), (4, 2));
                assert_eq!(local.table_width, whole.table_width);
                let whole = whole.buffers().unwrap().into_iter().collect::<std::collections::BTreeMap<_, _>>();
                let local = local.buffers().unwrap().into_iter().collect::<std::collections::BTreeMap<_, _>>();
                assert_eq!(whole.len(), local.len());
                for (name, bytes) in &whole {
                    let split_kv = name.ends_with(".keys") || name.ends_with(".values");
                    assert_eq!(local[name], if split_kv { bytes / 2 } else { *bytes }, "{name}");
                }
                assert_eq!(local["layer.59.keys"], 17 * 16384);
                assert_eq!(local["layer.59.values"], 17 * 16384);
                assert_eq!(local["layer.59.index"], 17 * 8448);
            }
        }
    }

    #[test]
    fn heads_unit_inventory_has_one_fp32_partial_and_exact_allocation_tails() {
        use crate::placement::planner::ComponentKind;
        for rows in [1, 64, 4096] {
            let layout = HeadsLayout::new(&cfg(), 1, rows).unwrap();
            let buffers = layout.buffers().into_iter().collect::<std::collections::BTreeMap<_, _>>();
            let rows = u64::from(rows);
            for (name, stride) in [("query", 8192), ("key", 512), ("value", 512),
                ("index_query", 512), ("index_key", 256), ("gate", 12288), ("up", 12288),
                ("residual", 24576), ("reduction", 24576), ("routed", 12288), ("logits", 512)] {
                assert_eq!(buffers[name], rows * stride, "{name}");
            }
            assert!(!buffers.contains_key("combined"));
            let components = layout.components("minimax.heads.unit", Lifetime::TARGET, 17).unwrap();
            assert_eq!(components.len(), buffers.len() + buffers.values().filter(|&&bytes| bytes > 1 << 20).count());
            for (name, bytes) in &buffers {
                let id = format!("minimax.heads.unit.{name}");
                assert_eq!(components.iter().filter(|component| matches!(&component.kind,
                    ComponentKind::Allocation { template, .. } if template.id == id
                        && template.bytes == *bytes && template.location == Location::coordinator(1)
                        && template.view.is_none())).count(), 1);
            }
        }
    }

    #[test]
    fn heads_peer_slots_are_fp32_reused_by_both_reductions_and_count_flags_once() {
        for diagnostics in [false, true] {
            let rows = [64, 2048, 4096];
            let buffers = HeadsLayout::peer_buffers(&rows, diagnostics).unwrap()
                .into_iter().collect::<std::collections::BTreeMap<_, _>>();
            let storage = crate::serving_capacity::PeerStorageLayout::new(3, 4096 * 6144 * 4, diagnostics).unwrap();
            for (slot, rows) in rows.into_iter().enumerate() {
                assert_eq!(buffers[&format!("recv.{slot}")], u64::from(rows) * 6144 * 4);
            }
            assert_eq!(buffers["control"], storage.control_bytes as u64);
            assert_eq!(buffers["abort"], 256);
            assert_eq!(buffers.get("progress").copied().unwrap_or(0), storage.progress_bytes as u64);
            assert_eq!(buffers.len(), 5 + usize::from(diagnostics));
            for rank in 0..2 {
                assert!(HeadsLayout::peer_components(rank, &rows, diagnostics, "minimax.heads.peer", 17).is_ok());
            }
        }
        for rows in [vec![], vec![0], vec![4097]] {
            assert!(HeadsLayout::peer_buffers(&rows, false).is_err());
        }
        assert!(HeadsLayout::peer_components(2, &[1], false, "minimax.heads.peer", 17).is_err());
    }

    #[test]
    fn decode_table_has_a_distinct_admitted_backing_without_reordering_canonical_rows() {
        for requests in [1, 3, 16, 64] {
            for context in [270, 1048576] {
                let cache = MsaCache::new(1, 5, requests, context).unwrap();
                let buffers = cache.buffers().unwrap().into_iter().collect::<std::collections::BTreeMap<_, _>>();
                let bytes = u64::from(requests) * u64::from(cache.table_width) * 4;
                assert_eq!(buffers["page_table"], bytes);
                assert_eq!(buffers["decode_table"], bytes);
                let mut components = cache.components("minimax.cache.0", 0, 5).unwrap();
                use crate::placement::planner::{ComponentKind, fixed_request_tails};
                assert!(components.iter().all(|component| matches!(component.kind, ComponentKind::Allocation { .. })));
                fixed_request_tails(&mut components, 5, &[]).unwrap();
                for name in ["page_table", "decode_table"] {
                    let id = format!("minimax.cache.0.{name}");
                    assert_eq!(components.iter().filter(|component| matches!(&component.kind,
                        ComponentKind::Allocation { template, .. } if template.id == id
                            && template.bytes == bytes && template.view.is_none())).count(), 1);
                    // Small allocations share driver blocks, so only large owners have a tail.
                    assert_eq!(components.iter().filter(|component| matches!(&component.kind,
                        ComponentKind::AllocationGranularity(tail) if tail.id == format!("{id}.tail"))).count(),
                        usize::from(bytes > 1 << 20));
                }
            }
        }
    }

    #[test]
    fn packed_metadata_keeps_segment_views_aligned_and_decode_contiguous() {
        assert_eq!(UnitMetadata::new(17).unwrap().buffers(), vec![("positions", 136),
            ("slots", 136), ("batch_ids", 68), ("causal_lengths", 68),
            ("cache_lengths", 4), ("query_offsets", 8), ("key_descale", 16), ("value_descale", 16)]);
        for requests in 2..=64 {
            let layout = UnitMetadata::for_requests(64, requests).unwrap();
            let buffers = layout.buffers().into_iter().collect::<std::collections::BTreeMap<_, _>>();
            let mut cursor = 0;
            for segment in 0..requests {
                let row = layout.segment_row(cursor).unwrap();
                assert_eq!((row * 4) % 16, 0);
                assert_eq!((row * 8) % 16, 0);
                cursor = row + if segment + 1 == requests { 65 - requests } else { 1 };
                let block = layout.segment_offset(segment).unwrap();
                assert_eq!(block % 16, 0);
                assert!(block + 16 <= buffers["query_offsets"]);
                assert!(block + 16 <= buffers["cache_lengths"]);
            }
            assert!(cursor <= layout.metadata_rows());
            assert!(buffers["query_offsets"] >= u64::from(requests + 1) * 4);
            assert!(buffers["cache_lengths"] >= u64::from(requests) * 4);
            assert_eq!(buffers["positions"], u64::from(layout.metadata_rows()) * 8);
            assert_eq!(buffers["key_descale"], u64::from(requests) * 16);
            assert!(layout.segment_offset(requests).is_err());
            assert!(layout.segment_row(layout.metadata_rows()).is_err());
        }
        for (rows, requests) in [(0, 1), (4097, 1), (64, 0), (64, 65)] {
            assert!(UnitMetadata::for_requests(rows, requests).is_err());
        }
    }
    #[test]
    fn head_inventory_has_one_owner_per_native_allocation() {
        let config = cfg();
        let head = HeadLayout::new(&config, 1024).unwrap();
        assert_eq!(head.buffers(), vec![("tokens", 4096), ("normalized", 1024 * 6144 * 2),
            ("logits", 1024 * 200064 * 4),
            ("workspace", cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64)]);
        assert!(HeadLayout::new(&config, 0).is_err());
        assert!(HeadLayout::new(&config, 4097).is_err());
        let mut wrong = config.clone(); wrong.vocab_size -= 1;
        assert!(HeadLayout::new(&wrong, 1024).is_err());
    }

    #[test]
    fn forward_inventory_validates_all_kernels_without_private_scratch() {
        let manifest: Value = serde_json::from_str(include_str!(
            "../../../../cuteafd-ffi/tests/fixtures/minimax_forward_programs.json")).unwrap();
        let (programs, specs) = ForwardPrograms::resolve(&manifest, 2048).unwrap();
        assert_eq!(specs.len(), 4);
        assert_eq!(programs.gather.rows, 4096);
        assert!(specs.iter().all(|spec| spec.scratch.is_empty()));
        assert!(ForwardPrograms::resolve(&manifest, 0).is_err());
        assert!(ForwardPrograms::resolve(&manifest, 4097).is_err());
        for i in 0..4 {
            let mut bad = manifest.clone(); bad["programs"].as_array_mut().unwrap().remove(i);
            assert!(ForwardPrograms::resolve(&bad, 1).is_err());
            let mut bad = manifest.clone(); bad["programs"].as_array_mut().unwrap().push(manifest["programs"][i].clone());
            assert!(ForwardPrograms::resolve(&bad, 1).is_err());
            for field in ["dtype", "shape", "role"] {
                let mut bad = manifest.clone(); bad["programs"][i]["pointers"][0][field] = json!("wrong");
                assert!(ForwardPrograms::resolve(&bad, 1).is_err());
            }
        }
    }
    #[test]
    fn producer_inventory_charges_each_scratch_owner_once() {
        let manifest: Value = serde_json::from_str(include_str!(
            "../../../../cuteafd-ffi/tests/fixtures/minimax_producer_programs.json")).unwrap();
        let (programs, specs) = ProducerPrograms::resolve(&manifest, 2048).unwrap();
        assert_eq!(specs.len(), 5);
        assert_eq!(programs.workspace(), vec![
            ("minimax_norm_rope_h64", 4096 * 64 * 256),
            ("minimax_norm_rope_h4", 4096 * 4 * 256),
            ("minimax_index_producer", 4096 * 2304),
        ]);
        let mut components = programs.workspace_components("minimax.producer.0", 0, Lifetime::TARGET, 16).unwrap();
        use crate::placement::planner::{ComponentKind, fixed_request_tails};
        assert!(components.iter().all(|component| matches!(component.kind, ComponentKind::Allocation { .. })));
        fixed_request_tails(&mut components, 16, &[]).unwrap();
        assert!(components.iter().any(|c| matches!(&c.kind,
            ComponentKind::AllocationGranularity(tail)
                if tail.id == "minimax.producer.0.minimax_index_producer.tail")));
        assert_eq!(components.iter().filter(|c| matches!(&c.kind,
            ComponentKind::Allocation { .. })).count(), 3);
        assert!(ProducerPrograms::resolve(&manifest, 4097).is_err());
        assert!(ProducerPrograms::resolve(&manifest, 0).is_err());
        let mut bad = manifest.clone(); bad["programs"].as_array_mut().unwrap().pop();
        assert!(ProducerPrograms::resolve(&bad, 1).is_err());
        let mut bad = manifest.clone(); bad["programs"].as_array_mut().unwrap().push(manifest["programs"][0].clone());
        assert!(ProducerPrograms::resolve(&bad, 1).is_err());
        let mut bad = manifest.clone(); bad["programs"][2]["pointers"][5]["dtype"] = json!("int32");
        assert!(ProducerPrograms::resolve(&bad, 1).is_err());
    }
    #[test]
    fn executable_inventory_requires_every_mode_and_exact_selection_strides() {
        let manifest = manifest();
        let (programs, specs) = MsaPrograms::resolve(&cfg(), &manifest, 16, 2048, 2049).unwrap();
        assert_eq!(specs.len(), 4);
        assert_eq!(programs.layouts[0].rows, 64);
        assert_eq!(programs.layouts[1].rows, 4096);
        let mut broken = manifest.clone();
        broken["programs"].as_array_mut().unwrap().pop();
        assert!(matches!(
            MsaPrograms::resolve(&cfg(), &broken, 16, 2048, 2049),
            Err(Refusal::MissingExport { .. })
        ));
        let mut broken = manifest.clone();
        broken["programs"]
            .as_array_mut()
            .unwrap()
            .push(manifest["programs"][0].clone());
        assert!(MsaPrograms::resolve(&cfg(), &broken, 16, 2048, 2049).is_err());
        for field in ["shape", "dtype", "role"] {
            let mut broken = manifest.clone();
            broken["programs"][0]["pointers"][0][field] = json!("wrong");
            assert!(MsaPrograms::resolve(&cfg(), &broken, 16, 2048, 2049).is_err());
        }
        assert!(MsaPrograms::resolve(&cfg(), &manifest, 65, 2048, 2049).is_err());
        assert!(MsaPrograms::resolve(&cfg(), &manifest, 16, 4097, 2049).is_err());
        assert!(MsaPrograms::resolve(&cfg(), &manifest, 16, 2048, 1048577).is_err());
        let mut config = cfg();
        config.heads = 32;
        assert!(MsaPrograms::resolve(&config, &manifest, 16, 2048, 2049).is_err());
    }
    #[test]
    fn owners_keep_compiled_strides_padding_and_independent_backing_tails() {
        let (programs, _) = MsaPrograms::resolve(&cfg(), &manifest(), 16, 2048, 2049).unwrap();
        let decode = programs
            .workspace(false)
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(decode["selections"], 64 * 256);
        assert_eq!(decode["index.scratch"], 25202688);
        let prefill = programs
            .workspace(true)
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(prefill["selections"], 4096 * 256);
        assert_eq!(prefill["index.keys"], 1048576 * 128);
        assert_eq!(prefill["index.key_scales"], 1048576 * 4);
        assert_eq!(prefill["index.scratch"], 539197952);
        let mut components = programs
            .workspace_components("minimax.prefill.0", 0, true, Lifetime::TARGET, 16)
            .unwrap();
        use crate::placement::planner::{ComponentKind, fixed_request_tails};
        assert_eq!(components.len(), prefill.len());
        assert!(components.iter().all(|component| matches!(component.kind, ComponentKind::Allocation { .. })));
        fixed_request_tails(&mut components, 16, &[]).unwrap();
        assert!(components.len() > prefill.len());
        assert!(components.iter().any(|c| matches!(&c.kind,
            crate::placement::planner::ComponentKind::AllocationGranularity(tail)
                if tail.id == "minimax.prefill.0.index.scratch.tail")));
        assert!(programs
            .workspace_components("", 0, true, Lifetime::TARGET, 16)
            .is_err());
    }
    #[test]
    fn cache_charges_planar_kv_and_packed_index_once_per_layer() {
        for (context, width) in [
            (127, 4),
            (128, 4),
            (129, 4),
            (2048, 32),
            (2049, 36),
            (1048576, 16384),
        ] {
            let cache = MsaCache::new(60, 17, 16, context).unwrap();
            assert_eq!(cache.table_width, width);
            let buffers = cache
                .buffers()
                .unwrap()
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            let globals = ["page_table", "decode_table", "lengths"];
            assert_eq!(buffers.len(), cache.layers as usize * 3 + globals.len());
            assert_eq!(buffers["layer.59.keys"], 17 * 32768);
            assert_eq!(buffers["layer.0.values"], 17 * 32768);
            assert_eq!(buffers["layer.17.index"], 17 * 8448);
            assert_eq!(buffers["page_table"], 16 * u64::from(width) * 4);
            assert_eq!(buffers["lengths"], 16 * 4);
            assert!(!cache.components("minimax.cache", 0, 17).unwrap().is_empty());
        }
        assert!(MsaCache::new(0, 1, 1, 128).is_err());
        assert!(MsaCache::new(1, u32::MAX, 1, 128).is_err());
    }
}

fn components<S: AsRef<str>>(
    buffers: Vec<(S, u64)>,
    prefix: &str,
    rank: u8,
    lifetime: Lifetime,
    maximum_pages: u64,
    category: Category,
) -> Result<Vec<Component>, Refusal> {
    if prefix.is_empty() || rank > 1 || maximum_pages == 0 {
        return Err(Refusal::InvalidFact(
            "invalid MiniMax MSA owner identity".into(),
        ));
    }
    let mut out = Vec::new();
    for (name, bytes) in buffers {
        if bytes == 0 || bytes > isize::MAX as u64 {
            return Err(Refusal::Overflow("MiniMax MSA owner"));
        }
        let allocation = Allocation::buffer(
            format!("{prefix}.{}", name.as_ref()),
            Location::coordinator(rank),
            bytes,
            lifetime,
            category,
            "MiniMax native MSA",
        );
        out.push(Component::allocation(
            allocation,
            ByteCurve::fixed(bytes, maximum_pages),
            false,
            maximum_pages,
        ));
    }
    Ok(out)
}
