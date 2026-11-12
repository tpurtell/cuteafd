//! Exact MSA program and backing inventory shared by admission and execution.
//! This describes the one-GPU coordinator; it is not a placement policy.
use super::MiniMaxM3Config;
use crate::placement::{
    inventory::ProgramEntry,
    planner::{fixed_request_tails, Allocation, ByteCurve, Component, Lifetime, Location, Refusal},
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
            || decode_rows > prefill_rows
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsaCache {
    pub layers: u32,
    pub pool_pages: u32,
    pub requests: u32,
    pub table_width: u32,
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
            table_width: max_context.div_ceil(64),
        })
    }
    pub fn buffers(self) -> Result<Vec<(String, u64)>, Refusal> {
        let mut out = Vec::new();
        for layer in 0..self.layers {
            for (name, stride) in [("keys", 32768u64), ("values", 32768), ("index", 8448)] {
                let bytes = u64::from(self.pool_pages)
                    .checked_mul(stride)
                    .filter(|&n| n <= isize::MAX as u64)
                    .ok_or(Refusal::Overflow("MiniMax MSA cache"))?;
                out.push((format!("layer.{layer}.{name}"), bytes));
            }
        }
        out.push((
            "page_table".into(),
            u64::from(self.requests) * u64::from(self.table_width) * 4,
        ));
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
        let components = programs.workspace_components("minimax.producer.0", 0, Lifetime::TARGET, 16).unwrap();
        use crate::placement::planner::ComponentKind;
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
        let components = programs
            .workspace_components("minimax.prefill.0", 0, true, Lifetime::TARGET, 16)
            .unwrap();
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
            (127, 2),
            (128, 2),
            (129, 3),
            (2048, 32),
            (2049, 33),
            (1048576, 16384),
        ] {
            let cache = MsaCache::new(60, 17, 16, context).unwrap();
            assert_eq!(cache.table_width, width);
            let buffers = cache
                .buffers()
                .unwrap()
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(buffers.len(), 60 * 3 + 2);
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
    fixed_request_tails(&mut out, maximum_pages, &[])?;
    Ok(out)
}
