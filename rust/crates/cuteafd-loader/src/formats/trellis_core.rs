//! Header-native dense trellis inventory and exact TP payload/vector layouts.
use crate::{plan::{checkpoint::Checkpoint, format}, formats::exl3_storage::exl3_module};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// How the current TP2 engine owns each eligible projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreSplit { Output, Input, Whole }

#[derive(Debug, Clone)]
pub struct CoreProjection {
    pub stem: String,
    pub bits: usize,
    pub codebook: crate::formats::exl3_storage::Exl3Codebook,
    /// Logical checkpoint shape [N,K], before any explicit output padding.
    pub shape: [usize; 2],
    pub stored_outputs: usize,
    pub split: CoreSplit,
    pub source_bytes: u64,
}

impl CoreProjection {
    /// Payload and vectors only; the four-byte codebook marker stays on host.
    pub fn resident_bytes(&self, ranks: usize) -> Result<u64> {
        ensure!(matches!(ranks, 1 | 2), "dense trellis supports one or two coordinator ranks");
        let [_, k] = self.shape;
        let n = self.stored_outputs;
        let payload = k as u64 * n as u64 * self.bits as u64 / 8;
        Ok(match self.split {
            CoreSplit::Whole => payload + 2 * (k + n) as u64,
            CoreSplit::Output => {
                ensure!(n % ranks == 0, "{} output cannot split TP{ranks}", self.stem);
                payload / ranks as u64 + 2 * (k + n / ranks) as u64
            }
            CoreSplit::Input => {
                ensure!(k % ranks == 0, "{} input cannot split TP{ranks}", self.stem);
                payload / ranks as u64 + 2 * (k / ranks + n) as u64
            }
        })
    }

    /// Model residency's split total and replicated amount. A Whole projection
    /// is later assigned to the existing MLA or latent owner, never duplicated.
    pub fn residency(&self) -> (u64, u64) {
        let [_, k] = self.shape;
        let n = self.stored_outputs;
        let payload = k as u64 * n as u64 * self.bits as u64 / 8;
        match self.split {
            CoreSplit::Output => (payload + 2 * n as u64, 2 * k as u64),
            CoreSplit::Input => (payload + 2 * k as u64, 2 * n as u64),
            CoreSplit::Whole => (0, payload + 2 * (k + n) as u64),
        }
    }
}

/// Exact compressed payload and H128 vector ownership after TP placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreWeightLayout {
    pub stem: String, pub side: crate::formats::dense_trellis::TrellisSide, pub rank: usize, pub ranks: usize,
    pub shape: [usize;2], pub bytes: usize, pub bits: usize,
    pub codebook: crate::formats::exl3_storage::Exl3Codebook,
}
impl CoreWeightLayout {
    /// Separate cudaMalloc owners used by the shared recorded loader.
    pub fn allocation_extents(&self) -> [u64; 3] {
        let [n, k] = self.shape;
        [(self.bytes - 2 * (n + k)) as u64, (2 * k) as u64, (2 * n) as u64]
    }

    pub fn arena_tensors(&self, prefix: &str) -> Result<Vec<(String, usize)>> {
        let [n, k] = self.shape;
        let payload = n.checked_mul(k).and_then(|v| v.checked_mul(self.bits))
            .context("dense trellis arena payload overflow")? / 8;
        let tensors = vec![(format!("{prefix}.trellis"), payload),
            (format!("{prefix}.suh"), k.checked_mul(2).context("trellis suh overflow")?),
            (format!("{prefix}.svh"), n.checked_mul(2).context("trellis svh overflow")?)];
        ensure!(tensors.iter().try_fold(0usize, |sum, (_, bytes)| sum.checked_add(*bytes)) == Some(self.bytes),
            "dense trellis arena differs from admitted bytes");
        Ok(tensors)
    }

    /// Vocabulary checkpoints may have an odd number of H128 blocks.
    pub fn output_shard(projection: &CoreProjection, rank: usize) -> Result<Self> {
        use crate::formats::dense_trellis::TrellisSide;
        let n = crate::shared::vocabulary::shard(projection.stored_outputs, 128, rank)?.len();
        let mut local = projection.clone();
        local.stored_outputs = n;
        let mut layout = Self::new(&local, TrellisSide::Output, 0, 1)?;
        layout.rank = rank;
        layout.ranks = 2;
        Ok(layout)
    }

    pub fn new(projection: &CoreProjection,
        side: crate::formats::dense_trellis::TrellisSide, rank: usize, ranks: usize) -> Result<Self> {
        ensure!(matches!(ranks,1|2) && rank<ranks,"dense trellis native rank geometry");
        let [_,k]=projection.shape;
        let n=projection.stored_outputs;
        ensure!(match side {
            crate::formats::dense_trellis::TrellisSide::Output=>n%ranks==0,
            crate::formats::dense_trellis::TrellisSide::Input=>k%ranks==0,
        },"dense trellis dimension does not divide ranks");
        let (n,k)=match side {
            crate::formats::dense_trellis::TrellisSide::Output=>(n/ranks,k),
            crate::formats::dense_trellis::TrellisSide::Input=>(n,k/ranks),
        };
        ensure!(matches!(projection.bits,4 | 6 | 8) && n>0 && k>0 && n%128==0 && k%128==0,
            "dense trellis K4/K6/K8 local H128 geometry");
        let payload=n.checked_mul(k).and_then(|v|v.checked_mul(projection.bits))
            .context("dense trellis payload extent overflow")?/8;
        let vectors=n.checked_add(k).and_then(|v|v.checked_mul(2))
            .context("dense trellis vector extent overflow")?;
        let bytes=payload.checked_add(vectors).context("dense trellis resident extent overflow")?;
        Ok(Self{stem:projection.stem.clone(),side,rank,ranks,shape:[n,k],bytes,bits:projection.bits,codebook:projection.codebook})
    }
}

/// Physical owners and bounded route-exclusive views for serialized projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionStorageLayout {
    pub owners: BTreeMap<&'static str, usize>,
    pub route_offsets: BTreeMap<&'static str, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionScratchLayout {
    pub extents: BTreeMap<&'static str, usize>,
    pub rows: usize,
}
impl ProjectionScratchLayout {
    /// Whether `(k, n, side, bits)` runs the fused K8 decode route (mirrors
    /// `export_b12x_dense_trellis_k6_aot.py`: K8 except Hy4's `6144x120832` LM head).
    pub fn has_fused_k8_route(k: usize, n: usize, side: usize, bits: usize) -> bool {
        bits == 8 && !(k == 6144 && n == 120832 && side == 1)
    }

    pub fn new(k: usize, n: usize, rows: usize) -> Result<Self> {
        Self::new_with_fused(k, n, rows, false)
    }

    /// Scratch layout adding the `"fused"` route-arena extent only when the
    /// resolved projection actually runs the fused K8 decode program.
    pub fn new_with_fused(k: usize, n: usize, rows: usize, fused: bool) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0,
            "dense trellis scratch geometry");
        let product = |values: &[usize]| values.iter().try_fold(1usize, |a, b| a.checked_mul(*b))
            .context("dense trellis scratch overflow");
        let partial_rows = rows.min(129);
        let partial = product(&[partial_rows, k.div_ceil(1024), n, 4])?;
        let mut extents = BTreeMap::from([
            ("rotated", product(&[rows, k, 2])?), ("table", 8), ("scales", k / 128 * 2),
            ("partial", partial),
            ("accumulator", product(&[rows, n, 4])?),
        ]);
        if fused {
            let fused_bytes = partial.checked_add(product(&[partial_rows, k, 2])?)
                .context("dense trellis fused scratch overflow")?;
            extents.insert("fused", fused_bytes);
        }
        let layout = Self { extents, rows };
        layout.bytes()?;
        Ok(layout)
    }

    /// Scratch layout for a resident weight layout, charging `"fused"` only when
    /// its width and geometry run the fused K8 decode route.
    pub fn for_weight(weight: &CoreWeightLayout, rows: usize) -> Result<Self> {
        let side = if weight.side == crate::formats::dense_trellis::TrellisSide::Input { 2 } else { 1 };
        Self::new_with_fused(
            weight.shape[1], weight.shape[0], rows,
            Self::has_fused_k8_route(weight.shape[1], weight.shape[0], side, weight.bits),
        )
    }
    /// A grouped native launch (KDA q/k/v/g):
    /// each member uses a fixed quarter of rotated/scales/accumulator, the
    /// table names the quarters, and the core indexes one shared partial.
    pub fn grouped(k: usize, n: usize, rows: usize, groups: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && groups == 4 && k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0,
            "dense trellis grouped scratch geometry");
        let live = rows;
        let product = |values: &[usize]| values.iter().try_fold(1usize, |a, b| a.checked_mul(*b))
            .context("dense trellis grouped scratch overflow");
        let layout = Self { rows, extents: BTreeMap::from([
            ("rotated", product(&[groups, live, k, 2])?), ("table", groups * 8),
            ("scales", groups * k / 128 * 2), ("partial", product(&[groups, live.min(129), k.div_ceil(1024), n, 4])?),
            ("accumulator", product(&[groups, live, n, 4])?),
        ]) };
        layout.bytes()?;
        Ok(layout)
    }
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        ensure!(self.rows == other.rows, "dense trellis scratch row capacities differ");
        for (&name, &bytes) in &other.extents {
            self.extents.entry(name).and_modify(|v| *v = (*v).max(bytes)).or_insert(bytes);
        }
        self.bytes()?;
        Ok(())
    }
    pub fn storage(&self) -> Result<ProjectionStorageLayout> {
        let mut owners = self.extents.clone();
        if !owners.contains_key("partial") {
            ensure!(owners.len() == 1 && owners.get("fp8").is_some_and(|bytes| *bytes == 256)
                && (1..=4096).contains(&self.rows), "dense trellis native partial absent");
            return Ok(ProjectionStorageLayout { owners, route_offsets: BTreeMap::new() });
        }
        let partial = owners.remove("partial").context("dense trellis native partial absent")?;
        let fused = owners.remove("fused");
        let arena = fused.map_or(partial, |bytes| partial.max(bytes));
        let mut route_offsets = BTreeMap::from([("partial", 0)]);
        if fused.is_some() {
            route_offsets.insert("fused", 0);
        }
        owners.insert("route arena", arena);
        Ok(ProjectionStorageLayout { owners, route_offsets })
    }
    /// One backing with cudaMalloc-equivalent alignment for each live owner.
    /// Mutually exclusive route buffers (partial/fused) share the route arena.
    pub fn packed(&self) -> Result<crate::serving_capacity::PackedWeightRegions> {
        let storage = self.storage()?;
        crate::serving_capacity::PackedWeightRegions::new(storage.owners.values().copied())
    }
    pub fn packed_bytes(&self) -> Result<usize> { Ok(self.packed()?.bytes) }

    pub fn bytes(&self) -> Result<usize> {
        self.storage()?.owners.values().try_fold(0usize, |a, b| a.checked_add(*b))
            .context("dense trellis scratch sum overflow")
    }
}


/// A family's explicit geometry and TP split, never inferred from a side map.
pub struct CoreContract {
    pub family: &'static str,
    pub widths: &'static [usize],
    pub split: CoreSplit,
    pub padded_outputs: bool,
    /// Only target-core widths follow CORE_VIEW; shared fixed-width drafts do not.
    pub view_varying: bool,
}

impl CoreContract {
    pub fn validate_width(&self, stem: &str, bits: usize) -> Result<()> {
        ensure!(self.widths.contains(&bits),
            "{} {stem}: unsupported dense core width K{bits}; shipping widths {:?}; add a native trellis_linear exporter program and family dispatch before loading this width",
            self.family, self.widths);
        validate_core_width(stem, bits)
    }
}

/// Dense core rates are independent of the routed-expert kernel inventory.
pub fn validate_core_width(stem: &str, bits: usize) -> Result<()> {
    ensure!(matches!(bits, 4 | 6 | 8),
        "{stem}: unsupported dense core width K{bits}; add dense_trellis_k6 trellis_linear K{bits} program in python/tools/aot/export_b12x_dense_trellis_k6_aot.py (supported: K4/K6/K8)");
    Ok(())
}

/// Read each missing selected shard header once, including index-only role
/// inventories. Companion validation and native/BF16 duplicates fail closed.
pub fn read_projections(
    checkpoint: &Checkpoint,
    expected: &BTreeMap<String, Vec<usize>>,
    select: impl Fn(&str) -> bool,
    contract: impl Fn(&str) -> Option<CoreContract>,
) -> Result<BTreeMap<String, CoreProjection>> {
    let mut selected = checkpoint.tensors.iter().filter(|t| select(&t.meta.name))
        .cloned().collect::<Vec<_>>();
    let loaded = selected.iter().map(|t| format::split_stem(&t.meta.name).0)
        .collect::<std::collections::BTreeSet<_>>();
    let missing = checkpoint.weight_map.keys().filter(|name| select(name))
        .filter_map(|name| name.strip_suffix(".trellis"))
        .filter(|stem| !loaded.contains(stem)).collect::<std::collections::BTreeSet<_>>();
    let mut shards = BTreeMap::<&str, std::collections::BTreeSet<&str>>::new();
    for (name, shard) in &checkpoint.weight_map {
        if missing.contains(format::split_stem(name).0) {
            shards.entry(shard).or_default().insert(name);
        }
    }
    for (shard, mut names) in shards {
        for meta in crate::read_safetensors_metadata(&checkpoint.snapshot.join(shard))? {
            if names.remove(meta.name.as_str()) {
                selected.push(crate::plan::checkpoint::CheckpointTensor { shard: shard.into(), meta });
            }
        }
        ensure!(names.is_empty(), "core shard {shard} lacks indexed tensors: {}",
            names.into_iter().collect::<Vec<_>>().join(", "));
    }
    let mut projections = BTreeMap::new();
    for (stem, members) in format::group_by_stem(&selected) {
        if !members.contains_key("trellis") { continue; }
        let contract = contract(&stem).with_context(|| format!("{stem}: native keep-list forbids core quantization"))?;
        let shape = expected.get(&format!("{stem}.weight"))
            .with_context(|| format!("{stem}: outside the family core tensor geometry"))?;
        ensure!(shape.len() == 2, "{stem}: core projection must be a matrix");
        let suffixes = members.keys().map(String::as_str).collect::<Vec<_>>();
        let module = exl3_module(&stem, &suffixes, |suffix| members.get(suffix).map(|t| &t.meta))?;
        contract.validate_width(&stem, module.bits)?;
        if contract.view_varying && checkpoint.view.as_ref().is_some_and(|view| view.is_mc()) {
            let expected_bits = match checkpoint.core_view() {
                crate::plan::checkpoint::CoreView::K6Compact => 6,
                crate::plan::checkpoint::CoreView::K8Compact => 8,
                _ => anyhow::bail!("{stem}: MC original core must be native, not trellis"),
            };
            ensure!(module.bits == expected_bits, "{stem}: selected core view requires K{expected_bits}, found K{}", module.bits);
        }
        let padded = contract.padded_outputs && module.output_features == shape[0].next_multiple_of(128);
        ensure!(module.input_features == shape[1] && (module.output_features == shape[0] || padded),
            "{stem}: core header shape [{},{}], expected {shape:?} (only explicit N128 padding is allowed)",
            module.output_features, module.input_features);
        ensure!(!checkpoint.weight_map.contains_key(&format!("{stem}.weight"))
            && !checkpoint.contains_tensor(&format!("{stem}.weight")),
            "{stem}: checkpoint contains both native and EXL3 storage");
        let projection = CoreProjection { stem: stem.clone(), bits: module.bits, codebook: module.codebook,
            shape: [shape[0], shape[1]], stored_outputs: module.output_features, split: contract.split,
            source_bytes: members.values().map(|t| t.meta.byte_length).sum() };
        projection.resident_bytes(1)?;
        projections.insert(stem, projection);
    }
    Ok(projections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::{dense_trellis::TrellisSide, exl3_storage::Exl3Codebook};

    fn projection(n: usize, k: usize, split: CoreSplit) -> CoreProjection {
        CoreProjection { stem: "model.projection".into(), bits: 6, codebook: Exl3Codebook::Mcg,
            shape: [n, k], stored_outputs: n, split, source_bytes: 0 }
    }

    #[test]
    fn fp8_only_scratch_has_one_explicit_owner_without_trellis_partial() -> Result<()> {
        // Step body and GLM target/draft FP8 head scratch share the storage
        // inventory contract, including recorded push planning before allocation.
        for family in ["step5", "glm5"] {
            for rows in [1,64,2048,4096] {
                let fp8=ProjectionScratchLayout {rows,extents:BTreeMap::from([("fp8",256)])};
                let storage=fp8.storage()?;
                assert_eq!(storage.owners,BTreeMap::from([("fp8",256)]),"{family}");
                assert!(storage.route_offsets.is_empty());
                assert_eq!(fp8.bytes()?,256);
                assert_eq!(fp8.packed_bytes()?,256);
                let mut mixed=ProjectionScratchLayout::new(4096,4096,rows)?;
                mixed.merge(&fp8)?;
                assert!(mixed.storage()?.route_offsets.contains_key("partial"));
                assert_eq!(mixed.storage()?.owners["fp8"],256);
                let mut broken=mixed;
                broken.extents.remove("partial");
                assert!(broken.storage().is_err(),"{family}: malformed trellis must not pass asFP8");
            }
        }
        for extents in [BTreeMap::new(),BTreeMap::from([("fp8",0)]),BTreeMap::from([("fp8",512)]),
            BTreeMap::from([("rotated",256)])] {
            assert!(ProjectionScratchLayout {rows:64,extents}.storage().is_err());
        }
        Ok(())
    }

    #[test]
    fn native_weight_layout_matches_exact_whole_and_split_residency() -> Result<()> {
        for (n, k) in [(16384, 2048), (6144, 16384), (12288, 6144), (2048, 6144), (154880, 6144)] {
            for (split, side) in [(CoreSplit::Output, TrellisSide::Output), (CoreSplit::Input, TrellisSide::Input)] {
                let p = projection(n, k, split);
                for ranks in [1, 2] {
                    for rank in 0..ranks {
                        let layout = CoreWeightLayout::new(&p, side, rank, ranks)?;
                        assert_eq!(layout.bytes as u64, p.resident_bytes(ranks)?);
                        assert_eq!(layout.shape, match side {
                            TrellisSide::Output => [n / ranks, k], TrellisSide::Input => [n, k / ranks],
                        });
                        assert_eq!(layout.codebook, Exl3Codebook::Mcg);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn header_widths_have_exact_residency_and_actionable_rejection() -> Result<()> {
        for bits in 1..=8 {
            if matches!(bits, 4 | 6 | 8) {
                validate_core_width("model.projection", bits)?;
                let mut p = projection(256, 512, CoreSplit::Output);
                p.bits = bits;
                for ranks in [1, 2] {
                    let layout = CoreWeightLayout::new(&p, TrellisSide::Output, 0, ranks)?;
                    assert_eq!(layout.bytes, 512 * 256 * bits / 8 / ranks + 2 * (512 + 256 / ranks));
                    assert_eq!(layout.bytes as u64, p.resident_bytes(ranks)?);
                    assert_eq!(layout.arena_tensors("projection")?.iter().map(|(_, b)| b).sum::<usize>(), layout.bytes);
                }
            } else {
                let error = validate_core_width("model.projection", bits).unwrap_err().to_string();
                assert!(error.contains(&format!("unsupported dense core width K{bits}")), "{error}");
                assert!(error.contains("model.projection") && error.contains("trellis_linear") && error.contains("export_b12x_dense_trellis_k6_aot.py"), "{error}");
            }
        }
        Ok(())
    }

    #[test]
    fn family_shipping_contracts_refuse_nonshipping_widths() -> Result<()> {
        for family in ["GLM 5.3", "GLM Flash", "Muse", "Mistral Large 4", "Step-5"] {
            let contract = CoreContract { family, widths: &[6], split: CoreSplit::Whole, view_varying: true, padded_outputs: false };
            contract.validate_width("model.q_proj", 6)?;
            for bits in [4, 5, 8] {
                let error = contract.validate_width("model.q_proj", bits).unwrap_err().to_string();
                assert!(error.contains(family) && error.contains(&format!("unsupported dense core width K{bits}"))
                    && error.contains("trellis_linear"), "{error}");
            }
        }
        Ok(())
    }

    #[test]
    fn padded_kv_a_layout_keeps_full_transform_extent() -> Result<()> {
        let mut p = projection(576, 6144, CoreSplit::Whole);
        p.stored_outputs = 640;
        let layout = CoreWeightLayout::new(&p, TrellisSide::Output, 0, 1)?;
        assert_eq!(layout.shape, [640, 6144]);
        assert_eq!(layout.bytes as u64, p.resident_bytes(1)?);
        assert_eq!(layout.bytes, 640 * 6144 * 6 / 8 + (640 + 6144) * 2);
        Ok(())
    }

    #[test]
    fn native_weight_layout_rejects_invalid_and_overflowing_geometry() -> Result<()> {
        let p = projection(640, 6144, CoreSplit::Whole);
        for (rank, ranks) in [(0, 0), (0, 3), (2, 2)] {
            assert!(CoreWeightLayout::new(&p, TrellisSide::Output, rank, ranks).is_err());
        }
        assert!(CoreWeightLayout::new(&p, TrellisSide::Output, 0, 2).is_err());
        let mut p = projection(128, 128, CoreSplit::Whole);
        for bits in [1, 2, 3, 5, 7] {
            p.bits = bits;
            assert!(CoreWeightLayout::new(&p, TrellisSide::Output, 0, 1).is_err());
        }
        p.bits = 4;
        assert_eq!(CoreWeightLayout::new(&p, TrellisSide::Output, 0, 1)?.bytes, 128 * 128 * 4 / 8 + 256 * 2);
        p.bits = 6;
        p.stored_outputs = usize::MAX & !127;
        assert!(CoreWeightLayout::new(&p, TrellisSide::Output, 0, 1).unwrap_err().to_string().contains("overflow"));
        Ok(())
    }

    #[test]
    fn serialized_projection_scratch_preserves_bounded_partial_and_route_owner() -> Result<()> {
        let mut layout = ProjectionScratchLayout::new(6144, 640, 4096)?;
        layout.merge(&ProjectionScratchLayout::new(2048, 16384, 4096)?)?;
        assert_eq!(layout.extents["rotated"], 4096 * 6144 * 2);
        assert_eq!(layout.extents["partial"], 129 * 2 * 16384 * 4);
        assert_eq!(layout.extents["accumulator"], 4096 * 16384 * 4);
        let storage = layout.storage()?;
        assert!(!storage.owners.contains_key("partial"));
        assert_eq!(storage.route_offsets["partial"], 0);
        assert_eq!(storage.owners["route arena"], layout.extents["partial"]);
        assert_eq!(layout.bytes()?, layout.extents.values().sum::<usize>());
        assert!(layout.merge(&ProjectionScratchLayout::new(6144, 640, 64)?).is_err());
        Ok(())
    }

    #[test]
    fn fused_k8_scratch_charges_route_arena_once_and_leaves_k6_and_hy4_head_unchanged() -> Result<()> {
        let unfused = ProjectionScratchLayout::new(6144, 120832, 64)?;
        let hy4_head_k8 = CoreWeightLayout {
            stem: "lm_head".into(), side: TrellisSide::Output, rank: 0, ranks: 1,
            shape: [120832, 6144], bytes: 0, bits: 8, codebook: Exl3Codebook::Mul1,
        };
        assert_eq!(ProjectionScratchLayout::for_weight(&hy4_head_k8, 64)?, unfused);
        let k6_body = CoreWeightLayout {
            stem: "q_a".into(), side: TrellisSide::Output, rank: 0, ranks: 1,
            shape: [2048, 6144], bytes: 0, bits: 6, codebook: Exl3Codebook::Mul1,
        };
        assert_eq!(ProjectionScratchLayout::for_weight(&k6_body, 64)?,
            ProjectionScratchLayout::new(6144, 2048, 64)?);
        let k8_body = CoreWeightLayout { bits: 8, ..k6_body };
        let fused = ProjectionScratchLayout::for_weight(&k8_body, 64)?;
        assert_eq!(fused.extents["fused"], 64 * 6 * 2048 * 4 + 64 * 6144 * 2);
        let storage = fused.storage()?;
        assert_eq!(storage.owners.len(), 5);
        assert_eq!(storage.route_offsets["partial"], 0);
        assert_eq!(storage.route_offsets["fused"], 0);
        assert_eq!(storage.owners["route arena"], fused.extents["fused"]);
        assert_eq!(fused.bytes()? - ProjectionScratchLayout::new(6144, 2048, 64)?.bytes()?, 64 * 6144 * 2);
        assert_eq!(fused.packed()?.regions.len(), 5);
        Ok(())
    }
}
