//! Checkpoint-native target operands, packed once per physical coordinator.
//! Layer ownership is supplied by placement; this module does not choose it.
use super::MiniMaxM3Config;
use crate::{plan::checkpoint::Checkpoint, placement::planner::EmbeddingHome, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::native_dense::{DenseGeometry, Encoding};
use std::collections::BTreeSet;

const TEXT: &str = "language_model.model.";

mod heads;
pub use heads::{HeadsWeights, TensorSlice};

/// Header-only receive owners for one onboarding candidate. Storage routing is
/// checked by the recipe, not here: an inactive mount-only count must not poison
/// a valid push count. Native dense operands require no transform/index storage.
pub fn push_host_envelope(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors,
    backings: &[crate::placement::families::minimax_m3::ExpertBacking], gpus: usize)
    -> Result<crate::push_load::PushHostEnvelope> {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8Projection};
    ensure!(matches!(gpus, 1 | 2) && (backings.is_empty() || backings.len() == gpus),
        "MiniMax push backing ranks differ from coordinator geometry");
    let tensors = tensors.for_coordinator();
    let mut scratch = vec![0; gpus];
    for (gpu, backing) in backings.iter().enumerate() {
        ensure!(backing.gpu == gpu && backing.count == backing.layers.len() && backing.count > 0
            && backing.layers.windows(2).all(|pair| pair[0].layer < pair[1].layer),
            "MiniMax push backing layer/rank identity differs");
        let mut regions = Vec::new();
        for layer in &backing.layers {
            let view = tensors.for_layer_metadata(layer.layer)?;
            ensure!(view.format() == ExpertFormat::Nvfp4, "MiniMax push requires calibrated NVFP4 experts");
            let experts = view.shape().experts;
            let mut scalars = Vec::new();
            let mut expand = Vec::new();
            for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
                let (weight, grid) = view.slice_bytes_with(projection, gpus, gpu, layer.slicing)?;
                let scale = view.scale_region_bytes_with(projection, gpus, gpu, layer.slicing)?;
                for (plane, bytes) in [(2 * index, experts.checked_mul(weight).context("MiniMax push weight overflow")?),
                    (2 * index + 1, scale)] {
                    ensure!(layer.planes[plane].len() == bytes, "MiniMax push plane differs from headers");
                    regions.push((format!("{}.{plane}", layer.layer), bytes));
                }
                let scalar = scale.checked_sub(experts.checked_mul(grid).context("MiniMax scale extent overflow")?)
                    .context("MiniMax scalar region precedes scale grid")?;
                scalars.push(scalar);
                let [_, image] = view.projection_slices(projection, gpus, gpu, layer.slicing)?;
                if image.rows > 0 && image.width > 0 && image.width < image.destination_pitch {
                    expand.push((image.width, image.destination_pitch));
                }
            }
            let input = experts.checked_mul(4).context("MiniMax input scale extent overflow")?;
            scratch[gpu] = scratch[gpu].max(crate::push_load::fp8_host_scratch(
                [], scalars, Some([input, input]), expand)? as u64);
        }
        let canonical = WeightArenaLayout::new(crate::serving_capacity::expert_weight_arena_id(
            "fp8", backing.physical_device, &regions)?, regions)?;
        ensure!(canonical == backing.arena && backing.layers.iter().all(|layer|
            layer.planes.iter().enumerate().all(|(plane, range)| canonical.view(&format!("{}.{plane}", layer.layer))
                .is_some_and(|view| view.offset() == range.start && view.bytes() == range.len()))),
            "MiniMax push backing differs from canonical shared arena");
    }
    Ok(crate::push_load::PushHostEnvelope::new(scratch, vec![0; gpus],
        cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64)?)
}

/// One native W8A8 projection. Values and UE8M0 scales remain as shipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp8Projection {
    pub stem: String,
    pub input: usize,
    pub output: usize,
}

impl Mxfp8Projection {
    pub fn geometry(&self) -> DenseGeometry {
        DenseGeometry { encoding: Encoding::Mxfp8A8, input_features: self.input as u64,
            output_features: self.output as u64, side: 1, max_rows: 4096 }
    }

    pub fn program_name(&self) -> String {
        format!("exlr8_dense_mxfp8_k{}_n{}_s1_a8", self.input, self.output)
    }
}

/// A target core's immutable views. Expert onboarding, media and the bundled
/// drafter are independently retired scopes and must not be added here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinatorWeights {
    pub rank: usize,
    pub layers: Vec<usize>,
    pub arena: WeightArenaLayout,
    pub projections: Vec<Mxfp8Projection>,
}

/// Reused storage for serial native projections in one execution owner. Input
/// activations and any results retained across projections belong to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseWorkspace {
    pub rows: u32,
    pub records: u64,
    pub output: u64,
}

impl DenseWorkspace {
    pub fn allocations(self, prefix: &str, rank: u8, lifetime: crate::placement::planner::Lifetime)
        -> Vec<crate::placement::planner::Allocation> {
        use crate::placement::planner::{Allocation, Location};
        [("inputs", 8), ("records", self.records), ("output", self.output)].into_iter()
            .map(|(name, bytes)| Allocation::buffer(format!("{prefix}.{name}"), Location::coordinator(rank),
                bytes, lifetime, cuteafd_core::memory_layout::Category::Workspace, "native serial dense projection")).collect()
    }

    pub fn components(self, prefix: &str, rank: u8, lifetime: crate::placement::planner::Lifetime,
        maximum_pages: u64) -> std::result::Result<Vec<crate::placement::planner::Component>, crate::placement::planner::Refusal> {
        use crate::placement::planner::{ByteCurve, Component, Refusal};
        if prefix.is_empty() || rank > 1 || maximum_pages == 0 || self.rows == 0 || self.rows > 4096
            || self.records == 0 || self.output == 0 {
            return Err(Refusal::InvalidFact("invalid MiniMax dense workspace owner".into()));
        }
        let components = self.allocations(prefix, rank, lifetime).into_iter().map(|allocation| {
            let bytes = ByteCurve::fixed(allocation.bytes, maximum_pages);
            Component::allocation(allocation, bytes, false, maximum_pages)
        }).collect();
        Ok(components)
    }
}

/// Stable BF16 producer destination addressed by the owner's one-entry input
/// table. Producers write packed [live_rows, projection_K] rows here directly;
/// this is not a staging copy or another charge for retained branch outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseInput {
    pub rows: u32,
    pub width: u64,
    pub bytes: u64,
}

impl DenseInput {
    pub fn allocation(self, prefix: &str, rank: u8, lifetime: crate::placement::planner::Lifetime)
        -> crate::placement::planner::Allocation {
        crate::placement::planner::Allocation::buffer(format!("{prefix}.activation"),
            crate::placement::planner::Location::coordinator(rank), self.bytes, lifetime,
            cuteafd_core::memory_layout::Category::Workspace, "native serial dense projection")
    }

    pub fn components(self, prefix: &str, rank: u8, lifetime: crate::placement::planner::Lifetime,
        maximum_pages: u64) -> std::result::Result<Vec<crate::placement::planner::Component>, crate::placement::planner::Refusal> {
        use crate::placement::planner::{ByteCurve, Component, Refusal};
        if prefix.is_empty() || rank > 1 || maximum_pages == 0 || self.rows == 0 || self.rows > 4096
            || self.width == 0 || u64::from(self.rows).checked_mul(self.width).and_then(|n| n.checked_mul(2)) != Some(self.bytes) {
            return Err(Refusal::InvalidFact("invalid MiniMax dense BF16 producer destination".into()));
        }
        let components = vec![Component::allocation(self.allocation(prefix, rank, lifetime),
            ByteCurve::fixed(self.bytes, maximum_pages), false, maximum_pages)];
        Ok(components)
    }
}

impl CoordinatorWeights {
    pub fn dense_input(&self, rows: u32) -> Result<DenseInput> {
        ensure!((1..=4096).contains(&rows), "MiniMax dense rows outside native capacity");
        let width = self.projections.iter().map(|p| p.input as u64).max()
            .filter(|width| *width > 0).context("MiniMax dense input owner has no projections")?;
        let bytes = u64::from(rows).checked_mul(width).and_then(|n| n.checked_mul(2))
            .context("MiniMax dense input extent overflow")?;
        Ok(DenseInput { rows, width, bytes })
    }

    /// Each decode lane or concurrently live prefill unit needs its own owner.
    /// The F32 output is temporary; callers retain live gate/up/Q/K/V separately.
    pub fn dense_workspace(&self, rows: u32) -> Result<DenseWorkspace> {
        let records = self.records_bytes(rows)?;
        let output = self.projections.iter().map(|projection| {
            u64::from(rows).checked_mul(projection.output as u64).and_then(|bytes| bytes.checked_mul(4))
                .context("MiniMax dense output extent overflow")
        }).try_fold(0, |peak, bytes| Ok::<_, anyhow::Error>(peak.max(bytes?)))?;
        ensure!(records > 0 && output > 0, "MiniMax dense owner has no projections");
        Ok(DenseWorkspace { rows, records, output })
    }

    /// Whole layers are assigned explicitly. The entry GPU owns embedding,
    /// the final GPU owns the head, and every GPU owns its final norm.
    pub fn plan(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, ranks: usize,
        layer_owners: &[usize]) -> Result<Vec<Self>> {
        Self::plan_with_embedding(checkpoint, cfg, ranks, layer_owners, EmbeddingHome::Device(0))
    }

    /// Host placement has exactly one mapped pinned table, outside every device
    /// arena. Device placement aliases rank zero's target backing.
    pub fn plan_with_embedding(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, ranks: usize,
        layer_owners: &[usize], embedding: EmbeddingHome) -> Result<Vec<Self>> {
        ensure!(matches!(ranks, 1 | 2), "MiniMax coordinator ranks must be 1 or 2");
        ensure!(matches!(embedding, EmbeddingHome::Device(0) | EmbeddingHome::HostPinned),
            "MiniMax embedding belongs on the entry rank or pinned host");
        if embedding == EmbeddingHome::HostPinned {
            checkpoint.require_untied_embedding(&format!("{TEXT}embed_tokens.weight"))?;
        }
        ensure!(layer_owners.len() == cfg.layers && layer_owners.iter().all(|rank| *rank < ranks),
            "MiniMax target layer owners must cover every layer exactly once");
        ensure!(MiniMaxM3Config::from_hf(&checkpoint.config)? == *cfg,
            "MiniMax arena configuration differs from checkpoint");
        (0..ranks).map(|rank| {
            let layers = layer_owners.iter().enumerate().filter_map(|(layer, owner)|
                (*owner == rank).then_some(layer)).collect::<Vec<_>>();
            let mut builder = Builder { checkpoint, tensors: Vec::new(), projections: Vec::new() };
            builder.plain(format!("{TEXT}norm.weight"), DType::Bf16, &[cfg.hidden])?;
            if rank == 0 {
                let name = format!("{TEXT}embed_tokens.weight");
                builder.plain(name, DType::Bf16, &[cfg.vocab_size, cfg.hidden])?;
                if embedding == EmbeddingHome::HostPinned { builder.tensors.pop(); }
            }
            if rank + 1 == ranks {
                builder.plain("language_model.lm_head.weight".into(), DType::Bf16, &[cfg.vocab_size, cfg.hidden])?;
            }
            for &layer in &layers { builder.layer(cfg, layer)?; }
            Ok(Self { rank, layers,
                arena: WeightArenaLayout::new(format!("minimax/target/{rank}"), builder.tensors)?,
                projections: builder.projections })
        }).collect()
    }

    /// Projections execute serially within a row-unit owner; their A8 records
    /// share one backing. Concurrent prefill slots need separate owners.
    pub fn records_bytes(&self, rows: u32) -> Result<u64> {
        ensure!((1..=4096).contains(&rows), "MiniMax dense rows outside native capacity");
        self.projections.iter().map(|projection| projection.geometry().scratch_bytes(rows)
            .map_err(anyhow::Error::from)).try_fold(0, |peak, bytes| Ok(peak.max(bytes?)))
    }

    pub fn dense_programs(&self) -> BTreeSet<String> {
        self.projections.iter().map(Mxfp8Projection::program_name).collect()
    }

    pub fn quantizer_programs(&self) -> BTreeSet<String> {
        self.projections.iter().map(|projection|
            format!("exlr8_dense_quant_k{}_bf16_a8", projection.input)).collect()
    }

    /// Resolve only exports this rank binds. Validate the full manifest ABI,
    /// including pointer dtypes and shapes, before admission or kernel loading.
    pub fn resolve_dense_programs(&self, manifest: &serde_json::Value, rows: u32)
        -> std::result::Result<Vec<cuteafd_ffi::programs::ProgramSpec>, crate::placement::planner::Refusal> {
        use crate::placement::{inventory::ProgramEntry, planner::Refusal};
        let records = self.records_bytes(rows).map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let names = self.dense_programs().union(&self.quantizer_programs()).cloned().collect::<BTreeSet<_>>();
        let entries = manifest["programs"].as_array().ok_or_else(||
            Refusal::InvalidFact("MiniMax native PROGRAMS.json has no program list".into()))?;
        let mut resolved = Vec::new();
        for name in names {
            let refusal = |detail| Refusal::MissingExport { export: name.clone(),
                geometry: "MiniMax BF16 ingress, native MXFP8 W8A8, rotation_count=1".into(),
                context: u64::from(rows), detail };
            let mut matches = entries.iter().filter(|entry| entry["name"].as_str() == Some(name.as_str()));
            let entry = matches.next().ok_or_else(|| refusal("absent from PROGRAMS.json".into()))?;
            if matches.next().is_some() { return Err(refusal("duplicate program in PROGRAMS.json".into())); }
            let parsed = ProgramEntry::from_manifest(entry).map_err(|error| refusal(error.to_string()))?;
            if name.starts_with("exlr8_dense_quant_k") {
                let expected = self.projections.iter().find_map(|projection|
                    (name == format!("exlr8_dense_quant_k{}_bf16_a8", projection.input))
                        .then_some(cuteafd_ffi::native_dense_glue::Op::Mx8QuantBf16 { features: projection.input as u32 }))
                    .expect("quantizer name derived from projections");
                let actual = cuteafd_ffi::native_dense_glue::validate_entry(&parsed.spec, entry)
                    .map_err(|error| refusal(error.to_string()))?;
                if actual != expected { return Err(refusal("quantizer geometry differs from checkpoint".into())); }
            } else {
                if entry["op"].as_str() != Some("dense_mxfp8") {
                    return Err(refusal("native MXFP8 GEMM operation differs".into()));
                }
                let expected = self.projections.iter().find(|projection| projection.program_name() == name)
                    .expect("GEMM name derived from projections").geometry();
                let actual = cuteafd_ffi::native_dense::validate_entry(&parsed.spec, entry)
                    .map_err(|error| refusal(error.to_string()))?;
                if actual != expected || actual.scratch_bytes(rows).map_err(|error| refusal(error.to_string()))? > records {
                    return Err(refusal("GEMM geometry or record owner differs from checkpoint".into()));
                }
            }
            resolved.push(parsed.spec);
        }
        Ok(resolved)
    }

    /// Raw checkpoint operands need only a bounded reused source buffer: no
    /// dequantization, row repacking, scale conversion or device staging.
    pub fn mount_load(&self, checkpoint: &Checkpoint) -> Result<crate::shared::dense_read::MountLoad> {
        use crate::shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES};
        let mut load = MountLoad::default();
        for (name, view) in self.arena.views() {
            let header = checkpoint.header(name).with_context(|| format!("MiniMax mount lacks {name}"))?;
            ensure!(header.byte_length == view.bytes() as u64, "MiniMax mount {name}: admitted bytes differ");
            let rows = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
            ensure!(rows > 0 && view.bytes() % rows == 0, "MiniMax mount {name}: invalid row extent");
            let row_bytes = view.bytes() / rows;
            load.merge(MountLoad::read(ReadBatch::new(rows, row_bytes, row_bytes, STEP_BYTES)?));
        }
        Ok(load)
    }

    /// Charge one arena backing and its one driver-allocation tail. Tensor
    /// views are aliases, not additional resident-weight charges.
    pub fn weight_components(&self, maximum_pages: u64)
        -> std::result::Result<Vec<crate::placement::planner::Component>, crate::placement::planner::Refusal> {
        use crate::placement::planner::{weight_arena_components, Lifetime, Location, Refusal};
        let rank = u8::try_from(self.rank).map_err(|_| Refusal::Overflow("MiniMax coordinator rank"))?;
        weight_arena_components(&self.arena, Location::coordinator(rank), Lifetime::RESIDENT,
            maximum_pages, cuteafd_core::memory_layout::MemoryKind::Weights)
    }
}

struct Builder<'a> {
    checkpoint: &'a Checkpoint,
    tensors: Vec<(String, usize)>,
    projections: Vec<Mxfp8Projection>,
}

impl Builder<'_> {
    fn plain(&mut self, name: String, dtype: DType, shape: &[usize]) -> Result<()> {
        let header = self.checkpoint.header(&name)
            .with_context(|| format!("MiniMax arena requires checkpoint header {name}"))?;
        ensure!(header.dtype == dtype && header.shape == shape,
            "MiniMax arena {name}: expected {dtype:?} {shape:?}, found {:?} {:?}", header.dtype, header.shape);
        let bytes_per_element = match dtype {
            DType::Bf16 => 2, DType::F32 => 4, DType::U8 | DType::F8E4M3 => 1,
            _ => anyhow::bail!("unsupported MiniMax arena dtype {dtype:?}"),
        };
        let bytes = shape.iter().try_fold(bytes_per_element, |bytes: usize, dim|
            bytes.checked_mul(*dim).context("MiniMax arena tensor extent overflow"))?;
        ensure!(header.byte_length == bytes as u64, "MiniMax arena {name}: header byte extent differs");
        self.tensors.push((name, bytes));
        Ok(())
    }

    fn matrix(&mut self, stem: String, n: usize, k: usize) -> Result<()> {
        ensure!(k > 0 && n > 0 && k % 128 == 0 && n % 32 == 0
            && k <= i32::MAX as usize && n <= i32::MAX as usize,
            "MiniMax native MXFP8 geometry {stem}: [{n}, {k}]");
        ensure!(self.checkpoint.module_numerics(&stem)? == crate::plan::checkpoint::ModuleNumerics::W8A8,
            "MiniMax arena {stem} must retain native dynamic W8A8 arithmetic");
        self.plain(format!("{stem}.weight"), DType::F8E4M3, &[n, k])?;
        self.plain(format!("{stem}.weight_scale_inv"), DType::U8, &[n, k / 32])?;
        self.projections.push(Mxfp8Projection { stem, input: k, output: n });
        Ok(())
    }

    fn layer(&mut self, cfg: &MiniMaxM3Config, layer: usize) -> Result<()> {
        let root = format!("{TEXT}layers.{layer}");
        let h = cfg.hidden;
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            self.plain(format!("{root}.{norm}.weight"), DType::Bf16, &[h])?;
        }
        let a = format!("{root}.self_attn");
        let q = cfg.heads.checked_mul(cfg.head_dim).context("MiniMax query width overflow")?;
        let kv = cfg.kv_heads.checked_mul(cfg.head_dim).context("MiniMax KV width overflow")?;
        for (name, n, k) in [("q", q, h), ("k", kv, h), ("v", kv, h), ("o", h, q)] {
            self.matrix(format!("{a}.{name}_proj"), n, k)?;
        }
        for norm in ["q_norm", "k_norm"] {
            self.plain(format!("{a}.{norm}.weight"), DType::Bf16, &[cfg.head_dim])?;
        }
        if cfg.sparse[layer] {
            self.matrix(format!("{a}.index_q_proj"), cfg.msa.index_heads.checked_mul(cfg.msa.index_dim)
                .context("MiniMax index width overflow")?, h)?;
            self.matrix(format!("{a}.index_k_proj"), cfg.msa.index_dim, h)?;
            for norm in ["index_q_norm", "index_k_norm"] {
                self.plain(format!("{a}.{norm}.weight"), DType::Bf16, &[cfg.msa.index_dim])?;
            }
        }
        let (ffn, intermediate) = if cfg.moe[layer] {
            let moe = format!("{root}.block_sparse_moe");
            self.plain(format!("{moe}.gate.weight"), DType::F32, &[cfg.experts, h])?;
            self.plain(format!("{moe}.e_score_correction_bias"), DType::F32, &[cfg.experts])?;
            (format!("{moe}.shared_experts"), cfg.shared_intermediate)
        } else { (format!("{root}.mlp"), cfg.dense_intermediate) };
        for (name, n, k) in [("gate", intermediate, h), ("up", intermediate, h), ("down", h, intermediate)] {
            self.matrix(format!("{ffn}.{name}_proj"), n, k)?;
        }
        Ok(())
    }
}
