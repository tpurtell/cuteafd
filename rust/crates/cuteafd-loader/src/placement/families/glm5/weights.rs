//! Header-native weights and serialized K6/KV-B/row-reader loading subsets.
//! Other loading temporaries and serving arenas join separately; this inventory
//! alone cannot authorize family Ready.
use crate::families::glm5::{GlmDsaConfig, GlmIndexer, core::CoreInventory};
use crate::formats::trellis_core::CoreWeightLayout;
use crate::placement::planner::{
    Activation, Allocation, Allowance, ByteCurve, Component, ComponentKind, EmbeddingHome,
    Lifetime, Location, Phase, Refusal,
};
use crate::plan::{
    Checkpoint,
    exlr8::dense::{DensePlan, DenseSelection, DenseTransform},
};
use anyhow::{Context, Result, ensure};
use cuteafd_core::{DType, memory_layout::Category};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct WeightOwner {
    pub allocation: Allocation,
    pub group: String,
    pub format: String,
}

/// Serialized K6 read/pack/upload facts, not a complete target load census.
/// Ordinary Vec payload capacities are unmeasured; metadata/allocator overhead
/// and a concurrent prefetched draft file must be charged separately.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CoreLoading {
    pub heap_payload_bytes: u64,
    /// Largest single payload/vector copy, never the sum of three uploads.
    /// All target, draft and workspace users share one reusable pinned buffer.
    pub sync_upload_bytes: u64,
}

impl CoreLoading {
    fn include(&mut self, source_bytes: u64, layout: &CoreWeightLayout) -> Result<()> {
        let [n, k] = layout.shape.map(|n| n as u64);
        let payload = product(&[n, k, layout.bits as u64])? / 8;
        let input = product(&[k, 2])?;
        let output = product(&[n, 2])?;
        ensure!(sum([payload, input, output])? == layout.bytes as u64,
            "GLM K6 loading extent differs from physical layout");
        let heap = sum([source_bytes, layout.bytes as u64])?;
        self.heap_payload_bytes = self.heap_payload_bytes.max(heap);
        self.sync_upload_bytes = self.sync_upload_bytes.max(payload).max(input).max(output);
        Ok(())
    }
}

/// BF16 KV-B absorption still runs when the other attention weights use K6.
/// Its source, transposed outputs and one rank's upload slice coexist.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KvBLoading {
    pub heap_payload_bytes: u64,
    pub sync_upload_bytes: u64,
}

impl KvBLoading {
    fn include(&mut self, source: u64, uk: u64, uv: u64, ranks: usize) -> Result<()> {
        ensure!((1..=2).contains(&ranks) && uk > 0 && uv > 0
            && uk % ranks as u64 == 0 && uv % ranks as u64 == 0,
            "GLM BF16 KV-B load split");
        let outputs = sum([uk, uv])?;
        ensure!(outputs == source, "GLM BF16 KV-B source differs from absorbed outputs");
        let upload = uk.max(uv) / ranks as u64;
        let heap = sum([source, outputs, upload])?;
        self.heap_payload_bytes = self.heap_payload_bytes.max(heap);
        self.sync_upload_bytes = self.sync_upload_bytes.max(upload);
        Ok(())
    }
}

/// The BF16 row reader holds all raw members while one FP8 member's raw
/// block grid and device dequantization pair are live. Device owners join later.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RowLoading {
    pub heap_payload_bytes: u64,
    pub sync_upload_bytes: u64,
}

impl RowLoading {
    fn include(&mut self, headers: &Headers<'_>, names: &[String]) -> Result<()> {
        ensure!(!names.is_empty(), "empty GLM BF16 row source");
        let mut source = 0;
        let mut grid = 0;
        let mut upload = 0;
        let mut columns = None;
        for name in names {
            let header = headers.get(name)?;
            let [n, k] = headers.matrix(name)?;
            ensure!(columns.is_none_or(|columns| columns == k),
                "GLM BF16 row sources differ in columns");
            columns = Some(k);
            let width = match header.dtype {
                DType::Bf16 => 2,
                DType::F8E4M3 => 1,
                _ => anyhow::bail!("unsupported GLM BF16 row source {name}"),
            };
            ensure!(header.byte_length == product(&[n, k, width])?,
                "GLM BF16 row source byte extent {name}");
            source = sum([source, header.byte_length])?;
            upload = upload.max(header.byte_length);
            if header.dtype == DType::F8E4M3 {
                let scale = headers.get(&format!("{name}_scale_inv"))?;
                let shape = [n.div_ceil(128), k.div_ceil(128)];
                ensure!(scale.dtype == DType::F32
                    && scale.shape == shape.map(|n| n as usize)
                    && scale.byte_length == product(&[shape[0], shape[1], 4])?,
                    "GLM BF16 row source scale grid {name}");
                grid = grid.max(scale.byte_length);
                upload = upload.max(scale.byte_length);
            }
        }
        let heap = sum([source, grid])?;
        self.heap_payload_bytes = self.heap_payload_bytes.max(heap);
        self.sync_upload_bytes = self.sync_upload_bytes.max(upload);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WeightInventory {
    pub ranks: usize,
    pub owners: Vec<WeightOwner>,
    pub core_loading: CoreLoading,
    pub kv_b_loading: KvBLoading,
    pub row_loading: RowLoading,
}

impl WeightInventory {
    /// These target operations serialize, so their heap peaks do not add.
    /// Concurrent draft prefetch and embedding filler join as separate charges.
    pub fn loading_components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 {
            return Err(Refusal::InvalidFact("empty GLM load page domain".into()));
        }
        let bytes = self.core_loading.heap_payload_bytes
            .max(self.kv_b_loading.heap_payload_bytes).max(self.row_loading.heap_payload_bytes);
        if bytes == 0 {
            return Ok(Vec::new());
        }
        Ok(vec![Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(bytes, maximum_pages),
            kind: ComponentKind::Allowance(Allowance {
                location: Location::HOST,
                bytes,
                lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
                source: "GLM serialized target payload heap subset: max(K6 full four-member source including marker + local packed payload/suh/svh, BF16 KV-B source + full uk/uv outputs + largest single-rank upload slice, BF16 row reader all raw members + largest one-member FP32 grid); ordinary Rust heap, unmeasured assumption; excludes allocator/metadata/thread overhead, retained thread-local staging, device dequantization temporaries, other target operations, concurrent draft prefetch and embedding filler".into(),
                assumed: true,
            }),
        }])
    }

    fn add(
        &mut self,
        rank: usize,
        name: &str,
        group: &str,
        format: &str,
        bytes: u64,
    ) -> Result<()> {
        ensure!(
            rank < self.ranks && bytes > 0,
            "invalid GLM weight owner {name}"
        );
        let id = format!("glm.weights.gpu.{rank}.{name}");
        ensure!(
            !self.owners.iter().any(|owner| owner.allocation.id == id),
            "duplicate GLM weight owner {id}"
        );
        self.owners.push(WeightOwner {
            allocation: Allocation::buffer(
                id,
                Location::coordinator(u8::try_from(rank)?),
                bytes,
                Lifetime::RESIDENT,
                if group == "embedding" {
                    Category::Embedding
                } else {
                    Category::Weights
                },
                "GLM target weights",
            ),
            group: group.into(),
            format: format.into(),
        });
        Ok(())
    }

    fn core(
        &mut self,
        rank: usize,
        name: &str,
        group: &str,
        layout: &CoreWeightLayout,
        source_bytes: u64,
    ) -> Result<()> {
        let [n, k] = layout.shape.map(|n| n as u64);
        let payload = product(&[n, k, layout.bits as u64])? / 8;
        let input = product(&[k, 2])?;
        let output = product(&[n, 2])?;
        ensure!(
            sum([payload, input, output])? == layout.bytes as u64,
            "GLM K6 physical owner extent"
        );
        // The native K6 uploader has three exact requests, without the ordinary 256-byte floor.
        for (suffix, bytes) in [("payload", payload), ("suh", input), ("svh", output)] {
            self.add(rank, &format!("{name}.{suffix}"), group, "exl3-k6", bytes)?;
        }
        self.core_loading.include(source_bytes, layout)?;
        Ok(())
    }

    fn embedding(&mut self, home: EmbeddingHome, bytes: u64, format: &str) -> Result<()> {
        ensure!(bytes > 0, "empty GLM embedding owner");
        let location = match home {
            EmbeddingHome::Device(rank) => {
                ensure!(
                    usize::from(rank) < self.ranks,
                    "GLM embedding device outside ranks"
                );
                Location::coordinator(rank)
            }
            EmbeddingHome::HostPinned => Location::PINNED,
        };
        self.owners.push(WeightOwner {
            allocation: Allocation::buffer(
                "glm.weights.embedding",
                location,
                bytes,
                Lifetime::RESIDENT,
                Category::Embedding,
                "GLM target weights",
            ),
            group: "embedding".into(),
            format: format.into(),
        });
        Ok(())
    }

    /// The executor adds peer routers only for the admitted local routed layers.
    pub fn add_peer_routers(&mut self, layers: &[usize]) -> Result<()> {
        ensure!(
            self.ranks == 2,
            "GLM peer routers require two coordinator ranks"
        );
        let mut replicas = Vec::new();
        for &layer in layers {
            for key in ["gate", "gate.bias"] {
                let id = format!("glm.weights.gpu.0.layer.{layer}.operand.{key}");
                let owner = self
                    .owners
                    .iter()
                    .find(|owner| owner.allocation.id == id)
                    .with_context(|| format!("GLM peer router source {id}"))?;
                replicas.push((
                    format!("layer.{layer}.operand.{key}"),
                    owner.allocation.bytes,
                    owner.format.clone(),
                ));
            }
        }
        for (name, bytes, format) in replicas {
            self.add(1, &name, "router", &format, bytes)?;
        }
        Ok(())
    }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 {
            return Err(Refusal::InvalidFact("empty GLM weight page domain".into()));
        }
        Ok(self
            .owners
            .iter()
            .flat_map(|owner| {
                let bytes = ByteCurve::fixed(owner.allocation.bytes, maximum_pages);
                [Component::allocation(owner.allocation.clone(), bytes.clone(), false, maximum_pages),
                    Component::resident_weight(crate::placement::planner::ResidentWeight::from_allocation(
                        &owner.allocation, crate::placement::ResidentWeights(owner.allocation.bytes)), bytes)]
            })
            .collect())
    }

    /// Display totals are reduced from physical owners, never the other way round.
    pub fn resident_groups(&self) -> Result<Vec<Vec<(String, String, u64)>>> {
        let mut groups = vec![BTreeMap::<(String, String), u64>::new(); self.ranks];
        for owner in &self.owners {
            let crate::placement::planner::Device::Coordinator(rank) =
                owner.allocation.location.device
            else {
                continue;
            };
            let total = groups[usize::from(rank)]
                .entry((owner.group.clone(), owner.format.clone()))
                .or_default();
            *total = total
                .checked_add(owner.allocation.bytes)
                .context("GLM resident group overflow")?;
        }
        Ok(groups
            .into_iter()
            .map(|rank| {
                rank.into_iter()
                    .map(|((group, format), bytes)| (group, format, bytes))
                    .collect()
            })
            .collect())
    }

    /// The executor's existing whole-attention owners are inputs, not a policy
    /// selected here. FFNs remain split and residual norms remain replicated.
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        ranks: usize,
        layers: usize,
        owners: Option<&[u8]>,
        native_bf16: bool,
        native_shared: bool,
        embedding: EmbeddingHome,
    ) -> Result<Self> {
        ensure!((1..=2).contains(&ranks), "GLM coordinator ranks");
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
        ensure!(
            layers <= cfg.layers
                && owners.is_none_or(|owners| ranks == 2
                    && owners.len() == layers
                    && owners.iter().all(|&rank| usize::from(rank) < ranks)),
            "GLM weight layer ownership"
        );
        let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        let headers = Headers(
            checkpoint
                .tensors
                .iter()
                .map(|tensor| (tensor.meta.name.as_str(), &tensor.meta))
                .collect(),
        );
        let mut out = Self {
            ranks,
            owners: Vec::new(),
            core_loading: CoreLoading::default(),
            kv_b_loading: KvBLoading::default(),
            row_loading: RowLoading::default(),
        };
        for layer in 0..layers {
            let root = format!("model.layers.{layer}");
            let attn = format!("{root}.self_attn");
            let qkv = [
                format!("{attn}.q_a_proj.weight"),
                format!("{attn}.kv_a_proj_with_mqa.weight"),
            ];
            let attention = [
                qkv[0].clone(),
                qkv[1].clone(),
                format!("{attn}.q_b_proj.weight"),
                format!("{attn}.kv_b_proj.weight"),
                format!("{attn}.o_proj.weight"),
            ];
            let core_attention = core.get_weight(&qkv[0]).is_some();
            let bf16 = core_attention || native_bf16 && headers.all_bf16(&attention)?;
            let layouts =
                core.layer_weights(&cfg, layer, ranks, owners.map(|owners| owners[layer]))?;
            if cfg.indexers[layer] == GlmIndexer::Full {
                out.row_loading.include(&headers, &[
                    cfg.weight(layer, crate::families::glm5::names::Operand::IndexK)
                        .context("GLM indexer row source")?,
                    cfg.weight(layer, crate::families::glm5::names::Operand::IndexWeights)
                        .context("GLM indexer row weights")?,
                ])?;
            }
            if bf16 {
                let name = format!("{attn}.kv_b_proj.weight");
                let header = headers.get(&name)?;
                let shape = headers.matrix(&name)?;
                let uk = product(&[cfg.heads as u64, cfg.kv_lora_rank as u64,
                    cfg.nope_dim(layer) as u64, 2])?;
                let uv = product(&[cfg.heads as u64, cfg.v_head_dim as u64,
                    cfg.kv_lora_rank as u64, 2])?;
                ensure!(header.dtype == DType::Bf16
                    && product(&[shape[0], shape[1], 2])? == header.byte_length
                    && shape == [product(&[cfg.heads as u64,
                        sum([cfg.nope_dim(layer) as u64, cfg.v_head_dim as u64])?])?,
                        cfg.kv_lora_rank as u64], "GLM BF16 kv_b source geometry");
                out.kv_b_loading.include(header.byte_length, uk, uv,
                    if owners.is_some() { 1 } else { ranks })?;
            }
            for rank in 0..ranks {
                let prefix = format!("layer.{layer}.operand");
                for (key, tensor) in [
                    ("input_norm", "input_layernorm.weight"),
                    ("post_norm", "post_attention_layernorm.weight"),
                ] {
                    out.add(
                        rank,
                        &format!("{prefix}.{key}"),
                        "norm",
                        "bf16",
                        headers
                            .get(&format!("{root}.{tensor}"))?
                            .byte_length
                            .max(256),
                    )?;
                }
                if owners.is_none_or(|owners| usize::from(owners[layer]) == rank) {
                    let parts = if owners.is_some() { 1 } else { ranks };
                    for (key, tensor) in [
                        ("q_a_norm", "q_a_layernorm.weight"),
                        ("kv_a_norm", "kv_a_layernorm.weight"),
                    ] {
                        out.add(
                            rank,
                            &format!("{prefix}.{key}"),
                            "norm",
                            "bf16",
                            headers
                                .get(&format!("{attn}.{tensor}"))?
                                .byte_length
                                .max(256),
                        )?;
                    }
                    if !core_attention {
                        headers.projection(
                            &mut out,
                            rank,
                            &format!("{prefix}.w_qkv_a"),
                            "attention",
                            &qkv,
                            1,
                            false,
                            bf16,
                        )?;
                        if !bf16 {
                            let rows = sum(qkv
                                .iter()
                                .map(|name| headers.matrix(name).map(|shape| shape[0]))
                                .collect::<Result<Vec<_>>>()?)?;
                            let k = headers.matrix(&qkv[0])?[1];
                            out.add(
                                rank,
                                &format!("{prefix}.w_qkv_a_kscale"),
                                "attention",
                                "fp8-scale",
                                product(&[rows, k.div_ceil(128), 4])?.max(256),
                            )?;
                        }
                        for (key, tensor, input) in [
                            ("w_q_b", "q_b_proj.weight", false),
                            ("w_o", "o_proj.weight", true),
                        ] {
                            headers.projection(
                                &mut out,
                                rank,
                                &format!("{prefix}.{key}"),
                                "attention",
                                &[format!("{attn}.{tensor}")],
                                parts,
                                input,
                                bf16,
                            )?;
                        }
                    }
                    if bf16 {
                        ensure!(
                            headers.get(&format!("{attn}.kv_b_proj.weight"))?.dtype == DType::Bf16,
                            "GLM BF16 kv_b source"
                        );
                    }
                    ensure!(cfg.heads % parts == 0, "GLM heads do not divide ranks");
                    for (key, width) in [("w_uk", cfg.qk_nope_head_dim), ("w_uv", cfg.v_head_dim)] {
                        let values = product(&[
                            (cfg.heads / parts) as u64,
                            cfg.kv_lora_rank as u64,
                            width as u64,
                        ])?;
                        if bf16 {
                            out.add(
                                rank,
                                &format!("{prefix}.{key}"),
                                "attention",
                                "bf16",
                                product(&[values, 2])?.max(256),
                            )?;
                        } else {
                            out.add(
                                rank,
                                &format!("{prefix}.{key}_fp8"),
                                "attention",
                                "fp8",
                                values.max(256),
                            )?;
                            out.add(
                                rank,
                                &format!("{prefix}.{key}_scale"),
                                "attention",
                                "fp8-scale",
                                product(&[values / 64, 4])?.max(256),
                            )?;
                        }
                    }
                    if cfg.indexers[layer] == GlmIndexer::Full {
                        let iq = [format!("{attn}.indexer.wq_b.weight")];
                        let index_bf16 = core_attention || native_bf16 && headers.all_bf16(&iq)?;
                        headers.projection(
                            &mut out,
                            rank,
                            &format!("{prefix}.w_iq"),
                            "indexer",
                            &iq,
                            1,
                            false,
                            index_bf16,
                        )?;
                        let ik = sum(["wk.weight", "weights_proj.weight"]
                            .into_iter()
                            .map(|suffix| {
                                let shape = headers.matrix(&format!("{attn}.indexer.{suffix}"))?;
                                product(&[shape[0], shape[1], 2])
                            })
                            .collect::<Result<Vec<_>>>()?)?;
                        // rows() requests the exact concatenated extent, unlike upload().
                        out.add(rank, &format!("{prefix}.w_ik"), "indexer", "bf16", ik)?;
                        for (key, tensor) in
                            [("k_norm_w", "k_norm.weight"), ("k_norm_b", "k_norm.bias")]
                        {
                            out.add(
                                rank,
                                &format!("{prefix}.{key}"),
                                "indexer",
                                "bf16",
                                headers
                                    .get(&format!("{attn}.indexer.{tensor}"))?
                                    .byte_length
                                    .max(256),
                            )?;
                        }
                    }
                }
                let dense = layer < cfg.first_moe_layer;
                let mlp = if dense {
                    format!("{root}.mlp")
                } else {
                    format!("{root}.mlp.shared_experts")
                };
                let names = [
                    format!("{mlp}.gate_proj.weight"),
                    format!("{mlp}.up_proj.weight"),
                    format!("{mlp}.down_proj.weight"),
                ];
                let group = if dense { "dense" } else { "shared_expert" };
                if (dense || !native_shared) && core.get_weight(&names[0]).is_none() {
                    let shared_bf16 = !dense && native_bf16 && headers.all_bf16(&names)?;
                    headers.projection(
                        &mut out,
                        rank,
                        &format!("{prefix}.w_gate_up"),
                        group,
                        &names[..2],
                        ranks,
                        false,
                        shared_bf16,
                    )?;
                    headers.projection(
                        &mut out,
                        rank,
                        &format!("{prefix}.w_down"),
                        group,
                        &names[2..],
                        ranks,
                        true,
                        shared_bf16,
                    )?;
                    if dense
                        && headers.tensor_scales(&names[..2])?
                        && headers.tensor_scales(&names[2..])?
                    {
                        for key in ["w_gate_up_tscale", "w_down_tscale"] {
                            out.add(
                                rank,
                                &format!("{prefix}.{key}"),
                                group,
                                "fp8-tensor-scale",
                                256,
                            )?;
                        }
                    }
                }
                if !dense && rank == 0 {
                    for (key, tensor) in
                        [("gate", "weight"), ("gate.bias", "e_score_correction_bias")]
                    {
                        let header = headers.get(&format!("{root}.mlp.gate.{tensor}"))?;
                        out.add(
                            rank,
                            &format!("{prefix}.{key}"),
                            "router",
                            if header.dtype == DType::F32 {
                                "f32"
                            } else {
                                "bf16"
                            },
                            header.byte_length.max(256),
                        )?;
                    }
                }
                for (name, layout) in &layouts[rank] {
                    out.core(
                        rank,
                        &format!("layer.{layer}.core.{name}"),
                        if ["q_a", "kv_a", "q_b", "o"].contains(name) {
                            "attention"
                        } else {
                            group
                        },
                        layout,
                        core.get_weight(&layout.stem).context("GLM core source projection")?.source_bytes,
                    )?;
                }
            }
        }
        let norm = headers.get("model.norm.weight")?;
        out.add(0, "norm", "norm", "bf16", norm.byte_length.max(256))?;
        if let Some(projection) = core.get_weight("lm_head") {
            let layout = CoreWeightLayout::new(
                projection,
                crate::formats::dense_trellis::TrellisSide::Output,
                0,
                1,
            )?;
            out.core(0, "core.head", "lm_head", &layout, projection.source_bytes)?;
        } else {
            let head = headers.get("lm_head.weight")?;
            if head.dtype == DType::F8E4M3 {
                out.row_loading.include(&headers, &["lm_head.weight".into()])?;
            }
            let bytes = if head.dtype == DType::F8E4M3 {
                product(&[
                    headers.matrix("lm_head.weight")?[0],
                    headers.matrix("lm_head.weight")?[1],
                    2,
                ])?
            } else {
                head.byte_length.max(256)
            };
            out.add(0, "head", "lm_head", "bf16", bytes)?;
        }
        let table = headers.get("model.embed_tokens.weight")?;
        ensure!(
            table.dtype == DType::Bf16,
            "GLM token embedding requires BF16 source"
        );
        out.embedding(embedding, table.byte_length, "bf16")?;
        Ok(out)
    }

    /// Consume the same native plan used by the reader: storage sides and lossless
    /// conversion/absorption extents are never rederived by GLM admission.
    pub fn from_native(
        selection: &DenseSelection,
        layers: usize,
        plans: &[DensePlan],
    ) -> Result<Self> {
        ensure!(
            (1..=2).contains(&plans.len()),
            "GLM native coordinator ranks"
        );
        let mut out = Self {
            ranks: plans.len(),
            owners: Vec::new(),
            core_loading: CoreLoading::default(),
            kv_b_loading: KvBLoading::default(),
            row_loading: RowLoading::default(),
        };
        out.extend_native(selection, layers, &plans.iter().collect::<Vec<_>>())?;
        Ok(out)
    }

    /// Append a separately validated owner group, such as the vocabulary, from
    /// the executor's existing per-rank plans without rebuilding their geometry.
    pub fn extend_native(
        &mut self,
        selection: &DenseSelection,
        layers: usize,
        plans: &[&DensePlan],
    ) -> Result<()> {
        ensure!(
            (1..=2).contains(&self.ranks) && plans.len() == self.ranks,
            "GLM native owner group ranks"
        );
        let mut out = Self {
            ranks: self.ranks,
            owners: Vec::new(),
            core_loading: CoreLoading::default(),
            kv_b_loading: KvBLoading::default(),
            row_loading: RowLoading::default(),
        };
        let mut embedding = false;
        for (rank, plan) in plans.iter().enumerate() {
            ensure!(
                plan.rank == rank && plan.world == self.ranks,
                "GLM native plan rank identity"
            );
            let before = sum(out.owners.iter().map(|owner| owner.allocation.bytes))?;
            for slice in &plan.tensors {
                let group = selection.resident_group(&slice.tensor, layers)?;
                let layer = slice
                    .tensor
                    .strip_prefix("model.layers.")
                    .map(|rest| -> Result<_> {
                        let (index, leaf) =
                            rest.split_once('.').context("GLM native layer tensor")?;
                        Ok((index.parse::<usize>()?, leaf))
                    })
                    .transpose()?;
                if let Some(layout) = &slice.absorbed {
                    let (layer, _) = layer.context("GLM native absorbed layer")?;
                    for (name, bytes) in ["w_uk_fp8", "w_uk_scale", "w_uv_fp8", "w_uv_scale"]
                        .into_iter()
                        .zip(layout.allocations)
                    {
                        out.add(
                            rank,
                            &format!("layer.{layer}.operand.{name}"),
                            group,
                            if name.ends_with("scale") {
                                "fp8-scale"
                            } else {
                                "fp8"
                            },
                            crate::plan::exlr8::dense::allocation(bytes) as u64,
                        )?;
                    }
                } else {
                    for component in &slice.components {
                        let name = if let Some((layer, leaf)) = layer {
                            if let Some((role, _)) = DenseSelection::PROJECTIONS
                                .iter()
                                .find(|(_, path)| leaf.strip_suffix(".weight") == Some(*path))
                            {
                                let key = if *role == "router" && component.suffix == "bf16" {
                                    "operand.gate".into()
                                } else {
                                    format!("native.n.{role}.{}", component.suffix)
                                };
                                format!("layer.{layer}.{key}")
                            } else {
                                let key = native_plain(leaf).with_context(|| {
                                    format!("unknown GLM native plain owner {}", slice.tensor)
                                })?;
                                format!("layer.{layer}.operand.{key}")
                            }
                        } else if group == "norm" {
                            "norm".into()
                        } else {
                            ensure!(
                                rank == 0 && ["embedding", "lm_head"].contains(&group),
                                "GLM native global owner"
                            );
                            embedding |= group == "embedding";
                            format!(
                                "vocabulary.{}.{}",
                                if group == "embedding" {
                                    "embedding"
                                } else {
                                    "head"
                                },
                                component.suffix
                            )
                        };
                        let format = if slice.transform == DenseTransform::WidenBf16 {
                            "f32".into()
                        } else {
                            component.dtype.to_lowercase()
                        };
                        out.add(
                            rank,
                            &name,
                            group,
                            &format,
                            slice.transform.allocation_bytes(component)? as u64,
                        )?;
                    }
                }
            }
            ensure!(
                sum(out.owners.iter().map(|owner| owner.allocation.bytes))? - before
                    == plan.resident_bytes as u64,
                "GLM native owners differ from shared plan"
            );
        }
        if embedding {
            out.embedding(EmbeddingHome::Device(0), 256, "placeholder")?;
        }
        ensure!(
            out.owners.iter().all(|new| self
                .owners
                .iter()
                .all(|old| old.allocation.id != new.allocation.id)),
            "duplicate GLM native owner group"
        );
        self.owners.extend(out.owners);
        Ok(())
    }
}

fn native_plain(leaf: &str) -> Option<&'static str> {
    Some(match leaf {
        "input_layernorm.weight" => "input_norm",
        "post_attention_layernorm.weight" => "post_norm",
        "self_attn.q_a_layernorm.weight" => "q_a_norm",
        "self_attn.kv_a_layernorm.weight" => "kv_a_norm",
        "self_attn.indexer.k_norm.weight" => "k_norm_w",
        "self_attn.indexer.k_norm.bias" => "k_norm_b",
        "mlp.gate.e_score_correction_bias" => "gate.bias",
        _ => return None,
    })
}

fn product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |total, &value| {
        total
            .checked_mul(value)
            .context("GLM weight extent overflow")
    })
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0u64, |total, value| {
        total.checked_add(value).context("GLM weight sum overflow")
    })
}
struct Headers<'a>(BTreeMap<&'a str, &'a crate::SafetensorsTensorMetadata>);
impl Headers<'_> {
    fn get(&self, name: &str) -> Result<&crate::SafetensorsTensorMetadata> {
        self.0
            .get(name)
            .copied()
            .with_context(|| format!("missing GLM operand {name}"))
    }
    fn matrix(&self, name: &str) -> Result<[u64; 2]> {
        let shape = &self.get(name)?.shape;
        ensure!(
            shape.len() == 2 && shape.iter().all(|&n| n > 0),
            "GLM matrix {name}"
        );
        Ok([shape[0] as u64, shape[1] as u64])
    }
    fn all_bf16(&self, names: &[String]) -> Result<bool> {
        names.iter().try_fold(true, |all, name| {
            Ok(all && self.get(name)?.dtype == DType::Bf16)
        })
    }
    fn tensor_scales(&self, names: &[String]) -> Result<bool> {
        for name in names {
            if self.get(name)?.dtype != DType::F8E4M3 {
                return Ok(false);
            }
            let stem = name
                .strip_suffix("weight")
                .context("GLM tensor scale stem")?;
            for name in [format!("{name}_scale"), format!("{stem}input_scale")] {
                let Some(header) = self.0.get(name.as_str()) else {
                    return Ok(false);
                };
                ensure!(
                    header.dtype == DType::F32 && header.byte_length == 4,
                    "GLM scalar {name}"
                );
            }
        }
        Ok(true)
    }
    #[allow(clippy::too_many_arguments)]
    fn projection(
        &self,
        out: &mut WeightInventory,
        rank: usize,
        key: &str,
        group: &str,
        names: &[String],
        ranks: usize,
        input_split: bool,
        bf16: bool,
    ) -> Result<()> {
        ensure!(
            ranks > 0 && !names.is_empty() && (!input_split || names.len() == 1),
            "GLM concatenated projection axis"
        );
        let shapes = names
            .iter()
            .map(|name| self.matrix(name))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            shapes.iter().all(|shape| shape[1] == shapes[0][1]
                && shape[usize::from(input_split)] % ranks as u64 == 0),
            "GLM split projection geometry"
        );
        let values = sum(shapes
            .iter()
            .map(|shape| product(shape))
            .collect::<Result<Vec<_>>>()?)?
            / ranks as u64;
        if bf16 {
            ensure!(self.all_bf16(names)?, "GLM BF16 projection source");
            return out.add(rank, key, group, "bf16", product(&[values, 2])?.max(256));
        }
        let mut native = true;
        for (index, (name, shape)) in names.iter().zip(&shapes).enumerate() {
            let dtype = &self.get(name)?.dtype;
            ensure!(
                matches!(dtype, DType::Bf16 | DType::F8E4M3),
                "unsupported GLM projection dtype {name}"
            );
            ensure!(
                index + 1 == names.len() || shape[0] % 128 == 0,
                "GLM concatenation cuts a 128-row block {name}"
            );
            if ranks > 1 {
                ensure!(
                    shape[0] % (if input_split { 128 } else { 128 * ranks as u64 }) == 0
                        && shape[1] % (if input_split { 128 * ranks as u64 } else { 128 }) == 0,
                    "GLM FP8 split cuts a 128-element block {name}"
                );
            }
            if *dtype == DType::Bf16 {
                native = false;
                continue;
            }
            if let Some(grid) = self.0.get(format!("{name}_scale_inv").as_str()) {
                ensure!(
                    grid.dtype == DType::F32
                        && grid.shape
                            == [
                                shape[0].div_ceil(128) as usize,
                                shape[1].div_ceil(128) as usize
                            ],
                    "invalid GLM FP8 scale grid {name}"
                );
            } else {
                let scalar = self.get(&format!("{name}_scale"))?;
                ensure!(
                    scalar.dtype == DType::F32 && scalar.byte_length == 4,
                    "invalid GLM FP8 tensor scale {name}"
                );
                native = false;
            }
        }
        let grids = sum(shapes
            .iter()
            .map(|shape| product(&[shape[0].div_ceil(128), shape[1].div_ceil(128), 4]))
            .collect::<Result<Vec<_>>>()?)?
            / ranks as u64;
        // fp8_split's converted row buffers and fp8_cols's value buffer are exact
        // requests; landing and ordinary uploads retain their 256-byte floor.
        let floor_values = ranks == 1 || native;
        let floor_grids = ranks == 1 || native || input_split;
        out.add(
            rank,
            &format!("{key}_fp8"),
            group,
            "fp8",
            if floor_values {
                values.max(256)
            } else {
                values
            },
        )?;
        out.add(
            rank,
            &format!("{key}_scale"),
            group,
            "fp8-scale",
            if floor_grids { grids.max(256) } else { grids },
        )
    }
}

#[cfg(test)]
mod tests;
