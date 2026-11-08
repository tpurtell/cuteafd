//! Retained owners of the standard GQA drafter, excluding program/graph internals.
use super::draft_physical::{GlmDflashConfig, RING, TAP_ROWS};
use super::draft_representation::{draft_fp8_scratch_bytes, draft_workspace_bytes, GlmDraftCapacity, GlmDraftRepresentation};
use crate::families::qwen4::Qwen4KvCache;
use crate::placement::planner::{Allocation, Lifetime, Location, Phase};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{memory_layout::Category, DType};
use std::{collections::BTreeMap, path::Path};

/// All extents are physical allocation sizes (including device allocation floors).
/// The target head/embedding are borrowed; only the host mask row belongs here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmDflashPhysicalLayout {
    owners: Vec<Allocation>,
    weight_owners: Vec<String>,
}

impl GlmDflashPhysicalLayout {
    pub fn read(snapshot: &Path, representation: GlmDraftRepresentation, capacity: GlmDraftCapacity,
        sm_count: usize, fp8_mode: u8, kv_cache: Qwen4KvCache) -> Result<Self> {
        Self::read_tp(snapshot, representation, capacity, sm_count, fp8_mode, kv_cache, 0, 1)
    }

    pub fn read_tp(snapshot: &Path, representation: GlmDraftRepresentation, capacity: GlmDraftCapacity,
        sm_count: usize, fp8_mode: u8, kv_cache: Qwen4KvCache, rank: usize, ranks: usize) -> Result<Self> {
        ensure!(rank < ranks, "invalid DFlash rank {rank}/{ranks}");
        let cfg = GlmDflashConfig::read(snapshot)?;
        let mut layout = Self::new(&cfg.for_tp(ranks)?, representation, capacity, sm_count, fp8_mode, kv_cache)?;
        if ranks == 2 {
            layout.owners.push(Allocation::buffer("workspace.reduced", Location::coordinator(0),
                (capacity.block_rows * cfg.hidden * 2).max(256) as u64,
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "draft owners"));
        }
        if rank > 0 {
            layout.owners.retain(|owner| !Self::head_owner(&owner.id));
            layout.weight_owners.retain(|id| !Self::head_owner(id));
        }
        let path = snapshot.join("model.safetensors");
        let file_bytes = std::fs::metadata(&path)?.len();
        let headers: BTreeMap<_, _> = crate::read_safetensors_metadata(&path)?.into_iter()
            .map(|tensor| (tensor.name.clone(), tensor)).collect();
        for (name, shape) in cfg.tensor_shapes() {
            let tensor = headers.get(&name).with_context(|| format!("drafter checkpoint has no {name}"))?;
            let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
                .context("drafter tensor extent overflow")?;
            ensure!(tensor.dtype == DType::Bf16 && tensor.shape == shape && tensor.byte_length == bytes as u64,
                "{name}: expected BF16 {shape:?}/{bytes} bytes, found {:?} {:?}/{} bytes",
                tensor.dtype, tensor.shape, tensor.byte_length);
            ensure!(tensor.byte_offset.checked_add(tensor.byte_length).is_some_and(|end| end <= file_bytes),
                "{name} lies past drafter checkpoint");
        }
        Ok(layout)
    }

    pub fn new(cfg: &GlmDflashConfig, representation: GlmDraftRepresentation, capacity: GlmDraftCapacity,
        sm_count: usize, fp8_mode: u8, kv_cache: Qwen4KvCache) -> Result<Self> {
        ensure!(GlmDraftCapacity::new(capacity.context_slots, capacity.max_batch_sequences, cfg.block)? == capacity,
            "DFlash capacity differs from checkpoint block size");
        // This checked contract validates products before the itemized arithmetic below.
        draft_workspace_bytes(cfg.geometry(), capacity, cfg.block as u64,
            representation == GlmDraftRepresentation::Fp8Only, sm_count as u64, fp8_mode, kv_cache)?;
        let layout = cfg.runtime_layout(representation, capacity)?;
        ensure!(cfg.window == RING && !cfg.sample_from_anchor, "standard GQA DFlash requires its selector and ring");
        let mut owners = Vec::new();
        let mut add = |id: String, location: Location, bytes: usize| {
            let bytes = if location == Location::HOST || location == Location::PINNED { bytes } else { bytes.max(256) };
            owners.push(Allocation::buffer(id, location, bytes as u64,
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "draft owners"));
        };
        let device = Location::coordinator(0);
        let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
        let mut matrix = |id: &str, n: usize, k: usize| {
            if representation == GlmDraftRepresentation::Fp8Only {
                add(format!("{id}.values"), device, n * k);
                add(format!("{id}.scales"), device, n * k / 128 * 4);
            } else { add(id.into(), device, n * k * 2); }
        };
        matrix("weight.fc", h, cfg.taps.len() * h);
        matrix("head.projection", cfg.rank, h);
        for index in 0..cfg.layers {
            let p = format!("layer.{index}");
            for (name, n, k) in [("gate_up", 2 * inter, h), ("down", h, inter),
                ("attn_conv", cfg.conv_width(), h), ("mlp_conv", cfg.conv_width(), h),
                ("qkv", cfg.qkv_width(), h), ("o", h, cfg.heads * cfg.head_dim)] {
                matrix(&format!("{p}.{name}"), n, k);
            }
        }
        let mut weight_owners: Vec<_> = owners.iter().map(|owner| owner.id.clone()).collect();
        let mut add = |id: String, location: Location, bytes: usize| {
            let bytes = if location == Location::HOST || location == Location::PINNED { bytes } else { bytes.max(256) };
            owners.push(Allocation::buffer(id, location, bytes as u64,
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "draft owners"));
        };
        for (name, bytes) in [("hidden_norm", h * 2), ("norm", h * 2),
            ("taps", TAP_ROWS * cfg.taps.len() * h * 2), ("fused", TAP_ROWS * h * 2),
            ("fused_norm", TAP_ROWS * h * 2), ("context_positions", TAP_ROWS * 8),
            ("head.predecessor", cfg.vocab * cfg.rank * 2), ("head.successor", cfg.vocab * cfg.rank * 2),
            ("context.kv", TAP_ROWS * 2 * kv * 2), ("context.slots", TAP_ROWS * 4)] {
            add(name.into(), device, bytes);
        }
        add("mask_row".into(), Location::HOST, h * 2);
        for index in 0..cfg.layers {
            let p = format!("layer.{index}");
            let ring = capacity.context_slots * RING * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim);
            for (name, bytes) in [("input_norm", h * 2), ("post_norm", h * 2),
                ("attn_base", 4 * h * 2), ("mlp_base", 4 * h * 2),
                ("q_norm", cfg.head_dim * 2), ("k_norm", cfg.head_dim * 2), ("k_ring", ring), ("v_ring", ring)] {
                add(format!("{p}.{name}"), device, bytes);
            }
        }
        let sequences = capacity.max_batch_sequences;
        let rows = capacity.block_rows;
        let drafted = sequences * cfg.drafts();
        for (name, bytes) in [("h", rows * h * 2), ("n", rows * h * 2), ("delta", rows * h * 2),
            ("gate_up", rows * 2 * inter * 2), ("act", rows * inter * 2), ("logits", rows * cfg.vocab * 4),
            ("anchors", sequences * 4), ("ids", rows * 4), ("tokens", drafted * 4), ("positions", rows * 8),
            ("head", 4 << 20), ("dflash.conv", rows * h * 2), ("dflash.dynamic", rows * cfg.conv_width() * 2),
            ("dflash.unary", drafted * 16 * 4), ("dflash.candidates", drafted * 16 * 4),
            ("dflash.projected", rows * cfg.rank * 2), ("dflash.features", drafted * 16),
            ("dflash.topk", drafted * 64 * 16 * 8), ("attention.qkv", rows * cfg.qkv_width() * 2),
            ("attention.q", rows * cfg.heads * cfg.head_dim * 2), ("attention.k", rows * kv * 2),
            ("attention.v", rows * kv * 2), ("attention.attn", rows * cfg.heads * cfg.head_dim * 2),
            ("attention.tables", 3 * sequences * 4),
            ("attention.scratch", usize::try_from(crate::serving_capacity::draft_attention_workspace_bytes(
                sequences as u64, cfg.kv_heads as u64, 2, (RING + cfg.block) as u64,
                cfg.head_dim as u64, sm_count as u64)?)?)] {
            add(format!("workspace.{name}"), device, bytes);
        }
        add("workspace.landing".into(), Location::PINNED, drafted * 20);
        if let Some(scratch) = layout.fp8_scratch {
            let bytes = scratch.shapes.iter().map(|s| draft_fp8_scratch_bytes(scratch.rows as u64,
                s.k, s.n, sm_count as u64, fp8_mode)).max().unwrap_or(0);
            // The runtime allocates this through shared/fp8_linear.rs as a
            // workspace owner; keep the receipt in the same bucket.
            owners.push(Allocation::buffer("fp8_workspace", device,
                usize::try_from(bytes)?.max(256) as u64,
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Workspace, "draft owners"));
        }
        // prime_metadata reserves full-sized, per-destination upload owners at readiness.
        for name in ["context_positions", "workspace.positions", "workspace.anchors", "workspace.ids",
            "context.slots", "workspace.attention.tables"] {
            let bytes = owners.iter().find(|owner| owner.id == name).context("missing metadata owner")?.bytes;
            owners.push(Allocation::buffer(format!("{name}.upload"), Location::PINNED, bytes,
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "draft owners"));
        }
        weight_owners.extend(["hidden_norm".into(), "norm".into(), "head.predecessor".into(), "head.successor".into()]);
        for index in 0..cfg.layers {
            weight_owners.extend(["input_norm", "post_norm", "attn_base", "mlp_base", "q_norm", "k_norm"]
                .map(|name| format!("layer.{index}.{name}")));
        }
        Ok(Self { owners, weight_owners })
    }

    fn head_owner(id: &str) -> bool {
        id.starts_with("head.") || matches!(id.strip_suffix(".upload").unwrap_or(id),
            "workspace.logits" | "workspace.head" | "workspace.anchors" | "workspace.ids" |
            "workspace.tokens" | "workspace.landing" | "workspace.dflash.unary" |
            "workspace.dflash.candidates" | "workspace.dflash.projected" |
            "workspace.dflash.features" | "workspace.dflash.topk")
    }

    /// Relative ids remain identical to the serving receipt visitor.
    /// Resident checkpoint-derived destinations only; scratch/cache/upload owners are excluded.
    pub fn weight_owners(&self) -> impl Iterator<Item = &Allocation> {
        self.owners.iter().filter(|owner| self.weight_owners.contains(&owner.id))
    }

    pub fn resident_weights(&self) -> crate::placement::ResidentWeights {
        crate::placement::ResidentWeights(self.weight_owners().map(|owner| owner.bytes).sum())
    }

    pub fn owners(&self) -> &[Allocation] { &self.owners }

    pub fn owner_bytes(&self, id: &str) -> Result<usize> {
        self.owners.iter().find(|owner| owner.id == id)
            .with_context(|| format!("unknown GQA draft owner {id}"))?
            .bytes.try_into().context("GQA draft owner extent exceeds usize")
    }

    pub fn allocations(&self, prefix: &str, coordinator: Location) -> Vec<Allocation> {
        self.owners.iter().cloned().map(|mut owner| {
            owner.id = format!("{prefix}.{}", owner.id);
            if owner.location == Location::coordinator(0) { owner.location = coordinator; }
            owner
        }).collect()
    }
}
