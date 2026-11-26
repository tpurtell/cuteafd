//! Checkpoint-native standard GQA DFlash owners; graph/program memory is opaque.
use super::draft_representation::{GlmDraftCapacity, GlmDraftGeometry, GlmDraftRepresentation, GlmDraftRuntimeLayout};
use anyhow::{ensure, Context, Result};
use std::path::Path;

/// Context entries each draft layer attends to (the checkpoint's sliding window).
pub const RING: usize = 2048;
/// Tapped rows one target step keeps (a prefill chunk's tail, or a verify step).
pub const TAP_ROWS: usize = RING;
#[derive(Debug, Clone, serde::Deserialize)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    vocab_size: usize,
    /// Newer configs nest the base in `rope_parameters`; others keep a flat `rope_theta`.
    #[serde(default)]
    rope_parameters: Option<RawRope>,
    #[serde(default)]
    rope_theta: Option<f64>,
    dflash_config: RawDflash,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawRope {
    rope_theta: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawDflash {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: u32,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
    #[serde(default)]
    selector_softcap_mode: Option<String>,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
}

/// The drafter geometry the kernels are written for, read from `config.json`.
#[derive(Debug, Clone)]
pub struct GlmDflashConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub block: usize,
    pub group: usize,
    pub mask_token: u32,
    pub rank: usize,
    pub taps: Vec<usize>,
    pub vocab: usize,
    /// The sliding window (the ring's length).
    pub window: usize,
    /// Mask each block row's context to its own window (upstream's
    /// `|p - q| < window`); off, every block row sees the last `window`
    /// context entries (the GLM 5.3 port's measured behavior).
    pub row_window: bool,
    /// Every block row is a prediction (dSpark's anchor-as-first: `block`
    /// drafts); off, the anchor row only carries context (`block - 1`).
    pub sample_from_anchor: bool,
    /// The selector applies this cap to the corrected score, not the unary logit.
    pub post_selector_softcap: Option<f32>,
}

impl GlmDflashConfig {
    /// GQA heads, retained KV and MLP neurons are independent rank-local owners.
    pub fn for_tp(&self, ranks: usize) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2) && self.heads % ranks == 0
            && self.kv_heads % ranks == 0 && self.intermediate % ranks == 0,
            "DFlash head/MLP geometry cannot split across {ranks} ranks");
        Ok(Self { heads: self.heads / ranks, kv_heads: self.kv_heads / ranks,
            intermediate: self.intermediate / ranks, ..self.clone() })
    }

    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_value(&serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?)
    }

    pub fn from_value(value: &serde_json::Value) -> Result<Self> {
        let raw: RawConfig = serde_json::from_value(value.clone())
            .context("parsing the DFlash2 config.json")?;
        let d = &raw.dflash_config;
        ensure!(matches!(raw.head_dim, 128 | 64) && raw.num_key_value_heads > 0
            && raw.num_attention_heads % raw.num_key_value_heads == 0
            && (raw.num_attention_heads / raw.num_key_value_heads).checked_mul(d.block_size).is_some_and(|rows| rows <= 64),
            "DFlash2 kernels take 128- or 64-wide heads and at most 64 queries per kv head and block \
             (got {} heads, {} kv, block {})", raw.num_attention_heads, raw.num_key_value_heads, d.block_size);
        ensure!(d.conv_kernel_size == 2 && d.selector_rank == 256 && d.selector_top_k == 16,
            "DFlash2 kernels take two-tap convolutions and a rank-256 top-16 selector");
        ensure!(raw.sliding_window == RING, "DFlash2 window {} (the ring holds {RING})", raw.sliding_window);
        let post_selector_softcap = match d.selector_softcap_mode.as_deref() {
            None | Some("before_correction") => None,
            Some("after_correction") => Some(d.final_logit_softcapping
                .filter(|cap| cap.is_finite() && *cap > 0.0)
                .context("DFlash2 after_correction requires positive final_logit_softcapping")?),
            Some(mode) => anyhow::bail!("unsupported DFlash2 selector_softcap_mode {mode}"),
        };
        Ok(Self {
            hidden: raw.hidden_size,
            intermediate: raw.intermediate_size,
            layers: raw.num_hidden_layers,
            heads: raw.num_attention_heads,
            kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            eps: raw.rms_norm_eps as f32,
            theta: raw.rope_parameters.as_ref().map(|r| r.rope_theta).or(raw.rope_theta)
                .context("DFlash2 config lacks rope_parameters.rope_theta and rope_theta")? as f32,
            block: d.block_size,
            group: d.conv_group_size,
            mask_token: d.mask_token_id,
            rank: d.selector_rank,
            taps: d.target_layer_ids.clone(),
            vocab: raw.vocab_size,
            window: raw.sliding_window,
            row_window: false,
            sample_from_anchor: false,
            post_selector_softcap,
        })
    }

    pub fn mla(cfg: &crate::formats::dflash::MlaDflashConfig) -> Self {
        Self { hidden: cfg.mla.hidden, intermediate: cfg.intermediate, layers: cfg.windows.len(),
            heads: cfg.mla.heads, kv_heads: cfg.mla.heads, head_dim: cfg.mla.nope + cfg.mla.rope,
            eps: cfg.eps as f32, theta: 0.0, block: cfg.block, group: cfg.group,
            mask_token: cfg.mask_token, rank: cfg.selector_rank, taps: cfg.taps.clone(), vocab: cfg.vocab,
            window: 0, row_window: false, sample_from_anchor: cfg.drafts() == cfg.block, post_selector_softcap: None }
    }

    /// Native BF16 body projections used by the shared GQA executor.
    pub fn projection_shapes(&self) -> [(usize, usize); 8] {
        let h = self.hidden;
        [(h, 2 * self.intermediate), (self.intermediate, h),
            (h * self.taps.len(), h), (h, self.conv_width()), (h, self.rank),
            (h, self.qkv_width()), (h, 2 * self.kv_width()), (self.heads * self.head_dim, h)]
    }

    pub fn drafts(&self) -> usize {
        if self.sample_from_anchor { self.block } else { self.block - 1 }
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    pub fn conv_width(&self) -> usize {
        4 * self.hidden / self.group
    }

    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let (h, kv, inter) = (self.hidden, self.kv_width(), self.intermediate);
        let mut shapes = vec![
            ("fc.weight".into(), vec![h, self.taps.len() * h]),
            ("hidden_norm.weight".into(), vec![h]),
            ("norm.weight".into(), vec![h]),
            ("candidate_selector.hidden_projection.weight".into(), vec![self.rank, h]),
            ("candidate_selector.predecessor_codebook".into(), vec![self.vocab, self.rank]),
            ("candidate_selector.successor_codebook".into(), vec![self.vocab, self.rank]),
        ];
        for l in 0..self.layers {
            let p = format!("layers.{l}");
            let a = format!("{p}.self_attn");
            shapes.extend([
                (format!("{p}.input_layernorm.weight"), vec![h]),
                (format!("{p}.post_attention_layernorm.weight"), vec![h]),
                (format!("{p}.attention_conv.kernel_projection.weight"), vec![self.conv_width(), h]),
                (format!("{p}.attention_conv.base_kernel"), vec![2, 2, h]),
                (format!("{p}.mlp_conv.kernel_projection.weight"), vec![self.conv_width(), h]),
                (format!("{p}.mlp_conv.base_kernel"), vec![2, 2, h]),
                (format!("{a}.q_proj.weight"), vec![self.heads * self.head_dim, h]),
                (format!("{a}.k_proj.weight"), vec![kv, h]),
                (format!("{a}.v_proj.weight"), vec![kv, h]),
                (format!("{a}.q_norm.weight"), vec![self.head_dim]),
                (format!("{a}.k_norm.weight"), vec![self.head_dim]),
                (format!("{a}.o_proj.weight"), vec![h, self.heads * self.head_dim]),
                (format!("{p}.mlp.gate_proj.weight"), vec![inter, h]),
                (format!("{p}.mlp.up_proj.weight"), vec![inter, h]),
                (format!("{p}.mlp.down_proj.weight"), vec![h, inter]),
            ]);
        }
        shapes
    }

    pub fn geometry(&self) -> GlmDraftGeometry {
        GlmDraftGeometry { hidden: self.hidden as u64, intermediate: self.intermediate as u64,
            layers: self.layers as u64, heads: self.heads as u64, kv_heads: self.kv_heads as u64,
            head_dim: self.head_dim as u64, taps: self.taps.len() as u64, vocab: self.vocab as u64,
            conv_group: self.group as u64, selector_rank: self.rank as u64 }
    }

    pub fn runtime_layout(&self, mode: GlmDraftRepresentation, capacity: GlmDraftCapacity)
        -> Result<GlmDraftRuntimeLayout> {
        Ok(GlmDraftRuntimeLayout::new(self.geometry(), mode, capacity, TAP_ROWS)?)
    }
}


#[derive(Debug, Clone, Copy)]
pub enum ShardAxis { Rows, Columns }

pub struct GqaLayerWeights<T, M> {
    pub input_norm: T, pub post_norm: T,
    pub attn_conv: M, pub attn_base: T, pub mlp_conv: M, pub mlp_base: T,
    pub qkv: M, pub q_norm: T, pub k_norm: T, pub o: M, pub gate_up: M, pub down: M,
}

pub struct GqaWeights<T, M> {
    pub layers: Vec<GqaLayerWeights<T, M>>,
    pub fc: M, pub hidden_norm: T, pub norm: T,
    pub selector: Option<(M, T, T)>,
}

/// The same traversal selects checkpoint slices for CPU requests and real loading.
/// Only immutable destinations pass through these two operations.
#[allow(clippy::too_many_arguments)]
pub fn gqa_weight_recipe<T, M>(
    source: &GlmDflashConfig,
    rank: usize, ranks: usize,
    tensor: impl Fn(&str, &[usize]) -> Result<T>,
    matrix: impl Fn(&[(&str, usize)], usize, Option<ShardAxis>) -> Result<M>,
) -> Result<GqaWeights<T, M>> {
    ensure!(rank < ranks, "invalid DFlash recipe rank {rank}/{ranks}");
    let cfg = source.for_tp(ranks)?;
    let h = cfg.hidden;
    let mut layers = Vec::with_capacity(cfg.layers);
    for l in 0..cfg.layers {
        let p = format!("layers.{l}");
        let a = format!("{p}.self_attn");
        layers.push(GqaLayerWeights {
            input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
            post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
            attn_conv: matrix(&[(&format!("{p}.attention_conv.kernel_projection.weight"), cfg.conv_width())], h, None)?,
            attn_base: tensor(&format!("{p}.attention_conv.base_kernel"), &[2, 2, h])?,
            mlp_conv: matrix(&[(&format!("{p}.mlp_conv.kernel_projection.weight"), cfg.conv_width())], h, None)?,
            mlp_base: tensor(&format!("{p}.mlp_conv.base_kernel"), &[2, 2, h])?,
            qkv: matrix(&[(&format!("{a}.q_proj.weight"), source.heads * cfg.head_dim),
                (&format!("{a}.k_proj.weight"), source.kv_width()),
                (&format!("{a}.v_proj.weight"), source.kv_width())], h, Some(ShardAxis::Rows))?,
            q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
            k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
            o: matrix(&[(&format!("{a}.o_proj.weight"), h)], source.heads * cfg.head_dim, Some(ShardAxis::Columns))?,
            gate_up: matrix(&[(&format!("{p}.mlp.gate_proj.weight"), source.intermediate),
                (&format!("{p}.mlp.up_proj.weight"), source.intermediate)], h, Some(ShardAxis::Rows))?,
            down: matrix(&[(&format!("{p}.mlp.down_proj.weight"), h)], source.intermediate, Some(ShardAxis::Columns))?,
        });
    }
    Ok(GqaWeights { layers, fc: matrix(&[("fc.weight", h)], cfg.taps.len() * h, None)?,
        hidden_norm: tensor("hidden_norm.weight", &[h])?, norm: tensor("norm.weight", &[h])?,
        selector: if rank == 0 { Some((matrix(&[("candidate_selector.hidden_projection.weight", cfg.rank)], h, None)?,
            tensor("candidate_selector.predecessor_codebook", &[cfg.vocab, cfg.rank])?,
            tensor("candidate_selector.successor_codebook", &[cfg.vocab, cfg.rank])?)) } else { None },
    })
}

/// Final immutable destinations, in the same order used by the serving binder.
/// This config-only form also supports planner fixtures; checkpoint callers use
/// `gqa_checkpoint_arena_requests` to validate the source headers first.
pub fn gqa_arena_requests(cfg: &GlmDflashConfig, representation: GlmDraftRepresentation,
    rank: usize, ranks: usize) -> Result<Vec<(String, usize)>> {
    ensure!(cfg.group > 0, "DFlash convolution group must be positive");
    let requests = std::cell::RefCell::new(Vec::new());
    gqa_weight_recipe(cfg, rank, ranks,
        |name, shape| {
            let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
                .context("DFlash tensor extent overflow")?;
            ensure!(bytes > 0, "{name}: empty DFlash tensor");
            requests.borrow_mut().push((name.to_owned(), bytes.max(256)));
            Ok(())
        },
        |parts, cols, axis| {
            ensure!(cols > 0 && !parts.is_empty(), "empty DFlash matrix");
            let mut rows = 0usize;
            let local_cols = if matches!(axis, Some(ShardAxis::Columns)) {
                ensure!(cols % ranks == 0, "DFlash column shard cuts a BF16 element");
                cols / ranks
            } else { cols };
            for (_, part_rows) in parts {
                let local = if matches!(axis, Some(ShardAxis::Rows)) {
                    ensure!(*part_rows % ranks == 0, "DFlash row shard cuts a tensor");
                    *part_rows / ranks
                } else { *part_rows };
                ensure!(local > 0, "empty DFlash matrix part");
                rows = rows.checked_add(local).context("DFlash matrix rows overflow")?;
            }
            let name = parts.iter().map(|(name, _)| *name).collect::<Vec<_>>().join("+");
            let values = rows.checked_mul(local_cols).context("DFlash matrix extent overflow")?;
            if representation == GlmDraftRepresentation::Bf16Only {
                requests.borrow_mut().push((name,
                    values.checked_mul(2).context("DFlash BF16 matrix extent overflow")?.max(256)));
            } else {
                ensure!(rows % 16 == 0 && local_cols > 0 && local_cols % 128 == 0,
                    "DFlash recipe FP8 shape {rows}/{local_cols}");
                requests.borrow_mut().extend([(format!("{name}.payload"), values),
                    (format!("{name}.scales"), values / 128 * 4)]);
            }
            Ok(())
        })?;
    Ok(requests.into_inner())
}

/// Header-validated counterpart; no tensor payload reads or device allocation.
pub fn gqa_checkpoint_arena_requests(snapshot: &Path, representation: GlmDraftRepresentation,
    rank: usize, ranks: usize) -> Result<Vec<(String, usize)>> {
    let cfg = GlmDflashConfig::read(snapshot)?;
    let requests = gqa_arena_requests(&cfg, representation, rank, ranks)?;
    validate_bf16_headers(snapshot, cfg.tensor_shapes())?;
    Ok(requests)
}

fn validate_bf16_headers(snapshot: &Path, shapes: Vec<(String, Vec<usize>)>) -> Result<()> {
    let path = snapshot.join("model.safetensors");
    let file_bytes = std::fs::metadata(&path)?.len();
    let headers = crate::read_safetensors_metadata(&path)?.into_iter()
        .map(|tensor| (tensor.name.clone(), tensor)).collect::<std::collections::HashMap<_, _>>();
    for (name, shape) in shapes {
        let tensor = headers.get(&name).with_context(|| format!("drafter checkpoint has no {name}"))?;
        let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: BF16 tensor byte count overflow"))?;
        ensure!(tensor.dtype == cuteafd_core::DType::Bf16, "{name}: drafter weights must be BF16, found {:?}", tensor.dtype);
        ensure!(tensor.shape == shape && tensor.byte_length == bytes as u64,
            "{name}: shape {:?}/{} bytes, expected BF16 {shape:?}/{bytes} bytes", tensor.shape, tensor.byte_length);
        ensure!(tensor.byte_offset.checked_add(tensor.byte_length).is_some_and(|end| end <= file_bytes),
            "{name} lies past the drafter checkpoint file");
    }
    Ok(())
}

pub struct MlaLayerWeights<T, M, A = T> {
    pub input_norm: T, pub post_norm: T,
    pub attn_conv: M, pub attn_base: T, pub mlp_conv: M, pub mlp_base: T,
    pub qkv_a: A, pub q_norm: T, pub kv_norm: T, pub q_b: A,
    pub uk: A, pub uv: A, pub output: A, pub gate_up: M, pub down: M,
}

pub struct MlaWeights<T, M, A = T> {
    pub layers: Vec<MlaLayerWeights<T, M, A>>,
    pub fc: M, pub hidden_norm: T, pub norm: T, pub selector: (M, T, T),
}

/// Shared MLA immutable traversal: header requests and `load_mla_with` bind the
/// same names, representations and absorption destinations, never KV or scratch.
pub fn mla_weight_recipe<T, M>(
    cfg: &crate::formats::dflash::MlaDflashConfig,
    tensor: impl Fn(&[(&str, Vec<usize>)]) -> Result<T>,
    matrix: impl Fn(&[(&str, usize)], usize) -> Result<M>,
    absorb: impl Fn(&str) -> Result<(T, T)>,
) -> Result<MlaWeights<T, M>> {
    ensure!(cfg.kind.is_dflash(), "MLA DFlash arena recipe requires DFlash2");
    mla_weight_recipe_with_attention(cfg, &tensor, matrix,
        |parts, columns| tensor(&parts.iter().map(|(name, rows)| (*name, vec![*rows, columns]))
            .collect::<Vec<_>>()), absorb)
}

/// Same immutable traversal for BF16 and row-major FP8 attention operands.
pub fn mla_weight_recipe_with_attention<T, M, A>(
    cfg: &crate::formats::dflash::MlaDflashConfig,
    tensor: impl Fn(&[(&str, Vec<usize>)]) -> Result<T>,
    matrix: impl Fn(&[(&str, usize)], usize) -> Result<M>,
    attention: impl Fn(&[(&str, usize)], usize) -> Result<A>,
    absorb: impl Fn(&str) -> Result<(A, A)>,
) -> Result<MlaWeights<T, M, A>> {
    ensure!(cfg.kind.is_dflash(), "MLA DFlash arena recipe requires DFlash2");
    let m = &cfg.mla;
    let h = m.hidden;
    let one = |name: &str, shape: &[usize]| tensor(&[(name, shape.to_vec())]);
    let mut layers = Vec::with_capacity(cfg.windows.len());
    for index in 0..cfg.windows.len() {
        let p = format!("layers.{index}");
        let a = format!("{p}.self_attn");
        let (uk, uv) = absorb(&format!("{a}.kv_b_proj.weight"))?;
        layers.push(MlaLayerWeights {
            input_norm: one(&format!("{p}.input_layernorm.weight"), &[h])?,
            post_norm: one(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
            attn_conv: matrix(&[(&format!("{p}.attention_conv.kernel_projection.weight"), 4 * h / cfg.group)], h)?,
            attn_base: one(&format!("{p}.attention_conv.base_kernel"), &[2, 2, h])?,
            mlp_conv: matrix(&[(&format!("{p}.mlp_conv.kernel_projection.weight"), 4 * h / cfg.group)], h)?,
            mlp_base: one(&format!("{p}.mlp_conv.base_kernel"), &[2, 2, h])?,
            qkv_a: attention(&[(&format!("{a}.q_a_proj.weight"), m.q_rank),
                (&format!("{a}.kv_a_proj_with_mqa.weight"), m.kv_rank + m.rope)], h)?,
            q_norm: one(&format!("{a}.q_a_layernorm.weight"), &[m.q_rank])?,
            kv_norm: one(&format!("{a}.kv_a_layernorm.weight"), &[m.kv_rank])?,
            q_b: attention(&[(&format!("{a}.q_b_proj.weight"), m.heads * (m.nope + m.rope))], m.q_rank)?,
            uk, uv, output: attention(&[(&format!("{a}.o_proj.weight"), h)], m.heads * m.value)?,
            gate_up: matrix(&[(&format!("{p}.mlp.gate_proj.weight"), cfg.intermediate),
                (&format!("{p}.mlp.up_proj.weight"), cfg.intermediate)], h)?,
            down: matrix(&[(&format!("{p}.mlp.down_proj.weight"), h)], cfg.intermediate)?,
        });
    }
    Ok(MlaWeights { layers, fc: matrix(&[("fc.weight", h)], cfg.taps.len() * h)?,
        hidden_norm: one("hidden_norm.weight", &[h])?, norm: one("norm.weight", &[h])?,
        selector: (matrix(&[("candidate_selector.hidden_projection.weight", cfg.selector_rank)], h)?,
            one("candidate_selector.predecessor_codebook", &[cfg.vocab, cfg.selector_rank])?,
            one("candidate_selector.successor_codebook", &[cfg.vocab, cfg.selector_rank])?),
    })
}

/// Config-only MLA requests, in the real loader's binding order.
pub fn mla_arena_requests(cfg: &crate::formats::dflash::MlaDflashConfig,
    representation: GlmDraftRepresentation) -> Result<Vec<(String, usize)>> {
    let requests = std::cell::RefCell::new(Vec::new());
    mla_weight_recipe(cfg,
        |parts| {
            let bytes = parts.iter().try_fold(0usize, |sum, (_, shape)| {
                let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
                    .context("MLA tensor extent overflow")?;
                sum.checked_add(bytes).context("MLA joined tensor extent overflow")
            })?;
            requests.borrow_mut().push((parts.iter().map(|(name, _)| *name)
                .collect::<Vec<_>>().join("+"), bytes.max(256)));
            Ok(())
        },
        |parts, cols| {
            let rows = parts.iter().try_fold(0usize, |n, (_, rows)| n.checked_add(*rows))
                .context("MLA matrix rows overflow")?;
            let values = rows.checked_mul(cols).context("MLA matrix extent overflow")?;
            let name = parts.iter().map(|(name, _)| *name).collect::<Vec<_>>().join("+");
            if representation == GlmDraftRepresentation::Bf16Only {
                requests.borrow_mut().push((name,
                    values.checked_mul(2).context("MLA BF16 matrix extent overflow")?.max(256)));
            } else {
                ensure!(rows > 0 && cols > 0 && rows % 16 == 0 && cols % 128 == 0,
                    "DFlash recipe FP8 shape {rows}/{cols}");
                requests.borrow_mut().extend([(format!("{name}.payload"), values),
                    (format!("{name}.scales"), values / 128 * 4)]);
            }
            Ok(())
        },
        |name| {
            let m = &cfg.mla;
            let (_, _, uk, uv) = absorb_geometry(m.heads, m.nope, m.value, m.kv_rank)?;
            requests.borrow_mut().extend([(format!("{name}.uk"), m.heads * uk),
                (format!("{name}.uv"), m.heads * uv)]);
            Ok(((), ()))
        })?;
    Ok(requests.into_inner())
}

/// Complete MLA body FP8 destinations. This is additive to the historical
/// representation until its lossless/performance qualification enables it.
/// Native auxiliary/prediction tensors retain their exact BF16 storage.
pub fn mla_complete_fp8_arena_requests(cfg: &crate::formats::dflash::MlaDflashConfig)
    -> Result<Vec<(String, usize)>> {
    use super::draft_representation::{draft_weight_role, DraftWeightRole};
    ensure!(cfg.kind.is_dflash(), "complete MLA FP8 body requires DFlash2");
    let mut requests = Vec::new();
    let mut qkv = Vec::new();
    for (name, shape) in cfg.tensor_shapes() {
        let values = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
            .context("MLA complete FP8 extent overflow")?;
        if name.ends_with(".self_attn.q_a_proj.weight") || name.ends_with(".self_attn.kv_a_proj_with_mqa.weight") {
            qkv.push((name, values));
            continue;
        }
        if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
            if name.ends_with(".mlp.gate_proj.weight") {
                let up = name.replace(".gate_proj.weight", ".up_proj.weight");
                let joined = format!("{name}+{up}");
                requests.extend([(format!("{joined}.payload"), values * 2),
                    (format!("{joined}.scales"), values * 2 / 128 * 4)]);
            }
            continue;
        }
        if name.ends_with(".self_attn.kv_b_proj.weight") {
            let root = &name;
            let m = &cfg.mla;
            for (part, count) in [("uk", m.heads * m.kv_rank * m.nope), ("uv", m.heads * m.value * m.kv_rank)] {
                requests.extend([(format!("{root}.{part}.payload"), count),
                    (format!("{root}.{part}.scales"), count / 64 * 4)]);
            }
            continue;
        }
        match draft_weight_role(&name).with_context(|| format!("MLA tensor {name} has no precision role"))? {
            DraftWeightRole::BodyLinear => {
                ensure!(shape.len() == 2 && shape[0] % 16 == 0 && shape[1] % 128 == 0,
                    "MLA complete FP8 matrix alignment: {name}");
                requests.extend([(format!("{name}.payload"), values), (format!("{name}.scales"), values / 128 * 4)]);
            }
            _ => requests.push((name, (values * 2).max(256))),
        }
    }
    for pair in qkv.chunks_exact(2) {
        let name = format!("{}+{}", pair[0].0, pair[1].0);
        let values = pair[0].1.checked_add(pair[1].1).context("MLA joined QKV overflow")?;
        requests.extend([(format!("{name}.payload"), values), (format!("{name}.scales"), values / 128 * 4)]);
    }
    ensure!(qkv.len() == cfg.windows.len() * 2, "MLA joined QKV inventory incomplete");
    Ok(requests)
}

/// The same MLA requests, checked against native BF16 checkpoint headers.
pub fn mla_checkpoint_arena_requests(snapshot: &Path, representation: GlmDraftRepresentation)
    -> Result<Vec<(String, usize)>> {
    let cfg = crate::formats::dflash::MlaDflashConfig::read(snapshot)?;
    let requests = mla_arena_requests(&cfg, representation)?;
    validate_bf16_headers(snapshot, cfg.tensor_shapes())?;
    Ok(requests)
}

pub fn absorb_geometry(heads: usize, nope: usize, value: usize, latent: usize) -> Result<(usize, usize, usize, usize)> {
    ensure!(heads > 0 && nope > 0 && value > 0 && latent > 0, "empty draft absorption geometry");
    let rows = nope.checked_add(value).context("draft absorption head rows overflow")?;
    let width = latent.checked_mul(2).context("draft absorption row bytes overflow")?;
    let head = |rows: usize| rows.checked_mul(width).context("draft absorption head bytes overflow");
    let (source, uk, uv) = (head(rows)?, head(nope)?, head(value)?);
    for bytes in [rows, source, uk, uv] {
        heads.checked_mul(bytes).context("draft absorption total extent overflow")?;
    }
    ensure!(source <= crate::shared::dense_read::STEP_BYTES, "draft absorption head exceeds staging bound");
    Ok((rows, source, uk, uv))
}

pub use super::draft_physical_layout::{draft_device_requests, GlmDflashPhysicalLayout};

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn glm_body_fp8_recipe_and_shared_roles_agree_for_every_operand() -> Result<()> {
        use crate::families::glm5::draft_representation::{draft_weight_role, DraftWeightRole};
        let cfg = GlmDflashConfig { hidden:256,intermediate:256,layers:2,heads:2,kv_heads:1,head_dim:128,
            eps:1e-6,theta:10000.,block:8,group:16,mask_token:0,rank:256,taps:vec![0,1],vocab:32,
            window:2048,row_window:true,sample_from_anchor:false, post_selector_softcap: None };
        let native = gqa_arena_requests(&cfg,GlmDraftRepresentation::Bf16Only,0,1)?
            .into_iter().collect::<std::collections::BTreeMap<_,_>>();
        let fp8 = gqa_arena_requests(&cfg,GlmDraftRepresentation::Fp8Only,0,1)?
            .into_iter().collect::<std::collections::BTreeMap<_,_>>();
        gqa_weight_recipe(&cfg,0,1,|name,_| {
            assert_ne!(draft_weight_role(name),Some(DraftWeightRole::BodyLinear),"{name}");
            assert_eq!(fp8[name],native[name]);
            assert!(!fp8.contains_key(&format!("{name}.payload")));
            Ok(())
        },|parts,cols,_| {
            for (name,_) in parts { assert_eq!(draft_weight_role(name),Some(DraftWeightRole::BodyLinear),"{name}"); }
            let name=parts.iter().map(|(name,_)|*name).collect::<Vec<_>>().join("+");
            let values=parts.iter().map(|(_,rows)|rows*cols).sum::<usize>();
            assert_eq!(fp8[&format!("{name}.payload")],values);
            assert_eq!(fp8[&format!("{name}.scales")],values/128*4);
            assert!(!fp8.contains_key(&name));
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn gqa_checkpoint_and_config_requests_agree_and_reject_invalid_headers() -> Result<()> {
        use crate::plan::testing::{t, write_safetensors};
        let value = serde_json::json!({"hidden_size":256, "intermediate_size":256,
            "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":2,
            "head_dim":128, "rms_norm_eps":1e-6, "rope_theta":10000.0,
            "sliding_window":2048, "vocab_size":16,
            "dflash_config":{"block_size":8, "conv_group_size":16, "conv_kernel_size":2,
                "mask_token_id":0, "selector_rank":256, "selector_top_k":16,
                "target_layer_ids":[0]}});
        let cfg = GlmDflashConfig::from_value(&value)?;
        assert_eq!(cfg.post_selector_softcap, None);
        let mut post = value.clone();
        post["dflash_config"]["selector_softcap_mode"] = "after_correction".into();
        assert!(GlmDflashConfig::from_value(&post).is_err());
        post["dflash_config"]["final_logit_softcapping"] = 30.0.into();
        assert_eq!(GlmDflashConfig::from_value(&post)?.post_selector_softcap, Some(30.0));
        post["dflash_config"]["selector_softcap_mode"] = "unknown".into();
        assert!(GlmDflashConfig::from_value(&post).is_err());
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join("config.json"), serde_json::to_vec(&value)?)?;
        let shapes = cfg.tensor_shapes();
        let path = root.path().join("model.safetensors");
        write_safetensors(&path, &shapes.iter().map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
        for mode in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
            for ranks in [1, 2] { for rank in 0..ranks {
                let requests = gqa_arena_requests(&cfg, mode, rank, ranks)?;
                assert_eq!(requests, gqa_checkpoint_arena_requests(root.path(), mode, rank, ranks)?);
                assert_eq!(requests.iter().any(|(name, _)| name.contains("candidate_selector")), rank == 0);
                let arena = crate::serving_capacity::WeightArenaLayout::new("draft", requests)?;
                assert!(arena.bytes() >= arena.payload_bytes());
            }}
        }
        assert!(gqa_arena_requests(&cfg, GlmDraftRepresentation::Bf16Only, 0, 0).is_err());
        assert!(gqa_arena_requests(&cfg, GlmDraftRepresentation::Bf16Only, 2, 2).is_err());
        let mut invalid = shapes.iter().map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>();
        invalid.pop();
        write_safetensors(&path, &invalid);
        assert!(gqa_checkpoint_arena_requests(root.path(), GlmDraftRepresentation::Bf16Only, 0, 1).is_err());
        write_safetensors(&path, &shapes.iter().map(|(name, shape)| t(name, "F32", shape)).collect::<Vec<_>>());
        assert!(gqa_checkpoint_arena_requests(root.path(), GlmDraftRepresentation::Bf16Only, 0, 1).is_err());
        Ok(())
    }
}
