//! Shared DFlash2 block runtime, currently used by GLM 5.3 (incoai/GLM-5.3-DFlash2: six layers,
//! 64 query heads) and GLM 5.3 Flash (incoai/GLM-5.3-Flash-DFlash2: five
//! layers, 32 query heads over 8 KV heads).
//!
//! Every target step taps the outputs of the `target_layer_ids` layers
//! (GLM 5.3: 5, 19, 33, 47, 61, 75; Flash: 5, 14, 24, 33, 42, where the tap is
//! the mean of the four mHC streams, [`DflashDrafter::tap_streams`]) into
//! [`DflashDrafter::taps`]. Once a
//! step's rows are committed, [`DflashDrafter::update`] projects their taps
//! (`hidden_norm(fc(taps))`) into every draft layer's context K/V, a ring of
//! the sequence's last 2048 positions. [`DflashDrafter::draft`] then runs the
//! Qwen3-style layers (two-tap dynamic convolutions around attention and
//! MLP) over the block `[anchor, mask x 7]` at the anchor's position,
//! non-causally against the ring, takes the top 16 of the target head's
//! logits per drafted row and walks the candidate selector greedily, all on
//! the device. Drafts only steer speculation; the verify step keeps output
//! identical to plain greedy decoding.
//!
//! The checkpoint BF16 GEMMs stay BF16 by default. Explicit FP8 convenience
//! quantization selects one E4M3 representation at load, with FP32 scales
//! per output row and 128-wide K block. The existing W8A16 kernel covers
//! every draft/context row count; no BF16 fallback or packed head copy is
//! retained. Both modes borrow the target's one resident head
//! ([`TargetHead`]): its BF16 matrix, or its own launcher (GLM 5.3 Flash's
//! FP8-only head). The target verifies every proposal. After Hugh Madden's
//! glm53f-afd (MIT, v1.1.0 16de2a6).
use crate::shared::fp8_linear::{self, Fp8Weight};
use crate::shared::memory::DeviceAllocation;
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::programs::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use cuteafd_loader::families::glm5::draft_representation::{
    GlmDraftCapacity, GlmDraftGeometry, GlmDraftRepresentation, GlmDraftRuntimeLayout,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

type Dev<'a> = DeviceAllocation<'a>;

/// The drafters' exact vocabulary head (`--draft-head exact`) of `rows` rows: 2..=24 rows on the
/// few-row FP32 kernel (one read of the head per 8 rows), others on the pedantic cuBLAS head;
/// CUTEAFD_GLM_HEAD=cublas keeps every row count on cuBLAS. Target logits take
/// [`crate::shared::target_head`] instead.
///
/// # Safety
/// `x` [rows, width] BF16, `weight` [vocab, width] BF16 and `logits` [rows,
/// vocab] FP32 are live device buffers on `stream`'s device; `head` was
/// created for at least `rows` rows of this width and vocabulary.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_head(library: &NativeLibrary, head: &cuteafd_ffi::programs::VocabularyHead<'_>,
    x: *const c_void, weight: *const c_void, logits: *mut f32, rows: usize, width: usize, vocab: usize,
    stream: *mut c_void) -> Result<()> {
    static CUBLAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let cublas = *CUBLAS.get_or_init(|| std::env::var("CUTEAFD_GLM_HEAD").is_ok_and(|v| v == "cublas"));
    if head_route(DraftHead::Exact, rows, cublas) == HeadRoute::FewRows {
        // SAFETY: the caller's contract.
        return unsafe { library.vocab_head_rows(x, weight, logits, rows, width, vocab, stream) };
    }
    // SAFETY: the caller's contract.
    unsafe { head.launch(x.cast(), weight.cast(), logits, rows as u32, stream) }
}

/// Diagnostic drafter head arithmetic (GLM 5.3 Flash's golden/replay `--draft-head`).
/// Serving uses the shared tensor head independently of this diagnostic selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DraftHead {
    /// As the target's head ([`launch_head`]): the few-row FP32 kernel for 2..=24 rows, one read
    /// of the head per 8 rows, and the pedantic FP32 cuBLAS GEMM (CUDA cores) past them.
    #[default]
    Exact,
    /// The few-row FP32 kernel for one draft block (up to 8 rows, one read of the head), and from
    /// [`DRAFT_HEAD_TENSOR_ROWS`] rows a BF16 tensor-core GEMM with FP32 accumulation that reads
    /// the head once for every row count.
    Tensor,
}

/// Rows from which [`DraftHead::Tensor`] takes tensor cores: past one pass of the few-row kernel
/// (two draft blocks and more; DFlash2 and dSpark blocks are 8 rows).
pub(crate) const DRAFT_HEAD_TENSOR_ROWS: usize = cuteafd_ffi::VOCAB_HEAD_ROWS_PASS + 1;

/// The kernel a vocabulary head of `rows` rows runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeadRoute {
    /// `vocab_head_rows`: FP32 products and accumulation, one read of the head per 8 rows.
    FewRows,
    /// cuBLAS with pedantic FP32 accumulation on CUDA cores.
    Pedantic,
    /// cuBLAS BF16 tensor cores with FP32 accumulation.
    TensorCores,
}

/// The route [`launch_head`] (`mode` [`DraftHead::Exact`]) or [`launch_draft_head`] takes for
/// `rows` rows (`cublas`: CUTEAFD_GLM_HEAD=cublas, every exact row count on cuBLAS).
pub(crate) fn head_route(mode: DraftHead, rows: usize, cublas: bool) -> HeadRoute {
    if mode == DraftHead::Tensor && rows >= DRAFT_HEAD_TENSOR_ROWS {
        HeadRoute::TensorCores
    } else if !cublas && (2..=cuteafd_ffi::VOCAB_HEAD_ROWS_MAX).contains(&rows) {
        HeadRoute::FewRows
    } else {
        HeadRoute::Pedantic
    }
}

/// A drafter's vocabulary head of `rows` rows under `mode` ([`DraftHead`]); [`DraftHead::Exact`]
/// is [`launch_head`].
///
/// # Safety
/// As [`launch_head`].
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_draft_head(library: &NativeLibrary, head: &cuteafd_ffi::programs::VocabularyHead<'_>,
    x: *const c_void, weight: *const c_void, logits: *mut f32, rows: usize, width: usize, vocab: usize,
    stream: *mut c_void, mode: DraftHead) -> Result<()> {
    if head_route(mode, rows, false) == HeadRoute::TensorCores {
        // SAFETY: the caller's contract.
        return unsafe { head.launch_tensor_op(x.cast(), weight.cast(), logits, rows as u32, stream) };
    }
    // SAFETY: the caller's contract.
    unsafe { launch_head(library, head, x, weight, logits, rows, width, vocab, stream) }
}

#[cfg(test)]
mod draft_head_tests {
    use super::*;

    /// The exact route is the target's: one row and past 24 rows on pedantic cuBLAS, 2..=24 on
    /// the few-row kernel, everything on cuBLAS under CUTEAFD_GLM_HEAD=cublas.
    #[test]
    fn the_exact_route_is_the_target_heads() {
        for rows in [2, 8, 9, 16, 24] {
            assert_eq!(head_route(DraftHead::Exact, rows, false), HeadRoute::FewRows, "{rows}");
        }
        for rows in [1, 25, 64, 127, 128] {
            assert_eq!(head_route(DraftHead::Exact, rows, false), HeadRoute::Pedantic, "{rows}");
        }
        for rows in [1, 8, 24, 128] {
            assert_eq!(head_route(DraftHead::Exact, rows, true), HeadRoute::Pedantic, "{rows}");
        }
    }

    /// One draft block keeps the exact route (C1 drafts do not change); two blocks and more take
    /// tensor cores, which read the head once for every row count.
    #[test]
    fn tensor_drafts_take_tensor_cores_from_two_draft_blocks() {
        assert_eq!(DRAFT_HEAD_TENSOR_ROWS, 9);
        for rows in 1..DRAFT_HEAD_TENSOR_ROWS {
            for cublas in [false, true] {
                assert_eq!(head_route(DraftHead::Tensor, rows, cublas), head_route(DraftHead::Exact, rows, cublas));
            }
        }
        assert_eq!(head_route(DraftHead::Tensor, 8, false), HeadRoute::FewRows);
        for rows in [9, 16, 24, 25, 32, 64, 127, 128] {
            for cublas in [false, true] {
                assert_eq!(head_route(DraftHead::Tensor, rows, cublas), HeadRoute::TensorCores, "{rows}");
            }
        }
    }
}


/// Context entries each draft layer attends to (the checkpoint's sliding window).
pub(crate) const RING: usize = 2048;
/// Tapped rows one target step keeps (a prefill chunk's tail, or a verify step).
pub(crate) const TAP_ROWS: usize = RING;
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
}

/// The drafter geometry the kernels are written for, read from `config.json`.
#[derive(Debug, Clone)]
pub(crate) struct DflashConfig {
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
}

impl DflashConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)
            .context("parsing the DFlash2 config.json")?;
        let d = &raw.dflash_config;
        ensure!(matches!(raw.head_dim, 128 | 64) && raw.num_key_value_heads > 0
            && raw.num_attention_heads % raw.num_key_value_heads == 0
            && raw.num_attention_heads / raw.num_key_value_heads * d.block_size <= 64,
            "DFlash2 kernels take 128- or 64-wide heads and at most 64 queries per kv head and block \
             (got {} heads, {} kv, block {})", raw.num_attention_heads, raw.num_key_value_heads, d.block_size);
        ensure!(d.conv_kernel_size == 2 && d.selector_rank == 256 && d.selector_top_k == 16,
            "DFlash2 kernels take two-tap convolutions and a rank-256 top-16 selector");
        ensure!(raw.sliding_window == RING, "DFlash2 window {} (the ring holds {RING})", raw.sliding_window);
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
        })
    }

    fn mla(cfg: &cuteafd_loader::formats::dflash::MlaDflashConfig) -> Self {
        Self { hidden: cfg.mla.hidden, intermediate: cfg.intermediate, layers: cfg.windows.len(),
            heads: cfg.mla.heads, kv_heads: cfg.mla.heads, head_dim: cfg.mla.nope + cfg.mla.rope,
            eps: cfg.eps as f32, theta: 0.0, block: cfg.block, group: cfg.group,
            mask_token: cfg.mask_token, rank: cfg.selector_rank, taps: cfg.taps.clone(), vocab: cfg.vocab,
            window: 0, row_window: false }
    }

    pub fn drafts(&self) -> usize {
        self.block - 1
    }

    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    fn conv_width(&self) -> usize {
        4 * self.hidden / self.group
    }

    fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
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

    pub(crate) fn runtime_layout(&self, mode: GlmDraftRepresentation, capacity: GlmDraftCapacity)
        -> Result<GlmDraftRuntimeLayout> {
        Ok(GlmDraftRuntimeLayout::new(GlmDraftGeometry {
            hidden: self.hidden as u64, intermediate: self.intermediate as u64, layers: self.layers as u64,
            heads: self.heads as u64, kv_heads: self.kv_heads as u64, head_dim: self.head_dim as u64,
            taps: self.taps.len() as u64, vocab: self.vocab as u64, conv_group: self.group as u64,
            selector_rank: self.rank as u64,
        }, mode, capacity, TAP_ROWS)?)
    }
}

/// Exactly one immutable resident representation of a GEMM.
enum DraftWeight<'a> {
    Bf16(Dev<'a>),
    Fp8(Fp8Weight<'a>),
}

struct DraftLayer<'a> {
    input_norm: Dev<'a>,
    post_norm: Dev<'a>,
    attn_conv: DraftWeight<'a>,
    attn_base: Dev<'a>,
    mlp_conv: DraftWeight<'a>,
    mlp_base: Dev<'a>,
    attention: DraftAttention<'a>,
    gate_up: DraftWeight<'a>,
    down: DraftWeight<'a>,
}

#[allow(dead_code)]
enum DraftAttention<'a> {
    Gqa(GqaAttention<'a>),
    Mla(MlaAttention<'a>),
}

impl<'a> DraftAttention<'a> {
    fn gqa(&self) -> Result<&GqaAttention<'a>> {
        match self {
            Self::Gqa(attention) => Ok(attention),
            Self::Mla(_) => anyhow::bail!("MLA draft context/forward binding is not installed"),
        }
    }
}

struct GqaAttention<'a> {
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: DraftWeight<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
    o: DraftWeight<'a>,
    k_ring: Dev<'a>,
    v_ring: Dev<'a>,
}

#[allow(dead_code)]
struct MlaAttention<'a> {
    program: cuteafd_ffi::mla_draft::MlaDraft<'a>,
    qkv_a: Dev<'a>, q_norm: Dev<'a>, kv_norm: Dev<'a>,
    q_b: Dev<'a>, uk: Dev<'a>, uv: Dev<'a>, output: Dev<'a>,
    cache: Dev<'a>, page_table: Dev<'a>,
}

impl MlaAttention<'_> {
    fn weights(&self) -> cuteafd_ffi::mla_draft::MlaDraftWeights {
        cuteafd_ffi::mla_draft::MlaDraftWeights { qkv_a: self.qkv_a.buffer,
            q_a_norm: self.q_norm.buffer, kv_a_norm: self.kv_norm.buffer,
            q_b: self.q_b.buffer, uk: self.uk.buffer, uv: self.uv.buffer, output: self.output.buffer }
    }
}

/// Admitted native MLA arenas, consumed into the common executor's owners.
/// This storage has no independent forward/head path and owns no target weights.
#[allow(dead_code)]
pub(crate) struct MlaDraftStorage<'a> {
    layout: cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout,
    buffers: Vec<Dev<'a>>,
}

#[allow(dead_code)]
impl<'a> MlaDraftStorage<'a> {
    /// The caller must first admit `layout.demands(target_gpu)` and validate the
    /// native programs. All native size queries finish before the first allocation.
    pub(crate) fn allocate(library: &'a NativeLibrary,
        layout: cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout) -> Result<Self> {
        ensure!(library.sm_count()? == layout.sm_count(), "MLA draft layout belongs to another SM count");
        let expected = |name: &str| -> Result<usize> {
            layout.buffers().iter().find(|b| b.name == name).map(|b| b.bytes)
                .with_context(|| format!("MLA draft missing admitted {name}"))
        };
        ensure!(expected("head_workspace")? == VOCABULARY_HEAD_WORKSPACE,
            "MLA draft vocabulary workspace disagrees with admission");
        let drafted = layout.capacity().block_rows - layout.capacity().max_batch_sequences;
        ensure!(library.glm_dflash_topk_workspace(drafted)?.max(256) == expected("topk_workspace")?,
            "MLA draft top-k workspace disagrees with admission");
        if let Some(scratch) = layout.fp8_scratch() {
            let mut bytes = 256;
            for shape in &scratch.shapes {
                let (k, n) = (usize::try_from(shape.k)?, usize::try_from(shape.n)?);
                for mode in 0..=layout.fp8_mode() {
                    let queried = if mode == 0 { library.fp8_w8a16_workspace(scratch.rows, k, n)? }
                        else { library.fp8_linear_workspace(scratch.rows, k, n, i32::from(mode))? };
                    bytes = bytes.max(queried);
                }
            }
            ensure!(bytes == expected("fp8_scratch")?, "MLA draft FP8 workspace disagrees with admission");
        }
        let buffers = layout.allocate_with(|b| DeviceAllocation::named(library, b.bytes, "drafter/mla"))?;
        Ok(Self { layout, buffers })
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        let index = self.layout.buffers().iter().position(|b| b.name == name)
            .with_context(|| format!("MLA draft unknown admitted buffer {name}"))?;
        Ok(self.buffers[index].buffer)
    }

    fn into_owners(self) -> Result<AdmittedBuffers<Dev<'a>>> {
        for (recipe, owner) in self.layout.buffers().iter().zip(&self.buffers) {
            ensure!(owner.buffer.bytes == recipe.bytes, "MLA draft {} extent differs from admission", recipe.name);
        }
        AdmittedBuffers::new(self.layout.buffers(), self.buffers)
    }
}

/// Move, never alias or reallocate, each named admitted owner exactly once.
#[allow(dead_code)]
struct AdmittedBuffers<T> {
    owners: HashMap<String, T>,
}

#[allow(dead_code)]
impl<T> AdmittedBuffers<T> {
    fn new(recipes: &[cuteafd_loader::formats::dflash::MlaDraftBuffer], owners: Vec<T>) -> Result<Self> {
        ensure!(recipes.len() == owners.len(), "MLA draft owner count differs from admission");
        let mut named = HashMap::with_capacity(owners.len());
        for (recipe, owner) in recipes.iter().zip(owners) {
            ensure!(!named.contains_key(&recipe.name), "MLA draft duplicate admitted owner {}", recipe.name);
            named.insert(recipe.name.clone(), owner);
        }
        Ok(Self { owners: named })
    }

    fn take(&mut self, name: &str) -> Result<T> {
        self.owners.remove(name).with_context(|| format!("MLA draft owner {name} missing or already consumed"))
    }

    fn finish(self) -> Result<()> {
        ensure!(self.owners.is_empty(), "MLA draft admitted owners were not consumed: {:?}", self.owners.keys());
        Ok(())
    }
}

enum DraftContext<'a> {
    Gqa(GqaContext<'a>),
    Mla(MlaContext<'a>),
}

struct GqaContext<'a> { kv: Dev<'a>, slots: Dev<'a> }

// Keep exhaustion downcastable across the family adapter's boxed error boundary.
fn box_draft_page_error(error: anyhow::Error) -> cuteafd_engine::prefix::BoxError {
    match error.downcast::<cuteafd_engine::prefix::DraftPageError>() {
        Ok(error) => Box::new(error),
        Err(error) => error.into(),
    }
}

struct MlaPagedState {
    rings: cuteafd_loader::formats::dflash::MlaDraftCacheState,
    pool: cuteafd_engine::prefix::DraftPagePool,
    bindings: Vec<Option<cuteafd_engine::prefix::DraftBinding>>,
}

impl MlaPagedState {
    fn new(layout: &cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout) -> Result<Self> {
        use std::hash::{Hash, Hasher};
        let mut identity = std::collections::hash_map::DefaultHasher::new();
        (layout.program().context, layout.target_hidden(), layout.target_vocab(), layout.target_taps(),
            layout.paged_cache().target_tokens, layout.paged_cache().record_bytes).hash(&mut identity);
        for layer in &layout.cache().layers { layer.window().hash(&mut identity); }
        Ok(Self { rings: cuteafd_loader::formats::dflash::MlaDraftCacheState::new(layout.cache().clone())?,
            pool: cuteafd_engine::prefix::DraftPagePool::new(layout.paged_cache().target_tokens,
                layout.capacity().context_slots, layout.program().context, identity.finish())?,
            bindings: vec![None; layout.capacity().context_slots] })
    }

    fn reset(&mut self, slot: usize, origin: usize) -> Result<()> {
        // Validate the ring geometry before dropping the previous request's ownership.
        let mut rings = self.rings.clone();
        rings.reset(slot, origin)?;
        self.discard(slot)?;
        self.bindings[slot] = Some(self.pool.bind(slot, origin)?);
        self.rings = rings;
        Ok(())
    }

    fn discard(&mut self, slot: usize) -> Result<()> {
        let binding = self.bindings.get_mut(slot).context("MLA draft slot outside capacity")?.take();
        if let Some(binding) = binding { self.pool.release(binding)?; }
        self.rings.discard(slot)?;
        Ok(())
    }

    fn block_batches(&self, layer: usize,
        requests: &[cuteafd_loader::formats::dflash::MlaDraftCacheRequest],
        reservations: &[cuteafd_engine::prefix::DraftReservation],
        full: bool) -> Result<Vec<cuteafd_loader::formats::dflash::MlaDraftCacheBatch>> {
        let mut batches = self.rings.block_batches(layer, requests)?;
        if !full { return Ok(batches); }
        for batch in &mut batches {
            batch.slots.clear();
            batch.page_table.clear();
            for index in batch.first_sequence..batch.first_sequence + batch.lengths.len() {
                batch.slots.extend(self.pool.destinations(reservations[index])?);
                let pages = self.pool.page_table(reservations[index])?;
                ensure!(pages.len() <= batch.page_table_width, "MLA draft pooled table exceeds context view");
                batch.page_table.extend_from_slice(&pages);
                batch.page_table.resize(batch.page_table.len() + batch.page_table_width - pages.len(), 0);
            }
        }
        Ok(batches)
    }

    fn reserve(&mut self, slot: usize, start: usize, rows: usize)
        -> Result<cuteafd_engine::prefix::DraftReservation> {
        let binding = self.bindings.get(slot).copied().flatten().context("MLA draft unbound slot")?;
        Ok(self.pool.reserve(binding, start, rows)?)
    }
}

/// A failed enqueue cannot recycle any destinations until the owned stream retires.
struct MlaPageLease<'c, 'a> {
    context: &'c MlaContext<'a>,
    library: &'a NativeLibrary,
    stream: *mut c_void,
    reservations: Vec<cuteafd_engine::prefix::DraftReservation>,
}

impl MlaPageLease<'_, '_> {
    fn commit(mut self) -> Result<()> {
        let mut state = self.context.state.borrow_mut();
        while let Some(&reservation) = self.reservations.last() {
            state.pool.commit(reservation, reservation.end() - reservation.start())?;
            self.reservations.pop();
        }
        Ok(())
    }
}

impl Drop for MlaPageLease<'_, '_> {
    fn drop(&mut self) {
        if self.reservations.is_empty() { return; }
        // SAFETY: this lease retains its destinations on the drafter's live owned stream.
        if unsafe { self.library.cuda_stream_synchronize(self.stream) }.is_err() {
            // Failed retirement deliberately retains ownership: never reuse live storage.
            return;
        }
        let mut state = self.context.state.borrow_mut();
        for reservation in self.reservations.drain(..) {
            let _ = state.pool.commit(reservation, 0);
        }
    }
}

struct MlaContext<'a> {
    cfg: cuteafd_loader::formats::dflash::MlaDflashConfig,
    layout: cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout,
    state: RefCell<MlaPagedState>,
    slots: Dev<'a>,
}

/// Buffers of one draft step for up to `sequences` sequences.
struct Workspace<'a> {
    sequences: usize,
    h: Dev<'a>,
    n: Dev<'a>,
    conv: Dev<'a>,
    dynamic: Dev<'a>,
    attention: AttentionWorkspace<'a>,
    delta: Dev<'a>,
    gate_up: Dev<'a>,
    act: Dev<'a>,
    logits: Dev<'a>,
    unary: Dev<'a>,
    candidates: Dev<'a>,
    projected: Dev<'a>,
    anchors: Dev<'a>,
    /// The block's input ids (anchor, then mask tokens) for the device embedding gather.
    ids: Dev<'a>,
    tokens: Dev<'a>,
    features: Dev<'a>,
    /// Pinned landing of `tokens` then `features`: both download behind the
    /// round's one host wait.
    landing: crate::shared::memory::HostAllocation<'a>,
    positions: Dev<'a>,
    topk_workspace: Dev<'a>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

#[allow(dead_code)]
enum AttentionWorkspace<'a> {
    Gqa(GqaWorkspace<'a>),
    Mla(MlaWorkspace<'a>),
}

impl<'a> AttentionWorkspace<'a> {
    fn gqa(&self) -> Result<&GqaWorkspace<'a>> {
        match self {
            Self::Gqa(workspace) => Ok(workspace),
            Self::Mla(_) => anyhow::bail!("MLA draft context/forward binding is not installed"),
        }
    }
}

struct GqaWorkspace<'a> {
    qkv: Dev<'a>, q: Dev<'a>, k: Dev<'a>, v: Dev<'a>, attn: Dev<'a>,
    tables: Dev<'a>, scratch: Dev<'a>,
}

#[allow(dead_code)]
struct MlaWorkspace<'a> {
    query: Dev<'a>, q_residual: Dev<'a>, attention: Dev<'a>, lse: Dev<'a>,
    scratch: Dev<'a>, lengths: Dev<'a>, query_offsets: Dev<'a>,
    slots: Dev<'a>, positions: Dev<'a>, inverse_frequencies: Dev<'a>,
}

impl MlaWorkspace<'_> {
    #[allow(clippy::too_many_arguments)]
    fn batch(&self, context: &MlaContext<'_>, layer: &MlaAttention<'_>, rows: usize,
        requests: usize, page_table_width: usize, x: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer,
        slots: CuteafdDeviceBuffer, positions: CuteafdDeviceBuffer) -> cuteafd_ffi::mla_draft::MlaDraftBatch {
        cuteafd_ffi::mla_draft::MlaDraftBatch { rows, requests, page_table_width,
            cache_pages: layer.cache.buffer.bytes / (64 * layer.program.cache_format().record_bytes()), active_splits: layer.program.splits(),
            rotary_factor: context.cfg.rotary.attention_factor, attention_scale: context.cfg.attention_scale(),
            x, slots, positions, inv_freq: self.inverse_frequencies.buffer, cache: layer.cache.buffer,
            page_table: layer.page_table.buffer, lengths: self.lengths.buffer, query_offsets: self.query_offsets.buffer,
            query: self.query.buffer, q_resid: self.q_residual.buffer, attention: self.attention.buffer,
            lse: self.lse.buffer, output, producer_scratch: self.scratch.buffer,
            attention_scratch: self.scratch.buffer, output_scratch: self.scratch.buffer }
    }
}

impl<'a> Workspace<'a> {
    fn admitted_mla(library: &'a NativeLibrary, cfg: &DflashConfig, sequences: usize,
        owners: &mut AdmittedBuffers<Dev<'a>>) -> Result<Self> {
        let head_workspace = owners.take("head_workspace")?;
        // SAFETY: the admitted head workspace outlives this borrowed native handle.
        let head = unsafe { library.vocabulary_head_rows(head_workspace.buffer.ptr,
            cfg.hidden as u32, (sequences * cfg.block) as u32, cfg.vocab as u32)? };
        Ok(Self { sequences, head, _head_workspace: head_workspace,
            h: owners.take("h")?, n: owners.take("n")?, conv: owners.take("conv")?,
            dynamic: owners.take("dynamic")?, delta: owners.take("delta")?,
            gate_up: owners.take("gate_up")?, act: owners.take("act")?, logits: owners.take("logits")?,
            unary: owners.take("unary")?, candidates: owners.take("candidates")?,
            projected: owners.take("projected")?, anchors: owners.take("anchors")?,
            ids: owners.take("ids")?, tokens: owners.take("tokens")?, features: owners.take("features")?,
            // Pinned host memory (drafted x 20 bytes), outside the device admission.
            landing: crate::shared::memory::HostAllocation::new(library, sequences * cfg.drafts() * 20)?,
            positions: owners.take("positions")?, topk_workspace: owners.take("topk_workspace")?,
            attention: AttentionWorkspace::Mla(MlaWorkspace {
                query: owners.take("query")?, q_residual: owners.take("q_residual")?,
                attention: owners.take("attention")?, lse: owners.take("lse")?,
                scratch: owners.take("native_scratch")?, lengths: owners.take("lengths")?,
                query_offsets: owners.take("query_offsets")?, slots: owners.take("native_slots")?,
                positions: owners.take("native_positions")?, inverse_frequencies: owners.take("inverse_frequencies")?,
            }),
        })
    }
}

fn draft_view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "MLA draft view exceeds admitted owner");
    // Non-owning descriptor; the retained allocation remains responsible for lifetime.
    let address = (buffer.ptr as usize).checked_add(offset).context("MLA draft view address overflow")?;
    ensure!(address.checked_add(bytes).is_some(), "MLA draft view end address overflow");
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

/// One sequence to draft for: its ring slot, the token at `position` whose
/// target step has not run yet, `position` itself (the context length), and
/// the first position whose context entry the ring holds for this sequence
/// (`context_valid_from`: 0 after a full prefill; a prefix-cache restore at
/// `P` starts the drafter cold at `P`, and ring entries before it belong to
/// the slot's previous sequence).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DraftSeq {
    pub slot: usize,
    pub anchor: u32,
    pub position: usize,
    pub valid_from: usize,
}

/// One committed tapped row: its row in the last step's taps, the sequence's
/// ring slot and the row's position.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContextRow {
    pub tap_row: usize,
    pub slot: usize,
    pub position: usize,
}

struct MlaContextUpdate {
    first: usize,
    positions: Vec<i64>,
    slots: Vec<Vec<i64>>,
    next: cuteafd_loader::formats::dflash::MlaDraftCacheState,
}

impl MlaContextUpdate {
    fn checked_paged(state: &mut MlaPagedState,
        layout: &cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout, rows: &[ContextRow])
        -> Result<(Self, Vec<cuteafd_engine::prefix::DraftReservation>)> {
        let mut plan = Self::checked(&state.rings, layout.cache().layers.len(), rows)?;
        let mut reservations = Vec::new();
        let result = (|| -> Result<()> {
            let mut grouped = std::collections::BTreeMap::<usize, Vec<&ContextRow>>::new();
            for row in rows { grouped.entry(row.slot).or_default().push(row); }
            for (slot, mut group) in grouped {
                group.sort_unstable_by_key(|r| r.position);
                let reservation = state.reserve(slot, group[0].position, group.len())?;
                reservations.push(reservation);
                let physical = state.pool.destinations(reservation)?;
                for (layer, destinations) in plan.slots.iter_mut().enumerate() {
                    if layout.cache().layers[layer].window().is_some() { continue; }
                    for (row, &physical) in group.iter().zip(&physical) {
                        destinations[row.tap_row - plan.first] = physical;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            for reservation in reservations { state.pool.commit(reservation, 0)?; }
            return Err(error);
        }
        Ok((plan, reservations))
    }

    fn checked(state: &cuteafd_loader::formats::dflash::MlaDraftCacheState,
        layers: usize, rows: &[ContextRow]) -> Result<Self> {
        ensure!(!rows.is_empty(), "MLA draft empty context update");
        let first = rows.iter().map(|r| r.tap_row).min().unwrap();
        let last = rows.iter().map(|r| r.tap_row).max().unwrap();
        ensure!(last < TAP_ROWS, "MLA draft tap row {last} past {TAP_ROWS}");
        let mut positions = vec![0; last - first + 1];
        let mut seen = std::collections::BTreeSet::new();
        let mut grouped = std::collections::BTreeMap::<usize, Vec<&ContextRow>>::new();
        for row in rows {
            ensure!(seen.insert(row.tap_row), "MLA draft duplicate tap row {}", row.tap_row);
            positions[row.tap_row - first] = i64::try_from(row.position)?;
            grouped.entry(row.slot).or_default().push(row);
        }
        let mut slots = vec![vec![-1; positions.len()]; layers];
        let mut next = state.clone();
        for (slot, mut group) in grouped {
            group.sort_unstable_by_key(|r| r.position);
            let start = group[0].position;
            ensure!(group.iter().enumerate().all(|(i, row)| start.checked_add(i) == Some(row.position)),
                "MLA draft context rows are not a contiguous append");
            for (layer, destinations) in slots.iter_mut().enumerate() {
                let physical = state.context_rows(layer, slot, start, group.len())?;
                for (row, physical) in group.iter().zip(physical) { destinations[row.tap_row - first] = physical; }
            }
            next.commit_context(slot, start, group.len())?;
        }
        Ok(Self { first, positions, slots, next })
    }
}

/// The drafted tokens after one anchor and the selector's per-token
/// features (margin, best probability, entropy, rank among the 16); a dSpark
/// draft has no selector and carries its confidence head's predicted
/// acceptance per token instead (empty for DFlash2).
#[derive(Debug, Clone)]
pub(crate) struct Draft {
    pub tokens: Vec<u32>,
    pub features: Vec<[f32; 4]>,
    pub confidence: Vec<f32>,
}

/// FP32 logits `[rows, vocab]` of BF16 rows `[rows, hidden]` on a stream: `(x, logits, rows, stream)`.
pub(crate) type HeadLaunch<'h> = Box<dyn Fn(*const c_void, *mut f32, usize, *mut c_void) -> Result<()> + 'h>;

/// Populate BF16 draft input rows on the supplied stream. A remote target
/// gathers on its own device and lands the rows before this stream consumes them.
/// The callback retains every source/transport buffer until the queued copy completes.
pub(crate) type InputLaunch<'h> = dyn Fn(&[u32], CuteafdDeviceBuffer, *mut c_void) -> Result<()> + 'h;

/// The target's vocabulary head a draft step borrows: its only resident copy,
/// never duplicated or repacked by the drafter.
pub(crate) enum TargetHead<'h> {
    /// The target's checkpoint BF16 `[vocab, hidden]`.
    Bf16(&'h Dev<'h>),
    /// A scoped borrow from a device-aware target owner, retained through the draft drain.
    Bf16Buffer(&'h CuteafdDeviceBuffer),
    /// A target-owned launcher over another representation (GLM 5.3 Flash's
    /// FP8-only head in 16-row spans); the drafter's head workspace is unused.
    Launch(HeadLaunch<'h>),
}

/// A resolved [`TargetHead`] inside a draft step.
enum HeadCall<'h> {
    Bf16(*const c_void),
    Launch(&'h HeadLaunch<'h>),
}

fn draft_input_ids(sequences: &[DraftSeq], mask_token: u32, block: usize) -> Vec<u32> {
    sequences.iter().flat_map(|seq| std::iter::once(seq.anchor)
        .chain(std::iter::repeat_n(mask_token, block - 1))).collect()
}

/// Where a draft step's input rows come from.
#[derive(Clone, Copy)]
enum DraftInput<'r, 'e> {
    /// The anchors' embedding rows (host BF16); the mask rows are the drafter's own copy.
    Rows(&'r [u8]),
    /// Gathered on the device from the target's embedding table by token id.
    Table(&'r TokenEmbedding<'e>),
    /// A device-ordered gather/transport bridge owned by the target.
    Launch(&'r InputLaunch<'r>),
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MlaReadback {
    pub frontier: (usize, usize),
    pub layers: Vec<Vec<u8>>,
    pub taps: Vec<u8>,
}

pub(crate) struct DflashDrafter<'a> {
    pub(crate) kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    library: &'a NativeLibrary,
    timing: crate::shared::draft::timing::DraftTiming<'a>,
    pub cfg: DflashConfig,
    stream: *mut c_void,
    /// The draft round's host wait (`--host-wait`).
    round: std::cell::RefCell<crate::shared::memory::BlockingWait<'a>>,
    /// Ring slots (sequences with a drafter context).
    pub slots: usize,
    max_sequences: usize,
    fc: DraftWeight<'a>,
    hidden_norm: Dev<'a>,
    norm: Dev<'a>,
    projection: DraftWeight<'a>,
    predecessor: Dev<'a>,
    successor: Dev<'a>,
    layers: Vec<DraftLayer<'a>>,
    /// [TAP_ROWS, taps * hidden] BF16: the last step's tapped rows.
    pub taps: Dev<'a>,
    /// Context update scratch: [TAP_ROWS, hidden] twice, [TAP_ROWS, 2 kv] and the tables.
    fused: Dev<'a>,
    fused_norm: Dev<'a>,
    context_positions: Dev<'a>,
    context: DraftContext<'a>,
    /// The mask token's embedding row.
    mask_row: Vec<u8>,
    workspace: RefCell<Option<Workspace<'a>>>,
    startup_primed: Cell<bool>,
    representation: GlmDraftRepresentation,
    fp8_workspace: Option<Dev<'a>>,
    /// How the borrowed BF16 head runs past one draft block ([`DraftHead`]; GLM 5.3
    /// Flash's --draft-head). [`DraftHead::Exact`] unless the target sets it.
    head_mode: Cell<DraftHead>,
    /// How the FP8 GEMMs run (GLM 5.3 Flash's --draft-linear), and the latest mode the FP8
    /// scratch was admitted for.
    fp8_rows: Cell<fp8_linear::Fp8Rows>,
    fp8_admitted: fp8_linear::Fp8Rows,
    confidence_directory: std::path::PathBuf,
    confidence_scales: fp8_linear::Fp8Scales,
}

#[allow(clippy::too_many_arguments)]
fn pack_draft_weight<'a>(library: &'a NativeLibrary, source: Dev<'a>, n: usize, k: usize,
    representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales, stream: *mut c_void)
    -> Result<DraftWeight<'a>> {
    match representation {
        GlmDraftRepresentation::Bf16Only => Ok(DraftWeight::Bf16(source)),
        GlmDraftRepresentation::Fp8Only => {
            let packed = Fp8Weight::pack(library, source.buffer.ptr, n, k, scales, stream);
            // SAFETY: startup packing retains source and destinations through completion.
            if let Err(error) = unsafe { library.cuda_stream_synchronize(stream) } {
                std::mem::forget(source);
                if let Ok(weight) = packed { weight.quarantine(); }
                return Err(error.context("DFlash packing owners quarantined after failed drain"));
            }
            let packed = packed?;
            drop(source);
            Ok(DraftWeight::Fp8(packed))
        }
    }
}

fn at(dev: &Dev<'_>, bytes: usize) -> *mut c_void {
    // SAFETY: callers stay inside the allocation.
    unsafe { dev.buffer.ptr.cast::<u8>().add(bytes) }.cast()
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

struct Checkpoint {
    data: Vec<u8>,
    tensors: HashMap<String, SafetensorsTensorMetadata>,
}

/// Reads the drafter's safetensors file on a thread (while the target loads).
pub(crate) fn prefetch(snapshot: &Path) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    let path = snapshot.join("model.safetensors");
    std::thread::spawn(move || {
        tracing::info!("drafter checkpoint read started");
        let started = std::time::Instant::now();
        let result = std::fs::read(path);
        let seconds = started.elapsed().as_secs_f64();
        let bytes = result.as_ref().map_or(0, Vec::len);
        tracing::info!(bytes, elapsed_ms = seconds * 1000.0,
            gb_per_second = bytes as f64 / seconds.max(f64::MIN_POSITIVE) / 1e9,
            success = result.is_ok(), "drafter checkpoint read finished");
        result
    })
}

impl Checkpoint {
    fn bytes(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("DFlash2 checkpoint has no {name}"))?;
        ensure!(t.dtype == DType::Bf16, "{name}: DFlash2 weights must be BF16, found {:?}", t.dtype);
        let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: BF16 tensor byte count overflow"))?;
        ensure!(t.shape == shape && usize::try_from(t.byte_length)? == bytes,
            "{name}: shape {:?}, expected BF16 {shape:?}", t.shape);
        let first = usize::try_from(t.byte_offset)?;
        let end = first.checked_add(bytes).with_context(|| format!("{name}: tensor range overflow"))?;
        let range = first..end;
        self.data.get(range).with_context(|| format!("{name} lies past the file"))
    }

    fn loaded(&self, name: &str) -> Result<cuteafd_loader::LoadedTensor> {
        let t = self.tensors.get(name).with_context(|| format!("DFlash2 checkpoint has no {name}"))?;
        Ok(cuteafd_loader::LoadedTensor {
            info: cuteafd_core::TensorInfo { name: name.into(), file: "model.safetensors".into(),
                dtype: t.dtype.clone(), shape: t.shape.clone(), byte_offset: t.byte_offset, byte_length: t.byte_length,
                role: cuteafd_core::TensorRole::Other, layer_id: None, expert_id: None,
                is_quantization_metadata: false },
            source_path: "model.safetensors".into(), bytes: self.bytes(name, &t.shape)?.to_vec(),
            elapsed_micros: 0, sha256: String::new(),
        })
    }

    /// Validate every owned tensor before the first drafter device allocation.
    /// Native FP8 checkpoint input needs a direct fragment packer/exporter;
    /// the BF16 convenience quantizer does not implicitly widen those values.
    fn validate(&self, cfg: &DflashConfig) -> Result<()> {
        for (name, shape) in cfg.tensor_shapes() {
            self.bytes(&name, &shape)?;
        }
        Ok(())
    }
}

/// Header-only admission before native modules or target/drafter allocation.
pub(crate) fn check_checkpoint(snapshot: &Path, fp8: Option<bool>, context_slots: Option<usize>,
    max_batch_sequences: usize) -> Result<()> {
    let cfg = DflashConfig::read(snapshot)?;
    let mode = GlmDraftRepresentation::from_fp8_option(fp8);
    let capacity = GlmDraftCapacity::new(context_slots.unwrap_or(20.max(max_batch_sequences)),
        max_batch_sequences, cfg.block)?;
    cfg.runtime_layout(mode, capacity)?;
    let path = snapshot.join("model.safetensors");
    let tensors: HashMap<_, _> = read_safetensors_metadata(&path)?.into_iter().map(|t| (t.name.clone(), t)).collect();
    let file_bytes = std::fs::metadata(&path)?.len();
    check_checkpoint_headers(&cfg, &tensors, file_bytes)
}

/// Admit the target's single resident BF16 head or native K6 launcher before allocation.
pub(crate) fn check_snapshot_target_head(snapshot: &Path, hidden: usize, vocab: usize,
    explicit_fp8_head: bool) -> Result<()> {
    use cuteafd_loader::formats::trellis_core::{read_projections, CoreContract, CoreSplit};
    use std::collections::BTreeMap;

    // Routed-expert catalogs need not contain coordinator tensors. Resolve the
    // head through its own index/header contract without reading weight data.
    let checkpoint = cuteafd_loader::plan::Checkpoint::open_for_role(snapshot, "DFlash target head",
        |name| name.starts_with("lm_head."))?;
    let expected = BTreeMap::from([("lm_head.weight".into(), vec![vocab, hidden])]);
    let native = read_projections(&checkpoint, &expected, |name| name.starts_with("lm_head."),
        |stem| (stem == "lm_head").then_some(CoreContract {
            split: CoreSplit::Whole, bits: 6, padded_outputs: false,
        }))?;
    if native.contains_key("lm_head") {
        ensure!(!explicit_fp8_head, "lm_head: --fp8-head cannot replace the shared native K6 target head");
        return Ok(());
    }
    ensure!(!checkpoint.tensor_names().any(|name| name.starts_with("lm_head.") && name != "lm_head.weight"),
        "DFlash target lm_head has incomplete or unsupported native storage");
    let head = checkpoint.tensors.iter().find(|t| t.meta.name == "lm_head.weight")
        .context("DFlash target checkpoint has no supported lm_head")?;
    check_target_bf16_head(&head.meta, hidden, vocab, explicit_fp8_head)
}

pub(crate) fn check_target_bf16_head(t: &SafetensorsTensorMetadata, hidden: usize, vocab: usize,
    explicit_fp8_head: bool) -> Result<()> {
    ensure!(!explicit_fp8_head,
        "{}: --fp8-head with this DFlash target is unsupported: its head has no shared FP8 launcher; \
         retaining a BF16 head beside an FP8 target copy is not supported", t.name);
    check_target_head_source(t, hidden, vocab)
}

/// The checkpoint head the target loads (BF16, or quantized from it to the
/// target's one FP8 head) and the drafter borrows: BF16 `[vocab, hidden]`.
pub(crate) fn check_target_head_source(t: &SafetensorsTensorMetadata, hidden: usize, vocab: usize) -> Result<()> {
    ensure!(t.dtype == DType::Bf16,
        "{}: shared DFlash head requires checkpoint BF16, found {:?}; add a source-native compact \
         target/drafter head consumer instead of widening or copying the checkpoint", t.name, t.dtype);
    let bytes = vocab.checked_mul(hidden).and_then(|n| n.checked_mul(2))
        .context("DFlash borrowed BF16 head byte count overflow")?;
    ensure!(t.shape == [vocab, hidden] && t.byte_length == bytes as u64,
        "{}: shared BF16 head shape {:?}/{} bytes, expected [{vocab}, {hidden}]/{bytes} bytes",
        t.name, t.shape, t.byte_length);
    Ok(())
}

fn check_checkpoint_headers(cfg: &DflashConfig, tensors: &HashMap<String, SafetensorsTensorMetadata>,
    file_bytes: u64) -> Result<()> {
    for (name, shape) in cfg.tensor_shapes() {
        let t = tensors.get(&name).with_context(|| format!("DFlash2 checkpoint has no {name}"))?;
        ensure!(t.dtype == DType::Bf16,
            "{name}: DFlash2 checkpoint must be BF16, found {:?}; source-native FP8 requires \
             a direct fragment packer, other value types are unsupported", t.dtype);
        let bytes = shape.iter().try_fold(2u64, |n, &d| n.checked_mul(d as u64))
            .with_context(|| format!("{name}: BF16 tensor byte count overflow"))?;
        ensure!(t.shape == shape && t.byte_length == bytes,
            "{name}: shape {:?}/{} bytes, expected BF16 {shape:?}/{bytes} bytes", t.shape, t.byte_length);
        let end = t.byte_offset.checked_add(t.byte_length).with_context(|| format!("{name}: tensor range overflow"))?;
        ensure!(end <= file_bytes, "{name} lies past the DFlash2 checkpoint file");
    }
    Ok(())
}

impl<'a> DflashDrafter<'a> {
    /// The keyed selector prior (`family/dflash2/numerics`), or the generic
    /// GLM-5.3 fit when the drafter ships none for this key.
    pub fn selector_fit(&self, family: &str, fp8_head: bool) -> Result<crate::shared::draft_confidence::SelectorFit> {
        crate::shared::draft_confidence::SelectorFit::load(&self.confidence_directory,
            &self.confidence_key(family, fp8_head))
    }

    fn confidence_key(&self, family: &str, fp8_head: bool) -> String {
        let head = if fp8_head { "head-fp8-row".into() } else { format!("head-bf16-{:?}", self.head_mode.get()) };
        let kv = if self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 { "-kv-fp8" } else { "" };
        let numerics = format!("{}-{:?}-{head}-{:?}{kv}-r1", self.representation.name(), self.fp8_rows.get(),
            self.confidence_scales).to_ascii_lowercase();
        format!("{family}/dflash2/{numerics}")
    }

    pub fn max_batch_sequences(&self) -> usize { self.max_sequences }
    /// Loads the drafter's weights from `file` (its safetensors bytes, see
    /// [`prefetch`]) and allocates `slots` ring contexts; draft steps take up
    /// to `max_sequences` sequences. `mask_row` is the target embedding of
    /// the mask token. `fp8_rows` is how the FP8 GEMMs run (their scratch serves
    /// it and the modes before it, see [`Self::set_draft_linear`]).
    #[allow(clippy::too_many_arguments)]
    pub fn load(library: &'a NativeLibrary, snapshot: &Path, file: Vec<u8>, stream: *mut c_void, slots: usize,
        max_sequences: usize, mask_row: Vec<u8>, row_window: bool, representation: GlmDraftRepresentation,
        scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let cfg = DflashConfig { row_window, ..DflashConfig::read(snapshot)? };
        let capacity = GlmDraftCapacity::new(slots, max_sequences, cfg.block)?;
        let layout = cfg.runtime_layout(representation, capacity)?;
        let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let (resident_bytes, scratch_bytes) = cuteafd_loader::families::glm5::draft_representation::draft_resident_bytes_with_mode(
            &config, slots, max_sequences, library.sm_count()? as u64, representation, fp8_rows.code() as u8, kv_cache)?;
        tracing::info!(resident_bytes, scratch_bytes, kv_cache = %kv_cache, "DFlash shared readiness inventory");
        ensure!(mask_row.len() == cfg.hidden * 2, "DFlash BF16 mask row has wrong hidden width");
        let path = snapshot.join("model.safetensors");
        let checkpoint = Checkpoint {
            data: file,
            tensors: read_safetensors_metadata(&path)?.into_iter().map(|t| (t.name.clone(), t)).collect(),
        };
        checkpoint.validate(&cfg)?;
        let upload = |bytes: &[u8]| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.len().max(256))?;
            library.copy_h2d(allocation.buffer, bytes)?;
            Ok(allocation)
        };
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
        let fp8_workspace = layout.fp8_scratch.as_ref().map(|scratch| {
            let shapes: Vec<_> = scratch.shapes.iter().map(|shape|
                Ok((usize::try_from(shape.k)?, usize::try_from(shape.n)?))).collect::<Result<_>>()?;
            fp8_linear::scratch_rows(library, scratch.rows, &shapes, fp8_rows)
        }).transpose()?;
        let tensor = |name: &str, shape: &[usize]| checkpoint.bytes(name, shape).and_then(upload);
        let concat = |parts: &[(&str, usize)], cols: usize| -> Result<Dev<'a>> {
            let total: usize = parts.iter().map(|(_, rows)| rows * cols * 2).sum();
            let allocation = DeviceAllocation::new(library, total)?;
            let mut offset = 0;
            for (name, rows) in parts {
                let bytes = checkpoint.bytes(name, &[*rows, cols])?;
                library.copy_h2d(CuteafdDeviceBuffer { ptr: at(&allocation, offset), bytes: bytes.len(),
                    ..allocation.buffer }, bytes)?;
                offset += bytes.len();
            }
            Ok(allocation)
        };
        let matrix = |source, n, k| pack_draft_weight(library, source, n, k, representation, scales, stream);
        let layers = (0..cfg.layers).map(|l| -> Result<DraftLayer<'a>> {
            let p = format!("layers.{l}");
            let a = format!("{p}.self_attn");
            Ok(DraftLayer {
                input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                attn_conv: matrix(tensor(&format!("{p}.attention_conv.kernel_projection.weight"), &[cfg.conv_width(), h])?,
                    cfg.conv_width(), h)?,
                attn_base: tensor(&format!("{p}.attention_conv.base_kernel"), &[2, 2, h])?,
                mlp_conv: matrix(tensor(&format!("{p}.mlp_conv.kernel_projection.weight"), &[cfg.conv_width(), h])?,
                    cfg.conv_width(), h)?,
                mlp_base: tensor(&format!("{p}.mlp_conv.base_kernel"), &[2, 2, h])?,
                attention: DraftAttention::Gqa(GqaAttention {
                    qkv: matrix(concat(&[(&format!("{a}.q_proj.weight"), cfg.heads * cfg.head_dim),
                        (&format!("{a}.k_proj.weight"), kv), (&format!("{a}.v_proj.weight"), kv)], h)?, cfg.qkv_width(), h)?,
                    q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
                    k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
                    o: matrix(tensor(&format!("{a}.o_proj.weight"), &[h, cfg.heads * cfg.head_dim])?,
                        h, cfg.heads * cfg.head_dim)?,
                    k_ring: zeroed(slots * RING * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim))?,
                    v_ring: zeroed(slots * RING * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim))?,
                }),
                gate_up: matrix(concat(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                    (&format!("{p}.mlp.up_proj.weight"), inter)], h)?, 2 * inter, h)?,
                down: matrix(tensor(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?, h, inter)?,
            })
        }).collect::<Result<Vec<_>>>()?;
        let taps = cfg.taps.len() * h;
        tracing::info!(?representation, context_slots = slots, max_batch_sequences = max_sequences,
            own_weight_bytes = layout.weights.resident_bytes()?, max_load_staging = layout.weights.max_load_staging,
            "DFlash single-copy storage admitted; vocabulary head borrowed from target");
        Ok(Self {
            kv_cache,
            library,
            stream,
            slots,
            max_sequences,
            fc: matrix(tensor("fc.weight", &[h, taps])?, h, taps)?,
            hidden_norm: tensor("hidden_norm.weight", &[h])?,
            norm: tensor("norm.weight", &[h])?,
            projection: matrix(tensor("candidate_selector.hidden_projection.weight", &[cfg.rank, h])?, cfg.rank, h)?,
            predecessor: tensor("candidate_selector.predecessor_codebook", &[cfg.vocab, cfg.rank])?,
            successor: tensor("candidate_selector.successor_codebook", &[cfg.vocab, cfg.rank])?,
            layers,
            taps: zeroed(TAP_ROWS * taps * 2)?,
            fused: zeroed(TAP_ROWS * h * 2)?,
            fused_norm: zeroed(TAP_ROWS * h * 2)?,
            context_positions: zeroed(TAP_ROWS * 8)?,
            context: DraftContext::Gqa(GqaContext {
                kv: zeroed(TAP_ROWS * 2 * kv * 2)?, slots: zeroed(TAP_ROWS * 4)?,
            }),
            mask_row,
            workspace: RefCell::new(None),
            startup_primed: Cell::new(false),
            representation,
            confidence_directory: snapshot.into(),
            confidence_scales: scales,
            fp8_workspace,
            timing: crate::shared::draft::timing::DraftTiming::from_env(library, "glm-dflash")?,
            round: std::cell::RefCell::new(crate::shared::memory::BlockingWait::new(library)),
            head_mode: Cell::new(DraftHead::Exact),
            fp8_rows: Cell::new(fp8_rows),
            fp8_admitted: fp8_rows,
            cfg,
        })
    }

    /// Called inside the one admitted `device.own` callback, before target weights.
    /// Programs and headers are checked before the exact arenas are allocated.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn load_mla(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>,
        program_namespace: &str, snapshot: &Path, file: Vec<u8>, stream: *mut c_void,
        layout: cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales,
        fp8_rows: fp8_linear::Fp8Rows) -> Result<Self> {
        use cuteafd_ffi::mla_draft::{MlaDraft, MlaDraftWindow};
        use cuteafd_loader::formats::{dflash::MlaDflashConfig, mla::{self, MlaOperand}};
        let _scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let native = MlaDflashConfig::read(snapshot)?;
        ensure!(native.runtime_layout(representation, layout.capacity(), layout.program().context,
            layout.paged_cache().target_tokens, library.sm_count()?, fp8_rows.code() as u8, layout.cache_format())? == layout,
            "MLA draft checkpoint/runtime differs from admitted immutable layout");
        let cfg = DflashConfig::mla(&native);
        let checkpoint = Checkpoint { data: file, tensors: read_safetensors_metadata(&snapshot.join("model.safetensors"))?
            .into_iter().map(|t| (t.name.clone(), t)).collect() };
        for (name, shape) in native.tensor_shapes() { checkpoint.bytes(&name, &shape)?; }
        let programs = native.windows.iter().map(|window| -> Result<_> {
            let program = MlaDraft::load(programs, program_namespace, if window.is_some() { MlaDraftWindow::Sliding4096 }
                else { MlaDraftWindow::Full }, layout.cache_format())?;
            ensure!(program.max_context() >= layout.program().context
                && program.splits() == layout.program().splits
                && program.scratch_bytes() == layout.program().scratch,
                "MLA draft program differs from admitted scratch/context");
            Ok(program)
        }).collect::<Result<Vec<_>>>()?;
        let mut owners = MlaDraftStorage::allocate(library, layout.clone())?.into_owners()?;
        let upload = |bytes: &[u8]| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.len().max(256))?;
            library.copy_h2d(allocation.buffer, bytes)?;
            Ok(allocation)
        };
        let tensor = |name: &str, shape: &[usize]| checkpoint.bytes(name, shape).and_then(upload);
        let concat = |parts: &[(&str, usize)], cols: usize| -> Result<Dev<'a>> {
            let total = parts.iter().map(|(_, rows)| rows * cols * 2).sum();
            let allocation = DeviceAllocation::new(library, total)?;
            let mut offset = 0;
            for (name, rows) in parts {
                let bytes = checkpoint.bytes(name, &[*rows, cols])?;
                library.copy_h2d(draft_view(allocation.buffer, offset, bytes.len())?, bytes)?;
                offset += bytes.len();
            }
            Ok(allocation)
        };
        let matrix = |source, n, k| pack_draft_weight(library, source, n, k, representation, scales, stream);
        let (h, inter) = (cfg.hidden, cfg.intermediate);
        let mut layers = Vec::with_capacity(cfg.layers);
        for (index, program) in programs.into_iter().enumerate() {
            let p = format!("layers.{index}");
            let a = format!("{p}.self_attn");
            let operand = |kind| -> Result<Dev<'a>> {
                let packed = mla::pack(&native.mla, &a, false, kind, 1, |name| checkpoint.loaded(name))?;
                let one = packed.into_iter().next().context("MLA draft missing packed operand")?;
                upload(&one.bytes)
            };
            layers.push(DraftLayer {
                input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                attn_conv: matrix(tensor(&format!("{p}.attention_conv.kernel_projection.weight"), &[cfg.conv_width(), h])?, cfg.conv_width(), h)?,
                attn_base: tensor(&format!("{p}.attention_conv.base_kernel"), &[2, 2, h])?,
                mlp_conv: matrix(tensor(&format!("{p}.mlp_conv.kernel_projection.weight"), &[cfg.conv_width(), h])?, cfg.conv_width(), h)?,
                mlp_base: tensor(&format!("{p}.mlp_conv.base_kernel"), &[2, 2, h])?,
                attention: DraftAttention::Mla(MlaAttention { program,
                    qkv_a: operand(MlaOperand::QkvA)?, q_norm: operand(MlaOperand::QNorm)?,
                    kv_norm: operand(MlaOperand::KvNorm)?, q_b: operand(MlaOperand::QB)?,
                    uk: operand(MlaOperand::Uk)?, uv: operand(MlaOperand::Uv)?, output: operand(MlaOperand::Output)?,
                    cache: owners.take(&format!("layer.{index}.cache"))?,
                    page_table: owners.take(&format!("layer.{index}.page_table"))?,
                }),
                gate_up: matrix(concat(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                    (&format!("{p}.mlp.up_proj.weight"), inter)], h)?, 2 * inter, h)?,
                down: matrix(tensor(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?, h, inter)?,
            });
        }
        let workspace = Workspace::admitted_mla(library, &cfg, layout.capacity().max_batch_sequences, &mut owners)?;
        let AttentionWorkspace::Mla(attention) = &workspace.attention else { unreachable!() };
        library.copy_h2d(draft_view(attention.inverse_frequencies.buffer, 0,
            native.rotary.inv_freq.len() * 4)?, bytes_of(&native.rotary.inv_freq))?;
        let result = Self { kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, library, stream, slots: layout.capacity().context_slots,
            max_sequences: layout.capacity().max_batch_sequences,
            fc: matrix(tensor("fc.weight", &[h, cfg.taps.len() * h])?, h, cfg.taps.len() * h)?,
            hidden_norm: tensor("hidden_norm.weight", &[h])?, norm: tensor("norm.weight", &[h])?,
            projection: matrix(tensor("candidate_selector.hidden_projection.weight", &[cfg.rank, h])?, cfg.rank, h)?,
            predecessor: tensor("candidate_selector.predecessor_codebook", &[cfg.vocab, cfg.rank])?,
            successor: tensor("candidate_selector.successor_codebook", &[cfg.vocab, cfg.rank])?, layers,
            taps: owners.take("taps")?, fused: owners.take("fused")?, fused_norm: owners.take("fused_norm")?,
            context_positions: owners.take("context_positions")?,
            context: DraftContext::Mla(MlaContext { cfg: native, state: RefCell::new(MlaPagedState::new(&layout)?),
                layout, slots: owners.take("context_slots")? }),
            mask_row: Vec::new(), workspace: RefCell::new(Some(workspace)), startup_primed: Cell::new(false), representation,
            fp8_workspace: if representation == GlmDraftRepresentation::Fp8Only { Some(owners.take("fp8_scratch")?) } else { None },
            head_mode: Cell::new(DraftHead::Exact), fp8_rows: Cell::new(fp8_rows), fp8_admitted: fp8_rows,
            confidence_directory: snapshot.into(), confidence_scales: scales,
            timing: crate::shared::draft::timing::DraftTiming::from_env(library, "mla-dflash")?, cfg,
            round: std::cell::RefCell::new(crate::shared::memory::BlockingWait::new(library)),
        };
        owners.finish()?;
        Ok(result)
    }

    /// How the FP8 GEMMs run from now on: the load's mode or one before it (whose scratch the
    /// load's covers). A BF16 drafter ignores it.
    pub fn set_draft_linear(&self, mode: fp8_linear::Fp8Rows) -> Result<()> {
        ensure!(mode <= self.fp8_admitted, "the DFlash2 FP8 scratch was admitted for {:?}, not {mode:?}",
            self.fp8_admitted);
        self.fp8_rows.set(mode);
        Ok(())
    }

    /// How draft steps run the borrowed BF16 head from now on (a target's FP8 head launcher
    /// ignores it).
    pub fn set_draft_head(&self, mode: DraftHead) {
        self.head_mode.set(mode);
    }

    pub(crate) fn draft_head_mode(&self) -> DraftHead {
        self.head_mode.get()
    }

    /// `out` [rows,n] = `x` [rows,k] @ selected weight rows. This loaded
    /// matrix owns one representation for every row count.
    ///
    /// # Safety
    /// Input/output pointers hold the documented shapes on this stream.
    #[allow(clippy::too_many_arguments)]
    unsafe fn linear(&self, x: *const c_void, weight: &DraftWeight<'_>, first: usize, out: *mut c_void,
        rows: usize, k: usize, n: usize) -> Result<()> {
        match weight {
            DraftWeight::Bf16(w) => {
                // SAFETY: source rows first..first+n lie in this matrix;
                // the caller supplies its selected width and live output.
                unsafe { self.library.linear_bf16(x, at(w, first * k * 2), out, rows, k, n, self.stream) }
            }
            DraftWeight::Fp8(w) => {
                let scratch = self.fp8_workspace.as_ref().context("FP8 DFlash scratch was not admitted")?;
                // SAFETY: scratch covers every selected matrix and up to
                // max(TAP_ROWS, max_batch_sequences*block) input rows in this
                // mode (set_draft_linear keeps it within the admitted one).
                unsafe { w.apply_rows(self.library, x, out, false, rows, first, n, scratch, self.stream,
                    self.fp8_rows.get()) }
            }
        }
    }

    /// Index of `layer` among the tapped target layers.
    pub fn tap_index(&self, layer: usize) -> Option<usize> {
        self.cfg.taps.iter().position(|&l| l == layer)
    }

    /// Copies target layer output rows `[first, first + n)` of `hidden`
    /// ([rows, hidden] BF16) into tap rows `0..n` when `layer` is tapped.
    pub fn tap(&self, layer: usize, hidden: *const c_void, first: usize, n: usize) -> Result<()> {
        self.tap_at(layer, hidden, first, n, 0)
    }

    /// [`Self::tap`] into tap rows `to..to + n`.
    pub fn tap_at(&self, layer: usize, hidden: *const c_void, first: usize, n: usize, to: usize) -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        let (h, width) = (self.cfg.hidden, self.cfg.taps.len() * self.cfg.hidden);
        ensure!(to + n <= TAP_ROWS, "tap rows {to}..{} exceed {TAP_ROWS}", to + n);
        // SAFETY: `hidden` holds first + n rows; the tap buffer TAP_ROWS rows.
        unsafe {
            self.library.glm_dflash_tap(hidden.cast::<u8>().add(first * h * 2).cast(),
                self.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(), n, h, width, index * h, self.stream)
        }
    }

    /// Taps the mean of the `hc` streams of rows `[first, first + n)` of
    /// `streams` ([rows, hc, hidden] BF16) into tap rows `0..n` when `layer`
    /// is tapped (GLM 5.3 Flash: the mHC contraction upstream captures).
    pub fn tap_streams(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize) -> Result<()> {
        self.tap_streams_at(layer, streams, hc, first, n, 0)
    }

    /// [`Self::tap_streams`] into tap rows `to..to + n`.
    pub fn tap_streams_at(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize, to: usize)
        -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        let (h, width) = (self.cfg.hidden, self.cfg.taps.len() * self.cfg.hidden);
        ensure!(to + n <= TAP_ROWS, "tap rows {to}..{} exceed {TAP_ROWS}", to + n);
        // SAFETY: `streams` holds first + n rows of hc streams; the tap buffer TAP_ROWS rows.
        unsafe {
            self.library.glm_dflash_tap_mean(streams.cast::<u8>().add(first * hc * h * 2).cast(),
                self.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(), n, h, hc, width, index * h, self.stream)
        }
    }

    /// Mutable owners for the opt-in GLM Flash replay alias diagnostic.
    pub(crate) fn replay_write_ranges(&self) -> Vec<(String, CuteafdDeviceBuffer)> {
        let mut ranges = Vec::new();
        for (name, buffer) in [("taps", &self.taps), ("fused", &self.fused),
            ("fused_norm", &self.fused_norm), ("context_positions", &self.context_positions)] {
            ranges.push((format!("drafter/{name}"), buffer.buffer));
        }
        match &self.context {
            DraftContext::Gqa(context) => {
                ranges.push(("drafter/context_kv".into(), context.kv.buffer));
                ranges.push(("drafter/context_slots".into(), context.slots.buffer));
            }
            DraftContext::Mla(context) => ranges.push(("drafter/context_slots".into(), context.slots.buffer)),
        }
        for (layer, buffers) in self.layers.iter().enumerate() {
            match &buffers.attention {
                DraftAttention::Gqa(a) => {
                    ranges.push((format!("drafter/layer{layer}/k_ring"), a.k_ring.buffer));
                    ranges.push((format!("drafter/layer{layer}/v_ring"), a.v_ring.buffer));
                }
                DraftAttention::Mla(a) => {
                    ranges.push((format!("drafter/layer{layer}/cache"), a.cache.buffer));
                    ranges.push((format!("drafter/layer{layer}/page_table"), a.page_table.buffer));
                }
            }
        }
        if let Some(buffer) = &self.fp8_workspace {
            ranges.push(("drafter/fp8_workspace".into(), buffer.buffer));
        }
        if let Some(w) = self.workspace.borrow().as_ref() {
            for (name, buffer) in [("h", &w.h), ("n", &w.n), ("conv", &w.conv),
                ("dynamic", &w.dynamic), ("delta", &w.delta), ("gate_up", &w.gate_up),
                ("act", &w.act), ("logits", &w.logits), ("unary", &w.unary),
                ("candidates", &w.candidates), ("projected", &w.projected), ("anchors", &w.anchors),
                ("ids", &w.ids), ("tokens", &w.tokens), ("features", &w.features),
                ("positions", &w.positions), ("topk_workspace", &w.topk_workspace),
                ("head_workspace", &w._head_workspace)] {
                ranges.push((format!("drafter/step/{name}"), buffer.buffer));
            }
            let attention = match &w.attention {
                AttentionWorkspace::Gqa(a) => vec![("qkv", &a.qkv), ("q", &a.q), ("k", &a.k),
                    ("v", &a.v), ("attn", &a.attn), ("tables", &a.tables), ("attention_workspace", &a.scratch)],
                AttentionWorkspace::Mla(a) => vec![("query", &a.query), ("q_residual", &a.q_residual),
                    ("attention", &a.attention), ("lse", &a.lse), ("native_scratch", &a.scratch),
                    ("lengths", &a.lengths), ("query_offsets", &a.query_offsets),
                    ("native_slots", &a.slots), ("native_positions", &a.positions)],
            };
            ranges.extend(attention.into_iter().map(|(name, buffer)| (format!("drafter/step/{name}"), buffer.buffer)));
        }
        ranges
    }

    /// Writes the context K/V of committed tapped rows (in tap-row order).
    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        self.update_with(rows, || Ok(()))
    }

    /// Upload metadata before a remote tap wait is enqueued. `ready` orders
    /// the tap producer before the first context reader on this stream.
    pub(crate) fn update_with(&self, rows: &[ContextRow], ready: impl FnOnce() -> Result<()>) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if let DraftContext::Mla(context) = &self.context {
            return self.update_mla(context, rows, ready);
        }
        let DraftContext::Gqa(context) = &self.context else { unreachable!() };
        let (first, last) = (rows.iter().map(|r| r.tap_row).min().unwrap(), rows.iter().map(|r| r.tap_row).max().unwrap());
        let n = last - first + 1;
        ensure!(last < TAP_ROWS, "tap row {last} past {TAP_ROWS}");
        let (mut positions, mut slots) = (vec![0i64; n], vec![-1i32; n]);
        for row in rows {
            ensure!(row.slot < self.slots, "ring slot {} of {}", row.slot, self.slots);
            positions[row.tap_row - first] = row.position as i64;
            slots[row.tap_row - first] = (row.slot * RING + row.position % RING) as i32;
        }
        self.put(&self.context_positions, bytes_of(&positions))?;
        self.put(&context.slots, bytes_of(&slots))?;
        ready()?;
        let (h, kv) = (self.cfg.hidden, self.cfg.kv_width());
        let width = self.cfg.taps.len() * h;
        let s = self.stream;
        let context_span = self.timing.begin("context", None, n, s)?;
        // SAFETY: every buffer holds TAP_ROWS rows of its width; the stream orders the chain.
        unsafe {
            self.linear(at(&self.taps, first * width * 2), &self.fc, 0,
                self.fused.buffer.ptr, n, width, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, self.cfg.eps, s)?;
            for layer in &self.layers {
                let layer = layer.attention.gqa()?;
                let q_rows = self.cfg.heads * self.cfg.head_dim;
                self.linear(self.fused_norm.buffer.ptr, &layer.qkv, q_rows,
                    context.kv.buffer.ptr, n, h, 2 * kv)?;
                self.library.glm_dflash_qk_rope(context.kv.buffer.ptr, layer.q_norm.buffer.ptr,
                    layer.k_norm.buffer.ptr, self.context_positions.buffer.ptr, context.slots.buffer.ptr,
                    std::ptr::null_mut(), layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, n, 0, self.cfg.kv_heads,
                    self.cfg.theta, self.cfg.eps, self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8,
                    self.cfg.head_dim, s)?;
            }
        }
        self.timing.end(context_span, s)?;
        Ok(())
    }

    /// Read only reachable committed records and explicitly initialized tap rows.
    #[cfg(test)]
    pub(crate) fn mla_readback(&self, slot: usize, tap_rows: usize) -> Result<MlaReadback> {
        let DraftContext::Mla(context) = &self.context else { anyhow::bail!("MLA readback needs MLA context"); };
        ensure!(tap_rows <= TAP_ROWS, "MLA readback tap extent");
        let frontier @ (origin, end) = context.state.borrow().rings.frontier(slot)?;
        let (snapshot, copies, pages) = {
            let mut state = context.state.borrow_mut();
            let binding = state.bindings[slot].context("MLA readback slot is not bound")?;
            let (snapshot, copies) = state.pool.capture(binding)?;
            let pages = state.pool.attachment(snapshot)?.pages;
            (snapshot, copies, pages)
        };
        self.copy_mla_pages(&copies)?;
        // SAFETY: test-only readback retains all operands and their producing stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        let cache = context.layout.cache();
        let record_bytes = context.layout.paged_cache().record_bytes;
        let mut layers = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let DraftAttention::Mla(attention) = &layer.attention else { anyhow::bail!("mixed MLA layers"); };
            let bound = cache.layers[index].bind(slot, origin)?;
            let start = context.cfg.windows[index].map_or(origin, |window| origin.max(end.saturating_sub(window)));
            let mut bytes = Vec::with_capacity((end - start) * record_bytes);
            for position in start..end {
                let row = if context.cfg.windows[index].is_some() { bound.row(position)? } else {
                    let relative = position - origin;
                    pages[relative / 64].index as usize * 64 + relative % 64
                };
                let view = crate::shared::prefix::view(attention.cache.buffer,
                    row * record_bytes, record_bytes)?;
                let mut row = vec![0; record_bytes];
                self.library.copy_d2h(&mut row, view)?;
                bytes.extend(row);
            }
            layers.push(bytes);
        }
        let count = tap_rows * self.cfg.taps.len() * self.cfg.hidden * 2;
        let mut taps = vec![0; count];
        if count > 0 {
            self.library.copy_d2h(&mut taps, crate::shared::prefix::view(self.taps.buffer, 0, count)?)?;
        }
        context.state.borrow_mut().pool.release_snapshot(snapshot)?;
        Ok(MlaReadback { frontier, layers, taps })
    }

    pub(crate) fn reset_context(&self, slot: usize, origin: usize) -> Result<()> {
        ensure!(slot < self.slots, "DFlash reset outside admitted slots");
        if let DraftContext::Mla(context) = &self.context {
            // SAFETY: rebind retires the old request before its pooled destinations are released.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            context.state.borrow_mut().reset(slot, origin)?;
        }
        Ok(())
    }

    pub(crate) fn discard_context(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots, "DFlash discard outside admitted slots");
        if let DraftContext::Mla(context) = &self.context {
            // SAFETY: cancellation cannot recycle pages still addressed by queued work.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            context.state.borrow_mut().discard(slot)?;
        }
        Ok(())
    }

    pub(crate) fn mla_frontier(&self, slot: usize) -> Result<(usize, usize)> {
        let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
        Ok(context.state.borrow().rings.frontier(slot)?)
    }

    pub(crate) fn mla_snapshot_buffers(&self) -> Result<Vec<CuteafdDeviceBuffer>> {
        let DraftContext::Mla(_) = &self.context else { anyhow::bail!("not an MLA drafter") };
        self.layers.iter().map(|layer| {
            let DraftAttention::Mla(layer) = &layer.attention else { anyhow::bail!("MLA draft layer mismatch") };
            Ok(layer.cache.buffer)
        }).collect()
    }

    pub(crate) fn mla_draft_page_bytes(&self) -> usize {
        let DraftContext::Mla(context) = &self.context else { return 0 };
        let layout = context.layout.paged_cache();
        layout.page_rows * layout.record_bytes * layout.full_layers
    }

    pub(crate) fn copy_mla_windows(&self, slot: usize, marks: &[CuteafdDeviceBuffer], capture: bool) -> Result<()> {
        let windows = self.mla_window_segments(slot)?;
        ensure!(windows.len() == marks.len(), "MLA window mark layer count");
        for (window, mark) in windows.into_iter().zip(marks) {
            ensure!(window.bytes == mark.bytes, "MLA window mark byte count");
            let live = CuteafdDeviceBuffer { ptr: window.addr as *mut c_void, bytes: window.bytes, ..*mark };
            let (dst, src) = if capture { (*mark, live) } else { (live, *mark) };
            // SAFETY: family retains both window and mark owners until its drafter drain.
            unsafe { self.library.copy_d2d_async(dst, src, src.bytes, self.stream)?; }
        }
        Ok(())
    }

    /// Typed full-context attachment. The caller's existing positional mark must also
    /// retain `mla_window_segments(slot)`; a pages-only attachment is not a warm snapshot.
    pub(crate) fn capture_mla_pages(&self, slot: usize, frontier: usize, write_frontier: usize)
        -> std::result::Result<cuteafd_engine::prefix::DraftPageAttachment, cuteafd_engine::prefix::BoxError> {
        (|| -> Result<cuteafd_engine::prefix::DraftPageAttachment> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
            // SAFETY: the owned stream retires producers before snapshotting their canonical rows.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            let mut state = context.state.borrow_mut();
            let binding = state.bindings.get(slot).copied().flatten().context("unbound MLA draft slot")?;
            ensure!(state.pool.frontier(binding)?.1 == frontier, "MLA snapshot differs from committed frontier");
            let (snapshot, copies) = state.pool.capture_writable(binding, write_frontier)?;
            let attachment = state.pool.attachment(snapshot)?;
            drop(state);
            if let Err(error) = self.copy_mla_pages(&copies) {
                // A failed copy may have queued work. Do not release its destinations unless drained.
                // SAFETY: all copies use this drafter's owned stream.
                if unsafe { self.library.cuda_stream_synchronize(self.stream) }.is_ok() {
                    context.state.borrow_mut().pool.release_snapshot(snapshot)?;
                }
                return Err(error);
            }
            Ok(attachment)
        })().map_err(box_draft_page_error)
    }

    pub(crate) fn release_mla_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<()> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
        // SAFETY: the family also retires its host-copy stream before calling this hook.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        let mut state = context.state.borrow_mut();
        state.pool.check_attachment(attachment)?;
        Ok(state.pool.release_snapshot(attachment.snapshot)?)
    }

    pub(crate) fn mla_pages_valid(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        let DraftContext::Mla(context) = &self.context else { return false };
        context.state.borrow().pool.check_attachment(attachment).is_ok()
    }

    pub(crate) fn mla_pages_revocable(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        let DraftContext::Mla(context) = &self.context else { return false };
        context.state.borrow().pool.snapshot_revocable(attachment.snapshot)
    }

    pub(crate) fn retain_mla_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<()> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
        let mut state = context.state.borrow_mut();
        state.pool.check_attachment(attachment)?;
        Ok(state.pool.retain_snapshot(attachment.snapshot)?)
    }

    /// The slot's live draft page reservation (`None` for windowed drafters).
    pub(crate) fn mla_reservation(&self, slot: usize) -> Option<cuteafd_engine::reserve::DraftDemand> {
        let DraftContext::Mla(context) = &self.context else { return None };
        let state = context.state.borrow();
        let binding = state.bindings.get(slot).copied().flatten()?;
        state.pool.reservation(binding, self.cfg.block).ok()
    }
    pub(crate) fn mla_lifetime_rows(&self) -> Option<usize> {
        let DraftContext::Mla(context) = &self.context else { return None };
        Some(context.state.borrow().pool.lifetime_rows(self.cfg.block))
    }
    pub(crate) fn mla_reservable(&self) -> Option<usize> {
        let DraftContext::Mla(context) = &self.context else { return None };
        Some(context.state.borrow().pool.reservable())
    }
    pub(crate) fn mla_admission(&self, origin: usize) -> Option<cuteafd_engine::reserve::DraftDemand> {
        matches!(self.context, DraftContext::Mla(_)).then(|| cuteafd_engine::reserve::DraftDemand {
            origin, headroom: self.cfg.block.div_ceil(cuteafd_engine::prefix::DRAFT_PAGE_ROWS) + 1, held: 0 })
    }

    pub(crate) fn admit_mla_pages(&self, slot: usize, capacity: usize) -> std::result::Result<(), cuteafd_engine::prefix::BoxError> {
        (|| -> Result<()> {
            let DraftContext::Mla(context) = &self.context else { return Ok(()) };
            let mut state = context.state.borrow_mut();
            let binding = state.bindings.get(slot).copied().flatten().context("unbound MLA draft slot")?;
            Ok(state.pool.admit(binding, capacity, self.cfg.block)?)
        })().map_err(box_draft_page_error)
    }

    pub(crate) fn promote_mla_pages(&self, saved: &cuteafd_engine::prefix::DraftPageAttachment)
        -> std::result::Result<cuteafd_engine::prefix::DraftPageAttachment, cuteafd_engine::prefix::BoxError> {
        (|| -> Result<cuteafd_engine::prefix::DraftPageAttachment> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
            Ok(context.state.borrow_mut().pool.promote(saved)?)
        })().map_err(box_draft_page_error)
    }

    /// Caller restores window segments from its positional mark before publishing the binding.
    pub(crate) fn restore_mla_pages(&self, slot: usize,
        attachment: &cuteafd_engine::prefix::DraftPageAttachment, write_frontier: usize) -> std::result::Result<(), cuteafd_engine::prefix::BoxError> {
        (|| -> Result<()> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
            // SAFETY: retire the previous request before replacing its ownership.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            let mut state = context.state.borrow_mut();
            state.pool.check_attachment(attachment)?;
            let mut rings = state.rings.clone();
            rings.restore_frontier(slot, attachment.origin, attachment.frontier)?;
            state.discard(slot)?;
            state.pool.pin_snapshot(attachment.snapshot)?;
            let (binding, copies) = match state.pool.restore_writable(slot, attachment.snapshot,
                attachment.frontier, write_frontier) {
                Ok(restored) => restored,
                Err(error) => { state.pool.unpin_snapshot(attachment.snapshot)?; return Err(error.into()); }
            };
            state.bindings[slot] = Some(binding);
            drop(state);
            let copied = self.copy_mla_pages(&copies);
            // SAFETY: retain the source pin and live destinations until every restore copy retires.
            // A failed drain deliberately leaves the pin/ownership quarantined in the pool.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            context.state.borrow_mut().pool.unpin_snapshot(attachment.snapshot)?;
            copied?;
            context.state.borrow_mut().rings = rings;
            Ok(())
        })().map_err(box_draft_page_error)
    }

    pub(crate) fn mla_page_segments(&self, page: cuteafd_engine::prefix::DraftPageId)
        -> Result<Vec<cuteafd_hostcache::copy::DeviceRange>> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
        let page_bytes = context.layout.paged_cache().page_rows * context.layout.paged_cache().record_bytes;
        ensure!((page.index as usize) < context.layout.paged_cache().pages, "MLA draft page outside pool");
        self.layers.iter().zip(&context.layout.cache().layers).filter(|(_, layout)| layout.window().is_none())
            .map(|(layer, _)| {
                let DraftAttention::Mla(layer) = &layer.attention else { anyhow::bail!("MLA draft layer mismatch") };
                let buffer = draft_view(layer.cache.buffer, page.index as usize * page_bytes, page_bytes)?;
                Ok(cuteafd_hostcache::copy::DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes })
            }).collect()
    }

    pub(crate) fn mla_window_segments(&self, slot: usize)
        -> Result<Vec<cuteafd_hostcache::copy::DeviceRange>> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
        ensure!(slot < self.slots, "MLA draft window slot outside capacity");
        self.layers.iter().zip(&context.layout.cache().layers).filter(|(_, layout)| layout.window().is_some())
            .map(|(layer, layout)| {
                let DraftAttention::Mla(layer) = &layer.attention else { anyhow::bail!("MLA draft layer mismatch") };
                let bytes = layout.rows_per_slot() * context.layout.paged_cache().record_bytes;
                let buffer = draft_view(layer.cache.buffer, slot * bytes, bytes)?;
                Ok(cuteafd_hostcache::copy::DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes })
            }).collect()
    }

    fn copy_mla_pages(&self, copies: &[cuteafd_engine::prefix::TailCopy]) -> Result<()> {
            let DraftContext::Mla(context) = &self.context else { anyhow::bail!("not an MLA drafter") };
        let bytes_per_row = context.layout.paged_cache().record_bytes;
        let page_bytes = context.layout.paged_cache().page_rows * bytes_per_row;
        for copy in copies {
            for (layer, layout) in self.layers.iter().zip(&context.layout.cache().layers) {
                if layout.window().is_some() { continue; }
                let DraftAttention::Mla(layer) = &layer.attention else { anyhow::bail!("MLA draft layer mismatch") };
                let bytes = copy.rows * bytes_per_row;
                let from = draft_view(layer.cache.buffer, copy.from as usize * page_bytes, bytes)?;
                let to = draft_view(layer.cache.buffer, copy.to as usize * page_bytes, bytes)?;
                // SAFETY: allocator owns both checked ranges until the family's retirement barrier.
                unsafe { self.library.copy_d2d_async(to, from, bytes, self.stream)?; }
            }
        }
        Ok(())
    }

    fn update_mla(&self, context: &MlaContext<'_>, rows: &[ContextRow], ready: impl FnOnce() -> Result<()>) -> Result<()> {
        let (plan, reservations) = MlaContextUpdate::checked_paged(&mut context.state.borrow_mut(), &context.layout, rows)?;
        let lease = MlaPageLease { context, library: self.library, stream: self.stream, reservations };
        let workspace = self.workspace.borrow();
        let w = workspace.as_ref().context("MLA draft requires admitted workspace")?;
        let AttentionWorkspace::Mla(attention) = &w.attention else { anyhow::bail!("MLA draft workspace mismatch") };
        let n = plan.positions.len();
        let h = self.cfg.hidden;
        // Preflight every layer and m64 chunk before the first cache mutation.
        let mut batches = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let DraftAttention::Mla(layer) = &layer.attention else { anyhow::bail!("MLA draft layer mismatch") };
            let mut chunks = Vec::new();
            for offset in (0..n).step_by(64) {
                let rows = (n - offset).min(64);
                let b = attention.batch(context, layer, rows, 0, 0,
                    draft_view(self.fused_norm.buffer, offset * h * 2, rows * h * 2)?, w.delta.buffer,
                    draft_view(context.slots.buffer, offset * 8, rows * 8)?,
                    draft_view(self.context_positions.buffer, offset * 8, rows * 8)?);
                layer.program.check_produce(layer.weights(), b)?;
                chunks.push(b);
            }
            batches.push(chunks);
        }
        self.put(&self.context_positions, bytes_of(&plan.positions))?;
        self.put(&context.slots, bytes_of(&plan.slots[0]))?;
        ready()?;
        let span = self.timing.begin("context", None, n, self.stream)?;
        // SAFETY: checked tap span and admitted full-sized common arenas; one ordered stream.
        unsafe {
            self.linear(at(&self.taps, plan.first * self.cfg.taps.len() * h * 2), &self.fc, 0,
                self.fused.buffer.ptr, n, self.cfg.taps.len() * h, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, self.cfg.eps, self.stream)?;
        }
        for (index, (layer, chunks)) in self.layers.iter().zip(batches).enumerate() {
            let DraftAttention::Mla(layer) = &layer.attention else { unreachable!() };
            if index > 0 {
                // ordered copy fence, not a compute drain; the staging owner retains its bytes.
                self.put(&context.slots, bytes_of(&plan.slots[index]))?;
            }
            for batch in chunks { layer.program.produce(layer.weights(), batch, self.stream)?; }
        }
        self.timing.end(span, self.stream)?;
        lease.commit()?;
        context.state.borrow_mut().rings = plan.next;
        Ok(())
    }

    fn put(&self, dev: &Dev<'_>, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        // SAFETY: metadata consumers and overwrites share the owned drafter
        // stream; pinned staging survives through the upload completion event.
        unsafe { self.library.copy_h2d_ordered(dev.buffer, bytes, self.stream) }
    }

    /// Shared startup path for every serving DFlash family; diagnostics may
    /// still allocate their GQA workspace lazily. Priming never borrows a target.
    pub(crate) fn prepare_workspace(&self) -> Result<()> {
        self.prime_remote()
    }

    /// Prime common and typed attention compute before any bridge wait. Only
    /// startup uses zero inputs/logits; all cache bindings return to cold state.
    pub(crate) fn prime_remote(&self) -> Result<()> {
        if self.startup_primed.get() { return Ok(()); }
        let result = self.prime_isolated();
        if result.is_err() {
            // SAFETY: startup still owns all operands; no bridge waits have been published.
            if unsafe { self.library.cuda_stream_synchronize(self.stream) }.is_err() {
                self.library.quarantine_module_after_failed_drain();
            }
        }
        result
    }

    fn prime_isolated(&self) -> Result<()> {
        if self.workspace.borrow().is_none() {
            ensure!(matches!(self.context, DraftContext::Gqa(_)), "MLA draft lost its admitted workspace");
            *self.workspace.borrow_mut() = Some(self.workspace(self.max_sequences)?);
        }
        self.prime_metadata()?;
        if matches!(self.context, DraftContext::Mla(_)) { self.reset_context(0, 0)?; }
        self.library.cuda_zero_bytes(self.taps.buffer, self.taps.buffer.bytes)?;
        self.update(&(0..TAP_ROWS).map(|tap_row| ContextRow { tap_row, slot: 0,
            position: tap_row }).collect::<Vec<_>>())?;
        if matches!(self.context, DraftContext::Mla(_)) {
            for slot in 0..self.slots { self.reset_context(slot, TAP_ROWS)?; }
        }
        // SAFETY: this drafter owns its live startup stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        let input = |_: &[u32], out: CuteafdDeviceBuffer, stream: *mut c_void| {
            // SAFETY: the draft workspace and its startup stream are live.
            unsafe { self.library.cuda_zero_bytes_async(out, out.bytes, stream) }
        };
        for count in 1..=self.max_sequences {
            let sequences = (0..count).map(|slot| DraftSeq { slot, anchor: self.cfg.mask_token,
                position: TAP_ROWS, valid_from: TAP_ROWS }).collect::<Vec<_>>();
            let head = |_: *const c_void, out: *mut f32, rows: usize, stream: *mut c_void| {
                let bytes = rows * self.cfg.vocab * 4;
                // SAFETY: initialized scratch logits precede selection on this stream.
                unsafe { self.library.cuda_zero_bytes_async(CuteafdDeviceBuffer { ptr: out.cast(), bytes,
                    ..Default::default() }, bytes, stream) }
            };
            self.draft_device_with(&sequences, &input, TargetHead::Launch(Box::new(head)))?;
        }
        for layer in &self.layers {
            if let DraftAttention::Gqa(layer) = &layer.attention {
                for ring in [&layer.k_ring, &layer.v_ring] {
                    self.library.cuda_zero_bytes(ring.buffer, ring.buffer.bytes)?;
                }
            }
        }
        for slot in 0..self.slots { self.discard_context(slot)?; }
        self.startup_primed.set(true);
        Ok(())
    }

    fn prime_metadata(&self) -> Result<()> {
        let workspace = self.workspace.borrow();
        let w = workspace.as_ref().context("DFlash startup workspace missing")?;
        let mut buffers = vec![&self.context_positions, &w.positions, &w.anchors, &w.ids];
        match (&self.context, &w.attention) {
            (DraftContext::Gqa(context), AttentionWorkspace::Gqa(attention)) => {
                buffers.extend([&context.slots, &attention.tables]);
            }
            (DraftContext::Mla(context), AttentionWorkspace::Mla(attention)) => {
                buffers.extend([&context.slots, &attention.positions, &attention.slots,
                    &attention.lengths, &attention.query_offsets]);
                for layer in &self.layers {
                    let DraftAttention::Mla(layer) = &layer.attention else { anyhow::bail!("MLA startup layer mismatch") };
                    buffers.push(&layer.page_table);
                }
            }
            _ => anyhow::bail!("DFlash startup context/workspace mismatch"),
        }
        for buffer in buffers {
            // Startup reserves each full pinned entry: ordered copy fence, not a compute drain.
            self.put(buffer, &vec![0; buffer.buffer.bytes])?;
        }
        // SAFETY: startup owns every destination and its stream; no peer waits exist yet.
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
    }

    fn workspace(&self, sequences: usize) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter/workspace");
        let c = &self.cfg;
        let rows = sequences * c.block;
        let drafted = sequences * c.drafts();
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256));
        let head_workspace = alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let attention_workspace = {
            let _scope = cuteafd_ffi::memory_ledger::scope("workspace/draft-attention");
            alloc(self.library.glm_dflash_attention_workspace(sequences, c.kv_heads, RING + c.block, c.head_dim)?)?
        };
        Ok(Workspace {
            sequences,
            h: alloc(rows * c.hidden * 2)?,
            n: alloc(rows * c.hidden * 2)?,
            conv: alloc(rows * c.hidden * 2)?,
            dynamic: alloc(rows * c.conv_width() * 2)?,
            attention: AttentionWorkspace::Gqa(GqaWorkspace {
                qkv: alloc(rows * c.qkv_width() * 2)?,
                q: alloc(rows * c.heads * c.head_dim * 2)?,
                k: alloc(rows * c.kv_width() * 2)?,
                v: alloc(rows * c.kv_width() * 2)?,
                attn: alloc(rows * c.heads * c.head_dim * 2)?,
                tables: alloc(3 * sequences * 4)?,
                scratch: attention_workspace,
            }),
            delta: alloc(rows * c.hidden * 2)?,
            gate_up: alloc(rows * 2 * c.intermediate * 2)?,
            act: alloc(rows * c.intermediate * 2)?,
            logits: alloc(rows * c.vocab * 4)?,
            unary: alloc(drafted * 16 * 4)?,
            candidates: alloc(drafted * 16 * 4)?,
            projected: alloc(rows * c.rank * 2)?,
            anchors: alloc(sequences * 4)?,
            ids: alloc(rows * 4)?,
            tokens: alloc(drafted * 4)?,
            features: alloc(drafted * 16)?,
            landing: crate::shared::memory::HostAllocation::new(self.library, drafted * 20)?,
            positions: alloc(rows * 8)?,
            topk_workspace: alloc(self.library.glm_dflash_topk_workspace(drafted)?)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, c.hidden as u32,
                rows as u32, c.vocab as u32)? },
            _head_workspace: head_workspace,
        })
    }

    /// Drafts `block - 1` tokens after each sequence's anchor. `anchor_rows`
    /// holds the anchors' embedding rows; `head` is the target's vocabulary head.
    pub fn draft(&self, sequences: &[DraftSeq], anchor_rows: &[u8], head: TargetHead<'_>) -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Rows(anchor_rows), self.head_call(&head)?, false)
    }

    /// [`Self::draft`] with the block's input rows gathered from the target's
    /// embedding table by token id: each anchor, then the mask token (the
    /// drafter's mask row is that table row).
    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_call(&head)?, true)
    }

    /// Diagnostic device-input replay retains the explicit head arithmetic selection.
    pub fn draft_device_diagnostic(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>) -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_call(&head)?, false)
    }

    /// Device input bridge for a drafter on another GPU. Existing same-device
    /// callers keep `draft_device`; the bridge must order input writes before
    /// returning and keep its owners alive through the draft's stream drain.
    pub(crate) fn draft_device_with(&self, sequences: &[DraftSeq], input: &InputLaunch<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Launch(input), self.head_call(&head)?, true)
    }

    fn head_call<'h>(&self, head: &'h TargetHead<'h>) -> Result<HeadCall<'h>> {
        Ok(match head {
            TargetHead::Bf16(owner) => HeadCall::Bf16(self.borrowed_head(owner.buffer)?),
            TargetHead::Bf16Buffer(buffer) => {
                ensure!(buffer.device_id == self.library.cuda_get_device()?, "draft head borrow on wrong device");
                HeadCall::Bf16(self.borrowed_head(**buffer)?)
            }
            TargetHead::Launch(launch) => HeadCall::Launch(launch),
        })
    }

    fn borrowed_head(&self, owner: CuteafdDeviceBuffer) -> Result<*const c_void> {
        let bytes = self.cfg.vocab.checked_mul(self.cfg.hidden).and_then(|n| n.checked_mul(2))
            .context("DFlash borrowed BF16 head byte count overflow")?;
        ensure!(!owner.ptr.is_null() && owner.bytes == bytes,
            "DFlash requires the target-owned BF16 head [{}, {}] ({bytes} bytes); compact target-head \
             consumer is unsupported, no private head copy will be allocated", self.cfg.vocab, self.cfg.hidden);
        Ok(owner.ptr)
    }

    fn draft_from(&self, sequences: &[DraftSeq], input: DraftInput<'_, '_>, head: HeadCall<'_>,
        serving: bool) -> Result<Vec<Draft>> {
        ensure!(!matches!(head, HeadCall::Bf16(p) if p.is_null()), "DFlash has no borrowed target head");
        let c = &self.cfg;
        let (s_count, block, h) = (sequences.len(), c.block, c.hidden);
        ensure!(!matches!((&self.context, input), (DraftContext::Mla(_), DraftInput::Rows(_))),
            "MLA draft borrows the target device embedding; host mask-row fallback is not admitted");
        let rows_ok = match input {
            DraftInput::Rows(r) => r.len() == s_count * h * 2,
            DraftInput::Table(_) | DraftInput::Launch(_) => true,
        };
        ensure!(s_count > 0 && s_count <= self.max_sequences && rows_ok, "draft step of {s_count} sequences");
        let rows = s_count * block;
        let mut slot = self.workspace.borrow_mut();
        if slot.as_ref().is_none_or(|w| w.sequences < s_count) {
            ensure!(matches!(self.context, DraftContext::Gqa(_)), "MLA draft exceeded its admitted workspace");
            *slot = None;
            *slot = Some(self.workspace(self.max_sequences)?);
        }
        let w = slot.as_ref().context("draft workspace")?;
        let mut mla_lease = match &self.context {
            DraftContext::Gqa(_) => None,
            DraftContext::Mla(context) => Some(MlaPageLease { context, library: self.library,
                stream: self.stream, reservations: Vec::new() }),
        };
        if let Some(lease) = &mut mla_lease {
            let mut state = lease.context.state.borrow_mut();
            // Validate every frontier before reserving or enqueueing any sequence.
            let mut seen = std::collections::BTreeSet::new();
            for seq in sequences {
                ensure!(seen.insert(seq.slot) && state.rings.frontier(seq.slot)? == (seq.valid_from, seq.position),
                    "MLA draft request differs from bound origin/frontier");
            }
            for seq in sequences {
                lease.reservations.push(state.reserve(seq.slot, seq.position, block)?);
            }
        }
        let mla_batches = match (&self.context, &w.attention) {
            (DraftContext::Gqa(_), AttentionWorkspace::Gqa(_)) => {
                for layer in &self.layers { layer.attention.gqa()?; }
                None
            }
            (DraftContext::Mla(context), AttentionWorkspace::Mla(attention)) => {
                let requests = sequences.iter().map(|seq| cuteafd_loader::formats::dflash::MlaDraftCacheRequest {
                    slot: seq.slot, position: seq.position, valid_from: seq.valid_from }).collect::<Vec<_>>();
                let state = context.state.borrow();
                let mut all = Vec::with_capacity(self.layers.len());
                for (index, layer) in self.layers.iter().enumerate() {
                    let DraftAttention::Mla(layer) = &layer.attention else { anyhow::bail!("MLA draft layer mismatch") };
                    let chunks = state.block_batches(index, &requests,
                        &mla_lease.as_ref().expect("MLA lease").reservations,
                        context.layout.cache().layers[index].window().is_none())?;
                    for chunk in &chunks {
                        let chunk_rows = chunk.positions.len();
                        let offset = chunk.first_sequence * block * h * 2;
                        let batch = attention.batch(context, layer, chunk_rows, chunk.lengths.len(), chunk.page_table_width,
                            draft_view(w.conv.buffer, offset, chunk_rows * h * 2)?,
                            draft_view(w.delta.buffer, offset, chunk_rows * h * 2)?, attention.slots.buffer, attention.positions.buffer);
                        layer.program.check_launch(layer.weights(), batch)?;
                    }
                    all.push(chunks);
                }
                Some(all)
            }
            _ => anyhow::bail!("DFlash context/workspace mismatch"),
        };
        let mut positions = Vec::with_capacity(rows);
        let mut tables = vec![0i32; 3 * s_count];
        for (i, seq) in sequences.iter().enumerate() {
            ensure!(seq.slot < self.slots, "ring slot {} of {}", seq.slot, self.slots);
            positions.extend((seq.position..seq.position + block).map(|p| p as i64));
            tables[i] = seq.slot as i32;
            tables[s_count + i] = seq.position.saturating_sub(seq.valid_from).min(RING) as i32;
            tables[2 * s_count + i] = seq.position as i32;
        }
        // Eager per-round launch: keep the arena capacity-sized, not this work bound.
        // A future graph path must read changing lengths on-device instead.
        let max_keys = tables[s_count..2 * s_count].iter().copied().max().unwrap() as usize + block;
        self.put(&w.positions, bytes_of(&positions))?;
        if let AttentionWorkspace::Gqa(attention) = &w.attention {
            self.put(&attention.tables, bytes_of(&tables))?;
        }
        let anchors: Vec<u32> = sequences.iter().map(|s| s.anchor).collect();
        self.put(&w.anchors, bytes_of(&anchors))?;
        match input {
            DraftInput::Rows(anchor_rows) => {
                let mut embed = Vec::with_capacity(rows * h * 2);
                for i in 0..s_count {
                    embed.extend_from_slice(&anchor_rows[i * h * 2..(i + 1) * h * 2]);
                    for _ in 1..block {
                        embed.extend_from_slice(&self.mask_row);
                    }
                }
                self.put(&w.h, &embed)?;
            }
            DraftInput::Table(embedding) => {
                let ids = draft_input_ids(sequences, c.mask_token, block);
                embedding.embed(&ids, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
            DraftInput::Launch(input) => {
                let ids = draft_input_ids(sequences, c.mask_token, block);
                input(&ids, w.h.buffer, self.stream)?;
            }
        }
        let (s, l) = (self.stream, self.library);
        let (eps, group, inter) = (c.eps, c.group, c.intermediate);
        let attention_width = c.heads * c.head_dim;
        let forward_span = self.timing.begin("forward", None, rows, s)?;
        // SAFETY: every workspace buffer holds `rows` rows of its width and the
        // weights their checkpoint shapes; the stream orders the chain.
        unsafe {
            l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                let layer_span = self.timing.begin("layer", Some(index), rows, s)?;
                self.linear(w.n.buffer.ptr, &layer.attn_conv, 0, w.dynamic.buffer.ptr,
                    rows, h, c.conv_width())?;
                l.glm_dflash_conv(w.n.buffer.ptr, w.dynamic.buffer.ptr, layer.attn_base.buffer.ptr, w.conv.buffer.ptr,
                    rows, block, h, group, s)?;
                match (&layer.attention, &w.attention, &self.context) {
                    (DraftAttention::Gqa(a), AttentionWorkspace::Gqa(attention), DraftContext::Gqa(_)) => {
                        self.linear(w.conv.buffer.ptr, &a.qkv, 0, attention.qkv.buffer.ptr, rows, h, c.qkv_width())?;
                        l.glm_dflash_qk_rope(attention.qkv.buffer.ptr, a.q_norm.buffer.ptr, a.k_norm.buffer.ptr,
                            w.positions.buffer.ptr, std::ptr::null(), attention.q.buffer.ptr, attention.k.buffer.ptr, attention.v.buffer.ptr,
                            rows, c.heads, c.kv_heads, c.theta, eps, false, c.head_dim, s)?;
                        l.glm_dflash_attention(attention.q.buffer.ptr, attention.k.buffer.ptr, attention.v.buffer.ptr,
                            a.k_ring.buffer.ptr, a.v_ring.buffer.ptr, attention.tables.buffer.ptr,
                            at(&attention.tables, s_count * 4), at(&attention.tables, 2 * s_count * 4),
                            attention.attn.buffer.ptr, attention.scratch.buffer.ptr, s_count, block, c.heads, c.kv_heads, RING,
                            max_keys, if c.row_window { c.window } else { 0 }, 1.0 / (c.head_dim as f32).sqrt(), self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, c.head_dim, s)?;
                        self.linear(attention.attn.buffer.ptr, &a.o, 0, w.delta.buffer.ptr, rows, attention_width, h)?;
                    }
                    (DraftAttention::Mla(a), AttentionWorkspace::Mla(attention), DraftContext::Mla(context)) => {
                        for chunk in &mla_batches.as_ref().context("MLA draft checked batches missing")?[index] {
                            // ordered copy fence, not a compute drain; m64 scratch is reused after each ordered launch.
                            self.put(&attention.positions, bytes_of(&chunk.positions))?;
                            self.put(&attention.slots, bytes_of(&chunk.slots))?;
                            self.put(&attention.lengths, bytes_of(&chunk.lengths))?;
                            self.put(&attention.query_offsets, bytes_of(&chunk.query_offsets))?;
                            self.put(&a.page_table, bytes_of(&chunk.page_table))?;
                            let chunk_rows = chunk.positions.len();
                            let offset = chunk.first_sequence * block * h * 2;
                            let batch = attention.batch(context, a, chunk_rows, chunk.lengths.len(), chunk.page_table_width,
                                draft_view(w.conv.buffer, offset, chunk_rows * h * 2)?,
                                draft_view(w.delta.buffer, offset, chunk_rows * h * 2)?, attention.slots.buffer, attention.positions.buffer);
                            a.program.launch(a.weights(), batch, s)?;
                        }
                    }
                    _ => anyhow::bail!("DFlash layer/workspace mismatch"),
                }
                l.glm_dflash_conv_residual_norm(w.delta.buffer.ptr, w.dynamic.buffer.ptr, layer.attn_base.buffer.ptr,
                    w.h.buffer.ptr, layer.post_norm.buffer.ptr, w.h.buffer.ptr, w.n.buffer.ptr, rows, block, h, group, eps, s)?;
                self.linear(w.n.buffer.ptr, &layer.mlp_conv, 0, w.dynamic.buffer.ptr,
                    rows, h, c.conv_width())?;
                l.glm_dflash_conv(w.n.buffer.ptr, w.dynamic.buffer.ptr, layer.mlp_base.buffer.ptr, w.conv.buffer.ptr,
                    rows, block, h, group, s)?;
                self.linear(w.conv.buffer.ptr, &layer.gate_up, 0, w.gate_up.buffer.ptr,
                    rows, h, 2 * inter)?;
                l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, inter, s)?;
                self.linear(w.act.buffer.ptr, &layer.down, 0, w.delta.buffer.ptr, rows,
                    inter, h)?;
                let next = self.layers.get(index + 1).map_or(self.norm.buffer.ptr, |n| n.input_norm.buffer.ptr);
                l.glm_dflash_conv_residual_norm(w.delta.buffer.ptr, w.dynamic.buffer.ptr, layer.mlp_base.buffer.ptr,
                    w.h.buffer.ptr, next, w.h.buffer.ptr, w.n.buffer.ptr, rows, block, h, group, eps, s)?;
                self.timing.end(layer_span, s)?;
            }
            self.timing.end(forward_span, s)?;
            let head_span = self.timing.begin("head_topk", None, rows, s)?;
            match &head {
                HeadCall::Bf16(weight) if serving => crate::shared::draft::launch_bf16_head(&w.head,
                    w.n.buffer.ptr.cast(), weight.cast(), w.logits.buffer.ptr.cast(), rows, s)?,
                HeadCall::Bf16(weight) => launch_draft_head(l, &w.head, w.n.buffer.ptr, *weight,
                    w.logits.buffer.ptr.cast(), rows, h, c.vocab, s, self.head_mode.get())?,
                HeadCall::Launch(launch) => launch(w.n.buffer.ptr.cast_const(), w.logits.buffer.ptr.cast(), rows, s)?,
            }
            l.glm_dflash_topk(w.logits.buffer.ptr, w.unary.buffer.ptr, w.candidates.buffer.ptr,
                w.topk_workspace.buffer.ptr, s_count, block, c.drafts(), c.vocab, s)?;
            self.timing.end(head_span, s)?;
            let select_span = self.timing.begin("selector_confidence", None, rows, s)?;
            self.linear(w.n.buffer.ptr, &self.projection, 0,
                w.projected.buffer.ptr, rows, h, c.rank)?;
            l.glm_dflash_select(self.predecessor.buffer.ptr, self.successor.buffer.ptr, w.projected.buffer.ptr,
                w.candidates.buffer.ptr, w.unary.buffer.ptr, w.anchors.buffer.ptr, w.tokens.buffer.ptr,
                w.features.buffer.ptr, s_count, block, c.drafts(), c.rank, s)?;
            self.timing.end(select_span, s)?;
            let drafted = s_count * c.drafts();
            // The round's results download on its stream; one host wait then
            // covers the round and both copies (no second, synchronous drain).
            // SAFETY: the device buffers hold `drafted` rows; the landing is
            // pinned, owned by the workspace and read only after the wait.
            let landing = |offset: usize, bytes: usize| cuteafd_ffi::CuteafdHostBuffer {
                ptr: w.landing.buffer.ptr.cast::<u8>().add(offset).cast(), bytes, ..w.landing.buffer };
            l.copy_d2h_host_buffer_async(landing(0, drafted * 4),
                CuteafdDeviceBuffer { bytes: drafted * 4, ..w.tokens.buffer }, drafted * 4, s)?;
            l.copy_d2h_host_buffer_async(landing(drafted * 4, drafted * 16),
                CuteafdDeviceBuffer { bytes: drafted * 16, ..w.features.buffer }, drafted * 16, s)?;
            self.round.borrow_mut().wait(s)?;
        }
        if let Some(mut lease) = mla_lease.take() {
            let mut state = lease.context.state.borrow_mut();
            while let Some(&reservation) = lease.reservations.last() {
                state.pool.commit(reservation, 0)?;
                lease.reservations.pop();
            }
        }
        self.timing.publish_completed()?;
        self.timing.dump_head_input(w.n.buffer, rows, h)?;
        let drafted = s_count * c.drafts();
        let (tokens, features) = w.landing.bytes()[..drafted * 20].split_at(drafted * 4);
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..s_count).map(|i| Draft {
            tokens: (0..c.drafts()).map(|j| word(&tokens, i * c.drafts() + j)).collect(),
            features: (0..c.drafts()).map(|j| {
                let n = (i * c.drafts() + j) * 4;
                [0, 1, 2, 3].map(|k| f32::from_bits(word(&features, n + k)))
            }).collect(),
            confidence: Vec::new(),
        }).collect())
    }

    /// The last draft step's final-norm rows [sequences * block, hidden] BF16.
    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("no draft step ran")?;
        let bytes = sequences * self.cfg.block * self.cfg.hidden * 2;
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..w.n.buffer })?;
        Ok(out)
    }

    /// Copies BF16 tap rows (host, [n, taps * hidden]) into the tap buffer.
    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        self.put(&self.taps, rows)
    }
}

/// Teacher-forced drafter replay on a golden sequence (after Hugh Madden's
/// glm53f-afd draft_record / draft_replay): the drafter's context follows
/// the golden taps one row at a time and it drafts after every token from
/// `start` on through the actual immutable resident mode. Legacy dual-copy
/// implementors retain their diagnostic modes. Prints the drafts accepted as a prefix of
/// the text and of the target's greedy picks (`greedy[p]`: the argmax of
/// the golden logits after token p; a draft past a miss of the text is not
/// scored against them), the first draft's greedy agreement and the draft
/// time; then how often the two modes drafted the same tokens.
/// `taps(first, n)` returns tap rows [n, taps * hidden] BF16.
/// What [`replay`] needs of a block drafter (GLM DFlash2, MiMo DFlash).
pub(crate) trait ReplayDrafter {
    fn block(&self) -> usize;
    /// Tokens one draft proposes (DFlash: the block's mask rows).
    fn drafts(&self) -> usize {
        self.block() - 1
    }
    /// Sequences a draft step takes (and ring slots).
    fn sequences(&self) -> usize;
    /// Legacy dual-format implementations may retain their diagnostic arms.
    fn has_fp8(&self) -> bool { false }
    fn set_fp8(&self, _on: bool) {}
    /// New immutable implementations report one resident arithmetic mode,
    /// without asking replay to switch precision or weight storage.
    fn resident_modes(&self) -> Vec<ReplayMode> {
        if self.has_fp8() {
            vec![ReplayMode { name: "BF16", legacy_fp8: Some(false) },
                ReplayMode { name: "FP8", legacy_fp8: Some(true) }]
        } else {
            vec![ReplayMode { name: "BF16", legacy_fp8: None }]
        }
    }
    /// Ring context of slot 0 from tap rows [n, taps * hidden] at positions `first..first + n`.
    fn context(&self, taps: &[u8], first: usize) -> Result<()>;
    /// Draft tokens after each (slot, anchor, position).
    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>>;
    /// Longest context update one call takes.
    fn tap_rows(&self) -> usize;
    /// The last draft step's final-norm rows [sequences * block, hidden] BF16.
    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReplayMode {
    pub name: &'static str,
    pub legacy_fp8: Option<bool>,
}

fn activate_replay_mode(drafter: &(impl ReplayDrafter + ?Sized), mode: ReplayMode) {
    if let Some(on) = mode.legacy_fp8 { drafter.set_fp8(on); }
}

impl ReplayDrafter for DflashDrafter<'_> {
    fn block(&self) -> usize {
        self.cfg.block
    }

    fn sequences(&self) -> usize {
        self.max_sequences.min(self.slots)
    }

    fn resident_modes(&self) -> Vec<ReplayMode> {
        vec![ReplayMode { name: self.representation.name(), legacy_fp8: None }]
    }

    fn context(&self, taps: &[u8], first: usize) -> Result<()> {
        let n = taps.len() / (self.cfg.taps.len() * self.cfg.hidden * 2);
        self.put_taps(taps)?;
        self.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r }).collect::<Vec<_>>())?;
        // SAFETY: diagnostic replay owns this stream. Complete the reads of
        // taps and metadata before the next context call overwrites them.
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
    }

    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>> {
        let seqs: Vec<DraftSeq> = seqs.iter().map(|&(slot, anchor, position)| DraftSeq { slot, anchor, position, valid_from: 0 }).collect();
        // Replay's caller keeps the target BF16 head owner live throughout.
        Ok(self.draft_from(&seqs, DraftInput::Rows(anchor_rows), HeadCall::Bf16(head), false)?.into_iter()
            .map(|d| d.tokens).collect())
    }

    fn tap_rows(&self) -> usize {
        TAP_ROWS
    }

    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        DflashDrafter::last_hidden(self, sequences)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replay(drafter: &(impl ReplayDrafter + ?Sized), tokens: &[u32], greedy: &[u32],
    taps: &dyn Fn(usize, usize) -> Result<Vec<u8>>, embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>, head: *const c_void,
    start: usize) -> Result<()> {
    replay_with(drafter, tokens, greedy, taps, embed, &|sequences, rows| {
        let sequences: Vec<_> = sequences.iter().map(|s| (s.slot, s.anchor, s.position)).collect();
        drafter.draft_tokens(&sequences, rows, head)
    }, start)
}

/// Diagnostic replay through a caller-owned draft path. A remote drafter can
/// scope its device and borrow the target head through its transport callback,
/// rather than treating a raw head pointer as resident on the draft device.
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay_with(drafter: &(impl ReplayDrafter + ?Sized), tokens: &[u32], greedy: &[u32],
    taps: &dyn Fn(usize, usize) -> Result<Vec<u8>>, embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>,
    draft: &dyn Fn(&[DraftSeq], &[u8]) -> Result<Vec<Vec<u32>>>, start: usize) -> Result<()> {
    let (block, drafts) = (drafter.block(), drafter.drafts());
    ensure!(tokens.len() > start + block && greedy.len() >= tokens.len(), "replay needs more than {} tokens", start + block);
    let anchors: Vec<usize> = (start..tokens.len() - block).collect();
    let modes = drafter.resident_modes();
    let mut outputs: Vec<Vec<Vec<u32>>> = Vec::new();
    // Final-norm rows of the first 64 anchors per mode.
    let mut hidden: Vec<Vec<Vec<f32>>> = Vec::new();
    for &mode in &modes {
        activate_replay_mode(drafter, mode);
        let (mut done, mut seconds) = (0usize, Vec::with_capacity(anchors.len()));
        let mut out = Vec::with_capacity(anchors.len());
        let mut rows_seen = Vec::new();
        let (mut text, mut greedy_ok, mut first) = (0usize, 0usize, 0usize);
        for &p in &anchors {
            while done < p {
                let n = (p - done).min(drafter.tap_rows());
                drafter.context(&taps(done, n)?, done)?;
                done += n;
            }
            let rows = embed(&[tokens[p]])?;
            let timer = std::time::Instant::now();
            let draft = draft(&[DraftSeq { slot: 0, anchor: tokens[p], position: p, valid_from: 0 }], &rows)?.remove(0);
            seconds.push(timer.elapsed().as_secs_f64());
            if rows_seen.len() < 64 {
                rows_seen.push(drafter.last_hidden(1)?.chunks_exact(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect::<Vec<f32>>());
            }
            text += draft.iter().zip(&tokens[p + 1..]).take_while(|(d, t)| d == t).count();
            let mut kept = 0;
            while kept < drafts && draft[kept] == greedy[p + kept] && (kept == 0 || draft[kept - 1] == tokens[p + kept]) {
                kept += 1;
            }
            greedy_ok += kept;
            first += usize::from(draft[0] == greedy[p]);
            out.push(draft);
        }
        seconds.sort_by(f64::total_cmp);
        let n = anchors.len() as f64;
        println!("draft replay {}: {} anchors, accepted vs text {:.3}, vs greedy {:.3} of {drafts}, first draft = \
            greedy {:.1}%, draft median {:.3} ms (p10 {:.3}, p90 {:.3})", mode.name,
            anchors.len(), text as f64 / n, greedy_ok as f64 / n, 100.0 * first as f64 / n,
            1e3 * seconds[seconds.len() / 2], 1e3 * seconds[seconds.len() / 10], 1e3 * seconds[seconds.len() * 9 / 10]);
        outputs.push(out);
        hidden.push(rows_seen);
        // Draft steps of several sequences (slot i drafts at the last anchor's position).
        let p = *anchors.last().unwrap();
        let mut line = String::new();
        for count in [1usize, 2, 4, 8, 16].into_iter().filter(|&c| c <= drafter.sequences()) {
            let seqs: Vec<DraftSeq> = (0..count)
                .map(|slot| DraftSeq { slot, anchor: tokens[p], position: p, valid_from: 0 }).collect();
            let rows = embed(&vec![tokens[p]; count])?;
            let mut times = Vec::new();
            for run in 0..9 {
                let timer = std::time::Instant::now();
                draft(&seqs, &rows)?;
                if run >= 2 {
                    times.push(timer.elapsed().as_secs_f64());
                }
            }
            times.sort_by(f64::total_cmp);
            line += &format!(" {count}: {:.2}", 1e3 * times[times.len() / 2]);
        }
        println!("draft replay {} step ms by sequences (median of 7):{line}", mode.name);
    }
    if let [bf16, fp8] = &outputs[..] {
        let same = bf16.iter().zip(fp8).filter(|(a, b)| a == b).count();
        let prefix: usize = bf16.iter().zip(fp8).map(|(a, b)| a.iter().zip(b).take_while(|(x, y)| x == y).count()).sum();
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
            let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            dot / (norm(a) * norm(b)).max(1e-30)
        };
        let worst = hidden[0].iter().zip(&hidden[1]).map(|(a, b)| cosine(a, b)).fold(1f64, f64::min);
        println!("draft replay BF16 vs FP8: identical drafts {same}/{} ({:.1}%), common prefix {:.2} of {drafts}, \
            worst final-norm cosine over the first {} anchors {worst:.6}", bf16.len(), 100.0 * same as f64 / bf16.len() as f64,
            prefix as f64 / bf16.len() as f64, hidden[0].len());
    }
    if modes.iter().any(|mode| mode.legacy_fp8.is_some()) { drafter.set_fp8(true); }
    Ok(())
}

/// The golden sequence and its greedy picks (the argmax of each row of
/// `logits.bin`, [tokens, vocab] F32) from a golden directory.
pub(crate) fn golden_sequence(dir: &Path, vocab: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    use std::io::Read;
    let tokens: Vec<u32> = std::fs::read(dir.join("tokens.bin"))?.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut file = std::io::BufReader::with_capacity(1 << 24, std::fs::File::open(dir.join("logits.bin"))?);
    let mut row = vec![0u8; vocab * 4];
    let mut greedy = Vec::with_capacity(tokens.len());
    for _ in 0..tokens.len() {
        file.read_exact(&mut row)?;
        let (mut best, mut value) = (0u32, f32::NEG_INFINITY);
        for (i, b) in row.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes(b.try_into().unwrap());
            if v > value {
                (best, value) = (i as u32, v);
            }
        }
        greedy.push(best);
    }
    Ok((tokens, greedy))
}

#[cfg(test)]
mod checkpoint_header_tests {
    use super::*;

    #[test]
    fn native_pool_exhaustion_survives_context_and_the_boxed_family_boundary() {
        use cuteafd_engine::prefix::{DraftPageError, PoolExhausted};
        let pages = PoolExhausted { needed: 4, free: 2, capacity: 8 };
        let native = anyhow::Error::new(DraftPageError::Exhausted(pages)).context("MLA reservation");
        let boxed = box_draft_page_error(native);
        assert!(matches!(boxed.downcast_ref::<DraftPageError>(),
            Some(DraftPageError::Exhausted(actual)) if *actual == pages));
    }

    struct HeaderSnapshot(std::path::PathBuf);
    impl Drop for HeaderSnapshot {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    fn expert_only_snapshot() -> HeaderSnapshot {
        use std::io::Write;
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::var_os("CARGO_TARGET_DIR").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("target"));
        let path = root.join("header-test-fixtures").join(format!("dflash-{}-{}", std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::create_dir_all(&path).unwrap();
        let fixture = HeaderSnapshot(path);
        let config = serde_json::json!({
            "model_type": "glm_moe_dsa", "quantization_config": {"quant_method": "fp8"},
            "vocab_size": 16, "hidden_size": 128, "num_hidden_layers": 1,
            "num_attention_heads": 1, "q_lora_rank": 128, "kv_lora_rank": 128,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "index_n_heads": 1, "index_head_dim": 128, "index_topk": 1,
            "first_k_dense_replace": 0, "intermediate_size": 128, "n_routed_experts": 1,
            "num_experts_per_tok": 1, "moe_intermediate_size": 128, "n_shared_experts": 1,
            "routed_scaling_factor": 1.0, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        });
        std::fs::write(fixture.0.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
        let mut headers = serde_json::Map::new();
        let mut index = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, bytes) in [
            ("lm_head.weight".to_owned(), "BF16", vec![16, 128], 4096),
            ("model.layers.0.mlp.experts.0.gate_proj.weight".into(), "F8_E4M3", vec![128, 128], 16384),
            ("model.layers.0.mlp.experts.0.up_proj.weight".into(), "F8_E4M3", vec![128, 128], 16384),
            ("model.layers.0.mlp.experts.0.down_proj.weight".into(), "F8_E4M3", vec![128, 128], 16384),
            ("model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv".into(), "F32", vec![1, 1], 4),
            ("model.layers.0.mlp.experts.0.up_proj.weight_scale_inv".into(), "F32", vec![1, 1], 4),
            ("model.layers.0.mlp.experts.0.down_proj.weight_scale_inv".into(), "F32", vec![1, 1], 4),
        ] {
            headers.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape,
                "data_offsets": [offset, offset + bytes]}));
            index.insert(name, serde_json::Value::String("weights.safetensors".into()));
            offset += bytes;
        }
        let header = serde_json::to_vec(&headers).unwrap();
        let mut file = std::fs::File::create(fixture.0.join("weights.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        file.set_len((8 + header.len() + offset) as u64).unwrap();
        std::fs::write(fixture.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap()).unwrap();
        fixture
    }

    #[test]
    fn target_head_guard_checks_checkpoint_independently_of_catalog_headers() {
        let fixture = expert_only_snapshot();
        let catalog = cuteafd_loader::read_expert_catalog(&fixture.0).unwrap();
        assert!(catalog.fp8().is_some());
        assert_eq!(catalog.tensors().len(), 1);
        assert_eq!(catalog.tensor("lm_head.weight").unwrap().metadata.dtype, DType::Bf16);
        check_snapshot_target_head(&fixture.0, 128, 16, false).unwrap();
        assert!(check_snapshot_target_head(&fixture.0, 128, 16, true).is_err());
        assert!(check_snapshot_target_head(&fixture.0, 128, 32, false).is_err());
        let index_path = fixture.0.join("model.safetensors.index.json");
        let mut index = cuteafd_loader::plan::checkpoint::read_json(&index_path).unwrap();
        index["weight_map"]["lm_head.weight"] = "missing.safetensors".into();
        std::fs::write(index_path, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(check_snapshot_target_head(&fixture.0, 128, 16, false).is_err());
    }

    fn native_head_snapshot(bits: usize) -> HeaderSnapshot {
        use std::io::Write;
        let fixture = expert_only_snapshot();
        let index_path = fixture.0.join("model.safetensors.index.json");
        let mut index = cuteafd_loader::plan::checkpoint::read_json(&index_path).unwrap();
        let map = index["weight_map"].as_object_mut().unwrap();
        map.remove("lm_head.weight");
        // Published K6 head storage, split across indexed shards to cover every companion.
        for (shard, tensors) in [
            ("head.safetensors", vec![("trellis", "I16", vec![8, 16, 16 * bits], 128 * 256 * bits / 8)]),
            ("head-vectors.safetensors", vec![("suh", "F16", vec![128], 256),
                ("svh", "F16", vec![256], 512), ("mcg", "I32", vec![], 4)]),
        ] {
            let mut headers = serde_json::Map::new();
            let mut offset = 0;
            for (suffix, dtype, shape, bytes) in tensors {
                let name = format!("lm_head.{suffix}");
                headers.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape,
                    "data_offsets": [offset, offset + bytes]}));
                map.insert(name, shard.into());
                offset += bytes;
            }
            let header = serde_json::to_vec(&headers).unwrap();
            let mut file = std::fs::File::create(fixture.0.join(shard)).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&header).unwrap();
            file.set_len((8 + header.len() + offset) as u64).unwrap();
        }
        std::fs::write(index_path, serde_json::to_vec(&index).unwrap()).unwrap();
        fixture
    }

    #[test]
    fn dflash_target_resolves_indexed_native_k6_head_without_bf16_or_side_metadata() {
        let fixture = native_head_snapshot(6);
        check_snapshot_target_head(&fixture.0, 128, 256, false).unwrap();
        assert!(check_snapshot_target_head(&fixture.0, 128, 256, true).is_err());
        assert!(check_snapshot_target_head(&fixture.0, 128, 128, false).is_err());
        assert!(check_snapshot_target_head(&fixture.0, 256, 256, false).is_err());
    }

    #[test]
    fn dflash_target_rejects_missing_k6_companions_duplicate_bf16_and_wrong_bits() {
        for suffix in ["trellis", "suh", "svh", "mcg"] {
            let fixture = native_head_snapshot(6);
            let path = fixture.0.join("model.safetensors.index.json");
            let mut index = cuteafd_loader::plan::checkpoint::read_json(&path).unwrap();
            index["weight_map"].as_object_mut().unwrap().remove(&format!("lm_head.{suffix}"));
            std::fs::write(path, serde_json::to_vec(&index).unwrap()).unwrap();
            assert!(check_snapshot_target_head(&fixture.0, 128, 256, false).is_err(), "missing {suffix}");
        }
        let fixture = native_head_snapshot(6);
        let path = fixture.0.join("model.safetensors.index.json");
        let mut index = cuteafd_loader::plan::checkpoint::read_json(&path).unwrap();
        index["weight_map"]["lm_head.weight"] = "weights.safetensors".into();
        std::fs::write(path, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(check_snapshot_target_head(&fixture.0, 128, 256, false).is_err());
        let fixture = native_head_snapshot(4);
        let error = check_snapshot_target_head(&fixture.0, 128, 256, false).unwrap_err().to_string();
        assert!(error.contains("requires K6") && error.contains("found K4"), "{error}");
    }

    #[test]
    fn dflash_target_rejects_malformed_or_unreadable_indexed_k6_companions() {
        use std::io::{Read, Seek, SeekFrom, Write};
        for (member, field, value) in [
            ("lm_head.suh", "dtype", serde_json::json!("BF16")),
            ("lm_head.svh", "shape", serde_json::json!([128])),
            ("lm_head.mcg", "dtype", serde_json::json!("F32")),
        ] {
            let fixture = native_head_snapshot(6);
            let path = fixture.0.join("head-vectors.safetensors");
            let mut file = std::fs::OpenOptions::new().read(true).write(true).open(path).unwrap();
            let mut size = [0; 8];
            file.read_exact(&mut size).unwrap();
            let mut bytes = vec![0; u64::from_le_bytes(size) as usize];
            file.read_exact(&mut bytes).unwrap();
            let mut header: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            header[member][field] = value;
            let header = serde_json::to_vec(&header).unwrap();
            file.seek(SeekFrom::Start(0)).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&header).unwrap();
            file.set_len((8 + header.len() + 772) as u64).unwrap();
            assert!(check_snapshot_target_head(&fixture.0, 128, 256, false).is_err(), "{member} {field}");
        }
        let fixture = native_head_snapshot(6);
        std::fs::remove_file(fixture.0.join("head-vectors.safetensors")).unwrap();
        assert!(check_snapshot_target_head(&fixture.0, 128, 256, false).is_err());
    }

    #[test]
    fn installed_glm_k6_index_resolves_a_dflash_target_head() {
        let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from).unwrap_or_else(||
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface"));
        let snapshot = home.join("hub/models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-Core-K6/snapshots")
            .join("5a4e83a09a11f3145332ed632d28a9894ea88225");
        if !snapshot.join("model.safetensors.index.json").is_file() {
            eprintln!("K6 checkpoint not installed; indexed native head fixture still runs");
            return;
        }
        let index = cuteafd_loader::plan::checkpoint::read_json(&snapshot.join("model.safetensors.index.json")).unwrap();
        assert!(index["weight_map"].get("lm_head.weight").is_none());
        assert!(index["weight_map"].get("lm_head.trellis").is_some());
        check_snapshot_target_head(&snapshot, 6144, 154880, false).unwrap();
    }

    fn fixture(dtype: DType) -> Checkpoint {
        Checkpoint { data: vec![0x80, 0x3f, 0, 0x40, 0x40, 0x40, 0x80, 0x40],
            tensors: HashMap::from([("fc.weight".into(), SafetensorsTensorMetadata {
                name: "fc.weight".into(), dtype, shape: vec![2, 2], byte_offset: 0, byte_length: 8,
            })]) }
    }

    #[test]
    fn bf16_payload_is_read_without_conversion() {
        let checkpoint = fixture(DType::Bf16);
        assert_eq!(checkpoint.bytes("fc.weight", &[2, 2]).unwrap(), checkpoint.data);
    }

    #[test]
    fn equal_width_other_dtypes_cannot_be_reinterpreted_as_bf16() {
        for dtype in [DType::F16, DType::I16] {
            let checkpoint = fixture(dtype.clone());
            let error = checkpoint.bytes("fc.weight", &[2, 2]).unwrap_err().to_string();
            assert!(error.contains("fc.weight") && error.contains("must be BF16"));
            assert!(error.contains(&format!("{dtype:?}")));
        }
    }

    #[test]
    fn bf16_dtype_does_not_bypass_shape_or_storage_guards() {
        let mut checkpoint = fixture(DType::Bf16);
        assert!(checkpoint.bytes("fc.weight", &[4, 1]).is_err());
        checkpoint.tensors.get_mut("fc.weight").unwrap().byte_length = 6;
        assert!(checkpoint.bytes("fc.weight", &[2, 2]).is_err());
    }

    #[test]
    fn admitted_buffers_move_once_and_drop_partial_failures_without_retry() {
        use cuteafd_loader::formats::dflash::MlaDraftBuffer;
        use std::rc::Rc;
        struct Owner(Rc<Cell<usize>>);
        impl Drop for Owner { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let dropped = Rc::new(Cell::new(0));
        let recipes = [MlaDraftBuffer { name: "h".into(), bytes: 256 },
            MlaDraftBuffer { name: "native_scratch".into(), bytes: 1024 }];
        let make = || Owner(dropped.clone());
        let mut buffers = AdmittedBuffers::new(&recipes, vec![make(), make()]).unwrap();
        let common = buffers.take("h").unwrap();
        assert!(buffers.take("h").is_err());
        let typed = buffers.take("native_scratch").unwrap();
        buffers.finish().unwrap();
        assert_eq!(dropped.get(), 0);
        drop((common, typed));
        assert_eq!(dropped.get(), 2);
        let mut buffers = AdmittedBuffers::new(&recipes, vec![make(), make()]).unwrap();
        let common = buffers.take("h").unwrap();
        assert!(buffers.finish().is_err());
        assert_eq!(dropped.get(), 3);
        drop(common);
        assert_eq!(dropped.get(), 4);
        assert!(AdmittedBuffers::new(&recipes, vec![make()]).is_err());
        assert_eq!(dropped.get(), 5);
        let duplicates = [recipes[0].clone(), recipes[0].clone()];
        assert!(AdmittedBuffers::new(&duplicates, vec![make(), make()]).is_err());
        assert_eq!(dropped.get(), 7);
    }

    fn mla_config() -> cuteafd_loader::formats::dflash::MlaDflashConfig {
        use cuteafd_loader::formats::{dflash::MlaDflashConfig,
            mla::MlaGeometry, block_dspark::DsparkRotaryPlan};
        let cfg = MlaDflashConfig { mla: MlaGeometry { hidden: 7168, heads: 64,
            q_rank: 1536, kv_rank: 512, nope: 128, rope: 64, value: 128 },
            intermediate: 14336, eps: 1e-5, block: 8, group: 16, selector_rank: 256,
            selector_top_k: 16, taps: vec![19, 37, 66, 78, 90], vocab: 163840,
            mask_token: 163592, max_context: 1048576, windows: vec![Some(4096), None],
            rotary: DsparkRotaryPlan { inv_freq: vec![1.; 32], attention_factor: 1., softmax_factor: 1. } };
        cfg
    }

    fn mla_cache_state() -> cuteafd_loader::formats::dflash::MlaDraftCacheState {
        cuteafd_loader::formats::dflash::MlaDraftCacheState::new(
            mla_config().cache_layout(2, 262144, cuteafd_core::MlaCacheFormat::Bf16).unwrap()).unwrap()
    }

    #[test]
    fn mla_pooled_producers_and_proposals_share_only_the_admitted_budget() {
        use cuteafd_loader::{families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation},
            formats::dflash::MlaDraftCacheRequest};
        let layout = mla_config().runtime_layout(GlmDraftRepresentation::Fp8Only,
            GlmDraftCapacity::new(2, 2, 8).unwrap(), 262144, 128, 188, 2, cuteafd_core::MlaCacheFormat::Bf16).unwrap();
        let mut state = MlaPagedState::new(&layout).unwrap();
        assert_eq!(state.pool.capacity(), 2);
        state.reset(0, 513).unwrap();
        state.reset(1, 17000).unwrap();
        let rows = (0..64).flat_map(|i| [
            ContextRow { tap_row: 2 * i, slot: 0, position: 513 + i },
            ContextRow { tap_row: 2 * i + 1, slot: 1, position: 17000 + i },
        ]).collect::<Vec<_>>();
        let (plan, reservations) = MlaContextUpdate::checked_paged(&mut state, &layout, &rows).unwrap();
        assert_eq!(state.pool.free(), 0);
        let full: std::collections::BTreeSet<_> = plan.slots[1].iter().copied().collect();
        assert_eq!(full.len(), 128);
        assert!(full.iter().all(|&row| (0..128).contains(&row)));
        assert_eq!(plan.slots[0][1], state.rings.context_rows(0, 1, 17000, 1).unwrap()[0]);
        for reservation in reservations { state.pool.commit(reservation, 64).unwrap(); }
        state.rings = plan.next;
        // Exhaustion is transactional, including the ring frontier and existing destinations.
        assert!(state.reserve(0, 577, 8).is_err());
        assert_eq!(state.rings.frontier(0).unwrap(), (513, 577));
        state.discard(1).unwrap();
        let proposal = state.reserve(0, 577, 8).unwrap();
        let request = [MlaDraftCacheRequest { slot: 0, position: 577, valid_from: 513 }];
        let batches = state.block_batches(1, &request, &[proposal], true).unwrap();
        assert_eq!(batches[0].slots, state.pool.destinations(proposal).unwrap());
        assert_eq!(&batches[0].page_table[..2], state.pool.page_table(proposal).unwrap());
        state.pool.commit(proposal, 0).unwrap();
        assert_eq!(state.pool.free(), 1);
        // Rebinding is a zero-page cold(P), even after fragmented/interleaved ownership.
        state.reset(0, 100001).unwrap();
        assert_eq!(state.pool.free(), state.pool.capacity());
        assert_eq!(state.rings.frontier(0).unwrap(), (100001, 100001));
        let (_, reservations) = MlaContextUpdate::checked_paged(&mut state, &layout,
            &[ContextRow { tap_row: 0, slot: 0, position: 100001 }]).unwrap();
        assert_eq!(state.pool.destinations(reservations[0]).unwrap()[0] % 64, 0);
        state.pool.commit(reservations[0], 0).unwrap();
        state.discard(0).unwrap();
        assert_eq!(state.pool.free(), state.pool.capacity());
    }

    #[test]
    fn glm_gqa_cold_restore_excludes_old_ring_rows_until_canonical_producers_replace_them() {
        for frontier in [63usize, 64, 65, 2047, 2048, 2049, 4097] {
            let mut reused = vec![u64::MAX; 2 * RING];
            let mut cold = vec![0u64; 2 * RING];
            let slot = 1;
            for count in 0..=RING + 1 {
                let position = frontier + count;
                let length = position.saturating_sub(frontier).min(RING);
                for p in position - length..position {
                    assert_eq!(reused[slot * RING + p % RING], cold[slot * RING + p % RING]);
                }
                let at = slot * RING + position % RING;
                reused[at] = position as u64 + 1;
                cold[at] = position as u64 + 1;
                assert!(reused[..RING].iter().all(|&row| row == u64::MAX));
            }
        }
    }

    #[test]
    fn glm_dflash_cache_hit_cold_origin_is_request_local_and_exact() {
        for frontier in [1usize, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1492] {
            let mut restored = mla_cache_state();
            let mut fresh = mla_cache_state();
            restored.reset(0, 0).unwrap();
            let old: Vec<_> = (0..4).map(|i| ContextRow { tap_row: i, slot: 0, position: i }).collect();
            restored = MlaContextUpdate::checked(&restored, 2, &old).unwrap().next;
            restored.reset(0, frontier).unwrap();
            fresh.reset(0, frontier).unwrap();
            // Interleaved no-spec work has its own slot and never alters the
            // drafting request's cold origin or its context address mapping.
            restored.reset(1, 0).unwrap();
            let probe: Vec<_> = (0..3).map(|i| ContextRow { tap_row: i, slot: 1, position: i }).collect();
            restored = MlaContextUpdate::checked(&restored, 2, &probe).unwrap().next;
            for offset in 0..3 {
                let rows = [ContextRow { tap_row: 2, slot: 0, position: frontier + offset }];
                let hit = MlaContextUpdate::checked(&restored, 2, &rows).unwrap();
                let cold = MlaContextUpdate::checked(&fresh, 2, &rows).unwrap();
                assert_eq!(hit.positions, cold.positions);
                assert_eq!(hit.slots, cold.slots);
                assert_eq!(hit.next.frontier(0).unwrap(), cold.next.frontier(0).unwrap());
                assert_eq!(hit.next.frontier(1).unwrap(), (0, 3));
                restored = hit.next;
                fresh = cold.next;
            }
        }
    }

    #[test]
    fn mla_context_preflight_preserves_holes_origins_and_all_slot_frontiers() {
        let mut state = mla_cache_state();
        state.reset(0, 4093).unwrap();
        state.reset(1, 17000).unwrap();
        let rows = (0..65).flat_map(|i| [
            ContextRow { tap_row: 3 * i, slot: 0, position: 4093 + i },
            ContextRow { tap_row: 3 * i + 2, slot: 1, position: 17000 + i },
        ]).collect::<Vec<_>>();
        let plan = MlaContextUpdate::checked(&state, 2, &rows).unwrap();
        assert_eq!(plan.first, 0);
        assert_eq!(plan.positions.len(), 195);
        assert_eq!(state.frontier(0).unwrap(), (4093, 4093));
        assert_eq!(state.frontier(1).unwrap(), (17000, 17000));
        assert_eq!(plan.next.frontier(0).unwrap(), (4093, 4158));
        assert_eq!(plan.next.frontier(1).unwrap(), (17000, 17065));
        for (layer, physical) in plan.slots.iter().enumerate() {
            let left = state.context_rows(layer, 0, 4093, 65).unwrap();
            let right = state.context_rows(layer, 1, 17000, 65).unwrap();
            for i in 0..65 {
                assert_eq!(physical[3 * i], left[i]);
                assert_eq!(physical[3 * i + 1], -1);
                assert_eq!(physical[3 * i + 2], right[i]);
            }
        }
        assert_eq!(plan.positions.chunks(64).map(|c| c.len()).collect::<Vec<_>>(), [64, 64, 64, 3]);
        let mut bad = rows.clone();
        bad.last_mut().unwrap().position += 1;
        assert!(MlaContextUpdate::checked(&state, 2, &bad).is_err());
        bad = rows.clone();
        bad.last_mut().unwrap().tap_row = 0;
        assert!(MlaContextUpdate::checked(&state, 2, &bad).is_err());
        bad = rows.clone();
        bad.last_mut().unwrap().slot = 2;
        assert!(MlaContextUpdate::checked(&state, 2, &bad).is_err());
        bad = rows.clone();
        bad.last_mut().unwrap().tap_row = TAP_ROWS;
        assert!(MlaContextUpdate::checked(&state, 2, &bad).is_err());
        assert_eq!(state.frontier(0).unwrap(), (4093, 4093));
        assert_eq!(state.frontier(1).unwrap(), (17000, 17000));
    }

    #[test]
    fn mla_draft_views_are_checked_nonowning_and_preserve_device() {
        let buffer = CuteafdDeviceBuffer { ptr: 0x1000usize as *mut c_void, bytes: 512, device_id: 1,
            ..Default::default() };
        let view = draft_view(buffer, 128, 256).unwrap();
        assert_eq!(view.ptr as usize, 0x1080);
        assert_eq!((view.bytes, view.device_id), (256, 1));
        assert!(draft_view(buffer, 256, 257).is_err());
        assert!(draft_view(buffer, usize::MAX, 2).is_err());
    }

    #[test]
    fn device_input_bridge_keeps_anchor_then_mask_row_order() {
        let seqs = [DraftSeq { slot: 1, anchor: 42, position: 7, valid_from: 0 },
            DraftSeq { slot: 0, anchor: 9, position: 13, valid_from: 0 }];
        assert_eq!(draft_input_ids(&seqs, 17, 3), [42, 17, 17, 9, 17, 17]);
        assert_eq!(draft_input_ids(&seqs, 17, 1), [42, 9]);
        assert!(draft_input_ids(&[], 17, 8).is_empty());
    }

    fn small_config() -> DflashConfig {
        DflashConfig { hidden: 128, intermediate: 128, layers: 2, heads: 1, kv_heads: 1, head_dim: 128,
            eps: 1e-6, theta: 10000., block: 8, group: 16, mask_token: 0, rank: 256,
            taps: vec![0], vocab: 16, window: RING, row_window: false }
    }

    fn complete_headers(cfg: &DflashConfig) -> (HashMap<String, SafetensorsTensorMetadata>, u64) {
        let mut end = 0u64;
        let tensors = cfg.tensor_shapes().into_iter().map(|(name, shape)| {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            let tensor = SafetensorsTensorMetadata { name: name.clone(), dtype: DType::Bf16,
                shape, byte_offset: end, byte_length: bytes };
            end += bytes;
            (name, tensor)
        }).collect();
        (tensors, end)
    }

    #[test]
    fn all_checkpoint_headers_are_validated_without_device_allocation() {
        let cfg = small_config();
        let (mut tensors, file_bytes) = complete_headers(&cfg);
        check_checkpoint_headers(&cfg, &tensors, file_bytes).unwrap();
        let name = "layers.1.self_attn.o_proj.weight";
        tensors.get_mut(name).unwrap().dtype = DType::F8E4M3;
        let error = check_checkpoint_headers(&cfg, &tensors, file_bytes).unwrap_err().to_string();
        assert!(error.contains(name) && error.contains("source-native FP8"));
    }

    #[test]
    fn later_checkpoint_shape_missing_and_truncated_inputs_are_named() {
        let cfg = small_config();
        let (mut tensors, file_bytes) = complete_headers(&cfg);
        let name = "layers.1.mlp.down_proj.weight";
        tensors.get_mut(name).unwrap().shape = vec![64, 256];
        let error = check_checkpoint_headers(&cfg, &tensors, file_bytes).unwrap_err().to_string();
        assert!(error.contains(name) && error.contains("shape"));
        let (mut tensors, file_bytes) = complete_headers(&cfg);
        tensors.remove(name);
        assert!(check_checkpoint_headers(&cfg, &tensors, file_bytes).unwrap_err().to_string().contains(name));
        let (tensors, file_bytes) = complete_headers(&cfg);
        assert!(check_checkpoint_headers(&cfg, &tensors, file_bytes - 1).unwrap_err().to_string().contains("past"));
    }

    #[test]
    fn shared_target_head_preserves_source_bf16_and_rejects_dual_or_compact_input() {
        let mut head = SafetensorsTensorMetadata { name: "lm_head.weight".into(), dtype: DType::Bf16,
            shape: vec![16, 128], byte_offset: 0, byte_length: 4096 };
        check_target_bf16_head(&head, 128, 16, false).unwrap();
        let error = check_target_bf16_head(&head, 128, 16, true).unwrap_err().to_string();
        assert!(error.contains("lm_head.weight") && error.contains("--fp8-head") && error.contains("unsupported"));
        head.dtype = DType::F8E4M3;
        head.byte_length = 2048;
        let error = check_target_bf16_head(&head, 128, 16, false).unwrap_err().to_string();
        assert!(error.contains("source-native compact") && error.contains("lm_head.weight"));
        head.dtype = DType::Bf16;
        head.byte_length = 4096;
        head.shape = vec![128, 16];
        assert!(check_target_bf16_head(&head, 128, 16, false).is_err());
    }
}

#[cfg(test)]
mod replay_mode_tests {
    use super::*;

    struct Probe {
        immutable: Option<GlmDraftRepresentation>,
        setters: RefCell<Vec<bool>>,
    }

    impl ReplayDrafter for Probe {
        fn block(&self) -> usize { 8 }
        fn sequences(&self) -> usize { 16 }
        fn has_fp8(&self) -> bool { self.immutable.is_none() }
        fn set_fp8(&self, on: bool) { self.setters.borrow_mut().push(on); }
        fn resident_modes(&self) -> Vec<ReplayMode> {
            match self.immutable {
                Some(mode) => vec![ReplayMode { name: mode.name(), legacy_fp8: None }],
                None => vec![ReplayMode { name: "BF16", legacy_fp8: Some(false) },
                    ReplayMode { name: "FP8", legacy_fp8: Some(true) }],
            }
        }
        fn context(&self, _: &[u8], _: usize) -> Result<()> { unreachable!() }
        fn draft_tokens(&self, _: &[(usize, u32, usize)], _: &[u8], _: *const c_void)
            -> Result<Vec<Vec<u32>>> { unreachable!() }
        fn tap_rows(&self) -> usize { TAP_ROWS }
        fn last_hidden(&self, _: usize) -> Result<Vec<u8>> { unreachable!() }
    }

    struct LegacyProbe(RefCell<Vec<bool>>);
    impl ReplayDrafter for LegacyProbe {
        fn block(&self) -> usize { 8 }
        fn sequences(&self) -> usize { 16 }
        fn has_fp8(&self) -> bool { true }
        fn set_fp8(&self, on: bool) { self.0.borrow_mut().push(on); }
        // resident_modes deliberately uses the unchanged default trait seam.
        fn context(&self, _: &[u8], _: usize) -> Result<()> { unreachable!() }
        fn draft_tokens(&self, _: &[(usize, u32, usize)], _: &[u8], _: *const c_void)
            -> Result<Vec<Vec<u32>>> { unreachable!() }
        fn tap_rows(&self) -> usize { TAP_ROWS }
        fn last_hidden(&self, _: usize) -> Result<Vec<u8>> { unreachable!() }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ReplayEvent {
        Mode(bool),
        Taps(usize, usize),
        Context(usize, Vec<u8>),
        Embed(Vec<u32>),
        Draft(Vec<(usize, u32, usize, usize)>, Vec<u8>),
        Hidden(usize),
    }

    struct ReplayProbe {
        legacy: bool,
        forbid_raw_head: bool,
        events: RefCell<Vec<ReplayEvent>>,
    }

    impl ReplayProbe {
        fn record_draft(&self, seqs: &[DraftSeq], rows: &[u8]) -> Result<Vec<Vec<u32>>> {
            self.events.borrow_mut().push(ReplayEvent::Draft(seqs.iter()
                .map(|s| (s.slot, s.anchor, s.position, s.valid_from)).collect(), rows.to_vec()));
            Ok(vec![vec![0; 3]; seqs.len()])
        }
    }

    impl ReplayDrafter for ReplayProbe {
        fn block(&self) -> usize { 4 }
        fn sequences(&self) -> usize { 4 }
        fn has_fp8(&self) -> bool { self.legacy }
        fn set_fp8(&self, on: bool) { self.events.borrow_mut().push(ReplayEvent::Mode(on)); }
        fn resident_modes(&self) -> Vec<ReplayMode> {
            if self.legacy {
                vec![ReplayMode { name: "BF16", legacy_fp8: Some(false) },
                    ReplayMode { name: "FP8", legacy_fp8: Some(true) }]
            } else {
                vec![ReplayMode { name: "FP8", legacy_fp8: None }]
            }
        }
        fn context(&self, taps: &[u8], first: usize) -> Result<()> {
            self.events.borrow_mut().push(ReplayEvent::Context(first, taps.to_vec()));
            Ok(())
        }
        fn draft_tokens(&self, seqs: &[(usize, u32, usize)], rows: &[u8], head: *const c_void)
            -> Result<Vec<Vec<u32>>> {
            assert!(!self.forbid_raw_head, "callback replay must not invoke the raw-head path");
            assert!(head.is_null());
            self.record_draft(&seqs.iter().map(|&(slot, anchor, position)|
                DraftSeq { slot, anchor, position, valid_from: 0 }).collect::<Vec<_>>(), rows)
        }
        fn tap_rows(&self) -> usize { 2 }
        fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
            self.events.borrow_mut().push(ReplayEvent::Hidden(sequences));
            Ok(vec![0, 0])
        }
    }

    fn expected_replay_events(legacy: bool) -> Vec<ReplayEvent> {
        let mut events = Vec::new();
        for mode in if legacy { vec![Some(false), Some(true)] } else { vec![None] } {
            if let Some(on) = mode { events.push(ReplayEvent::Mode(on)); }
            let mut done = 0;
            for position in 3..8 {
                while done < position {
                    let n = (position - done).min(2);
                    events.push(ReplayEvent::Taps(done, n));
                    events.push(ReplayEvent::Context(done, (done..done + n).map(|r| r as u8).collect()));
                    done += n;
                }
                events.push(ReplayEvent::Embed(vec![position as u32]));
                events.push(ReplayEvent::Draft(vec![(0, position as u32, position, 0)], vec![position as u8]));
                events.push(ReplayEvent::Hidden(1));
            }
            for count in [1, 2, 4] {
                events.push(ReplayEvent::Embed(vec![7; count]));
                for _ in 0..9 {
                    events.push(ReplayEvent::Draft((0..count).map(|slot| (slot, 7, 7, 0)).collect(), vec![7; count]));
                }
            }
        }
        if legacy { events.push(ReplayEvent::Mode(true)); }
        events
    }

    #[test]
    fn callback_replay_preserves_single_wide_context_order_and_valid_from() {
        let tokens: Vec<u32> = (0..12).collect();
        for legacy in [false, true] {
            for callback in [false, true] {
                let p = ReplayProbe { legacy, forbid_raw_head: callback, events: RefCell::new(Vec::new()) };
                let taps = |first: usize, n: usize| {
                    p.events.borrow_mut().push(ReplayEvent::Taps(first, n));
                    Ok((first..first + n).map(|r| r as u8).collect())
                };
                let embed = |ids: &[u32]| {
                    p.events.borrow_mut().push(ReplayEvent::Embed(ids.to_vec()));
                    Ok(ids.iter().map(|&id| id as u8).collect())
                };
                if callback {
                    replay_with(&p, &tokens, &[0; 12], &taps, &embed,
                        &|seqs, rows| p.record_draft(seqs, rows), 3).unwrap();
                } else {
                    replay(&p, &tokens, &[0; 12], &taps, &embed, std::ptr::null(), 3).unwrap();
                }
                assert_eq!(*p.events.borrow(), expected_replay_events(legacy));
            }
        }
    }

    #[test]
    fn callback_replay_stops_at_draft_error_without_hidden_or_wide_reads() {
        let p = ReplayProbe { legacy: false, forbid_raw_head: true, events: RefCell::new(Vec::new()) };
        let error = replay_with(&p, &[0; 12], &[0; 12], &|_, n| Ok(vec![0; n]), &|_| Ok(vec![0]),
            &|seqs, rows| {
                p.record_draft(seqs, rows)?;
                anyhow::bail!("injected remote draft failure")
            }, 3).unwrap_err();
        assert_eq!(error.to_string(), "injected remote draft failure");
        assert_eq!(*p.events.borrow(), [ReplayEvent::Context(0, vec![0; 2]),
            ReplayEvent::Context(2, vec![0]), ReplayEvent::Draft(vec![(0, 0, 3, 0)], vec![0])]);
    }

    #[test]
    fn immutable_fp8_replay_reports_actual_mode_and_never_calls_a_setter() {
        let p = Probe { immutable: Some(GlmDraftRepresentation::Fp8Only), setters: RefCell::new(Vec::new()) };
        let modes = p.resident_modes();
        assert_eq!(modes, [ReplayMode { name: "FP8", legacy_fp8: None }]);
        for mode in modes { activate_replay_mode(&p, mode); }
        assert!(p.setters.borrow().is_empty());
    }

    #[test]
    fn legacy_replay_modes_keep_existing_arithmetic_selection() {
        let p = LegacyProbe(RefCell::new(Vec::new()));
        let modes = p.resident_modes();
        assert_eq!(modes.iter().map(|m| m.name).collect::<Vec<_>>(), ["BF16", "FP8"]);
        for mode in modes { activate_replay_mode(&p, mode); }
        assert_eq!(*p.0.borrow(), [false, true]);
    }
}
