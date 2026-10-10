//! DFlash2 block drafter for GLM 5.3 (incoai/GLM-5.3-DFlash2: six layers,
//! 64 query heads) and GLM 5.3 Flash (incoai/GLM-5.3-Flash-DFlash2: five
//! layers, 32 query heads over 8 KV heads).
//!
//! Every target step taps the outputs of the `target_layer_ids` layers
//! (GLM 5.3: 5, 19, 33, 47, 61, 75; Flash: 5, 14, 24, 33, 42, where the tap is
//! the mean of the four mHC streams, [`GlmDrafter::tap_streams`]) into
//! [`GlmDrafter::taps`]. Once a
//! step's rows are committed, [`GlmDrafter::update`] projects their taps
//! (`hidden_norm(fc(taps))`) into every draft layer's context K/V, a ring of
//! the sequence's last 2048 positions. [`GlmDrafter::draft`] then runs the
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

/// Context entries each draft layer attends to (the checkpoint's sliding window).
pub(crate) const RING: usize = 2048;
/// Tapped rows one target step keeps (a prefill chunk's tail, or a verify step).
pub(crate) const TAP_ROWS: usize = RING;
/// Historical skinny-row limit, kept for the older MiMo compatibility path.
/// Immutable GLM drafters use the arbitrary-row native W8A16 entrypoint.
pub(crate) const FP8_ROWS: usize = 128;

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
    rope_parameters: RawRope,
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
        ensure!(raw.head_dim == 128 && raw.num_key_value_heads > 0
            && raw.num_attention_heads % raw.num_key_value_heads == 0
            && raw.num_attention_heads / raw.num_key_value_heads * d.block_size <= 64,
            "DFlash2 kernels take 128-wide heads and at most 64 queries per kv head and block \
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
            theta: raw.rope_parameters.rope_theta as f32,
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
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: DraftWeight<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
    o: DraftWeight<'a>,
    gate_up: DraftWeight<'a>,
    down: DraftWeight<'a>,
    k_ring: Dev<'a>,
    v_ring: Dev<'a>,
}

/// Buffers of one draft step for up to `sequences` sequences.
struct Workspace<'a> {
    sequences: usize,
    h: Dev<'a>,
    n: Dev<'a>,
    conv: Dev<'a>,
    dynamic: Dev<'a>,
    qkv: Dev<'a>,
    q: Dev<'a>,
    k: Dev<'a>,
    v: Dev<'a>,
    attn: Dev<'a>,
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
    positions: Dev<'a>,
    tables: Dev<'a>,
    attention_workspace: Dev<'a>,
    topk_workspace: Dev<'a>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
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

/// The target's vocabulary head a draft step borrows: its only resident copy,
/// never duplicated or repacked by the drafter.
pub(crate) enum TargetHead<'h> {
    /// The target's checkpoint BF16 `[vocab, hidden]`.
    Bf16(&'h Dev<'h>),
    /// A target-owned launcher over another representation (GLM 5.3 Flash's
    /// FP8-only head in 16-row spans); the drafter's head workspace is unused.
    Launch(HeadLaunch<'h>),
}

/// A resolved [`TargetHead`] inside a draft step.
enum HeadCall<'h> {
    Bf16(*const c_void),
    Launch(&'h HeadLaunch<'h>),
}

/// Where a draft step's input rows come from.
#[derive(Clone, Copy)]
enum DraftInput<'r, 'e> {
    /// The anchors' embedding rows (host BF16); the mask rows are the drafter's own copy.
    Rows(&'r [u8]),
    /// Gathered on the device from the target's embedding table by token id.
    Table(&'r TokenEmbedding<'e>),
}

pub(crate) struct GlmDrafter<'a> {
    library: &'a NativeLibrary,
    pub cfg: DflashConfig,
    stream: *mut c_void,
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
    context_kv: Dev<'a>,
    context_positions: Dev<'a>,
    context_slots: Dev<'a>,
    /// The mask token's embedding row.
    mask_row: Vec<u8>,
    workspace: RefCell<Option<Workspace<'a>>>,
    representation: GlmDraftRepresentation,
    fp8_workspace: Option<Dev<'a>>,
    /// How the borrowed BF16 head runs past one draft block ([`super::DraftHead`]; GLM 5.3
    /// Flash's --draft-head). [`super::DraftHead::Exact`] unless the target sets it.
    head_mode: Cell<super::DraftHead>,
    /// How the FP8 GEMMs run (GLM 5.3 Flash's --draft-linear), and the latest mode the FP8
    /// scratch was admitted for.
    fp8_rows: Cell<fp8_linear::Fp8Rows>,
    fp8_admitted: fp8_linear::Fp8Rows,
    confidence_directory: std::path::PathBuf,
    confidence_scales: fp8_linear::Fp8Scales,
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
    std::thread::spawn(move || std::fs::read(path))
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

/// The only shared head consumer in this tranche reads checkpoint BF16.
/// Reject compact input/dual-copy target modes before native allocation.
pub(crate) fn check_snapshot_target_bf16_head(snapshot: &Path, hidden: usize, vocab: usize,
    explicit_fp8_head: bool) -> Result<()> {
    // Routed-expert catalogs need not contain coordinator tensors. Resolve the
    // head through its own index/header contract without reading weight data.
    let index_path = snapshot.join("model.safetensors.index.json");
    let shard = if index_path.is_file() {
        let index = cuteafd_loader::plan::checkpoint::read_json(&index_path)?;
        index.get("weight_map").and_then(|map| map.get("lm_head.weight"))
            .and_then(serde_json::Value::as_str)
            .context("DFlash target index has no lm_head.weight shard")?.to_owned()
    } else { "model.safetensors".to_owned() };
    let path = snapshot.join(&shard);
    let head = read_safetensors_metadata(&path)?.into_iter()
        .find(|t| t.name == "lm_head.weight")
        .with_context(|| format!("DFlash target lm_head.weight missing from indexed shard {shard}"))?;
    check_target_bf16_head(&head, hidden, vocab, explicit_fp8_head)
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

impl<'a> GlmDrafter<'a> {
    pub fn confidence_policy(&self, family: &str, fp8_head: bool) -> Result<crate::shared::draft_confidence::ConfidencePolicy> {
        crate::shared::draft_confidence::ConfidencePolicy::load(&self.confidence_directory,
            self.confidence_key(family, fp8_head))
    }

    /// The keyed selector prior (`family/dflash2/numerics`), or the generic
    /// GLM-5.3 fit when the drafter ships none for this key.
    pub fn selector_fit(&self, family: &str, fp8_head: bool) -> Result<crate::shared::draft_confidence::SelectorFit> {
        crate::shared::draft_confidence::SelectorFit::load(&self.confidence_directory,
            &self.confidence_key(family, fp8_head))
    }

    fn confidence_key(&self, family: &str, fp8_head: bool) -> String {
        let head = if fp8_head { "head-fp8-row".into() } else { format!("head-bf16-{:?}", self.head_mode.get()) };
        let numerics = format!("{}-{:?}-{head}-{:?}-r1", self.representation.name(), self.fp8_rows.get(),
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
        scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let cfg = DflashConfig { row_window, ..DflashConfig::read(snapshot)? };
        let capacity = GlmDraftCapacity::new(slots, max_sequences, cfg.block)?;
        let layout = cfg.runtime_layout(representation, capacity)?;
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
        let matrix = |source: Dev<'a>, n: usize, k: usize| -> Result<DraftWeight<'a>> {
            match representation {
                GlmDraftRepresentation::Bf16Only => Ok(DraftWeight::Bf16(source)),
                GlmDraftRepresentation::Fp8Only => {
                    let packed = Fp8Weight::pack(library, source.buffer.ptr, n, k, scales, stream);
                    // SAFETY: source and any packed result are still live.
                    // Drain packing even after a launch error before Drop.
                    if let Err(error) = unsafe { library.cuda_stream_synchronize(stream) } {
                        // These quarantines retain allocations only. Complete
                        // family terminal teardown must also retain the native
                        // library/module owner until queued work has retired.
                        std::mem::forget(source);
                        if let Ok(weight) = packed { weight.quarantine(); }
                        return Err(error.context("DFlash packing owners quarantined after failed drain"));
                    }
                    let packed = packed?;
                    drop(source);
                    Ok(DraftWeight::Fp8(packed))
                }
            }
        };
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
                qkv: matrix(concat(&[(&format!("{a}.q_proj.weight"), cfg.heads * cfg.head_dim),
                    (&format!("{a}.k_proj.weight"), kv), (&format!("{a}.v_proj.weight"), kv)], h)?, cfg.qkv_width(), h)?,
                q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
                k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
                o: matrix(tensor(&format!("{a}.o_proj.weight"), &[h, cfg.heads * cfg.head_dim])?,
                    h, cfg.heads * cfg.head_dim)?,
                gate_up: matrix(concat(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                    (&format!("{p}.mlp.up_proj.weight"), inter)], h)?, 2 * inter, h)?,
                down: matrix(tensor(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?, h, inter)?,
                k_ring: zeroed(slots * RING * kv * 2)?,
                v_ring: zeroed(slots * RING * kv * 2)?,
            })
        }).collect::<Result<Vec<_>>>()?;
        let taps = cfg.taps.len() * h;
        tracing::info!(?representation, context_slots = slots, max_batch_sequences = max_sequences,
            own_weight_bytes = layout.weights.resident_bytes()?, max_load_staging = layout.weights.max_load_staging,
            "DFlash single-copy storage admitted; vocabulary head borrowed from target");
        Ok(Self {
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
            context_kv: zeroed(TAP_ROWS * 2 * kv * 2)?,
            context_positions: zeroed(TAP_ROWS * 8)?,
            context_slots: zeroed(TAP_ROWS * 4)?,
            mask_row,
            workspace: RefCell::new(None),
            representation,
            confidence_directory: snapshot.into(),
            confidence_scales: scales,
            fp8_workspace,
            head_mode: Cell::new(super::DraftHead::Exact),
            fp8_rows: Cell::new(fp8_rows),
            fp8_admitted: fp8_rows,
            cfg,
        })
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
    pub fn set_draft_head(&self, mode: super::DraftHead) {
        self.head_mode.set(mode);
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

    /// Writes the context K/V of committed tapped rows (in tap-row order).
    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
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
        self.put(&self.context_slots, bytes_of(&slots))?;
        let (h, kv) = (self.cfg.hidden, self.cfg.kv_width());
        let width = self.cfg.taps.len() * h;
        let s = self.stream;
        // SAFETY: every buffer holds TAP_ROWS rows of its width; the stream orders the chain.
        unsafe {
            self.linear(at(&self.taps, first * width * 2), &self.fc, 0,
                self.fused.buffer.ptr, n, width, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, self.cfg.eps, s)?;
            for layer in &self.layers {
                let q_rows = self.cfg.heads * self.cfg.head_dim;
                self.linear(self.fused_norm.buffer.ptr, &layer.qkv, q_rows,
                    self.context_kv.buffer.ptr, n, h, 2 * kv)?;
                self.library.glm_dflash_qk_rope(self.context_kv.buffer.ptr, layer.q_norm.buffer.ptr,
                    layer.k_norm.buffer.ptr, self.context_positions.buffer.ptr, self.context_slots.buffer.ptr,
                    std::ptr::null_mut(), layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, n, 0, self.cfg.kv_heads,
                    self.cfg.theta, self.cfg.eps, s)?;
            }
        }
        Ok(())
    }

    fn put(&self, dev: &Dev<'_>, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        self.library.copy_h2d(CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    /// Explicit serving startup admission; other callers retain lazy allocation.
    pub(crate) fn prepare_workspace(&self) -> Result<()> {
        if self.workspace.borrow().is_none() {
            *self.workspace.borrow_mut() = Some(self.workspace(self.max_sequences)?);
        }
        Ok(())
    }

    fn workspace(&self, sequences: usize) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter/workspace");
        let c = &self.cfg;
        let rows = sequences * c.block;
        let drafted = sequences * c.drafts();
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256));
        let head_workspace = alloc(VOCABULARY_HEAD_WORKSPACE)?;
        Ok(Workspace {
            sequences,
            h: alloc(rows * c.hidden * 2)?,
            n: alloc(rows * c.hidden * 2)?,
            conv: alloc(rows * c.hidden * 2)?,
            dynamic: alloc(rows * c.conv_width() * 2)?,
            qkv: alloc(rows * c.qkv_width() * 2)?,
            q: alloc(rows * c.heads * c.head_dim * 2)?,
            k: alloc(rows * c.kv_width() * 2)?,
            v: alloc(rows * c.kv_width() * 2)?,
            attn: alloc(rows * c.heads * c.head_dim * 2)?,
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
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            attention_workspace: alloc(self.library.glm_dflash_attention_workspace(sequences, c.kv_heads,
                RING + c.block)?)?,
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
        self.draft_from(sequences, DraftInput::Rows(anchor_rows), self.head_call(&head)?)
    }

    /// [`Self::draft`] with the block's input rows gathered from the target's
    /// embedding table by token id: each anchor, then the mask token (the
    /// drafter's mask row is that table row).
    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_call(&head)?)
    }

    fn head_call<'h>(&self, head: &'h TargetHead<'h>) -> Result<HeadCall<'h>> {
        Ok(match head {
            TargetHead::Bf16(owner) => HeadCall::Bf16(self.borrowed_head(owner)?),
            TargetHead::Launch(launch) => HeadCall::Launch(launch),
        })
    }

    fn borrowed_head(&self, owner: &Dev<'_>) -> Result<*const c_void> {
        let bytes = self.cfg.vocab.checked_mul(self.cfg.hidden).and_then(|n| n.checked_mul(2))
            .context("DFlash borrowed BF16 head byte count overflow")?;
        ensure!(!owner.buffer.ptr.is_null() && owner.buffer.bytes == bytes,
            "DFlash requires the target-owned BF16 head [{}, {}] ({bytes} bytes); compact target-head \
             consumer is unsupported, no private head copy will be allocated", self.cfg.vocab, self.cfg.hidden);
        Ok(owner.buffer.ptr)
    }

    fn draft_from(&self, sequences: &[DraftSeq], input: DraftInput<'_, '_>, head: HeadCall<'_>) -> Result<Vec<Draft>> {
        ensure!(!matches!(head, HeadCall::Bf16(p) if p.is_null()), "DFlash has no borrowed target head");
        let c = &self.cfg;
        let (s_count, block, h) = (sequences.len(), c.block, c.hidden);
        let rows_ok = match input {
            DraftInput::Rows(r) => r.len() == s_count * h * 2,
            DraftInput::Table(_) => true,
        };
        ensure!(s_count > 0 && s_count <= self.max_sequences && rows_ok, "draft step of {s_count} sequences");
        let rows = s_count * block;
        let mut slot = self.workspace.borrow_mut();
        if slot.as_ref().is_none_or(|w| w.sequences < s_count) {
            *slot = None;
            *slot = Some(self.workspace(self.max_sequences)?);
        }
        let w = slot.as_ref().context("draft workspace")?;
        let mut positions = Vec::with_capacity(rows);
        let mut tables = vec![0i32; 3 * s_count];
        for (i, seq) in sequences.iter().enumerate() {
            ensure!(seq.slot < self.slots, "ring slot {} of {}", seq.slot, self.slots);
            positions.extend((seq.position..seq.position + block).map(|p| p as i64));
            tables[i] = seq.slot as i32;
            tables[s_count + i] = seq.position.saturating_sub(seq.valid_from).min(RING) as i32;
            tables[2 * s_count + i] = seq.position as i32;
        }
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
                let ids: Vec<u32> = sequences.iter()
                    .flat_map(|seq| std::iter::once(seq.anchor).chain(std::iter::repeat_n(c.mask_token, block - 1)))
                    .collect();
                embedding.embed(&ids, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
        }
        self.put(&w.positions, bytes_of(&positions))?;
        self.put(&w.tables, bytes_of(&tables))?;
        let anchors: Vec<u32> = sequences.iter().map(|s| s.anchor).collect();
        self.put(&w.anchors, bytes_of(&anchors))?;
        let (s, l) = (self.stream, self.library);
        let (eps, group, inter) = (c.eps, c.group, c.intermediate);
        let attention_width = c.heads * c.head_dim;
        // SAFETY: every workspace buffer holds `rows` rows of its width and the
        // weights their checkpoint shapes; the stream orders the chain.
        unsafe {
            l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                self.linear(w.n.buffer.ptr, &layer.attn_conv, 0, w.dynamic.buffer.ptr,
                    rows, h, c.conv_width())?;
                l.glm_dflash_conv(w.n.buffer.ptr, w.dynamic.buffer.ptr, layer.attn_base.buffer.ptr, w.conv.buffer.ptr,
                    rows, block, h, group, s)?;
                self.linear(w.conv.buffer.ptr, &layer.qkv, 0, w.qkv.buffer.ptr, rows, h,
                    c.qkv_width())?;
                l.glm_dflash_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                    w.positions.buffer.ptr, std::ptr::null(), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, rows,
                    c.heads, c.kv_heads, c.theta, eps, s)?;
                l.glm_dflash_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                    layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4),
                    w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads, c.kv_heads, RING,
                    RING + block, if c.row_window { c.window } else { 0 }, 1.0 / (c.head_dim as f32).sqrt(), s)?;
                self.linear(w.attn.buffer.ptr, &layer.o, 0, w.delta.buffer.ptr, rows,
                    attention_width, h)?;
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
            }
            match &head {
                HeadCall::Bf16(weight) => super::launch_draft_head(l, &w.head, w.n.buffer.ptr, *weight,
                    w.logits.buffer.ptr.cast(), rows, h, c.vocab, s, self.head_mode.get())?,
                HeadCall::Launch(launch) => launch(w.n.buffer.ptr.cast_const(), w.logits.buffer.ptr.cast(), rows, s)?,
            }
            l.glm_dflash_topk(w.logits.buffer.ptr, w.unary.buffer.ptr, w.candidates.buffer.ptr,
                w.topk_workspace.buffer.ptr, s_count, block, c.drafts(), c.vocab, s)?;
            self.linear(w.n.buffer.ptr, &self.projection, 0,
                w.projected.buffer.ptr, rows, h, c.rank)?;
            l.glm_dflash_select(self.predecessor.buffer.ptr, self.successor.buffer.ptr, w.projected.buffer.ptr,
                w.candidates.buffer.ptr, w.unary.buffer.ptr, w.anchors.buffer.ptr, w.tokens.buffer.ptr,
                w.features.buffer.ptr, s_count, block, c.drafts(), c.rank, s)?;
            l.cuda_stream_synchronize(s)?;
        }
        let drafted = s_count * c.drafts();
        let mut tokens = vec![0u8; drafted * 4];
        let mut features = vec![0u8; drafted * 16];
        l.copy_d2h(&mut tokens, CuteafdDeviceBuffer { bytes: drafted * 4, ..w.tokens.buffer })?;
        l.copy_d2h(&mut features, CuteafdDeviceBuffer { bytes: drafted * 16, ..w.features.buffer })?;
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

impl ReplayDrafter for GlmDrafter<'_> {
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
        Ok(self.draft_from(&seqs, DraftInput::Rows(anchor_rows), HeadCall::Bf16(head))?.into_iter()
            .map(|d| d.tokens).collect())
    }

    fn tap_rows(&self) -> usize {
        TAP_ROWS
    }

    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        GlmDrafter::last_hidden(self, sequences)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replay(drafter: &(impl ReplayDrafter + ?Sized), tokens: &[u32], greedy: &[u32],
    taps: &dyn Fn(usize, usize) -> Result<Vec<u8>>, embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>, head: *const c_void,
    start: usize) -> Result<()> {
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
            let draft = drafter.draft_tokens(&[(0, tokens[p], p)], &rows, head)?.remove(0);
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
            let seqs: Vec<(usize, u32, usize)> = (0..count).map(|slot| (slot, tokens[p], p)).collect();
            let rows = embed(&vec![tokens[p]; count])?;
            let mut times = Vec::new();
            for run in 0..9 {
                let timer = std::time::Instant::now();
                drafter.draft_tokens(&seqs, &rows, head)?;
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
    fn target_head_guard_reads_checkpoint_when_actual_expert_catalog_has_no_head() {
        let fixture = expert_only_snapshot();
        let catalog = cuteafd_loader::read_expert_catalog(&fixture.0).unwrap();
        assert!(catalog.fp8().is_some());
        assert!(catalog.tensors().is_empty());
        assert!(catalog.tensor("lm_head.weight").is_err());
        check_snapshot_target_bf16_head(&fixture.0, 128, 16, false).unwrap();
        assert!(check_snapshot_target_bf16_head(&fixture.0, 128, 16, true).is_err());
        assert!(check_snapshot_target_bf16_head(&fixture.0, 128, 32, false).is_err());
        let index_path = fixture.0.join("model.safetensors.index.json");
        let mut index = cuteafd_loader::plan::checkpoint::read_json(&index_path).unwrap();
        index["weight_map"]["lm_head.weight"] = "missing.safetensors".into();
        std::fs::write(index_path, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(check_snapshot_target_bf16_head(&fixture.0, 128, 16, false).is_err());
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
