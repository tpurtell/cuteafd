//! dSpark block drafter for GLM 5.3 Flash
//! (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview: Speculators' DSparkDraftModel,
//! five Qwen3 layers of 64 heads x 64, a vanilla rank-256 Markov head and a
//! confidence head).
//!
//! Every target step taps the four-stream mean of the outputs of layers
//! `aux_hidden_state_layer_ids - 1` (vLLM's aux ids name a layer's input:
//! 20, 28, 32, 36, 40, 44 -> outputs of 19, 27, 31, 35, 39, 43) into
//! [`DsparkDrafter::taps`]; [`DsparkDrafter::update`] projects committed rows
//! (`hidden_norm(fc(taps))`) into every draft layer's context K/V, a ring of
//! the sequence's last 2048 positions (the checkpoint's sliding window: every
//! block row sees the same 2048 entries, as in training). A draft step runs
//! the layers over `[anchor, mask x 7]` at the anchor's position, causal
//! inside the block (sliding layers with `sliding_window_non_causal` off),
//! then the target's LM head; block row k predicts token `position + k + 1`
//! (`sample_from_anchor`): the Markov head adds `W2 . W1[previous token]` to
//! each row's logits and the drafts are taken greedily left to right, all on
//! the device. The confidence head predicts each row's acceptance
//! (`sigmoid(w . [h_k, W1[previous]] + b)`) for the draft policy.
//!
//! The embedding and LM head are the target's (the checkpoint's copies are
//! the official GLM 5.3 Flash tensors bit for bit, as in every GLM 5.3 Flash
//! quant), borrowed as the target keeps them ([`TargetHead`]: BF16, or its
//! FP8-only head). The GEMM weights are resident once, BF16 or (default) E4M3
//! packed at load, like the DFlash2 drafter ([`crate::families::glm5::dflash`]).
//! Drafts only steer speculation: the
//! verify step keeps output identical to plain greedy decoding.
//! python/reference/families/glm5_flash/dspark/reference.py is the oracle.
use crate::families::glm5::dflash::{ContextRow, Draft, DraftSeq, HeadLaunch, ReplayDrafter, ReplayMode, TargetHead, RING,
    TAP_ROWS};
use crate::families::glm5::DraftHead;
use cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation;
use crate::shared::fp8_linear::{self, Fp8Weight};
use crate::shared::memory::DeviceAllocation;
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::read_safetensors_metadata;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

type Dev<'a> = DeviceAllocation<'a>;

/// Most sequences one draft step takes (the Markov kernel's limit).
pub(crate) const MAX_SEQUENCES: usize = 32;

#[derive(Debug, Clone, serde::Deserialize)]
struct RawLayers {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    sliding_window: Option<usize>,
    #[serde(default)]
    layer_types: Vec<String>,
    vocab_size: usize,
    hidden_act: String,
    rope_parameters: RawRope,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawRope {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawConfig {
    speculators_model_type: String,
    transformer_layer_config: RawLayers,
    aux_hidden_state_layer_ids: Vec<usize>,
    block_size: usize,
    mask_token_id: u32,
    draft_vocab_size: usize,
    markov_rank: usize,
    markov_head_type: String,
    sample_from_anchor: bool,
    enable_confidence_head: bool,
    confidence_head_with_markov: bool,
    #[serde(default)]
    sliding_window_non_causal: bool,
}

/// Whether `snapshot` holds a Speculators dSpark checkpoint.
pub(crate) fn is_dspark(snapshot: &Path) -> Result<bool> {
    let path = snapshot.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)
        .with_context(|| format!("reading drafter {}", path.display()))?)
        .with_context(|| format!("parsing drafter {}", path.display()))?;
    Ok(config["speculators_model_type"] == "dspark")
}

/// The drafter geometry the kernels are written for, read from `config.json`.
#[derive(Debug, Clone)]
pub(crate) struct DsparkConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
    /// Query rows per draft and drafted tokens (sample_from_anchor).
    pub block: usize,
    pub mask_token: u32,
    pub rank: usize,
    /// Target layers whose outputs are tapped (aux ids - 1).
    pub taps: Vec<usize>,
    pub vocab: usize,
    /// Causal attention inside the block.
    pub causal: bool,
}

impl DsparkConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)
            .context("parsing the dSpark config.json")?;
        let t = &raw.transformer_layer_config;
        ensure!(raw.speculators_model_type == "dspark", "not a dSpark checkpoint");
        ensure!(t.head_dim == 64 && t.num_key_value_heads > 0 && t.num_attention_heads % t.num_key_value_heads == 0
            && t.num_attention_heads / t.num_key_value_heads * raw.block_size <= 32,
            "dSpark kernels take 64-wide heads and at most 32 queries per kv head and block (got {} heads, {} kv, \
             block {})", t.num_attention_heads, t.num_key_value_heads, raw.block_size);
        ensure!(raw.markov_head_type == "vanilla" && raw.markov_rank == 256 && raw.sample_from_anchor
            && raw.enable_confidence_head && raw.confidence_head_with_markov,
            "dSpark kernels take a vanilla rank-256 Markov head, sample_from_anchor and a Markov confidence head");
        ensure!(raw.draft_vocab_size == t.vocab_size, "reduced dSpark draft vocabularies (d2t maps) are not supported");
        ensure!(t.hidden_act == "silu" && t.rope_parameters.rope_type == "default", "dSpark layers: SiLU, default RoPE");
        let sliding = !t.layer_types.is_empty() && t.layer_types.iter().all(|k| k == "sliding_attention");
        ensure!(sliding && t.sliding_window == Some(RING),
            "dSpark layers must all slide over a {RING}-entry window (the ring), got {:?} / {:?}", t.layer_types,
            t.sliding_window);
        ensure!(raw.aux_hidden_state_layer_ids.iter().all(|&l| l >= 1), "aux layer id 0 (the embedding) is not tapped");
        Ok(Self {
            hidden: t.hidden_size,
            intermediate: t.intermediate_size,
            layers: t.num_hidden_layers,
            heads: t.num_attention_heads,
            kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            eps: t.rms_norm_eps as f32,
            theta: t.rope_parameters.rope_theta as f32,
            block: raw.block_size,
            mask_token: raw.mask_token_id,
            rank: raw.markov_rank,
            taps: raw.aux_hidden_state_layer_ids.iter().map(|&l| l - 1).collect(),
            vocab: t.vocab_size,
            causal: !raw.sliding_window_non_causal,
        })
    }

    /// Drafted tokens per draft step (every block row predicts one).
    pub fn drafts(&self) -> usize {
        self.block
    }

    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }

    fn qkv_width(&self) -> usize {
        self.q_width() + 2 * self.kv_width()
    }
}

/// Exactly one resident representation of a GEMM weight.
enum Weight<'a> {
    Bf16(Dev<'a>),
    Fp8(Fp8Weight<'a>),
}

struct DraftLayer<'a> {
    input_norm: Dev<'a>,
    post_norm: Dev<'a>,
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: Weight<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
    o: Weight<'a>,
    gate_up: Weight<'a>,
    down: Weight<'a>,
    k_ring: Dev<'a>,
    v_ring: Dev<'a>,
}

struct Workspace<'a> {
    sequences: usize,
    h: Dev<'a>,
    n: Dev<'a>,
    qkv: Dev<'a>,
    q: Dev<'a>,
    k: Dev<'a>,
    v: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    gate_up: Dev<'a>,
    act: Dev<'a>,
    logits: Dev<'a>,
    anchors: Dev<'a>,
    ids: Dev<'a>,
    tokens: Dev<'a>,
    confidence: Dev<'a>,
    positions: Dev<'a>,
    tables: Dev<'a>,
    attention_workspace: Dev<'a>,
    markov_workspace: Dev<'a>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

/// The target head inside a draft step.
#[derive(Clone, Copy)]
enum HeadRef<'h> {
    Bf16(*const c_void),
    Launch(&'h HeadLaunch<'h>),
}

#[derive(Clone, Copy)]
enum DraftInput<'r, 'e> {
    Rows(&'r [u8]),
    Table(&'r TokenEmbedding<'e>),
}

/// The tensors a dSpark draft reads (everything but the checkpoint's copies
/// of the target's embedding and LM head).
fn wanted(name: &str) -> bool {
    name != "embed_tokens.weight" && name != "lm_head.weight"
}

/// Header-only admission before any native allocation: the config fits the
/// target and every tensor the drafter reads is BF16 of its shape.
pub(crate) fn check_checkpoint(snapshot: &Path, hidden: usize, vocab: usize, layers: usize) -> Result<()> {
    let cfg = DsparkConfig::read(snapshot)?;
    ensure!(cfg.hidden == hidden && cfg.vocab == vocab && cfg.taps.iter().all(|&l| l < layers),
        "the dSpark drafter (hidden {}, vocab {}, taps {:?}) does not fit this target", cfg.hidden, cfg.vocab,
        cfg.taps);
    let tensors: HashMap<String, (cuteafd_core::DType, Vec<usize>)> =
        read_safetensors_metadata(&snapshot.join("model.safetensors"))?.into_iter()
            .map(|t| (t.name, (t.dtype, t.shape))).collect();
    let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
    let mut wanted = vec![("fc.weight".to_string(), vec![h, cfg.taps.len() * h]), ("hidden_norm.weight".into(), vec![h]),
        ("norm.weight".into(), vec![h]), ("markov_head.markov_w1.weight".into(), vec![cfg.vocab, cfg.rank]),
        ("markov_head.markov_w2.weight".into(), vec![cfg.vocab, cfg.rank]),
        ("confidence_head.proj.weight".into(), vec![1, h + cfg.rank]), ("confidence_head.proj.bias".into(), vec![1])];
    for l in 0..cfg.layers {
        let (p, a) = (format!("layers.{l}"), format!("layers.{l}.self_attn"));
        wanted.extend([(format!("{p}.input_layernorm.weight"), vec![h]),
            (format!("{p}.post_attention_layernorm.weight"), vec![h]),
            (format!("{a}.q_proj.weight"), vec![cfg.q_width(), h]), (format!("{a}.k_proj.weight"), vec![kv, h]),
            (format!("{a}.v_proj.weight"), vec![kv, h]), (format!("{a}.o_proj.weight"), vec![h, cfg.q_width()]),
            (format!("{a}.q_norm.weight"), vec![cfg.head_dim]), (format!("{a}.k_norm.weight"), vec![cfg.head_dim]),
            (format!("{p}.mlp.gate_proj.weight"), vec![inter, h]), (format!("{p}.mlp.up_proj.weight"), vec![inter, h]),
            (format!("{p}.mlp.down_proj.weight"), vec![h, inter])]);
    }
    for (name, shape) in wanted {
        let (dtype, found) = tensors.get(&name).with_context(|| format!("dSpark checkpoint has no {name}"))?;
        ensure!(*dtype == cuteafd_core::DType::Bf16 && *found == shape,
            "dSpark {name}: {dtype:?} {found:?}, expected BF16 {shape:?}");
    }
    Ok(())
}

/// Reads the drafter's tensors on a thread (while the target loads).
pub(crate) fn prefetch(snapshot: &Path) -> std::thread::JoinHandle<Result<HashMap<String, Vec<u8>>>> {
    let path = snapshot.join("model.safetensors");
    std::thread::spawn(move || {
        use std::os::unix::fs::FileExt;
        let file = std::fs::File::open(&path)?;
        read_safetensors_metadata(&path)?.into_iter().filter(|t| wanted(&t.name)).map(|t| {
            let mut bytes = vec![0u8; t.byte_length as usize];
            file.read_exact_at(&mut bytes, t.byte_offset)?;
            Ok((t.name, bytes))
        }).collect()
    })
}

pub(crate) struct DsparkDrafter<'a> {
    library: &'a NativeLibrary,
    pub cfg: DsparkConfig,
    stream: *mut c_void,
    pub slots: usize,
    max_sequences: usize,
    fc: Weight<'a>,
    hidden_norm: Dev<'a>,
    norm: Dev<'a>,
    markov_w1: Dev<'a>,
    markov_w2: Dev<'a>,
    /// Row norms of `markov_w2` (FP32): the exact pruning bound of the Markov argmax.
    markov_norms: Dev<'a>,
    confidence_w: Dev<'a>,
    confidence_b: Dev<'a>,
    layers: Vec<DraftLayer<'a>>,
    /// [TAP_ROWS, taps * hidden] BF16: the last step's tapped rows.
    pub taps: Dev<'a>,
    fused: Dev<'a>,
    fused_norm: Dev<'a>,
    context_kv: Dev<'a>,
    context_positions: Dev<'a>,
    context_slots: Dev<'a>,
    mask_row: Vec<u8>,
    workspace: RefCell<Option<Workspace<'a>>>,
    representation: GlmDraftRepresentation,
    /// GEMV scratch of the FP8 representation (every context/draft row count).
    fp8_workspace: Option<Dev<'a>>,
    /// How the borrowed BF16 head runs past one draft block (--draft-head).
    head_mode: Cell<DraftHead>,
    /// How the FP8 GEMMs run (--draft-linear), and the latest mode the FP8 scratch serves.
    fp8_rows: Cell<fp8_linear::Fp8Rows>,
    fp8_admitted: fp8_linear::Fp8Rows,
}

fn at(dev: &Dev<'_>, bytes: usize) -> *mut c_void {
    // SAFETY: callers stay inside the allocation.
    unsafe { dev.buffer.ptr.cast::<u8>().add(bytes) }.cast()
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<'a> DsparkDrafter<'a> {
    /// Uploads the drafter's tensors (see [`prefetch`]) and allocates `slots`
    /// ring contexts; draft steps take up to `max_sequences` sequences.
    /// `mask_row` is the target embedding of the mask token; `fp8_rows` how the
    /// FP8 GEMMs run.
    #[allow(clippy::too_many_arguments)]
    pub fn load(library: &'a NativeLibrary, snapshot: &Path, tensors: HashMap<String, Vec<u8>>, stream: *mut c_void,
        slots: usize, max_sequences: usize, mask_row: Vec<u8>, representation: GlmDraftRepresentation,
        scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let cfg = DsparkConfig::read(snapshot)?;
        let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let (resident_bytes, scratch_bytes) = cuteafd_loader::families::glm5::draft_representation::draft_resident_bytes_with_mode(
            &config, slots, max_sequences, library.sm_count()? as u64, representation, fp8_rows.code() as u8)?;
        tracing::info!(resident_bytes, scratch_bytes, "dSpark shared readiness inventory");
        ensure!((1..=MAX_SEQUENCES).contains(&max_sequences), "dSpark drafts take 1..={MAX_SEQUENCES} sequences");
        let path = snapshot.join("model.safetensors");
        let shapes: HashMap<String, Vec<usize>> = read_safetensors_metadata(&path)?.into_iter()
            .filter(|t| t.dtype == cuteafd_core::DType::Bf16).map(|t| (t.name, t.shape)).collect();
        let bytes = |name: &str, shape: &[usize]| -> Result<&[u8]> {
            let found = shapes.get(name).with_context(|| format!("dSpark checkpoint has no BF16 {name}"))?;
            ensure!(found == shape, "{name}: shape {found:?}, expected {shape:?}");
            tensors.get(name).map(Vec::as_slice).with_context(|| format!("{name} was not read"))
        };
        let upload = |data: &[u8]| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, data.len().max(256))?;
            library.copy_h2d(allocation.buffer, data)?;
            Ok(allocation)
        };
        let zeroed = |n: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, n.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
        let tensor = |name: &str, shape: &[usize]| bytes(name, shape).and_then(upload);
        let matrix = |source: Dev<'a>, n: usize, k: usize| -> Result<Weight<'a>> {
            match representation {
                GlmDraftRepresentation::Bf16Only => Ok(Weight::Bf16(source)),
                GlmDraftRepresentation::Fp8Only => {
                    let packed = Fp8Weight::pack(library, source.buffer.ptr, n, k, scales, stream);
                    // SAFETY: the packing kernel reads `source`: drain it before the source drops.
                    if let Err(error) = unsafe { library.cuda_stream_synchronize(stream) } {
                        std::mem::forget(source);
                        if let Ok(weight) = packed { weight.quarantine(); }
                        return Err(error.context("dSpark packing owners quarantined after a failed drain"));
                    }
                    let packed = packed?;
                    drop(source);
                    Ok(Weight::Fp8(packed))
                }
            }
        };
        let concat = |parts: &[(&str, usize)], cols: usize| -> Result<Dev<'a>> {
            let total: usize = parts.iter().map(|(_, rows)| rows * cols * 2).sum();
            let allocation = DeviceAllocation::new(library, total)?;
            let mut offset = 0;
            for (name, rows) in parts {
                let data = bytes(name, &[*rows, cols])?;
                library.copy_h2d(CuteafdDeviceBuffer { ptr: at(&allocation, offset), bytes: data.len(),
                    ..allocation.buffer }, data)?;
                offset += data.len();
            }
            Ok(allocation)
        };
        let layers = (0..cfg.layers).map(|l| -> Result<DraftLayer<'a>> {
            let p = format!("layers.{l}");
            let a = format!("{p}.self_attn");
            Ok(DraftLayer {
                input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                qkv: matrix(concat(&[(&format!("{a}.q_proj.weight"), cfg.q_width()), (&format!("{a}.k_proj.weight"), kv),
                    (&format!("{a}.v_proj.weight"), kv)], h)?, cfg.qkv_width(), h)?,
                q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
                k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
                o: matrix(tensor(&format!("{a}.o_proj.weight"), &[h, cfg.q_width()])?, h, cfg.q_width())?,
                gate_up: matrix(concat(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                    (&format!("{p}.mlp.up_proj.weight"), inter)], h)?, 2 * inter, h)?,
                down: matrix(tensor(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?, h, inter)?,
                k_ring: zeroed(slots * RING * kv * 2)?,
                v_ring: zeroed(slots * RING * kv * 2)?,
            })
        }).collect::<Result<Vec<_>>>()?;
        let taps = cfg.taps.len() * h;
        let fp8_workspace = match representation {
            GlmDraftRepresentation::Bf16Only => None,
            GlmDraftRepresentation::Fp8Only => {
                let shapes = [(taps, h), (h, cfg.qkv_width()), (h, 2 * kv), (cfg.q_width(), h), (h, 2 * inter),
                    (inter, h)];
                Some(fp8_linear::scratch_rows(library, TAP_ROWS.max(max_sequences * cfg.block), &shapes, fp8_rows)?)
            }
        };
        let markov_w2 = tensor("markov_head.markov_w2.weight", &[cfg.vocab, cfg.rank])?;
        let markov_norms = DeviceAllocation::new(library, cfg.vocab * 4)?;
        // SAFETY: both buffers hold `vocab` rows; the stream orders the kernel before any draft.
        unsafe { library.glmf_dspark_markov_norms(markov_w2.buffer.ptr, markov_norms.buffer.ptr, cfg.vocab, cfg.rank,
            stream)? };
        Ok(Self {
            library,
            stream,
            slots,
            max_sequences,
            fc: matrix(tensor("fc.weight", &[h, taps])?, h, taps)?,
            hidden_norm: tensor("hidden_norm.weight", &[h])?,
            norm: tensor("norm.weight", &[h])?,
            markov_w1: tensor("markov_head.markov_w1.weight", &[cfg.vocab, cfg.rank])?,
            markov_w2,
            markov_norms,
            confidence_w: tensor("confidence_head.proj.weight", &[1, h + cfg.rank])?,
            confidence_b: tensor("confidence_head.proj.bias", &[1])?,
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
            fp8_workspace,
            head_mode: Cell::new(DraftHead::Exact),
            fp8_rows: Cell::new(fp8_rows),
            fp8_admitted: fp8_rows,
            cfg,
        })
    }

    /// How draft steps run the borrowed BF16 head from now on (the FP8 head launcher ignores it).
    pub fn set_draft_head(&self, mode: DraftHead) {
        self.head_mode.set(mode);
    }

    /// How the FP8 GEMMs run from now on: the load's mode or one before it.
    pub fn set_draft_linear(&self, mode: fp8_linear::Fp8Rows) -> Result<()> {
        ensure!(mode <= self.fp8_admitted, "the dSpark FP8 scratch was admitted for {:?}, not {mode:?}",
            self.fp8_admitted);
        self.fp8_rows.set(mode);
        Ok(())
    }

    /// `out` [rows, n] = `x` [rows, k] @ rows `first..first + n` of `weight`^T.
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn linear(&self, x: *const c_void, weight: &Weight<'_>, first: usize, out: *mut c_void, rows: usize,
        k: usize, n: usize) -> Result<()> {
        match weight {
            // SAFETY: rows first..first + n lie in the matrix; the caller's contract.
            Weight::Bf16(w) => unsafe { self.library.linear_bf16(x, at(w, first * k * 2), out, rows, k, n, self.stream) },
            Weight::Fp8(w) => {
                let scratch = self.fp8_workspace.as_ref().context("FP8 dSpark scratch was not admitted")?;
                // SAFETY: the scratch covers every shape for up to max(TAP_ROWS, sequences x block) rows
                // in this mode (set_draft_linear keeps it within the admitted one).
                unsafe { w.apply_rows(self.library, x, out, false, rows, first, n, scratch, self.stream,
                    self.fp8_rows.get()) }
            }
        }
    }

    pub fn tap_index(&self, layer: usize) -> Option<usize> {
        self.cfg.taps.iter().position(|&l| l == layer)
    }

    /// Taps the mean of the `hc` streams of rows `[first, first + n)` of
    /// `streams` ([rows, hc, hidden] BF16) into tap rows `to..to + n` when
    /// `layer` is tapped.
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
            self.linear(at(&self.taps, first * width * 2), &self.fc, 0, self.fused.buffer.ptr, n, width, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, self.cfg.eps, s)?;
            let q_rows = self.cfg.q_width();
            for layer in &self.layers {
                self.linear(self.fused_norm.buffer.ptr, &layer.qkv, q_rows, self.context_kv.buffer.ptr, n, h, 2 * kv)?;
                self.library.glmf_dspark_qk_rope(self.context_kv.buffer.ptr, layer.q_norm.buffer.ptr,
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
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256));
        let head_workspace = alloc(VOCABULARY_HEAD_WORKSPACE)?;
        Ok(Workspace {
            sequences,
            h: alloc(rows * c.hidden * 2)?,
            n: alloc(rows * c.hidden * 2)?,
            qkv: alloc(rows * c.qkv_width() * 2)?,
            q: alloc(rows * c.q_width() * 2)?,
            k: alloc(rows * c.kv_width() * 2)?,
            v: alloc(rows * c.kv_width() * 2)?,
            attn: alloc(rows * c.q_width() * 2)?,
            delta: alloc(rows * c.hidden * 2)?,
            gate_up: alloc(rows * 2 * c.intermediate * 2)?,
            act: alloc(rows * c.intermediate * 2)?,
            logits: alloc(rows * c.vocab * 4)?,
            anchors: alloc(sequences * 4)?,
            ids: alloc(rows * 4)?,
            tokens: alloc(rows * 4)?,
            confidence: alloc(rows * 4)?,
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            attention_workspace: alloc(self.library.glmf_dspark_attention_workspace(sequences, c.kv_heads,
                RING + c.block)?)?,
            markov_workspace: alloc(self.library.glmf_dspark_markov_workspace(sequences, c.block)?)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, c.hidden as u32,
                rows as u32, c.vocab as u32)? },
            _head_workspace: head_workspace,
        })
    }

    fn head_ref<'h>(&self, head: &'h TargetHead<'h>) -> Result<HeadRef<'h>> {
        Ok(match head {
            TargetHead::Bf16(owner) => {
                ensure!(owner.buffer.bytes == self.cfg.vocab * self.cfg.hidden * 2,
                    "dSpark borrows the target BF16 head [{}, {}]", self.cfg.vocab, self.cfg.hidden);
                HeadRef::Bf16(owner.buffer.ptr)
            }
            TargetHead::Launch(launch) => HeadRef::Launch(launch),
        })
    }

    /// [`Self::draft`] with the block's input rows gathered on the device
    /// from the target's embedding table.
    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_ref(&head)?)
    }

    fn draft_from(&self, sequences: &[DraftSeq], input: DraftInput<'_, '_>, head: HeadRef<'_>)
        -> Result<Vec<Draft>> {
        ensure!(!matches!(head, HeadRef::Bf16(p) if p.is_null()), "dSpark has no borrowed target head");
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
        let (eps, inter) = (c.eps, c.intermediate);
        // SAFETY: every workspace buffer holds `rows` rows of its width and the
        // weights their checkpoint shapes; the stream orders the chain.
        unsafe {
            l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                self.linear(w.n.buffer.ptr, &layer.qkv, 0, w.qkv.buffer.ptr, rows, h, c.qkv_width())?;
                l.glmf_dspark_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                    w.positions.buffer.ptr, std::ptr::null(), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, rows,
                    c.heads, c.kv_heads, c.theta, eps, s)?;
                l.glmf_dspark_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                    layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4),
                    w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads, c.kv_heads, RING,
                    RING + block, c.causal, 1.0 / (c.head_dim as f32).sqrt(), s)?;
                self.linear(w.attn.buffer.ptr, &layer.o, 0, w.delta.buffer.ptr, rows, c.q_width(), h)?;
                l.glmf_dspark_add_rmsnorm(w.h.buffer.ptr, w.delta.buffer.ptr, layer.post_norm.buffer.ptr,
                    w.h.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
                self.linear(w.n.buffer.ptr, &layer.gate_up, 0, w.gate_up.buffer.ptr, rows, h, 2 * inter)?;
                l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, inter, s)?;
                self.linear(w.act.buffer.ptr, &layer.down, 0, w.delta.buffer.ptr, rows, inter, h)?;
                let next = self.layers.get(index + 1).map_or(self.norm.buffer.ptr, |n| n.input_norm.buffer.ptr);
                l.glmf_dspark_add_rmsnorm(w.h.buffer.ptr, w.delta.buffer.ptr, next, w.h.buffer.ptr, w.n.buffer.ptr,
                    rows, h, eps, s)?;
            }
            match head {
                HeadRef::Bf16(weight) => crate::families::glm5::launch_draft_head(l, &w.head, w.n.buffer.ptr,
                    weight, w.logits.buffer.ptr.cast(), rows, h, c.vocab, s, self.head_mode.get())?,
                HeadRef::Launch(launch) => launch(w.n.buffer.ptr.cast_const(), w.logits.buffer.ptr.cast(), rows, s)?,
            }
            l.glmf_dspark_markov(w.logits.buffer.ptr, self.markov_w1.buffer.ptr, self.markov_w2.buffer.ptr,
                self.markov_norms.buffer.ptr, w.anchors.buffer.ptr, w.tokens.buffer.ptr, w.markov_workspace.buffer.ptr, s_count, block, c.vocab,
                c.rank, s)?;
            l.glmf_dspark_confidence(w.n.buffer.ptr, self.markov_w1.buffer.ptr, self.confidence_w.buffer.ptr,
                self.confidence_b.buffer.ptr, w.anchors.buffer.ptr, w.tokens.buffer.ptr, w.confidence.buffer.ptr,
                s_count, block, h, c.rank, s)?;
            l.cuda_stream_synchronize(s)?;
        }
        let mut tokens = vec![0u8; rows * 4];
        let mut confidence = vec![0u8; rows * 4];
        l.copy_d2h(&mut tokens, CuteafdDeviceBuffer { bytes: rows * 4, ..w.tokens.buffer })?;
        l.copy_d2h(&mut confidence, CuteafdDeviceBuffer { bytes: rows * 4, ..w.confidence.buffer })?;
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..s_count).map(|i| Draft {
            tokens: (0..block).map(|j| word(&tokens, i * block + j)).collect(),
            features: vec![[0.0; 4]; block],
            confidence: (0..block).map(|j| f32::from_bits(word(&confidence, i * block + j))).collect(),
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

impl ReplayDrafter for DsparkDrafter<'_> {
    fn block(&self) -> usize {
        self.cfg.block
    }

    fn drafts(&self) -> usize {
        self.cfg.drafts()
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
        self.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r }).collect::<Vec<_>>())
    }

    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>> {
        let seqs: Vec<DraftSeq> = seqs.iter().map(|&(slot, anchor, position)| DraftSeq { slot, anchor, position, valid_from: 0 }).collect();
        // Replay's caller keeps the target BF16 head live throughout.
        Ok(self.draft_from(&seqs, DraftInput::Rows(anchor_rows), HeadRef::Bf16(head))?.into_iter()
            .map(|d| d.tokens).collect())
    }

    fn tap_rows(&self) -> usize {
        TAP_ROWS
    }

    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        DsparkDrafter::last_hidden(self, sequences)
    }
}

/// The GLM 5.3 Flash drafter: DFlash2 or dSpark, behind one interface the
/// engine (taps), the serve loop and glmf-golden use.
pub(crate) enum Drafter<'a> {
    Dflash2(crate::families::glm5::dflash::GlmDrafter<'a>),
    Dspark(DsparkDrafter<'a>),
}

impl<'a> Drafter<'a> {
    /// The shared draft policy's prior: DFlash2's keyed selector fit, or the
    /// dSpark confidence head blended with the history rate.
    pub fn draft_prior(&self, fp8_head: bool) -> Result<super::draft_binding::Prior> {
        Ok(match self {
            Self::Dflash2(d) => super::draft_binding::Prior::Selector(d.selector_fit("glm5_flash", fp8_head)?),
            Self::Dspark(_) => super::draft_binding::Prior::Head,
        })
    }

    /// Loads the drafter `snapshot` names (dSpark when its config says so,
    /// else DFlash2) for a target of `hidden` x `vocab` with `layers` layers:
    /// `slots` ring contexts, draft steps of up to `sequences` sequences.
    #[allow(clippy::too_many_arguments)]
    pub fn load(library: &'a NativeLibrary, snapshot: &Path, stream: *mut c_void, slots: usize, sequences: usize,
        embedding: &TokenEmbedding<'_>, hidden: usize, vocab: usize, layers: usize,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows)
        -> Result<Self> {
        if is_dspark(snapshot)? {
            let cfg = DsparkConfig::read(snapshot)?;
            ensure!(cfg.hidden == hidden && cfg.vocab == vocab && cfg.taps.iter().all(|&l| l < layers),
                "the dSpark drafter does not fit this target");
            let mask = embedding.host_rows(&[cfg.mask_token])?;
            tracing::info!(mask_source = "target mask_token row", mask_token = cfg.mask_token, "drafter mask source");
            let tensors = prefetch(snapshot).join().map_err(|_| anyhow::anyhow!("drafter read panicked"))??;
            return Ok(Self::Dspark(DsparkDrafter::load(library, snapshot, tensors, stream, slots,
                sequences.min(MAX_SEQUENCES), mask, representation, scales, fp8_rows)?));
        }
        let cfg = crate::families::glm5::dflash::DflashConfig::read(snapshot)?;
        ensure!(cfg.hidden == hidden && cfg.vocab == vocab && cfg.taps.iter().all(|&l| l < layers),
            "the DFlash2 drafter does not fit this target");
        let mask = embedding.host_rows(&[cfg.mask_token])?;
        tracing::info!(mask_source = "target mask_token row", mask_token = cfg.mask_token, "drafter mask source");
        let file = crate::families::glm5::dflash::prefetch(snapshot).join()
            .map_err(|_| anyhow::anyhow!("drafter read panicked"))??;
        Ok(Self::Dflash2(crate::families::glm5::dflash::GlmDrafter::load(library, snapshot, file, stream, slots,
            sequences, mask, true, representation, scales, fp8_rows)?))
    }

    pub fn prepare_workspace(&self) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.prepare_workspace(),
            Self::Dspark(d) => d.prepare_workspace(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Dflash2(_) => "DFlash2",
            Self::Dspark(_) => "dSpark",
        }
    }

    /// Most sequences one draft step takes.
    pub fn max_batch_sequences(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.max_batch_sequences(),
            Self::Dspark(d) => d.max_sequences,
        }
    }

    /// How draft steps run the target's BF16 head past one draft block (--draft-head).
    pub fn set_draft_head(&self, mode: DraftHead) {
        match self {
            Self::Dflash2(d) => d.set_draft_head(mode),
            Self::Dspark(d) => d.set_draft_head(mode),
        }
    }

    /// How the FP8 GEMMs run from now on (--draft-linear): the load's mode or one before it.
    pub fn set_draft_linear(&self, mode: fp8_linear::Fp8Rows) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.set_draft_linear(mode),
            Self::Dspark(d) => d.set_draft_linear(mode),
        }
    }

    /// Ring slots (sequences with a drafter context).
    pub fn slots(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.slots,
            Self::Dspark(d) => d.slots,
        }
    }

    /// Tokens one draft proposes per sequence.
    pub fn drafts(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.cfg.drafts(),
            Self::Dspark(d) => d.cfg.drafts(),
        }
    }

    /// Rows of one draft block.
    pub fn block(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.cfg.block,
            Self::Dspark(d) => d.cfg.block,
        }
    }

    /// Target layers whose outputs are tapped.
    pub fn taps(&self) -> &[usize] {
        match self {
            Self::Dflash2(d) => &d.cfg.taps,
            Self::Dspark(d) => &d.cfg.taps,
        }
    }

    pub fn tap_streams(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize) -> Result<()> {
        self.tap_streams_at(layer, streams, hc, first, n, 0)
    }

    pub fn tap_streams_at(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize, to: usize)
        -> Result<()> {
        match self {
            Self::Dflash2(d) => d.tap_streams_at(layer, streams, hc, first, n, to),
            Self::Dspark(d) => d.tap_streams_at(layer, streams, hc, first, n, to),
        }
    }

    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.update(rows),
            Self::Dspark(d) => d.update(rows),
        }
    }

    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        match self {
            Self::Dflash2(d) => d.draft_device(sequences, embedding, head),
            Self::Dspark(d) => d.draft_device(sequences, embedding, head),
        }
    }

    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.put_taps(rows),
            Self::Dspark(d) => d.put_taps(rows),
        }
    }

    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        match self {
            Self::Dflash2(d) => d.last_hidden(sequences),
            Self::Dspark(d) => d.last_hidden(sequences),
        }
    }

    /// The drafter as [`ReplayDrafter`] (teacher-forced replays).
    pub fn replay(&self) -> &dyn ReplayDrafter {
        match self {
            Self::Dflash2(d) => d,
            Self::Dspark(d) => d,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_detection_reports_missing_and_corrupt_config() {
        let dir = tempfile::tempdir().unwrap();
        assert!(is_dspark(dir.path()).unwrap_err().to_string().contains("config.json"));
        std::fs::write(dir.path().join("config.json"), b"bad").unwrap();
        assert!(is_dspark(dir.path()).unwrap_err().to_string().contains("parsing drafter"));
    }

    #[test]
    fn reads_the_redhat_config() {
        let dir = std::env::temp_dir().join(format!("cuteafd-dspark-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{
          "aux_hidden_state_layer_ids": [20, 28, 32, 36, 40, 44], "block_size": 8,
          "confidence_head_with_markov": true, "draft_vocab_size": 154880, "enable_confidence_head": true,
          "markov_head_type": "vanilla", "markov_rank": 256, "mask_token_id": 154856, "sample_from_anchor": true,
          "sliding_window_non_causal": false, "speculators_model_type": "dspark",
          "transformer_layer_config": {"head_dim": 64, "hidden_act": "silu", "hidden_size": 4096,
            "intermediate_size": 12288, "layer_types": ["sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "sliding_attention"], "num_attention_heads": 64,
            "num_hidden_layers": 5, "num_key_value_heads": 64, "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "sliding_window": 2048,
            "vocab_size": 154880}}"#).unwrap();
        assert!(is_dspark(&dir).unwrap());
        let cfg = DsparkConfig::read(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(cfg.taps, vec![19, 27, 31, 35, 39, 43]);
        assert_eq!((cfg.block, cfg.drafts(), cfg.qkv_width(), cfg.kv_width()), (8, 8, 12288, 4096));
        assert!(cfg.causal);
        assert_eq!(cfg.mask_token, 154856);
    }
}
