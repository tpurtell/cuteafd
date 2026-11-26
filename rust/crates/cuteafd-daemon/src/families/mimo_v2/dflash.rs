//! DFlash block drafter of MiMo V2.6 Pro (the snapshot's `dflash/`: five
//! Qwen3-style layers of 128 query / 8 KV heads, attention sinks, value
//! scale 0.612, NeoX RoPE on the first 64 of 128 dims, a 1024-position
//! sliding window, block 8).
//!
//! Every target step taps the outputs (pre-norm residual) of the
//! `target_layer_ids` layers (0, 15, 31, 47, 69) into [`MimoDrafter::taps`].
//! Once a step's rows are committed, [`MimoDrafter::update`] projects their
//! taps (`hidden_norm(fc(taps))`) into every draft layer's context K/V (a ring
//! of the sequence's last 1024 positions). [`MimoDrafter::draft`] runs the
//! layers over the block `[anchor, mask x 7]` at the anchor's position (the
//! mask token's row is `mask_embedding.pt`'s trained vector) non-causally
//! against the ring, and takes the argmax of the target head's logits for
//! block rows 1..8. Semantics: python/reference/families/mimo_v2/mimo_dflash/reference.py
//! (SGLang's DFlash path for MiMo). Drafts only steer speculation; verify
//! steps keep output identical to plain greedy decoding.
use crate::shared::fp8_linear;
use crate::shared::speculation::weights::{Load, ShardAxis, Weight as DraftWeight};
use crate::shared::memory::device::{Device, DeviceOwner};
use crate::shared::peer_split::{PeerExchange, DIRECT};
use std::rc::Rc;
use crate::shared::experts::rtx::dense::{DenseLoad, Operand, Rect};
use crate::shared::memory::{DeviceAllocation, Tag};
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use super::head::BorrowedHead;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::read_safetensors_metadata;
use cuteafd_loader::families::mimo_v2::draft_representation::{
    MimoDraftCapacity, MimoDraftRepresentation, MimoDraftRuntimeLayout,
};
use std::cell::RefCell;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) use cuteafd_loader::families::mimo_v2::draft_config::{RING, TAP_ROWS};
const WEIGHTS: &str = "dflash_draft_model.safetensors";

pub(crate) use cuteafd_loader::families::mimo_v2::draft_config::DflashConfig;

fn draft_fp8_scratch<'a>(library: &'a NativeLibrary, layout: &MimoDraftRuntimeLayout)
    -> Result<Option<Dev<'a>>> {
    layout.fp8_scratch.as_ref().map(|scratch| {
        let shapes = scratch.shapes.iter().map(|shape| Ok((
            usize::try_from(shape.k).context("DFlash scratch K width")?,
            usize::try_from(shape.n).context("DFlash scratch N width")?,
        ))).collect::<Result<Vec<_>>>()?;
        fp8_linear::scratch(library, scratch.rows, &shapes)
    }).transpose()
}

/// The drafter directory of a MiMo snapshot (`SNAP/dflash`), or `path` itself.
pub(crate) fn drafter_dir(path: &Path) -> PathBuf {
    if path.join(WEIGHTS).exists() { path.to_path_buf() } else { path.join("dflash") }
}

struct DraftLayer<'a> {
    input_norm: Dev<'a>,
    post_norm: Dev<'a>,
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: DraftWeight<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
    sinks: Option<Dev<'a>>,
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
    qkv: Dev<'a>,
    q: Dev<'a>,
    k: Dev<'a>,
    v: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    reduced: Option<Dev<'a>>,
    gate_up: Dev<'a>,
    act: Dev<'a>,
    output: Option<DraftOutput<'a>>,
    positions: Dev<'a>,
    tables: Dev<'a>,
    /// Token ids of the block rows (anchors; the mask rows' sentinel).
    ids: Dev<'a>,
    attention_workspace: Dev<'a>,
}

/// Only the rank borrowing the target head allocates sampling/output storage.
struct DraftOutput<'a> {
    logits: Dev<'a>,
    unary: Dev<'a>,
    candidates: Dev<'a>,
    landing: crate::shared::memory::HostAllocation<'a>,
    topk_workspace: Dev<'a>,
    head: Option<VocabularyHead<'a>>,
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

/// Drafted tokens after one anchor, their policy scores and optional trace features.
#[derive(Debug, Clone)]
pub(crate) struct Draft {
    pub tokens: Vec<u32>,
    pub features: Vec<[f32; 4]>,
    pub confidence: Vec<f32>,
}

pub(crate) struct MimoDrafter<'a> {
    pub(crate) kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    library: &'a NativeLibrary,
    projections: cuteafd_ffi::draft_projection::DraftProjections<'a>,
    timing: crate::shared::draft::timing::DraftTiming<'a>,
    pub cfg: DflashConfig,
    stream: *mut c_void,
    stream_owner: Option<Rc<crate::shared::memory::device::Stream<'a>>>,
    /// The draft round's host wait: sleeps instead of spinning the engine thread.
    round: std::cell::RefCell<crate::shared::memory::BlockingWait<'a>>,
    /// Ring slots (sequences with a drafter context).
    pub slots: usize,
    max_sequences: usize,
    fc: DraftWeight<'a>,
    hidden_norm: Dev<'a>,
    norm: Dev<'a>,
    layers: Vec<DraftLayer<'a>>,
    /// [TAP_ROWS, taps * hidden] BF16: the last step's tapped rows.
    pub taps: Dev<'a>,
    fused: Dev<'a>,
    fused_norm: Dev<'a>,
    context_kv: Dev<'a>,
    context_positions: Dev<'a>,
    context_slots: Dev<'a>,
    /// The mask token's (trained) embedding row, and its device copy (the
    /// gather's fallback row for the mask id).
    mask_row: Vec<u8>,
    mask_device: Dev<'a>,
    workspace: RefCell<Option<Workspace<'a>>>,
    /// The speculation trace is open: rounds also land the top-16 logits so the
    /// trace records [`features`] (diagnostic only; no decision reads them).
    trace_features: std::cell::Cell<bool>,
    policy_scores: std::cell::Cell<bool>,
    representation: MimoDraftRepresentation,
    bf16_head: bool,
    fp8_workspace: Option<Dev<'a>>,

    rank: usize,
    ranks: usize,
    peer: Option<DeviceOwner<'a, Box<MimoDrafter<'a>>>>,
    exchange: Option<Rc<PeerExchange<'a>>>,
}

/// The block rows' mask id: past every table, so the gather takes the mask row.
const MASK_ID: u32 = u32::MAX;

/// Where a draft step's block rows come from.
#[derive(Clone, Copy)]
enum Anchors<'r> {
    /// The anchors' embedding rows (host).
    Rows(&'r [u8]),
    /// The anchors' token ids through the target's embedding table.
    Tokens(&'r TokenEmbedding<'r>),
}

fn at(dev: &Dev<'_>, bytes: usize) -> *mut c_void {
    // SAFETY: callers stay inside the allocation.
    unsafe { dev.buffer.ptr.cast::<u8>().add(bytes) }.cast()
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Read only the header before target/native admission. This loader currently
/// supports BF16 drafter checkpoints; FP8 storage is an explicit load transform.
pub(crate) fn checkpoint_representation(dir: &Path) -> Result<MimoDraftRepresentation> {
    let tensors = read_safetensors_metadata(&dir.join(WEIGHTS))?;
    ensure!(!tensors.is_empty(), "DFlash checkpoint has no tensors");
    for tensor in tensors {
        ensure!(tensor.dtype == cuteafd_core::DType::Bf16,
            "DFlash checkpoint tensor {} has unsupported source {:?}; the drafter loader requires checkpoint BF16, add a native-format reader before selecting this checkpoint",
            tensor.name, tensor.dtype);
    }
    Ok(MimoDraftRepresentation::Bf16Only)
}

fn tensor_shapes(c: &DflashConfig) -> Vec<(String, Vec<usize>)> {
    let (h, kv, inter) = (c.hidden, c.kv_width(), c.intermediate);
    let mut shapes = vec![("fc.weight".into(), vec![h, c.taps.len() * h]),
        ("hidden_norm.weight".into(), vec![h]), ("norm.weight".into(), vec![h])];
    for l in 0..c.layers {
        let p = format!("layers.{l}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            shapes.push((format!("{p}.{norm}.weight"), vec![h]));
        }
        let a = format!("{p}.self_attn");
        for (projection, rows, cols) in [("q_proj", c.heads * c.head_dim, h),
            ("k_proj", kv, h), ("v_proj", kv, h), ("o_proj", h, c.heads * c.head_dim)] {
            shapes.push((format!("{a}.{projection}.weight"), vec![rows, cols]));
        }
        for norm in ["q_norm", "k_norm"] {
            shapes.push((format!("{a}.{norm}.weight"), vec![c.head_dim]));
        }
        if c.sinks { shapes.push((format!("{a}.attention_sink_bias"), vec![c.heads])); }
        for (projection, rows, cols) in [("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)] {
            shapes.push((format!("{p}.mlp.{projection}.weight"), vec![rows, cols]));
        }
    }
    shapes
}

/// Reads the drafter's safetensors file on a thread (while the target loads).
pub(crate) fn prefetch(dir: &Path) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    let path = dir.join(WEIGHTS);
    std::thread::spawn(move || std::fs::read(path))
}

/// The trained mask embedding (`mask_embedding.pt`: a torch zip whose
/// `*/data/0` entry holds `hidden` BF16 values, stored uncompressed), found
/// through the zip's central directory.
#[derive(Debug, thiserror::Error)]
#[error("{path}: the drafter's trained mask_embedding.pt is required: {source}")]
pub(crate) struct MaskEmbeddingReadError {
    pub path: std::path::PathBuf,
    #[source]
    pub source: std::io::Error,
}

#[cfg(test)]
pub(crate) fn mask_embedding(dir: &Path, hidden: usize) -> Result<Vec<u8>> {
    mask_source(dir, hidden).map(|(row, _)| row)
}

fn mask_source(dir: &Path, hidden: usize) -> Result<(Vec<u8>, Rect)> {
    let path = dir.join("mask_embedding.pt");
    let bytes = std::fs::read(&path).map_err(|source| MaskEmbeddingReadError { path: path.clone(), source })?;
    let path = path.canonicalize()?;
    let u16_at = |o: usize| -> Result<usize> {
        Ok(usize::from(u16::from_le_bytes(bytes.get(o..o + 2).context("truncated zip")?.try_into().unwrap())))
    };
    let u32_at = |o: usize| -> Result<usize> {
        Ok(u32::from_le_bytes(bytes.get(o..o + 4).context("truncated zip")?.try_into().unwrap()) as usize)
    };
    let end = (0..bytes.len().saturating_sub(21)).rev().find(|&o| bytes[o..o + 4] == [0x50, 0x4b, 0x05, 0x06])
        .with_context(|| format!("{}: not a zip file", path.display()))?;
    let (count, mut at) = (u16_at(end + 10)?, u32_at(end + 16)?);
    for _ in 0..count {
        ensure!(u32_at(at)? == 0x0201_4b50, "{}: bad central directory", path.display());
        let (method, size) = (u16_at(at + 10)?, u32_at(at + 20)?);
        let (name_len, extra_len, comment_len, local) = (u16_at(at + 28)?, u16_at(at + 30)?, u16_at(at + 32)?,
            u32_at(at + 42)?);
        let name = std::str::from_utf8(bytes.get(at + 46..at + 46 + name_len).context("truncated zip")?).unwrap_or("");
        if name.ends_with("/data/0") {
            let data = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
            ensure!(method == 0 && size == hidden * 2 && data + size <= bytes.len(),
                "{}: expected {} stored bytes in {name}, found method {method} size {size}", path.display(), hidden * 2);
            tracing::info!(path = %path.display(), mask_source = "trained mask_embedding.pt", "drafter mask source");
            return Ok((bytes[data..data + size].to_vec(), Rect { name: "mask_embedding".into(),
                path, base: data as u64, row_bytes: size, rows: 0..1, cols: 0..size }));
        }
        at += 46 + name_len + extra_len + comment_len;
    }
    anyhow::bail!("{}: no data/0 entry", path.display())
}

#[cfg(test)]
mod mask_tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    #[test]
    fn trace_features_match_pre_2443_selector_row() {
        // Two equal leaders and a far tail: margin 0, p = 1/2, entropy ln 2.
        let mut top = vec![-80.0f32; 16];
        top[0] = 3.0; top[1] = 3.0;
        let [margin, best, entropy, rank] = features(&top);
        assert_eq!((margin, rank), (0.0, 0.0));
        assert_eq!(candidate_probability(&top).to_bits(), best.to_bits());
        assert!((best - 0.5).abs() < 1e-6 && (entropy - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn trained_mask_missing_is_typed_and_corrupt_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let error = mask_embedding(dir.path(), 2).unwrap_err();
        let typed = error.downcast_ref::<MaskEmbeddingReadError>().unwrap();
        assert_eq!(typed.path, dir.path().join("mask_embedding.pt"));
        assert_eq!(typed.source.kind(), std::io::ErrorKind::NotFound);
        assert!(error.to_string().contains("trained mask_embedding.pt is required"));
        std::fs::write(&typed.path, b"corrupt").unwrap();
        assert!(mask_embedding(dir.path(), 2).unwrap_err().to_string().contains("not a zip"));
    }

    fn cfg() -> DflashConfig {
        DflashConfig { hidden: 128, intermediate: 256, layers: 2, heads: 2, kv_heads: 1,
            head_dim: 128, rope_dim: 64, theta: 10000., eps: 1e-6, block: 8,
            mask_token: 0, taps: vec![0, 1], vocab: 256, window: RING, v_scale: 0.612, sinks: true }
    }

    fn groups(c: &DflashConfig) -> Vec<Vec<(String, Vec<usize>)>> {
        let mut groups = tensor_shapes(c).into_iter().map(|shape| vec![shape]).collect::<Vec<_>>();
        for l in 0..c.layers {
            for suffixes in [vec!["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
                vec!["mlp.gate_proj.weight", "mlp.up_proj.weight"]] {
                let mut fused = Vec::new();
                for suffix in suffixes {
                    let name = format!("layers.{l}.{suffix}");
                    let at = groups.iter().position(|g| g[0].0 == name).unwrap();
                    fused.extend(groups.remove(at));
                }
                groups.push(fused);
            }
        }
        groups
    }

    #[test]
    fn bundled_drafter_headers_cover_every_bf16_class_and_sink_choice() -> Result<()> {
        use cuteafd_loader::plan::testing::{t, write_safetensors};
        for sinks in [false, true] {
            let c = DflashConfig { sinks, ..cfg() };
            let root = tempfile::tempdir()?;
            let shapes = tensor_shapes(&c);
            let tensors = shapes.iter().map(|(name, shape)| t(name.clone(), "BF16", shape)).collect::<Vec<_>>();
            let path = root.path().join(WEIGHTS);
            write_safetensors(&path, &tensors);
            let source = crate::shared::speculation::weights::Source::open_file(&path)?;
            for (name, shape) in &shapes { source.rect(name, shape)?; }
            assert_eq!(shapes.len(), 3 + c.layers * (11 + usize::from(sinks)));
            assert_eq!(checkpoint_representation(root.path())?, MimoDraftRepresentation::Bf16Only);
            let bytes = shapes.iter().map(|(_, s)| s.iter().product::<usize>() * 2).sum::<usize>();
            let grouped = groups(&c).iter().flatten().map(|(_, s)| s.iter().product::<usize>() * 2).sum::<usize>();
            assert_eq!(bytes, grouped);
        }
        Ok(())
    }

    /// The planned arena names every operand `load_rank` allocates (the
    /// loader's `+` groups, FP8 planes, the mask row) at exactly its bytes;
    /// TP2 halves split the sharded projections and replicate the rest.
    #[test]
    fn drafter_arena_names_each_loaded_operand_at_its_extent() -> Result<()> {
        use cuteafd_loader::families::mimo_v2::weight_arena::dflash_arena_requests;
        for sinks in [false, true] {
            let c = DflashConfig { sinks, ..cfg() };
            for representation in [MimoDraftRepresentation::Bf16Only, MimoDraftRepresentation::Fp8Only] {
                let one = dflash_arena_requests(&c, representation, 0, 1)?;
                let mut expected = groups(&c).into_iter().flat_map(|group| {
                    let name = group.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join("+");
                    let bytes = group.iter().map(|(_, s)| s.iter().product::<usize>() * 2).sum::<usize>();
                    if representation == MimoDraftRepresentation::Fp8Only && group[0].1.len() == 2 {
                        vec![(format!("{name}.payload"), bytes / 2), (format!("{name}.scales"), bytes / 2 / 128 * 4)]
                    } else { vec![(name, bytes.max(256))] }
                }).collect::<Vec<_>>();
                expected.push(("mask_embedding".into(), (c.hidden * 2).max(256)));
                expected.sort();
                let actual = one.views().iter().map(|(n, v)| (n.clone(), v.bytes())).collect::<Vec<_>>();
                assert_eq!(actual, expected, "{representation:?} sinks={sinks}");
                let layout = c.weight_layout(representation)?;
                assert!(one.payload_bytes() as u64 >= layout.resident_bytes()?, "{representation:?}");
                // TP2 needs dividing KV heads (the shared fixture has one).
                let split = DflashConfig { kv_heads: 2, ..c.clone() };
                let one = dflash_arena_requests(&split, representation, 0, 1)?;
                let halves = (0..2).map(|rank| dflash_arena_requests(&split, representation, rank, 2))
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(halves[0].views().keys().collect::<Vec<_>>(), one.views().keys().collect::<Vec<_>>());
                assert_eq!(halves[0].views().values().map(|v| v.bytes()).collect::<Vec<_>>(),
                    halves[1].views().values().map(|v| v.bytes()).collect::<Vec<_>>());
                assert_eq!(halves[1].id(), "mimo.dflash.rank1");
                assert!(halves[0].payload_bytes() < one.payload_bytes());
            }
        }
        Ok(())
    }

    #[test]
    fn bundled_and_external_drafters_select_their_own_namespace_and_normalize_aliases() -> Result<()> {
        use crate::shared::experts::rtx::push::PushPolicy;
        use cuteafd_loader::plan::testing::{t, write_safetensors};
        use std::os::unix::{ffi::OsStrExt, fs::symlink};
        let root = tempfile::tempdir()?;
        let namespace = root.path().join("namespace");
        let bundled = namespace.join("target/dflash");
        let local = root.path().join("local-draft");
        std::fs::create_dir_all(&bundled)?;
        std::fs::create_dir_all(&local)?;
        let path = std::ffi::CString::new(namespace.as_os_str().as_bytes())?;
        let value = br#"{"me":"raptor"}"#;
        // SAFETY: the test-owned directory and xattr bytes remain live for setxattr.
        ensure!(unsafe { libc::setxattr(path.as_ptr(), c"user.sparknest.cluster".as_ptr(),
            value.as_ptr().cast(), value.len(), 0) } == 0, "setxattr: {}", std::io::Error::last_os_error());
        assert_eq!(PushPolicy::for_snapshot(&namespace.join("target"))?, PushPolicy::Required);
        assert_eq!(PushPolicy::for_snapshot(&bundled)?, PushPolicy::Required);
        assert_eq!(PushPolicy::for_snapshot(&local)?, PushPolicy::Mount);
        let alias = root.path().join("hf-hub-alias");
        symlink(&bundled, &alias)?;
        assert_eq!(PushPolicy::for_snapshot(&alias)?, PushPolicy::Required);
        write_safetensors(&bundled.join(WEIGHTS), &[t("norm.weight", "BF16", &[128])]);
        write_mask(&bundled, &[0; 256]);
        let source = crate::shared::speculation::weights::Source::open_file(&alias.join(WEIGHTS))?;
        let rect = source.rect("norm.weight", &[128])?;
        assert_eq!(rect.path, bundled.join(WEIGHTS).canonicalize()?);
        let (_, mask) = mask_source(&alias, 128)?;
        assert_eq!(mask.path, bundled.join("mask_embedding.pt").canonicalize()?);
        let operand = |rect: Rect| Operand { device: 0,
            buffer: CuteafdDeviceBuffer { bytes: rect.bytes(), ..Default::default() },
            bytes: rect.bytes(), lands: vec![(rect, 0)] };
        let rank = crate::shared::experts::rtx::dense::RankLayout::new(
            vec![operand(rect), operand(mask)], Vec::new(), 0)?;
        let plan = rank.checked(&namespace.canonicalize()?, 1, 0)?;
        assert_eq!(plan.plan.files.len(), 2);
        assert_eq!(plan.totals.rank_bytes[0], 512);
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn bundled_bf16_drafter_records_without_payload_and_replays_mount_bytes() -> Result<()> {
        use crate::shared::{experts::rtx::dense, memory::device::Device};
        use cuteafd_loader::plan::testing::{t, write_safetensors};
        use std::os::unix::fs::FileExt;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let snapshot = root.path().join("models--mimo/snapshots/test");
        let dir = snapshot.join("dflash");
        std::fs::create_dir_all(&dir)?;
        let cfg = cfg();
        let shapes = tensor_shapes(&cfg);
        let tensors = shapes.iter().map(|(name, shape)| t(name.clone(), "BF16", shape)).collect::<Vec<_>>();
        let path = dir.join(WEIGHTS);
        write_safetensors(&path, &tensors);
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        for tensor in read_safetensors_metadata(&path)? {
            let seed = tensor.name.bytes().fold(0u8, |a, b| a.wrapping_add(b));
            let bytes = (0..tensor.byte_length).map(|i| (i as u8).wrapping_mul(29) ^ seed).collect::<Vec<_>>();
            file.write_all_at(&bytes, tensor.byte_offset)?;
        }
        let mask = (0..cfg.hidden * 2).map(|i| i as u8).collect::<Vec<_>>();
        write_mask(&dir, &mask);
        let device = Device { library: &library, id: 0 };
        let groups = groups(&cfg);
        let load = || device.run(|| {
            let recording = dense::is_recording();
            let bytes = if recording { Vec::new() } else { std::fs::read(&path)? };
            let load = Load::with_prefetched_file(&library, &path, bytes)?;
            load.validate(shapes.clone())?;
            // No payload can be opened after header validation while recording.
            let hidden = path.with_extension("hidden");
            if recording { std::fs::rename(&path, &hidden)?; }
            let owners = groups.iter().map(|group| {
                if group.len() == 1 { return load.tensor(&group[0].0, &group[0].1); }
                let parts = group.iter().map(|(name, shape)| (name.as_str(), shape[0])).collect::<Vec<_>>();
                let DraftWeight::Bf16(owner) = load.matrix(&parts, group[0].1[1], false,
                    fp8_linear::Fp8Scales::Amax, std::ptr::null_mut())? else { unreachable!() };
                Ok(owner)
            }).collect::<Result<Vec<_>>>();
            if recording { std::fs::rename(hidden, &path)?; }
            let mut owners = owners?;
            let (row, source) = mask_source(&dir, cfg.hidden)?;
            let owner = DeviceAllocation::new(&library, row.len().max(256), crate::shared::memory::TEST)?;
            DenseLoad::mount(&library).operand(Operand { device: 0, buffer: owner.buffer,
                bytes: row.len(), lands: vec![(source, 0)] })?;
            owners.push(owner);
            Ok(owners)
        });
        let mounted = load()?;
        let (pushed, records) = dense::recording(load)?;
        assert_eq!(records.operands.len(), shapes.len() - 3 * cfg.layers + 1);
        assert_eq!(records.operands.iter().map(|o| o.lands.len()).sum::<usize>(), shapes.len() + 1);
        assert!(records.steps.is_empty() && records.finishes.is_empty());
        let source_bytes = shapes.iter().map(|(_, s)| s.iter().product::<usize>() * 2).sum::<usize>();
        let rank = dense::RankLayout::new(records.operands.clone(), Vec::new(), 0)?;
        let plan = rank.checked(root.path(), 7, 6)?;
        assert_eq!(plan.totals.rank_bytes[6], (source_bytes + mask.len()) as u64);
        assert_eq!(plan.plan.files.len(), 2);
        assert!(plan.plan.files.iter().any(|p| p.ends_with("dflash/dflash_draft_model.safetensors")));
        assert!(plan.plan.files.iter().any(|p| p.ends_with("dflash/mask_embedding.pt")));
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        crate::shared::experts::rtx::push::tests::receive_pushed(device, None, root.path(), sockets.path(),
            &records, None, 6, 1)?;
        for ((expected, actual), operand) in mounted.iter().zip(&pushed).zip(&records.operands) {
            let read = |owner: &Dev<'_>| {
                let mut bytes = vec![0; operand.bytes];
                library.copy_d2h_quiescent(&mut bytes, owner.buffer)?;
                Ok::<_, anyhow::Error>(bytes)
            };
            assert_eq!(read(expected)?, read(actual)?);
        }
        Ok(())
    }

    #[derive(Clone)]
    struct LogWriter(Arc<Mutex<Vec<u8>>>);
    impl Write for LogWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }

    fn write_mask(dir: &Path, data: &[u8]) {
        let name = b"mask_embedding/data/0";
        let mut bytes = vec![0u8; 30];
        bytes[..4].copy_from_slice(&0x0403_4b50u32.to_le_bytes());
        bytes[26..28].copy_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&data);
        let central = bytes.len() as u32;
        let mut header = vec![0u8; 46];
        header[..4].copy_from_slice(&0x0201_4b50u32.to_le_bytes());
        header[20..24].copy_from_slice(&(data.len() as u32).to_le_bytes());
        header[28..30].copy_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(name);
        let mut end = vec![0u8; 22];
        end[..4].copy_from_slice(&0x0605_4b50u32.to_le_bytes());
        end[10..12].copy_from_slice(&1u16.to_le_bytes());
        end[16..20].copy_from_slice(&central.to_le_bytes());
        bytes.extend_from_slice(&end);
        std::fs::write(dir.join("mask_embedding.pt"), bytes).unwrap();
    }

    #[test]
    fn trained_mask_loads_and_logs_its_source() {
        let dir = tempfile::tempdir().unwrap();
        let data = [0u8, 0x3f, 0u8, 0x40];
        write_mask(dir.path(), &data);
        let log = Arc::new(Mutex::new(Vec::new()));
        let sink = log.clone();
        let subscriber = tracing_subscriber::fmt().without_time().with_ansi(false)
            .with_writer(move || LogWriter(sink.clone())).finish();
        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(mask_embedding(dir.path(), 2).unwrap(), data);
        });
        let log = String::from_utf8(log.lock().unwrap().clone()).unwrap();
        assert!(log.contains("trained mask_embedding.pt") && log.contains("drafter mask source"), "{log}");
    }
}

impl<'a> MimoDrafter<'a> {

    /// Loads the drafter's weights from `file` (its safetensors bytes, see
    /// [`prefetch`]) and allocates `slots` ring contexts; draft steps take up
    /// to `max_sequences` sequences. The trained mask keeps its host fallback row.
    pub fn load(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, dir: &Path, file: Vec<u8>, stream_owner: Rc<crate::shared::memory::device::Stream<'a>>, slots: usize,
        max_sequences: usize, representation: MimoDraftRepresentation, bf16_head: bool,
        scales: fp8_linear::Fp8Scales, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
        exchange: Option<Rc<PeerExchange<'a>>>) -> Result<Self> {
        let ranks = if exchange.is_some() { 2 } else { 1 };
        let mut lead = Self::load_rank(library, programs, dir, Some(file), stream_owner.raw, Some(stream_owner), slots, max_sequences,
            representation, scales, kv_cache, bf16_head, 0, ranks)?;
        if let Some(exchange) = exchange {
            let device = Device { library, id: exchange.device(1) };
            let peer = device.own(|| Self::load_rank(library, programs, dir, None, exchange.stream(1), None,
                slots, max_sequences, representation, scales, kv_cache, bf16_head, 1, ranks).map(Box::new))?;
            lead.peer = Some(peer);
            lead.exchange = Some(exchange);
        }
        Ok(lead)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_rank(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>,
        dir: &Path, file: Option<Vec<u8>>, stream: *mut c_void,
        stream_owner: Option<Rc<crate::shared::memory::device::Stream<'a>>>, slots: usize, max_sequences: usize,
        representation: MimoDraftRepresentation, scales: fp8_linear::Fp8Scales,
        kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache, bf16_head: bool, rank: usize, ranks: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let source_cfg = DflashConfig::read(dir)?;
        let cfg = source_cfg.for_tp(ranks)?;
        let h = cfg.hidden;
        let shapes = [(h * cfg.taps.len(), h), (h, cfg.qkv_width()), (h, 2 * cfg.kv_heads * cfg.head_dim),
            (cfg.heads * cfg.head_dim, h), (h, 2 * cfg.intermediate), (cfg.intermediate, h)];
        let projections = cuteafd_ffi::draft_projection::DraftProjections::load(programs,
            if representation == MimoDraftRepresentation::Bf16Only { &shapes } else { &[] },
            cuteafd_ffi::draft_projection::batch_rows(TAP_ROWS, cfg.block, max_sequences,
                programs.capacities().decode_rows.unwrap_or(0)))?;

        ensure!(matches!(representation, MimoDraftRepresentation::Bf16Only | MimoDraftRepresentation::Fp8Only),
            "MiMo serving supports only immutable bf16-only or fp8-only drafter storage; historical dual/mixed controls are not production modes");
        let mut capacity = MimoDraftCapacity::new(slots, max_sequences, cfg.block)?;
        capacity.kv_cache = kv_cache;
        let runtime_layout = cfg.runtime_layout(representation, capacity)?;
        let weight_layout = runtime_layout.weights;
        let (mask_row, mask_source) = mask_source(dir, cfg.hidden)?;
        // One planned arena per rank (DraftLoad, after target KV); views retain it.
        let device = Device { library, id: library.cuda_get_device()? };
        let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(device,
            cuteafd_loader::families::mimo_v2::weight_arena::dflash_arena_requests(&source_cfg, representation, rank, ranks)?, cuteafd_core::memory_layout::ArenaTag::drafter_weights("mimo/dflash"))?;
        let mut allocate = |id: i32, name: &str, bytes: usize| {
            ensure!(id == device.id, "MiMo DFlash rank {rank} allocates on GPU {id}, not {}", device.id);
            arena.view(name, bytes)
        };
        let checkpoint = match file {
            Some(file) => Load::with_prefetched_file(library, &dir.join(WEIGHTS), file)?,
            None => Load::open_file(library, &dir.join(WEIGHTS))?,
        }.with_allocator(&mut allocate);
        checkpoint.validate(tensor_shapes(&source_cfg))?;
        let zeroed = |bytes: usize, tag: Tag| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256), tag)?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (h, kv) = (cfg.hidden, cfg.kv_width());
        let fp8_workspace = if representation == MimoDraftRepresentation::Fp8Only {
            draft_fp8_scratch(library, &runtime_layout)?
        } else { None };
        let tensor = |name: &str, shape: &[usize]| checkpoint.tensor(name, shape);
        let matrix = |parts: &[(&str, usize)], cols: usize| checkpoint.matrix(parts, cols,
            representation == MimoDraftRepresentation::Fp8Only, scales, stream);
        let sharded = |parts: &[(&str, usize)], cols: usize, axis| checkpoint.matrix_shard(parts, cols,
            axis, rank, ranks, representation == MimoDraftRepresentation::Fp8Only, scales, stream);
        let layers = (0..cfg.layers).map(|l| -> Result<DraftLayer<'a>> {
            let p = format!("layers.{l}");
            let a = format!("{p}.self_attn");
            Ok(DraftLayer {
                input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                qkv: sharded(&[(&format!("{a}.q_proj.weight"), source_cfg.heads * cfg.head_dim),
                    (&format!("{a}.k_proj.weight"), source_cfg.kv_width()),
                    (&format!("{a}.v_proj.weight"), source_cfg.kv_width())], h, ShardAxis::Rows)?,
                q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
                k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
                sinks: if cfg.sinks { Some(checkpoint.tensor_shard(&format!("{a}.attention_sink_bias"),
                    &[source_cfg.heads], ShardAxis::Columns, rank, ranks)?) } else { None },
                o: sharded(&[(&format!("{a}.o_proj.weight"), h)], source_cfg.heads * cfg.head_dim, ShardAxis::Columns)?,
                gate_up: sharded(&[(&format!("{p}.mlp.gate_proj.weight"), source_cfg.intermediate),
                    (&format!("{p}.mlp.up_proj.weight"), source_cfg.intermediate)], h, ShardAxis::Rows)?,
                down: sharded(&[(&format!("{p}.mlp.down_proj.weight"), h)], source_cfg.intermediate, ShardAxis::Columns)?,
                k_ring: zeroed(slots * RING * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim), Tag::drafter_kv("mimo/dflash-rings"))?,
                v_ring: zeroed(slots * RING * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim), Tag::drafter_kv("mimo/dflash-rings"))?,
            })
        }).collect::<Result<Vec<_>>>()?;
        let taps = cfg.taps.len() * h;
        tracing::info!(?representation, own_weight_bytes = weight_layout.resident_bytes()?,
            max_load_staging = weight_layout.max_load_staging, slots, max_sequences,
            "DFlash immutable weight storage");
        let scratch = |bytes: usize| zeroed(bytes, Tag::drafter_workspace("mimo/dflash-workspace"));
        Ok(Self {
            kv_cache,
            library, projections,
            stream,
            stream_owner,
            round: std::cell::RefCell::new(crate::shared::memory::BlockingWait::new(library)),
            slots,
            max_sequences,
            fc: matrix(&[("fc.weight", h)], taps)?,
            hidden_norm: tensor("hidden_norm.weight", &[h])?,
            norm: tensor("norm.weight", &[h])?,
            layers,
            taps: scratch(TAP_ROWS * taps * 2)?,
            fused: scratch(TAP_ROWS * h * 2)?,
            fused_norm: scratch(TAP_ROWS * h * 2)?,
            context_kv: scratch(TAP_ROWS * 2 * kv * 2)?,
            context_positions: scratch(TAP_ROWS * 8)?,
            context_slots: scratch(TAP_ROWS * 4)?,
            mask_device: {
                let owner = DeviceAllocation::from_allocation(arena.view("mask_embedding", mask_row.len().max(256))?);
                DenseLoad::mount(library).operand(Operand { device: owner.buffer.device_id,
                    buffer: owner.buffer, bytes: mask_row.len(), lands: vec![(mask_source, 0)] })?;
                owner
            },
            mask_row,
            workspace: RefCell::new(None),
            trace_features: std::cell::Cell::new(false),
            policy_scores: std::cell::Cell::new(false),
            representation,
            bf16_head,
            fp8_workspace,
            timing: crate::shared::draft::timing::DraftTiming::from_env(library, "mimo-dflash")?,
            cfg,
            rank, ranks, peer: None, exchange: None,
        })
    }

    /// Land each round's top-16 logits for an open speculation trace. Only the
    /// trace records the features; tokens and every policy input are unchanged.
    pub(crate) fn set_selector_readback(&self, traced: bool, scored: bool) {
        self.trace_features.set(traced);
        self.policy_scores.set(scored);
    }

    pub(crate) fn is_split(&self) -> bool { self.peer.is_some() }

    pub(crate) fn prepare_workspace(&self) -> Result<()> {
        if self.workspace.borrow().is_none() {
            *self.workspace.borrow_mut() = Some(self.workspace(self.max_sequences)?);
        }
        if let Some(peer) = &self.peer {
            peer.device.run(|| peer.prepare_workspace())?;
        }
        Ok(())
    }

    /// Drafter weights as their `mimo.dflash.rank{r}` arena backing plus view
    /// aliases, live from DraftLoad (after target KV) through Growth.
    pub(crate) fn allocation_receipts(&self, devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>)
        -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_loader::placement::planner::{Lifetime, Phase};
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "mimo.dflash",
            Lifetime { first: Phase::DraftLoad, last: Phase::Growth },
            cuteafd_core::memory_layout::Category::Drafter, "drafter weights");
        self.weight_receipts(&mut inventory)?;
        if let Some(peer) = &self.peer { peer.get().weight_receipts(&mut inventory)?; }
        Ok(inventory.receipts)
    }

    fn weight_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        let r = self.rank;
        self.fc.allocation_receipts(inventory, &format!("rank.{r}.fc"))?;
        self.hidden_norm.inventory(inventory, &format!("rank.{r}.hidden_norm"))?;
        self.norm.inventory(inventory, &format!("rank.{r}.norm"))?;
        self.mask_device.inventory(inventory, &format!("rank.{r}.mask"))?;
        for (l, layer) in self.layers.iter().enumerate() {
            let p = format!("rank.{r}.layer.{l}");
            layer.input_norm.inventory(inventory, &format!("{p}.input_norm"))?;
            layer.post_norm.inventory(inventory, &format!("{p}.post_norm"))?;
            layer.qkv.allocation_receipts(inventory, &format!("{p}.qkv"))?;
            layer.q_norm.inventory(inventory, &format!("{p}.q_norm"))?;
            layer.k_norm.inventory(inventory, &format!("{p}.k_norm"))?;
            if let Some(sinks) = &layer.sinks { sinks.inventory(inventory, &format!("{p}.sinks"))?; }
            layer.o.allocation_receipts(inventory, &format!("{p}.o"))?;
            layer.gate_up.allocation_receipts(inventory, &format!("{p}.gate_up"))?;
            layer.down.allocation_receipts(inventory, &format!("{p}.down"))?;
        }
        Ok(())
    }

    pub fn tap_index(&self, layer: usize) -> Option<usize> {
        self.cfg.taps.iter().position(|&l| l == layer)
    }

    pub fn max_batch_sequences(&self) -> usize {
        self.max_sequences
    }

    /// Copies target layer output rows `[first, first + n)` of `hidden`
    /// ([rows, hidden] BF16) into tap rows `0..n` when `layer` is tapped.
    pub fn tap(&self, layer: usize, hidden: *const c_void, first: usize, n: usize) -> Result<()> {
        self.tap_into(&self.taps, layer, hidden, first, n)
    }

    pub fn tap_at(&self, layer: usize, hidden: *const c_void, first: usize, destination: usize, n: usize) -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        ensure!(destination + n <= TAP_ROWS, "MiMo unit tap exceeds the tail ring");
        let h = self.cfg.hidden;
        // SAFETY: caller supplies live unit rows; the destination is the unit's
        // disjoint part of the admitted whole-span tail staging.
        unsafe { self.library.glm_dflash_tap(hidden.cast::<u8>().add(first * h * 2).cast(),
            self.taps.buffer.ptr.cast::<u8>().add(destination * self.cfg.taps.len() * h * 2).cast(),
            n, h, self.cfg.taps.len() * h, index * h, self.stream) }
    }

    fn tap_into(&self, taps: &Dev<'_>, layer: usize, hidden: *const c_void, first: usize, n: usize) -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        let h = self.cfg.hidden;
        ensure!(n <= TAP_ROWS, "{n} tapped rows exceed {TAP_ROWS}");
        // SAFETY: `hidden` holds first + n rows; the tap buffer TAP_ROWS rows.
        unsafe {
            self.library.glm_dflash_tap(hidden.cast::<u8>().add(first * h * 2).cast(), taps.buffer.ptr, n, h,
                self.cfg.taps.len() * h, index * h, self.stream)
        }
    }

    /// `out` [rows, n] = `x` [rows, k] @ weight rows `first..first+n`.
    /// FP8-only storage covers every row count through the native kernel's
    /// bounded 64-row chunks. There is no resident BF16 fallback.
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn linear(&self, x: *const c_void, weight: &DraftWeight<'_>, first: usize, out: *mut c_void,
        rows: usize, k: usize, n: usize) -> Result<()> {
        let fp8 = match weight { DraftWeight::Fp8(w) => Some(w), _ => None };
        if let Some(w8) = fp8 {
            let scratch = self.fp8_workspace.as_ref().context("FP8-only DFlash scratch was not admitted")?;
            // SAFETY: the caller's contract; scratch covers each native chunk
            // of all admitted draft and TAP_ROWS context-update shapes.
            return unsafe { w8.apply(self.library, x, out, false, rows, first, n, scratch, self.stream) };
        }
        let bf16 = match weight {
            DraftWeight::Bf16(w) => w,
            DraftWeight::Fp8(_) => anyhow::bail!("FP8-only DFlash has no BF16 fallback"),
        };
        // SAFETY: the selected rows belong to the live BF16 matrix.
        unsafe { Ok(self.projections.launch(x, at(bf16, first * k * 2), out, rows, k, n, self.stream)?) }
    }

    pub fn context_rings(&self) -> Vec<(usize, CuteafdDeviceBuffer)> {
        let mut rings: Vec<_> = self.layers.iter().flat_map(|layer|
            [(self.rank, layer.k_ring.buffer), (self.rank, layer.v_ring.buffer)]).collect();
        if let Some(peer) = &self.peer { rings.extend(peer.context_rings()); }
        rings
    }

    /// Writes the context K/V of committed tapped rows.
    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        if rows.is_empty() { return Ok(()); }
        if let (Some(peer), Some(exchange)) = (&self.peer, &self.exchange) {
            // The completion acknowledgement protects peer taps from the next
            // target commit; both updates use each rank's local KV projections.
            let first = rows.iter().map(|row| row.tap_row).min().context("draft tap span")?;
            let last = rows.iter().map(|row| row.tap_row).max().context("draft tap span")?;
            ensure!(last < TAP_ROWS, "draft tap span exceeds its admitted rows");
            let row_bytes = self.cfg.taps.len() * self.cfg.hidden * 2;
            exchange.push_to(0, DIRECT, at(&self.taps, first * row_bytes),
                at(&peer.taps, first * row_bytes), (last - first + 1) * row_bytes)?;
            exchange.wait(1, DIRECT)?;
            peer.device.run(|| peer.update_from(&peer.taps, rows))?;
            exchange.signal(1, DIRECT)?;
            self.update_from(&self.taps, rows)?;
            exchange.wait(0, DIRECT)
        } else {
            self.update_from(&self.taps, rows)
        }
    }


    fn update_from(&self, taps: &Dev<'_>, rows: &[ContextRow]) -> Result<()> {
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
        let (h, kv, c) = (self.cfg.hidden, self.cfg.kv_width(), &self.cfg);
        let width = c.taps.len() * h;
        let s = self.stream;
        let context_span = self.timing.begin("context", None, n, s)?;
        // SAFETY: every buffer holds TAP_ROWS rows of its width; the stream orders the chain.
        unsafe {
            self.linear(at(taps, first * width * 2), &self.fc, 0,
                self.fused.buffer.ptr, n, width, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, c.eps, s)?;
            for layer in &self.layers {
                let q_rows = c.heads * c.head_dim;
                self.linear(self.fused_norm.buffer.ptr, &layer.qkv, q_rows,
                    self.context_kv.buffer.ptr, n, h, 2 * kv)?;
                self.library.mimo_dflash_qk_rope(self.context_kv.buffer.ptr, layer.q_norm.buffer.ptr,
                    layer.k_norm.buffer.ptr, self.context_positions.buffer.ptr, self.context_slots.buffer.ptr,
                    std::ptr::null_mut(), layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, n, 0, c.kv_heads,
                    c.rope_dim, c.theta, c.eps, c.v_scale, self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
            }
        }
        self.timing.end(context_span, s)?;
        Ok(())
    }

    fn put(&self, dev: &Dev<'_>, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        // SAFETY: every reader of the drafter's tables, inputs and taps runs on its
        // stream outside capture; the ordered copy follows them and keeps its source.
        unsafe { self.library.copy_h2d_ordered(dev.buffer, bytes, self.stream) }
    }

    fn workspace(&self, sequences: usize) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter/workspace");
        let c = &self.cfg;
        let rows = sequences * c.block;
        let drafted = sequences * c.drafts();
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256), Tag::drafter_workspace("mimo/dflash-workspace"));
        Ok(Workspace {
            sequences,
            h: alloc(rows * c.hidden * 2)?,
            n: alloc(rows * c.hidden * 2)?,
            qkv: alloc(rows * c.qkv_width() * 2)?,
            q: alloc(rows * c.heads * c.head_dim * 2)?,
            k: alloc(rows * c.kv_width() * 2)?,
            v: alloc(rows * c.kv_width() * 2)?,
            attn: alloc(rows * c.heads * c.head_dim * 2)?,
            delta: alloc(rows * c.hidden * 2)?,
            reduced: (self.ranks == 2).then(|| alloc(rows * c.hidden * 2)).transpose()?,
            gate_up: alloc(rows * 2 * c.intermediate * 2)?,
            act: alloc(rows * c.intermediate * 2)?,
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            ids: alloc(rows * 4)?,
            attention_workspace: alloc(self.library.mimo_dflash_attention_workspace(sequences, c.heads, c.kv_heads,
                c.block, RING + c.block)?)?,
            output: (self.rank == 0).then(|| -> Result<_> {
                let head_workspace = alloc(VOCABULARY_HEAD_WORKSPACE)?;
                Ok(DraftOutput {
                    logits: alloc(rows * c.vocab * 4)?,
                    unary: alloc(drafted * 16 * 4)?,
                    candidates: alloc(drafted * 16 * 4)?,
                    landing: crate::shared::memory::HostAllocation::new(self.library, 2 * drafted * 16 * 4,
                        Tag::staging("mimo/dflash-landing"))?,
                    topk_workspace: alloc(self.library.glm_dflash_topk_workspace(drafted)?)?,
                    // SAFETY: the workspace outlives this head's plan and uses the lead device.
                    head: self.bf16_head.then(|| {
                        // SAFETY: private stable workspace and the borrowed BF16 target geometry.
                        unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, c.hidden as u32,
                            rows as u32, c.vocab as u32, cuteafd_ffi::programs::HeadMode::Gemm) }
                    }).transpose()?,
                    _head_workspace: head_workspace,
                })
            }).transpose()?,
        })
    }

    /// Drafts `block - 1` tokens after each sequence's anchor, its rows
    /// gathered on the device from the target's embedding table (the mask
    /// rows from the trained mask row); `head` is the target's vocabulary head
    /// in its immutable BF16 or target row-major FP8 representation.
    pub fn draft(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: BorrowedHead<'_, '_>)
        -> Result<Vec<Draft>> {
        ensure!(head.hidden == self.cfg.hidden && head.vocab == self.cfg.vocab,
            "DFlash borrowed target head geometry differs from its checkpoint");
        self.draft_from(sequences, Anchors::Tokens(embedding), |plan, x, out, rows|
            head.launch_draft(plan, x, out, rows, self.stream))
    }

    /// [`Self::draft`] from the anchors' embedding rows (oracles and replays).
    pub fn draft_rows(&self, sequences: &[DraftSeq], anchor_rows: &[u8], head: *const c_void) -> Result<Vec<Draft>> {
        self.draft_from(sequences, Anchors::Rows(anchor_rows), |plan, x, out, rows| {
            // SAFETY: ReplayDrafter's caller retains a complete BF16 target
            // head. FP8-only targets reject that diagnostic before loading.
            unsafe { plan.context("BF16 diagnostic head was not prepared")?.launch(
                x.ptr.cast(), head.cast(), out.ptr.cast(), rows as u32, self.stream) }
        })
    }

    fn draft_from(&self, sequences: &[DraftSeq], anchors: Anchors<'_>,
        head: impl FnOnce(Option<&VocabularyHead<'_>>, CuteafdDeviceBuffer, CuteafdDeviceBuffer, usize) -> Result<()>) -> Result<Vec<Draft>> {
        let c = &self.cfg;
        let (s_count, block, h) = (sequences.len(), c.block, c.hidden);
        let anchors_fit = match anchors {
            Anchors::Rows(rows) => rows.len() == s_count * h * 2,
            Anchors::Tokens(_) => true,
        };
        ensure!(s_count > 0 && s_count <= self.max_sequences && anchors_fit, "draft step of {s_count} sequences");
        let rows = s_count * block;
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("draft workspace must be prepared before use")?;
        ensure!(w.sequences >= s_count, "draft workspace exceeds prepared sequence capacity");
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
        match anchors {
            Anchors::Rows(anchor_rows) => {
                let mut embed = Vec::with_capacity(rows * h * 2);
                for i in 0..s_count {
                    embed.extend_from_slice(&anchor_rows[i * h * 2..(i + 1) * h * 2]);
                    for _ in 1..block {
                        embed.extend_from_slice(&self.mask_row);
                    }
                }
                self.put(&w.h, &embed)?;
            }
            Anchors::Tokens(embedding) => {
                // Each block: the anchor, then the mask rows (an id past the table takes the mask row).
                let ids: Vec<u32> = sequences.iter()
                    .flat_map(|seq| std::iter::once(seq.anchor).chain(std::iter::repeat_n(MASK_ID, block - 1))).collect();
                embedding.check(&sequences.iter().map(|seq| seq.anchor).collect::<Vec<_>>())?;
                self.put(&w.ids, bytes_of(&ids))?;
                // SAFETY: the ids are up (ordered on this stream); `h` holds the block rows; the
                // mask row is a live [hidden] BF16 buffer.
                unsafe { embedding.embed_device_ids(w.ids.buffer, None, rows, 1,
                    Some((self.mask_device.buffer.ptr.cast_const(), &self.mask_row)), w.h.buffer, self.stream)? };
            }
        }
        self.put(&w.positions, bytes_of(&positions))?;
        self.put(&w.tables, bytes_of(&tables))?;
        let (s, l) = (self.stream, self.library);
        let forward_span = self.timing.begin("forward", None, rows, s)?;
        if let (Some(peer), Some(exchange)) = (&self.peer, &self.exchange) {
            let peer_slot = peer.workspace.borrow();
            let pw = peer_slot.as_ref().context("peer draft workspace must be prepared before use")?;
            ensure!(pw.sequences >= s_count, "peer draft workspace exceeds prepared sequence capacity");
            peer.device.run(|| {
                peer.put(&pw.positions, bytes_of(&positions))?;
                peer.put(&pw.tables, bytes_of(&tables))
            })?;
            exchange.push_to(0, DIRECT, w.h.buffer.ptr, pw.h.buffer.ptr, rows * h * 2)?;
            exchange.wait(1, DIRECT)?;
            self.begin_forward(w, rows)?;
            peer.device.run(|| peer.begin_forward(pw, rows))?;
            for index in 0..self.layers.len() {
                self.attention_partial(w, index, s_count, max_keys)?;
                peer.device.run(|| peer.attention_partial(pw, index, s_count, max_keys))?;
                Self::reduce(exchange, [w, pw], rows * h)?;
                self.ffn_partial(w, index, rows)?;
                peer.device.run(|| peer.ffn_partial(pw, index, rows))?;
                Self::reduce(exchange, [w, pw], rows * h)?;
                self.finish_layer(w, index, rows)?;
                peer.device.run(|| peer.finish_layer(pw, index, rows))?;
            }
            // Fence peer residual/norm readers before a subsequent target or
            // context-update operation reuses any of the shared exchange slots.
            exchange.signal(1, DIRECT)?;
            exchange.wait(0, DIRECT)?;
        } else {
            self.begin_forward(w, rows)?;
            for index in 0..self.layers.len() {
                let layer_span = self.timing.begin("layer", Some(index), rows, s)?;
                self.attention_partial(w, index, s_count, max_keys)?;
                self.ffn_partial(w, index, rows)?;
                self.finish_layer(w, index, rows)?;
                self.timing.end(layer_span, s)?;
            }
        }
        self.timing.end(forward_span, s)?;
        let output = w.output.as_ref().context("only the lead drafter can sample")?;
        // SAFETY: only rank0 owns the borrowed target head, and its full hidden
        // rows are reduced/normalized before the stream reaches this launch.
        unsafe {
            let head_span = self.timing.begin("head_topk", None, rows, s)?;
            head(output.head.as_ref(), w.n.buffer, output.logits.buffer, rows)?;
            l.glm_dflash_topk(output.logits.buffer.ptr, output.unary.buffer.ptr, output.candidates.buffer.ptr,
                output.topk_workspace.buffer.ptr, s_count, block, c.drafts(), c.vocab, s)?;
            self.timing.end(head_span, s)?;
        }
        let drafted = s_count * c.drafts();
        let landing = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: both halves stay inside the 2 x drafted x 64-byte landing.
            ptr: unsafe { output.landing.buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes: drafted * 64, ..output.landing.buffer };
        // The selector's top-16 logits already exist on device. Land them once
        // to compute the policy probability and optional trace features on host.
        let traced = self.trace_features.get();
        let scored = self.policy_scores.get();
        // SAFETY: the device buffers hold `drafted` 64-byte rows; the landing
        // is pinned, owned by the workspace, and read only after the wait.
        unsafe {
            if traced || scored {
                let span = self.timing.begin("candidate_logits_d2h", None, drafted, s)?;
                l.copy_d2h_host_buffer_async(landing(0), CuteafdDeviceBuffer { bytes: drafted * 64, ..output.unary.buffer },
                    drafted * 64, s)?;
                self.timing.end(span, s)?;
            }
            l.copy_d2h_host_buffer_async(landing(drafted * 64),
                CuteafdDeviceBuffer { bytes: drafted * 64, ..output.candidates.buffer }, drafted * 64, s)?;
            self.round.borrow_mut().wait(s)?;
        }
        self.timing.publish_completed()?;
        self.timing.dump_head_input(w.n.buffer, rows, h, s)?;
        let (unary, candidates) = output.landing.bytes()[..2 * drafted * 64].split_at(drafted * 64);
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let drafts = (0..s_count).map(|i| {
            let rows = (0..c.drafts()).map(|j| i * c.drafts() + j);
            let logits = |row| std::array::from_fn::<_, 16, _>(|k| f32::from_bits(word(unary, row * 16 + k)));
            let features: Vec<_> = if traced { rows.clone().map(|r| features(&logits(r))).collect() } else { Vec::new() };
            let confidence = if !scored { Vec::new() } else if traced { features.iter().map(|f| f[1]).collect() }
                else { rows.clone().map(|r| candidate_probability(&logits(r))).collect() };
            Draft { tokens: rows.map(|r| word(candidates, r * 16)).collect(), features, confidence }
        }).collect();
        Ok(drafts)
    }

    fn begin_forward(&self, w: &Workspace<'_>, rows: usize) -> Result<()> {
        // SAFETY: local full-hidden activations and replicated norm are live on this stream.
        unsafe { self.library.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr,
            w.n.buffer.ptr, rows, self.cfg.hidden, self.cfg.eps, self.stream) }
    }

    fn attention_partial(&self, w: &Workspace<'_>, index: usize, sequences: usize, max_keys: usize) -> Result<()> {
        let (c, l, s) = (&self.cfg, self.library, self.stream);
        let rows = sequences * c.block;
        let layer = &self.layers[index];
        // SAFETY: rank-local Q/K/V, rings and O columns use this rank's geometry;
        // all producers and consumers are ordered on its target stream.
        unsafe {
            self.linear(w.n.buffer.ptr, &layer.qkv, 0, w.qkv.buffer.ptr, rows, c.hidden, c.qkv_width())?;
            l.mimo_dflash_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                w.positions.buffer.ptr, std::ptr::null(), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr,
                rows, c.heads, c.kv_heads, c.rope_dim, c.theta, c.eps, c.v_scale, false, s)?;
            l.mimo_dflash_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, sequences * 4),
                at(&w.tables, 2 * sequences * 4), layer.sinks.as_ref().map_or(std::ptr::null(), |v| v.buffer.ptr),
                w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, sequences, c.block, c.heads, c.kv_heads,
                RING, max_keys, c.window, 1.0 / (c.head_dim as f32).sqrt(),
                self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
            self.linear(w.attn.buffer.ptr, &layer.o, 0, w.delta.buffer.ptr, rows, c.heads * c.head_dim, c.hidden)
        }
    }

    fn delta<'w>(&self, w: &'w Workspace<'_>) -> *mut c_void {
        w.reduced.as_ref().map_or(w.delta.buffer.ptr, |r| r.buffer.ptr)
    }

    fn ffn_partial(&self, w: &Workspace<'_>, index: usize, rows: usize) -> Result<()> {
        let (c, l, s) = (&self.cfg, self.library, self.stream);
        let layer = &self.layers[index];
        // SAFETY: reduced full-hidden delta precedes this replicated residual;
        // the local gate/up rows and down columns form one disjoint FFN partial.
        unsafe {
            l.mimo_dflash_add_norm(w.h.buffer.ptr, self.delta(w), layer.post_norm.buffer.ptr, w.n.buffer.ptr,
                rows, c.hidden, c.eps, s)?;
            self.linear(w.n.buffer.ptr, &layer.gate_up, 0, w.gate_up.buffer.ptr, rows, c.hidden, 2 * c.intermediate)?;
            l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, c.intermediate, s)?;
            self.linear(w.act.buffer.ptr, &layer.down, 0, w.delta.buffer.ptr, rows, c.intermediate, c.hidden)
        }
    }

    fn finish_layer(&self, w: &Workspace<'_>, index: usize, rows: usize) -> Result<()> {
        let next = self.layers.get(index + 1).map_or(self.norm.buffer.ptr, |n| n.input_norm.buffer.ptr);
        // SAFETY: full-hidden reduced FFN delta and replicated residual/norm have matching extents.
        unsafe { self.library.mimo_dflash_add_norm(w.h.buffer.ptr, self.delta(w), next,
            w.n.buffer.ptr, rows, self.cfg.hidden, self.cfg.eps, self.stream) }
    }

    fn reduce(exchange: &PeerExchange<'_>, work: [&Workspace<'_>; 2], elements: usize) -> Result<()> {
        // Draft blocks can exceed the target's 64-row decode slot. Reuse the
        // existing bounded slot in windows, acknowledging both readers each time.
        let window = exchange.slot_bytes() / 2;
        ensure!(window > 0, "empty TP2 draft exchange window");
        for first in (0..elements).step_by(window) {
            let count = window.min(elements - first);
            for (rank, w) in work.iter().enumerate() {
                exchange.push(rank, 0, at(&w.delta, first * 2), count * 2)?;
            }
            for (rank, w) in work.iter().enumerate() {
                exchange.wait(rank, 0)?;
                exchange.add(rank, at(&w.delta, first * 2), exchange.recv(rank, 0)?,
                    at(w.reduced.as_ref().context("TP2 draft reduction was not admitted")?, first * 2), count)?;
            }
            // Publication alone does not authorize receive-slot reuse.
            for rank in 0..2 { exchange.signal(rank, DIRECT)?; }
            for rank in 0..2 { exchange.wait(rank, DIRECT)?; }
        }
        Ok(())
    }

    /// The last draft step's final-norm rows [sequences * block, hidden] BF16.
    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("no draft step ran")?;
        let bytes = sequences * self.cfg.block * self.cfg.hidden * 2;
        let mut out = vec![0u8; bytes];
        self.stream_owner
            .as_ref()
            .context("MiMo drafter lead stream missing")?
            .copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..w.n.buffer })?;
        Ok(out)
    }

    /// Copies BF16 tap rows (host, [n, taps * hidden]) into the tap buffer.
    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        self.put(&self.taps, rows)
    }
}

fn candidate_probability(top: &[f32]) -> f32 {
    let best = top[0];
    1.0 / top.iter().map(|&u| (u - best).exp()).sum::<f32>()
}

/// [margin, best probability, entropy, rank] of one drafted row from its top-16
/// logits (best first), for the speculation trace. Copied unchanged from
/// the pre-#2443 MiMo drafter (206d36a323 `mimo_v2/dflash.rs` `features`); rank
/// is always 0 because MiMo drafts the top candidate.
pub(crate) fn features(top: &[f32]) -> [f32; 4] {
    let best = top[0];
    let mass: Vec<f32> = top.iter().map(|&u| (u - best).exp()).collect();
    let total: f32 = mass.iter().sum();
    let entropy = total.ln() - top.iter().zip(&mass).map(|(&u, &m)| m * (u - best)).sum::<f32>() / total;
    [best - top.get(1).copied().unwrap_or(f32::NEG_INFINITY), 1.0 / total, entropy.max(0.0), 0.0]
}

impl crate::shared::speculation::dflash::ReplayDrafter for MimoDrafter<'_> {
    fn block(&self) -> usize {
        self.cfg.block
    }

    fn sequences(&self) -> usize {
        self.max_sequences.min(self.slots)
    }

    fn resident_modes(&self) -> Vec<crate::shared::speculation::dflash::ReplayMode> {
        use crate::shared::speculation::dflash::ReplayMode;
        let name = match self.representation {
            MimoDraftRepresentation::Bf16Only => "BF16",
            MimoDraftRepresentation::Fp8Only => "FP8",
        };
        vec![ReplayMode { name, legacy_fp8: None }]
    }

    fn context(&self, taps: &[u8], first: usize) -> Result<()> {
        let n = taps.len() / (self.cfg.taps.len() * self.cfg.hidden * 2);
        self.put_taps(taps)?;
        self.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r }).collect::<Vec<_>>())
    }

    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>> {
        let seqs: Vec<DraftSeq> = seqs.iter().map(|&(slot, anchor, position)| DraftSeq { slot, anchor, position, valid_from: 0 }).collect();
        Ok(self.draft_rows(&seqs, anchor_rows, head)?.into_iter().map(|d| d.tokens).collect())
    }

    fn tap_rows(&self) -> usize {
        TAP_ROWS
    }

    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        MimoDrafter::last_hidden(self, sequences)
    }
}
