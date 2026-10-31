//! Kimi K3 coordinator operands from the installed K6 checkpoint
//! (CUTEAFD_PUSH_KIMI) on real GPUs (CUTEAFD_NATIVE_LIB): both global ranks
//! (embedding; native head, output mix and norm), the dense KDA layer 0, the
//! first routed KDA layer and the first MLA layer in each placement it admits,
//! on both coordinator GPUs. Digests are pinned to the loader as it was before
//! push records (CUTEAFD_KIMI_GOLDEN overrides the golden path;
//! CUTEAFD_KIMI_GOLDEN_WRITE=1 records it, to CUTEAFD_KIMI_GOLDEN_OUT when set).
use super::*;
use crate::shared::memory::device::Device;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::{placement::FfnMode, plan::Checkpoint};
use sha2::{Digest, Sha256};
use std::os::unix::fs::FileExt;

pub(crate) type Digests = BTreeMap<String, String>;

fn digest(library: &NativeLibrary, allocation: &Allocation<'_>) -> Result<String> {
    let mut bytes = vec![0u8; allocation.buffer.bytes];
    allocation.device.run(|| library.copy_d2h(&mut bytes, allocation.buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_KIMI").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_KIMI_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/kimi_k3/coordinator_golden.json"))
}

pub(crate) fn check(actual: &Digests) -> Result<()> {
    if std::env::var_os("CUTEAFD_KIMI_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_KIMI_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
        std::fs::write(&out, serde_json::to_vec_pretty(actual)?)?;
        eprintln!("wrote {} operand digests to {}", actual.len(), out.display());
        return Ok(());
    }
    let golden: Digests = serde_json::from_slice(&std::fs::read(path())?)?;
    ensure!(actual.keys().eq(golden.keys()), "operand inventory changed: {} actual, {} golden",
        actual.len(), golden.len());
    let differ: Vec<_> = actual.iter().filter(|(k, v)| golden.get(*k) != Some(v)).map(|(k, _)| k.clone()).collect();
    ensure!(differ.is_empty(), "{} operands differ from the golden, first {:?}", differ.len(),
        &differ[..differ.len().min(8)]);
    eprintln!("PASS {} operand digests match the golden", actual.len());
    Ok(())
}

/// The golden's layers and their placements: the dense KDA layer 0, the first
/// routed KDA layer, and the first MLA layer split by heads and whole on
/// either GPU.
pub(crate) fn golden_layers(cfg: &KimiK3Config) -> Result<Vec<(usize, LayerMode)>> {
    let kda = (0..cfg.layers).find(|&l| cfg.attention[l] == KimiAttention::Kda && !cfg.dense[l])
        .context("no routed KDA layer")?;
    let mla = (0..cfg.layers).find(|&l| cfg.attention[l] == KimiAttention::Mla).context("no MLA layer")?;
    ensure!(cfg.attention[0] == KimiAttention::Kda && cfg.dense[0], "Kimi layer 0 is a dense KDA layer");
    Ok(vec![(0, LayerMode::HeadSplit), (kda, LayerMode::HeadSplit), (mla, LayerMode::HeadSplit),
        (mla, LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }), (mla, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split })])
}

fn mode_tag(mode: LayerMode) -> String {
    match mode {
        LayerMode::HeadSplit => "heads".into(),
        LayerMode::Whole { gpu, .. } => format!("whole{gpu}"),
        other => format!("{other:?}"),
    }
}

fn core_digests(library: &NativeLibrary, prefix: &str, core: &CoreWeights<'_>, out: &mut Digests) -> Result<()> {
    for (part, allocation) in [("payload", &core.payload), ("suh", &core.suh), ("svh", &core.svh)] {
        out.insert(format!("{prefix}.{part}"), digest(library, allocation)?);
    }
    Ok(())
}

pub(crate) fn global_digests(library: &NativeLibrary, rank: usize, weights: &GlobalWeights<'_>,
    out: &mut Digests) -> Result<()> {
    for (name, allocation) in &weights.operands {
        out.insert(format!("global.{rank}.{name}"), digest(library, allocation)?);
    }
    if let Some(head) = &weights.core_head { core_digests(library, &format!("global.{rank}.head"), head, out)?; }
    Ok(())
}

pub(crate) fn layer_digests(library: &NativeLibrary, prefix: &str, weights: &LayerWeights<'_>,
    out: &mut Digests) -> Result<()> {
    for (name, allocation) in &weights.operands {
        out.insert(format!("{prefix}.{name}"), digest(library, allocation)?);
    }
    for (name, core) in &weights.core { core_digests(library, &format!("{prefix}.{name}"), core, out)?; }
    Ok(())
}

pub(crate) struct Installed {
    pub checkpoint: Checkpoint,
    pub cfg: KimiK3Config,
    pub core: CoreInventory,
}

pub(crate) fn installed(snapshot: &std::path::Path) -> Result<Installed> {
    let checkpoint = Checkpoint::coordinator(snapshot, false, false)?;
    let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
    let core = cuteafd_loader::plan::families::kimi::core_inventory(&checkpoint)?;
    ensure!(!core.projections.is_empty(), "the golden checkpoint carries the K6 core");
    Ok(Installed { checkpoint, cfg, core })
}

/// Every golden operand through the coordinator loader, each owner on its GPU.
pub(crate) fn mount(library: &NativeLibrary, model: &Installed, out: &mut Digests) -> Result<()> {
    let devices = [0, 1].map(|id| Device { library, id });
    let read = |name: &str| read_tensor(&model.checkpoint, name);
    let globals = GlobalLayout::pair_with_core(&model.cfg, &model.core)?;
    for (rank, layout) in globals.iter().enumerate() {
        let weights = layout.upload(devices[rank], layout.pack(&model.cfg, read)?)?;
        global_digests(library, rank, &weights, out)?;
    }
    for (layer, mode) in golden_layers(&model.cfg)? {
        let layouts = LayerLayout::pair_with_core(&model.cfg, layer, mode, &model.core)?;
        let packed = PackedLayer::pair_with_core(&model.cfg, layer, mode, &model.core, read)?;
        for ((rank, packed), layout) in packed.into_iter().enumerate().zip(&layouts) {
            let weights = LayerWeights::upload(devices[rank], packed, layout, layout.bytes()?)?;
            layer_digests(library, &format!("layers.{layer}.{}.{rank}", mode_tag(mode)), &weights, out)?;
        }
    }
    Ok(())
}

fn peers(library: &NativeLibrary) -> Result<()> {
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_KIMI, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let model = installed(&snapshot)?;
    let library = library()?;
    peers(&library)?;
    let mut actual = Digests::new();
    mount(&library, &model, &mut actual)?;
    check(&actual)
}

/// Every golden operand through the shared records (`records::KimiRecords`)
/// on `load`: filled through the mount at once, or recorded for push ranks.
pub(crate) fn recorded<'a>(library: &'a NativeLibrary, model: &Installed,
    load: &crate::shared::experts::rtx::dense::DenseLoad<'a>)
    -> Result<(Vec<(usize, GlobalWeights<'a>)>, Vec<(String, usize, LayerWeights<'a>)>)> {
    let devices = [0, 1].map(|id| Device { library, id });
    let records = records::KimiRecords { checkpoint: &model.checkpoint, cfg: &model.cfg, core: &model.core, dense: load,
        streams: [(0, std::ptr::null_mut()), (1, std::ptr::null_mut())] };
    let globals = GlobalLayout::pair_with_core(&model.cfg, &model.core)?;
    let mut global = Vec::new();
    for (rank, layout) in globals.iter().enumerate() { global.push((rank, records.global(devices[rank], layout)?)); }
    let mut layers = Vec::new();
    for (layer, mode) in golden_layers(&model.cfg)? {
        let layouts = LayerLayout::pair_with_core(&model.cfg, layer, mode, &model.core)?;
        for (rank, weights) in records.layer(devices, layer, &layouts)?.into_iter().enumerate() {
            layers.push((format!("layers.{layer}.{}.{rank}", mode_tag(mode)), rank, weights));
        }
    }
    Ok((global, layers))
}

fn recorded_digests(library: &NativeLibrary, global: &[(usize, GlobalWeights<'_>)],
    layers: &[(String, usize, LayerWeights<'_>)], out: &mut Digests) -> Result<()> {
    for (rank, weights) in global { global_digests(library, *rank, weights, out)?; }
    for (prefix, _, weights) in layers { layer_digests(library, prefix, weights, out)?; }
    Ok(())
}

/// The same operands through the shared records, filled by the mount.
#[test]
#[ignore = "requires CUTEAFD_PUSH_KIMI, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_recorded_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let model = installed(&snapshot)?;
    let library = library()?;
    peers(&library)?;
    let load = crate::shared::experts::rtx::dense::DenseLoad::mount(&library);
    let (global, layers) = recorded(&library, &model, &load)?;
    load.release();
    let mut actual = Digests::new();
    recorded_digests(&library, &global, &layers, &mut actual)?;
    check(&actual)
}

/// The same records pushed through a ring into each GPU's rank in file
/// order (trellis tiles, products and absorption on the GPU): every operand
/// matches the golden and each rank's staging stays bounded.
#[test]
#[ignore = "requires CUTEAFD_PUSH_KIMI (a SparkNest namespace), CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_pushed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    use crate::shared::experts::rtx::dense::{DenseLoad, Records, STEP_BYTES};
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let model = installed(&snapshot)?;
    let library = library()?;
    peers(&library)?;
    let (root, _) = cuteafd_loader::push_load::namespace(&snapshot)?;
    let sockets = tempfile::tempdir()?;
    let load = DenseLoad::push(&library);
    let (global, layers) = recorded(&library, &model, &load)?;
    let records = load.sink.as_ref().context("push sink")?.take_all();
    let (bytes, operands, staged, steps, largest, finishes) = records.summary();
    eprintln!("class2 {bytes} B in {operands} operands; class3 {staged} B in {steps} steps (largest {largest} B); \
        {finishes} finishes");
    ensure!(largest <= STEP_BYTES, "a step stages {largest} bytes");
    for gpu in 0..2 {
        let mine = Records { operands: records.operands.iter().filter(|o| o.device == gpu).cloned().collect(),
            steps: records.steps.iter().filter(|s| s.device == gpu).cloned().collect(), finishes: Vec::new() };
        crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &library, id: gpu },
            None, &root, sockets.path(), &mine, None, 0, gpu as usize)?;
    }
    for (staged, pinned) in crate::shared::experts::rtx::dense::testing::PEAKS.with(|p| p.take()) {
        eprintln!("rank staging peak {staged} B in use, {pinned} B pinned");
        ensure!(staged <= STEP_BYTES as u64 * 8 && pinned <= STEP_BYTES as u64 * 8,
            "rank staging peaked at {staged} B in use, {pinned} B pinned");
    }
    let mut actual = Digests::new();
    recorded_digests(&library, &global, &layers, &mut actual)?;
    check(&actual)
}

/// A small Kimi checkpoint (H256, K6 core on every core projection and the
/// head, layers 0..4 covering dense KDA, routed KDA and MLA) with patterned
/// payloads, as a SparkNest namespace. Returns the scaled config and the core
/// inventory from the shards' own headers.
fn small_checkpoint(root: &std::path::Path) -> Result<(KimiK3Config, CoreInventory)> {
    use cuteafd_loader::{families::kimi_k3::core::{split, CoreProjection},
        formats::exl3_storage::{exl3_module, Exl3Codebook}, plan::testing::{t, Tensor}};
    let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
    cfg.hidden = 256; cfg.heads = 4; cfg.kda_heads = 16; cfg.kda_head_dim = 128; cfg.q_lora_rank = 128;
    cfg.kv_lora_rank = 64; cfg.qk_nope_dim = 64; cfg.qk_rope_dim = 64; cfg.v_head_dim = 128; cfg.latent = 128;
    cfg.experts = 4; cfg.moe_intermediate = 128; cfg.shared_experts = 2; cfg.dense_intermediate = 256;
    cfg.vocab_size = 384; cfg.layers = 4;
    cfg.attention.truncate(4); cfg.dense.truncate(4);
    ensure!(cfg.attention[..3].iter().all(|a| *a == KimiAttention::Kda) && cfg.attention[3] == KimiAttention::Mla
        && cfg.dense == [true, false, false, false], "fixture layer kinds");
    let (h, d) = (cfg.hidden, cfg.kda_heads * cfg.kda_head_dim);
    let p = "language_model.model";
    let mut tensors: Vec<Tensor> = vec![t(format!("{p}.embed_tokens.weight"), "BF16", &[cfg.vocab_size, h]),
        t(format!("{p}.norm.weight"), "BF16", &[h]), t(format!("{p}.output_attn_res_norm.weight"), "BF16", &[h]),
        t(format!("{p}.output_attn_res_proj.weight"), "BF16", &[1, h])];
    let mut core = Vec::<(String, [usize; 2], usize, Exl3Codebook)>::new();
    let mut native = |tensors: &mut Vec<Tensor>, stem: String, n: usize, k: usize, stored: usize, book: Exl3Codebook| {
        tensors.push(t(format!("{stem}.trellis"), "I16", &[k / 16, stored / 16, 96]));
        tensors.push(t(format!("{stem}.suh"), "F16", &[k]));
        tensors.push(t(format!("{stem}.svh"), "F16", &[stored]));
        tensors.push(t(format!("{stem}.{}", book.marker()), "I32", &[]));
        core.push((stem, [n, k], stored, book));
    };
    native(&mut tensors, "language_model.lm_head".into(), cfg.vocab_size, h, cfg.vocab_size, Exl3Codebook::Mul1);
    for layer in 0..cfg.layers {
        let b = format!("{p}.layers.{layer}");
        for n in ["input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"] {
            tensors.push(t(format!("{b}.{n}.weight"), "BF16", &[h]));
        }
        for n in ["self_attention_res_proj", "mlp_res_proj"] { tensors.push(t(format!("{b}.{n}.weight"), "BF16", &[1, h])); }
        let a = format!("{b}.self_attn");
        let book = if layer % 2 == 0 { Exl3Codebook::Mcg } else { Exl3Codebook::Mul1 };
        if cfg.attention[layer] == KimiAttention::Kda {
            for proj in ["q", "k", "v", "g"] { native(&mut tensors, format!("{a}.{proj}_proj"), d, h, d, book); }
            native(&mut tensors, format!("{a}.o_proj"), h, d, h, book);
            for proj in ["q", "k", "v"] { tensors.push(t(format!("{a}.{proj}_conv1d.weight"), "F32", &[d, 1, 4])); }
            tensors.push(t(format!("{a}.f_a_proj.weight"), "BF16", &[128, h]));
            tensors.push(t(format!("{a}.f_b_proj.weight"), "BF16", &[d, 128]));
            tensors.push(t(format!("{a}.b_proj.weight"), "BF16", &[cfg.kda_heads, h]));
            tensors.push(t(format!("{a}.A_log"), "F32", &[32]));
            tensors.push(t(format!("{a}.dt_bias"), "F32", &[d]));
            tensors.push(t(format!("{a}.o_norm.weight"), "F32", &[128]));
        } else {
            let v = cfg.heads * cfg.v_head_dim;
            native(&mut tensors, format!("{a}.q_a_proj"), cfg.q_lora_rank, h, cfg.q_lora_rank, book);
            let kv_a = cfg.kv_lora_rank + cfg.qk_rope_dim;
            native(&mut tensors, format!("{a}.kv_a_proj_with_mqa"), kv_a, h, kv_a.next_multiple_of(128), book);
            native(&mut tensors, format!("{a}.q_b_proj"), cfg.heads * (cfg.qk_nope_dim + cfg.qk_rope_dim),
                cfg.q_lora_rank, cfg.heads * (cfg.qk_nope_dim + cfg.qk_rope_dim), book);
            native(&mut tensors, format!("{a}.g_proj"), v, h, v, book);
            native(&mut tensors, format!("{a}.o_proj"), h, v, h, book);
            tensors.push(t(format!("{a}.q_a_layernorm.weight"), "BF16", &[cfg.q_lora_rank]));
            tensors.push(t(format!("{a}.kv_a_layernorm.weight"), "BF16", &[cfg.kv_lora_rank]));
            tensors.push(t(format!("{a}.kv_b_proj.weight"), "BF16", &[cfg.heads * (cfg.qk_nope_dim + cfg.v_head_dim),
                cfg.kv_lora_rank]));
        }
        if cfg.dense[layer] {
            for (proj, n, k) in [("gate", cfg.dense_intermediate, h), ("up", cfg.dense_intermediate, h),
                ("down", h, cfg.dense_intermediate)] { tensors.push(t(format!("{b}.mlp.{proj}_proj.weight"), "BF16", &[n, k])); }
            continue;
        }
        let m = format!("{b}.block_sparse_moe");
        tensors.push(t(format!("{m}.gate.weight"), "BF16", &[cfg.experts, h]));
        tensors.push(t(format!("{m}.gate.e_score_correction_bias"), "F32", &[cfg.experts]));
        tensors.push(t(format!("{m}.routed_expert_norm.weight"), "BF16", &[cfg.latent]));
        native(&mut tensors, format!("{m}.routed_expert_down_proj"), cfg.latent, h, cfg.latent, book);
        native(&mut tensors, format!("{m}.routed_expert_up_proj"), h, cfg.latent, h, book);
        let width = cfg.moe_intermediate * cfg.shared_experts;
        for (proj, n, k) in [("gate", width, h), ("up", width, h), ("down", h, width)] {
            native(&mut tensors, format!("{m}.shared_experts.{proj}_proj"), n, k, n, book);
        }
    }
    cuteafd_loader::plan::testing::write_snapshot(root, &serde_json::json!({}), &tensors, None);
    // Pattern every payload; markers carry their codebook, A_log keeps finite
    // live heads and +0 padding.
    let checkpoint = Checkpoint::open(root)?;
    for tensor in &checkpoint.tensors {
        let meta = &tensor.meta;
        let mut bytes: Vec<u8> = (0..meta.byte_length).map(|i| (i as u8).wrapping_mul(29) ^ (meta.name.len() as u8)).collect();
        if let Some(book) = [Exl3Codebook::Mcg, Exl3Codebook::Mul1].into_iter().find(|b| meta.name.ends_with(&format!(".{}", b.marker()))) {
            bytes = book.multiplier().to_le_bytes().to_vec();
        } else if meta.name.ends_with(".A_log") {
            bytes = (0..32).flat_map(|i| if i < cfg.kda_heads { (i as f32 * 0.25 - 1.0).to_le_bytes() } else { 0f32.to_le_bytes() }).collect();
        }
        std::fs::File::options().write(true).open(root.join(&tensor.shard))?.write_all_at(&bytes, meta.byte_offset)?;
    }
    let mut inventory = CoreInventory::default();
    for (stem, shape, stored, book) in core {
        let members = ["trellis", "suh", "svh", book.marker()].map(|s| checkpoint.tensors.iter()
            .find(|t| t.meta.name == format!("{stem}.{s}")).map(|t| (s, t.meta.clone())));
        let members: BTreeMap<_, _> = members.into_iter().flatten().collect();
        let module = exl3_module(&stem, &members.keys().copied().collect::<Vec<_>>(), |s| members.get(s))?;
        ensure!(module.bits == 6 && module.output_features == stored, "{stem}: fixture module");
        inventory.projections.insert(stem.clone(), CoreProjection { split: split(&stem).context("fixture core split")?,
            stem, bits: 6, codebook: book, shape, stored_outputs: stored, source_bytes: 0 });
    }
    inventory.validate_runtime_groups(&cfg)?;
    Ok((cfg, inventory))
}

/// On the CPU fake DSO: every operand of the small K6 checkpoint (globals,
/// dense KDA, routed KDA, MLA split and whole on each GPU) through the shared
/// records, filled by the mount and pushed through each GPU's rank in file
/// order, is byte-identical to the host packer's upload.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn recorded_k6_operands_match_the_host_packer_on_mount_and_push() -> Result<()> {
    use crate::shared::experts::rtx::dense::{DenseLoad, Records, STEP_BYTES};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (cfg, core) = small_checkpoint(root.path())?;
    let checkpoint = Checkpoint::open(root.path())?;
    let model = Installed { checkpoint, cfg, core };
    let mut expected = Digests::new();
    mount(&library, &model, &mut expected)?;
    ensure!(expected.keys().any(|k| k.ends_with("mla.uk")) && expected.keys().any(|k| k.contains("head.payload"))
        && expected.keys().any(|k| k.ends_with("kda.a_log")) && expected.keys().any(|k| k.ends_with("ffn.down.payload")),
        "fixture covers absorption, the K6 head, A_log and input-side K6");
    // Mount fill of the records.
    let load = DenseLoad::mount(&library);
    let (global, layers) = recorded(&library, &model, &load)?;
    load.release();
    let mut actual = Digests::new();
    recorded_digests(&library, &global, &layers, &mut actual)?;
    ensure!(actual == expected, "records by mount differ: {:?}", actual.iter().filter(|(k, v)| expected.get(*k) != Some(v))
        .map(|(k, _)| k).take(8).collect::<Vec<_>>());
    drop((global, layers));
    // The same records pushed through each GPU's rank.
    let marker = |path: &std::path::Path, name: &std::ffi::CStr, value: &[u8]| -> Result<()> {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        // SAFETY: both strings are NUL-terminated and live across the call; the
        // value pointer and length describe one live buffer.
        let status = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
        ensure!(status == 0, "setxattr: {}", std::io::Error::last_os_error());
        Ok(())
    };
    marker(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    for entry in std::fs::read_dir(root.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors") { marker(&path, c"user.sparknest.hosts", b"coordinator")?; }
    }
    let load = DenseLoad::push(&library);
    let (global, layers) = recorded(&library, &model, &load)?;
    let records = load.sink.as_ref().context("push sink")?.take_all();
    let (_, _, _, steps, largest, _) = records.summary();
    ensure!(steps > 0 && largest <= STEP_BYTES, "a step stages {largest} bytes");
    ensure!(records.steps.iter().any(|s| s.jobs.iter().any(|j| matches!(j, crate::shared::experts::rtx::dense::Job::TrellisTiles { input: true, .. })))
        && records.steps.iter().any(|s| s.jobs.iter().any(|j| matches!(j, crate::shared::experts::rtx::dense::Job::Product { .. }))),
        "records exercise input-side tiles and products");
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    for gpu in 0..2 {
        let mine = Records { operands: records.operands.iter().filter(|o| o.device == gpu).cloned().collect(),
            steps: records.steps.iter().filter(|s| s.device == gpu).cloned().collect(), finishes: Vec::new() };
        crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &library, id: gpu },
            None, root.path(), sockets.path(), &mine, None, 0, gpu as usize)?;
    }
    let mut pushed = Digests::new();
    recorded_digests(&library, &global, &layers, &mut pushed)?;
    ensure!(pushed == expected, "records by push differ: {:?}", pushed.iter().filter(|(k, v)| expected.get(*k) != Some(v))
        .map(|(k, _)| k).take(8).collect::<Vec<_>>());
    Ok(())
}
