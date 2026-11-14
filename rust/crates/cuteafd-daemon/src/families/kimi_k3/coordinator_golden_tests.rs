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
    for (name, native) in &weights.native {
        for (part, allocation) in [("values", Some(&native.values)), ("scales", native.scales.as_ref())] {
            if let Some(allocation) = allocation {
                let mut bytes = vec![0u8; allocation.buffer.bytes];
                Device { library, id: allocation.buffer.device_id }.run(||
                    library.copy_d2h(&mut bytes, allocation.buffer))?;
                out.insert(format!("{prefix}.{name}.{part}"), format!("{:x}", Sha256::digest(&bytes)));
            }
        }
    }
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

// The standalone path is a test-only before-packing byte oracle, not a loader.
fn baseline_operands<'a>(device: Device<'a>, operands: BTreeMap<&'static str, Vec<u8>>)
    -> Result<BTreeMap<&'static str, Allocation<'a>>> {
    operands.into_iter().map(|(name, bytes)| {
        let owner = Allocation::new(device, bytes.len(), crate::shared::memory::TEST)?;
        device.run(|| device.library.copy_h2d(owner.buffer, &bytes))?;
        Ok((name, owner))
    }).collect()
}

pub(crate) fn baseline_global<'a>(device: Device<'a>, layout: &GlobalLayout, packed: PackedGlobal)
    -> Result<GlobalWeights<'a>> {
    layout.check(&packed)?;
    let operands = baseline_operands(device, packed.operands)?;
    let core_head = packed.core_head.map(|head| layout.core_head.as_ref().unwrap().upload(device, head)).transpose()?;
    Ok(GlobalWeights { operands, core_head })
}

pub(in crate::families::kimi_k3) fn baseline_layer<'a>(device: Device<'a>, layout: &LayerLayout, packed: PackedLayer)
    -> Result<LayerWeights<'a>> {
    layout.check(&packed)?;
    let operands = baseline_operands(device, packed.operands)?;
    let core = packed.core.into_iter().map(|(name, loaded)|
        Ok((name, layout.core[name].upload(device, loaded)?))).collect::<Result<_>>()?;
    let native = packed.native.into_iter().map(|(name, bytes)| device.run(|| {
        let values = crate::shared::memory::DeviceAllocation::new(device.library, bytes.len(), crate::shared::memory::TEST)?;
        device.library.copy_h2d(values.buffer, &bytes)?;
        Ok((name, crate::shared::native_attention::NativeAttentionWeights {
            layout: layout.native[name].clone(), values, scales: None }))
    })).collect::<Result<_>>()?;
    Ok(LayerWeights { operands, attention: packed.attention, core, native })
}

/// Every golden operand through the independent host packer, on its own GPU.
pub(crate) fn mount(library: &NativeLibrary, model: &Installed, out: &mut Digests) -> Result<()> {
    mount_for_ranks(library, model, 2, out)
}

fn fixture_layers(cfg: &KimiK3Config, ranks: usize) -> Result<Vec<(usize, LayerMode)>> {
    if ranks == 2 { return golden_layers(cfg); }
    ensure!(ranks == 1, "Kimi fixture physical rank count");
    Ok((0..cfg.layers).map(|layer| (layer,
        LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner })).collect())
}

fn mount_for_ranks(library: &NativeLibrary, model: &Installed, ranks: usize, out: &mut Digests) -> Result<()> {
    let devices = (0..ranks).map(|id| Device { library, id: id as i32 }).collect::<Vec<_>>();
    let read = |name: &str| read_tensor(&model.checkpoint, name);
    let globals = GlobalLayout::ranks_with_core(&model.cfg, &model.core, ranks)?;
    for (rank, layout) in globals.iter().enumerate() {
        let weights = baseline_global(devices[rank], layout, layout.pack(&model.cfg, read)?)?;
        global_digests(library, rank, &weights, out)?;
    }
    for (layer, mode) in fixture_layers(&model.cfg, ranks)? {
        let layouts = LayerLayout::ranks_with_core(&model.cfg, layer, mode, &model.core, ranks)?;
        let packed = PackedLayer::ranks_with_core(&model.cfg, layer, mode, &model.core, ranks, read)?;
        for ((rank, packed), layout) in packed.into_iter().enumerate().zip(&layouts) {
            let weights = baseline_layer(devices[rank], layout, packed)?;
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
    recorded_for_ranks(library, model, load, 2)
}

fn recorded_for_ranks<'a>(library: &'a NativeLibrary, model: &Installed,
    load: &crate::shared::experts::rtx::dense::DenseLoad<'a>, ranks: usize)
    -> Result<(Vec<(usize, GlobalWeights<'a>)>, Vec<(String, usize, LayerWeights<'a>)>)> {
    let devices = (0..ranks).map(|id| Device { library, id: id as i32 }).collect::<Vec<_>>();
    let mut records = records::KimiRecords::new(&model.checkpoint, &model.cfg, &model.core, load,
        devices.iter().map(|device| (device.id, std::ptr::null_mut())).collect());
    let globals = GlobalLayout::ranks_with_core(&model.cfg, &model.core, ranks)?;
    let selected = (0..model.cfg.layers).map(|layer| LayerLayout::ranks_with_core(&model.cfg, layer,
        if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } } else { LayerMode::HeadSplit },
        &model.core, ranks)).collect::<Result<Vec<_>>>()?;
    records.arenas = devices.iter().enumerate().map(|(rank, &device)| {
        let global = globals[rank].shared_layout();
        let layers = selected.iter().map(|layouts| layouts[rank].shared_layout()).collect::<Vec<_>>();
        let layout = cuteafd_loader::families::kimi_k3::weight_layout::coordinator_arena(
            rank, &global, &layers.iter().collect::<Vec<_>>())?;
        Ok((device.id, crate::shared::memory::weight_arena::DeviceWeightArena::new(device, layout, crate::shared::memory::TEST_ARENA)?))
    }).collect::<Result<BTreeMap<_, _>>>()?.into();
    let mut global = Vec::new();
    for (rank, layout) in globals.iter().enumerate() { global.push((rank, records.global(devices[rank], layout)?)); }
    let mut layers = Vec::new();
    for (layer, mode) in fixture_layers(&model.cfg, ranks)? {
        let layouts = LayerLayout::ranks_with_core(&model.cfg, layer, mode, &model.core, ranks)?;
        // Whole-MLA variants are independent golden scopes. The selected heads
        // path exercises the same complete per-GPU arena as production.
        let alternate = records::KimiRecords::new(&model.checkpoint, &model.cfg, &model.core, load,
            records.streams.clone());
        let target = if ranks == 1 || mode == LayerMode::HeadSplit { &records } else { &alternate };
        for (rank, weights) in target.layer_ranks(&devices, layer, &layouts)?.into_iter().enumerate() {
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
            steps: records.steps.iter().filter(|s| s.device == gpu).cloned().collect(), ..Default::default() };
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
fn small_checkpoint(root: &std::path::Path, native_fp8: bool) -> Result<(KimiK3Config, CoreInventory)> {
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
    let mut selected_native = BTreeMap::new();
    let mut native = |tensors: &mut Vec<Tensor>, stem: String, n: usize, k: usize, stored: usize, book: Exl3Codebook| {
        if stem.ends_with(".q_b_proj") || stem == "language_model.model.layers.1.self_attn.o_proj" {
            tensors.push(t(format!("{stem}.weight"), if native_fp8 { "F8_E4M3" } else { "BF16" }, &[n, k]));
            if native_fp8 {
                let suffix = if stem.ends_with("q_b_proj") { "weight_scale_inv" } else { "qscale_weight" };
                tensors.push(t(format!("{stem}.{suffix}"), "F32", &[n.div_ceil(128), k.div_ceil(128)]));
            }
            selected_native.insert(stem.clone(), cuteafd_loader::formats::native_attention::NativeProjection {
                shape: [n, k], fp8: native_fp8, split: split(&stem).expect("fixture native split") });
            return;
        }
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
        let mut bytes: Vec<u8> = (0..meta.byte_length).map(|i|
            (i as u8).wrapping_mul(29) ^ ((i >> 8) as u8).wrapping_mul(17)
                ^ ((i >> 16) as u8).wrapping_mul(11) ^ (meta.name.len() as u8)).collect();
        if let Some(book) = [Exl3Codebook::Mcg, Exl3Codebook::Mul1].into_iter().find(|b| meta.name.ends_with(&format!(".{}", b.marker()))) {
            bytes = book.multiplier().to_le_bytes().to_vec();
        } else if meta.name.ends_with(".A_log") {
            bytes = (0..32).flat_map(|i| if i < cfg.kda_heads { (i as f32 * 0.25 - 1.0).to_le_bytes() } else { 0f32.to_le_bytes() }).collect();
        }
        std::fs::File::options().write(true).open(root.join(&tensor.shard))?.write_all_at(&bytes, meta.byte_offset)?;
    }
    let mut inventory = CoreInventory { native: selected_native, ..Default::default() };
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
    let (cfg, core) = small_checkpoint(root.path(), false)?;
    let checkpoint = Checkpoint::open(root.path())?;
    let model = Installed { checkpoint, cfg, core };
    let rejected = DenseLoad::push(&library);
    let records = records::KimiRecords::new(&model.checkpoint, &model.cfg,
        &model.core, &rejected, vec![(0, std::ptr::null_mut())]);
    let device = Device { library: &library, id: 0 };
    let whole = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
    let mut invalid = LayerLayout::ranks_with_core(&model.cfg, 0, whole, &model.core, 1)?;
    *invalid[0].operands.get_mut("attention.score").unwrap() -= 1;
    *invalid[0].operands.get_mut("ffn.score").unwrap() += 1;
    // The fake DSO creates its log lazily on the first native operation.
    let _validation_stream = crate::shared::memory::device::Stream::new(device)?;
    let before = fixture.events()?;
    ensure!(records.layer_ranks(&[device], 0, &invalid).is_err(), "equal-total extent swap must fail");
    let valid = LayerLayout::ranks_with_core(&model.cfg, 0, whole, &model.core, 1)?;
    ensure!(records.layer_ranks(&[device], model.cfg.layers, &valid).is_err(), "out-of-range layer must fail");
    ensure!(records.layer_ranks(&[device, device], 0, &valid).is_err(), "aliased physical ranks must fail");
    let mut global = GlobalLayout::ranks_with_core(&model.cfg, &model.core, 1)?;
    global[0].operands.remove("embedding");
    ensure!(records.global(device, &global[0]).is_err(), "missing entry operand must fail");
    ensure!(fixture.events()? == before, "rejected layouts must not allocate or submit work");
    for ranks in [1, 2] {
        let mut expected = Digests::new();
        mount_for_ranks(&library, &model, ranks, &mut expected)?;
        ensure!(expected.contains_key("global.0.embedding")
            && expected.contains_key(&format!("global.{}.head.payload", ranks - 1))
            && (ranks == 2 || !expected.keys().any(|key| key.starts_with("global.1."))),
            "entry and terminal operands follow physical owners");
        ensure!(expected.keys().any(|k| k.ends_with("mla.uk")) && expected.keys().any(|k| k.contains("head.payload"))
            && expected.keys().any(|k| k.ends_with("kda.a_log")) && expected.keys().any(|k| k.ends_with("ffn.down.payload"))
            && expected.keys().any(|k| k.ends_with("mla.q_b.values"))
            && expected.keys().any(|k| k.ends_with("kda.output.values")),
            "fixture covers absorption, the K6 head, A_log, input-side K6 and native TP slices");
        // Mount fill of the records.
        let load = DenseLoad::mount(&library);
        let (global, layers) = recorded_for_ranks(&library, &model, &load, ranks)?;
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
        let (global, layers) = recorded_for_ranks(&library, &model, &load, ranks)?;
        let records = load.sink.as_ref().context("push sink")?.take_all();
        let (_, _, _, steps, largest, _) = records.summary();
        ensure!(steps > 0 && largest <= STEP_BYTES, "a step stages {largest} bytes");
        // Whole input-side weights copy directly; only TP2 slices gather tiles.
        ensure!(ranks == 1 || records.steps.iter().any(|s| s.jobs.iter().any(|j|
            matches!(j, crate::shared::experts::rtx::dense::Job::TrellisTiles { input: true, .. }))),
            "split records exercise input-side tiles");
        ensure!(records.steps.iter().any(|s| s.jobs.iter().any(|j|
            matches!(j, crate::shared::experts::rtx::dense::Job::Product { .. }))),
            "records exercise scale products");
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        for gpu in 0..ranks as i32 {
            let mine = Records { operands: records.operands.iter().filter(|o| o.device == gpu).cloned().collect(),
                steps: records.steps.iter().filter(|s| s.device == gpu).cloned().collect(), ..Default::default() };
            crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &library, id: gpu },
                None, root.path(), sockets.path(), &mine, None, 0, gpu as usize)?;
        }
        let mut pushed = Digests::new();
        recorded_digests(&library, &global, &layers, &mut pushed)?;
        ensure!(pushed == expected, "records by push differ: {:?}", pushed.iter().filter(|(k, v)| expected.get(*k) != Some(v))
            .map(|(k, _)| k).take(8).collect::<Vec<_>>());
    }
    Ok(())
}
// Independent of attention_records, Rect, fp8_scale and the BF16 host packer.
fn checkpoint_rectangle(checkpoint: &Checkpoint, name: &str, dtype: cuteafd_core::DType,
    rows: std::ops::Range<usize>, cols: std::ops::Range<usize>, width: usize) -> Result<Vec<u8>> {
    let tensor = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name)
        .with_context(|| format!("oracle tensor missing: {name}"))?;
    let header = &tensor.meta;
    ensure!(header.dtype == dtype && header.shape.len() == 2 && rows.end <= header.shape[0]
        && cols.end <= header.shape[1] && header.byte_length == (header.shape[0] * header.shape[1] * width) as u64,
        "oracle rectangle outside checkpoint header: {name}");
    let file = std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?;
    let mut bytes = vec![0; rows.len() * cols.len() * width];
    for (destination, row) in bytes.chunks_exact_mut(cols.len() * width).zip(rows) {
        file.read_exact_at(destination, header.byte_offset + ((row * header.shape[1] + cols.start) * width) as u64)?;
    }
    Ok(bytes)
}

#[test]
fn native_fp8_arena_bytes_match_checkpoint_tp_rectangles_on_mount_and_push() -> Result<()> {
    use crate::shared::experts::rtx::dense::{DenseLoad, Records};
    use cuteafd_loader::formats::trellis_core::CoreSplit;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    for device in 0..2 { fixture.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (cfg, core) = small_checkpoint(root.path(), true)?;
    let model = Installed { checkpoint: Checkpoint::open(root.path())?, cfg, core };
    let marker = |path: &std::path::Path, name: &std::ffi::CStr, value: &[u8]| -> Result<()> {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        // SAFETY: live, NUL-terminated path/name and the exact value slice.
        let status = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
        ensure!(status == 0, "setxattr: {}", std::io::Error::last_os_error());
        Ok(())
    };
    marker(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    for entry in std::fs::read_dir(root.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "safetensors") {
            marker(&path, c"user.sparknest.hosts", b"coordinator")?;
        }
    }
    for ranks in [1, 2] { for push in [false, true] {
        let load = if push { DenseLoad::push(&library) } else { DenseLoad::mount(&library) };
        let (globals, layers) = recorded_for_ranks(&library, &model, &load, ranks)?;
        if push {
            let records = load.sink.as_ref().context("native FP8 push sink")?.take_all();
            let sockets = tempfile::tempdir()?;
            for gpu in 0..ranks as i32 {
                let mine = Records { operands: records.operands.iter().filter(|owner| owner.device == gpu).cloned().collect(),
                    steps: records.steps.iter().filter(|step| step.device == gpu).cloned().collect(), ..Default::default() };
                crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &library, id: gpu },
                    None, root.path(), sockets.path(), &mine, None, 0, gpu as usize)?;
            }
        } else { load.release(); }
        let mut checked = std::collections::BTreeSet::new();
        for (_, _, layer) in &layers { for native in layer.native.values() {
            let layout = &native.layout;
            assert!(layout.fp8);
            let [n, k] = layout.shape;
            let (row, col) = match layout.split {
                CoreSplit::Output => (layout.rank * n, 0),
                CoreSplit::Input => (0, layout.rank * k),
                CoreSplit::Whole => (0, 0),
            };
            let suffix = if layout.stem.ends_with("q_b_proj") { "weight_scale_inv" } else { "qscale_weight" };
            let expected = checkpoint_rectangle(&model.checkpoint, &format!("{}.weight", layout.stem),
                cuteafd_core::DType::F8E4M3, row..row + n, col..col + k, 1)?;
            let expected_scales = checkpoint_rectangle(&model.checkpoint, &format!("{}.{suffix}", layout.stem),
                cuteafd_core::DType::F32, row / 128..(row + n) / 128, col / 128..(col + k) / 128, 4)?;
            for (owner, expected) in [(&native.values, expected), (native.scales.as_ref().context("FP8 scales")?, expected_scales)] {
                // The min-256 allocation tail is not initialized tensor data.
                let mut actual = vec![0; expected.len()];
                let buffer = cuteafd_ffi::CuteafdDeviceBuffer { bytes: actual.len(), ..owner.buffer };
                Device { library: &library, id: buffer.device_id }.run(|| library.copy_d2h(&mut actual, buffer))?;
                assert_eq!(actual, expected, "{} TP{ranks} rank{} push={push}", layout.stem, layout.rank);
                let ids = [0, 1];
                let mut inventory = crate::shared::memory::receipts::Inventory::new(
                    cuteafd_loader::placement::planner::CoordinatorDevices::new(&ids[..ranks])?, "native",
                    cuteafd_loader::placement::planner::Lifetime::RESIDENT,
                    cuteafd_core::memory_layout::Category::Weights, "native attention");
                owner.inventory(&mut inventory, "plane")?;
                assert!(inventory.receipts.iter().any(|receipt| receipt.allocation.view.is_some()), "must be an arena view");
            }
            checked.insert(suffix);
        }}
        assert_eq!(checked.len(), 2);
        drop((globals, layers));
    }}
    Ok(())
}
