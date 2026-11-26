//! Qwen 3.8 Flash Next coordinator operands from the installed checkpoint
//! (CUTEAFD_PUSH_QWEN4) on real GPUs (CUTEAFD_NATIVE_LIB): one GDN, one full
//! attention and one PLE layer, the final mixer, the FP8 head and the MTP
//! layer, BF16 and FP8-decode projections, on one GPU and placed over two.
//! Digests are pinned to the mount loader as it was before push records
//! (CUTEAFD_QWEN4_GOLDEN overrides the golden path; CUTEAFD_QWEN4_GOLDEN_WRITE=1
//! records it, to CUTEAFD_QWEN4_GOLDEN_OUT when set).
use super::*;
use crate::shared::memory::device::Device;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) type Digests = BTreeMap<String, String>;

fn digest(library: &NativeLibrary, allocation: &Allocation<'_>) -> Result<String> {
    let mut bytes = vec![0u8; allocation.buffer.bytes];
    allocation.device.run(|| library.copy_d2h_quiescent(&mut bytes, allocation.buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_QWEN4").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

pub(crate) fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_QWEN4_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/qwen4/coordinator_golden.json"))
}

pub(crate) fn check(actual: &Digests) -> Result<()> {
    if std::env::var_os("CUTEAFD_QWEN4_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_QWEN4_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
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

/// The golden's layers: the first GDN, full attention and PLE layers (as a
/// placed prefix, `0..=max` with whole-width owners alternating over ranks).
pub(crate) fn golden_layers(cfg: &Qwen4Config) -> Result<usize> {
    let gdn = cfg.attention.iter().position(|a| *a == Qwen4Attention::Gdn).context("no GDN layer")?;
    let full = cfg.attention.iter().position(|a| *a == Qwen4Attention::Full).context("no full layer")?;
    let ple = *cfg.ple_layers.first().context("no PLE layer")?;
    Ok(gdn.max(full).max(ple) + 1)
}

pub(crate) fn weights_digests(library: &NativeLibrary, prefix: &str, weights: &Qwen4Weights<'_>,
    out: &mut Digests) -> Result<()> {
    for (layer, part) in weights.layers.iter().enumerate() {
        for (name, allocation) in part.operands() {
            out.insert(format!("{prefix}.layers.{layer}.{name}"), digest(library, allocation)?);
        }
    }
    for (i, allocation) in weights.mixer.iter().enumerate() {
        out.insert(format!("{prefix}.mixer.{i}"), digest(library, allocation)?);
    }
    for (i, allocation) in weights.head.allocations().into_iter().enumerate() {
        out.insert(format!("{prefix}.head.{i}"), digest(library, allocation)?);
    }
    if let Some(mtp) = &weights.mtp {
        for (name, allocation) in mtp.layer.operands() {
            out.insert(format!("{prefix}.mtp.layer.{name}"), digest(library, allocation)?);
        }
        for (name, allocation) in [("norm_hidden", &mtp.norm_hidden), ("norm_embed", &mtp.norm_embed),
            ("fc_hidden", &mtp.fc_hidden), ("fc_embed", &mtp.fc_embed)] {
            out.insert(format!("{prefix}.mtp.{name}"), digest(library, allocation)?);
        }
        for (i, allocation) in mtp.mixer.iter().enumerate() {
            out.insert(format!("{prefix}.mtp.mixer.{i}"), digest(library, allocation)?);
        }
    }
    Ok(())
}

/// Every golden operand through `loader(fp8_decode)`: one GPU, then placed
/// over two (alternating owners), each with BF16 and FP8-decode projections.
pub(crate) fn mount<'l>(library: &'l NativeLibrary, checkpoint: &Checkpoint, cfg: &Qwen4Config,
    loader: impl Fn(bool) -> Qwen4Loader<'l>, out: &mut Digests) -> Result<()> {
    let layers = golden_layers(cfg)?;
    for fp8 in [false, true] {
        let tag = if fp8 { "fp8" } else { "bf16" };
        let single = loader(fp8);
        let weights = single.model(cfg, layers, true, true)?;
        weights_digests(library, &format!("{tag}.single"), &weights, out)?;
        drop(weights);
        let ranks: Vec<_> = (0..2).map(|id| (Device { library, id }, std::ptr::null_mut())).collect();
        let owners: Vec<_> = (0..layers).map(|layer| layer % 2).collect();
        let placed = loader(fp8).model_placed(cfg, &owners, &ranks, true, true)?;
        weights_digests(library, &format!("{tag}.placed"), &placed, out)?;
    }
    let _ = checkpoint;
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_QWEN4, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let checkpoint = Checkpoint::coordinator(&snapshot, true, true)?;
    let cfg = Qwen4Config::read(&snapshot)?;
    let library = library()?;
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    let mut actual = Digests::new();
    mount(&library, &checkpoint, &cfg, |fp8_decode| Qwen4Loader { library: &library, checkpoint: &checkpoint,
        fp8_decode, fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, stream: std::ptr::null_mut(),
        dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(&library)), arenas: None }, &mut actual)?;
    check(&actual)
}

/// The same operands recorded (allocated as through the mount, bytes
/// deferred), then each GPU's records pushed through a ring into its rank in
/// file order (block and row quantization, widening on the GPU): every operand
/// matches the mount golden and each rank's staging stays bounded.
#[test]
#[ignore = "requires CUTEAFD_PUSH_QWEN4 (a SparkNest namespace), CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_pushed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    use crate::shared::experts::rtx::dense::{DenseLoad, STEP_BYTES};
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let checkpoint = Checkpoint::coordinator(&snapshot, true, true)?;
    let cfg = Qwen4Config::read(&snapshot)?;
    let library = library()?;
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    let (root, _) = cuteafd_loader::push_load::namespace(&snapshot)?;
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let layers = golden_layers(&cfg)?;
    let sockets = tempfile::tempdir()?;
    let mut actual = Digests::new();
    for fp8 in [false, true] {
        let tag = if fp8 { "fp8" } else { "bf16" };
        for placed in [false, true] {
            let load = std::rc::Rc::new(DenseLoad::push(&library));
            let loader = Qwen4Loader { library: &library, checkpoint: &checkpoint, fp8_decode: fp8,
                fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, stream: std::ptr::null_mut(), dense: load.clone(), arenas: None };
            let weights = if placed {
                let ranks: Vec<_> = (0..2).map(|id| (Device { library: &library, id }, std::ptr::null_mut())).collect();
                loader.model_placed(&cfg, &(0..layers).map(|l| l % 2).collect::<Vec<_>>(), &ranks, true, true)?
            } else { loader.model(&cfg, layers, true, true)? };
            let records = load.sink.as_ref().context("push sink")?.take_all();
            let (bytes, operands, staged, steps, largest, finishes) = records.summary();
            eprintln!("{tag} placed={placed}: class2 {bytes} B in {operands} operands; class3 {staged} B in {steps} steps \
                (largest {largest} B); {finishes} finishes");
            ensure!(largest <= STEP_BYTES, "a step stages {largest} bytes");
            for gpu in 0..if placed { 2 } else { 1 } {
                if !records.operands.iter().any(|o| o.device == gpu as i32) { continue; }
                crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &library, id: gpu as i32 },
                    Some(&catalog), &root, sockets.path(), &records, None, 0, gpu)?;
            }
            for (staged, pinned) in crate::shared::experts::rtx::dense::testing::PEAKS.with(|p| p.take()) {
                ensure!(staged <= STEP_BYTES as u64 && pinned <= STEP_BYTES as u64,
                    "rank staging peaked at {staged} B in use, {pinned} B pinned");
            }
            weights_digests(&library, &format!("{tag}.{}", if placed { "placed" } else { "single" }), &weights, &mut actual)?;
        }
    }
    check(&actual)
}
