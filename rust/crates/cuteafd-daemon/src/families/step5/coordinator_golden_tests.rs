//! Step-5 coordinator operands from the installed checkpoint (CUTEAFD_PUSH_STEP5)
//! on real RTX GPUs: every operand `load_operand` builds for representative
//! layers (a head-split SWA layer with its dense MLP, a whole CSA layer with
//! its routed front end, a routed SWA layer) on both ranks of the two-GPU
//! ownership, those layers on one GPU, and MTP stage 0. Digests are pinned to
//! the mount loader as it was before push recipes, so the push recipes
//! (landings, F32 narrowing on the GPU) must reproduce it bit for bit.
//! CUTEAFD_STEP5_GOLDEN_WRITE=1 records instead.
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::step5::coordinator::RankWeightLayout};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) type Digests = BTreeMap<String, String>;

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_STEP5").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

/// The golden's layers: dense SWA (0), routed SWA (4), routed CSA (3, 7).
pub(crate) const LAYERS: [usize; 4] = [0, 3, 4, 7];

/// (key prefix, device, layout) of every layout the golden covers.
pub(crate) fn layouts(checkpoint: &Checkpoint) -> Result<Vec<(String, i32, RankWeightLayout)>> {
    let cfg = cuteafd_loader::plan::families::step5::Config::parse(checkpoint)?;
    let ownership = super::golden::split_ownership(&cfg, 16)?;
    let single = cuteafd_loader::plan::families::step5::coordinator::RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
    let mut out = Vec::new();
    for rank in 0..2 {
        out.push((format!("split.rank{rank}"), rank as i32, RankWeightLayout::for_execution(checkpoint, rank, &ownership)?));
    }
    out.push(("single".into(), 0, single));
    out.push(("mtp0".into(), 1, RankWeightLayout::for_mtp_stage(checkpoint, 0)?));
    Ok(out)
}

/// Whether the golden covers operand `name` (its layers, the head, MTP stage 0).
pub(crate) fn covered(name: &str) -> bool {
    match name.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok()) {
        Some(layer) => LAYERS.contains(&layer) || layer >= 92,
        None => !name.contains("embed_tokens"),
    }
}

pub(crate) fn digest(library: &NativeLibrary, buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<String> {
    let mut bytes = vec![0u8; buffer.bytes];
    Device { library, id: buffer.device_id }.run(|| library.copy_d2h_quiescent(&mut bytes, buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub(crate) fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_STEP5_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/step5/coordinator_golden.json"))
}

pub(crate) fn check(actual: &Digests, subset: bool) -> Result<()> {
    if std::env::var_os("CUTEAFD_STEP5_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_STEP5_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
        std::fs::write(&out, serde_json::to_vec_pretty(actual)?)?;
        eprintln!("wrote {} operand digests to {}", actual.len(), out.display());
        return Ok(());
    }
    let golden: Digests = serde_json::from_slice(&std::fs::read(path())?)?;
    if !subset {
        anyhow::ensure!(actual.keys().eq(golden.keys()), "operand inventory changed: {} actual, {} golden",
            actual.len(), golden.len());
    }
    let differ = actual.iter().filter(|(k, v)| golden.get(*k) != Some(v)).map(|(k, _)| k.clone()).collect::<Vec<_>>();
    anyhow::ensure!(differ.is_empty(), "{} operands differ from the golden, first {:?}", differ.len(),
        &differ[..differ.len().min(8)]);
    eprintln!("PASS {} operand digests match the golden", actual.len());
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_STEP5, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let checkpoint = Checkpoint::coordinator(&snapshot, false, true)?;
    let library = library()?;
    let mut actual = Digests::new();
    for (prefix, device, layout) in layouts(&checkpoint)? {
        let device = Device { library: &library, id: device };
        for operand in layout.operands.iter().filter(|o| covered(&o.name)) {
            let allocation = super::weights::load_operand(device, &checkpoint, &layout, operand, None)?;
            actual.insert(format!("{prefix}.{}", operand.name), digest(&library, allocation.buffer)?);
        }
    }
    check(&actual, false)
}

/// The dense records `load_operand` makes, filled through the mount on the
/// fake native DSO (the narrowing through its CPU twin of the GPU kernel),
/// equal the host packing they replaced for every covered operand of the
/// installed checkpoint's headers (both split ranks, one GPU, MTP stage 0).
#[test]
#[ignore = "requires CUTEAFD_PUSH_STEP5, an allocated CPU build slot and explicit NVMe fixture directory"]
fn records_match_the_host_packing_on_installed_headers() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let checkpoint = Checkpoint::coordinator(&snapshot, false, true)?;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 256 << 30)?; }
    let reader = crate::shared::qwen::weights::WeightReader { library: &library, checkpoint: &checkpoint };
    let (mut checked, mut narrowed) = (0, 0);
    for (prefix, device, layout) in layouts(&checkpoint)? {
        let device = Device { library: &library, id: device };
        for operand in layout.operands.iter().filter(|o| covered(&o.name)) {
            let expected = super::weights::pack_reference(operand, &layout, |name| reader.raw(name))?;
            let allocation = super::weights::load_operand(device, &checkpoint, &layout, operand, None)?;
            let mut actual = vec![0u8; expected.len()];
            library.copy_d2h_quiescent(&mut actual, cuteafd_ffi::CuteafdDeviceBuffer { bytes: expected.len(), ..allocation.buffer })?;
            anyhow::ensure!(actual == expected, "{prefix}.{}: records differ from the host packing", operand.name);
            checked += 1;
            narrowed += usize::from(operand.sources.iter().any(|s| s.meta.dtype != operand.dtype));
        }
    }
    anyhow::ensure!(narrowed > 0, "no F32 narrowing exercised");
    eprintln!("PASS {checked} operands ({narrowed} narrowed from F32) match the host packing");
    Ok(())
}

/// The GPU narrowing's CPU twin equals `OutputPrecision::Bf16::round` bit for
/// bit on edge values: ties both ways, the largest finite values rounding to
/// infinity, infinities, signed zeros, subnormals and NaN payloads.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn narrowing_matches_bf16_round_on_edge_values() -> Result<()> {
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
    let library = fixture.load()?;
    let mut values: Vec<u32> = vec![0, 0x8000_0000, 1, 0x8000_0001, 0x007f_ffff, 0x7f7f_ffff, 0xff7f_ffff,
        0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc0_0000, 0x7fff_ffff, 0x3f80_8000, 0x3f81_8000, 0x3f80_7fff,
        0x3f80_8001, 0x7f7f_8000, 0x7fbf_ffff];
    let mut x = 0x9e37_79b9u32;
    for _ in 0..1 << 20 { x ^= x << 13; x ^= x >> 17; x ^= x << 5; values.push(x); }
    let mut out = vec![0u16; values.len()];
    // SAFETY: host buffers of the shapes the CPU twin reads and writes.
    unsafe { library.load_narrow_f32_bf16(values.as_ptr().cast(), out.as_mut_ptr().cast(), values.len(),
        std::ptr::null_mut())?; }
    for (&bits, &half) in values.iter().zip(&out) {
        let want = cuteafd_core::dense_attention::OutputPrecision::Bf16.round(f32::from_bits(bits)).to_bits();
        anyhow::ensure!(u32::from(half) << 16 == want, "{bits:#010x}: narrowed {half:#06x}, round gives {want:#010x}");
    }
    Ok(())
}

/// The two-GPU split operands recorded (allocated as through the mount,
/// bytes deferred), then each GPU's records pushed through a ring in file
/// order into its rank (narrowing on the GPU): every operand matches the mount
/// golden and each rank's staging stays one buffer.
#[test]
#[ignore = "requires CUTEAFD_PUSH_STEP5 (a SparkNest namespace), CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_pushed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    use crate::shared::experts::rtx::dense::{Recorder, STEP_BYTES};
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let checkpoint = Checkpoint::coordinator(&snapshot, false, true)?;
    let library = library()?;
    let (root, _) = cuteafd_loader::push_load::namespace(&snapshot)?;
    let recorder = Recorder::start(true)?.context("recorder")?;
    let mut owners = Vec::new();
    for (prefix, device, layout) in layouts(&checkpoint)?.into_iter().filter(|(p, ..)| p.starts_with("split")) {
        let device = Device { library: &library, id: device };
        for operand in layout.operands.iter().filter(|o| covered(&o.name)) {
            owners.push((format!("{prefix}.{}", operand.name),
                super::weights::load_operand(device, &checkpoint, &layout, operand, None)?));
        }
    }
    let records = recorder.finish();
    let (bytes, operands, staged, steps, largest, finishes) = records.summary();
    eprintln!("records: class2 {bytes} B in {operands} operands; class3 {staged} B in {steps} steps (largest {largest} B); \
        {finishes} finishes");
    anyhow::ensure!(steps > 0 && largest <= STEP_BYTES, "narrowing steps within the bound");
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let sockets = tempfile::tempdir()?;
    for gpu in 0..2 {
        crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &library, id: gpu }, Some(&catalog), &root,
            sockets.path(), &records, None, 6, gpu as usize)?;
    }
    for (staged, pinned) in crate::shared::experts::rtx::dense::testing::PEAKS.with(|p| p.take()) {
        eprintln!("rank staging peak {staged} B in use, {pinned} B pinned");
        anyhow::ensure!(staged <= STEP_BYTES as u64 && pinned <= STEP_BYTES as u64, "staging beyond one step");
    }
    let mut actual = Digests::new();
    for (key, allocation) in &owners { actual.insert(key.clone(), digest(&library, allocation.buffer)?); }
    check(&actual, true)
}
