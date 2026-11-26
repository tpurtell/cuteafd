//! GLM Flash drafters (DFlash2 and block dSpark) from their installed SparkNest
//! snapshots: the serving constructor recorded under the shared recorder, one
//! standalone coordinator rank replayed through a ring and committed, versus the
//! same constructor through the mount. Every checkpoint-sourced owner must be
//! byte-identical, the recording must not read the payload, and the recorded
//! staging must fit the header-only admitted drafter push envelope.
//!
//! On the CPU fixture (default) only the FP8 serving representation runs and its
//! W8A16 packs are no-ops: BF16 tensor owners are compared, the packs prove only
//! the recipe and staging bound. With CUTEAFD_NATIVE_LIB and
//! CUTEAFD_PROGRAM_MANIFEST (a real GPU through golden-run --native-test) both
//! representations run and every owner, packs included, must be byte-identical.
use crate::shared::experts::rtx::dense::{self, RankLayout};
use crate::shared::memory::device::Device;
use crate::shared::speculation::{block_dspark::DsparkDrafter, dflash::{DflashConfig, DflashDrafter}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::{glm5::draft_representation::{flash_draft_capacity, GlmDraftRepresentation},
    qwen4::Qwen4KvCache};
use std::path::PathBuf;

fn installed(repo: &str) -> Result<PathBuf> {
    let home = std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(|| "/mnt/sparknest/hf-home".into());
    let model = home.join("hub").join(format!("models--{}", repo.replace('/', "--")));
    let revision = std::fs::read_to_string(model.join("refs/main")).with_context(|| format!("{repo} not installed"))?;
    Ok(model.join("snapshots").join(revision.trim()))
}

fn read(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; buffer.bytes];
    library.copy_d2h_quiescent(&mut bytes, buffer)?;
    Ok(bytes)
}

enum Loaded<'a> { Dflash(DflashDrafter<'a>), Dspark(DsparkDrafter<'a>) }

impl Loaded<'_> {
    /// Checkpoint-sourced owners only: rings, taps and workspaces are zeroed, not loaded.
    fn owners(&self) -> Result<Vec<(String, CuteafdDeviceBuffer)>> {
        Ok(match self {
            Self::Dflash(draft) => {
                let ids = [draft.taps.buffer.device_id];
                let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&ids)?;
                draft.allocation_receipts(devices, "draft")?.into_iter().filter(|r| {
                    let id = &r.allocation.id;
                    r.allocation.location != cuteafd_loader::placement::planner::Location::HOST
                        && !["ring", "taps", "fused", "context", "workspace", "fp8_workspace"].iter()
                            .any(|skip| id.contains(skip))
                }).map(|r| (r.allocation.id.clone(), CuteafdDeviceBuffer { ptr: r.address as *mut _,
                    bytes: r.allocation.bytes as usize, device_id: draft.taps.buffer.device_id,
                    flags: draft.taps.buffer.flags })).collect()
            }
            Self::Dspark(draft) => draft.weight_buffers(),
        })
    }
}

fn library(fixture: &Option<cuteafd_ffi::native_library_lifetime_fixture::Fixture>) -> Result<NativeLibrary> {
    match fixture {
        Some(fixture) => {
            let library = fixture.load()?;
            fixture.configure_memory(&library, 0, 64 << 30)?;
            fixture.immediate_copies(&library)?;
            Ok(library)
        }
        // SAFETY: the slot's native library, loaded once for this test process.
        None => unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?) },
    }
}

#[test]
#[ignore = "requires installed GLM Flash drafters in SparkNest and an NVMe fixture directory (or CUTEAFD_NATIVE_LIB)"]
fn installed_flash_drafters_push_byte_identical_to_mount_within_admitted_staging() -> Result<()> {
    let real = std::env::var_os("CUTEAFD_NATIVE_LIB").is_some();
    let fixture = (!real).then(|| {
        let root = PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/cuteafd/builds/glmf-draft-push-fixture");
        cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()
    }).transpose()?;
    let library = library(&fixture)?;
    // The byte-copying CPU fixture has no BF16 projection programs (those need its
    // extent-only mode): it runs the FP8 (serving default) arms, real GPUs both.
    let programs = match std::env::var_os("CUTEAFD_PROGRAM_MANIFEST") {
        Some(manifest) => library.programs()?.with_manifest(std::path::Path::new(&manifest))?,
        None => library.programs()?,
    };
    let representations: &[GlmDraftRepresentation] = if real { &[GlmDraftRepresentation::Bf16Only,
        GlmDraftRepresentation::Fp8Only] } else { &[GlmDraftRepresentation::Fp8Only] };
    let device = Device { library: &library, id: 0 };
    let socket_root = PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/cuteafd/builds/glmf-draft-push-sockets");
    std::fs::create_dir_all(&socket_root)?;
    let (mut arms, mut skipped) = (0, 0);
    for repo in ["incoai/GLM-5.3-Flash-DFlash2", "RedHatAI/GLM-5.3-Flash-speculator.dspark-preview"] {
        let snapshot = installed(repo)?;
        let (root, _) = cuteafd_loader::push_load::namespace(&snapshot).context("drafter outside SparkNest")?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let dspark = crate::shared::speculation::block_dspark::is_dspark(&snapshot)?;
        let capacity = flash_draft_capacity(&config, Some(4), 4)?;
        let (slots, batch) = (capacity.context_slots, capacity.max_batch_sequences);
        let mask = |hidden: usize| (0..hidden * 2).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        for &representation in representations {
            // BF16 dSpark projections are not exported by serving builds (dSpark serves FP8).
            if dspark && representation == GlmDraftRepresentation::Bf16Only {
                let cfg = crate::shared::speculation::block_dspark::DsparkConfig::read(&snapshot)?;
                let name = format!("draft_bf16_k{}_n{}", cfg.taps.len() * cfg.hidden, cfg.hidden);
                if !programs.contains(&name) { eprintln!("{repo} Bf16Only: skipped, build lacks {name}"); skipped += 1; continue; }
            }
            let load = |recording: bool| -> Result<Loaded<'_>> {
                device.run(|| Ok(if dspark {
                    let cfg = crate::shared::speculation::block_dspark::DsparkConfig::read(&snapshot)?;
                    let tensors = if recording { Default::default() }
                        else { crate::shared::speculation::block_dspark::prefetch(&snapshot).join()
                            .map_err(|_| anyhow::anyhow!("prefetch panicked"))?? };
                    Loaded::Dspark(DsparkDrafter::load(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Bf16, &snapshot, tensors, std::ptr::null_mut(),
                        slots, batch, mask(cfg.hidden), representation, crate::shared::fp8_linear::Fp8Scales::Pow2,
                        crate::shared::fp8_linear::Fp8Rows::W8a16, Qwen4KvCache::Bf16)?)
                } else {
                    let cfg = DflashConfig::read(&snapshot)?;
                    let file = if recording { Vec::new() } else { crate::shared::speculation::dflash::prefetch(&snapshot)
                        .join().map_err(|_| anyhow::anyhow!("prefetch panicked"))?? };
                    Loaded::Dflash(DflashDrafter::load(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Bf16, &snapshot, file, std::ptr::null_mut(),
                        slots, batch, mask(cfg.hidden), true, representation, crate::shared::fp8_linear::Fp8Scales::Pow2,
                        crate::shared::fp8_linear::Fp8Rows::W8a16, Qwen4KvCache::Bf16)?)
                }))
            };
            let mounted = load(false)?;
            let (pushed, records) = dense::recording(|| load(true))?;
            ensure!(!records.operands.is_empty(), "{repo}: nothing recorded");
            ensure!(representation == GlmDraftRepresentation::Bf16Only || !records.steps.is_empty(),
                "{repo}: FP8 pack steps were not recorded");
            ensure!(records.peer_copies.is_empty(), "{repo}: a local drafter recorded peer copies");
            // Report the runtime transform frontier for the recorded drafter.
            let layout = RankLayout::new(records.operands.clone(), records.steps.clone(), 0)?.with_finishes(&records.finishes);
            eprintln!("{repo} {representation:?}: {} staged bytes (runtime bound)", layout.host_staging.bytes()?);
            let sockets = tempfile::tempdir_in(&socket_root)?;
            dense::testing::PEAKS.with(|p| p.borrow_mut().clear());
            crate::shared::experts::rtx::push::tests::receive_pushed(device, None, &root, sockets.path(), &records, None, 0, 0)?;
            for (staged, pinned) in dense::testing::PEAKS.with(|p| p.take()) {
                ensure!(pinned <= staged, "{repo}: free-list retention exceeded observed open-window peak");
            }
            let (expected, actual) = (mounted.owners()?, pushed.owners()?);
            ensure!(expected.len() == actual.len() && !expected.is_empty(), "{repo}: owner inventories differ");
            let mut compared = 0usize;
            for ((name, mount), (other, push)) in expected.iter().zip(&actual) {
                ensure!(name == other && mount.bytes == push.bytes, "{repo}: owner {name} differs from {other}");
                let packed = name.ends_with(".values") || name.ends_with(".scales");
                // The CPU fixture's W8A16 pack writes nothing; compare packs on a real GPU only.
                if packed && !real { continue; }
                ensure!(device.run(|| Ok(read(&library, *mount)? == read(&library, *push)?))?,
                    "{repo} {representation:?}: pushed {name} differs from the mount");
                compared += mount.bytes;
            }
            eprintln!("{repo} {representation:?}: {} owners, {compared} B byte-identical", expected.len());
            arms += 1;
        }
    }
    ensure!(arms + skipped == 2 * representations.len() && skipped <= 1 && arms >= 2,
        "expected every drafter/representation arm, ran {arms}, skipped {skipped}");
    Ok(())
}

#[test]
fn serve_loop_holds_no_draft_policy_of_its_own() {
    // Lengths come from the shared history policy's plan, shape and observe alone.
    let serve = include_str!("serve.rs");
    for call in ["policy.plan(&requests, verify_rows)", "policy.shape(&mut shaped, verify_rows)", "policy.observe_timed(&outcomes,"] {
        assert_eq!(serve.matches(call).count(), 1, "{call}");
    }
    // The trace's round times reuse the step's lane start, draft seconds and verify ms.
    assert!(serve.contains("policy.observe_timed(&outcomes, Some(history::RoundTiming::since(tally.lane(),\n            tally.started(), lane.draft_s - draft0, step_ms / 1e3, starts.clone())));"));
    for source in [include_str!("mod.rs"), serve, include_str!("dspark.rs")] {
        for removed in ["draft_binding", "resource::", "SelectorFit", "choose_block", "DraftEvidence"] {
            assert!(!source.contains(removed), "{removed}");
        }
    }
}
