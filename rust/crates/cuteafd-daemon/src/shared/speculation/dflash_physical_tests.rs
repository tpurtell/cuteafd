//! Header-driven real serving loads against the allocation-only CPU native DSO.
use super::*;
use cuteafd_loader::{families::glm5::draft_physical::GlmDflashPhysicalLayout,
    families::qwen4::Qwen4KvCache, placement::planner::{CoordinatorDevices, Location}};
use std::{collections::BTreeMap, path::PathBuf};

fn installed_snapshot(repo: &str) -> Result<PathBuf> {
    let home = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface"));
    let model = home.join("hub").join(format!("models--{}", repo.replace('/', "--")));
    let revision = std::fs::read_to_string(model.join("refs/main"))
        .with_context(|| format!("required real draft checkpoint {repo} missing"))?;
    let snapshot = model.join("snapshots").join(revision.trim());
    ensure!(snapshot.join("model.safetensors").is_file(), "required real draft headers missing: {}", snapshot.display());
    Ok(snapshot)
}

fn assert_glm_draft_lifetimes(common: &[cuteafd_loader::placement::planner::Receipt],
    family: &[cuteafd_loader::placement::planner::Receipt]) {
    use cuteafd_loader::placement::planner::Lifetime;
    assert_eq!(common.len(), family.len());
    let mut immutable = 0;
    let mut unchanged = 0;
    for (before, after) in common.iter().zip(family) {
        let mut expected = before.allocation.clone();
        if expected.id.starts_with("glm/draft/") || expected.view.as_ref()
            .is_some_and(|view| view.backing.starts_with("glm/draft/")) {
            expected.lifetime = Lifetime::RESIDENT;
            immutable += 1;
        } else {
            unchanged += 1;
        }
        assert_eq!(after.allocation, expected);
        assert_eq!(after.address, before.address);
    }
    assert!(immutable > 1 && unchanged > 0, "prove both arena and nonweight chronology");
}

#[test]
fn arena_ordered_upload_receipt_uses_the_offset_zero_alias() -> Result<()> {
    use crate::shared::memory::{device::{Device, Stream}, weight_arena::DeviceWeightArena};
    use cuteafd_loader::{plan::testing::{t, write_safetensors}, serving_capacity::WeightArenaLayout};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.configure_memory(&library, 1, 1usize << 30)?;
    let device = Device { library: &library, id: 1 };
    let root = tempfile::tempdir_in(fixture.directory())?;
    let snapshot = root.path();
    std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&serde_json::json!({
        "hidden_size":128, "intermediate_size":128, "num_hidden_layers":1,
        "num_attention_heads":1, "num_key_value_heads":1, "head_dim":128,
        "rms_norm_eps":1e-6, "rope_theta":10000., "sliding_window":2048, "vocab_size":16,
        "dflash_config":{"block_size":8, "conv_group_size":16, "conv_kernel_size":2,
            "mask_token_id":0, "selector_rank":256, "selector_top_k":16, "target_layer_ids":[0]}
    }))?)?;
    let cfg = DflashConfig::read(snapshot)?;
    write_safetensors(&snapshot.join("model.safetensors"), &cfg.tensor_shapes().iter()
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    let shapes = [(cfg.hidden, 2 * cfg.intermediate), (cfg.intermediate, cfg.hidden),
        (cfg.hidden * cfg.taps.len(), cfg.hidden), (cfg.hidden, cfg.conv_width()),
        (cfg.hidden, cfg.rank), (cfg.hidden, cfg.qkv_width()), (cfg.hidden, 2 * cfg.kv_width()),
        (cfg.heads * cfg.head_dim, cfg.hidden)].into_iter().collect::<std::collections::BTreeSet<_>>();
    let entries = shapes.into_iter().enumerate().map(|(index, (k, n))| {
        fixture.draft_projection(&library, index.try_into()?, k.try_into()?, n.try_into()?)?;
        Ok(serde_json::json!({"name":format!("draft_bf16_k{k}_n{n}"),
            "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}], "capacity_rows":TAP_ROWS,
            "geometry":{"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
                "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}}))
    }).collect::<Result<Vec<_>>>()?;
    let manifest = snapshot.join("programs.json");
    std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
    let programs = library.programs()?.with_manifest(&manifest)?;
    let representation = GlmDraftRepresentation::Bf16Only;
    let requests = DflashDrafter::arena_requests(snapshot, representation, &[1], 0)?;
    // Deliberately outside the logical receipt prefix, as shared arena IDs are.
    let arena = DeviceWeightArena::new(device, WeightArenaLayout::new("glm/draft/1", requests[&1].clone())?, crate::shared::memory::TEST_ARENA)?;
    let first = arena.view(&requests[&1][0].0, requests[&1][0].1)?.buffer;
    let stream = Stream::new(device)?;
    let mut bound = Vec::new();
    let (drafter, records) = device.run(|| crate::shared::experts::rtx::dense::recording(||
        DflashDrafter::load_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, snapshot, Vec::new(), stream.raw, 2, 1,
            vec![0; cfg.hidden * 2], false, representation, fp8_linear::Fp8Scales::Amax,
            fp8_linear::Fp8Rows::W8a16, Qwen4KvCache::Bf16, |physical, name, bytes| {
                assert_eq!(physical, 1);
                let view = arena.view(name, bytes)?;
                bound.push((name.to_owned(), view.buffer.ptr as usize - first.ptr as usize, view.buffer.bytes));
                Ok(view)
            })))?;
    assert!(!records.operands.is_empty());
    assert_arena_bindings(arena.layout(), &requests[&1], &bound);
    drop(arena);
    let devices = CoordinatorDevices::new(&[1])?;
    let before = drafter.allocation_receipts(devices, "draft")?;
    let alias = before.iter().find(|receipt| receipt.address == first.ptr as u64
        && receipt.allocation.view.as_ref().is_some_and(|view| view.offset == 0))
        .context("offset-zero draft weight alias missing")?;
    let expected = format!("{}.upload", alias.allocation.id);
    device.run(|| fixture.reserve_ordered_upload(&library, first, stream.raw as usize))?;
    let after = drafter.allocation_receipts(devices, "draft")?;
    let uploads = after.iter().filter(|receipt| receipt.allocation.id.ends_with(".upload"))
        .collect::<Vec<_>>();
    assert_eq!(uploads.len(), 1);
    assert_eq!(uploads[0].allocation.id, expected);
    assert_eq!(uploads[0].allocation.location, Location::PINNED);
    assert_eq!(uploads[0].allocation.bytes, first.bytes as u64);
    assert_eq!(after.len(), before.len() + 1);
    assert_eq!(after.iter().filter(|receipt| receipt.allocation.id == "glm/draft/1").count(), 1);
    let drafter = crate::families::glm5::drafter::GlmDrafter::Local(drafter);
    let family = drafter.allocation_receipts(devices, "draft")?;
    assert_glm_draft_lifetimes(&after, &family);
    drop(drafter);
    stream.drain()?;
    fixture.finish_ordered_uploads(&library)?;
    Ok(())
}

#[test]
fn installed_standard_dflash_physical_layout_equals_serving_receipts() -> Result<()> {
    // Isolate the process-wide fixture ledger from other tests.
    const CHILD: &str = "CUTEAFD_DFLASH_PHYSICAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["shared::speculation::dflash::physical_tests::installed_standard_dflash_physical_layout_equals_serving_receipts",
                "--exact", "--nocapture"])
            .env(CHILD, "1").output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        ensure!(output.status.success() && stdout.contains("1 passed; 0 failed"),
            "isolated physical equality failed: {stdout}\n{stderr}");
        println!("{stdout}");
        return Ok(());
    }
    let mut arms = 0;
    let root = std::env::var_os("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/glm-dflash-physical-components/fixtures"));
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 0, 96usize << 30)?;
    for repo in ["incoai/GLM-5.3-DFlash2", "incoai/GLM-5.3-Flash-DFlash2"] {
        let snapshot = installed_snapshot(repo)?;
        let cfg = DflashConfig::read(&snapshot)?;
        let shapes: std::collections::BTreeSet<_> = [
            (cfg.hidden, 2 * cfg.intermediate), (cfg.intermediate, cfg.hidden),
            (cfg.hidden * cfg.taps.len(), cfg.hidden), (cfg.hidden, cfg.conv_width()),
            (cfg.hidden, cfg.rank), (cfg.hidden, cfg.qkv_width()),
            (cfg.hidden, 2 * cfg.kv_width()), (cfg.heads * cfg.head_dim, cfg.hidden),
        ].into_iter().collect();
        let entries = shapes.into_iter().enumerate().map(|(index, (k, n))| {
            fixture.draft_projection(&library, index.try_into()?, k.try_into()?, n.try_into()?)?;
            Ok(serde_json::json!({"name": format!("draft_bf16_k{k}_n{n}"),
                "pointers": [{"name":"x"},{"name":"w"},{"name":"out"}], "capacity_rows": TAP_ROWS,
                "geometry": {"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
                    "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}}))
        }).collect::<Result<Vec<_>>>()?;
        let manifest = fixture.directory().join("draft-programs.json");
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        for (slots, sequences) in [(2, 1), (20, 16)] {
            for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                for mode in [fp8_linear::Fp8Rows::W8a16, fp8_linear::Fp8Rows::Wide, fp8_linear::Fp8Rows::W8a8] {
                    let capacity = GlmDraftCapacity::new(slots, sequences, cfg.block)?;
                    let forecast = GlmDflashPhysicalLayout::read(&snapshot, representation,
                        capacity, library.sm_count()?, mode.code() as u8, kv)?;
                    // Real loader, matrix pack destinations and workspace constructor; recording
                    // suppresses checkpoint payload reads, not any device owner allocations.
                    let planned = DflashDrafter::arena_requests(&snapshot, representation, &[0], 0)?;
                    let mut requested = BTreeMap::<i32, Vec<(String, usize)>>::new();
                    let (drafter, records) = crate::shared::experts::rtx::dense::recording(|| {
                        DflashDrafter::load_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, &snapshot, Vec::new(), std::ptr::null_mut(),
                            slots, sequences, vec![0; cfg.hidden * 2], false, representation,
                            fp8_linear::Fp8Scales::Amax, mode, kv, |device, name, bytes| {
                                assert_eq!(device, library.cuda_get_device()?);
                                requested.entry(device).or_default().push((name.to_owned(), bytes));
                                super::super::weights::standalone(&library, device, name, bytes)
                            })
                    })?;
                    assert_eq!(requested, planned, "TP1 exact immutable requests");
                    ensure!(!records.operands.is_empty(), "real weight load did not run");
                    ensure!(representation == GlmDraftRepresentation::Bf16Only || !records.steps.is_empty(),
                        "real FP8 packing destinations did not run");
                    // Serving retains one admitted-maximum workspace, including for C1 probes.
                    for rejected in [0, sequences.saturating_sub(1), sequences + 1] {
                        let error = match drafter.workspace(rejected) {
                            Err(error) => error,
                            Ok(_) => anyhow::bail!("accepted non-admitted workspace capacity {rejected}"),
                        };
                        ensure!(error.to_string().contains("must use admitted capacity"), "{error}");
                    }
                    let missing = forecast.owner_bytes("missing-owner").unwrap_err();
                    ensure!(missing.to_string().contains("missing-owner"), "{missing}");
                    *drafter.workspace.borrow_mut() = Some(drafter.workspace(sequences)?);
                    drafter.prime_metadata()?;
                    let actual: BTreeMap<_, _> = drafter.allocation_receipts(CoordinatorDevices::new(&[0])?, "draft")?
                        .into_iter().map(|receipt| (receipt.allocation.id.clone(), receipt.allocation)).collect();
                    let expected: BTreeMap<_, _> = forecast.allocations("draft", Location::coordinator(0))
                        .into_iter().map(|allocation| (allocation.id.clone(), allocation)).collect();
                    assert_eq!(actual, expected, "{repo}, slots={slots}, C={sequences}, {kv}, {mode:?}, {representation:?}");
                    assert_eq!(actual["draft.workspace.tokens"].category, cuteafd_core::memory_layout::Category::Drafter);
                    assert_eq!(actual["draft.workspace.tokens"].bytes, (sequences * cfg.drafts() * 4).max(256) as u64);
                    if representation == GlmDraftRepresentation::Fp8Only {
                        assert_eq!(actual["draft.fp8_workspace"].category, cuteafd_core::memory_layout::Category::Workspace);
                    }
                    assert!(forecast.resident_weights().0 > 0);
                    assert!(forecast.weight_owners().all(|owner| owner.location == Location::coordinator(0)));
                    drop(drafter);
                    fixture.finish_ordered_uploads(&library)?;
                    assert_eq!(fixture.mapped_bytes(&library)?, 0, "virtual device mapping leaked");
                    arms += 1;
                }
                }
            }
        }
    }
    assert_eq!(arms, 48);
    println!("{arms} real-load arms; no retained virtual mappings");
    Ok(())
}

#[test]
fn installed_tp2_dflash_physical_layout_equals_both_rank_receipts() -> Result<()> {
    use crate::shared::{memory::device::{Device, Stream}, peer_split::{PeerExchange, RankDevice}};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    for device in 0..2 { fixture.configure_memory(&library, device, 96usize << 30)?; }
    for repo in ["incoai/GLM-5.3-DFlash2", "incoai/GLM-5.3-Flash-DFlash2", "nvidia/Muse-Spark-Open-816B-A42B-DFlash2"] {
        let snapshot = installed_snapshot(repo)?;
        let cfg = DflashConfig::read(&snapshot)?;
        let local = cfg.for_tp(2)?;
        assert_eq!((local.heads * 2, local.kv_heads * 2, local.intermediate * 2),
            (cfg.heads, cfg.kv_heads, cfg.intermediate));
        let shapes: std::collections::BTreeSet<_> = [(local.hidden, 2 * local.intermediate),
            (local.intermediate, local.hidden), (local.hidden * local.taps.len(), local.hidden),
            (local.hidden, local.conv_width()), (local.hidden, local.rank),
            (local.hidden, local.qkv_width()), (local.hidden, 2 * local.kv_width()),
            (local.heads * local.head_dim, local.hidden)].into_iter().collect();
        let entries = shapes.into_iter().enumerate().map(|(index, (k, n))| {
            fixture.draft_projection(&library, index.try_into()?, k.try_into()?, n.try_into()?)?;
            Ok(serde_json::json!({"name": format!("draft_bf16_k{k}_n{n}"),
                "pointers": [{"name":"x"},{"name":"w"},{"name":"out"}], "capacity_rows": TAP_ROWS,
                "geometry": {"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
                    "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}}))
        }).collect::<Result<Vec<_>>>()?;
        let manifest = fixture.directory().join("draft-programs.json");
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let devices = [0, 1].map(|id| Device { library: &library, id });
        let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let exchange = Rc::new(PeerExchange::new(&library,
            [0, 1].map(|rank| RankDevice { device: rank as i32, stream: streams[rank].raw }),
            1, 65536, 65536)?);
        for lead in 0..2 {
            library.cuda_set_device(lead as i32)?;
            for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                    let capacity = GlmDraftCapacity::new(20, 16, cfg.block)?;
                    let layouts = cuteafd_loader::placement::families::glm5_flash::drafter_tp2_inventory(
                        &snapshot, capacity, &[library.sm_count()? as u64; 2], representation,
                        fp8_linear::Fp8Rows::Wide.code() as u8, kv)?;
                    let planned = DflashDrafter::arena_requests(&snapshot, representation, &[0, 1], lead)?;
                    let mut requested = BTreeMap::<i32, Vec<(String, usize)>>::new();
                    let (drafter, _) = crate::shared::experts::rtx::dense::recording(|| DflashDrafter::load_tp2_on_with(
                        &library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, &snapshot, Vec::new(), 20, 16, vec![0; cfg.hidden * 2],
                        true, representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::Wide,
                        kv, exchange.clone(), lead, |device, name, bytes| {
                            assert_eq!(device, library.cuda_get_device()?);
                            requested.entry(device).or_default().push((name.to_owned(), bytes));
                            super::super::weights::standalone(&library, device, name, bytes)
                        }))?;
                    assert_eq!(requested, planned, "{repo}: TP2 head rank {lead} exact immutable requests");
                    *drafter.workspace.borrow_mut() = Some(drafter.workspace(16)?);
                    drafter.prime_metadata()?;
                    let peer = &drafter.head_split.as_ref().context("missing head split")?.peer;
                    peer.device.run(|| {
                        *peer.workspace.borrow_mut() = Some(peer.workspace(16)?);
                        peer.prime_metadata()
                    })?;
                    let actual: BTreeMap<_, _> = drafter.allocation_receipts(CoordinatorDevices::new(&[0, 1])?, "draft")?
                        .into_iter().map(|receipt| (receipt.allocation.id.clone(), receipt.allocation)).collect();
                    let expected: BTreeMap<_, _> = layouts.iter().enumerate().flat_map(|(rank, layout)|
                        layout.allocations(if rank == 0 { "draft" } else { "draft.peer" }, Location::coordinator((rank ^ lead) as u8)))
                        .map(|allocation| (allocation.id.clone(), allocation)).collect();
                    assert_eq!(actual, expected, "{representation:?}, {kv}");
                    assert_eq!(actual["draft.workspace.tokens"].category, cuteafd_core::memory_layout::Category::Drafter);
                    assert_eq!(actual["draft.workspace.tokens"].location, Location::coordinator(lead as u8));
                    assert!(!actual.contains_key("draft.peer.workspace.tokens"));
                    if representation == GlmDraftRepresentation::Fp8Only {
                        for name in ["draft.fp8_workspace", "draft.peer.fp8_workspace"] {
                            assert_eq!(actual[name].category, cuteafd_core::memory_layout::Category::Workspace);
                        }
                    }
                    assert!(matches!(peer.head_ops, DraftHeadOps::Peer));
                    assert!(peer.workspace.borrow().as_ref().unwrap().head.is_none());
                    assert!(layouts[1].owners().iter().all(|owner| !owner.id.starts_with("head.")));
                    assert!(layouts[0].resident_weights().0 > layouts[1].resident_weights().0);
                    assert!(layouts[1].owner_bytes("workspace.logits").is_err());
                    assert!(layouts[1].owner_bytes("workspace.head").is_err());
                    let full = GlmDflashPhysicalLayout::read(&snapshot, representation, capacity,
                        library.sm_count()?, fp8_linear::Fp8Rows::Wide.code() as u8, kv)?;
                    assert!(layouts[0].resident_weights().0 < full.resident_weights().0);
                    assert!(2 * layouts[0].resident_weights().0 > full.resident_weights().0,
                        "replicated physical weights must not disappear into opaque halving");
                    drop(drafter);
                    fixture.finish_ordered_uploads(&library)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires sealed draft exports, native library and two RTX GPUs"]
fn native_tp2_gqa_cross_owner_restore_is_byte_exact() -> Result<()> {
    use crate::shared::{memory::device::{Device, Stream}, peer_split::{PeerExchange, RankDevice}};
    let snapshot = installed_snapshot("incoai/GLM-5.3-Flash-DFlash2")?;
    // SAFETY: the native-test runner supplies the sealed library; all borrowers drop first.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let manifest = PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
    let programs = library.programs()?.with_manifest(&manifest)?;
    let devices = [0, 1].map(|id| Device { library: &library, id });
    ensure!(crate::shared::peer_split::probed_device(&library, 0, Some(1))? == Some(1), "GQA restore proof needs P2P");
    let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
    // Deliberately small slots exercise tiled reductions and receive-slot reuse.
    let exchange = Rc::new(PeerExchange::new_abortable(&library,
        [0, 1].map(|rank| RankDevice { device: rank as i32, stream: streams[rank].raw }),
        1, 65536, 65536)?);
    let cfg = DflashConfig::read(&snapshot)?;
    for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
        for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
            let draft = DflashDrafter::load_tp2(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, &snapshot,
                std::fs::read(snapshot.join("model.safetensors"))?, 2, 2, vec![0; cfg.hidden * 2],
                true, representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16,
                kv, exchange.clone())?;
            draft.prepare_workspace()?;
            let split = draft.head_split.as_ref().context("missing head split")?;
            let ranks = [&draft, &*split.peer];
            let first = 4093;
            let count = 83;
            let taps = (0..count * cfg.taps.len() * cfg.hidden).flat_map(|i| {
                let value = ((i * 17 % 127) as f32 - 63.0) / 64.0;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect::<Vec<_>>();
            library.copy_h2d(draft_view(draft.taps.buffer, 0, taps.len())?, &taps)?;
            let rows = |slot| (0..count).map(|tap_row| ContextRow { tap_row, slot,
                position: first + tap_row }).collect::<Vec<_>>();
            draft.update(&rows(0))?;
            draft.drain_mla()?;
            let mut captured = Vec::new();
            for (rank, owner) in ranks.iter().enumerate() {
                devices[rank].run(|| {
                    for layer in &owner.layers {
                        let attention = layer.attention.gqa()?;
                        for ring in [&attention.k_ring, &attention.v_ring] {
                            let bytes = ring.buffer.bytes / 2;
                            let mut snapshot = vec![0; bytes];
                            library.copy_d2h_quiescent(&mut snapshot, draft_view(ring.buffer, 0, bytes)?)?;
                            // Restore to another sequence's physical ring on each owner, then
                            // poison the old owner. A fresh projection must recreate those bytes.
                            library.copy_h2d(draft_view(ring.buffer, bytes, bytes)?, &snapshot)?;
                            library.cuda_zero_bytes(draft_view(ring.buffer, 0, bytes)?, bytes)?;
                            captured.push((rank, ring.buffer, snapshot));
                        }
                    }
                    Ok(())
                })?;
            }
            draft.update(&rows(0))?;
            draft.drain_mla()?;
            for (rank, ring, snapshot) in captured {
                devices[rank].run(|| {
                    let bytes = snapshot.len();
                    for offset in [0, bytes] {
                        let mut actual = vec![0; bytes];
                        library.copy_d2h_quiescent(&mut actual, draft_view(ring, offset, bytes)?)?;
                        assert_eq!(actual, snapshot, "rank {rank}, {representation:?}, {kv}, offset {offset}");
                    }
                    Ok(())
                })?;
            }
            let input = |_: &[u32], output: CuteafdDeviceBuffer, stream| {
                // SAFETY: the admitted draft workspace owns every output byte.
                unsafe { library.cuda_zero_bytes_async(output, output.bytes, stream) }
            };
            let head = |_: *const c_void, output: *mut f32, rows: usize, stream| {
                let bytes = rows * cfg.vocab * 4;
                // SAFETY: deterministic synthetic head fills the admitted output extent.
                unsafe { library.cuda_zero_bytes_async(CuteafdDeviceBuffer { ptr: output.cast(), bytes,
                    ..Default::default() }, bytes, stream) }
            };
            let sequences = [0, 1].map(|slot| DraftSeq { slot, anchor: cfg.mask_token,
                position: first + count, valid_from: first });
            let result = draft.draft_device_with(&sequences, &input, TargetHead::Launch(Box::new(head)))?;
            assert_eq!(result[0].tokens, result[1].tokens);
            let workspace = draft.workspace.borrow();
            let work = workspace.as_ref().context("missing proof workspace")?;
            let bytes = cfg.block * cfg.hidden * 2;
            let mut hidden = vec![0; 2 * bytes];
            library.copy_d2h_quiescent(&mut hidden, draft_view(work.n.buffer, 0, 2 * bytes)?)?;
            assert_eq!(&hidden[..bytes], &hidden[bytes..], "restored draft hidden rows differ");
            println!("TP2 GQA cross-owner exact restore: {representation:?}, {kv}; {} layers on both GPUs", cfg.layers);
        }
    }
    Ok(())
}

fn assert_arena_bindings(layout: &cuteafd_loader::serving_capacity::WeightArenaLayout,
    requests: &[(String, usize)], bound: &[(String, usize, usize)]) {
    let mut end = 0usize;
    let mut names = std::collections::BTreeSet::new();
    assert_eq!(requests.len(), bound.len());
    assert_eq!(requests.len(), layout.views().len());
    for ((name, bytes), (actual, offset, extent)) in requests.iter().zip(bound) {
        let aligned = end.div_ceil(256) * 256;
        assert!(names.insert(name));
        assert_eq!((actual, *offset, *extent), (name, aligned, *bytes));
        let view = layout.view(name).expect("planned view");
        assert_eq!((view.offset(), view.bytes()), (aligned, *bytes));
        assert!(*offset >= end && *offset - end < 256, "only alignment gaps are allowed");
        end = offset + extent;
    }
    assert_eq!(end, layout.bytes(), "no unexplained trailing gap");
}

#[test]
fn installed_kimi_mla_projection_recipe_matches_native_manifest() -> Result<()> {
    let cfg = cuteafd_loader::formats::dflash::MlaDflashConfig::read(
        &installed_snapshot("lightseekorg/kimi-k3-dflash2")?)?;
    let path = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/placement-inputs/current/share/PROGRAMS.json"));
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let shapes = std::cell::RefCell::new(std::collections::BTreeSet::new());
    super::super::weights::mla_weight_recipe(&cfg, |_| Ok(()), |parts, k| {
        shapes.borrow_mut().insert((k, parts.iter().map(|(_, n)| *n).sum::<usize>()));
        Ok(())
    }, |_| Ok(((), ())))?;
    assert_eq!(shapes.borrow().len(), 5);
    for (k, n) in shapes.into_inner() {
        cuteafd_ffi::draft_projection::validate_package_geometry(&manifest, k, n, TAP_ROWS)?;
    }
    Ok(())
}

/// Full production MLA geometry, sparse checkpoint and allocation-only DSO.
/// Exercises the real loader, not a second test-only traversal or numerical oracle.
pub(crate) fn with_mla_arena_fixture(verify: impl FnMut(&DflashDrafter<'_>,
    &cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout,
    &cuteafd_loader::serving_capacity::WeightArenaLayout) -> Result<()>) -> Result<()> {
    with_mla_arena_fixture_precision(false, verify)
}

fn with_mla_arena_fixture_precision(complete: bool, mut verify: impl FnMut(&DflashDrafter<'_>,
    &cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout,
    &cuteafd_loader::serving_capacity::WeightArenaLayout) -> Result<()>) -> Result<()> {
    with_mla_arena_fixture_config(complete, false, verify)
}

fn with_mla_arena_fixture_config(complete: bool, windows_only: bool, mut verify: impl FnMut(&DflashDrafter<'_>,
    &cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout,
    &cuteafd_loader::serving_capacity::WeightArenaLayout) -> Result<()>) -> Result<()> {
    use cuteafd_loader::{formats::dflash::MlaDflashConfig,
        plan::testing::{t, write_safetensors}, serving_capacity::WeightArenaLayout};
    use crate::shared::memory::{device::Stream, weight_arena::DeviceWeightArena};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 1, 96usize << 30)?;
    let device = Device { library: &library, id: 1 };
    let root = tempfile::tempdir_in(fixture.directory())?;
    let snapshot = root.path();
    std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&serde_json::json!({
        "architectures":["DFlash2DraftModel"], "hidden_size":7168,
        "intermediate_size":128, "num_hidden_layers":2, "num_attention_heads":64,
        "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
        "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
        "rms_norm_eps":1e-5, "vocab_size":256, "draft_vocab_size":256,
        "max_position_embeddings":65536, "num_target_layers":93,
        "layer_types":if windows_only { vec!["sliding_attention"; 2] } else { vec!["sliding_attention","full_attention"] }, "sliding_window":4096,
        "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
        "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
        "rope_parameters":{"rope_type":"yarn", "rope_theta":50000., "factor":32.,
            "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
            "mscale":1., "mscale_all_dim":1.},
        "dflash_config":{"attention_mode":"mla", "block_size":8,
            "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
            "selector_top_k":16, "target_layer_ids":[0,1], "mask_token_id":0}
    }))?)?;
    let cfg = MlaDflashConfig::read(snapshot)?;
    write_safetensors(&snapshot.join("model.safetensors"), &cfg.tensor_shapes().iter()
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    for cache in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
        for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
            if complete && representation != GlmDraftRepresentation::Fp8Only { continue; }
            let layout = if complete {
                cfg.complete_fp8_runtime_layout(GlmDraftCapacity::new(2,1,8)?,8192,16384,library.sm_count()?,0,cache)?
            } else { cfg.runtime_layout(representation, GlmDraftCapacity::new(2, 1, 8)?,
                8192, 16384, library.sm_count()?, fp8_linear::Fp8Rows::W8a16.code() as u8, cache)? };
            let p = layout.program();
            let shapes = [(7168, 256), (128, 7168), (14336, 7168), (7168, 1792), (7168, 256)]
                .into_iter().collect::<std::collections::BTreeSet<_>>();
            let mut entries = shapes.into_iter().map(|(k,n)| serde_json::json!({
                "name":format!("draft_bf16_k{k}_n{n}"), "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
                "scalars":[{"name":"rows","type":"int32"}], "scratch_bytes_at_capacity":{},
                "capacity_rows":TAP_ROWS, "geometry":{"width":k,"outputs":n,
                    "output_dtype":"torch.bfloat16", "reduction":"ordered_k16", "tiles_by_rows":[[16,1,4,16,64,6]]}
            })).collect::<Vec<_>>();
            for (index, suffix, pointers, scalars, window) in [
                (0,"producer",vec!["x","kv_slots","positions","inv_freq","w_qkv_a","q_a_norm","kv_a_norm",
                    "w_q_b","w_uk","kv_cache","query","q_resid","scratch"],"fi",None),
                (1,"attention_full",vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"],"iilif",None),
                (1,"attention_w4096",vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"],"iilif",Some(4096)),
                (2,"output",vec!["attn","w_uv","w_o","out","scratch"],"i",None),
            ] {
                let mut geometry = serde_json::json!({"name":"mla_block_draft", "heads":64,"hidden":7168,
                    "q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_dim":128,"qk_rope_dim":64,
                    "v_head_dim":128,"page_rows":64,"norm_eps":1e-5,"window_size":window,"kv_cache":cache.label()});
                match index {
                    0 => { geometry["rotary_style"] = "interleaved".into(); geometry["rotary_dim"] = 64.into(); }
                    1 => {
                        for (key,value) in [("max_context",p.context),("num_splits",p.splits),
                            ("chunks_per_split",p.chunks_per_split),("partial_output_bytes",p.partial_output_bytes),
                            ("partial_lse_bytes",p.partial_lse_bytes),("scratch_bytes",p.scratch[1])] { geometry[key] = value.into(); }
                        geometry["noncausal_block"] = true.into();
                    }
                    _ => geometry["output_gate"] = false.into(),
                }
                let (suffix, pointers) = if complete && index != 1 {
                    geometry["weight_dtype"] = "fp8".into();
                    geometry["projection_scale_k"] = 128.into();
                    geometry["batched_scale_k"] = 64.into();
                    if index == 0 {
                        ("producer_fp8_weights", vec!["x","kv_slots","positions","inv_freq","w_qkv_a","s_qkv_a",
                            "q_a_norm","kv_a_norm","w_q_b","s_q_b","w_uk","s_uk","kv_cache","query","q_resid","scratch"])
                    } else { ("output_fp8_weights", vec!["attn","w_uv","s_uv","w_o","s_o","out","scratch"]) }
                } else { (suffix, pointers) };
                let cache_suffix = if index == 2 { "" } else { cache.program_suffix() };
                entries.push(serde_json::json!({"name":format!("kimik3_mla_draft_{suffix}{cache_suffix}_m64"),
                    "pointers":pointers.iter().map(|name| serde_json::json!({"name":name})).collect::<Vec<_>>(),
                    "scalars":scalars.bytes().enumerate().map(|(i,kind)| serde_json::json!({"name":format!("s{i}"),
                        "type":match kind { b'i'=>"int32", b'l'=>"int64", _=>"float32" }})).collect::<Vec<_>>(),
                    "capacity_rows":64,"geometry":geometry,"scratch_bytes_at_capacity":{"scratch":p.scratch[index]}}));
            }
            fixture.register_programs(&library, &entries)?;
            let manifest = snapshot.join("programs.json");
            std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
            let programs = library.programs()?.with_manifest(&manifest)?;
            let requests = if complete {
                std::collections::BTreeMap::from([(1, cuteafd_loader::families::glm5::draft_physical::mla_complete_fp8_arena_requests(&cfg)?)])
            } else { DflashDrafter::mla_arena_requests(snapshot, representation, 1)? };
            assert_eq!(layout.weight_arena_requests(), requests[&1]);
            if !complete { assert_eq!(cuteafd_loader::families::glm5::draft_physical::mla_arena_requests(
                &cfg, representation)?, requests[&1]); }
            let planned = WeightArenaLayout::new("kimi/draft/1", requests[&1].clone())?;
            assert_eq!(planned.payload_bytes() as u64, layout.weights().resident_bytes()?);
            assert_eq!(planned.padding_bytes(), 0);
            let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
            let base = arena.view(&requests[&1][0].0, requests[&1][0].1)?.buffer.ptr as usize;
            let stream = Stream::new(device)?;
            let mut bound = Vec::new();
            let (drafter, records) = device.run(|| crate::shared::experts::rtx::dense::recording(||
                DflashDrafter::load_mla_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, "kimik3", snapshot, stream.raw,
                    layout.clone(), representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16, None,
                    |physical, name, bytes| {
                        assert_eq!(physical, 1);
                        let view = arena.view(name, bytes)?;
                        bound.push((name.to_owned(), view.buffer.ptr as usize - base, view.buffer.bytes));
                        Ok(view)
                    })))?;
            assert!(!records.operands.is_empty());
            if complete {
                let mut names = std::collections::BTreeSet::new();
                assert_eq!(bound.len(), planned.views().len());
                for (name, offset, bytes) in &bound {
                    assert!(names.insert(name));
                    let view = planned.view(name).context("complete MLA binder requested an unplanned owner")?;
                    assert_eq!((*offset,*bytes),(view.offset(),view.bytes()));
                }
                assert!(records.steps.iter().any(|step| step.jobs.iter().any(|job|
                    matches!(job,crate::shared::experts::rtx::dense::Job::AbsorbQuant { .. }))));
                assert!(records.steps.iter().all(|step|step.bytes()<=crate::shared::experts::rtx::dense::STEP_BYTES));
            } else { assert_arena_bindings(&planned, &requests[&1], &bound); }
            drop(arena);
            device.run(|| verify(&drafter, &layout, &planned))?;
            stream.drain()?;
            drop(drafter);
            fixture.finish_ordered_uploads(&library)?;
            assert_eq!(fixture.mapped_bytes(&library)?, 0);
        }
    }
    let tensors = cfg.tensor_shapes();
    write_safetensors(&snapshot.join("model.safetensors"), &tensors.iter().skip(1)
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    assert!(DflashDrafter::mla_arena_requests(snapshot, GlmDraftRepresentation::Bf16Only, 1).is_err());
    write_safetensors(&snapshot.join("model.safetensors"), &tensors.iter()
        .map(|(name, shape)| t(name, "F32", shape)).collect::<Vec<_>>());
    assert!(DflashDrafter::mla_arena_requests(snapshot, GlmDraftRepresentation::Bf16Only, 1).is_err());
    Ok(())
}

#[test]
fn prompt_window_snapshot_restores_exact_ring_bytes_and_committed_frontier() -> Result<()> {
    use crate::shared::memory::{device::Allocation, TEST};
    // The sparse CPU DSO's device extents are host-mapped bytes. Its synchronous
    // copy primitive writes those bytes even though numerical kernels are disabled.
    with_mla_arena_fixture_config(true, true, |draft, layout, _| {
        assert!(draft.mla_windows_only());
        assert_eq!(draft.mla_draft_page_bytes(), 0);
        draft.reset_context(0, 0)?;
        let DraftContext::Mla(context) = &draft.context else { unreachable!() };
        let end = 1034;
        let expected_rings = {
            let mut state = context.state.borrow_mut();
            for first in (0..end).step_by(64) {
                let rows = (first..(first+64).min(end)).enumerate()
                    .map(|(tap_row,position)| ContextRow { tap_row,slot:0,position }).collect::<Vec<_>>();
                let (plan,reservations) = MlaContextUpdate::checked_paged(&mut state, layout, &rows)?;
                assert!(reservations.is_empty());
                state.rings = plan.next;
            }
            assert_eq!(state.rings.frontier(0)?, (0,end));
            assert_eq!(state.pool.frontier(state.bindings[0].unwrap())?, (0,0));
            state.rings.clone()
        };
        let device = crate::shared::memory::device::Device { library:draft.library,id:1 };
        let windows = draft.mla_window_buffers(0)?;
        let marks = windows.iter().map(|window| Allocation::new(device,window.bytes,TEST)).collect::<Result<Vec<_>>>()?;
        let buffers = marks.iter().map(|mark|mark.buffer).collect::<Vec<_>>();
        let bytes = windows.iter().enumerate().map(|(layer,window)|
            (0..window.bytes).map(|i|((i*17+layer*31)%251) as u8).collect::<Vec<_>>()).collect::<Vec<_>>();
        let write = |destination: CuteafdDeviceBuffer,data:&[u8]| draft.library.copy_d2d(destination,
            CuteafdDeviceBuffer { ptr:data.as_ptr().cast_mut().cast(),bytes:data.len(),device_id:1,flags:0 },data.len());
        for (&window,data) in windows.iter().zip(&bytes) { write(window,data)?; }
        draft.copy_mla_windows(0,&buffers,true)?;
        draft.drain_mla()?;
        // Simulate recycling the request slot after prompt-end snapshot capture.
        draft.reset_context(0, 7000)?;
        for &window in &windows { write(window,&vec![0xa5;window.bytes])?; }
        draft.copy_mla_windows(0,&buffers,false)?;
        draft.restore_mla_windows(0,end)?;
        for (&window,expected) in windows.iter().zip(&bytes) {
            let mut actual=vec![0;window.bytes];draft.library.copy_d2h_quiescent(&mut actual,window)?;
            assert_eq!(&actual,expected,"restored FP8/BF16 ring bytes differ");
        }
        let state=context.state.borrow();
        assert_eq!(state.rings.frontier(0)?,expected_rings.frontier(0)?);
        assert_eq!(state.rings,expected_rings);
        assert_eq!(state.pool.frontier(state.bindings[0].unwrap())?,(end,end));
        Ok(())
    })?;
    // Mixed/full MLA still owns page attachments; it must not enter window restore.
    with_mla_arena_fixture(|draft,_,_| {
        assert!(!draft.mla_windows_only());
        assert!(draft.mla_draft_page_bytes()>0);
        assert!(draft.restore_mla_windows(0,1034).is_err());
        Ok(())
    })
}

#[test]
fn complete_mla_fp8_arena_covers_actual_loader_scale_views_and_bounded_steps() -> Result<()> {
    with_mla_arena_fixture_precision(true, |drafter, layout, planned| {
        assert!(layout.complete_fp8_body());
        assert_eq!(layout.weights().bf16_values,0);
        assert_eq!(layout.weights().max_load_staging,crate::shared::experts::rtx::dense::STEP_BYTES as u64);
        assert!(drafter.layers.iter().all(|layer|matches!(&layer.attention,
            DraftAttention::Mla(attention) if attention.fp8_scales.is_some())));
        assert_eq!(planned.payload_bytes() as u64,layout.weights().resident_bytes()?);
        Ok(())
    })
}

#[test]
fn ml4_arena_requests_exactly_cover_real_loader_bindings() -> Result<()> {
    use cuteafd_loader::{formats::dflash::MlaDflashConfig,
        families::mistral_large4::{Ml4Config, dspark::Ml4DsparkConfig},
        plan::testing::{t, write_safetensors}, serving_capacity::WeightArenaLayout};
    use crate::shared::memory::{device::Stream, weight_arena::DeviceWeightArena};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 1, 96usize << 30)?;
    let device = Device { library: &library, id: 1 };
    let root = tempfile::tempdir_in(fixture.directory())?;
    let mut target = Ml4Config::from_hf(&serde_json::from_str(include_str!(
        "../../../../cuteafd-loader/src/families/mistral_large4/official-config.json"))?)?;
    target.vocab_size = 256;
    target.dense_intermediate = 128;
    let native = MlaDflashConfig::ml4(&Ml4DsparkConfig { layers: 3, depth: 3,
        taps: vec![1, 18, 36, 54, 71], window: 2048, markov_rank: 512 }, &target)?;
    let path = root.path().join("model.safetensors");
    write_safetensors(&path, &native.tensor_shapes().iter()
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    let source = super::super::weights::Source::open(root.path())?;
    let layout = native.runtime_layout(GlmDraftRepresentation::Bf16Only,
        GlmDraftCapacity::new(2, 1, 3)?, 8192, 16384, library.sm_count()?, 0,
        cuteafd_core::MlaCacheFormat::Fp8)?;
    let p = layout.program();
    let mut entries = [(6144, 256), (128, 6144)].into_iter().map(|(k, n)| serde_json::json!({
        "name":format!("draft_bf16_k{k}_n{n}"), "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
        "scalars":[{"name":"rows","type":"int32"}], "scratch_bytes_at_capacity":{},
        "capacity_rows":TAP_ROWS, "geometry":{"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
            "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}
    })).collect::<Vec<_>>();
    entries.push(serde_json::json!({"name":"ml4_draft_context_fc", "capacity_rows":2048,
        "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
        "scalars":[{"name":"rows","type":"int32"}], "geometry":{"width":30720,"outputs":6144,
            "output_dtype":"torch.bfloat16","skinny_rows":0,"exact_row_cap":i32::MAX}}));
    for (index, suffix, pointers, scalars) in [
        (0, "producer", vec!["x","kv_slots","positions","cos_sin","unit_s","unit_b","w_q","w_kv_a",
            "kv_a_norm","w_uk","kv_cache","query","scratch"], "i"),
        (1, "attention", vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"], "iilif"),
        (2, "output", vec!["attn","w_uv","w_o","out","scratch"], "i"),
    ] {
        let mut geometry = serde_json::json!({"name":"ml4_draft","heads":64,"hidden":6144,
            "q_lora_rank":0,"kv_lora_rank":512,"qk_nope_dim":64,"qk_rope_dim":64,
            "v_head_dim":128,"page_rows":64,"window_size":2048,"norm_eps":1e-5,"kv_cache":"fp8"});
        if index == 1 {
            for (key, value) in [("max_context",p.context),("num_splits",p.splits),
                ("chunks_per_split",p.chunks_per_split),("partial_output_bytes",p.partial_output_bytes),
                ("partial_lse_bytes",p.partial_lse_bytes),("scratch_bytes",p.scratch[1])] { geometry[key] = value.into(); }
            geometry["noncausal_block"] = true.into(); geometry["block_window"] = true.into();
        }
        if index == 2 { geometry["output_gate"] = false.into(); }
        entries.push(serde_json::json!({"name":format!("ml4_draft_{suffix}_m64"),"capacity_rows":64,
            "pointers":pointers.iter().map(|name| serde_json::json!({"name":name})).collect::<Vec<_>>(),
            "scalars":scalars.bytes().enumerate().map(|(i, kind)| serde_json::json!({"name":format!("s{i}"),
                "type":match kind { b'i'=>"int32", b'l'=>"int64", _=>"float32" }})).collect::<Vec<_>>(),
            "geometry":geometry,"scratch_bytes_at_capacity":{"scratch":p.scratch[index]}}));
    }
    fixture.register_programs(&library, &entries)?;
    let manifest = root.path().join("programs.json");
    std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
    let programs = library.programs()?.with_manifest(&manifest)?;
    let requests = layout.immutable_arena_requests();
    let planned = WeightArenaLayout::new("glm/draft/1", requests.clone())?;
    let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
    let base = arena.view(&requests[0].0, requests[0].1)?.buffer.ptr as usize;
    let stream = Stream::new(device)?;
    let mut bound = Vec::new();
    let (drafter, records) = device.run(|| crate::shared::experts::rtx::dense::recording(||
        DflashDrafter::load_ml4_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, native.clone(), &target,
            target.logit_softcap as f32, stream.raw, layout.clone(), &|name, shape| source.rect(name, shape),
            |physical, name, bytes| {
                assert_eq!(physical, 1);
                let view = arena.view(name, bytes)?;
                bound.push((name.to_owned(), view.buffer.ptr as usize - base, view.buffer.bytes));
                Ok(view)
            })))?;
    assert!(!records.operands.is_empty());
    assert_arena_bindings(&planned, &requests, &bound);
    assert!(bound.iter().any(|(name, _, _)| name == "markov_norms"));
    drop(arena);
    let devices = CoordinatorDevices::new(&[1])?;
    let common = drafter.allocation_receipts(devices, "draft")?;
    let drafter = crate::families::glm5::drafter::GlmDrafter::Local(drafter);
    let receipts = drafter.allocation_receipts(devices, "draft")?;
    assert_glm_draft_lifetimes(&common, &receipts);
    let backing = receipts.iter().filter(|receipt| receipt.allocation.id == planned.id()).collect::<Vec<_>>();
    assert_eq!(backing.len(), 1);
    assert_eq!(backing[0].allocation.bytes, planned.bytes() as u64);
    assert_eq!(receipts.iter().filter(|receipt| receipt.allocation.view.as_ref()
        .is_some_and(|view| view.backing == planned.id())).count(), requests.len());
    stream.drain()?;
    drop(drafter);
    fixture.finish_ordered_uploads(&library)?;
    assert_eq!(fixture.mapped_bytes(&library)?, 0);
    Ok(())
}

#[test]
fn mla_arena_requests_exactly_cover_real_loader_bindings() -> Result<()> {
    with_mla_arena_fixture(|drafter, _, planned| {
        let mut census = crate::shared::memory::receipts::Census::default();
        census.owners(drafter.allocation_receipts(CoordinatorDevices::new(&[1])?, "draft")?);
        census.normalize(cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT)?;
        let root = census.receipts.iter().find(|receipt| receipt.allocation.id == planned.id()).context("draft root")?;
        assert_eq!(root.allocation.bytes, planned.bytes() as u64);
        assert_eq!(root.allocation.lifetime.first, cuteafd_loader::placement::planner::Phase::DraftLoad);
        let aliases = census.receipts.iter().filter(|receipt| receipt.allocation.view.as_ref()
            .is_some_and(|view| view.backing == planned.id())).collect::<Vec<_>>();
        assert_eq!(aliases.len(), planned.views().len());
        assert_eq!(aliases.iter().map(|receipt| receipt.allocation.bytes).sum::<u64>(), root.allocation.bytes);
        Ok(())
    })
}

#[test]
fn mla_mount_and_push_bound_device_staging_to_step_bytes_and_free_before_return() -> Result<()> {
    use cuteafd_core::memory_layout::MemoryKind;
    use cuteafd_ffi::memory_ledger::{AllocationAudit, AllocationEvent, Space};
    use cuteafd_loader::{formats::dflash::MlaDflashConfig, plan::testing::{t, write_safetensors},
        serving_capacity::WeightArenaLayout};
    use crate::shared::{experts::rtx::dense::{self, STEP_BYTES},
        memory::{device::Stream, weight_arena::DeviceWeightArena}};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 1, 96usize << 30)?;
    let device = Device { library: &library, id: 1 };
    let root = tempfile::tempdir_in(fixture.directory())?;
    let snapshot = root.path();
    // 2 taps -> fc.weight is [7168, 14336] BF16 = 196 MiB >> STEP_BYTES (32 MiB).
    std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&serde_json::json!({
        "architectures":["DFlash2DraftModel"], "hidden_size":7168,
        "intermediate_size":128, "num_hidden_layers":2, "num_attention_heads":64,
        "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
        "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
        "rms_norm_eps":1e-5, "vocab_size":256, "draft_vocab_size":256,
        "max_position_embeddings":65536, "num_target_layers":93,
        "layer_types":["sliding_attention","full_attention"], "sliding_window":4096,
        "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
        "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
        "rope_parameters":{"rope_type":"yarn", "rope_theta":50000., "factor":32.,
            "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
            "mscale":1., "mscale_all_dim":1.},
        "dflash_config":{"attention_mode":"mla", "block_size":8,
            "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
            "selector_top_k":16, "target_layer_ids":[0,1], "mask_token_id":0}
    }))?)?;
    let cfg = MlaDflashConfig::read(snapshot)?;
    write_safetensors(&snapshot.join("model.safetensors"), &cfg.tensor_shapes().iter()
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    let cache = cuteafd_core::MlaCacheFormat::Fp8;
    let peak_staging = |events: &[AllocationEvent], space: Space, owner: &str| -> (usize, usize) {
        let (mut cur, mut peak, mut allocs) = (0usize, 0usize, 0usize);
        for event in events {
            match event {
                AllocationEvent::Allocated(rec) if rec.key.space == space
                    && rec.key.tag.kind == MemoryKind::Staging && rec.key.tag.owner == owner => {
                    cur += rec.bytes;
                    peak = peak.max(cur);
                    allocs += 1;
                }
                AllocationEvent::Freed(rec) if rec.key.space == space
                    && rec.key.tag.kind == MemoryKind::Staging && rec.key.tag.owner == owner => {
                    cur = cur.saturating_sub(rec.bytes);
                }
                _ => {}
            }
        }
        (peak, allocs)
    };
    for (label, representation, complete) in [
        ("partial-fp8", GlmDraftRepresentation::Fp8Only, false),
        ("complete-fp8", GlmDraftRepresentation::Fp8Only, true),
        ("bf16-only", GlmDraftRepresentation::Bf16Only, false),
    ] {
        let layout = if complete {
            cfg.complete_fp8_runtime_layout(GlmDraftCapacity::new(2, 1, 8)?, 8192, 16384, library.sm_count()?, 0, cache)?
        } else {
            cfg.runtime_layout(representation, GlmDraftCapacity::new(2, 1, 8)?, 8192, 16384,
                library.sm_count()?, fp8_linear::Fp8Rows::W8a16.code() as u8, cache)?
        };
        let p = layout.program();
        let shapes = [(7168, 256), (128, 7168), (14336, 7168), (7168, 1792)]
            .into_iter().collect::<std::collections::BTreeSet<_>>();
        let mut entries = shapes.into_iter().map(|(k, n)| serde_json::json!({
            "name":format!("draft_bf16_k{k}_n{n}"), "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
            "scalars":[{"name":"rows","type":"int32"}], "scratch_bytes_at_capacity":{},
            "capacity_rows":TAP_ROWS, "geometry":{"width":k,"outputs":n,
                "output_dtype":"torch.bfloat16", "reduction":"ordered_k16", "tiles_by_rows":[[16,1,4,16,64,6]]}
        })).collect::<Vec<_>>();
        for (index, suffix, pointers, scalars, window) in [
            (0,"producer",vec!["x","kv_slots","positions","inv_freq","w_qkv_a","q_a_norm","kv_a_norm",
                "w_q_b","w_uk","kv_cache","query","q_resid","scratch"],"fi",None),
            (1,"attention_full",vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"],"iilif",None),
            (1,"attention_w4096",vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"],"iilif",Some(4096)),
            (2,"output",vec!["attn","w_uv","w_o","out","scratch"],"i",None),
        ] {
            let mut geometry = serde_json::json!({"name":"mla_block_draft", "heads":64,"hidden":7168,
                "q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_dim":128,"qk_rope_dim":64,
                "v_head_dim":128,"page_rows":64,"norm_eps":1e-5,"window_size":window,"kv_cache":cache.label()});
            match index {
                0 => { geometry["rotary_style"] = "interleaved".into(); geometry["rotary_dim"] = 64.into(); }
                1 => {
                    for (key, value) in [("max_context", p.context), ("num_splits", p.splits),
                        ("chunks_per_split", p.chunks_per_split), ("partial_output_bytes", p.partial_output_bytes),
                        ("partial_lse_bytes", p.partial_lse_bytes), ("scratch_bytes", p.scratch[1])] { geometry[key] = value.into(); }
                    geometry["noncausal_block"] = true.into();
                }
                _ => geometry["output_gate"] = false.into(),
            }
            let (suffix, pointers) = if complete && index != 1 {
                geometry["weight_dtype"] = "fp8".into();
                geometry["projection_scale_k"] = 128.into();
                geometry["batched_scale_k"] = 64.into();
                if index == 0 {
                    ("producer_fp8_weights", vec!["x","kv_slots","positions","inv_freq","w_qkv_a","s_qkv_a",
                        "q_a_norm","kv_a_norm","w_q_b","s_q_b","w_uk","s_uk","kv_cache","query","q_resid","scratch"])
                } else { ("output_fp8_weights", vec!["attn","w_uv","s_uv","w_o","s_o","out","scratch"]) }
            } else { (suffix, pointers) };
            let cache_suffix = if index == 2 { "" } else { cache.program_suffix() };
            entries.push(serde_json::json!({"name":format!("kimik3_mla_draft_{suffix}{cache_suffix}_m64"),
                "pointers":pointers.iter().map(|name| serde_json::json!({"name":name})).collect::<Vec<_>>(),
                "scalars":scalars.bytes().enumerate().map(|(i, kind)| serde_json::json!({"name":format!("s{i}"),
                    "type":match kind { b'i'=>"int32", b'l'=>"int64", _=>"float32" }})).collect::<Vec<_>>(),
                "capacity_rows":64,"geometry":geometry,"scratch_bytes_at_capacity":{"scratch":p.scratch[index]}}));
        }
        fixture.register_programs(&library, &entries)?;
        let manifest = snapshot.join("programs.json");
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let planned = WeightArenaLayout::new("kimi/draft/1", layout.weight_arena_requests())?;
        let stream = Stream::new(device)?;
        // 1. Mount mode: must stream through bounded STEP_BYTES device/pinned staging and free before return.
        let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
        let audit = AllocationAudit::begin();
        let mount_drafter = device.run(|| DflashDrafter::load_mla_with(&library, &programs,
            crate::shared::speculation::dflash::TargetProjection::Native, "kimik3", snapshot, stream.raw,
            layout.clone(), representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16, None,
            |_, name, bytes| arena.view(name, bytes)))?;
        let history = audit.finish();
        let expected_dev = if representation == GlmDraftRepresentation::Fp8Only { STEP_BYTES } else { 0 };
        let (dev_peak, dev_allocs) = peak_staging(&history.events, Space::Device,
            cuteafd_loader::shared::dense_read::DEVICE_STAGING_OWNER);
        let (full_src_peak, _) = peak_staging(&history.events, Space::Device, "shared/drafter-source");
        let (pinned_peak, _) = peak_staging(&history.events, Space::Pinned, "staging/coordinator-dense");
        assert_eq!(full_src_peak, 0, "{label} mount must never allocate full source matrix on device");
        assert_eq!(dev_peak, expected_dev, "{label} mount device staging peak");
        assert_eq!(dev_allocs, usize::from(expected_dev > 0), "{label} mount device staging allocated once and reused");
        assert!(pinned_peak > 0 && pinned_peak <= STEP_BYTES, "{label} mount pinned peak {pinned_peak}");
        assert!(!history.final_allocations.iter().any(|rec|
            matches!(rec.key.tag.owner, "staging/coordinator-dense-device" | "staging/coordinator-dense" | "shared/drafter-source")),
            "{label} mount must free transient staging before load_mla_with returns");
        drop(arena);
        stream.drain()?;
        drop(mount_drafter);
        // 2. Push mode (recording): records bounded steps (<= STEP_BYTES) and allocates 0 device staging.
        let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
        let audit = AllocationAudit::begin();
        let (push_drafter, records) = device.run(|| dense::recording(||
            DflashDrafter::load_mla_with(&library, &programs,
                crate::shared::speculation::dflash::TargetProjection::Native, "kimik3", snapshot, stream.raw,
                layout.clone(), representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16, None,
                |_, name, bytes| arena.view(name, bytes))))?;
        let history = audit.finish();
        assert!(records.steps.iter().all(|step| step.bytes() <= STEP_BYTES), "{label} push step exceeds STEP_BYTES");
        assert_eq!(peak_staging(&history.events, Space::Device,
            cuteafd_loader::shared::dense_read::DEVICE_STAGING_OWNER).0, 0, "{label} push recording allocates 0 device staging");
        assert_eq!(peak_staging(&history.events, Space::Device, "shared/drafter-source").0, 0);
        if representation == GlmDraftRepresentation::Fp8Only {
            assert!(records.steps.iter().any(|step| step.jobs.iter().any(|j| matches!(j, dense::Job::W8a16Pack { .. }))));
        } else {
            assert!(records.steps.iter().all(|step| step.jobs.iter().all(|j| matches!(j, dense::Job::Absorb { .. }))));
        }
        drop(arena);
        stream.drain()?;
        drop(push_drafter);
        fixture.finish_ordered_uploads(&library)?;
        assert_eq!(fixture.mapped_bytes(&library)?, 0);
    }
    Ok(())
}

#[test]
#[ignore = "requires sealed native library, programs manifest, real MLA drafter snapshot and one RTX GPU via golden-run --native-test"]
fn mla_drafter_load_peak_and_weight_hash_on_gpu() -> Result<()> {
    use cuteafd_core::memory_layout::MemoryKind;
    use cuteafd_ffi::memory_ledger::{AllocationAudit, AllocationEvent, Space};
    use cuteafd_loader::{formats::dflash::MlaDflashConfig, serving_capacity::WeightArenaLayout};
    use crate::shared::{experts::rtx::dense::{self, STEP_BYTES},
        memory::{device::Stream, weight_arena::DeviceWeightArena}};
    use sha2::{Digest, Sha256};

    let env = |name| std::env::var(name).with_context(|| format!("MLA drafter GPU check needs {name}"));
    let snapshot = std::path::PathBuf::from(env("CUTEAFD_KIMI_DRAFT_SNAPSHOT")?);
    let native_lib = std::path::PathBuf::from(env("CUTEAFD_NATIVE_LIB")?);
    let manifest = std::path::PathBuf::from(env("CUTEAFD_PROGRAM_MANIFEST")?);
    // SAFETY: golden-run supplies the sealed library and every borrower drops before it.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(&native_lib)? };
    let programs = library.programs()?.with_manifest(&manifest)?;
    let device = Device { library: &library, id: 0 };
    let stream = Stream::new(device)?;
    let native = MlaDflashConfig::read(&snapshot)?;
    let namespace = if native.kind == cuteafd_core::coordinator_programs::MlaDraftKind::CausalDflash2 {
        "kimik31"
    } else {
        "kimik3"
    };
    let sms = device.run(|| library.sm_count())?;
    let cap = GlmDraftCapacity::new(2, 1, native.block)?;
    let cache = cuteafd_core::MlaCacheFormat::Fp8;

    let hash_arena = |arena: &DeviceWeightArena<'_>, planned: &WeightArenaLayout| -> Result<String> {
        let mut views: Vec<_> = planned.views().iter().collect();
        views.sort_by_key(|(_, v)| v.offset());
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 32 << 20];
        device.run(|| -> Result<()> {
            for (name, view) in views {
                let alloc = arena.view(name, view.bytes())?;
                let mut offset = 0usize;
                while offset < view.bytes() {
                    let n = (view.bytes() - offset).min(buf.len());
                    let slice = CuteafdDeviceBuffer {
                        ptr: unsafe { alloc.buffer.ptr.cast::<u8>().add(offset) }.cast(),
                        bytes: n,
                        ..alloc.buffer
                    };
                    library.copy_d2h_quiescent(&mut buf[..n], slice)?;
                    hasher.update(&buf[..n]);
                    offset += n;
                }
            }
            Ok(())
        })?;
        Ok(format!("{:x}", hasher.finalize()))
    };

    let staging_peaks = |events: &[AllocationEvent]| -> (usize, usize, usize, usize) {
        let peak_of = |space: Space, owner: &str| -> usize {
            let (mut cur, mut peak) = (0usize, 0usize);
            for ev in events {
                match ev {
                    AllocationEvent::Allocated(r) if r.key.space == space
                        && r.key.tag.kind == MemoryKind::Staging && r.key.tag.owner == owner => {
                        cur += r.bytes;
                        peak = peak.max(cur);
                    }
                    AllocationEvent::Freed(r) if r.key.space == space
                        && r.key.tag.kind == MemoryKind::Staging && r.key.tag.owner == owner => {
                        cur = cur.saturating_sub(r.bytes);
                    }
                    _ => {}
                }
            }
            peak
        };
        let (mut cur_dev, mut peak_dev) = (0usize, 0usize);
        for ev in events {
            match ev {
                AllocationEvent::Allocated(r) if r.key.space == Space::Device
                    && r.key.tag.kind == MemoryKind::Staging => {
                    cur_dev += r.bytes;
                    peak_dev = peak_dev.max(cur_dev);
                }
                AllocationEvent::Freed(r) if r.key.space == Space::Device
                    && r.key.tag.kind == MemoryKind::Staging => {
                    cur_dev = cur_dev.saturating_sub(r.bytes);
                }
                _ => {}
            }
        }
        (
            peak_dev,
            peak_of(Space::Device, "shared/drafter-source"),
            peak_of(Space::Device, cuteafd_loader::shared::dense_read::DEVICE_STAGING_OWNER),
            peak_of(Space::Pinned, "staging/coordinator-dense"),
        )
    };

    // Untimed warm-up (CUDA kernels + page cache) on complete-fp8.
    {
        let layout = native.complete_fp8_runtime_layout(cap, 8192, 16384, sms, 0, cache)?;
        let planned = WeightArenaLayout::new("kimi/draft/warm", layout.weight_arena_requests())?;
        let arena = DeviceWeightArena::new(device, planned, crate::shared::memory::TEST_ARENA)?;
        let warm = device.run(|| DflashDrafter::load_mla_with(&library, &programs,
            crate::shared::speculation::dflash::TargetProjection::Native, namespace, &snapshot, stream.raw,
            layout, GlmDraftRepresentation::Fp8Only, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16, None,
            |_, name, bytes| arena.view(name, bytes)))?;
        stream.drain()?;
        drop(warm);
        drop(arena);
    }

    for (label, representation, complete) in [
        ("complete-fp8", GlmDraftRepresentation::Fp8Only, true),
        ("partial-fp8", GlmDraftRepresentation::Fp8Only, false),
        ("bf16-only", GlmDraftRepresentation::Bf16Only, false),
    ] {
        let layout = if complete {
            native.complete_fp8_runtime_layout(cap, 8192, 16384, sms, 0, cache)?
        } else {
            native.runtime_layout(representation, cap, 8192, 16384, sms,
                fp8_linear::Fp8Rows::W8a16.code() as u8, cache)?
        };
        let planned = WeightArenaLayout::new("kimi/draft/0", layout.weight_arena_requests())?;
        let resident_gb = planned.bytes() as f64 / 1e9;
        let expected_dev = if representation == GlmDraftRepresentation::Fp8Only { STEP_BYTES } else { 0 };

        // 1. Legacy dev mount path (std::fs::read + Load::with_prefetched full-matrix staging).
        let (legacy_hash, legacy_s, legacy_peaks, legacy_used_delta) = {
            let (free_before, total) = device.run(|| library.cuda_memory_info())?;
            let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
            let audit = AllocationAudit::begin();
            let t0 = std::time::Instant::now();
            let file = std::fs::read(snapshot.join("model.safetensors"))?;
            let weights = device.run(|| -> Result<_> {
                let mut alloc = |_: i32, name: &str, bytes: usize| arena.view(name, bytes);
                let load = super::super::weights::Load::with_prefetched(&library, &snapshot, file)?
                    .with_allocator(&mut alloc);
                use super::super::weights::MlaMatrix;
                let w = cuteafd_loader::families::glm5::draft_physical::mla_weight_recipe_with_attention(&native,
                    |parts| load.tensor_parts(parts),
                    |parts, cols| load.matrix(parts, cols, representation == GlmDraftRepresentation::Fp8Only,
                        fp8_linear::Fp8Scales::Amax, stream.raw),
                    |parts, cols| if complete {
                        load.mla_fp8_matrix(parts, cols, fp8_linear::Fp8Scales::Amax, stream.raw).map(MlaMatrix::Fp8)
                    } else { load.concat(parts, cols).map(MlaMatrix::Bf16) },
                    |name| if complete {
                        let (uk, uv) = load.mla_fp8_absorb(name, native.mla.heads, native.mla.nope,
                            native.mla.value, native.mla.kv_rank, fp8_linear::Fp8Scales::Amax, stream.raw)?;
                        Ok((MlaMatrix::Fp8(uk), MlaMatrix::Fp8(uv)))
                    } else {
                        let (uk, uv) = load.absorb(name, native.mla.heads, native.mla.nope,
                            native.mla.value, native.mla.kv_rank, stream.raw)?;
                        Ok((MlaMatrix::Bf16(uk), MlaMatrix::Bf16(uv)))
                    })?;
                drop(load);
                stream.drain()?;
                Ok(w)
            })?;
            let elapsed = t0.elapsed().as_secs_f64();
            let history = audit.finish();
            let (free_after, _) = device.run(|| library.cuda_memory_info())?;
            let hash = hash_arena(&arena, &planned)?;
            drop(weights);
            drop(arena);
            (hash, elapsed, staging_peaks(&history.events), (total - free_after).saturating_sub(total - free_before))
        };

        // 2. HEAD bounded mount path (DflashDrafter::load_mla_with via Load::open).
        let (mount_hash, mount_s, mount_peaks, mount_used_delta) = {
            let (free_before, total) = device.run(|| library.cuda_memory_info())?;
            let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
            let audit = AllocationAudit::begin();
            let t0 = std::time::Instant::now();
            let drafter = device.run(|| DflashDrafter::load_mla_with(&library, &programs,
                crate::shared::speculation::dflash::TargetProjection::Native, namespace, &snapshot, stream.raw,
                layout.clone(), representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16, None,
                |_, name, bytes| arena.view(name, bytes)))?;
            stream.drain()?;
            let elapsed = t0.elapsed().as_secs_f64();
            let history = audit.finish();
            let (free_after, _) = device.run(|| library.cuda_memory_info())?;
            assert!(!history.final_allocations.iter().any(|r|
                matches!(r.key.tag.owner, "staging/coordinator-dense-device" | "staging/coordinator-dense" | "shared/drafter-source")),
                "{label}: transient staging still live after load_mla_with returned");
            let hash = hash_arena(&arena, &planned)?;
            drop(drafter);
            drop(arena);
            (hash, elapsed, staging_peaks(&history.events), (total - free_after).saturating_sub(total - free_before))
        };

        // 3. HEAD push path (dense::recording + dense::testing::replay).
        let (push_hash, push_s, push_peaks) = {
            let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
            let audit = AllocationAudit::begin();
            let t0 = std::time::Instant::now();
            let (drafter, records) = device.run(|| dense::recording(||
                DflashDrafter::load_mla_with(&library, &programs,
                    crate::shared::speculation::dflash::TargetProjection::Native, namespace, &snapshot, stream.raw,
                    layout.clone(), representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16, None,
                    |_, name, bytes| arena.view(name, bytes))))?;
            let rank_layout = dense::RankLayout::new(records.operands, records.steps, device.id)?;
            let (push_root, _) = cuteafd_loader::push_load::namespace(&snapshot)?;
            dense::testing::replay(device, rank_layout, &push_root, 1, 0)?;
            stream.drain()?;
            let elapsed = t0.elapsed().as_secs_f64();
            let history = audit.finish();
            let hash = hash_arena(&arena, &planned)?;
            drop(drafter);
            drop(arena);
            (hash, elapsed, staging_peaks(&history.events))
        };

        eprintln!(
            "MLA_GPU_RECEIPT mode={label} resident_bytes={} admitted_staging={} \
             legacy_dev_peak={} legacy_src_peak={} legacy_dense_peak={} legacy_pinned_peak={} legacy_s={:.3} legacy_s_per_gb={:.3} legacy_cuda_delta={} \
             mount_dev_peak={} mount_src_peak={} mount_dense_peak={} mount_pinned_peak={} mount_s={:.3} mount_s_per_gb={:.3} mount_cuda_delta={} \
             push_dev_peak={} push_dense_peak={} push_pinned_peak={} push_s={:.3} push_s_per_gb={:.3} \
             sha256_legacy={} sha256_mount={} sha256_push={}",
            planned.bytes(), layout.weights().max_load_staging,
            legacy_peaks.0, legacy_peaks.1, legacy_peaks.2, legacy_peaks.3, legacy_s, legacy_s / resident_gb, legacy_used_delta,
            mount_peaks.0, mount_peaks.1, mount_peaks.2, mount_peaks.3, mount_s, mount_s / resident_gb, mount_used_delta,
            push_peaks.0, push_peaks.2, push_peaks.3, push_s, push_s / resident_gb,
            legacy_hash, mount_hash, push_hash
        );
        assert_eq!(layout.weights().max_load_staging, expected_dev as u64, "{label} admitted staging");
        assert_eq!(mount_peaks.1, 0, "{label} HEAD mount must not allocate shared/drafter-source");
        assert_eq!(mount_peaks.0, expected_dev, "{label} HEAD mount device staging must equal admitted bound");
        assert!(mount_peaks.3 <= STEP_BYTES, "{label} HEAD mount pinned staging <= STEP_BYTES");
        assert_eq!(push_peaks.0, expected_dev, "{label} HEAD push device staging must equal admitted bound");
        assert_eq!(mount_hash, legacy_hash, "{label} HEAD mount weights must be byte-exact vs legacy dev");
        assert_eq!(push_hash, legacy_hash, "{label} HEAD push weights must be byte-exact vs legacy dev");
    }
    Ok(())
}
