//! Hardware fault fixture: no model projections or synthetic numerical claims.
use super::*;
use crate::shared::memory::device::Stream;
use crate::shared::peer_split::TerminalState;
use crate::shared::token_io::{EmbedPlacement, EmbedSource};
use std::{mem::ManuallyDrop, time::Instant};

fn cfg() -> MimoV2Config {
    MimoV2Config {
        vocab_size: 1,
        hidden: 4096,
        layers: 0,
        heads: 64,
        full_kv_heads: 4,
        swa_kv_heads: 8,
        head_dim: 192,
        v_head_dim: 128,
        rope_dim: 64,
        full_rope_theta: 5000000.0,
        swa_rope_theta: 10000.0,
        window: 128,
        attention: vec![],
        full_sinks: false,
        swa_sinks: true,
        dense: vec![],
        dense_intermediate: 16384,
        experts: 256,
        topk: 8,
        moe_intermediate: 2048,
        routed_scale: 1.0,
        rms_norm_eps: 1e-5,
        v_scale: 0.707,
        router_fp32: true,
    }
}

fn run(label: &str, captured: bool, lanes: usize, injected: u32) -> Result<()> {
    run_case(label, captured, lanes, injected, false, false)
}

fn run_case(label: &str, captured: bool, lanes: usize, injected: u32, serving: bool, normal: bool) -> Result<()> {
    // SAFETY: this explicit fixture library owns all CUDA/verbs symbols and is
    // retained with the complete engine if drainage cannot be proven.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    let programs = library.programs()?;
    ensure!(
        programs.names().count() == 0,
        "fault fixture must carry no model AOT programs"
    );
    let lead = ManuallyDrop::new(Stream::new(Device {
        library: &library,
        id: 0,
    })?);
    let peer = ManuallyDrop::new(Stream::new(Device {
        library: &library,
        id: 1,
    })?);
    let output_dir = std::path::PathBuf::from(std::env::var("CUTEAFD_TERMINAL_FAULT_DIR")?);
    std::fs::create_dir_all(&output_dir)?;
    let path = output_dir.join(format!("{label}-embedding.bin"));
    std::fs::write(&path, vec![0u8; 8192])?;
    let (embedding, ()) = TokenEmbedding::load(
        &library,
        EmbedSource {
            path,
            offset: 0,
            vocab: 1,
            hidden: 4096,
            snapshot: None,
            tensor_name: "synthetic-terminal-fault-fixture".into(),
        },
        EmbedPlacement::Host,
        || Ok(()),
    )?;
    let weights = MimoWeights {
        layers: vec![],
        norm: Dev::new(&library, 256)?,
        head: super::super::head::MimoHead::Bf16(Dev::new(&library, 256)?),
        output_fp8: false,
    };
    let mut engine = ManuallyDrop::new(MimoEngine::new(
        &library,
        &programs,
        cfg(),
        weights,
        lead.raw,
        64,
        64,
        1,
        1,
        embedding,
        MimoKvCache::Int8,
        MimoPrefillOutput::AllRows,
    )?);
    let fixture_head = engine.weights.head.allocations()[0].buffer;
    // No target layer executes. Attach actual persistent peer slots/flags and
    // streams using the production constructor, with an empty peer layer list.
    engine.split_family = Some("mimo2");
    engine.attach_peer(1, peer.raw, vec![], false)?;
    engine.mtp_staging.borrow_mut().0.bytes_mut()[..256].fill(0x97);
    let exchange = engine.exchange()?;
    let control_bytes = exchange.fault_fixture_control(0)?.len();
    let mut owner_events = Vec::new();
    let mut copy_streams = [std::ptr::null_mut(); 4];
    let mut serving_owners = None;
    if serving {
        use cuteafd_engine::prefix::{PrefixCache, PrefixConfig, PrefixFamily};
        use cuteafd_hostcache::pool::PinnedMemory;
        use crate::shared::token_io::{DeviceLogits, RowSelect, SelectBatch, SelectPlacement, TokenSelector};
        let family = super::super::prefix::MimoPrefix::terminal_fixture(&engine)?;
        let mut copy = crate::shared::prefix::CudaCopyEngine::registered_owned(
            &library, family.host_owners())?;
        copy.allocate_chunk(65536)?;
        for (index, (device, stream)) in copy.terminal_fixture_streams().into_iter().enumerate() {
            ensure!(device == (index / 2) as i32, "registered fixture streams changed device order");
            copy_streams[index] = stream;
        }
        let cache = PrefixCache::new(family.layout(), PrefixConfig {
            entries: 1, mark_slots: 1, keep_logits: false, min_tokens: 1,
        }, Some((cuteafd_hostcache::config::Config {
            bytes: 65536, chunk_bytes: 65536, min_tokens: 1, max_tokens: 64,
            ..Default::default()
        }, copy)))?;
        let mut selector = TokenSelector::new(&library, SelectPlacement::Device, 1, 64)?;
        library.copy_h2d(fixture_head, &[0u8; 256])?;
        let logits = DeviceLogits { ptr: fixture_head.ptr.cast_const(), rows: 1,
            vocab: 1, stride: 1, stream: lead.raw, greedy: None };
        let mut batch = SelectBatch::default();
        batch.rows.push(RowSelect { sampling: cuteafd_core::TargetSamplingParams::new(0.7, 0.9, Some(1), 0.0, 42)?,
            position: 0, mask: Some(vec![1]), logprob: false });
        ensure!(selector.select(&logits, &batch)?.into_iter().all(|row| row.is_ok()), "fixture sampler initialization failed");
        for rank in 0..2 {
            owner_events.push(crate::shared::memory::device::Event::new(Device { library: &library, id: rank })?);
        }
        let mut owners = super::super::serving_owners::ServingOwners::new(&engine);
        owners.prefix(family, cache);
        owners.selector(selector);
        serving_owners = Some(owners);
    }
    let queue = |rank: usize| -> Result<()> {
        for lane in 0..lanes {
            exchange.wait(rank, lane * 4)?;
        }
        let destination = cuteafd_ffi::CuteafdDeviceBuffer {
            ptr: exchange.recv(rank, 2)?,
            bytes: 256,
            device_id: rank as i32,
            flags: 0,
        };
        engine.on(rank, || {
            // SAFETY: source is persistent engine-owned pinned staging,
            // destination is an engine-owned peer slot; both outlive drainage.
            unsafe {
                library.copy_host_buffer_h2d_async(
                    destination,
                    engine.mtp_staging.borrow().0.buffer,
                    256,
                    engine.stream_of(rank),
                )
            }
        })
    };
    if captured {
        for rank in 0..2 {
            let exec = engine.on(rank, || {
                // SAFETY: capture is confined to this owned stream; every
                // pointer is persistent engine storage, including abort words.
                unsafe {
                    library.cuda_graph_begin_capture(engine.stream_of(rank))?;
                }
                queue(rank)?;
                unsafe { library.cuda_graph_end_capture(engine.stream_of(rank)) }
            })?;
            let key = GraphKey {
                layer: 0,
                rows: lanes,
                table_stride: 1,
                previous: Previous::First,
                head: false,
            };
            let graph = GraphExec(exec, &library);
            if rank == 0 {
                engine.graphs.borrow_mut().insert(key, graph);
            } else {
                engine
                    .peer
                    .as_ref()
                    .unwrap()
                    .graphs
                    .borrow_mut()
                    .insert(key, graph);
            }
            engine.on(rank, || {
                // SAFETY: captured exec and bound owners remain in the engine.
                unsafe { library.cuda_graph_launch(exec, engine.stream_of(rank)) }
            })?;
        }
    } else {
        for rank in 0..2 {
            queue(rank)?;
        }
    }
    for rank in 0..2 {
        engine.on(rank, || {
            // SAFETY: query reads the live stream without blocking the owner.
            ensure!(
                !unsafe { library.cuda_stream_query(engine.stream_of(rank))? },
                "unmatched peer wait must actually remain queued"
            );
            Ok(())
        })?;
    }
    if let Some(owners) = &mut serving_owners {
        use cuteafd_hostcache::copy::{CopyEngine, DeviceRange, Stream as CopyStream};
        use cuteafd_hostcache::pool::HostRange;
        for (rank, event) in owner_events.iter_mut().enumerate() {
            event.record(if rank == 0 { &lead } else { &peer })?;
        }
        let events: Vec<_> = owner_events.iter().map(|event| (event.device.id, event.raw)).collect();
        let (family, cache, _) = owners.parts();
        let buffers = family.host_owners();
        let copy = cache.host_engine_mut().context("fixture host cache missing")?;
        copy.terminal_fixture_wait(&events)?;
        // Both host queues address actual callback-owned registered arenas and
        // a pinned host chunk, and remain queued behind the peer compute waits.
        for (rank, owner) in buffers.iter().enumerate() {
            let device = DeviceRange { addr: owner.buffer.ptr as u64, bytes: 256 };
            copy.d2h(CopyStream::Store, device, HostRange { chunk: 0, offset: rank * 512, bytes: 256 })?;
            copy.d2h(CopyStream::Restore, device, HostRange { chunk: 0, offset: rank * 512 + 256, bytes: 256 })?;
        }
    }
    library.terminal_fault_fixture_configure(injected, lead.raw, peer.raw)?;
    if serving { library.terminal_fault_fixture_copy_streams(copy_streams)?; }
    let start = Instant::now();
    let error = if normal {
        drop(serving_owners.take());
        ensure!(!engine.submission_failed(), "normal serving retirement latched execution failure");
        ensure!(super::super::body_after_shutdown(Ok(()), engine.submission_failed(), || engine.terminal_error()).is_ok(),
            "normal serving retirement became a terminal body error");
        None
    } else {
        let error = engine.submit::<()>(|| {
            if let Some(owners) = &mut serving_owners {
                use crate::shared::token_io::{DeviceLogits, RowSelect, SelectBatch};
                let (_, _, selector) = owners.parts();
                // Target work is genuinely queued; this post-target selection
                // failure must close the engine before serving may release it.
                let logits = DeviceLogits { ptr: fixture_head.ptr.cast_const(), rows: 65,
                    vocab: 1, stride: 1, stream: lead.raw, greedy: None };
                let mut batch = SelectBatch::default();
                for _ in 0..65 { batch.rows.push(RowSelect { sampling: cuteafd_core::TargetSamplingParams::default(),
                    position: 0, mask: Some(vec![1]), logprob: false }); }
                selector.select(&logits, &batch)?;
                anyhow::bail!("post-target selector capacity fault did not fire")
            }
            anyhow::bail!("injected failure after queued peer work")
        }).expect_err("fault submission must fail");
        drop(serving_owners.take());
        ensure!(engine.submission_failed(), "failed submission did not latch");
        ensure!(super::super::body_after_shutdown(Ok(()), engine.submission_failed(), || engine.terminal_error()).is_err(),
            "caller swallowed terminal execution failure");
        Some(error)
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let frees = library.terminal_fault_fixture_finish()?;
    let late_frees = if serving { library.terminal_fault_fixture_late_frees()? } else { [0, 0] };
    ensure!(
        elapsed_ms < 2000.0,
        "terminal fault took {elapsed_ms:.3} ms"
    );
    ensure!(
        frees == [0, 0],
        "device/pinned owners freed before shutdown result: {frees:?}"
    );
    if let Some(error) = &error {
        ensure!(error.downcast_ref::<TerminalStepError>().is_some(), "typed primary submission error lost: {error:#}");
    }
    if serving && injected == 0 {
        ensure!(late_frees[0] >= 3 && late_frees[1] >= 2, "serving owners were never released after drainage: {late_frees:?}");
    } else if serving {
        ensure!(late_frees == [0, 0], "quarantined callback storage was released: {late_frees:?}");
    }
    ensure!(
        engine.require_live().is_err(),
        "terminal engine was reusable"
    );
    let expected = if injected == 0 {
        TerminalState::Drained
    } else {
        TerminalState::Retained
    };
    ensure!(
        engine.terminal.get() == expected && exchange.terminal_state() == expected,
        "engine/exchange terminal owner state disagrees"
    );
    if injected != 0 {
        ensure!(
            engine.retain_queued_storage(),
            "failed publication/drain did not retain full engine"
        );
        // Fixture-only external release proves the leaked streams are no
        // longer blocked before process exit; engine state is never reopened.
        exchange.fault_fixture_external_drain()?;
        for (index, &stream) in copy_streams.iter().enumerate() {
            if stream.is_null() { continue; }
            library.cuda_set_device((index / 2) as i32)?;
            // SAFETY: test-only external cleanup of deliberately retained
            // callback copy streams; engine state is never made reusable.
            unsafe { library.cuda_stream_synchronize(stream)?; }
        }
        library.cuda_set_device(0)?;
        ensure!(engine.terminal.get() == TerminalState::Retained && engine.require_live().is_err());
    }
    for rank in 0..2 {
        engine.on(rank, || {
            // SAFETY: normal shutdown or explicit fixture drainage has completed.
            ensure!(
                unsafe { library.cuda_stream_query(engine.stream_of(rank))? },
                "rank {rank} not drained"
            );
            let mut marker = vec![0; 256];
            library.copy_d2h(
                &mut marker,
                cuteafd_ffi::CuteafdDeviceBuffer {
                    ptr: exchange.recv(rank, 2)?,
                    bytes: 256,
                    device_id: rank as i32,
                    flags: 0,
                },
            )?;
            ensure!(
                marker.iter().all(|&v| v == 0x97),
                "rank {rank} marker behind wait not completed"
            );
            let control = exchange.fault_fixture_control(rank)?;
            ensure!(
                control.len() == control_bytes && control.iter().all(|&v| v == 0),
                "abort rewrote push/wait sequence counters on rank {rank}"
            );
            Ok(())
        })?;
    }
    let record = serde_json::json!({ "case":label, "captured":captured, "lanes":lanes,
        "fault":injected, "elapsed_ms":elapsed_ms, "serving_owners":serving, "normal_retirement":normal,
        "post_target_selection_failure":serving && !normal, "failed_submission_latched":engine.submission_failed(),
        "late_device_frees":late_frees[0], "late_pinned_frees":late_frees[1], "early_device_frees":frees[0],
        "early_pinned_frees":frees[1], "state":format!("{expected:?}"),
        "both_streams_drained":true, "both_markers_exact":true, "sequence_words_unchanged":true,
        "reused":false, "model_layers":0, "real_qp_fault":false });
    std::fs::write(
        output_dir.join(format!("{label}.json")),
        serde_json::to_vec_pretty(&record)?,
    )?;
    eprintln!("terminal_fault={record}");
    // Retain the complete quarantined engine, bound graphs/native library and
    // streams until process exit, even after test-only external drain. Ordinary
    // successful fixtures also retain here to avoid adding a teardown oracle.
    // The empty guard option no longer owns callback buffers; explicitly end
    // its borrow before retaining the engine/library. Events have now retired.
    drop(serving_owners);
    drop(owner_events);
    std::mem::forget(engine);
    std::mem::forget(peer);
    std::mem::forget(lead);
    std::mem::forget(programs);
    std::mem::forget(library);
    Ok(())
}

#[test]
#[ignore = "requires component fixture native library and both RTX GPUs"]
fn terminal_fault_queued_peer() -> Result<()> {
    run("queued-peer", false, 1, 0)
}
#[test]
#[ignore = "requires component fixture native library and both RTX GPUs"]
fn terminal_fault_captured_peer() -> Result<()> {
    run("captured-peer", true, 1, 0)
}
#[test]
#[ignore = "requires component fixture native library and both RTX GPUs"]
fn terminal_fault_two_lanes() -> Result<()> {
    run("two-lanes", true, 2, 0)
}
#[test]
#[ignore = "requires component fixture native library and both RTX GPUs"]
fn terminal_fault_failed_publication() -> Result<()> {
    run("failed-publication", true, 2, 1)
}
#[test]
#[ignore = "requires component fixture native library and both RTX GPUs"]
fn terminal_fault_failed_drain() -> Result<()> {
    run("failed-drain", true, 2, 2)
}

#[test]
#[ignore = "requires serving-owner component native library and both RTX GPUs"]
fn terminal_fault_normal_serving_retirement() -> Result<()> {
    run_case("normal-serving-retirement", true, 2, 0, true, true)
}
#[test]
#[ignore = "requires serving-owner component native library and both RTX GPUs"]
fn terminal_fault_post_target_sampler() -> Result<()> {
    run_case("post-target-sampler", true, 2, 0, true, false)
}
#[test]
#[ignore = "requires serving-owner component native library and both RTX GPUs"]
fn terminal_fault_serving_failed_publication() -> Result<()> {
    run_case("serving-failed-publication", true, 2, 1, true, false)
}
#[test]
#[ignore = "requires serving-owner component native library and both RTX GPUs"]
fn terminal_fault_serving_failed_host_drain() -> Result<()> {
    run_case("serving-failed-host-drain", true, 2, 3, true, false)
}
