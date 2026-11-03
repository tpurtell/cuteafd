use super::super::weight_arena::{ArenaLayout, ExpectedCoverage};
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::push_load::{
    builder::{Builder, Destination, Source},
    proto::Request,
    Conn,
};
use std::{
    os::fd::AsRawFd,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

struct SocketRank(Conn);
impl RankEvents for SocketRank {
    fn fd(&self) -> Option<std::os::fd::BorrowedFd<'_>> { Some(self.0.fd()) }
    fn next(&self) -> std::io::Result<Event> {
        self.0
            .event()?
            .ok_or_else(|| std::io::Error::other("closed"))
    }
    fn try_next(&self) -> std::io::Result<Poll> {
        self.0.try_event()
    }
    fn release(&self, slot: u32) -> std::io::Result<()> {
        self.0.request(&Request::Release { slot }, None)
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Good,
    Delayed,
    Missing,
    Malformed,
    Failed,
    Closed,
    DrainFailed,
    Reused,
}

fn exercise(window: usize, mode: Mode, bytes: usize) -> Result<(usize, f64)> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device {
        library: &library,
        id: 0,
    };
    let dir = tempfile::tempdir()?;
    use cuteafd_loader::{
        formats::fp8_experts::{Fp8ExpertTensors, Fp8Projection, Slicing},
        plan::testing::{fp8, t, write_snapshot},
        RoutedExpertShape,
    };
    use std::os::unix::fs::FileExt;
    // Header-backed TP4 down-column plan: source rows are pitched, destination
    // rows are packed. Compare with the existing mount loader, not Pattern::runs.
    let hidden = bytes / 1024;
    let mut headers = Vec::new();
    for (projection, n, k) in [
        ("gate", 4096, hidden),
        ("up", 4096, hidden),
        ("down", hidden, 4096),
    ] {
        headers.extend(fp8(
            &format!("model.layers.0.mlp.experts.0.{projection}_proj"),
            n,
            k,
            None,
        ));
    }
    headers.push(t("validation.mcg", "I32", &[]));
    write_snapshot(
        dir.path(),
        &serde_json::json!({"model_type": "fixture"}),
        &headers,
        None,
    );
    let path = dir.path().join("model-00001-of-00001.safetensors");
    let metadata = cuteafd_loader::read_safetensors_metadata(&path)?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)?;
    for tensor in &metadata {
        let data = (0..tensor.byte_length)
            .map(|i| ((i + tensor.byte_offset) % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all_at(&data, tensor.byte_offset)?;
    }
    let tail = metadata
        .iter()
        .find(|t| t.name == "validation.mcg")
        .unwrap();
    file.write_all_at(&0xcbac1fedu32.to_le_bytes(), tail.byte_offset)?;
    let tensors = Fp8ExpertTensors::read_layers(
        dir.path(),
        RoutedExpertShape {
            layers: 1,
            first_layer: 0,
            experts: 1,
            topk: 1,
            hidden,
            intermediate: 4096,
            draft_stages: 0,
            draft_experts: 0,
        },
        &[0].into(),
    )?;
    let projection = Fp8Projection::Down;
    let (weights, scales) = tensors.slice_bytes_with(projection, 4, 1, Slicing::Padded)?;
    assert_eq!(weights, bytes);
    let device_bytes = weights + scales;
    let mut expected = vec![0; device_bytes];
    let (w, s) = expected.split_at_mut(weights);
    tensors.read_slice_with(
        0,
        0,
        projection,
        4,
        1,
        Slicing::Padded,
        w,
        s,
        &mut Vec::new(),
    )?;
    let mut builder = Builder::new(dir.path(), vec![(device_bytes + 4) as u64]);
    builder.fp8_projection(
        &tensors,
        0,
        0,
        projection,
        4,
        1,
        Slicing::Padded,
        Destination {
            rank: 0,
            offset: 0,
            bytes: weights as u64,
        },
        Destination {
            rank: 0,
            offset: weights as u64,
            bytes: scales as u64,
        },
    )?;
    builder.contiguous(
        &Source {
            path: path.clone(),
            offset: tail.byte_offset,
            bytes: 4,
        },
        0,
        4,
        Destination {
            rank: 0,
            offset: device_bytes as u64,
            bytes: 4,
        },
    )?;
    let source = std::fs::read(&path)?;
    let plan = builder.finish()?;
    let footprint = ExpectedCoverage::from_plan(&plan, 0)?;
    let arena = LoadingArena::new(
        device,
        ArenaLayout::new([device_bytes], device_bytes)?.with_validation(4, 4)?,
        footprint,
    )?;
    let ring = RegisteredRing::new(&library, 8, 1024 * 1024)?;
    let fd = ring.fd().as_raw_fd();
    let (client, server) = Conn::pair()?;
    client.set_timeout(Some(Duration::from_secs(10)))?;
    server.set_timeout(Some(Duration::from_secs(10)))?;
    let released = AtomicUsize::new(0);
    let (prefilled, ready) = std::sync::mpsc::channel();
    if matches!(mode, Mode::Delayed) { fixture.block_first_stream(&library, true)?; }
    let started = Instant::now();
    let result = std::thread::scope(|scope| -> Result<_> {
        let released = &released;
        let fixture = &fixture;
        let library = &library;
        let task = scope.spawn(move || -> Result<()> {
            let slots = 8usize;
            let slot_bytes = 1024 * 1024;
            // SAFETY: fd is retained by consume's ring until server disconnect; own mapping.
            let mapped = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    slots * slot_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                )
            };
            ensure!(mapped != libc::MAP_FAILED, "fake ring mapping failed");
            let run = (|| -> Result<()> {
                let mut slots_data = Vec::new();
                let mut packer = slot::Packer::new(vec![0; slot_bytes]);
                let mut flush = |p: slot::Packer| {
                    slots_data.push(p.finish().0);
                };
                for entry in &plan.plan.entries {
                    let pattern = &plan.plan.patterns[entry.pattern as usize];
                    for (src, dst) in pattern.runs() {
                        let mut data = &source[(entry.src_base + src) as usize
                            ..(entry.src_base + src + u64::from(pattern.run)) as usize];
                        let mut offset = entry.dst_base + dst;
                        while !data.is_empty() {
                            let n = packer.push(offset, data);
                            if n == 0 {
                                flush(packer);
                                packer = slot::Packer::new(vec![0; slot_bytes]);
                            }
                            offset += n as u64;
                            data = &data[n..];
                        }
                    }
                }
                if !packer.is_empty() {
                    flush(packer);
                }
                // A whole-file replay must not change coverage or data.
                if matches!(mode, Mode::Good | Mode::Delayed) {
                    let replay = slots_data.clone();
                    slots_data.extend(replay);
                }
                if matches!(mode, Mode::Missing) {
                    server.send_event(&Event::Done)?;
                    return Ok(());
                }
                let mut sent = 0;
                let mut inflight = std::collections::BTreeSet::new();
                while sent < slots_data.len() || !inflight.is_empty() {
                    while sent < slots_data.len() && inflight.len() < slots {
                        let index = (0..slots).find(|i| !inflight.contains(i)).unwrap();
                        let mut data = slots_data[sent].clone();
                        if matches!(mode, Mode::Malformed) {
                            data[slot_bytes - 16] ^= 1;
                        }
                        // SAFETY: only a free credit is written; bounds inside fake mapping.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                mapped.cast::<u8>().add(index * slot_bytes),
                                slot_bytes,
                            );
                        }
                        server.send_event(&Event::Filled {
                            slot: index as u32,
                            len: slot_bytes as u32,
                        })?;
                        inflight.insert(index);
                        sent += 1;
                        if matches!(mode, Mode::Good | Mode::Delayed) && sent == slots_data.len() {
                            // Real Done may arrive while copies and credits remain queued.
                            server.send_event(&Event::Done)?;
                        }
                        if matches!(mode, Mode::Good | Mode::Delayed) && sent == slots_data.len().min(slots) {
                            prefilled.send(())?;
                        }
                        if matches!(mode, Mode::Failed) {
                            server.send_event(&Event::Failed {
                                reason: "injected".into(),
                                missing: vec![],
                            })?;
                            return Ok(());
                        }
                        if matches!(mode, Mode::Closed) {
                            return Ok(());
                        }
                        if matches!(mode, Mode::Reused) {
                            server.send_event(&Event::Filled {
                                slot: index as u32,
                                len: slot_bytes as u32,
                            })?;
                            return Ok(());
                        }
                    }
                    let Some((request, _)) = server.next_request()? else {
                        return Ok(());
                    };
                    let Request::Release { slot } = request else {
                        anyhow::bail!("not a release");
                    };
                    ensure!(inflight.remove(&(slot as usize)), "duplicate release");
                    let events = fixture_events(&fixture)?;
                    let count = released.fetch_add(1, Ordering::Relaxed) + 1;
                    ensure!(events.matches('L').count() >= count,
                        "Release {count} preceded successful CUDA copy completion: {events}");
                    if matches!(mode, Mode::Delayed) && count <= 3 {
                        ensure!(slot != 0 && slot != 4,
                            "stalled stream released before three independent completions");
                        if count == 3 { fixture.block_first_stream(library, false)?; }
                    }
                    // Poison released storage; delayed source access must never see this.
                    // SAFETY: Release returned this credit to the producer.
                    unsafe {
                        std::ptr::write_bytes(
                            mapped.cast::<u8>().add(slot as usize * slot_bytes),
                            0xee,
                            slot_bytes,
                        );
                    }
                }
                if !matches!(mode, Mode::Good | Mode::Delayed) {
                    server.send_event(&Event::Done)?;
                }
                Ok(())
            })();
            // SAFETY: server stops writes and has no views after run.
            unsafe {
                libc::munmap(mapped, slots * slot_bytes);
            }
            run
        });
        if matches!(mode, Mode::DrainFailed) {
            fixture.configure_pack(&library, 0, 1)?;
        }
        if matches!(mode, Mode::Good | Mode::Delayed) {
            ready.recv_timeout(Duration::from_secs(10))?;
        }
        let value = consume(device, arena, ring, SocketRank(client), window);
        let server_result = task.join().expect("fake server panicked");
        if matches!(mode, Mode::Good | Mode::Delayed) {
            server_result?;
            let delivered = value?;
            assert_eq!(delivered.validation(), &0xcbac1fedu32.to_le_bytes());
            let buffer = delivered.region(0)?;
            // SAFETY: CPU fake DSO device storage is host malloc, delivery has drained.
            let actual =
                unsafe { std::slice::from_raw_parts(buffer.ptr.cast::<u8>(), device_bytes) };
            assert_eq!(actual, expected);
            drop(delivered);
        } else {
            let Err(error) = value else { anyhow::bail!("partial or invalid load published") };
            // The cause says what this rank received before failing.
            let cause = format!("{error:#}");
            ensure!(cause.contains("delivered") || cause.contains("drain"), "cause omits delivery: {cause}");
        }
        Ok(())
    });
    result?;
    let events = fixture.events()?;
    if matches!(mode, Mode::DrainFailed) {
        assert!(library.is_quarantined_after_failed_drain());
        assert!(!events.contains('g') && !events.contains('d'));
    } else {
        if matches!(mode, Mode::Good | Mode::Delayed) && window == 8 {
            assert_eq!(events.matches('V').count(), COPY_STREAMS);
            assert!(events.matches('R').count() > 1, "pipeline never queued multiple slot events");
        }
        assert!(events.rfind('S').unwrap() < events.rfind('g').unwrap());
        assert!(events.rfind('S').unwrap() < events.rfind('d').unwrap());
    }
    Ok((
        released.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64(),
    ))
}
fn fixture_events(fixture: &Fixture) -> Result<String> {
    fixture.events()
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn socket_fake_replays_checked_plan_into_registered_ring() -> Result<()> {
    for window in [1, 8] {
        exercise(window, Mode::Good, 2 * 1024 * 1024)?;
    }
    Ok(())
}
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn failures_drain_without_partial_publication() -> Result<()> {
    for mode in [
        Mode::Missing,
        Mode::Malformed,
        Mode::Failed,
        Mode::Closed,
        Mode::Reused,
        Mode::DrainFailed,
    ] {
        exercise(8, mode, 256 * 1024)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn delayed_stream_does_not_hold_other_slot_credits() -> Result<()> {
    // Three other streams must release (and permit source poisoning) while
    // stream zero is held. Final bytes and coverage still match mounted loading.
    exercise(8, Mode::Delayed, 8 * 1024 * 1024)?;
    Ok(())
}

// Explicit CPU throughput experiment, not a CUDA or RDMA performance gate.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn socket_fake_copy_out_throughput() -> Result<()> {
    for window in [1, 8] {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let bytes = 32 * 1024 * 1024;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("payload");
        std::fs::File::create(&path)?.set_len(bytes as u64)?;
        let mut builder = Builder::new(dir.path(), vec![bytes as u64]);
        builder.contiguous(
            &Source {
                path,
                offset: 0,
                bytes: bytes as u64,
            },
            0,
            bytes as u64,
            Destination {
                rank: 0,
                offset: 0,
                bytes: bytes as u64,
            },
        )?;
        let plan = builder.finish()?;
        let arena = LoadingArena::new(
            device,
            ArenaLayout::new([bytes], bytes)?,
            ExpectedCoverage::from_plan(&plan, 0)?,
        )?;
        let ring = RegisteredRing::new(&library, 128, 1024 * 1024)?;
        let fd = ring.fd().as_raw_fd();
        let (client, server) = Conn::pair()?;
        client.set_timeout(Some(Duration::from_secs(30)))?;
        server.set_timeout(Some(Duration::from_secs(30)))?;
        let payload = vec![0x5a; bytes];
        let mut templates = Vec::new();
        let mut offset = 0;
        while offset < bytes {
            let mut packer = slot::Packer::new(vec![0; 1024 * 1024]);
            let count = packer.push(offset as u64, &payload[offset..]);
            ensure!(count > 0, "benchmark slot cannot fit payload");
            templates.push(packer.finish().0);
            offset += count;
        }
        let measurement = std::thread::scope(|scope| -> Result<_> {
            let task = scope.spawn(move || -> Result<_> {
                let extent = 128 * 1024 * 1024;
                // SAFETY: ring owns fd until the producer has disconnected.
                let mapped = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        extent,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        fd,
                        0,
                    )
                };
                ensure!(mapped != libc::MAP_FAILED, "benchmark mapping failed");
                let run = (|| -> Result<_> {
                    let mut elapsed = 0.0;
                    for pass in 0..17 {
                        let start = Instant::now();
                        let mut sent = 0;
                        let mut inflight = std::collections::BTreeSet::new();
                        while sent < templates.len() || !inflight.is_empty() {
                            while sent < templates.len() && inflight.len() < 8 {
                                let index = (0..8).find(|i| !inflight.contains(i)).unwrap();
                                // SAFETY: producer owns this free credit, extent is checked.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        templates[sent].as_ptr(),
                                        mapped.cast::<u8>().add(index * 1024 * 1024),
                                        1024 * 1024,
                                    );
                                }
                                server.send_event(&Event::Filled {
                                    slot: index as u32,
                                    len: 1024 * 1024,
                                })?;
                                inflight.insert(index);
                                sent += 1;
                            }
                            let Some((Request::Release { slot }, _)) = server.next_request()?
                            else {
                                anyhow::bail!("benchmark missing Release");
                            };
                            ensure!(
                                inflight.remove(&(slot as usize)),
                                "benchmark duplicate Release"
                            );
                        }
                        // First complete arena pass faults in mappings and fake device pages.
                        if pass > 0 {
                            elapsed += start.elapsed().as_secs_f64();
                        }
                    }
                    server.send_event(&Event::Done)?;
                    Ok(elapsed)
                })();
                // SAFETY: no producer views or writes survive the run.
                unsafe {
                    libc::munmap(mapped, extent);
                }
                run
            });
            let delivered = consume(device, arena, ring, SocketRank(client), window)?;
            let elapsed = task.join().expect("benchmark server panicked")?;
            let region = delivered.region(0)?;
            // SAFETY: fake device storage is malloc and all events have drained.
            let actual = unsafe { std::slice::from_raw_parts(region.ptr.cast::<u8>(), bytes) };
            assert_eq!(actual, payload);
            Ok(elapsed)
        })?;
        let transferred = bytes * 16;
        eprintln!(
            "CPU_FAKE window={window} bytes={transferred} seconds={measurement:.6} GB/s={:.3}",
            transferred as f64 / measurement / 1e9
        );
    }
    Ok(())
}
