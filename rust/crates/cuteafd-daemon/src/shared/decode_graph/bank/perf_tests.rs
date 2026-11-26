//! Identical host-side workloads for baseline and candidate cache implementations.
use super::*;
use super::super::device::{Basis, Cycle, Preference, Site};
use std::{hint::black_box, sync::{Arc, Barrier}, time::Instant};

fn fixture(device: i32) -> Vec<GraphBank<u32, u32>> {
    (0..64).map(|_| {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(device, 1 << 40, Site { name: "perf", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 4096, basis: Basis::Bytes,
            spans_devices: false }).unwrap();
        for key in 0..128 {
            let ticket = bank.cache_ticket(&key, false, None).unwrap();
            bank.cache_commit(key, key, ticket, 8, 1, false);
        }
        bank
    }).collect()
}

#[test]
#[ignore = "optimized CPU cache microbenchmark; run explicitly with --nocapture"]
fn cache_hot_paths_nanoseconds() {
    // Two workers own disjoint banks sharing one device controller. No CUDA work.
    for threads in [1, 2] {
        let banks = fixture(-39000 - threads as i32);
        let chunks = banks.len() / threads;
        let barrier = Arc::new(Barrier::new(threads));
        let mut banks = banks;
        std::thread::scope(|scope| {
            for (worker, banks) in banks.chunks_mut(chunks).enumerate() {
                let barrier = barrier.clone();
                scope.spawn(move || {
                    for operation in ["get", "contains", "cached_launch", "cycle_switch", "capture_cancel"] {
                        let iterations = if operation == "cycle_switch" { 2_048 } else { 131_072 };
                        barrier.wait();
                        let started = Instant::now();
                        for step in 0..iterations {
                            let key = black_box((step % 128) as u32);
                            let bank = &mut banks[step % chunks];
                            match operation {
                                "get" => { black_box(bank.get(&key)); }
                                "contains" => { black_box(bank.contains(&key)); }
                                "cached_launch" => { black_box(bank.cached_launch(&key)); }
                                "cycle_switch" => bank.cache_cycle(if (step / chunks) % 2 == 0 { Cycle::Verify } else { Cycle::Decode }),
                                "capture_cancel" => { drop(black_box(bank.cache_ticket(&1024, false, None))); }
                                _ => unreachable!(),
                            }
                        }
                        let ns = started.elapsed().as_nanos() as f64 / iterations as f64;
                        println!("cache_perf threads={threads} worker={worker} records=8192 banks=64 op={operation} ns={ns:.2}");
                        barrier.wait();
                    }
                });
            }
        });
    }
}

#[test]
#[ignore = "optimized CPU bank_cycle microbenchmark; run explicitly with --nocapture"]
fn bank_cycle_flip_microseconds_600_vs_15000() -> anyhow::Result<()> {
    use super::super::GraphOwner;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory_usage(&library, 0, 256 << 20, 0)?;

    for (idx, live_execs) in [600u32, 15_000u32].into_iter().enumerate() {
        let device = -39_100 - idx as i32;
        let mut bank: GraphBank<u32, GraphOwner<'_, ()>> = GraphBank::new();
        bank.attach_device_for_test(device, 1 << 40, Site {
            name: "flip-bench", cycle: Cycle::Decode, preference: Preference::Capture,
            bootstrap: 4096, basis: Basis::Bytes, spans_devices: false,
        })?;
        for key in 0..live_execs {
            let mut ticket = bank.cache_ticket(&key, false, None).unwrap();
            // SAFETY: CPU fixture records no device work.
            let source = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
            let (raw, nodes, width, spans, _) = ticket.complete_source(&library, source, std::ptr::null_mut())?;
            // SAFETY: fixture graph owns no device allocations.
            let owner = unsafe { GraphOwner::new(&library, 0, raw, ())? };
            bank.cache_commit(key, owner, ticket, nodes as u64, width, spans);
        }
        bank.cache_measured(10);
        bank.cache_cycle(Cycle::Verify);
        bank.cache_cycle(Cycle::Decode);

        for launched in [0u32, 12u32, 166u32] {
            let iterations = 2_000usize;
            let mut total_flip_ns = 0u128;
            for step in 0..iterations {
                for k in 0..launched {
                    let key = (k + (step as u32 % 4)) % live_execs;
                    let raw = bank.cached_launch(&key).unwrap().raw;
                    // SAFETY: fixture submits no device work.
                    unsafe { super::super::launch(&library, raw, std::ptr::null_mut(), bank.launch_tally())?; }
                }
                let next = if step % 2 == 0 { Cycle::Verify } else { Cycle::Decode };
                let started = Instant::now();
                bank.cache_cycle(black_box(next));
                total_flip_ns += started.elapsed().as_nanos();
            }
            let us = (total_flip_ns as f64 / iterations as f64) / 1_000.0;
            println!("bank_cycle_flip live_execs={live_execs} launched={launched} us_per_flip={us:.3}");
        }
        drop(bank);
    }
    Ok(())
}
