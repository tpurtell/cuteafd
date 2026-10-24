//! The exchange worker against a fake GPU (counters a test thread advances)
//! and a fake transport: wave order, slot reuse by counters, a decode step
//! queued far ahead of its exchange, and failure release, without hardware.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The "GPU": the control words, written by the engine side of the test as
/// a stream would (staged after copies, consumed after a reduce) and read by
/// the worker; the worker publishes `landed` here.
struct World {
    words: Vec<AtomicU32>,
    slots: usize,
    /// Posted sequences, in order, and the slot each went to.
    posted: Mutex<Vec<(u32, usize)>>,
    /// Per slot: the sequence currently posted into it and not yet landed.
    occupied: Mutex<Vec<Option<u32>>>,
    fail_post: Option<u32>,
    fail_poll: Option<u32>,
    released: AtomicBool,
    polls: AtomicUsize,
}

impl World {
    fn new(slots: usize) -> Arc<Self> {
        Arc::new(Self { words: (0..Counter::words(slots)).map(|_| AtomicU32::new(0)).collect(), slots,
            posted: Mutex::new(Vec::new()), occupied: Mutex::new(vec![None; slots]), fail_post: None, fail_poll: None,
            released: AtomicBool::new(false), polls: AtomicUsize::new(0) })
    }
    fn load(&self, counter: Counter) -> u32 { self.words[counter.index(self.slots)].load(Ordering::Acquire) }
    fn store(&self, counter: Counter, value: u32) { self.words[counter.index(self.slots)].store(value, Ordering::Release) }
}

struct Fake { world: Arc<World> }
struct FakeWave { sequence: u32, slot: usize, polls: usize }

impl Backend for Fake {
    type Wave = FakeWave;
    fn counter(&self, counter: Counter) -> u32 { self.world.load(counter) }
    fn post(&mut self, job: Job) -> Result<FakeWave> {
        if self.world.fail_post == Some(job.sequence) { anyhow::bail!("injected post failure"); }
        let mut occupied = self.world.occupied.lock().unwrap();
        assert!(occupied[job.slot].is_none(), "slot {} reposted before its wave landed", job.slot);
        // The planes are free: the slot's previous reduce was consumed.
        assert!(reached(self.world.load(Counter::Consumed(job.slot)), job.consumed));
        if let Source::Staged { .. } = job.source {
            assert!(reached(self.world.load(Counter::Staged(job.slot)), job.sequence), "posted before staged");
        }
        let mut posted = self.world.posted.lock().unwrap();
        if let Some(&(previous, _)) = posted.last() { assert_eq!(job.sequence, previous.wrapping_add(1), "posts out of order"); }
        posted.push((job.sequence, job.slot));
        occupied[job.slot] = Some(job.sequence);
        Ok(FakeWave { sequence: job.sequence, slot: job.slot, polls: 0 })
    }
    fn poll(&mut self, wave: &mut FakeWave, slot: usize, _rows: usize) -> Result<bool> {
        assert_eq!(slot, wave.slot);
        self.world.polls.fetch_add(1, Ordering::Relaxed);
        wave.polls += 1;
        if self.world.fail_poll == Some(wave.sequence) && wave.polls == 2 { anyhow::bail!("injected receive failure"); }
        if wave.polls < 3 { return Ok(false); }
        self.world.occupied.lock().unwrap()[slot] = None;
        Ok(true)
    }
    fn publish(&mut self, sequence: u32) {
        if !reached(self.world.load(Counter::Landed), sequence) { self.world.store(Counter::Landed, sequence); }
        self.world.released.store(true, Ordering::Release);
    }
}

fn job(sequence: u32, slot: usize, consumed: u32) -> Job {
    Job { sequence, slot, rows: 1, consumed, sent: Instant::now(),
        source: Source::Staged { layout: RouteStaging::new(1, 8, 16).unwrap(),
            build: Box::new(|_, _| anyhow::bail!("the fake posts without building")) } }
}

/// Spins (bounded) until `counter` reaches `target`, as a stream wait would.
fn wait(world: &World, counter: Counter, target: u32) -> Result<()> {
    let started = Instant::now();
    while !reached(world.load(counter), target) {
        ensure!(started.elapsed() < Duration::from_secs(5), "{counter:?} never reached {target}");
        std::thread::yield_now();
    }
    Ok(())
}

/// The engine as a single stream: a decode step of `layers` layers queued on
/// it as send (wait for the slot's previous landing, stage, write staged),
/// land (wait for landed), reduce (write consumed). The host side (sending
/// jobs) runs ahead of the "stream" thread, which executes the queued ops in
/// order: exactly the run-ahead serving does.
fn drive(world: &Arc<World>, slots: usize, layers: u32, timeout: Duration)
    -> (Result<()>, Result<()>, Option<u32>) {
    #[derive(Debug)]
    enum Op { WaitLanded(u32), Staged(usize, u32), Consumed(usize, u32) }
    let (jobs, inbox) = mpsc::channel();
    let (ops, queue) = mpsc::channel::<Op>();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, timeout, false));
        let stream = scope.spawn(|| -> Result<()> {
            for op in queue {
                match op {
                    Op::WaitLanded(sequence) => wait(world, Counter::Landed, sequence)?,
                    Op::Staged(slot, sequence) => world.store(Counter::Staged(slot), sequence),
                    Op::Consumed(slot, uses) => world.store(Counter::Consumed(slot), uses),
                }
            }
            Ok(())
        });
        // The host queues the whole step without waiting for anything.
        let mut uses = vec![0u32; slots];
        for sequence in 1..=layers {
            let slot = (sequence as usize - 1) % slots;
            ops.send(Op::WaitLanded(sequence.wrapping_sub(slots as u32))).unwrap();
            ops.send(Op::Staged(slot, sequence)).unwrap();
            jobs.send(Command::Wave(job(sequence, slot, uses[slot]))).unwrap();
            ops.send(Op::WaitLanded(sequence)).unwrap();
            uses[slot] += 1;
            ops.send(Op::Consumed(slot, uses[slot])).unwrap();
        }
        drop(ops);
        let stream = stream.join().unwrap();
        drop(jobs);
        let (worker, highest) = worker.join().unwrap();
        (stream, worker, highest)
    })
}

#[test]
fn a_decode_step_queued_ahead_lands_every_layer_in_order() {
    for slots in 1..=4 {
        let world = World::new(slots);
        let (stream, worker, highest) = drive(&world, slots, 80, Duration::from_secs(5));
        stream.unwrap();
        worker.unwrap();
        assert_eq!(highest, Some(80));
        let posted = world.posted.lock().unwrap();
        assert_eq!(posted.iter().map(|p| p.0).collect::<Vec<_>>(), (1..=80).collect::<Vec<_>>());
        assert!(posted.iter().all(|&(sequence, slot)| slot == (sequence as usize - 1) % slots));
        assert_eq!(world.load(Counter::Landed), 80);
    }
}

#[test]
fn a_wave_waits_for_its_staged_copies_and_its_slots_previous_reduce() {
    let world = World::new(1);
    let (jobs, inbox) = mpsc::channel();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_secs(5), false));
        // Wave 2 reuses slot 0, whose wave 1 is landed but not yet reduced.
        world.store(Counter::Staged(0), 1);
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        wait(&world, Counter::Landed, 1).unwrap();
        jobs.send(Command::Wave(job(2, 0, 1))).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(world.posted.lock().unwrap().len(), 1, "posted before staged");
        world.store(Counter::Staged(0), 2);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(world.posted.lock().unwrap().len(), 1, "posted into planes a pending reduce still reads");
        world.store(Counter::Consumed(0), 1);
        wait(&world, Counter::Landed, 2).unwrap();
        drop(jobs);
        let (result, _) = worker.join().unwrap();
        result.unwrap();
    });
}

#[test]
fn a_failure_releases_every_queued_and_later_wait() {
    for (fail_post, fail_poll) in [(Some(3), None), (None, Some(2)), (Some(1), None), (None, Some(40))] {
        let mut world = World::new(2);
        Arc::get_mut(&mut world).unwrap().fail_post = fail_post;
        Arc::get_mut(&mut world).unwrap().fail_poll = fail_poll;
        let (jobs, inbox) = mpsc::channel();
        std::thread::scope(|scope| {
            let world2 = world.clone();
            let worker = scope.spawn(move || {
                let mut backend = Fake { world: world2 };
                let (result, highest) = run_worker(&mut backend, &inbox, Duration::from_secs(5), false);
                assert!(result.is_err());
                release_until_closed(&mut backend, &inbox, highest);
            });
            // The engine keeps queueing (it does not know yet) and waits on
            // the GPU: every wait must release.
            let mut uses = [0u32; 2];
            for sequence in 1..=60u32 {
                let slot = (sequence as usize - 1) % 2;
                wait(&world, Counter::Landed, sequence.wrapping_sub(2)).unwrap();
                world.store(Counter::Staged(slot), sequence);
                if jobs.send(Command::Wave(job(sequence, slot, uses[slot]))).is_err() { break; }
                wait(&world, Counter::Landed, sequence).unwrap();
                uses[slot] += 1;
                world.store(Counter::Consumed(slot), uses[slot]);
            }
            drop(jobs);
            worker.join().unwrap();
        });
        assert!(world.released.load(Ordering::Acquire));
    }
}

#[test]
fn closing_with_waves_in_flight_fails_instead_of_hanging() {
    let world = World::new(2);
    let (jobs, inbox) = mpsc::channel();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_secs(5), false));
        // Wave 2's copies never complete: the engine errored before.
        world.store(Counter::Staged(0), 1);
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        jobs.send(Command::Wave(job(2, 1, 0))).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        drop(jobs);
        let (result, highest) = worker.join().unwrap();
        assert!(result.is_err() && highest == Some(2));
    });
}

#[test]
fn a_stalled_gpu_times_out_instead_of_hanging() {
    let world = World::new(1);
    let (jobs, inbox) = mpsc::channel();
    let (result, _) = std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_millis(50), false));
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        let joined = worker.join().unwrap();
        drop(jobs);
        joined
    });
    assert!(format!("{:#}", result.unwrap_err()).contains("no progress"));
}

#[test]
fn an_idle_worker_blocks_instead_of_spinning() {
    let world = World::new(1);
    let (jobs, inbox) = mpsc::channel();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_secs(5), false));
        world.store(Counter::Staged(0), 1);
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        wait(&world, Counter::Landed, 1).unwrap();
        let polls = world.polls.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(50));
        // Blocked on its channel: no further progress passes while idle.
        assert_eq!(world.polls.load(Ordering::Relaxed), polls);
        drop(jobs);
        worker.join().unwrap().0.unwrap();
    });
}

#[test]
fn counters_compare_cyclically_across_wraparound() {
    assert!(reached(5, 5) && reached(6, 5) && !reached(4, 5));
    assert!(reached(1, u32::MAX) && !reached(u32::MAX, 1));
    // A slot's first use waits for `sequence - waves`, already reached.
    for waves in 1..=4u32 { assert!(reached(0, 1u32.wrapping_sub(waves))); }
    assert!(reached(0, 0u32.wrapping_sub(3)));
}

/// No family exchanges inline any more: every Spark wave goes through the
/// link's persistent exchange, and nothing waits on the host for routes or
/// a receive around it.
#[test]
fn no_family_exchanges_on_the_engine_thread() {
    let banned = [
        [".dis", "patch(&"].concat(), ["block_", "on(self.moe_land"].concat(), ["block_", "on(spark_land"].concat(),
        [".rece", "ive(wave"].concat(), ["receive_", "wave("].concat(), ["with_", "waves("].concat(),
        ["with_link_", "waves("].concat(),
    ];
    for (family, source) in [
        ("glm5", include_str!("../families/glm5/engine.rs")),
        ("glm5 layers", include_str!("../families/glm5/engine/layers.rs")),
        ("glm5 mod", include_str!("../families/glm5/mod.rs")),
        ("glm5_flash", include_str!("../families/glm5_flash/engine.rs")),
        ("glm5_flash mod", include_str!("../families/glm5_flash/mod.rs")),
        ("mimo_v2", include_str!("../families/mimo_v2/engine.rs")),
        ("mimo_v2 mod", include_str!("../families/mimo_v2/mod.rs")),
        ("qwen4", include_str!("../families/qwen4/engine.rs")),
        ("step5", include_str!("../families/step5/execution.rs")),
        ("deepseek_v4", include_str!("../families/deepseek_v4/engine.rs")),
        ("deepseek_v4 prefill", include_str!("../families/deepseek_v4/engine/prefill.rs")),
        ("kimi_k3", include_str!("../families/kimi_k3/spark.rs")),
        ("deepseek_v41", include_str!("../families/deepseek_v41/v41_experts/coordinator.rs")),
    ] {
        for needle in &banned {
            assert!(!source.contains(needle.as_str()), "{family} still exchanges inline ({needle})");
        }
    }
}

/// An in-process "Spark world" for the GPU exchange test: each rank answers a
/// request with its rows scaled by `rank + 1` (BF16 bit patterns), from its
/// own thread-free reply queue, a few polls after the post. Ranks in
/// `landing` write their rows straight into device planes, the way GPUDirect
/// landing does (here by a blocking H2D copy the "NIC" finishes before it
/// reports the wave).
struct EchoWorld {
    library: std::sync::Arc<NativeLibrary>,
    ranks: usize,
    row_bytes: usize,
    landing: u8,
    /// Device plane addresses per slot and rank (integers: they cross to the worker).
    planes: Vec<Vec<(usize, usize)>>,
    next_slot: usize,
}
struct EchoWave { slot: usize, rows: Vec<Vec<u8>>, polls: usize }

fn rank_rows(hidden: &[u8], rank: usize) -> Vec<u8> {
    hidden.chunks_exact(2).flat_map(|word| {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16) * (rank as f32 + 1.0);
        ((value.to_bits() >> 16) as u16).to_le_bytes()
    }).collect()
}

impl Transport for EchoWorld {
    type Wave = EchoWave;
    fn dispatch(&mut self, request: &ExpertProtocolV2Request) -> Result<EchoWave> {
        let slot = self.next_slot;
        self.next_slot = (slot + 1) % self.planes.len();
        let hidden = request.hidden_payload.as_ref();
        ensure!(hidden.len() == request.header.row_count as usize * self.row_bytes, "echo rows differ");
        Ok(EchoWave { slot, rows: (0..self.ranks).map(|rank| rank_rows(hidden, rank)).collect(), polls: 0 })
    }
    fn poll(&mut self, wave: &mut EchoWave, sink: &mut dyn FnMut(usize, u32, &[u8]) -> Result<()>)
        -> Result<Option<u8>> {
        wave.polls += 1;
        if wave.polls < 3 { return Ok(None); }
        for (rank, rows) in wave.rows.iter().enumerate() {
            if self.landing & (1 << rank) != 0 {
                let (ptr, _) = self.planes[wave.slot][rank];
                self.library.copy_h2d(CuteafdDeviceBuffer { ptr: ptr as *mut c_void, bytes: rows.len(),
                    ..Default::default() }, rows)?;
            } else {
                // Two frames per rank, row-indexed, as chunked replies arrive.
                let half = (rows.len() / self.row_bytes / 2) * self.row_bytes;
                sink(rank, 0, &rows[..half])?;
                sink(rank, (half / self.row_bytes) as u32, &rows[half..])?;
            }
        }
        Ok(Some(self.landing))
    }
    fn timeout(&self) -> Duration { Duration::from_secs(10) }
    fn abandon(&mut self) {}
}

/// GPU gate: a whole decode step (80 layers, one slot) and a prefill run
/// (3 slots) queued ahead on one stream through the real exchange: counters
/// in mapped memory, stream waits and writes, staging copies, pinned-rank
/// uploads behind the wait, and the reduce. The reduced output of every
/// layer must equal the host reference byte for byte. Opt in with
/// `CUTEAFD_PLANE_UPLOAD_LIBRARY` (the built native library).
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY)"]
fn gpu_exchange_queues_a_step_ahead_and_lands_byte_exact() -> Result<()> {
    use crate::shared::memory::DeviceAllocation;
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&path)? });
    cuteafd_transport::set_verbs_host_native_library_path(std::path::PathBuf::from(&path))?;
    let (hidden, topk) = (512usize, 4usize);
    let row_bytes = hidden * 2;
    for (ranks, landing, waves, layers, rows) in [(3usize, 0u8, 1usize, 80u32, 16usize), (4, 0b0101, 1, 80, 64),
        (3, 0b111, 1, 80, 1), (4, 0, 3, 24, 256), (6, 0b100001, 2, 30, 128)] {
        let capacity = rows;
        let planes_owned = (0..ranks).map(|_| DeviceAllocation::new(&library, capacity * row_bytes * waves))
            .collect::<Result<Vec<_>>>()?;
        let planes: Vec<Vec<CuteafdDeviceBuffer>> = (0..waves).map(|slot| planes_owned.iter().map(|plane|
            CuteafdDeviceBuffer { ptr: (plane.buffer.ptr as usize + slot * capacity * row_bytes) as *mut c_void,
                bytes: capacity * row_bytes, ..plane.buffer }).collect()).collect();
        let layout = RouteStaging::new(rows, topk, row_bytes)?;
        let arenas = (0..waves).map(|_| StagingArena::new(layout.bytes)).collect::<Result<Vec<_>>>()?;
        let echo = EchoWorld { library: library.clone(), ranks, row_bytes, landing, next_slot: 0,
            planes: planes.iter().map(|slot| slot.iter().map(|plane| (plane.ptr as usize, plane.bytes)).collect()).collect() };
        let intake = Intake { ranks, rows: capacity, row_bytes, waves, planes, landing, next_slot: 0 };
        let mut exchange = Waves::start(&library, intake, echo, Vec::new(), arenas, false)?;
        let stream = crate::shared::spark_intake::WarmStream::new(library.as_ref())?;
        let ids = DeviceAllocation::new(&library, rows * topk * 4)?;
        let weights = DeviceAllocation::new(&library, rows * topk * 4)?;
        // One input and one output per layer, all live until the end.
        let inputs = (0..layers).map(|_| DeviceAllocation::new(&library, rows * row_bytes)).collect::<Result<Vec<_>>>()?;
        let outputs = (0..layers).map(|_| DeviceAllocation::new(&library, rows * row_bytes)).collect::<Result<Vec<_>>>()?;
        let mut expected = Vec::new();
        for (layer, input) in inputs.iter().enumerate() {
            let host: Vec<u8> = (0..rows * hidden).flat_map(|i| {
                let value = ((i * 7 + layer * 13) % 61) as f32 / 16.0 - 1.5;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect();
            library.copy_h2d(input.buffer, &host)?;
            let sum: Vec<u8> = (0..rows * hidden).flat_map(|i| {
                let total: f32 = (0..ranks).map(|rank| {
                    let word = rank_rows(&host[2 * i..2 * i + 2], rank);
                    f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
                }).sum();
                ((total.to_bits() >> 16) as u16).to_le_bytes()
            }).collect();
            expected.push(sum);
        }
        let reducer = library.v41_compact_reducer()?;
        // The host queues every layer before the GPU runs the first: send,
        // land, reduce, consumed, back to back (decode run-ahead), with up
        // to `waves` waves in flight (prefill units).
        let mut tickets = std::collections::VecDeque::new();
        for layer in 0..layers as usize {
            if tickets.len() == waves {
                let (layer, ticket): (usize, WaveTicket) = tickets.pop_front().unwrap();
                let slot = ticket.slot();
                exchange.land(ticket, stream.raw())?;
                // SAFETY: planes, output and the null shared plane are live.
                unsafe { reducer.reduce_planes(exchange.planes(slot), ranks as u32, std::ptr::null(),
                    outputs[layer].buffer.ptr.cast(), rows as u32, stream.raw())?; }
                exchange.consumed(slot, stream.raw())?;
            }
            let (h, l) = (hidden, layer);
            // SAFETY: ids, weights and the layer's input are live device buffers of the layout's extents.
            let ticket = unsafe { exchange.send(rows, layout, [ids.buffer, weights.buffer, inputs[layer].buffer],
                stream.raw(), Box::new(move |routes, wire| ExpertProtocolV2Request::new_bytes(l as u64 + 1, 1, l as u32,
                    h as u32, cuteafd_transport::ExpertV2Dtype::Bf16, (0..rows as u32).map(|row|
                        cuteafd_transport::ExpertProtocolV2RowDescriptor { row_id: row.into(),
                        source_kind: cuteafd_transport::ExpertV2SourceKind::Decode, source_request_id: 1,
                        token_position: row.into(), route_offset: row * topk as u32, route_count: topk as u32 })
                        .collect(), routes, wire)))? };
            tickets.push_back((layer, ticket));
        }
        while let Some((layer, ticket)) = tickets.pop_front() {
            let slot = ticket.slot();
            exchange.land(ticket, stream.raw())?;
            // SAFETY: as above.
            unsafe { reducer.reduce_planes(exchange.planes(slot), ranks as u32, std::ptr::null(),
                outputs[layer].buffer.ptr.cast(), rows as u32, stream.raw())?; }
            exchange.consumed(slot, stream.raw())?;
        }
        stream.finish()?;
        healthy()?;
        for (layer, (output, expected)) in outputs.iter().zip(&expected).enumerate() {
            let mut actual = vec![0u8; rows * row_bytes];
            library.copy_d2h(&mut actual, output.buffer)?;
            assert!(actual == *expected, "ranks={ranks} landing={landing:#b} waves={waves} layer {layer} differs");
        }
        let (_echo, _) = exchange.stop::<EchoWorld>()?;
        eprintln!("exchange ranks={ranks} landing={landing:#b} waves={waves}: {layers} layers of {rows} rows byte-exact");
    }
    Ok(())
}
