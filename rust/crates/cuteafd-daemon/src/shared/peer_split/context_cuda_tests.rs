//! Two-GPU exchange microtest for [`ContextExchange`] (K2 gate): query,
//! candidate and partial exchanges at decode rows, byte-checked against the
//! same bytes assembled on the host and merged/combined on one GPU, then timed
//! as captured graph rounds against PLAN's peer model (3.4 us + bytes / 42
//! GB/s each way).
//!
//! Needs `CUTEAFD_NATIVE_LIB` (a native library built with
//! `CUTEAFD_CONTEXT_SPLIT_AOT=ON|ONLY` and the `glm` geometry),
//! `CUTEAFD_PROGRAMS_JSON` (its manifest) and two peer-capable SM120 GPUs.
use super::*;
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::CuteafdDeviceBuffer;

const HEADS: usize = 32; // per GPU
const D: usize = 576;
const K: usize = 2048;
const ROWS: usize = 64;
const Q_HALF: usize = HEADS * D * 2;
const PARTIAL: usize = HEADS * (PARTIAL_DIM * 2 + 4);
const CANDIDATES: usize = K * 8;
const UNIT: usize = 64;
const TABLE_STRIDE: usize = 64;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn bf16(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32s(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A device buffer on `device` (current device on return: `home`).
struct Buffer<'a> {
    library: &'a NativeLibrary,
    device: i32,
    home: i32,
    allocation: Option<DeviceAllocation<'a>>,
}

impl<'a> Buffer<'a> {
    fn new(library: &'a NativeLibrary, device: i32, home: i32, bytes: usize) -> Result<Self> {
        let allocation = on_device(library, device, home, || {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        })?;
        Ok(Self { library, device, home, allocation: Some(allocation) })
    }
    fn with(library: &'a NativeLibrary, device: i32, home: i32, bytes: &[u8]) -> Result<Self> {
        let buffer = Self::new(library, device, home, bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }
    fn buffer(&self) -> CuteafdDeviceBuffer {
        self.allocation.as_ref().expect("live").buffer
    }
    fn ptr(&self) -> *mut c_void {
        self.buffer().ptr
    }
    fn write(&self, bytes: &[u8]) -> Result<()> {
        on_device(self.library, self.device, self.home, || self.library.copy_h2d(self.buffer(), bytes))
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        let (library, device, home) = (self.library, self.device, self.home);
        let allocation = self.allocation.take();
        let _ = on_device(library, device, home, || {
            drop(allocation);
            Ok(())
        });
    }
}

fn read(library: &NativeLibrary, device: i32, home: i32, ptr: *mut c_void, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; bytes];
    on_device(library, device, home, || library.copy_d2h(&mut out, CuteafdDeviceBuffer { ptr, bytes, device_id: device,
        flags: 0 }))?;
    Ok(out)
}

/// Host-side candidates for one GPU: `[rows, K]` scores and indices on pages
/// of parity `rank`, each run in (score desc, index asc) order, some rows short
/// (padded with (-inf, -1)) and with tied scores.
fn candidates(rng: &mut Rng, rows: usize, rank: usize) -> (Vec<f32>, Vec<i32>) {
    let (mut scores, mut indices) = (Vec::new(), Vec::new());
    for row in 0..rows {
        let live = if row % 5 == 3 { K / 3 } else if row % 7 == 6 { 0 } else { K };
        let mut run: Vec<(f32, i32)> = (0..live).map(|i| {
            let unit = 2 * (i / UNIT) + rank; // logical unit of this GPU's parity
            let index = (unit * UNIT + (i % UNIT)) as i32;
            // Coarse scores: many exact ties across the two GPUs.
            ((rng.next() % 512) as f32 / 8.0, index)
        }).collect();
        run.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        run.resize(K, (f32::NEG_INFINITY, -1));
        scores.extend(run.iter().map(|c| c.0));
        indices.extend(run.iter().map(|c| c.1));
    }
    (scores, indices)
}

/// The host's merged top-K of both GPUs' lists by (score desc, logical index asc).
fn merged(scores: [&[f32]; 2], indices: [&[i32]; 2], rows: usize) -> (Vec<f32>, Vec<i32>) {
    let (mut out_scores, mut out_indices) = (Vec::new(), Vec::new());
    for row in 0..rows {
        let mut all: Vec<(f32, i32)> = (0..2).flat_map(|g| (0..K).map(move |i| (g, i)))
            .map(|(g, i)| (scores[g][row * K + i], indices[g][row * K + i])).filter(|c| c.1 >= 0).collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        all.resize(K, (f32::NEG_INFINITY, -1));
        out_scores.extend(all.iter().map(|c| c.0));
        out_indices.extend(all.iter().map(|c| c.1));
    }
    (out_scores, out_indices)
}

/// A partial `[rows, HEADS, 512]` BF16 + LSE `[rows, HEADS]`; some rows empty (-inf, zeros).
fn partial(rng: &mut Rng, rows: usize, empty_every: usize) -> Vec<u8> {
    let mut values = Vec::new();
    let mut lse = Vec::new();
    for row in 0..rows {
        let empty = empty_every > 0 && row % empty_every == empty_every - 1;
        for _ in 0..HEADS {
            for _ in 0..PARTIAL_DIM {
                values.extend(bf16(if empty { 0.0 } else { rng.unit() * 4.0 - 2.0 }));
            }
            lse.push(if empty { f32::NEG_INFINITY } else { rng.unit() * 24.0 - 4.0 });
        }
    }
    values.extend(f32s(&lse));
    values
}

struct Ctx<'a> {
    library: &'a NativeLibrary,
    ranks: [RankDevice; 2],
    combine: [Program<'a>; 2],
    merge: [Program<'a>; 2],
}

impl<'a> Ctx<'a> {
    /// lse_combine2 on `rank` (o0, l0, o1, l1 -> out), rows.
    fn combine(&self, rank: usize, operands: [(*mut c_void, *mut c_void); 2], sink: *mut c_void, out: *mut c_void,
        rows: usize) -> Result<()> {
        let [(o0, l0), (o1, l1)] = operands;
        // SAFETY: live buffers of this GPU sized for `rows`; ordered on its stream.
        on_device(self.library, self.ranks[rank].device, self.ranks[0].device, || unsafe {
            self.combine[rank].launch(&[o0, l0, o1, l1, sink, out], &[Scalar::I32(rows as i32)], self.ranks[rank].stream)
        })
    }
    /// dsa_candidate_merge on `rank`.
    #[allow(clippy::too_many_arguments)]
    fn merge(&self, rank: usize, scores: *mut c_void, indices: *mut c_void, out_scores: *mut c_void,
        out_indices: *mut c_void, table: *mut c_void, slots: *mut c_void, lengths: *mut c_void, rows: usize,
        shard: usize) -> Result<()> {
        // SAFETY: as `combine`.
        on_device(self.library, self.ranks[rank].device, self.ranks[0].device, || unsafe {
            self.merge[rank].launch(&[scores, indices, out_scores, out_indices, table, slots, lengths],
                &[Scalar::I32(rows as i32), Scalar::I32(TABLE_STRIDE as i32), Scalar::I32(shard as i32)],
                self.ranks[rank].stream)
        })
    }
    fn sync(&self) -> Result<()> {
        for rank in 0..2 {
            // SAFETY: test-owned streams.
            on_device(self.library, self.ranks[rank].device, self.ranks[0].device, || unsafe {
                self.library.cuda_stream_synchronize(self.ranks[rank].stream)
            })?;
        }
        Ok(())
    }
}

/// Captures `rounds` repetitions of `body(rank)` on each GPU's stream and
/// returns the per-round time on GPU0's stream (GPU1's graph launched first).
fn time_rounds(ctx: &Ctx<'_>, rounds: usize, body: &dyn Fn(usize) -> Result<()>) -> Result<f64> {
    let library = ctx.library;
    let home = ctx.ranks[0].device;
    let mut graphs = [std::ptr::null_mut(); 2];
    for rank in [1, 0] {
        graphs[rank] = on_device(library, ctx.ranks[rank].device, home, || {
            // SAFETY: test-owned stream with no other work queued during capture.
            unsafe { library.cuda_graph_begin_capture(ctx.ranks[rank].stream)? };
            let captured = (0..rounds).try_for_each(|_| body(rank));
            library.cuda_set_device(ctx.ranks[rank].device)?;
            // SAFETY: ends the capture begun above.
            let graph = unsafe { library.cuda_graph_end_capture(ctx.ranks[rank].stream) };
            captured.and(graph)
        })?;
    }
    let mut samples = Vec::new();
    for repeat in 0..6 {
        let (start, stop) = (library.cuda_event_create()?, library.cuda_event_create()?);
        // SAFETY: graphs and events are live; GPU1's graph waits on GPU0's pushes.
        unsafe {
            on_device(library, ctx.ranks[1].device, home, || library.cuda_graph_launch(graphs[1], ctx.ranks[1].stream))?;
            library.cuda_event_record(start, ctx.ranks[0].stream)?;
            library.cuda_graph_launch(graphs[0], ctx.ranks[0].stream)?;
            library.cuda_event_record(stop, ctx.ranks[0].stream)?;
        }
        ctx.sync()?;
        // SAFETY: both events completed with the synchronized stream.
        let ms = unsafe { library.cuda_event_elapsed_ms(start, stop)? };
        unsafe {
            library.cuda_event_destroy(start)?;
            library.cuda_event_destroy(stop)?;
        }
        if repeat > 0 {
            samples.push(f64::from(ms) * 1e3 / rounds as f64);
        }
    }
    for rank in 0..2 {
        // SAFETY: the graph's work drained (sync above).
        on_device(library, ctx.ranks[rank].device, home, || unsafe { library.cuda_graph_exec_destroy(graphs[rank]) })?;
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

fn model_us(bytes: usize) -> f64 {
    3.4 + bytes as f64 / 42e3
}

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB with context-split programs, CUTEAFD_PROGRAMS_JSON and two peer GPUs"]
fn context_exchange_is_byte_exact_and_timed_against_the_peer_model() -> Result<()> {
    // SAFETY: every CUDA owner below is dropped before the library.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_PROGRAMS_JSON")?);
    library.cuda_set_device(0)?;
    let mut ranks = [RankDevice { device: 0, stream: std::ptr::null_mut() }, RankDevice { device: 1, stream: std::ptr::null_mut() }];
    for rank in &mut ranks {
        rank.stream = on_device(&library, rank.device, 0, || library.cuda_stream_create())?;
    }
    let programs: Programs<'_> = library.programs()?.with_manifest(&manifest)?;
    let load = |name: &str, pointers: &[&str]| -> Result<[Program<'_>; 2]> {
        let first = on_device(&library, 1, 0, || programs.program(name, pointers))?;
        Ok([programs.program(name, pointers)?, first])
    };
    let ctx = Ctx {
        library: &library,
        ranks,
        combine: load("glm_lse_combine2", &["o0", "l0", "o1", "l1", "sink", "out"])?,
        merge: load("glm_dsa_candidate_merge", &["scores", "indices", "out_scores", "out_indices", "table",
            "local_slots", "local_lengths"])?,
    };
    let buffers = ContextBuffers { query_row_bytes: Q_HALF as u64, partial_row_bytes: PARTIAL as u64,
        candidate_row_bytes: CANDIDATES as u64, decode_rows: ROWS as u64, lanes: 1, ..Default::default() };
    let result = (|| -> Result<()> {
        let exchange = ContextExchange::new(&library, ranks, &buffers, None)?;
        assert_eq!(exchange.bytes() as u64, buffers.demands()?[1].bytes, "allocation == solver charge");
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let dev = |rank: usize, bytes: &[u8]| Buffer::with(&library, ranks[rank].device, 0, bytes);
        let zeros = |rank: usize, bytes: usize| Buffer::new(&library, ranks[rank].device, 0, bytes);
        println!("rows | exchange | bytes/way | measured us | model us");
        for rows in [1usize, 4, 16, 64] {
            for layer in [6usize, 7] {
                let s = slot(layer, 0);
                // --- query: each GPU's half lands in both assembled buffers.
                let q: Vec<Vec<u8>> = (0..2).map(|_| (0..rows * Q_HALF).map(|_| rng.next() as u8).collect()).collect();
                let q_dev: Vec<_> = (0..2).map(|r| dev(r, &q[r])).collect::<Result<_>>()?;
                for rank in 0..2 {
                    exchange.push_query(rank, s, q_dev[rank].ptr(), Q_HALF, rows)?;
                }
                let assembled: Vec<_> = (0..2).map(|rank| exchange.wait_query(rank, s)).collect::<Result<_>>()?;
                ctx.sync()?;
                let expected: Vec<u8> = (0..rows).flat_map(|r| q[0][r * Q_HALF..(r + 1) * Q_HALF].iter()
                    .chain(&q[1][r * Q_HALF..(r + 1) * Q_HALF]).copied().collect::<Vec<_>>()).collect();
                for rank in 0..2 {
                    assert_eq!(read(&library, ranks[rank].device, 0, assembled[rank], rows * 2 * Q_HALF)?, expected,
                        "assembled query on GPU{rank}, {rows} rows");
                }
                // --- candidates: exchanged lists merged on each GPU == host-assembled merge on GPU0.
                let lists: Vec<(Vec<f32>, Vec<i32>)> = (0..2).map(|r| candidates(&mut rng, rows, r)).collect();
                let table: Vec<i32> = (0..rows * TABLE_STRIDE).map(|i| ((i * 37 + 11) % 997) as i32).collect();
                let mut owned = Vec::new();
                for rank in 0..2 {
                    let (scores, indices) = (dev(rank, &f32s(&lists[rank].0))?, dev(rank, &i32s(&lists[rank].1))?);
                    exchange.push_candidates(rank, s, scores.ptr(), indices.ptr(), rows)?;
                    owned.push((scores, indices));
                }
                let mut outputs = Vec::new();
                for rank in 0..2 {
                    let (scores, indices) = exchange.wait_candidates(rank, s)?;
                    let out = [zeros(rank, rows * K * 4)?, zeros(rank, rows * K * 4)?, dev(rank, &i32s(&table))?,
                        zeros(rank, rows * K * 4)?, zeros(rank, rows * 4)?];
                    ctx.merge(rank, scores, indices, out[0].ptr(), out[1].ptr(), out[2].ptr(), out[3].ptr(), out[4].ptr(),
                        rows, rank)?;
                    outputs.push(out);
                }
                ctx.sync()?;
                let (host_scores, host_indices) = merged([&lists[0].0, &lists[1].0], [&lists[0].1, &lists[1].1], rows);
                for rank in 0..2 {
                    let joined_scores: Vec<f32> = (0..rows).flat_map(|r| lists[0].0[r * K..(r + 1) * K].iter()
                        .chain(&lists[1].0[r * K..(r + 1) * K]).copied().collect::<Vec<_>>()).collect();
                    let joined_indices: Vec<i32> = (0..rows).flat_map(|r| lists[0].1[r * K..(r + 1) * K].iter()
                        .chain(&lists[1].1[r * K..(r + 1) * K]).copied().collect::<Vec<_>>()).collect();
                    let single = [dev(0, &f32s(&joined_scores))?, dev(0, &i32s(&joined_indices))?,
                        zeros(0, rows * K * 4)?, zeros(0, rows * K * 4)?, dev(0, &i32s(&table))?, zeros(0, rows * K * 4)?,
                        zeros(0, rows * 4)?];
                    ctx.merge(0, single[0].ptr(), single[1].ptr(), single[2].ptr(), single[3].ptr(), single[4].ptr(),
                        single[5].ptr(), single[6].ptr(), rows, rank)?;
                    ctx.sync()?;
                    for (out, (reference, what)) in [(0usize, (2usize, "scores")), (1, (3, "indices")), (3, (5, "local slots")),
                        (4, (6, "local lengths"))] {
                        let bytes = if out == 4 { rows * 4 } else { rows * K * 4 };
                        assert_eq!(read(&library, ranks[rank].device, 0, outputs[rank][out].ptr(), bytes)?,
                            read(&library, 0, 0, single[reference].ptr(), bytes)?,
                            "merged {what} on GPU{rank} vs one-GPU merge, {rows} rows");
                    }
                    assert_eq!(read(&library, ranks[rank].device, 0, outputs[rank][1].ptr(), rows * K * 4)?,
                        i32s(&host_indices), "merged selection on GPU{rank} vs host total order");
                    let scores = read(&library, ranks[rank].device, 0, outputs[rank][0].ptr(), rows * K * 4)?;
                    assert_eq!(scores, f32s(&host_scores), "merged scores on GPU{rank}");
                }
                drop(owned);
                // --- partials: each GPU pushes its partial of the peer's heads; combine in GPU0, GPU1 order.
                // partials[g][h]: GPU g's partial of head half h.
                let partials: Vec<Vec<Vec<u8>>> = (0..2).map(|g| (0..2).map(|_| partial(&mut rng, rows, 3 + g)).collect()).collect();
                let held: Vec<Vec<Buffer<'_>>> = (0..2).map(|g| (0..2).map(|h| dev(g, &partials[g][h])).collect::<Result<_>>())
                    .collect::<Result<_>>()?;
                let lse_at = rows * HEADS * PARTIAL_DIM * 2;
                for rank in 0..2 {
                    let p = held[rank][1 - rank].ptr();
                    exchange.push_partial(rank, s, p, p.cast::<u8>().wrapping_add(lse_at).cast(), rows)?;
                }
                let sink = [zeros(0, HEADS * 4)?, zeros(1, HEADS * 4)?];
                let out = [zeros(0, rows * HEADS * PARTIAL_DIM * 2)?, zeros(1, rows * HEADS * PARTIAL_DIM * 2)?];
                for rank in 0..2 {
                    let received = exchange.wait_partial(rank, s)?;
                    let own = held[rank][rank].ptr();
                    let own = (own, own.cast::<u8>().wrapping_add(lse_at).cast());
                    ctx.combine(rank, combine_order(rank, own, received), sink[rank].ptr(), out[rank].ptr(), rows)?;
                }
                ctx.sync()?;
                for rank in 0..2 {
                    // One GPU, the same operands assembled on the host, the same order.
                    let o = [dev(0, &partials[0][rank])?, dev(0, &partials[1][rank])?];
                    let reference = zeros(0, rows * HEADS * PARTIAL_DIM * 2)?;
                    let operand = |b: &Buffer<'_>| (b.ptr(), b.ptr().cast::<u8>().wrapping_add(lse_at).cast());
                    ctx.combine(0, [operand(&o[0]), operand(&o[1])], sink[0].ptr(), reference.ptr(), rows)?;
                    ctx.sync()?;
                    let bytes = rows * HEADS * PARTIAL_DIM * 2;
                    let combined = read(&library, ranks[rank].device, 0, out[rank].ptr(), bytes)?;
                    assert_eq!(combined, read(&library, 0, 0, reference.ptr(), bytes)?,
                        "combine of exchanged partials on GPU{rank} vs one-GPU combine, {rows} rows");
                    assert!(combined.chunks(2).all(|h| {
                        let v = f32::from_bits(u32::from(u16::from_le_bytes([h[0], h[1]])) << 16);
                        v.is_finite()
                    }), "empty shards and empty rows combine to finite values");
                }
                if layer == 7 {
                    continue;
                }
                // --- timing: captured graph rounds, each round one even and one odd layer (slots 0
                // and 1, as a decode step alternates them); q written in place by its producer.
                let own_half = |rank: usize, s: usize| exchange.buffers(rank, s).map(|b| b.query.cast::<u8>()
                    .wrapping_add(rank * Q_HALF).cast_const().cast::<c_void>());
                let scores = [dev(0, &f32s(&lists[0].0))?, dev(1, &f32s(&lists[1].0))?];
                let indices = [dev(0, &i32s(&lists[0].1))?, dev(1, &i32s(&lists[1].1))?];
                let partial_push = |rank: usize, s: usize| {
                    let p = held[rank][1 - rank].ptr();
                    exchange.push_partial(rank, s, p, p.cast::<u8>().wrapping_add(lse_at).cast(), rows)
                };
                let combine = |rank: usize, s: usize| -> Result<()> {
                    let received = exchange.wait_partial(rank, s)?;
                    let own = held[rank][rank].ptr();
                    let own = (own, own.cast::<u8>().wrapping_add(lse_at).cast());
                    ctx.combine(rank, combine_order(rank, own, received), sink[rank].ptr(), out[rank].ptr(), rows)
                };
                let rounds = 32;
                let per_layer = |body: &dyn Fn(usize, usize) -> Result<()>| -> Result<f64> {
                    Ok(time_rounds(&ctx, rounds, &|rank| (0..2).try_for_each(|s| body(rank, s)))? / 2.0)
                };
                let query_us = per_layer(&|rank, s| {
                    exchange.push_query(rank, s, own_half(rank, s)?, 2 * Q_HALF, rows)?;
                    exchange.wait_query(rank, s).map(|_| ())
                })?;
                let candidate_us = per_layer(&|rank, s| {
                    exchange.push_candidates(rank, s, scores[rank].ptr(), indices[rank].ptr(), rows)?;
                    exchange.wait_candidates(rank, s).map(|_| ())
                })?;
                let partial_us = per_layer(&|rank, s| {
                    partial_push(rank, s)?;
                    exchange.wait_partial(rank, s).map(|_| ())
                })?;
                let combine_us = per_layer(&|rank, s| {
                    partial_push(rank, s)?;
                    combine(rank, s)
                })?;
                let layer_us = per_layer(&|rank, s| {
                    exchange.push_query(rank, s, own_half(rank, s)?, 2 * Q_HALF, rows)?;
                    exchange.wait_query(rank, s)?;
                    partial_push(rank, s)?;
                    combine(rank, s)
                })?;
                for (what, bytes, us) in [("q", rows * Q_HALF, query_us), ("candidates", rows * CANDIDATES, candidate_us),
                    ("partial", rows * PARTIAL, partial_us)] {
                    println!("{rows:>4} | {what:<10} | {bytes:>9} | {us:>11.2} | {:>8.2}", model_us(bytes));
                }
                println!("{rows:>4} | partial+combine | {:>9} | {combine_us:>11.2} | {:>8.2}", rows * PARTIAL,
                    model_us(rows * PARTIAL));
                println!("{rows:>4} | q+partial+combine (shared layer) | {:>9} | {layer_us:>11.2} | {:>8.2}",
                    rows * (Q_HALF + PARTIAL), model_us(rows * Q_HALF) + model_us(rows * PARTIAL));
            }
        }
        ctx.sync()?;
        drop(exchange);
        Ok(())
    })();
    drop(ctx);
    for rank in ranks {
        on_device(&library, rank.device, 0, || {
            // SAFETY: nothing else uses the test streams.
            unsafe { library.cuda_stream_synchronize(rank.stream)?; library.cuda_stream_destroy(rank.stream) }
        })?;
    }
    result
}
