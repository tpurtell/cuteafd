//! Single-Spark GPU screen of the per-expert partial-reply epilogue on the
//! shared EXL3 worker path (Muse Spark geometry, synthetic routes), run in the
//! Spark dev container on a build seed (kakapo/cassowary), never on fabric:
//!
//! ```text
//! CUTEAFD_NATIVE_LIB=.../libcuteafd_native.so      # SM121 build with MUSE expert scope
//! CUTEAFD_PROGRAMS=.../PROGRAMS.json               # carries expert_partial_*
//! CUTEAFD_EXL3_AOT=.../exl3/exl3-muse-k45/tp6-compact-rank0/m80
//! CUTEAFD_SCREEN_DIR=/nvme/scratch                  # synthetic snapshot (sparse)
//! cargo test -p cuteafd-daemon -- --ignored partial_reply_screen
//! ```
//!
//! The snapshot is the Muse header fixture with real bytes in layer 0's experts
//! 0..16 (seeded trellis payloads, small F16 rotations, the MCG marker) and its
//! `post_expert_norm`; every other byte is a sparse hole (zero). Gates:
//! 1. a mixed wave (decode, verify and a wide prefill segment) writes both
//!    regions, and each row's bytes equal that row run alone (row invariance)
//!    in both wide formats;
//! 2. decoding the NVFP4/FP8 region of a row lies within the format's step of
//!    its BF16-region bytes, which are non-zero (real data flowed);
//! 3. the routing-weighted top-k sum of the BF16 partials matches the
//!    unflagged weighted reply within BF16 rounding (same expert math);
//! 4. a zero-row region writes nothing (the other region and the slot tail
//!    keep a sentinel);
//! 5. the unflagged weighted reply is byte-identical with and without the
//!    epilogue installed.
use super::{execution::{Exl3Execution, Exl3InputFormat}, partial::PartialEpilogue, Exl3Weights};
use crate::shared::{experts::layer::ExpertLayer, memory::{DeviceAllocation, LoadStream}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{PartialReplyPlan, PartialWire};
use std::{io::{Seek, SeekFrom, Write}, path::Path, rc::Rc};

const WIDTH: usize = 4096;
const TOPK: usize = 8;
const ROUTED: usize = 16;

fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }

/// Fill layer 0's routed experts 0..ROUTED and its post-expert norm with
/// deterministic, decodable payloads inside the sparse fixture shard.
fn fill_layer0(dir: &Path) -> Result<()> {
    let shard = dir.join("model-00001.safetensors");
    let bytes = std::fs::read(&shard).map(|b| b[..8].to_vec())?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let mut file = std::fs::OpenOptions::new().read(true).write(true).open(&shard)?;
    let mut header = vec![0u8; header_len];
    use std::io::Read;
    file.seek(SeekFrom::Start(8))?;
    file.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    let base = 8 + header_len as u64;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    for (name, meta) in header.as_object().context("shard header")? {
        let layer0 = name.starts_with("model.language_model.layers.0.mlp.experts.");
        let routed = name.split(".experts.").nth(1).and_then(|t| t.split('.').next())
            .and_then(|e| e.parse::<usize>().ok()).is_some_and(|e| e < ROUTED);
        let norm = name.ends_with("post_expert_norm.weight") && layer0;
        if !(layer0 && (routed || norm)) { continue; }
        let range = meta["data_offsets"].as_array().context("offsets")?;
        let (start, end) = (range[0].as_u64().context("start")?, range[1].as_u64().context("end")?);
        let len = (end - start) as usize;
        let payload: Vec<u8> = if name.ends_with(".trellis") {
            (0..len).map(|_| (next() >> 24) as u8).collect()
        } else if name.ends_with(".suh") || name.ends_with(".svh") {
            // F16 +/-1/64: a bounded rotation sign pattern.
            (0..len / 2).flat_map(|_| if next() & 1 == 0 { 0x2400u16 } else { 0xa400 }.to_le_bytes()).collect()
        } else if name.ends_with(".mcg") {
            (0xcbac_1fedu32).to_le_bytes().to_vec()
        } else {
            // post_expert_norm: BF16 weights in [-0.25, 0.25].
            (0..len / 2).flat_map(|_| ((((next() >> 40) % 64) as f32 / 128.0 - 0.25).to_bits() >> 16)
                .to_le_bytes()[..2].to_vec()).collect()
        };
        ensure!(payload.len() == len, "{name}: payload size");
        file.seek(SeekFrom::Start(base + start))?;
        file.write_all(&payload)?;
    }
    Ok(())
}

struct Screen<'a> {
    library: &'a NativeLibrary,
    execution: Exl3Execution<'a>,
    epilogue: PartialEpilogue<'a>,
    inputs: [DeviceAllocation<'a>; 3],
    slot: DeviceAllocation<'a>,
    stream: LoadStream<'a>,
}

impl Screen<'_> {
    fn read(&self, buffer: CuteafdDeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
        let mut out = vec![0; bytes];
        self.library.copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..buffer })?;
        Ok(out)
    }

    /// Upload `rows` wire rows (row r's activations depend on r only) and routes.
    fn stage(&self, rows: &[usize]) -> Result<()> {
        let pitch = WIDTH + WIDTH / 32;
        let mut wire = vec![0u8; rows.len() * pitch];
        for (i, &r) in rows.iter().enumerate() {
            let row = &mut wire[i * pitch..(i + 1) * pitch];
            for (c, v) in row[..WIDTH].iter_mut().enumerate() { *v = 0x30 + ((r * 7 + c) % 16) as u8; }
            row[WIDTH..].fill(124);
        }
        self.library.copy_h2d(self.inputs[0].buffer, &wire)?;
        let ids: Vec<u8> = rows.iter().flat_map(|&r| (0..TOPK).map(move |k| ((r + 3 * k) % ROUTED) as i32))
            .flat_map(i32::to_ne_bytes).collect();
        let weights: Vec<u8> = rows.iter().flat_map(|&r| (0..TOPK).map(move |k| 1.0 / (2.0 + ((r + k) % 5) as f32)))
            .flat_map(f32::to_ne_bytes).collect();
        self.library.copy_h2d(self.inputs[1].buffer, &ids)?;
        self.library.copy_h2d(self.inputs[2].buffer, &weights)
    }

    fn inputs(&self) -> [CuteafdDeviceBuffer; 3] { std::array::from_fn(|i| self.inputs[i].buffer) }

    /// The weighted reply (unchanged path): BF16 [rows, WIDTH].
    fn weighted(&mut self, rows: &[usize]) -> Result<Vec<u8>> {
        self.stage(rows)?;
        let out = unsafe { self.execution.launch_layer(0, self.inputs(), rows.len(), self.stream.raw)? };
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw)?; }
        self.read(out, rows.len() * WIDTH * 2)
    }

    /// The partial reply payload of `regions` (sentinel-filled slot first).
    fn partial(&mut self, rows: &[usize], wide: PartialWire, regions: Vec<u8>) -> Result<(Vec<u8>, PartialReplyPlan)> {
        self.stage(rows)?;
        let plan = PartialReplyPlan { wide, regions };
        let total = plan.payload(TOPK, WIDTH)?.3;
        self.library.copy_h2d(self.slot.buffer, &vec![0xa5; self.slot.buffer.bytes])?;
        unsafe {
            let sources = self.execution.launch_layer_core(0, self.inputs(), rows.len(), self.stream.raw)?;
            self.epilogue.launch(&plan, rows.len(), TOPK, WIDTH, sources, self.inputs[1].buffer,
                self.slot.buffer, self.stream.raw)?;
            self.library.cuda_stream_synchronize(self.stream.raw)?;
        }
        let bytes = self.read(self.slot.buffer, self.slot.buffer.bytes)?;
        ensure!(bytes[total..].iter().all(|&b| b == 0xa5), "partial reply wrote past its payload");
        Ok((bytes[..total].to_vec(), plan))
    }
}

/// Row `index` (in wave order) of a region, `record` bytes per (row, slot).
fn region_row(payload: &[u8], offset: usize, rows_before: usize, record: usize) -> &[u8] {
    &payload[offset + rows_before * TOPK * record..offset + (rows_before + 1) * TOPK * record]
}

fn decode_wide(record: &[u8], wide: PartialWire) -> Vec<f32> {
    match wide {
        PartialWire::Nvfp4 => {
            const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let (values, scales) = record.split_at(WIDTH / 2);
            (0..WIDTH).map(|i| {
                let nibble = (values[i / 2] >> (4 * (i % 2))) & 0xf;
                let scale = e4m3_decode(scales[i / 16]);
                let magnitude = GRID[(nibble & 7) as usize] * scale;
                if nibble & 8 != 0 { -magnitude } else { magnitude }
            }).collect()
        }
        PartialWire::Fp8 => {
            let (values, scales) = record.split_at(WIDTH);
            (0..WIDTH).map(|i| e4m3_decode(values[i]) * (scales[i / 32] as f32 - 127.0).exp2()).collect()
        }
    }
}

fn e4m3_decode(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let (exp, mant) = ((byte >> 3) & 0xf, byte & 7);
    sign * if exp == 0 { mant as f32 * (-9f32).exp2() } else { (1.0 + mant as f32 / 8.0) * (exp as f32 - 7.0).exp2() }
}

#[test]
#[ignore = "requires an SM121 device, a native library with the Muse expert scope, its PROGRAMS.json and the exl3-muse-k45 package"]
fn partial_reply_screen() -> Result<()> {
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let programs = library.programs()?.with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAMS")?))?;
    let package = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?);
    let dir = tempfile::tempdir_in(std::env::var("CUTEAFD_SCREEN_DIR")?)?;
    cuteafd_loader::plan::testing::write_muse_spark_snapshot(dir.path(), &[0]);
    fill_layer0(dir.path())?;
    let _ = cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::MUSE_SPARK);
    let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
    ensure!(catalog.per_expert_output_norm(), "screen snapshot lacks its post-expert norm");
    library.cuda_set_device(0)?;
    let weights = Rc::new(vec![Exl3Weights::load(&library, &catalog,
        ExpertLayer::BackboneExl3Tp { layer: 0, rank: 0, world: 6 }, 30_000_000_000)?]);
    let execution = unsafe { Exl3Execution::with_input_format(&library, weights, &package, Exl3InputFormat::Fp8K32)? };
    let capacity = 80;
    let pitch = WIDTH + WIDTH / 32;
    let mut screen = Screen { library: &library, execution,
        epilogue: PartialEpilogue::load(&library, &programs, capacity)?,
        inputs: [DeviceAllocation::new(&library, capacity * pitch)?, DeviceAllocation::new(&library, capacity * TOPK * 4)?,
            DeviceAllocation::new(&library, capacity * TOPK * 4)?],
        slot: DeviceAllocation::new(&library, capacity * TOPK * WIDTH * 2 + 4096)?,
        stream: LoadStream { library: &library, raw: library.cuda_stream_create()? } };

    // Gate 5 needs the weighted reply before and after the epilogue first ran.
    let wave: Vec<usize> = (0..1).chain(1..9).chain(9..79).collect(); // decode 1, verify 8, prefill 70
    let weighted_before = screen.weighted(&wave)?;
    // Mixed wave: decode and verify rows BF16, the 70-row segment wide.
    let regions: Vec<u8> = (0..wave.len()).map(|i| u8::from(i >= 9)).collect();
    for wide in [PartialWire::Nvfp4, PartialWire::Fp8] {
        let (payload, plan) = screen.partial(&wave, wide, regions.clone())?;
        let (_, wide_at, _, _) = plan.payload(TOPK, WIDTH)?;
        let record = wide.record_bytes(WIDTH);
        // Gate 1: every row equals that row run alone (both regions).
        for (i, &row) in wave.iter().enumerate().step_by(7) {
            let region = regions[i];
            let (alone, _) = screen.partial(&[row], wide, vec![region])?;
            let (before, offset, size) = if region == 0 {
                (regions[..i].iter().filter(|&&r| r == 0).count(), 0, WIDTH * 2)
            } else { (regions[..i].iter().filter(|&&r| r == 1).count(), wide_at, record) };
            ensure!(region_row(&payload, offset, before, size) == &alone[..TOPK * size],
                "{wide:?}: row {row} differs alone vs in the mixed wave");
        }
        // Gates 2 and 3 on the all-BF16 and all-wide forms of the same wave.
        let (bf16, _) = screen.partial(&wave, wide, vec![0; wave.len()])?;
        let (all_wide, _) = screen.partial(&wave, wide, vec![1; wave.len()])?;
        let (mut worst, mut nonzero) = (0f32, 0usize);
        for i in 0..wave.len() {
            for slot in 0..TOPK {
                let exact: Vec<f32> = bf16[(i * TOPK + slot) * WIDTH * 2..][..WIDTH * 2].chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect();
                let coded = decode_wide(&all_wide[(i * TOPK + slot) * record..][..record], wide);
                for block in (0..WIDTH).step_by(if wide == PartialWire::Nvfp4 { 16 } else { 32 }) {
                    let width = if wide == PartialWire::Nvfp4 { 16 } else { 32 };
                    let amax = exact[block..block + width].iter().fold(0f32, |m, v| m.max(v.abs()));
                    // One format step at the block's scale, plus the BF16 rounding of `exact`.
                    let step = if wide == PartialWire::Nvfp4 { amax / 6.0 } else { amax / 448.0 * 16.0 };
                    for j in block..block + width {
                        let error = (coded[j] - exact[j]).abs();
                        worst = worst.max(if step > 0.0 { error / (step + exact[j].abs() / 128.0) } else { error });
                    }
                }
                nonzero += exact.iter().filter(|v| **v != 0.0).count();
            }
        }
        ensure!(nonzero > wave.len() * TOPK * WIDTH / 2, "{wide:?}: BF16 partials are mostly zero (no data flowed)");
        ensure!(worst <= 1.0, "{wide:?}: wide region exceeds its format step (worst {worst})");
        // Gate 3: the routing-weighted top-k sum of BF16 partials ~ the weighted reply
        // (the package's v41_exl3_sum is the plain weighted sum; the norm is the combine's).
        let weighted = screen.weighted(&wave)?;
        let mut sum_error = 0f32;
        for (i, &row) in wave.iter().enumerate() {
            for c in 0..WIDTH {
                let want = bf16_to_f32(u16::from_le_bytes([weighted[(i * WIDTH + c) * 2], weighted[(i * WIDTH + c) * 2 + 1]]));
                let got: f32 = (0..TOPK).map(|k| {
                    let at = ((i * TOPK + k) * WIDTH + c) * 2;
                    bf16_to_f32(u16::from_le_bytes([bf16[at], bf16[at + 1]])) / (2.0 + ((row + k) % 5) as f32)
                }).sum();
                sum_error = sum_error.max((got - want).abs() / want.abs().max(1e-3));
            }
        }
        ensure!(sum_error < 0.05, "{wide:?}: weighted sum of partials differs from the weighted reply ({sum_error})");
        eprintln!("{wide:?}: rows invariant; wide/BF16 worst step ratio {worst:.3}; weighted-sum rel {sum_error:.4}");
    }
    // Gate 4: zero-row regions write nothing (all-BF16 leaves the wide area at its sentinel).
    let (payload, plan) = screen.partial(&wave, PartialWire::Nvfp4, vec![0; wave.len()])?;
    ensure!(plan.payload(TOPK, WIDTH)?.2 == 0 && payload.len() == wave.len() * TOPK * WIDTH * 2, "empty wide region sized");
    // All-wide: the empty BF16 region takes no bytes; the wide region starts at 0.
    let (payload, plan) = screen.partial(&wave, PartialWire::Fp8, vec![1; wave.len()])?;
    ensure!(plan.payload(TOPK, WIDTH)?.0 == 0 && plan.payload(TOPK, WIDTH)?.1 == 0
        && payload.len() == wave.len() * TOPK * PartialWire::Fp8.record_bytes(WIDTH), "empty BF16 region sized");
    // Gate 5: the weighted path is byte-identical after the epilogue ran.
    ensure!(screen.weighted(&wave)? == weighted_before, "weighted reply changed after partial launches");
    eprintln!("partial-reply screen: mixed waves, both wide formats, empty regions and the weighted path pass");
    Ok(())
}
