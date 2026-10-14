//! Run ignored fixtures in the coordinator-dev image, one process per family.
use anyhow::{ensure, Result};
use cuteafd_core::{set_expert_geometry, ExpertGeometry};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41ExpertLaunchArgs, V41_EXPERT_POINTER_COUNT,
};
use std::ffi::c_void;

struct Arena<'a> {
    library: &'a NativeLibrary,
    device: i32,
    stream: *mut c_void,
    buffers: Vec<CuteafdDeviceBuffer>,
}
impl<'a> Arena<'a> {
    fn new(library: &'a NativeLibrary, device: i32) -> Result<Self> {
        library.cuda_set_device(device)?;
        Ok(Self {
            library,
            device,
            stream: library.cuda_stream_create()?,
            buffers: Vec::new(),
        })
    }
    fn alloc(&mut self, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        self.library.cuda_set_device(self.device)?;
        let buffer = self.library.alloc_device_buffer(bytes)?;
        self.buffers.push(buffer);
        Ok(buffer)
    }
    fn upload(&mut self, data: &[u8]) -> Result<CuteafdDeviceBuffer> {
        let buffer = self.alloc(data.len())?;
        self.library.copy_h2d(buffer, data)?;
        Ok(buffer)
    }
}
impl Drop for Arena<'_> {
    fn drop(&mut self) {
        if self.library.cuda_set_device(self.device).is_ok() {
            // SAFETY: drain all queued work before freeing the fixture's storage/stream.
            unsafe {
                let _ = self.library.cuda_stream_synchronize(self.stream);
                for buffer in &mut self.buffers {
                    let _ = self.library.free_device_buffer(buffer);
                }
                let _ = self.library.cuda_stream_destroy(self.stream);
            }
        }
    }
}

fn library() -> Result<NativeLibrary> {
    // SAFETY: the fixture uses the matching native ABI and retains the library until all CUDA work drains.
    unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?) }
}

#[test]
#[ignore = "requires two RTX GPUs and dsv4f/dsv4p native exports; set CUTEAFD_TEST_FAMILY"]
fn rtx_expert_capacities_initialize_and_launch_on_both_devices() -> Result<()> {
    let geometry = match std::env::var("CUTEAFD_TEST_FAMILY").as_deref() {
        Ok("dsv4p") => ExpertGeometry::DEEPSEEK_V4_PRO,
        Ok("dsv4f") | Err(_) => ExpertGeometry::DEEPSEEK_V4_FLASH,
        _ => anyhow::bail!("CUTEAFD_TEST_FAMILY must be dsv4f or dsv4p"),
    };
    set_expert_geometry(geometry)
        .map_err(|g| anyhow::anyhow!("geometry already fixed to {g:?}"))?;
    let lib = library()?;
    for tp2 in [true, false] {
        for capacity in [1, 16, 80, 256, 1024, 4096] {
            let mut arenas = [Arena::new(&lib, 0)?, Arena::new(&lib, 1)?];
            let mut launches = Vec::new();
            for arena in &mut arenas {
                lib.cuda_set_device(arena.device)?;
                let kernel = if tp2 {
                    lib.v41_tp2_expert_kernel(capacity)?
                } else {
                    lib.v41_local_expert_kernel(capacity)?
                };
                let info = *kernel.info();
                let logical = geometry.intermediate / if tp2 { 2 } else { 1 };
                ensure!(
                    info.role == if tp2 { 3 } else { 2 }
                        && info.logical_intermediate == logical
                        && info.kernel_intermediate == logical.div_ceil(128) * 128
                        && info.hidden_size == geometry.hidden
                        && info.experts == geometry.experts
                        && info.topk == geometry.topk
                        && info.capacity_rows == capacity
                        && info.input_dtype == 7,
                    "unexpected expert info: {info:?}"
                );
                let scratch = arena.alloc(info.scratch_bytes as usize)?;
                let dummy = arena.alloc(16)?;
                let mut tensors = [dummy.ptr; V41_EXPERT_POINTER_COUNT];
                // SAFETY: aligned scratch on the initialized device, retained by the arena.
                unsafe {
                    kernel.bind_scratch(scratch.ptr, scratch.bytes as u64, &mut tensors)?;
                    kernel.initialize_scratch(scratch.ptr, scratch.bytes as u64, arena.stream)?;
                    lib.cuda_stream_synchronize(arena.stream)?;
                }
                let rows = capacity.min(3) as usize;
                let mut wire = vec![0x38u8; rows * info.input_row_bytes()?];
                for row in wire.chunks_exact_mut(info.input_row_bytes()?) {
                    row[geometry.hidden as usize..].fill(127); // E8M0 scale 1, E4M3 input 1.
                }
                tensors[0] = arena.upload(&wire)?.ptr;
                tensors[1] = arena
                    .upload(&vec![0; rows * geometry.topk as usize * 4])?
                    .ptr;
                let routing: Vec<u8> = (0..rows * geometry.topk as usize)
                    .flat_map(|_| (1.0f32 / geometry.topk as f32).to_ne_bytes())
                    .collect();
                tensors[2] = arena.upload(&routing)?.ptr;
                // Only expert 0 is routed. Allocate its packed stride, not a full model layer.
                let extent = info.kernel_intermediate as usize * geometry.hidden as usize;
                for (slot, bytes) in [
                    (30, extent),
                    (31, extent / 16),
                    (32, extent / 2),
                    (33, extent / 32),
                ] {
                    let weights = arena.alloc(bytes)?;
                    lib.cuda_zero_bytes(weights, bytes)?;
                    tensors[slot] = weights.ptr;
                }
                let output_bytes = rows
                    * geometry.hidden as usize
                    * 4
                    * if kernel.accumulates_tokens() {
                        1
                    } else {
                        geometry.topk as usize
                    };
                let output = CuteafdDeviceBuffer {
                    ptr: tensors[41],
                    bytes: output_bytes,
                    device_id: arena.device,
                    flags: 0,
                };
                lib.copy_h2d(output, &vec![0xa5; output_bytes])?;
                let args = V41ExpertLaunchArgs::new(&info, tensors, rows as u32, arena.stream)?;
                launches.push((kernel, args, output));
            }
            // Queue both devices before either synchronization: handles coexist in one process.
            for (device, (kernel, args, _)) in launches.iter().enumerate() {
                lib.cuda_set_device(device as i32)?;
                // SAFETY: all typed weight/input/scratch storage is live and initialized on this stream.
                unsafe {
                    kernel.launch(args)?;
                }
            }
            for (device, (_, _, output)) in launches.iter().enumerate() {
                lib.cuda_set_device(device as i32)?;
                // SAFETY: the arena owns a live stream on this device.
                unsafe {
                    lib.cuda_stream_synchronize(arenas[device].stream)?;
                }
                let mut actual = vec![0; output.bytes];
                lib.copy_d2h(&mut actual, *output)?;
                ensure!(
                    actual
                        .chunks_exact(4)
                        .all(|b| f32::from_ne_bytes(b.try_into().unwrap()) == 0.0),
                    "zero-weight output differs: device {device} capacity {capacity} tp2 {tp2}"
                );
            }
            println!(
                "{} role={} capacity={capacity}: GPU0/GPU1 passed",
                geometry.family().unwrap(),
                if tp2 { 3 } else { 2 }
            );
        }
    }
    Ok(())
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
fn from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[test]
#[ignore = "requires two RTX GPUs and TP2 RTX combine/ordered route-sum symbols"]
fn rtx_combine_and_geometry_route_sums_match_exact_formula() -> Result<()> {
    use cuteafd_ffi::RtxPartialDtype;
    let lib = library()?;
    let combine = lib.rtx_tp2_combine()?;
    for dtype in [RtxPartialDtype::Bf16, RtxPartialDtype::F32] {
        for count in [1usize, 3, 255, 257, 1025] {
            for (has_routed, has_shared) in
                [(true, true), (true, false), (false, true), (false, false)]
            {
                let routed: [Vec<f32>; 2] = std::array::from_fn(|rank| {
                    (0..count)
                        .map(|i| {
                            ((i % 19) as f32 - 9.0) * if rank == 0 { 0.015625 } else { -0.0078125 }
                                + 0.0009765625
                        })
                        .collect()
                });
                let shared: [Vec<u16>; 2] = std::array::from_fn(|rank| {
                    (0..count)
                        .map(|i| {
                            bf16(((i % 7) as f32 - 3.0) * if rank == 0 { 0.25 } else { -0.125 })
                        })
                        .collect()
                });
                let payloads: [Vec<u8>; 2] = std::array::from_fn(|rank| {
                    (0..count)
                        .flat_map(|i| {
                            let p = if has_routed { routed[rank][i] } else { 0.0 }
                                + if has_shared {
                                    from_bf16(shared[rank][i])
                                } else {
                                    0.0
                                };
                            match dtype {
                                RtxPartialDtype::Bf16 => bf16(p).to_ne_bytes().to_vec(),
                                RtxPartialDtype::F32 => p.to_ne_bytes().to_vec(),
                            }
                        })
                        .collect()
                });
                let value = |rank: usize, i: usize| match dtype {
                    RtxPartialDtype::Bf16 => from_bf16(u16::from_ne_bytes(
                        payloads[rank][i * 2..i * 2 + 2].try_into().unwrap(),
                    )),
                    RtxPartialDtype::F32 => {
                        f32::from_ne_bytes(payloads[rank][i * 4..i * 4 + 4].try_into().unwrap())
                    }
                };
                let expected: Vec<u8> = (0..count)
                    .flat_map(|i| bf16(value(0, i) + value(1, i)).to_ne_bytes())
                    .collect();
                for device in 0..2 {
                    let mut arena = Arena::new(&lib, device)?;
                    let mut partials = Vec::new();
                    for rank in 0..2 {
                        let r = arena.upload(
                            &routed[rank]
                                .iter()
                                .flat_map(|v| v.to_ne_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        let s = arena.upload(
                            &shared[rank]
                                .iter()
                                .flat_map(|v| v.to_ne_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        let p = arena.alloc(count * dtype.element_bytes())?;
                        // SAFETY: fixture-owned disjoint buffers, with uploads completed before launch.
                        unsafe {
                            combine.partial(
                                if has_routed {
                                    r.ptr.cast()
                                } else {
                                    std::ptr::null()
                                },
                                if has_shared {
                                    s.ptr.cast()
                                } else {
                                    std::ptr::null()
                                },
                                p.ptr,
                                count,
                                dtype,
                                arena.stream,
                            )?;
                        }
                        partials.push(p);
                    }
                    let output = arena.alloc(count * 2)?;
                    // SAFETY: both rank payloads are on this GPU, ordered on the same stream.
                    unsafe {
                        combine.sum(
                            partials[0].ptr,
                            partials[1].ptr,
                            output.ptr.cast(),
                            count,
                            dtype,
                            arena.stream,
                        )?;
                        lib.cuda_stream_synchronize(arena.stream)?;
                    }
                    for rank in 0..2 {
                        let mut actual = vec![0; partials[rank].bytes];
                        lib.copy_d2h(&mut actual, partials[rank])?;
                        ensure!(
                            actual == payloads[rank],
                            "partial mismatch {dtype:?} count={count} rank={rank} device={device}"
                        );
                    }
                    let mut actual = vec![0; output.bytes];
                    lib.copy_d2h(&mut actual, output)?;
                    ensure!(
                        actual == expected,
                        "sum mismatch {dtype:?} count={count} device={device}"
                    );
                }
            }
        }
    }
    let reducer = lib.v41_tp2_expert_reducer()?;
    for geometry in [
        ExpertGeometry::DEEPSEEK_V41,
        ExpertGeometry::DEEPSEEK_V4_FLASH,
        ExpertGeometry::DEEPSEEK_V4_PRO,
        ExpertGeometry::GLM5,
        ExpertGeometry::QWEN4,
    ] {
        let rows = 3;
        let hidden = geometry.hidden as usize;
        let topk = geometry.topk as usize;
        let input: Vec<f32> = (0..rows * topk * hidden)
            .map(|i| {
                // Cancellation makes the specified left-to-right route order observable.
                match (i / hidden) % topk {
                    0 => 16_777_216.0,
                    1 => 1.0,
                    2 => -16_777_216.0,
                    _ => (i % 11) as f32 * 0.125,
                }
            })
            .collect();
        let expected: Vec<u8> = (0..rows * hidden)
            .flat_map(|i| {
                let base = (i / hidden) * topk * hidden + i % hidden;
                let mut value = input[base];
                for route in 1..topk {
                    value += input[base + route * hidden];
                }
                value.to_ne_bytes()
            })
            .collect();
        for device in 0..2 {
            let mut arena = Arena::new(&lib, device)?;
            let routes = arena.upload(
                &input
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let sums = arena.alloc(expected.len())?;
            // SAFETY: disjoint, correctly sized FP32 arrays on the arena's stream/device.
            unsafe {
                reducer.sum_routes_geometry(routes, sums, rows as u32, geometry, arena.stream)?;
                lib.cuda_stream_synchronize(arena.stream)?;
            }
            let mut actual = vec![0; sums.bytes];
            lib.copy_d2h(&mut actual, sums)?;
            ensure!(
                actual == expected,
                "geometry sum differs for {geometry:?} on device {device}"
            );
            if geometry == ExpertGeometry::DEEPSEEK_V41 {
                // SAFETY: legacy V4.1 extents match exactly, same route order.
                unsafe {
                    reducer.sum_routes(routes, sums, rows as u32, arena.stream)?;
                    lib.cuda_stream_synchronize(arena.stream)?;
                }
                lib.copy_d2h(&mut actual, sums)?;
                ensure!(
                    actual == expected,
                    "new sum is not byte-exact against V4.1 legacy sum"
                );
            }
        }
    }
    println!("BF16/F32 combine exact on both GPUs; geometry sums top-6/8/10 exact, legacy V4.1 byte-exact");
    Ok(())
}

struct Captured<'a> {
    library: &'a NativeLibrary,
    stream: *mut c_void,
    exec: *mut c_void,
}
impl Drop for Captured<'_> {
    fn drop(&mut self) {
        // SAFETY: the fixture retains its arena until the graph is drained and destroyed.
        unsafe {
            let _ = self.library.cuda_stream_synchronize(self.stream);
            let _ = self.library.cuda_graph_exec_destroy(self.exec);
        }
    }
}

fn read_bytes(lib: &NativeLibrary, buffer: CuteafdDeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
    let mut value = vec![0; bytes];
    lib.copy_d2h(&mut value, CuteafdDeviceBuffer { bytes, ..buffer })?;
    Ok(value)
}

#[test]
#[ignore = "requires CUDA and V4 masked native exports; exhaustive fixed-bucket replay gate"]
fn v4_masked_router_and_sum_are_bitwise_at_every_width() -> Result<()> {
    use cuteafd_ffi::RtxPartialDtype;
    const BUCKETS: [usize; 22] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        24, 32, 40, 48, 56, 64];
    let lib = library()?;
    let combine = lib.rtx_tp2_combine()?;
    for device in 0..2 {
        lib.cuda_set_device(device)?;
        lib.tp2_native_initialize()?;
        for geometry in [ExpertGeometry::DEEPSEEK_V4_FLASH, ExpertGeometry::DEEPSEEK_V4_PRO] {
            let (hidden, experts, topk) = (geometry.hidden as usize, geometry.experts as usize, geometry.topk as usize);
            for hash in [false, true] {
                for dtype in [RtxPartialDtype::Bf16, RtxPartialDtype::F32] {
                    let mut arena = Arena::new(&lib, device)?;
                    let logits = arena.alloc(64 * experts * 4)?;
                    let reference_logits = arena.alloc(logits.bytes)?;
                    let ids = arena.alloc(64 * topk * 4)?;
                    let weights = arena.alloc(ids.bytes)?;
                    let reference_ids = arena.alloc(ids.bytes)?;
                    let reference_weights = arena.alloc(ids.bytes)?;
                    let slots = arena.alloc(64 * 8)?;
                    let tokens = arena.upload(&(0..64u32).flat_map(|i| (i % 7).to_ne_bytes()).collect::<Vec<_>>())?;
                    let table = arena.upload(&(0..7 * topk).flat_map(|i| ((i * 37 % experts) as i32).to_ne_bytes()).collect::<Vec<_>>())?;
                    let bias = arena.upload(&(0..experts).flat_map(|i| (((i % 5) as f32 - 2.0) * 0.0625).to_ne_bytes()).collect::<Vec<_>>())?;
                    let (bias_ptr, table_ptr) = if hash { (std::ptr::null(), table.ptr.cast_const()) }
                        else { (bias.ptr.cast_const(), std::ptr::null()) };
                    let payload: [Vec<u8>; 2] = std::array::from_fn(|rank| (0..64 * hidden).flat_map(|i| {
                        let value = match i % 9 {
                            0 => if rank == 0 { -0.0 } else { 0.0 },
                            1 => if rank == 0 { 16_777_216.0 } else { -16_777_216.0 },
                            _ => ((i % 53) as f32 - 26.0) * if rank == 0 { 0.015625 } else { -0.0078125 },
                        };
                        match dtype { RtxPartialDtype::Bf16 => bf16(value).to_ne_bytes().to_vec(),
                            RtxPartialDtype::F32 => value.to_ne_bytes().to_vec() }
                    }).collect());
                    let rank0 = arena.upload(&payload[0])?;
                    let rank1 = arena.upload(&payload[1])?;
                    let output = arena.alloc(64 * hidden * 2)?;
                    let reference_sum = arena.alloc(output.bytes)?;
                    for bucket in BUCKETS {
                        let init_logits = vec![0; logits.bytes];
                        let init_slots: Vec<u8> = (0..64i64).flat_map(|i| i.to_ne_bytes()).collect();
                        lib.copy_h2d(logits, &init_logits)?;
                        lib.copy_h2d(slots, &init_slots)?;
                        // SAFETY: warm lazy kernels first; arena-owned arrays have the fixed bucket extents.
                        unsafe {
                            lib.dsv4_router_select_masked(logits.ptr, bias_ptr, table_ptr, tokens.ptr,
                                slots.ptr.cast(), ids.ptr, weights.ptr, bucket, experts, topk, 1.5, arena.stream)?;
                            combine.dsv4_sum_masked(rank0.ptr, rank1.ptr, output.ptr.cast(), slots.ptr.cast(),
                                bucket, hidden, dtype, arena.stream)?;
                            lib.cuda_stream_synchronize(arena.stream)?;
                            lib.cuda_graph_begin_capture(arena.stream)?;
                            lib.dsv4_router_select_masked(logits.ptr, bias_ptr, table_ptr, tokens.ptr,
                                slots.ptr.cast(), ids.ptr, weights.ptr, bucket, experts, topk, 1.5, arena.stream)?;
                            combine.dsv4_sum_masked(rank0.ptr, rank1.ptr, output.ptr.cast(), slots.ptr.cast(),
                                bucket, hidden, dtype, arena.stream)?;
                        }
                        // SAFETY: this stream owns the just-started capture, all arrays stay retained.
                        let graph = Captured { library: &lib, stream: arena.stream,
                            exec: unsafe { lib.cuda_graph_end_capture(arena.stream)? } };
                        for rows in (1..=64).filter(|&rows| BUCKETS.iter().copied().find(|&b| b >= rows) == Some(bucket)) {
                            // Repeated values exercise ties; +/- large logits exercise both transform branches.
                            let input: Vec<u8> = (0..64 * experts).flat_map(|i| {
                                let value: f32 = if i / experts >= rows { 0.0 } else { match i % 11 {
                                    0 => 30.0, 1 => -30.0, _ => ((i % 7) as f32 - 3.0) * 0.25,
                                }};
                                value.to_ne_bytes()
                            }).collect();
                            lib.copy_h2d(reference_logits, &input)?;
                            lib.copy_h2d(rank0, &payload[0])?;
                            lib.copy_h2d(rank1, &payload[1])?;
                            // SAFETY: unmasked exact-width oracle uses the same initialized operands and row-local arithmetic.
                            unsafe {
                                lib.dsv4_router_select(reference_logits.ptr, bias_ptr, table_ptr, tokens.ptr,
                                    reference_ids.ptr, reference_weights.ptr, rows, experts, topk, 1.5, arena.stream)?;
                                combine.sum(rank0.ptr, rank1.ptr, reference_sum.ptr.cast(), rows * hidden, dtype, arena.stream)?;
                                lib.cuda_stream_synchronize(arena.stream)?;
                            }
                            let expected_ids = read_bytes(&lib, reference_ids, rows * topk * 4)?;
                            let expected_weights = read_bytes(&lib, reference_weights, rows * topk * 4)?;
                            let expected_logits = read_bytes(&lib, reference_logits, rows * experts * 4)?;
                            let expected_sum = read_bytes(&lib, reference_sum, rows * hidden * 2)?;
                            for holes in [false, true] {
                                let valid: Vec<bool> = (0..64).map(|row| row < rows && (!holes || row % 3 != 1)).collect();
                                let metadata: Vec<u8> = valid.iter().enumerate().flat_map(|(row, &live)|
                                    (if live { row as i64 } else { -(1i64 << 62) }).to_ne_bytes()).collect();
                                lib.copy_h2d(slots, &metadata)?;
                                lib.copy_h2d(logits, &input)?;
                                for (buffer, bytes) in [(rank0, &payload[0]), (rank1, &payload[1])] {
                                    let mut poison = bytes.clone();
                                    for row in 0..64 {
                                        if !valid[row] {
                                            let nan = match dtype { RtxPartialDtype::Bf16 => 0x7fc1u16.to_ne_bytes().to_vec(),
                                                RtxPartialDtype::F32 => f32::NAN.to_ne_bytes().to_vec() };
                                            for element in poison[row * hidden * nan.len()..(row + 1) * hidden * nan.len()].chunks_exact_mut(nan.len()) {
                                                element.copy_from_slice(&nan);
                                            }
                                        }
                                    }
                                    lib.copy_h2d(buffer, &poison)?;
                                }
                                // SAFETY: same executable/grid/pointers; only uploaded validity and input bytes change.
                                unsafe { lib.cuda_graph_launch(graph.exec, arena.stream)?; lib.cuda_stream_synchronize(arena.stream)?; }
                                let actual_ids = read_bytes(&lib, ids, bucket * topk * 4)?;
                                let actual_weights = read_bytes(&lib, weights, bucket * topk * 4)?;
                                let actual_logits = read_bytes(&lib, logits, rows * experts * 4)?;
                                let actual_sum = read_bytes(&lib, output, bucket * hidden * 2)?;
                                ensure!(actual_logits == expected_logits, "router logits differ {geometry:?} rows={rows} hash={hash}");
                                for row in 0..bucket {
                                    let route = row * topk * 4..(row + 1) * topk * 4;
                                    let sum = row * hidden * 2..(row + 1) * hidden * 2;
                                    if valid[row] {
                                        ensure!(actual_ids[route.clone()] == expected_ids[route.clone()] &&
                                            actual_weights[route.clone()] == expected_weights[route],
                                            "live router bits differ device={device} {geometry:?} rows={rows} bucket={bucket} hash={hash}");
                                        ensure!(actual_sum[sum.clone()] == expected_sum[sum],
                                            "live sum bits differ device={device} {geometry:?} rows={rows} bucket={bucket} dtype={dtype:?}");
                                    } else {
                                        ensure!(actual_ids[route.clone()].iter().all(|&v| v == 255) &&
                                            actual_weights[route].iter().all(|&v| v == 0) && actual_sum[sum].iter().all(|&v| v == 0),
                                            "dummy row is not invalid/+0 device={device} rows={rows} bucket={bucket} holes={holes}");
                                    }
                                }
                            }
                        }
                    }
                    println!("V4 fixed-bucket mask bytes passed device={device} {geometry:?} hash={hash} dtype={dtype:?} widths=1..64");
                }
            }
        }
    }
    Ok(())
}
