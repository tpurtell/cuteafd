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
