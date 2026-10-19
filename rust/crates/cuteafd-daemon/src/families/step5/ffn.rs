//! Admitted post-attention dense FFN. This is not a serving executor.
use super::weights::load_operand;
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, FfnActivation};
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_loader::plan::{
    checkpoint::Checkpoint,
    families::step5::coordinator::{CoordinatorContract, RankWeightLayout, WeightOperand},
};

const HIDDEN: usize = 4096;
const DENSE: usize = 13824;

#[derive(Debug)]
struct Admission {
    rows: usize,
    planes: [usize; 6],
    bytes: u64,
}

impl Admission {
    fn new(
        rows: usize,
        weights: &[&WeightOperand],
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= 4096,
            "Step-5 dense row capacity must be 1..=4096"
        );
        let shapes = [
            (HIDDEN, 4),
            (HIDDEN, 2),
            (2 * DENSE, 2),
            (DENSE, 2),
            (HIDDEN, 2),
            (HIDDEN, 4),
        ];
        let mut planes = [0; 6];
        let mut bytes = 0u64;
        for (plane, (width, elem)) in planes.iter_mut().zip(shapes) {
            *plane = rows
                .checked_mul(width)
                .and_then(|n| n.checked_mul(elem))
                .context("dense workspace overflow")?
                .max(256);
            bytes = bytes
                .checked_add(*plane as u64)
                .context("dense workspace overflow")?;
        }
        for weight in weights {
            bytes = bytes
                .checked_add(weight.allocation_bytes)
                .context("dense weight admission overflow")?;
        }
        let total = reservations
            .iter()
            .try_fold(bytes, |n, v| n.checked_add(*v))
            .context("dense reservation overflow")?;
        ensure!(
            total <= available,
            "Step-5 dense chain and reservations need {total} bytes, available {available}"
        );
        Ok(Self {
            rows,
            planes,
            bytes,
        })
    }
}

pub(super) struct DenseFfn<'a> {
    // Fields drop in declaration order: drain before storage or programs retire.
    stream: Stream<'a>,
    weights: Vec<Allocation<'a>>,
    planes: Vec<Allocation<'a>>,
    programs: Vec<Program<'a>>,
    admission: Admission,
    epsilon: f32,
}

pub(super) struct DensePlan {
    layout: RankWeightLayout,
    weights: Vec<WeightOperand>,
    admission: Admission,
    activation: String,
    epsilon: f32,
}

impl DensePlan {
    pub fn bytes(&self) -> u64 {
        self.admission.bytes
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.admission.planes.iter().map(|&n| n as u64).sum()
    }

    pub fn prepare(
        programs: &Programs<'_>,
        checkpoint: &Checkpoint,
        layer: usize,
        rows: usize,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let layer_contract = contract
            .layers
            .get(layer)
            .context("dense target layer out of range")?;
        let Some(FfnActivation::SiluPostClampFp32Product { limit }) =
            layer_contract.dense_activation
        else {
            anyhow::bail!("Step-5 dense chain requires a dense target layer");
        };
        ensure!(
            limit == 7.0 || limit == 10.0,
            "unsupported Step-5 dense clamp {limit}"
        );
        // Dense weights are currently replicated under TP2; no collective is implied here.
        let layout = RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
        let names = [
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            format!("model.layers.{layer}.mlp.gate_up_proj.weight"),
            format!("model.layers.{layer}.mlp.down_proj.weight"),
        ];
        let weights = names
            .iter()
            .map(|name| {
                layout
                    .operands
                    .iter()
                    .find(|w| &w.name == name)
                    .context("dense operand missing")
            })
            .collect::<Result<Vec<_>>>()?;
        for (weight, (dtype, shape)) in weights.iter().zip([
            (DType::F32, vec![HIDDEN]),
            (DType::Bf16, vec![2 * DENSE, HIDDEN]),
            (DType::Bf16, vec![HIDDEN, DENSE]),
        ]) {
            ensure!(
                weight.dtype == dtype && weight.shape == shape,
                "Step-5 dense operand shape/dtype mismatch"
            );
        }
        let admission = Admission::new(rows, &weights, available, reservations)?;
        let activation = format!("step5_post_silu_w13824_c{}", limit as u32);
        let definitions: [(&str, &[&str], &[u8]); 5] = [
            (
                "step5_hidden_norm",
                &["x", "weight", "bias", "cos", "sin", "positions", "out"],
                b"if",
            ),
            ("step5_dense_gate_up", &["x", "w", "out"], b"i"),
            (&activation, &["gate_up", "out"], b"i"),
            ("step5_dense_down", &["x", "w", "out"], b"i"),
            ("step5_residual_bf16", &["residual", "delta", "out"], b"i"),
        ];
        // Validate the complete ABI before loading any module or allocating any plane.
        for (name, pointers, scalars) in definitions {
            let spec = programs.spec(name)?;
            ensure!(
                spec.pointers == pointers
                    && spec.scalar_kinds == scalars
                    && spec.capacity_rows as usize >= rows
                    && spec.scratch.is_empty(),
                "{name}: dense ABI/capacity mismatch"
            );
        }
        let norm = &programs.spec("step5_hidden_norm")?.geometry;
        ensure!(
            norm["width"] == HIDDEN
                && norm["heads"] == 1
                && norm["zero_centered"] == true
                && norm["bias"] == false
                && norm["subtract_mean"] == false
                && norm["rotary"] == 0
                && norm["epsilon"] == "runtime_float32",
            "dense norm geometry mismatch"
        );
        for (name, width, outputs) in [
            ("step5_dense_gate_up", HIDDEN, 2 * DENSE),
            ("step5_dense_down", DENSE, HIDDEN),
        ] {
            let spec = &programs.spec(name)?.geometry;
            ensure!(
                spec["width"] == width
                    && spec["outputs"] == outputs
                    && spec["output_dtype"] == "torch.bfloat16"
                    && spec["skinny_rows"] == 0,
                "{name}: dense projection geometry mismatch"
            );
        }
        let post = &programs.spec(&activation)?.geometry;
        ensure!(
            post["width"] == DENSE
                && post["limit"].as_f64() == Some(limit as f64)
                && programs.spec("step5_residual_bf16")?.geometry["width"] == HIDDEN,
            "dense pointwise geometry mismatch"
        );
        let weights = weights.into_iter().cloned().collect();
        Ok(Self {
            layout,
            weights,
            admission,
            activation,
            epsilon: contract.norms.post_attention,
        })
    }
}

impl<'a> DenseFfn<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer: usize,
        rows: usize,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        Self::load_plan(
            device,
            programs,
            checkpoint,
            DensePlan::prepare(programs, checkpoint, layer, rows, available, reservations)?,
        )
    }

    pub fn load_plan(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        plan: DensePlan,
    ) -> Result<Self> {
        let DensePlan {
            layout,
            weights,
            admission,
            activation,
            epsilon,
        } = plan;
        let definitions: [(&str, &[&str], &[u8]); 5] = [
            (
                "step5_hidden_norm",
                &["x", "weight", "bias", "cos", "sin", "positions", "out"],
                b"if",
            ),
            ("step5_dense_gate_up", &["x", "w", "out"], b"i"),
            (&activation, &["gate_up", "out"], b"i"),
            ("step5_dense_down", &["x", "w", "out"], b"i"),
            ("step5_residual_bf16", &["residual", "delta", "out"], b"i"),
        ];
        let resolved = definitions
            .iter()
            .map(|(name, pointers, _)| device.run(|| programs.program(name, pointers)))
            .collect::<Result<Vec<_>>>()?;
        let loaded = weights
            .into_iter()
            .map(|w| load_operand(device, checkpoint, &layout, &w))
            .collect::<Result<Vec<_>>>()?;
        let _scope = cuteafd_ffi::memory_ledger::scope("step5_dense_workspace");
        let planes = admission
            .planes
            .iter()
            .map(|&bytes| Allocation::new(device, bytes))
            .collect::<Result<Vec<_>>>()?;
        let stream = Stream::new(device)?;
        Ok(Self {
            stream,
            weights: loaded,
            planes,
            programs: resolved,
            admission,
            epsilon,
        })
    }

    pub fn enqueue(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.admission.rows,
            "dense live rows exceed admitted capacity"
        );
        for stage in 0..self.programs.len() {
            self.enqueue_stage(stage, rows)?;
        }
        Ok(())
    }

    pub fn input(&self) -> cuteafd_ffi::CuteafdDeviceBuffer {
        self.planes[0].buffer
    }
    pub fn output(&self) -> cuteafd_ffi::CuteafdDeviceBuffer {
        self.planes[5].buffer
    }

    // The containing layer owns this stream ahead of the FFN fields and drains
    // it before releasing us. Standalone callers keep using the owned stream.
    pub fn enqueue_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.enqueue_range_on(0, rows, stream)
    }

    pub fn enqueue_range_on(&self, start: usize, rows: usize, stream: &Stream<'a>) -> Result<()> {
        ensure!(
            stream.device.id == self.stream.device.id,
            "dense stream device mismatch"
        );
        for stage in 0..5 {
            self.stage_range_on(stage, start, rows, stream)?;
        }
        Ok(())
    }

    fn enqueue_stage(&self, stage: usize, rows: usize) -> Result<()> {
        self.stage_on(stage, rows, &self.stream)
    }

    fn stage_on(&self, stage: usize, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.stage_range_on(stage, 0, rows, stream)
    }

    fn stage_range_on(
        &self,
        stage: usize,
        start: usize,
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        ensure!(
            stage < 5
                && rows > 0
                && start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.admission.rows),
            "dense stage/capacity mismatch"
        );
        let p = self
            .planes
            .iter()
            .zip([
                HIDDEN * 4,
                HIDDEN * 2,
                2 * DENSE * 2,
                DENSE * 2,
                HIDDEN * 2,
                HIDDEN * 4,
            ])
            .map(|(a, stride)| a.buffer.ptr.wrapping_byte_add(start * stride))
            .collect::<Vec<_>>();
        let w = self
            .weights
            .iter()
            .map(|a| a.buffer.ptr)
            .collect::<Vec<_>>();
        let pointers = [
            vec![p[0], w[0], w[0], w[0], w[0], w[0], p[1]],
            vec![p[1], w[1], p[2]],
            vec![p[2], p[3]],
            vec![p[3], w[2], p[4]],
            vec![p[0], p[4], p[5]],
        ];
        self.stream.device.run(|| {
            let mut scalars = vec![Scalar::I32(rows as i32)];
            if stage == 0 {
                scalars.push(Scalar::F32(self.epsilon));
            }
            // SAFETY: all buffers are admitted, owned on this device, and retained
            // beyond the stream drain. Ignored nonrotary norm operands are never read.
            unsafe {
                self.programs[stage].launch(&pointers[stage], &scalars, stream.raw)?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_admission_charges_capacity_and_allocator_floors() -> Result<()> {
        let plan = Admission::new(1, &[], u64::MAX, &[])?;
        assert_eq!(plan.bytes, (HIDDEN * 12 + DENSE * 6) as u64);
        assert!(Admission::new(0, &[], u64::MAX, &[]).is_err());
        assert!(Admission::new(4097, &[], u64::MAX, &[]).is_err());
        assert!(Admission::new(1, &[], plan.bytes - 1, &[]).is_err());
        assert!(Admission::new(1, &[], plan.bytes, &[1]).is_err());
        assert!(Admission::new(1, &[], u64::MAX, &[u64::MAX]).is_err());
        Ok(())
    }

    fn fixture(root: &std::path::Path, manifest: &serde_json::Value, key: &str) -> Result<Vec<u8>> {
        use sha2::{Digest, Sha256};
        let entry = &manifest["files"][key];
        let bytes = std::fs::read(root.join(format!("{key}.bin")))?;
        ensure!(
            entry["bytes"].as_u64() == Some(bytes.len() as u64)
                && entry["sha256"].as_str()
                    == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
            "fixture identity mismatch: {key}"
        );
        Ok(bytes)
    }

    fn values(bytes: &[u8], f32: bool) -> Result<Vec<f64>> {
        ensure!(
            bytes.len() % if f32 { 4 } else { 2 } == 0,
            "incomplete fixture value"
        );
        let values = if f32 {
            bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
                .collect::<Vec<_>>()
        } else {
            bytes
                .chunks_exact(2)
                .map(|v| {
                    f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16)
                        as f64
                })
                .collect::<Vec<_>>()
        };
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite native/fixture output"
        );
        Ok(values)
    }

    #[test]
    #[ignore = "requires sealed native library, pinned real-weight component fixtures and CUDA"]
    fn native_dense_ffn_official_boundaries_and_graph_replay() -> Result<()> {
        native_dense_component(0)
    }

    #[test]
    #[ignore = "requires pinned dense layer-1/2 fixtures, sealed native library and CUDA"]
    fn native_dense_layers1_2_boundaries_and_graph_replay() -> Result<()> {
        for layer in 1..3 {
            native_dense_component(layer)?;
        }
        Ok(())
    }

    impl DenseFfn<'_> {
        pub(crate) fn qualify_saved_boundaries(
            &self,
            root: &std::path::Path,
            manifest: &serde_json::Value,
            rows: usize,
        ) -> Result<()> {
            let load = |name: &str| fixture(root, manifest, &format!("r{rows}_{name}"));
            let read = |plane: usize, bytes: usize| -> Result<Vec<u8>> {
                let mut output = vec![0; bytes];
                self.stream.device.run(|| {
                    self.stream
                        .device
                        .library
                        .copy_d2h(&mut output, self.planes[plane].buffer)
                })?;
                Ok(output)
            };
            for (plane, name, width, size) in [
                (1, "ffn_norm", HIDDEN, 2),
                (2, "gate_up", 2 * DENSE, 2),
                (3, "activation", DENSE, 2),
                (4, "down", HIDDEN, 2),
                (5, "output", HIDDEN, 4),
            ] {
                let actual = read(plane, rows * width * size)?;
                let expected = load(name)?;
                let a = values(&actual, size == 4)?;
                let b = values(&expected, size == 4)?;
                let rel = (a.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
                    / b.iter().map(|v| v * v).sum::<f64>())
                .sqrt();
                eprintln!("layer91 chained boundary={name} rel_l2={rel:.12e}");
                ensure!(rel < 0.03, "layer91 chained boundary {name}");
            }
            self.stream.device.run(|| {
                self.stream
                    .device
                    .library
                    .copy_h2d(self.input(), &load("post")?)
            })?;
            for (stage, input, name, plane) in [
                (0, "post", "ffn_norm", 0),
                (1, "ffn_norm", "gate_up", 1),
                (2, "gate_up", "activation", 2),
                (3, "activation", "down", 3),
                (4, "down", "output", 4),
            ] {
                self.stream.device.run(|| {
                    self.stream
                        .device
                        .library
                        .copy_h2d(self.planes[plane].buffer, &load(input)?)
                })?;
                self.enqueue_stage(stage, rows)?;
                self.stream.drain()?;
                let official = load(name)?;
                let actual = read(stage + 1, official.len())?;
                if stage == 0 {
                    super::super::norm_accuracy::check(
                        &actual,
                        &official,
                        &load("ffn_norm_fp64")?,
                        HIDDEN,
                    )?;
                } else if matches!(stage, 2 | 4) {
                    ensure!(actual == official, "layer91 isolated pointwise {name}");
                } else {
                    let reference = load(&format!("{name}_fp64"))?;
                    let reference = reference
                        .chunks_exact(8)
                        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let error = |bytes: &[u8]| -> Result<f64> {
                        Ok(values(bytes, false)?
                            .iter()
                            .zip(&reference)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>())
                    };
                    let native = error(&actual)?;
                    let baseline = error(&official)?;
                    eprintln!(
                        "layer91 isolated projection={name} FP64_error_ratio={}",
                        (native / baseline).sqrt()
                    );
                    ensure!(
                        native <= baseline * 1.01f64.powi(2),
                        "layer91 projection FP64 bar {name}"
                    );
                    let width = if stage == 1 { 2 * DENSE } else { HIDDEN };
                    let actual = values(&actual, false)?;
                    let official = values(&official, false)?;
                    for (row, ((a, b), r)) in actual.chunks_exact(width)
                        .zip(official.chunks_exact(width)).zip(reference.chunks_exact(width)).enumerate() {
                        let error = |v: &[f64]| v.iter().zip(r).map(|(v, r)| (v-r).powi(2)).sum::<f64>();
                        ensure!(error(a) <= error(b) * 1.01f64.powi(2),
                            "layer91 projection per-row FP64 bar {name}, row {row}");
                    }
                }
            }
            Ok(())
        }
    }

    fn native_dense_component(layer: usize) -> Result<()> {
        let mut root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_FFN_FIXTURE_DIR")?);
        if layer > 0 {
            root.push(format!("layer{layer}"));
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["schema"] == 1
                && manifest["seed"] == 274
                && manifest["layer"] == layer
                && manifest["checkpoint"] == "stepfun-ai/Step-5-Preview-BF16"
                && manifest["revision"] == "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
                && manifest["rows"] == serde_json::json!([1, 7, 128]),
            "unsupported dense fixture identity"
        );
        // SAFETY: this explicit GPU qualification supplies a sealed native library.
        let library =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library
            .programs()?
            .with_manifest(std::path::Path::new(&std::env::var(
                "CUTEAFD_PROGRAM_MANIFEST",
            )?))?;
        let checkpoint =
            Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        ensure!(
            checkpoint.snapshot.file_name().and_then(|v| v.to_str())
                == manifest["revision"].as_str(),
            "native checkpoint revision differs from fixture"
        );
        let mut chain = DenseFfn::load(device, &programs, &checkpoint, layer, 128, u64::MAX, &[])?;
        ensure!(
            manifest["epsilon"].as_f64().map(|v| v as f32) == Some(chain.epsilon),
            "fixture epsilon differs"
        );
        let bytes = chain.admission.bytes;
        assert!(
            DenseFfn::load(device, &programs, &checkpoint, layer, 128, bytes - 1, &[]).is_err()
        );
        assert!(DenseFfn::load(device, &programs, &checkpoint, layer, 128, bytes, &[1]).is_err());
        assert!(chain.enqueue(129).is_err());
        let mut elementwise_exact = true;
        let mut gemms_accurate = true;
        for rows in [1, 7, 128] {
            let residual = fixture(
                &root,
                &manifest,
                &format!("official_native_r{rows}_residual"),
            )?;
            device.run(|| library.copy_h2d(chain.planes[0].buffer, &residual))?;
            chain.enqueue(rows)?;
            chain.stream.drain()?;
            let mut eager = Vec::new();
            for (index, name) in ["norm", "gate_up", "activation", "down", "output"]
                .iter()
                .enumerate()
            {
                let f32 = *name == "output";
                let width = match *name {
                    "gate_up" => 2 * DENSE,
                    "activation" => DENSE,
                    _ => HIDDEN,
                };
                let mut actual = vec![0; rows * width * if f32 { 4 } else { 2 }];
                device.run(|| library.copy_d2h(&mut actual, chain.planes[index + 1].buffer))?;
                let a = values(&actual, f32)?;
                let mut exact = false;
                for oracle in ["official_native", "official_f32"] {
                    let expected = fixture(&root, &manifest, &format!("{oracle}_r{rows}_{name}"))?;
                    let b = values(&expected, f32 || oracle == "official_f32")?;
                    ensure!(a.len() == b.len(), "fixture shape mismatch");
                    let rel_l2 = (a.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
                        / b.iter().map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE))
                    .sqrt();
                    let equal = a
                        .iter()
                        .zip(&b)
                        .filter(|(a, b)| a.to_bits() == b.to_bits())
                        .count();
                    eprintln!("dense rows={rows} boundary={name} oracle={oracle} equal={equal}/{} rel_l2={rel_l2:.9e}", a.len());
                    if oracle == "official_native" {
                        exact = actual == expected;
                        if *name == "norm" {
                            super::super::norm_accuracy::check(
                                &actual,
                                &expected,
                                &fixture(&root, &manifest, &format!("fp64_r{rows}_norm"))?,
                                HIDDEN,
                            )?;
                        }
                    }
                }
                eprintln!("dense rows={rows} boundary={name} chained_byte_exact={exact}");
                eager.push(actual);
            }
            // Capture contains only pre-resolved launches and stable admitted pointers.
            device.run(|| {
                // SAFETY: the stream is idle, and all chain resources outlive capture/replay.
                unsafe { library.cuda_graph_begin_capture(chain.stream.raw) }
            })?;
            let enqueue = chain.enqueue(rows);
            let graph = device.run(|| {
                // SAFETY: capture was begun on this stream above, even if enqueue failed.
                unsafe { library.cuda_graph_end_capture(chain.stream.raw) }
            })?;
            // SAFETY: the mutable borrow pins the entire chain until this graph is dropped.
            let graph = unsafe {
                crate::shared::decode_graph::GraphOwner::new(
                    &library, device.id, graph, &mut chain,
                )?
            };
            enqueue?;
            let mut drain = graph
                .pins
                .stream
                .cancellation_guard("interrupted dense graph replay");
            device.run(|| {
                // SAFETY: graph pins the same device/stream/storage until replay drains.
                unsafe { library.cuda_graph_launch(graph.raw, graph.pins.stream.raw) }
            })?;
            graph.pins.stream.drain()?;
            drain.complete = true;
            drop(drain);
            for (index, expected) in eager.iter().enumerate() {
                let mut actual = vec![0; expected.len()];
                device
                    .run(|| library.copy_d2h(&mut actual, graph.pins.planes[index + 1].buffer))?;
                ensure!(
                    actual == *expected,
                    "dense graph replay differs at boundary {index}"
                );
            }
            drop(graph);
            // Isolate each consumer with its official native-dtype producer output.
            for (stage, input, output, plane) in [
                (1, "norm", "gate_up", 1),
                (2, "gate_up", "activation", 2),
                (3, "activation", "down", 3),
                (4, "down", "output", 4),
            ] {
                let input = fixture(
                    &root,
                    &manifest,
                    &format!("official_native_r{rows}_{input}"),
                )?;
                device.run(|| library.copy_h2d(chain.planes[plane].buffer, &input))?;
                chain.enqueue_stage(stage, rows)?;
                chain.stream.drain()?;
                let expected = fixture(
                    &root,
                    &manifest,
                    &format!("official_native_r{rows}_{output}"),
                )?;
                let mut actual = vec![0; expected.len()];
                device.run(|| library.copy_d2h(&mut actual, chain.planes[stage + 1].buffer))?;
                if matches!(stage, 2 | 4) {
                    eprintln!(
                        "isolated rows={rows} op={output} byte_exact={}",
                        actual == expected
                    );
                    elementwise_exact &= actual == expected;
                } else {
                    let fp64 = fixture(&root, &manifest, &format!("fp64_r{rows}_{output}"))?;
                    ensure!(
                        fp64.len() == expected.len() * 4,
                        "FP64 fixture shape mismatch"
                    );
                    let reference = fp64
                        .chunks_exact(8)
                        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    ensure!(
                        reference.iter().all(|v| v.is_finite()),
                        "nonfinite FP64 oracle"
                    );
                    let norm = reference.iter().map(|v| v * v).sum::<f64>();
                    let error = |bytes: &[u8]| -> Result<f64> {
                        Ok((values(bytes, false)?
                            .iter()
                            .zip(&reference)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            / norm)
                            .sqrt())
                    };
                    let native = error(&actual)?;
                    let official = error(&expected)?;
                    // Canonical MMA and cuBLAS accumulate in different orders.
                    // Measured native/official errors were 1.4820e-3..1.8017e-3,
                    // differing by at most 0.00208% relative; the explicit gate is 1%.
                    let equivalent = native <= official * 1.01;
                    eprintln!("isolated rows={rows} op={output} native_fp64_rel_l2={native:.12e} official_fp64_rel_l2={official:.12e} error_ratio={:.9} within_one_percent={equivalent}", native / official);
                    gemms_accurate &= equivalent;
                }
            }
        }
        ensure!(
            elementwise_exact,
            "isolated pointwise boundary differs from official native dtype"
        );
        ensure!(
            gemms_accurate,
            "native GEMM FP64 error exceeds 1.01 times official same-operand cuBLAS error"
        );
        Ok(())
    }
}
