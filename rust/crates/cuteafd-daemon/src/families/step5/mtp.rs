//! Predictor input and independent cache coordinates; no scheduler frontier advances here.
use super::weights::load_operand;
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer};
use cuteafd_loader::plan::{checkpoint::Checkpoint,
    families::step5::{Config, coordinator::{RankWeightLayout, WeightOperand}}};
use std::rc::Rc;

const H: usize = 4096;
const NORM_POINTERS: &[&str] = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
const EH_POINTERS: &[&str] = &["x", "w", "out"];

fn planes(rows: usize) -> Result<[usize; 6]> {
    ensure!((1..=4096).contains(&rows), "Step-5 predictor input row capacity");
    // BF16 hidden, normalized embedding/hidden, concat, EH, F32 residual.
    Ok([2, 2, 2, 4, 2, 4].map(|element| rows * H * element))
}

fn validate_programs(norm: &ProgramSpec, eh: &ProgramSpec, rows: usize) -> Result<()> {
    planes(rows)?;
    let g = &norm.geometry;
    ensure!(norm.pointers == NORM_POINTERS && norm.scalar_kinds == b"if"
        && norm.capacity_rows as usize >= rows && norm.scratch.is_empty()
        && g["width"] == H && g["heads"] == 1 && g["zero_centered"] == true
        && g["subtract_mean"] == false && g["bias"] == false && g["rotary"] == 0
        && g["epsilon"] == "runtime_float32", "Step-5 predictor input norm ABI/geometry");
    let g = &eh.geometry;
    ensure!(eh.pointers == EH_POINTERS && eh.scalar_kinds == b"i"
        && eh.capacity_rows as usize >= rows && eh.scratch.is_empty()
        && g["width"] == 2 * H && g["outputs"] == H
        && g["output_dtype"] == "torch.bfloat16" && g["skinny_rows"] == 0,
        "Step-5 predictor EH ABI/geometry");
    Ok(())
}

pub(super) struct InputPlan {
    layout: RankWeightLayout,
    weights: Vec<WeightOperand>,
    rows: usize,
    epsilon: f32,
    bytes: u64,
}
impl InputPlan {
    pub(super) fn prepare(programs: &Programs<'_>, checkpoint: &Checkpoint,
        stage: usize, rows: usize) -> Result<Self> {
        let layout = RankWeightLayout::for_mtp_stage(checkpoint, stage)?;
        let cfg = Config::parse(checkpoint)?;
        let prefix = format!("model.layers.{}.", cfg.layers + stage);
        let weights = [("enorm.weight", DType::F32, vec![H]),
            ("hnorm.weight", DType::F32, vec![H]),
            ("eh_proj.weight", DType::Bf16, vec![H, 2 * H])].into_iter()
            .map(|(suffix, dtype, shape)| {
                let name = format!("{prefix}{suffix}");
                let weight = layout.operands.iter().find(|w| w.name == name)
                    .context("Step-5 predictor input operand missing")?;
                ensure!(weight.dtype == dtype && weight.shape == shape,
                    "Step-5 predictor input operand shape/dtype: {name}");
                Ok(weight.clone())
            }).collect::<Result<Vec<_>>>()?;
        validate_programs(programs.spec("step5_final_norm")?,
            programs.spec("step5_mtp_eh")?, rows)?;
        let bytes = weights.iter().map(|w| w.allocation_bytes)
            .chain(planes(rows)?.into_iter().map(|n| n as u64))
            .try_fold(0u64, u64::checked_add).context("Step-5 predictor input byte overflow")?;
        Ok(Self { layout, weights, rows, epsilon: cfg.norm_eps, bytes })
    }

    pub(super) fn bytes(&self) -> u64 { self.bytes }
}

struct InputWeights<'a> {
    norm: Program<'a>,
    eh: Program<'a>,
    weights: Vec<Allocation<'a>>,
    epsilon: f32,
}

pub(super) struct InputOwner<'a> {
    // This lane stream drains before any borrowed programs or storage retire.
    stream: Rc<Stream<'a>>,
    immutable: Rc<InputWeights<'a>>,
    planes: Vec<Allocation<'a>>,
    rows: usize,
}
impl<'a> InputOwner<'a> {
    pub(super) fn load(device: Device<'a>, programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint, plan: InputPlan, stream: Rc<Stream<'a>>,
        available: u64, reservations: &[u64]) -> Result<Self> {
        let bytes = reservations.iter().try_fold(plan.bytes, |n, v| n.checked_add(*v))
            .context("Step-5 predictor input reservation overflow")?;
        ensure!(bytes <= available && stream.device.id == device.id
            && std::ptr::eq(stream.device.library, device.library),
            "Step-5 predictor input admission/device mismatch");
        let immutable = device.run(|| {
            let norm = programs.program("step5_final_norm", NORM_POINTERS)?;
            let eh = programs.program("step5_mtp_eh", EH_POINTERS)?;
            let weights = plan.weights.iter()
                .map(|w| load_operand(device, checkpoint, &plan.layout, w))
                .collect::<Result<Vec<_>>>()?;
            Ok(Rc::new(InputWeights { norm, eh, weights, epsilon: plan.epsilon }))
        })?;
        Self::workspace(stream, immutable, plan.rows)
    }

    fn workspace(stream: Rc<Stream<'a>>, immutable: Rc<InputWeights<'a>>,
        rows: usize) -> Result<Self> {
        let planes = planes(rows)?.into_iter().map(|n| Allocation::new(stream.device, n))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stream, immutable, planes, rows })
    }

    pub(super) fn new_lane(&self, stream: Rc<Stream<'a>>) -> Result<Self> {
        ensure!(stream.device.id == self.stream.device.id
            && std::ptr::eq(stream.device.library, self.stream.device.library)
            && stream.raw != self.stream.raw, "Step-5 predictor input lane device/identity");
        Self::workspace(stream, self.immutable.clone(), self.rows)
    }

    /// Embedding BF16 and the previous predictor's own F32 residual are retained
    /// by the enclosing transaction through this stream's drain. Stage zero uses
    /// target hidden, not a different predictor's cache or normalized head input.
    pub(super) fn enqueue(&self, embedding: CuteafdDeviceBuffer,
        previous_hidden: CuteafdDeviceBuffer, rows: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!((1..=self.rows).contains(&rows), "Step-5 predictor input active rows");
        for (buffer, element) in [(embedding, 2), (previous_hidden, 4)] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= rows * H * element
                && buffer.device_id == self.stream.device.id,
                "Step-5 predictor input buffer extent/device");
        }
        let mut guard = self.stream.cancellation_guard("Step-5 predictor input borrow");
        self.stream.device.run(|| {
            let library = self.stream.device.library;
            let p = |n: usize| self.planes[n].buffer.ptr;
            let w = |n: usize| self.immutable.weights[n].buffer.ptr;
            let scalars = [Scalar::I32(rows as i32), Scalar::F32(self.immutable.epsilon)];
            // SAFETY: checked external inputs and all private, disjoint planes
            // stay live until this stream drains. Norm's unused operands are ignored.
            unsafe {
                library.cuda_f32_to_bf16_async(previous_hidden, self.planes[0].buffer,
                    rows * H, self.stream.raw)?;
                self.immutable.norm.launch(&[embedding.ptr, w(0), w(0), w(0), w(0), w(0), p(1)],
                    &scalars, self.stream.raw)?;
                self.immutable.norm.launch(&[p(0), w(1), w(1), w(1), w(1), w(1), p(2)],
                    &scalars, self.stream.raw)?;
                library.glm_dflash_tap(p(1), p(3), rows, H, 2 * H, 0, self.stream.raw)?;
                library.glm_dflash_tap(p(2), p(3), rows, H, 2 * H, H, self.stream.raw)?;
                self.immutable.eh.launch(&[p(3), w(2), p(4)],
                    &[Scalar::I32(rows as i32)], self.stream.raw)?;
                library.cuda_zero_bytes_async(self.planes[5].buffer, rows * H * 4, self.stream.raw)?;
                library.cuda_accumulate_bf16_to_f32_async(self.planes[4].buffer,
                    self.planes[5].buffer, rows * H, self.stream.raw)?;
            }
            Ok(())
        })?;
        guard.complete = true;
        Ok(self.planes[5].buffer)
    }
}
impl Drop for InputOwner<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Step-5 predictor input owner"); }
}

/// Teacher stage s starts at global s+1. KV rows are local to that stage;
/// page-aligned rebasing bounds table storage without changing RoPE or key order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Coordinates {
    global: Vec<i64>,
    local: Vec<i64>,
    table: Vec<i32>,
    base: u64,
    end: u64,
}
impl Coordinates {
    fn new(stage: usize, base: u64, rows: usize, context: u64,
        ring: cuteafd_core::prefix::AppendRing) -> Result<Self> {
        ensure!(stage < 3 && (1..=4096).contains(&rows)
            && ring.window == 512 && ring.rows > 0 && ring.rows % 64 == 0
            && ring.step > 0 && ring.rows >= ring.window.saturating_add(ring.step),
            "Step-5 predictor coordinate geometry");
        let end = base.checked_add(rows as u64).context("Step-5 predictor frontier overflow")?;
        let origin = stage as u64 + 1;
        ensure!(rows as u64 <= ring.step && end.checked_add(origin).is_some_and(|n|
            n <= context && n <= i64::MAX as u64), "Step-5 predictor context/step extent");
        let shift = base.saturating_sub(ring.window) / 64 * 64;
        let width = (end - shift).div_ceil(64);
        let pages = ring.rows / 64;
        let table = (0..width).map(|p| ((shift / 64 + p) % pages) as i32).collect();
        Ok(Self { global: (base..end).map(|p| (p + origin) as i64).collect(),
            local: (base..end).map(|p| (p - shift) as i64).collect(), table, base, end })
    }
}

/// One stage's native owners and lane-private metadata. Never shares a stage
/// head, F32 hidden plane or SWA workspace with a different predictor.
pub(super) struct StageOwner<'a> {
    stream: Rc<Stream<'a>>,
    input: InputOwner<'a>,
    attention: super::swa::SwaLayer<'a>,
    head: super::head::TargetHeadOwner<'a>,
    metadata: Vec<Allocation<'a>>,
    stage: usize,
    rows: usize,
    immutable_bytes: u64,
    workspace_bytes: u64,
    cache_owner: std::cell::Cell<Option<usize>>,
}
impl<'a> StageOwner<'a> {
    pub(super) fn load(device: Device<'a>, programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint, stage: usize, rows: usize,
        rope: Rc<super::rope::TargetTables<'a>>, available: u64, reservations: &[u64]) -> Result<Self> {
        let input = InputPlan::prepare(programs, checkpoint, stage, rows)?;
        let head = super::head::HeadPlan::for_mtp_stage(programs, checkpoint, stage, rows)?;
        let immutable_bytes = RankWeightLayout::for_mtp_stage(checkpoint, stage)?.resident_bytes;
        let metadata = cuteafd_loader::plan::families::step5::native::mtp_metadata_planes(rows)?;
        let workspace_bytes = planes(rows)?.into_iter()
            .chain(cuteafd_loader::plan::families::step5::native::swa_planes(rows, 64)?)
            .chain(cuteafd_loader::plan::families::step5::native::dense_planes(rows)?)
            .chain(cuteafd_loader::plan::families::step5::native::head_planes(rows)?)
            .chain(metadata).try_fold(0u64, |n, bytes| n.checked_add(bytes as u64))
            .context("Step-5 predictor owner workspace overflow")?;
        let total = reservations.iter().copied().chain([immutable_bytes, workspace_bytes])
            .try_fold(0u64, u64::checked_add).context("Step-5 predictor owner admission overflow")?;
        ensure!(total <= available, "Step-5 predictor owner and reservations exceed admission");
        let stream = Rc::new(Stream::new(device)?);
        let input = InputOwner::load(device, programs, checkpoint, input, stream.clone(), available, &[])?;
        let attention = super::swa::SwaLayer::for_mtp_stage(device, programs, checkpoint,
            stage, rows, rope, available, &[])?;
        let head = super::head::TargetHeadOwner::load(device, programs, checkpoint,
            head, available, &[])?;
        let metadata = metadata.into_iter().map(|n| Allocation::new(device, n))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stream, input, attention, head, metadata, stage, rows,
            immutable_bytes, workspace_bytes, cache_owner: std::cell::Cell::new(None) })
    }

    pub(super) fn new_lane(&self, available: u64, reservations: &[u64]) -> Result<Self> {
        let total = reservations.iter().try_fold(self.workspace_bytes, |n, bytes| n.checked_add(*bytes))
            .context("Step-5 predictor lane admission overflow")?;
        ensure!(total <= available, "Step-5 predictor lane exceeds admission");
        let stream = Rc::new(Stream::new(self.stream.device)?);
        let input = self.input.new_lane(stream.clone())?;
        let attention = self.attention.new_lane()?;
        let head = self.head.new_lane()?;
        let metadata = self.metadata.iter().map(|p| Allocation::new(stream.device, p.buffer.bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stream, input, attention, head, metadata, stage: self.stage, rows: self.rows,
            immutable_bytes: self.immutable_bytes, workspace_bytes: self.workspace_bytes,
            cache_owner: std::cell::Cell::new(None) })
    }

    /// Register once before any prefix capture or reset can observe this lane.
    pub(super) fn bind_cache(&self, cache: &super::cache::CacheRank<'a>) -> Result<()> {
        ensure!(self.cache_owner.get().is_none(), "Step-5 predictor lane already bound");
        self.cache_components(cache, 0)?;
        cache.register_lane(&self.stream)?;
        self.cache_owner.set(Some(cache.identity()));
        Ok(())
    }

    fn cache_components(&self, cache: &super::cache::CacheRank<'a>, sequence: u64)
        -> Result<[CuteafdDeviceBuffer; 4]> {
        ensure!(cache.stream.device.id == self.stream.device.id
            && std::ptr::eq(cache.stream.device.library, self.stream.device.library),
            "Step-5 predictor cache device mismatch");
        let state = &cache.layout().ranks[cache.rank()];
        let sliding = state.sliding.iter().find(|s| s.mtp && s.layer == 92 + self.stage)
            .context("Step-5 predictor cache layer missing")?;
        ensure!(sliding.query_heads == 64 && sliding.key.bytes == cache.layout().ring.rows * 4 * 192 * 2
            && sliding.value.bytes == sliding.key.bytes,
            "Step-5 predictor cache KV geometry");
        let length = *state.mtp_lengths.get(self.stage).context("Step-5 predictor length missing")?;
        let history = *state.mtp_stage_hidden.get(self.stage).context("Step-5 predictor history missing")?;
        ensure!(length.bytes == 8 && history.bytes == cache.layout().mtp_hidden_rows * H as u64 * 4,
            "Step-5 predictor cache history geometry");
        [sliding.key, sliding.value, length, history].map(|span|
            cache.request_component(span, sequence)).into_iter().collect::<Result<Vec<_>>>()?
            .try_into().map_err(|_| anyhow::anyhow!("Step-5 predictor component inventory"))
    }

    /// Qualification path only: verify the restored frontier at this boundary.
    /// Serving must retain stage cursor evidence rather than download lengths.
    pub(super) fn forward<'e>(&'e mut self, cache: &'e super::cache::CacheRank<'a>,
        sequence: u64, base: u64, embedding: &'e Allocation<'a>,
        previous_hidden: &'e Allocation<'a>, rows: usize) -> Result<StageStep<'e, 'a>> {
        ensure!(rows <= self.rows && self.cache_owner.get() == Some(cache.identity()),
            "Step-5 predictor lane row capacity/cache binding");
        cache.validate_reset(sequence)?;
        let coordinates = Coordinates::new(self.stage, base, rows,
            cache.layout().context, cache.layout().ring)?;
        let components = self.cache_components(cache, sequence)?;
        cache.drain()?;
        let mut encoded = [0; 8];
        cache.stream.device.run(|| cache.stream.device.library.copy_d2h(&mut encoded, components[2]))?;
        ensure!(u64::from_le_bytes(encoded) == base, "Step-5 predictor frontier differs from committed cache");
        for (buffer, bytes) in [(embedding.buffer, rows * H * 2),
            (previous_hidden.buffer, rows * H * 4)] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= bytes && buffer.device_id == self.stream.device.id,
                "Step-5 predictor input allocation extent/device");
        }
        // Borrow all producers/consumers through the proposal's Drop drain.
        cache.busy[sequence as usize].set(true);
        let mut step = StageStep { owner: self, cache, sequence, coordinates,
            components, inputs: [embedding, previous_hidden], complete: false, enqueued: false };
        step.enqueue()?;
        Ok(step)
    }
}
impl Drop for StageOwner<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Step-5 predictor stage owner"); }
}

fn slice(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|n| n <= buffer.bytes),
        "Step-5 predictor slice extent");
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.wrapping_byte_add(offset), bytes, ..buffer })
}

/// Appends only to speculative slack. Publication is independent from target
/// acceptance; until coupled by the shared scheduler this is not a serving path.
pub(super) struct StageStep<'e, 'a> {
    owner: &'e mut StageOwner<'a>,
    cache: &'e super::cache::CacheRank<'a>,
    sequence: u64,
    coordinates: Coordinates,
    components: [CuteafdDeviceBuffer; 4],
    inputs: [&'e Allocation<'a>; 2],
    complete: bool,
    enqueued: bool,
}
impl<'e, 'a> StageStep<'e, 'a> {
    fn enqueue(&mut self) -> Result<()> {
        let owner = &*self.owner;
        let stream = &owner.stream;
        let rows = self.coordinates.global.len();
        let mut guard = stream.cancellation_guard("Step-5 predictor stage proposal");
        for (plane, bytes) in [(0, self.coordinates.global.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>()),
            (1, self.coordinates.local.iter().flat_map(|n| n.to_le_bytes()).collect()),
            (2, vec![0; rows * 4]),
            (3, self.coordinates.table.iter().flat_map(|n| n.to_le_bytes()).collect())] {
            stream.device.run(|| {
                // SAFETY: ordered upload retains source bytes; this proposal
                // borrows every metadata allocation through stream drainage.
                unsafe { stream.device.library.copy_h2d_ordered(owner.metadata[plane].buffer, &bytes, stream.raw) }
            })?;
        }
        let hidden = owner.input.enqueue(self.inputs[0].buffer, self.inputs[1].buffer, rows)?;
        stream.device.run(|| {
            // SAFETY: disjoint admitted EH and SWA input planes survive the proposal.
            unsafe { stream.device.library.copy_d2d_async(owner.attention.input(), hidden, rows * H * 4, stream.raw) }
        })?;
        owner.attention.project_on(rows, owner.metadata[0].buffer, stream)?;
        let projected = owner.attention.predictor_projection()?;
        for run in self.cache.layout().ring.proposal(self.coordinates.base, rows as u64)
            .context("Step-5 predictor proposal ring")? {
            for (target, source) in self.components[..2].iter().copied().zip(projected) {
                let source = slice(source, run.offset as usize * 1536, run.rows as usize * 1536)?;
                let target = slice(target, run.slot as usize * 1536, source.bytes)?;
                stream.device.run(|| {
                    // SAFETY: append-only proposal slots do not overlap committed
                    // visible KV rows; all source and destination owners are retained.
                    unsafe { stream.device.library.copy_d2d_async(target, source, source.bytes, stream.raw) }
                })?;
            }
        }
        owner.attention.predictor_finish_on(super::transaction::AttentionRead {
            inputs: super::transaction::StepInputs { stream, positions: owner.metadata[1].buffer,
                requests: owner.metadata[2].buffer, rows, base: self.coordinates.base as usize,
                extent: (self.coordinates.end + owner.stage as u64 + 1) as usize, heads: 64 },
            start: 0, key: self.components[0], value: self.components[1], summaries: None,
            table: owner.metadata[3].buffer, table_width: self.coordinates.table.len(),
            pages: self.cache.layout().ring.rows as usize / 64,
        })?;
        let history_ring = cuteafd_core::prefix::AppendRing {
            window: self.cache.layout().mtp_hidden_rows - self.cache.layout().ring.step,
            step: self.cache.layout().ring.step, rows: self.cache.layout().mtp_hidden_rows };
        for run in history_ring.proposal(self.coordinates.base, rows as u64)
            .context("Step-5 predictor own history ring")? {
            let source = slice(owner.attention.output(), run.offset as usize * H * 4, run.rows as usize * H * 4)?;
            let target = slice(self.components[3], run.slot as usize * H * 4, source.bytes)?;
            stream.device.run(|| {
                // SAFETY: append-safe F32 own-stage history and output are
                // disjoint and retained until this proposal's stream drains.
                unsafe { stream.device.library.copy_d2d_async(target, source, source.bytes, stream.raw) }
            })?;
        }
        self.enqueued = true;
        guard.complete = true;
        Ok(())
    }

    pub(super) fn hidden(&self) -> CuteafdDeviceBuffer { self.owner.attention.output() }

    pub(super) fn logits(&self) -> Result<crate::shared::token_io::DeviceLogits> {
        ensure!(self.enqueued && !self.complete, "Step-5 predictor proposal unavailable");
        self.owner.head.device_logits_on(self.hidden(), self.coordinates.global.len(), &self.owner.stream)
    }

    pub(super) fn publish(mut self, accepted: usize) -> Result<()> {
        ensure!(self.enqueued && accepted <= self.coordinates.global.len(),
            "Step-5 predictor accepted row extent");
        self.owner.stream.drain()?;
        if accepted > 0 {
            let length = self.coordinates.base + accepted as u64;
            self.cache.stream.device.run(|| self.cache.stream.device.library.copy_h2d(
                self.components[2], &length.to_le_bytes())).inspect_err(|_| self.cache.poisoned.set(true))?;
        }
        self.complete = true;
        Ok(())
    }
}
impl Drop for StageStep<'_, '_> {
    fn drop(&mut self) {
        self.owner.stream.drain_or_abort("Step-5 predictor proposal retirement");
        self.cache.busy[self.sequence as usize].set(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires sealed SM120 exports, real Step-5 weights and pinned MTP fixtures"]
    fn native_predictors_match_reference_replay_and_fresh_prefix() -> Result<()> {
        use sha2::{Digest, Sha256};
        use super::super::{cache::CacheRank, rope::TargetTables};
        use cuteafd_loader::plan::families::step5::coordinator::CoordinatorContract;
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_MTP_FIXTURE_DIR")?);
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(manifest["schema"] == 1 && manifest["seed"] == 274
            && manifest["row_capacity"] == 64
            && manifest["source"]["vllm_commit"] == "dde9aca41cd3ccceb1ffa902933b6678a09c83ce"
            && manifest["source"]["head_dtype"] == "float32", "MTP fixture provenance");
        // SAFETY: this explicit GPU qualification selects a sealed matching native ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let checkpoint = Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        ensure!(checkpoint.snapshot.file_name().and_then(|s| s.to_str()) == manifest["snapshot"].as_str(),
            "MTP fixture checkpoint identity");
        let cfg = Config::parse(&checkpoint)?;
        let layout = cuteafd_loader::serving_capacity::step5::Step5CacheLayout::new(&cfg, 1, 1024, 3, 64)?;
        let mut cache = CacheRank::new(device, layout.clone(), 0, 1, 2, 1, usize::MAX)?;
        let contract = CoordinatorContract::from_checkpoint(&checkpoint)?;
        let rope = Rc::new(TargetTables::load(device, &programs, &contract, "step5_", u64::MAX, &[])?);
        let cases = manifest["cases"].as_array().context("MTP cases")?;
        ensure!(cases.len() == 3, "MTP must qualify all three stages");
        let target_end = cases[0]["rows"].as_u64().context("MTP rows")? + 1;
        let target_hidden = std::fs::read(root.join("target_hidden.bin"))?;
        ensure!(manifest["target_rows"] == target_end && target_hidden.len() == target_end as usize * H * 2
            && manifest["target_hidden_sha256"].as_str()
                == Some(format!("{:x}", Sha256::digest(&target_hidden)).as_str()), "MTP target history fixture");
        let read = |buffer: CuteafdDeviceBuffer, bytes: usize| -> Result<Vec<u8>> {
            let mut data = vec![0; bytes];
            device.run(|| library.copy_d2h(&mut data, slice(buffer, 0, bytes)?))?;
            Ok(data)
        };
        let same_input = match std::env::var("CUTEAFD_STEP5_MTP_INPUT_MODE").as_deref() {
            Ok("reference") => true,
            Ok("chained") | Err(_) => false,
            _ => anyhow::bail!("MTP input mode must be reference or chained"),
        };
        let dump = std::env::var_os("CUTEAFD_STEP5_MTP_TRACE_DIR").map(std::path::PathBuf::from);
        let mut agreement = true;
        eprintln!("MTP mode={}", if same_input { "reference" } else { "chained" });
        for pass in 0..2 {
            cache.reset_sequence(0)?;
            device.run(|| library.copy_h2d(cache.length(0, false)?, &target_end.to_le_bytes()))?;
            let history = cache.request_component(layout.ranks[0].mtp_hidden.context("MTP target history")?, 0)?;
            for position in 0..target_end as usize {
                let row = slice(history, position % layout.mtp_hidden_rows as usize * H * 2, H * 2)?;
                device.run(|| library.copy_h2d(row, &target_hidden[position*H*2..(position+1)*H*2]))?;
            }
            let mut preceding_native: Option<Vec<u8>> = None;
            for (stage, case) in cases.iter().enumerate() {
                let rows = case["rows"].as_u64().context("MTP fixture rows")? as usize;
                ensure!(case["stage"] == stage && case["origin"] == stage + 1 && rows >= 600,
                    "MTP stage fixture geometry");
                let load = |name: &str, bytes: usize| -> Result<Vec<u8>> {
                    let data = std::fs::read(root.join(format!("stage{stage}/{name}")))?;
                    ensure!(data.len() == bytes && case["sha256"][name].as_str()
                        == Some(format!("{:x}", Sha256::digest(&data)).as_str()), "MTP fixture extent/hash: {name}");
                    Ok(data)
                };
                let inputs = load("embedding.bin", rows * H * 2)?;
                let reference_previous = load("previous.bin", rows * H * 4)?;
                let previous = if let Some(previous) = preceding_native.as_ref().filter(|_| !same_input) {
                    ensure!(previous.len() == (rows + 1) * H * 4, "MTP preceding native history extent");
                    &previous[..rows * H * 4]
                } else { &reference_previous };
                let expected = load("hidden.bin", rows * H * 4)?;
                let probes = case["probes"].as_array().context("MTP probes")?.iter()
                    .map(|p| p.as_u64().context("MTP probe").map(|p| p as usize)).collect::<Result<Vec<_>>>()?;
                let expected_logits = load("logits.bin", probes.len() * 128896 * 4)?;
                let embedding = Allocation::new(device, 64 * H * 2)?;
                let hidden = Allocation::new(device, 64 * H * 4)?;
                let mut owner = StageOwner::load(device, &programs, &checkpoint, stage, 64,
                    rope.clone(), u64::MAX, &[])?;
                owner.bind_cache(&cache)?;
                let mut boundaries: Vec<(&str, usize, Vec<u8>)> = [
                    ("enorm", H), ("hnorm", H), ("concat", 2 * H), ("eh", H),
                    ("input_norm", H), ("qraw", 64*192), ("kraw", 4*192),
                    ("value", 4*192), ("query", 64*192), ("key", 4*192),
                    ("attention", 64*192), ("gated", 64*192), ("attention_delta", H)]
                    .into_iter().map(|(name, width)| (name, width, Vec::new())).collect();
                let mut base = 0;
                let mut actual = Vec::with_capacity(expected.len());
                let mut logits = vec![Vec::new(); probes.len()];
                while base < rows {
                    let boundary = [64, 512, 576, 600, rows].into_iter()
                        .filter(|&end| end > base).min().context("MTP next boundary")?;
                    let count = (boundary - base).min(64);
                    let accepted = if base == 0 { 32 } else { count };
                    if base == 576 && pass == 0 {
                        cache.capture(0, 1, target_end)?;
                        cache.reset_sequence(0)?;
                        cache.restore(0, 1, target_end)?;
                    }
                    device.run(|| library.copy_h2d(embedding.buffer, &inputs[base*H*2..(base+count)*H*2]))?;
                    device.run(|| library.copy_h2d(hidden.buffer, &previous[base*H*4..(base+count)*H*4]))?;
                    let first = owner.forward(&cache, 0, base as u64, &embedding, &hidden, count)?;
                    first.owner.stream.drain()?;
                    let replay = read(first.hidden(), count * H * 4)?;
                    drop(first); // Aborted wave must not advance any committed frontier.
                    let step = owner.forward(&cache, 0, base as u64, &embedding, &hidden, count)?;
                    step.owner.stream.drain()?;
                    let values = read(step.hidden(), count * H * 4)?;
                    ensure!(values == replay, "MTP same-shape abort/replay differs, stage {stage} base {base}");
                    actual.extend_from_slice(&values[..accepted * H * 4]);
                    if pass == 0 && dump.is_some() {
                        let input = &step.owner.input.planes;
                        let planes = [("enorm", input[1].buffer, H), ("hnorm", input[2].buffer, H),
                            ("concat", input[3].buffer, 2*H), ("eh", input[4].buffer, H)]
                            .into_iter().chain(step.owner.attention.predictor_trace());
                        for (name, buffer, width) in planes {
                            let bytes = read(buffer, accepted * width * 2)?;
                            boundaries.iter_mut().find(|(key, _, _)| *key == name).unwrap().2.extend(bytes);
                        }
                    }
                    if probes.iter().any(|&p| (base..base+accepted).contains(&p)) {
                        let result = step.logits()?;
                        for (i, &p) in probes.iter().enumerate().filter(|(_, p)| (base..base+accepted).contains(p)) {
                            logits[i] = result.row_host(&library, p-base)?.into_iter().flat_map(f32::to_le_bytes).collect();
                        }
                    }
                    step.publish(accepted)?;
                    base += accepted;
                }
                if pass == 0 {
                    if let Some(directory) = &dump {
                        let directory = directory.join(format!("stage{stage}"));
                        std::fs::create_dir_all(&directory)?;
                        std::fs::write(directory.join("hidden.bin"), &actual)?;
                        for (name, width, bytes) in boundaries {
                            let reference = load(&format!("{name}.bf16"), rows * width * 2)?;
                            std::fs::write(directory.join(format!("{name}.bf16")), &bytes)?;
                            let decode = |bytes: &[u8]| bytes.chunks_exact(2).map(|v|
                                f32::from_bits((u16::from_le_bytes(v.try_into().unwrap()) as u32) << 16) as f64)
                                .collect::<Vec<_>>();
                            let (a,b) = (decode(&bytes), decode(&reference));
                            let e = a.iter().zip(&b).map(|(a,b)| (a-b).powi(2)).sum::<f64>();
                            let n = b.iter().map(|v| v*v).sum::<f64>().max(f64::MIN_POSITIVE);
                            let different = bytes.chunks_exact(2).zip(reference.chunks_exact(2))
                                .filter(|(a,b)| a != b).count();
                            eprintln!("MTP boundary stage={stage} name={name} rel_l2={:.9e} different={different}/{}",
                                (e/n).sqrt(), rows*width);
                        }
                    }
                }
                preceding_native = Some(actual.clone());
                for (name, a, b) in [("hidden", actual, expected),
                    ("logits", logits.into_iter().flatten().collect(), expected_logits)] {
                    let a: Vec<_> = a.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64).collect();
                    let b: Vec<_> = b.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64).collect();
                    ensure!(a.len() == b.len() && a.iter().chain(&b).all(|v| v.is_finite()), "MTP nonfinite/shape");
                    let error = a.iter().zip(&b).map(|(a,b)| (a-b).powi(2)).sum::<f64>();
                    let norm = b.iter().map(|b| b*b).sum::<f64>().max(f64::MIN_POSITIVE);
                    let relative = (error/norm).sqrt();
                    let maximum = a.iter().zip(&b).map(|(a,b)| (a-b).abs()).fold(0f64, f64::max);
                    let width = if name == "hidden" { H } else { 128896 };
                    let worst = a.chunks_exact(width).zip(b.chunks_exact(width)).map(|(a,b)| {
                        let e = a.iter().zip(b).map(|(a,b)| (a-b).powi(2)).sum::<f64>();
                        let n = b.iter().map(|v| v*v).sum::<f64>().max(f64::MIN_POSITIVE);
                        (e/n).sqrt()
                    }).fold(0f64, f64::max);
                    eprintln!("MTP pass={pass} stage={stage} {name} rel_l2={relative:.9e} max_abs={maximum:.9e} worst_row_rel_l2={worst:.9e}");
                    if same_input {
                        agreement &= relative <= 0.01 && worst <= 0.01;
                    }
                }
                if pass == 0 {
                    // Deliberately dirty rejected slack after the committed end.
                    let proposal = owner.forward(&cache, 0, rows as u64, &embedding, &hidden, 1)?;
                    drop(proposal);
                }
            }
            cache.capture(0, pass, target_end)?;
            if pass == 0 {
                cache.restore(0, 0, target_end)?;
                cache.capture(0, 1, target_end)?;
                ensure!(read(cache.mark_view(0)?, layout.ranks[0].bytes as usize)?
                    == read(cache.mark_view(1)?, layout.ranks[0].bytes as usize)?, "MTP restore canonical bytes");
            }
        }
        ensure!(read(cache.mark_view(0)?, layout.ranks[0].bytes as usize)?
            == read(cache.mark_view(1)?, layout.ranks[0].bytes as usize)?,
            "MTP restored prefix differs byte-for-byte from fresh prefill");
        eprintln!("MTP exactness passed: abort/replay, canonical restore, restored/fresh prefix");
        // The target verifier gates output; chained draft error is informational.
        ensure!(agreement, "MTP same-input reference agreement exceeds unchanged 1% bar");
        Ok(())
    }

    fn specs() -> (ProgramSpec, ProgramSpec) {
        let spec = |name: &str, pointers: &[&str], scalars: &[u8], geometry| ProgramSpec {
            name: name.into(), pointers: pointers.iter().map(|s| (*s).into()).collect(),
            scalar_kinds: scalars.to_vec(), scratch: Default::default(),
            capacity_rows: 4096, geometry,
        };
        (spec("step5_final_norm", NORM_POINTERS, b"if", serde_json::json!({
            "width": H, "heads": 1, "zero_centered": true, "subtract_mean": false,
            "bias": false, "rotary": 0, "epsilon": "runtime_float32"})),
        spec("step5_mtp_eh", EH_POINTERS, b"i", serde_json::json!({
            "width": 2 * H, "outputs": H, "output_dtype": "torch.bfloat16", "skinny_rows": 0})))
    }

    #[test]
    fn predictor_coordinates_preserve_global_rope_and_exact_causal_keys() -> Result<()> {
        for rows in [1, 16, 64, 256, 513, 4096] {
            let ring = cuteafd_core::prefix::AppendRing::new(512, rows as u64, 64).unwrap();
            for stage in 0..3 {
                for base in [0, 1, 63, 64, 511, 512, 513, 575, 576, 767,
                    ring.rows - 1, ring.rows, ring.rows + 1, 1_000_000] {
                    let c = Coordinates::new(stage, base, rows, 1_048_576, ring)?;
                    assert_eq!(c.global[0], (base + stage as u64 + 1) as i64);
                    assert_eq!(c.end, base + rows as u64);
                    assert!(c.table.len() * 4 <=
                        cuteafd_loader::plan::families::step5::native::mtp_metadata_planes(rows)?[3]);
                    let shift = base - c.local[0] as u64;
                    for row in [0, rows / 2, rows - 1] {
                        let query = base + row as u64;
                        let mut actual = Vec::new();
                        for selected in c.local[row] - 511..=c.local[row] {
                            if selected < 0 { continue; }
                            let selected = selected as u64;
                            let logical = selected + shift;
                            let physical = c.table[(selected / 64) as usize] as u64 * 64 + selected % 64;
                            assert_eq!(physical, logical % ring.rows);
                            actual.push(logical);
                        }
                        assert_eq!(actual, (query.saturating_sub(511)..=query).collect::<Vec<_>>());
                    }
                    let mut slots = vec![false; ring.rows as usize];
                    for run in ring.visible(base) {
                        slots[run.slot as usize..(run.slot + run.rows) as usize].fill(true);
                    }
                    for run in ring.proposal(base, rows as u64).unwrap() {
                        assert!(slots[run.slot as usize..(run.slot + run.rows) as usize]
                            .iter().all(|old| !old));
                    }
                }
            }
        }
        let ring = cuteafd_core::prefix::AppendRing::new(512, 256, 64).unwrap();
        for (stage, base, rows, context) in [(3, 0, 1, 1024), (0, 0, 0, 1024),
            (0, 0, 257, 1024), (0, 1023, 1, 1024), (0, u64::MAX, 1, u64::MAX)] {
            assert!(Coordinates::new(stage, base, rows, context, ring).is_err());
        }
        Ok(())
    }

    #[test]
    fn predictor_metadata_is_charged_even_at_tiny_row_counts() -> Result<()> {
        for rows in [1, 16, 64, 256, 4096] {
            let metadata = cuteafd_loader::plan::families::step5::native::mtp_metadata_planes(rows)?;
            assert!(metadata.iter().all(|n| *n >= 256));
            let private = planes(rows)?.into_iter().sum::<usize>() as u64;
            let blocks = cuteafd_loader::plan::families::step5::native::swa_planes(rows, 64)?
                .into_iter().chain(cuteafd_loader::plan::families::step5::native::dense_planes(rows)?)
                .chain(cuteafd_loader::plan::families::step5::native::head_planes(rows)?)
                .sum::<usize>() as u64;
            assert!(private + metadata.into_iter().sum::<usize>() as u64 + blocks <=
                cuteafd_loader::plan::families::step5::native::mtp_forecast_workspace(rows, 1)?);
        }
        assert_eq!(cuteafd_loader::plan::families::step5::native::mtp_metadata_planes(256)?
            .into_iter().sum::<usize>(), 5632);
        Ok(())
    }

    #[test]
    fn native_input_workspace_fits_the_conservative_forecast() -> Result<()> {
        for rows in [1, 64, 256, 4096] {
            let private = planes(rows)?.into_iter().sum::<usize>() as u64;
            assert_eq!(private, (rows * H * 16) as u64);
            let blocks = cuteafd_loader::plan::families::step5::native::swa_planes(rows, 64)?
                .into_iter().chain(cuteafd_loader::plan::families::step5::native::dense_planes(rows)?)
                .chain(cuteafd_loader::plan::families::step5::native::head_planes(rows)?)
                .sum::<usize>() as u64;
            assert!(private + blocks <=
                cuteafd_loader::plan::families::step5::native::mtp_forecast_workspace(rows, 1)?);
        }
        for rows in [0, 4097, usize::MAX] { assert!(planes(rows).is_err()); }
        Ok(())
    }

    #[test]
    fn native_input_programs_require_bf16_eh_and_whole_concatenation() -> Result<()> {
        let (norm, eh) = specs();
        validate_programs(&norm, &eh, 256)?;
        let mut bad = eh.clone();
        bad.geometry["width"] = H.into();
        assert!(validate_programs(&norm, &bad, 256).is_err());
        bad = eh.clone();
        bad.geometry["output_dtype"] = "torch.float32".into();
        assert!(validate_programs(&norm, &bad, 256).is_err());
        bad = eh.clone();
        bad.capacity_rows = 255;
        assert!(validate_programs(&norm, &bad, 256).is_err());
        bad = eh.clone();
        bad.scratch.insert("undeclared".into(), 256);
        assert!(validate_programs(&norm, &bad, 256).is_err());
        let mut bad_norm = norm.clone();
        bad_norm.geometry["zero_centered"] = false.into();
        assert!(validate_programs(&bad_norm, &eh, 256).is_err());
        bad_norm = norm.clone();
        bad_norm.scalar_kinds.pop();
        assert!(validate_programs(&bad_norm, &eh, 256).is_err());
        Ok(())
    }
}
