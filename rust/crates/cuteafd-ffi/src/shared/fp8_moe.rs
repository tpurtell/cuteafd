//! Exact FP8 routed-expert package library (`native/shared/include/cuteafd_fp8_moe.h`):
//! one dlopen'd `libcuteafd_fp8moe.so` per layout, one program per capacity.
//! Drain every stream that used the module before dropping it, on the thread
//! (and with the device) that loaded it.
use anyhow::{ensure, Context, Result};
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

pub const FP8_MOE_POINTERS: usize = 11;
pub const FP8_MOE_LIBRARY: &str = "libcuteafd_fp8moe.so";

/// The expert weight format a package reads (ABI word 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8MoeWeights {
    /// Storage 1: E4M3 with FP32 128x128 block scales.
    Fp8,
    /// Storage 2: packed E2M1 with UE8M0 scales per 32.
    Mxfp4,
    /// Storage 4: ModelOpt NVFP4: packed E2M1, E4M3 scales per 16, each
    /// scale region followed by the experts' FP32 `weight_scale_2` and
    /// `input_scale`. ABI v5 accepts calibrated W4A4 only, at every row count.
    Nvfp4,
    /// Native BF16 row-major matrices; scale pointers are unused.
    Bf16,
}

/// Clamp order and intermediate rounding, independent of projection precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Fp8MoeActivation {
    /// Clamp gate before SiLU, round SiLU to BF16, multiply, then round to BF16.
    SiluBf16Product = 1,
    /// Clamp FP32 SiLU, multiply in FP32, then round once to BF16.
    SiluPostClampFp32Product = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Fp8MoeProjection {
    /// Gate/up round before activation; down rounds before the FP32 route sum.
    Bf16 = 1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fp8MoeInfo {
    pub hidden: usize,
    pub slice: usize,
    pub experts: usize,
    pub topk: usize,
    pub intermediate: usize,
    pub tp: usize,
    /// Input rows are FP8 K32 wire rows (Spark) rather than BF16 (coordinator).
    pub wire_input: bool,
    pub swiglu_limit: f32,
    pub activation: Fp8MoeActivation,
    pub projection: Fp8MoeProjection,
    pub capacities: Vec<usize>,
    /// The weight format; MXFP4 also supports exact K32 slices.
    pub weights: Fp8MoeWeights,
}

impl Fp8MoeInfo {
    fn from_words(words: [u32; 16]) -> Result<Self> {
        let count = words[9] as usize;
        let [version, format, activation, projection] = words[0].to_le_bytes();
        ensure!(version == 5 && matches!(format, 1 | 2 | 4 | 5)
            && matches!(activation, 1 | 2) && !(format == 2 && activation == 2) && projection == 1
            && matches!(words[7], 1 | 7) && (1..=6).contains(&count),
            "unsupported FP8 expert package ABI {words:?}");
        let info = Self {
            hidden: words[1] as usize,
            slice: words[2] as usize,
            experts: words[3] as usize,
            topk: words[4] as usize,
            intermediate: words[5] as usize,
            tp: words[6] as usize,
            wire_input: words[7] == 7,
            swiglu_limit: f32::from_bits(words[8]),
            activation: match activation {
                1 => Fp8MoeActivation::SiluBf16Product,
                _ => Fp8MoeActivation::SiluPostClampFp32Product,
            },
            projection: Fp8MoeProjection::Bf16,
            capacities: words[10..10 + count].iter().map(|&c| c as usize).collect(),
            weights: match format {
                1 => Fp8MoeWeights::Fp8,
                2 => Fp8MoeWeights::Mxfp4,
                4 => Fp8MoeWeights::Nvfp4,
                _ => Fp8MoeWeights::Bf16,
            },
        };
        // Legacy packages store H128 exact or padded slices. Native MXFP4 may
        // additionally store exact K32 units; recipe validation narrows support.
        let exact = |unit: usize| {
            let blocks = info.intermediate / unit;
            info.intermediate % unit == 0 && info.tp > 0 && blocks >= info.tp
                && (info.slice == blocks / info.tp * unit || info.slice == blocks.div_ceil(info.tp) * unit)
        };
        let sliced = (info.slice % 128 == 0 && info.slice >= 128
            && (exact(128) || (info.slice * info.tp >= info.intermediate
                && (info.slice - 128) * info.tp < info.intermediate)))
            || (info.weights == Fp8MoeWeights::Mxfp4 && exact(32));
        ensure!(info.tp > 0 && sliced && info.hidden > 0 && info.hidden % 128 == 0
            && info.experts > 0 && (1..=16).contains(&info.topk)
            && info.swiglu_limit.is_finite() && info.swiglu_limit >= 0.0
            && info.capacities[0] > 0 && info.capacities.iter().all(|&c| c <= i32::MAX as usize)
            && info.capacities.windows(2).all(|w| w[0] < w[1])
            && (!matches!(info.weights, Fp8MoeWeights::Bf16 | Fp8MoeWeights::Nvfp4) || !info.wire_input),
            "inconsistent FP8 expert package info {info:?}");
        Ok(info)
    }

    /// Check checkpoint semantics on static metadata before allocating device storage.
    pub fn validate_activation(&self, activation: Fp8MoeActivation, limit: f32,
        projection: Fp8MoeProjection) -> Result<()> {
        ensure!(limit.is_finite() && limit >= 0.0 && self.activation == activation
            && self.swiglu_limit == limit && self.projection == projection,
            "expert package activation/projection contract disagrees with checkpoint: package {:?}/{:?} limit {}, requested {:?}/{:?} limit {}",
            self.activation, self.projection, self.swiglu_limit, activation, projection, limit);
        Ok(())
    }

    /// Wire row bytes: hidden E4M3 values then hidden/32 UE8M0 scales.
    pub fn wire_row_bytes(&self) -> usize {
        self.hidden + self.hidden / 32
    }

    /// Smallest compiled capacity that holds `rows`.
    pub fn capacity_for(&self, rows: usize) -> Option<usize> {
        self.capacities.iter().copied().find(|&c| c >= rows)
    }
}

type Launch = unsafe extern "C" fn(*mut c_void, u32, *const *mut c_void, i32, *mut c_void) -> i32;
type LaunchSegments = unsafe extern "C" fn(*mut c_void, *const *mut c_void, i32, *mut c_void) -> i32;
type ReplyLayout = unsafe extern "C" fn(u32, u32, u32, u32, *mut u64, u32) -> i32;
type SegmentInfo = unsafe extern "C" fn(u32, *mut u32, *mut u32, *mut u32) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Scratch = unsafe extern "C" fn(u32, *mut u64) -> i32;
type SetOptions = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type WeightReadGroup = unsafe extern "C" fn(u32, u32, u32, *mut u32, *mut u32) -> i32;
type ArithmeticRoute = unsafe extern "C" fn(u32, u32, u32, *mut u32, *mut u32) -> i32;

/// A traffic bound must never masquerade as an installed kernel tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8MoeTrafficContract {
    ExactGrouping,
    ConservativeRouteUpperBound,
}

impl Fp8MoeTrafficContract {
    fn from_words(group: u32, contract: u32) -> Result<(usize, Self)> {
        ensure!((1..=4096).contains(&group), "invalid FP8 expert weight-read group {group}");
        let contract = match contract {
            1 => Self::ExactGrouping,
            2 if group == 1 => Self::ConservativeRouteUpperBound,
            _ => anyhow::bail!("invalid FP8 expert traffic contract {contract} with group {group}"),
        };
        Ok((group as usize, contract))
    }
}

/// How FP8 expert packages run their large (prefill) row counts
/// (`cuteafd_fp8moe_set_options`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fp8MoePrefill {
    /// The package default: FP8 K32 wire rows W8A8 (block-scaled E4M3 x E4M3
    /// gate/up), BF16 rows W8A16 (exact rows).
    #[default]
    Auto,
    /// The former W8A16 programs (weights widened to `bf16(w * s)`).
    W8a16,
    /// W8A8 for BF16 rows too: quantized to wire rows inside the program.
    W8a8,
}

impl Fp8MoePrefill {
    fn option(self) -> u32 {
        match self {
            Self::Auto => 0,
            Self::W8a16 => 1,
            Self::W8a8 => 2,
        }
    }
}

impl std::str::FromStr for Fp8MoePrefill {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "w8a8" => Ok(Self::W8a8),
            "w8a16" => Ok(Self::W8a16),
            other => anyhow::bail!("FP8 expert prefill is auto, w8a8 or w8a16, not {other:?}"),
        }
    }
}

pub struct Fp8MoeModule {
    // Function addresses and the context are valid only while the library is loaded.
    _library: Library,
    context: NonNull<c_void>,
    info: Fp8MoeInfo,
    scratch: Vec<usize>,
    launch: Launch,
    launch_segments: LaunchSegments,
    launch_core: Launch,
    launch_segments_core: LaunchSegments,
    reply_layout: ReplyLayout,
    segments: Vec<Fp8MoeSegment>,
    destroy: Destroy,
    set_options: Option<SetOptions>,
    weight_read_group: Option<WeightReadGroup>,
    arithmetic_route: Option<ArithmeticRoute>,
    prefill: Fp8MoePrefill,
    _owner_thread: PhantomData<Rc<()>>,
}

/// Allocation metadata from a trusted package's static ABI. Reading this
/// object never calls `cuteafd_fp8moe_create` or initializes CUDA kernels.
#[derive(Debug, Clone)]
pub struct Fp8MoeMetadata {
    pub info: Fp8MoeInfo,
    scratch: Vec<usize>,
    pub segments: Vec<Fp8MoeSegment>,
}

/// One installed wide-stream arithmetic recipe. Only constituent units at
/// least `minimum_rows` may join; grouping cannot promote a decode tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8MoeSegment {
    pub form: Fp8MoePrefill,
    pub maximum_rows: u32,
    pub minimum_rows: u32,
    pub act_row_bytes: u32,
}

fn segment_scratch(info: &Fp8MoeInfo, segments: &[Fp8MoeSegment], rows: u32) -> Result<usize> {
    segments.iter().map(|segment| {
        cuteafd_core::spark_executions::segment_scratch_bytes(info.hidden.try_into()?, info.experts.try_into()?,
            info.topk.try_into()?, segment.act_row_bytes, rows, segment.maximum_rows)
            .context("invalid segmented scratch extent").and_then(|bytes| Ok(usize::try_from(bytes)?))
    }).collect::<Result<Vec<_>>>()?.into_iter().max().context("package has no segmented stream")
}

/// Exact scratch retained by one execution slot, from the package's static ABI.
/// This query neither initializes CUDA nor allocates device storage.
pub fn fp8_slot_scratch_bytes(package: impl IntoIterator<Item = (usize, usize)>, capacity: usize)
    -> Result<usize> {
    ensure!(capacity > 0, "native slot capacity must be nonzero");
    let mut previous = 0;
    let mut selected = None;
    for (compiled, scratch) in package {
        ensure!(compiled > previous, "native package capacities must be sorted and unique");
        if selected.is_none() && compiled >= capacity { selected = Some(scratch.max(256)); }
        previous = compiled;
    }
    selected.with_context(|| format!("FP8 expert package has no capacity for {capacity} rows"))
}

/// Exact native slot input/routes/output allocation, independent of architecture.
pub fn fp8_slot_io_bytes(hidden: usize, topk: usize, wire_input: bool, bf16_sibling: bool,
    capacity: usize) -> Result<usize> {
    ensure!(hidden > 0 && hidden % 32 == 0 && topk > 0 && capacity > 0,
        "invalid native slot I/O geometry");
    let input = if !wire_input || bf16_sibling { hidden.checked_mul(2) }
        else { hidden.checked_add(hidden / 32) }.context("native slot input overflow")?;
    input.checked_add(topk.checked_mul(8).context("native slot routes overflow")?)
        .and_then(|bytes| hidden.checked_mul(2).and_then(|out| bytes.checked_add(out)))
        .and_then(|bytes| bytes.checked_mul(capacity)).context("native slot I/O size overflow")
}

impl Fp8MoeMetadata {
    /// Read the same geometry and workspace contract used by `load`, before
    /// any CUDA state, weights or workspaces are allocated.
    ///
    /// # Safety
    /// Only load trusted generated native code whose info/scratch entry
    /// points implement the static package ABI.
    pub unsafe fn read(directory: &Path) -> Result<Self> {
        let path = directory.join(FP8_MOE_LIBRARY);
        // SAFETY: the caller supplies trusted native code; no function or
        // pointer from this library escapes the metadata read.
        let library = unsafe { Library::new(&path) }
            .with_context(|| format!("loading metadata from {}", path.display()))?;
        // SAFETY: the caller's trusted static ABI remains live in `library`.
        unsafe { Self::from_library(&library) }
    }

    unsafe fn from_library(library: &Library) -> Result<Self> {
        // SAFETY: the trusted package exports the documented static ABI;
        // both functions are called while this library remains live.
        let (query, scratch_bytes) = unsafe {
            (*library.get::<unsafe extern "C" fn(*mut u32, u32) -> i32>(b"cuteafd_fp8moe_info")?,
                *library.get::<Scratch>(b"cuteafd_fp8moe_scratch_bytes")?)
        };
        let mut words = [0u32; 16];
        // SAFETY: the query fills at most sixteen live words.
        ensure!(unsafe { query(words.as_mut_ptr(), 16) } == 0,
            "FP8 expert package info query failed");
        let mut metadata = Self::from_words_and_scratch(words, |capacity| {
            let mut bytes = 0u64;
            // SAFETY: capacity came from the validated package info and
            // the scratch query writes exactly one live u64.
            ensure!(unsafe { scratch_bytes(capacity as u32, &mut bytes) } == 0,
                "FP8 expert scratch query failed");
            Ok(bytes)
        })?;
        // SAFETY: static queries from the trusted library write only supplied scalars.
        let (segment_info, segment_bytes) = unsafe {
            (*library.get::<SegmentInfo>(b"cuteafd_fp8moe_segment_info")?,
             *library.get::<Scratch>(b"cuteafd_fp8moe_segment_scratch_bytes")?)
        };
        for form in [Fp8MoePrefill::Auto, Fp8MoePrefill::W8a16, Fp8MoePrefill::W8a8] {
            let (mut maximum, mut minimum, mut act_row_bytes) = (0, 0, 0);
            // SAFETY: the query writes three live u32 values and performs no GPU work.
            ensure!(unsafe { segment_info(form.option(), &mut maximum, &mut minimum, &mut act_row_bytes) } == 0,
                "FP8 segment info query failed");
            if maximum == 0 { continue; }
            ensure!(metadata.info.wire_input && matches!(metadata.info.weights, Fp8MoeWeights::Fp8 | Fp8MoeWeights::Mxfp4)
                && maximum == cuteafd_core::spark_executions::SEGMENT_MAX_ROWS
                && minimum > 0 && minimum <= 4096 && act_row_bytes > 0,
                "invalid segmented stream metadata");
            metadata.segments.push(Fp8MoeSegment { form, maximum_rows: maximum, minimum_rows: minimum, act_row_bytes });
        }
        if !metadata.segments.is_empty() {
            ensure!(metadata.segments.len() == 3, "segmented package must describe every arithmetic form");
            for rows in [1u32, 4096, 8256, cuteafd_core::spark_executions::SEGMENT_MAX_ROWS] {
                let mut bytes = 0u64;
                // SAFETY: the static query writes one live u64; no allocation or CUDA work.
                ensure!(unsafe { segment_bytes(rows, &mut bytes) } == 0,
                    "FP8 segmented scratch query failed");
                ensure!(u64::try_from(metadata.segment_scratch_bytes(rows)?)? == bytes,
                    "FP8 installed segmented scratch disagrees with shared geometry");
            }
        }
        Ok(metadata)
    }

    fn from_words_and_scratch(words: [u32; 16], mut query: impl FnMut(usize) -> Result<u64>) -> Result<Self> {
        // Validate exactly the runtime's package contract before querying any
        // workspace extent; this helper does not create CUDA state.
        let info = Fp8MoeInfo::from_words(words)?;
        let scratch = info.capacities.iter().map(|&capacity| {
            usize::try_from(query(capacity)?).context("FP8 expert scratch does not fit this process")
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { info, scratch, segments: Vec::new() })
    }

    pub fn segment_scratch_bytes(&self, rows: u32) -> Result<usize> {
        segment_scratch(&self.info, &self.segments, rows)
    }

    /// Static `(compiled capacity, scratch bytes)` records for planner/runtime admission.
    pub fn scratch_layout(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.info.capacities.iter().copied().zip(self.scratch.iter().copied())
    }

    /// Scratch for the smallest compiled program holding `rows`.
    pub fn scratch_for(&self, rows: usize) -> Result<usize> {
        let capacity = self.info.capacity_for(rows)
            .with_context(|| format!("FP8 expert package has no capacity for {rows} rows"))?;
        let index = self.info.capacities.iter().position(|&c| c == capacity)
            .context("FP8 expert capacity metadata disagrees")?;
        self.scratch.get(index).copied().context("FP8 expert scratch metadata disagrees")
    }
}

impl Fp8MoeModule {
    /// Loads `directory/libcuteafd_fp8moe.so` and every program on the current device.
    ///
    /// # Safety
    /// Load only trusted generated code with a current CUDA device; keep that
    /// device current for launches and drop, and drain every stream using
    /// the module before dropping it.
    pub unsafe fn load(directory: &Path) -> Result<Self> {
        let path = directory.join(FP8_MOE_LIBRARY);
        let library = Library::new(&path).with_context(|| format!("loading {}", path.display()))?;
        let metadata = Fp8MoeMetadata::from_library(&library)?;
        let create = *library.get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"cuteafd_fp8moe_create")?;
        let launch = *library.get::<Launch>(b"cuteafd_fp8moe_launch")?;
        let launch_segments = *library.get::<LaunchSegments>(b"cuteafd_fp8moe_launch_segments")?;
        let launch_core = *library.get::<Launch>(b"cuteafd_fp8moe_launch_core")?;
        let launch_segments_core = *library.get::<LaunchSegments>(b"cuteafd_fp8moe_launch_segments_core")?;
        let reply_layout = *library.get::<ReplyLayout>(b"cuteafd_fp8moe_reply_layout")?;
        let destroy = *library.get::<Destroy>(b"cuteafd_fp8moe_destroy")?;
        // Packages built before the W8A16 fallback forms have no options.
        let set_options = library.get::<SetOptions>(b"cuteafd_fp8moe_set_options").ok().map(|f| *f);
        let weight_read_group = library.get::<WeightReadGroup>(b"cuteafd_fp8moe_weight_read_group").ok().map(|f| *f);
        let arithmetic_route = library.get::<ArithmeticRoute>(b"cuteafd_fp8moe_arithmetic_route").ok().map(|f| *f);
        let Fp8MoeMetadata { info, scratch, segments } = metadata;
        let mut context = std::ptr::null_mut();
        let status = create(&mut context);
        ensure!(status == 0, "FP8 expert package initialization failed with CUDA status {status}");
        let context = NonNull::new(context).context("FP8 expert package returned a null context")?;
        Ok(Self { _library: library, context, info, scratch, launch, launch_segments, launch_core, launch_segments_core, reply_layout, segments, destroy, set_options,
            weight_read_group, arithmetic_route, prefill: Fp8MoePrefill::Auto, _owner_thread: PhantomData })
    }

    /// Selects the programs large row counts run. A package without the
    /// option runs its single (default) form.
    pub fn set_prefill(&mut self, prefill: Fp8MoePrefill) -> Result<()> {
        let options = prefill.option();
        let Some(set) = self.set_options else {
            ensure!(options == 0, "this FP8 expert package predates the prefill forms; rebuild it");
            return Ok(());
        };
        // SAFETY: the context came from this library's create and is still live.
        let status = unsafe { set(self.context.as_ptr(), options) };
        ensure!(status == 0, "FP8 expert package rejected options {options:#x} ({status})");
        self.prefill = prefill;
        Ok(())
    }

    pub fn info(&self) -> &Fp8MoeInfo {
        &self.info
    }

    /// Static `(compiled capacity, scratch bytes)` records, identical to CPU metadata.
    pub fn scratch_layout(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.info.capacities.iter().copied().zip(self.scratch.iter().copied())
    }

    /// The loaded capacity and selected form's static traffic contract.
    pub fn weight_read_group(&self, rows: usize) -> Result<(usize, Fp8MoeTrafficContract)> {
        let capacity = if rows == 0 { self.info.capacities.last().copied() }
            else { self.info.capacity_for(rows) }
            .context("FP8 expert traffic query exceeds installed capacities")?;
        self.weight_read_group_for_capacity(capacity, rows)
    }

    pub fn weight_read_group_for_capacity(&self, capacity: usize, rows: usize)
        -> Result<(usize, Fp8MoeTrafficContract)> {
        ensure!(self.info.capacities.contains(&capacity) && rows <= capacity,
            "FP8 expert traffic query names an uninstalled capacity");
        let query = self.weight_read_group.context(
            "FP8 expert package lacks installed traffic metadata; rebuild the package")?;
        let (mut group, mut contract) = (0u32, 0u32);
        // SAFETY: this static function belongs to the live loaded library and
        // writes exactly two live words; it neither touches CUDA nor the context.
        let status = unsafe { query(u32::try_from(capacity)?, self.prefill.option(),
            u32::try_from(rows)?, &mut group, &mut contract) };
        ensure!(status == 0, "FP8 expert weight-read grouping query failed ({status})");
        Fp8MoeTrafficContract::from_words(group, contract)
    }

    /// Arithmetic identity and inclusive interval end of the selected program.
    /// Missing/unknown metadata disables exact packing, not ordinary execution.
    pub fn arithmetic_route_for_capacity(&self, capacity: usize, rows: usize)
        -> Result<Option<(u32, usize)>> {
        ensure!(self.info.capacities.contains(&capacity) && (1..=capacity).contains(&rows),
            "FP8 arithmetic query names an uninstalled row/capacity");
        let Some(query) = self.arithmetic_route else { return Ok(None); };
        let (mut route, mut end) = (0u32, 0u32);
        // SAFETY: static function belongs to the live library and writes two live
        // words without CUDA/context access, exactly like the traffic query.
        let status = unsafe { query(u32::try_from(capacity)?, self.prefill.option(),
            u32::try_from(rows)?, &mut route, &mut end) };
        ensure!(status == 0, "FP8 arithmetic route query failed ({status})");
        ensure!(route <= 3 && (rows..=capacity).contains(&(end as usize)),
            "invalid FP8 arithmetic route interval");
        Ok((route != 0).then_some((route, end as usize)))
    }

    /// Scratch bytes of the program for `capacity`.
    pub fn scratch_bytes(&self, capacity: usize) -> Result<usize> {
        let index = self.info.capacities.iter().position(|&c| c == capacity)
            .with_context(|| format!("FP8 expert package has no capacity {capacity}"))?;
        Ok(self.scratch[index])
    }

    pub fn segment_minimum_rows(&self) -> Option<u32> {
        self.segments.iter().map(|segment| segment.minimum_rows).min()
    }

    pub fn segment_recipe(&self) -> Option<Fp8MoeSegment> {
        self.segments.iter().find(|segment| segment.form == self.prefill).copied()
    }

    pub fn segment_scratch_bytes(&self, rows: u32) -> Result<usize> {
        segment_scratch(&self.info, &self.segments, rows)
    }

    /// # Safety
    /// Input/output are device-readable row-address tables, retained with all
    /// addressed rows and other operands until this stream completes. Tables
    /// must be populated before launch and scratch sized at the admitted sum.
    pub unsafe fn launch_segments(&self, pointers: &[*mut c_void; FP8_MOE_POINTERS], rows: usize,
        stream: *mut c_void) -> Result<()> {
        let recipe = self.segment_recipe().context("package has no selected segmented recipe")?;
        ensure!(rows > 0 && rows <= recipe.maximum_rows as usize, "segmented rows exceed code extent");
        // SAFETY: caller guarantees ordered, live operands; context belongs to this library.
        let status = unsafe { (self.launch_segments)(self.context.as_ptr(), pointers.as_ptr(),
            i32::try_from(rows)?, stream) };
        ensure!(status == 0, "FP8 segmented launch failed ({status})");
        Ok(())
    }

    /// Existing BF16 route plane and pair-position map inside admitted scratch.
    pub fn reply_sources(&self, scratch: crate::CuteafdDeviceBuffer, rows: usize, segmented: bool)
        -> Result<(crate::CuteafdDeviceBuffer, crate::CuteafdDeviceBuffer)> {
        let capacity = if segmented { self.segment_recipe().map(|s| s.maximum_rows as usize) }
            else { self.info.capacity_for(rows) }.context("FP8 reply rows exceed installed capacity")?;
        let mut words = [0u64; 4];
        // SAFETY: static package query writes only the supplied four words.
        let status = unsafe { (self.reply_layout)(capacity.try_into()?, self.prefill.option(), rows.try_into()?,
            u32::from(segmented), words.as_mut_ptr(), 4) };
        ensure!(status == 0 && words[0] == 1 && rows > 0, "FP8 reply layout query failed ({status})");
        let view = |offset: u64, bytes: usize| -> Result<crate::CuteafdDeviceBuffer> {
            let offset = usize::try_from(offset)?;
            ensure!(!scratch.ptr.is_null() && offset % 16 == 0
                && offset.checked_add(bytes).is_some_and(|end| end <= scratch.bytes),
                "FP8 reply source outside admitted scratch");
            Ok(crate::CuteafdDeviceBuffer { ptr: scratch.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..scratch })
        };
        let pairs = rows.checked_mul(self.info.topk).context("FP8 reply routes overflow")?;
        let expected = pairs.checked_mul(self.info.hidden).and_then(|n| n.checked_mul(2))
            .context("FP8 reply source overflow")?;
        let y_bytes = usize::try_from(words[3])?;
        ensure!(y_bytes >= expected && y_bytes % (self.info.hidden * 2) == 0, "FP8 reply plane extent mismatch");
        Ok((view(words[2], y_bytes)?, view(words[1], pairs.checked_mul(4).context("FP8 reply positions overflow")?)?))
    }

    /// # Safety
    /// As `launch`; the caller consumes the route plane before scratch reuse.
    pub unsafe fn launch_core(&self, pointers: &[*mut c_void; FP8_MOE_POINTERS], rows: usize,
        segmented: bool, stream: *mut c_void) -> Result<()> {
        let status = if segmented {
            let recipe = self.segment_recipe().context("package has no selected segmented recipe")?;
            ensure!(rows > 0 && rows <= recipe.maximum_rows as usize, "segmented core rows exceed extent");
            // SAFETY: caller retains every operand through the consuming epilogue.
            unsafe { (self.launch_segments_core)(self.context.as_ptr(), pointers.as_ptr(), rows.try_into()?, stream) }
        } else {
            let capacity = self.info.capacity_for(rows).context("FP8 core rows exceed capacity")?;
            ensure!(rows > 0, "FP8 core rows must be positive");
            // SAFETY: same retained operands and capacity contract as the ordinary launch.
            unsafe { (self.launch_core)(self.context.as_ptr(), capacity.try_into()?, pointers.as_ptr(), rows.try_into()?, stream) }
        };
        ensure!(status == 0, "FP8 core launch failed ({status})");
        Ok(())
    }

    /// Runs the smallest program holding `rows`.
    ///
    /// # Safety
    /// Every pointer names live device memory of the documented extent for
    /// `rows` (`scratch` sized for the chosen capacity); `stream` belongs to
    /// the loading device and is drained before any of it is released.
    pub unsafe fn launch(&self, pointers: &[*mut c_void; FP8_MOE_POINTERS], rows: usize, stream: *mut c_void)
        -> Result<()> {
        let capacity = self.info.capacity_for(rows)
            .with_context(|| format!("{rows} rows exceed the FP8 expert package capacities"))?;
        let status = (self.launch)(self.context.as_ptr(), capacity as u32, pointers.as_ptr(), i32::try_from(rows)?,
            stream);
        ensure!(status == 0, "FP8 expert launch ({rows} rows, capacity {capacity}) failed with {status}");
        Ok(())
    }
}

impl Drop for Fp8MoeModule {
    fn drop(&mut self) {
        // SAFETY: the context came from this library's create and is destroyed once.
        unsafe { (self.destroy)(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    #[test]
    fn slot_scratch_selects_capacity_and_rejects_invalid_manifests() {
        let package = [(16, 0), (2048, 8192), (4096, 32768)];
        for (rows, bytes) in [(1, 256), (16, 256), (17, 8192), (2048, 8192),
            (2752, 32768), (4096, 32768)] {
            assert_eq!(fp8_slot_scratch_bytes(package, rows).unwrap(), bytes);
        }
        assert!(fp8_slot_scratch_bytes(package, 0).is_err());
        assert!(fp8_slot_scratch_bytes(package, 4097).is_err());
        for package in [vec![], vec![(0, 1)], vec![(16, 1), (16, 2)], vec![(32, 1), (16, 2)]] {
            assert!(fp8_slot_scratch_bytes(package, 1).is_err());
        }
    }

    #[test]
    fn slot_io_matches_retained_buffers_with_checked_arithmetic() {
        for rows in [1, 16, 2048, 2752, 4096] {
            assert_eq!(fp8_slot_io_bytes(4096, 8, true, false, rows).unwrap(),
                rows * (4096 + 128 + 8 * 8 + 4096 * 2));
            for (wire, sibling) in [(false, false), (true, true)] {
                assert_eq!(fp8_slot_io_bytes(4096, 8, wire, sibling, rows).unwrap(),
                    rows * (4096 * 4 + 8 * 8));
            }
        }
        for (hidden, topk, rows) in [(0, 8, 1), (33, 8, 1), (128, 0, 1), (128, 8, 0),
            (128, usize::MAX, 1), (usize::MAX - 31, 8, 1), (128, 8, usize::MAX)] {
            assert!(fp8_slot_io_bytes(hidden, topk, true, false, rows).is_err());
        }
    }

    fn words() -> [u32; 16] {
        // ABI 5, MXFP4, pre-SiLU clamp, BF16 projection and input.
        [0x01010205, 6144, 2048, 256, 8, 2048, 1, 1, 0, 2, 16, 4096, 0, 0, 0, 0]
    }

    #[test]
    fn activation_contract_is_explicit_and_fail_closed() {
        for format in [1, 2, 4, 5] {
            for activation in [1, 2] {
                let mut w = words();
                w[0] = 5 | format << 8 | activation << 16 | 1 << 24;
                w[8] = 7.0f32.to_bits();
                if format == 2 && activation == 2 {
                    assert!(Fp8MoeInfo::from_words(w).is_err());
                    continue;
                }
                let info = Fp8MoeInfo::from_words(w).unwrap();
                let requested = if activation == 1 { Fp8MoeActivation::SiluBf16Product }
                    else { Fp8MoeActivation::SiluPostClampFp32Product };
                info.validate_activation(requested, 7.0, Fp8MoeProjection::Bf16).unwrap();
                assert!(info.validate_activation(requested, 2.0, Fp8MoeProjection::Bf16).is_err());
                if activation == 1 {
                    assert!(info.validate_activation(Fp8MoeActivation::SiluPostClampFp32Product,
                        7.0, Fp8MoeProjection::Bf16).is_err());
                }
            }
        }
        for abi in [1, 2, 3, 4, 0x01000105, 0x01030105, 0x02010105, 0x01010305] {
            let mut w = words();
            w[0] = abi;
            assert!(Fp8MoeMetadata::from_words_and_scratch(w, |_| panic!("invalid contract queried scratch")).is_err());
        }
        for limit in [f32::NAN, f32::INFINITY, -1.0] {
            let mut w = words();
            w[8] = limit.to_bits();
            assert!(Fp8MoeInfo::from_words(w).is_err());
        }
    }

    #[test]
    fn exact_k32_slices_require_mxfp4_and_balanced_unit_widths() {
        for input in [1, 7] {
            for (tp, width) in [(6, 256), (6, 320), (6, 352), (6, 384),
                (3, 640), (3, 672), (3, 704), (3, 768)] {
                let mut w = words();
                w[2] = width;
                w[3] = 384;
                w[6] = tp;
                w[7] = input;
                assert_eq!(Fp8MoeInfo::from_words(w).unwrap().slice, width as usize);
            }
            for (tp, width) in [(6, 0), (6, 288), (6, 336), (6, 368), (6, 416),
                (3, 656), (3, 688), (3, 736)] {
                let mut w = words();
                w[2] = width;
                w[6] = tp;
                w[7] = input;
                assert!(Fp8MoeMetadata::from_words_and_scratch(w, |_| {
                    panic!("invalid slice must not query workspace")
                }).is_err());
            }
        }
        for format in [1, 4, 5] {
            for (tp, width) in [(6, 320), (6, 352), (3, 672), (3, 704)] {
                let mut w = words();
                w[0] = 5 | format << 8 | 1 << 16 | 1 << 24;
                w[2] = width;
                w[6] = tp;
                assert!(Fp8MoeInfo::from_words(w).is_err());
            }
        }
    }

    #[test]
    fn metadata_validates_before_querying_workspace_extents() {
        let mut invalid = words();
        invalid[0] = 99;
        let error = Fp8MoeMetadata::from_words_and_scratch(invalid, |_| {
            panic!("invalid ABI must not query a workspace")
        }).unwrap_err();
        assert!(error.to_string().contains("unsupported FP8 expert package ABI"));
    }

    #[test]
    fn metadata_queries_every_program_and_selects_the_smallest_fitting_one() {
        let mut queried = Vec::new();
        let metadata = Fp8MoeMetadata::from_words_and_scratch(words(), |capacity| {
            queried.push(capacity);
            Ok((capacity * 1024) as u64)
        }).unwrap();
        assert_eq!(queried, [16, 4096]);
        assert_eq!(metadata.info.weights, Fp8MoeWeights::Mxfp4);
        assert_eq!(metadata.scratch_for(1).unwrap(), 16 * 1024);
        assert_eq!(metadata.scratch_for(16).unwrap(), 16 * 1024);
        assert_eq!(metadata.scratch_for(17).unwrap(), 4096 * 1024);
        assert_eq!(metadata.scratch_for(4096).unwrap(), 4096 * 1024);
        assert!(metadata.scratch_for(4097).is_err());
    }

    #[test]
    fn metadata_propagates_a_missing_workspace_extent() {
        let error = Fp8MoeMetadata::from_words_and_scratch(words(), |capacity| {
            ensure!(capacity != 4096, "missing prefill workspace extent");
            Ok(0)
        }).unwrap_err();
        assert!(error.to_string().contains("missing prefill workspace extent"));
    }
}
