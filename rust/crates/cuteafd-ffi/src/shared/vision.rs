//! Resident ViT owner. Construct, use and drop on its dedicated CUDA thread.
use libloading::Library;
use std::{ffi::c_void, fmt, marker::PhantomData, path::Path, rc::Rc, sync::Arc};

pub const ENCODER_NUMERICS: u32 = 1;
pub const NO_VISION_OFFSET: u64 = u64::MAX;
/// Native tower selector of Mistral Large 4's Pixtral tower (numerics 2).
pub const MISTRAL_RESERVED: u32 = 4;

/// LM rows an encode writes for a ViT patch grid: merged rows, plus one marker
/// row per merged row for a span-writing tower.
pub fn output_rows(grid: [usize; 2], span_markers: bool) -> usize {
    let [h, w] = grid;
    let (Ok(h), Ok(w)) = (u32::try_from(h), u32::try_from(w)) else { return 0; };
    cuteafd_core::media::ImageTokenGeometry { merge: 2, row_markers: u32::from(span_markers) }
        .extent([1, h, w]).map_or(0, |(_, _, rows)| rows)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisionBlock {
    pub qkv: u64,
    pub qkv_bias: u64,
    pub proj: u64,
    pub proj_bias: u64,
    pub gate_up: u64,
    pub gate_up_bias: u64,
    pub down: u64,
    pub down_bias: u64,
    pub norm1: u64,
    pub norm2: u64,
    pub key0_bias: u64,
    pub window: i32,
    pub column_order: i32,
}

#[derive(Clone, Debug, Default)]
pub struct VisionSpec {
    pub abi_version: u32,
    pub max_tokens: u32,
    pub output_width: u32,
    pub reserved: u32,
    pub weight_bytes: u64,
    pub patch: u64,
    pub merger_norm: u64,
    pub merger_fc1: u64,
    pub merger_fc2: u64,
    pub inv_freq: u64,
    pub blocks: Vec<VisionBlock>,
    // Numerics selection is unchanged; layer storage is owned at checkpoint depth.
    pub hidden: u32,
    pub depth: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub patch_size: u32,
    pub merger_width: u32,
    pub norm_eps: f32,
    pub geometry_reserved: u32,
    pub patch_bias: u64,
    pub pos_embed: u64,
    pub merger_norm_bias: u64,
    pub merger_fc1_bias: u64,
    pub merger_fc2_bias: u64,
    pub merger_extra: [u64; 8],
    pub norm1_bias: Vec<u64>,
    pub norm2_bias: Vec<u64>,
    pub q_norm: Vec<u64>,
    pub k_norm: Vec<u64>,
}

#[repr(C)]
struct RawVisionSpec {
    storage_abi: u32,
    abi_version: u32,
    max_tokens: u32,
    output_width: u32,
    reserved: u32,
    weight_bytes: u64,
    patch: u64,
    merger_norm: u64,
    merger_fc1: u64,
    merger_fc2: u64,
    inv_freq: u64,
    blocks: *const VisionBlock,
    // Numerics selection is unchanged; layer storage is owned at checkpoint depth.
    hidden: u32,
    depth: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    patch_size: u32,
    merger_width: u32,
    norm_eps: f32,
    geometry_reserved: u32,
    patch_bias: u64,
    pos_embed: u64,
    merger_norm_bias: u64,
    merger_fc1_bias: u64,
    merger_fc2_bias: u64,
    merger_extra: [u64; 8],
    norm1_bias: *const u64,
    norm2_bias: *const u64,
    q_norm: *const u64,
    k_norm: *const u64,
}

impl VisionSpec {
    /// Layer storage is owned and follows the checkpoint depth, never an ABI cap.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = u32::try_from(depth).expect("vision depth must fit u32");
        self.blocks = vec![VisionBlock::default(); depth];
        self.norm1_bias.resize(depth, 0);
        self.norm2_bias.resize(depth, 0);
        self.q_norm.resize(depth, 0);
        self.k_norm.resize(depth, 0);
        self
    }

    // The raw view stays private and is borrowed only for synchronous FFI calls.
    // Native registration deep-copies every layer before returning.
    fn raw(&self) -> Result<RawVisionSpec, VisionError> {
        let depth = self.depth as usize;
        if depth == 0 || self.blocks.len() != depth
            || [&self.norm1_bias, &self.norm2_bias, &self.q_norm, &self.k_norm]
                .iter().any(|v| v.len() != depth) {
            return Err(VisionError::InvalidInput("vision layer storage/depth mismatch"));
        }
        Ok(RawVisionSpec {
            storage_abi: 3,
            abi_version: self.abi_version,
            max_tokens: self.max_tokens,
            output_width: self.output_width,
            reserved: self.reserved,
            weight_bytes: self.weight_bytes,
            patch: self.patch,
            merger_norm: self.merger_norm,
            merger_fc1: self.merger_fc1,
            merger_fc2: self.merger_fc2,
            inv_freq: self.inv_freq,
            blocks: self.blocks.as_ptr(),
            hidden: self.hidden,
            depth: self.depth,
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            intermediate: self.intermediate,
            patch_size: self.patch_size,
            merger_width: self.merger_width,
            norm_eps: self.norm_eps,
            geometry_reserved: self.geometry_reserved,
            patch_bias: self.patch_bias,
            pos_embed: self.pos_embed,
            merger_norm_bias: self.merger_norm_bias,
            merger_fc1_bias: self.merger_fc1_bias,
            merger_fc2_bias: self.merger_fc2_bias,
            merger_extra: self.merger_extra,
            norm1_bias: self.norm1_bias.as_ptr(),
            norm2_bias: self.norm2_bias.as_ptr(),
            q_norm: self.q_norm.as_ptr(),
            k_norm: self.k_norm.as_ptr(),
        })
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisionLedger {
    pub weights: u64,
    pub scratch: u64,
    pub blas_workspace: u64,
    pub device_allocations: u64,
    pub encodes: u64,
}
impl VisionLedger {
    pub fn total_bytes(&self) -> u64 {
        self.weights + self.scratch + self.blas_workspace
    }
}

#[derive(Debug)]
pub enum VisionError {
    Native(i32),
    Failure(anyhow::Error),
    InvalidInput(&'static str),
    Runtime(String),
    Library(libloading::Error),
}
impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failure(error) => std::fmt::Display::fmt(error, f),
            Self::Native(status) => write!(f, "vision native status {status}"),
            Self::InvalidInput(reason) => f.write_str(reason),
            Self::Runtime(reason) => f.write_str(reason),
            Self::Library(e) => write!(f, "vision library: {e}"),
        }
    }
}
impl std::error::Error for VisionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self { Self::Failure(error) => Some(error.as_ref()), _ => None }
    }
}
impl From<libloading::Error> for VisionError {
    fn from(e: libloading::Error) -> Self {
        Self::Library(e)
    }
}
#[cfg(feature = "test-support")]
pub fn check_for_test(status: i32) -> Result<(), VisionError> { check(status) }
fn check(status: i32) -> Result<(), VisionError> {
    if status == 0 {
        crate::memory_ledger::fault::clear_failure();
        Ok(())
    } else {
        Err(VisionError::Failure(crate::memory_ledger::capture_failure(anyhow::Error::new(VisionError::Native(status)))))
    }
}

type Required = unsafe extern "C" fn(*const RawVisionSpec, *mut VisionLedger) -> i32;
type Create = unsafe extern "C" fn(*const RawVisionSpec, i32, u64, *const super::media_backings::RawTags, *mut *mut c_void) -> i32;
type Upload = unsafe extern "C" fn(*mut c_void, u64, *const u8, u64) -> i32;
type Observer = unsafe extern "C" fn(*mut c_void, i32, *const c_void, i32, i32, i32) -> i32;
type Encode = unsafe extern "C" fn(
    *mut c_void,
    *const u8,
    u64,
    *const f32,
    i32,
    i32,
    *mut u16,
    u64,
    Option<Observer>,
    *mut c_void,
) -> i32;
type Ledger = unsafe extern "C" fn(*mut c_void, *mut VisionLedger) -> i32;

pub struct NativeVision {
    _library: Arc<Library>,
    owner: *mut c_void,
    encode: Encode,
    ledger: Ledger,
    max_tokens: usize,
    output_width: usize,
    patch_size: usize,
    /// Mistral's Pixtral tower (reserved 4) emits its whole LM span: one
    /// [IMG_BREAK]/[IMG_END] row after each merged row of [IMG] rows.
    span_markers: bool,
    backings: super::media_backings::MediaBackings,
    // CUDA and the BLAS handle never migrate away from their owner thread.
    _thread: PhantomData<Rc<()>>,
}
impl NativeVision {
    /// Pure header/arena query, before loading tensor data or initializing CUDA.
    pub fn required(path: &Path, spec: &VisionSpec) -> Result<VisionLedger, VisionError> {
        // SAFETY: trusted library and versioned header ABI; query initializes no CUDA state.
        unsafe {
            let library = Arc::new(Library::new(crate::host_isa::native_library(path))?);
            crate::memory_ledger::fault::install(&library);
            let required = library.get::<Required>(b"cuteafd_vision_required")?;
            let mut ledger = VisionLedger::default();
            check(required(&spec.raw()?, &mut ledger))?;
            Ok(ledger)
        }
    }
    /// `tags` name the tower's weight arena and scratch arena in the ledger.
    pub fn prepare(path: &Path, spec: &VisionSpec, device: i32, admitted: u64, tags: crate::memory_ledger::MediaTags)
        -> Result<Self, VisionError> {
        Self::load_inner(path, spec, device, admitted, None, tags)
    }
    /// Existing admitted arena; the caller retains this owner until external writes drain.
    pub fn weight_arena(&self) -> Result<usize, VisionError> {
        // SAFETY: exact header ABI; the owner is live and confined to this thread.
        unsafe {
            let get = self._library.get::<unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32>(b"cuteafd_vision_weight_arena")?;
            let mut arena = std::ptr::null_mut();
            check(get(self.owner, &mut arena))?;
            Ok(arena as usize)
        }
    }
    pub fn upload_range(&self, offset: u64, bytes: &[u8]) -> Result<(), VisionError> {
        // SAFETY: native checks the range and drains the host upload before return.
        unsafe { check(self._library.get::<Upload>(b"cuteafd_vision_upload")?(self.owner, offset, bytes.as_ptr(), bytes.len() as u64)) }
    }
    /// External push streams must already be drained by the group receiver.
    pub fn commit_weights(&mut self) -> Result<(), VisionError> {
        // SAFETY: exact header ABI and live thread-confined owner.
        unsafe { check(self._library.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cuteafd_vision_commit_weights")?(self.owner)) }
    }
    /// Check admission before native weights, scratch or cuBLAS allocations.
    /// `weights` must be the complete initialized arena described by `spec`.
    pub fn load(
        path: &Path,
        spec: &VisionSpec,
        device: i32,
        admitted_bytes: u64,
        weights: &[u8],
        tags: crate::memory_ledger::MediaTags,
    ) -> Result<Self, VisionError> {
        Self::load_inner(path, spec, device, admitted_bytes, Some(weights), tags)
    }
    fn load_inner(path: &Path, spec: &VisionSpec, device: i32, admitted_bytes: u64, weights: Option<&[u8]>,
        tags: crate::memory_ledger::MediaTags) -> Result<Self, VisionError> {
        if weights.is_some_and(|w| w.len() as u64 != spec.weight_bytes) {
            return Err(VisionError::InvalidInput(
                "vision weight arena length mismatch",
            ));
        }
        // SAFETY: caller selects a trusted native library. Every symbol has the
        // versioned header's exact ABI; the library outlives the native owner.
        unsafe {
            let library = Arc::new(Library::new(crate::host_isa::native_library(path))?);
            crate::memory_ledger::fault::install(&library);
            let required = *library.get::<Required>(b"cuteafd_vision_required")?;
            let create = *library.get::<Create>(b"cuteafd_vision_create")?;
            let upload = *library.get::<Upload>(b"cuteafd_vision_upload")?;
            let mut ledger = VisionLedger::default();
            check(required(&spec.raw()?, &mut ledger))?;
            if ledger.total_bytes() > admitted_bytes {
                return Err(VisionError::InvalidInput(
                    "vision weight/scratch admission shortfall",
                ));
            }
            let mut result = Self {
                owner: std::ptr::null_mut(),
                encode: *library.get::<Encode>(b"cuteafd_vision_encode")?,
                ledger: *library.get::<Ledger>(b"cuteafd_vision_get_ledger")?,
                _library: library,
                max_tokens: spec.max_tokens as usize,
                output_width: spec.output_width as usize,
                patch_size: if spec.abi_version == 1 { 16 } else { spec.patch_size as usize },
                span_markers: spec.abi_version == 2 && spec.reserved == MISTRAL_RESERVED,
                backings: Default::default(),
                _thread: PhantomData,
            };
            let native = super::media_backings::NativeTags::new(tags).map_err(VisionError::Native)?;
            let status = create(&spec.raw()?, device, admitted_bytes, &native.raw(), &mut result.owner);
            // Each backing native still holds is recorded, success or not; on
            // failure Drop destroys the owner and retires what native frees.
            match super::media_backings::MediaBackings::created(&result._library, "cuteafd_vision", status,
                result.owner, device, tags) {
                Ok(backings) => result.backings = backings,
                Err((status, backings)) => { result.backings = backings; return Err(VisionError::Native(status)); }
            }
            if let Some(weights) = weights {
                check(upload(result.owner, 0, weights.as_ptr(), weights.len() as u64))?;
            }
            Ok(result)
        }
    }
    /// Encode into caller-owned output. No Rust or CUDA allocation in this call.
    /// The LUT is normalized FP32 `[channel][rgb8]`, with reference operation order.
    pub fn encode_into(
        &mut self,
        rgb: &[u8],
        grid: [usize; 2],
        lut: &[f32; 768],
        output: &mut [u16],
    ) -> Result<(), VisionError> {
        let [h, w] = grid;
        let patches = h
            .checked_mul(w)
            .ok_or(VisionError::InvalidInput("vision grid overflow"))?;
        if h < 2
            || w < 2
            || h % 2 != 0
            || w % 2 != 0
            || patches > self.max_tokens * 4
            || rgb.len() != patches * self.patch_size * self.patch_size * 3
            || output.len() != output_rows(grid, self.span_markers) * self.output_width
        {
            return Err(VisionError::InvalidInput(
                "vision RGB/grid/output extent mismatch",
            ));
        }
        // SAFETY: validated host extents stay live through synchronous encode.
        // Native owner serializes its stream and drains on success and errors.
        check(unsafe {
            (self.encode)(
                self.owner,
                rgb.as_ptr(),
                rgb.len() as u64,
                lut.as_ptr(),
                h as i32,
                w as i32,
                output.as_mut_ptr(),
                std::mem::size_of_val(output) as u64,
                None,
                std::ptr::null_mut(),
            )
        })
    }
    pub fn ledger(&self) -> Result<VisionLedger, VisionError> {
        let mut ledger = VisionLedger::default();
        // SAFETY: exclusive owner is live on this thread; native writes one ledger.
        check(unsafe { (self.ledger)(self.owner, &mut ledger) })?;
        Ok(ledger)
    }
}
impl Drop for NativeVision {
    fn drop(&mut self) {
        if !self.owner.is_null() {
            // SAFETY: destroy drains before freeing; library remains loaded.
            // Only backings native reports as freed leave the ledger.
            let status = unsafe { self.backings.destroy(&self._library, "cuteafd_vision", self.owner) };
            if status != 0 {
                tracing::error!(status, "vision teardown failed");
            }
            self.owner = std::ptr::null_mut();
        }
    }
}

type Inject =
    unsafe extern "C" fn(*const u16, *const u32, *mut u16, i32, i32, i32, i32, *mut c_void) -> i32;
/// Pre-resolved graph-compatible scatter after the normal token embedding gather.
pub struct EmbeddingInjection<'a> {
    _library: &'a crate::NativeLibrary,
    launch: Inject,
}
impl crate::NativeLibrary {
    pub fn embedding_injection(&self) -> Result<EmbeddingInjection<'_>, VisionError> {
        // SAFETY: library stays loaded for the lifetime of the resolved exact-ABI symbol.
        let launch = unsafe { *self.lib.get::<Inject>(b"cuteafd_embed_inject")? };
        Ok(EmbeddingInjection {
            _library: self,
            launch,
        })
    }
}
impl EmbeddingInjection<'_> {
    /// Stages host rows into caller-owned scratch before any consumer reuses it.
    /// Indices are checked on the host; all producers/consumers use this stream.
    pub fn inject_host(&self, features: &[u8], indices: &[u32], feature_scratch: crate::CuteafdDeviceBuffer,
        index_scratch: crate::CuteafdDeviceBuffer, output: crate::CuteafdDeviceBuffer,
        rows: usize, width: usize, copies: usize, stream: *mut c_void) -> Result<(), VisionError> {
        if indices.is_empty() { return Ok(()); }
        validate_injection(feature_scratch, index_scratch, output, indices.len(), rows, width, copies)?;
        if features.len() != indices.len() * width * 2
            || indices.iter().any(|&i| i as usize >= rows)
            || indices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(VisionError::InvalidInput("host embedding rows/indices"));
        }
        // SAFETY: the caller's stream owns the scratch; drain its previous readers before staging.
        unsafe { self._library.cuda_stream_synchronize(stream) }
            .map_err(|e| VisionError::Runtime(e.to_string()))?;
        self._library.copy_h2d(crate::CuteafdDeviceBuffer { bytes: features.len(), ..feature_scratch }, features)
            .map_err(|e| VisionError::Runtime(e.to_string()))?;
        let index_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        self._library.copy_h2d(crate::CuteafdDeviceBuffer { bytes: index_bytes.len(), ..index_scratch }, &index_bytes)
            .map_err(|e| VisionError::Runtime(e.to_string()))?;
        // SAFETY: extents and sorted unique indices validated; scratch/output remain owned by the engine.
        unsafe { self.launch(feature_scratch, index_scratch, output, indices.len(), rows, width, copies, stream)? };
        // SAFETY: finish reading the temporary index table before its normal producer overwrites it.
        unsafe { self._library.cuda_stream_synchronize(stream) }.map_err(|e| VisionError::Runtime(e.to_string()))
    }

    /// # Safety
    /// Buffers reside on the current stream device and remain live until it drains.
    /// Producers precede this launch; indices are unique and less than rows. Input
    /// and output storage are disjoint and no conflicting accesses are pending.
    pub unsafe fn launch(
        &self,
        features: crate::CuteafdDeviceBuffer,
        indices: crate::CuteafdDeviceBuffer,
        output: crate::CuteafdDeviceBuffer,
        feature_rows: usize,
        rows: usize,
        width: usize,
        copies: usize,
        stream: *mut c_void,
    ) -> Result<(), VisionError> {
        validate_injection(features, indices, output, feature_rows, rows, width, copies)?;
        // SAFETY: caller owns CUDA publication/lifetimes; extents/devices validated here.
        check(unsafe {
            (self.launch)(
                features.ptr.cast(),
                indices.ptr.cast(),
                output.ptr.cast(),
                feature_rows as i32,
                rows as i32,
                width as i32,
                copies as i32,
                stream,
            )
        })
    }
}
fn validate_injection(
    features: crate::CuteafdDeviceBuffer,
    indices: crate::CuteafdDeviceBuffer,
    output: crate::CuteafdDeviceBuffer,
    feature_rows: usize,
    rows: usize,
    width: usize,
    copies: usize,
) -> Result<(), VisionError> {
    if feature_rows == 0
        || feature_rows > rows
        || rows > 16384
        || width == 0
        || width > 16384
        || !(1..=4).contains(&copies)
    {
        return Err(VisionError::InvalidInput("embedding injection shape"));
    }
    for (buffer, bytes, align) in [
        (features, feature_rows * width * 2, 2),
        (indices, feature_rows * 4, 4),
        (output, rows * copies * width * 2, 2),
    ] {
        if buffer.ptr.is_null()
            || buffer.ptr as usize % align != 0
            || buffer.bytes < bytes
            || buffer.device_id < 0
            || buffer.device_id != output.device_id
            || buffer.flags != 0
        {
            return Err(VisionError::InvalidInput(
                "embedding injection buffer/device extent",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn injection_rejects_bad_extents_and_devices_before_launch() {
        let buffer = crate::CuteafdDeviceBuffer {
            ptr: 0x1000usize as *mut c_void,
            bytes: 128,
            device_id: 0,
            flags: 0,
        };
        assert!(validate_injection(buffer, buffer, buffer, 2, 4, 4, 4).is_ok());
        assert!(validate_injection(buffer, buffer, buffer, 2, 4, 4, 5).is_err());
        assert!(validate_injection(buffer, buffer, buffer, 5, 4, 4, 1).is_err());
        assert!(validate_injection(buffer, buffer, buffer, 2, 4, 64, 4).is_err());
        assert!(validate_injection(
            buffer,
            crate::CuteafdDeviceBuffer {
                device_id: 1,
                ..buffer
            },
            buffer,
            2,
            4,
            4,
            1
        )
        .is_err());
    }
    #[test]
    fn layer_storage_follows_depth_and_rejects_ragged_views() {
        for depth in [24, 27, 28, 30, 50] {
            let mut spec = VisionSpec { q_norm: vec![NO_VISION_OFFSET; depth],
                ..Default::default() }.with_depth(depth);
            spec.blocks[depth - 1].norm1 = 1234;
            let raw = spec.raw().unwrap();
            assert_eq!(raw.depth as usize, depth);
            assert_eq!(raw.blocks, spec.blocks.as_ptr());
            assert_eq!(spec.q_norm, vec![NO_VISION_OFFSET; depth]);
            let cloned = spec.clone();
            assert_ne!(cloned.blocks.as_ptr(), raw.blocks);
            drop(spec);
            assert_eq!(cloned.blocks[depth - 1].norm1, 1234);
            let mut ragged = cloned;
            ragged.k_norm.pop();
            assert!(ragged.raw().is_err());
        }
        assert!(VisionSpec::default().raw().is_err());
    }

    #[test]
    fn c_header_layout_is_stable() {
        assert_eq!(std::mem::size_of::<VisionBlock>(), 96);
        assert_eq!(std::mem::size_of::<VisionLedger>(), 40);
        assert_eq!(std::mem::size_of::<RawVisionSpec>(), 256);
        assert_eq!(std::mem::offset_of!(RawVisionSpec, blocks), 72);
        assert_eq!(std::mem::offset_of!(RawVisionSpec, hidden), 80);
        assert_eq!(std::mem::offset_of!(RawVisionSpec, patch_bias), 120);
        assert_eq!(std::mem::offset_of!(RawVisionSpec, norm1_bias), 224);
    }
}
