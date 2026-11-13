//! Resident MiMo audio owner, confined to the shared encoder CUDA thread.
use libloading::Library;
use std::{ffi::c_void, fmt, marker::PhantomData, path::Path, rc::Rc, sync::Arc};

pub const AUDIO_ABI: u32 = 1;
pub const AUDIO_FP32_NUMERICS: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioCodecBlock {
    pub q: u64,
    pub qb: u64,
    pub k: u64,
    pub v: u64,
    pub vb: u64,
    pub o: u64,
    pub ob: u64,
    pub norm1: u64,
    pub norm1b: u64,
    pub norm2: u64,
    pub norm2b: u64,
    pub fc1: u64,
    pub fc1b: u64,
    pub fc2: u64,
    pub fc2b: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioPatchBlock {
    pub norm1: u64,
    pub norm2: u64,
    pub q: u64,
    pub qb: u64,
    pub k: u64,
    pub kb: u64,
    pub v: u64,
    pub vb: u64,
    pub o: u64,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioSpec {
    pub abi_version: u32,
    pub numerics: u32,
    pub max_samples: u32,
    pub output_width: u32,
    pub weight_bytes: u64,
    pub conv1: u64,
    pub conv1b: u64,
    pub conv2: u64,
    pub conv2b: u64,
    pub downsample: u64,
    pub norm: u64,
    pub normb: u64,
    pub downnorm: u64,
    pub downnormb: u64,
    pub codec: [AudioCodecBlock; 24],
    pub codebooks: [u64; 20],
    pub speech: [u64; 20],
    pub patch: [AudioPatchBlock; 6],
    pub patch_norm: u64,
    pub projection1: u64,
    pub projection2: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AudioLedger {
    pub weights: u64,
    pub scratch: u64,
    pub blas_workspace: u64,
    pub fft_workspace: u64,
    pub device_allocations: u64,
    pub encodes: u64,
}
impl AudioLedger {
    pub fn total_bytes(&self) -> Result<u64, AudioError> {
        [
            self.weights,
            self.scratch,
            self.blas_workspace,
            self.fft_workspace,
        ]
        .into_iter()
        .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or(AudioError::InvalidInput("audio ledger overflow"))
    }
}
/// Qualification overrides live through synchronous upload only. All-empty
/// slices select embedded native constants; partially empty sets are rejected.
pub struct AudioTables<'a> {
    pub hann: &'a [f32],
    pub mel_filterbank: &'a [f32],
    pub codec_rotary: &'a [f32],
    pub patch_rotary: &'a [f32],
}
#[derive(Debug)]
pub enum AudioError {
    Native(i32),
    Failure(anyhow::Error),
    InvalidInput(&'static str),
    Library(libloading::Error),
}
impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failure(error) => std::fmt::Display::fmt(error, f),
            Self::Native(code) => write!(f, "audio native status {code}"),
            Self::InvalidInput(reason) => f.write_str(reason),
            Self::Library(error) => write!(f, "audio library: {error}"),
        }
    }
}
impl std::error::Error for AudioError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self { Self::Failure(error) => Some(error.as_ref()), _ => None }
    }
}
impl From<libloading::Error> for AudioError {
    fn from(error: libloading::Error) -> Self {
        Self::Library(error)
    }
}
#[cfg(feature = "test-support")]
pub fn check_for_test(code: i32) -> Result<(), AudioError> { check(code) }
fn check(code: i32) -> Result<(), AudioError> {
    if code == 0 {
        crate::memory_ledger::fault::clear_failure();
        Ok(())
    } else {
        Err(AudioError::Failure(crate::memory_ledger::capture_failure(anyhow::Error::new(AudioError::Native(code)))))
    }
}
/// Geometry includes per-segment downsampling, not downsampling the concatenation.
pub fn audio_geometry(samples: usize) -> Result<(usize, usize), AudioError> {
    if !(481..=7_200_000).contains(&samples) {
        return Err(AudioError::InvalidInput(
            "audio PCM must contain 481..7200000 samples",
        ));
    }
    let frames = samples / 240 + 1;
    let tail = frames % 6000;
    let codes = frames / 6000 * 1500 + tail.div_ceil(2).div_ceil(2);
    Ok((codes, codes.div_ceil(4)))
}

type Required = unsafe extern "C" fn(*const AudioSpec, *mut AudioLedger) -> i32;
type Create = unsafe extern "C" fn(*const AudioSpec, i32, u64, *const super::media_backings::RawTags, *mut *mut c_void) -> i32;
type Upload = unsafe extern "C" fn(
    *mut c_void,
    *const u8,
    u64,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
) -> i32;
type Observer = unsafe extern "C" fn(*mut c_void, i32, i32, *const c_void, i32, i32) -> i32;
type Encode = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    u32,
    *mut f32,
    u64,
    *mut i32,
    u64,
    Option<Observer>,
    *mut c_void,
) -> i32;
type Ledger = unsafe extern "C" fn(*mut c_void, *mut AudioLedger) -> i32;
type Backend = unsafe extern "C" fn(*mut u8, u64) -> i32;

pub struct NativeAudio {
    _library: Arc<Library>,
    owner: *mut c_void,
    encode: Encode,
    ledger: Ledger,
    backings: super::media_backings::MediaBackings,
    max_samples: usize,
    output_width: usize,
    _thread: PhantomData<Rc<()>>,
}
impl NativeAudio {
    /// Select attested build-time tables without loading any serving sidecar.
    pub fn load_embedded(
        path: &Path,
        spec: &AudioSpec,
        device: i32,
        admitted_bytes: u64,
        weights: &[u8],
        tags: crate::memory_ledger::MediaTags,
    ) -> Result<Self, AudioError> {
        Self::load(
            path,
            spec,
            device,
            admitted_bytes,
            weights,
            tags,
            AudioTables {
                hann: &[],
                mel_filterbank: &[],
                codec_rotary: &[],
                patch_rotary: &[],
            },
        )
    }
    pub fn required(path: &Path, spec: &AudioSpec) -> Result<AudioLedger, AudioError> {
        // SAFETY: trusted versioned native ABI, no CUDA context or payload read.
        unsafe {
            let library = Arc::new(Library::new(crate::host_isa::native_library(path))?);
            crate::memory_ledger::fault::install(&library);
            let required = library.get::<Required>(b"cuteafd_audio_required")?;
            let mut ledger = AudioLedger::default();
            check(required(spec, &mut ledger))?;
            ledger.total_bytes()?;
            Ok(ledger)
        }
    }
    pub fn backend(path: &Path) -> Result<String, AudioError> {
        let mut buffer = [0u8; 256];
        // SAFETY: exact backend symbol ABI and bounded caller-owned output.
        unsafe {
            let library = Arc::new(Library::new(crate::host_isa::native_library(path))?);
            crate::memory_ledger::fault::install(&library);
            let backend = library.get::<Backend>(b"cuteafd_audio_backend")?;
            check(backend(buffer.as_mut_ptr(), buffer.len() as u64))?;
        }
        let end = buffer
            .iter()
            .position(|byte| *byte == 0)
            .ok_or(AudioError::InvalidInput(
                "unterminated audio backend identity",
            ))?;
        std::str::from_utf8(&buffer[..end])
            .map(str::to_owned)
            .map_err(|_| AudioError::InvalidInput("audio backend identity is not UTF8"))
    }
    pub fn load(
        path: &Path,
        spec: &AudioSpec,
        device: i32,
        admitted_bytes: u64,
        weights: &[u8],
        tags: crate::memory_ledger::MediaTags,
        tables: AudioTables<'_>,
    ) -> Result<Self, AudioError> {
        Self::load_inner(path, spec, device, admitted_bytes, Some(weights), tags, tables)
    }
    /// `tags` name the tower's weight arena and scratch arena in the ledger.
    pub fn prepare(path: &Path, spec: &AudioSpec, device: i32, admitted: u64, tags: crate::memory_ledger::MediaTags)
        -> Result<Self, AudioError> {
        Self::load_inner(path, spec, device, admitted, None, tags, AudioTables {
            hann: &[], mel_filterbank: &[], codec_rotary: &[], patch_rotary: &[],
        })
    }
    pub fn weight_arena(&self) -> Result<usize, AudioError> {
        // SAFETY: exact header ABI; the owner stays live until external push writes drain.
        unsafe {
            let get = self._library.get::<unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32>(b"cuteafd_audio_weight_arena")?;
            let mut arena = std::ptr::null_mut();
            check(get(self.owner, &mut arena))?;
            Ok(arena as usize)
        }
    }
    /// External push streams must be drained before deriving RVQ tables and publishing.
    pub fn commit_weights(&mut self) -> Result<(), AudioError> {
        // SAFETY: exact header ABI and live thread-confined owner.
        unsafe { check(self._library.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cuteafd_audio_commit_weights")?(self.owner)) }
    }
    fn load_inner(path: &Path, spec: &AudioSpec, device: i32, admitted_bytes: u64,
        weights: Option<&[u8]>, tags: crate::memory_ledger::MediaTags, tables: AudioTables<'_>) -> Result<Self, AudioError> {
        audio_geometry(spec.max_samples as usize)?;
        let mel_capacity = 6000.min(spec.max_samples as usize / 240 + 1);
        let embedded = tables.hann.is_empty()
            && tables.mel_filterbank.is_empty()
            && tables.codec_rotary.is_empty()
            && tables.patch_rotary.is_empty();
        if weights.is_some_and(|weights| weights.len() as u64 != spec.weight_bytes)
            || (!embedded
                && (tables.hann.len() != 960
                    || tables.mel_filterbank.len() != 481 * 128
                    || tables.codec_rotary.len() != mel_capacity.div_ceil(2) * 64
                    || tables.patch_rotary.len() != 4 * 64))
        {
            return Err(AudioError::InvalidInput(
                "audio weight/table extent mismatch",
            ));
        }
        if [
            tables.hann,
            tables.mel_filterbank,
            tables.codec_rotary,
            tables.patch_rotary,
        ]
        .into_iter()
        .any(|table| table.iter().any(|x| !x.is_finite()))
        {
            return Err(AudioError::InvalidInput(
                "audio table contains nonfinite values",
            ));
        }
        // SAFETY: checked immutable host extents, exact ABI symbols and library
        // retained until native owner destruction. Native create/upload drain errors.
        unsafe {
            let library = Arc::new(Library::new(crate::host_isa::native_library(path))?);
            crate::memory_ledger::fault::install(&library);
            let required = *library.get::<Required>(b"cuteafd_audio_required")?;
            let create = *library.get::<Create>(b"cuteafd_audio_create")?;
            let upload = *library.get::<Upload>(b"cuteafd_audio_upload")?;
            let mut ledger = AudioLedger::default();
            check(required(spec, &mut ledger))?;
            if ledger.total_bytes()? > admitted_bytes {
                return Err(AudioError::InvalidInput(
                    "audio weight/scratch admission shortfall",
                ));
            }
            let mut result = Self {
                owner: std::ptr::null_mut(),
                encode: *library.get::<Encode>(b"cuteafd_audio_encode")?,
                ledger: *library.get::<Ledger>(b"cuteafd_audio_get_ledger")?,
                backings: Default::default(),
                _library: library,
                max_samples: spec.max_samples as usize,
                output_width: spec.output_width as usize,
                _thread: PhantomData,
            };
            let native = super::media_backings::NativeTags::new(tags).map_err(AudioError::Native)?;
            let status = create(spec, device, admitted_bytes, &native.raw(), &mut result.owner);
            // Each backing native still holds is recorded, success or not; on
            // failure Drop destroys the owner and retires what native frees.
            match super::media_backings::MediaBackings::created(&result._library, "cuteafd_audio", status,
                result.owner, device, tags) {
                Ok(backings) => result.backings = backings,
                Err((status, backings)) => { result.backings = backings; return Err(AudioError::Native(status)); }
            }
            if let Some(weights) = weights {
            check(upload(
                result.owner,
                weights.as_ptr(),
                weights.len() as u64,
                if embedded {
                    std::ptr::null()
                } else {
                    tables.hann.as_ptr()
                },
                if embedded {
                    std::ptr::null()
                } else {
                    tables.mel_filterbank.as_ptr()
                },
                if embedded {
                    std::ptr::null()
                } else {
                    tables.codec_rotary.as_ptr()
                },
                if embedded {
                    std::ptr::null()
                } else {
                    tables.patch_rotary.as_ptr()
                },
            ))?;
            }
            Ok(result)
        }
    }
    /// Synchronous encode has no host/native allocation and no graph captures.
    pub fn encode_into(
        &mut self,
        pcm: &[f32],
        output: &mut [f32],
        codes: Option<&mut [i32]>,
    ) -> Result<(), AudioError> {
        let (frames, tokens) = audio_geometry(pcm.len())?;
        if pcm.len() > self.max_samples
            || pcm.iter().any(|x| !x.is_finite())
            || output.len() != tokens * self.output_width
            || codes
                .as_ref()
                .is_some_and(|values| values.len() != frames * 20)
        {
            return Err(AudioError::InvalidInput(
                "audio PCM/output/code extent mismatch",
            ));
        }
        let (code_pointer, code_bytes) = codes
            .map(|values| (values.as_mut_ptr(), std::mem::size_of_val(values) as u64))
            .unwrap_or((std::ptr::null_mut(), 0));
        // SAFETY: validated borrowed extents stay live until native stream drains.
        check(unsafe {
            (self.encode)(
                self.owner,
                pcm.as_ptr(),
                pcm.len() as u32,
                output.as_mut_ptr(),
                std::mem::size_of_val(output) as u64,
                code_pointer,
                code_bytes,
                None,
                std::ptr::null_mut(),
            )
        })
    }
    pub fn ledger(&self) -> Result<AudioLedger, AudioError> {
        let mut ledger = AudioLedger::default();
        // SAFETY: owner is live and thread-confined; output is one writable ledger.
        check(unsafe { (self.ledger)(self.owner, &mut ledger) })?;
        Ok(ledger)
    }
}
impl Drop for NativeAudio {
    fn drop(&mut self) {
        if !self.owner.is_null() {
            // SAFETY: exclusively owned native pointer, library still live. Native
            // destruction drains queued work before releasing modules/storage.
            // Only backings native reports as freed leave the ledger.
            let status = unsafe { self.backings.destroy(&self._library, "cuteafd_audio", self.owner) };
            if status != 0 {
                tracing::warn!(status, "audio owner destruction reported an error");
            }
            self.owner = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_layout_and_segment_geometry() {
        assert_eq!(std::mem::size_of::<AudioCodecBlock>(), 120);
        assert_eq!(std::mem::size_of::<AudioPatchBlock>(), 96);
        assert_eq!(std::mem::size_of::<AudioSpec>(), 3896);
        assert_eq!(std::mem::offset_of!(AudioSpec, codec), 96);
        assert_eq!(std::mem::offset_of!(AudioSpec, patch), 3296);
        assert_eq!(std::mem::size_of::<AudioLedger>(), 48);
        assert_eq!(audio_geometry(481).unwrap(), (1, 1));
        assert_eq!(audio_geometry(7_200_000).unwrap(), (7501, 1876));
        assert_eq!(audio_geometry(5999 * 240).unwrap(), (1500, 375));
        assert_eq!(audio_geometry(6000 * 240).unwrap(), (1501, 376));
        assert!(audio_geometry(480).is_err());
        assert!(audio_geometry(7_200_001).is_err());
    }
    #[test]
    fn table_contract_rejects_partial_and_nonfinite_before_library_load() {
        let spec = AudioSpec {
            max_samples: 481,
            ..Default::default()
        };
        let absent = Path::new("/missing/audio-library.so");
        let empty = || AudioTables {
            hann: &[],
            mel_filterbank: &[],
            codec_rotary: &[],
            patch_rotary: &[],
        };
        assert!(matches!(
            NativeAudio::load_embedded(absent, &spec, 0, 0, &[], crate::memory_ledger::MediaTags::new("test/tower")),
            Err(AudioError::Library(_))
        ));
        assert!(matches!(
            NativeAudio::load(absent, &spec, 0, 0, &[], crate::memory_ledger::MediaTags::new("test/tower"), empty()),
            Err(AudioError::Library(_))
        ));
        assert!(matches!(
            NativeAudio::load(
                absent,
                &spec,
                0,
                0,
                &[],
                crate::memory_ledger::MediaTags::new("test/tower"),
                AudioTables {
                    hann: &[0.0; 960],
                    ..empty()
                }
            ),
            Err(AudioError::InvalidInput(
                "audio weight/table extent mismatch"
            ))
        ));
        assert!(matches!(
            NativeAudio::load(
                absent,
                &spec,
                0,
                0,
                &[],
                crate::memory_ledger::MediaTags::new("test/tower"),
                AudioTables {
                    hann: &[f32::NAN; 960],
                    mel_filterbank: &[0.0; 481 * 128],
                    codec_rotary: &[0.0; 2 * 64],
                    patch_rotary: &[0.0; 4 * 64],
                }
            ),
            Err(AudioError::InvalidInput(
                "audio table contains nonfinite values"
            ))
        ));
        assert!(matches!(
            NativeAudio::load_embedded(absent, &spec, 0, 0, &[0], crate::memory_ledger::MediaTags::new("test/tower")),
            Err(AudioError::InvalidInput(
                "audio weight/table extent mismatch"
            ))
        ));
    }
    #[test]
    fn ledger_overflow_fails_closed() {
        assert!(AudioLedger {
            weights: u64::MAX,
            scratch: 1,
            ..Default::default()
        }
        .total_bytes()
        .is_err());
    }
}
