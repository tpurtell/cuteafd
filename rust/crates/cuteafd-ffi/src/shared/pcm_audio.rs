//! Shared BF16 PCM owner. Confined to the encoder CUDA thread; no unsafe Send.
use super::audio::AudioError;
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, rc::Rc, sync::Arc};
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PcmBlock {
    pub conv: u64,
    pub conv_bias: u64,
    pub norm: u64,
    pub fc1: u64,
    pub fc1_bias: u64,
    pub fc2: u64,
    pub fc2_bias: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PcmStage {
    pub projection: u64,
    pub bias: u64,
    pub norm: u64,
    pub blocks: [PcmBlock; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PcmSpec {
    pub abi_version: u32,
    pub max_samples: u32,
    pub output_width: u32,
    pub feature_norm_eps: f32,
    pub weight_bytes: u64,
    pub stages: [PcmStage; 4],
    pub norm: u64,
    pub projection: u64,
    pub bias: u64,
    pub post_norm: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PcmLedger {
    pub weights: u64,
    pub scratch: u64,
    pub device_allocations: u64,
    pub encodes: u64,
}
type Required = unsafe extern "C" fn(*const PcmSpec, *mut PcmLedger) -> i32;
type Create = unsafe extern "C" fn(*const PcmSpec, i32, u64, *const super::media_backings::RawTags, *mut *mut c_void) -> i32;
type Upload = unsafe extern "C" fn(*mut c_void, u64, *const u8, u64) -> i32;
type Observer = unsafe extern "C" fn(*mut c_void, i32, u32, *const c_void, u32, u32) -> i32;
type Encode = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    u32,
    *mut u16,
    u64,
    Option<Observer>,
    *mut c_void,
) -> i32;
type Ledger = unsafe extern "C" fn(*mut c_void, *mut PcmLedger) -> i32;
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
pub struct NativePcmAudio {
    _library: Arc<Library>,
    owner: *mut c_void,
    encode: Encode,
    ledger: Ledger,
    backings: super::media_backings::MediaBackings,
    max_samples: usize,
    width: usize,
    _thread: PhantomData<Rc<()>>,
}
impl NativePcmAudio {
    /// Query the native ledger without allocating an owner or touching CUDA.
    pub fn required(path: &Path, spec: &PcmSpec) -> Result<PcmLedger, AudioError> {
        // SAFETY: the library stays live through the versioned ABI call and the
        // output ledger is writable for its exact C layout.
        unsafe {
            let library = Arc::new(Library::new(path)?);
            crate::memory_ledger::fault::install(&library);
            let required: Required = *library.get(b"cuteafd_pcm_required\0")?;
            let mut ledger = PcmLedger::default();
            check(required(spec, &mut ledger))?;
            Ok(ledger)
        }
    }
    pub fn load(
        path: &Path,
        spec: &PcmSpec,
        device: i32,
        admitted: u64,
        weights: &[u8],
        tags: crate::memory_ledger::MediaTags,
    ) -> Result<Self, AudioError> {
        Self::load_inner(path, spec, device, admitted, Some(weights), tags)
    }
    /// `tags` name the tower's weight arena and scratch arena in the ledger.
    pub fn prepare(path: &Path, spec: &PcmSpec, device: i32, admitted: u64, tags: crate::memory_ledger::MediaTags)
        -> Result<Self, AudioError> {
        Self::load_inner(path, spec, device, admitted, None, tags)
    }
    pub fn weight_arena(&self) -> Result<usize, AudioError> {
        // SAFETY: exact header ABI, live owner; caller retains it through external writes.
        unsafe {
            let get = self._library.get::<unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32>(b"cuteafd_pcm_weight_arena")?;
            let mut arena = std::ptr::null_mut();
            check(get(self.owner, &mut arena))?;
            Ok(arena as usize)
        }
    }
    /// Call only after the existing push group drains all external writes.
    pub fn commit_weights(&mut self) -> Result<(), AudioError> {
        // SAFETY: exact header ABI and live thread-confined owner.
        unsafe { check(self._library.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cuteafd_pcm_commit_weights")?(self.owner)) }
    }
    fn load_inner(path: &Path, spec: &PcmSpec, device: i32, admitted: u64, weights: Option<&[u8]>,
        tags: crate::memory_ledger::MediaTags) -> Result<Self, AudioError> {
        if spec.abi_version != 2 || !spec.feature_norm_eps.is_finite() || spec.feature_norm_eps <= 0.0 {
            return Err(AudioError::InvalidInput("PCM ABI 2 and positive finite feature epsilon required"));
        }
        if weights.is_some_and(|w| w.len() as u64 != spec.weight_bytes) {
            return Err(AudioError::InvalidInput("PCM weight arena length"));
        }
        // SAFETY: symbols have the versioned C header ABI, and Library is held
        // until the owner has drained and destroyed all of its CUDA work.
        unsafe {
            let library = Arc::new(Library::new(path)?);
            crate::memory_ledger::fault::install(&library);
            let required: Required = *library.get(b"cuteafd_pcm_required\0")?;
            let create: Create = *library.get(b"cuteafd_pcm_create\0")?;
            let upload: Upload = *library.get(b"cuteafd_pcm_upload\0")?;
            let encode: Encode = *library.get(b"cuteafd_pcm_encode\0")?;
            let ledger: Ledger = *library.get(b"cuteafd_pcm_get_ledger\0")?;
            let mut charge = PcmLedger::default();
            check(required(spec, &mut charge))?;
            if charge
                .weights
                .checked_add(charge.scratch)
                .is_none_or(|n| n > admitted)
            {
                return Err(AudioError::InvalidInput(
                    "PCM admission below native requirement",
                ));
            }
            let mut owner = std::ptr::null_mut();
            let native = super::media_backings::NativeTags::new(tags).map_err(AudioError::Native)?;
            let status = create(spec, device, admitted, &native.raw(), &mut owner);
            // From here Drop destroys a non-null owner and retires what native frees.
            let mut result = Self {
                _library: library,
                owner,
                encode,
                ledger,
                backings: Default::default(),
                max_samples: spec.max_samples as usize,
                width: spec.output_width as usize,
                _thread: PhantomData,
            };
            // Each backing native still holds is recorded, success or not.
            match super::media_backings::MediaBackings::created(&result._library, "cuteafd_pcm", status,
                result.owner, device, tags) {
                Ok(backings) => result.backings = backings,
                Err((status, backings)) => { result.backings = backings; return Err(AudioError::Native(status)); }
            }
            if let Some(weights) = weights {
                check(upload(result.owner, 0, weights.as_ptr(), weights.len() as u64))?;
            }
            Ok(result)
        }
    }
    pub fn ledger(&self) -> Result<PcmLedger, AudioError> {
        let mut ledger = PcmLedger::default();
        // SAFETY: live thread-confined owner and writable output of exact ABI size.
        unsafe {
            check((self.ledger)(self.owner, &mut ledger))?;
        }
        Ok(ledger)
    }
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u16>, AudioError> {
        if pcm.is_empty() || pcm.len() > self.max_samples || pcm.iter().any(|v| !v.is_finite()) {
            return Err(AudioError::InvalidInput(
                "PCM must be finite, nonempty and admitted",
            ));
        }
        let tokens = pcm.len().div_ceil(1920);
        let mut padded = vec![0u16; tokens * 1920];
        for (out, &value) in padded.iter_mut().zip(pcm) {
            // Round-to-nearest-even exactly once, before the first patch linear.
            let bits = value.to_bits();
            *out = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
        }
        let mut output = vec![0; tokens * self.width];
        // SAFETY: buffers cover rounded input/output extents; native drains its
        // stream before returning, including all error paths. No callback retained.
        unsafe {
            check((self.encode)(
                self.owner,
                padded.as_ptr(),
                pcm.len() as u32,
                output.as_mut_ptr(),
                (output.len() * 2) as u64,
                None,
                std::ptr::null_mut(),
            ))?;
        }
        Ok(output)
    }
}
impl Drop for NativePcmAudio {
    fn drop(&mut self) {
        // SAFETY: owner is uniquely held on its creation thread; native drains
        // before releasing modules/storage, then Library drops after this body.
        // Only backings native reports as freed leave the ledger.
        if !self.owner.is_null() {
            let status = unsafe { self.backings.destroy(&self._library, "cuteafd_pcm", self.owner) };
            if status != 0 { tracing::warn!(status, "PCM owner destruction reported an error"); }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_old_abi_and_invalid_feature_epsilon_before_loading_library() {
        for (version, epsilon) in [(1, 1e-5), (3, 1e-5), (2, 0.0),
            (2, f32::NAN), (2, f32::INFINITY), (2, -1.0)] {
            let spec = PcmSpec { abi_version: version, feature_norm_eps: epsilon, ..Default::default() };
            assert!(matches!(NativePcmAudio::load(Path::new("missing-library"), &spec, 0, 0, &[], crate::memory_ledger::MediaTags::new("test/tower")),
                Err(AudioError::InvalidInput("PCM ABI 2 and positive finite feature epsilon required"))));
        }
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn required_queries_the_library_without_creating_a_cuda_owner() {
        let root = std::path::PathBuf::from(std::env::var_os("TMPDIR").unwrap_or_else(||
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                .join(".cache/cuteafd/builds/pcm-required/tmp").into_os_string()))
            .join(format!("pcm-required-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("mock.c");
        let library = root.join("mock.so");
        // Only the query symbol is provided: owner creation cannot be invoked.
        std::fs::write(&source, r#"
#include <stdint.h>
struct Ledger { uint64_t weights, scratch, device_allocations, encodes; };
int cuteafd_pcm_required(const void *spec, struct Ledger *out) {
    if (!spec) return 1;
    *out = (struct Ledger){4096, 8192, 2, 0};
    return 0;
}
"#).unwrap();
        let output = std::process::Command::new("cc").args(["-shared", "-fPIC"])
            .arg(&source).arg("-o").arg(&library).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let ledger = NativePcmAudio::required(&library, &PcmSpec::default()).unwrap();
        assert_eq!(ledger, PcmLedger { weights:4096, scratch:8192, device_allocations:2, encodes:0 });
        assert!(NativePcmAudio::required(&root.join("missing.so"), &PcmSpec::default()).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn c_abi_sizes() {
        assert_eq!(std::mem::size_of::<PcmBlock>(), 56);
        assert_eq!(std::mem::size_of::<PcmStage>(), 136);
        assert_eq!(std::mem::size_of::<PcmSpec>(), 600);
        assert_eq!(std::mem::offset_of!(PcmSpec, feature_norm_eps), 12);
        assert_eq!(std::mem::offset_of!(PcmSpec, weight_bytes), 16);
        assert_eq!(std::mem::size_of::<PcmLedger>(), 32);
    }
}
