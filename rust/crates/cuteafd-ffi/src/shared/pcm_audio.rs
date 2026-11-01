//! Shared BF16 PCM owner. Confined to the encoder CUDA thread; no unsafe Send.
use super::audio::AudioError;
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, rc::Rc};
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
    pub reserved: u32,
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
type Create = unsafe extern "C" fn(*const PcmSpec, i32, u64, *mut *mut c_void) -> i32;
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
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
fn check(code: i32) -> Result<(), AudioError> {
    if code == 0 {
        Ok(())
    } else {
        Err(AudioError::Native(code))
    }
}
pub struct NativePcmAudio {
    _library: Library,
    owner: *mut c_void,
    encode: Encode,
    ledger: Ledger,
    destroy: Destroy,
    max_samples: usize,
    width: usize,
    _thread: PhantomData<Rc<()>>,
}
impl NativePcmAudio {
    pub fn load(
        path: &Path,
        spec: &PcmSpec,
        device: i32,
        admitted: u64,
        weights: &[u8],
    ) -> Result<Self, AudioError> {
        if weights.len() as u64 != spec.weight_bytes {
            return Err(AudioError::InvalidInput("PCM weight arena length"));
        }
        // SAFETY: symbols have the versioned C header ABI, and Library is held
        // until the owner has drained and destroyed all of its CUDA work.
        unsafe {
            let library = Library::new(path)?;
            let required: Required = *library.get(b"cuteafd_pcm_required\0")?;
            let create: Create = *library.get(b"cuteafd_pcm_create\0")?;
            let upload: Upload = *library.get(b"cuteafd_pcm_upload\0")?;
            let encode: Encode = *library.get(b"cuteafd_pcm_encode\0")?;
            let ledger: Ledger = *library.get(b"cuteafd_pcm_get_ledger\0")?;
            let destroy: Destroy = *library.get(b"cuteafd_pcm_destroy\0")?;
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
            check(create(spec, device, admitted, &mut owner))?;
            if let Err(error) = check(upload(owner, 0, weights.as_ptr(), weights.len() as u64)) {
                destroy(owner);
                return Err(error);
            }
            Ok(Self {
                _library: library,
                owner,
                encode,
                ledger,
                destroy,
                max_samples: spec.max_samples as usize,
                width: spec.output_width as usize,
                _thread: PhantomData,
            })
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
        unsafe {
            (self.destroy)(self.owner);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn c_abi_sizes() {
        assert_eq!(std::mem::size_of::<PcmBlock>(), 56);
        assert_eq!(std::mem::size_of::<PcmStage>(), 136);
        assert_eq!(std::mem::size_of::<PcmSpec>(), 600);
        assert_eq!(std::mem::size_of::<PcmLedger>(), 32);
    }
}
