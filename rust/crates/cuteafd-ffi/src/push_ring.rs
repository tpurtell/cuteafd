//! Registered shared mappings for push delivery. No device bytes live here.
use crate::{CuteafdHostBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use libloading::Symbol;
use std::{
    ffi::{c_void, CString},
    os::fd::{AsFd, BorrowedFd, FromRawFd, OwnedFd},
};

pub const DEFAULT_SLOT_BYTES: usize = 4 * 1024 * 1024;
pub const DEFAULT_SLOTS: usize = 32;

pub struct RegisteredRing<'a> {
    library: &'a NativeLibrary,
    fd: OwnedFd,
    ptr: *mut c_void,
    bytes: usize,
    slots: usize,
    slot_bytes: usize,
    device: i32,
    registered: bool,
}
impl<'a> RegisteredRing<'a> {
    pub fn new(library: &'a NativeLibrary, slots: usize, slot_bytes: usize) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&slots)
                && slot_bytes >= 4096
                && slot_bytes <= DEFAULT_SLOT_BYTES
                && slot_bytes % 4096 == 0,
            "invalid RDMA push ring geometry"
        );
        let bytes = slots
            .checked_mul(slot_bytes)
            .context("push ring size overflow")?;
        let name = CString::new("weight-push")?;
        // SAFETY: valid NUL-terminated name; ownership of the new fd is transferred once.
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        ensure!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        // SAFETY: successful memfd_create returned a newly owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        use std::os::fd::AsRawFd;
        // SAFETY: owned fd, bounded nonzero extent; sealing prevents remote resize.
        ensure!(
            unsafe { libc::ftruncate(fd.as_raw_fd(), bytes as libc::off_t) } == 0,
            "ring ftruncate failed"
        );
        // SAFETY: sealing resize does not prevent shared payload writes.
        ensure!(
            unsafe {
                libc::fcntl(
                    fd.as_raw_fd(),
                    libc::F_ADD_SEALS,
                    libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL,
                )
            } == 0,
            "ring sealing failed"
        );
        // SAFETY: file has the requested extent and remains owned through unmapping.
        let device = library.cuda_get_device()?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        ensure!(ptr != libc::MAP_FAILED, "ring mmap failed");
        let mut ring = Self {
            library,
            fd,
            ptr,
            bytes,
            slots,
            slot_bytes,
            device,
            registered: false,
        };
        type Register = unsafe extern "C" fn(*mut c_void, usize) -> i32;
        // SAFETY: native registration accepts this live shared mapping.
        let status = unsafe {
            let register: Symbol<Register> = library.lib.get(b"cuteafd_register_host_mapping")?;
            register(ptr, bytes)
        };
        library.status_to_result("cuteafd_register_host_mapping", status)?;
        ring.registered = true;
        Ok(ring)
    }
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
    pub fn slots(&self) -> usize {
        self.slots
    }
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    pub fn belongs_to(&self, library: &NativeLibrary, device: i32) -> bool {
        std::ptr::eq(self.library, library) && self.device == device
    }
    /// Read a slot whose Filled credit the caller owns.
    ///
    /// # Safety
    /// The producer must not write this slot until all references and queued
    /// copies end. The caller must retain its credit until then.
    pub unsafe fn slot(&self, index: u32, len: u32) -> Result<&[u8]> {
        ensure!(
            (index as usize) < self.slots && len as usize == self.slot_bytes,
            "invalid Filled slot extent"
        );
        // SAFETY: checked slot is inside the mapped extent; Filled transfers write ownership.
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.ptr.cast::<u8>().add(index as usize * self.slot_bytes),
                len as usize,
            )
        })
    }
    pub fn payload(&self, index: u32, offset: usize, bytes: usize) -> Result<CuteafdHostBuffer> {
        ensure!(
            (index as usize) < self.slots
                && offset
                    .checked_add(bytes)
                    .is_some_and(|n| n <= self.slot_bytes),
            "ring payload outside slot"
        );
        // SAFETY: checked range is inside this registered mapping retained by the consumer.
        Ok(CuteafdHostBuffer {
            ptr: unsafe {
                self.ptr
                    .cast::<u8>()
                    .add(index as usize * self.slot_bytes + offset)
                    .cast()
            },
            bytes,
            flags: 1,
        })
    }
}
impl Drop for RegisteredRing<'_> {
    fn drop(&mut self) {
        if self.library.is_quarantined_after_failed_drain() {
            return;
        }
        if self.registered {
            let result = (|| -> Result<()> {
                type Unregister = unsafe extern "C" fn(*mut c_void) -> i32;
                // Resolve before changing devices, including the missing-symbol path.
                // SAFETY: the library exposes this native ABI signature.
                let unregister: Symbol<Unregister> =
                    unsafe { self.library.lib.get(b"cuteafd_unregister_host_mapping")? };
                let previous = self.library.cuda_get_device()?;
                self.library.cuda_set_device(self.device)?;
                // SAFETY: all queued copies must have drained before the ring is dropped.
                let status = unsafe { unregister(self.ptr) };
                let result = self
                    .library
                    .status_to_result("cuteafd_unregister_host_mapping", status);
                self.library.cuda_set_device(previous)?;
                result
            })();
            if let Err(error) = result {
                tracing::error!(%error, "retaining registered push mapping");
                return;
            }
        }
        // SAFETY: unique mapping owner, no registration or queued accesses remain.
        unsafe {
            libc::munmap(self.ptr, self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_library_lifetime_fixture::Fixture;
    use std::os::fd::AsRawFd;

    #[test]
    fn default_geometry_keeps_128_mib_with_uniform_4_mib_slots() {
        assert_eq!(DEFAULT_SLOT_BYTES, 4 * 1024 * 1024);
        assert_eq!(DEFAULT_SLOTS, 32);
        assert_eq!(DEFAULT_SLOT_BYTES * DEFAULT_SLOTS, 128 * 1024 * 1024);
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn mapping_seals_bounds_and_device_restoration() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        for (slots, bytes) in [
            (0, 4096),
            (4097, 4096),
            (1, 4095),
            (1, 8193),
            (1, 2 * DEFAULT_SLOT_BYTES),
        ] {
            assert!(RegisteredRing::new(&library, slots, bytes).is_err());
        }
        library.cuda_set_device(1)?;
        let ring = RegisteredRing::new(&library, 2, 4096)?;
        assert!(ring.belongs_to(&library, 1));
        assert!(!ring.belongs_to(&library, 0));
        // SAFETY: valid descriptor, querying its immutable resize seals.
        let seals = unsafe { libc::fcntl(ring.fd().as_raw_fd(), libc::F_GET_SEALS) };
        assert_eq!(
            seals & (libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL),
            libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL
        );
        // SAFETY: no producer has this fd; both slots remain exclusively owned here.
        unsafe {
            assert_eq!(ring.slot(0, 4096)?.len(), 4096);
            assert!(ring.slot(2, 4096).is_err());
            assert!(ring.slot(0, 4095).is_err());
        }
        assert!(ring.payload(0, 4095, 2).is_err());
        library.cuda_set_device(0)?;
        drop(ring);
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?, "Gg");
        Ok(())
    }
}
