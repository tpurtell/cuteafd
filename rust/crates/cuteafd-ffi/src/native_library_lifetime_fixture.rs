//! Explicit-slot CPU fake-DSO support; never enabled in production builds.
use crate::*;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicUsize;

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

pub struct Fixture {
    module: PathBuf,
    events: PathBuf,
}

impl Fixture {
    pub fn directory(&self) -> &std::path::Path {
        self.module.parent().expect("fixture module has a directory")
    }

    /// Hermetic CPU fixture beside this test executable, inside its asserted
    /// NVMe Cargo target. No task-specific path or shared mutable DSO is used.
    pub fn build_for_tests() -> Result<Self> {
        let executable = std::env::current_exe()?;
        let root = executable.parent().context("test executable has no parent")?
            .join("native-fixtures");
        Self::build_in(&root)
    }

    pub fn build() -> Result<Self> {
        let root = PathBuf::from(
            std::env::var_os("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")
                .context("set CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR under ~/.cache/cuteafd/builds")?,
        );
        Self::build_at(root, false)
    }

    /// Build a test-owned fixture, taking a shared build slot only for compilation.
    pub fn build_in(root: &std::path::Path) -> Result<Self> {
        Self::build_at(root.to_path_buf(), true)
    }

    fn build_at(root: PathBuf, take_build_slot: bool) -> Result<Self> {
        let allowed = PathBuf::from(std::env::var_os("HOME").context("HOME is unset")?)
            .join(".cache/cuteafd/builds")
            .canonicalize()?;
        anyhow::ensure!(
            root.is_absolute()
                && root.starts_with(&allowed)
                && !root
                    .components()
                    .any(|part| part == std::path::Component::ParentDir),
            "fixture directory must be under {}",
            allowed.display()
        );
        let existing = root
            .ancestors()
            .find(|path| path.exists())
            .context("fixture directory has no existing parent")?;
        anyhow::ensure!(
            existing.canonicalize()?.starts_with(&allowed),
            "fixture directory's existing parent escapes NVMe build root"
        );
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let assert_path = repo.join("scripts/build/assert-build-filesystem.py");
        let status = Command::new("python3")
            .arg(&assert_path)
            .arg(&root)
            .status()?;
        anyhow::ensure!(status.success(), "fixture build filesystem check failed");
        fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        anyhow::ensure!(
            root.starts_with(&allowed),
            "fixture directory escapes NVMe build root"
        );
        // Container PID namespaces can reuse the same PID on the next run.
        // Keep all prior evidence and claim a fresh directory atomically.
        let directory = loop {
            let directory = root.join(format!(
                "lifetime-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&directory) {
                Ok(()) => break directory,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        };
        let module = directory.join("liblifetime_fixture.so");
        let status = Command::new("python3")
            .arg(assert_path)
            .arg(&module)
            .status()?;
        anyhow::ensure!(status.success(), "fixture module filesystem check failed");
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let (c_object, cc_object) = (directory.join("fixture.o"), directory.join("kernels.o"));
        let status = Command::new("python3")
            .arg(repo.join("python/tools/lib/mxfp4_layouts.py"))
            .arg("--header")
            .arg(directory.join("cuteafd_mxfp4_layouts.h"))
            .status()?;
        anyhow::ensure!(status.success(), "generating CPU fixture MXFP4 layouts failed");
        let compiler = |program: &str| {
            if take_build_slot {
                let mut command = Command::new(repo.join("scripts/build/build-slot.sh"));
                command.arg(program);
                command
            } else {
                Command::new(program)
            }
        };
        let status = compiler("cc")
            .args(["-c", "-fPIC", "-O0", "-Wall", "-Wextra", "-Werror"])
            .arg(fixtures.join("native_library_lifetime.c"))
            .arg("-I")
            .arg(&directory)
            .arg("-o")
            .arg(&c_object)
            .env("TMPDIR", &directory)
            .status()?;
        anyhow::ensure!(status.success(), "building CPU lifetime fixture failed");
        // Load-only kernels run the shared CUDA rule on the CPU.
        let status = compiler("c++")
            .args(["-std=c++17", "-c", "-fPIC", "-O1", "-Wall", "-Wextra", "-Werror"])
            .arg(fixtures.join("native_library_kernels.cc"))
            .arg("-o")
            .arg(&cc_object)
            .env("TMPDIR", &directory)
            .status()?;
        anyhow::ensure!(status.success(), "building CPU fixture kernels failed");
        let status = compiler("c++")
            .args(["-shared", "-fPIC"])
            .arg(&c_object)
            .arg(&cc_object)
            .arg("-o")
            .arg(&module)
            .env("TMPDIR", &directory)
            .status()?;
        anyhow::ensure!(status.success(), "linking CPU lifetime fixture failed");
        let events = directory.join("events");
        fs::write(&events, [])?;
        Ok(Self { module, events })
    }

    /// Configure static package ABI before loading a CPU expert module.
    pub fn configure_fp8(&self, library: &NativeLibrary, words: &[u32; 16], scratch: &[u64; 6]) -> Result<()> {
        // SAFETY: the CPU-only export copies the two fixed-size arrays synchronously.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(*const u32, *const u64)> =
                library.lib.get(b"fixture_configure_fp8")?;
            configure(words.as_ptr(), scratch.as_ptr());
        }
        Ok(())
    }

    pub fn configure_fp8_segments(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this CPU-only fixture export toggles its static segment metadata.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn()> = library.lib.get(b"fixture_configure_fp8_segments")?;
            configure();
        }
        Ok(())
    }

    pub fn load(&self) -> Result<NativeLibrary> {
        // SAFETY: this test just compiled the fixture with the native ABI.
        let library = unsafe { NativeLibrary::load(&self.module)? };
        let path = CString::new(self.events.as_os_str().as_bytes())?;
        // SAFETY: the fixture export copies this valid C string synchronously.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(*const c_char)> =
                library.lib.get(b"fixture_set_event_path")?;
            configure(path.as_ptr());
        }
        Ok(library)
    }

    pub fn quiet_events(&self, library: &NativeLibrary, quiet: bool) -> Result<()> {
        // SAFETY: this export belongs to the CPU-only fixture loaded above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32)> =
                library.lib.get(b"fixture_quiet_events")?;
            configure(i32::from(quiet));
        }
        Ok(())
    }

    /// Opt-in current-device checking; a rejected free leaves its allocation live.
    pub fn strict_device_free(&self, library: &NativeLibrary, enabled: bool) -> Result<()> {
        // SAFETY: the private fixture export updates only its device-free controls.
        unsafe { library.lib.get::<unsafe extern "C" fn(i32)>(b"fixture_strict_device_free")?(i32::from(enabled)); }
        Ok(())
    }

    /// Number of rejected frees and the last (owning device, current device).
    pub fn device_free_mismatch(&self, library: &NativeLibrary) -> Result<(usize, i32, i32)> {
        let (mut owner, mut current) = (-1, -1);
        // SAFETY: the private export writes two live i32 slots synchronously.
        let count = unsafe {
            library.lib.get::<unsafe extern "C" fn(*mut i32, *mut i32) -> usize>(b"fixture_device_free_mismatch")?(
                &mut owner, &mut current)
        };
        Ok((count, owner, current))
    }

    pub fn block_first_stream(&self, library: &NativeLibrary, block: bool) -> Result<()> {
        // SAFETY: the CPU fixture owns this mutex-protected injection control.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32)> =
                library.lib.get(b"fixture_block_first_stream")?;
            configure(i32::from(block));
        }
        Ok(())
    }

    pub fn resident(&self) -> Result<bool> {
        Ok(fs::read_to_string("/proc/self/maps")?
            .lines()
            .any(|line| line.ends_with(self.module.to_str().unwrap())))
    }

    pub fn events(&self) -> Result<String> {
        Ok(fs::read_to_string(&self.events)?)
    }

    /// Reserve untouched virtual device extents per owner and skip copies/zeroes.
    /// Only allocation/receipt tests may enable this; launching math is unsupported.
    pub fn extent_only(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this export belongs to the private CPU fixture and has no arguments.
        unsafe { library.lib.get::<unsafe extern "C" fn()>(b"fixture_extent_only")?(); }
        Ok(())
    }

    /// Allow pair construction while preserving real byte copies; peer launches still fail.
    pub fn peer_abort_initialization(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this private fixture export only enables the no-argument initializer.
        unsafe { library.lib.get::<unsafe extern "C" fn()>(b"fixture_peer_abort_initialization")?(); }
        Ok(())
    }

    pub fn mapped_bytes(&self, library: &NativeLibrary) -> Result<usize> {
        // SAFETY: this private fixture export reads its allocation counter.
        Ok(unsafe { library.lib.get::<unsafe extern "C" fn() -> usize>(b"fixture_mapped_bytes")?() })
    }

    pub fn module_path(&self) -> &std::path::Path { &self.module }

    /// Register program ABI metadata; launches fail unless a reply behavior is registered.
    pub fn register_programs(&self, library: &NativeLibrary, entries: &[serde_json::Value]) -> Result<()> {
        for (index, entry) in entries.iter().enumerate() {
            let name = CString::new(entry["name"].as_str().context("fixture program name missing")?)?;
            let pointers = u32::try_from(entry["pointers"].as_array().context("fixture pointers missing")?.len())?;
            let kinds = entry["scalars"].as_array().context("fixture scalars missing")?.iter().map(|scalar| {
                match scalar["type"].as_str() {
                    Some("int32") => Ok(b'i'), Some("int64") => Ok(b'l'), Some("float32") => Ok(b'f'),
                    _ => anyhow::bail!("fixture program scalar type missing"),
                }
            }).collect::<Result<Vec<_>>>()?;
            // SAFETY: fixture copies the bounded name/scalar metadata before returning.
            let status = unsafe {
                library.lib.get::<unsafe extern "C" fn(u32, *const c_char, u32, *const u8, u32) -> i32>(
                    b"fixture_register_program")?(u32::try_from(index)?, name.as_ptr(), pointers,
                        kinds.as_ptr(), u32::try_from(kinds.len())?)
            };
            anyhow::ensure!(status == 0, "fixture rejected installed program {index}");
        }
        Ok(())
    }

    /// Attach an explicit readiness check and dispatch marker to a registered reply program.
    pub fn register_reply(&self, library: &NativeLibrary, index: u32, ready: u32, event: u8) -> Result<()> {
        // SAFETY: the fixture validates the index and copies these scalar values.
        let status = unsafe {
            library.lib.get::<unsafe extern "C" fn(u32, u32, c_char) -> i32>(
                b"fixture_register_reply")?(index, ready, event as c_char)
        };
        anyhow::ensure!(status == 0, "fixture rejected reply program {index}");
        Ok(())
    }

    pub fn configure_exl3(&self, library: &NativeLibrary, words: &[u32; 16]) -> Result<()> {
        // SAFETY: CPU fixture copies this complete fixed-size ABI record synchronously.
        unsafe { library.lib.get::<unsafe extern "C" fn(*const u32)>(b"fixture_configure_exl3")?(words.as_ptr()); }
        Ok(())
    }

    /// Admit a BF16 projection in the allocation-only catalog; launches still fail.
    pub fn draft_projection(&self, library: &NativeLibrary, index: u32, k: u32, n: u32) -> Result<()> {
        // SAFETY: the private fixture validates the index and copies only scalar metadata.
        let status = unsafe {
            library.lib.get::<unsafe extern "C" fn(u32, u32, u32) -> i32>(b"fixture_draft_projection")?(index, k, n)
        };
        anyhow::ensure!(status == 0, "fixture draft projection {index} rejected");
        Ok(())
    }

    /// CPU byte-roundtrip mode; ordinary lifetime fixtures retain deferred uploads.
    pub fn immediate_copies(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this export belongs only to the CPU fixture loaded above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn()> =
                library.lib.get(b"fixture_configure_immediate_copies")?;
            configure();
        }
        Ok(())
    }

    /// Record ready checks only for tests asserting host-wait placement.
    pub fn trace_stream_queries(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this export toggles recording in the CPU-only fixture above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn()> = library.lib.get(b"fixture_trace_stream_queries")?;
            configure();
        }
        Ok(())
    }

    /// Queue copies and deterministic route-dependent fake expert reads, rather
    /// than executing either at submission. Only explicit advance/drain runs work.
    pub fn queued_experts(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: these exports belong only to the CPU fixture built above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn()> = library.lib.get(b"fixture_configure_queued_experts")?;
            configure();
        }
        Ok(())
    }

    pub fn advance_queued_experts(&self, library: &NativeLibrary, operations: usize) -> Result<()> {
        // SAFETY: the fixture executes at most this many previously queued operations.
        unsafe {
            let advance: Symbol<unsafe extern "C" fn(usize)> = library.lib.get(b"fixture_advance_queued_experts")?;
            advance(operations);
        }
        Ok(())
    }

    pub fn configure_memory(&self, library: &NativeLibrary, device: i32, total: usize) -> Result<()> {
        // SAFETY: this export belongs to the CPU-only fixture loaded above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32, usize, usize, usize)> =
                library.lib.get(b"fixture_configure_memory")?;
            configure(device, total, 0, 0);
        }
        Ok(())
    }

    pub fn configure_endpoint_failures(&self, library: &NativeLibrary, view: i32, destroy: i32) -> Result<()> {
        // SAFETY: this export belongs to the CPU-only fixture loaded above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32, i32)> =
                library.lib.get(b"fixture_configure_endpoint_failures")?;
            configure(view, destroy);
        }
        Ok(())
    }

    pub fn configure_capture(&self, library: &NativeLibrary, status: i32) -> Result<()> {
        // SAFETY: the CPU-only fixture export only updates its local injected status.
        unsafe { library.lib.get::<unsafe extern "C" fn(i32)>(b"fixture_configure_capture")?(status); }
        Ok(())
    }

    pub fn configure_pack(&self, library: &NativeLibrary, pack: i32, drain: i32) -> Result<()> {
        // SAFETY: this export belongs to the CPU-only fixture loaded above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32, i32)> =
                library.lib.get(b"fixture_configure_pack")?;
            configure(pack, drain);
        }
        Ok(())
    }

    pub fn stage(&self, library: &NativeLibrary) -> Result<()> {
        library
            .sync_h2d_staging
            .lock()
            .unwrap()
            .ensure(library, 64)?;
        Ok(())
    }

    pub fn reserve_ordered_upload(&self, library: &NativeLibrary, dst: CuteafdDeviceBuffer,
        stream: usize) -> Result<()> {
        // SAFETY: the CPU fixture owns this destination and accepts a synthetic
        // stream; reservation records no pending copy or consumer.
        unsafe { library.reserve_h2d_ordered(dst, stream as *mut c_void) }
    }

    pub fn finish_ordered_uploads(&self, library: &NativeLibrary) -> Result<()> {
        library.release_ordered_h2d_staging()
    }

    pub fn configure_drain_after(&self, library: &NativeLibrary, completed: i32) -> Result<()> {
        // SAFETY: this export belongs to the CPU-only fixture loaded above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32)> =
                library.lib.get(b"fixture_configure_drain_after")?;
            configure(completed);
        }
        Ok(())
    }

    pub fn configure_bind_after(&self, library: &NativeLibrary, completed: i32) -> Result<()> {
        // SAFETY: this export belongs to the CPU-only fixture loaded above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32)> =
                library.lib.get(b"fixture_configure_bind_after")?;
            configure(completed);
        }
        Ok(())
    }

    pub fn drain(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: the fixture's stream synchronizer accepts a null stream and
        // performs no CUDA work. Only use with this fixture's NativeLibrary.
        unsafe { library.cuda_stream_synchronize(std::ptr::null_mut()) }
    }
}

#[cfg(test)]
mod device_free_tests {
    use super::*;

    #[test]
    fn strict_device_free_records_mismatch_and_preserves_the_live_owner() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        // Use private DSO calls, not the process-global allocation ledger.
        // SAFETY: both exports belong to this fixture and the buffer lives until its matching free.
        let (allocate, free) = unsafe {
            (*library.lib.get::<unsafe extern "C" fn(usize, *mut CuteafdDeviceBuffer) -> i32>(b"cuteafd_alloc_device_buffer")?,
             *library.lib.get::<unsafe extern "C" fn(*mut CuteafdDeviceBuffer) -> i32>(b"cuteafd_free_device_buffer")?)
        };
        for strict in [false, true] {
            fixture.strict_device_free(&library, strict)?;
            library.cuda_set_device(1)?;
            let mut buffer = CuteafdDeviceBuffer::default();
            // SAFETY: this CPU fixture writes a fresh host-backed device buffer.
            assert_eq!(unsafe { allocate(256, &mut buffer) }, 0);
            let pointer = buffer.ptr;
            library.cuda_set_device(0)?;
            // SAFETY: the fixture either frees its own buffer or rejects without modifying it.
            let status = unsafe { free(&mut buffer) };
            if strict {
                assert_ne!(status, 0, "wrong-device free must fail");
                assert_eq!(fixture.device_free_mismatch(&library)?, (1, 1, 0));
                assert_eq!(buffer.ptr, pointer, "failed free must leave its owner live");
                library.cuda_set_device(1)?;
                // SAFETY: the rejected free retained this live allocation; the owning device is current.
                assert_eq!(unsafe { free(&mut buffer) }, 0);
            } else {
                assert_eq!(status, 0, "legacy tests keep permissive frees");
                assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
            }
            assert!(buffer.ptr.is_null());
        }
        Ok(())
    }
}
