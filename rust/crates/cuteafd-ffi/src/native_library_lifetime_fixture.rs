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

    pub fn build() -> Result<Self> {
        let root = PathBuf::from(
            std::env::var_os("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")
                .context("set CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR under ~/.cache/cuteafd/builds")?,
        );
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
        let status = Command::new("cc")
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
        let status = Command::new("c++")
            .args(["-std=c++17", "-c", "-fPIC", "-O1", "-Wall", "-Wextra", "-Werror"])
            .arg(fixtures.join("native_library_kernels.cc"))
            .arg("-o")
            .arg(&cc_object)
            .env("TMPDIR", &directory)
            .status()?;
        anyhow::ensure!(status.success(), "building CPU fixture kernels failed");
        let status = Command::new("c++")
            .args(["-shared", "-fPIC"])
            .arg(&c_object)
            .arg(&cc_object)
            .arg("-o")
            .arg(&module)
            .env("TMPDIR", &directory)
            .status()?;
        anyhow::ensure!(status.success(), "linking CPU lifetime fixture failed");
        Ok(Self {
            module,
            events: directory.join("events"),
        })
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

    pub fn resident(&self) -> Result<bool> {
        Ok(fs::read_to_string("/proc/self/maps")?
            .lines()
            .any(|line| line.ends_with(self.module.to_str().unwrap())))
    }

    pub fn events(&self) -> Result<String> {
        Ok(fs::read_to_string(&self.events)?)
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
