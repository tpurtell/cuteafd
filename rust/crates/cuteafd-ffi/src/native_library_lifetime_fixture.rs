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
        let status = Command::new("cc")
            .args(["-shared", "-fPIC", "-O0", "-Wall", "-Wextra", "-Werror"])
            .arg(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/native_library_lifetime.c"),
            )
            .arg("-o")
            .arg(&module)
            .env("TMPDIR", &directory)
            .status()?;
        anyhow::ensure!(status.success(), "building CPU lifetime fixture failed");
        Ok(Self {
            module,
            events: directory.join("events"),
        })
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

    pub fn drain(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: the fixture's stream synchronizer accepts a null stream and
        // performs no CUDA work. Only use with this fixture's NativeLibrary.
        unsafe { library.cuda_stream_synchronize(std::ptr::null_mut()) }
    }
}
