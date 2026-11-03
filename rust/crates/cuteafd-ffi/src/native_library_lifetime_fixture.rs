//! Explicit-slot CPU fake-DSO support; never enabled in production builds.
use crate::*;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicUsize;

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyKind {
    Initialize = 0,
    SmLinear = 1,
    SmRows = 2,
    DmaLinear = 3,
    DmaRows = 4,
}

/// Submission metadata resolved from actual fake-device allocation owners.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyRecord {
    pub kind: CopyKind,
    pub device: i32,
    pub destination_device: i32,
    pub source_device: i32,
    pub destination: u64,
    pub source: u64,
    pub destination_backing: u64,
    pub source_backing: u64,
    pub destination_offset: u64,
    pub source_offset: u64,
    pub bytes: u64,
    pub width: u64,
    pub rows: u64,
    pub destination_pitch: u64,
    pub source_pitch: u64,
    pub stream: u64,
}

#[repr(C)]
#[derive(Default)]
struct RawCopyRecord {
    kind: u32,
    device: i32,
    destination_device: i32,
    source_device: i32,
    geometry: [u64; 12],
}

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
            .args(["-c", "-fPIC", "-O0", "-Wall", "-Wextra", "-Werror", "-pthread"])
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
            .args(["-shared", "-fPIC", "-pthread"])
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

    /// Opt-in allocation-only target program metadata; launches still refuse.
    pub fn glm_target_programs(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this export belongs only to the CPU fixture built above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn()> =
                library.lib.get(b"fixture_configure_glm_target_programs")?;
            configure();
        }
        Ok(())
    }

    /// Opt-in EXLR8 owner table; launches remain unsupported, metadata is manifest-validated.
    pub fn exlr8_programs(&self, library: &NativeLibrary, flash: bool) -> Result<()> {
        // SAFETY: this export only selects the CPU fixture's static program table.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(i32)> =
                library.lib.get(b"fixture_configure_exlr8_programs")?;
            configure(i32::from(flash));
        }
        Ok(())
    }

    /// Opt-in small row-FP8 vocabulary table; numerical launches remain unsupported.
    pub fn native_vocabulary_programs(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this export only selects the CPU fixture's static program table.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn()> =
                library.lib.get(b"fixture_configure_native_vocabulary_programs")?;
            configure();
        }
        Ok(())
    }

    /// Canonical projection metadata for allocation-only drafter construction; launches refuse.
    pub fn draft_projection_programs<'a>(&self, library: &'a NativeLibrary,
        shapes: &[(usize, usize)], rows: usize) -> Result<crate::programs::Programs<'a>> {
        let shapes = shapes.iter().copied().collect::<std::collections::BTreeSet<_>>();
        anyhow::ensure!(!shapes.is_empty() && shapes.len() <= 16, "fixture projection table extent");
        let words = shapes.iter().map(|&(k, n)| Ok([u32::try_from(k)?, u32::try_from(n)?]))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: the fixture copies these live contiguous pairs before returning.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(*const u32, u32) -> i32> =
                library.lib.get(b"fixture_configure_draft_projections")?;
            anyhow::ensure!(configure(words.as_ptr().cast(), words.len() as u32) == 0,
                "configuring CPU draft projection table failed");
        }
        let entries = shapes.iter().map(|&(k, n)| serde_json::json!({
            "name":format!("draft_bf16_k{k}_n{n}"), "capacity_rows":rows,
            "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
            "scratch_bytes_at_capacity":{}, "geometry":{"width":k,"outputs":n,
                "output_dtype":"torch.bfloat16","reduction":"ordered_k16",
                "tiles_by_rows":[[16,1,4,16,64,6]]}})).collect::<Vec<_>>();
        let manifest = self.directory().join("draft-projections.json");
        fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        library.programs()?.with_manifest(&manifest)
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

    pub fn configure_exl3_capacity(&self, library: &NativeLibrary, capacity: u32) -> Result<()> {
        // SAFETY: CPU-only ABI selection; already loaded modules retain their own metadata.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn(u32)> =
                library.lib.get(b"fixture_configure_exl3_capacity")?;
            configure(capacity);
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

    /// Enable one allocation-aware trace for both SM and DMA submissions.
    /// Enabling resets only the trace, not queued operations or their lifetimes.
    pub fn record_peer_copies(&self, library: &NativeLibrary) -> Result<()> {
        // SAFETY: this control export belongs only to the CPU fixture built above.
        unsafe {
            let configure: Symbol<unsafe extern "C" fn()> = library.lib.get(b"fixture_record_peer_copies")?;
            configure();
        }
        Ok(())
    }

    pub fn peer_copy_records(&self, library: &NativeLibrary) -> Result<Vec<CopyRecord>> {
        // SAFETY: both exports belong to this fixture; readback copies one repr(C)
        // record synchronously. Call after submission threads have joined.
        unsafe {
            let count: Symbol<unsafe extern "C" fn() -> usize> =
                library.lib.get(b"fixture_peer_copy_record_count")?;
            let get: Symbol<unsafe extern "C" fn(usize, *mut RawCopyRecord) -> i32> =
                library.lib.get(b"fixture_peer_copy_record")?;
            (0..count()).map(|index| {
                let mut raw = RawCopyRecord::default();
                anyhow::ensure!(get(index, &mut raw) == 0, "fixture copy record {index} is missing");
                let kind = match raw.kind {
                    0 => CopyKind::Initialize,
                    1 => CopyKind::SmLinear,
                    2 => CopyKind::SmRows,
                    3 => CopyKind::DmaLinear,
                    4 => CopyKind::DmaRows,
                    other => anyhow::bail!("unknown fixture copy kind {other}"),
                };
                let [destination, source, destination_backing, source_backing,
                    destination_offset, source_offset, bytes, width, rows,
                    destination_pitch, source_pitch, stream] = raw.geometry;
                Ok(CopyRecord { kind, device: raw.device, destination_device: raw.destination_device,
                    source_device: raw.source_device, destination, source, destination_backing,
                    source_backing, destination_offset, source_offset, bytes, width, rows,
                    destination_pitch, source_pitch, stream })
            }).collect()
        }
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
