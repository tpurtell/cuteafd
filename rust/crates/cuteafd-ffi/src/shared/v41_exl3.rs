//! Owned mixed EXL3 AOT module. Captured graphs and borrowed device buffers must
//! be released/drained before dropping this owner on its original CUDA thread.
use anyhow::{ensure, Context, Result};
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41Exl3Info {
    /// Bytes per output element: BF16 (2) or FP32 (4).
    pub output_element_bytes: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub capacity: usize,
    pub topk: usize,
    pub tier_count: usize,
    pub bits: [u32; 4],
    pub core_pointers: usize,
    pub core_scalars: usize,
    pub sum_pointers: usize,
    pub sum_scalars: usize,
}

/// Rank shard widths an EXL3 export may carry for `geometry`: whole H128
/// rotation blocks of the intermediate split over the published rank counts,
/// the first ranks owning any extra block. V4.1 (2304) includes a clean TP6
/// split of three H128 blocks per rank (384 rows), alongside TP3's six blocks.
pub fn exl3_shard_widths(geometry: cuteafd_core::ExpertGeometry) -> Vec<u32> {
    let blocks = geometry.intermediate / 128;
    let mut widths: Vec<u32> = [1u32, 2, 3, 4, 6]
        .into_iter()
        .filter(|&world| geometry.intermediate % 128 == 0 && blocks >= world)
        .flat_map(|world| [blocks / world, blocks.div_ceil(world)])
        .map(|count| count * 128)
        .collect();
    widths.sort_unstable();
    widths.dedup();
    widths
}

impl V41Exl3Info {
    fn from_words(words: [u32; 16]) -> Result<Self> {
        Self::from_words_for_geometry(words, cuteafd_core::expert_geometry())
    }

    fn from_words_for_geometry(words: [u32; 16], geometry: cuteafd_core::ExpertGeometry) -> Result<Self> {
        ensure!(
            words[0] == 2
                && matches!(words[15], 2 | 4)
                && words[1] == geometry.hidden
                && exl3_shard_widths(geometry).contains(&words[2])
                && words[3] <= 1024
                && words[3] >= words[5]
                && words[4] > 0
                && (1..=16).contains(&words[5])
                && matches!(words[6], 2..=4),
            "invalid EXL3 native geometry: {words:?}"
        );
        let tier_count = words[6] as usize;
        let bits: [u32; 4] = words[11..15].try_into().unwrap();
        ensure!(
            bits[..tier_count].iter().all(|bit| (2..=5).contains(bit))
                && bits[tier_count..].iter().all(|bit| *bit == 0)
                && bits[..tier_count]
                    .iter()
                    .enumerate()
                    .all(|(i, bit)| !bits[..i].contains(bit)),
            "invalid EXL3 native tier bits"
        );
        ensure!(
            words[7..11].iter().all(|count| *count > 0 && *count <= 64),
            "invalid EXL3 pointer/scalar ABI"
        );
        Ok(Self {
            output_element_bytes: words[15] as usize,
            hidden: words[1] as usize,
            intermediate: words[2] as usize,
            experts: words[3] as usize,
            capacity: words[4] as usize,
            topk: words[5] as usize,
            tier_count,
            bits,
            core_pointers: words[7] as usize,
            core_scalars: words[8] as usize,
            sum_pointers: words[9] as usize,
            sum_scalars: words[10] as usize,
        })
    }

}

type Launch = unsafe extern "C" fn(*mut c_void, *const *mut c_void, *const i32, *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);

pub struct V41Exl3Kernel {
    // Function addresses and context are only valid while the module is loaded.
    _library: Library,
    context: NonNull<c_void>,
    info: V41Exl3Info,
    core: Launch,
    sum: Launch,
    destroy: Destroy,
    _owner_thread: PhantomData<Rc<()>>,
}

impl V41Exl3Kernel {
    /// # Safety
    /// Load only trusted generated code, with a current CUDA device. Caller must
    /// keep that device current during launches and drop, and drain every stream
    /// and destroy graphs referencing this module before dropping it.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        let library = Library::new(path.as_ref())?;
        let query =
            *library.get::<unsafe extern "C" fn(*mut u32, u32) -> i32>(b"cuteafd_exl3_info")?;
        let create =
            *library.get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"cuteafd_exl3_create")?;
        let core = *library.get::<Launch>(b"cuteafd_exl3_core")?;
        let sum = *library.get::<Launch>(b"cuteafd_exl3_sum")?;
        let destroy = *library.get::<Destroy>(b"cuteafd_exl3_destroy")?;
        let mut words = [0; 16];
        ensure!(query(words.as_mut_ptr(), 16) == 0, "EXL3 native info query failed");
        let info = V41Exl3Info::from_words(words)?;
        let mut context = std::ptr::null_mut();
        let status = create(&mut context);
        ensure!(
            status == 0,
            "EXL3 native initialization failed with CUDA status {status}"
        );
        let context = NonNull::new(context).context("EXL3 initialized a null context")?;
        Ok(Self {
            _library: library,
            context,
            info,
            core,
            sum,
            destroy,
            _owner_thread: PhantomData,
        })
    }

    pub fn info(&self) -> V41Exl3Info {
        self.info
    }

    /// # Safety
    /// Pointer/scalar order must match the verified export manifest. Device
    /// storage must meet its sizes, types, alignment and non-aliasing contract
    /// and remain alive through stream completion or the last graph replay.
    pub unsafe fn launch_core(
        &self,
        pointers: &[*mut c_void],
        scalars: &[i32],
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            pointers.len() == self.info.core_pointers && scalars.len() == self.info.core_scalars,
            "EXL3 core argument count mismatch"
        );
        let status = (self.core)(
            self.context.as_ptr(),
            pointers.as_ptr(),
            scalars.as_ptr(),
            stream,
        );
        ensure!(status == 0, "EXL3 core failed with CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Same buffer lifetime/order requirements as `launch_core`; the epilogue
    /// must follow the producing core on an ordered stream.
    pub unsafe fn launch_sum(
        &self,
        pointers: &[*mut c_void],
        scalars: &[i32],
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            pointers.len() == self.info.sum_pointers && scalars.len() == self.info.sum_scalars,
            "EXL3 sum argument count mismatch"
        );
        let status = (self.sum)(
            self.context.as_ptr(),
            pointers.as_ptr(),
            scalars.as_ptr(),
            stream,
        );
        ensure!(status == 0, "EXL3 sum failed with CUDA status {status}");
        Ok(())
    }
}

impl Drop for V41Exl3Kernel {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}

fn check_route_initialization(status: i32) -> Result<()> {
    ensure!(status != 201,
        "EXL3 route initialization failed: no current CUDA context on the route-creating thread (CUDA_ERROR_INVALID_CONTEXT, 201)");
    ensure!(status == 0, "EXL3 route initialization failed: {status}");
    Ok(())
}

/// Native route preparation, owned on the creating CUDA thread/context.
pub struct V41Exl3Routes {
    _library: Library,
    context: NonNull<c_void>,
    launch:
        unsafe extern "C" fn(*mut c_void, *const *mut c_void, *const u64, i32, *mut c_void) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    _owner_thread: PhantomData<Rc<()>>,
}
impl V41Exl3Routes {
    /// # Safety
    /// Trusted module only; keep its owning context current and drain streams /
    /// destroy referencing graphs before dropping this handle.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        let library = Library::new(path.as_ref())?;
        let create = *library
            .get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"cuteafd_exl3_routes_create")?;
        let launch = *library.get(b"cuteafd_exl3_routes_launch")?;
        let destroy = *library.get(b"cuteafd_exl3_routes_destroy")?;
        let mut context = std::ptr::null_mut();
        let status = create(&mut context);
        check_route_initialization(status)?;
        Ok(Self {
            _library: library,
            context: NonNull::new(context).context("null EXL3 route context")?,
            launch,
            destroy,
            _owner_thread: PhantomData,
        })
    }
    /// # Safety
    /// Seven int32 buffers in manifest order, correctly sized/aligned, distinct,
    /// on the owning device and alive through all uses of this stream/graph.
    pub unsafe fn launch(
        &self,
        pointers: &[*mut c_void; 7],
        bytes: &[u64; 7],
        rows: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        let status = (self.launch)(
            self.context.as_ptr(),
            pointers.as_ptr(),
            bytes.as_ptr(),
            rows,
            stream,
        );
        ensure!(status == 0, "EXL3 route launch failed: {status}");
        Ok(())
    }
}
impl Drop for V41Exl3Routes {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}

#[cfg(test)]
mod info_tests {
    use super::*;

    #[test]
    fn route_initialization_reports_missing_current_context() {
        assert!(check_route_initialization(0).is_ok());
        let missing = check_route_initialization(201).unwrap_err().to_string();
        assert!(missing.contains("no current CUDA context on the route-creating thread"));
        assert!(missing.contains("201"));
        assert_eq!(check_route_initialization(2).unwrap_err().to_string(),
            "EXL3 route initialization failed: 2");
    }

    #[test]
    #[ignore = "requires a CUDA GPU, CUTEAFD_NATIVE_LIB and four-tier CUTEAFD_EXL3_AOT"]
    fn native_four_tier_library_loads_through_rust_owner() -> Result<()> {
        let native = unsafe { crate::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        native.cuda_set_device(0)?;
        let path =
            std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?).join("libcuteafd_exl3.so");
        let first = unsafe { V41Exl3Kernel::load(&path)? };
        let second = unsafe { V41Exl3Kernel::load(&path)? };
        assert_eq!(first.info().tier_count, 4);
        assert_eq!(first.info().bits, [2, 3, 4, 5]);
        assert_eq!(first.info(), second.info());
        drop(first);
        drop(second);
        let reloaded = unsafe { V41Exl3Kernel::load(&path)? };
        assert_eq!(reloaded.info().tier_count, 4);
        Ok(())
    }

    #[test]
    fn native_info_accepts_two_through_four_distinct_tiers() {
        for bits in [&[2, 5][..], &[2, 3, 5], &[2, 3, 4, 5]] {
            let mut words = [
                2,
                5120,
                512,
                6,
                16,
                6,
                bits.len() as u32,
                44,
                18,
                7,
                3,
                0,
                0,
                0,
                0,
                2,
            ];
            words[11..11 + bits.len()].copy_from_slice(bits);
            let info = V41Exl3Info::from_words(words).unwrap();
            assert_eq!(info.tier_count, bits.len());
            assert_eq!(&info.bits[..bits.len()], bits);
        }
    }

    /// Whole-H128 disjoint shards include TP3's 768 and TP6's 384 rows;
    /// fractional-block TP4 width 576 is still not a valid EXL3 publication.
    #[test]
    fn native_info_accepts_every_published_shard_width() {
        let base = [2u32, 5120, 768, 384, 16, 6, 2, 44, 18, 7, 3, 2, 3, 0, 0, 2];
        for intermediate in [384u32, 512, 640, 768, 1152, 2304] {
            let mut words = base;
            words[2] = intermediate;
            let info = V41Exl3Info::from_words(words)
                .unwrap_or_else(|error| panic!("shard width {intermediate} must be accepted: {error}"));
            assert_eq!(info.intermediate, intermediate as usize);
        }
        for intermediate in [0u32, 256, 576, 896, 1024, 1280] {
            let mut words = base;
            words[2] = intermediate;
            assert!(
                V41Exl3Info::from_words(words).is_err(),
                "shard width {intermediate} must be rejected"
            );
        }
    }

    #[test]
    fn native_info_accepts_published_v41_tp6_abi() {
        let words = [2, 5120, 384, 384, 1, 6, 2, 32, 11, 6, 3, 2, 3, 0, 0, 2];
        let info = V41Exl3Info::from_words(words).unwrap();
        assert_eq!(info.intermediate, 3 * 128);
        assert_eq!(info.hidden, 5120);
        assert_eq!(info.experts, 384);
        assert_eq!(info.capacity, 1);
        assert_eq!(info.topk, 6);
        assert_eq!(info.output_element_bytes, 2);
        assert_eq!(info.bits, [2, 3, 0, 0]);
    }

    #[test]
    fn native_info_accepts_e896_with_explicit_kimi_geometry() {
        let geometry = cuteafd_core::ExpertGeometry::KIMI_K3;
        for width in [512, 3072] {
            for capacity in [1, 16] {
                let words = [2, 3584, width, 896, capacity, 16, 2, 32, 11, 6, 3, 2, 3, 0, 0, 4];
                let info = V41Exl3Info::from_words_for_geometry(words, geometry).unwrap();
                assert_eq!(info.experts, 896);
                assert_eq!(info.intermediate, width as usize);
                assert!(V41Exl3Info::from_words_for_geometry(
                    words, cuteafd_core::ExpertGeometry::DEEPSEEK_V41).is_err());
                for experts in [512, 1024] {
                    let mut valid = words;
                    valid[3] = experts;
                    assert!(V41Exl3Info::from_words_for_geometry(valid, geometry).is_ok());
                }
                for experts in [0, 15, 1025, u32::MAX] {
                    let mut invalid = words;
                    invalid[3] = experts;
                    assert!(V41Exl3Info::from_words_for_geometry(invalid, geometry).is_err());
                }
            }
        }
    }

    #[test]
    fn shard_widths_follow_the_expert_geometry() {
        use cuteafd_core::ExpertGeometry;
        assert_eq!(exl3_shard_widths(ExpertGeometry::DEEPSEEK_V41), [384, 512, 640, 768, 1152, 2304]);
        assert_eq!(exl3_shard_widths(ExpertGeometry::DEEPSEEK_V4_PRO), [512, 768, 1024, 1536, 3072]);
        assert_eq!(exl3_shard_widths(ExpertGeometry::DEEPSEEK_V4_FLASH), [256, 384, 512, 640, 768, 1024, 2048]);
        assert_eq!(exl3_shard_widths(ExpertGeometry::QWEN4), [128, 256, 384, 640]);
    }

    #[test]
    fn native_info_rejects_invalid_tier_headers_before_slicing() {
        let valid = [2, 5120, 512, 6, 16, 6, 4, 44, 18, 7, 3, 2, 3, 4, 5, 2];
        for count in [0, 1, 5, u32::MAX] {
            let mut words = valid;
            words[6] = count;
            assert!(V41Exl3Info::from_words(words).is_err());
        }
        for (index, value) in [(11, 1), (14, 6), (12, 2), (7, 65)] {
            let mut words = valid;
            words[index] = value;
            assert!(V41Exl3Info::from_words(words).is_err());
        }
        let mut nonzero_padding = valid;
        nonzero_padding[6] = 3;
        assert!(V41Exl3Info::from_words(nonzero_padding).is_err());
    }
}
