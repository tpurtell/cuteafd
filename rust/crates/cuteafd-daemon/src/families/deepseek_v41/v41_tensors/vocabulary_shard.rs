//! Direct checkpoint loading of contiguous vocabulary rows onto one GPU.
use super::*;
use std::ops::Range;

pub(crate) struct VocabularyShard<'a> {
    allocation: Option<DeviceAllocation<'a>>,
    device_id: i32,
    tokens: Range<usize>,
    fp8: Option<crate::shared::fp8_linear::Fp8Weight<'a>>,
    packed: Option<crate::shared::packed_head::PackedHead<'a>>,
}
impl<'a> VocabularyShard<'a> {
    const VOCAB: usize = 129280;
    const ROW_BYTES: usize = 5120 * 2;

    pub fn device_bytes(catalog: &OfficialV41Catalog, tokens: Range<usize>) -> Result<usize> {
        ensure!(tokens.start < tokens.end && tokens.end <= Self::VOCAB,
            "invalid vocabulary shard token range");
        ensure!(NativeRtxTensors::plan(catalog, &["head.weight".into()])? == Self::VOCAB * Self::ROW_BYTES,
            "unexpected vocabulary checkpoint geometry");
        Ok(tokens.len() * Self::ROW_BYTES)
    }

    /// Peak GPU allocation while packing. Final residency is smaller in `all` and `packed`.
    pub fn load_bytes(catalog: &OfficialV41Catalog, tokens: Range<usize>) -> Result<usize> {
        let source = Self::device_bytes(catalog, tokens)?;
        Ok(Self::peak_bytes(source, super::fp8_head()))
    }
    fn peak_bytes(source: usize, mode: super::Fp8Head) -> usize {
        mode.load_bytes(source as u64) as usize
    }

    /// Device-free recipe shared with header-only startup admission checks.
    pub(crate) fn pack_steps(whole: crate::shared::experts::rtx::dense::Rect, device: i32,
        values: CuteafdDeviceBuffer, scale: CuteafdDeviceBuffer,
    ) -> Result<Vec<crate::shared::experts::rtx::dense::Step>> {
        use crate::shared::experts::rtx::dense::{Job, Step};
        let rows = whole.rows.len();
        ensure!(whole.width() == Self::ROW_BYTES && rows % 16 == 0,
            "vocabulary rows are not whole 16-row tiles");
        let per = 16;
        Ok((0..rows).step_by(per).map(|first| Step { device,
            rects: vec![whole.sub(first..first + per)], jobs: vec![Job::W8a16Pack { n: per, k: 5120,
                rule: crate::shared::fp8_linear::Fp8Scales::Best,
                packed: CuteafdDeviceBuffer { ptr: values.ptr.cast::<u8>().wrapping_add(first * 5120).cast(),
                    bytes: per * 5120, ..values },
                scale: CuteafdDeviceBuffer { ptr: scale.ptr.cast::<u8>().wrapping_add(first * 160).cast(),
                    bytes: per * 160, ..scale },
            }] }).collect())
    }

    /// The caller scopes construction/destruction to the owning GPU. Admission
    /// precedes allocation and payload reads. No full-head GPU staging is used.
    pub fn load(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        tokens: Range<usize>, budget: usize, staging_bytes: usize) -> Result<Self> {
        let bytes = Self::device_bytes(catalog, tokens.clone())?;
        ensure!(Self::load_bytes(catalog, tokens.clone())? <= budget, "vocabulary shard peak load exceeds device budget");
        ensure!((1..=64 * 1024 * 1024).contains(&staging_bytes),
            "vocabulary pinned staging must be 1 byte through 64 MiB");
        let device_id = library.cuda_get_device()?;
        let dense = crate::shared::experts::rtx::dense::DenseLoad::mount(library);
        let rows = tokens.len();
        let whole = super::rect(catalog, "head.weight", Some(tokens.clone()))?;
        let mut packed = None;
        let (allocation, fp8) = if super::fp8_head() == super::Fp8Head::All {
            // The FP8 copy is the only resident one: pack 16-row windows of the
            // staged BF16 rows straight into it (tiles are independent).
            let packed = crate::shared::fp8_linear::Fp8Weight::allocate(library, rows, 5120)?;
            let (values, scale) = packed.buffers();
            let steps = Self::pack_steps(whole, device_id, values, scale)?;
            let stream = library.cuda_stream_create()?;
            let staged = (|| -> Result<()> {
                for step in steps { dense.step(step, stream)?; }
                Ok(())
            })();
            // SAFETY: every job drained before returning; the stream has no other users.
            unsafe { library.cuda_stream_destroy(stream)?; }
            staged?;
            (None, Some(packed))
        } else {
            let allocation = DeviceAllocation::new(library, bytes)?;
            dense.operand(crate::shared::experts::rtx::dense::Operand { device: device_id, buffer: allocation.buffer,
                lands: vec![(whole, 0)], bytes })?;
            match super::fp8_head() {
                super::Fp8Head::Off => (Some(allocation), None),
                super::Fp8Head::Packed => {
                    // The exact copy packs (and verifies) from the whole landed
                    // BF16 shard, which it then releases.
                    ensure!(!crate::shared::experts::rtx::dense::is_recording(), "the packed head loads through the mount");
                    packed = Some(super::pack_packed(library, allocation, rows)?);
                    (None, None)
                }
                _ => {
                    ensure!(!crate::shared::experts::rtx::dense::is_recording(), "the draft-only FP8 head loads through the mount");
                    let (source, packed) = super::pack_fp8(library, allocation, rows)?;
                    (source, Some(packed))
                }
            }
        };
        let _ = staging_bytes;
        let (exceptions, row_exceptions) = packed.as_ref().map_or((0, 0), |head| head.exceptions());
        tracing::info!(device_id, rows = tokens.len(), bf16_bytes = allocation.as_ref().map_or(0, |a| a.buffer.bytes),
            fp8_bytes = fp8.as_ref().map_or(0, |w| w.bytes()), packed12_bytes = packed.as_ref().map_or(0, |h| h.bytes()),
            exceptions, row_exceptions, "V4.1 vocabulary residency");
        Ok(Self { allocation, device_id, tokens, fp8, packed })
    }
    pub fn weight(&self) -> Option<CuteafdDeviceBuffer> { self.allocation.as_ref().map(|a| a.buffer) }
    pub fn device_id(&self) -> i32 { self.device_id }
    /// The resident copy (`CUTEAFD_V41_FP8_HEAD`), when one was packed.
    pub fn copy(&self) -> Option<super::HeadCopy<'_, 'a>> {
        self.packed.as_ref().map(super::HeadCopy::Packed).or(self.fp8.as_ref().map(super::HeadCopy::Fp8))
    }
    pub fn tokens(&self) -> Range<usize> { self.tokens.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::Device;

    #[test]
    fn vocabulary_pack_admits_source_and_destinations() {
        let source = 129280 * 10240;
        assert_eq!(VocabularyShard::peak_bytes(source, Fp8Head::Off), source);
        for mode in [Fp8Head::Draft, Fp8Head::All] {
            assert_eq!(VocabularyShard::peak_bytes(source, mode), source + 129280 * (5120 + 160));
        }
        // The exact copy's single allocation lands beside the whole BF16 source.
        for rows in [129280, 64640] {
            let layout = cuteafd_core::packed_head::PackedHeadLayout::new(rows, 5120).unwrap();
            assert_eq!(VocabularyShard::peak_bytes(rows * 10240, Fp8Head::Packed), rows * 10240 + layout.total_bytes());
        }
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and two CUDA GPUs"]
    fn dual_vocabulary_shards_match_complete_checkpoint_ranges() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let reader = catalog.coordinator_tensor_reader("head.weight")?;
        lib.cuda_set_device(0)?;
        let mut shards = Vec::new();
        for (id, tokens) in [(0, 0..64640), (1, 64640..129280)] {
            let device = Device { library: &lib, id };
            let bytes = VocabularyShard::device_bytes(&catalog, tokens.clone())?;
            assert_eq!(bytes, 661913600);
            assert!(device.run(|| VocabularyShard::load(&lib, &catalog, tokens.clone(), bytes - 1, 7 << 20)).is_err());
            let shard = device.own(|| VocabularyShard::load(&lib, &catalog, tokens.clone(), bytes, 7 << 20))?;
            assert_eq!(shard.tokens(), tokens);
            assert_eq!(shard.weight().context("test requires BF16 head")?.bytes, bytes);
            let mut expected = vec![0u8; 7 << 20];
            let mut actual = vec![0u8; expected.len()];
            for offset in (0..bytes).step_by(expected.len()) {
                let count = expected.len().min(bytes - offset);
                reader.read_into((tokens.start * VocabularyShard::ROW_BYTES + offset) as u64,
                    &mut expected[..count])?;
                let buffer = shard.weight().context("test requires BF16 head")?;
                let source = CuteafdDeviceBuffer {
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes: count, ..buffer
                };
                device.run(|| lib.copy_d2h(&mut actual[..count], source))?;
                ensure!(actual[..count] == expected[..count], "vocabulary shard differs on GPU {id} at byte {offset}");
            }
            eprintln!("PASS GPU {id} vocabulary tokens {tokens:?}: all {bytes} bytes match checkpoint");
            shards.push(shard);
        }
        for tokens in [0..0, 129280..129281, 129281..129282] {
            assert!(VocabularyShard::device_bytes(&catalog, tokens).is_err());
        }
        drop(shards);
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}

#[cfg(test)]
impl VocabularyShard<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        if let Some(a) = &self.allocation { f(format!("{prefix}bf16"), a.buffer); }
        if let Some(w) = &self.fp8 { w.visit(&format!("{prefix}fp8."), f); }
        if let Some(head) = &self.packed { head.visit(prefix, f); }
    }
}
