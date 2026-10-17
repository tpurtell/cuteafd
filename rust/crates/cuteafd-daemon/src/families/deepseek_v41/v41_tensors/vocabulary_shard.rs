//! Direct checkpoint loading of contiguous vocabulary rows onto one GPU.
use super::*;
use std::ops::Range;

pub(crate) struct VocabularyShard<'a> {
    allocation: Option<DeviceAllocation<'a>>,
    device_id: i32,
    tokens: Range<usize>,
    fp8: Option<crate::shared::fp8_linear::Fp8Weight<'a>>,
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

    /// Peak GPU allocation while packing. Final residency is smaller in `all`.
    pub fn load_bytes(catalog: &OfficialV41Catalog, tokens: Range<usize>) -> Result<usize> {
        let source = Self::device_bytes(catalog, tokens)?;
        Ok(Self::peak_bytes(source, super::fp8_head()))
    }
    fn peak_bytes(source: usize, mode: super::Fp8Head) -> usize {
        mode.load_bytes(source as u64) as usize
    }

    /// The caller scopes construction/destruction to the owning GPU. Admission
    /// precedes allocation and payload reads. No full-head GPU staging is used.
    pub fn load(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        tokens: Range<usize>, budget: usize, staging_bytes: usize) -> Result<Self> {
        let bytes = Self::device_bytes(catalog, tokens.clone())?;
        ensure!(Self::load_bytes(catalog, tokens.clone())? <= budget, "vocabulary shard peak load exceeds device budget");
        ensure!((1..=64 * 1024 * 1024).contains(&staging_bytes),
            "vocabulary pinned staging must be 1 byte through 64 MiB");
        let reader = catalog.coordinator_tensor_reader("head.weight")?;
        let mut staging = HostAllocation::new(library, staging_bytes.min(bytes))?;
        let allocation = DeviceAllocation::new(library, bytes)?;
        let source_start = tokens.start * Self::ROW_BYTES;
        let mut offset = 0;
        while offset < bytes {
            let count = staging_bytes.min(bytes - offset);
            let source = &mut staging.bytes_mut()[..count];
            reader.read_into((source_start + offset) as u64, source)?;
            let destination = CuteafdDeviceBuffer {
                // SAFETY: offset/count stay inside the admitted source allocation.
                ptr: unsafe { allocation.buffer.ptr.cast::<u8>().add(offset).cast() },
                bytes: count, ..allocation.buffer
            };
            library.copy_h2d(destination, source)?;
            offset += count;
        }
        let device_id = allocation.buffer.device_id;
        let (allocation, fp8) = if super::fp8_head() == super::Fp8Head::Off {
            (Some(allocation), None)
        } else {
            let (source, packed) = super::pack_fp8(library, allocation, tokens.len())?;
            (source, Some(packed))
        };
        tracing::info!(device_id, rows = tokens.len(), bf16_bytes = allocation.as_ref().map_or(0, |a| a.buffer.bytes),
            fp8_bytes = fp8.as_ref().map_or(0, |w| w.bytes()), "V4.1 vocabulary residency");
        Ok(Self { allocation, device_id, tokens, fp8 })
    }
    pub fn weight(&self) -> Option<CuteafdDeviceBuffer> { self.allocation.as_ref().map(|a| a.buffer) }
    pub fn device_id(&self) -> i32 { self.device_id }
    /// The FP8 copy (`CUTEAFD_V41_FP8_HEAD`), when one was packed.
    pub fn fp8(&self) -> Option<&crate::shared::fp8_linear::Fp8Weight<'a>> { self.fp8.as_ref() }
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
