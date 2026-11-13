//! BF16 embedding and zero-centered final norm through the shared target head.
use super::weights::TargetWeights;
use crate::shared::{memory::{device::{Allocation, Stream}, receipts::Inventory},
    target_head::{self, TargetHead}, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{minimax_m3::forward::{ForwardProducer, Kind, NormBatch},
    programs::{Programs, VocabularyHead}, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::minimax_m3::MiniMaxM3Config;
use std::rc::Rc;

/// The handle drops before its borrowed BLAS workspace. All storage survives
/// queued embedding/norm/head operations through the retained stream's drain.
pub(crate) struct Head<'a> {
    stream: Rc<Stream<'a>>,
    head: VocabularyHead<'a>,
    norm: ForwardProducer<'a>,
    tokens: Allocation<'a>,
    normalized: Allocation<'a>,
    logits: Allocation<'a>,
    workspace: Allocation<'a>,
    rows: usize,
    vocab: usize,
}
impl Drop for Head<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax head retirement"); }
}
impl<'a> Head<'a> {
    pub fn new(stream: Rc<Stream<'a>>, programs: &'a Programs<'a>, cfg: &MiniMaxM3Config, rows: usize) -> Result<Self> {
        let layout = cuteafd_loader::families::minimax_m3::execution::HeadLayout::new(cfg, u32::try_from(rows)?)?;
        let device = stream.device;
        let mut buffers = layout.buffers().into_iter().map(|(_, bytes)|
            Allocation::new(device, usize::try_from(bytes)?, Tag::workspace("minimax/head")));
        let tokens = buffers.next().unwrap()?;
        let normalized = buffers.next().unwrap()?;
        let logits = buffers.next().unwrap()?;
        let workspace = buffers.next().unwrap()?;
        let head = device.run(|| {
            // SAFETY: handle borrows this retained private workspace until drop;
            // row and checkpoint dimensions are checked before construction.
            unsafe { device.library.vocabulary_head_rows(workspace.buffer.ptr, 6144,
                rows as u32, cfg.vocab_size as u32, TargetHead::configured().native_mode()) }
        })?;
        let norm = device.run(|| Ok(ForwardProducer::bind(programs, Kind::ResidualNorm)?))?;
        Ok(Self { stream, head, norm, tokens, normalized, logits, workspace, rows, vocab: cfg.vocab_size })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, allocation) in [("tokens", &self.tokens), ("normalized", &self.normalized),
            ("logits", &self.logits), ("workspace", &self.workspace)] {
            allocation.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
    pub fn embed(&self, weights: &TargetWeights<'_>, tokens: &[u32], residual: Buffer) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= self.rows
            && tokens.iter().all(|&token| (token as usize) < self.vocab), "MiniMax embedding tokens");
        let device = self.stream.device;
        self.stream.drain()?;
        let bytes = tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let embedding = weights.operand("language_model.model.embed_tokens.weight")?.buffer;
        device.run(|| {
            device.library.copy_h2d(self.tokens.buffer, &bytes)?;
            // SAFETY: target arena and all scratch remain live on this stream;
            // residual starts at zero so the existing BF16->FP32 add is a cast.
            unsafe {
                device.library.cuda_embedding_lookup_bf16_async(embedding, self.tokens.buffer,
                    self.normalized.buffer, tokens.len(), self.vocab, 6144, self.stream.raw)?;
                device.library.cuda_zero_bytes_async(residual, tokens.len() * 6144 * 4, self.stream.raw)?;
                device.library.cuda_accumulate_bf16_to_f32_async(self.normalized.buffer, residual,
                    tokens.len() * 6144, self.stream.raw)
            }
        })
    }
    pub fn copy_normalized(&self, bytes: &mut [u8]) -> Result<()> {
        ensure!(!bytes.is_empty() && bytes.len() <= self.rows * 6144 * 2
            && bytes.len() % (6144 * 2) == 0, "MiniMax normalized tap extent");
        self.stream.drain()?;
        self.stream.device.run(|| self.stream.device.library.copy_d2h(bytes, self.normalized.buffer))
    }
    /// Logits borrow this admitted head until the next projection. Selection or
    /// scoring must consume them on the producing stream before scratch reuse.
    pub fn project_device(&self, weights: &TargetWeights<'_>, residual: Buffer, rows: usize) -> Result<DeviceLogits> {
        ensure!(rows > 0 && rows <= self.rows && residual.device_id == self.stream.device.id
            && residual.bytes >= rows * 6144 * 4, "MiniMax head live rows/device");
        let norm = weights.operand("language_model.model.norm.weight")?.buffer;
        let weight = weights.operand("language_model.lm_head.weight")?.buffer;
        let device = self.stream.device;
        device.run(|| {
            // SAFETY: all exact arena/scratch extents are retained; same ordered
            // stream as the shared prefill's final residual write.
            unsafe {
                self.norm.norm(NormBatch { rows, input: residual, weight: norm,
                    output: self.normalized.buffer }, self.stream.raw)?;
                target_head::launch(TargetHead::configured(), &self.head, self.normalized.buffer.ptr,
                    weight.ptr, self.logits.buffer.ptr.cast(), rows, self.stream.raw)?;
            }
            Ok(())
        })?;
        Ok(DeviceLogits { ptr: self.logits.buffer.ptr, rows, vocab: self.vocab,
            stride: self.vocab, stream: self.stream.raw, greedy: None })
    }
    /// Stream head tiles without altering the transformer's prefill units.
    /// `consume` finishes reading each tile before the next overwrites logits.
    pub fn project_spans(&self, weights: &TargetWeights<'_>, residual: Buffer, rows: usize,
        consume: &mut dyn FnMut(usize, DeviceLogits) -> Result<()>) -> Result<()> {
        ensure!(self.rows <= 64 && rows > 0 && residual.device_id == self.stream.device.id
            && rows.checked_mul(6144 * 4).is_some_and(|bytes| bytes <= residual.bytes),
            "MiniMax serving head must use an admitted <=64-row owner and bounded residual");
        for span in crate::shared::probe::head_spans(rows, self.rows)? {
            let input = super::cache::view(residual, span.start * 6144 * 4, span.len() * 6144 * 4)?;
            consume(span.start, self.project_device(weights, input, span.len())?)?;
        }
        Ok(())
    }
    pub fn project_scoring(&self, weights: &TargetWeights<'_>, residual: Buffer, rows: usize,
        scoring: crate::shared::probe::ScoringRows<'_>) -> Result<crate::shared::probe::ScoreLogits> {
        self.project_spans(weights, residual, rows, &mut |offset, logits|
            scoring.record(self.stream.device.library, offset, &logits))?;
        Ok(crate::shared::probe::ScoreLogits::Recorded)
    }
    /// Cold golden alone materializes complete host rows; serving consumes
    /// project_device/project_spans through the shared sampler or probe.
    pub fn project(&self, weights: &TargetWeights<'_>, residual: Buffer, rows: usize) -> Result<Vec<f32>> {
        self.project_device(weights, residual, rows)?;
        self.stream.drain()?;
        let mut bytes = vec![0; rows * self.vocab * 4];
        self.stream.device.run(|| self.stream.device.library.copy_d2h(&mut bytes, self.logits.buffer))?;
        let logits = bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(logits.iter().all(|value| value.is_finite()), "MiniMax nonfinite head logits");
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::Device;
    use cuteafd_core::memory_layout::Category;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};

    #[test]
    fn serving_head_has_one_fixed_bounded_owner_set_and_retires_on_its_device() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 128 << 20)?;
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../cuteafd-ffi/tests/fixtures/minimax_forward_programs.json"))?;
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let path = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let cfg = MiniMaxM3Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json"))?)?;
        let device = Device { library: &library, id: 1 };
        let stream = Rc::new(Stream::new(device)?);
        let head = Head::new(Rc::clone(&stream), &programs, &cfg, 64)?;
        let devices = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax",
            Lifetime::TARGET, Category::Workspace, "bounded serving head");
        head.inventory(&mut inventory, "head")?;
        let buffers = cuteafd_loader::families::minimax_m3::execution::HeadLayout::new(&cfg, 64)?.buffers();
        assert_eq!(inventory.receipts.len(), buffers.len());
        for (receipt, (name, bytes)) in inventory.receipts.iter().zip(&buffers) {
            assert_eq!(receipt.allocation.id, format!("minimax.head.{name}"));
            assert_eq!(receipt.allocation.bytes, *bytes);
        }
        let bytes = buffers.iter().map(|(_, bytes)| *bytes).sum::<u64>() as usize;
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, (128 << 20) - bytes);
        assert!(head.copy_normalized(&mut [0; 2]).is_err());
        assert_eq!(library.cuda_get_device()?, 0);
        drop(head);
        drop(stream);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 128 << 20);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn streamed_head_tiles_keep_prediction_positions_and_fixed_capacity() -> Result<()> {
        for rows in [1, 63, 64, 65, 128, 129, 2752, 4096] {
            let spans = crate::shared::probe::head_spans(rows, 64)?.collect::<Vec<_>>();
            assert!(spans.iter().all(|span| !span.is_empty() && span.len() <= 64));
            assert_eq!(spans.iter().cloned().flatten().collect::<Vec<_>>(), (0..rows).collect::<Vec<_>>());
            let prediction = 193;
            assert_eq!(spans.iter().flat_map(|span| span.clone().map(|row| prediction + row))
                .collect::<Vec<_>>(), (prediction..prediction + rows).collect::<Vec<_>>());
        }
        Ok(())
    }
}
