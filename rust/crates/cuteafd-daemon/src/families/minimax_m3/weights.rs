//! Raw target operands: one admitted backing, checkpoint bytes unchanged.
use crate::shared::{
    experts::rtx::dense::{is_recording, DenseLoad, Operand, Rect},
    memory::{device::{Allocation, Device}, receipts::Inventory, weight_arena::DeviceWeightArena},
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{families::minimax_m3::weight_layout::CoordinatorWeights, plan::Checkpoint};
use std::collections::BTreeMap;
use crate::shared::token_io::{EmbedPlacement, EmbedSource, TokenEmbedding};

#[path = "push.rs"]
pub(crate) mod push;

const EMBEDDING: &str = "language_model.model.embed_tokens.weight";

/// Views retain the packed backing after the loading arena retires. There is
/// deliberately no per-tensor allocation or converted weight representation.
pub(crate) struct TargetWeights<'a> {
    pub rank: usize,
    pub layers: Vec<usize>,
    operands: BTreeMap<String, Allocation<'a>>,
}

impl<'a> TargetWeights<'a> {
    /// The caller admits the layout's weight components and mount owners first.
    /// This synchronous mount publishes no views until every operand has landed.
    pub fn mount(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        ensure!(!is_recording(), "MiniMax synchronous target mount cannot publish recorded operands");
        Self::load(device, checkpoint, layout)
    }

    /// Under dense recording these same native views are push destinations.
    /// The caller must retain them unread until the admitted group commits.
    pub fn load(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        let arena = DeviceWeightArena::new(device, layout.arena.clone(), ArenaTag::weights("minimax/target"))?;
        Self::load_into(device, checkpoint, layout, &arena, false)
    }

    /// Entry-rank embedding and target share the admitted backing on GPU. Host
    /// placement uses the shared single-residency mapped pinned table instead.
    pub fn load_with_embedding(device: Device<'a>, checkpoint: &Checkpoint,
        layout: &CoordinatorWeights, placement: EmbedPlacement) -> Result<(Self, TokenEmbedding<'a>)> {
        ensure!(layout.rank == 0, "MiniMax embedding must load on the entry rank");
        ensure!(layout.arena.views().contains_key(EMBEDDING) == (placement == EmbedPlacement::Gpu),
            "MiniMax embedding placement differs from admitted arena");
        let tensor = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == EMBEDDING)
            .context("MiniMax embedding header")?;
        let hidden = *tensor.meta.shape.get(1).context("MiniMax embedding shape")?;
        let source = EmbedSource::new(&checkpoint.snapshot, &tensor.shard, &tensor.meta, hidden)?;
        let arena = DeviceWeightArena::new(device, layout.arena.clone(), ArenaTag::weights("minimax/target"))?;
        let (embedding, target) = device.run(|| TokenEmbedding::load_with(device.library, source, placement,
            || Self::load_into(device, checkpoint, layout, &arena, true),
            |_, bytes| arena.view(EMBEDDING, bytes)))?;
        Ok((target, embedding))
    }

    fn load_into(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights,
        arena: &DeviceWeightArena<'a>, embedding_loaded: bool) -> Result<Self> {
        let mut rects = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            if embedding_loaded && name == EMBEDDING { continue; }
            let rect = Rect::tensor(checkpoint, name, None, None)?;
            ensure!(rect.bytes() == view.bytes(), "MiniMax mount {name}: source differs from admitted view");
            rects.insert(name.clone(), rect);
        }
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(layout.mount_load(checkpoint)?)?;
        let mut operands = BTreeMap::new();
        for (name, rect) in rects {
            let bytes = rect.bytes();
            let allocation = arena.view(&name, bytes)?;
            dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(rect, 0)], bytes })?;
            operands.insert(name, allocation);
        }
        dense.release();
        Ok(Self { rank: layout.rank, layers: layout.layers.clone(), operands })
    }

    pub fn operand(&self, name: &str) -> Result<&Allocation<'a>> {
        self.operands.get(name).with_context(|| format!("MiniMax target lacks operand {name}"))
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.operands { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::{placement::planner::{CoordinatorDevices, Lifetime, PhaseOrder},
        serving_capacity::WeightArenaLayout};
    use crate::shared::memory::receipts::Census;
    use cuteafd_core::memory_layout::Category;
    use std::os::unix::fs::FileExt;

    fn checkpoint(dir: &std::path::Path) -> Result<(Checkpoint, CoordinatorWeights, BTreeMap<String, Vec<u8>>)> {
        let tensors = vec![
            ("q.weight".into(), "F8_E4M3", vec![32, 128]),
            ("q.weight_scale_inv".into(), "U8", vec![32, 4]),
            ("router.weight".into(), "F32", vec![4, 128]),
            ("norm.weight".into(), "BF16", vec![128]),
        ];
        // This fixture exercises landing only, not the separately tested family parser.
        cuteafd_loader::plan::testing::write_snapshot(dir, &serde_json::json!({"model_type":"fixture"}), &tensors, None);
        let checkpoint = Checkpoint::open(dir)?;
        let mut expected = BTreeMap::new();
        for (index, tensor) in checkpoint.tensors.iter().enumerate() {
            let bytes = (0..tensor.meta.byte_length as usize).map(|at| ((at * 29 + index * 17) % 251) as u8).collect::<Vec<_>>();
            let file = std::fs::OpenOptions::new().write(true).open(dir.join(&tensor.shard))?;
            file.write_all_at(&bytes, tensor.meta.byte_offset)?;
            expected.insert(tensor.meta.name.clone(), bytes);
        }
        let arena = WeightArenaLayout::new("minimax/target/0", expected.iter().map(|(name, bytes)| (name.clone(), bytes.len())))?;
        let layout = CoordinatorWeights { rank: 0, layers: vec![2, 5], arena, projections: Vec::new() };
        Ok((checkpoint, layout, expected))
    }

    #[test]
    fn raw_mount_preserves_bytes_and_one_retained_backing() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 1 << 20)?;
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let (checkpoint, layout, expected) = checkpoint(dir.path())?;
        let device = Device { library: &library, id: 1 };
        let weights = TargetWeights::mount(device, &checkpoint, &layout)?;
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(weights.rank, 0);
        assert_eq!(weights.layers, [2, 5]);
        assert!(weights.operand("absent").is_err());
        for (name, bytes) in &expected {
            let operand = weights.operand(name)?;
            let mut actual = vec![0; bytes.len()];
            device.run(|| library.copy_d2h(&mut actual, operand.buffer))?;
            assert_eq!(&actual, bytes, "{name}");
        }
        let devices = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax", Lifetime::TARGET,
            Category::Weights, "checkpoint-native target");
        weights.inventory(&mut inventory, "target")?;
        let mut census = Census::default();
        census.owners(inventory.receipts);
        census.normalize(PhaseOrder::DRAFT_THEN_TARGET)?;
        assert_eq!(census.receipts.len(), expected.len() + 1);
        let roots = census.receipts.iter().filter(|r| r.allocation.view.is_none()).collect::<Vec<_>>();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].allocation.id, layout.arena.id());
        assert_eq!(roots[0].allocation.bytes, layout.arena.bytes() as u64);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, (1 << 20) - layout.arena.bytes());
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        drop(weights);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
        assert_eq!(fixture.events()?.matches('d').count(), 1);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn raw_mount_failure_drops_partial_views_and_restores_device() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 1 << 20)?;
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let (checkpoint, layout, _) = checkpoint(dir.path())?;
        // The header was accepted, but a later source read now fails after the
        // earlier tensors landed. No initialized target may escape this error.
        let last = checkpoint.tensors.last().unwrap();
        std::fs::OpenOptions::new().write(true).open(dir.path().join(&last.shard))?.set_len(last.meta.byte_offset)?;
        let device = Device { library: &library, id: 1 };
        assert!(TargetWeights::mount(device, &checkpoint, &layout).is_err());
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        assert_eq!(fixture.events()?.matches('d').count(), 1);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn raw_mount_streams_operands_larger_than_source_window() -> Result<()> {
        use cuteafd_loader::shared::dense_read::STEP_BYTES;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let capacity = STEP_BYTES * 2;
        fixture.configure_memory(&library, 1, capacity)?;
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let tensors = vec![("payload".into(), "U8", vec![STEP_BYTES / 4096 + 1, 4096])];
        cuteafd_loader::plan::testing::write_snapshot(dir.path(), &serde_json::json!({"model_type":"fixture"}), &tensors, None);
        let checkpoint = Checkpoint::open(dir.path())?;
        let header = checkpoint.header("payload").unwrap();
        let bytes = header.byte_length as usize;
        let file = std::fs::OpenOptions::new().write(true).open(dir.path().join(&checkpoint.tensors[0].shard))?;
        // Distinct markers straddle the reused pinned-buffer boundary.
        for (offset, marker) in [(0, 17), (STEP_BYTES - 1, 29), (STEP_BYTES, 31), (bytes - 1, 43)] {
            file.write_all_at(&[marker], header.byte_offset + offset as u64)?;
        }
        let layout = CoordinatorWeights { rank: 0, layers: Vec::new(),
            arena: WeightArenaLayout::new("minimax/target/0", [("payload".into(), bytes)])?, projections: Vec::new() };
        assert_eq!(layout.mount_load(&checkpoint)?.pinned_source, STEP_BYTES as u64);
        let device = Device { library: &library, id: 1 };
        let weights = TargetWeights::mount(device, &checkpoint, &layout)?;
        let mut actual = vec![0; bytes];
        device.run(|| library.copy_d2h(&mut actual, weights.operand("payload")?.buffer))?;
        assert_eq!((actual[0], actual[STEP_BYTES - 1], actual[STEP_BYTES], actual[bytes - 1]), (17, 29, 31, 43));
        assert_eq!(actual.iter().filter(|value| **value != 0).count(), 4);
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        drop(weights);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, capacity);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn recorded_target_push_matches_mount_without_receiver_reads() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_memory(&library, 1, 1 << 20)?;
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let (checkpoint, layout, expected) = checkpoint(dir.path())?;
        let device = Device { library: &library, id: 1 };
        let mounted = TargetWeights::mount(device, &checkpoint, &layout)?;
        let (pushed, records) = crate::shared::experts::rtx::dense::recording(||
            TargetWeights::load(device, &checkpoint, &layout))?;
        assert_eq!(records.operands.len(), expected.len());
        assert!(records.steps.is_empty());
        let phase = crate::shared::memory::load_timing::Split::start();
        let delivered = crate::shared::experts::rtx::push::tests::receive_pushed(device, None,
            dir.path(), dir.path(), &records, None, 0, 0)?;
        assert_eq!(delivered.len(), 0);
        assert_eq!(phase.read_bytes(), 0,
            "push receiver must not read checkpoint mount bytes");
        for (name, expected) in &expected {
            let mut actual = vec![0; expected.len()];
            let mut baseline = vec![0; expected.len()];
            device.run(|| library.copy_d2h(&mut actual, pushed.operand(name)?.buffer))?;
            device.run(|| library.copy_d2h(&mut baseline, mounted.operand(name)?.buffer))?;
            assert_eq!(&actual, expected, "{name}");
            assert_eq!(actual, baseline, "{name}");
        }
        assert_eq!(fixture.events()?.matches('D').count(), 2);
        drop((pushed, mounted));
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
        Ok(())
    }

    #[test]
    fn embedding_uses_one_residency_and_shared_push_records() -> Result<()> {
        for placement in [EmbedPlacement::Gpu, EmbedPlacement::Host] {
            let fixture = Fixture::build_for_tests()?;
            let library = fixture.load()?;
            fixture.immediate_copies(&library)?;
            fixture.configure_memory(&library, 1, 1 << 20)?;
            let dir = tempfile::tempdir_in(fixture.directory())?;
            let tensors = vec![(EMBEDDING.into(), "BF16", vec![4, 128]),
                ("norm.weight".into(), "BF16", vec![128])];
            cuteafd_loader::plan::testing::write_snapshot(dir.path(),
                &serde_json::json!({"model_type":"fixture", "tie_word_embeddings":false}), &tensors, None);
            let checkpoint = Checkpoint::open(dir.path())?;
            let regions = tensors.iter().filter(|(name, _, _)|
                placement == EmbedPlacement::Gpu || name != EMBEDDING)
                .map(|(name, _, shape)| (name.clone(), shape.iter().product::<usize>() * 2));
            let layout = CoordinatorWeights { rank: 0, layers: Vec::new(), projections: Vec::new(),
                arena: WeightArenaLayout::new("minimax/target/0", regions)? };
            let device = Device { library: &library, id: 1 };
            let ((target, embedding), records) = crate::shared::experts::rtx::dense::recording(||
                TargetWeights::load_with_embedding(device, &checkpoint, &layout, placement))?;
            assert_eq!(library.cuda_get_device()?, 0);
            assert_eq!(embedding.placement(), placement);
            assert_eq!(embedding.owner().id, 1);
            assert!(target.operand(EMBEDDING).is_err(), "embedding table has a single inventory owner");
            assert_eq!(records.operands.len(), if placement == EmbedPlacement::Gpu { 2 } else { 1 });
            assert_eq!(records.host_operands.len(), usize::from(placement == EmbedPlacement::Host));
            let phase = crate::shared::memory::load_timing::Split::start();
            crate::shared::experts::rtx::push::tests::receive_pushed(device, None,
                dir.path(), dir.path(), &records, None, 0, 0)?;
            assert_eq!(phase.read_bytes(), 0);
            assert_eq!(embedding.host_rows(&[0, 3])?, vec![0; 2 * 128 * 2]);
            let physical = [1];
            let devices = CoordinatorDevices::new(&physical)?;
            let mut inventory = Inventory::new(devices, "minimax", Lifetime::RESIDENT,
                Category::Weights, "checkpoint-native target");
            target.inventory(&mut inventory, "target")?;
            let mut census = Census::default();
            census.owners(inventory.receipts);
            census.owners(embedding.allocation_receipts(devices, "minimax.embedding")?);
            embedding.allocation_aliases(&mut census, devices, "minimax.embedding")?;
            census.normalize(PhaseOrder::DRAFT_THEN_TARGET)?;
            let roots = census.receipts.iter().filter(|r| r.allocation.view.is_none()).collect::<Vec<_>>();
            assert_eq!(roots.len(), if placement == EmbedPlacement::Gpu { 1 } else { 2 });
            assert_eq!(roots.iter().filter(|r| r.allocation.location.domain
                == cuteafd_loader::placement::planner::Domain::Device).count(), 1);
            assert_eq!(fixture.events()?.matches('D').count(), 1);
            drop((target, embedding));
            assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
        }
        Ok(())
    }

    #[test]
    fn raw_mount_rejects_recording_before_allocating() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let (checkpoint, layout, _) = checkpoint(dir.path())?;
        let (rejected, records) = crate::shared::experts::rtx::dense::recording(|| {
            Ok(TargetWeights::mount(Device { library: &library, id: 1 }, &checkpoint, &layout).is_err())
        })?;
        assert!(rejected);
        assert!(records.operands.is_empty());
        assert_eq!(fixture.events()?.matches('D').count(), 0);
        Ok(())
    }
}
