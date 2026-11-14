//! Muse Spark's DFlash2 drafter on the shared executor (`DflashDrafter`),
//! beside the target head on the last coordinator rank.
//!
//! - Stream: the head rank's own stream, so taps, context updates, drafts and
//!   the target step are ordered without events (as GLM Flash).
//! - Taps: after each tapped layer the head rank's full-width `hidden` rows
//!   (identical on both ranks of a pair after the layer) are copied into the
//!   drafter's tap rows; a prefill call keeps only its tail (`TAP_ROWS`, the
//!   drafter's sliding window).
//! - Input rows: the target's embedding table lives on rank 0; the drafter
//!   gathers its block's rows (anchor, then the mask token) from it over peer
//!   access on its own stream (`draft_device_with`), as rank 1's step embed.
//! - Head: the drafter has no head of its own. Its `TargetHead::Launch` runs
//!   the target's selected BF16/trellis head on the drafter's own logit planes, into
//!   stored-width F32 rows, trimmed to the vocabulary, then `muse_softcap`
//!   (output multiplier and softcap, the values the drafter was trained on;
//!   admission refuses a drafter with others).
use super::engine::RankResources;
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::memory::Tag;
use crate::shared::memory::weight_arena::{weight_allocation, DeviceWeightArena};
use crate::shared::speculation::dflash::{ContextRow, DflashDrafter, Draft, DraftSeq, TargetHead, TAP_ROWS};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::muse_spark::MuseSparkConfig;
use std::{ffi::c_void, path::Path};
use cuteafd_core::memory_layout::ArenaTag;

pub(crate) struct MuseDraft<'a> {
    pub drafter: DflashDrafter<'a>,
    /// The selected head's stored-width F32 rows for a full draft batch.
    stored: Allocation<'a>,
    /// The trimmed F32 rows the softcap reads.
    trimmed: Allocation<'a>,
    /// Token ids of a draft batch's input rows (on the drafter's device).
    ids: Allocation<'a>,
    pub stream: *mut c_void,
    device: Device<'a>,
    pub block: usize,
    rows: usize,
}

fn drafter_arenas<'a>(device: Device<'a>, snapshot: &Path, devices: &[i32], head_rank: usize,
    representation: cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation)
    -> Result<std::collections::BTreeMap<i32, DeviceWeightArena<'a>>> {
    DflashDrafter::arena_requests(snapshot, representation, devices, head_rank)?.into_iter()
        .map(|(id, requests)| {
            let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(
                format!("muse/drafter/{id}"), requests)?;
            DeviceWeightArena::new(Device { library: device.library, id }, layout, ArenaTag::drafter_weights("muse/drafter"))
                .map(|arena| (id, arena))
        }).collect()
}

impl<'a> MuseDraft<'a> {
    /// Loads `snapshot` on the head rank's device and stream. `sequences`
    /// bounds a draft batch and the context slots (the admitted charge).
    pub(crate) fn load(programs: &'a cuteafd_ffi::programs::Programs<'a>, cfg: &MuseSparkConfig, head: &RankResources<'a>, stream: *mut c_void, snapshot: &Path,
        table: &crate::shared::token_io::TokenEmbedding<'_>, admitted: &cuteafd_loader::placement::families::muse_spark::MuseDrafter,
        kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
        exchange: Option<std::rc::Rc<crate::shared::peer_split::PeerExchange<'a>>>) -> Result<Self> {
        head.check_head_stream(stream)?;
        let device = head.device;
        let config = crate::shared::speculation::dflash::DflashConfig::read(snapshot)?;
        ensure!(config.hidden == cfg.hidden && config.vocab == cfg.vocab_size && config.block == admitted.block
            && config.taps.iter().all(|&tap| tap < cfg.layers), "Muse drafter does not fit the admitted target");
        let file = if crate::shared::experts::rtx::dense::is_recording() { Vec::new() } else {
            crate::shared::speculation::dflash::prefetch(snapshot).join()
                .map_err(|_| anyhow::anyhow!("Muse drafter read panicked"))??
        };
        let mask = table.host_rows(&[config.mask_token])?;
        let _scope = cuteafd_ffi::memory_ledger::scope("drafter");
        ensure!(exchange.is_some() == !admitted.tp2.is_empty(), "Muse draft split differs from admission");
        let representation = admitted.representation;
        let (devices, head_rank) = if let Some(exchange) = exchange.as_ref() {
            ensure!(exchange.device(1) == device.id && exchange.stream(1) == stream,
                "Muse drafter lead differs from the target head rank");
            (vec![exchange.device(0), exchange.device(1)], 1)
        } else {
            (vec![device.id], 0)
        };
        let arenas = drafter_arenas(device, snapshot, &devices, head_rank, representation)?;
        for rank in 0..devices.len() {
            let exchange_rank = if rank == 0 { head_rank } else { 1 - head_rank };
            let id = devices[exchange_rank];
            let arena = arenas.get(&id).context("Muse admitted drafter arena device")?;
            ensure!(*arena.layout() == admitted.arena(rank, u8::try_from(id)?)?,
                "Muse drafter arena differs from its admitted immutable recipe");
        }
        let allocate = |id, name: &str, bytes| {
            let arena = arenas.get(&id).context("Muse drafter requested an unplanned physical device")?;
            weight_allocation(Device { library: device.library, id }, Some(arena), name, bytes, ArenaTag::drafter_weights("muse/drafter"))
        };
        let drafter = device.run(|| {
            use crate::shared::fp8_linear::{Fp8Scales, Fp8Rows};
            if let Some(exchange) = exchange {
                DflashDrafter::load_tp2_on_with(device.library, programs, crate::shared::speculation::dflash::TargetProjection::Native, snapshot, file, admitted.context_slots,
                    admitted.sequences, mask, true, representation, Fp8Scales::Amax, Fp8Rows::W8a16,
                    kv_cache, exchange, head_rank, allocate)
            } else {
                DflashDrafter::load_with(device.library, programs, crate::shared::speculation::dflash::TargetProjection::Native, snapshot, file, stream, admitted.context_slots,
                    admitted.sequences, mask, true, representation, Fp8Scales::Amax, Fp8Rows::W8a16, kv_cache, allocate)
            }
        })?;
        let stored_width = head.globals.stored_head_outputs(cfg.vocab_size);
        let rows = admitted.sequences * admitted.block;
        ensure!(admitted.head_scratch == (rows * (stored_width + cfg.vocab_size) * 4) as u64,
            "Muse drafter head scratch differs from its admission");
        // The ids below and the backend's route probe ring (`RoundProbe::new` at HEAD_ROWS).
        let probe = cfg.layers * cuteafd_loader::families::muse_spark::lane::HEAD_ROWS * cfg.topk * 4;
        ensure!(admitted.runtime == (rows * 4 + probe) as u64, "Muse drafter runtime buffers differ from their admission");
        Ok(Self { drafter, stored: Allocation::new(device, rows * stored_width * 4, Tag::drafter_workspace("muse/draft-head"))?,
            trimmed: Allocation::new(device, rows * cfg.vocab_size * 4, Tag::drafter_workspace("muse/draft-head"))?,
            ids: Allocation::new(device, rows * 4, Tag::drafter_workspace("muse/draft-head"))?,
            stream, device, block: admitted.block, rows })
    }

    pub(crate) fn prepare_workspace(&self) -> Result<()> {
        self.device.run(|| self.drafter.prepare_workspace())
    }

    /// Copies the head rank's `hidden` rows `[first, first + n)` after `layer`
    /// into tap rows `to..to + n` when the layer is tapped.
    pub(crate) fn tap(&self, layer: usize, hidden: Buffer, first: usize, n: usize, to: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        self.device.run(|| self.drafter.tap_at(layer, hidden.ptr.cast_const(), first, n, to))
    }

    /// Commits tap rows as context: row `r` is position `first + r` of `slot`.
    pub(crate) fn update(&self, slot: usize, first: usize, rows: usize) -> Result<()> {
        if rows == 0 { return Ok(()); }
        ensure!(rows <= TAP_ROWS, "Muse drafter commits at most {TAP_ROWS} tap rows");
        self.device.run(|| self.drafter.update(&(0..rows).map(|r| ContextRow { tap_row: r, slot, position: first + r })
            .collect::<Vec<_>>()))
    }

    /// Commits verify-step tap rows `tap_row..tap_row + rows` of `slot` as
    /// positions `first..first + rows` (a sequence's kept rows at its step offset).
    pub(crate) fn update_rows(&self, slot: usize, tap_row: usize, first: usize, rows: usize) -> Result<()> {
        ensure!(tap_row + rows <= TAP_ROWS, "Muse drafter tap rows beyond {TAP_ROWS}");
        self.device.run(|| self.drafter.update(&(0..rows).map(|r| ContextRow { tap_row: tap_row + r, slot,
            position: first + r }).collect::<Vec<_>>()))
    }

    /// A block per sequence, through the target's embedding (peer gather) and head.
    pub(crate) fn draft(&self, cfg: &MuseSparkConfig, head: &RankResources<'a>,
        table: &crate::shared::token_io::TokenEmbedding<'_>, sequences: &[DraftSeq]) -> Result<Vec<Draft>> {
        ensure!(sequences.len() * self.block <= self.rows, "Muse draft batch beyond its admitted rows");
        let (vocab, stored) = (cfg.vocab_size,
            head.globals.stored_head_outputs(cfg.vocab_size));
        let (stored_plane, trimmed_plane, ids) = (self.stored.buffer, self.trimmed.buffer, self.ids.buffer);
        let input = |tokens: &[u32], output: Buffer, stream: *mut c_void| -> Result<()> {
            ensure!(stream == self.stream, "Muse draft input on another stream");
            table.embed(tokens, ids, 1, output, stream)
        };
        let launch = move |x: *const c_void, logits: *mut f32, rows: usize, stream: *mut c_void| -> Result<()> {
            ensure!(stream == self.stream && rows <= self.rows, "Muse draft head on another stream or too wide");
            let device = self.device.id;
            let input = Buffer { ptr: x.cast_mut(), bytes: rows * cfg.hidden * 2, device_id: device, flags: 0 };
            let staged = Buffer { bytes: rows * stored * 4, ..stored_plane };
            let trimmed = Buffer { bytes: rows * vocab * 4, ..trimmed_plane };
            let out = Buffer { ptr: logits.cast(), bytes: rows * vocab * 4, device_id: device, flags: 0 };
            super::forward::project_head(cfg, head, rows, input, staged, trimmed, out, stream)
        };
        self.device.run(|| self.drafter.draft_device_with(sequences, &input, TargetHead::Launch(Box::new(launch))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::receipts::{Census, Inventory};
    use cuteafd_core::memory_layout::Category;
    use cuteafd_loader::{families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only,
        placement::planner::{CoordinatorDevices, Lifetime, Phase, PhaseOrder}};

    #[test]
    fn installed_muse_drafter_arenas_follow_physical_devices_and_retain_views() -> Result<()> {
        let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/huggingface"));
        let model = home.join("hub/models--nvidia--Muse-Spark-Open-816B-A42B-DFlash2");
        let revision = std::fs::read_to_string(model.join("refs/main"))?;
        let snapshot = model.join("snapshots").join(revision.trim());
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.extent_only(&library)?;
        fixture.strict_device_free(&library, true)?;
        for id in 0..2 { fixture.configure_memory(&library, id, 96usize << 30)?; }
        let target = MuseSparkConfig::from_hf(&cuteafd_loader::plan::testing::muse_spark_config())?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let draft = cuteafd_loader::placement::families::muse_spark::drafter_inventory(
            &target, &config, 16, 16, 202_112, 188,
            cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8)?;
        for devices in [vec![1], vec![0, 1], vec![1, 0]] {
            let admitted = if devices.len() == 2 {
                draft.clone().with_tp2(&snapshot, &[188; 2],
                    cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8)?
            } else { draft.clone() };
            let head_rank = devices.len() - 1;
            let head = Device { library: &library, id: devices[head_rank] };
            let requests = DflashDrafter::arena_requests(&snapshot, Bf16Only, &devices, head_rank)?;
            let arenas = drafter_arenas(head, &snapshot, &devices, head_rank, Bf16Only)?;
            let mut views = Vec::new();
            let mut census = Census::default();
            let topology = CoordinatorDevices::new(&devices)?;
            for (&id, requests) in &requests {
                let arena = arenas.get(&id).context("missing Muse drafter arena")?;
                let rank = if id == devices[head_rank] { 0 } else { 1 };
                let planned = admitted.arena(rank, u8::try_from(id)?)?;
                assert_eq!(arena.layout(), &planned, "planner/runtime exact views and offsets");
                let expected = match (devices.len(), rank) {
                    (1, 0) => (8_768_326_144usize, 1_963_520usize),
                    (2, 0) => (4_993_452_544, 1_963_520),
                    (2, 1) => (4_782_361_088, 1_242_624),
                    _ => unreachable!(),
                };
                assert_eq!((planned.bytes(), planned.bytes().div_ceil(2 << 20) * (2 << 20) - planned.bytes()), expected);
                let device = Device { library: &library, id };
                let mut inventory = Inventory::new(topology, &format!("muse.draft.{id}"),
                    Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "Muse drafter");
                for (name, bytes) in requests {
                    let view = weight_allocation(device, Some(arena), name, *bytes, crate::shared::memory::TEST_ARENA)?;
                    assert_eq!(view.buffer.device_id, id);
                    assert_eq!(view.buffer.bytes, *bytes);
                    assert_eq!(view.buffer.ptr as usize % 256, 0);
                    view.inventory(&mut inventory, name)?;
                    views.push(view);
                }
                assert!(weight_allocation(Device { library: &library, id: 1 - id },
                    Some(arena), &requests[0].0, requests[0].1, crate::shared::memory::TEST_ARENA).is_err());
                let payload = requests.iter().map(|(_, bytes)| bytes).sum::<usize>();
                let large = requests.iter().filter(|(_, bytes)| *bytes > 1 << 20).collect::<Vec<_>>();
                let old_tail = large.iter().map(|(_, bytes)| bytes.div_ceil(2 << 20) * (2 << 20) - bytes).sum::<usize>();
                let bytes = arena.layout().bytes();
                println!("Muse drafter TP{} GPU{id}: N={} payload={payload} large_N={} large_bytes={} old_tail={old_tail} arena={bytes} padding={} tail={}",
                    devices.len(), requests.len(), large.len(), large.iter().map(|(_, bytes)| bytes).sum::<usize>(),
                    arena.layout().padding_bytes(), bytes.div_ceil(2 << 20) * (2 << 20) - bytes);
                census.owners(inventory.receipts);
            }
            drop(arenas);
            assert!(views.iter().all(|view| view.weight_backing().is_some()));
            census.normalize(PhaseOrder::TARGET_THEN_DRAFT)?;
            assert_eq!(census.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).count(), devices.len());
            library.cuda_set_device(0)?;
            drop(views);
            assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
        }
        Ok(())
    }
}
