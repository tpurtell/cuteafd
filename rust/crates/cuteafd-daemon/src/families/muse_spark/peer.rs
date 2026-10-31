//! Muse TP2's exchange over the shared two-GPU `PeerExchange`: one receive
//! slot per [`Channel`] on each GPU, sized by that channel's widest payload at
//! the admitted rows (`tp2::slot_bytes`; one slot each, as the interleaving
//! model in `cuteafd_loader::families::muse_spark::tp2` proves). A channel's
//! slot index is its position in [`Channel::ALL`], so a push on one channel
//! never touches another channel's slot or flag.
use super::device::PeerExchange as RankExchange;
use crate::shared::peer_split::{PeerExchange, RankDevice};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{peer_exchange::PeerPlane, CuteafdDeviceBuffer as Buffer, NativeLibrary};
use cuteafd_loader::families::muse_spark::tp2::{receive_slots, slot_bytes, Channel};

/// Slot index of `channel` (one slot per channel).
pub(crate) fn slot(channel: Channel) -> usize {
    Channel::ALL.iter().position(|c| *c == channel).expect("every channel is listed")
}

/// Per-slot receive bytes of the pair, in [`Channel::ALL`] order. `Partials`
/// is sized only when a layer onboards; otherwise it keeps a 16-byte slot so
/// every channel's index (and flag) stays fixed.
pub(crate) fn slot_sizes(hidden: usize, expert: usize, topk: usize, rows: usize, onboarded: bool) -> Result<Vec<usize>> {
    Channel::ALL.iter().map(|&channel| {
        ensure!(receive_slots(channel) == 1, "Muse TP2 channel {channel:?} needs one slot per GPU");
        if channel == Channel::Partials && !onboarded { return Ok(16); }
        slot_bytes(channel, hidden, expert, topk, rows)
    }).collect()
}

/// The pair's exchange (both ends); each rank uses it through [`RankEnd`].
pub(crate) struct MusePeer<'a> {
    exchange: PeerExchange<'a>,
    sizes: Vec<usize>,
}

impl<'a> MusePeer<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, ranks: [RankDevice; 2], sizes: Vec<usize>) -> Result<Self> {
        ensure!(sizes.len() == Channel::ALL.len(), "Muse TP2 needs one slot size per channel");
        let exchange = PeerExchange::new_abortable_sized(library, ranks, &sizes)?;
        Ok(Self { exchange, sizes })
    }

    pub(crate) fn rank(&self, rank: usize) -> RankEnd<'_, 'a> { RankEnd { peer: self, rank } }

    pub(crate) fn exchange(&self) -> &PeerExchange<'a> { &self.exchange }
}

/// One rank's end of the pair's exchange.
pub(crate) struct RankEnd<'p, 'a> { peer: &'p MusePeer<'a>, rank: usize }

impl RankEnd<'_, '_> {
    fn received(&self, channel: Channel, bytes: usize) -> Result<Buffer> {
        let at = slot(channel);
        ensure!(bytes <= self.peer.sizes[at], "Muse TP2 {channel:?} payload of {bytes} bytes past its slot");
        Ok(Buffer { ptr: self.peer.exchange.recv(self.rank, at)?, bytes,
            device_id: self.peer.exchange.device(self.rank), flags: 0 })
    }
}

impl RankExchange for RankEnd<'_, '_> {
    fn swap(&mut self, channel: Channel, local: Buffer) -> Result<Buffer> {
        let at = slot(channel);
        ensure!(local.bytes <= self.peer.sizes[at], "Muse TP2 {channel:?} swap past its slot");
        self.peer.exchange.push(self.rank, at, local.ptr.cast_const(), local.bytes)?;
        self.peer.exchange.wait(self.rank, at)?;
        self.received(channel, local.bytes)
    }

    fn send(&mut self, channel: Channel, source: Buffer, rows: usize, row_bytes: usize, pitch: usize) -> Result<()> {
        let at = slot(channel);
        let bytes = rows.checked_mul(row_bytes).context("Muse TP2 send extent")?;
        ensure!(bytes <= self.peer.sizes[at] && row_bytes <= pitch && row_bytes % 16 == 0 && pitch % 16 == 0,
            "Muse TP2 {channel:?} send geometry");
        if row_bytes == pitch {
            return self.peer.exchange.push(self.rank, at, source.ptr.cast_const(), bytes);
        }
        let destination = self.peer.exchange.recv(1 - self.rank, at)?;
        self.peer.exchange.push_planes(self.rank, at, &[PeerPlane { destination, source: source.ptr.cast_const(),
            rows: rows as u64, row_bytes: row_bytes as u64, destination_pitch: row_bytes as u64,
            source_pitch: pitch as u64 }])
    }

    fn receive(&mut self, channel: Channel, bytes: usize) -> Result<Buffer> {
        self.peer.exchange.wait(self.rank, slot(channel))?;
        self.received(channel, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_slot_per_channel_sized_by_its_payload() -> Result<()> {
        let sizes = slot_sizes(8192, 4096, 8, 2752, false)?;
        assert_eq!(sizes, [2752 * 8192 * 2, 2752 * 4096, 16, 2752 * 4096, 2752 * 8192 * 2]);
        crate::shared::peer_split::validate_slot_sizes([0, 1], &sizes)?;
        let onboarded = slot_sizes(8192, 4096, 8, 2752, true)?;
        assert_eq!(onboarded[slot(Channel::Partials)], 2752 * 8 * 4096 * 2);
        assert_eq!((0..5).map(|i| slot(Channel::ALL[i])).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        Ok(())
    }

    /// The runtime's slots are what admission charged: the fixed channels as
    /// "Muse TP2 exchange slots", `Partials` as the peer half of `tp2_workspace`.
    #[test]
    fn runtime_slots_equal_the_admitted_exchange() -> Result<()> {
        use cuteafd_loader::placement::families::muse_spark::{tp2_exchange_bytes, tp2_partial_bytes};
        let cfg = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
            &cuteafd_loader::plan::testing::muse_spark_config())?;
        for unit in [64, 2752, 4096] {
            let fixed = slot_sizes(cfg.hidden, cfg.expert_hidden, cfg.topk, unit, false)?;
            assert_eq!(fixed.iter().sum::<usize>() - 16, tp2_exchange_bytes(&cfg, unit)?);
            let onboarded = slot_sizes(cfg.hidden, cfg.expert_hidden, cfg.topk, unit, true)?;
            assert_eq!(2 * onboarded[slot(Channel::Partials)], tp2_partial_bytes(&cfg, unit));
        }
        Ok(())
    }
}
