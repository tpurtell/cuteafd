//! Native whole-layer execution on shared device-ordered residual hops.
use super::*;
use crate::shared::peer_split::{hop::HopLink, RankDevice};
use cuteafd_loader::placement::{plan_hops, HopPoint};

pub(super) struct Peer<'a> {
    pub stream: Stream<'a>,
    pub graphs: graphs::Graphs<'a>,
    pub programs: &'a Programs<'a>,
    pub planes: Planes,
    pub _planes: Vec<Allocation<'a>>,
    pub bank: GdnBank<'a>,
    pub prefix: Option<std::rc::Rc<Allocation<'a>>>,
    pub mark: std::rc::Rc<Allocation<'a>>,
    pub hops: HopLink<'a>,
    pub boundary: usize,
}

impl<'a> DenseEngine<'a> {
    pub(crate) fn devices(&self) -> Vec<i32> {
        std::iter::once(self.stream.device.id).chain(self.peer.iter().map(|peer| peer.stream.device.id)).collect()
    }

    pub(super) fn drain(&self) -> Result<()> {
        // The primary stream joins the return hop before exposing any output.
        self.stream.drain()?;
        if let Some(peer) = &self.peer { peer.stream.drain()?; }
        Ok(())
    }

    pub(super) fn state_regions(&self, slot: usize) -> Vec<CuteafdDeviceBuffer> {
        self.homes.layers.iter().filter_map(|home| home.gdn_ordinal.map(|ordinal| (home.owner, ordinal)))
            .flat_map(|(owner, ordinal)| self.owner_bank(owner).slot_regions(ordinal, slot)).collect()
    }

    pub(super) fn owner_bank(&self, rank: usize) -> &GdnBank<'a> {
        if rank == 0 { &self.bank } else { &self.peer.as_ref().expect("admitted peer owner").bank }
    }

    pub(super) fn owner_stream(&self, rank: usize) -> &Stream<'a> {
        if rank == 0 { &self.stream } else { &self.peer.as_ref().expect("admitted peer owner").stream }
    }

    pub(super) fn owner_planes(&self, rank: usize) -> Planes {
        if rank == 0 { self.planes } else { self.peer.as_ref().expect("admitted peer owner").planes }
    }

    pub(super) fn owner_programs(&self, rank: usize) -> &Programs<'a> {
        if rank == 0 { self.programs } else { self.peer.as_ref().expect("admitted peer owner").programs }
    }

    pub(super) fn layer_blocks(&self, index: usize) -> Blocks<'_, 'a> {
        let rank = self.homes.layers[index].owner;
        Blocks { programs: self.owner_programs(rank), admitted: &self.admitted, cfg: &self.cfg,
            homes: &self.homes, bank: self.owner_bank(rank), stream: self.owner_stream(rank).raw }
    }

    pub(super) fn enter_layer(&self, index: usize, rows: usize) -> Result<()> {
        if let Some(peer) = &self.peer {
            if index == peer.boundary {
                let ticket = peer.hops.send_into(0, 0, self.planes.residual.ptr, rows, Some(peer.planes.residual))?;
                peer.hops.land(ticket)?;
            }
        }
        Ok(())
    }

    pub(super) fn finish_layers(&self, rows: usize) -> Result<()> {
        if let Some(peer) = &self.peer {
            let ticket = peer.hops.send_into(1, 0, peer.planes.residual.ptr, rows, Some(self.planes.residual))?;
            peer.hops.land(ticket)?;
        }
        Ok(())
    }

    /// dSpark and MTP borrow the primary embedding/head stream. A post-block
    /// tap on the peer uses its admitted hidden scratch and an event join.
    pub(super) fn primary_tap(&self, index: usize, rows: usize) -> Result<CuteafdDeviceBuffer> {
        let owner = self.homes.layers[index].owner;
        let src = crate::shared::prefix::view(self.owner_planes(owner).residual, 0, rows * self.cfg.hidden * 2)?;
        if owner == 0 { return Ok(src); }
        let dst = crate::shared::prefix::view(self.planes.norm, 0, src.bytes)?;
        let peer = self.peer.as_ref().context("dense tap peer")?;
        let ticket = peer.hops.send_into(1, 0, src.ptr, rows, Some(dst))?;
        peer.hops.land(ticket)?;
        Ok(dst)
    }
}

pub(super) fn make_peer<'a>(device: Device<'a>, programs: &'a Programs<'a>, cfg: &Qwen35Config,
    admitted: &DensePrograms, buffers: &NativeBuffers, homes: &LayerStateMap,
    primary: &Stream<'a>, owners: &[u8], slots: usize) -> Result<Peer<'a>> {
    device.run(|| {
        let stream = Stream::new(device)?;
        let allocations = buffers.planes.into_iter().map(|bytes|
            Allocation::new(device, bytes, Tag::workspace("qwen35/peer-planes"))).collect::<Result<Vec<_>>>()?;
        let views = allocations.iter().map(|allocation| allocation.buffer).collect::<Vec<_>>();
        let planes = Planes { residual: views[0], norm: views[1], delta: views[2], query: views[3],
            gate: views[4], attention: views[5], scratch: views[6], slots: views[7], seq_first: views[8],
            rope_positions: views[9], kv_slots: views[10], positions: views[11], page_table: views[12] };
        let commit = if admitted.prefix == "qwen35" { "qwen35_gdn_commit" } else { "qwen38_gdn_commit" };
        let bank = GdnBank::new(device.library, programs, stream.raw, cfg.gdn, homes.gdn_layers[1], slots, commit)?;
        let prefix = if buffers.prefix == 0 { None } else { Some(std::rc::Rc::new(
            Allocation::new(device, buffers.prefix, Tag::prefix("qwen35/peer-prefix"))?)) };
        let mark = std::rc::Rc::new(Allocation::new(device, buffers.verify[0], Tag::kv("qwen35/peer-verify-mark"))?);
        let modes = owners.iter().map(|&gpu| cuteafd_loader::placement::LayerMode::Whole {
            gpu, ffn: cuteafd_loader::placement::FfnMode::Owner }).collect::<Vec<_>>();
        let spec = qwen3_5::dual::hop_spec(cfg);
        let hops = plan_hops(&modes, &spec)?;
        ensure!(hops.len() == 2 && matches!(hops[0].at, HopPoint::BeforeLayer(_)) && hops[1].at == HopPoint::Exit,
            "dense layer ranges require one boundary and one return hop");
        let boundary = owners.iter().position(|&rank| rank == 1).context("dense peer range")?;
        let link = primary.device.run(|| HopLink::new(device.library,
            [RankDevice { device: primary.device.id, stream: primary.raw }, RankDevice { device: device.id, stream: stream.raw }],
            &hops, spec, HEAD_ROWS * cfg.hidden * 2))?;
        let variant = format!("{}-layers-rtx2-rank1", cuteafd_loader::serving_capacity::qwen35_graphs::variant(
            cfg, admitted.format.fp8(), admitted.kv_format));
        let graphs = graphs::Graphs::new(device, variant)?;
        Ok(Peer { stream, graphs, programs, planes, _planes: allocations, bank, prefix, mark, hops: link, boundary })
    })
}
