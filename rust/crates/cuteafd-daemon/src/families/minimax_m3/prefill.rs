//! MiniMax arithmetic hooks on the shared causal unit executor and Spark waves.
use super::{cache::{Cache, Metadata}, layer::{Kernels, UnitOwner}, weights::TargetWeights};
use crate::shared::{experts::staging::RouteStaging,
    prefill_pipeline::{run_prefill_units, FfnRoute, PrefillHooks, Unit},
    spark_waves::{self, WaveTicket, Waves}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::minimax_m3::{MiniMaxM3Config, weight_layout::CoordinatorWeights};
use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
    expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};

/// Input/output residual spans are caller-admitted FP32 rows. Embedding and head
/// stay outside this arithmetic adapter, never a private layer or unit loop.
pub(crate) struct Prefill<'s, 'a> {
    pub cfg: &'s MiniMaxM3Config,
    pub admitted: &'s CoordinatorWeights,
    pub weights: &'s TargetWeights<'a>,
    pub kernels: &'s Kernels<'a>,
    pub cache: &'s Cache<'a>,
    pub owners: &'s mut [UnitOwner<'a>],
    pub metadata: &'s mut [Metadata<'a>],
    pub waves: &'s mut Waves<'a>,
    pub residual: Buffer,
    pub placement_version: u64,
    /// Monotonic nonzero request identity, retained across scoring windows.
    pub next_identity: &'s mut u64,
}
impl Prefill<'_, '_> {
    pub fn run(&mut self, units: &[Unit]) -> Result<()> {
        ensure!(!units.is_empty() && !self.owners.is_empty()
            && self.metadata.len() == self.owners.len() && self.placement_version != 0,
            "MiniMax prefill owners/placement");
        let rows = units.last().context("MiniMax prefill has no units")?.rows.end;
        ensure!(self.residual.bytes >= rows.checked_mul(6144 * 4).context("MiniMax residual extent")?
            && units.iter().all(|unit| unit.sequence == 0 && unit.segments.is_empty()
                && unit.slot < self.owners.len() && unit.layers == (0..self.cfg.layers)),
            "MiniMax cold prefill units differ from admitted owner geometry");
        let stream = &self.owners[0].stream;
        ensure!(self.owners.iter().all(|owner| std::rc::Rc::ptr_eq(&owner.stream, stream)),
            "MiniMax prefill requires one ordered issuing stream");
        let waves = self.owners.len();
        ensure!(waves <= self.waves.waves(), "MiniMax unit slots exceed the Spark exchange");
        let result = run_prefill_units(self, units, waves);
        // The caller owns terminal Spark shutdown on error before freeing owners;
        // do not wait here on a device counter a failed exchange must release.
        if result.is_ok() {
            self.owners[0].stream.drain()?;
            spark_waves::healthy()?;
        }
        result
    }
    fn slice(&self, unit: &Unit) -> Result<Buffer> {
        let offset = unit.rows.start.checked_mul(6144 * 4).context("MiniMax unit offset")?;
        let bytes = unit.rows.len().checked_mul(6144 * 4).context("MiniMax unit bytes")?;
        ensure!(offset.checked_add(bytes).is_some_and(|end| end <= self.residual.bytes),
            "MiniMax unit residual slice exceeds span");
        Ok(Buffer { ptr: self.residual.ptr.wrapping_byte_add(offset), bytes, ..self.residual })
    }
}
impl PrefillHooks for Prefill<'_, '_> {
    type Ticket = WaveTicket;
    fn route(&self, layer: usize) -> FfnRoute {
        if self.cfg.moe[layer] { FfnRoute::Spark } else { FfnRoute::Local }
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        self.cache.prepare(&mut self.metadata[unit.slot], unit.rows.clone())?;
        let owner = &self.owners[unit.slot];
        let input = self.slice(unit)?;
        owner.stream.device.run(|| {
            // SAFETY: distinct admitted residual span and slot storage; one
            // stream retains both until the shared executor completes the call.
            unsafe { owner.stream.device.library.copy_d2d_async(owner.plane("residual")?, input,
                unit.rows.len() * 6144 * 4, owner.stream.raw) }
        })
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let request = self.cache.request(&self.metadata[unit.slot], layer)?;
        // SAFETY: cache/metadata owners retain bounded physical IDs and all
        // buffers; the shared executor orders earlier units at this layer.
        unsafe { self.owners[unit.slot].attention(self.kernels, self.weights, self.admitted, layer, request) }
    }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        self.owners[unit.slot].ffn(self.kernels, self.weights, self.admitted, layer, unit.rows.len(), false)
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<WaveTicket> {
        let rows = unit.rows.len();
        let owner = &mut self.owners[unit.slot];
        owner.router(self.kernels, self.weights, layer, rows)?;
        let identity = *self.next_identity;
        ensure!(identity != 0, "MiniMax Spark request identity must be nonzero");
        *self.next_identity = identity.checked_add(1).context("MiniMax Spark request identity exhausted")?;
        let version = self.placement_version;
        let start = unit.rows.start;
        let staging = RouteStaging::new(rows, 4, 6144 * 2)?;
        let sources = [owner.plane("route_ids")?, owner.plane("route_weights")?, owner.dense.producer_destination()];
        let ticket = owner.stream.device.run(|| {
            // SAFETY: router ids/weights and its normalized BF16 activation are
            // staged on the same stream before shared FFN reuses the activation.
            unsafe { self.waves.send(rows, rows * 6144 * 2, staging, sources, owner.stream.raw,
                spark_waves::single_reply_eligible(rows, ExpertV2SourceKind::Prefill),
                Box::new(move |routes, wire| request(identity, version, layer, start, rows, routes, wire))) }
        })?;
        owner.ffn(self.kernels, self.weights, self.admitted, layer, rows, true)?;
        Ok(ticket)
    }
    fn land_experts(&mut self, _: usize, unit: &Unit, ticket: WaveTicket) -> Result<()> {
        let slot = ticket.slot();
        let owner = &self.owners[unit.slot];
        owner.stream.device.run(|| {
            self.waves.land(ticket, owner.stream.raw)?;
            // SAFETY: the shared reducer reads the landed slot and writes a
            // distinct unit-owned BF16 plane. Shared addition is a later round.
            unsafe { self.waves.reduce(slot, std::ptr::null(), owner.plane("routed")?.ptr.cast(),
                unit.rows.len(), owner.stream.raw) }
        })?;
        owner.join_routed(self.kernels, unit.rows.len())
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let destination = self.slice(unit)?;
        owner.stream.device.run(|| {
            // SAFETY: disjoint slot/span slices retained through stream drain.
            unsafe { owner.stream.device.library.copy_d2d_async(destination, owner.plane("residual")?,
                unit.rows.len() * 6144 * 4, owner.stream.raw) }
        })
    }
    fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
}

fn request(identity: u64, version: u64, layer: usize, start: usize, rows: usize,
    routes: Vec<ExpertProtocolV2RouteEntry>, wire: bytes::Bytes) -> Result<ExpertProtocolV2Request> {
    ensure!(identity != 0 && version != 0 && (3..60).contains(&layer), "MiniMax Spark request identity/layer");
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: (start + row) as u64, source_kind: ExpertV2SourceKind::Prefill,
        source_request_id: 1, token_position: (start + row) as u64,
        route_offset: (row * 4) as u32, route_count: 4,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(identity, version, layer as u32, 6144,
        ExpertV2Dtype::Bf16, descriptors, routes, wire)?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn requests_retain_causal_positions_and_checkpoint_native_bf16_ingress() -> Result<()> {
        let routes = (0..8).map(|route| ExpertProtocolV2RouteEntry {
            row_index: route / 4, expert_id: route % 128, gate_weight: 0.5,
        }).collect();
        let request = request(8, 77, 3, 63, 2, routes, vec![0; 2 * 6144 * 2].into())?;
        assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Bf16);
        assert_eq!(request.header.hidden_row_stride_bytes, 6144 * 2);
        assert_ne!(request.header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, 0);
        assert_eq!(request.rows[0].token_position, 63);
        assert_eq!(request.rows[1].route_offset, 4);
        assert_eq!(request.rows[1].route_count, 4);
        Ok(())
    }
    #[test]
    fn forward_uses_shared_executor_and_wave_exchange_without_a_private_layer_loop() {
        let source = include_str!("prefill.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("run_prefill_units(self, units,"));
        assert!(source.contains("self.waves.send("));
        assert!(source.contains("self.waves.reduce("));
        assert!(!source.contains("for layer in"));
    }
}
