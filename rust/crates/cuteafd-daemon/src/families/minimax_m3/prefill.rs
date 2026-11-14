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
    pub taps: Option<&'s mut LayerTaps>,
}

/// Diagnostic-only synchronous copies; never enabled by serving or ordinary golden.
/// One admitted host buffer is reused after draining the issuing stream.
pub(crate) struct LayerTaps {
    root: std::path::PathBuf,
    layers: std::collections::BTreeSet<usize>,
    bytes: Vec<u8>,
    pub residual_only: bool,
}
impl LayerTaps {
    pub fn host_bytes(rows: u32) -> u64 { u64::from(rows) * 6144 * 4 }
    pub fn new(root: std::path::PathBuf, layers: &[usize], rows: u32) -> Result<Self> {
        ensure!(!layers.is_empty() && layers.iter().all(|&layer| layer < 60), "MiniMax tap layers outside decoder");
        std::fs::create_dir(&root)?;
        Ok(Self { root, layers: layers.iter().copied().collect(),
            bytes: vec![0; usize::try_from(Self::host_bytes(rows))?], residual_only: false })
    }
    pub fn write(&mut self, layer: usize, stage: &str, plane: &str, width: usize,
        unit: &Unit, owner: &UnitOwner<'_>) -> Result<()> {
        self.buffer(layer, stage, owner.plane(plane)?, width, unit, owner)
    }
    pub fn buffer(&mut self, layer: usize, stage: &str, buffer: Buffer, width: usize,
        unit: &Unit, owner: &UnitOwner<'_>) -> Result<()> {
        let count = unit.rows.len().checked_mul(6144 * width).context("MiniMax tap extent")?;
        self.extent(layer, stage, buffer, count, unit, owner)
    }
    pub fn extent(&mut self, layer: usize, stage: &str, buffer: Buffer, count: usize,
        unit: &Unit, owner: &UnitOwner<'_>) -> Result<()> {
        if !self.layers.contains(&layer) || (self.residual_only && stage != "residual") { return Ok(()); }
        ensure!(count <= self.bytes.len() && count <= buffer.bytes, "MiniMax tap exceeds admitted host buffer");
        owner.stream.drain()?;
        owner.stream.device.run(|| owner.stream.device.library.copy_d2h(
            &mut self.bytes[..count], buffer))?;
        self.write_bytes(layer, stage, unit.rows.start, &self.bytes[..count])
    }
    pub fn router(&mut self, layer: usize, weights: &TargetWeights<'_>, unit: &Unit,
        owner: &UnitOwner<'_>) -> Result<()> {
        for (name, bytes) in [("logits", unit.rows.len() * 128 * 4),
            ("route_ids", unit.rows.len() * 4 * 4), ("route_weights", unit.rows.len() * 4 * 4)] {
            self.extent(layer, &format!("router.{name}"), owner.plane(name)?, bytes, unit, owner)?;
        }
        let bias = weights.operand(&format!("language_model.model.layers.{layer}.block_sparse_moe.e_score_correction_bias"))?.buffer;
        self.extent(layer, "router.bias", bias, 128 * 4, unit, owner)
    }
    pub fn ffn_inputs(&mut self, layer: usize, prefix: &str, shared: bool,
        admitted: &CoordinatorWeights, unit: &Unit, owner: &UnitOwner<'_>) -> Result<()> {
        let stem = if shared { "block_sparse_moe.shared_experts" } else { "mlp" };
        let name = format!("language_model.model.layers.{layer}.{stem}.down_proj");
        let input = admitted.projections.iter().find(|p| p.stem == name)
            .context("MiniMax diagnostic down projection missing")?.input;
        let bytes = unit.rows.len() * input * 2;
        for plane in ["gate", "up"] {
            self.extent(layer, &format!("{prefix}{plane}.bf16"), owner.plane(plane)?, bytes, unit, owner)?;
        }
        self.extent(layer, &format!("{prefix}oai.bf16"), owner.dense.producer_destination(), bytes, unit, owner)?;
        let records = unit.rows.len().div_ceil(16) * (input / 32) * 528;
        self.extent(layer, &format!("{prefix}down.records"), owner.dense.workspace.records.buffer,
            records, unit, owner)
    }
    pub fn final_norm(&mut self, head: &super::head::Head<'_>, first: usize, rows: usize) -> Result<()> {
        let count = rows.checked_mul(6144 * 2).context("MiniMax final norm tap extent")?;
        ensure!(count <= self.bytes.len(), "MiniMax final norm tap exceeds admitted host buffer");
        head.copy_normalized(&mut self.bytes[..count])?;
        self.write_bytes(60, "final_norm.bf16", first, &self.bytes[..count])
    }
    fn write_bytes(&self, layer: usize, stage: &str, first: usize, bytes: &[u8]) -> Result<()> {
        use std::io::Write;
        let path = self.root.join(format!("layer{layer:02}.{stage}.row{first:06}.bin"));
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(bytes)?;
        Ok(())
    }
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
        if let Some(taps) = &mut self.taps {
            taps.write(layer, "input", "residual", 4, unit, &self.owners[unit.slot])?;
        }
        let request = self.cache.request(&self.metadata[unit.slot], layer)?;
        // SAFETY: cache/metadata owners retain bounded physical IDs and all
        // buffers; the shared executor orders earlier units at this layer.
        unsafe { self.owners[unit.slot].attention(self.kernels, self.weights, self.admitted, layer, request)?; }
        if let Some(taps) = &mut self.taps {
            taps.write(layer, "attention_projected.bf16", "ffn", 2, unit, &self.owners[unit.slot])?;
            taps.write(layer, "post_attention", "residual", 4, unit, &self.owners[unit.slot])?;
        }
        Ok(())
    }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        self.owners[unit.slot].ffn(self.kernels, self.weights, self.admitted, layer, unit.rows.len(), false)?;
        if let Some(taps) = &mut self.taps {
            taps.ffn_inputs(layer, "", false, self.admitted, unit, &self.owners[unit.slot])?;
            taps.write(layer, "mlp.bf16", "ffn", 2, unit, &self.owners[unit.slot])?;
            taps.write(layer, "residual", "residual", 4, unit, &self.owners[unit.slot])?;
        }
        Ok(())
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<WaveTicket> {
        let rows = unit.rows.len();
        let owner = &mut self.owners[unit.slot];
        owner.router(self.kernels, self.weights, layer, rows)?;
        if let Some(taps) = &mut self.taps { taps.router(layer, self.weights, unit, owner)?; }
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
    fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: WaveTicket) -> Result<()> {
        let slot = ticket.slot();
        let owner = &self.owners[unit.slot];
        owner.stream.device.run(|| {
            self.waves.land(ticket, owner.stream.raw)?;
            // SAFETY: the shared reducer reads the landed slot and writes a
            // distinct unit-owned BF16 plane. Shared addition is a later round.
            unsafe { self.waves.reduce(slot, std::ptr::null(), owner.plane("routed")?.ptr.cast(),
                unit.rows.len(), owner.stream.raw) }
        })?;
        owner.join_routed(self.kernels, unit.rows.len())?;
        if let Some(taps) = &mut self.taps {
            taps.write(layer, "residual", "residual", 4, unit, owner)?;
        }
        Ok(())
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
    fn layer_taps_are_bounded_and_never_overwrite() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let taps = LayerTaps::new(directory.path().join("taps"), &[0, 1, 2, 3], 256)?;
        assert_eq!(taps.bytes.len() as u64, LayerTaps::host_bytes(256));
        assert_eq!(taps.bytes.len(), 256 * 6144 * 4);
        taps.write_bytes(0, "residual", 64, &[1, 2, 3, 4])?;
        assert_eq!(std::fs::read(directory.path().join("taps/layer00.residual.row000064.bin"))?, [1, 2, 3, 4]);
        assert!(taps.write_bytes(0, "residual", 64, &[9]).is_err());
        assert!(LayerTaps::new(directory.path().join("bad"), &[60], 256).is_err());
        Ok(())
    }
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
