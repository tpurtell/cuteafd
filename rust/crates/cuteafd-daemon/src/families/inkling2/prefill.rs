//! Family arithmetic on the shared causal unit executor; no private schedule.
use super::{experts::Dispatcher, graphs::{self, DecodeGraphs, Mode, Section}, layers::{self, Kernels}, owners::UnitBuffers, weights::TargetWeights};
use crate::shared::prefill_pipeline::{run_prefill_units, sequence_units, FfnRoute, PrefillHooks, Unit};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

/// Persistent whole-rank prefill over shared Spark waves. Each in-flight unit
/// has an admitted scratch owner; global pages and causal histories are shared
/// request owners, ordered by run_prefill_units at every layer.
pub(crate) trait UnitIo<'a> {
    fn begin(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()>;
    fn end(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>) -> Result<()>;
    fn tap(&mut self,_layer:usize,_unit:&Unit,_owner:&super::owners::UnitOwner<'a>)->Result<()> {Ok(())}
    fn stage(&mut self,_layer:usize,_stage:&str,_unit:&Unit,_owner:&super::owners::UnitOwner<'a>)->Result<()> {Ok(())}
}
/// Which rows of `head` segments anybody reads. Spans are the head's 64-row
/// tiles; a span is projected through the vocabulary only when it holds such a
/// row, so its GEMM (and the bytes of every returned row) matches projecting
/// every span. Drafter final-norm taps still normalize every span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadRows {
    /// Every row of every head segment (decode, verify).
    All,
    /// Each head segment's last row (serving prefill, packed prefill).
    MemberLast,
    /// Rows at 1-based request position `>= first` (streamed prefill scoring).
    FromPosition(usize),
}

/// Per 64-row head span: whether its vocabulary projection is read.
pub(crate) fn projected_spans(rows: usize, members: &[crate::shared::prefill_pipeline::UnitSegment],
    head_rows: HeadRows) -> Result<Vec<(std::ops::Range<usize>, bool)>> {
    let needed = members.iter().filter(|member| member.head && !member.rows.is_empty()).filter_map(|member| {
        let rows = match head_rows {
            HeadRows::All => member.rows.clone(),
            HeadRows::MemberLast => member.rows.end-1..member.rows.end,
            HeadRows::FromPosition(first) => {
                // Row r holds 1-based position member.tokens.start + (r - member.rows.start) + 1.
                let skip = first.saturating_sub(member.tokens.start+1);
                member.rows.start.saturating_add(skip).min(member.rows.end)..member.rows.end
            }
        };
        (!rows.is_empty()).then_some(rows)
    }).collect::<Vec<_>>();
    Ok(crate::shared::probe::head_spans(rows, 64)?.map(|span| {
        let project = needed.iter().any(|rows| rows.start < span.end && span.start < rows.end);
        (span, project)
    }).collect())
}

/// Token/head edge for ordinary prefill and bounded streamed scoring. The
/// callback consumes scratch logits before another head tile overwrites them.
pub(crate) struct TokenIo<'s, 'a> {
    pub kernels: &'s Kernels<'a>,
    pub head: &'s super::head::Head<'a>,
    pub peer_head: Option<super::head::PeerHead<'s, 'a>>,
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s TargetWeights<'a>,
    pub cfg: &'s Inkling2Config,
    /// Submission-local token streams in the shared unit sequence order.
    pub tokens: &'s [&'s [u32]],
    pub starts: &'s [usize],
    pub slots: &'s [usize],
    pub retained_rows: bool,
    pub head_rows: HeadRows,
    pub draft:Option<&'s super::draft::Binding<'a>>,
    pub publish_draft:bool,
    pub trace:Option<&'s std::cell::RefCell<super::width_invariance::Trace>>,
    pub media: &'s [Option<cuteafd_engine::media::RequestMedia>],
    pub on_logits: &'s mut dyn FnMut(usize, usize, crate::shared::token_io::DeviceLogits) -> Result<()>,
}
impl<'a> UnitIo<'a> for TokenIo<'_, 'a> {
    fn stage(&mut self,layer:usize,stage:&str,unit:&Unit,owner:&super::owners::UnitOwner<'a>)->Result<()>{
        if let Some(trace)=self.trace{trace.borrow_mut().capture(layer,stage,unit.rows.len(),owner)?;}Ok(())
    }
    fn tap(&mut self,layer:usize,unit:&Unit,owner:&super::owners::UnitOwner<'a>)->Result<()> {
        if let Some(draft)=self.draft {
            draft.taps(self.retained_rows,unit.slot)?.entry(layer,owner.buffer("partial")?,unit.rows.len())?;
        }
        Ok(())
    }
    fn begin(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()> {
        let tokens = unit_tokens(unit, self.tokens, self.starts)?;
        let mut chunk = cuteafd_engine::media::MediaChunk::default();
        for member in members(unit) {
            if let Some(media) = self.media.get(member.sequence).and_then(Option::as_ref) {
                let mut part = cuteafd_engine::media::MediaChunk::default();
                media.write_chunk(member.tokens.start, member.tokens.end, &mut part)?;
                chunk.indices.extend(part.indices.into_iter().map(|row| row + member.rows.start as u32));
                chunk.features.extend(part.features);
            }
        }
        self.head.embed_unit(self.kernels, owner, self.target, self.cfg, &tokens, &chunk, uploads)
    }
    fn end(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>) -> Result<()> {
        let members = members(unit);
        let spans = projected_spans(unit.rows.len(), &members, self.head_rows)?;
        if !spans.iter().any(|(_, project)| *project) && self.draft.is_none() {return Ok(());}
        let sequence = *self.slots.get(members[0].sequence).context("Inkling head sequence slot absent")?;
        for (span, project) in spans {
            if !project && self.draft.is_none() { continue; }
            let state = owner.segment(self.cache, sequence, self.cfg.layers-1, span.clone())?;
            let output_row = if self.retained_rows { span.start } else { 0 };
            let logits = if project {Some(self.head.project_at(self.kernels,&state,self.target,self.peer_head,self.cfg,span.len(),
                self.cfg.residual_blocks()+1,output_row)?)} else {
                self.head.normalize(self.kernels,&state,self.target,span.len(),self.cfg.residual_blocks()+1)?;None
            };
            if let Some(draft)=self.draft {
                draft.taps(self.retained_rows,unit.slot)?.final_norm(self.head.normalized(span.len())?,span.start,span.len())?;
            }
            let Some(logits)=logits else {continue;};
            for member in members.iter().filter(|member| member.head) {
                let first = span.start.max(member.rows.start);
                let end = span.end.min(member.rows.end);
                if first >= end { continue; }
                let offset = (first-span.start)*self.cfg.unpadded_vocab_size*4;
                (self.on_logits)(member.sequence, member.tokens.start+first-member.rows.start+1,
                    crate::shared::token_io::DeviceLogits { ptr: logits.ptr.wrapping_byte_add(offset), rows: end-first,
                        vocab: self.cfg.unpadded_vocab_size, stride: self.cfg.unpadded_vocab_size,
                        stream: owner.stream.raw, greedy: None })?;
            }
        }
        if self.publish_draft {
            if let Some(draft)=self.draft {
                let taps=draft.taps(self.retained_rows,unit.slot)?;
                for member in &members {
                    let slot=*self.slots.get(member.sequence).context("draft member slot missing")?;
                    draft.update(taps,slot,member.tokens.start,member.rows.start,member.rows.len())?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct PeerUnits<'s, 'a> {
    pub exchange: &'s crate::shared::peer_split::PeerExchange<'a>,
    pub(super) graphs: Option<&'s mut DecodeGraphs<'a>>,
    pub kernels: &'s Kernels<'a>,
    pub target: &'s TargetWeights<'a>,
    pub cache: &'s super::cache::CacheBank<'a>,
    pub owners: &'s [super::owners::UnitOwner<'a>],
    pub uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
}

impl<'a> PeerUnits<'_, 'a> {
    #[allow(clippy::too_many_arguments)]
    fn attention(&mut self, kernels: &Kernels<'a>, lead_graphs: Option<&mut DecodeGraphs<'a>>,
        lead: &super::owners::PackedLayerUnit<'_, 'a>, target: &TargetWeights<'_>,
        cfg: &Inkling2Config, layer: usize, unit: &Unit, members: &[super::owners::PackedMember]) -> Result<()> {
        use crate::shared::peer_split::DIRECT;
        let peer = &self.owners[unit.slot];
        let rows = unit.rows.len();
        let count = rows * cfg.hidden;
        let bytes = count * 4;
        let mode = if lead.unit.replaying() { Mode::Verify } else { Mode::Decode };
        let mut lead_graphs = lead_graphs;
        let input = layers::mix_attention_input(kernels, lead, target, cfg, layer, rows)?;
        let source = if layer == 0 { "partial" } else if cfg.local[layer] { "mixed" } else { "global_mixes" };
        let dst = peer.buffer(source)?;
        if layer == 0 || cfg.local[layer] {
            self.exchange.push_to(0, DIRECT, input.ptr, dst.ptr, bytes)?;
        } else {
            let src = lead.buffer(source)?;
            let pitch = peer.layout.rows * cfg.hidden * 4;
            ensure!(pitch == lead.layout().rows * cfg.hidden * 4 && src.bytes >= 3 * pitch && dst.bytes >= 3 * pitch,
                "Inkling peer attention input plane geometry differs");
            self.exchange.push_planes(0, DIRECT, &[cuteafd_ffi::peer_exchange::PeerPlane {
                destination: dst.ptr, source: src.ptr, rows: 3, row_bytes: bytes as u64,
                destination_pitch: pitch as u64, source_pitch: pitch as u64,
            }])?;
        }
        self.exchange.on(1, || peer.packed_metadata(layer, self.cache, members, &mut self.uploads[unit.slot]))?;
        self.exchange.wait(1, DIRECT)?;
        let peer_state = super::owners::PackedLayerUnit { unit: peer, cache: self.cache, layer, members,
            uploads: std::cell::RefCell::new(&mut self.uploads[unit.slot]) };
        // Both ranks consume rank-local KV and write an F32 output partial.
        self.exchange.on(1, || {
            graphs::local(self.graphs.as_deref_mut(), peer_state.stream(), layer, rows, Section::RankAttention, mode, || {
                layers::attend_heads(self.kernels, &peer_state, self.target, cfg, layer, rows, dst)?;
                layers::project_heads(self.kernels, &peer_state, self.target, cfg, layer, rows)
            })
        })?;
        // Single BF16 wire (all phases): narrow F32 output partial once into dead `projection_input` (BF16).
        self.exchange.push_partial(1, 0, peer.buffer("branch")?, peer.buffer("projection_input")?, count)?;
        graphs::local(lead_graphs.as_deref_mut(), lead.stream(), layer, rows, Section::RankAttention, mode, || {
            layers::attend_heads(kernels, lead, target, cfg, layer, rows, input)?;
            layers::project_heads(kernels, lead, target, cfg, layer, rows)
        })?;
        self.exchange.narrow_partial(0, lead.buffer("branch")?, lead.buffer("projection_input")?, count)?;
        self.exchange.wait(0, 0)?;
        let remote = cuteafd_ffi::CuteafdDeviceBuffer { ptr: self.exchange.recv(0, 0)?,
            bytes: count * 2, device_id: lead.stream().device.id, flags: 0 };
        let reduced = kernels.reduce_heads(lead, layers::output_weight(target, layer)?, remote, rows, cfg.hidden)?;
        graphs::local(lead_graphs, lead.stream(), layer, rows, Section::AfterAttention, mode, || {
            layers::finish_attention(kernels, lead, target, cfg, layer, rows, reduced)?;
            layers::prepare_ffn_input(kernels, lead, target, cfg, layer, rows)
        })
    }
}

pub(crate) struct Stateful<'s, 'a> {
    pub(super) graphs: Option<&'s mut DecodeGraphs<'a>>,
    pub peer: Option<PeerUnits<'s, 'a>>,
    pub kernels: &'s Kernels<'a>,
    pub owners: &'s [super::owners::UnitOwner<'a>],
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s TargetWeights<'a>,
    pub cfg: &'s Inkling2Config,
    pub ingress: &'s [cuteafd_core::SparkIngress],
    pub placements: &'s mut [super::cache::Placement],
    pub uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
    pub waves: &'s mut crate::shared::spark_waves::Waves<'a>,
    pub onboarded: usize,
    pub local: &'s [Box<dyn Dispatcher + 'a>],
    pub next_identity: &'s mut u64,
    pub placement_version: u64,
    pub kind: cuteafd_transport::ExpertV2SourceKind,
    pub io: &'s mut dyn UnitIo<'a>,
}
impl Stateful<'_, '_> {
    pub fn run(&mut self, units: &[Unit]) -> Result<()> {
        ensure!(!units.is_empty() && self.owners.len() == self.uploads.len()
            && self.owners.len() <= self.waves.waves() && self.onboarded == self.local.len()
            && self.onboarded <= self.cfg.layers && self.ingress.len() == self.cfg.layers && self.placement_version != 0,
            "Inkling stateful prefill owners/identity differ");
        let stream = &self.owners.first().context("Inkling prefill has no unit owners")?.stream;
        ensure!(self.owners.iter().all(|owner| std::rc::Rc::ptr_eq(&owner.stream, stream)),
            "Inkling causal state requires one ordered issuing stream");
        ensure!(units.iter().all(|unit| unit.slot < self.owners.len() && unit.layers == (0..self.cfg.layers)
            && members(unit).iter().all(|member| member.sequence < self.placements.len())),
            "Inkling prefill unit differs from admitted requests");
        let waves = self.owners.len();
        run_prefill_units(self, units, waves)
    }
    fn packed_members(&self, unit: &Unit) -> Result<Vec<super::owners::PackedMember>> {
        members(unit).into_iter().map(|member| {
            let placement = &self.placements[member.sequence];
            Ok(super::owners::PackedMember { sequence: placement.slot, rows: member.rows,
                metadata: cuteafd_loader::families::inkling2::execution::UnitMetadata::new(self.cfg,
                    member.tokens, &placement.units, self.cache.layout.pool_pages)? })
        }).collect()
    }
    fn prepare_routes(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let rows = unit.rows.len();
        graphs::local(self.graphs.as_deref_mut(), &owner.stream, layer, rows, Section::Routes, Mode::Decode, || {
            layers::prepare_routes(self.kernels, owner, self.target, self.cfg, layer, rows)
        })
    }
    fn join(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let members = self.packed_members(unit)?;
        let state = super::owners::PackedLayerUnit { unit: owner, cache: self.cache, layer, members: &members,
            uploads: std::cell::RefCell::new(&mut self.uploads[unit.slot]) };
        self.io.stage(layer,"experts",unit,owner)?;
        let rows = unit.rows.len();
        let mode = if owner.replaying() { Mode::Verify } else { Mode::Decode };
        graphs::local(self.graphs.as_deref_mut(), state.stream(), layer, rows, Section::ExpertFinish, mode, || {
            layers::join_experts(self.kernels, &state, self.target, self.cfg, layer, rows)
        })?;
        self.io.stage(layer,"exit",unit,owner)
    }
}

fn unit_tokens(unit: &Unit, sources: &[&[u32]], starts: &[usize]) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(unit.rows.len());
    for member in members(unit) {
        let source = sources.get(member.sequence).context("Inkling unit token source absent")?;
        let origin = *starts.get(member.sequence).context("Inkling token source origin absent")?;
        let first = member.tokens.start.checked_sub(origin).context("Inkling token span precedes submitted suffix")?;
        let end = member.tokens.end.checked_sub(origin).context("Inkling token end precedes submitted suffix")?;
        tokens.extend_from_slice(source.get(first..end).context("Inkling unit token span outside request")?);
    }
    ensure!(tokens.len() == unit.rows.len(), "Inkling token members do not cover unit");
    Ok(tokens)
}

fn members(unit: &Unit) -> Vec<crate::shared::prefill_pipeline::UnitSegment> {
    if !unit.segments.is_empty() { return unit.segments.clone(); }
    vec![crate::shared::prefill_pipeline::UnitSegment { sequence: unit.sequence,
        rows: 0..unit.rows.len(), tokens: unit.rows.clone(), prompt_end: false, snapshot: false, head: false }]
}

impl PrefillHooks for Stateful<'_, '_> {
    type Ticket = crate::shared::spark_waves::WaveTicket;
    fn route(&self, layer: usize) -> FfnRoute {
        if layer < self.onboarded { FfnRoute::Local } else { FfnRoute::Spark }
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        self.io.begin(unit, &self.owners[unit.slot], &mut self.uploads[unit.slot])
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        self.io.tap(layer,unit,owner)?;
        self.io.stage(layer,"entry",unit,owner)?;
        let members = self.packed_members(unit)?;
        owner.packed_metadata(layer, self.cache, &members, &mut self.uploads[unit.slot])?;
        let state = super::owners::PackedLayerUnit { unit: owner, cache: self.cache, layer, members: &members,
            uploads: std::cell::RefCell::new(&mut self.uploads[unit.slot]) };
        if let Some(peer) = &mut self.peer {
            peer.attention(self.kernels, self.graphs.as_deref_mut(), &state, self.target, self.cfg, layer, unit, &members)?;
        } else {
            let rows = unit.rows.len();
            let mode = if owner.replaying() { Mode::Verify } else { Mode::Decode };
            graphs::local(self.graphs.as_deref_mut(), state.stream(), layer, rows, Section::WholeAttention, mode, || {
                layers::begin_layer(self.kernels, &state, self.target, self.cfg, layer, rows)
            })?;
        }
        self.io.stage(layer,"attention",unit,owner)
    }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        self.prepare_routes(layer, unit)?;
        let owner = &self.owners[unit.slot];
        layers::dispatch_experts(owner, self.local[layer].as_ref(), unit.rows.len())?;
        self.join(layer, unit)
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        use crate::shared::{experts::staging::RouteStaging, spark_waves};
        use cuteafd_transport::ExpertV2SourceKind;
        self.prepare_routes(layer, unit)?;
        let owner = &self.owners[unit.slot];
        let rows = unit.rows.len();
        let identity = *self.next_identity;
        ensure!(identity > 0, "Inkling Spark request identity is zero");
        *self.next_identity = identity.checked_add(1).context("Inkling Spark identity exhausted")?;
        let (source, start, version) = (unit.sequence as u64 + 1, unit.rows.start, self.placement_version);
        let dtype = match self.ingress[layer] {
            cuteafd_core::SparkIngress::Fp8K32 => cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            cuteafd_core::SparkIngress::Bf16 => cuteafd_transport::ExpertV2Dtype::Bf16,
            _ => bail!("Inkling layer ingress must name its native package input"),
        };
        let input = owner.buffer(if dtype == cuteafd_transport::ExpertV2Dtype::Bf16 { "projection_input" } else { "expert_wire" })?;
        let staging = RouteStaging::new(rows, self.cfg.topk, dtype.row_bytes(self.cfg.hidden)?)?;
        let kind = self.kind;
        // Explicit singleton members can carry a nonzero restored frontier;
        // preserve their request-local position too, not only multi-member packs.
        let origins = members(unit).iter().flat_map(|member| member.tokens.clone().map(move |position|
            (member.sequence as u64+1, position as u64))).collect::<Vec<_>>();
        let packed = unit.reply_origins()?;
        let combined = if kind == ExpertV2SourceKind::Prefill {
            packed.as_ref().map_or_else(|| spark_waves::single_reply_eligible(rows, kind), |origins| origins.combined_reply_eligible())
        } else { spark_waves::single_reply_eligible(rows, kind) };
        owner.stream.device.run(|| {
            // SAFETY: admitted unit owns routes/wire and the issuing stream;
            // shared SM staging finishes before the same unit reuses them.
            unsafe { self.waves.send(rows, rows*self.cfg.hidden*2, staging,
                [owner.buffer("route_ids")?, owner.buffer("route_weights")?, input],
                owner.stream.raw, combined, Box::new(move |routes, wire| {
                    let mut request = request(identity, version, layer, source, start, rows, dtype, routes, wire)?;
                    if let Some(packed) = packed { packed.apply(&mut request)?; }
                    ensure!(origins.len() == request.rows.len(), "Inkling Spark origin count differs");
                    for (row, &(source, position)) in request.rows.iter_mut().zip(&origins) {
                        row.source_request_id = source; row.token_position = position; row.source_kind = kind;
                    }
                    Ok(request)
                })) }
        })
    }
    fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let slot = ticket.slot();
        owner.stream.device.run(|| {
            self.waves.land(ticket, owner.stream.raw)?;
            // SAFETY: shared wave landing precedes reduction into the private
            // unit plane; the reducer also releases the shared return credit.
            unsafe { self.waves.reduce(slot, std::ptr::null(), owner.buffer("expert_output")?.ptr.cast(),
                unit.rows.len(), owner.stream.raw) }
        })?;
        self.join(layer, unit)
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> { self.io.end(unit, &self.owners[unit.slot]) }
    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        for member in members(unit) {
            let placement = &mut self.placements[member.sequence];
            ensure!(placement.state_len == member.tokens.start, "Inkling causal commit is out of order");
            placement.state_len = member.tokens.end;
            placement.len = member.tokens.end;
        }
        Ok(())
    }
}

fn request(identity: u64, version: u64, layer: usize, source: u64, start: usize, rows: usize,
    dtype: cuteafd_transport::ExpertV2Dtype, routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, wire: bytes::Bytes)
    -> Result<cuteafd_transport::ExpertProtocolV2Request> {
    use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2SourceKind};
    ensure!(identity > 0 && version > 0 && source > 0 && layer < 48 && (1..=4096).contains(&rows),
        "Inkling Spark request geometry/identity differs");
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: (start+row) as u64, source_kind: ExpertV2SourceKind::Prefill, source_request_id: source,
        token_position: (start+row) as u64, route_offset: (row*8) as u32, route_count: 8,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(identity, version, u32::try_from(layer)?, 4608,
        dtype, descriptors, routes, wire)?;
    request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

pub(crate) fn layer<'a>(kernels: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'a>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    struct Hooks<'b, 'a, O> {
        kernels: &'b Kernels<'a>, owner: &'b O, target: &'b TargetWeights<'a>,
        experts: &'b dyn Dispatcher, cfg: &'b Inkling2Config,
    }
    impl<'a, O: UnitBuffers<'a>> PrefillHooks for Hooks<'_, 'a, O> {
        type Ticket = ();
        fn route(&self, _: usize) -> FfnRoute { FfnRoute::Local }
        fn begin_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
        fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            layers::begin_layer(self.kernels, self.owner, self.target, self.cfg, layer, unit.rows.len())
        }
        fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            layers::finish_layer(self.kernels, self.owner, self.target, self.experts, self.cfg, layer, unit.rows.len())
        }
        fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> { bail!("cold Inkling golden has no Spark route") }
        fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> Result<()> { bail!("cold Inkling golden has no Spark ticket") }
        fn end_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
        fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
    }
    let mut units = sequence_units(rows, owner.layout().rows.next_multiple_of(64), 1, cfg.layers)?;
    for unit in &mut units { unit.layers = index..index+1; }
    run_prefill_units(&mut Hooks { kernels, owner, target, experts, cfg }, &units, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_transport::{ExpertProtocolV2RouteEntry, ExpertV2Dtype};

    #[test]
    fn requests_keep_real_request_origins_and_exact_mxfp8_wire() -> Result<()> {
        let routes = (0..16).map(|index| ExpertProtocolV2RouteEntry {
            row_index: index/8, expert_id: index, gate_weight: 0.125,
        }).collect();
        let request = request(17, 4, 5, 83, 1087, 2, ExpertV2Dtype::Fp8E4m3Ue8m0K32, routes, vec![0; 2*4752].into())?;
        assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Fp8E4m3Ue8m0K32);
        assert_eq!(request.header.hidden_row_stride_bytes, 4752);
        assert_eq!(request.rows[0].source_request_id, 83);
        assert_eq!(request.rows[0].token_position, 1087);
        assert_eq!(request.rows[1].token_position, 1088);
        assert_eq!(request.rows[1].route_offset, 8);
        assert_eq!(request.rows[1].route_count, 8);
        assert_ne!(request.header.flags & cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, 0);
        Ok(())
    }

    #[test]
    fn native_bf16_requests_preserve_unquantized_input_bytes() -> Result<()> {
        let routes = (0..16).map(|index| ExpertProtocolV2RouteEntry {
            row_index: index/8, expert_id: index, gate_weight: 0.125,
        }).collect();
        let wire = vec![0x3d; 2*4608*2];
        let request = request(17, 4, 0, 83, 1087, 2, ExpertV2Dtype::Bf16, routes, wire.clone().into())?;
        assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Bf16);
        assert_eq!(request.header.hidden_row_stride_bytes, 9216);
        assert_eq!(request.hidden_payload.as_ref(), wire.as_slice());
        Ok(())
    }

    #[test]
    fn packed_members_keep_request_local_frontiers_and_shared_wire_origins() -> Result<()> {
        use crate::shared::prefill_pipeline::UnitSegment;
        let segments = vec![
            UnitSegment { sequence: 2, rows: 0..2, tokens: 1087..1089, prompt_end: false, snapshot: false, head: false },
            UnitSegment { sequence: 7, rows: 2..5, tokens: 63..66, prompt_end: true, snapshot: true, head: true },
        ];
        let unit = Unit { index: 0, sequence: 2, segments: segments.clone(), rows: 0..5,
            layers: 0..48, slot: 0, barrier: false };
        assert_eq!(members(&unit), segments);
        let a = (0..1090).collect::<Vec<u32>>();
        let b = (1000..1100).collect::<Vec<u32>>();
        let sources: Vec<&[u32]> = vec![&[], &[], &a, &[], &[], &[], &[], &b];
        assert_eq!(unit_tokens(&unit, &sources, &[0; 8])?, [1087, 1088, 1063, 1064, 1065]);
        assert!(unit_tokens(&unit, &[&a], &[0]).is_err());
        let routes = (0..40).map(|route| ExpertProtocolV2RouteEntry {
            row_index: route/8, expert_id: route%384, gate_weight: 0.125 }).collect();
        let mut request = request(99, 3, 5, 3, 0, 5, ExpertV2Dtype::Fp8E4m3Ue8m0K32, routes, vec![0; 5*4752].into())?;
        unit.reply_origins()?.unwrap().apply(&mut request)?;
        assert_eq!(request.rows.iter().map(|row| row.source_request_id).collect::<Vec<_>>(), [3, 3, 8, 8, 8]);
        assert_eq!(request.rows.iter().map(|row| row.token_position).collect::<Vec<_>>(), [1087, 1088, 63, 64, 65]);
        let singleton = Unit { segments: vec![], rows: 63..66, ..unit };
        assert_eq!(members(&singleton)[0].rows, 0..3);
        assert_eq!(members(&singleton)[0].tokens, 63..66);
        Ok(())
    }

    #[test]
    fn persistent_units_use_shared_waves_without_host_waits_or_a_private_scheduler() {
        let source = include_str!("prefill.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("impl PrefillHooks for Stateful"));
        assert!(source.contains("run_prefill_units(self, units, waves)"));
        assert!(source.contains("self.waves.send("));
        assert!(source.contains("self.waves.land("));
        assert!(source.contains("self.waves.reduce("));
        assert!(source.contains("layers::prepare_routes("));
        assert!(source.contains("layers::join_experts("));
        assert!(!source.contains("for layer in"));
        assert!(!source.contains("cuda_stream_synchronize"));
        assert!(!source.contains(".drain("));
    }
}
