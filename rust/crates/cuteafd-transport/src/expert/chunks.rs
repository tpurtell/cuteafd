//! Bounded chunks carry request-selected rank partials in token order.
use super::{BackboneRequest, SparkTopology, V41_NATIVE_GROUP_REQUEST_FLAG, V41Tp4Planes};
use crate::{
    EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES, EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, ExpertProtocolV2ResponseHeader,
    ExpertProtocolV2ResponseRef, ExpertProtocolV2ResponseView, ExpertProtocolV2Status,
};
use anyhow::{Context, Result, ensure};

impl BackboneRequest<'_> {
    fn response_header_bytes(&self) -> usize {
        if self.view.header.flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0 {
            EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN
        } else {
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
        }
    }
    /// Maximum token rows per indexed frame, including header and row-index overhead.
    pub fn response_chunk_rows(&self, max_frame_bytes: usize) -> Result<u32> {
        ensure!(!self.view.partial_replies(), "per-expert partial regions cannot be chunked");
        let payload = max_frame_bytes
            .checked_sub(self.response_header_bytes())
            .context("response frame cannot fit header")?;
        let rows = payload / (self.output_row_bytes()? as usize + 4);
        ensure!(rows > 0, "response frame cannot fit one native token row");
        Ok(rows.min(self.rows() as usize) as u32)
    }
    /// Produce one contiguous chunk from a slice of the full route plane.
    /// `partials` contains only this chunk; caller reuses the supplied index scratch.
    pub fn response_chunk<'a>(
        &self,
        executor_id: u64,
        start_row: u32,
        partials: &'a [u8],
        row_indices: &'a mut [u32],
        max_frame_bytes: usize,
    ) -> Result<ExpertProtocolV2ResponseRef<'a>> {
        ensure!(
            executor_id != 0,
            "native response needs an executor identity"
        );
        ensure!(
            partials.len() % self.output_row_bytes()? as usize == 0,
            "native response chunk has a partial token row"
        );
        let rows = u32::try_from(partials.len() / self.output_row_bytes()? as usize)?;
        ensure!(
            rows > 0 && rows <= self.response_chunk_rows(max_frame_bytes)?,
            "native response chunk exceeds frame budget"
        );
        let end = start_row
            .checked_add(rows)
            .context("native response row overflow")?;
        ensure!(
            end <= self.rows(),
            "native response chunk exceeds request rows"
        );
        ensure!(
            row_indices.len() >= rows as usize,
            "response row-index scratch is too short"
        );
        for (index, row) in row_indices[..rows as usize].iter_mut().enumerate() {
            *row = start_row + index as u32;
        }
        let request = &self.view.header;
        let response = ExpertProtocolV2ResponseRef {
            header: ExpertProtocolV2ResponseHeader {
                request_id: request.request_id,
                placement_version: request.placement_version,
                layer_id: request.layer_id,
                row_count: rows,
                output_dim: cuteafd_core::expert_geometry().hidden,
                output_dtype: self.output_dtype(),
                output_row_stride_bytes: self.output_row_bytes()?,
                output_payload_bytes: partials.len() as u64,
                status: ExpertProtocolV2Status::Ok,
                flags: (request.flags & !V41_NATIVE_GROUP_REQUEST_FLAG)
                    | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES
                    | if end < self.rows() {
                        EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS
                    } else {
                        0
                    },
                executor_id,
                pace: None,
            },
            row_indices: Some(&row_indices[..rows as usize]),
            partial_output_payload: partials,
        };
        response.validate()?;
        ensure!(
            response.wire_stats().wire_bytes <= max_frame_bytes,
            "native response frame budget mismatch"
        );
        Ok(response)
    }
}

/// Constant-size assembly state; a sink copies each validated chunk into its rank plane.
/// Frames from different ranks may interleave; each rank's rows must arrive in order.
/// A rank is complete only after every row and the matching final-chunk marker.
/// As with complete responses, request IDs must identify unique in-flight waves.
pub struct V41Tp4ChunkReceiver {
    identity: V41Tp4Planes<'static>,
    received: [u32; cuteafd_core::MAX_SPARK_WORLD],
    finished: [bool; cuteafd_core::MAX_SPARK_WORLD],
    max_frame_bytes: usize,
}
impl V41Tp4ChunkReceiver {
    pub fn new(
        request: &BackboneRequest<'_>,
        executors: [u64; 4],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        Self::new_ranks(request, &executors, max_frame_bytes)
    }
    /// Collect exactly two complete rank planes while preserving TP4 APIs.
    pub fn new_tp2(
        request: &BackboneRequest<'_>,
        executors: [u64; 2],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        Self::new_ranks(request, &executors, max_frame_bytes)
    }
    /// Generic constructor for the validated physical rank counts 1..=8.
    ///
    /// A request admitted under the native replicated-group contract must match
    /// its topology's world size.
    pub fn new_ranks(
        request: &BackboneRequest<'_>,
        executors: &[u64],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=cuteafd_core::MAX_SPARK_WORLD).contains(&executors.len()),
            "native TP/EP requires 1..=8 executors"
        );
        if let Some(topology) = request.native_topology() {
            ensure!(
                executors.len() == topology.world_size(),
                "native group executor count does not match its topology"
            );
        }
        let identity = V41Tp4Planes::new_ranks(request, executors)?;
        identity.check_frame_budget(max_frame_bytes, request.response_header_bytes())?;
        Ok(Self {
            identity,
            received: [0; cuteafd_core::MAX_SPARK_WORLD],
            finished: [false; cuteafd_core::MAX_SPARK_WORLD],
            max_frame_bytes,
        })
    }
    /// Validate the owned request without serializing its activation payload.
    pub(crate) fn from_owned(
        request: &crate::ExpertProtocolV2Request,
        max_rows: u32,
        executors: [u64; 4],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        Self::from_owned_ranks(request, max_rows, &executors, max_frame_bytes)
    }
    pub(crate) fn from_owned_ranks(
        request: &crate::ExpertProtocolV2Request,
        max_rows: u32,
        executors: &[u64],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        // Ownership words are undecodable without the group count, so a native
        // group request needs the topology-bound constructor below.
        ensure!(
            request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG == 0,
            "native group ownership requires a topology-bound receiver"
        );
        BackboneRequest::validate_owned(request, max_rows)?;
        ensure!(
            request.wire_stats().wire_bytes <= max_frame_bytes,
            "native request exceeds RoCE frame budget"
        );
        let header_bytes = if request.header.flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0 {
            EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN
        } else {
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
        };
        let mut identity = V41Tp4Planes::from_header_ranks(&request.header, executors)?;
        identity.set_partial_plan(request.partial.as_ref())?;
        identity.check_frame_budget(max_frame_bytes, header_bytes)?;
        Ok(Self {
            identity,
            received: [0; cuteafd_core::MAX_SPARK_WORLD],
            finished: [false; cuteafd_core::MAX_SPARK_WORLD],
            max_frame_bytes,
        })
    }
    /// Ownership-aware constructor for a worker bound to one native topology.
    pub(crate) fn from_owned_topology(
        request: &crate::ExpertProtocolV2Request,
        max_rows: u32,
        executors: &[u64],
        topology: SparkTopology,
        max_frame_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            executors.len() == topology.world_size(),
            "native group executor count does not match its topology"
        );
        ensure!(
            request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0,
            "topology-bound receiver requires a native group request"
        );
        BackboneRequest::validate_owned_native_group(request, max_rows, topology)?;
        ensure!(
            request.wire_stats().wire_bytes <= max_frame_bytes,
            "native request exceeds RoCE frame budget"
        );
        let header_bytes = if request.header.flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0 {
            EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN
        } else {
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
        };
        ensure!(
            max_frame_bytes
                >= header_bytes
                    + V41Tp4Planes::from_header_ranks(&request.header, executors)?.output_row_bytes
                        as usize
                    + 4,
            "response frame cannot fit one native token row"
        );
        Ok(Self {
            identity: V41Tp4Planes::from_header_ranks(&request.header, executors)?,
            received: [0; cuteafd_core::MAX_SPARK_WORLD],
            finished: [false; cuteafd_core::MAX_SPARK_WORLD],
            max_frame_bytes,
        })
    }
    pub fn complete(&self) -> bool {
        self.finished[..self.identity.world_size()]
            .iter()
            .all(|value| *value)
    }
    /// Legacy four-rank row counters; ranks past four are always zero.
    pub fn received_rows(&self) -> [u32; 4] {
        [
            self.received[0],
            self.received[1],
            self.received[2],
            self.received[3],
        ]
    }
    /// Row counters for every physical rank in rank order.
    pub fn received_rows_slice(&self) -> &[u32] {
        &self.received[..self.identity.world_size()]
    }
    /// Physical ranks this receiver expects.
    pub fn world_size(&self) -> usize {
        self.identity.world_size()
    }

    /// The verbs client has already checked framing and checksum before exposing
    /// these owned/recycled payload bytes. Keep their owner alive through the sink.
    pub(crate) fn push_rdma<'f, F>(
        &mut self,
        chunk: &'f crate::VerbsHostProtocolV2ResponseChunk,
        sink: F,
    ) -> Result<usize>
    where
        F: FnOnce(usize, u32, &'f [u8]) -> Result<()>,
    {
        ensure!(
            chunk.wire_bytes <= self.max_frame_bytes,
            "native response exceeds receive frame budget"
        );
        self.push_parts(
            &chunk.header,
            chunk.row_indices.as_deref(),
            chunk.partial_output_payload.as_ref(),
            sink,
        )
    }

    /// A rank's whole plane, received straight into its device landing range
    /// (the verbs session validated the header's extent and request identity).
    pub(crate) fn push_landed(
        &mut self,
        chunk: &crate::VerbsHostProtocolV2ResponseChunk,
    ) -> Result<usize> {
        ensure!(
            chunk.landed && chunk.row_indices.is_none(),
            "not a GPU-landed whole-plane response"
        );
        ensure!(
            chunk.wire_bytes <= self.max_frame_bytes,
            "native response exceeds receive frame budget"
        );
        let h = &chunk.header;
        let rank = self.identity.response_rank(h)?;
        ensure!(
            !self.finished[rank] && self.received[rank] == 0,
            "native TP rank already completed"
        );
        let allowed = self.identity.partial_flag;
        ensure!(
            h.flags & !allowed == 0,
            "unsupported GPU-landed response flags"
        );
        ensure!(
            h.row_count == self.identity.rows,
            "GPU-landed response must contain the entire plane"
        );
        ensure!(
            h.output_payload_bytes == self.identity.partial_payload_bytes
                .unwrap_or(u64::from(h.row_count) * u64::from(self.identity.output_row_bytes)),
            "native payload size mismatch"
        );
        self.received[rank] = h.row_count;
        self.finished[rank] = true;
        Ok(rank)
    }

    /// Validate before invoking the sink, then commit progress only if it succeeds.
    /// The sink receives (rank, first token row, contiguous request-selected rank-partial bytes).
    /// It must copy the bytes or keep their storage alive until asynchronous writes
    /// finish; `complete()` does not itself synchronize GPU copies.
    /// On sink error no progress is committed, but destination bytes may be partial.
    pub fn push<'f, F>(&mut self, frame: &'f [u8], sink: F) -> Result<usize>
    where
        F: FnOnce(usize, u32, &'f [u8]) -> Result<()>,
    {
        ensure!(
            frame.len() <= self.max_frame_bytes,
            "native response exceeds receive frame budget"
        );
        let response = ExpertProtocolV2ResponseView::parse(frame)?;
        let indices = if response.row_indexed() {
            Some(
                (0..response.header.row_count as usize)
                    .map(|i| response.request_row_index(i))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        self.push_parts(
            &response.header,
            indices.as_deref(),
            response.partial_output_payload(),
            sink,
        )
    }

    fn push_parts<'f, F>(
        &mut self,
        h: &ExpertProtocolV2ResponseHeader,
        indices: Option<&[u32]>,
        payload: &'f [u8],
        sink: F,
    ) -> Result<usize>
    where
        F: FnOnce(usize, u32, &'f [u8]) -> Result<()>,
    {
        let rank = self.identity.response_rank(h)?;
        ensure!(!self.finished[rank], "native TP rank already completed");
        let allowed = self.identity.partial_flag
            | EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
            | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES
            | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        ensure!(
            h.flags & !allowed == 0,
            "unsupported native response chunk flags"
        );
        let start = self.received[rank];
        let end = start
            .checked_add(h.row_count)
            .context("native received row overflow")?;
        ensure!(
            h.row_count > 0 && end <= self.identity.rows,
            "native response chunk exceeds remaining rows"
        );
        ensure!(
            indices.is_some() == (h.flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES != 0),
            "native row-index flag mismatch"
        );
        let more_chunks = h.flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS != 0;
        ensure!(
            payload.len() == h.output_payload_bytes as usize,
            "native payload size mismatch"
        );
        if let Some(indices) = indices {
            ensure!(
                indices.len() == h.row_count as usize,
                "native row index count mismatch"
            );
            for i in 0..h.row_count {
                ensure!(
                    indices[i as usize] == start + i,
                    "native response rows overlap, skip or reorder"
                );
            }
        } else {
            ensure!(
                start == 0 && h.row_count == self.identity.rows && !more_chunks,
                "unindexed native response must contain the entire plane"
            );
        }
        ensure!(
            more_chunks == (end < self.identity.rows),
            "native response has an early or missing final marker"
        );
        sink(rank, start, payload)?;
        self.received[rank] = end;
        self.finished[rank] = end == self.identity.rows;
        Ok(rank)
    }
}
