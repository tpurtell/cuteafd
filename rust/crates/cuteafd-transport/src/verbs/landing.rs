//! GPU landing: response payloads received straight into device memory.
//!
//! A session given a [`DeviceLanding`] registers that device range with its
//! NIC protection domain over dma-buf (`cuMemGetHandleForAddressRange` +
//! `ibv_reg_dmabuf_mr`; nvidia-peermem is not needed) and posts every response
//! receive as two scatter entries: the fixed response header into the host
//! slot and the payload into the device range. Native compact-BF16 workers
//! answer each request with one unindexed frame whose payload directly follows
//! the header, so a rank's rows land at the start of its range in row order
//! with no host copy and no upload; workers are unchanged. If registration
//! fails the session keeps host receives and says why.
use super::*;

/// A device range one rank's response payloads land in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLanding {
    /// Device address, carried as an integer so a landing can cross threads.
    pub ptr: usize,
    pub bytes: usize,
}

impl DeviceLanding {
    pub fn new(buffer: CuteafdDeviceBuffer) -> Self {
        Self { ptr: buffer.ptr as usize, bytes: buffer.bytes }
    }
    fn buffer(self) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer { ptr: self.ptr as *mut c_void, bytes: self.bytes, ..Default::default() }
    }
}

/// Write mode: the peer RDMA-writes each response's rows into
/// `[base + plane_offset, + plane_bytes)` and then its request id into the
/// u64 at `base + flag_offset`; `[base, base + bytes)` is one device range the
/// session exposes (dma-buf, remote write, no relaxed ordering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceWriteTarget {
    pub base: usize,
    pub bytes: usize,
    pub plane_offset: usize,
    pub plane_bytes: usize,
    pub flag_offset: usize,
}

/// Exposes `target`'s range on `endpoint` and returns the handshake's view of it.
pub(super) fn expose(endpoint: &NativeRdmaEndpoint, target: DeviceWriteTarget) -> Result<VerbsHostWriteTarget> {
    anyhow::ensure!(target.plane_offset + target.plane_bytes <= target.bytes && target.flag_offset + 8 <= target.bytes
        && target.flag_offset % 8 == 0, "write target ranges exceed the exposed range");
    let buffer = CuteafdDeviceBuffer { ptr: target.base as *mut c_void, bytes: target.bytes, ..Default::default() };
    // SAFETY: `SparkExperts::set_write_targets`'s contract keeps the range
    // allocated while this endpoint (owned by the session) exists.
    let rkey = unsafe { endpoint.library.rdma_rc_endpoint_expose_device(endpoint.info.handle, Some(buffer))? };
    Ok(VerbsHostWriteTarget { plane_addr: (target.base + target.plane_offset) as u64, plane_bytes: target.plane_bytes as u64,
        flag_addr: (target.base + target.flag_offset) as u64, rkey })
}

/// Registers `landing` on `endpoint` before its receives are posted; false
/// (with the reason logged) keeps the session on host receives.
pub(super) fn attach(endpoint: &NativeRdmaEndpoint, landing: DeviceLanding, addr: SocketAddr) -> bool {
    // SAFETY: `SparkExperts::set_gpu_landing`'s contract makes its caller keep
    // the range allocated, and unread while a receive may write it, for as
    // long as this endpoint (owned by the session) exists.
    let attached = unsafe {
        endpoint.library.rdma_rc_endpoint_set_recv_landing(endpoint.info.handle, Some(landing.buffer()),
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN)
    };
    match attached {
        Ok(()) => {
            tracing::debug!(%addr, bytes = landing.bytes, "response payloads land in GPU memory (dma-buf)");
            true
        }
        Err(error) => {
            tracing::warn!(%addr, %error, "GPU landing unavailable; receiving into pinned host memory");
            false
        }
    }
}

impl ProtocolV2ResponseChunkAssembler {
    /// A whole-plane response whose payload landed in device memory.
    pub(super) fn accept_landed(&mut self, request: &ExpertProtocolV2Request, header: &ExpertProtocolV2ResponseHeader)
        -> Result<()> {
        anyhow::ensure!(!self.final_chunk_received, "ProtocolV2 response chunk arrived after the final chunk");
        validate_response_matches_request(header, request)?;
        anyhow::ensure!(!self.stream_frame && self.partial_output_payload.is_none(),
            "GPU landing takes only streamed whole-plane responses");
        anyhow::ensure!(header.row_count as usize == self.request_row_count,
            "GPU-landed response carries {} of {} request rows", header.row_count, self.request_row_count);
        self.completed_rows.iter_mut().for_each(|row| *row = true);
        self.completed_row_count = self.request_row_count;
        self.header = Some(header.clone());
        self.debug_checksum_enabled = Some(false);
        self.final_chunk_received = true;
        Ok(())
    }
}

impl VerbsHostProtocolV2PersistentClientSession {
    /// [`Self::accept_chunk_response_frame`] for a session whose payloads land
    /// in device memory: only the header is in the host slot.
    pub(super) fn accept_landed_chunk_frame(
        &mut self,
        pending: &mut VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>,
        header_bytes: &[u8],
        recv_offset: usize,
        recv_slot: usize,
        config: &TcpTransportConfig,
    ) -> Result<()> {
        let header = ExpertProtocolV2ResponseView::parse_landed_header(header_bytes)
            .context("validating a GPU-landed ProtocolV2 response header")?;
        let wire_bytes = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + header.output_payload_bytes as usize;
        anyhow::ensure!(wire_bytes <= config.max_frame_bytes,
            "GPU-landed response of {wire_bytes} bytes exceeds the frame budget");
        // The device range is written again only by the response to this
        // session's next request, which its owner posts after consuming it.
        self.endpoint.post_recv_at(recv_offset, self.response_ring.slot_capacity_bytes,
            VERBS_HOST_RECV_WR_ID + recv_slot as u64)?;
        self.response_recv_sequence = self.response_recv_sequence.wrapping_add(1);
        let mut completed = pending.pop_front()
            .context("persistent verbs-host response arrived without a pending request")?;
        completed.assembler.accept_landed(&completed.request, &header)?;
        completed.chunk_tx.send(VerbsHostProtocolV2ResponseChunk {
            stream_id: completed.stream_id,
            header,
            row_indices: None,
            partial_output_payload: VerbsHostProtocolV2ResponsePayload::from_owned(Vec::new()),
            wire_bytes,
            landed: true,
        }).context("forwarding a GPU-landed ProtocolV2 response")?;
        completed.response_frames += 1;
        completed.response_wire_bytes += wire_bytes;
        let response_executor_id = completed.assembler.finish_validation()?.executor_id;
        let _ = completed.response_tx.send(Ok(VerbsHostProtocolV2ResponseStreamStats {
            response_frames: completed.response_frames,
            response_wire_bytes: completed.response_wire_bytes,
            response_executor_id,
        }));
        Ok(())
    }
}

/// What the startup probe found about landing responses in device memory.
#[derive(Debug, Clone, Serialize)]
pub struct GpuLandingProbe {
    pub rdma_device: String,
    pub cuda_device: i32,
    pub dma_buf: bool,
    pub gpudirect_rdma: bool,
    /// `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING`; 100 (owner) or
    /// more lets kernels launched after a completion read the landed bytes.
    pub writes_ordering: i32,
    pub registered: bool,
    /// Loopback SEND rate landing in device memory, GB/s.
    pub gpu_gbps: f64,
    /// Loopback SEND rate landing in pinned host memory, GB/s.
    pub host_gbps: f64,
    pub status: String,
    pub error: Option<String>,
}

impl GpuLandingProbe {
    /// Landing verified end to end and ordered for this device's kernels.
    pub fn usable(&self) -> bool {
        self.error.is_none() && self.registered && self.writes_ordering >= 100
    }
}

/// Probes dma-buf landing on the current CUDA device with a loopback QP pair
/// on the transport's RDMA device, timing `bytes` sends into device and into
/// pinned host memory.
pub fn gpu_landing_probe(library: &NativeLibrary, rdma_device: Option<&str>, bytes: usize, iterations: u32)
    -> Result<GpuLandingProbe> {
    let port = verbs_host_ib_port_num()?;
    let (probe, error) = library.rdma_gpu_landing_probe(rdma_device, port, bytes, iterations)?;
    Ok(GpuLandingProbe {
        rdma_device: c_char_array_to_string(&probe.device_name),
        cuda_device: probe.cuda_device,
        dma_buf: probe.dma_buf_supported != 0,
        gpudirect_rdma: probe.gpudirect_rdma_supported != 0,
        writes_ordering: probe.writes_ordering,
        registered: probe.registered != 0,
        gpu_gbps: probe.gpu_gbps,
        host_gbps: probe.host_gbps,
        status: c_char_array_to_string(&probe.status),
        error,
    })
}
