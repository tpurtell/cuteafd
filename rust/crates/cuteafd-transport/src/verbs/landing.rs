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
    pub slot_bytes: usize,
    /// Every request names its own [`LandingSlice`] of the range, and its
    /// receive is posted with that request: no receive is posted ahead and
    /// there is no rotation, so the wave geometry inside the range may change
    /// between waves and a session may (re)open at any wave slot.
    pub placed: bool,
}

/// Where one wave's payload lands inside a placed [`DeviceLanding`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LandingSlice {
    /// Byte offset from the start of the rank's landing range.
    pub offset: usize,
    /// Payload bytes the receive accepts there.
    pub bytes: usize,
}

impl DeviceLanding {
    pub fn new(buffer: CuteafdDeviceBuffer) -> Self {
        Self { ptr: buffer.ptr as usize, bytes: buffer.bytes, slot_bytes: buffer.bytes, placed: false }
    }
    pub fn with_slots(buffer: CuteafdDeviceBuffer, slots: usize) -> Result<Self> {
        anyhow::ensure!((1..=4).contains(&slots) && buffer.bytes > 0 && buffer.bytes % slots == 0,
            "GPU landing requires equally sized slots");
        Ok(Self { slot_bytes: buffer.bytes / slots, ..Self::new(buffer) })
    }
    /// The whole range, registered once; each wave lands at the slice its
    /// request names ([`LandingSlice`]).
    pub fn placed(buffer: CuteafdDeviceBuffer) -> Result<Self> {
        anyhow::ensure!(buffer.bytes > 0, "GPU landing range is empty");
        Ok(Self { placed: true, ..Self::new(buffer) })
    }
    /// Whether `slice` lies inside the range.
    pub fn holds(&self, slice: LandingSlice) -> bool {
        slice.bytes > 0 && slice.offset.checked_add(slice.bytes).is_some_and(|end| end <= self.bytes)
    }
    fn buffer(self) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer { ptr: self.ptr as *mut c_void, bytes: self.bytes, ..Default::default() }
    }
}

/// Registers `landing` on `endpoint` before its receives are posted; false
/// (with the reason logged) keeps the session on host receives.
pub(super) fn attach(endpoint: &NativeRdmaEndpoint, landing: DeviceLanding, addr: SocketAddr) -> bool {
    // A placed range is one slot: its receives name their own slices.
    let slot_bytes = if landing.placed { landing.bytes } else { landing.slot_bytes };
    // SAFETY: `SparkExperts::set_gpu_landing`'s contract makes its caller keep
    // the range allocated, and unread while a receive may write it, for as
    // long as this endpoint (owned by the session) exists.
    let attached = unsafe {
        endpoint.library.rdma_rc_endpoint_set_recv_landing(endpoint.info.handle, Some(landing.buffer()),
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, slot_bytes)
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
        self.endpoint.traffic.received(wire_bytes);
        // Native posts rotate the N landing slots in request order. The owner
        // fences a slot's consumer before posting its replacement request.
        // A placed landing posts each receive with its own request instead.
        if !self.placed_landing {
            self.endpoint.post_recv_at(recv_offset, self.response_ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + recv_slot as u64)?;
        }
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
/// on `rdma` (the device, port and source GID that reach the transport's
/// peers, [`crate::fabric::select_rdma_route`]), timing `bytes` sends into
/// device and into pinned host memory.
pub fn gpu_landing_probe(library: &NativeLibrary, rdma: &crate::fabric::RdmaSelection, bytes: usize,
    iterations: u32) -> Result<GpuLandingProbe> {
    let (probe, error) = library.rdma_gpu_landing_probe(&rdma.device, rdma.port, rdma.gid_index, bytes, iterations)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA/RDMA container and an exclusive hardware lease"]
    fn cuda_malloc_dma_buf_landing_probe() -> Result<()> {
        let library = load_verbs_host_native_library()?;
        let peer = std::env::var("CUTEAFD_LANDING_PROBE_PEER")
            .context("set CUTEAFD_LANDING_PROBE_PEER to a Spark worker address (host:port)")?;
        let peer = peer.to_socket_addrs()?.next().context("landing probe peer resolved no address")?;
        let rdma = crate::fabric::select_rdma_route(&crate::fabric::discover()?.ports, peer)?;
        let probe = gpu_landing_probe(&library, &rdma, 64 << 20, 8)?;
        println!("GPU landing probe: {}", serde_json::to_string(&probe)?);
        anyhow::ensure!(probe.usable(), "CUDA-allocated RDMA landing unavailable: {probe:?}");
        Ok(())
    }

    #[test]
    fn placed_landing_registers_the_whole_range_and_checks_slices() {
        let range = CuteafdDeviceBuffer { ptr: 4096usize as *mut c_void, bytes: 8256 * 128, ..Default::default() };
        let landing = DeviceLanding::placed(range).unwrap();
        assert!(landing.placed && landing.slot_bytes == landing.bytes);
        // 3×2752 and 4×2048 and 2×4096 slots all fit one reserve range.
        for (waves, rows) in [(3usize, 2752usize), (4, 2048), (2, 4096), (1, 4096)] {
            for slot in 0..waves {
                assert!(landing.holds(LandingSlice { offset: slot * rows * 128, bytes: rows * 128 }));
            }
        }
        assert!(!landing.holds(LandingSlice { offset: 8256 * 128 - 64, bytes: 128 }));
        assert!(!landing.holds(LandingSlice { offset: 0, bytes: 0 }));
        assert!(!landing.holds(LandingSlice { offset: usize::MAX, bytes: 2 }));
        assert!(DeviceLanding::placed(CuteafdDeviceBuffer::default()).is_err());
    }

    #[test]
    fn landing_slots_partition_the_rank_allocation() {
        for slots in [1, 2, 3, 4] {
            let range = CuteafdDeviceBuffer { ptr: 4096usize as *mut c_void,
                bytes: slots * 12288, ..Default::default() };
            let landing = DeviceLanding::with_slots(range, slots).unwrap();
            assert_eq!(landing.slot_bytes, 12288);
            assert_eq!(landing.bytes, slots * 12288);
        }
        for (bytes, slots) in [(0, 2), (4096, 0), (4096, 3), (4096, 5)] {
            assert!(DeviceLanding::with_slots(CuteafdDeviceBuffer { bytes, ..Default::default() }, slots).is_err());
        }
    }
}
