//! One serving lane's TP2 encoder FFN, including peer inputs and final addition.
use crate::families::deepseek_v41::v41_backbone_shared::tp2::{
    Wave as SharedWave, Weights as SharedWeights,
};
use crate::shared::experts::rtx::{
    exl3::Exl3Tp2,
    native::NativeTp2,
    pair::{OwnerInputs, OwnerPair, OwnerPolicy, OwnerWeights},
};
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, Bf16Add};
use std::rc::Rc;

/// Weight-only budgets have already been admitted separately from every lane's
/// workspace by the family placement request. Shared loading rechecks peak usage.
pub(crate) fn load_routed_weights<'a>(
    devices: [Device<'a>; 2],
    catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: usize,
    capacity: u32,
    budgets: [usize; 2],
    package: &std::path::Path,
) -> Result<[OwnerWeights<'a>; 2]> {
    if catalog.exl3().is_some() {
        let OwnerPolicy::Exl3 { selection, .. } = OwnerPolicy::exl3(package) else { unreachable!() };
        let workspace = Exl3Tp2::workspace_bytes_for_selection(
            package,
            5120,
            capacity as usize,
            cuteafd_core::Exl3RowPolicy::Nearest,
            selection,
        )?;
        let budgets = budgets
            .map(|budget| {
                budget
                    .checked_add(workspace)
                    .ok_or_else(|| anyhow::anyhow!("TP2 EXL3 load budget overflow"))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("two TP2 budgets"))?;
        return Ok(Exl3Tp2::load_pair_weights(
            devices,
            catalog,
            cuteafd_core::expert_geometry(),
            [package; 2],
            (0..layers).collect(),
            capacity as usize,
            budgets,
            selection,
        )?
        .map(OwnerWeights::Exl3));
    }
    let nvfp4 = catalog.nvfp4().is_some();
    let OwnerPolicy::Native { selection, output, .. } =
        devices[0].run(|| OwnerPolicy::native(devices[0].library, nvfp4))? else { unreachable!() };
    let workspace = NativeTp2::workspace_bytes_for_format(
        devices[0].library,
        capacity as usize,
        selection,
        output,
        nvfp4,
    )?;
    let budgets = budgets
        .map(|budget| {
            budget
                .checked_add(workspace)
                .ok_or_else(|| anyhow::anyhow!("TP2 native load budget overflow"))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("two TP2 budgets"))?;
    Ok(NativeTp2::load_pair_weights(
        devices,
        catalog,
        (0..layers).collect(),
        capacity as usize,
        budgets,
        selection,
        output,
    )?
    .map(OwnerWeights::Native))
}

/// [`load_routed_weights`] over native halves a push group delivered.
pub(crate) fn pushed_routed_weights<'a>(
    devices: [Device<'a>; 2],
    catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: usize,
    capacity: u32,
    budgets: [usize; 2],
    pushed: [Vec<crate::shared::experts::layer::ExpertWeights<'a>>; 2],
) -> Result<[OwnerWeights<'a>; 2]> {
    ensure!(catalog.exl3().is_none() && catalog.nvfp4().is_none(), "pushed coordinator halves are native MXFP4");
    let OwnerPolicy::Native { selection, output, .. } =
        devices[0].run(|| OwnerPolicy::native(devices[0].library, false))? else { unreachable!() };
    let workspace = NativeTp2::workspace_bytes_for_format(devices[0].library, capacity as usize, selection, output, false)?;
    let budgets = [budgets[0].checked_add(workspace).context("TP2 native budget overflow")?,
        budgets[1].checked_add(workspace).context("TP2 native budget overflow")?];
    Ok(NativeTp2::pushed_pair_weights(devices, catalog, (0..layers).collect(), capacity as usize, budgets,
        selection, output, pushed)?.map(OwnerWeights::Native))
}

struct PeerInputs<'a> {
    values: Allocation<'a>,
    wire: Allocation<'a>,
    ids: Allocation<'a>,
    routing: Allocation<'a>,
}
impl<'a> PeerInputs<'a> {
    fn allocation_sizes(rows: u32) -> [usize; 4] { [10240,5280,24,24].map(|width| width*rows as usize) }
    fn new(device: Device<'a>, capacity: u32) -> Result<Self> {
        let [values, wire, ids, routing] = Self::allocation_sizes(capacity);
        Ok(Self {
            values: Allocation::new(device, values)?,
            wire: Allocation::new(device, wire)?,
            ids: Allocation::new(device, ids)?,
            routing: Allocation::new(device, routing)?,
        })
    }
    fn buffers(&self) -> [CuteafdDeviceBuffer; 4] {
        [
            self.values.buffer,
            self.wire.buffer,
            self.ids.buffer,
            self.routing.buffer,
        ]
    }
}

/// One real TP2 FFN execution arena per GPU, shared by all prefill slots.
/// Peer values/wire/ids/routing and final sums are reserve planes, since later
/// hooks may read them. OwnerPair rank GEMM/count/offset/reduction arenas are
/// unit scratch guarded through the final add (including both peer readers).
/// Shared TP2 final sums are reserve planes with their existing 1E guards.
/// Submission bookkeeping is mutex-owned, then protected by the GPU release
/// frontier after the host submit completes; host phase/window stays per slot.
/// One arena does not remove measured cross-unit compute overlap: prof19 k2
/// had 0/680 overlapping route-to-output windows per GPU (3113/2245 ms busy);
/// prof20 k3 had 0/475 (2240/1918 ms). Kernel-only overlap was also exactly zero.
/// Both traces used private per-slot OwnerPairs. Keep one unit arena, not k.
#[cfg(test)]
thread_local! { static SKIP_FFN_ACQUIRE: std::cell::Cell<bool> = const {std::cell::Cell::new(false)}; }
pub(crate) struct Storage<'a> {
    execution: tokio::sync::Mutex<Execution<'a>>,
    streams: Rc<[Stream<'a>; 2]>,
    rows: u32,
    unit: u32,
    layers: usize,
    shared_layers: usize,
}
impl<'a> Storage<'a> {
    pub(crate) fn storage_bytes(library: &cuteafd_ffi::NativeLibrary, rows: u32, unit: u32, nvfp4: bool) -> Result<usize> {
        ensure!(unit > 0 && unit <= 4096 && unit <= rows
            && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS, "invalid TP2 FFN storage geometry");
        Ok(OwnerPair::device_bytes_for(library, unit, cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
                OwnerPolicy::native(library, nvfp4)?)?
            + crate::families::deepseek_v41::v41_backbone_shared::tp2::Storage::storage_bytes(library, rows, unit)?
            + PeerInputs::allocation_sizes(rows).into_iter().sum::<usize>() + rows as usize * 10240)
    }
    pub(crate) fn exl3_storage_bytes(directory: &std::path::Path, library: &cuteafd_ffi::NativeLibrary,
        rows: u32, unit: u32) -> Result<usize> {
        ensure!(unit > 0 && unit <= 4096 && unit <= rows
            && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS, "invalid TP2 EXL3 storage geometry");
        Ok(OwnerPair::device_bytes_for(library, unit, cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
                OwnerPolicy::exl3(directory))?
            + crate::families::deepseek_v41::v41_backbone_shared::tp2::Storage::storage_bytes(library, rows, unit)?
            + PeerInputs::allocation_sizes(rows).into_iter().sum::<usize>() + rows as usize * 10240)
    }
    pub(crate) fn new(routed: [OwnerWeights<'a>; 2], shared: [Rc<Vec<SharedWeights<'a>>>; 2],
        layers: usize, rows: u32, unit: u32,
        modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; 2]) -> Result<Rc<Self>> {
        ensure!(unit > 0 && unit <= rows && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS,
            "invalid TP2 FFN reserve/unit");
        let shared_layers = shared[0].len().min(shared[1].len());
        let execution = Execution::new(routed, shared, layers, unit, rows, modules)?;
        let streams = execution.streams.clone();
        Ok(Rc::new(Self { execution: tokio::sync::Mutex::new(execution), streams, rows, unit, layers, shared_layers }))
    }
}
pub(crate) struct Wave<'a> {
    storage: Rc<Storage<'a>>,
    window: crate::shared::memory::row_window::RowWindow,
}
impl<'a> Wave<'a> {
    pub(crate) fn device_bytes(library: &cuteafd_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        Storage::storage_bytes(library, capacity, capacity, false)
    }
    pub(crate) fn nvfp4_device_bytes(library: &cuteafd_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        Storage::storage_bytes(library, capacity, capacity, true)
    }
    pub(crate) fn exl3_device_bytes(directory: &std::path::Path, library: &cuteafd_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        Storage::exl3_storage_bytes(directory, library, capacity, capacity)
    }
    pub(crate) fn new(routed: [OwnerWeights<'a>; 2], shared: [Rc<Vec<SharedWeights<'a>>>; 2],
        layers: usize, capacity: u32,
        modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; 2]) -> Result<Self> {
        Ok(Self::on_storage(Storage::new(routed, shared, layers, capacity, capacity, modules)?))
    }
    pub(crate) fn on_storage(storage: Rc<Storage<'a>>) -> Self {
        Self { storage, window: crate::shared::memory::row_window::RowWindow::NONE }
    }
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows as usize
                && range.len() <= self.storage.unit as usize, "TP2 FFN window exceeds reserve/unit");
        }
        self.window = window; Ok(())
    }
    pub(crate) fn contains(&self, layer: usize) -> bool { layer < self.storage.layers }
    #[cfg(test)]
    pub(crate) fn window(&self) -> &crate::shared::memory::row_window::RowWindow { &self.window }
    pub(crate) fn contains_shared(&self, layer: usize) -> bool { layer < self.storage.shared_layers }
    pub(crate) fn stream(&self, device: usize) -> &Stream<'a> { &self.storage.streams[device] }
    pub(crate) fn release_scratch(&self) -> Result<()> {
        self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?.released.release()
    }
    pub(crate) fn release_shared_output(&self, destination: usize) -> Result<()> {
        self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?.release_shared_output(destination)
    }
    async fn execution(&self, rows: u32) -> Result<tokio::sync::MutexGuard<'_, Execution<'a>>> {
        ensure!(self.window.admits(rows as usize), "TP2 FFN rows differ from slot window");
        let mut execution = self.storage.execution.lock().await;
        execution.window = self.window.clone(); execution.shared.select_window(self.window.clone())?;
        Ok(execution)
    }
    /// # Safety
    /// Inputs are complete and retained until the enclosing chain drains.
    pub(crate) async unsafe fn execute_shared(&mut self, layer: usize, rows: u32, values: CuteafdDeviceBuffer) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.execute_shared(layer, rows, values).await }
    }
    /// # Safety
    /// Same input/chain lifetime contract as execute_shared.
    pub(crate) async unsafe fn execute_shared_on(&mut self, layer: usize, rows: u32, values: CuteafdDeviceBuffer, destination: usize) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.execute_shared_on(layer, rows, values, destination).await }
    }
    /// # Safety
    /// Source stays immutable and retained until the enclosing chain drains.
    pub(crate) async unsafe fn return_result(&mut self, source: CuteafdDeviceBuffer, destination: usize, rows: u32) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.return_result(source, destination, rows).await }
    }
    /// # Safety
    /// Inputs stay immutable and retained until the enclosing chain drains.
    pub(crate) async unsafe fn execute(&mut self, layer: usize, rows: u32, values: CuteafdDeviceBuffer,
        wire: CuteafdDeviceBuffer, ids: CuteafdDeviceBuffer, routing: CuteafdDeviceBuffer) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.execute(layer, rows, values, wire, ids, routing).await }
    }
}
struct Execution<'a> {
    // Drain uploads and final addition before any referenced allocation drops.
    streams: Rc<[Stream<'a>; 2]>,
    peers: [PeerInputs<'a>; 2],
    shared: SharedWave<'a>,
    routed: OwnerPair<'a>,
    output: [Allocation<'a>; 2],
    add: Bf16Add<'a>,
    /// SM peer copies per destination GPU for device-ordered passes.
    sm: [Option<cuteafd_ffi::V41PeerCopy<'a>>; 2],
    layers: usize,
    capacity: u32,
    window: crate::shared::memory::row_window::RowWindow,
    released: crate::shared::memory::row_window::ScratchGuard<'a>,
}
impl Drop for Execution<'_> {
    fn drop(&mut self) {
        for stream in self.streams.iter() { stream.drain_or_abort("TP2 FFN shared storage"); }
    }
}
/// Copies `bytes` from `source` (the other GPU) into `destination` on
/// `stream`: an SM copy ordered by events under a device-ordered pass (no
/// copy-engine queue held behind an event), else a copy-engine peer copy.
unsafe fn peer_copy(
    stream: &Stream<'_>,
    sm: Option<&cuteafd_ffi::V41PeerCopy<'_>>,
    destination: CuteafdDeviceBuffer,
    source: CuteafdDeviceBuffer,
    bytes: usize,
) -> Result<()> {
    // Device-ordered passes copy decode-width transfers on the SMs and
    // prefill-width ones on the copy engines (`device::sm_peer_copy_for`).
    // The issuing stream is this wave's own, ordered by a
    // `cudaStreamWaitEvent` on the chain head, never a memory wait.
    match sm.filter(|_| crate::shared::memory::device::sm_peer_copy_for(crate::shared::memory::chain::deferred(), bytes)) {
        Some(sm) => unsafe { sm.launch(destination, source, bytes, stream.raw) },
        None => unsafe {
            stream
                .device
                .library
                .copy_peer_async(destination, source, bytes, stream.raw)
        },
    }
}
impl<'a> Execution<'a> {
    pub fn new(
        routed: [OwnerWeights<'a>; 2],
        shared: [Rc<Vec<SharedWeights<'a>>>; 2],
        layers: usize,
        capacity: u32,
        reserve: u32,
        modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; 2],
    ) -> Result<Self> {
        let devices = [routed[0].device(), routed[1].device()];
        ensure!(
            devices[0].id == 0
                && devices[1].id == 1
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid TP2 FFN devices"
        );
        ensure!(
            (1..=40).contains(&layers) && (1..=4096).contains(&capacity),
            "invalid TP2 FFN layer count/capacity"
        );
        for rank in 0..2 {
            ensure!(
                routed[rank].layers().get(..layers).is_some_and(|ids| ids.iter().copied().eq(0..layers))
                    && shared[rank].len() >= layers
                    && shared[rank][0].device().id == devices[rank].id
                    && std::ptr::eq(shared[rank][0].device().library, devices[rank].library),
                "TP2 FFN weights do not cover the requested devices/layers"
            );
        }
        for rank in 0..2 {
            devices[rank].run(|| devices[rank].library.cuda_enable_peer(1 - rank as i32))?;
        }
        Ok(Self {
            streams: Rc::new([Stream::new(devices[0])?, Stream::new(devices[1])?]),
            peers: [
                PeerInputs::new(devices[0], reserve)?,
                PeerInputs::new(devices[1], reserve)?,
            ],
            shared: SharedWave::on_storage(crate::families::deepseek_v41::v41_backbone_shared::tp2::Storage::new(shared, reserve, capacity)?),
            // One resident load serves every arena: each pair is admitted at
            // its own capacity (prefill unit cap, or a decode lane's width).
            routed: OwnerPair::with_modules(
                [routed[0].with_max_rows(capacity as usize)?, routed[1].with_max_rows(capacity as usize)?],
                capacity,
                cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
                std::env::var_os("CUTEAFD_TP2_TIMING").is_some(),
                modules,
            )?,
            output: [
                Allocation::new(devices[0], reserve as usize * 10240)?,
                Allocation::new(devices[1], reserve as usize * 10240)?,
            ],
            add: devices[0].library.bf16_add()?,
            sm: [Some(devices[0].run(|| devices[0].library.v41_peer_copy())?),
                Some(devices[1].run(|| devices[1].library.v41_peer_copy())?)],
            layers,
            capacity, window: crate::shared::memory::row_window::RowWindow::NONE,
            released: crate::shared::memory::row_window::ScratchGuard::new(devices[0].library, &[0,1])?,
        })
    }
    pub fn contains(&self, layer: usize) -> bool {
        layer < self.layers
    }
    /// The lane's stream on GPU `device` (0 or 1).
    pub(crate) fn stream(&self, device: usize) -> &Stream<'a> {
        &self.streams[device]
    }

    pub fn contains_shared(&self, layer: usize) -> bool {
        self.shared.contains(layer)
    }

    /// Execute only the shared TP2 contribution while routed experts
    /// run on Sparks. Reuses the encoder FFN's lane-local input/shared storage.
    /// # Safety
    /// Completed normalized input remains immutable through completion or drain.
    pub async unsafe fn execute_shared(
        &mut self,
        layer: usize,
        rows: u32,
        values: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        unsafe {
            self.execute_shared_on(layer, rows, values, values.device_id as usize)
                .await
        }
    }

    /// Same input lifetime as `execute_shared`; select the reduction owner so
    /// Spark collection can remain on its transport GPU for encoder layers too.
    /// # Safety
    /// Input storage remains complete, immutable and live through completion or drain.
    pub async unsafe fn execute_shared_on(
        &mut self,
        layer: usize,
        rows: u32,
        values: CuteafdDeviceBuffer,
        destination: usize,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            self.contains_shared(layer)
                && destination < 2
                && rows > 0
                && rows <= self.capacity
                && matches!(values.device_id, 0 | 1)
                && !values.ptr.is_null()
                && values.bytes >= rows as usize * 10240,
            "TP2 shared input layer/device/extent differs"
        );
        // Peer DMA is never queued behind an unresolved dependency: settle any
        // chained producer of `values` on the host first.
        if !crate::shared::memory::chain::deferred() {
            crate::shared::memory::chain::settle(self.streams[0].device.library)?;
        }
        let local = values.device_id as usize;
        let remote = 1 - local;
        let upload = &self.streams[remote];
        let mut guard = upload.cancellation_guard("TP2 shared input upload");
        if crate::shared::memory::chain::deferred() {
            // Both GPUs' streams follow the input's producers (the chain head).
            self.streams[0].join_chain()?;
            self.streams[1].join_chain()?;
        } else {
            upload.wait().await?;
        }
        let peer = self.window.apply(self.peers[remote].values.buffer, 10240);
        let sm = self.sm[remote].as_ref();
        upload
            .device
            .run(|| unsafe { peer_copy(upload, sm, peer, values, rows as usize * 10240) })?;
        let mut inputs = [values; 2];
        inputs[remote] = peer;
        let output = unsafe {
            self.shared
                .execute(
                    layer,
                    rows,
                    destination,
                    inputs,
                    [&self.streams[0], &self.streams[1]],
                )
                .await?
        };
        guard.complete = true;
        Ok(output)
    }

    /// Return a completed remote-expert reduction to its block's GPU using
    /// lane-owned scratch. Stream::wait drains this copy on cancellation.
    /// # Safety
    /// Source storage remains immutable and live until the transfer completes or drains.
    pub async unsafe fn return_result(
        &mut self,
        source: CuteafdDeviceBuffer,
        destination: usize,
        rows: u32,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            destination < 2
                && rows > 0
                && rows <= self.capacity
                && source.bytes >= rows as usize * 10240,
            "invalid TP2 result transfer"
        );
        if !crate::shared::memory::chain::deferred() {
            crate::shared::memory::chain::settle(self.streams[0].device.library)?;
        }
        let stream = &self.streams[destination];
        let mut output = self.window.apply(self.output[destination].buffer, 10240);
        output.bytes = rows as usize * 10240;
        let sm = self.sm[destination].as_ref();
        let queued = stream.device.run(|| unsafe {
            if crate::shared::memory::chain::deferred() {
                crate::shared::memory::chain::join(stream.device.library, stream.raw)?;
            }
            peer_copy(stream, sm, output, source, output.bytes)
        });
        let drained = stream.complete().await;
        queued.and(drained)?;
        Ok(output)
    }

    /// # Safety
    /// Local inputs are complete, immutable, and retained through return or
    /// cancellation. They all represent the same normalized FFN rows. Each lane
    /// owns a distinct Wave. Returned storage is valid until its next execute.
    pub async unsafe fn execute(
        &mut self,
        layer: usize,
        rows: u32,
        values: CuteafdDeviceBuffer,
        wire: CuteafdDeviceBuffer,
        ids: CuteafdDeviceBuffer,
        routing: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            self.contains(layer) && rows > 0 && rows <= self.capacity,
            "TP2 FFN layer/rows unavailable"
        );
        ensure!(
            matches!(values.device_id, 0 | 1),
            "invalid TP2 FFN input device"
        );
        let local = values.device_id as usize;
        let remote = 1 - local;
        let inputs = [values, wire, ids, routing];
        let widths = [10240, 5280, 24, 24];
        for (buffer, width) in inputs.into_iter().zip(widths) {
            ensure!(
                buffer.device_id == local as i32
                    && !buffer.ptr.is_null()
                    && buffer.bytes >= rows as usize * width,
                "TP2 FFN input extent/device mismatch"
            );
        }
        #[cfg(test)]
        let skip = SKIP_FFN_ACQUIRE.with(|s|s.get());
        #[cfg(not(test))]
        let skip = false;
        if !skip {for stream in self.streams.iter() {
            stream.device.run(|| unsafe { self.released.acquire(stream.raw) })?;
        }}
        let upload = &self.streams[remote];
        let mut guard = upload.cancellation_guard("TP2 FFN input upload");
        // Do not submit DMA behind an unresolved stream dependency: a blocked
        // copy packet can hold up independent lanes on the shared copy engine.
        // Chained producers of these inputs are settled on the host as well.
        if !crate::shared::memory::chain::deferred() {
            crate::shared::memory::chain::settle(upload.device.library)?;
        }
        if crate::shared::memory::chain::deferred() {
            self.streams[0].join_chain()?;
            self.streams[1].join_chain()?;
        } else {
            upload.wait().await?;
        }
        let peer = self.peers[remote].buffers().into_iter().zip(widths)
            .map(|(buffer, width)| self.window.apply(buffer, width)).collect::<Vec<_>>();
        let peer: [CuteafdDeviceBuffer; 4] = peer.try_into().ok().unwrap();
        let sm = self.sm[remote].as_ref();
        upload.device.run(|| {
            for ((source, destination), width) in inputs.into_iter().zip(peer).zip(widths) {
                unsafe {
                    peer_copy(upload, sm, destination, source, rows as usize * width)?;
                }
            }
            Ok(())
        })?;
        let mut ranks = [inputs; 2];
        ranks[remote] = peer;
        // NVFP4 consumes the BF16 values already broadcast for shared experts,
        // not the native family's FP8 wire payload.
        let routed_input = if self.routed.uses_bf16_input() { 0 } else { 1 };
        // Each backend records its own producer event after the queued copies;
        // local compute can begin before peer input transfers finish.
        let (routed, shared) = tokio::try_join!(
            unsafe {
                self.routed.execute(
                    layer,
                    rows,
                    local,
                    std::array::from_fn(|rank| OwnerInputs {
                        input: ranks[rank][routed_input],
                        ids: ranks[rank][2],
                        weights: ranks[rank][3],
                        producer: &self.streams[rank],
                    }),
                )
            },
            unsafe {
                self.shared.execute(
                    layer,
                    rows,
                    local,
                    [ranks[0][0], ranks[1][0]],
                    [&self.streams[0], &self.streams[1]],
                )
            }
        )?;
        guard.complete = true;
        let stream = &self.streams[local];
        let output = self.window.apply(self.output[local].buffer, 10240);
        let queued = stream.device.run(|| unsafe {
            // Deferred: the routed and shared sums were merged into the chain head.
            if crate::shared::memory::chain::deferred() {
                crate::shared::memory::chain::join(stream.device.library, stream.raw)?;
            }
            self.add
                .launch(routed, shared, output, rows as usize * 5120, stream.raw)
        });
        let drained = stream.complete().await;
        queued.and(drained)?;
        // The add above read the shared output: release it for the next layer
        // (it is on the chain head now; no-op outside a chain).
        self.shared.release_output(local)?;
        self.released.release()?;
        let mut result = output;
        result.bytes = rows as usize * 10240;
        Ok(result)
    }
    /// The Spark reduce that read the shared output on `destination` finished
    /// on the unit's chain: the next layer's shared add may overwrite it.
    pub(crate) fn release_shared_output(&self, destination: usize) -> Result<()> {
        self.shared.release_output(destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires combined TP2 CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT, two GPUs"]
    fn real_tp2_ffn_broadcast_and_combine_independent_lanes() -> Result<()> {
        let lib =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let devices = [
            Device {
                library: &lib,
                id: 0,
            },
            Device {
                library: &lib,
                id: 1,
            },
        ];
        let routed = load_routed_weights(
            devices,
            &catalog,
            1,
            16,
            [4_000_000_000; 2],
            std::path::Path::new(&std::env::var("CUTEAFD_EXL3_AOT").unwrap_or_default()),
        )?;
        let shared: [Rc<Vec<SharedWeights<'_>>>; 2] = [0, 1]
            .map(|rank| {
                (0..40)
                    .map(|layer| {
                        SharedWeights::load(
                            devices[rank],
                            &catalog,
                            layer,
                            SharedWeights::load_peak_device_bytes(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(Rc::new)
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("two shared ranks");
        let mut lanes = [
            Wave::new(routed.clone(), shared.clone(), 1, 16, [None, None])?,
            Wave::new(routed.clone(), shared.clone(), 1, 16, [None, None])?,
        ];
        let mut reference_routed = OwnerPair::new(
            routed,
            16,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
            std::env::var_os("CUTEAFD_TP2_TIMING").is_some(),
        )?;
        let mut reference_shared = SharedWave::new(shared, 16)?;
        let inputs = [
            PeerInputs::new(devices[0], 16)?,
            PeerInputs::new(devices[1], 16)?,
        ];
        let reference_inputs = [
            PeerInputs::new(devices[0], 16)?,
            PeerInputs::new(devices[1], 16)?,
        ];
        let producers = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        assert!(lanes[0].contains_shared(39));
        assert!(!lanes[0].contains_shared(40));
        assert!(!lanes[0].contains(20));
        for layer in [20, 39] {
            for rows in [1u32, 16, 1] {
                for local in 0..2 {
                    let host: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            let value = ((i + layer + local) % 17) as f32 / 32.0 - 0.25;
                            ((value.to_bits() >> 16) as u16).to_ne_bytes()
                        })
                        .collect();
                    for rank in 0..2 {
                        devices[rank].run(|| {
                            devices[rank]
                                .library
                                .copy_h2d(reference_inputs[rank].values.buffer, &host)
                        })?;
                    }
                    devices[local].run(|| {
                        devices[local]
                            .library
                            .copy_h2d(inputs[local].values.buffer, &host)
                    })?;
                    let actual = runtime.block_on(unsafe {
                        lanes[local].execute_shared(layer, rows, inputs[local].values.buffer)
                    })?;
                    let expected = runtime.block_on(unsafe {
                        reference_shared.execute(
                            layer,
                            rows,
                            local,
                            [
                                reference_inputs[0].values.buffer,
                                reference_inputs[1].values.buffer,
                            ],
                            [&producers[0], &producers[1]],
                        )
                    })?;
                    devices[local].run(|| {
                        let mut left = vec![0; actual.bytes];
                        let mut right = vec![0; expected.bytes];
                        devices[local].library.copy_d2h(&mut left, actual)?;
                        devices[local].library.copy_d2h(&mut right, expected)?;
                        assert_eq!(
                            left, right,
                            "decoder shared TP2 broadcast/reduction differs"
                        );
                        Ok(())
                    })?;
                    let returned = runtime.block_on(async {
                        let peer = unsafe {
                            lanes[local]
                                .execute_shared_on(
                                    layer,
                                    rows,
                                    inputs[local].values.buffer,
                                    1 - local,
                                )
                                .await?
                        };
                        unsafe { lanes[local].return_result(peer, local, rows).await }
                    })?;
                    devices[local].run(|| {
                        let mut left = vec![0; returned.bytes];
                        let mut right = vec![0; expected.bytes];
                        lib.copy_d2h(&mut left, returned)?;
                        lib.copy_d2h(&mut right, expected)?;
                        ensure!(
                            left == right,
                            "peer-owned shared reduction and return differs"
                        );
                        Ok(())
                    })?;
                    assert_eq!(lib.cuda_get_device()?, 0);
                }
            }
        }
        for (rows, bf16, fp8) in [
            (1, 0u16, 0u8),
            (16, 0x3f80, 0x38),
            (1, 0x3f00, 0x30),
            (16, 0, 0),
        ] {
            let frames: [[Vec<u8>; 4]; 2] = std::array::from_fn(|rank| {
                let negative = rank == 1 && bf16 != 0;
                let values: Vec<u8> = (bf16 ^ if negative { 0x8000 } else { 0 })
                    .to_ne_bytes()
                    .into_iter()
                    .cycle()
                    .take(16 * 10240)
                    .collect();
                let mut wire = vec![0; 16 * 5280];
                for row in wire.chunks_exact_mut(5280) {
                    row[..5120].fill(fp8 ^ if negative { 0x80 } else { 0 });
                    row[5120..].fill(127);
                }
                let ids: Vec<u8> = (0..96i32)
                    .flat_map(|i| (i % 6 + rank as i32 * 6).to_ne_bytes())
                    .collect();
                let routing: Vec<u8> = (0..96)
                    .flat_map(|i| {
                        ([0.1f32, 0.15, 0.2, 0.25, 0.1, 0.2][(i + rank) % 6]).to_ne_bytes()
                    })
                    .collect();
                [values, wire, ids, routing]
            });
            for rank in 0..2 {
                devices[rank].run(|| {
                    for (buffer, bytes) in inputs[rank].buffers().into_iter().zip(&frames[rank]) {
                        lib.copy_h2d(buffer, bytes)?;
                    }
                    Ok(())
                })?;
            }
            let [left, right] = &mut lanes;
            let a = inputs[0].buffers();
            let b = inputs[1].buffers();
            let (x, y) = runtime.block_on(async {
                tokio::join!(
                    unsafe { left.execute(0, rows, a[0], a[1], a[2], a[3]) },
                    unsafe { right.execute(0, rows, b[0], b[1], b[2], b[3]) }
                )
            });
            let mut actual = [
                vec![0; rows as usize * 10240],
                vec![0; rows as usize * 10240],
            ];
            for (rank, buffer) in [x?, y?].into_iter().enumerate() {
                devices[rank].run(|| lib.copy_d2h(&mut actual[rank], buffer))?;
            }
            assert_eq!(
                actual[0] != actual[1],
                bf16 != 0,
                "different lane inputs must remain independent"
            );
            for lane in 0..2 {
                for rank in 0..2 {
                    devices[rank].run(|| {
                        for (buffer, bytes) in reference_inputs[rank]
                            .buffers()
                            .into_iter()
                            .zip(&frames[lane])
                        {
                            lib.copy_h2d(buffer, bytes)?;
                        }
                        Ok(())
                    })?;
                }
                let route = runtime.block_on(unsafe {
                    reference_routed.execute(
                        0,
                        rows,
                        0,
                        std::array::from_fn(|rank| OwnerInputs {
                            input: reference_inputs[rank].wire.buffer,
                            ids: reference_inputs[rank].ids.buffer,
                            weights: reference_inputs[rank].routing.buffer,
                            producer: &producers[rank],
                        }),
                    )
                })?;
                let shared = runtime.block_on(unsafe {
                    reference_shared.execute(
                        0,
                        rows,
                        0,
                        [
                            reference_inputs[0].values.buffer,
                            reference_inputs[1].values.buffer,
                        ],
                        [&producers[0], &producers[1]],
                    )
                })?;
                let mut route_bytes = vec![0; actual[0].len()];
                let mut shared_bytes = vec![0; actual[0].len()];
                lib.copy_d2h(&mut route_bytes, route)?;
                lib.copy_d2h(&mut shared_bytes, shared)?;
                for ((a, b), out) in route_bytes
                    .chunks_exact(2)
                    .zip(shared_bytes.chunks_exact(2))
                    .zip(actual[lane].chunks_exact(2))
                {
                    let decode =
                        |v: &[u8]| f32::from_bits((u16::from_ne_bytes([v[0], v[1]]) as u32) << 16);
                    let sum = decode(a) + decode(b);
                    assert!(sum.is_finite());
                    let bits = sum.to_bits();
                    let expected = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
                    assert_eq!(u16::from_ne_bytes([out[0], out[1]]), expected);
                }
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        for capacity in [1, 16, 80, 256, 1024, 4096] {
            eprintln!(
                "TP2 FFN capacity={capacity} workspace_bytes_per_gpu_per_lane={}",
                Wave::device_bytes(&lib, capacity)?
            );
        }
        // Deliberately hold one lane's upload while the other completes real
        // kernels. A bounded release avoids hanging even if a join regresses.
        use std::ffi::c_void;
        use std::{
            future::Future,
            sync::atomic::{AtomicBool, Ordering},
            task::{Context, Poll, Waker},
        };
        unsafe extern "C" fn hold(data: *mut c_void) {
            let release = unsafe { &*data.cast::<AtomicBool>() };
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        struct CudaLibrary(*mut c_void);
        impl Drop for CudaLibrary {
            fn drop(&mut self) {
                unsafe {
                    libc::dlclose(self.0);
                }
            }
        }
        let cuda =
            CudaLibrary(unsafe { libc::dlopen(c"libcudart.so.13".as_ptr(), libc::RTLD_NOW) });
        ensure!(!cuda.0.is_null(), "CUDA runtime unavailable");
        let symbol = unsafe { libc::dlsym(cuda.0, c"cudaLaunchHostFunc".as_ptr()) };
        ensure!(!symbol.is_null(), "CUDA callback API unavailable");
        let launch: unsafe extern "C" fn(
            *mut c_void,
            unsafe extern "C" fn(*mut c_void),
            *mut c_void,
        ) -> i32 = unsafe { std::mem::transmute(symbol) };
        let release = AtomicBool::new(false);
        struct Release<'a>(&'a AtomicBool);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _release = Release(&release);
        let [left, right] = &mut lanes;
        devices[1].run(|| {
            ensure!(
                unsafe {
                    launch(
                        left.storage.streams[1].raw,
                        hold,
                        (&release as *const AtomicBool).cast_mut().cast(),
                    )
                } == 0,
                "could not stall TP2 upload"
            );
            Ok(())
        })?;
        let a = inputs[0].buffers();
        let b = inputs[1].buffers();
        let _entered = runtime.enter();
        let mut blocked = Box::pin(unsafe { left.execute(0, 16, a[0], a[1], a[2], a[3]) });
        let _unwind_release = Release(&release);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(blocked.as_mut().poll(&mut context), Poll::Pending));
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                release.store(true, Ordering::Release);
            });
            for rank in 0..2 {
                devices[rank].run(|| unsafe {
                    lib.copy_peer_async(
                        right.storage.execution.try_lock()?.output[rank].buffer,
                        inputs[1 - rank].values.buffer,
                        10240,
                        right.storage.streams[rank].raw,
                    )
                })?;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            for rank in 0..2 {
                assert!(
                    devices[rank]
                        .run(|| unsafe { lib.cuda_stream_query(right.storage.streams[rank].raw) })?,
                    "unrelated peer copy stalled on GPU {rank}"
                );
            }
            let started = std::time::Instant::now();
            let mut other = Box::pin(unsafe { right.execute(0, 16, b[0], b[1], b[2], b[3]) });
            let initial = other.as_mut().poll(&mut context);
            eprintln!(
                "independent lane first poll {:?}, pending={}",
                started.elapsed(),
                initial.is_pending()
            );
            let result = match initial {
                Poll::Ready(result) => result,
                Poll::Pending => runtime.block_on(other),
            };
            eprintln!("independent lane completed {:?}", started.elapsed());
            let independent = !release.load(Ordering::Acquire);
            drop(blocked); // Cancellation must drain before releasing input owners.
            assert!(release.load(Ordering::Acquire));
            result?;
            assert!(
                independent,
                "one lane's stalled input upload held the other lane"
            );
            Ok(())
        })?;
        // Reuse the cancelled lane's same streams, events, and scratch.
        runtime.block_on(unsafe { left.execute(0, 16, a[0], a[1], a[2], a[3]) })?;
        release.store(false, Ordering::Release);
        devices[1].run(|| {
            ensure!(
                unsafe {
                    launch(
                        left.storage.streams[1].raw,
                        hold,
                        (&release as *const AtomicBool).cast_mut().cast(),
                    )
                } == 0,
                "stalling shared upload failed"
            );
            Ok(())
        })?;
        let mut blocked_shared = Box::pin(unsafe { left.execute_shared(20, 16, a[0]) });
        let shared_unwind_release = Release(&release);
        assert!(matches!(
            blocked_shared.as_mut().poll(&mut context),
            Poll::Pending
        ));
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                release.store(true, Ordering::Release);
            });
            let result = runtime.block_on(unsafe { right.execute_shared(39, 1, b[0]) });
            let independent = !release.load(Ordering::Acquire);
            drop(blocked_shared);
            result?;
            assert!(independent, "decoder shared work joined the other lane");
            Ok(())
        })?;
        drop(shared_unwind_release);
        runtime.block_on(unsafe { left.execute_shared(20, 16, a[0]) })?;
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::shared::memory::{chain::StageChain,row_window::RowWindow};
    fn run(guarded:bool)->Result<()> {
        let lib=unsafe {cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};lib.cuda_set_device(0)?;
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device {library:&lib,id:0},Device {library:&lib,id:1}];
        let routed=load_routed_weights(devices,&catalog,1,16,[4_000_000_000;2],
            std::path::Path::new(&std::env::var("CUTEAFD_EXL3_AOT").unwrap_or_default()))?;
        let shared=[Rc::new(vec![SharedWeights::load(devices[0],&catalog,0,SharedWeights::load_peak_device_bytes())?]),
            Rc::new(vec![SharedWeights::load(devices[1],&catalog,0,SharedWeights::load_peak_device_bytes())?])];
        let inputs=[PeerInputs::new(devices[0],16)?,PeerInputs::new(devices[1],16)?];
        for (seed,input) in inputs.iter().enumerate() {
            let value=if seed==0 {0x3e80u16}else {0xbe80u16};
            let values=value.to_ne_bytes().repeat(16*5120);
            let mut wire=vec![0;16*5280];for row in wire.chunks_exact_mut(5280) {
                row[..5120].fill(if seed==0 {0x28}else {0xa8});row[5120..].fill(127);
            }
            let ids:Vec<u8>=(0..96u32).flat_map(|i|(i%6).to_ne_bytes()).collect();
            let routing=(1.0f32/6.0).to_ne_bytes().repeat(96);
            input.values.device.run(|| {
                for (buffer,host) in input.buffers().into_iter().zip([&values,&wire,&ids,&routing]) {lib.copy_h2d(buffer,host)?;}
                Ok(())
            })?;
        }
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let mut private=Wave::new(routed.clone(),shared.clone(),1,16,[None,None])?;
        let a=inputs[0].buffers();
        let private_a=runtime.block_on(unsafe {private.execute(0,16,a[0],a[1],a[2],a[3])})?;
        let mut want_a=vec![0;16*10240];devices[0].run(||lib.copy_d2h(&mut want_a,private_a))?;
        let private_rank=private.storage.execution.try_lock().unwrap().routed.rank_output(0);
        let mut want=vec![0;private_rank.bytes];lib.copy_d2h(&mut want,private_rank)?;
        let b=inputs[1].buffers();
        let private_b=runtime.block_on(unsafe {private.execute(0,16,b[0],b[1],b[2],b[3])})?;
        let mut want_b=vec![0;16*10240];devices[1].run(||lib.copy_d2h(&mut want_b,private_b))?;
        let storage=Storage::new(routed,shared,1,48,16,[None,None])?;
        let mut slots=[Wave::on_storage(storage.clone()),Wave::on_storage(storage.clone())];
        for (i,slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::within(i*16..i*16+16,48)?)?;
            let b=inputs[i].buffers();runtime.block_on(unsafe {slot.execute(0,16,b[0],b[1],b[2],b[3])})?;
        }
        let rank=storage.execution.try_lock().unwrap().routed.rank_output(0);
        let reader=Stream::new(devices[1])?;let saved=Allocation::new(devices[1],want.len())?;
        let reader_copy=devices[1].run(||lib.v41_peer_copy())?;
        devices[1].run(||unsafe {reader_copy.launch(saved.buffer,rank,want.len(),reader.raw)})?;reader.drain()?;
        let chains=[StageChain::on_devices(&lib,&[0,1])?,StageChain::on_devices(&lib,&[0,1])?];
        let gate=devices[1].run(||unsafe {cuteafd_ffi::test_support::CudaStreamGate::new(&lib,reader.raw)})?;
        runtime.block_on(chains[0].handle().scope(async {
            let b=inputs[0].buffers();unsafe {slots[0].execute(0,16,b[0],b[1],b[2],b[3]).await?;}
            devices[1].run(||unsafe {crate::shared::memory::chain::join(&lib,reader.raw)?;
                reader_copy.launch(saved.buffer,rank,want.len(),reader.raw)?;
                crate::shared::memory::chain::finish(&lib,reader.raw)})?;
            slots[0].release_scratch()
        }))?;
        SKIP_FFN_ACQUIRE.with(|s|s.set(!guarded));
        let queued=runtime.block_on(chains[1].handle().scope(async {
            let b=inputs[1].buffers();unsafe {slots[1].execute(0,16,b[0],b[1],b[2],b[3]).await?;}anyhow::Ok(())
        }));SKIP_FFN_ACQUIRE.with(|s|s.set(false));queued?;
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(if guarded {2}else {10});let mut done=false;
        while std::time::Instant::now()<deadline {
            let execution = storage.execution.try_lock().unwrap();
            let stream = execution.routed.rank_stream(0);
            if stream.device.run(||unsafe {lib.cuda_stream_query(stream.raw)})? {done=true;break;}
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let final_done = devices[1].run(||unsafe {lib.cuda_stream_query(storage.streams[1].raw)})?;
        eprintln!("OwnerPair guarded={guarded}: rank0_ready={done}, final_add_ready={final_done} before reader release");
        devices[1].run(|| {drop(gate);Ok(())})?;for chain in &chains {chain.drain()?;}reader.drain()?;
        assert_eq!(done,!guarded,"TP2 FFN B rank completion before A rank-reader release");
        let mut got=vec![0;want.len()];devices[1].run(||lib.copy_d2h(&mut got,saved.buffer))?;
        if guarded {assert_eq!(got,want);}else {assert_ne!(got,want,"unguarded actual OwnerPair did not clobber");}
        let execution=storage.execution.try_lock().unwrap();
        for (slot,expected) in [(0,want_a),(1,want_b)] {
            let output=RowWindow::within(slot*16..slot*16+16,48)?.apply(execution.output[slot].buffer,10240);
            let mut actual=vec![0;expected.len()];
            devices[slot].run(||lib.copy_d2h(&mut actual,output))?;
            assert_eq!(actual,expected,"slot {slot} full FFN output differs from private run");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires actual TP2 routed/shared kernels, snapshot and two GPUs"]
    fn tp2_ffn_shared_owner_preserves_gated_peer_reader()->Result<()> {for _ in 0..3 {run(true)?;}Ok(())}
    #[test]
    #[ignore = "requires actual TP2 routed/shared kernels, snapshot and two GPUs"]
    fn tp2_ffn_without_owner_guard_clobbers_gated_peer_reader()->Result<()> {for _ in 0..3 {run(false)?;}Ok(())}
    /// 2-RTX assembly at k = 2/3/4: one routed load at the unit cap serves the
    /// slots' shared arena (reserve 8192, unit U) and both decode lanes at
    /// their own width, each pair admitted at its arena's capacity.
    #[test]
    #[ignore = "requires actual TP2 routed/shared kernels, snapshot and two GPUs"]
    fn one_routed_load_builds_slot_storage_and_decode_lanes_at_every_k()->Result<()> {
        let lib=unsafe {cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};lib.cuda_set_device(0)?;
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device {library:&lib,id:0},Device {library:&lib,id:1}];
        let decode=crate::families::deepseek_v41::v41_native_serve::DECODE_LANE_ROWS;
        for unit in [4096u32,2752,2048] {
            let routed=load_routed_weights(devices,&catalog,1,unit,[8_000_000_000;2],
                std::path::Path::new(&std::env::var("CUTEAFD_EXL3_AOT").unwrap_or_default()))?;
            let shared=[Rc::new(vec![SharedWeights::load(devices[0],&catalog,0,SharedWeights::load_peak_device_bytes())?]),
                Rc::new(vec![SharedWeights::load(devices[1],&catalog,0,SharedWeights::load_peak_device_bytes())?])];
            let storage=Storage::new(routed.clone(),shared.clone(),1,8192,unit,[None,None])?;
            let lanes=[Wave::new(routed.clone(),shared.clone(),1,decode,[None,None])?,
                Wave::new(routed.clone(),shared.clone(),1,decode,[None,None])?];
            assert_eq!(storage.execution.try_lock().unwrap().capacity,unit);
            for lane in &lanes {assert_eq!(lane.storage.execution.try_lock().unwrap().capacity,decode);}
            assert!(routed.iter().all(|w| w.max_rows()==unit as usize),"re-admission must not change the loaded handle");
            eprintln!("k unit {unit}: slot storage + 2 decode lanes at {decode} built");
        }
        Ok(())
    }
    #[test]
    fn peer_input_storage_bytes_equal_planned_allocations() {
        assert_eq!(PeerInputs::allocation_sizes(8192).into_iter().sum::<usize>(),8192*15568);
    }
}
