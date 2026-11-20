//! Native coordinator experts; format-independent boundary retained by the forward.
use crate::shared::{experts::rtx::dense::{DenseLoad, Operand, Rect},
    memory::{device::{Allocation, Device, Stream}, receipts::Inventory, weight_arena::DeviceWeightArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, memory_layout::ArenaTag};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, fp8_moe::{Fp8MoeMetadata, Fp8MoeModule, Fp8MoeWeights}};
use cuteafd_loader::{families::inkling2::Inkling2Config, plan::Checkpoint,
    serving_capacity::WeightArenaLayout, shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES}};
use std::{path::Path, rc::Rc};

/// The forward knows only normalized inputs/routes and the stream-ordered
/// output. EXL3 K4 can supply the same boundary without editing layer math.
pub(crate) trait Dispatcher {
    fn wire_input(&self) -> bool { true }
    fn enqueue(&self, input: Buffer, ids: Buffer, weights: Buffer, output: Buffer,
        rows: usize, stream: &Stream<'_>) -> Result<()>;
}

/// Native MXFP8 and MC EXL3 onboarding use the shared whole-layer backend.
/// Its output dtype is independent of K32 ingress; narrow EXL3 token sums only
/// once at the same BF16 family boundary used by remote expert replies.
pub(crate) struct Resident<'a> {
    layer: usize,
    backend: Rc<std::cell::RefCell<Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>>>,
    stream: Rc<Stream<'a>>,
}
impl<'a> Resident<'a> {
    pub fn new(layer: usize, backend: Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>,
        stream: Rc<Stream<'a>>) -> Result<Self> {
        ensure!(backend.shard() == crate::shared::experts::rtx::RtxShard::Whole
            && backend.layers().contains(&layer) && backend.device() == stream.device.id,
            "Inkling resident adapter requires a whole native backend");
        let element = if backend.partial() == crate::shared::experts::rtx::PartialDtype::F32 { 4 } else { 2 };
        ensure!(!backend.output().is_null() && backend.workspace_bytes() >= 4608*element,
            "Inkling resident workspace cannot hold one output row");
        Ok(Self { layer, backend: Rc::new(std::cell::RefCell::new(backend)), stream })
    }
    /// Bind already committed push payloads through the shared whole executor;
    /// there is no load/mount fallback and no duplicate family weight arena.
    pub fn from_pushed(stream: Rc<Stream<'a>>, catalog: &cuteafd_loader::OfficialV41Catalog,
        native_lib: &Path, backing: &cuteafd_loader::placement::families::inkling2::LocalBacking,
        rows: usize, pushed: crate::shared::experts::rtx::push::PushedExperts<'a>) -> Result<Vec<Self>> {
        use crate::shared::experts::rtx::{exl3::Exl3Tp2, whole::Whole, RtxExpertLayer, RtxSelectionPolicy};
        ensure!(backing.gpu == 0 && !backing.layers.is_empty(), "Inkling whole backend differs from selected rank");
        let workspace = backing.workspace.iter().try_fold(0usize, |bytes, (_, size)|
            bytes.checked_add(usize::try_from(*size)?).context("Inkling resident workspace overflow"))?;
        let budget = backing.arena.bytes().checked_add(workspace).context("Inkling resident owner budget overflow")?;
        let device = stream.device;
        let backend: Box<dyn RtxExpertLayer + 'a> = match pushed {
            crate::shared::experts::rtx::push::PushedExperts::Fp8(layers) => {
                let tensors = catalog.fp8().context("Inkling native pushed catalog absent")?
                    .for_layer_metadata(backing.layers[0])?;
                let package = crate::shared::experts::fp8::package_directory(native_lib, 1, tensors.format());
                Box::new(Whole::from_pushed(device, &tensors, &package, backing.layers.clone(), rows, budget, layers)?)
            }
            crate::shared::experts::rtx::push::PushedExperts::Exl3(layers) => {
                let stem = catalog.exl3().context("Inkling EXL3 pushed catalog absent")?.package_name("inkling2");
                let package = native_lib.parent().context("Inkling native lib directory absent")?
                    .join("exl3").join(stem).join("rtx-tp1");
                Box::new(Exl3Tp2::pushed_whole_weights(device, catalog, &package, backing.layers.clone(), rows,
                    budget, RtxSelectionPolicy::CanonicalDecode, layers)?.execution()?)
            }
            _ => anyhow::bail!("Inkling pushed backend storage differs from its selected native/MC catalog"),
        };
        ensure!(backend.workspace_bytes() == workspace, "Inkling resident workspace differs from admitted physical requests");
        Self::for_layers(backend, stream)
    }

    /// All admitted layers of one backend share its single physical scratch
    /// and output owner; handles never instantiate a backend per layer.
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        self.backend.borrow().allocation_receipts(inventory, prefix)
    }

    pub fn for_layers(backend: Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>,
        stream: Rc<Stream<'a>>) -> Result<Vec<Self>> {
        let layers = backend.layers().to_vec();
        ensure!(layers.windows(2).all(|pair| pair[0] < pair[1]), "Inkling resident layer inventory is not ordered");
        let first = *layers.first().context("Inkling resident backend has no layers")?;
        let owner = Self::new(first, backend, stream)?;
        Ok(layers.into_iter().map(|layer| Self { layer, backend: owner.backend.clone(), stream: owner.stream.clone() }).collect())
    }
}
struct SplitResident<'a> {
    layer: usize,
    owner: Rc<std::cell::RefCell<crate::shared::experts::rtx::owner_tp2::OwnerTp2<'a>>>,
    // The shared owner holds raw stream handles through its terminal drain.
    _streams: [Rc<Stream<'a>>; 2],
}
impl Dispatcher for SplitResident<'_> {
    fn wire_input(&self) -> bool { self.owner.borrow().wire }
    fn enqueue(&self, input: Buffer, ids: Buffer, weights: Buffer, output: Buffer,
        rows: usize, stream: &Stream<'_>) -> Result<()> {
        let mut owner = self.owner.borrow_mut();
        ensure!(owner.exchange.stream(0) == stream.raw && owner.exchange.device(0) == stream.device.id
            && [input, ids, weights, output].iter().all(|part| part.device_id == stream.device.id)
            && input.bytes >= rows * if owner.wire { 4752 } else { 4608 * 2 }
            && ids.bytes >= rows * 8 * 4 && weights.bytes >= rows * 8 * 4 && output.bytes >= rows * 4608 * 2,
            "Inkling TP2 routed operand differs from the issuing owner");
        // SAFETY: admitted private unit planes remain live through both peer
        // streams, and the shared executor serializes its retained scratch.
        unsafe { owner.enqueue_routed(0, self.layer, rows, input.ptr,
            crate::shared::experts::rtx::Routes { ids: ids.ptr, weights: weights.ptr }, output.ptr) }
    }
}

pub(crate) fn serving_dispatchers<'a>(stream: Rc<Stream<'a>>, peer: Option<Rc<Stream<'a>>>,
    catalog: &cuteafd_loader::OfficialV41Catalog, native: &Path,
    backings: &[cuteafd_loader::placement::families::inkling2::LocalBacking], rows: usize,
    pushed: Vec<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>, cfg: &Inkling2Config,
    admitted: &cuteafd_loader::placement::planner::ResolvedPlan)
    -> Result<Vec<Box<dyn Dispatcher + 'a>>> {
    use crate::shared::experts::rtx::{exl3::Exl3Tp2, fp8moe::Fp8MoeTp2, owner_tp2::OwnerTp2,
        push::PushedExperts, RtxExpertLayer, RtxSelectionPolicy};
    use crate::shared::peer_split::RankDevice;
    let gpus = 1 + usize::from(peer.is_some());
    let groups = super::weights::local_groups(backings, gpus)?;
    ensure!(pushed.len() == gpus && pushed.iter().all(|rank| rank.len() == groups.len()),
        "Inkling expert push rank/group dimensions differ");
    let mut pushed = pushed.into_iter().map(Vec::into_iter).collect::<Vec<_>>();
    let mut dispatchers = Vec::<(usize, Box<dyn Dispatcher + 'a>)>::new();
    for (group_index, group) in groups.into_iter().enumerate() {
        let layers = group[0].layers.clone();
        let lead = pushed[0].next().context("Inkling pushed lead expert set absent")?;
        if let Some(peer) = &peer {
            let peer_set = pushed[1].next().context("Inkling pushed peer expert set absent")?;
            let devices = [stream.device, peer.device];
            let workspace = |rank: usize| group[rank].workspace.iter().try_fold(0usize, |n, (_, bytes)|
                n.checked_add(usize::try_from(*bytes)?).context("Inkling TP2 expert workspace overflow"));
            let expected = [workspace(0)?, workspace(1)?];
            let budgets = [group[0].arena.bytes() + expected[0], group[1].arena.bytes() + expected[1]];
            let experts: [Box<dyn RtxExpertLayer + 'a>; 2] = match (lead, peer_set) {
                (PushedExperts::Fp8(left), PushedExperts::Fp8(right)) => {
                    let tensors = catalog.fp8().context("Inkling native pushed catalog missing")?.for_layer_metadata(layers[0])?;
                    let dir = crate::shared::experts::fp8::package_directory(native, 2, tensors.format());
                    Fp8MoeTp2::pushed_pair(devices, &tensors, &dir, layers.clone(), rows, budgets, [left, right])?
                        .map(|rank| Box::new(rank) as Box<dyn RtxExpertLayer>)
                }
                (PushedExperts::Exl3(left), PushedExperts::Exl3(right)) => {
                    let stem = catalog.exl3().context("Inkling EXL3 pushed catalog missing")?.package_name("inkling2");
                    let dir = native.parent().context("Inkling native directory missing")?
                        .join("exl3").join(stem).join("rtx-tp2");
                    let [left, right] = Exl3Tp2::pushed_pair_weights(devices, catalog, cuteafd_core::ExpertGeometry::INKLING2,
                        [&dir, &dir], layers.clone(), rows, budgets, RtxSelectionPolicy::CanonicalDecode, [left, right])?;
                    [Box::new(left.execution()?), Box::new(right.execution()?)]
                }
                _ => anyhow::bail!("Inkling TP2 pushed expert formats differ"),
            };
            ensure!(experts.iter().zip(expected).all(|(expert, bytes)| expert.workspace_bytes() == bytes),
                "Inkling TP2 native workspace differs from admission");
            let wire = experts[0].uses_fp8_input();
            let endpoints = [&stream, peer].map(|s| RankDevice { device: s.device.id, stream: s.raw });
            let owner = OwnerTp2::new(stream.device.library, endpoints, experts,
                cfg.hidden, cfg.topk, rows, 64, wire, "inkling2")?;
            let prefix = format!("inkling2.tp2.{}.", group_index * 2);
            let transport = owner.transport_buffers();
            let selected = admitted.facts.allocations.iter().filter(|a| a.id.starts_with(&prefix)
                || a.id.starts_with(&format!("inkling2.tp2.{}.", group_index * 2 + 1))).count();
            ensure!(transport.len() == selected, "Inkling TP2 expert exchange backing count differs from admission");
            for (rank, name, buffer) in transport {
                let id = format!("inkling2.tp2.{}.{name}", group_index * 2 + rank);
                let spec = admitted.facts.allocations.iter().find(|allocation| allocation.id == id)
                    .with_context(|| format!("Inkling TP2 expert exchange owner not admitted: {id}"))?;
                ensure!(spec.bytes == buffer.bytes as u64 && buffer.device_id == devices[rank].id
                    && spec.location == cuteafd_loader::placement::planner::Location::coordinator(rank as u8),
                    "Inkling TP2 expert exchange owner differs: {id}");
            }
            let owner = Rc::new(std::cell::RefCell::new(owner));
            for layer in layers { dispatchers.push((layer, Box::new(SplitResident { layer, owner: owner.clone(), _streams: [stream.clone(), peer.clone()] }))); }
        } else {
            let resident = Resident::from_pushed(stream.clone(), catalog, native, &group[0], rows, lead)?;
            dispatchers.extend(layers.into_iter().zip(resident.into_iter().map(|owner| Box::new(owner) as Box<dyn Dispatcher>)));
        }
    }
    dispatchers.sort_by_key(|(layer, _)| *layer);
    ensure!(dispatchers.iter().enumerate().all(|(index, (layer, _))| index == *layer),
        "Inkling resident layers are not the admitted ordered prefix");
    Ok(dispatchers.into_iter().map(|(_, owner)| owner).collect())
}

impl Dispatcher for Resident<'_> {
    fn wire_input(&self) -> bool { self.backend.borrow().uses_fp8_input() }
    fn enqueue(&self, input: Buffer, ids: Buffer, weights: Buffer, output: Buffer,
        rows: usize, stream: &Stream<'_>) -> Result<()> {
        use crate::shared::experts::rtx::{ExpertInput, PartialDtype, PartialLayout, Routes};
        ensure!(self.stream.raw == stream.raw && std::ptr::eq(self.stream.device.library, stream.device.library),
            "Inkling resident workspace changed issuing stream");
        let mut backend = self.backend.borrow_mut();
        ensure!(rows > 0 && backend.device() == stream.device.id
            && backend.partial_layout(rows)? == PartialLayout::Tokens && output.bytes >= rows*4608*2,
            "Inkling resident owner/output layout differs");
        let element = if backend.partial() == PartialDtype::F32 { 4 } else { 2 };
        let result_bytes = rows.checked_mul(4608*element).context("Inkling resident output extent overflow")?;
        ensure!(!backend.output().is_null() && backend.workspace_bytes() >= result_bytes,
            "Inkling resident workspace cannot hold submitted output rows");
        let wire = backend.uses_fp8_input();
        let input_bytes = rows * if wire { 4752 } else { 4608*2 };
        for (buffer, bytes) in [(input, input_bytes), (ids, rows*8*4), (weights, rows*8*4), (output, rows*4608*2)] {
            ensure!(buffer.device_id == stream.device.id && buffer.bytes >= bytes && !buffer.ptr.is_null()
                && buffer.flags == 0 && buffer.ptr as usize % 16 == 0, "Inkling resident native operand differs");
        }
        stream.device.run(|| {
            // SAFETY: unit retains package-typed input/routes and output; shared
            // backend retains its immutable arena and serial execution workspace.
            let input = if wire { ExpertInput::Fp8K32(input.ptr) } else { ExpertInput::Bf16(input.ptr) };
            unsafe { backend.enqueue(self.layer, rows, input,
                Routes { ids: ids.ptr, weights: weights.ptr }, stream.raw, &mut || {})?; }
            let element = if backend.partial() == PartialDtype::F32 { 4 } else { 2 };
            let source = Buffer { ptr: backend.output(), bytes: rows*4608*element, ..output };
            ensure!(source.ptr != output.ptr, "Inkling resident output aliases backend storage");
            // SAFETY: token sums and private output are disjoint, retained and
            // issued after the expert launch on the same ordered stream.
            unsafe {
                if element == 4 { stream.device.library.cuda_f32_to_bf16_async(source, output, rows*4608, stream.raw) }
                else { stream.device.library.copy_d2d_async(output, source, source.bytes, stream.raw) }
            }
        })
    }
}

impl Drop for Resident<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Inkling resident backend retirement"); }
}

#[cfg(test)]
mod resident_tests {
    use super::*;
    use crate::shared::experts::rtx::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use std::ffi::c_void;

    struct Backend<'a> { output: Allocation<'a>, wire: bool }
    impl RtxExpertLayer for Backend<'_> {
        fn shard(&self) -> RtxShard { RtxShard::Whole }
        fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
        fn uses_fp8_input(&self) -> bool { self.wire }
        fn layers(&self) -> &[usize] { &[0] }
        fn device(&self) -> i32 { self.output.device.id }
        fn workspace_bytes(&self) -> usize { self.output.buffer.bytes }
        fn output(&self) -> *mut c_void { self.output.buffer.ptr }
        unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput, _routes: Routes,
            _stream: *mut c_void, _abort: &mut dyn FnMut()) -> Result<()> {
            ensure!(layer == 0 && rows == 2 && matches!((self.wire, input),
                (true, ExpertInput::Fp8K32(_)) | (false, ExpertInput::Bf16(_))),
                "resident test lost native ingress");
            self.output.device.library.copy_h2d(self.output.buffer, &vec![0x4d; rows*4608*2])
        }
    }
    #[test]
    fn resident_native_backend_keeps_k32_ingress_and_copies_combined_token_sums() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        let device = Device { library: &library, id: 0 };
        let stream = Rc::new(Stream::new(device)?);
        let alloc = |bytes| Allocation::new(device, bytes, crate::shared::memory::TEST);
        let backend = Backend { output: alloc(2*4608*2)?, wire: true };
        let resident = Resident::new(0, Box::new(backend), stream.clone())?;
        let input = alloc(2*4752)?;
        let ids = alloc(2*8*4)?;
        let weights = alloc(2*8*4)?;
        let output = alloc(2*4608*2)?;
        resident.enqueue(input.buffer, ids.buffer, weights.buffer, output.buffer, 2, &stream)?;
        stream.drain()?;
        let mut observed = vec![0; output.buffer.bytes];
        library.copy_d2h(&mut observed, output.buffer)?;
        assert!(observed.iter().all(|&byte| byte == 0x4d));
        let other = Stream::new(device)?;
        assert!(resident.enqueue(input.buffer, ids.buffer, weights.buffer, output.buffer, 2, &other).is_err());
        let backend = Backend { output: alloc(2*4608*2)?, wire: false };
        let native = Resident::new(0, Box::new(backend), stream.clone())?;
        assert!(!native.wire_input());
        let bf16 = alloc(2*4608*2)?;
        native.enqueue(bf16.buffer, ids.buffer, weights.buffer, output.buffer, 2, &stream)?;
        assert!(native.enqueue(input.buffer, ids.buffer, weights.buffer, output.buffer, 2, &stream).is_err());
        for wire in [false, true] {
            assert!(Resident::new(0, Box::new(Backend { output: alloc(32)?, wire }), stream.clone()).is_err());
            let one = Resident::new(0, Box::new(Backend { output: alloc(4608*2)?, wire }), stream.clone())?;
            let source = if wire { input.buffer } else { bf16.buffer };
            assert!(one.enqueue(source, ids.buffer, weights.buffer, output.buffer, 2, &stream).is_err());
        }
        let handles = Resident::for_layers(Box::new(Backend { output: alloc(2*4608*2)?, wire: true }), stream.clone())?;
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].layer, 0);
        Ok(())
    }
}

pub(crate) struct Mxfp8Layout {
    pub arena: WeightArenaLayout,
    pub mount: MountLoad,
    lands: Vec<Vec<(Rect, usize)>>,
}
impl Mxfp8Layout {
    pub fn plan(checkpoint: &Checkpoint, cfg: &Inkling2Config, layer: usize) -> Result<Self> {
        ensure!(layer < cfg.layers && cfg.intermediate % 128 == 0 && cfg.hidden % 128 == 0,
            "Inkling MXFP8 layer geometry unsupported");
        let (e, h, i) = (cfg.experts, cfg.hidden, cfg.intermediate);
        let root = format!("model.llm.layers.{layer}.mlp.experts");
        let mut lands = Vec::new();
        let mut tensors = Vec::new();
        let mut mount = MountLoad::default();
        for (plane, projection, scale, part) in [(0, "w13", false, 0), (1, "w13", true, 0),
            (2, "w13", false, 1), (3, "w13", true, 1), (4, "w2", false, 0), (5, "w2", true, 0)] {
            let name = format!("{root}.{projection}_weight{}", if scale { "_scale" } else { "" });
            let header = checkpoint.header(&name).with_context(|| format!("Inkling expert requires {name}"))?;
            let dtype = if scale { DType::U8 } else { DType::F8E4M3 };
            let (rows, cols) = if projection == "w13" { (2*i, if scale { h/32 } else { h }) }
                else { (h, if scale { i/32 } else { i }) };
            ensure!(header.dtype == dtype && header.shape == [e, rows, cols]
                && header.byte_length == (e*rows*cols) as u64,
                "Inkling expert {name}: native MXFP8 header differs");
            let tensor = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name).context("Inkling tensor path")?;
            let (pitch, width, n) = if projection == "w13" { (2*cols, cols, i) } else { (cols, cols, h) };
            let bytes = e*n*width;
            tensors.push((plane.to_string(), bytes));
            let mut source = Vec::new();
            for expert in 0..e {
                source.push((Rect { name: name.clone(), path: checkpoint.snapshot.join(&tensor.shard),
                    base: header.byte_offset+(expert*rows*cols) as u64, row_bytes: pitch,
                    rows: 0..n, cols: part*cols..part*cols+width }, expert*n*width));
            }
            mount.merge(MountLoad::read(ReadBatch::new(n, pitch, width, STEP_BYTES)?));
            lands.push(source);
        }
        Ok(Self { arena: WeightArenaLayout::new("inkling2/golden/experts", tensors)?, mount, lands })
    }
}

pub(crate) struct Mxfp8<'a> {
    stream: Rc<Stream<'a>>,
    module: Fp8MoeModule,
    planes: Vec<Allocation<'a>>,
    scratch: Allocation<'a>,
    rows: usize,
}
impl<'a> Mxfp8<'a> {
    /// Metadata and all arena/workspace requests are admitted by the caller.
    pub fn load(stream: Rc<Stream<'a>>, layout: &Mxfp8Layout, metadata: &Fp8MoeMetadata,
        directory: &Path, cfg: &Inkling2Config, rows: usize) -> Result<Self> {
        let info = &metadata.info;
        ensure!(info.weights == Fp8MoeWeights::Mxfp8 && info.wire_input && info.tp == 1
            && (info.hidden, info.experts, info.topk, info.intermediate, info.slice)
                == (cfg.hidden, cfg.experts, cfg.topk, cfg.intermediate, cfg.intermediate),
            "Inkling native expert package differs from checkpoint");
        info.validate_activation(cuteafd_core::FfnActivation::Situ { beta: cfg.gate_softcap as f32,
            linear_beta: Some(cfg.up_softcap as f32) }, cuteafd_ffi::fp8_moe::Fp8MoeProjection::Bf16)?;
        let scratch_bytes = metadata.scratch_for(rows)?;
        let device: Device<'a> = stream.device;
        let arena = DeviceWeightArena::new(device, layout.arena.clone(), ArenaTag::experts("inkling2/experts"))?;
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(layout.mount)?;
        let mut planes = Vec::new();
        for (plane, lands) in layout.lands.iter().enumerate() {
            let view = layout.arena.view(&plane.to_string()).context("Inkling expert plane missing")?;
            let owner = arena.view(&plane.to_string(), view.bytes())?;
            dense.operand(Operand { device: device.id, buffer: owner.buffer, lands: lands.clone(), bytes: view.bytes() })?;
            planes.push(owner);
        }
        dense.release();
        let scratch = Allocation::new(device, scratch_bytes, cuteafd_core::memory_layout::Tag::workspace("inkling2/experts"))?;
        // SAFETY: trusted package, current device scoped by Device::run. Self
        // retains the package and drains its owning stream before unloading.
        let module = device.run(|| unsafe { Fp8MoeModule::load(directory) })?;
        ensure!(module.info() == info, "Inkling expert package changed since admission");
        Ok(Self { stream, module, planes, scratch, rows })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (index, owner) in self.planes.iter().enumerate() { owner.inventory(inventory, &format!("{prefix}.{index}"))?; }
        Ok(())
    }
    pub fn workspace_inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        self.scratch.inventory(inventory, "expert.scratch")
    }
}
impl Dispatcher for Mxfp8<'_> {
    fn enqueue(&self, input: Buffer, ids: Buffer, weights: Buffer, output: Buffer,
        rows: usize, stream: &Stream<'_>) -> Result<()> {
        ensure!(rows > 0 && rows <= self.rows && stream.raw == self.stream.raw
            && stream.device.id == self.stream.device.id, "Inkling expert stream/capacity differs");
        let info = self.module.info();
        for (buffer, bytes) in [(input, rows*info.wire_row_bytes()), (ids, rows*info.topk*4),
            (weights, rows*info.topk*4), (output, rows*info.hidden*2)] {
            ensure!(buffer.device_id == stream.device.id && buffer.bytes >= bytes && !buffer.ptr.is_null()
                && buffer.ptr as usize % 16 == 0 && buffer.flags == 0, "Inkling expert operand extent/device differs");
        }
        let pointers = [input.ptr, ids.ptr, weights.ptr, self.planes[0].buffer.ptr,
            self.planes[1].buffer.ptr, self.planes[2].buffer.ptr, self.planes[3].buffer.ptr,
            self.planes[4].buffer.ptr, self.planes[5].buffer.ptr, output.ptr, self.scratch.buffer.ptr];
        // SAFETY: admitted native planes, checked I/O extents and retained
        // stream/workspace/module owners. Caller orders route/input producers.
        unsafe { self.module.launch(&pointers, rows, stream.raw) }
    }
}
impl Drop for Mxfp8<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Inkling native expert module and weight arena"); }
}
