//! Native MSA producer -> cache -> selection -> paged attention chain.
use crate::shared::memory::{device::{Allocation, Device, Stream}, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, index_cache,
    minimax_m3::{self, producers, forward, Mode, Msa}, programs::Programs};
use cuteafd_loader::families::minimax_m3::{MiniMaxM3Config,
    execution::{MsaPrograms, ProducerPrograms, ForwardPrograms}};
use std::collections::BTreeMap;

/// Handles only: admission resolves every manifest before any module loads.
pub(crate) struct Kernels<'a> {
    device: Device<'a>,
    query: producers::Producer<'a>,
    key: producers::Producer<'a>,
    index: producers::Producer<'a>,
    kv: forward::ForwardProducer<'a>,
    gather: index_cache::Gather<'a>,
    select: Msa<'a>,
    attention: Msa<'a>,
    decode_select: Msa<'a>,
    decode_attention: Msa<'a>,
    msa_layout: MsaPrograms,
    producer_layout: ProducerPrograms,
}

impl<'a> Kernels<'a> {
    pub fn bind(device: Device<'a>, programs: &'a Programs<'a>, manifest: &serde_json::Value,
        cfg: &MiniMaxM3Config, rows: u32, context: u32) -> Result<Self> {
        ensure!(std::ptr::eq(device.library, programs.library()), "MiniMax attention library differs");
        let (msa_layout, msa) = MsaPrograms::resolve(cfg, manifest, rows.min(64), rows, context)?;
        let (producer_layout, producers) = ProducerPrograms::resolve(manifest, rows)?;
        let (_, forward) = ForwardPrograms::resolve(manifest, rows)?;
        for expected in msa.iter().chain(&producers).chain(&forward) {
            let actual = programs.spec(&expected.name)?;
            ensure!(actual.pointers == expected.pointers && actual.scalar_kinds == expected.scalar_kinds
                && actual.scratch == expected.scratch && actual.capacity_rows == expected.capacity_rows
                && actual.geometry == expected.geometry,
                "MiniMax attention program {} differs from admission", expected.name);
        }
        device.run(|| Ok(Self {
            device,
            query: producers::Producer::bind(programs, producers::Kind::NormRope { heads: 64 })?,
            key: producers::Producer::bind(programs, producers::Kind::NormRope { heads: 4 })?,
            index: producers::Producer::bind(programs, producers::Kind::Index)?,
            kv: forward::ForwardProducer::bind(programs, forward::Kind::KvPack)?,
            gather: index_cache::Gather::bind(programs, "minimax_index_cache_gather")?,
            select: Msa::bind(programs, Mode::IndexPrefill)?,
            attention: Msa::bind(programs, Mode::AttentionExtend)?,
            decode_select: Msa::bind(programs, Mode::IndexDecode)?,
            decode_attention: Msa::bind(programs, Mode::AttentionDecode)?,
            msa_layout, producer_layout,
        }))
    }
}

/// One independently live unit's scratch. No weights or cache pages live here.
/// The stream owner must drain before these allocations retire.
pub(crate) struct Workspace<'a> {
    device: Device<'a>,
    planes: BTreeMap<String, Allocation<'a>>,
    prefill: bool,
}
impl<'a> Workspace<'a> {
    pub fn new(kernels: &Kernels<'a>) -> Result<Self> { Self::for_mode(kernels, true) }
    pub fn new_decode(kernels: &Kernels<'a>) -> Result<Self> { Self::for_mode(kernels, false) }
    fn for_mode(kernels: &Kernels<'a>, prefill: bool) -> Result<Self> {
        let mut planes = BTreeMap::new();
        for (name, bytes) in kernels.msa_layout.workspace(prefill).into_iter()
            .chain(kernels.producer_layout.workspace()) {
            ensure!(!planes.contains_key(name), "duplicate MiniMax attention scratch owner");
            planes.insert(name.into(), Allocation::new(kernels.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/attention"))?);
        }
        Ok(Self { device: kernels.device, planes, prefill })
    }
    #[cfg(test)]
    pub(super) fn diagnostic_plane(&self, name: &str) -> Result<Buffer> { self.plane(name) }
    fn plane(&self, name: &str) -> Result<Buffer> {
        self.planes.get(name).map(|owner| owner.buffer)
            .with_context(|| format!("MiniMax attention lacks admitted plane {name}"))
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.planes { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}

/// Projected BF16 rows remain live through the chain. Norms are arena views.
#[derive(Clone, Copy)]
pub(crate) struct Projections {
    pub query: Buffer,
    pub key: Buffer,
    pub value: Buffer,
    pub index_query: Buffer,
    pub index_key: Buffer,
    pub query_norm: Buffer,
    pub key_norm: Buffer,
    pub index_query_norm: Buffer,
    pub index_key_norm: Buffer,
}

/// Metadata is produced by the shared cache owner, not inferred from a slot.
/// This entrypoint stages one request, including keys from earlier units.
#[derive(Clone, Copy)]
pub(crate) struct Request {
    pub rows: usize,
    pub context: usize,
    pub positions_capacity: usize,
    pub table_width: usize,
    pub table_stride: usize,
    pub pool_pages: usize,
    pub positions: Buffer,
    pub slots: Buffer,
    pub batch_ids: Buffer,
    pub cos_sin: Buffer,
    pub pages: Buffer,
    pub causal_lengths: Buffer,
    pub cache_lengths: Buffer,
    pub query_offsets: Buffer,
    pub key_descale: Buffer,
    pub value_descale: Buffer,
    pub key_cache: Buffer,
    pub value_cache: Buffer,
    pub index_cache: Buffer,
}

/// One sequence's attention inside an otherwise batched rowwise forward.
#[derive(Clone, Copy)]
pub(crate) struct Segment {
    pub first_row: usize,
    pub request: Request,
}

/// Native decode has exactly one row per sequence; multi-token verification
/// uses prefill segments so each query retains its own causal frontier.
#[derive(Clone, Copy)]
pub(crate) struct DecodeRequest {
    pub request: Request,
    pub batch: usize,
}

fn decode_page_bound(request: Request, exported_pages: u32) -> Result<usize> {
    let pages = request.positions_capacity.div_ceil(64);
    ensure!(pages > 0 && pages <= exported_pages as usize
        && pages <= request.table_width && request.table_width <= request.table_stride,
        "MiniMax decode capacity exceeds admitted index pages/table");
    Ok(pages)
}

pub(crate) fn row_view(buffer: Buffer, first: usize, rows: usize, stride: usize) -> Result<Buffer> {
    let offset = first.checked_mul(stride).context("MiniMax row offset overflow")?;
    let bytes = rows.checked_mul(stride).context("MiniMax row extent overflow")?;
    ensure!(rows != 0 && offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "MiniMax row view exceeds retained owner");
    Ok(Buffer { ptr: buffer.ptr.wrapping_byte_add(offset), bytes, ..buffer })
}

impl Workspace<'_> {
    fn decode_batch(&self, r: Request, page_bound: usize) -> Result<minimax_m3::IndexDecodeBatch> {
        Ok(minimax_m3::IndexDecodeBatch {
            rows: r.rows, table_width: page_bound, table_stride: r.table_stride,
            pool_pages: r.pool_pages, query: self.plane("index.query")?, scales: self.plane("index.scales")?,
            cache: r.index_cache, pages: r.pages, lengths: r.cache_lengths,
            selections: self.plane("selections")?, scratch: self.plane("index.scratch")?,
        })
    }

    fn attention_batch(&self, r: Request, batch: usize) -> Result<minimax_m3::AttentionBatch> {
        Ok(minimax_m3::AttentionBatch {
            rows: r.rows, batch, table_stride: r.table_stride, pool_pages: r.pool_pages,
            query: self.plane("attention.query")?, keys: r.key_cache, values: r.value_cache,
            key_scales: r.key_descale, value_scales: r.value_descale,
            pages: r.pages, lengths: r.cache_lengths, query_offsets: r.query_offsets,
            selections: self.plane("selections")?, output: self.plane("attention.output")?,
            lse: self.plane("attention.lse")?, scratch: self.plane("attention.scratch")?,
        })
    }

    /// # Safety
    /// Positions/slots/page IDs, query offsets and lengths must describe one
    /// nonempty causal request. Newly appended slots are unique, belong to this
    /// layer and address the physical pool. Earlier rows are initialized. All
    /// operands, arena views and this workspace survive the stream's drain.
    pub unsafe fn enqueue(&self, kernels: &Kernels<'_>, p: Projections, r: Request,
        stream: &Stream<'_>) -> Result<Buffer> {
        ensure!(self.prefill, "MiniMax prefill requires a prefill workspace");
        // SAFETY: the caller supplies the documented retained request operands.
        unsafe { self.enqueue_impl(kernels, p, r, 1, false, stream) }
    }

    /// # Safety
    /// As `enqueue`, with one query per sequence, distinct append slots and
    /// batch-sized tables, lengths, descales and arange query offsets.
    pub unsafe fn decode(&self, kernels: &Kernels<'_>, p: Projections, decode: DecodeRequest,
        stream: &Stream<'_>) -> Result<Buffer> {
        ensure!(!self.prefill && decode.batch == decode.request.rows,
            "MiniMax native decode requires one row per request and a decode workspace");
        // SAFETY: caller retains the batch metadata and all cache records.
        unsafe { self.enqueue_impl(kernels, p, decode.request, decode.batch, true, stream) }
    }

    unsafe fn enqueue_impl(&self, kernels: &Kernels<'_>, p: Projections, r: Request,
        batch: usize, decode: bool, stream: &Stream<'_>) -> Result<Buffer> {
        ensure!(self.device.id == kernels.device.id && self.device.id == stream.device.id
            && std::ptr::eq(self.device.library, kernels.device.library)
            && std::ptr::eq(self.device.library, stream.device.library),
            "MiniMax attention owner device/library differs");
        let attention = if decode { &kernels.decode_attention } else { &kernels.attention };
        ensure!(r.rows > 0 && r.rows <= attention.layout.rows as usize && batch > 0
            && batch <= attention.layout.extent as usize && (decode || r.rows <= r.context)
            && r.context > 0 && r.context <= r.positions_capacity && r.context.div_ceil(64) <= r.table_width
            && r.table_width <= r.table_stride,
            "MiniMax attention request extents");
        // Decode graphs keep this launch bound fixed; the native scheduler and
        // scorer read cache_lengths on device to visit only this step's pages.
        let decode_pages = if decode { decode_page_bound(r, kernels.decode_select.layout.extent)? } else { 0 };
        // Prefill still uses live host bounds and must remain an eager graph site.
        let padded = if decode { 0 } else {
            r.context.checked_add(255).context("MiniMax staging overflow")? / 256 * 256
        };
        let query = self.plane("attention.query")?;
        let key = self.plane("attention.key")?;
        let index_query = self.plane("index.query")?;
        let index_scales = self.plane("index.scales")?;
        let selections = self.plane("selections")?;
        let output = self.plane("attention.output")?;
        let mut drain = stream.cancellation_guard("MiniMax attention submission error");
        let result = self.device.run(|| {
            // SAFETY: caller retains the request's metadata/cache and projected
            // rows; each launch checks extents/aliases and executes on one stream.
            unsafe {
                kernels.query.norm_rope(producers::NormRopeBatch {
                    rows: r.rows, positions_capacity: r.positions_capacity, input: p.query,
                    weight: p.query_norm, positions: r.positions, cos_sin: r.cos_sin,
                    output: query, scratch: self.plane("minimax_norm_rope_h64")?,
                }, stream.raw)?;
                kernels.key.norm_rope(producers::NormRopeBatch {
                    rows: r.rows, positions_capacity: r.positions_capacity, input: p.key,
                    weight: p.key_norm, positions: r.positions, cos_sin: r.cos_sin,
                    output: key, scratch: self.plane("minimax_norm_rope_h4")?,
                }, stream.raw)?;
                kernels.kv.kv(forward::KvBatch {
                    rows: r.rows, batch, pool_pages: r.pool_pages,
                    key, value: p.value, slots: r.slots, batch_ids: r.batch_ids,
                    key_descale: r.key_descale, value_descale: r.value_descale,
                    key_cache: r.key_cache, value_cache: r.value_cache,
                }, stream.raw)?;
                kernels.index.index(producers::IndexProducerBatch {
                    rows: r.rows, positions_capacity: r.positions_capacity, pool_pages: r.pool_pages,
                    query: p.index_query, key: p.index_key, query_norm: p.index_query_norm,
                    key_norm: p.index_key_norm, positions: r.positions, slots: r.slots,
                    cos_sin: r.cos_sin, query_fp8: index_query, query_scale: index_scales,
                    key_fp8: self.plane("index.produced_keys")?,
                    key_scale: self.plane("index.produced_scales")?, cache: r.index_cache,
                    scratch: self.plane("minimax_index_producer")?,
                }, stream.raw)?;
                if decode {
                    kernels.decode_select.index_decode(self.decode_batch(r, decode_pages)?, stream.raw)?;
                } else {
                    let keys = self.plane("index.keys")?;
                    let scales = self.plane("index.key_scales")?;
                    let starts = self.plane("index.starts")?;
                    let ends = self.plane("index.ends")?;
                    kernels.gather.launch(index_cache::Batch {
                        rows: r.rows, table_width: r.table_width, page_begin: 0,
                        pool_pages: r.pool_pages, live_tokens: r.context, output_tokens: padded,
                        cache: r.index_cache, pages: r.pages, lengths: r.causal_lengths,
                        keys, scales, starts, ends,
                    }, stream.raw)?;
                    kernels.select.index_prefill(minimax_m3::IndexPrefillBatch {
                        rows: r.rows, k_rows: r.context, query: index_query, scales: index_scales,
                        keys, key_scales: scales, starts, ends, selections,
                        scratch: self.plane("index.scratch")?,
                    }, stream.raw)?;
                }
                attention.attention(self.attention_batch(r, batch)?, stream.raw)?;
            }
            Ok(output)
        });
        if result.is_ok() { drain.complete = true; }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::Category;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};

    fn manifest() -> serde_json::Value {
        let mut entries = Vec::new();
        for source in [
            include_str!("../../../../cuteafd-ffi/tests/fixtures/minimax_msa_programs.json"),
            include_str!("../../../../cuteafd-ffi/tests/fixtures/minimax_producer_programs.json"),
            include_str!("../../../../cuteafd-ffi/tests/fixtures/minimax_forward_programs.json"),
        ] {
            let value: serde_json::Value = serde_json::from_str(source).unwrap();
            entries.extend(value["programs"].as_array().unwrap().clone());
        }
        serde_json::json!({"programs":entries})
    }
    fn config() -> MiniMaxM3Config {
        MiniMaxM3Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json")).unwrap()).unwrap()
    }

    #[test]
    fn handles_bind_without_allocations_and_refuse_admission_drift() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let manifest = manifest();
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let path = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let kernels = Kernels::bind(device, &programs, &manifest, &config(), 17, 2049)?;
        assert_eq!(kernels.select.layout.rows, 4096);
        assert_eq!(kernels.attention.layout.rows, 4096);
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?.matches('D').count(), 0);
        for i in 0..manifest["programs"].as_array().unwrap().len() {
            let mut missing = manifest.clone();
            missing["programs"].as_array_mut().unwrap().remove(i);
            assert!(Kernels::bind(device, &programs, &missing, &config(), 17, 2049).is_err());
        }
        let mut drift = manifest.clone();
        drift["programs"][0]["geometry"]["admission_marker"] = 1.into();
        assert!(Kernels::bind(device, &programs, &drift, &config(), 17, 2049).is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 0);
        Ok(())
    }

    #[test]
    fn row_views_preserve_device_and_slice_only_live_rows() {
        let buffer = Buffer { ptr: 4096usize as *mut std::ffi::c_void, bytes: 17 * 1024,
            device_id: 1, ..Default::default() };
        for first in [0, 1, 3, 16] {
            let view = row_view(buffer, first, 1, 1024).unwrap();
            assert_eq!(view.ptr as usize, 4096 + first * 1024);
            assert_eq!(view.bytes, 1024);
            assert_eq!(view.device_id, 1);
            assert_eq!(view.flags, buffer.flags);
        }
        for (first, rows, stride) in [(17, 1, 1024), (0, 18, 1024), (0, 0, 1024),
            (usize::MAX, 1, 1024), (0, usize::MAX, 1024)] {
            assert!(row_view(buffer, first, rows, stride).is_err());
        }
    }

    #[test]
    fn decode_page_bound_ignores_context_and_excludes_table_padding() -> Result<()> {
        let buffer = Buffer::default();
        let mut request = Request { rows: 1, context: 1, positions_capacity: 4097,
            table_width: 68, table_stride: 68, pool_pages: 65,
            positions: buffer, slots: buffer, batch_ids: buffer, cos_sin: buffer,
            pages: buffer, causal_lengths: buffer, cache_lengths: buffer,
            query_offsets: buffer, key_descale: buffer, value_descale: buffer,
            key_cache: buffer, value_cache: buffer, index_cache: buffer };
        for context in [1, 63, 64, 65, 4095, 4096, 4097] {
            request.context = context;
            assert_eq!(decode_page_bound(request, 65)?, 65);
        }
        assert!(decode_page_bound(request, 64).is_err());
        request.table_width = 64;
        assert!(decode_page_bound(request, 65).is_err());
        request.table_width = 68;
        request.table_stride = 67;
        assert!(decode_page_bound(request, 65).is_err());
        request.positions_capacity = 0;
        assert!(decode_page_bound(request, 65).is_err());
        Ok(())
    }

    fn native_read(device: Device<'_>, buffer: Buffer, bytes: usize) -> Result<Vec<u8>> {
        let mut out = vec![0; bytes];
        device.run(|| device.library.copy_d2h(&mut out, buffer))?;
        Ok(out)
    }

    fn native_capture<'a>(stream: &Stream<'a>, launch: impl FnOnce() -> Result<()>)
        -> Result<crate::shared::decode_graph::GraphOwner<'a, ()>> {
        let device = stream.device;
        stream.drain()?;
        // SAFETY: the test retains every captured allocation and drains each replay before retirement.
        device.run(|| unsafe { device.library.cuda_graph_begin_capture(stream.raw) })?;
        let launched = launch();
        // End even a failed capture before returning, so the stream never remains in capture mode.
        // SAFETY: capture began on this owned stream, with retained compiled programs and operands.
        let raw = device.run(|| unsafe { device.library.cuda_graph_end_capture_for_test(stream.raw) })?;
        // SAFETY: all external storage owners outlive the executable and every drained replay.
        let graph = unsafe { crate::shared::decode_graph::GraphOwner::new(device.library, device.id, raw, ())? };
        launched?;
        Ok(graph)
    }

    fn native_seed(device: Device<'_>, request: Request, tokens: usize) -> Result<()> {
        let pages = tokens.div_ceil(64);
        for (buffer, stride) in [(request.key_cache, 32768), (request.value_cache, 32768),
            (request.index_cache, 8448)] {
            let mut bytes = vec![0; pages * stride];
            for (page, record) in bytes.chunks_exact_mut(stride).enumerate() {
                let key_bytes = if stride == 8448 { 8192 } else { stride };
                for (at, value) in record[..key_bytes].iter_mut().enumerate() {
                    *value = [0x28, 0xb0, 0x38, 0xa8][(page * 7 + at * 3 + at / 128) % 4];
                }
                if stride == 8448 {
                    for value in record[key_bytes..].chunks_exact_mut(4) { value.copy_from_slice(&1f32.to_le_bytes()); }
                }
            }
            device.run(|| {
                device.library.cuda_zero_bytes(buffer, buffer.bytes)?;
                device.library.copy_h2d(buffer, &bytes)
            })?;
        }
        Ok(())
    }

    fn native_projections(device: Device<'_>) -> Result<(Vec<Allocation<'_>>, Projections)> {
        let shapes = [8192, 512, 512, 512, 128, 128, 128, 128, 128];
        let owners = shapes.into_iter().enumerate().map(|(plane, elements)| {
            let owner = Allocation::new(device, elements * 2, Tag::workspace("minimax/replay-oracle"))?;
            let bytes = (0..elements).flat_map(|column| {
                let value = if plane >= 5 { 0.0 } else { ((column * 17 + plane * 13) % 251) as f32 / 128.0 - 1.0 };
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect::<Vec<_>>();
            device.run(|| device.library.copy_h2d(owner.buffer, &bytes))?;
            Ok(owner)
        }).collect::<Result<Vec<_>>>()?;
        let p = Projections { query: owners[0].buffer, key: owners[1].buffer, value: owners[2].buffer,
            index_query: owners[3].buffer, index_key: owners[4].buffer, query_norm: owners[5].buffer,
            key_norm: owners[6].buffer, index_query_norm: owners[7].buffer, index_key_norm: owners[8].buffer };
        Ok((owners, p))
    }

    #[test]
    #[ignore = "requires native SM120 exports through golden-run --native-test"]
    fn native_decode_graph_replays_live_metadata_and_bounds_cost() -> Result<()> {
        use super::super::cache::Cache;
        use crate::shared::memory::device::Event;
        use std::{path::Path, rc::Rc};
        // SAFETY: golden-run supplies a sealed native library; every owner retires before it.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = Device { library: &library, id: std::env::var("CUTEAFD_CACHE_TEST_DEVICE")
            .unwrap_or_else(|_| "0".into()).parse()? };
        let path = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?;
        let manifest = serde_json::from_slice(&std::fs::read(&path)?)?;
        let programs = library.programs()?.with_manifest(Path::new(&path))?;
        let mut cfg = config();
        cfg.layers = 1;
        let kernels = Kernels::bind(device, &programs, &manifest, &cfg, 1, 1048576)?;
        let stream = Rc::new(Stream::new(device)?);
        let (_operands, p) = native_projections(device)?;
        let decode = |workspace: &Workspace<'_>, request| {
            // SAFETY: the cold cache and metadata retain unique initialized pages on this stream.
            unsafe { workspace.decode(&kernels, p, DecodeRequest { request, batch: 1 }, &stream).map(|_| ()) }
        };
        {
            let eager = Workspace::new_decode(&kernels)?;
            let replay = Workspace::new_decode(&kernels)?;
            let a = Cache::new(Rc::clone(&stream), &cfg, 8193)?;
            let b = Cache::new(Rc::clone(&stream), &cfg, 8193)?;
            let mut ma = a.metadata(1)?;
            let mut mb = b.metadata(1)?;
            a.prepare(&mut ma, 62..63)?;
            b.prepare(&mut mb, 62..63)?;
            native_seed(device, a.request(&ma, 0)?, 8193)?;
            native_seed(device, b.request(&mb, 0)?, 8193)?;
            decode(&replay, b.request(&mb, 0)?)?;
            stream.drain()?;
            let graph = native_capture(&stream, || decode(&replay, b.request(&mb, 0)?))?;
            // Undo warm-up's append so both caches begin from identical state.
            native_seed(device, b.request(&mb, 0)?, 8193)?;
            for context in [63, 64, 65, 4095, 4096, 4097, 8191, 8192, 8193] {
                a.prepare(&mut ma, context - 1..context)?;
                b.prepare(&mut mb, context - 1..context)?;
                decode(&eager, a.request(&ma, 0)?)?;
                // SAFETY: all captured pointers are unchanged; only retained device metadata changes.
                let result = device.run(|| unsafe { library.cuda_graph_launch(graph.raw, stream.raw) });
                stream.drain_or_abort("MiniMax live-metadata graph oracle");
                result?;
                let actual = native_read(device, replay.plane("attention.output")?, 16384)?;
                assert_eq!(actual, native_read(device, eager.plane("attention.output")?, 16384)?,
                    "context={context}: graph output bytes");
                let values = actual.chunks_exact(2).map(|v| f32::from_bits(
                    u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16)).collect::<Vec<_>>();
                ensure!(values.iter().all(|v| v.is_finite()) && values.iter().any(|v| *v != 0.0),
                    "context={context}: nonfinite or empty attention output");
                for head in 0..4 {
                    let first = head * kernels.decode_select.layout.rows as usize;
                    assert_eq!(native_read(device, row_view(replay.plane("selections")?, first, 1, 64)?, 64)?,
                        native_read(device, row_view(eager.plane("selections")?, first, 1, 64)?, 64)?,
                        "context={context} head={head}: graph selection bytes");
                }
                for (x, y) in [(a.request(&ma, 0)?.key_cache, b.request(&mb, 0)?.key_cache),
                    (a.request(&ma, 0)?.value_cache, b.request(&mb, 0)?.value_cache),
                    (a.request(&ma, 0)?.index_cache, b.request(&mb, 0)?.index_cache)] {
                    assert_eq!(native_read(device, x, x.bytes)?, native_read(device, y, y.bytes)?,
                        "context={context}: graph changed persistent cache bytes");
                }
                eprintln!("MiniMax graph context={context}: output, four selections and complete K/V/index caches byte-exact");
            }
        }
        // Ceiling hypothesis: only the native device-live scheduler sees the wider bound;
        // clear/selector grids already cover compiled capacity. Stop outside a +/-3% delta.
        let workspace = Workspace::new_decode(&kernels)?;
        let cache = Cache::new(Rc::clone(&stream), &cfg, 1048576)?;
        let mut metadata = cache.metadata(1)?;
        cache.prepare(&mut metadata, 8191..8192)?;
        let request = cache.request(&metadata, 0)?;
        native_seed(device, request, 8192)?;
        decode(&workspace, request)?;
        stream.drain()?;
        let launch = |pages| device.run(|| {
            // SAFETY: index/attention arguments refer to retained, initialized C1 operands and scratch.
            unsafe {
                kernels.decode_select.index_decode(workspace.decode_batch(request, pages)?, stream.raw)?;
                kernels.decode_attention.attention(workspace.attention_batch(request, 1)?, stream.raw)?;
            }
            Ok(())
        });
        launch(128)?;
        stream.drain()?;
        let expected = native_read(device, workspace.plane("attention.output")?, 16384)?;
        launch(16384)?;
        stream.drain()?;
        assert_eq!(native_read(device, workspace.plane("attention.output")?, 16384)?, expected,
            "8K-live/1M-capacity: fixed bound changes output");
        let exact = native_capture(&stream, || launch(128))?;
        let fixed = native_capture(&stream, || launch(16384))?;
        crate::shared::thermal::monitor("minimax-replay-test");
        let mut elapsed = Vec::new();
        for (label, graph) in [("exact", &exact), ("fixed", &fixed)] {
            // One untimed warm batch per arm, excluding capture and first-use work.
            for _ in 0..32 {
                // SAFETY: this scope retains both executables and all captured operands.
                device.run(|| unsafe { library.cuda_graph_launch(graph.raw, stream.raw) })?;
            }
            stream.drain()?;
            let from = cuteafd_core::thermal::now_s();
            let mut start = Event::timing(device)?;
            let mut end = Event::timing(device)?;
            start.record(&stream)?;
            for _ in 0..8192 {
                // SAFETY: serial replays share retained scratch on this owned stream.
                device.run(|| unsafe { library.cuda_graph_launch(graph.raw, stream.raw) })?;
            }
            end.record(&stream)?;
            stream.drain_or_abort("MiniMax admitted-bound timing");
            // SAFETY: both timing events completed on the same retained device/stream.
            let ms = device.run(|| unsafe { library.cuda_event_elapsed_ms(start.raw, end.raw) })?;
            let window = cuteafd_core::thermal::window(from, cuteafd_core::thermal::now_s());
            eprintln!("MiniMax bound={label} C1 live=8192 capacity=1048576 us={}; thermal={window}", ms * 1000.0 / 8192.0);
            elapsed.push(ms);
        }
        let ratio = elapsed[1] / elapsed[0];
        eprintln!("MiniMax fixed/exact admitted-bound latency ratio={ratio}");
        ensure!(ratio.is_finite() && (0.97..=1.03).contains(&ratio),
            "MiniMax admitted-bound cost exceeds the declared +/-3% stop bar: ratio={ratio}");
        Ok(())
    }

    #[test]
    fn decode_owner_uses_only_admitted_decode_scratch() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.extent_only(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 8usize << 30)?;
        let device = Device { library: &library, id: 1 };
        let manifest = manifest();
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let path = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let kernels = Kernels::bind(device, &programs, &manifest, &config(), 64, 2049)?;
        let workspace = Workspace::new_decode(&kernels)?;
        assert!(!workspace.prefill);
        assert!(workspace.plane("index.keys").is_err());
        assert!(workspace.plane("index.starts").is_err());
        assert_eq!(workspace.plane("selections")?.bytes, 64 * 256);
        assert_eq!(workspace.plane("attention.query")?.bytes, 64 * 16384);
        let ids = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&ids)?, "minimax", Lifetime::TARGET,
            Category::Workspace, "MiniMax native decode MSA");
        workspace.inventory(&mut inventory, "decode.1")?;
        let expected = kernels.msa_layout.workspace(false).into_iter()
            .chain(kernels.producer_layout.workspace()).collect::<BTreeMap<_, _>>();
        assert_eq!(inventory.receipts.len(), expected.len());
        for (name, bytes) in expected { assert_eq!(workspace.plane(name)?.bytes as u64, bytes); }
        drop(workspace);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 8usize << 30);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn unit_scratch_matches_each_admitted_request_and_retires_on_its_device() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.extent_only(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 8usize << 30)?;
        let device = Device { library: &library, id: 1 };
        let manifest = manifest();
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let path = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let kernels = Kernels::bind(device, &programs, &manifest, &config(), 17, 2049)?;
        let workspace = Workspace::new(&kernels)?;
        let other = Workspace::new(&kernels)?;
        for name in workspace.planes.keys() {
            assert_ne!(workspace.plane(name)?.ptr, other.plane(name)?.ptr);
        }
        let ids = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&ids)?, "minimax", Lifetime::TARGET,
            Category::Workspace, "MiniMax native MSA");
        workspace.inventory(&mut inventory, "prefill.0")?;
        let expected = kernels.msa_layout.workspace_components("minimax.prefill.0", 0, true,
            Lifetime::TARGET, 1)?;
        let producer = kernels.producer_layout.workspace_components("minimax.prefill.0", 0,
            Lifetime::TARGET, 1)?;
        use cuteafd_loader::placement::planner::ComponentKind;
        let mut expected = expected.into_iter().chain(producer).filter_map(|component| match component.kind {
            ComponentKind::Allocation { template, .. } => Some(template), _ => None,
        }).collect::<Vec<_>>();
        expected.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(inventory.receipts.len(), expected.len());
        for (actual, expected) in inventory.receipts.iter().zip(expected) {
            assert_eq!(actual.allocation, expected);
        }
        assert_eq!(workspace.plane("attention.key")?.bytes, 4096 * 1024);
        assert_eq!(workspace.plane("index.produced_keys")?.bytes, 4096 * 128);
        assert_eq!(workspace.plane("index.produced_scales")?.bytes, 4096 * 4);
        drop(other);
        drop(workspace);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 8usize << 30);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        assert_eq!(library.cuda_get_device()?, 0);
        Ok(())
    }
}
