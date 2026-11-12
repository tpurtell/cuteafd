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
    msa_layout: MsaPrograms,
    producer_layout: ProducerPrograms,
}

impl<'a> Kernels<'a> {
    pub fn bind(device: Device<'a>, programs: &'a Programs<'a>, manifest: &serde_json::Value,
        cfg: &MiniMaxM3Config, rows: u32, context: u32) -> Result<Self> {
        ensure!(std::ptr::eq(device.library, programs.library()), "MiniMax attention library differs");
        let (msa_layout, msa) = MsaPrograms::resolve(cfg, manifest, 1, rows, context)?;
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
            msa_layout, producer_layout,
        }))
    }
}

/// One independently live unit's scratch. No weights or cache pages live here.
/// The stream owner must drain before these allocations retire.
pub(crate) struct Workspace<'a> {
    device: Device<'a>,
    planes: BTreeMap<String, Allocation<'a>>,
}
impl<'a> Workspace<'a> {
    pub fn new(kernels: &Kernels<'a>) -> Result<Self> {
        let mut planes = BTreeMap::new();
        for (name, bytes) in kernels.msa_layout.workspace(true).into_iter()
            .chain(kernels.producer_layout.workspace()) {
            ensure!(!planes.contains_key(name), "duplicate MiniMax attention scratch owner");
            planes.insert(name.into(), Allocation::new(kernels.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/attention"))?);
        }
        Ok(Self { device: kernels.device, planes })
    }
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

impl Workspace<'_> {
    /// # Safety
    /// Positions/slots/page IDs, query offsets and lengths must describe one
    /// nonempty causal request. Newly appended slots are unique, belong to this
    /// layer and address the physical pool. Earlier rows are initialized. All
    /// operands, arena views and this workspace survive the stream's drain.
    pub unsafe fn enqueue(&self, kernels: &Kernels<'_>, p: Projections, r: Request,
        stream: &Stream<'_>) -> Result<Buffer> {
        ensure!(self.device.id == kernels.device.id && self.device.id == stream.device.id
            && std::ptr::eq(self.device.library, kernels.device.library)
            && std::ptr::eq(self.device.library, stream.device.library),
            "MiniMax attention owner device/library differs");
        ensure!(r.rows > 0 && r.rows <= r.context && r.context <= r.positions_capacity
            && r.context.div_ceil(64) <= r.table_width && r.table_width <= r.table_stride,
            "MiniMax attention request extents");
        let padded = r.context.checked_add(255).context("MiniMax staging overflow")? / 256 * 256;
        let query = self.plane("attention.query")?;
        let key = self.plane("attention.key")?;
        let index_query = self.plane("index.query")?;
        let index_scales = self.plane("index.scales")?;
        let keys = self.plane("index.keys")?;
        let scales = self.plane("index.key_scales")?;
        let starts = self.plane("index.starts")?;
        let ends = self.plane("index.ends")?;
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
                    rows: r.rows, batch: 1, pool_pages: r.pool_pages,
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
                kernels.attention.attention(minimax_m3::AttentionBatch {
                    rows: r.rows, batch: 1, table_stride: r.table_stride, pool_pages: r.pool_pages,
                    query, keys: r.key_cache, values: r.value_cache,
                    key_scales: r.key_descale, value_scales: r.value_descale,
                    pages: r.pages, lengths: r.cache_lengths, query_offsets: r.query_offsets,
                    selections, output, lse: self.plane("attention.lse")?,
                    scratch: self.plane("attention.scratch")?,
                }, stream.raw)?;
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
