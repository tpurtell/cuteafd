//! Native dense BF16/E4M3 operands with checkpoint-format scales and BF16 gates.
use crate::shared::memory::{device::{Allocation, Device}, receipts::Inventory, weight_arena::DeviceWeightArena};
use crate::shared::qwen::weights::WeightReader;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_loader::families::qwen::HybridAttention;
use cuteafd_loader::{families::qwen3_5::Qwen35Config, serving_capacity::WeightArenaLayout};
#[cfg(test)]
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::{collections::HashMap, ffi::c_void};
use cuteafd_core::memory_layout::{arena_class, ArenaTag};

const PREFIX: &str = "model.language_model.";

use cuteafd_loader::families::qwen3_5::weight_layout::{layer_requests, native_operands,
    validate_native_matrices, Storage};
pub(crate) use cuteafd_loader::families::qwen3_5::weight_layout::{mtp_arena, target_arena, DenseFormat};
#[cfg(test)]
use cuteafd_loader::families::qwen3_5::weight_layout::layer_operands;

pub(crate) struct DenseLayer<'a> {
    pub(crate) attention: HybridAttention,
    operands: HashMap<&'static str, Allocation<'a>>,
}

impl DenseLayer<'_> {
    pub(crate) fn ptr(&self, name: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(name).with_context(|| format!("dense Qwen layer has no operand {name}"))?.buffer.ptr)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct DenseWeights<'a> {
    pub(crate) layers: Vec<DenseLayer<'a>>,
    pub(crate) norm: Allocation<'a>,
    pub(crate) head: Allocation<'a>,
}

#[cfg(test)]
fn packed_native_layout(tensors: &[(Vec<u8>, DType, Vec<usize>)], storage: Storage) -> Result<Vec<u8>> {
    validate_native_matrices(tensors.iter().map(|(bytes, dtype, shape)|
        (dtype, shape.as_slice(), bytes.len() as u64)), storage)?;
    let mut out = Vec::new();
    for (bytes, dtype, _) in tensors {
        if storage == Storage::Scales && *dtype == DType::Bf16 {
            out.extend(bytes.chunks_exact(2).flat_map(|v|
                (u32::from(u16::from_le_bytes([v[0], v[1]])) << 16).to_le_bytes()));
        } else { out.extend_from_slice(bytes); }
    }
    ensure!(!out.is_empty(), "empty native matrix packing");
    Ok(out)
}

/// Caller-named final weight owner; an arena view in serving, standalone otherwise.
pub(crate) type Allocate<'a, 'b> = dyn FnMut(&str, usize) -> Result<Allocation<'a>> + 'b;

/// A preplanned per-device weight arena for one independently retired scope.
/// Every planned view must be requested exactly once, at its planned extent.
pub(crate) struct ScopeArena<'a> {
    arena: DeviceWeightArena<'a>,
    taken: std::cell::RefCell<std::collections::BTreeSet<String>>,
}

impl<'a> ScopeArena<'a> {
    pub(crate) fn new(device: Device<'a>, layout: WeightArenaLayout, tag: ArenaTag<arena_class::Immutable>) -> Result<Self> {
        Ok(Self { arena: DeviceWeightArena::new(device, layout, tag)?, taken: Default::default() })
    }

    pub(crate) fn allocate(&self, name: &str, bytes: usize) -> Result<Allocation<'a>> {
        ensure!(self.taken.borrow_mut().insert(name.to_owned()), "{name}: weight view requested twice");
        self.arena.view(name, bytes)
    }

    /// The loader has bound every planned view; nothing in the backing is unowned.
    pub(crate) fn complete(&self) -> Result<()> {
        let taken = self.taken.borrow();
        let missing: Vec<_> = self.arena.layout().views().keys().filter(|name| !taken.contains(*name)).collect();
        ensure!(missing.is_empty(), "{} views never loaded: {missing:?}", self.arena.layout().id());
        Ok(())
    }

    pub(crate) fn layout(&self) -> &WeightArenaLayout { self.arena.layout() }
}

impl DenseWeights<'_> {
    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        for (index, layer) in self.layers.iter().enumerate() {
            layer.inventory(inventory, &format!("layer.{index}"))?;
        }
        self.norm.inventory(inventory, "norm")?;
        self.head.inventory(inventory, "head")
    }
}

impl DenseLayer<'_> {
    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        let mut names: Vec<_> = self.operands.keys().collect();
        names.sort();
        for name in names { self.operands[name].inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}

impl<'a> WeightReader<'a> {
    /// Validate every header before uploading any weights.
    pub(crate) fn dense_native(&self, cfg: &Qwen35Config, norm_dtype: DType, format: DenseFormat,
        allocate: &mut Allocate<'a, '_>) -> Result<DenseWeights<'a>> {
        cuteafd_loader::placement::families::qwen3_5::NativeWeights::native(self.checkpoint, cfg, norm_dtype)?;
        let norm_name = format!("{PREFIX}norm.weight");
        let head_name = if cfg.tied_embeddings { format!("{PREFIX}embed_tokens.weight") } else { "lm_head.weight".into() };
        let layers = cfg.attention.iter().enumerate().map(|(index, &kind)|
            self.dense_layer(&format!("{PREFIX}layers.{index}"), kind, format, allocate)).collect::<Result<_>>()?;
        Ok(DenseWeights { layers, norm: self.bf16_with(&norm_name, &norm_name, &mut *allocate)?,
            head: self.bf16_with("head.weight", &head_name, &mut *allocate)? })
    }

    pub(crate) fn dense_layer_bytes(&self, prefix: &str, attention: HybridAttention, format: DenseFormat) -> Result<u64> {
        layer_requests(self.checkpoint, prefix, attention, format)?.into_iter().try_fold(0u64, |total, (_, bytes)| {
            total.checked_add(u64::try_from(bytes)?).context("dense Qwen layer size overflow")
        })
    }

    /// MTP and target blocks share the checkpoint-native projection packing.
    /// Each operand is one caller-named owner `{prefix}/{operand}`.
    pub(crate) fn dense_layer(&self, prefix: &str, attention: HybridAttention, format: DenseFormat,
        allocate: &mut Allocate<'a, '_>) -> Result<DenseLayer<'a>> {
        let operands = native_operands(prefix, attention, format).into_iter().map(|spec| {
                let name = format!("{prefix}/{}", spec.name);
                let allocate = &mut *allocate;
                let weight = match spec.storage {
                    Storage::Bf16 if spec.tensors.len() > 1 => self.rows_with(&name, &spec.tensors, 0, allocate)?,
                    Storage::Bf16 => self.bf16_with(&name, &spec.tensors[0], allocate)?,
                    Storage::F32 => self.f32_with(&name, &spec.tensors[0], allocate)?,
                    Storage::Fp8 | Storage::Scales => {
                        let headers = spec.tensors.iter().map(|name| self.tensor(name)).collect::<Result<Vec<_>>>()?;
                        validate_native_matrices(headers.iter().map(|tensor|
                            (&tensor.meta.dtype, tensor.meta.shape.as_slice(), tensor.meta.byte_length)), spec.storage)?;
                        self.native_with(&name, &spec.tensors, spec.storage == Storage::Scales, allocate)?
                    },
                    Storage::NativeNorm => self.native_with(&name, &spec.tensors, false, allocate)?,
                };
            Ok((spec.name, weight))
        }).collect::<Result<_>>()?;
        Ok(DenseLayer { attention, operands })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(format: DenseFormat, tied_embeddings: bool) -> (Checkpoint, Qwen35Config) {
        use cuteafd_loader::{families::qwen::{GdnGeometry, GdnOutputGate}, plan::checkpoint::CheckpointTensor,
            SafetensorsTensorMetadata};
        let cfg = Qwen35Config { hidden: 128, intermediate: 128, vocab: 128,
            attention: vec![HybridAttention::Gdn, HybridAttention::Full], heads: 1, kv_heads: 1, head_dim: 256,
            gdn: GdnGeometry { key_heads: 1, value_heads: 1, key_dim: 128, value_dim: 128,
                conv_kernel: 4, output_gate: GdnOutputGate::Silu }, rope_dim: 64, rope_theta: 1e7,
            rms_norm_eps: 1e-6, max_context: 4096, mtp_layers: 1, tied_embeddings };
        let mut tensors = std::collections::BTreeMap::new();
        let mut add = |name: String, dtype: DType, shape: Vec<usize>| {
            let width = match dtype { DType::F8E4M3 => 1, DType::F32 => 4, _ => 2 };
            let byte_length = (shape.iter().product::<usize>() * width) as u64;
            tensors.insert(name.clone(), CheckpointTensor { shard: "model.safetensors".into(),
                meta: SafetensorsTensorMetadata { name, dtype, shape, byte_offset: 0, byte_length } });
        };
        for (prefix, kind) in [(format!("{PREFIX}layers.0"), HybridAttention::Gdn),
            (format!("{PREFIX}layers.1"), HybridAttention::Full), ("mtp.layers.0".into(), HybridAttention::Full)] {
            for spec in native_operands(&prefix, kind, format) {
                for name in spec.tensors {
                    let (dtype, shape) = match spec.storage {
                        Storage::Fp8 => (DType::F8E4M3, vec![128, 128]),
                        Storage::Scales => (DType::Bf16, vec![1, 1]),
                        Storage::F32 => (DType::Bf16, vec![129]),
                        Storage::NativeNorm => (DType::Bf16, vec![128]),
                        Storage::Bf16 => (DType::Bf16, vec![128, 128]),
                    };
                    add(name, dtype, shape);
                }
            }
        }
        for name in [format!("{PREFIX}embed_tokens.weight"), "lm_head.weight".into()] {
            add(name, DType::Bf16, vec![128, 128]);
        }
        add(format!("{PREFIX}norm.weight"), DType::Bf16, vec![128]);
        for name in ["norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding"] {
            add(format!("mtp.{name}.weight"), DType::Bf16, vec![128]);
        }
        add("mtp.fc.weight".into(), DType::Bf16, vec![128, 256]);
        (Checkpoint { view: None, snapshot: Default::default(), config: serde_json::json!({}), quantize_config: None,
            weight_map: Default::default(), tensors: tensors.into_values().collect(), missing_shards: vec![], shard_bytes: 0 }, cfg)
    }

    fn check_layout(layout: &WeightArenaLayout) {
        let mut views = layout.views().values().copied().collect::<Vec<_>>();
        views.sort_by_key(|view| view.offset());
        let mut end = 0;
        for view in views {
            assert_eq!(view.offset() % 256, 0);
            assert!(view.offset() >= end);
            assert!(view.bytes() >= 256);
            end = view.offset() + view.bytes();
        }
        assert_eq!(end, layout.bytes());
        assert_eq!(layout.bytes(), layout.payload_bytes() + layout.padding_bytes());
    }

    #[test]
    fn arena_layouts_preserve_native_formats_floor_and_separate_mtp_ownership() -> Result<()> {
        for format in [DenseFormat::Bf16, DenseFormat::Fp8] {
            for tied in [false, true] {
                let (mut checkpoint, cfg) = fixture(format, tied);
                let target = target_arena(&checkpoint, &cfg, DType::Bf16, format)?;
                let mtp = mtp_arena(&checkpoint, &cfg, format)?;
                check_layout(&target);
                check_layout(&mtp);
                assert_ne!(target.id(), mtp.id());
                assert_eq!(target.views().len(), if format.fp8() { 30 } else { 21 });
                assert_eq!(mtp.views().len(), if format.fp8() { 16 } else { 12 });
                assert!(target.views().keys().all(|name| !name.starts_with("mtp.")));
                assert!(mtp.views().keys().all(|name| name.starts_with("mtp.")));
                assert_eq!(target.view("embedding.table").unwrap().bytes(), 32768);
                assert_eq!(target.view("head.weight").unwrap().bytes(), 32768);
                let p = format!("{PREFIX}layers.0");
                assert_eq!(target.view(&format!("{p}/a_log")).unwrap().bytes(), 516);
                assert!(target.padding_bytes() > 0);
                for tensor in &mut checkpoint.tensors {
                    if ["conv1d.weight", "A_log", "dt_bias"].iter().any(|suffix| tensor.meta.name.ends_with(suffix)) {
                        tensor.meta.dtype = DType::F32;
                        tensor.meta.byte_length *= 2;
                    }
                }
                assert_eq!(target_arena(&checkpoint, &cfg, DType::Bf16, format)?, target);
                if format.fp8() {
                    assert_eq!(target.view(&format!("{p}/w_qkvz_scale")).unwrap().bytes(), 256);
                    assert!(target.view(&format!("{p}/w_in")).is_none());
                    for tensor in &mut checkpoint.tensors {
                        if tensor.meta.name.ends_with("weight_scale_inv") {
                            tensor.meta.dtype = DType::F32;
                            tensor.meta.byte_length *= 2;
                        }
                    }
                    assert_eq!(target_arena(&checkpoint, &cfg, DType::Bf16, format)?, target);
                    assert_eq!(mtp_arena(&checkpoint, &cfg, format)?, mtp);
                }
                for tensor in &mut checkpoint.tensors {
                    if tensor.meta.name.ends_with("linear_attn.norm.weight") {
                        tensor.meta.dtype = DType::F32;
                        tensor.meta.byte_length *= 2;
                    }
                }
                let fp32_norm = target_arena(&checkpoint, &cfg, DType::F32, format)?;
                assert_eq!(fp32_norm.view(&format!("{p}/norm_w")).unwrap().bytes(), 512);
                assert!(target_arena(&checkpoint, &cfg, DType::Bf16, format).is_err());
                checkpoint.tensors.retain(|t| !t.meta.name.ends_with("linear_attn.A_log"));
                assert!(target_arena(&checkpoint, &cfg, DType::F32, format).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn native_arena_admission_equals_loaded_target_and_mtp_backings() -> Result<()> {
        use cuteafd_loader::placement::{families::qwen3_5::{self, DenseFacts, DrafterBytes, NativeBuffers, NativeWeights},
            planner::{CoordinatorDevices, Lifetime, Operation, Phase, Pin, Request}};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        fixture.quiet_events(&library, true)?;
        let device = Device { library: &library, id: 0 };
        for format in [DenseFormat::Bf16, DenseFormat::Fp8] {
            let (mut checkpoint, cfg) = self::fixture(format, false);
            let snapshot = tempfile::tempdir_in(fixture.directory())?;
            checkpoint.snapshot = snapshot.path().into();
            let mut end = 0;
            for tensor in &mut checkpoint.tensors {
                tensor.meta.byte_offset = end;
                end += tensor.meta.byte_length;
                checkpoint.weight_map.insert(tensor.meta.name.clone(), tensor.shard.clone());
            }
            std::fs::File::create(snapshot.path().join("model.safetensors"))?.set_len(end)?;
            let target = target_arena(&checkpoint, &cfg, DType::Bf16, format)?;
            let mtp = mtp_arena(&checkpoint, &cfg, format)?;
            let weights = NativeWeights::native(&checkpoint, &cfg, DType::Bf16)?;
            let buffers = NativeBuffers::new(&cfg, 256, 1, 0, false)?;
            let draft = DrafterBytes::native_mtp(&checkpoint, &cfg)?;
            assert_eq!(draft.weight_arena, Some(("qwen3_5/mtp/gpu0", mtp.bytes() as u64)));
            assert_eq!(draft.weights.0, mtp.payload_bytes() as u64);
            let request = Request { context: 4096.into(), pool: Pin::Exact(4096), onboard: Pin::Auto,
                prefill: Pin::Auto, operation: Operation::Serving, resident_weight_limit: None,
                spark_reply_formats: Default::default() };
            let plan = qwen3_5::admit(request, qwen3_5::forecast_budgets(&cuteafd_loader::placement::ARCH_CONTEXTS[1],
                96 << 30, 0, (128 << 30, 128 << 30))?, DenseFacts { cfg: &cfg, weights: &weights,
                target_arena: &target, buffers: &buffers, slots: 1, record_bytes: 1024, drafter: Some(draft),
                blas_head: cuteafd_loader::placement::planner::serving::BlasHead::TensorLt,
                serving: None, host_prefix: cuteafd_loader::placement::planner::serving::HostPrefixOwners {
                    quota_bytes: 0, chunk_bytes: 0 }, verify_check: false })?;
            device.run(|| -> Result<()> {
                let reader = WeightReader { library: &library, checkpoint: &checkpoint };
                for (layout, lifetime, tag) in [(target, Lifetime::RESIDENT,
                    ArenaTag::weights("qwen35/target")), (mtp,
                    Lifetime { first: Phase::DraftLoad, last: Phase::Growth },
                    ArenaTag::drafter_weights("qwen35/mtp"))] {
                    let kind = tag.kind();
                    let arena = ScopeArena::new(device, layout, tag)?;
                    let mut inventory = Inventory::new(CoordinatorDevices::new(&[0])?, "test", lifetime,
                        kind.category(), "native weights");
                    if lifetime == Lifetime::RESIDENT {
                        let weights = reader.dense_native(&cfg, DType::Bf16, format,
                            &mut |name: &str, bytes: usize| arena.allocate(name, bytes))?;
                        let embedding = reader.bf16_with("embedding.table", &format!("{PREFIX}embed_tokens.weight"),
                            &mut |name: &str, bytes: usize| arena.allocate(name, bytes))?;
                        weights.inventory(&mut inventory)?;
                        embedding.inventory(&mut inventory, "embedding")?;
                    } else {
                        let layer = reader.dense_layer("mtp.layers.0", HybridAttention::Full, format,
                            &mut |name: &str, bytes: usize| arena.allocate(name, bytes))?;
                        layer.inventory(&mut inventory, "mtp.layer")?;
                        for name in ["mtp.fc.weight", "mtp.pre_fc_norm_hidden.weight",
                            "mtp.pre_fc_norm_embedding.weight", "mtp.norm.weight"] {
                            reader.bf16_with(name, name, &mut |name: &str, bytes: usize| arena.allocate(name, bytes))?
                                .inventory(&mut inventory, name)?;
                        }
                    }
                    arena.complete()?;
                    let roots: Vec<_> = inventory.receipts.iter().filter(|r| r.allocation.view.is_none()).collect();
                    assert_eq!(roots.len(), 1);
                    let planned = plan.facts.allocations.iter().find(|owner| owner.id == arena.layout().id()).unwrap();
                    assert_eq!(&roots[0].allocation, planned);
                    assert_eq!(inventory.receipts.len() - 1, arena.layout().views().len());
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "reads installed checkpoint headers; set CUTEAFD_QWEN_DENSE_ARENA_SNAPSHOT"]
    fn installed_arena_inventory() -> Result<()> {
        let snapshot = std::env::var("CUTEAFD_QWEN_DENSE_ARENA_SNAPSHOT")?;
        let checkpoint = Checkpoint::open(std::path::Path::new(&snapshot))?;
        let cfg = Qwen35Config::from_hf(&checkpoint.config)?;
        let gate = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{PREFIX}layers.0.mlp.gate_proj.weight"))
            .context("missing first gate")?;
        let format = if gate.meta.dtype == DType::F8E4M3 { DenseFormat::Fp8 } else { DenseFormat::Bf16 };
        let norm = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{PREFIX}layers.0.linear_attn.norm.weight"))
            .context("missing GDN norm")?.meta.dtype.clone();
        use cuteafd_loader::placement::{families::qwen3_5::{self, DenseFacts, DrafterBytes, NativeBuffers, NativeWeights},
            planner::{Lifetime, Operation, Phase, Pin, Request, allocation_granularity_forecast, ALLOCATION_GRANULARITY_GROUP}};
        let target = target_arena(&checkpoint, &cfg, norm.clone(), format)?;
        let programs = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS")
            .context("set CUTEAFD_DEPLOYMENT_TEST_PROGRAMS to installed exports")?;
        let mut options = cuteafd_loader::plan::PlanOptions {
            placement: cuteafd_loader::plan::ExpertPlacement::Local,
            vision: cuteafd_loader::plan::MediaMode::Off, audio: cuteafd_loader::plan::MediaMode::Off,
            layout: Some(cuteafd_loader::plan::layout::LayoutOptions {
                rtx_bytes: vec![96 << 30], host_memory: Some((128 << 30, 128 << 30)), concurrency: 1,
                state_slots: Some(1), prefix_slots: Some(0), host_cache_bytes: Some(0), native_mtp_layers: 0,
                pool_tokens: Some(262144), workspace_manifest: Some(programs.into()),
                dense_qwen_kv: cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, ..Default::default()
            }), ..Default::default()
        };
        let weights = NativeWeights::native(&checkpoint, &cfg, norm)?;
        let buffers = NativeBuffers::new(&cfg, 256, 1, 0, false)?;
        let mut layouts = vec![target.clone()];
        let mut drafts = vec![None];
        if cfg.mtp_layers == 1 {
            layouts.push(mtp_arena(&checkpoint, &cfg, format)?);
            drafts.push(Some(DrafterBytes::native_mtp(&checkpoint, &cfg)?));
        }
        if let Ok(draft) = std::env::var("CUTEAFD_QWEN_DENSE_ARENA_DRAFT") {
            use crate::shared::speculation::block_dspark::DsparkDrafter;
            let snapshot = std::path::Path::new(&draft);
            let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
            let mut requests = DsparkDrafter::arena_requests(snapshot,
                cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only, 0)?;
            layouts.push(WeightArenaLayout::new("qwen3_5/dspark/gpu0", requests.remove(&0).context("dSpark device")?)?);
            drafts.push(Some(DrafterBytes::block_dspark(snapshot, &config, 1, 142, cfg.max_context)?));
        }
        for drafter in drafts {
            let expected = layouts.iter().filter(|layout| layout.id() == target.id()
                || drafter.as_ref().and_then(|draft| draft.weight_arena).is_some_and(|(id, _)| id == layout.id())).collect::<Vec<_>>();
            let layout = options.layout.as_mut().unwrap();
            layout.dense_qwen_mtp = drafter.as_ref().and_then(|draft| draft.weight_arena)
                .is_some_and(|(id, _)| id == "qwen3_5/mtp/gpu0");
            layout.dense_qwen_drafter_snapshot = drafter.as_ref().and_then(|draft| draft.weight_arena)
                .filter(|(id, _)| *id == "qwen3_5/dspark/gpu0")
                .map(|_| std::env::var_os("CUTEAFD_QWEN_DENSE_ARENA_DRAFT").unwrap().into());
            let report = cuteafd_loader::plan::plan(std::path::Path::new(&snapshot), &options)?;
            assert!(report.executable(), "{:?}", report.hints);
            let memory = report.memory_layout.context("installed dense layout")?;
            assert_eq!(memory.devices[0].items.iter().filter(|item| item.group == "weight arena")
                .map(|item| item.bytes).sum::<u64>(), expected.iter().map(|layout| layout.bytes() as u64).sum::<u64>());
            let tails = expected.iter().map(|layout| allocation_granularity_forecast(&[layout.bytes() as u64]))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            assert_eq!(memory.devices[0].items.iter().filter(|item| item.group == ALLOCATION_GRANULARITY_GROUP)
                .map(|item| item.bytes).sum::<u64>(), tails.iter().sum::<u64>());
            let request = Request { context: 4096.into(), pool: Pin::Exact(4096), onboard: Pin::Auto,
                prefill: Pin::Auto, operation: Operation::Serving, resident_weight_limit: None,
                spark_reply_formats: Default::default() };
            let plan = qwen3_5::admit(request, qwen3_5::forecast_budgets(&cuteafd_loader::placement::ARCH_CONTEXTS[1],
                96 << 30, 0, (128 << 30, 128 << 30))?, DenseFacts { cfg: &cfg, weights: &weights,
                target_arena: &target, buffers: &buffers, slots: 1, record_bytes: 4096, drafter,
                blas_head: cuteafd_loader::placement::planner::serving::BlasHead::TensorLt,
                serving: None, host_prefix: cuteafd_loader::placement::planner::serving::HostPrefixOwners {
                    quota_bytes: 0, chunk_bytes: 0 }, verify_check: false })?;
            assert_eq!(plan.facts.resident_weights.len(), expected.len());
            assert_eq!(plan.facts.allocation_granularity.len(), expected.len());
            for layout in expected {
                let root = plan.facts.allocations.iter().find(|owner| owner.id == layout.id()).unwrap();
                assert_eq!(root.bytes, layout.bytes() as u64);
                assert!(root.view.is_none());
                assert_eq!(root.lifetime, if layout.id() == target.id() { Lifetime::RESIDENT }
                    else { Lifetime { first: Phase::DraftLoad, last: Phase::Growth } });
                let tail = plan.facts.allocation_granularity.iter().find(|tail| tail.id == format!("{}.tail", layout.id())).unwrap();
                assert_eq!(tail.requests(), [layout.bytes() as u64]);
                assert_eq!(tail.bytes()?, allocation_granularity_forecast(&[root.bytes])?);
                assert_eq!(plan.facts.resident_weights.iter().find(|owner| owner.id == root.id).unwrap().weights.0, root.bytes);
            }
        }
        for layout in layouts {
            check_layout(&layout);
            let large = layout.views().values().filter(|view| view.bytes() > 1 << 20).collect::<Vec<_>>();
            let slack = |bytes: usize| bytes.div_ceil(2 << 20) * (2 << 20) - bytes;
            println!("{}: allocations={} payload={} large_count={} large_bytes={} before_slack={} arena={} padding={} tail={}",
                layout.id(), layout.views().len(), layout.payload_bytes(), large.len(),
                large.iter().map(|v| v.bytes()).sum::<usize>(), large.iter().map(|v| slack(v.bytes())).sum::<usize>(),
                layout.bytes(), layout.padding_bytes(), slack(layout.bytes()));
        }
        Ok(())
    }

    #[test]
    fn fp8_operands_keep_raw_weight_bytes_exact_scales_and_bf16_gates() -> Result<()> {
        let specs = native_operands("layer", HybridAttention::Gdn, DenseFormat::Fp8);
        let find = |name| specs.iter().find(|spec| spec.name == name).unwrap();
        assert_eq!(find("w_ba").tensors, ["layer.linear_attn.in_proj_b.weight", "layer.linear_attn.in_proj_a.weight"]);
        assert_eq!(find("w_ba").storage, Storage::Bf16);
        assert_eq!(find("w_qkvz_scale").tensors, ["layer.linear_attn.in_proj_qkv.weight_scale_inv",
            "layer.linear_attn.in_proj_z.weight_scale_inv"]);
        assert!(specs.iter().all(|spec| !matches!(spec.name, "w_in" | "w_out")));
        assert_eq!(packed_native_layout(&[(vec![1, 2], DType::F8E4M3, vec![1, 2]),
            (vec![3, 4], DType::F8E4M3, vec![1, 2])], Storage::Fp8)?, vec![1, 2, 3, 4]);
        let bits = [0x3f80u16, 0x8000, 0x0001, 0x7f7f];
        let bytes = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
        let widened = packed_native_layout(&[(bytes, DType::Bf16, vec![2, 2])], Storage::Scales)?;
        assert_eq!(widened, bits.iter().flat_map(|&value| (u32::from(value) << 16).to_le_bytes()).collect::<Vec<_>>());
        assert!(packed_native_layout(&[(vec![0; 2], DType::Bf16, vec![1, 1])], Storage::Fp8).is_err());
        assert!(packed_native_layout(&[(vec![0], DType::F8E4M3, vec![1, 2])], Storage::Fp8).is_err());
        Ok(())
    }

    #[test]
    fn dense_operands_keep_native_projection_order_and_norm_precision() {
        for kind in [HybridAttention::Gdn, HybridAttention::Full] {
            let specs = layer_operands("layer", kind);
            let names = |operand: &str| specs.iter().find(|s| s.name == operand).unwrap().tensors.clone();
            assert_eq!(names("ffn.w_gate_up"), ["layer.mlp.gate_proj.weight", "layer.mlp.up_proj.weight"]);
            if kind == HybridAttention::Gdn {
                assert_eq!(names("w_in"), ["layer.linear_attn.in_proj_qkv.weight", "layer.linear_attn.in_proj_z.weight",
                    "layer.linear_attn.in_proj_b.weight", "layer.linear_attn.in_proj_a.weight"]);
                assert_eq!(specs.iter().find(|s| s.name == "norm_w").unwrap().storage, Storage::NativeNorm);
                assert_eq!(specs.iter().filter(|s| s.storage == Storage::F32).count(), 3);
            } else {
                assert_eq!(names("w_in"), ["layer.self_attn.q_proj.weight", "layer.self_attn.k_proj.weight", "layer.self_attn.v_proj.weight"]);
            }
            assert!(specs.iter().all(|s| !s.name.contains("expert") && !s.name.contains("hc") && !s.name.contains("index")));
        }
    }
}
