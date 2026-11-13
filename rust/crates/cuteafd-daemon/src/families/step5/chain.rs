//! Admitted single-rank target prefix. Serving remains fail-closed.
use super::{
    csa::CsaLayer,
    ffn::DensePlan,
    rope::TargetTables,
    routed::{RoutedFfn, RoutedPlan},
    swa::SwaLayer,
    transaction::Step,
};
use crate::shared::{
    memory::device::{Allocation, Device},
    spark_intake::SparkLink,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer};
use cuteafd_hostcache::copy::CopyEngine;
use cuteafd_loader::plan::{
    checkpoint::Checkpoint,
    families::step5::coordinator::{CoordinatorContract, RankWeightLayout},
};
use std::rc::Rc;

enum Attention<'a> {
    Sliding(SwaLayer<'a>),
    Sparse(CsaLayer<'a>),
}
struct Layer<'a> {
    attention: Attention<'a>,
    routed: Option<RoutedFfn<'a>>,
}
fn validate_ffn_ownership(
    dense_planned: bool,
    dense_owned: bool,
    routed_owned: bool,
) -> Result<()> {
    ensure!(
        dense_owned == dense_planned && routed_owned != dense_planned,
        "Step-5 planned FFN must have exactly one matching executor: dense planned={dense_planned}, dense owned={dense_owned}, routed owned={routed_owned}"
    );
    Ok(())
}
impl Layer<'_> {
    fn input(&self) -> CuteafdDeviceBuffer {
        match &self.attention {
            Attention::Sliding(l) => l.input(),
            Attention::Sparse(l) => l.input(),
        }
    }
    fn output(&self) -> CuteafdDeviceBuffer {
        self.routed.as_ref().map_or_else(
            || match &self.attention {
                Attention::Sliding(l) => l.output(),
                Attention::Sparse(l) => l.output(),
            },
            RoutedFfn::output,
        )
    }
}

fn weight_layout(layout: &RankWeightLayout, rope: &cuteafd_loader::plan::rope::Tables,
    device: i32, count: usize, head: bool) -> Result<cuteafd_loader::serving_capacity::WeightArenaLayout> {
    // The separately admitted vocabulary head is not part of the prefix owner.
    let mut prefix = layout.clone();
    let selected = |name: &str| name.strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.')).and_then(|(layer, _)| layer.parse::<usize>().ok())
        .is_some_and(|layer| layer < count);
    let selected = |name: &str| selected(name) || (head && matches!(name, "model.norm.weight" | "lm_head.weight"));
    prefix.operands.retain(|operand| selected(&operand.name));
    prefix.core.retain(|name, _| selected(name));
    prefix.target_arena(Some(rope), device, count, false)
}

fn workspace_requests(layout: &RankWeightLayout,
    layer: &cuteafd_loader::plan::families::step5::coordinator::LayerContract,
    rows: usize) -> Result<Vec<u64>> {
    use cuteafd_loader::{formats::trellis_core::ProjectionScratchLayout,
        plan::families::step5::native};
    let mut requests = if layer.sliding.is_some() {
        native::swa_planes(rows, layer.geometry.query_heads)?.to_vec()
    } else { native::csa_planes(rows)?.to_vec() };
    requests.extend(if layer.dense_activation.is_some() {
        native::dense_planes(rows)?.to_vec()
    } else { native::routed_planes(rows)?.to_vec() });
    // Cold owners allocate every plane independently, unlike serving's
    // serialized arena. Native attention and FFN each retain packed scratch.
    let mut scratch = std::collections::BTreeMap::<bool, ProjectionScratchLayout>::new();
    let prefix = format!("model.layers.{}.", layer.layer);
    for (name, core) in layout.core.iter().filter(|(name, _)| name.starts_with(&prefix)) {
        let next = ProjectionScratchLayout::new(core.shape[1], core.shape[0], rows)?;
        let attention = name.contains(".self_attn.");
        match scratch.get_mut(&attention) {
            Some(owner) => owner.merge(&next)?,
            None => { scratch.insert(attention, next); }
        }
    }
    for layout in scratch.values() { requests.push(layout.packed_bytes()?); }
    Ok(requests.into_iter().map(|bytes| bytes as u64).collect())
}

pub(super) fn physical_bytes(requests: &[u64]) -> Result<u64> {
    let payload = requests.iter().try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
        .context("Step-5 cold allocation payload overflow")?;
    payload.checked_add(cuteafd_loader::placement::planner::allocation_granularity_forecast(requests)?)
        .context("Step-5 cold allocation tail overflow")
}

pub(super) struct TargetChain<'a> {
    layers: Vec<Layer<'a>>,
    rows: usize,
    bytes: u64,
    _weights: crate::shared::memory::weight_arena::DeviceWeightArena<'a>,
}
impl<'a> TargetChain<'a> {
    fn plan(
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        count: usize,
        rows: usize,
        head: bool,
    ) -> Result<(u64, Vec<Option<RoutedPlan>>)> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        ensure!(
            (1..=contract.layers.len()).contains(&count) && (1..=4096).contains(&rows),
            "Step-5 target prefix count/rows"
        );
        let layout = RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
        let arena = weight_layout(&layout, &contract.rope_tables, 0, count, head)?;
        let weights = arena.bytes() as u64;
        let tail = cuteafd_loader::placement::planner::allocation_granularity_forecast(&[weights])?;
        let mut requests = vec![weights];
        tracing::info!(weights, tail, count, "Step-5 cold chain weight arena");
        let catalog = (count > 3).then(|| cuteafd_loader::read_expert_catalog_view(
            checkpoint.view.as_ref().context("Step-5 target chain requires its selected checkpoint view")?)).transpose()?;
        let mut routed = Vec::new();
        for layer in &contract.layers[..count] {
            if layer.dense_activation.is_some() {
                routed.push(None);
                DensePlan::prepare(programs, checkpoint, layer.layer, rows, u64::MAX, &[])?;
            } else {
                let plan =
                    RoutedPlan::prepare(programs, checkpoint, layer.layer, rows, u64::MAX, &[])?
                        .with_bf16_input({
                            let catalog = catalog.as_ref().context("Step-5 routed expert headers missing")?;
                            super::experts::StepExperts::from_catalog(catalog)?.layer_bf16_input(catalog, layer.layer)?
                        });
                routed.push(Some(plan));
            }
            requests.extend(workspace_requests(&layout, layer, rows)?);
        }
        Ok((physical_bytes(&requests)?, routed))
    }
    pub fn required_bytes(
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        count: usize,
        rows: usize,
    ) -> Result<u64> {
        Ok(Self::plan(programs, checkpoint, count, rows, false)?.0)
    }
    pub(super) fn required_with_head(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        count: usize, rows: usize, head: bool) -> Result<u64> {
        Ok(Self::plan(programs, checkpoint, count, rows, head)?.0)
    }

    pub(super) fn weight_arena(&self) -> &crate::shared::memory::weight_arena::DeviceWeightArena<'a> {
        &self._weights
    }

    pub fn load(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        count: usize,
        rows: usize,
        available: u64,
        reservations: &[u64],
        kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache,
    ) -> Result<Self> {
        Self::load_with_head(device, programs, checkpoint, count, rows, available, reservations, kv_format, false)
    }
    pub(super) fn load_with_head(device: Device<'a>, programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        count: usize, rows: usize, available: u64, reservations: &[u64],
        kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, head: bool) -> Result<Self> {
        let (bytes, routed) = Self::plan(programs, checkpoint, count, rows, head)?;
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let total = reservations
            .iter()
            .try_fold(bytes, |n, r| n.checked_add(*r))
            .context("chain reservation overflow")?;
        ensure!(
            total <= available,
            "Step-5 target prefix needs {total} bytes, available {available}"
        );
        let layout = RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
        let weights = crate::shared::memory::weight_arena::DeviceWeightArena::new(device,
            weight_layout(&layout, &contract.rope_tables, device.id, count, head)?,
            cuteafd_core::memory_layout::ArenaTag::weights("step5/target"))?;
        let rope = Rc::new(TargetTables::load_in(
            device,
            programs,
            &contract,
            "step5_",
            available,
            reservations,
            Some(&weights),
        )?);
        let mut layers = Vec::new();
        for (id, plan) in routed.into_iter().enumerate() {
            let attention = if contract.layers[id].sliding.is_some() {
                Attention::Sliding(SwaLayer::load_weights_in(
                    device,
                    programs,
                    checkpoint,
                    id,
                    rows,
                    Some(rope.clone()),
                    available,
                    &[], kv_format, Some(&weights))?)
            } else {
                Attention::Sparse(CsaLayer::load_weights_in(
                    device,
                    programs,
                    checkpoint,
                    id,
                    rows,
                    Some(rope.clone()),
                    available,
                    &[], kv_format, Some(&weights))?)
            };
            let routed = plan
                .map(|p| RoutedFfn::load_shared(device, programs, checkpoint, id, p, None, None, Some(&weights)))
                .transpose()?;
            let dense_owned = match &attention {
                Attention::Sliding(owner) => owner.has_dense_ffn(),
                Attention::Sparse(owner) => owner.has_dense_ffn(),
            };
            validate_ffn_ownership(
                contract.layers[id].dense_activation.is_some(),
                dense_owned,
                routed.is_some(),
            )
            .with_context(|| format!("Step-5 layer {id} FFN ownership"))?;
            layers.push(Layer { attention, routed });
        }
        Ok(Self {
            layers,
            rows,
            bytes,
            _weights: weights,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn output(&self, layer: usize) -> Result<CuteafdDeviceBuffer> {
        Ok(self
            .layers
            .get(layer)
            .context("Step-5 chain output layer")?
            .output())
    }
    pub(super) fn diagnostic_layer0(&self) -> Result<&SwaLayer<'a>> {
        match &self.layers.first().context("Step-5 missing layer zero")?.attention {
            Attention::Sliding(owner) => Ok(owner),
            _ => anyhow::bail!("Step-5 layer zero must be SWA"),
        }
    }
    pub fn enqueue_step<'e, E: CopyEngine>(
        &'e mut self,
        step: &mut Step<'e, 'a, E>,
        rank: usize,
        input: &'e Allocation<'a>,
        link: Option<(&'e std::cell::RefCell<SparkLink<'a>>, super::routed::GateRequest)>,
    ) -> Result<()> {
        ensure!(
            rank == 0,
            "Step-5 target chain currently binds one rank only"
        );
        ensure!(
            !self.layers.iter().any(|l| l.routed.is_some()) || link.is_some(),
            "Step-5 routed chain needs an expert link"
        );
        if let Some((link, _)) = link {
            step.compose(rank, link, |_, _| Ok(()))?;
        }
        step.compose(rank, input, |owner, meta| {
            ensure!(
                meta.rows <= self.rows
                    && meta.heads == 64
                    && owner.buffer.bytes >= meta.rows * 4096 * 4,
                "Step-5 chain input capacity/head geometry"
            );
            ensure!(
                owner.device.id == meta.stream.device.id
                    && std::ptr::eq(owner.device.library, meta.stream.device.library),
                "Step-5 chain input device"
            );
            Ok(())
        })?;
        let mut source = input.buffer;
        for layer in &mut self.layers {
            let target = layer.input();
            let output = layer.output();
            step.compose(rank, input, |_, meta| {
                meta.stream.device.run(|| {
                    // SAFETY: prior layer/input owners are borrowed through this transaction; views are disjoint.
                    unsafe {
                        meta.stream.device.library.copy_d2d_async(
                            target,
                            source,
                            meta.rows * 4096 * 4,
                            meta.stream.raw,
                        )
                    }
                })
            })?;
            let post = match &mut layer.attention {
                Attention::Sliding(owner) => {
                    let out = owner.output();
                    owner.enqueue_step(step, rank)?;
                    out
                }
                Attention::Sparse(owner) => {
                    let out = owner.output();
                    owner.enqueue_step(step, rank)?;
                    out
                }
            };
            if let Some(ffn) = &layer.routed {
                step.compose(rank, ffn, |ffn, meta| {
                    meta.stream.device.run(|| {
                        // SAFETY: disjoint owned FFN input and preceding attention output share this stream.
                        unsafe {
                            meta.stream.device.library.copy_d2d_async(
                                ffn.input(),
                                post,
                                meta.rows * 4096 * 4,
                                meta.stream.raw,
                            )
                        }
                    })?;
                    let (link, request) = link.context("Step-5 expert link")?;
                    ffn.enqueue_on(meta.rows, link, request, meta.stream)
                })?;
            }
            source = output;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        cache::CacheRank,
        prefix::{Frontier, Placement, Step5Prefix},
        transaction::Staging,
    };
    use super::*;
    use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
    use cuteafd_engine::prefix::{PrefixCache, PrefixConfig, PrefixFamily};
    use cuteafd_hostcache::copy::StubCopyEngine;
    use cuteafd_transport::{
        expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
        worker_selection::ModelAdmission,
        ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
        ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
    };
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{cell::RefCell, path::Path};
    use super::super::routed::GateRequest;

    #[test]
    fn cold_mc_views_charge_selected_core_backing_once() -> Result<()> {
        use cuteafd_loader::plan::{testing, checkpoint::{CheckpointView, CoreView},
            families::step5::{Config, exl3}};
        let root = tempfile::tempdir()?;
        testing::write_step5_exl3_snapshot(root.path());
        let source = Checkpoint::open(root.path())?;
        let cfg = Config::parse(&source)?;
        let tensors = source.tensors.iter().map(|t| (t.meta.name.clone(),
            match t.meta.dtype {
                cuteafd_core::DType::Bf16 => "BF16", cuteafd_core::DType::F16 => "F16",
                cuteafd_core::DType::F32 => "F32", cuteafd_core::DType::I16 => "I16",
                cuteafd_core::DType::I32 => "I32", cuteafd_core::DType::U8 => "U8",
                _ => panic!("fixture dtype"),
            }, t.meta.shape.clone())).collect::<Vec<_>>();
        let target = exl3::expected(&cfg).into_iter().filter(|(name, _)|
            name == "lm_head.weight" || name.strip_prefix("model.layers.")
                .and_then(|s| s.split('.').next()).and_then(|s| s.parse::<usize>().ok())
                .is_some_and(|layer| layer < cfg.layers)).collect();
        let mc = tempfile::tempdir()?;
        testing::write_mc_snapshot(mc.path(), &source.config, &tensors, &target);
        let mut charges = Vec::new();
        for view in [CoreView::Original, CoreView::K6Compact, CoreView::K8Compact] {
            let checkpoint = Checkpoint::open_view(&CheckpointView::select(mc.path(), view)?)?;
            let contract = CoordinatorContract::from_checkpoint(&checkpoint)?;
            let layout = RankWeightLayout::from_checkpoint(&checkpoint, 0, 1)?;
            let arena = weight_layout(&layout, &contract.rope_tables, 0, cfg.layers, true)?;
            let mut tensors = layout.arena_tensors(|name| name == "model.norm.weight"
                || name == "lm_head.weight" || name.strip_prefix("model.layers.")
                    .and_then(|s| s.split('.').next()).and_then(|s| s.parse::<usize>().ok())
                    .is_some_and(|layer| layer < cfg.layers))?;
            for (index, recipe) in contract.rope_tables.recipes.iter().enumerate() {
                let bytes = usize::try_from(recipe.allocation_bytes(contract.rope_tables.extent)? / 2)?;
                for part in ["cos", "sin"] { tensors.push((format!("rope.{index}.{part}"), bytes)); }
            }
            let expected = cuteafd_loader::serving_capacity::PackedWeightRegions::new(
                tensors.iter().map(|(_, bytes)| *bytes))?.bytes as u64;
            assert_eq!(arena.bytes() as u64, expected, "{view:?}");
            assert_eq!(physical_bytes(&[expected])?, expected +
                cuteafd_loader::placement::planner::allocation_granularity_forecast(&[expected])?);
            if view != CoreView::Original {
                assert!(!layout.core.is_empty());
                assert!(arena.payload_bytes() as u64 > layout.operands.iter()
                    .filter(|operand| operand.name != "model.embed_tokens.weight")
                    .map(|operand| operand.allocation_bytes).sum::<u64>());
            }
            charges.push(expected);
        }
        assert!(charges[0] > charges[2] && charges[2] > charges[1], "{charges:?}");
        Ok(())
    }


    const H: usize = 4096;
    const REVISION: &str = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1";

    #[test]
    fn every_planned_ffn_requires_one_matching_live_executor() -> Result<()> {
        // Use the admission check applied to live owners, including sparse+dense.
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/plan/families/step5/fixtures/official-fp8.json"
        ))?;
        let config = &fixture["config"]["text_config"];
        let routed = config["moe_layer_list"]
            .as_array()
            .context("routed schedule")?;
        for layer in 0..config["num_hidden_layers"]
            .as_u64()
            .context("target count")?
        {
            let dense = !routed.iter().any(|id| id.as_u64() == Some(layer));
            validate_ffn_ownership(dense, dense, !dense)?;
            assert!(
                validate_ffn_ownership(dense, false, false).is_err(),
                "layer {layer}: missing FFN"
            );
            assert!(
                validate_ffn_ownership(dense, true, true).is_err(),
                "layer {layer}: double FFN"
            );
            assert!(
                validate_ffn_ownership(dense, !dense, dense).is_err(),
                "layer {layer}: wrong FFN"
            );
        }
        Ok(())
    }

    fn fixture(
        root: &Path,
        manifest: &Value,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<Vec<u8>> {
        let element = match dtype {
            "torch.bfloat16" => 2usize,
            "torch.float32" | "torch.int32" => 4,
            "torch.float64" => 8,
            _ => anyhow::bail!("unsupported chain fixture dtype"),
        };
        let expected = shape
            .iter()
            .try_fold(element, |n, dim| n.checked_mul(*dim))
            .context("chain fixture size overflow")?;
        let entry = &manifest["files"][name];
        ensure!(
            entry["dtype"] == dtype
                && entry["shape"] == serde_json::json!(shape)
                && entry["bytes"] == expected,
            "Step-5 chain fixture dtype/shape: {name}"
        );
        let bytes = std::fs::read(root.join(format!("{name}.bin")))?;
        ensure!(
            bytes.len() == expected
                && entry["sha256"].as_str()
                    == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
            "Step-5 chain fixture hash/length: {name}"
        );
        Ok(bytes)
    }

    fn verify_weights(snapshot: &Path, expected: &serde_json::Map<String, Value>) -> Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        let checkpoint = Checkpoint::open_for_role(snapshot, "chain fixture identity", |name| {
            expected.contains_key(name)
        })?;
        ensure!(
            checkpoint.tensors.len() == expected.len(),
            "chain fixture weight inventory"
        );
        let mut buffer = vec![0u8; 1 << 20];
        for tensor in &checkpoint.tensors {
            let mut file = std::fs::File::open(snapshot.join(&tensor.shard))?;
            file.seek(SeekFrom::Start(tensor.meta.byte_offset))?;
            let mut left = tensor.meta.byte_length;
            let mut hash = Sha256::new();
            while left > 0 {
                let bytes = left.min(buffer.len() as u64) as usize;
                file.read_exact(&mut buffer[..bytes])?;
                hash.update(&buffer[..bytes]);
                left -= bytes as u64;
            }
            ensure!(
                expected[&tensor.meta.name].as_str()
                    == Some(format!("{:x}", hash.finalize()).as_str()),
                "chain fixture source weight differs: {}",
                tensor.meta.name
            );
        }
        Ok(())
    }

    fn values(bytes: &[u8], size: usize) -> Result<Vec<f64>> {
        ensure!(
            matches!(size, 2 | 4 | 8) && bytes.len() % size == 0,
            "chain value extent"
        );
        let values: Vec<_> = bytes
            .chunks_exact(size)
            .map(|v| match size {
                2 => f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16)
                    as f64,
                4 => f32::from_le_bytes(v.try_into().unwrap()) as f64,
                8 => f64::from_le_bytes(v.try_into().unwrap()),
                _ => unreachable!(),
            })
            .collect();
        ensure!(
            !values.is_empty() && values.iter().all(|v| v.is_finite()),
            "nonfinite/empty chain values"
        );
        Ok(values)
    }

    fn relative(a: &[f64], b: &[f64]) -> f64 {
        let error = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        let norm = b.iter().map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE);
        (error / norm).sqrt()
    }

    fn accuracy(actual: &[u8], official: &[u8], fp64: &[u8], width: usize) -> Result<()> {
        let (a, b, r) = (values(actual, 4)?, values(official, 4)?, values(fp64, 8)?);
        ensure!(
            width > 0 && a.len() == b.len() && a.len() == r.len() && a.len() % width == 0,
            "chain oracle shape"
        );
        eprintln!(
            "same-quant native_fp64_rel_l2={} official_fp64_rel_l2={}",
            relative(&a, &r),
            relative(&b, &r)
        );
        ensure!(
            relative(&a, &r) <= 1.01 * relative(&b, &r),
            "chain global same-quant FP64 bar"
        );
        for (row, ((a, b), r)) in a
            .chunks_exact(width)
            .zip(b.chunks_exact(width))
            .zip(r.chunks_exact(width))
            .enumerate()
        {
            ensure!(
                relative(a, r) <= 1.01 * relative(b, r),
                "chain row {row} same-quant FP64 bar"
            );
        }
        Ok(())
    }

    fn selection(actual: &[u8], official: &[u8], margins: &[u8], rows: usize) -> Result<usize> {
        ensure!(
            actual.len() == rows * 8 * 4
                && actual.len() == official.len()
                && margins.len() == rows * 4,
            "chain routing fixture shape"
        );
        let margins = values(margins, 4)?;
        ensure!(margins.iter().all(|v| *v >= 0.), "negative router margin");
        let mut ties = 0;
        for row in 0..rows {
            let ids = |bytes: &[u8]| {
                let mut ids: Vec<_> = bytes[row * 32..(row + 1) * 32]
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    .collect();
                ids.sort_unstable();
                ids
            };
            let (a, b) = (ids(actual), ids(official));
            ensure!(
                a.iter().all(|id| *id < 352)
                    && a.windows(2).all(|p| p[0] != p[1])
                    && b.iter().all(|id| *id < 352)
                    && b.windows(2).all(|p| p[0] != p[1]),
                "invalid selected expert ids"
            );
            if margins[row] < 1e-3 {
                ties += 1;
            } else {
                ensure!(a == b, "chain top8 mismatch outside tie row {row}");
            }
        }
        eprintln!("router near-tie rows={ties}/{rows}");
        Ok(ties)
    }

    fn request(
        layer: usize,
        rows: usize,
        dtype: ExpertV2Dtype,
        routes: Vec<ExpertProtocolV2RouteEntry>,
        wire: bytes::Bytes,
    ) -> Result<ExpertProtocolV2Request> {
        let mut request = ExpertProtocolV2Request::new_bytes(
            layer as u64 + 1,
            17,
            layer as u32,
            H as u32,
            dtype,
            (0..rows as u32)
                .map(|row| ExpertProtocolV2RowDescriptor {
                    row_id: row as u64,
                    source_kind: ExpertV2SourceKind::Prefill,
                    source_request_id: 1,
                    token_position: row as u64,
                    route_offset: row * 8,
                    route_count: 8,
                })
                .collect(),
            routes,
            wire,
        )?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        Ok(request)
    }

    fn link<'a>(device: Device<'a>, snapshot: &Path, layers: &[usize]) -> Result<RefCell<SparkLink<'a>>> {
        use crate::shared::placement::SparkAdmission;
        let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let world = SparkWorld::canonical(peers.len())?;
        ensure!(
            cuteafd_core::tp_units::spark_package_worlds("step5", "fp8").contains(&peers.len()),
            "chain gate requires a shared width-class Spark recipe"
        );
        let hosted = std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?
            .split(',').map(str::parse).collect::<std::result::Result<Vec<usize>, _>>()?;
        ensure!(hosted == layers, "chain gate hosted layer selection differs");
        let catalog = cuteafd_loader::read_expert_catalog(snapshot)?;
        let config = TcpTransportConfig::default();
        // Read architecture-independent sealed package metadata: Spark libraries
        // cannot be dlopened in this coordinator process.
        let packages = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SPARK_PACKAGES")?);
        let metadata: Value = serde_json::from_slice(&std::fs::read(
            packages.join("fp8/fp8-step5/manifest.json"),
        )?)?;
        let tp4 = &metadata["layouts"][format!("tp{}", peers.len())];
        ensure!(
            tp4["hidden"] == H
                && tp4["experts"] == 352
                && tp4["top_k"] == 8
                && tp4["intermediate"] == 1536
                && tp4["slice"] == 1536 / peers.len()
                && tp4["tp"] == peers.len(),
            "sealed Step-5 expert geometry"
        );
        let exports = crate::families::step5::serve::package_exports(&serde_json::json!({"layouts": {"tp": tp4.clone()}}))?;
        SparkAdmission::plan_original_for_test(
            snapshot,
            ExpertGeometry::STEP5,
            Some(&catalog),
            &cuteafd_loader::placement::SparkWorkingSet {
                rtx_layers: vec![],
                spark_layers: layers.to_vec(),
                spark_ranks: peers.len(),
                prefill_lanes: 1,
            },
            peers.len(),
            crate::families::step5::serve::spark_model(&exports,
                SparkEndpoint { ring_depth: None, capacity: 64, ingress: SparkIngress::Native, partial_reply_bytes: None },
                1, 1, config.max_frame_bytes)?,
        )?
        .select(&peers)?;
        let warm = request(
            layers[0],
            64,
            ExpertV2Dtype::Bf16,
            (0..64 * 8)
                .map(|i| ExpertProtocolV2RouteEntry {
                    row_index: (i / 8) as u32,
                    expert_id: (i % 352) as u32,
                    gate_weight: 0.,
                })
                .collect(),
            vec![0; 64 * H * 2].into(),
        )?;
        // The widest wave sizes every rank's rings and the exchange staging.
        let mut link = SparkLink::new(device.library, world, &peers, 64, config, H * 2)?;
        link.bootstrap(&cuteafd_transport::SparkSessionShape::from_request(&warm)?)?;
        Ok(RefCell::new(link))
    }

    fn native(count: usize) -> Result<()> {
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
        cuteafd_core::set_expert_geometry(ExpertGeometry::STEP5)
            .map_err(|_| anyhow::anyhow!("chain gate needs its own Step-5 process"))?;
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
        let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["schema"] == 1
                && manifest["seed"] == 274
                && manifest["revision"] == REVISION
                && manifest["checkpoint"] == "stepfun-ai/Step-5-Preview-BF16"
                && manifest["layers"] == serde_json::json!([0, 1, 2, 3])
                && manifest["rows"] == serde_json::json!([1, 7, 8, 9, 64])
                && manifest["source_pins"]
                    == serde_json::json!({
                        "model.py": "330a166878910b21e37f65669033a84fcaada3b58d7903fc16ef55c496cb1ea5",
                        "kernel.py": "2cd01d57d2d135397d2fd36c203684af7564b5e6c6f5d3853b3d35bedeb548be",
                        "vision_encoder.py": "8943ee555ffd2a32213f6f9f0a7146d1f49566c65c8b7691c37055208cdb6b53"
                    }),
            "chain fixture identity"
        );
        // SAFETY: this ignored gate receives a verified native artifact from the supported runner.
        let library =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        device.run(|| crate::shared::inventory::preload_native_modules(&library))?;
        let programs = library
            .programs()?
            .with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
        let checkpoint = Checkpoint::coordinator(&snapshot, false, false)?;
        let dense_hashes = manifest["weights_sha256"]
            .as_object()
            .context("chain dense source hashes")?
            .iter()
            .filter(|(name, _)| {
                (0..count.min(3)).any(|layer| name.starts_with(&format!("model.layers.{layer}.")))
            })
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<serde_json::Map<_, _>>();
        let layout = RankWeightLayout::from_checkpoint(&checkpoint, 0, 1)?;
        let dense_sources: std::collections::BTreeSet<_> = layout
            .operands
            .iter()
            .filter(|w| {
                (0..count.min(3)).any(|layer| w.name.starts_with(&format!("model.layers.{layer}.")))
            })
            .flat_map(|w| w.sources.iter().map(|s| s.meta.name.clone()))
            .collect();
        ensure!(
            dense_sources == dense_hashes.keys().cloned().collect(),
            "chain dense hash inventory"
        );
        verify_weights(&snapshot, &dense_hashes)?;
        if count > 3 {
            let identity = &manifest["same_quant"];
            ensure!(
                identity["checkpoint"] == "stepfun-ai/Step-5-Preview"
                    && identity["revision"] == "364b3b03fed2477aa0953fe855a2c4c8c48ee3cc"
                    && identity["revision"].as_str()
                        == snapshot.file_name().and_then(|s| s.to_str())
                    && identity["layer"] == 3
                    && [4, 6].iter().any(|tp| identity["tp"] == *tp)
                    && identity["slice"].as_u64() == identity["tp"].as_u64().map(|tp| 1536 / tp)
                    && identity["ingress"] == "bf16"
                    && identity["maximum_rows"] == 64
                    && identity["error_ratio_bound"] == 1.01
                    && identity["router_relative_margin"] == 1e-3,
                "chain same-quant fixture identity"
            );
            let hashes = manifest["quant_weights_sha256"]
                .as_object()
                .context("chain quant source hashes")?;
            let layout = RankWeightLayout::from_checkpoint(&checkpoint, 0, 1)?;
            let mut expected: std::collections::BTreeSet<String> = layout
                .operands
                .iter()
                .filter(|w| {
                    w.name.starts_with("model.layers.3.")
                        && (w.name.contains("post_attention_layernorm.")
                            || w.name.contains("moe.gate.")
                            || w.name.ends_with("moe.router_bias")
                            || w.name.contains("share_expert."))
                })
                .flat_map(|w| w.sources.iter().map(|s| s.meta.name.clone()))
                .collect();
            for projection in ["gate", "up", "down"] {
                for suffix in ["weight", "weight_scale_inv"] {
                    expected.insert(format!("model.layers.3.moe.{projection}_proj.{suffix}"));
                }
            }
            ensure!(
                expected == hashes.keys().cloned().collect(),
                "chain quant hash inventory"
            );
            verify_weights(&snapshot, hashes)?;
        }
        let link = if count > 3 {
            Some(device.run(|| link(device, &snapshot, &[3]))?)
        } else {
            None
        };
        let layout = super::super::cache::tests::layout_for(128, 1, 0, 64);
        let cache = RefCell::new(CacheRank::new(device, layout, 0, 1, 2, 4, usize::MAX)?);
        let staging = Staging::new(device, &programs, &cache.borrow(), usize::MAX)?;
        let family = Step5Prefix::new(vec![&cache])?;
        let pool = RefCell::new(PrefixCache::<StubCopyEngine, Frontier>::new(
            family.layout(),
            PrefixConfig {
                entries: 0,
                mark_slots: 2,
                keep_logits: false,
                min_tokens: 1,
            },
            None,
        )?);
        let mut chain = TargetChain::load(
            device,
            &programs,
            &checkpoint,
            count,
            64,
            u64::MAX,
            &[
                cache.borrow().bytes() as u64,
                staging.allocation_bytes() as u64,
            ], cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16)?;
        let input = Allocation::new(device, 64 * H * 4, crate::shared::memory::TEST)?;
        let read = |buffer, bytes| {
            let mut result = vec![0; bytes];
            device.run(|| library.copy_d2h(&mut result, buffer))?;
            Ok::<_, anyhow::Error>(result)
        };
        for rows in [1, 7, 8, 9, 64] {
            let mut placement = pool.borrow_mut()
                .admit_cold(&family, 1, 1, |pages| Placement {
                    pages,
                    sequence: 0,
                    committed: 0,
                })?
                .placement;
            let initial_pages = placement.pages.clone();
            let free = pool.borrow().pool().free();
            let data = fixture(
                &root,
                &manifest,
                &format!("r{rows}_input"),
                "torch.float32",
                &[rows, H],
            )?;
            ensure!(data.len() == rows * H * 4, "chain input fixture extent");
            device.run(|| library.copy_h2d(input.buffer, &data))?;
            let mut first = Vec::new();
            for replay in 0..2 {
                let mut step = Step::begin(
                    vec![&cache],
                    vec![&staging],
                    &pool,
                    &mut placement,
                    rows,
                )?;
                chain.enqueue_step(&mut step, 0, &input, link.as_ref().map(|link| (link, request as GateRequest)))?;
                cache.borrow().drain()?;
                crate::shared::spark_waves::healthy()?;
                // Dropping an incomplete target aborts its private proposal after drainage.
                drop(step);
                for layer in 0..count {
                    let actual = read(chain.output(layer)?, rows * H * 4)?;
                    if replay == 0 {
                        let expected = fixture(
                            &root,
                            &manifest,
                            &format!("r{rows}_layer{layer}"),
                            "torch.float32",
                            &[rows, H],
                        )?;
                        let (a, b) = (values(&actual, 4)?, values(&expected, 4)?);
                        ensure!(a.len() == b.len(), "chain hidden extent");
                        eprintln!(
                            "BF16-root report-only rows={rows} layer={layer} rel_l2={}",
                            relative(&a, &b)
                        );
                        first.push(actual);
                    } else {
                        ensure!(actual == first[layer], "chain abort/replay byte drift");
                    }
                }
                ensure!(
                    placement.committed == 0
                        && placement.pages == initial_pages
                        && pool.borrow().pool().free() == free,
                    "chain abort changed committed cache"
                );
            }
            if let Some(link) = &link {
                ensure!(
                    manifest["same_quant"]["tp"] == link.borrow().world_size()
                        && manifest["same_quant"]["error_ratio_bound"] == 1.01
                        && manifest["same_quant"]["revision"].as_str()
                            == snapshot.file_name().and_then(|s| s.to_str()),
                    "chain same-quant fixture identity"
                );
                let ffn = chain.layers[3]
                    .routed
                    .as_ref()
                    .context("chain routed owner")?;
                let load = |name: &str| {
                    let (dtype, shape) = match name {
                        "ids" => ("torch.int32", vec![rows, 8]),
                        "margin" => ("torch.float32", vec![rows]),
                        "fp64_output" => ("torch.float64", vec![rows, H]),
                        _ => ("torch.float32", vec![rows, H]),
                    };
                    fixture(
                        &root,
                        &manifest,
                        &format!("q_r{rows}_{name}"),
                        dtype,
                        &shape,
                    )
                };
                device.run(|| library.copy_h2d(ffn.input(), &load("residual")?))?;
                let mut step = Step::begin(
                    vec![&cache],
                    vec![&staging],
                    &pool,
                    &mut placement,
                    rows,
                )?;
                step.compose(0, link, |_, _| Ok(()))?;
                step.compose(0, ffn, |ffn, meta| {
                    ffn.enqueue_on(rows, link, request, meta.stream)
                })?;
                cache.borrow().drain()?;
                crate::shared::spark_waves::healthy()?;
                drop(step);
                let actual = read(ffn.output(), rows * H * 4)?;
                let official = load("wiring_output")?;
                accuracy(&actual, &official, &load("fp64_output")?, H)?;
                selection(
                    &read(ffn.route_ids(), rows * 8 * 4)?,
                    &load("ids")?,
                    &load("margin")?,
                    rows,
                )?;
                let exact = values(&load("exact_decode_output")?, 4)?;
                eprintln!("exact-decode report-only rows={rows} native_rel_l2={} materialization_rel_l2={}",
                    relative(&values(&actual,4)?,&exact),relative(&values(&official,4)?,&exact));
            }
            pool.borrow_mut().release(&family, &placement.pages)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sealed native artifacts, CUDA and pinned BF16-root fixtures"]
    fn native_step5_dense_chain() -> Result<()> {
        native(3)
    }

    #[test]
    #[ignore = "requires golden-run real TP4/TP6 experts and same-quant fixtures"]
    fn native_step5_layer_chain() -> Result<()> {
        native(4)
    }

    #[test]
    #[ignore = "requires golden-run real TP6 experts and native BF16 tail fixtures"]
    fn native_step5_bf16_tails() -> Result<()> {
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
        cuteafd_core::set_expert_geometry(ExpertGeometry::STEP5)
            .map_err(|_| anyhow::anyhow!("tail gate needs its own Step-5 process"))?;
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
        ensure!(snapshot.file_name().and_then(|s| s.to_str())
            == Some("364b3b03fed2477aa0953fe855a2c4c8c48ee3cc"), "tail checkpoint pin");
        // SAFETY: the supported runner verifies the native artifact and source seal.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        device.run(|| crate::shared::inventory::preload_native_modules(&library))?;
        let programs = library.programs()?.with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let checkpoint = Checkpoint::coordinator(&snapshot, false, false)?;
        let link = device.run(|| link(device, &snapshot, &[88, 89, 90]))?;
        ensure!(link.borrow().world_size() == 6, "BF16 tail qualification requires TP6");
        let read = |buffer, bytes| {
            let mut result = vec![0; bytes];
            device.run(|| library.copy_d2h(&mut result, buffer))?;
            Ok::<_, anyhow::Error>(result)
        };
        for layer in 88..=90 {
            let directory = root.join(format!("layer{layer}"));
            let manifest: Value = serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
            let identity = &manifest["same_quant"];
            ensure!(manifest["schema"] == 1 && manifest["seed"] == 274
                && manifest["source_pins"] == serde_json::json!({
                    "model.py": "330a166878910b21e37f65669033a84fcaada3b58d7903fc16ef55c496cb1ea5",
                    "kernel.py": "2cd01d57d2d135397d2fd36c203684af7564b5e6c6f5d3853b3d35bedeb548be",
                    "vision_encoder.py": "8943ee555ffd2a32213f6f9f0a7146d1f49566c65c8b7691c37055208cdb6b53"
                })
                && manifest["rows"] == serde_json::json!([1,7,8,9,64])
                && identity["checkpoint"] == "stepfun-ai/Step-5-Preview"
                && identity["revision"].as_str() == snapshot.file_name().and_then(|s| s.to_str())
                && identity["layer"] == layer && identity["storage"] == "bf16"
                && identity["tp"] == 6 && identity["slice"] == 256
                && identity["ingress"] == "bf16" && identity["maximum_rows"] == 64
                && identity["error_ratio_bound"] == 1.01 && identity["router_relative_margin"] == 1e-3,
                "native BF16 tail fixture identity");
            let hashes = manifest["quant_weights_sha256"].as_object().context("tail source hashes")?;
            let layout = RankWeightLayout::from_checkpoint(&checkpoint, 0, 1)?;
            let mut expected: std::collections::BTreeSet<_> = layout.operands.iter()
                .filter(|w| w.name.starts_with(&format!("model.layers.{layer}."))
                    && (w.name.contains("post_attention_layernorm.") || w.name.contains("moe.gate.")
                        || w.name.ends_with("moe.router_bias") || w.name.contains("share_expert.")))
                .flat_map(|w| w.sources.iter().map(|s| s.meta.name.clone())).collect();
            for projection in ["gate", "up", "down"] {
                expected.insert(format!("model.layers.{layer}.moe.{projection}_proj.weight"));
            }
            ensure!(expected == hashes.keys().cloned().collect(), "tail source hash inventory");
            verify_weights(&snapshot, hashes)?;
            let plan = RoutedPlan::prepare(&programs, &checkpoint, layer, 64, u64::MAX, &[])?;
            let ffn = RoutedFfn::load(device, &programs, &checkpoint, layer, plan)?;
            let stream = crate::shared::memory::device::Stream::new(device)?;
            for rows in [1,7,8,9,64] {
                let load = |name: &str| {
                    let (dtype, shape) = match name {
                        "ids" => ("torch.int32", vec![rows,8]),
                        "margin" => ("torch.float32", vec![rows]),
                        "fp64_output" => ("torch.float64", vec![rows,H]),
                        _ => ("torch.float32", vec![rows,H]),
                    };
                    fixture(&directory, &manifest, &format!("q_r{rows}_{name}"), dtype, &shape)
                };
                device.run(|| library.copy_h2d(ffn.input(), &load("residual")?))?;
                let mut first = Vec::new();
                for replay in 0..3 {
                    ffn.enqueue_on(rows, &link, request, &stream)?;
                    stream.drain()?;
                    crate::shared::spark_waves::healthy()?;
                    let actual = read(ffn.output(), rows * H * 4)?;
                    if replay == 0 {
                        accuracy(&actual, &load("wiring_output")?, &load("fp64_output")?, H)?;
                        selection(&read(ffn.route_ids(), rows * 8 * 4)?, &load("ids")?, &load("margin")?, rows)?;
                        eprintln!("native BF16 tail layer={layer} rows={rows} qualified");
                        first = actual;
                    } else { ensure!(actual == first, "native BF16 tail replay drift"); }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn same_quant_bar_checks_each_row_and_router_margin() -> Result<()> {
        let f32s = |x: &[f32]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let f64s = |x: &[f64]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let b = f32s(&[1.01, 1001.]);
        let r = f64s(&[1., 1000.]);
        accuracy(&b, &b, &r, 1)?;
        assert!(accuracy(&f32s(&[1.1, 1000.]), &b, &r, 1).is_err());
        let ids: Vec<u8> = (0..8u32).flat_map(u32::to_le_bytes).collect();
        let other: Vec<u8> = (1..9u32).flat_map(u32::to_le_bytes).collect();
        assert!(selection(&other, &ids, &f32s(&[0.1]), 1).is_err());
        assert_eq!(selection(&other, &ids, &f32s(&[0.0001]), 1)?, 1);
        assert!(selection(&ids, &ids, &f32s(&[-0.1]), 1).is_err());
        assert!(selection(&ids, &vec![0; 32], &f32s(&[0.0001]), 1).is_err());
        assert!(values(&f32s(&[f32::NAN]), 4).is_err());
        Ok(())
    }

    #[test]
    fn fixture_rejects_wrong_shape_dtype_length_and_hash() -> Result<()> {
        let root = tempfile::tempdir()?;
        let bytes = 1f32.to_le_bytes();
        std::fs::write(root.path().join("input.bin"), bytes)?;
        let mut manifest = serde_json::json!({"files": {"input": {
            "dtype": "torch.float32", "shape": [1, 1], "bytes": 4,
            "sha256": format!("{:x}", Sha256::digest(bytes))
        }}});
        assert_eq!(
            fixture(root.path(), &manifest, "input", "torch.float32", &[1, 1])?,
            bytes
        );
        assert!(fixture(root.path(), &manifest, "input", "torch.float32", &[1]).is_err());
        assert!(fixture(root.path(), &manifest, "input", "torch.int32", &[1, 1]).is_err());
        manifest["files"]["input"]["bytes"] = serde_json::json!(8);
        assert!(fixture(root.path(), &manifest, "input", "torch.float32", &[1, 1]).is_err());
        manifest["files"]["input"]["bytes"] = serde_json::json!(4);
        std::fs::write(root.path().join("input.bin"), 2f32.to_le_bytes())?;
        assert!(fixture(root.path(), &manifest, "input", "torch.float32", &[1, 1]).is_err());
        Ok(())
    }
}
