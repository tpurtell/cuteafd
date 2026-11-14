//! Retained role-specific serving owners. The shared scheduler is attached separately.
use super::{cache::{self, RequestState, RequestStreams}, graphs::{self, Graphs, Op},
    head::Head, layer::{Kernels, UnitOwner}, prefix::MiniMaxPrefix, weights::TargetWeights};
use crate::shared::{decode_graph::device::Basis,
    memory::receipts::{Census, Inventory}, token_io::{EmbedPlacement, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Category;
use cuteafd_ffi::{programs::Programs, expert_usage::{ExpertUsage, UsageDevice, UsageStorage}};
use cuteafd_loader::{families::minimax_m3::weight_layout::CoordinatorWeights,
    placement::{families::minimax_m3::{AdmittedMiniMax, MiniMaxInputs, Owner},
        planner::{Component, ComponentKind, CoordinatorDevices, Facts, Lifetime, Location, Recipe}}};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

pub(crate) struct LoadedRank<'a> {
    pub weights: TargetWeights<'a>,
    pub streams: RequestStreams<'a>,
}

pub(crate) struct Rank<'a> {
    pub weights: TargetWeights<'a>,
    pub layout: CoordinatorWeights,
    pub kernels: Kernels<'a>,
    pub decode: RefCell<UnitOwner<'a>>,
    pub prefill: Vec<RefCell<UnitOwner<'a>>>,
}

/// Graphs retain this entire bundle, not just the buffers in one captured site.
/// Mutable request/slot borrows are synchronous and never cross an await.
pub(crate) struct Storage<'a> {
    pub ranks: Vec<Rank<'a>>,
    pub requests: RefCell<RequestState<'a>>,
    pub prefix: MiniMaxPrefix<'a>,
    pub head: Head<'a>,
    pub embedding: TokenEmbedding<'a>,
    pub usage: Option<ExpertUsage<'a>>,
    devices: Vec<i32>,
}
impl Drop for Storage<'_> {
    fn drop(&mut self) {
        for rank in &self.ranks {
            rank.decode.borrow().stream.drain_or_abort("MiniMax serving storage retirement");
            for unit in &rank.prefill {
                unit.borrow().stream.drain_or_abort("MiniMax serving storage retirement");
            }
        }
    }
}

/// Graph fields retire before the final storage pin. Attention stays eager:
/// its context-dependent launch scalars are not part of a decode graph key.
pub(crate) struct Owners<'a> {
    pub graphs: Vec<Graphs<'a, Storage<'a>>>,
    pub storage: Rc<Storage<'a>>,
    selector_contract: SelectorContract,
}

pub(crate) struct SelectorContract {
    owners: Vec<cuteafd_loader::placement::planner::Allocation>,
}
impl SelectorContract {
    fn from_plan(plan: &cuteafd_loader::placement::planner::ResolvedPlan) -> Result<Self> {
        let owners = plan.facts.allocations.iter().filter(|owner| owner.id.starts_with("selector.")).cloned().collect::<Vec<_>>();
        ensure!(!owners.is_empty(), "MiniMax admitted selector owner inventory missing");
        Ok(Self { owners })
    }

    pub fn reconcile(&self, receipts: &[cuteafd_loader::placement::planner::Receipt]) -> Result<()> {
        let actual = receipts.iter().map(|receipt| (receipt.allocation.id.as_str(), receipt)).collect::<BTreeMap<_, _>>();
        ensure!(actual.len() == receipts.len() && actual.len() == self.owners.len(),
            "MiniMax scheduler selector owner inventory differs from admission");
        let mut spans = Vec::new();
        for planned in &self.owners {
            let receipt = actual.get(planned.id.as_str()).with_context(|| format!("MiniMax selector missing {}", planned.id))?;
            let owner = &receipt.allocation;
            // The shared Inventory's diagnostic group/category may differ;
            // physical identity, payload, location and lifetime must not.
            ensure!(owner.view.is_none() && planned.view.is_none() && owner.bytes == planned.bytes
                && owner.location == planned.location && owner.lifetime == planned.lifetime && receipt.address != 0,
                "MiniMax selector {} differs from admitted owner", planned.id);
            let end = receipt.address.checked_add(owner.bytes).context("MiniMax selector address overflow")?;
            spans.push((owner.location, receipt.address, end));
        }
        spans.sort_unstable();
        ensure!(spans.windows(2).all(|pair| pair[0].0 != pair[1].0 || pair[0].2 <= pair[1].1),
            "MiniMax selector physical owners overlap");
        Ok(())
    }
}

impl<'a> Owners<'a> {
    pub fn new(inputs: &MiniMaxInputs, admitted: &AdmittedMiniMax,
        programs: &'a Programs<'a>, loaded: Vec<LoadedRank<'a>>, embedding: TokenEmbedding<'a>) -> Result<Self> {
        Self::with_usage_reader(inputs, admitted, programs, loaded, embedding, Default::default())
    }

    pub fn with_usage_reader(inputs: &MiniMaxInputs, admitted: &AdmittedMiniMax,
        programs: &'a Programs<'a>, loaded: Vec<LoadedRank<'a>>, embedding: TokenEmbedding<'a>,
        usage_reader: cuteafd_core::expert_usage::Reader) -> Result<Self> {
        let capacity = inputs.request_capacity()?;
        let contract = inputs.prepare(&admitted.solved.request)?;
        ensure!(admitted.solved.pool_page_rows == contract.page_rows
            && admitted.solved.pool_tokens % contract.page_rows == 0,
            "MiniMax serving pool differs from its prepared contract");
        let pages = admitted.solved.pool_tokens / contract.page_rows;
        let waves = usize::try_from(inputs.prefill.waves)?;
        ensure!(matches!(loaded.len(), 1 | 2) && loaded.len() == inputs.gpus
            && inputs.targets.len() == loaded.len() && waves > 0
            && admitted.solved.decisions.prefill == inputs.prefill
            && admitted.solved.request.context.maximum == inputs.context.maximum
            && embedding.placement() == EmbedPlacement::Gpu,
            "MiniMax serving constructor differs from admitted rank/unit/context or embedding placement");
        if let Some(geometry) = &inputs.expert_usage {
            ensure!(geometry.validate() && geometry.layers == inputs.cfg.layers
                && geometry.experts == inputs.cfg.experts && geometry.topk == inputs.cfg.topk
                && geometry.routed_layers == inputs.routed,
                "MiniMax expert usage differs from admitted router geometry");
        }
        let devices = loaded.iter().map(|rank| rank.streams.decode.device.id).collect::<Vec<_>>();
        let mapping = CoordinatorDevices::new(&devices)?;
        let library = programs.library();
        let mut weight_inventory = Inventory::new(mapping, "minimax", Lifetime::RESIDENT,
            Category::Weights, "checkpoint-native target");
        for (gpu, rank) in loaded.iter().enumerate() {
            let target = &inputs.targets[gpu];
            let device = rank.streams.decode.device;
            ensure!(target.rank == gpu && rank.weights.rank == gpu && rank.weights.layers == target.layers
                && std::ptr::eq(device.library, library) && rank.streams.prefill.len() == waves
                && rank.streams.prefill.iter().all(|stream| stream.device.id == device.id
                    && std::ptr::eq(stream.device.library, library)),
                "MiniMax loaded rank {gpu} differs from admitted weights/streams");
            rank.weights.inventory(&mut weight_inventory, &format!("target.{gpu}"))?;
        }
        embedding.validate_residency(devices[0], inputs.cfg.vocab_size as u64 * inputs.cfg.hidden as u64 * 2)?;
        embedding.inventory(&mut weight_inventory, "embedding", Lifetime::RESIDENT)?;
        let mut weights = Census::default();
        weights.owners(weight_inventory.receipts);
        weights.normalize(admitted.solved.facts.phase_order)?;
        reconcile(&weights, &admitted.solved.facts, &contract.components, pages,
            |id| id.starts_with("minimax/target/"))?;

        let mut ranks = Vec::with_capacity(loaded.len());
        let mut streams = Vec::with_capacity(loaded.len());
        for (gpu, rank) in loaded.into_iter().enumerate() {
            let layout = inputs.targets[gpu].clone();
            let bind = if inputs.split() { Kernels::bind_split } else { Kernels::bind };
            let kernels = bind(rank.streams.decode.device, programs, &inputs.manifest,
                &inputs.cfg, &layout, capacity.execution_rows(), u32::try_from(inputs.context.maximum)?)?;
            let make_unit = |stream: Rc<_>, rows, prefill| -> Result<_> {
                if inputs.split() {
                    let roles = (0..inputs.cfg.layers).map(|layer| inputs.roles(layer, gpu, rows))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    ensure!(roles.iter().all(|roles| roles.iter().any(|role|
                        role.role == cuteafd_loader::families::minimax_m3::execution::ExecutionRole::PartialFfn)),
                        "MiniMax split rank lacks admitted partial role");
                    UnitOwner::new_split(&kernels, stream, &inputs.cfg, &layout, rows,
                        inputs.split_wire.context("MiniMax split wire not selected")?, prefill,
                        gpu == 1 && admitted.onboard > 0)
                } else if prefill { UnitOwner::new(&kernels, stream, &inputs.cfg, &layout, rows) }
                else { UnitOwner::new_decode(&kernels, stream, &inputs.cfg, &layout, rows) }
            };
            let decode = RefCell::new(make_unit(Rc::clone(&rank.streams.decode), capacity.decode.rows, false)?);
            let prefill = rank.streams.prefill.iter().map(|stream|
                make_unit(Rc::clone(stream), capacity.prefill.rows, true).map(RefCell::new))
                .collect::<Result<Vec<_>>>()?;
            if gpu == 0 {
                for unit in &prefill {
                    let unit = unit.borrow();
                    let ids = cache::view(unit.plane("route_ids")?, 0, capacity.prefill.rows as usize * 4)?;
                    unit.stream.device.run(|| {
                        // SAFETY: each full-capacity U32 destination is a stable
                        // unit-owned view; no work is queued during construction.
                        unsafe { library.reserve_h2d_ordered(ids, unit.stream.raw) }
                    })?;
                }
            }
            streams.push(rank.streams);
            ranks.push(Rank { weights: rank.weights, layout, kernels, decode, prefill });
        }
        let requests = RequestState::new(&inputs.cfg, capacity, admitted, streams)?;
        let prefix = MiniMaxPrefix::new(requests.banks(), ranks[0].prefill[0].borrow().prefix_scratch())?;
        let head_stream = Rc::clone(&ranks.last().context("MiniMax head rank")?.decode.borrow().stream);
        let head = Head::new(head_stream, programs, &inputs.cfg, capacity.decode.rows as usize)?;
        // The shared Qwen scheduler constructs and owns the one admitted
        // selector. Retaining a second one here would double-charge storage.
        let usage = inputs.expert_usage.as_ref().map(|geometry| {
            let producers = inputs.usage_devices().into_iter()
                .map(|gpu| Ok(UsageDevice { logical: gpu, cuda: devices[usize::from(gpu)] }))
                .collect::<Result<Vec<_>>>()?;
            ExpertUsage::new(library, geometry.clone(), &producers, usage_reader.clone())
        }).transpose()?;
        let storage = Rc::new(Storage { ranks, requests: RefCell::new(requests), prefix, head,
            embedding, usage, devices });
        let census = storage.inventory()?;
        reconcile(&census, &admitted.solved.facts, &contract.components, pages, retained_owner)?;
        let mut graphs = Vec::with_capacity(inputs.gpus);
        for (gpu, rank) in storage.ranks.iter().enumerate() {
            let layers = if inputs.split() { (0..inputs.cfg.layers).collect::<Vec<_>>() }
                else { rank.layout.layers.clone() };
            let sites = layers.iter().flat_map(|&layer| {
                let mut sites = vec![(layer, Op::Ffn)];
                if inputs.cfg.moe[layer] {
                    sites.push((layer, Op::JoinRouted));
                    if !inputs.split() || gpu == 0 { sites.push((layer, Op::Router)); }
                }
                sites
            }).collect::<Vec<_>>();
            graphs.push(Graphs::new(Rc::clone(&rank.decode.borrow().stream), Rc::clone(&storage),
                gpu, inputs.cfg.layers, capacity.decode.rows as usize, &sites, graphs::Config {
                    reserve: inputs.graph_reserve[gpu], bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: false })?);
        }
        Ok(Self { graphs, storage, selector_contract: SelectorContract::from_plan(&admitted.solved)? })
    }
}

impl Owners<'_> {
    pub fn reconcile_selector(&self, selector: &crate::shared::token_io::TokenSelector<'_>)
        -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        let device_ids = self.storage.ranks.iter().map(|rank| rank.decode.borrow().stream.device.id).collect::<Vec<_>>();
        let mut inventory = Inventory::new(CoordinatorDevices::new(&device_ids)?, "selector",
            cuteafd_loader::placement::planner::serving::ServingOwners::LIFETIME,
            Category::Workspace, "token selector");
        // Inventory requires a nonempty suffix; strip only that diagnostic
        // namespace after observing the selector's actual backings.
        selector.allocation_receipts(&mut inventory, "next")?;
        for receipt in &mut inventory.receipts {
            receipt.allocation.id = receipt.allocation.id.replacen("selector.next.", "selector.", 1);
        }
        self.selector_contract.reconcile(&inventory.receipts)?;
        Ok(inventory.receipts)
    }
}

impl Storage<'_> {
    pub fn inventory(&self) -> Result<Census> {
        let devices = CoordinatorDevices::new(&self.devices)?;
        let mut weights = Inventory::new(devices, "minimax", Lifetime::RESIDENT,
            Category::Weights, "checkpoint-native target");
        let mut execution = Inventory::new(devices, "minimax", Lifetime::TARGET,
            Category::Workspace, "retained serving execution");
        for (gpu, rank) in self.ranks.iter().enumerate() {
            rank.weights.inventory(&mut weights, &format!("target.{gpu}"))?;
            rank.decode.borrow().inventory(&mut execution, &Owner::Decode { gpu }.inventory_name().unwrap())?;
            for (slot, unit) in rank.prefill.iter().enumerate() {
                let unit = unit.borrow();
                let name = Owner::Unit { gpu, slot: slot as u64 }.inventory_name().unwrap();
                unit.inventory(&mut execution, &name)?;
                if gpu == 0 {
                    let rows = self.requests.borrow().capacity.prefill.rows as usize;
                    let ids = cache::view(unit.plane("route_ids")?, 0, rows * 4)?;
                    let upload = unit.stream.device.run(|| unit.stream.device.library
                        .ordered_h2d_staging_buffer(ids, unit.stream.raw))?.context("MiniMax token staging missing")?;
                    ensure!(upload.bytes == ids.bytes, "MiniMax token staging differs from admitted unit rows");
                    execution.pinned(&format!("{name}.route_ids.upload"), upload)?;
                }
            }
        }
        self.embedding.inventory(&mut weights, "embedding", Lifetime::RESIDENT)?;
        self.requests.borrow().inventory(&mut execution)?;
        self.prefix.upload_inventory(&mut execution)?;
        self.head.inventory(&mut execution, &Owner::Head.inventory_name().unwrap())?;
        let mut census = Census::default();
        census.owners(weights.receipts);
        census.owners(execution.receipts);
        if let Some(usage) = &self.usage {
            let mut observed = Inventory::new(devices, "expert-usage", Lifetime::TARGET,
                Category::Workspace, "retained expert telemetry");
            for receipt in usage.allocation_receipts() {
                ensure!(devices.location(receipt.device.cuda)? == Location::coordinator(receipt.device.logical),
                    "MiniMax expert usage CUDA/logical device mapping differs");
                let name = receipt.owner.strip_prefix("expert-usage.")
                    .context("MiniMax expert usage receipt namespace")?;
                match receipt.storage {
                    UsageStorage::Device(buffer) => observed.device(name, buffer)?,
                    UsageStorage::Pinned(buffer) => observed.pinned(name, buffer)?,
                }
            }
            census.owners(observed.receipts);
        }
        census.normalize(cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT)?;
        Ok(census)
    }
}

/// This is deliberately not a complete serving reconciliation: the scheduler
/// must also reconcile its own canonically charged selector (without allocating
/// another here), attach transport/experts/draft/media/host-cache owners, account
/// for expert-usage host history/worker metadata, and reconcile startup/driver/graph
/// allowances before signalling readiness.
fn retained_owner(id: &str) -> bool {
    ["minimax/target/", "minimax.kv.", "minimax.cache.", "minimax.rope.", "minimax.decode.",
        "minimax.unit.", "minimax.prefix.h2d.", "minimax.head.", "expert-usage.gpu"]
        .iter().any(|prefix| id.starts_with(prefix))
}

fn unique_curve<'b>(matches: &[&'b Component]) -> Result<&'b cuteafd_loader::placement::planner::ByteCurve> {
    let first = matches.first().context("MiniMax retained owner needs a canonical allocation curve")?;
    for (index, component) in matches.iter().enumerate() {
        ensure!(component.bytes == first.bytes, "MiniMax repeated owner has unequal payload curves");
        for other in &matches[..index] {
            let exclusive = component.activation.clauses.iter().all(|left|
                other.activation.clauses.iter().all(|right|
                    left.iter().any(|a| right.iter().any(|b| a.variable == b.variable && a.option != b.option))));
            ensure!(exclusive, "MiniMax repeated owner has overlapping activations");
        }
    }
    Ok(&first.bytes)
}

fn reconcile(census: &Census, facts: &Facts, components: &[Component], pages: u64,
    selected: impl Fn(&str) -> bool) -> Result<()> {
    let actual = census.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).map(|receipt|
        (receipt.allocation.id.as_str(), receipt)).collect::<BTreeMap<_, _>>();
    let planned = facts.allocations.iter().filter(|owner| owner.view.is_none() && selected(&owner.id))
        .map(|owner| (owner.id.as_str(), owner)).collect::<BTreeMap<_, _>>();
    ensure!(actual.len() == census.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).count()
        && planned.len() == facts.allocations.iter().filter(|owner| owner.view.is_none() && selected(&owner.id)).count(),
        "MiniMax retained physical owner identity is duplicated");
    ensure!(actual.len() == planned.len(), "MiniMax retained physical owner count differs: actual {}, admitted {}; missing {:?}; extra {:?}",
        actual.len(), planned.len(), planned.keys().filter(|id| !actual.contains_key(*id)).collect::<Vec<_>>(),
        actual.keys().filter(|id| !planned.contains_key(*id)).collect::<Vec<_>>());
    for (id, receipt) in actual {
        let owner = planned.get(id).with_context(|| format!("MiniMax retained owner not admitted: {id}"))?;
        let actual = &receipt.allocation;
        let matching = components.iter().filter(|component| matches!(&component.kind,
            ComponentKind::Allocation { template, .. } if template.id == id)).collect::<Vec<_>>();
        let curve = unique_curve(&matching).with_context(|| format!("MiniMax retained owner {id}"))?;
        let piece = curve.0.iter().find(|piece| (piece.first..=piece.last).contains(&pages))
            .context("MiniMax retained owner is outside its curve domain")?;
        let payload = pages.checked_sub(piece.first).and_then(|pages| piece.bytes_per_page.checked_mul(pages))
            .and_then(|bytes| piece.bytes_at_first.checked_add(bytes)).context("MiniMax owner payload overflow")?;
        // Rounded curves charge backing pages directly; raw curves leave tails
        // to their own owner. Evaluate the canonical curve, never guess a unit.
        ensure!(piece.granularity > 0 && piece.backings == 1 && receipt.address != 0
            && actual.bytes == payload && curve.at(pages)? == owner.bytes && actual.location == owner.location
            && actual.lifetime == owner.lifetime, "MiniMax retained owner {id} differs from admission: actual {actual:?}; planned {owner:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{experts::rtx::dense::recording, memory::device::{Device, Stream}};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::{families::minimax_m3::MiniMaxM3Config,
        placement::{families::minimax_m3::{self as canonical, ExpertBacking, LocalPackage, SparkCharge},
            planner::{serving, Context as ModelContext, Operation, PrefillShape}}, plan::{Checkpoint, testing}};
    use serde_json::{json, Value};

    #[test]
    fn selector_receipts_are_exact_and_missing_sampler_is_not_accepted() -> Result<()> {
        use cuteafd_loader::placement::planner::{Allocation, Receipt};
        let owner = Allocation::buffer("selector.output", Location::coordinator(0), 192,
            serving::ServingOwners::LIFETIME, Category::Workspace, "token selector");
        let sampler = Allocation::buffer("selector.sampler.rows", Location::coordinator(0), 512,
            serving::ServingOwners::LIFETIME, Category::Workspace, "target sampler");
        let contract = SelectorContract { owners: vec![owner.clone(), sampler.clone()] };
        let receipts = vec![Receipt { allocation: owner, address: 4096 },
            Receipt { allocation: sampler, address: 8192 }];
        contract.reconcile(&receipts)?;
        assert!(contract.reconcile(&receipts[..1]).is_err());
        let mut changed = receipts.clone();
        changed[1].allocation.bytes += 1;
        assert!(contract.reconcile(&changed).is_err());
        changed = receipts.clone();
        changed[1].address = 4100;
        assert!(contract.reconcile(&changed).is_err());
        changed = receipts.clone();
        changed[1].allocation.location = Location::coordinator(1);
        assert!(contract.reconcile(&changed).is_err());
        changed = receipts.clone();
        changed.push(receipts[0].clone());
        assert!(contract.reconcile(&changed).is_err());
        Ok(())
    }

    // Sparse native checkpoint headers skip payload IO under dense recording.
    // Production loading, allocation and receipt observation still run unchanged.
    fn snapshot(dir: &std::path::Path) -> Result<(Checkpoint, MiniMaxM3Config)> {
        let mut config: Value = serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json"))?;
        config.as_object_mut().unwrap().remove("vision_config");
        config["text_config"]["num_hidden_layers"] = json!(5);
        config["text_config"]["moe_layer_freq"].as_array_mut().unwrap().truncate(5);
        config["text_config"]["moe_layer_freq"][2] = json!(1);
        for key in ["sparse_attention_freq", "sparse_disable_index_value"] {
            config["text_config"]["sparse_attention_config"][key].as_array_mut().unwrap().truncate(5);
        }
        let cfg = MiniMaxM3Config::from_hf(&config)?;
        let mut tensors = vec![("language_model.model.embed_tokens.weight".into(), "BF16", vec![200064, 6144]),
            ("language_model.lm_head.weight".into(), "BF16", vec![200064, 6144]),
            ("language_model.model.norm.weight".into(), "BF16", vec![6144])];
        for layer in 0..5 {
            let p = format!("language_model.model.layers.{layer}.");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                tensors.push((format!("{p}{norm}.weight"), "BF16", vec![6144]));
            }
            let mut matrix = |stem: String, n, k| {
                tensors.push((format!("{stem}.weight"), "F8_E4M3", vec![n, k]));
                tensors.push((format!("{stem}.weight_scale_inv"), "U8", vec![n, k / 32]));
            };
            for (name, n, k) in [("q_proj", 8192, 6144), ("k_proj", 512, 6144), ("v_proj", 512, 6144),
                ("o_proj", 6144, 8192), ("index_q_proj", 512, 6144), ("index_k_proj", 128, 6144)] {
                matrix(format!("{p}self_attn.{name}"), n, k);
            }
            let (stem, width) = if cfg.moe[layer] { ("block_sparse_moe.shared_experts", 3072) } else { ("mlp", 12288) };
            for (name, n, k) in [("gate", width, 6144), ("up", width, 6144), ("down", 6144, width)] {
                matrix(format!("{p}{stem}.{name}_proj"), n, k);
            }
            for norm in ["q_norm", "k_norm", "index_q_norm", "index_k_norm"] {
                tensors.push((format!("{p}self_attn.{norm}.weight"), "BF16", vec![128]));
            }
            if cfg.moe[layer] {
                tensors.push((format!("{p}block_sparse_moe.gate.weight"), "F32", vec![128, 6144]));
                tensors.push((format!("{p}block_sparse_moe.e_score_correction_bias"), "F32", vec![128]));
                for expert in 0..128 {
                    for (name, n, k) in [("w1", 3072, 6144), ("w3", 3072, 6144), ("w2", 6144, 3072)] {
                        let stem = format!("{p}block_sparse_moe.experts.{expert}.{name}");
                        tensors.push((format!("{stem}.weight_packed"), "U8", vec![n, k / 2]));
                        tensors.push((format!("{stem}.weight_scale"), "F8_E4M3", vec![n, k / 16]));
                        for scale in ["weight_global_scale", "input_global_scale"] {
                            tensors.push((format!("{stem}.{scale}"), "F32", vec![]));
                        }
                    }
                }
            }
        }
        testing::write_snapshot(dir, &config, &tensors, None);
        Ok((Checkpoint::open(dir)?, cfg))
    }

    fn manifest(targets: &[CoordinatorWeights]) -> Result<Value> {
        let mut entries = BTreeMap::new();
        for text in [include_str!("../../../../cuteafd-ffi/tests/fixtures/minimax_msa_programs.json"),
            include_str!("../../../../cuteafd-ffi/tests/fixtures/minimax_producer_programs.json"),
            include_str!("../../../../cuteafd-ffi/tests/fixtures/minimax_forward_programs.json")] {
            for entry in serde_json::from_str::<Value>(text)?["programs"].as_array().unwrap() {
                entries.insert(entry["name"].as_str().unwrap().to_owned(), entry.clone());
            }
        }
        for (whole, half) in [(3072, 1536), (12288, 6144)] {
            let source = &entries[&format!("minimax_oai_split_i{whole}")];
            let entry: Value = serde_json::from_str(&source.to_string().replace(&whole.to_string(), &half.to_string()))?;
            entries.insert(format!("minimax_oai_split_i{half}"), entry);
        }
        let scalars = json!([{"name":"rows","type":"int32"},
            {"name":"rotation_id","type":"int32"}, {"name":"cta_limit","type":"int32"}]);
        for projection in targets.iter().flat_map(|target| &target.projections) {
            let (k, n) = (projection.input, projection.output);
            let name = format!("exlr8_dense_quant_k{k}_bf16_a8");
            entries.insert(name.clone(), json!({"name":name,"op":"dense_mxfp8_quant",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":k,"input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"out"}]}));
            let name = projection.program_name();
            entries.insert(name.clone(), json!({"name":name,"op":"dense_mxfp8",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":k,"output_features":n,"sharded_side":1,"encoding":"mxfp8",
                    "input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"payload","dtype":"uint8","shape":format!("[{}]", k*n),"role":"in"},
                    {"name":"scales","dtype":"uint8","shape":format!("[{}]", n*k/32),"role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"in"},
                    {"name":"output","dtype":"float32","shape":format!("[rows,{n}]"),"role":"out"}]}));
        }
        Ok(json!({"programs":entries.into_values().collect::<Vec<_>>()}))
    }

    fn inputs(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, ids: &[i32], concurrency: u64) -> Result<MiniMaxInputs> {
        let gpus = ids.len();
        let namespace_error = cuteafd_loader::push_load::namespace(&checkpoint.snapshot)
            .expect_err("constructor fixture must use mount loading without a push namespace");
        ensure!(namespace_error.kind() == std::io::ErrorKind::NotFound
            && namespace_error.raw_os_error().is_none() && checkpoint.snapshot.exists(),
            "constructor fixture did not resolve the canonical mount route: {namespace_error}");
        let owners = canonical::layer_owners(cfg.layers, gpus);
        let split = if gpus == 2 { Some(cuteafd_loader::families::minimax_m3::weight_layout::SplitFfnWeights::plan(
            checkpoint, cfg, &owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>(),
            cuteafd_loader::placement::planner::EmbeddingHome::Device(0))?) } else { None };
        let targets = match &split {
            Some(split) => split.iter().map(|w| w.target.clone()).collect(),
            None => CoordinatorWeights::plan(checkpoint, cfg, gpus,
                &owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>())?,
        };
        let routed = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
        let catalog = cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?;
        let tensors = catalog.fp8().context("MiniMax fixture native expert catalog")?;
        let expert_backings = (0..=routed.len()).map(|count| if count == 0 { Ok(vec![]) } else {
            (0..gpus).map(|gpu| ExpertBacking::plan(tensors, &routed[..count], gpus, gpu, ids[gpu])).collect()
        }).collect::<Result<Vec<_>>>()?;
        let prefill = PrefillShape { rows: 128, waves: 3 };
        let budgets = (0..gpus).map(|gpu| canonical::planned_budget(gpu as u8, 96 << 30, 96 << 30, 1 << 30))
            .chain((0..2).map(|rank| canonical::spark_budget(rank, 100 << 30)))
            .chain([cuteafd_loader::placement::families::kimi::host_budget(512 << 30, 480 << 30)]).collect();
        Ok(MiniMaxInputs { cfg: cfg.clone(), gpus, split_wire: Some(cuteafd_loader::families::minimax_m3::execution::WirePrecision::Bf16),
            attention: canonical::MINIMAX_M3.check_attention(None, gpus, gpus == 2)?,
            spark_ranks: 2, context: ModelContext::pinned(257),
            maximum_pool_tokens: 256 * 64, prefill, sequences: concurrency, manifest: std::sync::Arc::new(manifest(&targets)?),
            routed: routed.clone(), expert_backings, owners, budgets,
            local: LocalPackage::Installed { scratch: 64 << 20, output: prefill.rows * 6144 * 2 },
            spark: (0..=routed.len()).map(|count| vec![if count == routed.len() {
                SparkCharge { experts: 0, owners: 0, runtime: 0 }
            } else { SparkCharge { experts: 1 << 30, owners: 1 << 30, runtime: 1 << 30 } }; 2]).collect(),
            links: canonical::spark_link(0, 2, prefill, Default::default())?,
            spark_reply_formats: Default::default(), drafter: None, vision: None, code: vec![64 << 20; gpus],
            graph_reserve: vec![500 << 20; gpus], head: serving::BlasHead::Gemm,
            host_prefix: serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }, device_sampling: true,
            expert_usage: None, resident_weight_limit: None,
            push_host_envelopes: vec![None; routed.len() + 1],
            mount: match split { Some(split) => split.iter().map(|w| w.mount_load(checkpoint)).collect::<Result<_>>()?,
                None => targets.iter().map(|target| target.mount_load(checkpoint)).collect::<Result<_>>()? }, targets })
    }

    #[test]
    fn repeated_owner_curves_require_identical_curves_and_exclusive_activations() -> Result<()> {
        use cuteafd_loader::placement::planner::{Activation, Allocation, ByteCurve, Choice};
        let owner = Allocation::buffer("receive", Location::coordinator(1), 64,
            Lifetime::TARGET, Category::Transport, "route receive");
        let mut a = Component::allocation(owner, ByteCurve::fixed(64, 16), false, 16);
        a.activation = Activation::choice(0, 1);
        let mut b = a.clone();
        b.activation = Activation::choice(0, 2);
        unique_curve(&[&a, &b])?;
        b.activation = a.activation.clone();
        assert!(unique_curve(&[&a, &b]).is_err());
        b.activation = Activation::always();
        assert!(unique_curve(&[&a, &b]).is_err());
        b.activation = Activation { clauses: vec![vec![Choice { variable: 0, option: 2 }],
            vec![Choice { variable: 1, option: 0 }]] };
        assert!(unique_curve(&[&a, &b]).is_err(), "every DNF clause must exclude every peer clause");
        b.activation = Activation::choice(0, 2);
        b.bytes = ByteCurve::fixed(65, 16);
        assert!(unique_curve(&[&a, &b]).is_err());
        b.bytes = ByteCurve::fixed(64, 15);
        assert!(unique_curve(&[&a, &b]).is_err(), "equal selected-point bytes do not imply equal curves");
        assert!(unique_curve(&[]).is_err());
        Ok(())
    }

    #[test]
    fn ownership_fixture_bf16_add_binding_fails_closed() -> Result<()> {
        use crate::shared::memory::device::Allocation;
        use cuteafd_core::memory_layout::Tag;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let stream = Stream::new(device)?;
        let a = Allocation::new(device, 2, Tag::workspace("minimax/test/add-a"))?;
        let b = Allocation::new(device, 2, Tag::workspace("minimax/test/add-b"))?;
        let out = Allocation::new(device, 2, Tag::workspace("minimax/test/add-out"))?;
        let result = device.run(|| {
            // SAFETY: real disjoint retained fixture allocations on this device.
            unsafe { library.bf16_add()?.launch(a.buffer, b.buffer, out.buffer, 1, stream.raw) }
        });
        assert!(result.unwrap_err().to_string().contains("CUDA status 1"));
        Ok(())
    }

    #[test]
    fn retained_constructor_matches_canonical_recipe_on_one_two_rtx_and_boot_concurrency() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.extent_only(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        fixture.strict_device_free(&library, true)?;
        for id in 0..2 { fixture.configure_memory(&library, id, 96 << 30)?; }
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let (checkpoint, cfg) = snapshot(dir.path())?;
        for ids in [vec![1], vec![1, 0]] {
            for concurrency in [1, 16, 64] {
                for (device_sampling, expert_usage) in [(false, false), (true, false), (false, true), (true, true)] {
                for (wire, onboard) in if ids.len() == 2 {
                    vec![(crate::families::minimax_m3::dense::PartialPrecision::Bf16, 0), (crate::families::minimax_m3::dense::PartialPrecision::F32, 0),
                        (crate::families::minimax_m3::dense::PartialPrecision::Bf16, 1), (crate::families::minimax_m3::dense::PartialPrecision::F32, 1),
                        (crate::families::minimax_m3::dense::PartialPrecision::Bf16, 2), (crate::families::minimax_m3::dense::PartialPrecision::F32, 2)]
                } else { vec![(crate::families::minimax_m3::dense::PartialPrecision::Bf16, 0)] } {
                let mut inputs = inputs(&checkpoint, &cfg, &ids, concurrency)?;
                inputs.split_wire = Some(wire);
                inputs.device_sampling = device_sampling;
                inputs.expert_usage = expert_usage.then(|| cuteafd_core::expert_usage::Geometry {
                    layers: cfg.layers, experts: cfg.experts, topk: cfg.topk, routed_layers: inputs.routed.clone(),
                });
                let request = canonical::request(inputs.context, Some(129 * 64), cuteafd_loader::placement::Onboard::Layers(onboard),
                    inputs.prefill, Operation::Serving, None, Default::default());
                let admitted = inputs.admit(request).with_context(|| format!("{} RTX C{concurrency} wire{wire:?} onboard{onboard}", ids.len()))?;
                assert_eq!(admitted.solved.pool_tokens, 129 * 64);
                let contract = inputs.prepare(&admitted.solved.request)?;
                fixture.register_programs(&library, inputs.manifest["programs"].as_array().unwrap())?;
                let path = dir.path().join("PROGRAMS.json");
                std::fs::write(&path, serde_json::to_vec(&*inputs.manifest)?)?;
                let programs = library.programs()?.with_manifest(&path)?;
                let split = if ids.len() == 2 { Some(cuteafd_loader::families::minimax_m3::weight_layout::SplitFfnWeights::plan(
                    &checkpoint, &cfg, &inputs.owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>(),
                    cuteafd_loader::placement::planner::EmbeddingHome::Device(0))?) } else { None };
                let mut loaded = Vec::new();
                let mut embedding = None;
                for (gpu, &id) in ids.iter().enumerate() {
                    let device = Device { library: &library, id };
                    let target = &inputs.targets[gpu];
                    let weights = if gpu == 0 {
                        let ((weights, table), records) = recording(||
                            match &split {
                                Some(split) => TargetWeights::load_split_ffn_with_embedding(device, &checkpoint, &split[gpu], EmbedPlacement::Gpu),
                                None => TargetWeights::load_with_embedding(device, &checkpoint, target, EmbedPlacement::Gpu),
                            })?;
                        assert!(!records.operands.is_empty());
                        embedding = Some(table);
                        weights
                    } else {
                        let (weights, records) = recording(|| match &split {
                            Some(split) => TargetWeights::load_split_ffn(device, &checkpoint, &split[gpu]),
                            None => TargetWeights::load(device, &checkpoint, target),
                        })?;
                        assert!(!records.operands.is_empty());
                        weights
                    };
                    loaded.push(LoadedRank { weights, streams: RequestStreams { decode: Rc::new(Stream::new(device)?),
                        prefill: (0..3).map(|_| Stream::new(device).map(Rc::new)).collect::<Result<_>>()? } });
                }
                let constructed = Owners::new(&inputs, &admitted, &programs, loaded, embedding.unwrap());
                let owners = constructed?;
                assert_eq!(owners.graphs.len(), ids.len());
                assert_eq!(owners.storage.usage.is_some(), expert_usage);
                let census = owners.storage.inventory()?;
                if inputs.split() {
                    assert!(!census.receipts.iter().any(|r| r.allocation.id.starts_with("expert-usage.gpu1")));
                    if let Some(usage) = &owners.storage.usage {
                        let layer = inputs.routed[0];
                        assert!(usage.routes(ids[0], layer, cuteafd_ffi::expert_usage::UsagePhase::Decode).is_ok());
                        assert!(usage.routes(ids[1], layer, cuteafd_ffi::expert_usage::UsagePhase::Decode).is_err());
                    }
                    for (gpu, rank) in owners.storage.ranks.iter().enumerate() {
                        assert_eq!(rank.kernels.has_router(), gpu == 0);
                        for unit in std::iter::once(&rank.decode).chain(&rank.prefill) {
                            let unit = unit.borrow();
                            assert!(unit.is_split());
                            assert!(unit.plane("combined").is_err());
                            assert_eq!(unit.plane("gate")?.bytes, unit.rows() as usize * 6144 * 2);
                            assert_eq!(unit.plane("up")?.bytes, unit.rows() as usize * 6144 * 2);
                            assert_eq!(unit.plane("ffn")?.bytes, unit.rows() as usize * 6144 * 2);
                            assert_eq!(unit.plane("logits").is_ok(), gpu == 0);
                            assert_eq!(unit.plane("route_ids").is_ok(), gpu == 0 || onboard > 0);
                            assert_eq!(unit.plane("route_weights").is_ok(), gpu == 0 || onboard > 0);
                            unit.validate_split(&inputs, gpu)?;
                        }
                    }
                }
                assert_eq!(census.receipts.iter().filter(|r|
                    r.allocation.id.starts_with("expert-usage.gpu")).count(),
                    if expert_usage { 3 * inputs.usage_devices().len() } else { 0 });
                assert!(!census.receipts.iter().any(|r| r.allocation.id.starts_with("selector.")),
                    "the shared scheduler must own the sole selector");
                let selector = admitted.solved.facts.allocations.iter().find(|owner|
                    owner.id == "selector.output").context("admitted scheduler selector missing")?;
                assert_eq!(selector.location, Location::coordinator((ids.len() - 1) as u8));
                assert_eq!(selector.bytes, concurrency * 12);
                assert_eq!(admitted.solved.facts.allocations.iter().any(|owner|
                    owner.id.starts_with("selector.sampler.")), device_sampling);
                reconcile(&census, &admitted.solved.facts, &contract.components, 129, retained_owner)?;
                let index = census.receipts.iter().position(|receipt|
                    receipt.allocation.id == Owner::Kv { layer: 0, plane: "index" }.id()).unwrap();
                assert_eq!(census.receipts[index].allocation.bytes, 129 * 8448);
                assert!(admitted.solved.facts.allocations.iter().find(|owner|
                    owner.id == census.receipts[index].allocation.id).unwrap().bytes > 129 * 8448);
                for defect in 0..6 {
                    let mut changed = Census::default();
                    changed.receipts = census.receipts.clone();
                    let receipt = &mut changed.receipts[index];
                    match defect {
                        0 => receipt.allocation.bytes += 1,
                        1 => receipt.allocation.location = Location::PINNED,
                        2 => receipt.allocation.lifetime = Lifetime::RESIDENT,
                        3 => receipt.address = 0,
                        4 => { changed.receipts.remove(index); }
                        5 => changed.receipts.push(changed.receipts[index].clone()),
                        _ => unreachable!(),
                    }
                    assert!(reconcile(&changed, &admitted.solved.facts, &contract.components, 129, retained_owner).is_err(),
                        "{} RTX C{concurrency} defect {defect}", ids.len());
                }
                let mut wrong_charge = admitted.solved.facts.clone();
                wrong_charge.allocations.iter_mut().find(|owner|
                    owner.id == census.receipts[index].allocation.id).unwrap().bytes += 1;
                assert!(reconcile(&census, &wrong_charge, &contract.components, 129, retained_owner).is_err());
                for rank in &owners.storage.ranks {
                    assert_eq!(rank.prefill.len(), 3);
                    assert_eq!(rank.decode.borrow().plane("residual")?.bytes, concurrency as usize * 6144 * 4);
                }
                let weak = Rc::downgrade(&owners.storage);
                let Owners { graphs, storage, .. } = owners;
                drop(storage);
                assert!(weak.upgrade().is_some(), "graph owner must pin all arithmetic/request/weight storage");
                drop(graphs);
                assert!(weak.upgrade().is_none());
                assert_eq!(fixture.mapped_bytes(&library)?, 0, "every virtual native owner retired");
                assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
                }
                }
            }
        }
        Ok(())
    }
}

