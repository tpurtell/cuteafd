//! Pool-dependent Kimi owners. Search physical backings, then retain exact
//! allocation requests and their single named tail in the solved receipt.
use super::*;
use crate::families::kimi_k3::{allocation::NativeLaneLayout, context::ContextLayout};

#[derive(Debug, Clone)]
pub enum ShapedOwner {
    Lane { rank: u8, lane: usize, layout: NativeLaneLayout, extra_requests: Vec<usize> },
    Context { layout: ContextLayout },
}

impl ShapedOwner {
    fn requests(&self, context: u64) -> Result<Vec<(u8, String, Category, Vec<u64>)>, Refusal> {
        let context = usize::try_from(context).map_err(|_| Refusal::Overflow("Kimi served context"))?;
        let error = |e: anyhow::Error| invalid(&e.to_string());
        match self {
            Self::Lane { rank, lane, layout, extra_requests } => {
                let layout = layout.with_served_context(context).map_err(error)?;
                let requests = std::iter::once(layout.arena_bytes)
                    .chain(layout.operands.iter().filter(|(name, _)| !layout.arena_offsets.contains_key(*name))
                        .map(|(_, &bytes)| bytes))
                    .chain(extra_requests.iter().copied()).filter(|&bytes| bytes > 0)
                    .map(|bytes| bytes as u64).collect();
                Ok(vec![(*rank, format!("Kimi lane {lane} native operands"), Category::Workspace, requests)])
            }
            Self::Context { layout } => {
                let layout = layout.with_served_context(context).map_err(error)?;
                let buffers = layout.buffers();
                let staging = buffers.compiled_extent.div_ceil(buffers.staging_unit_rows) * buffers.staging_unit_bytes;
                let workspace = layout.workspace().iter().map(|(_, bytes)| *bytes as u64).collect::<Vec<_>>();
                Ok((0..2).flat_map(|rank| [
                    (rank, "context staging".into(), Category::Workspace, vec![staging; 2]),
                    (rank, "context workspace".into(), Category::Workspace, workspace.clone()),
                    (rank, "context exchange".into(), Category::Transport,
                        vec![crate::placement::ContextExchangeLayout::new(&buffers).bytes().expect("validated layout")]),
                ]).collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn affine_curves_preserve_every_sample_without_rounding_sawtooth_expansion() {
        let values = (1..=32768).map(|pages| (pages.min(16384) * 256) as u64).collect::<Vec<_>>();
        let curve = super::curve(&values);
        assert!(curve.0.len() <= 2);
        for (index, &value) in values.iter().enumerate() {
            assert_eq!(curve.at(index as u64 + 1).unwrap(), value);
        }
    }

    use super::*;

    #[test]
    fn context_curves_and_receipts_follow_solved_pool_and_pins() -> anyhow::Result<()> {
        let mut manifest = serde_json::json!({"programs": []});
        crate::families::kimi_k3::context::add_manifest_programs(&mut manifest);
        let initial = ContextLayout::from_manifest(&manifest, cuteafd_core::MlaCacheFormat::Fp8,
            1_048_576, 2752, 3)?;
        let mut recipe = super::super::tests::recipe();
        recipe.context = 1_048_576;
        recipe.maximum_pages = 16_384;
        recipe.shaped.push(ShapedOwner::Context { layout: initial.clone() });
        for context in [Context::automatic(1_048_576), Context::pinned(262_145)] {
            let request = Request { context, ..super::super::tests::single() };
            let components = recipe.shaped_components(&request)?;
            for pages in [1, 63, 64, 4096, 4097, 8192, 16384] {
                let served = context.served(pages * recipe.page_rows());
                let expected = initial.with_served_context(served as usize)?;
                let demands = expected.buffers().demands()?;
                for component in &components {
                    let ComponentKind::Allocation { template, .. } = &component.kind else { panic!("allocation"); };
                    let requests = recipe.shaped[0].requests(served)?.into_iter()
                        .find(|(rank, group, _, _)| Location::coordinator(*rank) == template.location
                            && *group == template.group).unwrap().3;
                    let demand = demands.iter().find(|d| Location::coordinator(d.gpu) == template.location
                        && d.group == template.group).unwrap();
                    assert_eq!(requests.iter().sum::<u64>(), demand.bytes);
                    let index: usize = template.id.rsplit('.').next().unwrap().parse()?;
                    assert_eq!(component.bytes.at(pages)?, sm120_request_backing(requests[index])?);
                }
                let contract = Contract { components: components.clone(), floor_fallback: None,
                    resident_weight_limit: None, phase_order: PhaseOrder::DRAFT_THEN_TARGET,
                    spark_reply_formats: None, page_rows: recipe.page_rows(), pool_page_multiple: 1,
                    maximum_pages: recipe.maximum_pages, gpus: 2,
                    variables: vec![Variable::Prefill(vec![recipe.prefill]), Variable::Lanes(vec![1])], requires: vec![] };
                let selected = [0, 0];
                let facts = contract.facts(&selected, pages)?;
                let mut solved = ResolvedPlan { request: request.clone(), pool_tokens: pages * recipe.page_rows(),
                    pool_page_rows: recipe.page_rows(), facts,
                    budgets: (0..2).map(|rank| Budget { id: format!("gpu{rank}"),
                        locations: vec![Location::coordinator(rank)], physical_bytes: 96 << 30,
                        logical_bytes: 96 << 30, headroom: 0,
                        baseline: Baseline::Forecast { occupied: 0, source: "fixture".into(), assumed: false } }).collect(),
                    phases: vec![], decisions: contract.decisions(&selected)?, encoder_diagnostics: vec![] };
                recipe.refine_shaped(&mut solved)?;
                for demand in demands {
                    let owners = solved.facts.allocations.iter().filter(|root|
                        root.group == demand.group && root.location == Location::coordinator(demand.gpu)).collect::<Vec<_>>();
                    assert_eq!(owners.len(), 1);
                    assert_eq!(owners[0].bytes, demand.bytes);
                }
                assert_eq!(solved.facts.allocation_granularity.len(), 6);
            }
        }
        Ok(())
    }
}

/// Compress exact samples into affine intervals. A capped served context makes
/// the tail flat; rounding boundaries remain explicit rather than approximated.
fn curve(values: &[u64]) -> ByteCurve {
    let mut pieces = Vec::new();
    let mut first = 0;
    while first < values.len() {
        let slope = values.get(first + 1).map_or(0, |next| next - values[first]);
        let mut last = first;
        while last + 1 < values.len() && values[last + 1] - values[last] == slope { last += 1; }
        pieces.push(BytePiece { first: first as u64 + 1, last: last as u64 + 1,
            bytes_at_first: values[first], bytes_per_page: slope, granularity: 1, backings: 1 });
        first = last + 1;
    }
    ByteCurve(pieces)
}

impl KimiRecipe {
    pub(super) fn shaped_components(&self, request: &Request) -> Result<Vec<Component>, Refusal> {
        let mut components = Vec::new();
        for owner in &self.shaped {
            let mut samples: Vec<Vec<Vec<u64>>> = Vec::new();
            let mut roots = Vec::new();
            for pages in 1..=self.maximum_pages {
                let context = request.context.served(pages * self.page_rows());
                let requests = owner.requests(context)?;
                if pages == 1 {
                    samples.resize_with(requests.len(), Vec::new);
                    roots = requests.clone();
                }
                for (values, (_, _, _, requests)) in samples.iter_mut().zip(requests) {
                    if pages == 1 { values.resize_with(requests.len(), Vec::new); }
                    for (values, bytes) in values.iter_mut().zip(requests) { values.push(bytes); }
                }
            }
            for ((rank, group, category, _), requests) in roots.into_iter().zip(samples) {
                for (index, values) in requests.into_iter().enumerate() {
                    let owner = Allocation::buffer(format!("{group}.{rank}.request.{index}"), Location::coordinator(rank),
                        values[0], Lifetime::RESIDENT, category, group.clone());
                    let mut physical = curve(&values);
                    for piece in &mut physical.0 { piece.granularity = SM120_ALLOCATION_GRANULARITY_BYTES; }
                    let mut component = Component::allocation(owner, ByteCurve::fixed(values[0], self.maximum_pages),
                        false, self.maximum_pages);
                    component.byte_extent(physical)?;
                    components.push(component);
                }
            }
        }
        Ok(components)
    }

    pub(super) fn refine_shaped(&self, solved: &mut ResolvedPlan) -> Result<(), Refusal> {
        let before = evaluate(&solved.facts, &solved.budgets)?;
        for owner in &self.shaped {
            for (rank, group, _, requests) in owner.requests(solved.served_context())? {
                let id = format!("{group}.{rank}");
                let mut root = None;
                for (index, &bytes) in requests.iter().enumerate() {
                    let request_id = format!("{id}.request.{index}");
                    let position = solved.facts.allocations.iter().position(|root| root.id == request_id)
                        .ok_or_else(|| invalid("shaped owner missing from selected receipt"))?;
                    let allocation = solved.facts.allocations.remove(position);
                    if allocation.bytes != sm120_request_backing(bytes)? {
                        return Err(invalid("shaped backing differs from search"));
                    }
                    root.get_or_insert(allocation);
                }
                let mut root = root.ok_or_else(|| invalid("shaped owner has no allocations"))?;
                root.id = id.clone();
                root.bytes = requests.iter().sum();
                root.shape = vec![root.bytes];
                solved.facts.allocation_granularity.push(AllocationGranularity {
                    id: format!("{id}.tail"), location: root.location, lifetime: root.lifetime,
                    backings: vec![GranularityBacking { backing: id, requests }],
                    evidence: GranularityEvidence::IndependentLarge,
                });
                solved.facts.allocations.push(root);
            }
        }
        let after = evaluate(&solved.facts, &solved.budgets)?;
        if before.len() != after.len() || before.iter().zip(&after).any(|(a, b)|
            a.budget != b.budget || a.phase != b.phase || a.available != b.available
                || a.owned.checked_add(a.opaque) != b.owned.checked_add(b.opaque)) {
            return Err(invalid("shaped receipt refinement changed physical costs"));
        }
        solved.phases = after;
        Ok(())
    }
}
