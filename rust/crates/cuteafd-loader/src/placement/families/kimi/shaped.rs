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

fn backing(requests: &[u64]) -> Result<u64, Refusal> {
    requests.iter().try_fold(0u64, |sum, &bytes| sum.checked_add(sm120_request_backing(bytes)?)
        .ok_or(Refusal::Overflow("Kimi shaped backing")))
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
            let mut samples: Vec<Vec<u64>> = Vec::new();
            let mut roots = Vec::new();
            for pages in 1..=self.maximum_pages {
                let context = request.context.served(pages * self.page_rows());
                let requests = owner.requests(context)?;
                if pages == 1 {
                    samples.resize_with(requests.len(), Vec::new);
                    roots = requests.clone();
                }
                for (values, (_, _, _, requests)) in samples.iter_mut().zip(requests) {
                    values.push(backing(&requests)?);
                }
            }
            for ((rank, group, category, _), values) in roots.into_iter().zip(samples) {
                let owner = Allocation::buffer(format!("{group}.{rank}"), Location::coordinator(rank),
                    values[0], Lifetime::RESIDENT, category, group);
                let mut component = Component::allocation(owner, ByteCurve::fixed(values[0], self.maximum_pages),
                    false, self.maximum_pages);
                component.byte_extent(curve(&values))?;
                components.push(component);
            }
        }
        Ok(components)
    }

    pub(super) fn refine_shaped(&self, solved: &mut ResolvedPlan) -> Result<(), Refusal> {
        let before = evaluate(&solved.facts, &solved.budgets)?;
        for owner in &self.shaped {
            for (rank, group, _, requests) in owner.requests(solved.served_context())? {
                let id = format!("{group}.{rank}");
                let root = solved.facts.allocations.iter_mut().find(|root| root.id == id)
                    .ok_or_else(|| invalid("shaped owner missing from selected receipt"))?;
                if root.bytes != backing(&requests)? { return Err(invalid("shaped backing differs from search")); }
                root.bytes = requests.iter().sum();
                root.shape = vec![root.bytes];
                solved.facts.allocation_granularity.push(AllocationGranularity {
                    id: format!("{id}.tail"), location: root.location, lifetime: root.lifetime,
                    backings: vec![GranularityBacking { backing: id, requests }],
                    evidence: GranularityEvidence::IndependentLarge,
                });
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
