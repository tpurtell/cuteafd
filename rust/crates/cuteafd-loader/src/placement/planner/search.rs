//! Propagate finite domains, price surviving assignments, materialize one winner.
use super::*;

#[derive(Debug, Clone, Copy, Default)]
struct Cost {
    owned: u64,
    opaque: u64,
    owned_slope: u64,
    opaque_slope: u64,
}
impl Cost {
    fn add(&mut self, other: Self) -> Result<(), Refusal> {
        self.owned = self
            .owned
            .checked_add(other.owned)
            .ok_or(Refusal::Overflow("owned cost"))?;
        self.opaque = self
            .opaque
            .checked_add(other.opaque)
            .ok_or(Refusal::Overflow("opaque cost"))?;
        self.owned_slope = self
            .owned_slope
            .checked_add(other.owned_slope)
            .ok_or(Refusal::Overflow("owned slope"))?;
        self.opaque_slope = self
            .opaque_slope
            .checked_add(other.opaque_slope)
            .ok_or(Refusal::Overflow("opaque slope"))?;
        Ok(())
    }
    fn required(self) -> Result<u64, Refusal> {
        self.owned
            .checked_add(self.opaque)
            .ok_or(Refusal::Overflow("phase cost"))
    }
    fn slope(self) -> Result<u64, Refusal> {
        self.owned_slope
            .checked_add(self.opaque_slope)
            .ok_or(Refusal::Overflow("phase slope"))
    }
    fn at(self, delta: u64) -> Result<Self, Refusal> {
        Ok(Self {
            owned: self
                .owned_slope
                .checked_mul(delta)
                .and_then(|n| self.owned.checked_add(n))
                .ok_or(Refusal::Overflow("owned curve"))?,
            opaque: self
                .opaque_slope
                .checked_mul(delta)
                .and_then(|n| self.opaque.checked_add(n))
                .ok_or(Refusal::Overflow("opaque curve"))?,
            ..self
        })
    }
}
struct Group {
    activation: Activation,
    /// [piece][budget x phase], precomputed once, without allocation metadata.
    costs: Vec<Vec<Cost>>,
}
pub(super) struct Compiled {
    starts: Vec<u64>,
    pool_page_multiple: u64,
    phase_order: PhaseOrder,
    groups: Vec<Group>,
    rows: usize,
    memory_rows: usize,
    resident_weight_limit: Option<ResidentWeights>,
}
impl Compiled {
    pub(super) fn new(contract: &Contract, budgets: &[Budget]) -> Result<Self, Refusal> {
        let mut starts = BTreeSet::from([1, contract.maximum_pages + 1]);
        for component in &contract.components {
            for piece in &component.bytes.0 {
                starts.insert(piece.first);
            }
        }
        let starts: Vec<_> = starts.into_iter().collect();
        let memory_rows = budgets.len() * contract.phase_order.phases().len();
        let rows = memory_rows + contract.gpus;
        let mut groups = BTreeMap::<Activation, Vec<Vec<Cost>>>::new();
        for component in &contract.components {
            let (charged_rows, opaque) = if let ComponentKind::ResidentWeight(owner) = &component.kind {
                let Device::Coordinator(gpu) = owner.location.device else { continue };
                (vec![memory_rows + usize::from(gpu)], false)
            } else if let Some((location, lifetime, opaque)) = component.charge() {
                let budget = budgets.iter().position(|b| b.locations.contains(&location))
                    .ok_or_else(|| Refusal::InvalidFact("component has no physical budget".into()))?;
                let charged = contract.phase_order.phases().iter().enumerate()
                    .filter(|(_, at)| lifetime.contains(**at, contract.phase_order))
                    .map(|(phase, _)| budget * contract.phase_order.phases().len() + phase)
                    .collect();
                (charged, opaque)
            } else {
                continue;
            };
            let costs = groups
                .entry(component.activation.clone())
                .or_insert_with(|| vec![vec![Cost::default(); rows]; starts.len() - 1]);
            for (index, &first) in starts[..starts.len() - 1].iter().enumerate() {
                let piece = component.bytes.piece(first)?;
                let bytes = component.bytes.at(first)?;
                let cost = if opaque {
                    Cost { opaque: bytes, opaque_slope: piece.bytes_per_page, ..Cost::default() }
                } else {
                    Cost { owned: bytes, owned_slope: piece.bytes_per_page, ..Cost::default() }
                };
                for &row in &charged_rows {
                    costs[index][row].add(cost)?;
                }
            }
        }
        Ok(Self {
            starts,
            pool_page_multiple: contract.pool_page_multiple,
            phase_order: contract.phase_order,
            rows,
            memory_rows,
            resident_weight_limit: contract.resident_weight_limit,
            groups: groups
                .into_iter()
                .map(|(activation, costs)| Group { activation, costs })
                .collect(),
        })
    }
    fn available(&self, row: usize, budgets: &[Budget]) -> Result<u64, Refusal> {
        if row < self.memory_rows {
            budgets[row / self.phase_order.phases().len()].available()
        } else {
            Ok(self.resident_weight_limit.map_or(u64::MAX, |limit| limit.0))
        }
    }
    fn capacity(&self, rows: &[u64], budgets: &[Budget]) -> Result<(), Refusal> {
        for (row, &required) in rows.iter().enumerate() {
            let available = self.available(row, budgets)?;
            if required <= available { continue; }
            return Err(if row < self.memory_rows {
                Refusal::Capacity {
                    budget: budgets[row / self.phase_order.phases().len()].id.clone(),
                    phase: self.phase_order.phases()[row % self.phase_order.phases().len()],
                    required, available, owners: vec![],
                }
            } else {
                Refusal::ResidentWeightLimit {
                    gpu: (row - self.memory_rows) as u8, required, limit: available,
                }
            });
        }
        Ok(())
    }
    fn costs(&self, selected: &[usize], piece: usize) -> Result<Vec<Cost>, Refusal> {
        let mut costs = vec![Cost::default(); self.rows];
        for group in &self.groups {
            if group.activation.active(selected) {
                for (sum, &cost) in costs.iter_mut().zip(&group.costs[piece]) {
                    sum.add(cost)?;
                }
            }
        }
        Ok(costs)
    }
    fn piece(&self, pages: u64) -> usize {
        self.starts.partition_point(|&start| start <= pages) - 1
    }
    pub(super) fn phases(
        &self,
        budgets: &[Budget],
        selected: &[usize],
        pages: u64,
    ) -> Result<Vec<PhaseBudget>, Refusal> {
        let piece = self.piece(pages);
        self.costs(selected, piece)?
            .into_iter()
            .take(self.memory_rows)
            .enumerate()
            .map(|(row, cost)| {
                let cost = cost.at(pages - self.starts[piece])?;
                let budget = &budgets[row / self.phase_order.phases().len()];
                Ok(PhaseBudget {
                    budget: budget.id.clone(),
                    phase: self.phase_order.phases()[row % self.phase_order.phases().len()],
                    owned: cost.owned,
                    opaque: cost.opaque,
                    available: budget.available()?,
                })
            })
            .collect()
    }
    fn weights(&self, selected: &[usize], pages: u64) -> Result<Vec<ResidentWeights>, Refusal> {
        let piece = self.piece(pages);
        self.costs(selected, piece)?.into_iter().skip(self.memory_rows)
            .map(|cost| Ok(ResidentWeights(cost.at(pages - self.starts[piece])?.required()?)))
            .collect()
    }
    pub(super) fn limit(
        &self,
        budgets: &[Budget],
        selected: &[usize],
        lower: u64,
        upper: u64,
    ) -> Result<Option<u64>, Refusal> {
        let mut answer = None;
        for piece in 0..self.starts.len() - 1 {
            let first = self.starts[piece].max(lower).div_ceil(self.pool_page_multiple)
                .checked_mul(self.pool_page_multiple).ok_or(Refusal::Overflow("pool page alignment"))?;
            let mut last = (self.starts[piece + 1] - 1).min(upper);
            if first > last {
                continue;
            }
            for (row, cost) in self.costs(selected, piece)?.into_iter().enumerate() {
                let cost = cost.at(first - self.starts[piece])?;
                let available = self.available(row, budgets)?;
                let required = cost.required()?;
                if required > available {
                    last = 0;
                    break;
                }
                let slope = cost.slope()?;
                if slope != 0 {
                    last = last.min(first.saturating_add((available - required) / slope));
                }
            }
            last = last / self.pool_page_multiple * self.pool_page_multiple;
            if first <= last {
                answer = Some(last);
            }
        }
        Ok(answer)
    }
    /// Independent intercept/slope minima underprice every surviving assignment.
    /// Their closed-form limit is therefore a true upper bound, even when the
    /// cheapest intercept and slope come from different options.
    fn upper_bound(
        &self,
        budgets: &[Budget],
        domains: &[Vec<usize>],
        lower: u64,
        upper: u64,
    ) -> Result<Option<u64>, Refusal> {
        let mut answer = None;
        for piece in 0..self.starts.len() - 1 {
            let first = self.starts[piece].max(lower).div_ceil(self.pool_page_multiple)
                .checked_mul(self.pool_page_multiple).ok_or(Refusal::Overflow("pool page alignment"))?;
            let mut last = (self.starts[piece + 1] - 1).min(upper);
            if first > last {
                continue;
            }
            let mut result = vec![Cost::default(); self.rows];
            let mut local = BTreeMap::<usize, Vec<Vec<Cost>>>::new();
            for group in &self.groups {
                if group.activation.guaranteed(domains) {
                    for (sum, &cost) in result.iter_mut().zip(&group.costs[piece]) {
                        sum.add(cost.at(first - self.starts[piece])?)?;
                    }
                } else if let Some(variable) = group.activation.local_variable() {
                    let values = local.entry(variable).or_insert_with(|| {
                        vec![vec![Cost::default(); self.rows]; domains[variable].len()]
                    });
                    for (&option, costs) in domains[variable].iter().zip(values) {
                        if group
                            .activation
                            .clauses
                            .iter()
                            .any(|c| c[0].option == option)
                        {
                            for (sum, &cost) in costs.iter_mut().zip(&group.costs[piece]) {
                                sum.add(cost.at(first - self.starts[piece])?)?;
                            }
                        }
                    }
                }
            }
            for values in local.into_values() {
                for (row, sum) in result.iter_mut().enumerate() {
                    let mut bytes = u64::MAX;
                    let mut slope = u64::MAX;
                    for option in &values {
                        bytes = bytes.min(option[row].required()?);
                        slope = slope.min(option[row].slope()?);
                    }
                    sum.add(Cost {
                        owned: bytes,
                        owned_slope: slope,
                        ..Cost::default()
                    })?;
                }
            }
            for (row, cost) in result.into_iter().enumerate() {
                let available = self.available(row, budgets)?;
                let required = cost.required()?;
                if required > available {
                    last = 0;
                    break;
                }
                let slope = cost.slope()?;
                if slope != 0 {
                    last = last.min(first.saturating_add((available - required) / slope));
                }
            }
            last = last / self.pool_page_multiple * self.pool_page_multiple;
            if first <= last {
                answer = Some(last);
            }
        }
        Ok(answer)
    }
    fn group_minimum(&self, group: &Group, lower: u64, upper: u64) -> Result<Vec<u64>, Refusal> {
        let mut result = vec![u64::MAX; self.rows];
        for piece in 0..self.starts.len() - 1 {
            let first = self.starts[piece].max(lower);
            if first > upper || first >= self.starts[piece + 1] {
                continue;
            }
            for (value, &cost) in result.iter_mut().zip(&group.costs[piece]) {
                *value = (*value).min(cost.at(first - self.starts[piece])?.required()?);
            }
        }
        Ok(result)
    }
    fn lower_bound(
        &self,
        domains: &[Vec<usize>],
        lower: u64,
        upper: u64,
    ) -> Result<Vec<u64>, Refusal> {
        let mut result = vec![0u64; self.rows];
        let mut local = BTreeMap::<usize, Vec<Vec<u64>>>::new();
        for group in &self.groups {
            let minimum = self.group_minimum(group, lower, upper)?;
            if group.activation.guaranteed(domains) {
                add_bytes(&mut result, &minimum)?;
            } else if let Some(variable) = group.activation.local_variable() {
                let values = local
                    .entry(variable)
                    .or_insert_with(|| vec![vec![0; self.rows]; domains[variable].len()]);
                for (&option, costs) in domains[variable].iter().zip(values) {
                    if group
                        .activation
                        .clauses
                        .iter()
                        .any(|c| c[0].option == option)
                    {
                        add_bytes(costs, &minimum)?;
                    }
                }
            }
        }
        for values in local.into_values() {
            let minimum: Vec<_> = (0..self.rows)
                .map(|row| values.iter().map(|v| v[row]).min().unwrap_or(0))
                .collect();
            add_bytes(&mut result, &minimum)?;
        }
        Ok(result)
    }
}
fn add_bytes(destination: &mut [u64], bytes: &[u64]) -> Result<(), Refusal> {
    for (sum, &n) in destination.iter_mut().zip(bytes) {
        *sum = sum
            .checked_add(n)
            .ok_or(Refusal::Overflow("cost lower bound"))?;
    }
    Ok(())
}
fn compatible(contract: &Contract, domains: &[Vec<usize>]) -> bool {
    contract.requires.iter().all(|r| {
        domains[r.when.variable] != [r.when.option]
            || r.any_of
                .iter()
                .any(|c| domains[c.variable].contains(&c.option))
    })
}
fn logical(contract: &Contract, domains: &mut [Vec<usize>], exact: Option<u64>) -> bool {
    loop {
        let before = domains.to_vec();
        for constraint in &contract.requires {
            let available: Vec<_> = constraint
                .any_of
                .iter()
                .filter(|c| domains[c.variable].contains(&c.option))
                .copied()
                .collect();
            if available.is_empty() {
                domains[constraint.when.variable].retain(|&o| o != constraint.when.option);
            } else if domains[constraint.when.variable] == [constraint.when.option]
                && available.len() == 1
            {
                domains[available[0].variable].retain(|&o| o == available[0].option);
            }
        }
        if domains.iter().any(Vec::is_empty) {
            return false;
        }
        // Different destination profiles may describe the same routed layers;
        // only their selected prefixes must be disjoint. Retain an option only
        // when every other profile still has a non-overlapping support.
        for variable in 0..domains.len() {
            let retained: Vec<_> = domains[variable]
                .iter()
                .copied()
                .filter(|&option| {
                    let layers = contract.variables[variable].onboard_layers(option);
                    layers.is_empty()
                        || domains.iter().enumerate().all(|(other, domain)| {
                            variable == other
                                || domain.iter().any(|&candidate| {
                                    contract.variables[other]
                                        .onboard_layers(candidate)
                                        .iter()
                                        .all(|layer| !layers.contains(layer))
                                })
                        })
                })
                .collect();
            domains[variable] = retained;
        }
        if domains.iter().any(Vec::is_empty) {
            return false;
        }
        if let Some(exact) = exact {
            let ranges: Vec<_> = contract
                .variables
                .iter()
                .zip(domains.iter())
                .map(|(v, d)| {
                    let counts: Vec<_> = d.iter().map(|&o| v.onboard(o)).collect();
                    (*counts.iter().min().unwrap(), *counts.iter().max().unwrap())
                })
                .collect();
            let minimum: u64 = ranges.iter().map(|r| r.0).sum();
            let maximum: u64 = ranges.iter().map(|r| r.1).sum();
            if exact < minimum || exact > maximum {
                return false;
            }
            for (index, domain) in domains.iter_mut().enumerate() {
                domain.retain(|&o| {
                    let count = contract.variables[index].onboard(o);
                    minimum - ranges[index].0 + count <= exact
                        && maximum - ranges[index].1 + count >= exact
                });
            }
        }
        if domains.iter().any(Vec::is_empty) {
            return false;
        }
        if before == domains {
            return compatible(contract, domains);
        }
    }
}
fn propagate(
    contract: &Contract,
    costs: &Compiled,
    budgets: &[Budget],
    domains: &mut [Vec<usize>],
    lower: u64,
    upper: u64,
    exact: Option<u64>,
) -> Result<(), Refusal> {
    loop {
        if !logical(contract, domains, exact) {
            return Err(Refusal::PinConflict);
        }
        costs.capacity(&costs.lower_bound(domains, lower, upper)?, budgets)?;
        let before = domains.to_vec();
        for variable in 0..domains.len() {
            if domains[variable].len() == 1 { continue; }
            let mut retained = Vec::new();
            let mut capacity_failure = None;
            for &option in &domains[variable] {
                let mut forced = domains.to_vec();
                forced[variable] = vec![option];
                if logical(contract, &mut forced, exact) {
                    match costs.capacity(&costs.lower_bound(&forced, lower, upper)?, budgets) {
                        Ok(()) => retained.push(option),
                        Err(error @ (Refusal::Capacity { .. } | Refusal::ResidentWeightLimit { .. })) => {
                            capacity_failure.get_or_insert(error);
                        }
                        Err(error) => return Err(error),
                    }
                }
            }
            domains[variable] = retained;
            if domains[variable].is_empty() {
                return Err(capacity_failure.unwrap_or(Refusal::PinConflict));
            }
        }
        if before == domains {
            return Ok(());
        }
    }
}
fn remaining(phases: &[PhaseBudget]) -> u64 {
    phases
        .iter()
        .map(|p| p.available - p.owned - p.opaque)
        .min()
        .unwrap_or(0)
}

fn coordinator_slack(phases: &[PhaseBudget], budgets: &[Budget]) -> u64 {
    phases
        .iter()
        .filter(|phase| {
            matches!(phase.phase, Phase::Warmup | Phase::Ready | Phase::Growth)
                && budgets.iter().any(|budget| budget.id == phase.budget
                    && budget.locations.iter().any(|l| matches!(l.device, Device::Coordinator(_))))
        })
        .map(|p| p.available - p.owned - p.opaque)
        .min()
        .unwrap_or(0)
}

fn branch(contract: &Contract, domains: &[Vec<usize>]) -> Option<usize> {
    // Executor choices precede the small destination-count tuple. No ordinary
    // optional layer becomes an independent binary decision.
    (0..domains.len())
        .filter(|&i| domains[i].len() > 1)
        .min_by_key(|&i| {
            (
                matches!(contract.variables[i], Variable::Onboard { .. }),
                domains[i].len(),
                i,
            )
        })
}
fn selected(domains: &[Vec<usize>]) -> Vec<usize> {
    domains.iter().map(|d| d[0]).collect()
}
fn infeasible(error: &Refusal) -> bool {
    matches!(error, Refusal::Capacity { .. } | Refusal::ResidentWeightLimit { .. } | Refusal::PinConflict)
}

struct Search<'a> {
    contract: &'a Contract,
    costs: &'a Compiled,
    budgets: &'a [Budget],
    exact: Option<u64>,
    assignments: u64,
    last: Refusal,
}
impl Search<'_> {
    fn restrict(
        &mut self,
        domains: &mut [Vec<usize>],
        lower: u64,
        upper: u64,
    ) -> Result<bool, Refusal> {
        match propagate(
            self.contract,
            self.costs,
            self.budgets,
            domains,
            lower,
            upper,
            self.exact,
        ) {
            Ok(()) => Ok(true),
            Err(error) if infeasible(&error) => {
                // Preserve a typed resource failure across later logical dead ends.
                if matches!(error, Refusal::Capacity { .. } | Refusal::ResidentWeightLimit { .. })
                    || !matches!(self.last, Refusal::Capacity { .. } | Refusal::ResidentWeightLimit { .. })
                {
                    self.last = error;
                }
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }
    fn count_assignment(&mut self) -> Result<(), Refusal> {
        self.assignments = self
            .assignments
            .checked_add(1)
            .ok_or(Refusal::Overflow("assignment count"))?;
        Ok(())
    }
    fn feasible(&mut self, mut domains: Vec<Vec<usize>>, pages: u64) -> Result<bool, Refusal> {
        if !self.restrict(&mut domains, pages, pages)? {
            return Ok(false);
        }
        let Some(variable) = branch(self.contract, &domains) else {
            self.count_assignment()?;
            return Ok(true);
        };
        for option in domains[variable].clone() {
            let mut child = domains.clone();
            child[variable] = vec![option];
            if self.feasible(child, pages)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn maximum(
        &mut self,
        mut domains: Vec<Vec<usize>>,
        lower: u64,
        upper: u64,
        best: &mut Option<u64>,
    ) -> Result<(), Refusal> {
        if best.is_some_and(|p| p >= upper) || !self.restrict(&mut domains, lower, upper)? {
            return Ok(());
        }
        let Some(bound) = self
            .costs
            .upper_bound(self.budgets, &domains, lower, upper)?
        else {
            return Ok(());
        };
        if best.is_some_and(|p| p >= bound) {
            return Ok(());
        }
        // Bind the analytical upper bound first. If correlations make it
        // unattainable, refine finite assignments, never decrement pool pages.
        if self.feasible(domains.clone(), bound)? {
            *best = Some(bound);
            return Ok(());
        }
        let Some(variable) = branch(self.contract, &domains) else {
            let assignment = selected(&domains);
            if let Some(pages) = self.costs.limit(self.budgets, &assignment, lower, bound)? {
                if best.is_none_or(|old| pages > old) {
                    *best = Some(pages);
                }
            }
            return Ok(());
        };
        for option in domains[variable].clone() {
            let mut child = domains.clone();
            child[variable] = vec![option];
            self.maximum(child, lower, bound, best)?;
        }
        Ok(())
    }
    fn complete(
        &mut self,
        mut domains: Vec<Vec<usize>>,
        pages: u64,
        best: &mut Option<((u64, u64, u64, u64), Vec<usize>)>,
    ) -> Result<(), Refusal> {
        if !self.restrict(&mut domains, pages, pages)? {
            return Ok(());
        }
        let maximum: u64 = self
            .contract
            .variables
            .iter()
            .zip(&domains)
            .map(|(v, d)| d.iter().map(|&o| v.onboard(o)).max().unwrap())
            .sum();
        if best.as_ref().is_some_and(|(score, _)| maximum < score.0) {
            return Ok(());
        }
        if let Some(variable) = branch(self.contract, &domains) {
            // High counts find a useful incumbent early; canonical final ties
            // are explicit and independent of this exploration order.
            let mut options = domains[variable].clone();
            options.sort_by_key(|&o| {
                (
                    std::cmp::Reverse(self.contract.variables[variable].onboard(o)),
                    o,
                )
            });
            for option in options {
                let mut child = domains.clone();
                child[variable] = vec![option];
                self.complete(child, pages, best)?;
            }
            return Ok(());
        }
        self.count_assignment()?;
        let assignment = selected(&domains);
        let layers = self
            .contract
            .variables
            .iter()
            .zip(&assignment)
            .map(|(v, &o)| v.onboard(o))
            .sum();
        let phases = self.costs.phases(self.budgets, &assignment, pages)?;
        let encoder_preference = self.contract.variables.iter().zip(&assignment).find_map(|(variable, &option)|
            if let Variable::Encoder(values) = variable { Some((values.len() - option) as u64) } else { None }).unwrap_or(0);
        let score = (
            layers,
            encoder_preference,
            coordinator_slack(&phases, self.budgets),
            remaining(&phases),
        );
        if best
            .as_ref()
            .is_none_or(|(old, choices)| score > *old || (score == *old && assignment < *choices))
        {
            *best = Some((score, assignment));
        }
        Ok(())
    }
}

/// `GPU_RESERVE` / `SPARK_RESERVE` replace a device kind's whole reserve:
/// every built-in reserve component (runtime, graph and reserved charges,
/// opaque allowances) on those devices is dropped and the budget keeps the
/// override free instead. Sampled occupancy stays in the baseline: those
/// bytes are already held, not a component.
pub(crate) fn apply_reserve_override(contract: &mut Contract, budgets: &mut [Budget]) {
    use cuteafd_core::memory_policy::{DeviceKind, MemoryPolicy};
    let policy = MemoryPolicy::current();
    let kind = |device: Device| match device {
        Device::Coordinator(_) => Some(DeviceKind::Gpu),
        Device::Spark(_) => Some(DeviceKind::Spark),
        _ => None,
    };
    let overridden = |location: &Location| kind(location.device)
        .is_some_and(|kind| policy.reserve_override(kind).is_some());
    contract.components.retain(|component| {
        let reserve = match &component.kind {
            ComponentKind::Allowance(allowance) => Some(allowance.location),
            ComponentKind::Allocation { template, .. } if matches!(template.category,
                Category::Runtime | Category::Reserved) => Some(template.location),
            _ => None,
        };
        !reserve.is_some_and(|location| overridden(&location))
    });
    for budget in budgets.iter_mut() {
        let Some(kind) = budget.locations.first().and_then(|location| kind(location.device)) else { continue };
        if let Some(bytes) = policy.reserve_override(kind) { budget.headroom = bytes; }
    }
}

pub fn solve(
    request: Request,
    budgets: Vec<Budget>,
    recipe: &impl Recipe,
) -> Result<ResolvedPlan, Refusal> {
    let started = std::time::Instant::now();
    let mut contract = recipe.prepare(&request)?;
    contract.resident_weight_limit = request.resident_weight_limit;
    let mut budgets = budgets;
    apply_reserve_override(&mut contract, &mut budgets);
    let requested = request.spark_reply_formats;
    let resolved = contract.spark_reply_formats;
    if [(requested.prefill, resolved.and_then(|formats| formats.prefill)),
        (requested.decode, resolved.and_then(|formats| formats.decode))]
        .into_iter().any(|(requested, resolved)| requested.is_some() && requested != resolved)
    {
        return Err(Refusal::PinConflict);
    }
    contract.validate()?;
    contract.validate_home_budgets(&budgets)?;
    // Validate physical budgets independently of conditional allocation choices.
    validate(
        &Facts { resident_weights: vec![],
            phase_order: contract.phase_order,
            allocations: vec![],
            deferred: vec![],
            sequential: vec![],
            reservations: vec![], allowances: vec![],
        },
        &budgets,
    )?;
    if request.context == 0 {
        return Err(Refusal::InvalidFact("zero context".into()));
    }
    let page = contract.page_rows;
    let multiple = contract.pool_page_multiple;
    let pool_rows = page.checked_mul(multiple).ok_or(Refusal::Overflow("pool row alignment"))?;
    if let Pin::Exact(tokens) = request.pool {
        if tokens % pool_rows != 0 {
            return Err(Refusal::MisalignedPool {
                tokens,
                page_rows: pool_rows,
            });
        }
        if tokens < request.context {
            return Err(Refusal::PoolBelowContext {
                tokens,
                context: request.context,
            });
        }
    }
    if matches!(request.prefill, Pin::Exact(s) if s.rows == 0 || s.waves == 0) {
        return Err(Refusal::InvalidPin("zero prefill shape"));
    }
    let minimum = request.context.div_ceil(pool_rows).checked_mul(multiple)
        .ok_or(Refusal::Overflow("minimum pool pages"))?;
    // KV target, min and max (`cuteafd_core::memory_policy`). The layout is
    // searched at the target; an automatic pool then grows to the max.
    // The card class picks the default target (a capped PRO keeps the PRO
    // target, as `PoolPolicy` and the live sample do).
    let smallest = budgets.iter().filter(|b| b.locations.iter()
        .any(|l| matches!(l.device, Device::Coordinator(_))))
        .map(|b| b.physical_bytes).min().unwrap_or(u64::MAX);
    let spark_free = !budgets.iter().any(|b| b.locations.iter().any(|l| matches!(l.device, Device::Spark(_))));
    let kv = cuteafd_core::memory_policy::MemoryPolicy::current().kv(smallest, request.context, spark_free)
        .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
    let aligned = |tokens: u64| tokens.div_ceil(page).div_ceil(multiple).checked_mul(multiple)
        .ok_or(Refusal::Overflow("target pool pages"));
    let compiled = contract.maximum_pages / multiple * multiple;
    let target = aligned(kv.target.max(request.context))?;
    let ceiling = aligned(kv.max.max(kv.target).max(request.context))?.min(compiled);
    let wanted = match request.pool {
        Pin::Auto => target.min(compiled),
        Pin::Exact(n) => n / page,
    };
    if wanted < minimum || wanted > contract.maximum_pages {
        return Err(Refusal::InvalidPin("pool exceeds compiled extent"));
    }
    let exact = match request.onboard {
        Pin::Auto => None,
        Pin::Exact(n) => Some(n),
    };
    let mut domains: Vec<Vec<usize>> = contract
        .variables
        .iter()
        .map(|v| (0..v.len()).collect())
        .collect();
    for (variable, domain) in contract.variables.iter().zip(&mut domains) {
        if let (Variable::Prefill(values), Pin::Exact(shape)) = (variable, request.prefill) {
            domain.retain(|&o| values[o] == shape);
        }
    }
    if domains.iter().any(Vec::is_empty) {
        return Err(Refusal::PinConflict);
    }
    // Logical pins do not depend on the pool. Target-specific capacity pruning
    // must not escape into the automatic below-target search.
    if !logical(&contract, &mut domains, exact) {
        return Err(Refusal::PinConflict);
    }
    let costs = Compiled::new(&contract, &budgets)?;
    let mut search = Search {
        contract: &contract,
        costs: &costs,
        budgets: &budgets,
        exact,
        assignments: 0,
        last: Refusal::Unsupported,
    };
    if let Some(fallback) = contract.floor_fallback {
        let domain = domains.get(fallback.variable)
            .filter(|domain| domain.contains(&fallback.option))
            .ok_or_else(|| Refusal::InvalidFact("invalid pool-floor fallback choice".into()))?;
        let mut preferred = domains.clone();
        preferred[fallback.variable] = domain.iter().copied()
            .filter(|&option| option != fallback.option).collect();
        let floor = if matches!(request.pool, Pin::Exact(_)) { wanted } else { minimum };
        if !preferred[fallback.variable].is_empty() && search.feasible(preferred.clone(), floor)? {
            domains = preferred;
        }
    }
    let pages = if search.feasible(domains.clone(), wanted)? {
        wanted
    } else if matches!(request.pool, Pin::Exact(_)) {
        return Err(search.last);
    } else {
        let mut maximum = None;
        if wanted > multiple {
            search.maximum(domains.clone(), multiple, wanted - multiple, &mut maximum)?;
        }
        maximum.unwrap_or(0)
    };
    if pages < minimum {
        // Reuse this search's validated facts and domains; never publish a
        // reduced-context plan or rebuild the family recipe with another context.
        if pages > 0 {
            let mut best = None;
            search.complete(domains.clone(), pages, &mut best)?;
            let (_, assignment) = best.ok_or_else(|| search.last.clone())?;
            // A pruning failure can describe a rejected alternative. Reprice
            // the best feasible layout at its first infeasible pool page.
            let next = pages + multiple;
            let facts = contract.facts(&assignment, next)?;
            search.last = match evaluate(&facts, &budgets) {
                Err(error) => error,
                Ok(_) => {
                    let fixed: Vec<_> = assignment.into_iter().map(|option| vec![option]).collect();
                    costs.capacity(&costs.lower_bound(&fixed, next, next)?, &budgets)
                        .err().ok_or_else(|| Refusal::InvalidFact("pool above maximum is feasible".into()))?
                }
            };
        } else {
            search.feasible(domains.clone(), pages + multiple)?;
        }
        if !matches!(search.last, Refusal::Capacity { .. }) {
            return Err(search.last);
        }
        return Err(Refusal::FullContextShortfall { context: request.context,
            tokens: pages.checked_mul(page).ok_or(Refusal::Overflow("feasible pool tokens"))?,
            limiting: Box::new(search.last) });
    }
    let mut best = None;
    search.complete(domains, pages, &mut best)?;
    tracing::trace!(target: "cuteafd::planner", surviving_assignments = search.assignments,
        components = contract.components.len(), preparation_search_us = started.elapsed().as_micros() as u64,
        "priced propagated contract assignments");
    let (_, selected) = best.ok_or(search.last)?;
    // The layout is fixed; an automatic pool grows into what it leaves.
    let pages = if matches!(request.pool, Pin::Auto) && ceiling > pages {
        costs.limit(&budgets, &selected, pages, ceiling)?.unwrap_or(pages).max(pages)
    } else { pages };
    // An explicit KV_MIN_TOKENS refuses a smaller automatic pool; unset, the
    // family's own context floor (`minimum`, above) is the refusal.
    let explicit_min = cuteafd_core::memory_policy::MemoryPolicy::current().kv_min;
    if matches!(request.pool, Pin::Auto) && explicit_min.is_some()
        && pages.saturating_mul(page) < kv.min.min(compiled.saturating_mul(page)) {
        return Err(Refusal::FullContextShortfall { context: kv.min,
            tokens: pages.checked_mul(page).ok_or(Refusal::Overflow("feasible pool tokens"))?,
            limiting: Box::new(Refusal::InvalidFact(format!("KV_MIN_TOKENS {} is above the admitted pool", kv.min))) });
    }
    let decisions = contract.decisions(&selected)?;
    let facts = contract.facts(&selected, pages)?;
    let phases = evaluate(&facts, &budgets)?;
    let weights = resident_weight_totals(&facts, contract.gpus)?;
    if weights != costs.weights(&selected, pages)? {
        return Err(Refusal::InvalidFact("component resource vectors differ from named resident weights".into()));
    }
    if let Some(limit) = request.resident_weight_limit {
        for (gpu, required) in weights.iter().enumerate() {
            if required.0 > limit.0 {
                return Err(Refusal::ResidentWeightLimit { gpu: gpu as u8, required: required.0, limit: limit.0 });
            }
        }
    }
    if phases != costs.phases(&budgets, &selected, pages)? {
        return Err(Refusal::InvalidFact(
            "component cost vectors differ from exact phase evaluator".into(),
        ));
    }
    // Diagnose alternatives at this exact solution. Do not shrink the pool or
    // rerun the optimizer merely to explain why Auto disabled image input.
    let mut encoder_diagnostics = Vec::new();
    for (variable, domain) in contract.variables.iter().enumerate() {
        if !matches!(domain, Variable::Encoder(_)) { continue; }
        for option in 0..selected[variable] {
            let mut alternative = selected.clone();
            alternative[variable] = option;
            if contract.requires.iter().any(|rule| alternative[rule.when.variable] == rule.when.option
                && !rule.any_of.iter().any(|choice| alternative[choice.variable] == choice.option)) {
                encoder_diagnostics.push((format!("encoder candidate {option}: no active Spark worker at the admitted onboarding"), 0));
                continue;
            }
            let alternate_facts = contract.facts(&alternative, pages)?;
            let mut refusal = evaluate(&alternate_facts, &budgets).err();
            if refusal.is_none() {
                if let Some(limit) = request.resident_weight_limit {
                    refusal = costs.weights(&alternative, pages)?.iter().enumerate().find_map(|(gpu, required)|
                        (required.0 > limit.0).then_some(Refusal::ResidentWeightLimit {
                            gpu: gpu as u8, required: required.0, limit: limit.0 }));
                }
            }
            if let Some(refusal) = refusal {
                let missing = match &refusal {
                    Refusal::Capacity { required, available, .. } => required.saturating_sub(*available),
                    Refusal::ResidentWeightLimit { required, limit, .. } => required.saturating_sub(*limit),
                    _ => 0,
                };
                encoder_diagnostics.push((format!("encoder candidate {option}: {refusal}; short by {missing} bytes"), missing));
            }
        }
    }
    Ok(ResolvedPlan {
        encoder_diagnostics,
        request,
        decisions,
        pool_tokens: pages
            .checked_mul(page)
            .ok_or(Refusal::Overflow("pool tokens"))?,
        budgets,
        facts,
        phases,
    })
}
