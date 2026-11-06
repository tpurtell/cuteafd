//! Immutable component contracts and finite decision domains, prepared once.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Choice {
    pub variable: usize,
    pub option: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Activation {
    /// Any clause activates the component; all choices in a clause must hold.
    /// An empty clause is unconditional, while no clauses means never.
    pub clauses: Vec<Vec<Choice>>,
}
impl Activation {
    pub fn always() -> Self {
        Self {
            clauses: vec![vec![]],
        }
    }
    pub fn choice(variable: usize, option: usize) -> Self {
        Self {
            clauses: vec![vec![Choice { variable, option }]],
        }
    }
    pub(super) fn active(&self, selected: &[usize]) -> bool {
        self.clauses
            .iter()
            .any(|clause| clause.iter().all(|c| selected[c.variable] == c.option))
    }
    pub(super) fn guaranteed(&self, domains: &[Vec<usize>]) -> bool {
        self.clauses
            .iter()
            .any(|clause| clause.iter().all(|c| domains[c.variable] == [c.option]))
    }
    fn overlaps(&self, other: &Self) -> bool {
        self.clauses.iter().any(|left| other.clauses.iter().any(|right| {
            let mut choices = BTreeMap::new();
            left.iter().chain(right).all(|choice| choices.insert(choice.variable, choice.option)
                .is_none_or(|old| old == choice.option))
        }))
    }
    pub(super) fn local_variable(&self) -> Option<usize> {
        let variable = self.clauses.first()?.first()?.variable;
        self.clauses
            .iter()
            .all(|c| c.len() == 1 && c[0].variable == variable)
            .then_some(variable)
    }
}

#[derive(Debug, Clone)]
pub enum Variable {
    Attention {
        layer: usize,
        values: Vec<LayerMode>,
    },
    Prefill(Vec<PrefillShape>),
    Lanes(Vec<u64>),
    Embedding(Vec<EmbeddingHome>),
    /// Ordered coordinator homes for one drafter; lower index is preferred.
    DraftHome(Vec<u8>),
    /// Ordered optional bundled-encoder candidates; lower index is preferred.
    Encoder(Vec<usize>),
    /// Fallback rank set for this routed group, after the selected ordered
    /// onboarding prefixes have claimed their local layers. No per-layer
    /// optional local home is introduced.
    ExpertHome {
        layers: Vec<usize>,
        rank_sets: Vec<Vec<u8>>,
    },
    /// One ordered count profile per destination. Count k owns layers[..k];
    /// the order and its exact costs are family facts, not a subset search.
    /// TP2 destinations name both GPUs but count each layer only once.
    Onboard {
        layers: Vec<usize>,
        destination: Vec<u8>,
        counts: Vec<usize>,
    },
}
impl Variable {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Attention { values, .. } => values.len(),
            Self::Prefill(values) => values.len(),
            Self::Lanes(values) => values.len(),
            Self::Embedding(values) => values.len(),
            Self::DraftHome(values) => values.len(),
            Self::Encoder(values) => values.len(),
            Self::ExpertHome { rank_sets, .. } => rank_sets.len(),
            Self::Onboard { counts, .. } => counts.len(),
        }
    }
    pub(super) fn onboard(&self, option: usize) -> u64 {
        match self {
            Self::Onboard { counts, .. } => counts[option] as u64,
            _ => 0,
        }
    }
    pub(super) fn onboard_layers(&self, option: usize) -> &[usize] {
        match self {
            Self::Onboard { layers, counts, .. } => &layers[..counts[option]],
            _ => &[],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Requires {
    pub when: Choice,
    pub any_of: Vec<Choice>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BytePiece {
    pub first: u64,
    pub last: u64,
    pub bytes_at_first: u64,
    pub bytes_per_page: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteCurve(pub Vec<BytePiece>);
impl ByteCurve {
    pub fn fixed(bytes: u64, maximum: u64) -> Self {
        Self::linear(bytes, 0, maximum)
    }
    pub fn linear(bytes_at_first: u64, bytes_per_page: u64, maximum: u64) -> Self {
        Self(vec![BytePiece {
            first: 1,
            last: maximum,
            bytes_at_first,
            bytes_per_page,
        }])
    }
    pub fn at(&self, pages: u64) -> Result<u64, Refusal> {
        let piece = self.piece(pages)?;
        piece
            .bytes_per_page
            .checked_mul(pages - piece.first)
            .and_then(|n| piece.bytes_at_first.checked_add(n))
            .ok_or(Refusal::Overflow("component curve"))
    }
    pub(super) fn piece(&self, pages: u64) -> Result<&BytePiece, Refusal> {
        self.0
            .iter()
            .find(|p| p.first <= pages && pages <= p.last)
            .ok_or_else(|| Refusal::InvalidFact("component curve has no page interval".into()))
    }
    pub(super) fn validate(&self, maximum: u64) -> Result<(), Refusal> {
        let mut next = 1;
        for piece in &self.0 {
            if piece.first != next || piece.first > piece.last || piece.last > maximum {
                return Err(Refusal::InvalidFact(
                    "component curve has a gap, overlap or invalid interval".into(),
                ));
            }
            self.at(piece.last)?;
            next = piece
                .last
                .checked_add(1)
                .ok_or(Refusal::Overflow("curve interval"))?;
        }
        if next
            != maximum
                .checked_add(1)
                .ok_or(Refusal::Overflow("maximum pages"))?
        {
            return Err(Refusal::InvalidFact(
                "component curve does not cover page domain".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ComponentKind {
    Allocation {
        template: Allocation,
        shape: Vec<ByteCurve>,
        offset: Option<ByteCurve>,
        deferred: bool,
    },
    Sequential(Sequential),
    Reservation(Reservation),
    Allowance(Allowance),
    /// Weight residency can outlive or precede the future allocation ledger.
    ResidentWeight(ResidentWeight),
}
#[derive(Debug, Clone)]
pub struct Component {
    pub activation: Activation,
    pub bytes: ByteCurve,
    pub kind: ComponentKind,
}
impl Component {
    pub fn allocation(
        template: Allocation,
        bytes: ByteCurve,
        deferred: bool,
        maximum: u64,
    ) -> Self {
        let shape = template
            .shape
            .iter()
            .map(|&n| ByteCurve::fixed(n, maximum))
            .collect();
        let offset = template
            .view
            .as_ref()
            .map(|v| ByteCurve::fixed(v.offset, maximum));
        Self {
            activation: Activation::always(),
            bytes,
            kind: ComponentKind::Allocation {
                template,
                shape,
                offset,
                deferred,
            },
        }
    }
    pub fn resident_weight(owner: ResidentWeight, bytes: ByteCurve) -> Self {
        Self {
            activation: Activation::always(),
            bytes,
            kind: ComponentKind::ResidentWeight(owner),
        }
    }

    /// Raw byte buffers keep their sole shape dimension equal to their extent.
    pub fn byte_extent(&mut self, bytes: ByteCurve) -> Result<(), Refusal> {
        let ComponentKind::Allocation {
            template, shape, ..
        } = &mut self.kind
        else {
            return Err(Refusal::InvalidFact("byte extent for an allowance".into()));
        };
        if template.dtype != "bytes" || shape.len() != 1 {
            return Err(Refusal::InvalidFact(
                "byte extent for a typed tensor".into(),
            ));
        }
        *shape = vec![bytes.clone()];
        self.bytes = bytes;
        Ok(())
    }
    pub(super) fn charge(&self) -> Option<(Location, Lifetime, bool)> {
        match &self.kind {
            ComponentKind::Allocation { template, .. } if template.view.is_none() => {
                Some((template.location, template.lifetime, false))
            }
            ComponentKind::Allocation { .. } => None,
            ComponentKind::Sequential(group) => group
                .members
                .first()
                .map(|a| (a.location, a.lifetime, false)),
            ComponentKind::Reservation(r) => Some((r.location, r.lifetime, false)),
            ComponentKind::Allowance(a) => Some((a.location, a.lifetime, true)),
            ComponentKind::ResidentWeight(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Contract {
    pub phase_order: PhaseOrder,
    /// Family-resolved once, before components price the selected reply storage.
    pub spark_reply_formats: Option<SparkReplyFormats>,
    pub resident_weight_limit: Option<ResidentWeights>,
    pub page_rows: u64,
    pub maximum_pages: u64,
    pub gpus: usize,
    pub variables: Vec<Variable>,
    pub requires: Vec<Requires>,
    pub components: Vec<Component>,
}
impl Contract {
    pub fn fixed(page_rows: u64, maximum_pages: u64, gpus: usize, facts: Facts) -> Self {
        let mut components: Vec<_> = facts
            .allocations
            .into_iter()
            .map(|a| {
                Component::allocation(
                    a.clone(),
                    ByteCurve::fixed(a.bytes, maximum_pages),
                    false,
                    maximum_pages,
                )
            })
            .chain(facts.deferred.into_iter().map(|a| {
                Component::allocation(
                    a.clone(),
                    ByteCurve::fixed(a.bytes, maximum_pages),
                    true,
                    maximum_pages,
                )
            }))
            .collect();
        components.extend(facts.sequential.into_iter().map(|group| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(group.peak(), maximum_pages),
            kind: ComponentKind::Sequential(group),
        }));
        components.extend(facts.reservations.into_iter().map(|r| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(r.bytes, maximum_pages),
            kind: ComponentKind::Reservation(r),
        }));
        components.extend(facts.allowances.into_iter().map(|a| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(a.bytes, maximum_pages),
            kind: ComponentKind::Allowance(a),
        }));
        components.extend(facts.resident_weights.into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.weights.0, maximum_pages);
            Component::resident_weight(owner, bytes)
        }));
        Self {
            phase_order: facts.phase_order,
            spark_reply_formats: None,
            resident_weight_limit: None,
            page_rows,
            maximum_pages,
            gpus,
            variables: vec![],
            requires: vec![],
            components,
        }
    }
    pub fn set_byte_extent(&mut self, id: &str, curve: ByteCurve) -> Result<(), Refusal> {
        let component = self
            .components
            .iter_mut()
            .find(|c| {
                matches!(&c.kind,
            ComponentKind::Allocation { template, .. } if template.id == id)
            })
            .ok_or_else(|| Refusal::InvalidFact(format!("unknown component {id}")))?;
        component.byte_extent(curve)
    }
    pub fn facts(&self, selected: &[usize], pages: u64) -> Result<Facts, Refusal> {
        self.decisions(selected)?;
        let mut result = Facts { resident_weights: vec![],
            phase_order: self.phase_order,
            allocations: vec![],
            deferred: vec![],
            sequential: vec![],
            reservations: vec![], allowances: vec![],
        };
        for component in &self.components {
            if !component.activation.active(selected) {
                continue;
            }
            let bytes = component.bytes.at(pages)?;
            if bytes == 0 && !matches!(component.kind, ComponentKind::ResidentWeight(_)) {
                continue;
            }
            match &component.kind {
                ComponentKind::Allocation {
                    template,
                    shape,
                    offset,
                    deferred,
                } => {
                    let mut allocation = template.clone();
                    allocation.bytes = bytes;
                    allocation.shape = shape
                        .iter()
                        .map(|curve| curve.at(pages))
                        .collect::<Result<_, _>>()?;
                    if let (Some(view), Some(offset)) = (&mut allocation.view, offset) {
                        view.offset = offset.at(pages)?;
                    }
                    if *deferred {
                        result.deferred.push(allocation);
                    } else {
                        result.allocations.push(allocation);
                    }
                }
                ComponentKind::Sequential(group) => {
                    if bytes != group.peak() {
                        return Err(Refusal::InvalidFact(
                            "sequential peak differs from members".into(),
                        ));
                    }
                    result.sequential.push(group.clone());
                }
                ComponentKind::Reservation(template) => {
                    let mut reservation = template.clone();
                    reservation.bytes = bytes;
                    result.reservations.push(reservation);
                }
                ComponentKind::Allowance(template) => {
                    let mut allowance = template.clone();
                    allowance.bytes = bytes;
                    result.allowances.push(allowance);
                }
                ComponentKind::ResidentWeight(template) => {
                    let mut owner = template.clone();
                    owner.weights = ResidentWeights(bytes);
                    result.resident_weights.push(owner);
                }
            }
        }
        Ok(result)
    }
    pub fn decisions(&self, selected: &[usize]) -> Result<Decisions, Refusal> {
        if selected.len() != self.variables.len()
            || selected
                .iter()
                .zip(&self.variables)
                .any(|(&o, v)| o >= v.len())
        {
            return Err(Refusal::InvalidFact("invalid contract assignment".into()));
        }
        let mut attention = BTreeMap::new();
        let mut prefill = None;
        let mut lanes = None;
        let mut onboard_layers = vec![vec![]; self.gpus];
        let mut expert_homes = BTreeMap::new();
        let mut remote_homes = BTreeMap::new();
        let mut embedding = None;
        let mut draft_home = None;
        let mut encoder = None;
        for (variable, &option) in self.variables.iter().zip(selected) {
            match variable {
                Variable::Attention { layer, values } => {
                    attention.insert(*layer, values[option]);
                }
                Variable::Prefill(values) => {
                    prefill = Some(values[option]);
                }
                Variable::Lanes(values) => {
                    lanes = Some(values[option]);
                }
                Variable::Embedding(values) => {
                    if embedding.replace(values[option]).is_some() {
                        return Err(Refusal::InvalidFact("duplicate embedding domain".into()));
                    }
                }
                Variable::DraftHome(values) => {
                    if draft_home.replace(values[option]).is_some() {
                        return Err(Refusal::InvalidFact("duplicate draft home domain".into()));
                    }
                }
                Variable::Encoder(values) => {
                    if encoder.replace(values[option]).is_some() {
                        return Err(Refusal::InvalidFact("duplicate encoder domain".into()));
                    }
                }
                Variable::ExpertHome { layers, rank_sets } => {
                    for &layer in layers {
                        if remote_homes.insert(layer, ExpertDestination::Sparks {
                            ranks: rank_sets[option].clone(),
                        }).is_some() {
                            return Err(Refusal::InvalidFact("overlapping remote expert groups".into()));
                        }
                    }
                }
                Variable::Onboard { destination, .. } => {
                    let layers = variable.onboard_layers(option);
                    for &layer in layers {
                        if expert_homes.insert(layer, ExpertDestination::Coordinator {
                            gpus: destination.clone(),
                        }).is_some() {
                            return Err(Refusal::PinConflict);
                        }
                    }
                    for &gpu in destination {
                        onboard_layers[gpu as usize].extend(layers);
                    }
                }
            }
        }
        if attention.keys().copied().ne(0..attention.len()) {
            return Err(Refusal::InvalidFact(
                "noncontiguous attention layers".into(),
            ));
        }
        for layers in &mut onboard_layers {
            layers.sort_unstable();
        }
        for (layer, home) in remote_homes {
            expert_homes.entry(layer).or_insert(home);
        }
        Ok(Decisions {
            spark_reply_formats: self.spark_reply_formats,
            attention: attention.into_values().collect(),
            onboard_layers,
            expert_homes,
            embedding,
            draft_home,
            encoder,
            prefill: prefill
                .ok_or_else(|| Refusal::InvalidFact("missing prefill domain".into()))?,
            lanes: lanes.ok_or_else(|| Refusal::InvalidFact("missing lane domain".into()))?,
        })
    }
    pub(super) fn validate_home_budgets(&self, budgets: &[Budget]) -> Result<(), Refusal> {
        let covered = |location| budgets.iter().any(|budget| budget.locations.contains(&location));
        for component in &self.components {
            if let ComponentKind::ResidentWeight(owner) = &component.kind {
                if !covered(owner.location) {
                    return Err(Refusal::InvalidFact(format!("resident weight {} has no physical budget", owner.id)));
                }
            }
        }
        for variable in &self.variables {
            match variable {
                Variable::Embedding(values) => {
                    if values.iter().any(|home| !covered(home.location())) {
                        return Err(Refusal::InvalidFact("embedding home has no physical budget".into()));
                    }
                }
                Variable::DraftHome(values) => {
                    if values.iter().any(|&gpu| !covered(Location::coordinator(gpu))) {
                        return Err(Refusal::InvalidFact("draft home has no physical budget".into()));
                    }
                }
                Variable::ExpertHome { rank_sets, .. } => {
                    if rank_sets.iter().flatten().any(|&rank| !budgets.iter().any(|budget| {
                        budget.locations.iter().any(|location| location.device == Device::Spark(rank)
                            && matches!(location.domain, Domain::Device | Domain::Managed))
                    })) {
                        return Err(Refusal::InvalidFact("expert rank has no physical budget".into()));
                    }
                }
                Variable::Onboard { destination, .. } => {
                    if destination.iter().any(|&gpu| !covered(Location::coordinator(gpu))) {
                        return Err(Refusal::InvalidFact("local expert home has no physical budget".into()));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub(super) fn validate(&self) -> Result<(), Refusal> {
        if self.page_rows == 0
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.gpus == 0
        {
            return Err(Refusal::InvalidFact(
                "zero or overflowing contract extent".into(),
            ));
        }
        let mut attention = BTreeSet::new();
        let mut remote_layers = BTreeSet::new();
        let (mut prefill, mut lanes, mut embedding) = (0, 0, 0);
        let mut encoder = false;
        let mut draft_home = false;
        for variable in &self.variables {
            if variable.len() == 0 {
                return Err(Refusal::Unsupported);
            }
            match variable {
                Variable::Attention { layer, values } => {
                    if !attention.insert(*layer)
                        || values.iter().any(|v| {
                            matches!(v,
                        LayerMode::Whole { gpu, .. } if usize::from(*gpu) >= self.gpus)
                        })
                    {
                        return Err(Refusal::InvalidFact("invalid attention domain".into()));
                    }
                }
                Variable::Prefill(values) => {
                    prefill += 1;
                    if values.iter().any(|v| v.rows == 0 || v.waves == 0) {
                        return Err(Refusal::InvalidPin("zero prefill shape"));
                    }
                }
                Variable::Lanes(values) => {
                    lanes += 1;
                    if values.contains(&0) {
                        return Err(Refusal::InvalidFact("zero lanes".into()));
                    }
                }
                Variable::Embedding(values) => {
                    embedding += 1;
                    if values.iter().enumerate().any(|(index, home)| {
                        values[..index].contains(home)
                            || matches!(home, EmbeddingHome::Device(gpu) if usize::from(*gpu) >= self.gpus)
                    }) {
                        return Err(Refusal::InvalidFact("invalid embedding domain".into()));
                    }
                }
                Variable::DraftHome(values) => {
                    if draft_home || values.iter().enumerate().any(|(index, &gpu)|
                        usize::from(gpu) >= self.gpus || values[..index].contains(&gpu)) {
                        return Err(Refusal::InvalidFact("invalid draft home domain".into()));
                    }
                    draft_home = true;
                }
                Variable::Encoder(values) => {
                    if encoder || values.iter().copied().ne(0..values.len()) {
                        return Err(Refusal::InvalidFact("invalid encoder domain".into()));
                    }
                    encoder = true;
                }
                Variable::ExpertHome { layers, rank_sets } => {
                    if layers.is_empty()
                        || layers.iter().any(|&layer| !remote_layers.insert(layer))
                        || rank_sets.iter().enumerate().any(|(index, ranks)| {
                            ranks.is_empty() || ranks.windows(2).any(|pair| pair[0] >= pair[1])
                                || rank_sets[..index].contains(ranks)
                        })
                    {
                        return Err(Refusal::InvalidFact("invalid remote expert domain".into()));
                    }
                }
                Variable::Onboard {
                    layers,
                    destination,
                    counts,
                } => {
                    if layers.is_empty()
                        || layers.iter().collect::<BTreeSet<_>>().len() != layers.len()
                        || destination.is_empty()
                        || destination.iter().any(|&g| usize::from(g) >= self.gpus)
                        || destination.windows(2).any(|pair| pair[0] >= pair[1])
                        || counts.iter().any(|&k| k > layers.len())
                        || counts.windows(2).any(|pair| pair[0] >= pair[1])
                    {
                        return Err(Refusal::InvalidFact("invalid onboard count profile".into()));
                    }
                }
            }
        }
        if prefill != 1 || lanes != 1 || embedding > 1 {
            return Err(Refusal::InvalidFact(
                "prefill/lane/embedding domains must be unique".into(),
            ));
        }
        let choice_valid = |c: &Choice| {
            self.variables
                .get(c.variable)
                .is_some_and(|v| c.option < v.len())
        };
        for constraint in &self.requires {
            if !choice_valid(&constraint.when) || !constraint.any_of.iter().all(&choice_valid) {
                return Err(Refusal::InvalidFact(
                    "invalid compatibility constraint".into(),
                ));
            }
        }
        for component in &self.components {
            if !component
                .activation
                .clauses
                .iter()
                .flatten()
                .all(&choice_valid)
            {
                return Err(Refusal::InvalidFact("invalid component activation".into()));
            }
            component.bytes.validate(self.maximum_pages)?;
            if component.charge().is_some_and(|(_, life, _)|
                !self.phase_order.at_or_before(life.first, life.last)) {
                return Err(Refusal::InvalidFact("component lifetime contradicts family phase order".into()));
            }
            if let ComponentKind::ResidentWeight(owner) = &component.kind {
                if owner.id.is_empty() || matches!(owner.location.device,
                    Device::Coordinator(gpu) if usize::from(gpu) >= self.gpus)
                    || !matches!(owner.location,
                        Location { device: Device::Coordinator(_) | Device::Spark(_), domain: Domain::Device | Domain::Managed }
                        | Location { device: Device::Host, domain: Domain::Host | Domain::Pinned })
                {
                    return Err(Refusal::InvalidFact(format!("invalid resident weight {}", owner.id)));
                }
            }
            if let ComponentKind::Reservation(reservation) = &component.kind {
                let facts = Facts { resident_weights: vec![], phase_order: self.phase_order, allocations: vec![], deferred: vec![],
                    sequential: vec![], reservations: vec![reservation.clone()], allowances: vec![] };
                validate_owners(&facts)?;
            }
            if let ComponentKind::Sequential(group) = &component.kind {
                group.validate(self.phase_order)?;
                if component.bytes != ByteCurve::fixed(group.peak(), self.maximum_pages) {
                    return Err(Refusal::InvalidFact("invalid sequential peak curve".into()));
                }
            }
            if let ComponentKind::Allocation { shape, offset, .. } = &component.kind {
                for curve in shape.iter().chain(offset) {
                    curve.validate(self.maximum_pages)?;
                }
            }
        }
        let mut allocations = BTreeMap::<&str, Vec<&Component>>::new();
        for component in &self.components {
            if let ComponentKind::Allocation { template, .. } = &component.kind {
                allocations.entry(&template.id).or_default().push(component);
            }
        }
        let mut weights = BTreeMap::<&str, Vec<&Component>>::new();
        for component in &self.components {
            let ComponentKind::ResidentWeight(owner) = &component.kind else { continue };
            let peers = weights.entry(&owner.id).or_default();
            if peers.iter().any(|other| component.activation.overlaps(&other.activation)) {
                return Err(Refusal::InvalidFact(format!("overlapping resident weight {}", owner.id)));
            }
            peers.push(component);
            for other in allocations.get(owner.id.as_str()).into_iter().flatten() {
                if let ComponentKind::Allocation { template, .. } = &other.kind {
                    if component.activation.overlaps(&other.activation)
                        && (template.view.is_some() || template.location != owner.location)
                    {
                        return Err(Refusal::InvalidFact(format!("resident weight {} is not a physical backing", owner.id)));
                    }
                }
            }
        }
        Ok(())
    }
}
