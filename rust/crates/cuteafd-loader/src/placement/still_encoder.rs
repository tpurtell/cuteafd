//! Packed still-tower owner choices for the existing joint admission solver.
use crate::{media::still_tower::StillTowerInventory,plan::{MediaMode,encoder::{self,EncoderHardware,EncoderPlacement,EncoderKind,EncoderRequest}}};
use super::planner::*;
use cuteafd_core::memory_layout::Category;

#[derive(Clone,Debug)]
pub struct StillEncoder {
    pub inventory: StillTowerInventory,
    pub candidates: Vec<EncoderPlacement>,
    pub media_cache: u64,
}
impl StillEncoder {
    pub fn planned(checkpoint: &crate::plan::Checkpoint, mode: MediaMode,
        replicas: usize, budgets: &[Budget], media_cache: u64) -> anyhow::Result<Option<Self>> {
        if mode == MediaMode::Off || checkpoint.config.get("vision_config").is_none() {
            return Ok(None);
        }
        let rows = match checkpoint.model_type() {
            Some("kimi_k3" | "kimi_k31") => crate::families::kimi_k3::vision::MAX_IMAGE_TOKENS as usize,
            Some("minimax_m3_vl") => crate::families::minimax_m3::vision::MAX_IMAGE_TOKENS,
            Some("inkling2" | "inkling2_loop" | "step5_vl") => 4096,
            Some("hy_v4_flash") => crate::families::hy4_flash::vision::MAX_MERGED_ROWS,
            Some("hy_v4_vl") => crate::families::hy4::vision::MAX_MERGED_ROWS,
            _ => anyhow::bail!("unsupported independent still tower"),
        };
        let inventory = StillTowerInventory::from_checkpoint(checkpoint, rows)?;
        let mut hardware = EncoderHardware { v41: false, gpus: Vec::new(), sparks: Vec::new() };
        for budget in budgets {
            for location in &budget.locations {
                match *location {
                    Location { device: Device::Coordinator(rank), domain: Domain::Device } => {
                        if usize::from(rank) != hardware.gpus.len() {
                            anyhow::bail!("still encoder coordinator budgets are not rank ordered");
                        }
                        hardware.gpus.push(encoder::EncoderGpuBudget { free_bytes: budget.available()?, kv_target_bytes: 0 });
                    }
                    Location { device: Device::Spark(rank), domain: Domain::Managed } => {
                        hardware.sparks.push(encoder::EncoderSparkBudget { rank: usize::from(rank),
                            host: format!("spark{rank}"), idle: false, expert_bytes: 0, free_bytes: budget.available()? });
                    }
                    _ => {}
                }
            }
        }
        Ok(Some(Self::new(inventory, mode, replicas, &hardware, media_cache)?))
    }

    pub fn new(inventory: StillTowerInventory, mode: MediaMode, replicas: usize,
        hardware: &EncoderHardware, media_cache: u64) -> Result<Self,Refusal> {
        let request=EncoderRequest {mode,weights:inventory.arena.bytes() as u64,
            scratch:inventory.scratch,copies:replicas};
        let mut candidates=encoder::candidates(&request,hardware);
        if mode==MediaMode::Auto && request.weights>0 {
            candidates.push(encoder::encoder_placement(MediaMode::Off,hardware,0,0,1));
        }
        if candidates.is_empty() {return Err(Refusal::InvalidFact("still encoder has no candidate device".into()));}
        Ok(Self {inventory,candidates,media_cache})
    }

    /// Append one encoder variable. Each candidate charges a single packed
    /// backing plus scratch; views never become independent allocations.
    pub fn extend(&self,variables:&mut Vec<Variable>,components:&mut Vec<Component>,maximum:u64)
        -> Result<(),Refusal> {
        let variable=variables.len();
        variables.push(Variable::Encoder(self.candidates.iter().map(EncoderPlacement::vision_on).collect()));
        for (option,placement) in self.candidates.iter().enumerate() {
            let locations=match placement.kind {
                EncoderKind::Off=>Vec::new(),
                EncoderKind::Rtx {gpu}=>vec![Location::coordinator(u8::try_from(gpu)
                    .map_err(|_| Refusal::Overflow("encoder GPU rank"))?)],
                EncoderKind::Spark {..}=>placement.spark_ranks().into_iter().map(|rank|
                    u8::try_from(rank).map(|rank|Location {device:Device::Spark(rank),domain:Domain::Managed})
                        .map_err(|_|Refusal::Overflow("encoder Spark rank"))).collect::<Result<Vec<_>,_>>()?,
                EncoderKind::SparkIdle {..}=>return Err(Refusal::InvalidFact("idle external still encoder has no admitted budget".into())),
            };
            let mut owners=Vec::new();
            for (copy,location) in locations.into_iter().enumerate() {
                let backing=format!("{}.candidate{option}.copy{copy}",self.inventory.arena.id());
                let bytes=self.inventory.arena.bytes() as u64;
                let root=Allocation::buffer(&backing,location,bytes,Lifetime::RESIDENT,Category::Weights,crate::media::still_tower::vision_weight_arena_label());
                owners.push(Component::allocation(root.clone(),ByteCurve::fixed(bytes,maximum),false,maximum));
                for (name, view) in self.inventory.arena.views() {
                    let mut alias = Allocation::buffer(format!("{backing}.{name}"), location,
                        view.bytes() as u64, Lifetime::RESIDENT, Category::Weights, "vision tensor view");
                    alias.view = Some(View { backing: backing.clone(), offset: view.offset() as u64 });
                    owners.push(Component::allocation(alias, ByteCurve::fixed(view.bytes() as u64, maximum), false, maximum));
                }
                if matches!(location.device,Device::Coordinator(_)) {
                    owners.push(Component::resident_weight(ResidentWeight::from_allocation(&root,
                        super::ResidentWeights(bytes)),ByteCurve::fixed(bytes,maximum)));
                    if let Some(tail)=exact_owner_tail(format!("{backing}.tail"),backing.clone(),location,
                        Lifetime::RESIDENT,vec![bytes],maximum)? {owners.push(tail);}
                }
                let scratch=Allocation::buffer(format!("{backing}.scratch"),location,self.inventory.scratch,
                    Lifetime::RESIDENT,Category::Workspace,"vision scratch");
                owners.push(Component::allocation(scratch,ByteCurve::fixed(self.inventory.scratch,maximum),false,maximum));
                if matches!(location.device,Device::Coordinator(_)) {
                    if let Some(tail)=exact_owner_tail(format!("{backing}.scratch.tail"),format!("{backing}.scratch"),location,
                        Lifetime::RESIDENT,vec![self.inventory.scratch],maximum)? {owners.push(tail);}
                }
            }
            if placement.kind!=EncoderKind::Off && self.media_cache>0 {
                owners.push(Component::allocation(Allocation::buffer(format!("{}.media.{option}",self.inventory.arena.id()),
                    Location::HOST,self.media_cache,Lifetime::TARGET,Category::Embedding,"media/embedding-cache"),
                    ByteCurve::fixed(self.media_cache,maximum),false,maximum));
            }
            for mut owner in owners {owner.activation=Activation::choice(variable,option);components.push(owner);}
        }
        Ok(())
    }

    pub fn require_spark_worker(&self, variables: &[Variable], requires: &mut Vec<Requires>) {
        let Some(encoder) = variables.iter().position(|v| matches!(v, Variable::Encoder(_))) else { return; };
        for (option, placement) in self.candidates.iter().enumerate() {
            for rank in placement.spark_ranks() {
                let mut retained = Vec::new();
                // Families here have one ordered onboarding prefix shared by every Spark.
                for (variable, domain) in variables.iter().enumerate() {
                    if let Variable::Onboard { layers, counts, .. } = domain {
                        retained.extend(counts.iter().enumerate().filter(|(_, count)| **count < layers.len())
                            .map(|(option, _)| Choice { variable, option }));
                    }
                }
                if !variables.iter().any(|domain| matches!(domain, Variable::Onboard { .. })) {
                    for (variable, domain) in variables.iter().enumerate() {
                        if let Variable::ExpertHome { layers, rank_sets } = domain {
                            if !layers.is_empty() {
                                retained.extend(rank_sets.iter().enumerate().filter(|(_, ranks)| ranks.contains(&(rank as u8)))
                                    .map(|(option, _)| Choice { variable, option }));
                            }
                        }
                    }
                }
                requires.push(Requires { when: Choice { variable: encoder, option }, any_of: retained });
            }
        }
    }
    pub fn admitted(&self,solved:&ResolvedPlan)->Option<EncoderPlacement> {
        solved.decisions.encoder.and_then(|index|self.candidates.get(index).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{media::still_tower::StillTowerKind,serving_capacity::WeightArenaLayout};
    fn inventory(kind:StillTowerKind)->StillTowerInventory {
        StillTowerInventory {kind,arena:WeightArenaLayout::new("vision/fixture",[("matrix".into(),4<<20),("rotary".into(),128)]).unwrap(),scratch:1<<20,max_output_rows:16698}
    }
    #[test]
    fn both_families_use_shared_spark_first_and_rtx_fallback_with_one_arena() {
        use crate::plan::encoder::{EncoderGpuBudget,EncoderSparkBudget};
        for kind in [StillTowerKind::Kimi,StillTowerKind::MiniMax,StillTowerKind::Inkling2,StillTowerKind::Step5] {
            for spark_free in [u64::MAX,0] {
                let hardware=EncoderHardware {v41:false,gpus:vec![EncoderGpuBudget {free_bytes:u64::MAX,kv_target_bytes:0}],
                    sparks:vec![EncoderSparkBudget {rank:0,host:"fixture".into(),idle:false,expert_bytes:0,free_bytes:spark_free}]};
                let plan=StillEncoder::new(inventory(kind),MediaMode::Auto,1,&hardware,0).unwrap();
                assert_eq!(plan.candidates[0].kind,if spark_free>0 {EncoderKind::Spark {rank:0}} else {EncoderKind::Rtx {gpu:0}});
                let mut components=Vec::new();let mut variables=Vec::new();plan.extend(&mut variables,&mut components,16).unwrap();
                for option in 0..plan.candidates.len() {
                    let roots=components.iter().filter_map(|c| match &c.kind {ComponentKind::Allocation {template,..}
                        if template.id.ends_with(&format!("candidate{option}.copy0"))=>Some(template),_=>None}).collect::<Vec<_>>();
                    assert_eq!(roots.len(),usize::from(plan.candidates[option].kind!=EncoderKind::Off));
                }
            }
        }
    }
    #[test]
    fn auto_towers_and_explicit_off_share_the_same_family_independent_candidates() {
        use crate::plan::encoder::EncoderGpuBudget;
        let hardware = EncoderHardware { v41: false,
            gpus: vec![EncoderGpuBudget { free_bytes: 96 << 30, kv_target_bytes: 0 }], sparks: vec![] };
        for kind in [StillTowerKind::Inkling2, StillTowerKind::MiniMax] {
            let auto = StillEncoder::new(inventory(kind), MediaMode::Auto, 1, &hardware, 0).unwrap();
            assert_eq!(auto.candidates[0].kind, EncoderKind::Rtx { gpu: 0 });
            let off = StillEncoder::new(inventory(kind), MediaMode::Off, 1, &hardware, 0).unwrap();
            assert_eq!(off.candidates.len(), 1);
            assert_eq!(off.candidates[0].kind, EncoderKind::Off);
            assert!(off.candidates[0].reason.contains("disabled"));
        }
        let absent = encoder::encoder_placement(MediaMode::Auto, &hardware, 0, 0, 1);
        assert_eq!(absent.kind, EncoderKind::Off);
        assert!(absent.reason.contains("no checkpoint tower"));
    }

    #[test]
    fn joint_solve_retries_rtx_when_spark_experts_leave_no_tower_room() {
        use crate::plan::encoder::{EncoderGpuBudget, EncoderSparkBudget};
        struct Prepared(Contract);
        impl Recipe for Prepared {
            fn prepare(&self, _: &Request) -> Result<Contract, Refusal> { Ok(self.0.clone()) }
        }
        for kind in [StillTowerKind::Kimi, StillTowerKind::MiniMax, StillTowerKind::Inkling2, StillTowerKind::Step5] {
            let hardware = EncoderHardware { v41: false,
                gpus: vec![EncoderGpuBudget { free_bytes: 64 << 20, kv_target_bytes: 0 }],
                sparks: vec![EncoderSparkBudget { rank: 0, host: "fixture".into(), idle: false,
                    expert_bytes: 0, free_bytes: 64 << 20 }] };
            let encoder = StillEncoder::new(inventory(kind), MediaMode::Auto, 1, &hardware, 0).unwrap();
            for expert_bytes in [1 << 20, 60 << 20] {
                let location = Location { device: Device::Spark(0), domain: Domain::Managed };
                let mut contract = Contract::fixed(64, 1, 1, Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT,
                    allocations: vec![Allocation::buffer("experts", location, expert_bytes,
                        Lifetime::RESIDENT, Category::Experts, "routed experts")],
                    deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
                    resident_weights: vec![], allocation_granularity: vec![], driver_page_tables: vec![] });
                contract.variables.extend([Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]),
                    Variable::Lanes(vec![1])]);
                contract.variables.push(Variable::ExpertHome { layers: vec![0], rank_sets: vec![vec![0]] });
                encoder.extend(&mut contract.variables, &mut contract.components, 1).unwrap();
                encoder.require_spark_worker(&contract.variables, &mut contract.requires);
                let budgets = [Location::coordinator(0), location].into_iter().enumerate().map(|(i, location)|
                    Budget { id: format!("device{i}"), locations: vec![location], physical_bytes: 64 << 20,
                        logical_bytes: 64 << 20, headroom: 0,
                        baseline: Baseline::Forecast { occupied: 0, source: "test".into(), assumed: false } }).collect();
                let solved = solve(Request { context: 64.into(), pool: Pin::Exact(64), onboard: Pin::Auto,
                    prefill: Pin::Auto, operation: Operation::Serving, resident_weight_limit: None,
                    spark_reply_formats: Default::default() }, budgets, &Prepared(contract)).unwrap();
                let expected = if expert_bytes == 1 << 20 { EncoderKind::Spark { rank: 0 } }
                    else { EncoderKind::Rtx { gpu: 0 } };
                assert_eq!(encoder.admitted(&solved).unwrap().kind, expected);
                let roots = solved.facts.allocations.iter().filter(|a| a.group == crate::media::still_tower::vision_weight_arena_label())
                    .collect::<Vec<_>>();
                assert_eq!(roots.len(), 1);
                assert_eq!(roots[0].bytes, encoder.inventory.arena.bytes() as u64);
                let aliases = solved.facts.allocations.iter().filter(|a| a.view.as_ref()
                    .is_some_and(|v| v.backing == roots[0].id)).count();
                assert_eq!(aliases, encoder.inventory.arena.views().len());
                let tails = solved.facts.allocation_granularity.iter().filter(|tail| tail.backings.iter()
                    .any(|b| b.backing == roots[0].id)).count();
                assert_eq!(tails, usize::from(matches!(expected, EncoderKind::Rtx { .. })));
            }
        }
    }

}
