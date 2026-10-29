//! Inert K6 projection owners and projected attention/FFN seams.
//! Serving admission remains closed until full-layer and TP2 gates qualify it.
use crate::shared::dense_trellis::{CoreWeightLayout, CoreWeights, Projection, ProjectionLayout, ProjectionScratch};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer,programs::Programs,
    dense_trellis::{TrellisDType,
        GroupedMember,GroupedPolicy,GroupedTrellis},
    KimiKdaCore,KimiKdaCoreBatch,KimiKdaCoreWeights,KimiMlaSplit,KimiMlaRankBatch,KimiSitu,Bf16F32Add,F32Add,V41PeerCopy};
use std::collections::BTreeMap;
use std::ffi::c_void;

pub(crate) struct Projected<'a,'b> {pub projection:&'b Projection<'a>,pub weights:&'b CoreWeights<'a>,pub scratch:&'b ProjectionScratch<'a>}
impl Projected<'_,'_> {
    fn run(&self,rows:usize,input:Buffer,output:Buffer,stream:*mut c_void)->Result<()> {
        self.projection.local(rows,input,self.weights,self.scratch,output,stream)
    }
}

/// q/k/v/g are independent calibrations: four rotations, four projections.
/// They share one grouped native launch per H128 stage at every row count,
/// bitwise-equal to the four standalone launches. The existing recurrence
/// owns convolution, state, replay and normalization.
pub(crate) fn kda_projected(core:&KimiKdaCore<'_>,projections:[Projected<'_,'_>;4],grouped:Option<&GroupedTrellis<'_>>,
    weights:KimiKdaCoreWeights,batch:KimiKdaCoreBatch,stream:*mut c_void)->Result<()> {
    let outputs=[batch.q,batch.k,batch.v,batch.gate];
    match grouped {
        Some(grouped)=> {
            let scratch=projections[0].scratch;
            ensure!(projections.iter().all(|p|std::ptr::eq(p.scratch,scratch)
                && (p.weights.shape[1],p.weights.shape[0])==grouped.geometry()),
                "K6 grouped KDA members differ in scratch or geometry");
            ensure!(batch.rows<=scratch.rows,"K6 scratch live-row ceiling");
            let get=|name|scratch.buffers[name];
            let members=std::array::from_fn(|i|GroupedMember{payload:projections[i].weights.payload.buffer,
                suh:projections[i].weights.suh.buffer,svh:projections[i].weights.svh.buffer,output:outputs[i]});
            grouped.launch(batch.rows,batch.input,members,get("rotated"),get("table"),get("scales"),
                get("partial"),get("accumulator"),stream)?;
        }
        None=>for (projection,output) in projections.iter().zip(outputs) {
            projection.run(batch.rows,batch.input,output,stream)?;
        },
    }
    core.launch(weights,batch,stream)?;Ok(())
}
/// Stored kv_a N640 is transformed in full; the split norm consumes its N576 prefix.
pub(crate) fn mla_rank_projected(core:&KimiMlaSplit<'_>,qa:Projected<'_,'_>,kv:Projected<'_,'_>,input:Buffer,
    batch:KimiMlaRankBatch,stream:*mut c_void)->Result<()> {
    qa.run(batch.rows,input,batch.q_a,stream)?;kv.run(batch.rows,input,batch.kv_a,stream)?;
    core.rank_norm(batch,stream)?;Ok(())
}
pub(crate) fn mla_query_projected(core:&KimiMlaSplit<'_>,qb:Projected<'_,'_>,rows:usize,q_residual:Buffer,
    q_b:Buffer,uk:Buffer,query:Buffer,stream:*mut c_void)->Result<()> {
    qb.run(rows,q_residual,q_b,stream)?;core.query(rows,q_b,uk,query,stream)?;Ok(())
}
pub(crate) fn mla_gate_projected(core:&KimiMlaSplit<'_>,gate:Projected<'_,'_>,rows:usize,input:Buffer,
    attention:Buffer,uv:Buffer,gates:Buffer,normalized:Buffer,stream:*mut c_void)->Result<()> {
    gate.run(rows,input,gates,stream)?;core.gate_output(rows,attention,uv,gates,normalized,stream)?;Ok(())
}
/// Separate gate/up outputs are interleaved by row for the unchanged Situ op;
/// down returns F32 scratch; the caller independently narrows its retained
/// partial to BF16 before publishing it for the rank-ordered F32 meet.
pub(crate) fn ffn_projected(gate:Projected<'_,'_>,up:Projected<'_,'_>,down:Projected<'_,'_>,situ:&KimiSitu<'_>,
    copy:&V41PeerCopy<'_>,rows:usize,input:Buffer,gate_output:Buffer,up_output:Buffer,gate_up:Buffer,
    activated:Buffer,stream:*mut c_void)->Result<Buffer> {
    let width=gate.projection.program.policy().output;
    ensure!(width==up.projection.program.policy().output && width==down.projection.program.policy().input,
        "K6 FFN intermediate widths differ");
    gate.run(rows,input,gate_output,stream)?;up.run(rows,input,up_output,stream)?;
    copy.interleave_bf16(gate_up,gate_output,up_output,rows,width,stream)?;
    situ.launch(rows,gate_up,activated,stream)?;
    down.projection.core(rows,activated,down.weights,down.scratch,stream)
}

/// Resolve both possible MLA ownership geometries before placement samples CUDA.
/// Serialized projections reuse one lane-private scratch set, never one set per
/// layer. Their BF16 outputs remain distinct until the downstream core consumes them.
pub(crate) struct CorePrograms<'a> {
    projections:BTreeMap<(usize,usize,usize,String),Projection<'a>>,
    grouped:BTreeMap<(usize,usize,usize,String),GroupedTrellis<'a>>,
    /// Diagnostic only: false forces the four standalone launches (bitwise A/B).
    grouped_enabled:std::cell::Cell<bool>,
    pub layout:ProjectionLayout,
    pub kda:[KimiKdaCore<'a>;2],
    pub mla:BTreeMap<(usize,usize,usize),KimiMlaSplit<'a>>,
    pub norm:cuteafd_ffi::KimiRmsNorm<'a>,pub situ:KimiSitu<'a>,pub copy:V41PeerCopy<'a>,pub add:F32Add<'a>,
    pub core_add:Bf16F32Add<'a>,
}
impl<'a> CorePrograms<'a> {
    pub(crate) fn load(programs:&'a Programs<'a>,cfg:&cuteafd_loader::families::kimi_k3::config::KimiK3Config,
        inventory:&cuteafd_loader::families::kimi_k3::core::CoreInventory,rows:usize,
        cache_format:cuteafd_core::MlaCacheFormat)->Result<Option<Self>> {
        use cuteafd_loader::placement::{LayerMode,FfnMode};
        inventory.validate_runtime_groups(cfg)?;
        if inventory.projections.is_empty() {return Ok(None);}
        let mut projections=BTreeMap::new();
        let mut grouped=BTreeMap::new();
        let mut layout:Option<ProjectionLayout>=None;
        for layer in 0..cfg.layers {
            let modes=if cfg.attention[layer]==cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla {
                vec![LayerMode::HeadSplit,LayerMode::Whole{gpu:0,ffn:FfnMode::Split}]
            } else {vec![LayerMode::HeadSplit]};
            for mode in modes {
                for rank in super::weights::LayerLayout::pair_with_core(cfg,layer,mode,inventory)? {
                    for weight in rank.core_layouts() {
                        let key=projection_key(weight.shape,weight.side,weight.codebook);
                        if projections.contains_key(&key) {continue;}
                        let projection=Projection::load_geometry(programs,key.0,key.1,key.2,&key.3,rows,TrellisDType::Bf16)?;
                        let mut required=ProjectionLayout::new(&projection.program,rows)?;
                        if let Some(policy)=GroupedPolicy::from_package_geometry(programs,key.0,key.1,key.2,&key.3)? {
                            let program=GroupedTrellis::load(programs,projection.program.policy(),&policy)?;
                            required.merge(&ProjectionLayout{rows,extents:cuteafd_loader::families::kimi_k3::core::
                                ProjectionScratchLayout::grouped(key.0,key.1,rows,policy.groups)?.extents})?;
                            grouped.insert(key.clone(),program);
                        }
                        if let Some(layout)=&mut layout {layout.merge(&required)?;} else {layout=Some(required);}
                        projections.insert(key,projection);
                    }
                }
            }
        }
        if let Some(head)=inventory.get_weight("language_model.lm_head") {
            let weight=CoreWeightLayout::new(head,
                cuteafd_loader::formats::dense_trellis::TrellisSide::Output,0,1)?;
            let key=projection_key(weight.shape,weight.side,weight.codebook);
            let projection=Projection::load_geometry(programs,key.0,key.1,key.2,&key.3,rows.min(64),TrellisDType::F32)?;
            let mut required=ProjectionLayout::new(&projection.program,rows.min(64))?;
            // Head logits are consumed in <=64-row spans, while this scratch
            // owner also serves full-capacity serialized layer projections.
            required.rows=rows;
            if let Some(layout)=&mut layout {layout.merge(&required)?;}else{layout=Some(required);}
            projections.insert(key,projection);
        }
        let Some(layout)=layout else {anyhow::bail!("Kimi K6 inventory has no bound projections");};
        let planned=cuteafd_loader::families::kimi_k3::core::projection_scratch(cfg,inventory,rows,
            programs.family_metadata("dense_trellis_k6")?)?.context("K6 planned scratch absent")?;
        ensure!(layout.extents==planned.extents && layout.rows==planned.rows,
            "K6 runtime/planner aggregate scratch differs");
        let kda=[KimiKdaCore::load(programs,"kda_core_h48_m64",64,true)?,
            KimiKdaCore::load(programs,"kda_core_h48_m4096",4096,false)?];
        let mut mla=BTreeMap::new();
        for heads in [48,96] {for stride in [576,640] {for capacity in [64,4096] {
            let names=[format!("mla_rank_norm_h{heads}_kv{stride}{}_m{capacity}",cache_format.program_suffix()),
                format!("mla_query_h{heads}_m{capacity}"),format!("mla_gate_output_h{heads}_m{capacity}")];
            mla.insert((heads,stride,capacity),KimiMlaSplit::load(programs,
                [&names[0],&names[1],&names[2]],heads,capacity,stride,cache_format)?);
        }}}
        Ok(Some(Self{projections,grouped,grouped_enabled:std::cell::Cell::new(true),layout,kda,mla,norm:cuteafd_ffi::KimiRmsNorm::load(programs,3584)?,situ:KimiSitu::load(programs,3072)?,
            copy:programs.library().v41_peer_copy()?,add:programs.library().f32_add()?,
            core_add:programs.library().bf16_f32_add()?}))
    }
    pub(crate) fn projection(&self,weights:&CoreWeights<'_>)->Result<&Projection<'a>> {
        self.projections.get(&projection_key(weights.shape,weights.side,weights.codebook))
            .ok_or_else(||anyhow::anyhow!("Kimi K6 weight geometry not resolved before admission"))
    }
    /// The grouped KDA launch for these member weights, when the package has one.
    pub(crate) fn grouped(&self,weights:&CoreWeights<'_>)->Option<&GroupedTrellis<'a>> {
        self.grouped.get(&projection_key(weights.shape,weights.side,weights.codebook)).filter(|_|self.grouped_enabled.get())
    }
    pub(crate) fn set_grouped(&self,enabled:bool) { self.grouped_enabled.set(enabled); }
    pub(crate) fn projected<'b>(&'b self,weights:&'b CoreWeights<'a>,scratch:&'b ProjectionScratch<'a>)->Result<Projected<'a,'b>> {
        Ok(Projected{projection:self.projection(weights)?,weights,scratch})
    }
}
fn projection_key(shape:[usize;2],side:cuteafd_loader::formats::dense_trellis::TrellisSide,
    book:cuteafd_loader::formats::exl3_storage::Exl3Codebook)->(usize,usize,usize,String) {
    (shape[1],shape[0],match side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
        cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2},
        match book {cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg=>"mcg",
            cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1=>"mul1"}.into())
}

#[cfg(test)]
mod runtime_layout_tests {
    use super::*;
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn route_views_share_one_owner_and_admission_matches_allocated_bytes()->Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::families::kimi_k3::core::ProjectionScratchLayout;
        let fixture=Fixture::build()?;
        let library=fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device=crate::shared::memory::device::Device{library:&library,id:0};
        for rows in [1,16,63,64,4096] {
            let planned=ProjectionScratchLayout::new(128,256,rows)?;
            let layout=ProjectionLayout{extents:planned.extents.clone(),rows};
            let scratch=layout.allocate(device)?;
            assert_eq!(layout.demand(0,"Kimi K6 projection scratch")?.bytes as usize,
                scratch._owners.values().map(|owner|owner.buffer.bytes).sum::<usize>());
            assert_eq!(layout.bytes()?,planned.bytes()?);
            let arena=scratch._owners["route arena"].buffer;
            assert_eq!(scratch.buffers["partial"].ptr,arena.ptr);
            assert_eq!(scratch.buffers["partial"].bytes,rows.min(129)*256*4);
            drop(scratch);
        }
        let events=fixture.events()?;
        assert_eq!(events.matches('D').count(),25);
        assert_eq!(events.matches('d').count(),25);
        Ok(())
    }
    #[test]
    fn planner_and_runtime_native_core_bytes_agree_on_and_off()->Result<()> {
        use cuteafd_loader::{families::kimi_k3::{config::{KimiK3Config,KimiAttention},core::{self,CoreCharges,CoreInventory}},
            plan::{testing,Checkpoint},placement::{LayerMode,FfnMode}};
        use cuteafd_ffi::dense_trellis::{DenseTrellisGeometry,TrellisDType};
        use super::super::weights::LayerLayout;
        use cuteafd_loader::formats::dense_trellis::TrellisSide;
        let dir=tempfile::tempdir()?;
        testing::write_kimi_core_snapshot(dir.path(),false);
        let checkpoint=Checkpoint::open(dir.path())?;
        let cfg=KimiK3Config::from_hf(&checkpoint.config)?;
        let inventory=cuteafd_loader::plan::families::kimi::core_inventory(&checkpoint)?;
        let mut geometries=BTreeMap::new();
        for layer in 0..cfg.layers {
            for mode in [LayerMode::HeadSplit,LayerMode::Whole{gpu:0,ffn:FfnMode::Split}] {
                if cfg.attention[layer]==KimiAttention::Kda && mode!=LayerMode::HeadSplit {continue;}
                for rank in LayerLayout::pair_with_core(&cfg,layer,mode,&inventory)? {
                    for weight in rank.core_layouts() {
                        let projection=&inventory.projections[&weight.stem];
                        // Local native allocations contain only compressed payload and vectors.
                        assert_eq!(weight.bytes as u64,projection.resident_bytes(weight.ranks)?);
                        geometries.insert(projection_key(weight.shape,weight.side,weight.codebook),());
                    }
                }
            }
        }
        let head=CoreWeightLayout::new(inventory.get_weight("language_model.lm_head").unwrap(),TrellisSide::Output,0,1)?;
        geometries.insert(projection_key(head.shape,head.side,head.codebook),());
        let policies=geometries.keys().map(|(k,n,side,book)|{
            serde_json::json!({"input_features":k,"output_features":n,"sharded_side":side,"codebook":book})
        }).collect::<Vec<_>>();
        let mut programs=Vec::new();
        let mut program=|name:String,bytes:usize| programs.push(serde_json::json!({"name":name,
            "scratch_bytes_at_capacity":{"scratch":bytes}}));
        for capacity in [64,4096] {
            program(format!("kimik3_kda_m{capacity}"),1024);
            program(format!("kda_core_h48_m{capacity}"),2048);
            for heads in [48,96] {
                let suffix=if heads==48 {""}else{"_h96"};
                program(format!("kimik3_mla_producer{suffix}_m{capacity}"),4096);
                program(format!("kimik3_mla_output{suffix}_m{capacity}"),4096);
                for mode in if capacity == 64 { &["decode", "verify"][..] } else { &["extend"][..] } {
                    program(format!("kimik3_dense_mla_{mode}{suffix}_m{capacity}"),4096);
                }
                for stride in [576,640] {for cache in ["","_fp8"] {
                    program(format!("mla_rank_norm_h{heads}_kv{stride}{cache}_m{capacity}"),2048);
                }}
            }
        }
        let metadata=serde_json::json!({"abi_version":1,"row_policies":policies});
        let manifest=serde_json::json!({"programs":programs,"families":{"dense_trellis_k6":metadata}});
        // `rows` lane rows of `unit`-row units: one-unit lanes and the 4096 x 2 lane.
        for (rows,unit) in [(1,1),(16,16),(63,63),(64,64),(4096,4096),(8192,4096),(8256,2752),(8192,2048)] {
            for on in [false,true] {
                let empty=CoreInventory::default();
                let core=if on {&inventory}else{&empty};
                let charged=CoreCharges::from_package(&cfg,core,rows,unit,&manifest)?;
                let mut runtime:Option<ProjectionLayout>=None;
                if on {
                    for ((k,n,side,_),()) in &geometries {
                        let live=if *n==cfg.vocab_size {unit.min(64)}else{unit};
                        let w=DenseTrellisGeometry{input:*k,output:*n,bits:6,side:*side,rows:live,
                            input_dtype:TrellisDType::Bf16,output_dtype:TrellisDType::Bf16}.workspace()?;
                        let extents=BTreeMap::from([("rotated",w.rotated),("table",w.pointer_table),
                            ("scales",w.block_scales),("partial",w.partial),("accumulator",w.accumulator)]);
                        let layout=ProjectionLayout{extents,rows:unit};
                        if let Some(runtime)=&mut runtime {runtime.merge(&layout)?;}else{runtime=Some(layout);}
                    }
                }
                assert_eq!(charged.projection.as_ref().map(|p|&p.extents),runtime.as_ref().map(|p|&p.extents));
                assert_eq!(charged.receive,super::super::startup::exchange_demands(unit,false)?[0].bytes as usize
                    -256-256-(((cuteafd_loader::placement::HOP_SLOTS+1)*16)as usize).max(256));
                let expected_stages=if on {core::lane_extents(rows,unit,1024,4096,2048,2048)?}else{BTreeMap::new()};
                assert_eq!(charged.stages,expected_stages);
                let sum=charged.demands()?.iter().filter(|d|d.gpu==0).map(|d|d.bytes).sum::<u64>();
                assert_eq!(sum,runtime.as_ref().map_or(Ok(0),|p|p.bytes())? as u64
                    +charged.stage_bytes as u64+charged.receive as u64);
                if on {
                    let base = cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout::new(&cfg, rows, unit,
                        1_048_576, cuteafd_loader::families::kimi_k3::allocation::NativeScratch::from_manifest(&manifest)?)?;
                    assert_eq!(charged.stage_bytes, base.clone().with_core_stages(expected_stages)?.bytes()? - base.bytes()?);
                }
            }
        }
        // A duplicate exact geometry is ambiguous, never selected by table order.
        let mut duplicate=manifest.clone();
        let first=duplicate["families"]["dense_trellis_k6"]["row_policies"][0].clone();
        duplicate["families"]["dense_trellis_k6"]["row_policies"].as_array_mut().unwrap().push(first);
        assert!(CoreCharges::from_package(&cfg,&inventory,64,64,&duplicate).is_err());
        Ok(())
    }

    #[test]
    fn serialized_projections_charge_memberwise_high_water_not_per_layer_sum()->Result<()> {
        let mut shared=ProjectionLayout{rows:64,extents:BTreeMap::from([("rotated",256),("partial",4096),
            ("table",8),("accumulator",512)])};
        let other=ProjectionLayout{rows:64,extents:BTreeMap::from([("rotated",512),("partial",2048),
            ("table",8),("accumulator",1024)])};
        shared.merge(&other)?;
        assert_eq!(shared.bytes()?,512+8+1024+4096);
        let once=shared.clone();shared.merge(&other)?;assert_eq!(shared,once);
        assert!(shared.merge(&ProjectionLayout{rows:63,..other}).is_err());
        Ok(())
    }
}
