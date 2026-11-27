//! Admitted compact projection owners and projected attention/FFN seams.
use crate::shared::dense_trellis::{CoreWeightLayout, CoreWeights, Projection, ProjectionLayout, ProjectionScratch};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer,programs::Programs,
    dense_trellis::{TrellisDType,
        GroupedMember,GroupedPolicy,GroupedTrellis},
    KimiKdaCore,KimiKdaCoreBatch,KimiKdaCoreWeights,KimiMlaSplit,KimiSitu,Bf16F32Add,F32Bf16Narrow,V41PeerCopy};
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
        Some(grouped) if projections.iter().all(|p| p.weights.bits==grouped.bits()
            && p.weights.codebook==projections[0].weights.codebook)=> {
            let scratch=projections[0].scratch;
            ensure!(projections.iter().all(|p|std::ptr::eq(p.scratch,scratch)
                && (p.weights.shape[1],p.weights.shape[0])==grouped.geometry()
                && p.weights.bits==grouped.bits()
                && p.projection.program.policy().bits==grouped.bits()),
                "K6 grouped KDA members differ in scratch or geometry");
            ensure!(batch.rows<=scratch.rows,"K6 scratch live-row ceiling");
            let get=|name|scratch.buffers[name];
            let members=std::array::from_fn(|i|GroupedMember{payload:projections[i].weights.payload.buffer,
                suh:projections[i].weights.suh.buffer,svh:projections[i].weights.svh.buffer,output:outputs[i]});
            grouped.launch(batch.rows,batch.input,members,get("rotated"),get("table"),get("scales"),
                get("partial"),get("accumulator"),stream)?;
        }
        _=>for (projection,output) in projections.iter().zip(outputs) {
            projection.run(batch.rows,batch.input,output,stream)?;
        },
    }
    core.launch(weights,batch,stream)?;Ok(())
}
/// Stored kv_a N640 is transformed in full; the split norm consumes its N576 prefix.
#[cfg(test)]
pub(crate) fn mla_rank_projected(core:&KimiMlaSplit<'_>,qa:Projected<'_,'_>,kv:Projected<'_,'_>,input:Buffer,
    batch:cuteafd_ffi::KimiMlaRankBatch,stream:*mut c_void)->Result<()> {
    qa.run(batch.rows,input,batch.q_a,stream)?;kv.run(batch.rows,input,batch.kv_a,stream)?;
    core.rank_norm(batch,stream)?;Ok(())
}
#[cfg(test)]
pub(crate) fn mla_query_projected(core:&KimiMlaSplit<'_>,qb:Projected<'_,'_>,rows:usize,q_residual:Buffer,
    q_b:Buffer,uk:Buffer,query:Buffer,stream:*mut c_void)->Result<()> {
    qb.run(rows,q_residual,q_b,stream)?;core.query(rows,q_b,uk,query,stream)?;Ok(())
}
#[cfg(test)]
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
    projections:BTreeMap<(usize,usize,usize,String,usize),Projection<'a>>,
    grouped:BTreeMap<(usize,usize,usize,String,usize),GroupedTrellis<'a>>,
    /// Diagnostic only: false forces the four standalone launches (bitwise A/B).
    grouped_enabled:std::cell::Cell<bool>,
    pub layouts:Vec<Option<ProjectionLayout>>,
    pub native_inventory:cuteafd_loader::families::kimi_k3::core::CoreInventory,
    native_bf16:[cuteafd_ffi::draft_projection::DraftProjections<'a>;2],
    native_fp8:[cuteafd_ffi::native_attention::Fp8Projections<'a>;2],
    pub native_scratch:usize,
    pub fp8_head: Option<crate::shared::target_head::BlockFp8Head<'a>>,
    pub dense_situ: Option<KimiSitu<'a>>,
    pub native_add:cuteafd_ffi::F32Add<'a>,
    pub kda:[KimiKdaCore<'a>;2],
    pub mla:BTreeMap<(usize,usize,usize),KimiMlaSplit<'a>>,
    pub norm:cuteafd_ffi::KimiRmsNorm<'a>,pub situ:KimiSitu<'a>,pub copy:V41PeerCopy<'a>,
    pub core_add:Bf16F32Add<'a>,
    pub narrow:F32Bf16Narrow<'a>,
}
impl<'a> CorePrograms<'a> {
    pub(crate) fn load(programs:&'a Programs<'a>,cfg:&cuteafd_loader::families::kimi_k3::config::KimiK3Config,
        inventory:&cuteafd_loader::families::kimi_k3::core::CoreInventory,rows:usize,
        cache_format:cuteafd_core::MlaCacheFormat)->Result<Option<Self>> {
        Self::load_for_ranks(programs, cfg, inventory, rows, cache_format, 2)
    }

    pub(crate) fn load_for_ranks(programs:&'a Programs<'a>,cfg:&cuteafd_loader::families::kimi_k3::config::KimiK3Config,
        inventory:&cuteafd_loader::families::kimi_k3::core::CoreInventory,rows:usize,
        cache_format:cuteafd_core::MlaCacheFormat,ranks:usize)->Result<Option<Self>> {
        Self::load_for_ranks_rows(programs, cfg, inventory, rows, cache_format, ranks, 64)
    }

    pub(crate) fn load_for_ranks_rows(programs:&'a Programs<'a>,cfg:&cuteafd_loader::families::kimi_k3::config::KimiK3Config,
        inventory:&cuteafd_loader::families::kimi_k3::core::CoreInventory,rows:usize,
        cache_format:cuteafd_core::MlaCacheFormat,ranks:usize,decode_rows:usize)->Result<Option<Self>> {
        use cuteafd_loader::placement::{LayerMode,FfnMode};
        ensure!(matches!(ranks,1|2),"Kimi K6 physical rank count");
        let decode_cap = cuteafd_loader::families::kimi_k3::decode_capacity(decode_rows);
        inventory.validate_runtime_groups(cfg)?;
        if !inventory.has_split_core() {return Ok(None);}
        let mut projections=BTreeMap::new();
        let mut grouped=BTreeMap::new();
        let mut layout:Option<ProjectionLayout>=None;
        for layer in 0..cfg.layers {
            let modes=if ranks==1 {vec![LayerMode::Whole{gpu:0,ffn:FfnMode::Owner}]}
            else if cfg.attention[layer]==cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla {
                vec![LayerMode::HeadSplit,LayerMode::Whole{gpu:0,ffn:FfnMode::Split}]
            } else {vec![LayerMode::HeadSplit]};
            for mode in modes {
                for rank in super::weights::LayerLayout::ranks_with_core(cfg,layer,mode,inventory,ranks)? {
                    for weight in rank.core_layouts() {
                        let key=projection_key(weight.shape,weight.side,weight.codebook,weight.bits);
                        if projections.contains_key(&key) {continue;}
                        let projection=Projection::load_geometry_bits(programs,key.0,key.1,key.2,key.4,&key.3,rows,TrellisDType::Bf16)?;
                        ensure!(weight.bits==projection.program.policy().bits,
                            "Kimi {}: loaded K{} projection resolved to K{} program",weight.stem,weight.bits,projection.program.policy().bits);
                        let mut required=ProjectionLayout::new(&projection.program,rows)?;
                        if let Some(policy)=GroupedPolicy::from_package_geometry_bits(programs,key.0,key.1,key.2,key.4,&key.3)? {
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
        let mut layouts=vec![layout;ranks];
        let head_rank=ranks-1;
        if let Some(head)=inventory.get_weight("language_model.lm_head") {
            let weight=CoreWeightLayout::new(head,
                cuteafd_loader::formats::dense_trellis::TrellisSide::Output,0,1)?;
            let key=projection_key(weight.shape,weight.side,weight.codebook,weight.bits);
            let projection=Projection::load_geometry_bits(programs,key.0,key.1,key.2,key.4,&key.3,rows.min(64),TrellisDType::F32)?;
            ensure!(weight.bits==projection.program.policy().bits,
                "Kimi head: loaded K{} projection resolved to K{} program",weight.bits,projection.program.policy().bits);
            let mut required=ProjectionLayout::new(&projection.program,rows.min(64))?;
            if let Some(layout)=&mut layouts[head_rank] {
                // Only a body owner also serves full-capacity layer projections.
                required.rows=layout.rows;
                layout.merge(&required)?;
            }else{layouts[head_rank]=Some(required);}
            projections.insert(key,projection);
        }
        let planned = if inventory.projections.is_empty() { vec![None; ranks] }
            else { cuteafd_loader::families::kimi_k3::core::projection_scratch_for_ranks(cfg, inventory, rows,
                programs.family_metadata("dense_trellis_k6")?, ranks)? };
        for rank in 0..ranks {
            ensure!(layouts[rank].as_ref().map(|p|(&p.extents,p.rows))
                ==planned[rank].as_ref().map(|p|(&p.extents,p.rows)),
                "K6 runtime/planner rank {rank} scratch differs");
        }
        use cuteafd_ffi::draft_projection::{DraftProjections, ProjectionOutput};
        let geometries = inventory.native_geometries(ranks)?;
        let shapes = |fp8, partial| geometries.iter().filter(|g| g.2 == fp8 && g.3 == partial)
            .map(|g| (g.0, g.1)).collect::<Vec<_>>();
        let native_bf16 = [DraftProjections::load_with_output(programs, &shapes(false, false), rows.max(64), ProjectionOutput::Bf16)?,
            DraftProjections::load_with_output(programs, &shapes(false, true), rows.max(64), ProjectionOutput::F32)?];
        let native_fp8 = [cuteafd_ffi::native_attention::Fp8Projections::load(programs, &shapes(true, false), rows.max(64), ProjectionOutput::Bf16)?,
            cuteafd_ffi::native_attention::Fp8Projections::load(programs, &shapes(true, true), rows.max(64), ProjectionOutput::F32)?];
        let mut native_scratch = 0;
        for (k, n, fp8, partial) in geometries {
            if fp8 { native_scratch = native_scratch.max(native_fp8[usize::from(partial)].scratch_bytes(k, n)?); }
        }
        let fp8_head = inventory.fp8_head()?.map(|head| crate::shared::target_head::BlockFp8Head::load(programs, head.shape)).transpose()?;
        if let Some(head) = &fp8_head { native_scratch = native_scratch.max(head.scratch_bytes()?); }
        let dense_situ = inventory.has_dense_fp8_ffn().then(|| KimiSitu::load(programs, cfg.dense_intermediate / ranks)).transpose()?;
        let heads=cfg.kda_heads/ranks;
        let kda=[KimiKdaCore::load(programs,&format!("kda_core_h{heads}_m{decode_cap}"),decode_cap,true)?,
            KimiKdaCore::load(programs,&format!("kda_core_h{heads}_m4096"),4096,false)?];
        let mut mla=BTreeMap::new();
        let k31 = cfg.dsa.is_some();
        let prefix = if k31 { "kimik31_" } else { "" };
        for heads in [48,96] {for stride in [576,640] {for capacity in [decode_cap,4096] {
            let names=[format!("{prefix}mla_rank_norm_h{heads}_kv{stride}{}_m{capacity}",cache_format.program_suffix()),
                format!("{prefix}mla_query_h{heads}_m{capacity}"),format!("{prefix}mla_gate_output_h{heads}_m{capacity}")];
            let load = if k31 { KimiMlaSplit::load_k31 } else { KimiMlaSplit::load };
            mla.insert((heads,stride,capacity),load(programs,
                [&names[0],&names[1],&names[2]],heads,capacity,stride,cache_format)?);
        }}}
        Ok(Some(Self{projections,grouped,grouped_enabled:std::cell::Cell::new(true),layouts,native_inventory:inventory.clone(),native_bf16,native_fp8,native_scratch,fp8_head,dense_situ,native_add:programs.library().f32_add()?,kda,mla,norm:cuteafd_ffi::KimiRmsNorm::load(programs,3584)?,situ:KimiSitu::load(programs,6144/ranks)?,
            copy:programs.library().v41_peer_copy()?,narrow:programs.library().f32_bf16_narrow()?,
            core_add:programs.library().bf16_f32_add()?}))
    }
    pub(crate) fn native_projection(&self, weight:&crate::shared::native_attention::NativeAttentionWeights<'_>,
        rows:usize, input:Buffer, output:Buffer, scratch:Option<Buffer>, stream:*mut c_void)->Result<()> {
        use cuteafd_loader::families::kimi_k3::core::CoreSplit;
        let layout = &weight.layout;
        let [n,k] = layout.shape;
        let partial = layout.ranks == 2 && layout.split == CoreSplit::Input;
        ensure!(input.device_id == weight.values.buffer.device_id && output.device_id == input.device_id
            && input.bytes >= rows*k*2, "Kimi native projection input/device differs from retained owner");
        // SAFETY: admitted lane and weight owners survive queued work and graph replay;
        // the selected registry checks output precision, capacity and alignment.
        unsafe {
            if layout.fp8 {
                self.native_fp8[usize::from(partial)].launch(input.ptr, weight.values.buffer.ptr,
                    weight.scales.as_ref().context("Kimi native FP8 scales absent")?.buffer.ptr, output,
                    scratch.context("Kimi native FP8 scratch not admitted")?, rows,k,n,stream)?;
            } else {
                self.native_bf16[usize::from(partial)].launch_checked(input.ptr, weight.values.buffer.ptr,
                    output,rows,k,n,stream)?;
            }
        }
        Ok(())
    }

    pub(crate) fn projection(&self,weights:&CoreWeights<'_>)->Result<&Projection<'a>> {
        self.projections.get(&projection_key(weights.shape,weights.side,weights.codebook,weights.bits))
            .ok_or_else(||anyhow::anyhow!("Kimi K6 weight geometry not resolved before admission"))
    }
    /// The grouped KDA launch for these member weights, when the package has one.
    pub(crate) fn grouped(&self,weights:&CoreWeights<'_>)->Option<&GroupedTrellis<'a>> {
        self.grouped.get(&projection_key(weights.shape,weights.side,weights.codebook,weights.bits)).filter(|_|self.grouped_enabled.get())
    }
    pub(crate) fn set_grouped(&self,enabled:bool) { self.grouped_enabled.set(enabled); }
    pub(crate) fn projected<'b>(&'b self,weights:&'b CoreWeights<'a>,scratch:&'b ProjectionScratch<'a>)->Result<Projected<'a,'b>> {
        Ok(Projected{projection:self.projection(weights)?,weights,scratch})
    }
}
fn projection_key(shape:[usize;2],side:cuteafd_loader::formats::dense_trellis::TrellisSide,
    book:cuteafd_loader::formats::exl3_storage::Exl3Codebook,bits:usize)->(usize,usize,usize,String,usize) {
    (shape[1],shape[0],match side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
        cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2},
        match book {cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg=>"mcg",
            cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1=>"mul1"}.into(),bits)
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
            let scratch=layout.allocate(device, crate::shared::memory::Tag::workspace("kimi/native-projection"))?;
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
        for ranks in [1, 2] {
        let mut geometries=BTreeMap::new();
        for layer in 0..cfg.layers {
            let modes = if ranks == 1 { vec![LayerMode::Whole{gpu:0,ffn:FfnMode::Owner}] }
                else { vec![LayerMode::HeadSplit,LayerMode::Whole{gpu:0,ffn:FfnMode::Split}] };
            for mode in modes {
                if ranks == 2 && cfg.attention[layer]==KimiAttention::Kda && mode!=LayerMode::HeadSplit {continue;}
                for rank in LayerLayout::ranks_with_core(&cfg,layer,mode,&inventory,ranks)? {
                    for weight in rank.core_layouts() {
                        let projection=&inventory.projections[&weight.stem];
                        // Local native allocations contain only compressed payload and vectors.
                        assert_eq!(weight.bytes as u64,projection.resident_bytes(weight.ranks)?);
                        geometries.insert(projection_key(weight.shape,weight.side,weight.codebook,weight.bits),());
                    }
                }
            }
        }
        let head=CoreWeightLayout::new(inventory.get_weight("language_model.lm_head").unwrap(),TrellisSide::Output,0,1)?;
        geometries.insert(projection_key(head.shape,head.side,head.codebook,head.bits),());
        let policies=geometries.keys().map(|(k,n,side,book,bits)|{
            serde_json::json!({"input_features":k,"output_features":n,"sharded_side":side,"codebook":book,"bits":bits})
        }).collect::<Vec<_>>();
        let mut programs=Vec::new();
        let mut program=|name:String,bytes:usize| programs.push(serde_json::json!({"name":name,
            "scratch_bytes_at_capacity":{"scratch":bytes}}));
        for capacity in [64,4096] {
            program(format!("kimik3_kda_m{capacity}"),1024);
            program(format!("kda_fused_h96_m{capacity}"),1536);
            program(format!("kda_core_h48_m{capacity}"),2048);
            program(format!("kda_core_h96_m{capacity}"),3072);
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
                let charged=CoreCharges::from_package_for_ranks(&cfg,core,rows,unit,&manifest,ranks)?;
                let expected_stages=if on {core::lane_extents_for_ranks(rows,unit,if ranks==1 {1536}else{1024},4096,if ranks==1 {3072}else{2048},2048,ranks)?}else{BTreeMap::new()};
                for rank in 0..ranks {
                    let mut runtime:Option<ProjectionLayout>=None;
                    if on {
                        for ((k,n,side,_,bits),()) in &geometries {
                            if *n==cfg.vocab_size && rank!=ranks-1 {continue;}
                            let live=if *n==cfg.vocab_size {unit.min(64)}else{unit};
                            let w=DenseTrellisGeometry{input:*k,output:*n,bits:*bits,side:*side,rows:live,
                                input_dtype:TrellisDType::Bf16,output_dtype:TrellisDType::Bf16}.workspace()?;
                            let extents=BTreeMap::from([("rotated",w.rotated),("table",w.pointer_table),
                                ("scales",w.block_scales),("partial",w.partial),("fused",w.fused),
                                ("accumulator",w.accumulator)]);
                            let layout=ProjectionLayout{extents,rows:unit};
                            if let Some(runtime)=&mut runtime {runtime.merge(&layout)?;}else{runtime=Some(layout);}
                        }
                    }
                    assert_eq!(charged.projection[rank].as_ref().map(|p|&p.extents),runtime.as_ref().map(|p|&p.extents));
                    let receive = if ranks == 2 {core::exchange_recv_bytes(unit)?}else{0};
                    assert_eq!(charged.receive, receive);
                    assert!(charged.demands()?.iter().all(|d| usize::from(d.gpu)<ranks));
                    assert_eq!(charged.projection.len(), ranks);
                    assert_eq!(charged.stages,expected_stages);
                    let sum=charged.demands()?.iter().filter(|d|usize::from(d.gpu)==rank).map(|d|d.bytes).sum::<u64>();
                    assert_eq!(sum,runtime.as_ref().map_or(Ok(0),|p|p.bytes())? as u64
                        +charged.stage_bytes as u64+charged.receive as u64);
                }
                if on {
                    let base = cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout::for_ranks(&cfg, rows, unit,
                        1_048_576, cuteafd_loader::families::kimi_k3::allocation::NativeScratch::from_manifest_for_ranks(&manifest,ranks)?,ranks)?;
                    assert_eq!(charged.stage_bytes, base.clone().with_core_stages(expected_stages)?.bytes()? - base.bytes()?);
                }
            }
        }
        // A duplicate exact geometry is ambiguous, never selected by table order.
        let mut duplicate=manifest.clone();
        let first=duplicate["families"]["dense_trellis_k6"]["row_policies"][0].clone();
        duplicate["families"]["dense_trellis_k6"]["row_policies"].as_array_mut().unwrap().push(first);
        assert!(CoreCharges::from_package_for_ranks(&cfg,&inventory,64,64,&duplicate,ranks).is_err());
        }
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
