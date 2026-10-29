//! Real-size core-layer gate; routed experts are deliberately outside this fixture.
use super::{core_native_tests::{bf16, errors}, core_runtime::Projected};
use crate::shared::dense_trellis::{Projection, ProjectionLayout, ProjectionScratch, CoreWeightLayout, CoreWeights};
use crate::shared::memory::device::{Allocation,Device,Stream};
use anyhow::{ensure,Context,Result};
use cuteafd_core::{DType,TensorInfo,TensorRole};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer,NativeLibrary,programs::Programs,
    dense_trellis::TrellisDType,KimiKda,KimiKdaBatch,KimiKdaCore,KimiKdaCoreBatch,KimiKdaCoreWeights};
use cuteafd_loader::{LoadedTensor,read_safetensors_metadata,
    families::kimi_k3::{config::KimiK3Config,core::{CoreProjection,CoreSplit},kda},
    formats::{dense_trellis::TrellisSide,exl3_storage::derive_storage_map},
    kda::{KdaOperand,KdaTensor}};
use std::{collections::BTreeMap,fs::File,os::unix::fs::FileExt,path::{Path,PathBuf}};

fn read(path:&Path,name:&str)->Result<LoadedTensor> {
    let meta=read_safetensors_metadata(path)?.into_iter().find(|m|m.name==name)
        .with_context(||format!("fixture tensor {name}"))?;
    let mut bytes=vec![0;usize::try_from(meta.byte_length)?];
    File::open(path)?.read_exact_at(&mut bytes,meta.byte_offset)?;
    Ok(LoadedTensor {info:TensorInfo{name:name.into(),file:path.file_name().context("fixture filename")?.to_string_lossy().into(),
        dtype:meta.dtype,shape:meta.shape,byte_offset:meta.byte_offset,byte_length:meta.byte_length,
        role:TensorRole::Other,layer_id:None,expert_id:None,is_quantization_metadata:false},source_path:path.into(),bytes,
        elapsed_micros:0,sha256:String::new()})
}
// Called only after errors() validates finite, nonempty vectors with nonzero reference.
fn error_bias(actual:&[f64],reference:&[f64])->(Option<f64>,f64) {
    let mut dot=0.;let mut error_squared=0.;let mut reference_squared=0.;
    let mut signed=0.;let mut absolute_reference=0.;
    for (&actual,&reference) in actual.iter().zip(reference) {
        let error=actual-reference;
        dot+=error*reference;error_squared+=error*error;reference_squared+=reference*reference;
        signed+=error;absolute_reference+=reference.abs();
    }
    let cosine=if error_squared==0. {None}else{Some(dot/(error_squared*reference_squared).sqrt())};
    (cosine,signed/absolute_reference)
}
struct Source {root:PathBuf,book:String,failures:std::cell::RefCell<Vec<String>>}
impl Source {
    fn report(&self,label:&str,actual:&[f64],reference:&[f64])->Result<()> {
        let (relative,peak)=errors(actual,reference)?;
        let max_abs=actual.iter().zip(reference).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
        let (cosine,signed)=error_bias(actual,reference);
        let cosine=cosine.map_or_else(||"undefined".to_string(),|value|value.to_string());
        let result=format!("book={} {label} rel_l2={relative} max_abs={max_abs} max_abs_over_peak={peak} error_ref_cosine={cosine} mean_signed_over_mean_abs_ref={signed}",self.book);
        eprintln!("K6 full-layer {result}");
        if relative>=0.004 || peak>=0.01 {self.failures.borrow_mut().push(result);}
        Ok(())
    }
    fn case(&self,stem:&str)->PathBuf {self.root.join(format!("{}.{stem}.safetensors",self.book))}
    fn weight(&self,name:&str)->Result<LoadedTensor> {
        let stem=name.strip_suffix(".weight").unwrap_or(name);
        if !self.case(stem).is_file() {return read(&self.root.join("bf16-keep.safetensors"),name);}
        let tensor=read(&self.case(stem),&format!("{}.{stem}.dequant",self.book))?;
        ensure!(tensor.info.dtype==DType::F16 && tensor.info.shape.len()==2,"dequant fixture header");
        let (k,stored)=(tensor.info.shape[0],tensor.info.shape[1]);
        let n=if stem.ends_with("kv_a_proj_with_mqa") {576}else{stored};
        ensure!(k>0 && stored>=n && tensor.bytes.len()==k.checked_mul(stored).and_then(|v|v.checked_mul(2))
            .context("dequant fixture extent overflow")?,"dequant fixture extent");
        let mut bytes=vec![0;n*k*2];
        for input in 0..k {for output in 0..n {
            let offset=2*(input*stored+output);
            let h=u16::from_le_bytes([tensor.bytes[offset],tensor.bytes[offset+1]]);
            let exponent=(h>>10)&31;let mantissa=u32::from(h&1023);let sign=u32::from(h&0x8000)<<16;
            ensure!(exponent!=31,"nonfinite dequant fixture");
            let value=if exponent==0 {(if sign==0 {1.0}else{-1.0})*(mantissa as f32)*2f32.powi(-24)}
                else {f32::from_bits(sign|((u32::from(exponent)+112)<<23)|(mantissa<<13))};
            let bits=value.to_bits();
            let b=((bits.wrapping_add(0x7fff+((bits>>16)&1)))>>16)as u16;
            bytes[2*(output*k+input)..2*(output*k+input)+2].copy_from_slice(&b.to_le_bytes());
        }}
        Ok(LoadedTensor{info:TensorInfo{name:name.into(),dtype:DType::Bf16,shape:vec![n,k],byte_length:bytes.len()as u64,
            ..tensor.info},bytes,..tensor})
    }
    fn projection<'a>(&self,device:Device<'a>,programs:&'a Programs<'a>,stem:&str,shape:&str,
        side:TrellisSide,rank:usize,ranks:usize,rows:usize)->Result<OwnedProjection<'a>> {
        let path=self.case(stem);let prefix=format!("{}.{stem}",self.book);
        let headers=read_safetensors_metadata(&path)?;
        let members=headers.iter().filter(|m|m.name.strip_prefix(&format!("{prefix}.")).is_some_and(|suffix|
            matches!(suffix,"trellis"|"suh"|"svh"|"mcg"|"mul1"))).collect::<Vec<_>>();
        let modules=derive_storage_map(members)?;let module=&modules[&prefix];
        let descriptor=CoreProjection{stem:prefix.clone(),bits:6,codebook:module.codebook,
            shape:[module.output_features,module.input_features],stored_outputs:module.output_features,
            split:CoreSplit::Whole,source_bytes:0};
        let layout=CoreWeightLayout::new(&descriptor,side,rank,ranks)?;
        let loaded=layout.pack(module,|name|read(&path,name))?;
        let weights=layout.upload(device,loaded)?;
        let program=device.run(||Projection::load(programs,shape,&self.book,rows,TrellisDType::Bf16))?;
        let scratch=ProjectionLayout::new(&program.program,rows)?.allocate(device)?;
        Ok(OwnedProjection{program,weights,scratch})
    }
}
struct OwnedProjection<'a> {program:Projection<'a>,weights:CoreWeights<'a>,scratch:ProjectionScratch<'a>}
impl<'a> OwnedProjection<'a> {
    fn projected(&self)->Projected<'a,'_> {Projected{projection:&self.program,weights:&self.weights,scratch:&self.scratch}}
    fn local(&self,rows:usize,input:Buffer,out:Buffer,stream:*mut std::ffi::c_void)->Result<()> {
        self.program.local(rows,input,&self.weights,&self.scratch,out,stream)
    }
    fn core(&self,rows:usize,input:Buffer,stream:*mut std::ffi::c_void)->Result<Buffer> {
        self.program.core(rows,input,&self.weights,&self.scratch,stream)
    }
}
fn upload<'a>(device:Device<'a>,bytes:&[u8])->Result<Allocation<'a>> {
    let allocation=Allocation::new(device,bytes.len().max(16))?;
    device.run(||device.library.copy_h2d(allocation.buffer,bytes))?;Ok(allocation)
}
fn zero<'a>(device:Device<'a>,bytes:usize)->Result<Allocation<'a>> {upload(device,&vec![0;bytes])}
fn input(rows:usize,k:usize)->Vec<u8> {(0..rows*k).flat_map(|i| {
    let bits=((i as f32)*0.003).sin().to_bits();
    ((bits.wrapping_add(0x7fff+((bits>>16)&1)))>>16).to_le_bytes()[..2].to_vec()
}).collect()}
fn reverse_k_blocks(bytes:&[u8],k:usize)->Vec<u8> {
    let mut result=bytes.to_vec();
    for (row,out) in bytes.chunks_exact(k*2).zip(result.chunks_exact_mut(k*2)) {
        for block in 0..k/128 {out[block*256..(block+1)*256]
            .copy_from_slice(&row[(k/128-1-block)*256..(k/128-block)*256]);}
    }
    result
}
fn compare(source:&Source,device:Device<'_>,actual:Buffer,reference:Buffer,count:usize,label:&str)->Result<()> {
    let mut a=vec![0;count*2];let mut b=vec![0;count*2];
    device.run(||{device.library.copy_d2h(&mut a,actual)?;device.library.copy_d2h(&mut b,reference)})?;
    source.report(label,&bf16(&a),&bf16(&b))
}

fn check_retained_partial(device:Device<'_>,source:Buffer,retained:Buffer,count:usize)->Result<()> {
    let mut f32_bytes=vec![0;count*4];let mut bf16_bytes=vec![0;count*2];
    device.run(|| {
        device.library.copy_d2h(&mut f32_bytes,source)?;
        device.library.copy_d2h(&mut bf16_bytes,retained)
    })?;
    for (source,retained) in f32_bytes.chunks_exact(4).zip(bf16_bytes.chunks_exact(2)) {
        let bits=u32::from_le_bytes(source.try_into().unwrap());
        ensure!(f32::from_bits(bits).is_finite(),"K6 retained partial must be finite");
        let expected=(bits.wrapping_add(0x7fff+((bits>>16)&1))>>16)as u16;
        assert_eq!(u16::from_le_bytes(retained.try_into().unwrap()),expected,"K6 partial BF16 RNE");
    }
    Ok(())
}

fn compare_meet(device:Device<'_>,source:&Source,stream:&Stream<'_>,rows:usize,
    ranks:&[(OwnedProjection<'_>,Allocation<'_>,Allocation<'_>)],label:&str)->Result<()> {
    let add=device.run(||device.library.bf16_f32_add())?;
    let reduced=zero(device,rows*7168*4)?;let combined=zero(device,rows*7168*2)?;
    let (projection,rank0,_)=&ranks[0];let rank1=ranks[1].1.buffer;
    let result=device.run(||projection.program.reduced(&add,rows,rank0.buffer,rank1,reduced.buffer,
        &projection.weights,combined.buffer,stream.raw));
    stream.drain_or_abort("K6 BF16 partial / F32 meet / H128 egress");result?;
    let mut retained=[vec![0;rows*7168*2],vec![0;rows*7168*2]];
    let mut references=retained.clone();
    for rank in 0..2 {device.run(|| {
        device.library.copy_d2h(&mut retained[rank],ranks[rank].1.buffer)?;
        device.library.copy_d2h(&mut references[rank],ranks[rank].2.buffer)
    })?;}
    let mut sum=vec![0;rows*7168*4];device.run(||device.library.copy_d2h(&mut sum,reduced.buffer))?;
    for ((a,b),actual) in bf16(&retained[0]).iter().zip(bf16(&retained[1])).zip(sum.chunks_exact(4)) {
        assert_eq!(u32::from_le_bytes(actual.try_into().unwrap()),((*a as f32)+(b as f32)).to_bits(),
            "K6 rank-ordered BF16 partials sum into F32");
    }
    let reference=bf16(&references[0]).iter().zip(bf16(&references[1])).map(|(a,b)|a+b).collect::<Vec<_>>();
    let mut bytes=vec![0;rows*7168*2];device.run(||device.library.copy_d2h(&mut bytes,combined.buffer))?;
    source.report(&format!("{label} rows={rows} BF16 partial F32 meet"),&bf16(&bytes),&reference)
}

fn kda_layer(device:Device<'_>,base:&Programs<'_>,programs:&Programs<'_>,source:&Source,cfg:&KimiK3Config,rows:usize)->Result<()> {
    let stream=Stream::new(device)?;
    let x=upload(device,&input(rows,7168))?;
    let fused=device.run(||Ok(KimiKda::load(base,64,true)?))?;
    let split=device.run(||Ok(KimiKdaCore::load(programs,"kda_core_h48_m64",64,true)?))?;
    let bf16_control=device.run(||Ok(cuteafd_ffi::KimiFfn::load(base,3072,cuteafd_ffi::KimiFfnMode::Direct)?))?;
    let pack=|operand|kda::pack(cfg,1,operand,2,|name| {
        let t=source.weight(name)?;Ok(KdaTensor{dtype:t.info.dtype,shape:t.info.shape,bytes:t.bytes})
    }).map_err(anyhow::Error::from);
    let mut packed=BTreeMap::new();
    for (name,operand) in [("input",KdaOperand::WIn),("forget",KdaOperand::WFg),("conv",KdaOperand::ConvW),
        ("a_log",KdaOperand::ALog),("dt_bias",KdaOperand::DtBias),("norm",KdaOperand::ONorm),("output",KdaOperand::WO)] {
        packed.insert(name,pack(operand)?);
    }
    let mut rank_cores=Vec::new();
    for rank in 0..2 {
        let weights=packed.iter().map(|(&name,pair)|Ok((name,upload(device,&pair[rank].bytes)?))).collect::<Result<BTreeMap<_,_>>>()?;
        let w=|name|weights[name].buffer;
        let local_width=6144;let h=7168;
        let wi=&packed["input"][rank].bytes;
        let mut fab=wi[3*local_width*h*2..(3*local_width+128)*h*2].to_vec();
        fab.extend_from_slice(&wi[(4*local_width+128)*h*2..]);
        let fab=upload(device,&fab)?;
        let stem="language_model.model.layers.1.self_attn";
        let projections=["q_proj","k_proj","v_proj","g_proj"].iter().map(|leaf|
            source.projection(device,programs,&format!("{stem}.{leaf}"),"kda_qkvg_tp2",TrellisSide::Output,rank,2,64))
            .collect::<Result<Vec<_>>>()?;
        let out_projection=source.projection(device,programs,&format!("{stem}.o_proj"),"kda_o_tp2",TrellisSide::Input,rank,2,64)?;
        let planes=(0..4).map(|_|zero(device,rows*local_width*2)).collect::<Result<Vec<_>>>()?;
        let state=zero(device,KimiKda::state_bytes(1)?)?;let state_ref=zero(device,state.buffer.bytes)?;
        let conv=zero(device,KimiKda::conv_state_bytes(1)?)?;let conv_ref=zero(device,conv.buffer.bytes)?;
        let normalized=zero(device,rows*local_width*2)?;let normalized_ref=zero(device,normalized.buffer.bytes)?;
        let output_ref=zero(device,rows*h*2)?;
        let output=zero(device,rows*h*2)?;
        let retained=zero(device,rows*h*2)?;
        let replay=zero(device,KimiKda::replay_bytes())?;let replay_ref=zero(device,replay.buffer.bytes)?;
        let scratch=zero(device,split.scratch_bytes())?;let scratch_ref=zero(device,fused.scratch_bytes())?;
        let slots=zero(device,rows*4)?;let first=zero(device,rows*4)?;
        let result=device.run(|| {
            super::core_runtime::kda_projected(&split,std::array::from_fn(|i|projections[i].projected()),None,
                KimiKdaCoreWeights{fab:fab.buffer,forget:w("forget"),convolution:w("conv"),a_log:w("a_log"),dt_bias:w("dt_bias"),norm:w("norm")},
                KimiKdaCoreBatch{rows,sequences:1,state_slots:1,speculative:false,input:x.buffer,q:planes[0].buffer,k:planes[1].buffer,
                    v:planes[2].buffer,gate:planes[3].buffer,conv_state:conv.buffer,state:state.buffer,slots:slots.buffer,first:first.buffer,
                    output:normalized.buffer,replay:replay.buffer,scratch:scratch.buffer},stream.raw)?;
            fused.launch(cuteafd_ffi::KimiKdaWeights{input:w("input"),forget:w("forget"),convolution:w("conv"),a_log:w("a_log"),
                dt_bias:w("dt_bias"),norm:w("norm"),output:w("output")},KimiKdaBatch{rows,sequences:1,state_slots:1,speculative:false,
                input:x.buffer,state:state_ref.buffer,conv_state:conv_ref.buffer,slots:slots.buffer,first:first.buffer,
                normalized:normalized_ref.buffer,output:output_ref.buffer,replay:Some(replay_ref.buffer),scratch:scratch_ref.buffer},stream.raw)?;
            let partial=out_projection.core(rows,normalized.buffer,stream.raw)?;
            out_projection.program.program.output(rows,partial,out_projection.weights.svh.buffer,output.buffer,stream.raw)?;
            crate::shared::dense_trellis::retain_partial(device.library,partial,retained.buffer,rows*h,stream.raw)?;
            Ok(partial)
        });
        stream.drain_or_abort("K6 full KDA case");let partial=result?;
        check_retained_partial(device,partial,retained.buffer,rows*h)?;
        // Per-rank egress is diagnostic only; the runtime narrows both cores
        // independently, sums in F32 and runs one H128 egress after the meet.
        compare(source,device,output.buffer,output_ref.buffer,rows*h,&format!("KDA rank={rank} rows={rows} output"))?;
        compare(source,device,normalized.buffer,normalized_ref.buffer,rows*local_width,&format!("KDA rank={rank} rows={rows} normalized"))?;
        compare(source,device,conv.buffer,conv_ref.buffer,conv.buffer.bytes/2,&format!("KDA rank={rank} rows={rows} conv"))?;
        // The unchanged fused program leaves its BF16 input projections at the
        // start of scratch. Compare those before recurrence can amplify error.
        let projection_width=4*local_width+128+48;
        let mut projected_reference=vec![0;rows*projection_width*2];
        device.run(||device.library.copy_d2h(&mut projected_reference,scratch_ref.buffer))?;
        for (index,label) in ["q","k","v","gate"].iter().enumerate() {
            let mut actual=vec![0;rows*local_width*2];
            device.run(||device.library.copy_d2h(&mut actual,planes[index].buffer))?;
            let offset=if index==3 {3*local_width+128}else{index*local_width};
            let reference=(0..rows).flat_map(|row| {
                let first=2*(row*projection_width+offset);
                bf16(&projected_reference[first..first+local_width*2])
            }).collect::<Vec<_>>();
            source.report(&format!("KDA rank={rank} rows={rows} projection {label}"),&bf16(&actual),&reference)?;
        }
        let mut a=vec![0;state.buffer.bytes];let mut b=vec![0;state_ref.buffer.bytes];
        device.run(||{device.library.copy_d2h(&mut a,state.buffer)?;device.library.copy_d2h(&mut b,state_ref.buffer)})?;
        let floats=|bytes:&[u8]|bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())as f64).collect::<Vec<_>>();
        source.report(&format!("KDA rank={rank} rows={rows} state"),&floats(&a),&floats(&b))?;
        // Reverse matching K128 blocks of BF16 x and weights. Products and
        // the exact dot product stay fixed; only MMA summation order changes.
        // The separate f_a/b projection still uses the original x and weights.
        let control_input=upload(device,&reverse_k_blocks(&input(rows,h),h))?;
        let control_planes=(0..4).map(|_|zero(device,rows*local_width*2)).collect::<Result<Vec<_>>>()?;
        let control_weights=(0..4).map(|index| {
            let first=(if index==3 {3*local_width+128}else{index*local_width})*h*2;
            upload(device,&reverse_k_blocks(&wi[first..first+local_width*h*2],h))
        }).collect::<Result<Vec<_>>>()?;
        let control_state=zero(device,state.buffer.bytes)?;let control_conv=zero(device,conv.buffer.bytes)?;
        let control_normalized=zero(device,rows*local_width*2)?;let control_output=zero(device,rows*h*2)?;
        let control_replay=zero(device,KimiKda::replay_bytes())?;
        let control_scratch=zero(device,split.scratch_bytes())?;
        let control_output_program=device.run(||Ok(cuteafd_ffi::KimiFfn::load(base,6144,cuteafd_ffi::KimiFfnMode::Direct)?))?;
        let result=device.run(|| {
            for index in 0..4 {bf16_control.project_gate_up(rows,control_input.buffer,control_weights[index].buffer,control_planes[index].buffer,stream.raw)?;}
            split.launch(KimiKdaCoreWeights{fab:fab.buffer,forget:w("forget"),convolution:w("conv"),a_log:w("a_log"),dt_bias:w("dt_bias"),norm:w("norm")},
                KimiKdaCoreBatch{rows,sequences:1,state_slots:1,speculative:false,input:x.buffer,
                    q:control_planes[0].buffer,k:control_planes[1].buffer,v:control_planes[2].buffer,gate:control_planes[3].buffer,
                    conv_state:control_conv.buffer,state:control_state.buffer,slots:slots.buffer,first:first.buffer,
                    output:control_normalized.buffer,replay:control_replay.buffer,scratch:control_scratch.buffer},stream.raw)?;
            control_output_program.project_down(rows,control_normalized.buffer,w("output"),control_output.buffer,stream.raw)?;
            Ok(())
        });
        stream.drain_or_abort("BF16 reduction-order KDA self-control");result?;
        for (index,label) in ["q","k","v","gate"].iter().enumerate() {
            let mut actual=vec![0;rows*local_width*2];
            device.run(||device.library.copy_d2h(&mut actual,control_planes[index].buffer))?;
            let offset=if index==3 {3*local_width+128}else{index*local_width};
            let reference=(0..rows).flat_map(|row| {
                let first=2*(row*projection_width+offset);
                bf16(&projected_reference[first..first+local_width*2])
            }).collect::<Vec<_>>();
            source.report(&format!("BF16 control rank={rank} rows={rows} projection {label}"),&bf16(&actual),&reference)?;
        }
        compare(source,device,control_normalized.buffer,normalized_ref.buffer,rows*local_width,&format!("BF16 control rank={rank} rows={rows} normalized"))?;
        compare(source,device,control_output.buffer,output_ref.buffer,rows*h,&format!("BF16 control rank={rank} rows={rows} output"))?;
        compare(source,device,control_conv.buffer,conv_ref.buffer,conv.buffer.bytes/2,&format!("BF16 control rank={rank} rows={rows} conv"))?;
        let mut control=vec![0;state.buffer.bytes];device.run(||device.library.copy_d2h(&mut control,control_state.buffer))?;
        source.report(&format!("BF16 control rank={rank} rows={rows} state"),&floats(&control),&floats(&b))?;
        rank_cores.push((out_projection,retained,output_ref));
    }
    compare_meet(device,source,&stream,rows,&rank_cores,"KDA")?;
    Ok(())
}

fn mla_layer(device:Device<'_>,base:&Programs<'_>,programs:&Programs<'_>,source:&Source,cfg:&KimiK3Config,rows:usize)->Result<()> {
    use cuteafd_ffi::{KimiMla,KimiMlaMode,KimiMlaBatch,KimiMlaWeights,KimiMlaSplit,KimiMlaRankBatch};
    use cuteafd_loader::{families::kimi_k3::mla,formats::mla::MlaOperand};
    let stream=Stream::new(device)?;
    let attention=device.run(||Ok(KimiMla::load(base,96,64,KimiMlaMode::Verify,cuteafd_core::MlaCacheFormat::Bf16)?))?;
    let split=device.run(||Ok(KimiMlaSplit::load(programs,["mla_rank_norm_h96_kv640_m64",
        "mla_query_h96_m64","mla_gate_output_h96_m64"],96,64,640,cuteafd_core::MlaCacheFormat::Bf16)?))?;
    let mut weights=BTreeMap::new();
    for (name,operand) in [("qa",MlaOperand::QkvA),("qn",MlaOperand::QNorm),("kn",MlaOperand::KvNorm),
        ("qb",MlaOperand::QB),("uk",MlaOperand::Uk),("uv",MlaOperand::Uv),("gate",MlaOperand::Gate),("out",MlaOperand::Output)] {
        let packed=mla::pack(cfg,3,operand,1,|name|source.weight(name))?;
        weights.insert(name,upload(device,&packed[0].bytes)?);
    }
    let w=|name|weights[name].buffer;
    let stem="language_model.model.layers.3.self_attn";
    let mut projections=BTreeMap::new();
    for (leaf,shape) in [("q_a_proj","mla_q_a"),("kv_a_proj_with_mqa","mla_kv_a_padded"),
        ("q_b_proj","mla_q_b_owned"),("g_proj","mla_g_owned"),("o_proj","mla_o_owned")] {
        projections.insert(leaf,source.projection(device,programs,&format!("{stem}.{leaf}"),shape,TrellisSide::Output,0,1,64)?);
    }
    let x=upload(device,&input(rows,7168))?;
    let qa=zero(device,rows*1536*2)?;let kv=zero(device,rows*640*2)?;
    let qb=zero(device,rows*96*192*2)?;let gates=zero(device,rows*12288*2)?;
    let norm=zero(device,rows*12288*2)?;let output=zero(device,rows*7168*2)?;
    let slots=upload(device,&(0..rows as i64).flat_map(i64::to_le_bytes).collect::<Vec<_>>())?;
    let pages=rows.div_ceil(64);let page_table=upload(device,&(0..pages as i32).flat_map(i32::to_le_bytes).collect::<Vec<_>>())?;
    let lengths=upload(device,&(rows as i32).to_le_bytes())?;
    let offsets=upload(device,&[0i32.to_le_bytes(),(rows as i32).to_le_bytes()].concat())?;
    let scratch=attention.scratch_bytes();
    let make_batch=||->Result<(BTreeMap<&str,Allocation<'_>>,KimiMlaBatch)> {
        let mut owners=BTreeMap::new();
        for (name,bytes) in [("cache",pages*64*576*2),("query",rows*96*576*2),("resid",rows*1536*2),
            ("attn",rows*96*512*2),("lse",rows*96*4),("out",rows*7168*2),("producer",scratch[0]),
            ("attention",scratch[1]),("output",scratch[2])] {owners.insert(name,zero(device,bytes.max(16))?);}
        let b=|name|owners[name].buffer;
        let batch=KimiMlaBatch{rows,requests:1,page_table_width:pages,cache_pages:pages,active_splits:attention.splits(),
            x:x.buffer,slots:slots.buffer,cache:b("cache"),page_table:page_table.buffer,lengths:lengths.buffer,query_offsets:offsets.buffer,
            query:b("query"),q_resid:b("resid"),attention:b("attn"),lse:b("lse"),output:b("out"),producer_scratch:b("producer"),
            attention_scratch:b("attention"),output_scratch:b("output")};
        Ok((owners,batch))
    };
    let (_owners,actual)=make_batch()?;let (_reference_owners,reference)=make_batch()?;
    let norm_scratch=zero(device,split.scratch_bytes())?;
    let result=device.run(|| {
        super::core_runtime::mla_rank_projected(&split,projections["q_a_proj"].projected(),projections["kv_a_proj_with_mqa"].projected(),x.buffer,
            KimiMlaRankBatch{rows,cache_pages:pages,q_a:qa.buffer,kv_a:kv.buffer,q_norm:w("qn"),kv_norm:w("kn"),slots:slots.buffer,
                q_resid:actual.q_resid,cache:actual.cache,scratch:norm_scratch.buffer},stream.raw)?;
        super::core_runtime::mla_query_projected(&split,projections["q_b_proj"].projected(),rows,actual.q_resid,qb.buffer,w("uk"),actual.query,stream.raw)?;
        attention.attention(actual,stream.raw)?;
        super::core_runtime::mla_gate_projected(&split,projections["g_proj"].projected(),rows,x.buffer,actual.attention,w("uv"),gates.buffer,norm.buffer,stream.raw)?;
        projections["o_proj"].local(rows,norm.buffer,output.buffer,stream.raw)?;
        attention.launch(KimiMlaWeights{qkv_a:w("qa"),q_a_norm:w("qn"),kv_a_norm:w("kn"),q_b:w("qb"),uk:w("uk"),uv:w("uv"),
            gate:w("gate"),output:w("out")},reference,stream.raw)?;
        Ok(())
    });
    stream.drain_or_abort("K6 full MLA case");result?;
    for (label,a,b,count) in [("output",output.buffer,reference.output,rows*7168),("cache",actual.cache,reference.cache,rows*576),
        ("q residual",actual.q_resid,reference.q_resid,rows*1536),("query",actual.query,reference.query,rows*96*576),
        ("attention",actual.attention,reference.attention,rows*96*512)] {
        compare(source,device,a,b,count,&format!("MLA rows={rows} {label}"))?;
    }
    Ok(())
}

fn compare_ffn_truth(source:&Source,device:Device<'_>,actual:Buffer,count:usize,layer:usize,rank:usize,
    stage:&str,path:&str)->Result<()> {
    let truth=read(&source.root.join("ffn-f64-truth.safetensors"),
        &format!("{}.layer{layer}.rank{rank}.{stage}",source.book))?;
    ensure!(truth.info.dtype==DType::Unknown("F64".into()) && truth.info.shape==[1,count] && truth.bytes.len()==count*8,
        "F64 FFN truth geometry");
    let reference=truth.bytes.chunks_exact(8).map(|v|f64::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    let mut bytes=vec![0;count*2];device.run(||device.library.copy_d2h(&mut bytes,actual))?;
    let values=bf16(&bytes);
    let (relative,peak)=errors(&values,&reference)?;
    let (cosine,signed)=error_bias(&values,&reference);
    let radial=cosine.map_or(0.,|value|relative*value);
    let cosine=cosine.map_or_else(||"undefined".into(),|value|value.to_string());
    eprintln!("K6 F64-truth book={} layer={layer} rank={rank} rows=1 path={path} stage={stage} rel_l2={relative} max_abs_over_peak={peak} error_ref_cosine={cosine} radial_fraction={radial} mean_signed_over_mean_abs_ref={signed}",source.book);
    Ok(())
}

fn shared_ffn(device:Device<'_>,base:&Programs<'_>,programs:&Programs<'_>,source:&Source,cfg:&KimiK3Config,
    layer:usize,rows:usize)->Result<()> {
    use cuteafd_ffi::{KimiFfn,KimiFfnMode,KimiFfnBatch,KimiSitu};
    use cuteafd_loader::families::kimi_k3::w8;
    let stream=Stream::new(device)?;
    let packed=w8::pack_ffn(cfg,layer,2,|name|source.weight(name))?;
    let fused=device.run(||Ok(KimiFfn::load(base,3072,KimiFfnMode::Direct)?))?;
    let situ=device.run(||Ok(KimiSitu::load(base,3072)?))?;
    let copy=device.run(||device.library.v41_peer_copy())?;
    let input_bytes=input(rows,7168);
    if rows==1 {
        let truth_input=read(&source.root.join("ffn-f64-truth.safetensors"),"input")?;
        ensure!(truth_input.info.dtype==DType::Bf16 && truth_input.info.shape==[1,7168]
            && truth_input.bytes==input_bytes,"F64 FFN truth must use identical input bytes");
    }
    let input=upload(device,&input_bytes)?;
    let mut rank_cores=Vec::new();
    for (rank,packed) in packed.iter().enumerate() {
        let stem=format!("language_model.model.layers.{layer}.block_sparse_moe.shared_experts");
        let gate=source.projection(device,programs,&format!("{stem}.gate_proj"),"shared_gate_up_tp2",TrellisSide::Output,rank,2,64)?;
        let up=source.projection(device,programs,&format!("{stem}.up_proj"),"shared_gate_up_tp2",TrellisSide::Output,rank,2,64)?;
        let down=source.projection(device,programs,&format!("{stem}.down_proj"),"shared_down_tp2",TrellisSide::Input,rank,2,64)?;
        let gate_weight=upload(device,&packed.gate_up)?;let down_weight=upload(device,&packed.down)?;
        let gate_out=zero(device,rows*3072*2)?;let up_out=zero(device,rows*3072*2)?;
        let gate_up=zero(device,rows*6144*2)?;let gate_up_ref=zero(device,rows*6144*2)?;
        let activated=zero(device,rows*3072*2)?;let activated_ref=zero(device,rows*3072*2)?;
        let output=zero(device,rows*7168*2)?;let reference=zero(device,rows*7168*2)?;
        let retained=zero(device,rows*7168*2)?;
        let result=device.run(|| {
            let partial=super::core_runtime::ffn_projected(gate.projected(),up.projected(),down.projected(),&situ,&copy,rows,
                input.buffer,gate_out.buffer,up_out.buffer,gate_up.buffer,activated.buffer,stream.raw)?;
            down.program.program.output(rows,partial,down.weights.svh.buffer,output.buffer,stream.raw)?;
            crate::shared::dense_trellis::retain_partial(device.library,partial,retained.buffer,rows*7168,stream.raw)?;
            fused.launch(KimiFfnBatch{rows,input:input.buffer,gate_up_weight:gate_weight.buffer,down_weight:down_weight.buffer,
                gate_up:gate_up_ref.buffer,activated:activated_ref.buffer,output:reference.buffer},stream.raw)?;
            Ok(partial)
        });
        stream.drain_or_abort("K6 shared FFN case");let partial=result?;
        check_retained_partial(device,partial,retained.buffer,rows*7168)?;
        compare(source,device,activated.buffer,activated_ref.buffer,rows*3072,&format!("layer={layer} FFN rank={rank} rows={rows} activation"))?;
        compare(source,device,output.buffer,reference.buffer,rows*7168,&format!("layer={layer} FFN rank={rank} rows={rows} output"))?;
        if rows==1 {
            let mut packed=vec![0;6144*2];
            device.run(||device.library.copy_d2h(&mut packed,gate_up_ref.buffer))?;
            let reference_gate=upload(device,&packed[..3072*2])?;
            let reference_up=upload(device,&packed[3072*2..])?;
            for (stage,count,k6,bf16) in [("gate",3072,gate_out.buffer,reference_gate.buffer),
                ("up",3072,up_out.buffer,reference_up.buffer),
                ("activation",3072,activated.buffer,activated_ref.buffer),
                ("output",7168,output.buffer,reference.buffer)] {
                compare_ffn_truth(source,device,k6,count,layer,rank,stage,"k6")?;
                compare_ffn_truth(source,device,bf16,count,layer,rank,stage,"bf16")?;
            }
        }
        rank_cores.push((down,retained,reference));
    }
    compare_meet(device,source,&stream,rows,&rank_cores,&format!("layer={layer} FFN"))?;
    Ok(())
}

#[test]
fn bf16_order_control_changes_only_product_sequence() {
    let k=384;
    let x=(0..k*2).map(|i|(i%251)as u8).collect::<Vec<_>>();
    let w=(0..k*2).map(|i|((i*3+7)%251)as u8).collect::<Vec<_>>();
    let (reversed_x,reversed_w)=(reverse_k_blocks(&x,k),reverse_k_blocks(&w,k));
    assert_ne!(x,reversed_x);
    assert_eq!(reverse_k_blocks(&reversed_x,k),x);
    let dot=|a:&[u8],b:&[u8]|a.chunks_exact(2).zip(b.chunks_exact(2))
        .map(|(a,b)|u64::from(u16::from_le_bytes([a[0],a[1]]))*u64::from(u16::from_le_bytes([b[0],b[1]]))).sum::<u64>();
    assert_eq!(dot(&x,&w),dot(&reversed_x,&reversed_w));
}

#[test]
fn error_bias_distinguishes_aligned_cancelled_and_zero_error() {
    let aligned=error_bias(&[2.,4.],&[1.,2.]);
    assert!((aligned.0.unwrap()-1.).abs()<1e-12);
    assert_eq!(aligned.1,1.);
    let cancelled=error_bias(&[2.,0.],&[1.,1.]);
    assert_eq!(cancelled,(Some(0.),0.));
    let negative=error_bias(&[0.,0.],&[1.,2.]);
    assert!((negative.0.unwrap()+1.).abs()<1e-12);
    assert_eq!(negative.1,-1.);
    assert_eq!(error_bias(&[1.,2.],&[1.,2.]),(None,0.));
}

#[test]
fn layer_diagnostics_collect_failures_without_relaxing_tolerance() {
    let source=Source{root:PathBuf::new(),book:"mcg".into(),failures:Default::default()};
    source.report("exact",&[1.,2.],&[1.,2.]).unwrap();
    assert!(source.failures.borrow().is_empty());
    source.report("outside bar",&[1.01,2.02],&[1.,2.]).unwrap();
    let failures=source.failures.borrow();
    assert_eq!(failures.len(),1);
    assert!(failures[0].contains("book=mcg") && failures[0].contains("max_abs="));
    drop(failures);
    assert!(source.report("nonfinite",&[f64::NAN],&[1.]).is_err());
}

#[test]
#[ignore="requires CPU F64 truth fixtures and test-only dense/split library"]
fn k6_shared_ffn_f64_truth_diagnostics() -> Result<()> {
    let root=PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
    let cfg=KimiK3Config::read(Path::new(&std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?))?;
    // SAFETY: authorized immutable runner-selected base and test-only library paths.
    let library=unsafe{NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};
    // SAFETY: test-only linkage, never used by production serving.
    let fixture=unsafe{NativeLibrary::load(root.join("full-programs/libkimi_k6_full.so"))?};
    let device=Device{library:&library,id:0};
    let base=device.run(||library.programs()?.with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?)))?;
    let programs=device.run(||fixture.programs()?.with_manifest(&root.join("full-programs/dsv4_programs.json")))?;
    for book in ["mcg","mul1"] {
        let source=Source{root:root.join("full-layers"),book:book.into(),failures:Default::default()};
        for layer in [1,3] {shared_ffn(device,&base,&programs,&source,&cfg,layer,1)?;}
        // Composed-stage errors are recorded; model-level KL is the acceptance gate.
    }
    Ok(())
}

#[test]
#[ignore="requires full K6 layer fixtures and test-only dense/split library"]
fn k6_real_core_layers_match_bf16_dequant_state() -> Result<()> {
    let root=PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
    let cfg=KimiK3Config::read(Path::new(&std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?))?;
    // SAFETY: both immutable library paths are selected by the native test runner.
    let library=unsafe{NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};
    // SAFETY: authorized test-only linkage, never used by production serving.
    let fixture=unsafe{NativeLibrary::load(root.join("full-programs/libkimi_k6_full.so"))?};
    let device=Device{library:&library,id:0};
    let base=device.run(||library.programs()?.with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?)))?;
    let programs=device.run(||fixture.programs()?.with_manifest(&root.join("full-programs/dsv4_programs.json")))?;
    let mut failures=Vec::new();
    for book in ["mcg","mul1"] {for rows in [1,8,64] {
        let source=Source{root:root.join("full-layers"),book:book.into(),failures:Default::default()};
        kda_layer(device,&base,&programs,&source,&cfg,rows)?;
        mla_layer(device,&base,&programs,&source,&cfg,rows)?;
        for layer in [1,3] {shared_ffn(device,&base,&programs,&source,&cfg,layer,rows)?;}
        failures.extend(source.failures.into_inner());
    }}
    ensure!(failures.is_empty(),"K6 full-layer numerical failures: {}",failures.join("; "));
    Ok(())
}
