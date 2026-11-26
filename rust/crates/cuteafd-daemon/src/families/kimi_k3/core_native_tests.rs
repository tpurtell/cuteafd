//! Small quantizer-written core fixtures through the real loader and native ABI.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, TensorInfo, TensorRole};
use cuteafd_ffi::{dense_trellis::{DenseTrellis, DenseTrellisGeometry, TrellisDType}, NativeLibrary};
use cuteafd_loader::{formats::{dense_trellis::{load_native_shard, TrellisSide},
    exl3_storage::{derive_storage_map, Exl3Codebook}}, LoadedTensor, read_safetensors_metadata};
use std::{collections::BTreeMap, fs::File, os::unix::fs::FileExt, path::{Path,PathBuf}};

fn tensors(path: &Path) -> Result<BTreeMap<String,LoadedTensor>> {
    let file=File::open(path)?;
    read_safetensors_metadata(path)?.into_iter().map(|meta| {
        let mut bytes=vec![0;usize::try_from(meta.byte_length)?];
        file.read_exact_at(&mut bytes,meta.byte_offset)?;
        let name=meta.name;
        Ok((name.clone(),LoadedTensor {info:TensorInfo{name,file:path.file_name().context("fixture file")?.to_string_lossy().into(),
            dtype:meta.dtype,shape:meta.shape,byte_offset:meta.byte_offset,byte_length:meta.byte_length,
            role:TensorRole::Other,layer_id:None,expert_id:None,is_quantization_metadata:false},
            source_path:path.into(),bytes,elapsed_micros:0,sha256:String::new()}))
    }).collect()
}
pub(super) fn bf16(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks_exact(2).map(|b|f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16)as f64).collect()
}
fn f16(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks_exact(2).map(|b| {
        let h=u16::from_le_bytes([b[0],b[1]]);
        let exponent=(h>>10)&31;
        let mantissa=h&1023;
        let sign=if h&0x8000==0 {1.}else{-1.};
        match exponent {0=>sign*f64::from(mantissa)*2f64.powi(-24),31=>f64::NAN,
            _=>sign*(1.+f64::from(mantissa)/1024.)*2f64.powi(i32::from(exponent)-15)}
    }).collect()
}
pub(super) fn errors(actual: &[f64], reference: &[f64]) -> Result<(f64,f64)> {
    ensure!(!actual.is_empty() && actual.len()==reference.len(),"native oracle extent");
    ensure!(actual.iter().chain(reference).all(|x|x.is_finite()),"native oracle nonfinite");
    let squared=actual.iter().zip(reference).map(|(a,b)|(a-b).powi(2)).sum::<f64>();
    let norm=reference.iter().map(|v|v*v).sum::<f64>();
    let peak=reference.iter().map(|v|v.abs()).fold(0.,f64::max);
    ensure!(norm>0. && peak>0.,"empty oracle signal");
    Ok(((squared/norm).sqrt(),actual.iter().zip(reference).map(|(a,b)|(a-b).abs()).fold(0.,f64::max)/peak))
}

#[test]
#[ignore = "requires K6 numerical fixtures and small-shape SM120 native program exports"]
fn k6_checkpoint_load_forward_matches_dequant() -> Result<()> {
    let root=PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES").context("fixture directory")?);
    let path=root.join("core-k6.safetensors");
    let core=tensors(&path)?;
    let oracle=tensors(&root.join("oracle.safetensors"))?;
    let headers=read_safetensors_metadata(&path)?;
    let modules=derive_storage_map(&headers)?;
    ensure!(modules.len()==22,"expected eleven slices for both codebooks");
    // SAFETY: this is the native library selected by the existing golden runner.
    let library=unsafe{NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};
    // SAFETY: test-only shim links the fixture objects and the runner's base
    // library; production allocation/device symbols remain in the base handle.
    let fixture_library=unsafe{NativeLibrary::load(std::env::var("CUTEAFD_K6_FIXTURE_LIB")?)?};
    let device=Device{library:&library,id:std::env::var("CUTEAFD_CACHE_TEST_DEVICE").unwrap_or_else(|_|"0".into()).parse()?};
    let programs=device.run(||fixture_library.programs()?.with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?)))?;
    let upload=|bytes: &[u8]| -> Result<Allocation<'_>> {
        let allocation=Allocation::new(device,bytes.len().max(16), crate::shared::memory::TEST)?;
        device.run(||library.copy_h2d(allocation.buffer,bytes))?;
        Ok(allocation)
    };
    let stream=device.run(||library.cuda_stream_create())?;
    let result=(|| -> Result<()> {
        let mut numerical_failures=Vec::new();
        let mut cases=0;
        for (stem,module) in &modules {
            ensure!(module.bits==6 && module.input_features==128 && module.output_features==256,"fixture geometry");
            let book=match module.codebook {Exl3Codebook::Mcg=>"mcg",Exl3Codebook::Mul1=>"mul1"};
            let x=&oracle[&format!("{stem}.input_bf16")];
            let weight=&oracle[&format!("{stem}.dequant")];
            let forward=&oracle[&format!("{stem}.forward_f32")];
            ensure!(x.info.dtype==DType::Bf16 && x.info.shape==[8,128]
                && weight.info.dtype==DType::F16 && weight.info.shape==[128,256]
                && forward.info.dtype==DType::F32 && forward.info.shape==[8,256],"oracle header geometry");
            let xf=bf16(&x.bytes);let wf=f16(&weight.bytes);
            let matmul = |weights: &[f64]| (0..8).flat_map(|row|(0..256).map({let xf=&xf;
                move|col|(0..128).map(|k|xf[row*128+k]*weights[k*256+col]).sum::<f64>()})).collect::<Vec<_>>();
            let source=forward.bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())as f64).collect::<Vec<_>>();
            ensure!(errors(&source,&matmul(&wf))?.0<1e-5,"fixture F32 forward differs from dequant F64");
            // Match the BF16-of-dequant baseline, but accumulate its forward in F64.
            let bf16_weight=wf.iter().map(|&v| {
                let bits=(v as f32).to_bits();
                let rounded=bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000;
                f32::from_bits(rounded)as f64
            }).collect::<Vec<_>>();
            let reference=matmul(&bf16_weight);
            for side in [TrellisSide::Output,TrellisSide::Input] {
                let shard=load_native_shard(stem,module,side,0,1,|name|core.get(name).cloned().context("fixture member"))?.shard;
                let side_id=if side==TrellisSide::Output{1}else{2};
                let core_name=format!("kimi_k6_fixture_trellis_s{side_id}_{book}");
                let native=device.run(||Ok(DenseTrellis::load(&programs,
                    ["kimi_k6_fixture_trellis_input_k128_bf16",&core_name,"kimi_k6_fixture_trellis_output_n256_f32"],
                    DenseTrellisGeometry{input:128,output:256,bits:6,side:side_id,rows:64,
                        input_dtype:TrellisDType::Bf16,output_dtype:TrellisDType::F32},book)?))?;
                let w=native.workspace();
                let input=upload(&x.bytes)?;let payload=upload(&shard.payload)?;
                let suh=upload(&shard.suh)?;let svh=upload(&shard.svh)?;
                let rotated=Allocation::new(device,w.rotated, crate::shared::memory::TEST)?;
                let table=upload(&(rotated.buffer.ptr as u64).to_le_bytes())?;
                let scales=Allocation::new(device,w.block_scales, crate::shared::memory::TEST)?;
                let partial=Allocation::new(device,w.partial, crate::shared::memory::TEST)?;
                let accumulator=Allocation::new(device,w.accumulator, crate::shared::memory::TEST)?;
                let output=Allocation::new(device,8*256*4, crate::shared::memory::TEST)?;
                // Drain on every result before the per-case owners leave scope.
                let launched=device.run(|| {
                    native.input(8,input.buffer,suh.buffer,rotated.buffer,scales.buffer,stream)?;
                    native.core(8,table.buffer,payload.buffer,scales.buffer,partial.buffer,accumulator.buffer,stream)?;
                    native.output(8,accumulator.buffer,svh.buffer,output.buffer,stream)?;
                    Ok(())
                });
                // SAFETY: the stream and all queued operands are still owned here.
                let drained=device.run(||unsafe{library.cuda_stream_synchronize(stream)});
                if let Err(error)=drained {
                    library.quarantine_module_after_failed_drain();
                    fixture_library.quarantine_module_after_failed_drain();
                    return Err(error.context("native fixture drain failed; allocation owners and both modules quarantined"));
                }
                launched?;
                let mut bytes=vec![0;8*256*4];
                device.run(||library.copy_d2h_quiescent(&mut bytes,output.buffer))?;
                let actual=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())as f64).collect::<Vec<_>>();
                let (rel,peak)=errors(&actual,&reference)?;
                eprintln!("K6 {stem} side={side_id} rows=8 rel_l2={rel} max_abs_over_peak={peak}");
                cases+=1;
                if rel>=0.004 || peak>=0.01 {
                    numerical_failures.push(format!("{stem} side={side_id} rel_l2={rel} max_abs_over_peak={peak}"));
                }
            }
        }
        ensure!(cases==44,"expected 22 fixtures on both native layouts");
        ensure!(numerical_failures.is_empty(),"K6 native forward exceeds F64 bar: {}",numerical_failures.join("; "));
        Ok(())
    })();
    if library.is_quarantined_after_failed_drain() {
        return result;
    }
    // SAFETY: all per-case work has drained before stream destruction.
    let destroyed=device.run(||unsafe{library.cuda_stream_destroy(stream)});
    result?;destroyed?;Ok(())
}

#[test]
fn forward_gate_rejects_nonfinite_or_wrong_extent() {
    assert!(errors(&[1.,f64::NAN],&[1.,2.]).is_err());
    assert!(errors(&[1.],&[1.,2.]).is_err());
    assert_eq!(errors(&[1.,2.],&[1.,2.]).unwrap(),(0.,0.));
}
