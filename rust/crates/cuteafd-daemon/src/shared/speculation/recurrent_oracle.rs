//! Real checkpoint forward oracle; run only through golden-run's native-test.
use super::{recurrent_forward::{Rank,ProgramsForRank},recurrent_weights::Weights,
    recurrent_cache::{ContextBank,ContextState},dflash::DraftSeq};
use crate::shared::{experts::rtx::dense::{DenseLoad,Operand,Rect},
    memory::{device::{Allocation,Device,Stream},Tag,weight_arena::DeviceWeightArena}};
use anyhow::{ensure,Context,Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{formats::block_dspark::recurrent::{loading::RecurrentSource,runtime::RecurrentRuntimeLayout},
    plan::checkpoint::{CheckpointView,Checkpoint,CoreView},serving_capacity::WeightArenaLayout};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use std::{collections::BTreeMap,path::Path,rc::Rc,os::unix::fs::FileExt};

fn tensor(path:&Path,name:&str)->Result<Vec<u8>> {
    let header=cuteafd_loader::read_safetensors_metadata(path)?.into_iter().find(|t|t.name==name).context("oracle fixture tensor absent")?;
    let mut bytes=vec![0;usize::try_from(header.byte_length)?];
    std::fs::File::open(path)?.read_exact_at(&mut bytes,header.byte_offset)?;Ok(bytes)
}
fn bf16_values(bytes:&[u8])->Vec<f32>{bytes.chunks_exact(2).map(|v|f32::from_bits(u32::from(u16::from_le_bytes([v[0],v[1]]))<<16)).collect()}
fn compare(actual:&[u8],expected:&[u8],name:&str)->Result<()> {
    ensure!(actual.len()==expected.len(),"oracle {name}: extent differs");
    let a=bf16_values(actual);let b=bf16_values(expected);
    let error=a.iter().zip(&b).map(|(a,b)|f64::from(a-b).powi(2)).sum::<f64>().sqrt();
    let norm=b.iter().map(|b|f64::from(*b).powi(2)).sum::<f64>().sqrt().max(1e-12);
    let equal=actual.chunks_exact(2).zip(expected.chunks_exact(2)).filter(|(a,b)|a==b).count();
    eprintln!("recurrent oracle {name}: relative_l2={} bit_equal={}/{}",error/norm,equal,a.len());
    ensure!(error/norm<0.02,"oracle {name}: relative L2 exceeds0.02");Ok(())
}

#[test]
#[ignore="requires real Inkling BF16 MTP tensors, native programs and GPU; golden-run only"]
fn installed_inkling_recurrent_forward_matches_real_weight_reference()->Result<()> {
    // SAFETY: golden-run mounts the matching sealed trusted native library.
    let library=unsafe{cuteafd_ffi::NativeLibrary::load_for_experts(std::env::var("CUTEAFD_NATIVE_LIB")?,
        cuteafd_core::ExpertGeometry::INKLING2)?};
    let programs=library.programs()?.with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
    let snapshot=std::path::PathBuf::from(std::env::var("CUTEAFD_TEST_DRAFT_SNAPSHOT")?);
    let fixture=std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
    let device=Device{library:&library,id:0};
    let stream=Rc::new(Stream::new(device)?);
    let source=RecurrentSource::open(&CheckpointView::select(&snapshot,CoreView::Original)?)?;
    let layout=source.rank(0,1,0)?;
    let weights=Weights::load(device,&source,layout,stream.raw)?;
    let runtime=RecurrentRuntimeLayout::new(&source.config,1,1,2048,2048,4096,library.sm_count()? as usize,
        cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,0,1,0)?;
    let native=ProgramsForRank::load(&programs,&weights,&source.config,&runtime,device)?;
    let rank=Rank::allocate(weights,Rc::clone(&stream),source.config.clone(),runtime,native)?;
    let target=Checkpoint::open_for_role(&snapshot,"recurrent-oracle-target",|name|
        matches!(name,"model.llm.embed.weight"|"model.llm.embed_norm.weight"|"model.llm.unembed.weight"))?;
    let names=["model.llm.embed.weight","model.llm.embed_norm.weight","model.llm.unembed.weight"];
    let target_arena=DeviceWeightArena::new(device,WeightArenaLayout::new("oracle/borrowed-target",
        names.iter().map(|&name|Ok((name.to_owned(),usize::try_from(target.header(name).context("oracle target missing")?.byte_length)?)))
            .collect::<Result<Vec<_>>>()?)?,ArenaTag::weights("oracle/target"))?;
    let dense=DenseLoad::mount(&library);let mut targets=BTreeMap::new();
    for name in names {
        let rect=Rect::tensor(&target,name,None,None)?;let bytes=rect.bytes();let owner=target_arena.view(name,bytes)?;
        dense.operand(Operand{device:0,buffer:owner.buffer,lands:vec![(rect,0)],bytes})?;targets.insert(name,owner);
    }
    dense.release();rank.initialize_derived()?;rank.fold_head(targets["model.llm.unembed.weight"].buffer)?;stream.drain()?;
    for rows in [3usize,1031,2048] {
        let file=fixture.join(format!("rows-{rows}.safetensors"));
        let features=tensor(&file,"features")?;
        library.copy_h2d(rank.buffer("context.taps")?,&features)?;
        let bank=ContextBank{state:std::cell::RefCell::new(ContextState::new(1,2048,4096,rows as u64)?)};
        bank.state.borrow_mut().reset(0,0)?;
        bank.append_ranked(&rank,None,0,0,rank.buffer("context.taps")?,rows)?;
        let sequence=[DraftSeq{slot:0,position:rows,anchor:101,valid_from:0}];
        let pages=bank.state.borrow().pages(&sequence)?;
        library.copy_h2d(rank.buffer("step.ids")?,&101u32.to_le_bytes())?;
        rank.initialize_block(rank.buffer("step.ids")?,targets["model.llm.embed.weight"].buffer,
            targets["model.llm.embed_norm.weight"].buffer,1)?;
        rank.forward_whole(&sequence,&pages)?;
        // SAFETY: last committed feature and head feature are distinct admitted
        // BF16 planes on this ordered stream.
        unsafe{library.copy_d2d_async(rank.buffer("head.f_rnn")?,rank.buffer("cache.last_features")?,4608*2,stream.raw)?;}
        rank.recurrent_tokens(1,rank.buffer("step.ids")?,targets["model.llm.unembed.weight"].buffer)?;
        stream.drain()?;
        let expected=tensor(&file,"hidden")?;let mut actual=vec![0;expected.len()];
        stream.copy_d2h(&mut actual,Buffer{bytes:expected.len(),..rank.buffer("step.normalized")?})?;
        compare(&actual,&expected,&format!("rows{rows}.hidden"))?;
        let expected=tensor(&file,"tokens")?;let mut actual=vec![0;expected.len()];
        stream.copy_d2h(&mut actual,Buffer{bytes:expected.len(),..rank.buffer("head.tokens")?})?;
        ensure!(actual==expected,"recurrent oracle rows{rows}: seven predicted token ids differ");
    }
    Ok(())
}

/// Real target head weights, all nine stages of the unchanged recurrent token
/// loop, and deterministic inputs to isolate the head from a served target.
#[test]
#[ignore = "requires an Inkling MC checkpoint, native programs and GPU; golden-run only"]
fn installed_inkling_recurrent_head_baseline() -> Result<()> {
    use super::recurrent_head::HeadStage;
    use crate::{families::inkling2::{layers::Kernels, weights::{ProjectionWeight, TargetWeights}},
        shared::memory::device::Event};
    use cuteafd_loader::{families::inkling2::{Inkling2Config, InklingKvCache,
        weight_layout::{CoordinatorWeights, HEAD}}};

    let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let mode = std::env::var("CUTEAFD_TEST_CORE_VIEW")?.parse::<CoreView>()?;
    let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
    // SAFETY: golden-run mounts the matching sealed trusted native library.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load_for_experts(std::env::var("CUTEAFD_NATIVE_LIB")?,
        cuteafd_core::ExpertGeometry::INKLING2)? };
    let programs = library.programs()?.with_manifest(&manifest)?;
    let device = Device { library: &library, id: 0 };
    let stream = Rc::new(Stream::new(device)?);
    let source = RecurrentSource::open(&CheckpointView::select(&snapshot, CoreView::Original)?)?;
    let weights = Weights::load(device, &source, source.rank(0, 1, 0)?, stream.raw)?;
    let config = &source.config;
    let runtime = RecurrentRuntimeLayout::new(config, 64, 64, 512, 2048, 4096,
        library.sm_count()? as usize, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, 0, 1, 0)?;
    let native = ProgramsForRank::load(&programs, &weights, config, &runtime, device)?;
    let rank = Rank::allocate(weights, Rc::clone(&stream), config.clone(), runtime, native)?;

    let view = CheckpointView::select(&snapshot, mode)?;
    let checkpoint = Checkpoint::open_view_for_role(&view, "recurrent-head-bench", |name| name.starts_with(HEAD))?;
    let cfg = Inkling2Config::from_hf(&checkpoint.config)?;
    let layout = CoordinatorWeights::head_only(&checkpoint, &cfg)?;
    let target = TargetWeights::load(device, &checkpoint, &layout)?;
    let stored = layout.stored_head().unwrap_or(config.padded_vocab);
    let f32_stage = Allocation::new(device, 64*stored*4, Tag::drafter_workspace("head-bench/f32"))?;
    let bf16_stage = Allocation::new(device, 64*stored*2, Tag::drafter_workspace("head-bench/bf16"))?;
    let mut kernels = Kernels::bind(&programs, InklingKvCache::Bf16)?;
    let fp8_bytes = layout.fp8_scratch(&serde_json::from_slice(&std::fs::read(&manifest)?)?, 64, 64)?;
    kernels.bind_compact(&programs, Rc::clone(&stream), &layout, 64, 64, fp8_bytes)?;

    if mode != CoreView::Original && std::env::var_os("CUTEAFD_TEST_FOLD_HEAD").is_some() {
        const SAMPLE: usize = 128;
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_OUT")?);
        let h = config.transformer.hidden;
        let r = config.transformer.markov_rank;
        let transpose = Allocation::new(device, 32*stored*4, Tag::drafter_workspace("head-fold-oracle/transpose"))?;
        let weight = target.projection("model.llm.unembed.weight")?;
        let download = |buffer:Buffer, bytes:usize| -> Result<Vec<u8>> {
            ensure!(bytes<=buffer.bytes,"fold oracle download extent");
            let mut host=vec![0;bytes];
            stream.copy_d2h(&mut host,Buffer{bytes,..buffer})?;
            Ok(host)
        };
        if let ProjectionWeight::Fp8(w)=weight {
            std::fs::write(root.join("fold-fp8.bin"),download(w.values.buffer,SAMPLE*h)?)?;
            std::fs::write(root.join("fold-scales.bin"),download(w.scales.buffer,h/128*4)?)?;
        }
        let mut f32 = Vec::new();
        rank.fold_compact_head(&programs, stored, f32_stage.buffer, transpose.buffer, |input,output,rows| {
            kernels.project_compact(&stream,input,weight,output,rows,h,stored)?;
            stream.drain()?;
            let data = download(output,rows*stored*4)?;
            for row in data.chunks_exact(stored*4) { f32.extend_from_slice(&row[..SAMPLE*4]); }
            Ok(())
        })?;
        ensure!(f32.len()==(r+1)*SAMPLE*4,"compact fold oracle projection count differs");
        std::fs::write(root.join("fold-f32.bin"),f32)?;
        std::fs::write(root.join("fold-weight.bin"),download(rank.weights.derived("derived.fold_weight")?.buffer,SAMPLE*r*2)?)?;
        std::fs::write(root.join("fold-bias.bin"),download(rank.weights.derived("derived.fold_bias")?.buffer,SAMPLE*2)?)?;
        eprintln!("RECURRENT_HEAD_FOLD_CAPTURE {mode} {SAMPLE} rows");
        if std::env::var_os("CUTEAFD_TEST_WIDE_HEAD").is_none() { return Ok(()); }
    }

    // No fold is timed. Synthetic nonzero folded operands do equivalent work
    // without adding a second head or depending on a future compact fold.
    for (index, buffer) in [rank.buffer("step.normalized")?, rank.buffer("head.f_rnn")?,
        rank.weights.derived("derived.fold_weight")?.buffer, rank.weights.derived("derived.fold_bias")?.buffer]
        .into_iter().enumerate() {
        let pattern = [0x3c00u16, 0x3b80, 0x0000, 0xbb80, 0xbc00, 0x3c80, 0xbc80];
        let data = (0..buffer.bytes/2).flat_map(|j| pattern[(j ^ (j >> 5) ^ index) % pattern.len()].to_le_bytes())
            .collect::<Vec<_>>();
        library.copy_h2d(buffer, &data)?;
    }
    let anchors = rank.buffer("step.ids")?;
    let mut ids = vec![0; anchors.bytes];
    for (i, dst) in ids.chunks_exact_mut(4).take(64).enumerate() {
        dst.copy_from_slice(&(101 + i as u32).to_le_bytes());
    }
    library.copy_h2d(anchors, &ids)?;

    if mode == CoreView::Original && std::env::var_os("CUTEAFD_TEST_BATCHED_HEAD").is_some() {
        crate::shared::thermal::monitor("recurrent-head-oracle");
    }
    const SAMPLES: usize = 12;
    const STAGES: usize = HeadStage::Copies as usize;
    let predictions = config.predictions();
    let h = config.transformer.hidden;
    let n = config.padded_vocab;
    let narrow = library.f32_bf16_narrow()?;
    let head = || target.projection("model.llm.unembed.weight");
    let head_bytes = match head()? {
        ProjectionWeight::Bf16(w) => w.buffer.bytes,
        ProjectionWeight::Trellis(w) => w.payload.buffer.bytes + w.suh.buffer.bytes + w.svh.buffer.bytes,
        ProjectionWeight::Fp8(w) => w.values.buffer.bytes + w.scales.buffer.bytes,
    };
    if mode != CoreView::Original && std::env::var_os("CUTEAFD_TEST_WIDE_HEAD").is_some() {
        use cuteafd_ffi::{draft_projection::ProjectionOutput, native_attention::Fp8Projections};
        use crate::shared::dense_trellis::{Projection, ProjectionLayout};
        use cuteafd_ffi::dense_trellis::{DenseTrellisDispatch, DenseTrellisPolicy, TrellisDType};
        let live = 448;
        let input = Allocation::new(device, live*h*2, Tag::drafter_workspace("head-screen/input"))?;
        let data = (0..live*h).flat_map(|i| [0x3b80u16, 0xbb80, 0x3c00, 0xbc00][(i ^ (i >> 6)) % 4].to_le_bytes())
            .collect::<Vec<_>>();
        library.copy_h2d(input.buffer, &data)?;
        let output = Allocation::new(device, live*stored*4, Tag::drafter_workspace("head-screen/output"))?;
        let (mut trellis, mut fp8) = (None, None);
        match head()? {
            ProjectionWeight::Trellis(w) => {
                let mut policy = DenseTrellisPolicy::from_package_geometry_bits(&programs, h, stored, 1, w.bits, "mcg")?;
                policy.output_program = format!("dense_trellis_k6_trellis_output_n{stored}_f32");
                let projection = Projection { program: DenseTrellisDispatch::load(&programs, policy, live, TrellisDType::F32)? };
                let scratch = ProjectionLayout::new(&projection.program, live)?.allocate_packed(device, Tag::drafter_workspace("head-screen/trellis"))?;
                trellis = Some((projection, scratch));
            }
            ProjectionWeight::Fp8(_) => {
                let fp8_kernel = Fp8Projections::load(&programs, &[(h, stored)], live, ProjectionOutput::F32)?;
                let scratch = Allocation::new(device, fp8_kernel.scratch_bytes(h, stored)?, Tag::drafter_workspace("head-screen/fp8"))?;
                fp8 = Some((fp8_kernel, scratch));
            }
            ProjectionWeight::Bf16(_) => unreachable!(),
        }
        let mut prefix = None;
        for rows in [7, 14, 21, 28, 35, 42, 49, 56, 64, 112, 224, 448] {
            let project = || -> Result<()> {
                match head()? {
                    ProjectionWeight::Trellis(w) => {
                        let (projection, scratch) = trellis.as_ref().unwrap();
                        projection.local(rows, input.buffer, w, scratch, output.buffer, stream.raw)?;
                    }
                    ProjectionWeight::Fp8(w) if rows<=64 => {
                        kernels.project_compact(&stream, input.buffer, ProjectionWeight::Fp8(w), output.buffer, rows, h, stored)?;
                    }
                    ProjectionWeight::Fp8(w) => {
                        let (project, scratch) = fp8.as_ref().unwrap();
                        // SAFETY: all active views are checked for this retained, same-stream projection.
                        unsafe { project.launch(input.buffer.ptr, w.values.buffer.ptr, w.scales.buffer.ptr,
                            output.buffer, scratch.buffer, rows, h, stored, stream.raw)?; }
                    }
                    ProjectionWeight::Bf16(_) => unreachable!(),
                }
                Ok(())
            };
            project()?;
            if rows<=56 {
                stream.drain()?;
                let mut values=vec![0;rows*stored*4];
                let bytes=values.len();
                stream.copy_d2h(&mut values,Buffer{bytes,..output.buffer})?;
                if let Some(previous)=&prefix {
                    let previous: &Vec<u8> = previous;
                    ensure!(values.starts_with(previous), "{mode} head prefix changes at M={rows}");
                }
                prefix=Some(values);
            }
            let mut start = Event::timing(device)?;
            let mut end = Event::timing(device)?;
            start.record(&stream)?;
            for _ in 0..64 { project()?; }
            end.record(&stream)?;
            stream.drain()?;
            // SAFETY: both events were recorded and drained on the same stream.
            let us = f64::from(unsafe { library.cuda_event_elapsed_ms(start.raw, end.raw)? }) * 1000.0 / 64.0;
            eprintln!("RECURRENT_HEAD_WIDTH {}", serde_json::json!({"view": mode.name(), "rows": rows,
                "base_us": us, "roof_us": head_bytes as f64 / 1_792_000.0, "prefix_exact": (rows<=56).then_some(true)}));
        }
    }
    for rows in [1, 4, 8, 16, 64] {
        let base = |hidden: Buffer, output: Buffer, rows: usize| -> Result<()> {
            let weight = head()?;
            if let ProjectionWeight::Bf16(w) = weight {
                // SAFETY: same retained BF16 operands as the original head.
                unsafe { rank.programs.plain.launch_checked(hidden.ptr, w.buffer.ptr, output, rows, h, n, stream.raw)?; }
                return Ok(());
            }
            kernels.project_compact(&stream, hidden, weight, f32_stage.buffer, rows, h, stored)?;
            // SAFETY: typed scratch covers physical row pitch; copy strips only
            // compact padding and leaves the unchanged BF16 baseline epilogue.
            unsafe {
                narrow.launch(f32_stage.buffer.ptr, bf16_stage.buffer.ptr, rows*stored, stream.raw)?;
                library.copy_d2d_2d_async(output, n*2, bf16_stage.buffer, stored*2, n*2, rows, stream.raw)
            }
        };
        rank.recurrent_tokens_measured(rows, anchors, None, base, |_, _| Ok(()))?;
        stream.drain()?;
        let mut totals = [[0.0; STAGES]; 7];
        for _ in 0..SAMPLES {
            let mut events = (0..predictions*(STAGES+1)).map(|_| Event::timing(device)).collect::<Result<Vec<_>>>()?;
            rank.recurrent_tokens_measured(rows, anchors, None, base,
                |step, stage| events[step*(STAGES+1)+stage as usize].record(&stream))?;
            stream.drain()?;
            for (step, stages) in totals.iter_mut().enumerate() {
                for (stage, total) in stages.iter_mut().enumerate() {
                    let i = step*(STAGES+1) + stage;
                    // SAFETY: both events are recorded and drained on this stream.
                    *total += f64::from(unsafe { library.cuda_event_elapsed_ms(events[i].raw, events[i+1].raw)? }) * 1e3;
                }
            }
        }
        if mode == CoreView::Original && std::env::var_os("CUTEAFD_TEST_BATCHED_HEAD").is_some() {
            let read_tokens = |out: Buffer| -> Result<Vec<u8>> {
                let mut bytes = vec![0; rows*predictions*4];
                stream.copy_d2h(&mut bytes, Buffer { bytes: rows*predictions*4, ..out })?;
                Ok(bytes)
            };
            let old = rank.recurrent_tokens_measured(rows, anchors, None, base, |_, _| Ok(()))?;
            stream.drain()?;
            let expected = read_tokens(old)?;
            let new = rank.recurrent_tokens(rows, anchors, match head()? {
                ProjectionWeight::Bf16(w) => w.buffer,
                _ => unreachable!("original uses a BF16 head"),
            })?;
            stream.drain()?;
            ensure!(expected == read_tokens(new)?, "batched original head changed recurrent tokens at M={rows}");
            let time = |batch: bool| -> Result<(f64, serde_json::Value)> {
                use cuteafd_core::thermal;
                let mut begin = Event::timing(device)?;
                let mut end = Event::timing(device)?;
                let from = thermal::now_s();
                begin.record(&stream)?;
                for _ in 0..64 {
                    rank.recurrent_tokens_measured(rows, anchors, batch.then_some(rows*predictions), base, |_, _| Ok(()))?;
                }
                end.record(&stream)?;
                stream.drain()?;
                // SAFETY: both events finished on the ordered CUDA stream.
                let us = f64::from(unsafe { library.cuda_event_elapsed_ms(begin.raw, end.raw)? }) * 1e3 / 64.0;
                Ok((us, thermal::window(from, thermal::now_s())))
            };
            let (old, old_thermal) = time(false)?;
            let (new, new_thermal) = time(true)?;
            eprintln!("RECURRENT_HEAD_BF16_PAIR {}", serde_json::json!({"rows": rows,
                "serial_round_us": old, "batched_round_us": new, "serial_thermal": old_thermal,
                "batched_thermal": new_thermal, "exact_tokens": true}));
        }
        let mean = (0..STAGES).map(|stage| totals.iter().map(|step| step[stage]).sum::<f64>() / (SAMPLES*predictions) as f64)
            .collect::<Vec<_>>();
        eprintln!("RECURRENT_HEAD_BASELINE {}", serde_json::json!({"view": mode.name(), "rows": rows,
            "head_bytes": head_bytes, "roof_us": head_bytes as f64 / 1_792_000.0,
            "gather_us": mean[0], "recurrent_us": mean[1], "base_us": mean[2], "correction_us": mean[3],
            "bias_us": mean[4], "finish_us": mean[5], "widen_us": mean[6], "argmax_us": mean[7],
            "copies_us": mean[8], "total_us": mean.iter().sum::<f64>(),
            "per_step_total_us": totals.iter().map(|step| step.iter().sum::<f64>() / SAMPLES as f64).collect::<Vec<_>>() }));
    }
    Ok(())
}
