//! Cold coordinator-only native prefix golden, with bounded layer residency.
use super::{admission, experts::{Mxfp8, Mxfp8Layout}, head::Head, layers::{self, Kernels},
    owners::UnitOwner, weights::TargetWeights};
use crate::shared::memory::device::{Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::fp8_moe::Fp8MoeMetadata;
use cuteafd_loader::{families::inkling2::{execution::UnitLayout, weight_layout::CoordinatorWeights, Inkling2Config}, plan::Checkpoint};
use std::{path::PathBuf, rc::Rc};

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)] pub attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env="CUTEAFD_NATIVE_LIB")] native_lib: PathBuf,
    #[arg(long, default_value="/opt/cuteafd/share/PROGRAMS.json")] manifest: PathBuf,
    /// JSON array of token ids; no tokenizer or prompt preprocessing in golden.
    #[arg(long)] tokens: PathBuf,
    /// Prefix stops after this zero-based layer; no full-model claim if <47.
    #[arg(long, default_value_t=0)] last_layer: usize,
    #[arg(long, default_value_t=0)] device: i32,
    /// Existing parent; files are created below this new directory.
    #[arg(long)] save_logits: PathBuf,
    #[arg(long)] nll: bool,
    /// Optional raw FP32 matching partial-prefix reference logits.
    #[arg(long)] golden: Option<PathBuf>,
    #[arg(long)] check_headers: bool,
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || run(args)).await?
}

fn raw(buffer: cuteafd_ffi::CuteafdDeviceBuffer, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
    ensure!(bytes <= buffer.bytes, "Inkling golden view exceeds admitted owner");
    Ok(cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..buffer })
}

fn run(args: GoldenArgs) -> Result<()> {
    let view = args.attention.selected_view.clone().map(Ok).unwrap_or_else(|| args.attention.select(&args.snapshot))?;
    args.attention.validate_view(&view)?;
    ensure!(view.mode() == cuteafd_loader::plan::checkpoint::CoreView::Original, "Inkling stage2 golden currently qualifies original BF16 core only");
    let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
    let cfg = Inkling2Config::from_hf(&checkpoint.config)?;
    ensure!(args.last_layer < cfg.layers, "Inkling --last-layer outside checkpoint");
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(&args.tokens)?)?;
    ensure!((2..=4096).contains(&tokens.len()) && tokens.iter().all(|&t| (t as usize)<cfg.unpadded_vocab_size),
        "Inkling golden needs2..4096 valid token ids");
    let rows = tokens.len()-1;
    let unit_layout = UnitLayout::new(&cfg, rows, tokens.len().max(64))?;
    let target_layout = CoordinatorWeights::golden(&checkpoint, &cfg, &(0..=args.last_layer).collect::<Vec<_>>())?;
    let package = args.native_lib.parent().context("Inkling native library directory")?
        .join("fp8/fp8-inkling2-mxfp8/tp1");
    // SAFETY: only the trusted slot's generated static package ABI is queried.
    let metadata = unsafe { Fp8MoeMetadata::read(&package) }?;
    let expert_layouts = (0..=args.last_layer).map(|index| Mxfp8Layout::plan(&checkpoint, &cfg, index))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!args.save_logits.exists(), "Inkling golden output must be new");
    if args.check_headers {
        println!("Inkling golden headers: layers={} rows={} nativeMXFP8=true partial={}", args.last_layer+1, rows,
            args.last_layer+1 != cfg.layers);
        return Ok(());
    }
    // SAFETY: library outlives every loaded program, stream and physical owner.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib) }?;
    let device = Device { library: &library, id: args.device };
    let programs = device.run(|| Ok(library.programs()?.with_manifest(&args.manifest)?))?;
    let kernels = device.run(|| Kernels::bind(&programs))?;
    let (free, total) = device.run(|| library.cuda_memory_info())?;
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    let _facts = admission::admit(&checkpoint, &cfg, &target_layout, &expert_layouts[0], metadata.scratch_for(rows)?,
        &unit_layout, rows, free as u64, total as u64, host)?;
    let stream = Rc::new(Stream::new(device)?);
    let owner = UnitOwner::new(Rc::clone(&stream), unit_layout)?;
    let weights = TargetWeights::load(device, &checkpoint, &target_layout)?;
    let head = Head::new(Rc::clone(&stream), &cfg, rows)?;
    let positions = (0..rows).flat_map(|n| (n as i64).to_le_bytes()).collect::<Vec<_>>();
    let rings = (0..rows).flat_map(|n| ((n%1088) as i64).to_le_bytes()).collect::<Vec<_>>();
    let pages = (0..owner.layout.buffers["page_table"]/4).flat_map(|n| (n as i32).to_le_bytes()).collect::<Vec<_>>();
    stream.drain()?;
    device.run(|| {
        library.copy_h2d(owner.buffer("positions")?, &positions)?;
        library.copy_h2d(owner.buffer("cache_slots")?, &positions)?;
        library.copy_h2d(owner.buffer("ring_slots")?, &rings)?;
        library.copy_h2d(owner.buffer("page_table")?, &pages)
    })?;
    head.embed(&kernels, &owner, &weights, &cfg, &tokens[..rows])?;
    for (index, layout) in expert_layouts.iter().enumerate() {
        // Reuse one independently retired expert arena: no one-card fullmodel
        // fit requirement, and native expert precision is never changed.
        // Every layer owns separate causal histories; the cold whole-sequence
        // unit reuses storage only after the preceding layer has drained.
        device.run(|| {
            for name in ["attn_conv_history", "mlp_conv_history", "k_conv_history", "v_conv_history", "ring", "global_cache"] {
                let buffer = owner.buffer(name)?;
                // SAFETY: no queued prior-layer reader remains after drain.
                unsafe { library.cuda_zero_bytes_async(buffer, buffer.bytes, stream.raw)?; }
            }
            Ok(())
        })?;
        let experts = device.run(|| Mxfp8::load(Rc::clone(&stream), layout, &metadata, &package, &cfg, rows))?;
        device.run(|| super::prefill::layer(&kernels, &owner, &weights, &experts, &cfg, index, rows))?;
        stream.drain()?;
        println!("Inkling native layer{index} complete rows={rows}");
    }
    let candidates = super::layers::residual_site(&cfg, args.last_layer)?.mlp_candidates;
    let logits = head.project(&kernels, &owner, &weights, &cfg, rows, candidates)?;
    stream.drain()?;
    let mut bytes = vec![0u8; rows*cfg.unpadded_vocab_size*4];
    let download_bytes = bytes.len();
    device.run(|| library.copy_d2h(&mut bytes, raw(logits, download_bytes)?))?;
    let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "Inkling golden nonfinite logits");
    let mut nll = 0f64;
    for (row, &next) in values.chunks_exact(cfg.unpadded_vocab_size).zip(&tokens[1..]) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        nll += maximum + row.iter().map(|&v| (v as f64-maximum).exp()).sum::<f64>().ln() - row[next as usize] as f64;
    }
    nll /= rows as f64;
    let comparison = if let Some(path) = &args.golden {
        let expected = std::fs::read(path)?;
        ensure!(expected.len() == bytes.len(), "Inkling partial reference logits shape differs");
        let mut square = 0f64;
        let mut norm = 0f64;
        let mut dot = 0f64;
        let mut actual_norm = 0f64;
        for (&actual, expected) in values.iter().zip(expected.chunks_exact(4)) {
            let expected = f32::from_le_bytes(expected.try_into().unwrap()) as f64;
            ensure!(expected.is_finite(), "Inkling reference nonfinite logits");
            square += (actual as f64-expected).powi(2); norm += expected*expected;
            dot += actual as f64*expected; actual_norm += (actual as f64).powi(2);
        }
        let rms = (square/norm.max(1e-30)).sqrt();
        let cosine = dot/(norm*actual_norm).sqrt().max(1e-30);
        ensure!(rms<=0.02 && cosine>=0.999, "Inkling partial logits fail RMS/cosine bar: {rms}/{cosine}");
        let mut reference_nll = 0f64;
        for (row, &next) in expected.chunks_exact(cfg.unpadded_vocab_size*4).zip(&tokens[1..]) {
            let row = row.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect::<Vec<_>>();
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            reference_nll += maximum + row.iter().map(|&v| (v-maximum).exp()).sum::<f64>().ln() - row[next as usize];
        }
        reference_nll /= rows as f64;
        ensure!((nll-reference_nll).abs() <= 0.03, "Inkling partial NLL exceeds0.03 bar: {nll}/{reference_nll}");
        serde_json::json!({"normalized_rms":rms,"cosine":cosine,"reference_nll":reference_nll,
            "nll_delta":nll-reference_nll})
    } else { serde_json::Value::Null };
    std::fs::create_dir(&args.save_logits)?;
    std::fs::write(args.save_logits.join("logits.bin"), bytes)?;
    let report = serde_json::json!({"family":"inkling2","native":true,"partial":args.last_layer+1 != cfg.layers,
        "last_layer":args.last_layer,"rows":rows,"vocab":cfg.unpadded_vocab_size,"mean_nll":nll,
        "reference_comparison":comparison});
    std::fs::write(args.save_logits.join("result.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("Inkling native golden mean NLL={nll:.8} partial={} rows={rows}", args.last_layer+1 != cfg.layers);
    Ok(())
}
