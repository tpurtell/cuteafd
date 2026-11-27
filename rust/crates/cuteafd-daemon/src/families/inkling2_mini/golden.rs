//! Full-model TP1 golden, reference logit/NLL comparison, width invariance (N=1 vs N),
//! exact-width graph vs eager parity, and accepted-prefix commit invariance for Inkling 2 Mini.
//!
//! Authoritative reference precision: commit `f1940f40eb`.
use crate::shared::{
    prefill_pipeline::{sequence_units, Unit, UnitSegment},
    prefix::PrefixArgs,
    token_io::DeviceLogits,
};
use anyhow::{ensure, Result};
use cuteafd_loader::{
    families::inkling2_mini::{Inkling2MiniConfig, InklingKvCache},
    placement::planner::PrefillShape,
    plan::Checkpoint,
};
use std::{path::PathBuf, sync::Arc};

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub attention: crate::cli::CoreViewArgs,
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long = "kv-cache", default_value = "fp8")]
    pub kv_format: InklingKvCache,
    /// JSON array of token ids.
    #[arg(long)]
    pub tokens: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub save_logits: PathBuf,
    #[arg(long)]
    pub nll: bool,
    /// Optional raw FP32 reference logits (`f1940f40eb`) for RMS / cosine / KL / NLL comparison.
    #[arg(long)]
    pub golden: Option<PathBuf>,
    #[arg(long)]
    pub check_headers: bool,
    /// Compare teacher-forced decode widths (e.g., `1,2,4,8`) byte-for-byte.
    #[arg(long, value_delimiter = ',')]
    pub width_invariance: Vec<usize>,
    /// Verify byte-exact equality between eager and graphed decode replays.
    #[arg(long)]
    pub graph_parity: bool,
    /// Optional total packed row count for multi-slot shared-graph invariance check (e.g. 40).
    #[arg(long, default_value_t = 0)]
    pub slot_parity_rows: usize,
    /// Comma-separated slot counts to check at `slot_parity_rows` (e.g. `15,16`), proving one
    /// captured graph replays identically across different slot counts with byte-exact eager parity.
    #[arg(long, value_delimiter = ',')]
    pub slot_parity_slots: Vec<usize>,
    #[arg(long, default_value_t = 64)]
    pub prefill: usize,
    #[arg(long)]
    pub pool_tokens: Option<u64>,
    #[arg(long, default_value_t = 4)]
    pub max_sequences: usize,
    #[arg(long, default_value_t = 64)]
    pub decode_rows: u32,
    #[arg(long, default_value_t = 2048)]
    pub prefill_rows: usize,
    #[arg(long, default_value_t = 2)]
    pub prefill_waves: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub dspark: bool,
    #[arg(long)]
    pub draft: Option<PathBuf>,
    #[arg(long, requires = "draft")]
    pub draft_oracle: Option<PathBuf>,
    #[arg(long, default_value = "fp8")]
    pub draft_kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    #[command(flatten)]
    pub prefix: PrefixArgs,
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || run(args)).await?
}

fn read_logits(library: &cuteafd_ffi::NativeLibrary, device_id: i32, logits: &DeviceLogits) -> Result<Vec<f32>> {
    let count = logits.rows * logits.vocab;
    let mut bytes = vec![0u8; count * 4];
    let buf = cuteafd_ffi::CuteafdDeviceBuffer {
        ptr: logits.ptr as *mut libc::c_void,
        bytes: bytes.len(),
        device_id,
        flags: 0,
    };
    // Engine streams are created with `cudaStreamNonBlocking`, so null-stream `cudaMemcpy`
    // (`copy_d2h`) does not order against `logits.stream`; drain before reading `logits.ptr`
    // while the head logits buffer remains live during the callback.
    library.cuda_device_synchronize()?;
    library.copy_d2h(&mut bytes, buf)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn run(mut args: GoldenArgs) -> Result<()> {
    let view = args
        .attention
        .selected_view
        .clone()
        .map(Ok)
        .unwrap_or_else(|| args.attention.select(&args.snapshot))?;
    args.attention.validate_view(&view)?;
    let checkpoint = Checkpoint::open_view(&view)?;
    let cfg = Inkling2MiniConfig::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(&args.tokens)?)?;
    ensure!(
        (2..=4096).contains(&tokens.len())
            && tokens
                .iter()
                .all(|&t| (t as usize) < cfg.text.unpadded_vocab_size),
        "Inkling 2 Mini golden needs 2..4096 valid token ids"
    );
    if args.check_headers {
        println!(
            "Inkling 2 Mini golden headers: layers={} visits={} rows={} kv={} precision_oracle=f1940f40eb",
            cfg.text.layers,
            cfg.total_visits(),
            tokens.len() - 1,
            args.kv_format.name()
        );
        return Ok(());
    }
    ensure!(!args.save_logits.exists(), "Inkling 2 Mini golden output path must be new");
    args.prefix.resolve_entries(args.max_sequences);

    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::INKLING2_MINI)
        .map_err(|prev| anyhow::anyhow!("Mini golden requires isolated expert geometry, found {prev:?}"))?;
    // SAFETY: library outlives all device/program owners created inside this blocking task.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let vision = crate::shared::vision::still::Options {
        vision: cuteafd_loader::plan::MediaMode::Off,
        vision_replicas: 1,
        media_cache_bytes: None,
        vision_peers: Vec::new(),
        encoder_plan_hash: None,
        encoder_revision: None,
    };
    let (mut execution, _inputs, admitted, _encoding, _media) = super::startup::whole(
        &library,
        &programs,
        &view,
        &checkpoint,
        &cfg,
        super::startup::Options {
            device: args.device,
            manifest: &args.manifest,
            native_lib: &args.native_lib,
            prefill: PrefillShape {
                rows: args.prefill_rows as u64,
                waves: args.prefill_waves,
            },
            decode_rows: args.decode_rows,
            sequences: args.max_sequences as u64,
            context_tokens: tokens.len().max(4096) as u64,
            pool_tokens: args.pool_tokens,
            kv_format: args.kv_format,
            drafter: (args.dspark && args.draft.is_none()).then_some(args.draft_kv_cache),
            dflash: args.draft.as_deref().map(|p| (p, args.draft_kv_cache)),
            prefix: &args.prefix,
            resident_weights: crate::shared::memory::budget::resident_weight_limit(),
            expert_usage: false,
            prewarm_graphs: false,
            vision: &vision,
        },
    )?;

    if let Some(oracle) = &args.draft_oracle {
        let binding = execution.draft.as_ref().ok_or_else(|| anyhow::anyhow!("Mini draft oracle missing body"))?;
        let embed = execution.target.operand("model.llm.embed.weight")?.buffer;
        let norm = execution.target.operand("model.llm.embed_norm.weight")?.buffer;
        let head = execution.target.operand("model.llm.unembed.weight")?.buffer;
        return binding.check_oracle(oracle, embed, norm, head);
    }

    let pages_needed = tokens.len().div_ceil(64) as u32;
    let pool_pages = admitted.solved.pool_tokens.div_ceil(64) as u32;
    ensure!(pages_needed * 2 <= pool_pages, "Mini golden pool too small for sequence + parity check");

    let rows = tokens.len() - 1;
    let vocab = cfg.text.unpadded_vocab_size;
    let mut values = vec![0f32; rows * vocab];
    let prefix = super::cache::MiniPrefix::for_execution(&execution);
    let mut placement = super::cache::Placement {
        units: (0..pages_needed).collect(),
        slot: 0,
        len: 0,
        state_len: 0,
        media: None,
    };
    prefix.bind_slot(&mut placement, Some(rows))?;

    let mut units = sequence_units(rows, args.prefill_rows, args.prefill_waves as usize, cfg.total_visits())?;
    for u in &mut units {
        u.segments = vec![UnitSegment {
            sequence: 0,
            rows: 0..u.rows.len(),
            tokens: u.rows.clone(),
            prompt_end: u.rows.end == rows,
            snapshot: false,
            head: true,
        }];
    }
    let mut placements = [placement.clone()];
    let copy_lib = &*library;
    let device_id = args.device;
    execution.prefill(
        &mut placements,
        &[&tokens[..rows]],
        &units,
        super::prefill::HeadProjection::AllHeadRows,
        &mut |_seq, at_1based, logits| {
        let tile = read_logits(copy_lib, device_id, &logits)?;
        let row0 = at_1based - 1;
        values[row0 * vocab..(row0 + logits.rows) * vocab].copy_from_slice(&tile);
        Ok(())
    })?;
    placement = placements[0].clone();
    ensure!(values.iter().all(|v| v.is_finite()), "Inkling 2 Mini golden nonfinite logits");

    // Optional N=1 vs N width invariance & graph vs eager byte-exact parity checks.
    let mut width_report = serde_json::Value::Null;
    if !args.width_invariance.is_empty() || args.graph_parity {
        let prefix_len = args.prefill.min(rows.saturating_sub(8)).max(1);
        let tail_len = (rows - prefix_len).min(8);
        ensure!(tail_len >= 1, "Mini width/graph parity requires tokens past --prefill");
        let widths = if args.width_invariance.is_empty() {
            vec![1usize]
        } else {
            args.width_invariance.clone()
        };
        let mut by_width: Vec<(usize, Vec<u32>)> = Vec::new();
        for &w in &widths {
            ensure!(w >= 1 && w <= tail_len, "width {w} exceeds tail_len {tail_len}");
            let mut p = super::cache::Placement {
                units: (pages_needed..2 * pages_needed).collect(),
                slot: 1,
                len: 0,
                state_len: 0,
                media: None,
            };
            prefix.bind_slot(&mut p, Some(prefix_len + tail_len))?;
            let pre_u = Unit {
                index: 0,
                sequence: 0,
                rows: 0..prefix_len,
                layers: 0..cfg.total_visits(),
                slot: 0,
                barrier: false,
                segments: vec![UnitSegment {
                    sequence: 0,
                    rows: 0..prefix_len,
                    tokens: 0..prefix_len,
                    prompt_end: true,
                    snapshot: false,
                    head: false,
                }],
            };
            let mut ps = [p];
            execution.prefill(
                &mut ps,
                &[&tokens[..prefix_len]],
                &[pre_u],
                super::prefill::HeadProjection::AllHeadRows,
                &mut |_, _, _| Ok(()),
            )?;
            p = ps[0].clone();
            let mut collected = Vec::<u32>::new();
            let mut pos = prefix_len;
            while pos < prefix_len + tail_len {
                let step = w.min(prefix_len + tail_len - pos);
                p.len = pos;
                let dec_u = Unit {
                    index: 0,
                    sequence: 0,
                    rows: 0..step,
                    layers: 0..cfg.total_visits(),
                    slot: 0,
                    barrier: false,
                    segments: vec![UnitSegment {
                        sequence: 0,
                        rows: 0..step,
                        tokens: pos..pos + step,
                        prompt_end: false,
                        snapshot: false,
                        head: true,
                    }],
                };
                let mut ps = [p];
                let logits = execution.decode_eager(&mut ps, &[&tokens[pos..pos + step]], &dec_u)?;
                p = ps[0].clone();
                let tile = read_logits(&library, device_id, &logits)?;
                collected.extend(tile.into_iter().map(f32::to_bits));
                pos += step;
            }
            by_width.push((w, collected));
        }
        let base_bits = &by_width[0].1;
        for (w, bits) in &by_width[1..] {
            ensure!(
                bits == base_bits,
                "Inkling 2 Mini width invariance failed: width {w} differs from width {}",
                by_width[0].0
            );
        }

        let mut per_width = Vec::with_capacity(widths.len());
        let mut graph_exact = false;
        if args.graph_parity {
            // Replay every checked width eagerly vs through graph capture + replay (3 steps).
            for &w in &widths {
                let mut eager_bits = Vec::new();
                let mut graphed_bits = Vec::new();
                for (graphed, out) in [(false, &mut eager_bits), (true, &mut graphed_bits)] {
                    for rep in 0..3 {
                        let mut p = super::cache::Placement {
                            units: (pages_needed..2 * pages_needed).collect(),
                            slot: 1,
                            len: 0,
                            state_len: 0,
                            media: None,
                        };
                        prefix.bind_slot(&mut p, Some(prefix_len + w))?;
                        let pre_u = Unit {
                            index: 0,
                            sequence: 0,
                            rows: 0..prefix_len,
                            layers: 0..cfg.total_visits(),
                            slot: 0,
                            barrier: false,
                            segments: vec![UnitSegment {
                                sequence: 0,
                                rows: 0..prefix_len,
                                tokens: 0..prefix_len,
                                prompt_end: true,
                                snapshot: false,
                                head: false,
                            }],
                        };
                        let mut ps = [p];
                        execution.prefill(
                            &mut ps,
                            &[&tokens[..prefix_len]],
                            &[pre_u],
                            super::prefill::HeadProjection::AllHeadRows,
                            &mut |_, _, _| Ok(()),
                        )?;
                        p = ps[0].clone();
                        let dec_u = Unit {
                            index: 0,
                            sequence: 0,
                            rows: 0..w,
                            layers: 0..cfg.total_visits(),
                            slot: 0,
                            barrier: false,
                            segments: vec![UnitSegment {
                                sequence: 0,
                                rows: 0..w,
                                tokens: prefix_len..prefix_len + w,
                                prompt_end: false,
                                snapshot: false,
                                head: true,
                            }],
                        };
                        p.len = prefix_len;
                        let mut ps = [p];
                        let logits = if graphed {
                            execution.decode(&mut ps, &[&tokens[prefix_len..prefix_len + w]], &dec_u)?
                        } else {
                            execution.decode_eager(&mut ps, &[&tokens[prefix_len..prefix_len + w]], &dec_u)?
                        };
                        if rep == 2 {
                            *out = read_logits(&library, device_id, &logits)?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect();
                        }
                    }
                }
                ensure!(
                    eager_bits == graphed_bits,
                    "Inkling 2 Mini graph vs eager replay at width {w} is NOT byte-exact!"
                );
                per_width.push(serde_json::json!({
                    "width": w,
                    "byte_exact": true,
                    "graph_vs_eager_byte_exact": true,
                }));
            }
            graph_exact = true;
        } else {
            for &w in &widths {
                per_width.push(serde_json::json!({
                    "width": w,
                    "byte_exact": true,
                    "graph_vs_eager_byte_exact": serde_json::Value::Null,
                }));
            }
        }
        width_report = serde_json::json!({
            "widths_checked": widths,
            "byte_exact": true,
            "graph_vs_eager_byte_exact": graph_exact,
            "per_width": per_width,
        });
    }

    // Optional direct slot-independence disproof: verify that the same total `rows`
    // across different active slot counts (e.g., rows=40 at slots=15 and slots=16)
    // replays the EXACT same single captured graph executable with byte-exact parity vs eager.
    let mut slot_report = serde_json::Value::Null;
    if args.slot_parity_rows > 0 && !args.slot_parity_slots.is_empty() {
        let total_rows = args.slot_parity_rows;
        let max_slots = *args.slot_parity_slots.iter().max().unwrap();
        ensure!(
            total_rows <= args.decode_rows as usize && max_slots >= 1 && max_slots <= args.max_sequences,
            "Mini slot parity requires rows<=decode_rows and slots<=max_sequences"
        );
        ensure!(
            pages_needed * (max_slots as u32 + 1) <= pool_pages,
            "Mini slot parity pool too small for {max_slots} sequences"
        );
        let use_verify = execution.draft.is_some();
        let mode_name = if use_verify { "Verify" } else { "Decode" };
        let captures_before = execution.graphs.as_ref().map_or(0, |g| g.capture_count());
        let mut shared_exec: Option<usize> = None;
        let mut captures_after_first = 0usize;
        let mut per_slot = Vec::with_capacity(args.slot_parity_slots.len());

        for (idx, &slot_count) in args.slot_parity_slots.iter().enumerate() {
            ensure!(
                slot_count >= 1 && total_rows >= slot_count && total_rows <= slot_count * 8,
                "Mini slot parity cannot partition {total_rows} rows across {slot_count} slots (1..=8/slot)"
            );
            let base = total_rows / slot_count;
            let rem = total_rows % slot_count;
            let row_counts: Vec<usize> = (0..slot_count).map(|i| base + usize::from(i < rem)).collect();
            let prefix_lens: Vec<usize> = (0..slot_count)
                .map(|i| (16 + (i % 7) * 5).min(rows.saturating_sub(8)).max(1))
                .collect();

            let build_prefilled = |exec: &mut super::engine::Execution<'_>| -> Result<Vec<super::cache::Placement>> {
                let mut ps = Vec::with_capacity(slot_count);
                for i in 0..slot_count {
                    let plen = prefix_lens[i];
                    let page_base = (i as u32 + 1) * pages_needed;
                    let mut p = super::cache::Placement {
                        units: (page_base..page_base + pages_needed).collect(),
                        slot: i,
                        len: 0,
                        state_len: 0,
                        media: None,
                    };
                    prefix.bind_slot(&mut p, Some(plen + row_counts[i]))?;
                    let pre_u = Unit {
                        index: 0,
                        sequence: 0,
                        rows: 0..plen,
                        layers: 0..cfg.total_visits(),
                        slot: 0,
                        barrier: false,
                        segments: vec![UnitSegment {
                            sequence: 0,
                            rows: 0..plen,
                            tokens: 0..plen,
                            prompt_end: true,
                            snapshot: false,
                            head: false,
                        }],
                    };
                    let mut one = [p];
                    exec.prefill(
                        &mut one,
                        &[&tokens[..plen]],
                        &[pre_u],
                        super::prefill::HeadProjection::AllHeadRows,
                        &mut |_, _, _| Ok(()),
                    )?;
                    ps.push(one[0].clone());
                }
                Ok(ps)
            };

            let run_step = |exec: &mut super::engine::Execution<'_>, graphed: bool| -> Result<Vec<u32>> {
                let mut ps = build_prefilled(exec)?;
                let mut segments = Vec::with_capacity(slot_count);
                let mut tok_slices: Vec<&[u32]> = Vec::with_capacity(slot_count);
                let mut cursor = 0usize;
                for i in 0..slot_count {
                    let plen = prefix_lens[i];
                    let r = row_counts[i];
                    ps[i].len = plen;
                    segments.push(UnitSegment {
                        sequence: i,
                        rows: cursor..cursor + r,
                        tokens: plen..plen + r,
                        prompt_end: false,
                        snapshot: false,
                        head: true,
                    });
                    tok_slices.push(&tokens[plen..plen + r]);
                    cursor += r;
                }
                ensure!(cursor == total_rows);
                let dec_u = Unit {
                    index: 0,
                    sequence: 0,
                    rows: 0..total_rows,
                    layers: 0..cfg.total_visits(),
                    slot: 0,
                    barrier: false,
                    segments,
                };
                if use_verify {
                    let draft = exec.draft.as_ref().expect("drafter").clone();
                    exec.requests
                        .decode
                        .set_replay(Some((draft.verify.replay.buffer, exec.requests.cache.layout.conv.width)))?;
                }
                let res = if graphed {
                    exec.decode(&mut ps, &tok_slices, &dec_u)
                } else {
                    exec.decode_eager(&mut ps, &tok_slices, &dec_u)
                };
                if use_verify {
                    let _ = exec.requests.decode.set_replay(None);
                }
                let logits = res?;
                Ok(read_logits(&library, device_id, &logits)?.into_iter().map(f32::to_bits).collect())
            };

            let eager_bits = run_step(&mut execution, false)?;
            let reps = if idx == 0 { 3 } else { 1 };
            let mut graphed_bits = Vec::new();
            for rep in 0..reps {
                let bits = run_step(&mut execution, true)?;
                if rep + 1 == reps {
                    graphed_bits = bits;
                }
            }
            ensure!(
                eager_bits == graphed_bits,
                "Inkling 2 Mini slot parity failed at rows={total_rows} slots={slot_count} ({mode_name})"
            );
            let exec_addr = execution
                .graphs
                .as_ref()
                .and_then(|g| g.last_launched_exec())
                .ok_or_else(|| anyhow::anyhow!("Mini slot parity did not launch a captured graph at slots={slot_count}"))?;
            let captures_now = execution.graphs.as_ref().map_or(0, |g| g.capture_count());
            if idx == 0 {
                ensure!(
                    captures_now == captures_before + 1,
                    "Mini slot parity expected 1 capture at slots={slot_count}, got {}",
                    captures_now - captures_before
                );
                shared_exec = Some(exec_addr);
                captures_after_first = captures_now;
            } else {
                ensure!(
                    captures_now == captures_after_first,
                    "Mini slot parity at slots={slot_count} triggered a new capture ({captures_after_first} -> {captures_now})"
                );
                ensure!(
                    shared_exec == Some(exec_addr),
                    "Mini slot parity at slots={slot_count} launched a different cudaGraphExec_t ({exec_addr:#x} vs {:#x})",
                    shared_exec.unwrap_or(0)
                );
            }
            per_slot.push(serde_json::json!({
                "slots": slot_count,
                "row_partition": row_counts,
                "exec_addr": format!("{exec_addr:#x}"),
                "captures_total": captures_now,
                "graph_vs_eager_byte_exact": true,
            }));
        }
        let captures_after = execution.graphs.as_ref().map_or(0, |g| g.capture_count());
        slot_report = serde_json::json!({
            "rows": total_rows,
            "mode": mode_name,
            "slots_checked": args.slot_parity_slots,
            "captures_added": captures_after - captures_before,
            "shared_single_capture": captures_after - captures_before == 1,
            "per_slot": per_slot,
        });
    }

    let mut nll = 0f64;
    for (row, &next) in values.chunks_exact(vocab).zip(&tokens[1..]) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        nll += maximum
            + row.iter().map(|&v| (v as f64 - maximum).exp()).sum::<f64>().ln()
            - row[next as usize] as f64;
    }
    nll /= rows as f64;

    let comparison = if let Some(path) = &args.golden {
        let expected = std::fs::read(path)?;
        ensure!(
            expected.len() == values.len() * 4,
            "Mini reference logits shape differs: expected {} bytes, got {}",
            values.len() * 4,
            expected.len()
        );
        let ref_vals: Vec<f32> = expected
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let (mut sq, mut norm, mut dot, mut actual_norm, mut ref_nll, mut kl_sum) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for (&a, &e) in values.iter().zip(&ref_vals) {
            ensure!(e.is_finite(), "Mini reference nonfinite logits");
            sq += (a as f64 - e as f64).powi(2);
            norm += (e as f64).powi(2);
            dot += (a as f64) * (e as f64);
            actual_norm += (a as f64).powi(2);
        }
        for ((a_row, e_row), &next) in values
            .chunks_exact(vocab)
            .zip(ref_vals.chunks_exact(vocab))
            .zip(&tokens[1..])
        {
            let emax = e_row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let amax = a_row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let elz = emax + e_row.iter().map(|&v| (v as f64 - emax).exp()).sum::<f64>().ln();
            let alz = amax + a_row.iter().map(|&v| (v as f64 - amax).exp()).sum::<f64>().ln();
            ref_nll += elz - e_row[next as usize] as f64;
            for (&a, &e) in a_row.iter().zip(e_row) {
                let p = (e as f64 - elz).exp();
                let log_p = e as f64 - elz;
                let log_q = a as f64 - alz;
                kl_sum += p * (log_p - log_q);
            }
        }
        ref_nll /= rows as f64;
        let mean_kl = kl_sum / rows as f64;
        let rms = (sq / norm.max(1e-30)).sqrt();
        let cosine = dot / (norm * actual_norm).sqrt().max(1e-30);
        serde_json::json!({
            "oracle_commit": "f1940f40eb",
            "normalized_rms": rms,
            "cosine": cosine,
            "reference_nll": ref_nll,
            "nll_delta": nll - ref_nll,
            "mean_kl_nats": mean_kl,
        })
    } else {
        serde_json::Value::Null
    };

    std::fs::create_dir(&args.save_logits)?;
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(args.save_logits.join("logits.bin"), &raw_bytes)?;
    std::fs::write(
        args.save_logits.join("tokens.bin"),
        tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    let _ = placement;
    let report = serde_json::json!({
        "family": "inkling2_mini",
        "precision_oracle": "f1940f40eb",
        "kv_format": args.kv_format.name(),
        "visits": cfg.total_visits(),
        "rows": rows,
        "vocab": vocab,
        "mean_nll": nll,
        "reference_comparison": comparison,
        "invariance": width_report,
        "slot_invariance": slot_report,
    });
    std::fs::write(args.save_logits.join("result.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("Inkling 2 Mini golden mean NLL={nll:.8} rows={rows} oracle=f1940f40eb");
    Ok(())
}
