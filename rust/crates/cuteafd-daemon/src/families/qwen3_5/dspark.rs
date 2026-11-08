//! Dense-target binding of the shared block drafter and online draft policy.
use super::{engine::{DenseEngine, DrafterBytes}, GoldenArgs};
use crate::shared::speculation::dflash::{ContextRow, DraftSeq, TargetHead, TAP_ROWS};
use crate::shared::{draft::{binding, clock::RoundTimes, evidence::{Censor, DraftSource, Evidence}, routes::RoundRoutes},
    fp8_linear::{Fp8Rows, Fp8Scales}, speculation::{block_dspark::{self, DsparkConfig, DsparkDrafter},
        target::{verify_commit, AcceptedPrefix, VerifyRequest, VerifyRow}}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{verify_dspark_greedy_vocab, DraftCandidate, DraftPolicy, LayerResource, PolicyGeometry,
    PolicySeed, ResourceClass};
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{families::{glm5::draft_representation::GlmDraftRepresentation,
    qwen3_5::Qwen35Config}, plan::checkpoint::Checkpoint};
use std::time::{Duration, Instant};

pub(super) fn load_drafter<'a>(engine: &DenseEngine<'a>, programs: &'a Programs<'a>, snapshot: &std::path::Path,
    lanes: usize, mask: Vec<u8>) -> Result<DsparkDrafter<'a>> {
    let device = engine.target_stream().device;
    // The drafter is an independently admitted scope with its own per-device arena.
    let mut requests = DsparkDrafter::arena_requests(snapshot, GlmDraftRepresentation::Bf16Only, device.id)?;
    let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("qwen3_5/dspark/gpu{}", device.id),
        requests.remove(&device.id).context("block dSpark arena lacks the target device")?)?;
    let arena = super::weights::ScopeArena::new(device, layout,
        cuteafd_core::memory_layout::ArenaTag::drafter_weights("qwen35/dspark"))?;
    tracing::info!(arena = arena.layout().id(), bytes = arena.layout().bytes(),
        views = arena.layout().views().len(), padding = arena.layout().padding_bytes(),
        "dense Qwen dSpark weight arena allocated");
    let drafter = crate::shared::experts::rtx::push::load_alone(snapshot, &[device], || {
        let tensors = if crate::shared::experts::rtx::dense::is_recording() {
            Default::default()
        } else {
            block_dspark::prefetch(snapshot).join().map_err(|_| anyhow::anyhow!("draft loader panicked"))??
        };
        DsparkDrafter::load_paged_with(device.library, programs, crate::shared::speculation::dflash::TargetProjection::Bf16, snapshot, tensors,
            engine.target_stream().raw, lanes, lanes, mask, GlmDraftRepresentation::Bf16Only,
            Fp8Scales::Amax, Fp8Rows::W8a16, engine.cfg.max_context, engine.pool_tokens,
            cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, |id, name, bytes| {
                ensure!(id == device.id, "dense dSpark weight {name} requested on GPU {id}, not {}", device.id);
                arena.allocate(name, bytes)
            })
    })?;
    arena.complete()?;
    Ok(drafter)
}

/// Retained dSpark arena receipts (weights only; rings and scratch are separate owners).
pub(super) fn weight_receipts(drafter: &DsparkDrafter<'_>,
    devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>)
    -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
    use cuteafd_loader::placement::planner::{Lifetime, Phase};
    let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "qwen3_5.dspark",
        Lifetime { first: Phase::DraftLoad, last: Phase::Growth },
        cuteafd_core::memory_layout::Category::Drafter, "dSpark weights");
    drafter.allocation_receipts(&mut inventory, "weights")?;
    Ok(inventory.receipts)
}

fn policy(layers: usize, block: usize) -> Result<DraftPolicy> {
    Ok(DraftPolicy::seeded(PolicyGeometry {
        layers: vec![LayerResource { class: None, slice_bytes: 0., group_rows: 1, timed: false }; layers],
        group_rows_by_rows: vec![], experts: 0, topk: 0, max_requests: 1, max_positions: block,
        widths: vec![block], classes: vec![ResourceClass::rtx()], regimes: 1,
    }, PolicySeed::weak(block))?)
}

fn argmax(row: &[f32]) -> Result<u32> {
    ensure!(!row.is_empty() && row.iter().all(|value| value.is_finite()), "nonfinite dense posterior");
    Ok(row.iter().enumerate().max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .context("empty dense posterior")?.0 as u32)
}

fn top_margin(row: &[f32], best: u32) -> (f32, Vec<u32>) {
    let top = row[best as usize];
    let second = row.iter().enumerate().filter(|&(index, _)| index != best as usize)
        .map(|(_, &value)| value).fold(f32::NEG_INFINITY, f32::max);
    (top - second, row.iter().enumerate().filter(|(_, value)| **value == top)
        .map(|(index, _)| index as u32).collect())
}

fn micros(duration: Duration) -> u64 { duration.as_micros().min(u128::from(u64::MAX)) as u64 }

fn tapped_rows(library: &NativeLibrary, drafter: &DsparkDrafter<'_>, rows: usize) -> Result<Vec<u8>> {
    let bytes = rows * drafter.cfg.taps.len() * drafter.cfg.hidden * 2;
    ensure!(rows <= TAP_ROWS, "dense draft tap read exceeds capacity");
    let mut host = vec![0; bytes];
    library.copy_d2h(&mut host, CuteafdDeviceBuffer { bytes, ..drafter.taps.buffer })?;
    Ok(host)
}

/// Exercise the live cache hooks against cold replay of identical canonical taps
/// and producer row shapes; cuBLAS projection rounding depends on the row count.
fn check_draft_prefix(library: &NativeLibrary, drafter: &DsparkDrafter<'_>, taps: &[u8],
    update_rows: &[usize], label: &str)
    -> Result<()> {
    let pitch = drafter.cfg.taps.len() * drafter.cfg.hidden * 2;
    ensure!(!taps.is_empty() && taps.len() % pitch == 0, "dense draft prefix tap shape");
    let frontier = taps.len() / pitch;
    ensure!(update_rows.iter().all(|&rows| rows > 0 && rows <= TAP_ROWS)
        && update_rows.iter().try_fold(0usize, |sum, &rows| sum.checked_add(rows)) == Some(frontier),
        "dense draft prefix update groups differ from retained taps");
    let saved = drafter.capture_pages(0, frontier, frontier)?;
    let read = |attachment: &cuteafd_engine::prefix::DraftPageAttachment| -> Result<Vec<Vec<u8>>> {
        let mut bytes = Vec::new();
        for (index, &page) in attachment.pages.iter().enumerate() {
            let rows = (attachment.frontier - attachment.origin - index * 64).min(64);
            for segment in drafter.page_segments(page)? {
                let len = segment.bytes / 64 * rows;
                let mut host = vec![0; len];
                library.copy_d2h(&mut host, CuteafdDeviceBuffer { ptr: segment.addr as *mut _,
                    bytes: len, device_id: library.cuda_get_device()?, ..CuteafdDeviceBuffer::default() })?;
                bytes.push(host);
            }
        }
        Ok(bytes)
    };
    let expected = read(&saved)?;
    drafter.reset_context(0, 0)?;
    let mut position = 0;
    for &count in update_rows {
        drafter.put_taps(&taps[position * pitch..(position + count) * pitch])?;
        let rows: Vec<_> = (0..count).map(|tap_row| ContextRow {
            tap_row, slot: 0, position: position + tap_row }).collect();
        drafter.update(&rows)?;
        position += count;
    }
    let cold = drafter.capture_pages(0, frontier, frontier)?;
    ensure!(read(&cold)? == expected, "dense {label} drafter same-shape cold bytes differ");
    drafter.release_pages(&cold)?;
    let continuation = &taps[..3.min(frontier) * pitch];
    let append = || -> Result<cuteafd_engine::prefix::DraftPageAttachment> {
        drafter.put_taps(continuation)?;
        let rows: Vec<_> = (0..continuation.len() / pitch).map(|tap_row| ContextRow {
            tap_row, slot: 0, position: frontier + tap_row }).collect();
        drafter.update(&rows)?;
        drafter.capture_pages(0, frontier + rows.len(), frontier + rows.len())
    };
    let grown = append()?;
    let expected_grown = read(&grown)?;
    drafter.release_pages(&grown)?;
    drafter.restore_pages(0, &saved, frontier)?;
    let restored = drafter.capture_pages(0, frontier, frontier)?;
    ensure!(read(&restored)? == expected, "dense {label} restored draft bytes differ");
    drafter.release_pages(&restored)?;
    let grown = append()?;
    ensure!(read(&grown)? == expected_grown && read(&saved)? == expected,
        "dense {label} draft continuation changed bytes or overwrote snapshot tail");
    drafter.release_pages(&grown)?;
    drafter.restore_pages(0, &saved, frontier)?;
    drafter.release_pages(&saved)?;
    println!("dense {label} drafter prefix: {frontier} rows, same-shape cold replay, restore and continuation byte-identical");
    Ok(())
}

/// Opt-in fixed-length numerical check, not a serving benchmark. Both arms
/// deliberately continue through EOS so they compare exactly `count` tokens.
#[allow(clippy::too_many_arguments)]
pub(super) fn spec_decode<'a>(args: &GoldenArgs, library: &'a NativeLibrary, programs: &'a Programs<'a>,
    checkpoint: &'a Checkpoint, cfg: &Qwen35Config, prompt: &[u32], count: usize) -> Result<()> {
    ensure!(count > 0 && !prompt.is_empty(), "spec decode needs a prompt and positive token count");
    let snapshot = args.draft.as_ref().context("spec decode requires --draft")?;
    block_dspark::check_checkpoint(snapshot, cfg.hidden, cfg.vocab, cfg.attention.len())?;
    let draft_cfg = DsparkConfig::read(snapshot)?;
    ensure!(draft_cfg.window.is_none(), "dense golden expects checkpoint-native full-context drafting");
    let required = prompt.len().checked_add(count).and_then(|n| n.checked_add(draft_cfg.block))
        .context("dense draft context size overflow")?;
    ensure!(required <= cfg.max_context, "spec decode exceeds target checkpoint context");
    let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    let demand = DrafterBytes::block_dspark(snapshot, &config, 1, library.sm_count()? as u64, cfg.max_context)?;
    let mut engine = DenseEngine::with_prefix(library, programs, checkpoint, cfg.clone(), args.pool_tokens,
        Some(demand.clone()), 1, None, args.kv_cache, false, None, args.verify_check)?;
    ensure!(required <= engine.pool_tokens, "spec decode exceeds admitted target pool");
    if args.verify_check { engine.verify_check(prompt)?; }
    let plain = (|| -> Result<(Vec<u32>, Duration, Vec<(f32, Vec<u32>)>)> {
        let mut last = vec![];
        for chunk in prompt.chunks(args.prefill_chunk) { last = engine.evaluate(chunk, |_, _| Ok(()))?; }
        let mut next = argmax(&last[last.len() - cfg.vocab..])?;
        let mut output = vec![next];
        let mut margins = vec![top_margin(&last[last.len() - cfg.vocab..], next)];
        let start = Instant::now();
        while output.len() < count {
            let logits = engine.evaluate(&[next], |_, _| Ok(()))?;
            next = argmax(&logits)?;
            margins.push(top_margin(&logits, next));
            output.push(next);
        }
        Ok((output, start.elapsed(), margins))
    })()?;
    // Normal evaluation invalidates target marks; use a fresh executor rather
    // than reconstructing the recurrent state or reusing a stale mark.
    drop(engine);
    let mut engine = DenseEngine::with_prefix(library, programs, checkpoint, cfg.clone(), args.pool_tokens,
        Some(demand), 1, None, args.kv_cache, false, None, args.verify_check)?;
    let stream = engine.target_stream().raw;
    let mask = engine.token_embedding().host_rows(&[draft_cfg.mask_token])?;
    let drafter = load_drafter(&engine, programs, snapshot, 1, mask)?;
    drafter.prepare_workspace()?;
    let mut last = vec![];
    let mut position = 0;
    let mut canonical_taps = Vec::new();
    let mut update_rows = Vec::new();
    for chunk in prompt.chunks(args.prefill_chunk.min(TAP_ROWS)) {
        last = engine.evaluate(chunk, |layer, hidden| {
            if drafter.tap_index(layer).is_some() {
                // SAFETY: evaluate retains its live post-block buffer through
                // this callback; capture reads on the same producer stream.
                drafter.tap_streams_at(layer, hidden.ptr, 1, 0, chunk.len(), 0)?;
            }
            Ok(())
        })?;
        let rows: Vec<_> = (0..chunk.len()).map(|tap_row| ContextRow { tap_row, slot: 0,
            position: position + tap_row }).collect();
        drafter.update(&rows)?;
        engine.target_stream().drain()?;
        if args.prefix_check {
            canonical_taps.extend(tapped_rows(library, &drafter, chunk.len())?);
            update_rows.push(chunk.len());
        }
        position += chunk.len();
    }
    if args.prefix_check { check_draft_prefix(library, &drafter, &canonical_taps, &update_rows, "prompt")?; }
    let mut next = argmax(&last[last.len() - cfg.vocab..])?;
    let mut output = vec![next];
    let mut policy = policy(cfg.attention.len(), draft_cfg.block)?;
    let mut routes = RoundRoutes::new(cfg.attention.len(), 0);
    let (mut cycles, mut proposed, mut accepted) = (0usize, 0usize, 0usize);
    let (mut draft_time, mut verify_time) = (Duration::ZERO, Duration::ZERO);
    let started = Instant::now();
    while output.len() < count {
        let round_start = Instant::now();
        let position = engine.position;
        let draft_start = Instant::now();
        let embedding = engine.token_embedding();
        let device_id = engine.target_stream().device.id;
        let head = TargetHead::Launch(Box::new(|hidden, logits, rows, actual_stream| {
            ensure!(actual_stream == stream, "dense draft head stream differs from target");
            engine.target_head(CuteafdDeviceBuffer { ptr: hidden.cast_mut(), bytes: rows * cfg.hidden * 2,
                device_id, ..CuteafdDeviceBuffer::default() }, CuteafdDeviceBuffer { ptr: logits.cast(),
                bytes: rows * cfg.vocab * 4, device_id, ..CuteafdDeviceBuffer::default() }, rows)?;
            Ok(())
        }));
        let draft = drafter.draft_device(&[DraftSeq { slot: 0, anchor: next, position, valid_from: 0 }],
            embedding, head)?.remove(0);
        let draft_elapsed = draft_start.elapsed();
        draft_time += draft_elapsed;
        let confidence: Vec<_> = draft.confidence.iter().map(|&value| f64::from(value)).collect();
        let candidates = [DraftCandidate { id: 0, confidence: &confidence }];
        let room = count - output.len();
        let selection = policy.select_within(false, &candidates, draft_cfg.block, (room + 1).min(64))?;
        let want = selection.as_ref().map_or(policy.cold_drafts(), |s| s.lengths[0])
            .min(draft.tokens.len()).min(room.saturating_sub(1));
        let inputs: Vec<_> = std::iter::once(next).chain(draft.tokens[..want].iter().copied()).collect();
        let rows: Vec<_> = inputs.iter().enumerate().map(|(index, &token)| VerifyRow {
            slot: 0, position: position + index, token: Some(token) }).collect();
        let verify_start = Instant::now();
        let result = verify_commit(&mut engine, &[0], VerifyRequest { rows: &rows, tap_layers: &draft_cfg.taps }, |verified| {
            let logits = verified.logits.to_host(library)?;
            let target = logits.chunks_exact(cfg.vocab).map(argmax).collect::<Result<Vec<_>>>()?;
            let result = verify_dspark_greedy_vocab(&inputs, &target, None, room, cfg.vocab)
                .map_err(anyhow::Error::msg)?;
            // SAFETY: borrowed taps stay valid through this consumer and the
            // producer stream orders their copies before target commit/reuse.
            unsafe { drafter.capture_target_taps(verified.taps, 0, rows.len(), 0)?; }
            let retained = result.accepted_inputs as usize;
            Ok((result, vec![AcceptedPrefix { slot: 0, rows: retained }]))
        })?;
        let retained = result.accepted_inputs as usize;
        let context: Vec<_> = (0..retained).map(|tap_row| ContextRow { tap_row, slot: 0,
            position: position + tap_row }).collect();
        drafter.update(&context)?;
        engine.target_stream().drain()?;
        if args.prefix_check && cycles == 0 {
            canonical_taps.extend(tapped_rows(library, &drafter, retained)?);
            update_rows.push(retained);
            check_draft_prefix(library, &drafter, &canonical_taps, &update_rows, "turn")?;
        }
        verify_time += verify_start.elapsed();
        proposed += want;
        accepted += retained - 1;
        cycles += 1;
        next = *result.emitted.last().context("verify emitted no correction/bonus")?;
        output.extend(&result.emitted);
        routes.begin(inputs.len());
        for layer in 0..cfg.attention.len() { routes.dense(layer)?; }
        let requests = [binding::RoundRequest { id: 0, rows: inputs.len(), accepted: retained,
            source: DraftSource::Neural, evidence: Some(Evidence::Head(&confidence)),
            censor: result.length_limit.then_some(Censor::OutputLimit) }];
        binding::observe(&mut policy, &binding::RoundRecord { shared: false, requests: &requests,
            routes: &routes, layer_us: &vec![None; cfg.attention.len()],
            times: RoundTimes { total_us: micros(round_start.elapsed()), draft_us: Some(micros(draft_elapsed)) },
            draft: Some(binding::DraftPass { width: draft_cfg.block, requests: Some(1) }),
            predicted: selection.as_ref().map(|selection| selection.predicted_us) })?;
    }
    let elapsed = started.elapsed();
    let same = plain.0.iter().zip(&output).take_while(|(a, b)| a == b).count();
    println!("dense spec decode: {same}/{count} greedy tokens identical; accepted {accepted}/{proposed} drafts, {cycles} cycles");
    println!("plain {:.2} tok/s; dSpark {:.2} tok/s; draft {:.3} ms/cycle, verify+commit {:.3} ms/cycle (diagnostic host posterior)",
        count.saturating_sub(1) as f64 / plain.1.as_secs_f64(), count.saturating_sub(1) as f64 / elapsed.as_secs_f64(),
        1000. * draft_time.as_secs_f64() / cycles.max(1) as f64,
        1000. * verify_time.as_secs_f64() / cycles.max(1) as f64);
    ensure!(output.len() == count, "dense speculative output length differs");
    if same < count {
        let (margin, tied) = &plain.2[same];
        println!("first mismatch token {same}: plain {}, dSpark {}, target top-2 margin {margin:.8}, tied top IDs {tied:?}",
            plain.0[same], output[same]);
        let tie_broken = *margin == 0. && tied.contains(&output[same]);
        ensure!(tie_broken, "non-tie dense speculative mismatch at generated token {same}");
        println!("dense speculative mismatch is tie-broken at the common-prefix target posterior; suffix equality is not claimed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires sealed dense Qwen target/drafter and CUTEAFD_PUSH_DUMP via golden-run --native-test"]
    fn native_push_dumps_each_requested_batch() -> Result<()> {
        use crate::shared::{golden_scores::push_regression as regression,
            experts::rtx::push::PushPolicy, speculation::target::BlockVerifyTarget};
        use std::path::PathBuf;
        let _ = tracing_subscriber::fmt().with_ansi(false)
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_test_writer().try_init();
        let path = |key| std::env::var_os(key).map(PathBuf::from).with_context(|| format!("push regression needs {key}"));
        let snapshot = path("CUTEAFD_GOLDEN_SNAPSHOT")?;
        let draft = path("CUTEAFD_DRAFT_SNAPSHOT")?;
        let output = path("CUTEAFD_PUSH_DUMP")?;
        ensure!(PushPolicy::for_snapshot(&snapshot)? == PushPolicy::Required
            && PushPolicy::for_snapshot(&draft)? == PushPolicy::Required,
            "native push regression requires target and drafter push namespaces");
        let checkpoint = Checkpoint::open(&snapshot)?;
        let cfg = cuteafd_loader::families::qwen3_5::Qwen35Checkpoint::read(&snapshot)?.config;
        block_dspark::check_checkpoint(&draft, cfg.hidden, cfg.vocab, cfg.attention.len())?;
        let draft_cfg = DsparkConfig::read(&draft)?;
        let native = path("CUTEAFD_NATIVE_LIB")?;
        let manifest = path("CUTEAFD_PROGRAM_MANIFEST")?;
        // SAFETY: golden-run supplies the sealed library; all GPU owners drop first.
        let library = unsafe { NativeLibrary::load(&native)? };
        library.cuda_set_device(0)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let config = serde_json::from_slice(&std::fs::read(draft.join("config.json"))?)?;
        let pool_tokens = cfg.max_context.max(16384).next_multiple_of(64);
        let demand = DrafterBytes::block_dspark(&draft, &config, 2, library.sm_count()? as u64, cfg.max_context)?;
        let mut engine = DenseEngine::with_prefix(&library, &programs, &checkpoint, cfg.clone(), Some(pool_tokens as u64),
            Some(demand), 2, None, cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, false, None, false)?;
        let mask = engine.token_embedding().host_rows(&[draft_cfg.mask_token])?;
        let drafter = load_drafter(&engine, &programs, &draft, 2, mask)?;
        drafter.prepare_workspace()?;
        for sequences in [1, 2] {
            let prompts = regression::prompts(sequences);
            let mut next_page = 0u32;
            for (slot, prompt) in prompts.iter().enumerate() {
                engine.reset_slot(slot)?;
                let end = prompt.len() + draft_cfg.block + 16;
                let pages = (next_page..next_page + end.div_ceil(64) as u32).collect::<Vec<_>>();
                next_page += pages.len() as u32;
                engine.bind_slot(slot, &pages)?;
                drafter.reset_context(slot, 0)?;
                drafter.admit_pages(slot, end)?;
                for (index, chunk) in prompt.chunks(1024).enumerate() {
                    engine.prefill_tapped(slot, chunk, 0, |layer, hidden| {
                        if drafter.tap_index(layer).is_some() {
                            // SAFETY: the target owns the post-block rows on the shared stream.
                            drafter.tap_streams_at(layer, hidden.ptr, 1, 0, chunk.len(), 0)?;
                        }
                        Ok(())
                    })?;
                    drafter.update(&(0..chunk.len()).map(|tap_row| ContextRow {
                        tap_row, slot, position: index * 1024 + tap_row }).collect::<Vec<_>>())?;
                }
            }
            regression::draft_dump(&output, sequences, draft_cfg.block, draft_cfg.block, cfg.vocab, || {
                let stream = engine.target_stream();
                let head = TargetHead::Launch(Box::new(|hidden, logits, rows, actual_stream| {
                    ensure!(actual_stream == stream.raw, "draft head stream mismatch");
                    engine.target_head(CuteafdDeviceBuffer { ptr: hidden.cast_mut(), bytes: rows * cfg.hidden * 2,
                        device_id: 0, ..Default::default() }, CuteafdDeviceBuffer { ptr: logits.cast(),
                        bytes: rows * cfg.vocab * 4, device_id: 0, ..Default::default() }, rows)?;
                    Ok(())
                }));
                let requests = prompts.iter().enumerate().map(|(slot, prompt)| DraftSeq {
                    slot, anchor: 42 + slot as u32, position: prompt.len(), valid_from: 0 }).collect::<Vec<_>>();
                let proposals = drafter.draft_device(&requests, engine.token_embedding(), head)?;
                Ok((proposals.into_iter().map(|draft| draft.tokens).collect(), drafter.last_logits(sequences)?))
            })?;
            let slots = (0..sequences).collect::<Vec<_>>();
            let mark = engine.checkpoint(&slots)?;
            let teacher = regression::teacher_rows();
            let rows = prompts.iter().enumerate().flat_map(|(slot, prompt)| teacher.iter().enumerate()
                .map(move |(offset, &token)| VerifyRow { slot, position: prompt.len() + offset, token: Some(token) }))
                .collect::<Vec<_>>();
            let values = engine.verify(&mark, VerifyRequest { rows: &rows, tap_layers: &[] })?.logits.to_host(&library)?;
            regression::target_dump(&output, sequences, cfg.vocab, &values)?;
            engine.restore(&mark)?;
            for slot in slots { drafter.discard_context(slot)?; engine.reset_slot(slot)?; }
        }
        Ok(())
    }

    #[test]
    fn dense_draft_binding_uses_empty_route_geometry_and_checkpoint_vocabulary() {
        let policy = policy(64, 7).unwrap();
        assert_eq!(policy.widths(), &[7]);
        assert_eq!(argmax(&[2., 2., 1.]).unwrap(), 0);
        assert_eq!(top_margin(&[2., 2., 1.], 0), (0., vec![0, 1]));
        assert_eq!(top_margin(&[3., 2., 1.], 0), (1., vec![0]));
        assert!(argmax(&[f32::NAN]).is_err());
        assert!(argmax(&[]).is_err());
    }
}
