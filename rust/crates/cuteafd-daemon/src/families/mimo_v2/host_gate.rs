//! Ignored full-model gate for the actual serving prefix-cache setup.
use super::*;
use clap::Parser;
use cuteafd_engine::prefix::{After, MarkSlot, PrefixFamily, SnapshotKind};
use crate::shared::memory::device::Device;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    prefix: crate::shared::prefix::PrefixArgs,
    #[arg(long)]
    gate_tokens: PathBuf,
    #[arg(long, default_value_t = 512)]
    gate_context_tokens: usize,
    #[arg(long, value_delimiter = ',', default_value = "191,256,383,191")]
    gate_prefixes: Vec<usize>,
    #[arg(long, default_value_t = 8)]
    gate_decode_tokens: usize,
    #[arg(long, default_value_t = 128)]
    gate_chunk: usize,
}

fn mark(engine: &engine::MimoEngine<'_>, family: &prefix::MimoPrefix<'_, '_>, slot: MarkSlot)
    -> Result<Vec<Vec<u8>>> {
    family.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    let segments = family.mark_segments(slot);
    ensure!(segments.len() == engine.ranks(), "mark must cover both ranks");
    ensure!(segments.iter().map(|s| s.bytes).sum::<usize>() == family.mark_bytes(), "mark byte budget differs");
    segments.iter().enumerate().map(|(rank, segment)| {
        let template = engine.kv_layer_on(rank, 0).1;
        let device = Device { library: engine.library, id: template.device_id };
        let mut bytes = vec![0u8; segment.bytes];
        // SAFETY: `engine.stream_of(rank)` owns rank's work on `template.device_id`.
        device.run(|| unsafe { engine.library.copy_d2h(&mut bytes, cuteafd_ffi::CuteafdDeviceBuffer {
            ptr: segment.addr as *mut std::ffi::c_void, bytes: segment.bytes, ..template }, engine.stream_of(rank)) })?;
        Ok(bytes)
    }).collect()
}

fn decode(engine: &engine::MimoEngine<'_>, placement: &mut engine::MimoPlacement,
    mut logits: Vec<f32>, steps: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut tokens = Vec::new();
    let mut bits = Vec::new();
    for _ in 0..steps {
        bits.extend(logits.iter().map(|value| value.to_bits()));
        let token = cuteafd_engine::prefix::greedy(&logits);
        tokens.push(token);
        logits = engine.verify_device(&mut [(&mut *placement, 1)], &[token], None)?
            .context("full model decode produced no logits")?.row_host(engine.library, 0)?;
    }
    bits.extend(logits.iter().map(|value| value.to_bits()));
    Ok((tokens, bits))
}

fn run(args: &Args, engine: &engine::MimoEngine<'_>, tokens: &[u32]) -> Result<()> {
    ensure!(engine.ranks() == 2 && engine.weights.layers.len() == engine.cfg.layers,
        "rank host gate requires every model layer on two RTX GPUs");
    ensure!(args.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
        "rank host gate requires exact prefix reuse");
    ensure!(engine.rings >= 2 && args.gate_decode_tokens > 0, "gate needs two rings and greedy continuation");
    ensure!(args.gate_chunk > 0 && args.gate_chunk <= engine.prefill_rows, "invalid gate chunk");
    let n = args.gate_context_tokens;
    let capacity = n + args.gate_decode_tokens;
    ensure!(tokens.len() >= n && capacity <= engine.max_context, "gate tokens/context do not fit");
    let tokens = &tokens[..n];
    ensure!(tokens.iter().all(|&token| (token as usize) < engine.cfg.vocab_size), "out-of-vocabulary gate token");
    let (family, mut cache) = serve::prefix_cache(engine, &args.prefix, 1)?;
    let host = cache.stats().host.context("requested host retention was not enabled")?;
    if let crate::shared::prefix::HostBudget::Bytes(bytes) = args.prefix.host_cache_bytes {
        ensure!(host.quota_bytes == bytes, "explicit host quota changed under the head split");
    }
    let layout = family.layout();
    ensure!(family.page_segments(0).iter().map(|segment| segment.bytes).sum::<usize>() == layout.page_bytes,
        "host layout must describe both ranks in one aggregate page and mark");
    let devices = family.host_owners().iter().map(|owner| owner.buffer.device_id)
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(devices.len() == 2, "registered snapshot owners must cover both physical GPUs");
    let inspector = prefix::MimoPrefix::new(engine, |_| 2, false)?;
    let current = engine.library.cuda_get_device()?;
    for (round, &at) in args.gate_prefixes.iter().enumerate() {
        ensure!(at >= engine.cfg.window && at < n, "gate frontier must cover its mark and leave a suffix");
        let make = |ring, pages| engine::MimoPlacement { ring, pages, len: 0 };
        let mut cold = cache.admit(&family, &[], capacity, false, |pages| make(0, pages))?.placement;
        let prefix = prefill_digest(engine, &mut cold, &tokens[..at], args.gate_chunk, true)?;
        let before_kv = kv_rows(engine, &cold, 0, at)?;
        inspector.capture(MarkSlot(0), &cold, at).map_err(|e| anyhow::anyhow!("{e}"))?;
        let before_mark = mark(engine, &inspector, MarkSlot(0))?;
        let kind = if round % 2 == 0 { SnapshotKind::Prompt } else { SnapshotKind::Turn };
        ensure!(cache.capture(&family, kind, &tokens[..at], &cold, After::from_logits(&prefix.last, true))?,
            "prefix snapshot was skipped");
        let straight = prefill_digest(engine, &mut cold, &tokens[at..], args.gate_chunk, true)?;
        let straight_kv = kv_rows(engine, &cold, 0, n)?;
        let straight_decode = decode(engine, &mut cold, straight.last.clone(), args.gate_decode_tokens)?;
        let straight_final = kv_rows(engine, &cold, 0, capacity)?;
        cache.release(&family, &cold.pages)?;
        cache.clear(&family)?; // Force a real host promotion; no device snapshot survives.
        let before = cache.stats();
        ensure!(before.entries_prompt == 0 && before.entries_turn == 0 && before.pages_free == before.pages,
            "device snapshot or request still owns pages");
        ensure!(before.host.as_ref().is_some_and(|host| host.resident_snapshots > 0), "no completed host snapshot");
        // Destroy every possible device source of the saved state before restore.
        // This includes all mark slots, the entire full-KV pool, and live SWA rings.
        for owner in family.host_owners() {
            owner.device.run(|| engine.library.cuda_zero_bytes(owner.buffer, owner.buffer.bytes))?;
        }
        for rank in 0..engine.ranks() {
            for layer in 0..engine.weights.layers.len() {
                let (attention, buffer, _) = engine.kv_layer_on(rank, layer);
                if attention == cuteafd_loader::families::mimo_v2::MimoAttention::Sliding {
                    Device { library: engine.library, id: buffer.device_id }
                        .run(|| engine.library.cuda_zero_bytes(buffer, buffer.bytes))?;
                }
            }
        }
        if let Some(mtp) = &engine.mtp {
            let buffer = mtp.hidden.buffer;
            Device { library: engine.library, id: buffer.device_id }
                .run(|| engine.library.cuda_zero_bytes(buffer, buffer.bytes))?;
        }
        let admitted = cache.admit(&family, &tokens[..at], capacity, true, |pages| make(1, pages))?;
        ensure!(admitted.resume == at && admitted.source.is_some_and(|source| source.host && !source.partial),
            "expected exact host promotion at {at}, got resume {}", admitted.resume);
        let after = admitted.after.context("exact host hit lost its next-token payload")?;
        ensure!(after.logits.as_ref().is_some_and(|logits| logits.iter().map(|v| v.to_bits())
            .eq(prefix.last.iter().map(|v| v.to_bits()))), "retained next-token logits changed");
        let mut restored = admitted.placement;
        family.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(kv_rows(engine, &restored, 0, at)? == before_kv, "rank KV differs immediately after host restore");
        inspector.capture(MarkSlot(1), &restored, at).map_err(|e| anyhow::anyhow!("{e}"))?;
        let restored_mark = mark(engine, &inspector, MarkSlot(1))?;
        for rank in 0..engine.ranks() {
            ensure!(before_mark[rank] == restored_mark[rank], "rank {rank} positional mark differs after host restore");
        }
        let suffix = prefill_digest(engine, &mut restored, &tokens[at..], args.gate_chunk, true)?;
        ensure!(straight.layers == suffix.layers && straight.logits == suffix.logits
            && straight.argmax == suffix.argmax, "restored suffix layer/logit bits differ");
        ensure!(kv_rows(engine, &restored, 0, n)? == straight_kv, "rank KV differs after suffix continuation");
        let continued = decode(engine, &mut restored, suffix.last, args.gate_decode_tokens)?;
        ensure!(straight_decode == continued, "restored greedy tokens or per-step logits differ");
        ensure!(kv_rows(engine, &restored, 0, capacity)? == straight_final, "rank KV differs after greedy continuation");
        ensure!(engine.library.cuda_get_device()? == current, "host restore changed the caller's CUDA device");
        let after = cache.stats();
        ensure!(after.host_promotions == before.host_promotions + 1, "host promotion was not recorded");
        ensure!(after.restore_failures == 0 && after.host_store_skips == 0,
            "host gate hid a failed restore or skipped store");
        println!("{}", serde_json::json!({"gate":"mimo-rank-host", "round":round, "frontier":at,
            "kind":format!("{kind:?}"), "ranks":engine.ranks(), "layers":engine.cfg.layers,
            "page_bytes":layout.page_bytes, "mark_bytes":layout.mark_bytes,
            "mark_rank_bytes":before_mark.iter().map(Vec::len).collect::<Vec<_>>(),
            "prefix_bytes":before_kv.len(), "suffix_layers":suffix.layers, "suffix_logits":suffix.logits,
            "greedy_tokens":continued.0, "host":after.host, "exact":true}));
        cache.release(&family, &restored.pages)?;
        cache.clear(&family)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a full MiMo checkpoint, two RTX GPUs and live Spark experts; CUTEAFD_MIMO_HOST_GATE_ARGS is a JSON argv list"]
fn dual_rank_host_restore_matches_cold_continuation() -> Result<()> {
    let argv: Vec<String> = serde_json::from_str(&std::env::var("CUTEAFD_MIMO_HOST_GATE_ARGS")?)?;
    let args = Args::try_parse_from(std::iter::once("mimo-host-gate".to_string()).chain(argv))?;
    ensure!(!args.engine.skip_experts && args.engine.peers.is_some(), "full-model gate needs live Spark experts");
    ensure!(args.engine.split_device.is_some(), "full-model gate needs a head split");
    ensure!(!args.gate_prefixes.is_empty(), "gate needs at least one frontier");
    let bytes = std::fs::read(&args.gate_tokens)?;
    ensure!(bytes.len() % 4 == 0, "token file must contain complete u32 rows");
    let tokens = bytes.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>();
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine| run(&args, engine, &tokens))
}

#[test]
#[ignore = "requires MiMo Pro and RTX kernels; CUTEAFD_MIMO_RESUME_GATE_ARGS is a JSON golden argv list"]
fn native_mtp_resume_all_page_and_ring_frontiers() -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    #[derive(Parser)]
    struct ResumeArgs {
        #[command(flatten)]
        golden: GoldenArgs,
    }
    fn sha256(path: &std::path::Path) -> Result<String> {
        let mut file = std::fs::File::open(path)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 { break; }
            hash.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hash.finalize()))
    }

    let argv: Vec<String> = serde_json::from_str(&std::env::var("CUTEAFD_MIMO_RESUME_GATE_ARGS")?)?;
    let args = ResumeArgs::try_parse_from(std::iter::once("mimo-resume-gate".to_string()).chain(argv))?.golden;
    ensure!(args.engine.skip_experts && args.engine.mtp == 3 && args.engine.draft.is_none(),
        "state-only gate requires skipped routed experts and all three native MTP stages");
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(args.golden.join("provenance.json"))?)?;
    ensure!(provenance["set_sha256"] == "4a16d05350aa7df48011ac49501f772812fa88bb0166b7e2103ac6ecaa36d4fa"
        && provenance["window_id"] == "legacy", "unexpected resume token source");
    let token_path = args.golden.join("tokens.bin");
    let tokens_sha = sha256(&token_path)?;
    ensure!(tokens_sha == "e377648dedddff2e9990f100c147c2e8ebecdb2c2683929305f681ae438a8ae2"
        && provenance["tokens_sha256"] == tokens_sha, "resume token bytes changed");
    let bytes = std::fs::read(&token_path)?;
    ensure!(bytes.len() == 513 * 4, "resume fixture must contain 513 complete token rows");
    let tokens = bytes.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>();
    println!("{}", serde_json::json!({"gate":"mimo-native-mtp-resume", "state_only_skip_experts":true,
        "set_sha256":provenance["set_sha256"], "window_id":provenance["window_id"],
        "tokens_sha256":tokens_sha, "binary_sha256":sha256(&std::env::current_exe()?)?,
        "native_lib_sha256":sha256(&args.engine.native_lib)?, "swa_format":format!("{:?}", args.engine.swa_kv_cache)}));
    let opened = open(&args.engine)?;
    ensure!(tokens.iter().all(|&token| (token as usize) < opened.cfg.vocab_size), "out-of-vocabulary resume token");
    opened.with_engine(&args.engine, |engine| {
        for at in [63, 64, 65, 255, 256, 257] {
            resume_check(&args, engine, &tokens, at)?;
        }
        Ok(())
    })
}
