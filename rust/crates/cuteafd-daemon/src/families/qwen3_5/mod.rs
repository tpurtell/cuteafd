//! Dense Qwen qualification entry point. Execution remains fail-closed until the executor qualifies.
mod programs;
mod weights;
mod block;
pub(crate) mod engine;
use anyhow::{ensure, Context, Result};
use std::path::PathBuf;
use cuteafd_loader::families::qwen3_5::Qwen35Checkpoint;

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct GoldenArgs {
    #[arg(long)]
    pub snapshot: PathBuf,
    /// Hugging Face identity for a snapshot outside the standard hub layout.
    #[arg(long)]
    pub model_id: Option<String>,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    /// Independent reference dump or sealed fidelity golden/ directory.
    #[arg(long)]
    pub golden: Option<PathBuf>,
    /// Window in a sealed fidelity golden; ignored for a legacy full-row dump.
    #[arg(long, default_value = "legacy")]
    pub window: String,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub save_logits: Option<PathBuf>,
    /// Validate checkpoint headers only (CPU); does not run a golden comparison.
    #[arg(long)]
    pub check_headers: bool,
    /// Strict explicit KV pool, otherwise the shared pool-first policy applies.
    #[arg(long)]
    pub pool_tokens: Option<u64>,
    /// Input chunk size, including serial decode-shaped qualification.
    #[arg(long, default_value_t = 4096)]
    pub prefill_chunk: usize,
    /// Check block verify padding, taps, exact restore and accepted-input replay before scoring.
    #[arg(long)]
    pub verify_check: bool,
    /// Maximum absolute mean-NLL difference from the supplied reference.
    #[arg(long, default_value_t = 0.01)]
    pub nll_tolerance: f64,
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    let header = Qwen35Checkpoint::read(&args.snapshot)?;
    let cfg = &header.config;
    tracing::info!(hidden = cfg.hidden, layers = cfg.attention.len(), intermediate = cfg.intermediate,
        tensor_contracts = header.tensors.len(), "dense Qwen checkpoint headers validated for qualification");
    if args.check_headers { return Ok(()); }
    ensure!((1..=4096).contains(&args.prefill_chunk), "prefill chunk must be 1..4096");
    ensure!(args.nll_tolerance.is_finite() && args.nll_tolerance >= 0.0, "invalid NLL tolerance");
    let golden = args.golden.as_ref().context("dense golden requires --golden reference directory")?;
    let reference = reference_snapshot(&args.snapshot, args.model_id.as_deref(), &cuteafd_bench::fidelity_match::hub())?;
    let (directory, positions) = reference_window(golden, &args.window, &args.snapshot, &reference, cfg.vocab)?;
    let token_bytes = std::fs::read(directory.join("tokens.bin"))?;
    ensure!(token_bytes.len() >= 8 && token_bytes.len() % 4 == 0, "invalid reference token payload");
    let tokens: Vec<_> = token_bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(tokens.iter().all(|&token| (token as usize) < cfg.vocab), "reference token outside vocabulary");
    let positions = positions.unwrap_or_else(|| (1..tokens.len()).collect());
    ensure!(!positions.is_empty() && positions.iter().all(|&p| p > 0 && p < tokens.len())
        && positions.windows(2).all(|p| p[0] < p[1]), "invalid reference score positions");
    let reference = std::fs::read(directory.join("logits.bin"))?;
    let reference_rows = if golden.join("windows.json").is_file() { positions.len() } else { tokens.len() };
    ensure!(reference.len() == reference_rows * cfg.vocab * 4, "reference logits shape mismatch");
    let reference: Vec<_> = reference.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(reference.iter().all(|v| v.is_finite()), "nonfinite reference logits");
    let checkpoint = cuteafd_loader::plan::checkpoint::Checkpoint::open(&args.snapshot)?;
    let native = args.native_lib.as_ref().context("dense golden requires --native-lib")?;
    // SAFETY: the selected library remains loaded until programs and all GPU owners drop.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(native)? };
    library.cuda_set_device(args.device)?;
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let mut engine = engine::DenseEngine::new(&library, &programs, &checkpoint, cfg.clone(), args.pool_tokens, None)?;
    if args.verify_check { engine.verify_check(&tokens)?; }
    let mut logits = Vec::with_capacity(tokens.len() * cfg.vocab);
    let mut first = 0;
    for chunk in tokens.chunks(args.prefill_chunk) {
        let values = engine.evaluate(chunk, |_layer, _hidden| Ok(()))?;
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite dense logits at input {first}");
        logits.extend(values);
        first += chunk.len();
    }
    if let Some(directory) = &args.save_logits {
        std::fs::create_dir_all(directory)?;
        std::fs::write(directory.join("tokens.bin"), &token_bytes)?;
        let bytes: Vec<_> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(directory.join("logits.bin"), bytes)?;
    }
    let ours = select_scored_rows(&logits, &positions, cfg.vocab)?;
    let theirs = if reference_rows == tokens.len() {
        select_scored_rows(&reference, &positions, cfg.vocab)?
    } else { reference };
    let targets: Vec<_> = positions.iter().map(|&p| tokens[p]).collect();
    let nll = scored_nll(&ours, &targets, cfg.vocab)?;
    let reference_nll = scored_nll(&theirs, &targets, cfg.vocab)?;
    let (mean_kl, max_kl) = scored_kl(&theirs, &ours, cfg.vocab)?;
    let argmax = |row: &[f32]| row.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0;
    let agree = ours.chunks_exact(cfg.vocab).zip(theirs.chunks_exact(cfg.vocab))
        .filter(|(ours, theirs)| argmax(ours) == argmax(theirs)).count();
    println!("dense prefill logits: {} input rows, top-1 agreement {agree}/{} ({:.4}%); mean NLL {nll:.8}, reference {reference_nll:.8}, delta {:.8}; KL(reference||engine) mean {mean_kl:.8}, max {max_kl:.8}",
        tokens.len(), positions.len(), agree as f64 * 100.0 / positions.len() as f64, nll - reference_nll);
    ensure!((nll - reference_nll).abs() <= args.nll_tolerance,
        "dense mean NLL differs by {:.8}, tolerance {}", (nll - reference_nll).abs(), args.nll_tolerance);
    Ok(())
}

fn reference_snapshot(snapshot: &std::path::Path, model: Option<&str>, hub: &std::path::Path) -> Result<PathBuf> {
    use cuteafd_bench::fidelity_match;
    let inferred = snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    });
    ensure!(model.zip(inferred.as_deref()).is_none_or(|(given, inferred)| given == inferred),
        "dense golden model identity disagrees with snapshot repository");
    let model = model.or(inferred.as_deref()).context("dense golden requires --model-id outside an HF snapshot repository")?;
    let resolution = fidelity_match::resolve_in(model, Some(snapshot), hub)?;
    ensure!(matches!(resolution.reference_root.as_str(), "Qwen/Qwen3.5-9B" | "Qwen/Qwen3.8-27B"),
        "dense reference resolves to an unsupported official root");
    let root = if resolution.reference_root == model { snapshot.to_owned() } else {
        fidelity_match::local_snapshot(hub, &resolution.reference_root)
            .context("mapped highest-precision reference snapshot is not available locally")?
    };
    tracing::info!(model, reference_root = resolution.reference_root, reference_match = resolution.reference_match,
        "dense golden resolved official fidelity root");
    Ok(root)
}

fn reference_window(root: &std::path::Path, window: &str, snapshot: &std::path::Path,
    reference: &std::path::Path, vocab: usize)
    -> Result<(PathBuf, Option<Vec<usize>>)> {
    use sha2::{Digest, Sha256};
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("meta.json"))?)?;
    ensure!(metadata["family"] == "qwen3_5" && metadata["mtp"] == false
        && metadata["reference_sha256"] == "7788020ca357c6c8fb6e750fd847dd80ac77625902213b47f4949c8085344a58",
        "dense reference must use the qualified pinned Transformers target without MTP");
    let sealed = root.join("windows.json").is_file();
    let identity = if sealed { &metadata["snapshot_identity"] } else { &metadata };
    for (file, key) in [("config.json", "config_sha256"), ("tokenizer.json", "tokenizer_sha256")] {
        let hash = format!("{:x}", Sha256::digest(std::fs::read(reference.join(file))?));
        ensure!(identity[key].as_str() == Some(hash.as_str()), "dense reference {key} differs from mapped official root");
    }
    let tokenizer = format!("{:x}", Sha256::digest(std::fs::read(snapshot.join("tokenizer.json"))?));
    ensure!(identity["tokenizer_sha256"].as_str() == Some(tokenizer.as_str()),
        "dense reference tokenizer differs from target checkpoint");
    if !sealed { return Ok((root.to_owned(), None)); }
    ensure!(metadata["schema"] == "cuteafd.fidelity.golden/2"
        && metadata["prefix_qualification"]["passed"] == true, "unqualified fidelity golden");
    let panel: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("windows.json"))?)?;
    ensure!(metadata["set_sha256"] == panel["set_sha256"], "reference window panel identity differs");
    let entry = metadata["windows"].as_array().context("reference windows missing")?.iter()
        .find(|entry| entry["id"].as_str() == Some(window)).context("reference window not found")?;
    ensure!(entry["vocab"].as_u64() == Some(vocab as u64), "reference vocabulary differs");
    let relative = std::path::Path::new(entry["path"].as_str().context("reference window path missing")?);
    ensure!(!relative.as_os_str().is_empty() && relative.components().all(|p| matches!(p, std::path::Component::Normal(_))),
        "reference window path escapes golden directory");
    let positions = entry["positions"].as_array().context("reference score positions missing")?.iter()
        .map(|p| usize::try_from(p.as_u64().context("invalid reference score position")?).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    Ok((root.join(relative), Some(positions)))
}

fn select_scored_rows(logits: &[f32], positions: &[usize], vocab: usize) -> Result<Vec<f32>> {
    ensure!(vocab > 0 && logits.len() % vocab == 0 && !positions.is_empty(), "dense row selection shape mismatch");
    let rows = logits.len() / vocab;
    ensure!(positions.iter().all(|&p| p > 0 && p <= rows)
        && positions.windows(2).all(|p| p[0] < p[1]), "invalid dense score positions");
    Ok(positions.iter().flat_map(|&p| logits[(p - 1) * vocab..p * vocab].iter().copied()).collect())
}

fn scored_nll(logits: &[f32], targets: &[u32], vocab: usize) -> Result<f64> {
    ensure!(vocab > 0 && !targets.is_empty() && logits.len() == targets.len() * vocab,
        "dense scoring shape mismatch");
    ensure!(logits.iter().all(|v| v.is_finite()) && targets.iter().all(|&t| (t as usize) < vocab),
        "dense scoring requires finite logits and in-range tokens");
    let sum = logits.chunks_exact(vocab).zip(targets).map(|(row, &next)| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + row.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln() - row[next as usize] as f64
    }).sum::<f64>();
    Ok(sum / targets.len() as f64)
}

fn scored_kl(reference: &[f32], logits: &[f32], vocab: usize) -> Result<(f64, f64)> {
    ensure!(vocab > 0 && !reference.is_empty() && reference.len() == logits.len()
        && reference.len() % vocab == 0, "dense KL shape mismatch");
    ensure!(reference.iter().chain(logits).all(|v| v.is_finite()), "dense KL requires finite logits");
    let log_z = |row: &[f32]| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + row.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln()
    };
    let mut sum = 0.0;
    let mut max = 0.0f64;
    for (reference, row) in reference.chunks_exact(vocab).zip(logits.chunks_exact(vocab)) {
        let (reference_z, z) = (log_z(reference), log_z(row));
        let kl = reference.iter().zip(row).map(|(&p, &q)| {
            let log_p = p as f64 - reference_z;
            log_p.exp() * (log_p - (q as f64 - z))
        }).sum::<f64>().max(0.0);
        sum += kl;
        max = max.max(kl);
    }
    Ok((sum / (reference.len() / vocab) as f64, max))
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    #[test]
    fn scored_window_rows_predict_the_token_at_position_not_after_it() {
        let logits = [4.0, 0.0, 0.0, 4.0, 9.0, -9.0];
        let selected = super::select_scored_rows(&logits, &[1, 2], 2).unwrap();
        assert_eq!(selected, logits[..4]);
        assert!(super::scored_nll(&selected, &[0, 1], 2).unwrap() < 0.02);
        assert_eq!(super::select_scored_rows(&logits, &[2], 2).unwrap(), [0.0, 4.0]);
        for positions in [&[0][..], &[4][..], &[2, 1][..], &[1, 1][..], &[][..]] {
            assert!(super::select_scored_rows(&logits, positions, 2).is_err());
        }
    }
    #[test]
    fn nll_scores_shifted_tokens_and_rejects_invalid_payloads() {
        let nll = super::scored_nll(&[0.0; 4], &[1, 0], 2).unwrap();
        assert!((nll - 2f64.ln()).abs() < 1e-12);
        assert!(super::scored_nll(&[0.0; 5], &[1, 0], 2).is_err());
        assert!(super::scored_nll(&[0.0; 4], &[2, 0], 2).is_err());
        assert!(super::scored_nll(&[f32::NAN; 4], &[1, 0], 2).is_err());
        assert!(super::scored_nll(&[], &[], 2).is_err());
        let peaked = [4.0, 0.0, 0.0, 4.0];
        assert!(super::scored_nll(&peaked, &[0, 1], 2).unwrap() < 0.02);
        assert!(super::scored_nll(&peaked, &[1, 0], 2).unwrap() > 4.0);
    }
    #[test]
    fn kl_scores_reference_distribution_and_rejects_invalid_payloads() {
        let reference = [0.0, 0.0, 4.0, 0.0];
        assert_eq!(super::scored_kl(&reference, &reference, 2).unwrap(), (0.0, 0.0));
        let shifted = [7.0, 7.0, 11.0, 7.0];
        let (mean, max) = super::scored_kl(&reference, &shifted, 2).unwrap();
        assert!(mean < 1e-12 && max < 1e-12);
        let (mean, max) = super::scored_kl(&[0.0, 0.0], &[4.0, 0.0], 2).unwrap();
        let expected = (2.0f64.exp() + (-2.0f64).exp()).ln() - 2.0f64.ln();
        assert!((mean - expected).abs() < 1e-12 && mean == max);
        assert!(super::scored_kl(&[], &[], 2).is_err());
        assert!(super::scored_kl(&[0.0], &[0.0], 2).is_err());
        assert!(super::scored_kl(&[0.0, 0.0], &[0.0, f32::NAN], 2).is_err());
    }
    #[test]
    fn sealed_reference_preserves_score_positions_and_rejects_identity_or_path_drift() -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};
        use serde_json::json;
        let root = tempfile::tempdir()?;
        let snapshot = tempfile::tempdir()?;
        for file in ["config.json", "tokenizer.json"] { std::fs::write(snapshot.path().join(file), b"{}")?; }
        let hash = format!("{:x}", Sha256::digest(b"{}"));
        let mut metadata = json!({"family":"qwen3_5", "mtp":false,
            "reference_sha256":"7788020ca357c6c8fb6e750fd847dd80ac77625902213b47f4949c8085344a58",
            "schema":"cuteafd.fidelity.golden/2", "set_sha256":"panel",
            "prefix_qualification":{"passed":true},
            "snapshot_identity":{"config_sha256":hash,"tokenizer_sha256":hash},
            "windows":[{"id":"a00","path":"windows/a00","positions":[64,65],"vocab":2}]});
        std::fs::write(root.path().join("windows.json"), br#"{"set_sha256":"panel"}"#)?;
        let write = |metadata: &serde_json::Value| std::fs::write(root.path().join("meta.json"), metadata.to_string());
        write(&metadata)?;
        let (path, positions) = super::reference_window(root.path(), "a00", snapshot.path(), snapshot.path(), 2)?;
        assert_eq!(path, root.path().join("windows/a00"));
        assert_eq!(positions, Some(vec![64,65]));
        assert!(super::reference_window(root.path(), "missing", snapshot.path(), snapshot.path(), 2).is_err());
        assert!(super::reference_window(root.path(), "a00", snapshot.path(), snapshot.path(), 3).is_err());
        metadata["windows"][0]["path"] = json!("../escape");
        write(&metadata)?;
        assert!(super::reference_window(root.path(), "a00", snapshot.path(), snapshot.path(), 2).is_err());
        metadata["windows"][0]["path"] = json!("windows/a00");
        metadata["snapshot_identity"]["config_sha256"] = json!("different");
        write(&metadata)?;
        assert!(super::reference_window(root.path(), "a00", snapshot.path(), snapshot.path(), 2).is_err());
        Ok(())
    }
    #[test]
    fn quant_reference_uses_shared_ancestry_and_keeps_root_config_and_target_tokenizer_checks() -> anyhow::Result<()> {
        use serde_json::json;
        use sha2::{Digest, Sha256};
        let hub = tempfile::tempdir()?;
        let make = |model: &str, config: &str| -> anyhow::Result<std::path::PathBuf> {
            let path = hub.path().join(format!("models--{}/snapshots/abcdef", model.replace('/', "--")));
            std::fs::create_dir_all(&path)?;
            std::fs::write(path.join("config.json"), config)?;
            std::fs::write(path.join("tokenizer.json"), "{}")?;
            Ok(path)
        };
        let official = make("Qwen/Qwen3.8-27B", "{}")?;
        let quant = make("Qwen/Qwen3.8-27B-FP8", r#"{"quantization_config":{"quant_method":"fp8"}}"#)?;
        assert_eq!(super::reference_snapshot(&quant, None, hub.path())?, official);
        assert!(super::reference_snapshot(&quant, Some("Qwen/Qwen3.5-9B"), hub.path()).is_err());
        let community = make("test/quant", "{}")?;
        assert!(super::reference_snapshot(&community, None, hub.path()).is_err());
        std::fs::write(community.join("README.md"), "---\nbase_model: Qwen/Qwen3.8-27B\nbase_model_relation: quantized\n---\n")?;
        assert_eq!(super::reference_snapshot(&community, None, hub.path())?, official);
        std::fs::write(community.join("README.md"), "---\nbase_model: Qwen/Qwen3.5-9B\nbase_model_relation: finetune\n---\n")?;
        assert!(super::reference_snapshot(&community, None, hub.path()).is_err());
        let golden = tempfile::tempdir()?;
        let hash = format!("{:x}", Sha256::digest(b"{}"));
        let metadata = json!({"family":"qwen3_5", "mtp":false,
            "reference_sha256":"7788020ca357c6c8fb6e750fd847dd80ac77625902213b47f4949c8085344a58",
            "config_sha256":hash,"tokenizer_sha256":hash});
        std::fs::write(golden.path().join("meta.json"), metadata.to_string())?;
        assert!(super::reference_window(golden.path(), "legacy", &quant, &official, 2).is_ok());
        std::fs::write(official.join("config.json"), r#"{"changed":true}"#)?;
        assert!(super::reference_window(golden.path(), "legacy", &quant, &official, 2).is_err());
        std::fs::write(official.join("config.json"), "{}")?;
        std::fs::write(quant.join("tokenizer.json"), r#"{"changed":true}"#)?;
        assert!(super::reference_window(golden.path(), "legacy", &quant, &official, 2).is_err());
        Ok(())
    }
    #[test]
    fn qwen35_golden_is_registered() {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "qwen35-golden", "--snapshot", "/m", "--check-headers"]).unwrap();
        let crate::cli::Commands::Qwen35Golden(args) = cli.command else { panic!("dense golden not registered") };
        assert_eq!(args.nll_tolerance, 0.01);
    }
}
