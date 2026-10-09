//! A representative report for render tests and page development.
use crate::report::*;

pub fn report(failed: bool) -> Report {
    let gpu = |index: u32, used: bool| Gpu { index, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition".into(),
        uuid: Some(format!("GPU-{index}")), memory_mib: Some(97_887), power_limit_w: Some(325.0),
        power_max_w: Some(600.0), sm_count: Some(188), compute_cap: Some("12.0".into()), pcie: Some("Gen5 x16".into()),
        used };
    let names = ["ostrich", "dodo", "emu", "kiwi"];
    let hardware = Hardware {
        host: Some("raptor".into()), gpus: vec![gpu(0, true), gpu(1, false)], driver: Some("580.95.05".into()),
        cuda: Some("13.0".into()),
        sparks: names.iter().enumerate().map(|(i, n)| Spark { address: format!("10.55.0.{}", i + 1),
            name: Some(n.to_string()), rank: i as u32 }).collect(),
        fabric: vec![FabricPort { device: "mlx5_0".into(), port: 1, active: true, link_gbps: 400.0,
            pcie: Some("32 GT/s x16".into()), netdev: Some("enp1s0f0np0".into()),
            subnets: vec!["10.55.0.0/24".into(), "10.55.1.0/24".into()] }],
        rails: Some("2 rails on one 400 Gb/s port; using 1 (link rate < PCIe ingress)".into()),
    };
    let setting = |name: &str, value: &str, default: Option<&str>, source: &str| Setting { name: name.into(),
        value: Some(value.into()), default: default.map(str::to_string), source: source.into() };
    let configuration = Configuration {
        snapshot: Some("/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa".into()),
        quant: vec![QuantGroup { group: "attention".into(), formats: vec!["fp8_block".into()] },
            QuantGroup { group: "routed expert".into(), formats: vec!["mxfp4".into()] },
            QuantGroup { group: "shared expert".into(), formats: vec!["fp8_block".into()] }],
        speculator: Some("dSpark (≤5 drafts)".into()),
        layout: Some("1 RTX · experts TP4 over 4 Sparks".into()),
        settings: vec![setting("dspark", "true", Some("false"), "cli"), setting("concurrency", "16", Some("16"), "cli"),
            setting("prefill-batch-tokens", "4096", Some("2048"), "cli"),
            setting("CUTEAFD_NVFP4_ACTIVATIONS", "a16", None, "env")],
    };
    let timing = |completion: u64, decode_s: f64| StreamTiming { prompt_tokens: 61, completion_tokens: completion,
        ttft_s: 0.12, total_s: 0.12 + decode_s, decode_s, ..StreamTiming::default() };
    let rate = |content: &str, tok_s: f64| ContentRate { content: content.into(), tok_s,
        runs: vec![timing(320, 319.0 / tok_s)], acceptance: None };
    let mut quality = Quality::default();
    let mut check = |id: &str, title: &str, status: CheckStatus, summary: &str, metrics: &[(&str, f64)]| {
        let mut c = Check::new(id, title);
        c.status = status;
        c.summary = summary.into();
        c.seconds = 4.2;
        for (k, v) in metrics {
            c.set(k, *v);
        }
        quality.checks.push(c);
    };
    check("fidelity", "Logit fidelity", if failed { CheckStatus::Fail } else { CheckStatus::Pass },
        if failed { "KL 0.912 · top-1 41.0% · NLL 5.120 vs 3.338 · 512 tokens vs Qwen 3.8 Flash Next reference" }
        else { "KL 0.058 · top-1 89.9% · NLL 3.402 vs 3.338 · 512 tokens vs Qwen 3.8 Flash Next reference" },
        &[("kl", if failed { 0.912 } else { 0.0581 }), ("top1", if failed { 0.41 } else { 0.899 })]);
    check("cache_exact", "Prefix-cache restore", CheckStatus::Pass,
        "prompt end: 1104/1472 restored byte-identical · turn end: 1536/1632 restored byte-identical", &[]);
    check("spec_lossless", "Speculation lossless", CheckStatus::Pass,
        "dSpark (≤5 drafts): 128 greedy tokens identical with drafts on and off (187 vs 61.2 tok/s)", &[]);
    check("template", "Template round trip", CheckStatus::Pass,
        "tool call parsed · reasoning kept · re-render identical (412 tokens, 410 cached)", &[]);
    check("c1_c4", "C1 vs C4 divergence", CheckStatus::Info, "4 of 4 concurrent greedy outputs identical to C1 (64 tokens)", &[]);
    quality.settle();
    let baseline = Baseline {
        fingerprint: "3f9a2c41d07be5a1".into(), run_id: "7c1e2d3f4a5b6c7d8e9f".into(), created: "2026-10-02T10:12:00Z".into(),
        card: BasicCard { decode: vec![rate("code", 187.3), rate("prose", 142.6), rate("json", 201.9)],
            concurrent: Some(ConcurrentRate { width: 8, aggregate_tok_s: 960.0, per_stream_median_tok_s: 120.0,
                decode_s: 319.0 / 120.0, runs: (0..8).map(|_| ConcurrentTiming { sent_s: 0.0,
                    timing: timing(320, 319.0 / 120.0) }).collect(), warmup_s: 3.0 }),
            prefill: Some(PrefillRate { prompt_tokens: 8192, tok_s: 2415.0, ttft_s: 3.392, runs: vec![] }),
            warmup_s: Some(14.2),
            capacity: Some(Capacity { kv_tokens: Some(14_710_000), kv_pages: Some(28_728),
                kv_format: Some("FP4 compressed (CSA/HCA) + FP8 window".into()), max_requests: Some(16),
                max_context: Some(1 << 20), max_output: Some(393_216), host_cache_bytes: Some(64 << 30) }) },
        quality, seconds: 151.0,
    };
    Report {
        schema: SCHEMA.into(), id: "7c1e2d3f4a5b6c7d8e9f".into(), created: "2026-10-02T10:12:00Z".into(),
        finished: Some("2026-10-02T10:14:31Z".into()), status: RunStatus::Done, profile: "smoke".into(),
        plan: vec![PlannedPanel { id: "hardware".into(), passes: 1 }, PlannedPanel { id: "configuration".into(), passes: 1 }],
        server: ServerInfo { model: "deepseek-ai/DeepSeek-V4.1-Flash".into(), family: Some("deepseek_v41".into()),
            revision: Some("dba1be0a40aa45a94ad051997016db3960a90277".into()),
            build: BuildInfo { version: "0.1.0".into(), release: None, image: None,
                remote: Some("https://github.com/tpurtell/cuteafd".into()),
                commit: Some("8facee6a1b2c3d4e5f60718293a4b5c6d7e8f901".into()), dirty: Some(false) },
            hardware, configuration, readiness_s: Some(112.4), started: Some("2026-10-02T10:10:07Z".into()) },
        fingerprint: "3f9a2c41d07be5a1".into(), baseline: Some(baseline),
        panels: vec![PanelResult { id: "hardware".into(), title: "Hardware".into(), status: PanelStatus::Done,
            passes: vec![serde_json::json!({})], ..PanelResult::default() }],
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use crate::render;

    #[test]
    fn old_reports_without_concurrent_rate_still_deserialize() {
        let mut value = serde_json::to_value(super::report(false)).unwrap();
        value["baseline"]["card"].as_object_mut().unwrap().remove("concurrent");
        let report: crate::report::Report = serde_json::from_value(value).unwrap();
        assert!(report.baseline.as_ref().unwrap().card.concurrent.is_none());
        let card = render::card::card_svg(&report);
        assert!(!card.contains("C8 CODE DECODE"));
        assert!(card.contains(r#"x="610" y="246""#));
        assert!(!render::report::panel_svg(&report, "baseline").contains("C8 CODE DECODE"));
    }

    #[test]
    fn concurrent_rate_renders_at_the_measured_width_everywhere() {
        let mut report = super::report(false);
        for width in [8, 4] {
            report.baseline.as_mut().unwrap().card.concurrent.as_mut().unwrap().width = width;
            let label = format!("C{width} CODE DECODE");
            let card = render::card::card_svg(&report);
            let panel = render::report::panel_svg(&report, "baseline");
            assert!(card.contains(&label) && card.contains(">960</text>"));
            assert!(panel.contains(&label) && panel.contains(">960</text>"));
            assert!(card.contains("aggregate · 120 tok/s per stream"));
            let color = render::Theme::for_report(false).concurrent;
            assert!(card.contains(&format!(r#"fill="{color}""#)));
            assert!(panel.contains(&format!(r#"fill="{color}""#)));
            assert!(!render::Theme::for_report(false).series[..4].contains(&color));
            assert!(!card.contains("code, aggregate"));
            assert!(card.find("C1 CODE DECODE").unwrap() < card.find(&label).unwrap());
            assert!(card.find(&label).unwrap() < card.find("8K PREFILL").unwrap());
            assert!(render::report::report_svg(&report).contains(&label));
            let placed = crate::publish::Placed { dir: "benchmarks/sample/card".into(), report: report.clone() };
            let readme = crate::publish::results(&[placed]);
            assert!(readme.contains("Concurrent code (aggregate)"));
            assert!(readme.contains(&format!("C{width}: 960")));
            assert!(readme.contains("<th>Model · quant</th><th>5090</th><th>1× RTX</th><th>2× RTX</th>"));
        }
    }

    #[test]
    fn publication_evidence_stays_in_full_report_not_card_options() {
        let mut report = super::report(false);
        let card = render::card::card_svg(&report);
        for name in ["provenance.artifacts", "qualification.conditions", "supersedes-checkpoint"] {
            report.server.configuration.settings.push(crate::report::Setting {
                name: name.into(), value: Some("x".repeat(2048)), default: None, source: "publication".into(),
            });
        }
        assert_eq!(render::card::card_svg(&report), card);
        let full = render::report::panel_svg(&report, "configuration");
        for name in ["provenance.artifacts", "qualification.conditions", "supersedes-checkpoint"] {
            assert!(full.contains(name), "{name} evidence missing from full report");
        }
    }

    /// Renders every export of a normal and a failed report; with
    /// `CUTEAFD_BENCH_SAMPLE_DIR` set, writes them there for a look.
    #[test]
    fn sample_exports_render_to_svg_and_png() {
        let dir = std::env::var_os("CUTEAFD_BENCH_SAMPLE_DIR").map(std::path::PathBuf::from);
        for failed in [false, true] {
            let report = super::report(failed);
            assert_eq!(report.quality_failed(), failed);
            let tag = if failed { "failed" } else { "ok" };
            let files = [("report", render::report::report_svg(&report)), ("card", render::card::card_svg(&report)),
                ("panel-baseline", render::report::panel_svg(&report, "baseline")),
                ("panel-hardware", render::report::panel_svg(&report, "hardware")),
                ("panel-configuration", render::report::panel_svg(&report, "configuration"))];
            for (name, svg) in files {
                assert!(!svg.contains("<script") && !svg.contains("http://www.w3.org/1999/xlink")
                    && !svg.contains("href="), "{name}: external reference");
                assert_eq!(svg.contains("UNVERIFIED — QUALITY GATE FAILED"), failed, "{name}");
                let png = render::png::png(&svg, 1.0).unwrap_or_else(|e| panic!("{name}: {e:#}"));
                assert!(png.starts_with(b"\x89PNG"));
                if let Some(dir) = &dir {
                    std::fs::create_dir_all(dir).unwrap();
                    std::fs::write(dir.join(format!("{name}-{tag}.svg")), &svg).unwrap();
                    std::fs::write(dir.join(format!("{name}-{tag}.png")), &png).unwrap();
                }
            }
            let card = render::card::card_svg(&report);
            assert!(card.contains(r#"width="1200" height="675""#));
        }
    }
}

/// A report with synthetic passes of every measurement panel (render work).
pub fn full_report() -> Report {
    use serde_json::json;
    let mut report = report(false);
    let pass = |id: &str, value: serde_json::Value| PanelResult { id: id.into(), title: id.into(),
        status: PanelStatus::Done, passes: vec![value], ..PanelResult::default() };
    let contents = ["code", "prose", "json", "math", "chat", "translation", "summary", "table"];
    let rows: Vec<_> = contents.iter().enumerate().map(|(i, c)| json!({"content": c, "tok_s": 140.0 + 9.0 * i as f64,
        "tokens": 256, "ttft_s": 0.12, "acceptance": 0.6 + 0.03 * i as f64})).collect();
    let levels = ["off", "10", "low 25", "high 50", "xhigh 75", "max 100"];
    let mut reasoning = Vec::new();
    for (l, level) in levels.iter().enumerate() {
        for (k, tier) in ["medium", "medium", "medium", "hard", "hard", "brutal", "aime", "aime"].iter().enumerate() {
            let tokens = (40.0 * 2f64.powi(l as i32) * (1.0 + k as f64 * 0.4)) as u64;
            reasoning.push(json!({"level": level, "tier": tier, "correct": (l + k) % 3 != 0 || l > 3,
                "hit_cap": l == 5 && k == 7, "reasoning_tokens": tokens, "solo_s": tokens as f64 / 180.0}));
        }
    }
    let fidelity_score = crate::reference::Fidelity::from_records((0..64).flat_map(|window| (0..8).map(move |position|
        crate::reference::Position { window: format!("a{window:02}"), block: "A".into(), bucket: "0-2K".into(),
            role: "gen".into(), position, agree: true, confident: true, top3_contained: true, agree_text: true,
            finite: true, kl: 0.005, nll: 0.5, ref_nll: 0.5, argmax: 1, reference_argmax: 1 })).collect());
    let mut fidelity_run = crate::fidelity::Run { schema: "cuteafd.fidelity.run/2".into(), arm: "synthetic".into(),
        checkpoint: "synthetic".into(), set_sha256: "synthetic".into(), reference_sha256: "synthetic".into(),
        tier: "standard".into(), path_shape: "decode-shaped".into(), kl_kind: "full-vocabulary".into(),
        verify_rows: None, reference_selection: None, standard_balance: None, dataset: Some(json!({"config": "synthetic-long-config-for-responsive-layout",
            "revision": "a".repeat(40)})), engine: "synthetic".into(), settings: json!({}), seconds: 346.0,
        score: fidelity_score, floor_top1: 0.985, floor_kl: 0.06, tripwire_expect: None };
    let full_score = fidelity_run.score.clone();
    fidelity_run.score = crate::reference::Fidelity::from_records(full_score.records.iter().filter(|p| p.window[1..].parse::<usize>().unwrap() < 32).cloned().collect());
    fidelity_run.tier = crate::fidelity_dataset::STANDARD_TIER.into();
    fidelity_run.dataset.as_mut().unwrap()["standard_subset"] = json!({"version":"standard-v2", "mode":"32 decode / 32 prefill"});
    let mut fidelity = crate::panels::fidelity::record(&fidelity_run, Some(&fidelity_run));
    fidelity_run.path_shape = "prefill-shaped".into();
    fidelity_run.score = crate::reference::Fidelity::from_records(full_score.records.iter().filter(|p| p.window[1..].parse::<usize>().unwrap() >= 32).cloned().collect());
    for (key, value) in crate::panels::fidelity::record(&fidelity_run, None).as_object().unwrap() {
        if key.starts_with("prefill") { fidelity[key] = value.clone(); }
    }
    fidelity_run.path_shape = "decode-shaped".into();
    fidelity_run.dataset.as_mut().unwrap().as_object_mut().unwrap().remove("standard_subset");
    fidelity_run.score = full_score;
    fidelity_run.tier = "full".into();
    let mut full_fidelity = crate::panels::fidelity::record(&fidelity_run, Some(&fidelity_run));
    fidelity_run.path_shape = "prefill-shaped".into(); fidelity_run.seconds = 240.0;
    for (key, value) in crate::panels::fidelity::record(&fidelity_run, None).as_object().unwrap() {
        if key.starts_with("prefill") { full_fidelity[key] = value.clone(); }
    }
    report.panels = vec![
        pass("fidelity", fidelity), pass("fidelity_full", full_fidelity),
        pass("decode_content", json!({"rows": rows})),
        pass("concurrency", json!({"points": ([1, 2, 4, 8, 16].iter().map(|&c| json!({"c": c,
            "aggregate_tok_s": 187.0 * (c as f64).powf(0.7), "per_request_tok_s": 187.0 / (c as f64).powf(0.3),
            "ttft_s": 0.1 * c as f64})).collect::<Vec<_>>())})),
        pass("prefill", json!({"points": ([1024, 2048, 4096, 8192, 16384, 32768, 65536].iter().map(|&l| json!({
            "target": l, "prompt_tokens": l, "cold_ttft_s": l as f64 / 2400.0 + 0.05, "cold_tok_s": 2400.0 - l as f64 / 100.0,
            "cached_tokens": l - 256, "cached_ttft_s": 0.11 + l as f64 / 1e6})).collect::<Vec<_>>())})),
        pass("retained", json!({"points": ([0, 4096, 16384, 65536].iter().map(|&c| json!({"context": c, "prompt_tokens": c + 60,
            "decode_tok_s": 187.0 - c as f64 / 1000.0})).collect::<Vec<_>>())})),
        pass("prefix_cache", json!({"turns": (1..=6).map(|t| json!({"turn": t, "prompt_tokens": 1000 + 300 * t,
            "cached_tokens": if t == 1 { 0 } else { 1000 + 300 * (t - 1) }, "ttft_s": if t == 1 { 0.6 } else { 0.15 },
            "cold_ttft_s": 0.5 + 0.1 * t as f64})).collect::<Vec<_>>()})),
        pass("structured", json!({"rows": [{"schema": "person", "valid": true, "tok_s": 150.0, "free_tok_s": 170.0},
            {"schema": "order", "valid": true, "tok_s": 140.0, "free_tok_s": 168.0},
            {"schema": "ticket", "valid": false, "tok_s": 120.0, "free_tok_s": 160.0}]})),
        pass("needle", json!({"cells": ([2048, 8192, 32768, 131072].iter().flat_map(|&l| [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter().map(move |&d| json!({"length": l, "depth": d, "found": !(l == 131072 && d == 0.5)}))).collect::<Vec<_>>())})),
        pass("math", json!({"correct": 10, "rows": (0..12).map(|i| json!({"question": format!("problem {i}"),
            "correct": i % 6 != 5})).collect::<Vec<_>>()})),
        pass("code", json!({"passed": 9, "sandbox": "subprocess, no network", "rows": (["is_palindrome", "fizzbuzz",
            "merge_intervals", "roman", "anagram_groups", "longest_unique", "primes_upto", "flatten", "rle", "balanced",
            "top_k_words", "matrix_spiral"].iter().enumerate().map(|(i, p)| json!({"problem": p, "passed": i % 4 != 3}))
            .collect::<Vec<_>>())})),
        pass("ifeval", json!({"prompt_accuracy": 0.85, "instruction_accuracy": 0.9, "rows": (0..20).map(|i|
            json!({"passed": i % 7 != 0})).collect::<Vec<_>>()})),
        pass("reasoning_effort", json!({"levels": levels, "rows": reasoning})),
        pass("agentic", json!({"task": "money", "success": true, "seconds": 182.0, "turns": (1..=7).map(|t| json!({"turn": t,
            "prompt_tokens": 6000 + 1800 * t, "cached_tokens": if t == 1 { 0 } else { 6000 + 1800 * (t - 1) },
            "ttft_s": if t == 1 { 2.1 } else { 0.4 }, "decode_tok_s": 120.0, "full_turn_reused": t > 1,
            "invalid_tool_calls": 0})).collect::<Vec<_>>()})),
        pass("startup", json!({"phases": [{"name": "engine loaded", "at_s": 98.0}, {"name": "API listening", "at_s": 99.1}],
            "readiness_s": 99.1, "warmup_s": 14.2})),
        PanelResult { id: "tool_eval".into(), title: "Tool eval".into(), status: PanelStatus::Done,
            passes: vec![json!({"standard": 121, "standard_max": 138, "hard": 19, "hard_max": 30, "total": 140, "total_max": 168})],
            history: vec![json!({"standard": 117, "hard": 21, "total": 138}), json!({"standard": 124, "hard": 17, "total": 141})],
            ..PanelResult::default() },
    ];
    report
}

#[cfg(test)]
mod full_tests {
    #[test]
    fn every_panel_renders() {
        let dir = std::env::var_os("CUTEAFD_BENCH_SAMPLE_DIR").map(std::path::PathBuf::from);
        let report = super::full_report();
        if let Some(dir) = &dir {
            std::fs::write(dir.join("full-report.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        }
        for panel in &report.panels {
            let svg = crate::render::report::panel_svg(&report, &panel.id);
            assert!(!svg.contains("Pass 1"), "{} fell back to the generic body", panel.id);
            let png = crate::render::png::png(&svg, 1.0).unwrap();
            if let Some(dir) = &dir {
                std::fs::write(dir.join(format!("full-{}.png", panel.id)), png).unwrap();
                if panel.id.starts_with("fidelity") {
                    let narrow = crate::render::report::panel_body_svg(&report, &panel.id, 320.0);
                    std::fs::write(dir.join(format!("narrow-{}.png", panel.id)), crate::render::png::png(&narrow, 1.0).unwrap()).unwrap();
                }
            }
        }
        let svg = crate::render::report::report_svg(&report);
        if let Some(dir) = &dir {
            std::fs::write(dir.join("full-report.png"), crate::render::png::png(&svg, 1.0).unwrap()).unwrap();
        }
    }
}
