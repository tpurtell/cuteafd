//! Every launchable (family, coordinator ranks) resolves a nonzero decode-graph
//! reserve with graphs on; and
//! each family × ranks envelope equals the maximum of that family's checked-in
//! measured charges, so the fallback table cannot drift from its sources.
use super::*;
use crate::placement::AttentionPlacement;
use serde_json::Value;
use std::collections::BTreeSet;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

/// The launcher family each matrix family name or example `MODEL_ID` serves.
fn family_of(name: &str) -> Option<GraphFamily> {
    use GraphFamily::*;
    Some(match name {
        "muse_spark" => Muse, "kimi_k3" => Kimi, "mimo_v2" => Mimo, "qwen3_5" => Qwen35, "step5" => Step5,
        "deepseek_v4" => V4, "deepseek_v41" => V41, "minimax_m3" => MiniMax, "glm5" => Glm5,
        "mistral_large4" => Ml4, "glm5_flash" => GlmFlash, "qwen4" => Qwen4, "hy4_flash" => Hy4Flash,
        "hy4" => Hy4, "inkling2" => Inkling2, "inkling2_mini" => Inkling2Mini,
        _ => return None,
    })
}

fn model_family(model: &str) -> GraphFamily {
    let name = model.rsplit('/').next().unwrap().to_ascii_lowercase();
    let family = [("muse-spark", "muse_spark"), ("kimi-k3", "kimi_k3"), ("mimo-v2", "mimo_v2"),
        ("step-5", "step5"), ("deepseek-v4.1", "deepseek_v41"), ("deepseek-v4-", "deepseek_v4"),
        ("minimax-m3", "minimax_m3"), ("glm-5.3-flash", "glm5_flash"), ("glm-5.3", "glm5"),
        ("mistral-large-4", "mistral_large4"), ("qwen3.8-flash-next", "qwen4"), ("qwen3.8-27b", "qwen3_5"),
        ("qwen3.5-", "qwen3_5"), ("hy4-flash", "hy4_flash"), ("hy4-", "hy4"), ("inkling-2-mini", "inkling2_mini"),
        ("inkling-2-small", "inkling2")].into_iter()
        .find_map(|(prefix, family)| name.starts_with(prefix).then_some(family))
        .unwrap_or_else(|| panic!("example config MODEL_ID {model} has no graph family; add it here"));
    family_of(family).unwrap()
}

/// (family, coordinator ranks, Sparks attached) the launcher admits: every
/// `current`/`pending` matrix cell and every live (non-retired) example config.
fn launchable() -> BTreeSet<(String, usize, bool)> {
    let matrix: Value = serde_json::from_slice(&std::fs::read(format!("{ROOT}/benchmarks/current-matrix.json")).unwrap()).unwrap();
    let mut out = BTreeSet::new();
    for cell in matrix["cells"].as_array().unwrap().iter().chain(matrix["supplemental"].as_array().unwrap()) {
        let family = cell["family"].as_str().unwrap();
        for (column, ranks) in [("1rtx", 1), ("2rtx", 2)] {
            if matches!(cell[column]["status"].as_str(), Some("current" | "pending")) {
                // Matrix cells do not record Spark counts; cover both attachments.
                out.insert((family.to_string(), ranks, true));
                out.insert((family.to_string(), ranks, false));
            }
        }
    }
    let retired = matrix["retired"].as_array().unwrap().iter()
        .filter_map(|entry| entry["config"].as_str().map(str::to_string)).collect::<BTreeSet<_>>();
    for entry in std::fs::read_dir(format!("{ROOT}/examples/configs")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "config") { continue }
        let relative = format!("examples/configs/{}", path.file_name().unwrap().to_string_lossy());
        if retired.contains(&relative) { continue }
        let text = std::fs::read_to_string(&path).unwrap();
        let get = |key: &str| text.lines().find_map(|line| line.strip_prefix(&format!("{key}=")).map(str::to_string));
        let model = get("MODEL_ID").unwrap_or_else(|| panic!("{relative}: MODEL_ID"));
        let ranks = get("RTX_GPUS").and_then(|v| v.parse().ok()).unwrap_or(1);
        let sparks = get("SPARK_COUNT").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) > 0;
        let family = model_family(&model);
        let name = family_of_name(family);
        out.insert((name.to_string(), ranks, sparks));
    }
    out
}

fn family_of_name(family: GraphFamily) -> &'static str {
    ["muse_spark", "kimi_k3", "mimo_v2", "qwen3_5", "step5", "deepseek_v4", "deepseek_v41", "minimax_m3", "glm5",
        "mistral_large4", "glm5_flash", "qwen4", "hy4_flash", "hy4", "inkling2", "inkling2_mini"]
        .into_iter().find(|name| family_of(name) == Some(family)).unwrap()
}

#[test]
fn every_launchable_family_rank_count_has_an_envelope_row() {
    let launchable = launchable();
    assert!(launchable.len() > 20, "launch inventory unexpectedly small: {launchable:?}");
    for (name, ranks, sparks) in launchable {
        let family = family_of(&name).unwrap_or_else(|| panic!("matrix family {name} has no graph family"));
        let row = ENVELOPES.iter().find(|(f, r, _)| *f == family && *r == ranks)
            .map(|(_, _, bytes)| *bytes)
            // Only Qwen 3.5 2x documents borrowing the 1x row.
            .or_else(|| (family == GraphFamily::Qwen35 && ranks == 2).then(|| {
                ENVELOPES.iter().find(|(f, r, _)| *f == family && *r == 1).unwrap().2
            }))
            .unwrap_or_else(|| panic!("{name} {ranks}x RTX (Sparks {sparks}) has no graph reserve envelope row"));
        for rank in 0..ranks {
            assert!(envelope_bytes(family, ranks, rank) > 0, "{name} {ranks}x rank {rank}");
        }
        assert!(row.iter().all(|&bytes| bytes > 0) && (row.len() == ranks || family == GraphFamily::Qwen35));
    }
}

fn charged(base: u64, tail: u64) -> u64 {
    resolve(None, 0, false, base, tail).unwrap().bytes
}

fn working(measured: WorkingSet) -> u64 {
    resolve_working_set(None, None, None, 0, measured).unwrap().bytes
}

fn json(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(format!("{}/src/serving_capacity/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
}

fn as_vec(value: &Value, ranks: usize) -> Vec<u64> {
    value.as_array().map_or_else(|| vec![value.as_u64().unwrap(); ranks],
        |values| values.iter().map(|v| v.as_u64().unwrap()).collect())
}

/// Per-rank maxima of each family's measured default charges, by rank count.
fn measured_maxima() -> Vec<(GraphFamily, usize, Vec<u64>)> {
    use GraphFamily::*;
    let mut points: Vec<(GraphFamily, usize, Vec<u64>)> = Vec::new();
    let mut push = |family, values: Vec<u64>| points.push((family, values.len(), values));
    for (family, file) in [(Muse, "muse_prewarm.json"), (Kimi, "kimi_prewarm.json"), (Mimo, "mimo_prewarm.json"),
        (Qwen35, "qwen35_prewarm.json"), (V4, "v4_prewarm.json")] {
        let list = json(file);
        let layouts: Vec<Value> = match &list["layouts"] {
            Value::Array(layouts) => layouts.clone(),
            Value::Object(layouts) => layouts.values().cloned().collect(),
            _ => unreachable!(),
        };
        for layout in layouts {
            if layout.get("traced") == Some(&Value::Bool(false)) { continue }
            let base = as_vec(&layout["reserve_base_bytes"], 1);
            let tail = as_vec(&layout["reserve_extra_bytes"], base.len());
            push(family, base.iter().zip(&tail).map(|(&b, &t)| charged(b, t)).collect());
        }
    }
    for trace in json("step5_prewarm.json")["layouts"].as_array().unwrap() {
        push(Step5, trace["ranks"].as_array().unwrap().iter().map(|rank|
            charged(rank["reserve_base_bytes"].as_u64().unwrap(), rank["reserve_extra_bytes"].as_u64().unwrap())).collect());
    }
    for ranks in [1, 2] {
        // V4.1: one lane at 64 and 128 rows, two lanes at 64 (the measured bands).
        for (lanes, rows) in [(1, 64), (1, 128), (2, 64)] {
            push(V41, (0..ranks).map(|rank| working(v41_working_set(ranks, rank, lanes, rows).unwrap())).collect());
        }
        for family in [crate::serving_capacity::glm_graphs::GlmPrewarmFamily::Glm5,
            crate::serving_capacity::glm_graphs::GlmPrewarmFamily::MistralLarge4] {
            let name = if family == crate::serving_capacity::glm_graphs::GlmPrewarmFamily::Glm5 { Glm5 } else { Ml4 };
            push(name, (0..ranks).map(|rank| working(glm_default(family, ranks, rank))).collect());
        }
        for split in [false, true] {
            push(GlmFlash, (0..ranks).map(|rank| working(crate::serving_capacity::glmf_graphs::working_set(ranks, split, rank))).collect());
        }
        let key = crate::serving_capacity::qwen_graphs::QwenGraphShapeKey::resolve(ranks, false);
        push(Qwen4, (0..ranks).map(|rank| working(crate::serving_capacity::qwen_graphs::qwen_working_set(key, rank).unwrap())).collect());
        for sparks in [0, 1, 4] {
            let layout = crate::serving_capacity::hy4_graphs::Hy4GraphLayout::resolve(ranks, sparks);
            let values = (0..ranks).map(|rank| crate::serving_capacity::hy4_graphs::reserve_policy(layout, ranks, rank, true).unwrap())
                .filter(|policy| policy.source == "working-set reserve").map(|policy| policy.bytes).collect::<Vec<_>>();
            if values.len() == ranks { push(Hy4Flash, values); }
        }
        let attention = if ranks == 1 { AttentionPlacement::Heads } else { AttentionPlacement::Context };
        let layout = crate::families::hy4::graph_reserve::Layout::resolve(ranks, if ranks == 1 { 4 } else { 6 }, attention);
        push(Hy4, (0..ranks).map(|rank| crate::families::hy4::graph_reserve::reserve_policy(layout, ranks, rank, true).unwrap().bytes).collect());
        push(Inkling2, (0..ranks).map(|rank| crate::serving_capacity::inkling2_graphs::policy(rank, ranks, true).unwrap().bytes).collect());
    }
    for (gpus, attention, wire) in [(1, AttentionPlacement::Heads, None), (2, AttentionPlacement::Heads, None),
        (2, AttentionPlacement::Layers, Some(crate::families::minimax_m3::execution::WirePrecision::F32))] {
        push(MiniMax, (0..gpus).map(|rank| crate::families::minimax_m3::admission::graph_policy(gpus, 16, rank, attention, wire, true)
            .unwrap().bytes).collect());
    }
    use crate::serving_capacity::inkling2_mini_graphs::{policy, KeySpace};
    push(Inkling2Mini, vec![policy(0, KeySpace::for_config(64, 64, true), true).unwrap().bytes]);
    let mut maxima: Vec<(GraphFamily, usize, Vec<u64>)> = Vec::new();
    for (family, ranks, values) in points {
        match maxima.iter_mut().find(|(f, r, _)| *f == family && *r == ranks) {
            Some((_, _, max)) => for (m, v) in max.iter_mut().zip(values) { *m = (*m).max(v) },
            None => maxima.push((family, ranks, values)),
        }
    }
    maxima
}

#[test]
fn envelopes_equal_measured_maxima() {
    cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
        let maxima = measured_maxima();
        for (family, ranks, values) in &maxima {
            let row = ENVELOPES.iter().find(|(f, r, _)| f == family && r == ranks)
                .unwrap_or_else(|| panic!("{family:?} {ranks}x measured but has no envelope row")).2;
            assert_eq!(row, values.as_slice(), "{family:?} {ranks}x envelope differs from its measured maximum");
        }
        for (family, ranks, _) in ENVELOPES {
            assert!(maxima.iter().any(|(f, r, _)| f == family && r == ranks), "{family:?} {ranks}x envelope has no source");
        }
    });
}

#[test]
fn reachable_resolver_inputs_never_yield_zero_with_graphs_on() {
    use crate::serving_capacity::{hy4_graphs::Hy4GraphLayout, inkling2_mini_graphs::{policy, KeySpace}};
    cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
        for ranks in [1, 2] {
            for rank in 0..ranks {
                // Muse: every onboarding count.
                for onboard in 0..=62 {
                    assert!(crate::serving_capacity::muse_graphs::policy(ranks, onboard, rank, true).unwrap().bytes > 0);
                }
                // Step-5: traced and untraced (MC views, FP8, NVFP4) layouts.
                use crate::serving_capacity::step5_graphs::{policy as step5, Step5GraphLayout::*};
                for layout in [None, Some(Fp8Max), Some(Nvfp4Min), Some(Exl3Min), Some(Exl3Max), Some(Exl3McMin), Some(Exl3McMax)] {
                    assert!(step5(layout, ranks, rank).unwrap().bytes > 0, "Step-5 {layout:?} {ranks}x");
                }
                // V4.1: every lane count and row budget.
                for lanes in [1, 2] {
                    for rows in [1, 64, 96, 128, 192, 256] {
                        assert!(v41_current(ranks, rank, lanes, rows, true).unwrap().bytes > 0);
                    }
                }
                // Hy4: every Spark count and attention placement.
                for sparks in 0..=8 {
                    for attention in [AttentionPlacement::Heads, AttentionPlacement::Context, AttentionPlacement::Layers] {
                        let layout = crate::families::hy4::graph_reserve::Layout::resolve(ranks, sparks, attention);
                        assert!(crate::families::hy4::graph_reserve::reserve_policy(layout, ranks, rank, true).unwrap().bytes > 0);
                    }
                    let layout = Hy4GraphLayout::resolve(ranks, sparks);
                    assert!(crate::serving_capacity::hy4_graphs::reserve_policy(layout, ranks, rank, true).unwrap().bytes > 0,
                        "Hy4-flash {ranks}x + {sparks} Sparks");
                }
                assert!(crate::serving_capacity::inkling2_graphs::policy(rank, ranks, true).unwrap().bytes > 0);
                // V4: every shape, traced or provisional.
                for sparks in [false, true] {
                    assert!(crate::serving_capacity::v4_graphs::v4_reserves(ranks, sparks, true).unwrap()[rank] > 0);
                }
                // Kimi K3: every core view, traced or bootstrap.
                use crate::serving_capacity::kimi_prewarm::KimiCoreFormat::*;
                for core in [Bf16, K6, Trellis, McOriginal, McK6, McK8, McFp8] {
                    let policies = crate::families::kimi_k3::allocation::graph_reserve_policies(core, ranks, true).unwrap();
                    assert!(policies[rank].bytes > 0, "Kimi {core:?} {ranks}x");
                }
                // MiniMax: 1x Heads, 2x Heads and 2x Layers.
                assert!(crate::families::minimax_m3::admission::graph_policy(ranks, 16, rank, AttentionPlacement::Heads, None, true).unwrap().bytes > 0);
                if ranks == 2 {
                    assert!(crate::families::minimax_m3::admission::graph_policy(
                        2, 16, rank, AttentionPlacement::Layers,
                        Some(crate::families::minimax_m3::execution::WirePrecision::F32), true
                    ).unwrap().bytes > 0);
                }
                // MiMo: both checkpoints, with and without Sparks.
                for config in [crate::plan::testing::mimo_flash_mopd_config(), crate::plan::testing::mimo_pro_config()] {
                    let cfg = crate::families::mimo_v2::config::MimoV2Config::from_hf(&config).unwrap();
                    for sparks in [false, true] {
                        let layout = crate::families::mimo_v2::decode_graph::graph_layout(&cfg, ranks, sparks, true).unwrap();
                        assert!(layout.reserves[rank] > 0, "MiMo {ranks}x sparks={sparks}");
                    }
                }
                // GLM 5.3 / ML4, GLM Flash and Qwen4 always resolve a measured working set.
                for family in [crate::serving_capacity::glm_graphs::GlmPrewarmFamily::Glm5,
                    crate::serving_capacity::glm_graphs::GlmPrewarmFamily::MistralLarge4] {
                    assert!(current_working_set(rank, glm_default(family, ranks, rank)).unwrap().bytes > 0);
                }
                for split in [false, true] {
                    assert!(current_working_set(rank, crate::serving_capacity::glmf_graphs::working_set(ranks, split, rank)).unwrap().bytes > 0);
                }
                for sparks in [false, true] {
                    let key = crate::serving_capacity::qwen_graphs::QwenGraphShapeKey::resolve(ranks, sparks);
                    assert!(crate::serving_capacity::qwen_graphs::qwen_resolve_reserve(key, rank).unwrap().bytes > 0);
                }
                // Qwen 3.5: traced variant, other checkpoints/KV, two-rank layouts.
                let suffix = if ranks == 2 { format!("-layers-rtx2-rank{rank}") } else { String::new() };
                for variant in ["h5120-l64-bf16-kvfp8-rtx1", "h5120-l64-fp8-kvfp8-rtx1", "h4096-l32-bf16-kvbf16-rtx1"] {
                    assert!(crate::serving_capacity::qwen35_graphs::reserve(&format!("{variant}{suffix}")).unwrap().bytes > 0);
                }
            }
        }
        for sequences in 0..=128 {
            for rows in [1, 64, 128] {
                for draft in [false, true] {
                    assert!(policy(0, KeySpace::for_config(sequences, rows, draft), true).unwrap().bytes > 0);
                }
            }
        }
    });
}

#[test]
fn six_newly_wired_families_shipping_cards_match_pre_change_reserves_on_and_zero_off() {
    use crate::serving_capacity::{hy4_graphs::Hy4GraphLayout, kimi_prewarm::KimiCoreFormat};
    let matrix: Value = serde_json::from_slice(&std::fs::read(format!("{ROOT}/benchmarks/current-matrix.json")).unwrap()).unwrap();
    let smoke: Value = serde_json::from_slice(&std::fs::read(format!("{ROOT}/scripts/bench/release-smoke-v7.json")).unwrap()).unwrap();
    let targets = ["deepseek_v4", "deepseek_v41", "kimi_k3", "minimax_m3", "hy4", "hy4_flash"];
    let mut matrix_cells = BTreeSet::new();
    for cell in matrix["cells"].as_array().unwrap() {
        let family = cell["family"].as_str().unwrap();
        if !targets.contains(&family) { continue; }
        for (column, ranks) in [("1rtx", 1usize), ("2rtx", 2)] {
            if matches!(cell[column]["status"].as_str(), Some("current" | "pending")) {
                matrix_cells.insert((family.to_string(), ranks));
            }
        }
    }
    let mut smoke_cards = BTreeSet::new();
    for entry in smoke["entries"].as_array().unwrap() {
        let family = entry["family"].as_str().unwrap();
        if !targets.contains(&family) { continue; }
        let ranks = entry["gpus"].as_array().unwrap().len();
        let sparks = entry["sparks"].as_array().unwrap().len();
        smoke_cards.insert((family.to_string(), ranks, sparks));
    }
    for family in targets {
        assert!(matrix_cells.iter().any(|(f, _)| f == family), "missing {family} in current-matrix.json");
        assert!(smoke_cards.iter().any(|(f, _, _)| f == family), "missing {family} in release-smoke-v7.json");
    }

    cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
        // 1. DeepSeek V4: shipping 2x + 6 Sparks (plus 1x/local coverage).
        assert!(smoke_cards.contains(&("deepseek_v4".into(), 2, 6)));
        let v4_list = crate::serving_capacity::v4_graphs::v4_prewarm_list();
        for (gpus, sparks) in [(2, true), (1, true), (2, false)] {
            let layout = crate::serving_capacity::v4_graphs::v4_layout(gpus, sparks).unwrap();
            let entry = &v4_list.layouts[layout.key()];
            let on = crate::serving_capacity::v4_graphs::v4_reserve_policies(gpus, sparks, true).unwrap();
            let off = crate::serving_capacity::v4_graphs::v4_reserve_policies(gpus, sparks, false).unwrap();
            let on_bytes = crate::serving_capacity::v4_graphs::v4_reserves(gpus, sparks, true).unwrap();
            let off_bytes = crate::serving_capacity::v4_graphs::v4_reserves(gpus, sparks, false).unwrap();
            for rank in 0..gpus {
                let expected = resolve(None, rank, false, entry.reserve_base_bytes[rank], entry.reserve_extra_bytes[rank]).unwrap();
                assert_eq!(on[rank], expected, "V4 {gpus}x sparks={sparks} rank {rank}");
                assert_eq!(on_bytes[rank], expected.bytes);
                assert!(on[rank].bytes > 0);
                assert_eq!(off[rank], disabled(rank));
                assert_eq!(off_bytes[rank], 0);
            }
        }

        // 2. DeepSeek V4.1: shipping 1x + 3 Sparks and 2x + 6 Sparks.
        assert!(smoke_cards.contains(&("deepseek_v41".into(), 1, 3)));
        assert!(smoke_cards.contains(&("deepseek_v41".into(), 2, 6)));
        for (ranks, lanes, rows) in [(1, 1, 64), (1, 1, 128), (2, 1, 64), (2, 2, 64), (2, 2, 128)] {
            for rank in 0..ranks {
                let on = v41_current(ranks, rank, lanes, rows, true).unwrap();
                let expected = resolve_working_set(None, None, None, rank, v41_working_set(ranks, rank, lanes, rows).unwrap()).unwrap();
                assert_eq!(on, expected, "V4.1 {ranks}x lanes={lanes} rows={rows} rank {rank}");
                assert!(on.bytes > 0);
                let off = v41_current(ranks, rank, lanes, rows, false).unwrap();
                assert_eq!(off, disabled(rank));
                let owner_on = crate::placement::families::deepseek_v41::admission::graph_allowance(ranks, rank, lanes, rows, true).unwrap();
                let owner_off = crate::placement::families::deepseek_v41::admission::graph_allowance(ranks, rank, lanes, rows, false).unwrap();
                assert_eq!(owner_on.bytes, expected.bytes);
                assert_eq!(owner_off.bytes, 0);
            }
        }

        // 3. Kimi K3: shipping K6 2x + 6 Sparks, K3.1 MC 1x + 6 Sparks and 2x + 6 Sparks.
        assert!(smoke_cards.contains(&("kimi_k3".into(), 1, 6)));
        assert!(smoke_cards.contains(&("kimi_k3".into(), 2, 6)));
        for (core, ranks) in [(KimiCoreFormat::K6, 2), (KimiCoreFormat::McK8, 1), (KimiCoreFormat::McK8, 2), (KimiCoreFormat::McOriginal, 2)] {
            let on = crate::families::kimi_k3::allocation::graph_reserve_policies(core, ranks, true).unwrap();
            let off = crate::families::kimi_k3::allocation::graph_reserve_policies(core, ranks, false).unwrap();
            let demands_off = crate::families::kimi_k3::allocation::graph_reserve_demands(core, ranks, false).unwrap();
            for rank in 0..ranks {
                let (base, tail) = kimi_default(core, ranks, rank);
                let expected = resolve(None, rank, false, base, tail).unwrap();
                assert_eq!(on[rank], expected, "Kimi {core:?} {ranks}x rank {rank}");
                assert!(on[rank].bytes > 0);
                assert_eq!(off[rank], disabled(rank));
                assert_eq!(demands_off[rank].bytes, 0);
            }
        }

        // 4. MiniMax M3.1: shipping 1x + 3 Sparks and 2x + 6 Sparks.
        assert!(smoke_cards.contains(&("minimax_m3".into(), 1, 3)));
        assert!(smoke_cards.contains(&("minimax_m3".into(), 2, 6)));
        for (gpus, attention, wire) in [
            (1, AttentionPlacement::Heads, None),
            (2, AttentionPlacement::Heads, None),
            (2, AttentionPlacement::Layers, Some(crate::families::minimax_m3::execution::WirePrecision::F32)),
        ] {
            for rank in 0..gpus {
                let on = crate::families::minimax_m3::admission::graph_policy(gpus, 16, rank, attention, wire, true).unwrap();
                assert!(on.bytes > 0);
                assert_eq!(on.bytes, envelope_bytes(GraphFamily::MiniMax, gpus, rank));
                let off = crate::families::minimax_m3::admission::graph_policy(gpus, 16, rank, attention, wire, false).unwrap();
                assert_eq!(off, disabled(rank));
            }
        }

        // 5. Hy4: shipping 1x + 4 Sparks and 2x + 6 Sparks.
        assert!(smoke_cards.contains(&("hy4".into(), 1, 4)));
        assert!(smoke_cards.contains(&("hy4".into(), 2, 6)));
        for (gpus, sparks, attention) in [(1, 4, AttentionPlacement::Heads), (2, 6, AttentionPlacement::Context)] {
            let layout = crate::families::hy4::graph_reserve::Layout::resolve(gpus, sparks, attention);
            for rank in 0..gpus {
                let on = crate::families::hy4::graph_reserve::reserve_policy(layout, gpus, rank, true).unwrap();
                assert!(on.bytes > 0);
                assert_eq!(on.bytes, envelope_bytes(GraphFamily::Hy4, gpus, rank));
                let off = crate::families::hy4::graph_reserve::reserve_policy(layout, gpus, rank, false).unwrap();
                assert_eq!(off, disabled(rank));
            }
        }

        // 6. Hy4-flash: shipping 1x+0, 2x+0, 1x+1, 1x+4.
        assert!(smoke_cards.contains(&("hy4_flash".into(), 1, 0)));
        assert!(smoke_cards.contains(&("hy4_flash".into(), 2, 0)));
        for (gpus, sparks, tp2_dfly2) in [(1, 0, false), (2, 0, false), (2, 0, true), (1, 1, false), (1, 4, false)] {
            let layout = Hy4GraphLayout::resolve(gpus, sparks);
            for rank in 0..gpus {
                let on = crate::serving_capacity::hy4_graphs::reserve_policy_for(layout, gpus, rank, tp2_dfly2, true).unwrap();
                assert!(on.bytes > 0, "Hy4-flash {gpus}x+{sparks} tp2_dfly2={tp2_dfly2} rank {rank}");
                let off = crate::serving_capacity::hy4_graphs::reserve_policy_for(layout, gpus, rank, tp2_dfly2, false).unwrap();
                assert_eq!(off, disabled(rank));
            }
        }
    });
}
