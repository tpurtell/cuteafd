//! Chart bodies of the measurement panels. Each reads its panel's newest
//! record (the running pass's partial one while it runs).
use super::bodies::content_color;
use super::charts::{self, hbars, legend, short, Plot, Scale};
use super::svg::{Anchor, Doc, Font};
use super::{rate, seconds, Theme};
use crate::report::PanelResult;
use serde_json::Value;

fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn empty(doc: &mut Doc, t: &Theme, text: &str) -> f64 {
    doc.text(0.0, 16.0, Font::new(12.0, t.ink2), text);
    24.0
}

/// Renders panel `id`'s chart; None for panels without a chart body here.
pub fn body(doc: &mut Doc, t: &Theme, id: &str, panel: &PanelResult, w: f64) -> Option<f64> {
    if id == "tool_eval" {
        return Some(tool_eval(doc, t, panel, w));
    }
    let latest = panel.latest()?;
    Some(match id {
        "decode_content" => decode_content(doc, t, latest, w),
        "concurrency" => concurrency(doc, t, latest, w),
        "prefill" => prefill(doc, t, latest, w),
        "retained" => retained(doc, t, latest, w),
        "prefix_cache" => prefix_cache(doc, t, latest, w),
        "fidelity" | "fidelity_full" => fidelity(doc, t, latest, w),
        "structured" => structured(doc, t, latest, w),
        "needle" => needle(doc, t, latest, w),
        "math" => scored(doc, t, latest, w, "correct", "rows", "question", "correct"),
        "code" => scored(doc, t, latest, w, "passed", "rows", "problem", "passed"),
        "ifeval" => ifeval(doc, t, latest, w),
        "reasoning_effort" => reasoning(doc, t, latest, w),
        "agentic" => agentic(doc, t, latest, w),
        "startup" => startup(doc, t, latest, w),
        _ => return None,
    })
}

fn fidelity(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let mut y = 0.0;
    let dataset = &v["dataset"];
    let config = dataset["config"].as_str().unwrap_or("unpublished");
    let revision = dataset["revision"].as_str().unwrap_or("?");
    let identity = format!("{config} · {}", revision.chars().take(12).collect::<String>());
    doc.titled(&identity);
    doc.text(0.0, 14.0, Font::new(11.0, t.ink2), &super::svg::fit(&identity, 11.0, w));
    doc.end();
    y += 28.0;
    if let Some(how) = v["reference_match"].as_str() {
        let label = format!("Reference via {how}: {}", v["reference_root"].as_str().unwrap_or("?"));
        doc.titled(&label);
        doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), &super::svg::fit(&label, 11.0, w));
        doc.end(); y += 22.0;
        if let Some(chains) = v["resolved_chain"].as_array() {
            for chain in chains {
                let label = chain.as_array().map(|c| c.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" -> ")).unwrap_or_default();
                doc.titled(&label);
                doc.text(0.0, y + 12.0, Font::new(10.0, t.ink2), &super::svg::fit(&label, 10.0, w));
                doc.end(); y += 20.0;
            }
        }
        if let Some(source) = v["text_checkpoint"].as_str() {
            let label = format!("Window text from {source}");
            doc.titled(&label);
            doc.text(0.0, y + 12.0, Font::new(10.0, t.ink2), &super::svg::fit(&label, 10.0, w));
            doc.end(); y += 20.0;
        }
    }
    for (path, verdict) in [("decode", &v["verdict"]), ("prefill", &v["prefill_verdict"])] {
        if verdict.is_null() { continue; }
        let status = if verdict["pass"] == true { "PASS" } else { "FAIL" };
        let score = &verdict["generated"];
        doc.text(0.0, y + 12.0, Font::new(12.0, t.ink).weight(600), &format!("{path} · {status} · {} · n={}",
            seconds(num(&v[path], "seconds").unwrap_or(0.0)), score["positions"]));
        y += 22.0;
        if let Some(label) = verdict["label"].as_str() {
            doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), label);
            y += 22.0;
        }
        // Different units never share an axis. Percent metrics and KL get separate rows.
        for (label, metric, bound, percent) in [("top-1", "top1", "top1_min", true),
            ("KL", "kl", "kl_max", false), ("confident top-1", "confident_top1", "confident_top1_min", true),
            ("top-3", "top3_contained", "top3_min", true)] {
            let val = num(score, metric).map(|x| if percent { format!("{:.2}%", x*100.0) } else { format!("{x:.5} nat") })
                .unwrap_or_else(|| "N/A".into());
            let limit = num(verdict, bound).unwrap_or(0.0);
            doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), label);
            if w < 420.0 { y += 18.0; }
            doc.text(w, y + 12.0, Font::new(11.0, t.ink).anchor(Anchor::End), &format!("{val} · {} {}",
                if metric == "kl" { "max" } else { "min" },
                if percent { format!("{:.1}%", limit*100.0) } else { format!("{limit:.3}") }));
            y += 20.0;
        }
        if let Some(reasons) = verdict["reasons"].as_array() {
            for reason in reasons { doc.text(0.0, y + 11.0, Font::new(10.0, t.ink2), reason.as_str().unwrap_or("?")); y += 18.0; }
        }
        y += 10.0;
    }
    if let Some(error) = v["paired"]["unavailable"].as_str() {
        doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), &format!("Pair unavailable: {}", super::svg::fit(error, 11.0, w - 110.0)));
        y += 26.0;
    } else if let Some(pair) = v["paired"]["comparison"].as_object() {
        doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), &format!("Paired decode: {}",
            if pair.get("pass") == Some(&Value::Bool(true)) { "PASS" } else { "FAIL" }));
        y += 20.0;
        doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), &format!("{} generated rows · {} windows", pair["positions"], pair["windows"]));
        y += 28.0;
        let current = &v["verdict"]["generated"];
        let prior = v["paired"]["earlier_score"].clone();
        if let Ok(prior) = serde_json::from_value::<crate::reference::Fidelity>(prior) {
            // Validated palette; direct labels and 2px gaps distinguish the pair in CVD/print.
            let colors = ["#309dcc", "#ae6299"];
            for (title, a, b, unit) in [("Top-1", num(current, "top1").unwrap_or(0.0)*100.0, prior.top1*100.0, "%"),
                ("KL", num(current, "kl").unwrap_or(0.0), prior.kl, "nat")] {
                doc.text(0.0, y + 12.0, Font::new(11.0, t.ink).weight(600), title);
                y += 20.0;
                y += legend(doc, t, 0.0, y + 8.0, &[("current", colors[0]), ("earlier", colors[1])]);
                let max = a.max(b).max(1e-9) * 1.1;
                let track = (w - 170.0).max(20.0);
                for (label, value, color) in [("current", a, colors[0]), ("earlier", b, colors[1])] {
                    let value_text = if unit == "nat" { format!("{value:.6} nat") } else { format!("{value:.3}%") };
                    doc.titled(&format!("{label}: {value_text}"));
                    doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), label);
                    doc.rect(64.0, y + 3.0, track, 10.0, 3.0, t.well);
                    doc.rect(64.0, y + 3.0, track * (value / max).clamp(0.0, 1.0), 10.0, 3.0, color);
                    doc.text(w, y + 12.0, Font::new(10.0, t.ink).anchor(Anchor::End), &value_text);
                    doc.end(); y += 22.0;
                }
                y += 10.0;
            }
        }
    }
    if let Some(error) = v["paired_prefill"]["unavailable"].as_str() {
        doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), &format!("Prefill pair unavailable: {}", super::svg::fit(error, 11.0, w - 140.0)));
        y += 26.0;
    } else if let Some(pair) = v["paired_prefill"]["comparison"].as_object() {
        doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), &format!("Paired prefill: {} · n={}",
            if pair.get("pass") == Some(&Value::Bool(true)) { "PASS" } else { "FAIL" }, pair["positions"]));
        y += 26.0;
    }
    for (path, key) in [("decode", "per_window"), ("prefill", "prefill_per_window")] {
        if let Some(groups) = v[key].as_object() {
            doc.titled("Per-window top-1 / KL across all rows; table includes counts");
            doc.text(0.0, y + 12.0, Font::new(11.0, t.ink2), &format!("{path} window · top-1 / KL (all rows)"));
            doc.end();
            y += 24.0;
            for (id, score) in groups {
                doc.titled(&format!("{id}: {} rows, top-1 {:.2}%, KL {:.6}", score["positions"],
                    num(score,"top1").unwrap_or(0.0)*100.0, num(score,"kl").unwrap_or(0.0)));
                doc.text(0.0, y + 12.0, Font::new(10.5, t.ink2), id);
                doc.text(w, y + 12.0, Font::new(10.5, t.ink).anchor(Anchor::End), &format!("{:.2}% / {:.5}",
                    num(score,"top1").unwrap_or(0.0)*100.0, num(score,"kl").unwrap_or(0.0)));
                doc.end(); y += 18.0;
            }
        }
    }
    y
}

fn decode_content(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let rows: Vec<(String, f64, &str, String)> = v["rows"].as_array().map(|rows| rows.iter().map(|r| {
        let content = r["content"].as_str().unwrap_or("?").to_string();
        let note = match num(r, "acceptance") {
            Some(a) => format!("{:.0}% accepted", 100.0 * a),
            None => format!("{} tokens", r["tokens"].as_u64().unwrap_or(0)),
        };
        let color = content_color(t, &content);
        (content, num(r, "tok_s").unwrap_or(0.0), color, note)
    }).collect()).unwrap_or_default();
    if rows.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    hbars(doc, t, 0.0, 0.0, w, &rows, "tok/s")
}

fn plot_height(w: f64) -> f64 {
    (w * 0.3).clamp(180.0, 280.0)
}

fn concurrency(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let points = v["points"].as_array().cloned().unwrap_or_default();
    if points.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let max_c = points.iter().filter_map(|p| num(p, "c")).fold(1.0, f64::max).max(2.0);
    let agg: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "c")?, num(p, "aggregate_tok_s")?))).collect();
    let per: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "c")?, num(p, "per_request_tok_s")?))).collect();
    let h = plot_height(w);
    let plot = Plot { x: 46.0, y: 26.0, w: w - 150.0, h, xs: Scale::Log2 { min: 1.0, max: max_c },
        ys: Scale::linear_from(agg.iter().chain(&per).map(|p| p.1), true) };
    legend(doc, t, 46.0, 10.0, &[("aggregate", t.series[0]), ("per request", t.series[1])]);
    plot.axes(doc, t, "concurrent requests", "tok/s", &|x| format!("C{}", x as u64));
    plot.line(doc, &agg, t.series[0], agg.last().map(|p| rate(p.1)).as_deref());
    plot.line(doc, &per, t.series[1], per.last().map(|p| rate(p.1)).as_deref());
    26.0 + h + 36.0
}

fn prefill(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let points = v["points"].as_array().cloned().unwrap_or_default();
    if points.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let cold: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "prompt_tokens")?, num(p, "cold_ttft_s")?))).collect();
    let cached: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "prompt_tokens")?, num(p, "cached_ttft_s")?))).collect();
    let max_x = cold.iter().map(|p| p.0).fold(2048.0, f64::max);
    let ys: Vec<f64> = cold.iter().chain(&cached).map(|p| p.1).collect();
    let (lo, hi) = (ys.iter().cloned().fold(f64::INFINITY, f64::min).max(0.005), ys.iter().cloned().fold(0.01, f64::max));
    let h = plot_height(w);
    let plot = Plot { x: 46.0, y: 26.0, w: w - 130.0, h, xs: Scale::Log2 { min: 1024.0, max: max_x * 1.05 },
        ys: Scale::Log10 { min: 10f64.powf(lo.log10().floor()), max: 10f64.powf(hi.log10().ceil()) } };
    legend(doc, t, 46.0, 10.0, &[("cold", t.series[3]), ("cached prefix (last 256 tokens new)", t.series[1])]);
    plot.axes(doc, t, "prompt tokens", "TTFT s", &|x| charts::tokens(x));
    plot.line(doc, &cold, t.series[3], None);
    plot.line(doc, &cached, t.series[1], None);
    for p in &points {
        if let (Some(x), Some(y), Some(rate_v)) = (num(p, "prompt_tokens"), num(p, "cold_ttft_s"), num(p, "cold_tok_s")) {
            doc.text(plot.px(x), plot.py(y) - 8.0, Font::new(9.5, t.ink2).anchor(Anchor::Middle), &rate(rate_v));
        }
    }
    26.0 + h + 36.0
}

fn retained(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let points: Vec<(f64, f64)> = v["points"].as_array().map(|p| p.iter()
        .filter_map(|p| Some((num(p, "prompt_tokens")?, num(p, "decode_tok_s")?))).collect()).unwrap_or_default();
    if points.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let h = plot_height(w) * 0.85;
    let plot = Plot { x: 46.0, y: 18.0, w: w - 110.0, h,
        xs: Scale::linear_from(points.iter().map(|p| p.0), true), ys: Scale::linear_from(points.iter().map(|p| p.1), true) };
    plot.axes(doc, t, "tokens of context", "decode tok/s", &|x| charts::tokens(x));
    plot.line(doc, &points, t.series[0], None);
    for (x, y) in &points {
        doc.text(plot.px(*x), plot.py(*y) - 9.0, Font::new(10.0, t.ink).anchor(Anchor::Middle), &rate(*y));
    }
    18.0 + h + 36.0
}

fn prefix_cache(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let turns = v["turns"].as_array().cloned().unwrap_or_default();
    if turns.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let max = turns.iter().filter_map(|x| num(x, "prompt_tokens")).fold(1.0, f64::max);
    let h = 150.0;
    let top = 22.0;
    legend(doc, t, 0.0, 10.0, &[("restored from cache", t.series[1]), ("prefilled", t.series[3])]);
    let slot = w / turns.len().max(6) as f64;
    for (i, turn) in turns.iter().enumerate() {
        let prompt = num(turn, "prompt_tokens").unwrap_or(0.0);
        let cached = num(turn, "cached_tokens").unwrap_or(0.0).min(prompt);
        let x = slot * i as f64 + slot * 0.2;
        let bw = slot * 0.6;
        let total_h = h * prompt / max;
        let cached_h = h * cached / max;
        let base = top + h + 8.0;
        doc.rect(x, base - total_h, bw, total_h - cached_h, 2.0, t.series[3]);
        doc.rect(x, base - cached_h, bw, cached_h, 2.0, t.series[1]);
        let mut label = seconds(num(turn, "ttft_s").unwrap_or(f64::NAN));
        let cold = num(turn, "cold_ttft_s");
        doc.titled(&format!("turn {}: {} of {} prompt tokens restored, TTFT {label}{}", turn["turn"], cached, prompt,
            cold.map(|c| format!(", cold {}", seconds(c))).unwrap_or_default()));
        if let Some(cold) = cold.filter(|_| slot >= 110.0) {
            label.push_str(&format!(" / {}", seconds(cold)));
        }
        doc.text(x + bw / 2.0, base - total_h - 6.0, Font::new(10.0, t.ink).anchor(Anchor::Middle), &label);
        doc.text(x + bw / 2.0, base + 14.0, Font::new(10.0, t.muted).anchor(Anchor::Middle),
            &if slot >= 90.0 { format!("turn {} · {:.0}%", turn["turn"], 100.0 * cached / prompt.max(1.0)) }
                else { format!("{:.0}%", 100.0 * cached / prompt.max(1.0)) });
        doc.end();
    }
    if w >= 560.0 {
        doc.text(w, 10.0, Font::new(10.0, t.muted).anchor(Anchor::End), "TTFT / cold TTFT above each turn");
    }
    top + h + 30.0
}


fn structured(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let rows: Vec<(String, f64, &str, String)> = v["rows"].as_array().map(|rows| rows.iter().map(|r| {
        let valid = r["valid"] == true;
        let note = format!("{} · free {}", if valid { "valid ✓" } else { "INVALID ✗" },
            rate(num(r, "free_tok_s").unwrap_or(0.0)));
        (r["schema"].as_str().unwrap_or("?").to_string(), num(r, "tok_s").unwrap_or(0.0),
            if valid { t.series[0] } else { t.bad }, note)
    }).collect()).unwrap_or_default();
    if rows.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    hbars(doc, t, 0.0, 0.0, w, &rows, "tok/s")
}

fn needle(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let cells = v["cells"].as_array().cloned().unwrap_or_default();
    if cells.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let mut lengths: Vec<u64> = cells.iter().filter_map(|c| c["length"].as_u64()).collect();
    lengths.dedup();
    let mut depths: Vec<f64> = cells.iter().filter_map(|c| c["depth"].as_f64()).collect();
    depths.sort_by(f64::total_cmp);
    depths.dedup();
    let xs: Vec<String> = lengths.iter().map(|l| charts::tokens(*l as f64)).collect();
    let ys: Vec<String> = depths.iter().map(|d| format!("{:.0}% deep", 100.0 * d)).collect();
    let cell = |i: usize, j: usize| cells.iter().find(|c| c["length"].as_u64() == Some(lengths[i])
        && c["depth"].as_f64() == Some(depths[j])).map(|c| if c["found"] == true { 1.0 } else { 0.0 });
    let found = cells.iter().filter(|c| c["found"] == true).count();
    doc.text(0.0, 12.0, Font::new(11.0, t.ink2), &format!("{found} of {} found · tokens of context →", cells.len()));
    18.0 + charts::heatmap(doc, t, 0.0, 18.0, w, &xs, &ys, &cell, t.good, t.bad)
}

/// A score headline and one named square per item (green pass, red fail);
/// the full prompt is the item's tooltip.
#[allow(clippy::too_many_arguments)]
fn scored(doc: &mut Doc, t: &Theme, v: &Value, w: f64, count: &str, list: &str, label: &str, ok: &str) -> f64 {
    let items = v[list].as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let good = num(v, count).unwrap_or_else(|| items.iter().filter(|i| i[ok] == true).count() as f64);
    let total = items.len() as f64;
    doc.text(0.0, 34.0, Font::new(32.0, t.series[0]).bold(), &format!("{good:.0}/{total:.0}"));
    doc.text(0.0, 54.0, Font::new(11.0, t.ink2), &format!("{:.0}%", 100.0 * good / total.max(1.0)));
    if let Some(sandbox) = v["sandbox"].as_str() {
        doc.text(60.0, 54.0, Font::new(10.0, t.muted), &super::svg::fit(sandbox, 10.0, w - 60.0));
    }
    let (size, gap, cell) = (18.0, 6.0, 136.0);
    let (x0, y0) = if w >= 520.0 { (150.0, 0.0) } else { (0.0, 70.0) };
    let per_row = ((w - x0 + 6.0) / cell).floor().max(1.0) as usize;
    for (i, item) in items.iter().enumerate() {
        let x = x0 + (i % per_row) as f64 * cell;
        let y = y0 + (i / per_row) as f64 * (size + gap);
        let pass = item[ok] == true;
        let full = item[label].as_str().or_else(|| item["task"].as_str()).unwrap_or("");
        let name = item["name"].as_str().map(str::to_string).unwrap_or_else(|| crate::panels::names::short_name(full));
        doc.titled(&format!("{} {name}: {full}", if pass { "✓" } else { "✗" }));
        doc.rect(x, y, size, size, 4.0, if pass { t.good } else { t.bad });
        doc.text(x + size / 2.0, y + 13.5, Font::new(11.0, t.bg).anchor(Anchor::Middle).bold(), if pass { "✓" } else { "✗" });
        doc.text(x + size + 6.0, y + 13.5, Font::new(10.5, t.ink2), &super::svg::fit(&name, 10.5, cell - size - 10.0));
        doc.end();
    }
    let rows = items.len().div_ceil(per_row) as f64;
    (y0 + rows * (size + gap)).max(64.0)
}

fn ifeval(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let rows = v["rows"].as_array().cloned().unwrap_or_default();
    if rows.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let prompt = num(v, "prompt_accuracy").unwrap_or(0.0);
    let instruction = num(v, "instruction_accuracy").unwrap_or(0.0);
    doc.text(0.0, 34.0, Font::new(32.0, t.series[0]).bold(), &format!("{:.0}%", 100.0 * prompt));
    doc.text(0.0, 54.0, Font::new(10.5, t.ink2), "prompt-level");
    doc.text(130.0, 34.0, Font::new(32.0, t.series[1]).bold(), &format!("{:.0}%", 100.0 * instruction));
    doc.text(130.0, 54.0, Font::new(10.5, t.ink2), "instruction-level");
    let (size, gap) = (18.0, 5.0);
    let (x0, y0) = if w >= 560.0 { (280.0, 6.0) } else { (0.0, 70.0) };
    let per_row = ((w - x0 + gap) / (size + gap)).floor().max(1.0) as usize;
    for (i, r) in rows.iter().enumerate() {
        let x = x0 + (i % per_row) as f64 * (size + gap);
        let y = y0 + (i / per_row) as f64 * (size + gap);
        let prompt = r["prompt"].as_str().unwrap_or("");
        let name = r["name"].as_str().map(str::to_string).unwrap_or_else(|| crate::panels::names::short_name(prompt));
        let rules: Vec<String> = r["rules"].as_array().into_iter().flatten().map(|x| format!("{} {}",
            if x["ok"] == true { "✓" } else { "✗" }, x["rule"].as_str().unwrap_or(""))).collect();
        doc.titled(&format!("{name}: {prompt} [{}]", rules.join(", ")));
        doc.rect(x, y, size, size, 3.0, if r["passed"] == true { t.good } else { t.bad });
        doc.end();
    }
    (y0 + rows.len().div_ceil(per_row) as f64 * (size + gap)).max(64.0)
}

/// Wilson 68% interval of k successes in n.
fn wilson(k: f64, n: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 1.0);
    }
    let z = 1.0;
    let p = k / n;
    let center = (p + z * z / (2.0 * n)) / (1.0 + z * z / n);
    let half = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()) / (1.0 + z * z / n);
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn reasoning(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let rows = v["rows"].as_array().cloned().unwrap_or_default();
    let levels: Vec<String> = v["levels"].as_array().map(|l| l.iter().filter_map(|x| x.as_str().map(str::to_string))
        .collect()).unwrap_or_default();
    if rows.is_empty() || levels.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let mut y = 0.0;
    // 1. Effort ladder: mean reasoning tokens (log) against accuracy, with intervals.
    let stats: Vec<(String, f64, f64, f64, f64)> = levels.iter().filter_map(|l| {
        let n = of(&rows, l).count() as f64;
        if n == 0.0 {
            return None;
        }
        let k = of(&rows, l).filter(|r| r["correct"] == true).count() as f64;
        let tokens = of(&rows, l).filter_map(|r| num(r, "reasoning_tokens")).sum::<f64>() / n;
        let (lo, hi) = wilson(k, n);
        Some((l.clone(), tokens.max(1.0), k / n, lo, hi))
    }).collect();
    doc.text(0.0, 12.0, Font::new(10.5, t.ink2).weight(600).spacing(1.2), "EFFORT LADDER");
    let max_tokens = stats.iter().map(|s| s.1).fold(10.0, f64::max);
    let h = 170.0;
    let plot = Plot { x: 46.0, y: 30.0, w: w * 0.55 - 60.0, h, xs: Scale::Log10 { min: 1.0, max: 10f64.powf(max_tokens.log10().ceil()) },
        ys: Scale::Linear { min: 0.0, max: 1.0 } };
    plot.axes(doc, t, "mean reasoning tokens", "accuracy", &|x| short(x));
    let line: Vec<(f64, f64)> = stats.iter().map(|s| (s.1, s.2)).collect();
    plot.line(doc, &line, t.series[0], None);
    for (i, (label, tokens, acc, lo, hi)) in stats.iter().enumerate() {
        let color = t.series[i % t.series.len()];
        plot.range(doc, *tokens, *lo, *acc, *hi, color);
        doc.text(plot.px(*tokens) + 7.0, plot.py(*acc) - 6.0, Font::new(10.0, color).weight(600), label);
    }
    // 2. Accuracy by tier per level (right of the ladder).
    let tiers = ["medium", "hard", "brutal", "aime"];
    let hx = w * 0.55 + 10.0;
    doc.text(hx, 12.0, Font::new(10.5, t.ink2).weight(600).spacing(1.2), "ACCURACY BY TIER");
    let cell = |i: usize, j: usize| {
        let cells: Vec<&Value> = of(&rows, &levels[j]).filter(|r| r["tier"].as_str() == Some(tiers[i])).collect();
        (!cells.is_empty()).then(|| cells.iter().filter(|r| r["correct"] == true).count() as f64 / cells.len() as f64)
    };
    let xs: Vec<String> = tiers.iter().map(|s| s.to_string()).collect();
    charts::heatmap(doc, t, hx, 26.0, w - hx, &xs, &levels, &cell, t.good, t.bad);
    y += 30.0 + h + 40.0;
    // 3. Time to solve: solo-equivalent seconds, solved and failed, per level.
    doc.text(0.0, y + 12.0, Font::new(10.5, t.ink2).weight(600).spacing(1.2),
        "TIME TO SOLVE · solo-equivalent (reasoning tokens ÷ C1 decode) · solved vs failed");
    let times: Vec<f64> = rows.iter().filter_map(|r| num(r, "solo_s")).filter(|v| *v > 0.0).collect();
    let (lo, hi) = (times.iter().cloned().fold(f64::INFINITY, f64::min).max(0.05), times.iter().cloned().fold(1.0, f64::max));
    let vh = 160.0;
    let vplot = Plot { x: 46.0, y: y + 28.0, w: w - 60.0, h: vh,
        xs: Scale::Linear { min: -0.5, max: levels.len() as f64 - 0.5 },
        ys: Scale::Log10 { min: 10f64.powf(lo.log10().floor()), max: 10f64.powf(hi.log10().ceil()) } };
    for tick in vplot.ys.ticks() {
        let py = vplot.py(tick);
        doc.line(vplot.x, py, vplot.x + vplot.w, py, t.line, 1.0);
        doc.text(vplot.x - 6.0, py + 3.5, Font::new(10.0, t.muted).anchor(Anchor::End), &format!("{}s", short(tick)));
    }
    let slot = vplot.w / levels.len() as f64;
    for (i, level) in levels.iter().enumerate() {
        let solved: Vec<f64> = of(&rows, level).filter(|r| r["correct"] == true).filter_map(|r| num(r, "solo_s")).collect();
        let failed: Vec<f64> = of(&rows, level).filter(|r| r["correct"] != true).filter_map(|r| num(r, "solo_s")).collect();
        charts::violin(doc, &vplot, i as f64 - 0.18, slot * 0.17, &solved, t.good);
        charts::violin(doc, &vplot, i as f64 + 0.18, slot * 0.17, &failed, t.bad);
        let caps = of(&rows, level).filter(|r| r["hit_cap"] == true).count();
        doc.text(vplot.px(i as f64), vplot.y + vh + 16.0, Font::new(10.5, t.ink2).anchor(Anchor::Middle),
            &format!("{level}{}", if caps > 0 { format!(" · {caps} cap") } else { String::new() }));
    }
    y += 28.0 + vh + 30.0;
    // 4. Overthinking: reasoning tokens on the easy (medium) tier.
    doc.text(0.0, y + 12.0, Font::new(10.5, t.ink2).weight(600).spacing(1.2), "OVERTHINKING · reasoning tokens on easy items");
    let bars: Vec<(String, f64, &str, String)> = levels.iter().enumerate().map(|(i, level)| {
        let easy: Vec<f64> = of(&rows, level).filter(|r| r["tier"].as_str() == Some("medium")).filter_map(|r| num(r, "reasoning_tokens")).collect();
        let mean = if easy.is_empty() { 0.0 } else { easy.iter().sum::<f64>() / easy.len() as f64 };
        let correct = of(&rows, level).filter(|r| r["tier"].as_str() == Some("medium") && r["correct"] == true).count();
        (level.clone(), mean, t.series[i % t.series.len()], format!("{correct}/{} right", easy.len()))
    }).collect();
    y += 22.0 + hbars(doc, t, 0.0, y + 22.0, w, &bars, "tokens");
    y + 6.0
}

fn of<'a>(rows: &'a [Value], level: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
    rows.iter().filter(move |r| r["level"].as_str() == Some(level))
}

/// Standard, Hard and Total points: the mean over every run on this
/// configuration with its spread, each run a dot.
fn tool_eval(doc: &mut Doc, t: &Theme, panel: &PanelResult, w: f64) -> f64 {
    let runs: Vec<&Value> = panel.history.iter().chain(panel.passes.iter()).collect();
    let Some(last) = panel.passes.last() else { return empty(doc, t, "no run yet") };
    let groups = [("STANDARD", "standard", num(last, "standard_max").unwrap_or(138.0)),
        ("HARD", "hard", num(last, "hard_max").unwrap_or(30.0)), ("TOTAL", "total", num(last, "total_max").unwrap_or(168.0))];
    let col = w / 3.0;
    for (i, (title, key, max)) in groups.iter().enumerate() {
        let x = col * i as f64;
        let values: Vec<f64> = runs.iter().filter_map(|r| num(r, key)).collect();
        let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
        let color = t.series[i];
        doc.text(x, 12.0, Font::new(10.0, t.muted).spacing(1.4), title);
        let text = format!("{mean:.1} /{max:.0}");
        let size = (col - 20.0) / (0.6 * text.chars().count() as f64);
        doc.spans(x, 52.0, size.min(38.0), Anchor::Start, &[(&format!("{mean:.1}"), color, 700),
            (&format!(" /{max:.0}"), t.ink2, 400)]);
        let track = col - 30.0;
        doc.rect(x, 64.0, track, 10.0, 5.0, t.well);
        if let Some((lo, _, hi)) = charts::spread(&values) {
            doc.rect_attrs(x + track * lo / max, 64.0, (track * (hi - lo) / max).max(3.0), 10.0, 5.0,
                &format!(r#"fill="{color}" opacity="0.35""#));
        }
        for v in &values {
            doc.circle(x + track * v / max, 69.0, 3.0, color);
        }
        doc.line(x + track * mean / max, 60.0, x + track * mean / max, 78.0, t.ink, 2.0);
        doc.text(x, 94.0, Font::new(10.5, t.ink2), &format!("{:.0}% · {} run{}", 100.0 * mean / max, values.len(),
            if values.len() == 1 { "" } else { "s" }));
    }
    104.0
}

fn agentic(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let turns = v["turns"].as_array().cloned().unwrap_or_default();
    let success = v.get("success").and_then(Value::as_bool);
    let (label, color) = match success {
        Some(true) => ("✓ tests pass", t.good),
        Some(false) => ("✗ tests fail", t.bad),
        None => ("running", t.series[0]),
    };
    let reused = turns.iter().filter(|x| x["full_turn_reused"] == true).count();
    let invalid: u64 = turns.iter().filter_map(|x| x["invalid_tool_calls"].as_u64()).sum();
    let rates: Vec<f64> = turns.iter().filter_map(|x| num(x, "decode_tok_s")).filter(|v| *v > 0.0).collect();
    let mean_rate = if rates.is_empty() { 0.0 } else { rates.iter().sum::<f64>() / rates.len() as f64 };
    doc.spans(0.0, 14.0, 12.0, Anchor::Start, &[(&format!("task {} · ", v["task"].as_str().unwrap_or("?")), t.ink2, 400),
        (label, color, 700), (&format!(" · {} turns · {} reused whole · {} invalid tool calls · {} tok/s mean{}",
            turns.len(), reused, invalid, rate(mean_rate),
            num(v, "seconds").map(|s| format!(" · {}", seconds(s))).unwrap_or_default()), t.ink2, 400)]);
    doc.group(0.0, 26.0);
    let h = prefix_cache(doc, t, v, w);
    doc.end();
    26.0 + h
}

fn startup(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let phases: Vec<(String, f64)> = v["phases"].as_array().map(|p| p.iter().filter_map(|x|
        Some((x["name"].as_str()?.to_string(), num(x, "at_s")?))).collect()).unwrap_or_default();
    let ready = num(v, "readiness_s").or_else(|| phases.last().map(|p| p.1)).unwrap_or(0.0);
    let warmup = num(v, "warmup_s").unwrap_or(0.0);
    let end = (ready + warmup).max(1.0);
    let (x0, bar_w) = (210.0, w - 290.0);
    let px = |s: f64| x0 + bar_w * s / end;
    let mut segments: Vec<(String, f64, f64, &str)> = Vec::new();
    let mut last = 0.0;
    for (i, (name, at)) in phases.iter().enumerate() {
        let label = if i == 0 { format!("load → {name}") } else { name.clone() };
        segments.push((label, last, *at, t.series[i % 2 * 3]));
        last = *at;
    }
    if warmup > 0.0 {
        segments.push(("first requests (warm-up)".into(), ready, ready + warmup, t.series[1]));
    }
    for (i, (name, a, b, color)) in segments.iter().enumerate() {
        let y = 6.0 + 26.0 * i as f64;
        doc.text(0.0, y + 13.0, Font::new(11.0, t.ink2), name);
        doc.rect(px(*a), y + 2.0, (px(*b) - px(*a)).max(2.0), 16.0, 3.0, color);
        doc.text(px(*b) + 6.0, y + 14.0, Font::new(10.5, t.ink), &seconds(b - a));
    }
    let y = 6.0 + 26.0 * segments.len() as f64 + 4.0;
    doc.line(x0, y, x0 + bar_w, y, t.line2, 1.0);
    doc.text(x0, y + 14.0, Font::new(10.0, t.muted), "process start");
    doc.text(x0 + bar_w, y + 14.0, Font::new(10.0, t.muted).anchor(Anchor::End), &format!("{} after start", seconds(end)));
    y + 20.0
}
