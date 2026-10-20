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
        "draft_mix" => draft_mix(doc, t, latest, w),
        "prefill" => prefill(doc, t, latest, w),
        "retained" => retained(doc, t, latest, w),
        "prefix_cache" => prefix_cache(doc, t, latest, w),
        "context_load" => context_load(doc, t, latest, w),
        "fidelity" | "fidelity_full" => fidelity(doc, t, latest, w),
        "structured" => structured(doc, t, latest, w),
        "needle" => needle(doc, t, latest, w),
        "math" => scored(doc, t, latest, w, "correct", "rows", "question", "correct"),
        "code" => scored(doc, t, latest, w, "passed", "rows", "problem", "passed"),
        "spec_lossless_set" => scored(doc, t, latest, w, "passed", "rows", "reason", "pass"),
        "ifeval" => ifeval(doc, t, latest, w),
        "reasoning_effort" => reasoning(doc, t, latest, w),
        "agentic" => agentic(doc, t, latest, w),
        "startup" => startup(doc, t, latest, w),
        _ => return None,
    })
}

// Multiple incomparable units belong in a table, not on competing axes.
fn context_load(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let points = v["points"].as_array().cloned().unwrap_or_default();
    if points.is_empty() { return empty(doc, t, "no measurements yet"); }
    // Aggregate tps per measured (context, C), one bar each; the table below has the rest.
    let bars: Vec<(String, f64, &str, String)> = points.iter().filter(|p| p["status"] == "ok").filter_map(|p| {
        let label = format!("{} · C{}", crate::report::short_tokens(p["context"].as_u64()?), p["c"].as_u64()?);
        let note = num(p, "ttft_s").map(|s| format!("TTFT {:.0} ms", s * 1e3)).unwrap_or_default();
        Some((label, num(p, "aggregate_tok_s")?, t.series[0], note))
    }).collect();
    let mut y = 0.0;
    if !bars.is_empty() {
        doc.text(0.0, 12.0, Font::new(10.5, t.muted).weight(600), "AGGREGATE DECODE AT SHARED CONTEXT");
        y = 20.0 + hbars(doc, t, 0.0, 20.0, w, &bars, "tps") + 14.0;
    }
    let columns = ["context", "C", "agg tps", "med tps", "TTFT ms", "cached", "cold ms"];
    let widths = [0.16, 0.06, 0.16, 0.15, 0.15, 0.16, 0.16];
    y += 16.0;
    let mut x = 0.0;
    for (name, fraction) in columns.iter().zip(widths) {
        doc.text(x, y, Font::new(10.0, t.ink2).weight(600), name);
        x += w * fraction;
    }
    y += 10.0;
    doc.line(0.0, y, w, y, t.line, 1.0);
    y += 22.0;
    for point in points {
        let values = [
            point["context"].as_u64().map(crate::report::short_tokens).unwrap_or_else(|| "-".into()),
            point["c"].as_u64().map(|n| n.to_string()).unwrap_or_else(|| "-".into()),
            num(&point, "aggregate_tok_s").map(rate).unwrap_or_else(|| "-".into()),
            num(&point, "per_request_tok_s").map(rate).unwrap_or_else(|| "-".into()),
            num(&point, "ttft_s").map(|n| format!("{:.1}", n * 1e3)).unwrap_or_else(|| "-".into()),
            num(&point, "cached_tokens").map(|n| format!("{n:.0}")).unwrap_or_else(|| "-".into()),
            num(&point, "cold_ttft_s").map(|n| format!("{:.1}", n * 1e3)).unwrap_or_else(|| "-".into()),
        ];
        x = 0.0;
        for (value, fraction) in values.iter().zip(widths) {
            doc.text(x, y, Font::new(11.0, t.ink), &super::svg::fit(value, 11.0, w * fraction - 6.0));
            x += w * fraction;
        }
        y += 20.0;
        let status = point["status"].as_str().unwrap_or("unknown");
        let detail = if status == "ok" {
            format!("{status}: {}", point["restore_pages"].as_str().unwrap_or("unknown page sharing"))
        } else {
            format!("{status}: {}", point["reason"].as_str().unwrap_or("cache reuse or request failure; see JSON"))
        };
        doc.titled(&detail);
        doc.text(0.0, y, Font::new(10.0, t.ink2), &super::svg::fit(&detail, 10.0, w));
        doc.end();
        y += 24.0;
    }
    y
}

fn fidelity(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let mut y = 0.0;
    let dataset = &v["dataset"];
    let config = dataset["config"].as_str().unwrap_or("unpublished");
    let revision = dataset["revision"].as_str().unwrap_or("?");
    // Where the numbers come from, said once and plainly: whose logits are
    // the reference, and whose generated text the windows replay.
    let scored = v["decode"]["checkpoint"].as_str().or_else(|| v["prefill"]["checkpoint"].as_str());
    let short = |id: &str| id.rsplit('/').next().unwrap_or(id).to_string();
    let reference = v["reference_root"].as_str();
    let text = v["text_checkpoint"].as_str().or_else(|| dataset["text_checkpoint"].as_str());
    let reference_only = dataset["qualification"] == "reference-only qualified";
    let mut sources = Vec::new();
    if let Some(reference) = reference {
        sources.push(format!("Reference logits: {}{}", short(reference), if reference_only { " (reference-only)" } else { "" }));
    }
    if let Some(text) = text {
        let shared = scored.is_some_and(|s| s != text);
        sources.push(format!("prompt text: {}", if shared { format!("shared {} set", short(text)) } else { short(text) }));
    }
    let label = if sources.is_empty() { format!("{config} · {}", revision.chars().take(12).collect::<String>()) } else { sources.join(" · ") };
    doc.titled(&format!("dataset {config} @ {revision}{}", reference.map(|r| format!(" · reference {r} via {}",
        v["reference_match"].as_str().unwrap_or("?"))).unwrap_or_default()));
    doc.text(0.0, 14.0, Font::new(11.0, t.ink2), &super::svg::fit(&label, 11.0, w));
    doc.end();
    y += 22.0;
    let mut notes = Vec::new();
    if let Some(precision) = v["decode"]["precision"].as_object().or_else(|| v["prefill"]["precision"].as_object())
        .and_then(|p| serde_json::from_value::<crate::fidelity_precision::Precision>(Value::Object(p.clone())).ok()) {
        notes.push(precision.label());
    }
    if dataset["qualification"] == "reference" {
        notes.push("reference".to_string());
    }
    if reference_only {
        notes.push("reference-only qualified: reference logits without paired engine calibration, so the verdict uses the common floors".to_string());
    }
    if let (Some(text), Some(scored)) = (text, scored) {
        if text != scored {
            notes.push(format!("provisional: window text was generated by {}, not by the scored checkpoint", short(text)));
        }
    }
    for (i, note) in notes.iter().enumerate() {
        let color = if note.starts_with("provisional") { t.warn } else { t.muted };
        doc.titled(note);
        doc.text(0.0, y + 11.0, Font::new(10.0, color), &super::svg::fit(note, 10.0, w));
        doc.end();
        y += if i + 1 == notes.len() { 22.0 } else { 16.0 };
    }
    if notes.is_empty() { y += 6.0; }
    for (path, verdict, windows) in [("decode", &v["verdict"], &v["per_window"]), ("prefill", &v["prefill_verdict"], &v["prefill_per_window"])] {
        if verdict.is_null() { continue; }
        y += fidelity_path(doc, t, 0.0, y, w, path, verdict, windows, num(&v[path], "seconds"));
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
    y
}

/// One scoring path: a headline (PASS/FAIL and the four metrics against their
/// floors), then per-window KL bars with top-1 as dots on its own scale, each
/// with its floor as a dashed reference line.
#[allow(clippy::too_many_arguments)]
fn fidelity_path(doc: &mut Doc, t: &Theme, x: f64, y0: f64, w: f64, path: &str, verdict: &Value, windows: &Value,
    seconds_taken: Option<f64>) -> f64 {
    let mut y = y0;
    let pass = verdict["pass"] == true;
    let score = &verdict["generated"];
    let (status, color) = if pass { ("PASS", t.good) } else { ("FAIL", t.bad) };
    doc.spans(x, y + 13.0, 12.5, Anchor::Start, &[(&format!("{path} "), t.ink, 600), (status, color, 700),
        (&format!("  {} generated rows{}", score["positions"], seconds_taken.map(|s| format!(" · {}", seconds(s))).unwrap_or_default()), t.ink2, 400)]);
    y += 22.0;
    // Headline: one cell per metric, measured over its bound.
    let metrics = [("KL", "kl", "kl_max", false), ("top-1", "top1", "top1_min", true),
        ("confident top-1", "confident_top1", "confident_top1_min", true), ("top-3", "top3_contained", "top3_min", true)];
    let columns = if w < 520.0 { 2 } else { 4 };
    let cell = w / columns as f64;
    for (i, (label, metric, bound, percent)) in metrics.iter().enumerate() {
        let cx = x + cell * (i % columns) as f64;
        let cy = y + 40.0 * (i / columns) as f64;
        let value = num(score, metric);
        let limit = num(verdict, bound);
        let ok = match (value, limit) { (Some(v), Some(l)) => if *metric == "kl" { v <= l } else { v + 1e-12 >= l }, _ => true };
        let show = |v: f64| if *percent { format!("{:.2}%", v * 100.0) } else { format!("{v:.4}") };
        doc.text(cx, cy + 10.0, Font::new(10.0, t.muted), label);
        doc.spans(cx, cy + 28.0, 15.0, Anchor::Start, &[(&value.map_or("N/A".into(), show), if ok { t.ink } else { t.bad }, 600),
            (&limit.map(|l| format!("  {} {}", if *metric == "kl" { "≤" } else { "≥" }, show(l))).unwrap_or_default(), t.muted, 400)]);
    }
    y += 40.0 * (metrics.len().div_ceil(columns)) as f64 + 4.0;
    if let Some(reasons) = verdict["reasons"].as_array().filter(|r| !r.is_empty()) {
        let line = reasons.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("; ");
        doc.text(x, y + 11.0, Font::new(10.0, t.bad), &super::svg::fit(&line, 10.0, w));
        y += 18.0;
    }
    let Some(windows) = windows.as_object().filter(|w| !w.is_empty()) else { return y - y0 + 10.0 };
    // Per-window: two aligned strips on one window axis (never two y-scales):
    // KL bars above, top-1 dots below, each with its floor dashed.
    let kl_max = num(verdict, "kl_max").unwrap_or(0.06);
    let top1_min = num(verdict, "top1_min").unwrap_or(0.9);
    let kls: Vec<f64> = windows.values().map(|s| num(s, "kl").unwrap_or(0.0)).collect();
    let tops: Vec<f64> = windows.values().map(|s| num(s, "top1").unwrap_or(0.0)).collect();
    let kl_top = kls.iter().cloned().fold(kl_max * 1.25, f64::max) * 1.08;
    let top_lo = ((tops.iter().cloned().fold(top1_min, f64::min) - 0.02) * 50.0).floor() / 50.0;
    let (px, pw) = (x + 44.0, w - 44.0);
    let n = windows.len() as f64;
    let step = pw / n;
    let bw = (step * 0.7).clamp(1.5, 18.0);
    let strip = |doc: &mut Doc, y: f64, h: f64, title: &str, ticks: [(f64, String); 2]| {
        doc.text(x, y - 10.0, Font::new(10.0, t.muted).weight(600), title);
        for (frac, label) in ticks {
            let gy = y + h - frac * h;
            doc.line(px, gy, px + pw, gy, t.line, 1.0);
            doc.text(px - 6.0, gy + 3.5, Font::new(9.5, t.muted).anchor(Anchor::End), &label);
        }
    };
    let (kh, th) = (72.0, 56.0);
    let ktop = y + 26.0;
    let ttop = ktop + kh + 36.0;
    strip(doc, ktop, kh, &format!("{} · KL PER WINDOW (nat)", path.to_uppercase()), [(0.0, "0".into()), (1.0, format!("{kl_top:.2}"))]);
    strip(doc, ttop, th, "TOP-1 PER WINDOW", [(0.0, format!("{:.0}%", top_lo * 100.0)), (1.0, "100%".into())]);
    let ky = |v: f64| ktop + kh - (v / kl_top).clamp(0.0, 1.0) * kh;
    let ty = |v: f64| ttop + th - ((v - top_lo) / (1.0 - top_lo)).clamp(0.0, 1.0) * th;
    for (i, (id, s)) in windows.iter().enumerate() {
        let cx = px + step * (i as f64 + 0.5);
        let (kl, top1) = (num(s, "kl").unwrap_or(0.0), num(s, "top1").unwrap_or(0.0));
        doc.titled(&format!("{path} window {id}: KL {kl:.5} nat · top-1 {:.2}% · {} rows", top1 * 100.0, s["positions"]));
        doc.rect(cx - bw / 2.0, ky(kl), bw, (ktop + kh - ky(kl)).max(1.0), (bw / 4.0).min(2.0), if kl > kl_max { t.bad } else { t.series[0] });
        doc.circle(cx, ty(top1), 3.2, if top1 + 1e-12 < top1_min { t.bad } else { t.series[1] });
        doc.rect_attrs(cx - step / 2.0, ktop, step, ttop + th - ktop, 0.0, r#"fill="transparent""#);
        doc.end();
    }
    for (gy, label, anchor_x, anchor) in [(ky(kl_max), format!("floor {kl_max:.3}"), px + pw, Anchor::End),
        (ty(top1_min), format!("floor {:.0}%", top1_min * 100.0), px + pw, Anchor::End)] {
        doc.raw(&format!(r#"<line x1="{px}" x2="{}" y1="{gy}" y2="{gy}" stroke="{}" stroke-width="1" stroke-dasharray="4 3"/>"#, px + pw, t.ink2));
        doc.text(anchor_x, gy - 4.0, Font::new(9.5, t.ink2).anchor(anchor), &label);
    }
    let note = if w < 480.0 { format!("{} windows · red: past its floor", windows.len()) }
        else { format!("{} windows · red: past its floor · hover a window for values", windows.len()) };
    doc.text(px, ttop + th + 14.0, Font::new(9.5, t.muted), &super::svg::fit(&note, 9.5, pw));
    ttop + th + 24.0 - y0
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
    hbars(doc, t, 0.0, 0.0, w, &rows, "tps")
}

/// Two groups of bars: the C16 mixed waves (aggregate, with the per-request
/// median as a tick) and the C1 rewrite passes (tokens/round and acceptance).
fn draft_mix(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let waves = v["waves"].as_array().cloned().unwrap_or_default();
    let rewrites = v["rewrite"].as_array().cloned().unwrap_or_default();
    if waves.is_empty() && rewrites.is_empty() { return empty(doc, t, "no measurements yet"); }
    let mut y = 0.0;
    if !waves.is_empty() {
        doc.text(0.0, y + 12.0, Font::new(10.5, t.muted).weight(600), &format!("C{} MIXED PROMPTS · AGGREGATE", waves[0]["c"].as_u64().unwrap_or(16)));
        y += 20.0;
        let rows: Vec<(String, f64, &str, String)> = waves.iter().enumerate().map(|(i, wave)| {
            let median = num(wave, "per_request_tok_s");
            (format!("wave {}", i + 1), num(wave, "aggregate_tok_s").unwrap_or(0.0), t.series[0],
                median.map(|m| format!("{} per request", rate(m))).unwrap_or_default())
        }).collect();
        let top = y;
        y += hbars(doc, t, 0.0, y, w, &rows, "tps");
        // Per-request median as a tick on the same scale as the bars.
        let max = rows.iter().map(|r| r.1).fold(1e-9f64, f64::max) * 1.1;
        let label_w = (w * 0.26).min(120.0);
        let track = w - label_w - charts::hbar_value_width(w, &rows, "tps");
        for (i, wave) in waves.iter().enumerate() {
            if let Some(median) = num(wave, "per_request_tok_s") {
                let x = label_w + track * (median / max).clamp(0.0, 1.0);
                doc.line(x, top + 24.0 * i as f64 + 1.0, x, top + 24.0 * i as f64 + 19.0, t.series[1], 2.0);
            }
        }
        legend(doc, t, 0.0, y + 10.0, &[("aggregate", t.series[0]), ("per-request median", t.series[1])]);
        y += 26.0;
    }
    if !rewrites.is_empty() {
        doc.text(0.0, y + 12.0, Font::new(10.5, t.muted).weight(600), "C1 COPY-HEAVY REWRITE");
        y += 20.0;
        let rows: Vec<(String, f64, &str, String)> = rewrites.iter().enumerate().map(|(i, r)| {
            let mut note = Vec::new();
            if let Some(per) = num(r, "tokens_per_round") { note.push(format!("{per:.1} tokens/round")); }
            if let Some(a) = num(r, "acceptance") { note.push(format!("{:.0}% accepted", 100.0 * a)); }
            (format!("pass {}", i + 1), num(r, "tok_s").unwrap_or(0.0), t.series[2], note.join(" · "))
        }).collect();
        y += hbars(doc, t, 0.0, y, w, &rows, "tps");
    }
    y + 6.0
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
    plot.axes(doc, t, "concurrent requests", "tps", &|x| format!("C{}", x as u64));
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
    // The legend starts after the y-axis title ("TTFT s"), never under it.
    legend(doc, t, 12.0 + super::svg::text_width("TTFT s", 10.0) + 14.0, 10.0,
        &[("cold", t.series[3]), (if w < 420.0 { "cached prefix" } else { "cached prefix (last 256 tokens new)" }, t.series[1])]);
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
    // Each point sits at its nominal retained context (prompt tokens for older reports).
    let points: Vec<(f64, f64)> = v["points"].as_array().map(|p| p.iter()
        .filter_map(|p| Some((num(p, "context").or_else(|| num(p, "prompt_tokens"))?, num(p, "decode_tok_s")?))).collect())
        .unwrap_or_default();
    if points.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let h = plot_height(w) * 0.85;
    // Contexts double (0, 4K, 16K, ... 256K): a log2 axis spreads them evenly;
    // the empty context sits at the axis start, labelled 0. Ticks are only the
    // measured contexts, never powers of two nobody ran.
    let first = points.iter().map(|p| p.0).filter(|x| *x > 0.0).fold(f64::INFINITY, f64::min);
    let min = if first.is_finite() { (first / 2.0).max(256.0) } else { 1024.0 };
    let max = points.iter().map(|p| p.0).fold(min * 2.0, f64::max) * 1.05;
    let points: Vec<(f64, f64)> = points.into_iter().map(|(x, y)| (x.max(min), y)).collect();
    // Headroom above the top point keeps its value label inside the plot.
    let top = points.iter().map(|p| p.1).fold(0.0, f64::max) * 1.15;
    // The axis starts a little before the zero point so its label clears the y ticks.
    let plot = Plot { x: 46.0, y: 30.0, w: w - 110.0, h, xs: Scale::Log2 { min: min / 1.6, max },
        ys: Scale::linear_from(points.iter().map(|p| p.1).chain([top]), true) };
    let ticks: Vec<f64> = points.iter().map(|p| p.0).collect();
    plot.axes_at(doc, t, &ticks, "tokens of context", "decode tps", &|x| if x <= min * 1.0001 { "0".into() } else { charts::tokens(x) });
    plot.line(doc, &points, t.series[0], None);
    for (x, y) in &points {
        doc.text(plot.px(*x), plot.py(*y) - 9.0, Font::new(10.0, t.ink).anchor(Anchor::Middle), &rate(*y));
    }
    30.0 + h + 36.0
}

fn prefix_cache(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let turns = v["turns"].as_array().cloned().unwrap_or_default();
    if turns.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let max = turns.iter().filter_map(|x| num(x, "prompt_tokens")).fold(1.0, f64::max);
    let h = 150.0;
    // Room above the tallest bar for its TTFT label, clear of the legend row.
    let top = 36.0;
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
    hbars(doc, t, 0.0, 0.0, w, &rows, "tps")
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
    let mut h = (y0 + rows * (size + gap)).max(64.0);
    // How the items ran, on its own line under the grid (never over the names).
    if let Some(sandbox) = v["sandbox"].as_str() {
        doc.text(0.0, h + 12.0, Font::new(10.0, t.muted), &super::svg::fit(sandbox, 10.0, w));
        h += 20.0;
    }
    h
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
    // The token axis spans the decades the levels actually used.
    let max_tokens = stats.iter().map(|s| s.1).fold(10.0, f64::max);
    let min_tokens = stats.iter().map(|s| s.1).fold(max_tokens, f64::min);
    let h = 170.0;
    // Phone width: the tier heatmap goes under the ladder instead of beside it.
    let narrow = w < 520.0;
    let ladder_w = if narrow { w } else { w * 0.55 };
    let plot = Plot { x: 46.0, y: 44.0, w: ladder_w - 60.0, h, xs: Scale::Log10 {
            min: 10f64.powf(min_tokens.log10().floor()).min(max_tokens / 10.0), max: 10f64.powf(max_tokens.log10().ceil()) },
        ys: Scale::Linear { min: 0.0, max: 1.0 } };
    plot.axes(doc, t, "mean reasoning tokens", "accuracy", &|x| short(x));
    let line: Vec<(f64, f64)> = stats.iter().map(|s| (s.1, s.2)).collect();
    plot.line(doc, &line, t.series[0], None);
    // Level labels: right of each point, moved below it when they would
    // overlap the previous label.
    let mut placed: Vec<(f64, f64, f64)> = Vec::new();
    for (i, (label, tokens, acc, lo, hi)) in stats.iter().enumerate() {
        let color = t.series[i % t.series.len()];
        plot.range(doc, *tokens, *lo, *acc, *hi, color);
        let (x, width) = (plot.px(*tokens) + 7.0, super::svg::text_width(label, 10.0));
        let overlaps = |y: f64| placed.iter().any(|(px, py, pw)| x < px + pw + 4.0 && *px < x + width + 4.0 && (y - py).abs() < 12.0);
        let above = plot.py(*acc) - 6.0;
        let y = [above, plot.py(*acc) + 15.0, above - 13.0, plot.py(*acc) + 28.0].into_iter()
            .find(|y| !overlaps(*y)).unwrap_or(above);
        doc.text(x, y, Font::new(10.0, color).weight(600), label);
        placed.push((x, y, width));
    }
    // 2. Accuracy by tier per level (right of the ladder).
    let tiers = ["medium", "hard", "brutal", "aime"];
    let (hx, hy) = if narrow { (0.0, 44.0 + h + 40.0) } else { (w * 0.55 + 10.0, 0.0) };
    doc.text(hx, hy + 12.0, Font::new(10.5, t.ink2).weight(600).spacing(1.2), "ACCURACY BY TIER");
    let cell = |i: usize, j: usize| {
        let cells: Vec<&Value> = of(&rows, &levels[j]).filter(|r| r["tier"].as_str() == Some(tiers[i])).collect();
        (!cells.is_empty()).then(|| cells.iter().filter(|r| r["correct"] == true).count() as f64 / cells.len() as f64)
    };
    let xs: Vec<String> = tiers.iter().map(|s| s.to_string()).collect();
    let heat_h = charts::heatmap(doc, t, hx, hy + 26.0, w - hx, &xs, &levels, &cell, t.good, t.bad);
    y += if narrow { hy + 26.0 + heat_h + 18.0 } else { 44.0 + h + 40.0 };
    // 3. Time to solve: solo-equivalent seconds, solved and failed, per level.
    doc.text(0.0, y + 12.0, Font::new(10.5, t.ink2).weight(600).spacing(1.2), "TIME TO SOLVE · solved vs failed");
    // How the time is derived, right-aligned where it fits beside the title.
    let note = if narrow { "solo-equivalent" } else { "solo-equivalent: reasoning tokens ÷ C1 decode" };
    if w >= 300.0 + super::svg::text_width(note, 10.0) {
        doc.text(w, y + 12.0, Font::new(10.0, t.muted).anchor(Anchor::End), note);
    }
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
        // Narrow slots stack the level name over its cap count.
        let cap = if caps > 0 { format!("{caps} cap") } else { String::new() };
        let size = if slot < 70.0 { 9.5 } else { 10.5 };
        if slot < 70.0 || super::svg::text_width(&format!("{level} · {cap}"), size) > slot - 4.0 {
            doc.text(vplot.px(i as f64), vplot.y + vh + 16.0, Font::new(size, t.ink2).anchor(Anchor::Middle),
                &super::svg::fit(level, size, slot - 2.0));
            if !cap.is_empty() {
                doc.text(vplot.px(i as f64), vplot.y + vh + 28.0, Font::new(size, t.muted).anchor(Anchor::Middle), &cap);
            }
        } else {
            doc.text(vplot.px(i as f64), vplot.y + vh + 16.0, Font::new(size, t.ink2).anchor(Anchor::Middle),
                &format!("{level}{}", if cap.is_empty() { String::new() } else { format!(" · {cap}") }));
        }
    }
    y += 28.0 + vh + 42.0;
    // 4. Overthinking: reasoning tokens on the easy (medium) tier.
    doc.text(0.0, y + 12.0, Font::new(10.5, t.ink2).weight(600).spacing(1.2),
        if w < 420.0 { "OVERTHINKING · easy items" } else { "OVERTHINKING · reasoning tokens on easy items" });
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
    // The verdict leads; the counts follow on the same line when they fit, else on a second line.
    let head = format!("task {} · ", v["task"].as_str().unwrap_or("?"));
    let detail = format!("{} turns · {} reused whole · {} invalid tool calls · {} tps mean{}",
        turns.len(), reused, invalid, rate(mean_rate), num(v, "seconds").map(|s| format!(" · {}", seconds(s))).unwrap_or_default());
    let one_line = super::svg::text_width(&format!("{head}{label} · {detail}"), 12.0) <= w;
    let top = if one_line {
        doc.spans(0.0, 14.0, 12.0, Anchor::Start, &[(&head, t.ink2, 400), (label, color, 700), (&format!(" · {detail}"), t.ink2, 400)]);
        26.0
    } else {
        doc.spans(0.0, 14.0, 12.0, Anchor::Start, &[(&head, t.ink2, 400), (label, color, 700)]);
        doc.text(0.0, 32.0, Font::new(11.0, t.ink2), &super::svg::fit(&detail, 11.0, w));
        44.0
    };
    doc.group(0.0, top);
    let h = prefix_cache(doc, t, v, w);
    doc.end();
    top + h
}

fn startup(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let phases: Vec<(String, f64)> = v["phases"].as_array().map(|p| p.iter().filter_map(|x|
        Some((x["name"].as_str()?.to_string(), num(x, "at_s")?))).collect()).unwrap_or_default();
    let ready = num(v, "readiness_s").or_else(|| phases.last().map(|p| p.1)).unwrap_or(0.0);
    let warmup = num(v, "warmup_s").unwrap_or(0.0);
    let end = (ready + warmup).max(1.0);
    // Phone width: labels above their bars, the bars full width.
    let narrow = w < 520.0;
    let (x0, bar_w) = if narrow { (0.0, w - 70.0) } else { (210.0, w - 290.0) };
    let row = if narrow { 40.0 } else { 26.0 };
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
        let y = 6.0 + row * i as f64;
        let bar_y = if narrow { y + 16.0 } else { y };
        doc.text(0.0, y + 13.0, Font::new(11.0, t.ink2), name);
        doc.rect(px(*a), bar_y + 2.0, (px(*b) - px(*a)).max(2.0), 16.0, 3.0, color);
        doc.text(px(*b) + 6.0, bar_y + 14.0, Font::new(10.5, t.ink), &seconds(b - a));
    }
    let y = 6.0 + row * segments.len() as f64 + 4.0;
    doc.line(x0, y, x0 + bar_w, y, t.line2, 1.0);
    doc.text(x0, y + 14.0, Font::new(10.0, t.muted), "process start");
    doc.text(x0 + bar_w, y + 14.0, Font::new(10.0, t.muted).anchor(Anchor::End), &format!("{} after start", seconds(end)));
    y + 20.0
}

#[cfg(test)]
mod context_load_tests {
    #[test]
    fn fidelity_panel_shows_recorded_precision_and_root_reason() {
        let mut value = serde_json::json!({"decode":{}});
        for root in [false, true] {
            let p = crate::fidelity_precision::Precision::new(4.0, "test", root, "test").unwrap();
            value["decode"]["precision"] = serde_json::json!(p);
            let mut doc = super::Doc::new(900.0, 300.0);
            super::fidelity(&mut doc, &super::super::theme::NORMAL, &value, 900.0);
            let svg = doc.finish();
            assert!(svg.contains("4.0 bpw"));
            assert!(svg.contains(if root { "reference-root format" } else { "tier:" }));
        }
    }

    #[test]
    fn fidelity_card_labels_reference() {
        let value = serde_json::json!({"dataset":{"config":"dense-bf16","revision":"a".repeat(40),
            "qualification":"reference"}});
        let mut doc = super::Doc::new(900.0, 300.0);
        super::fidelity(&mut doc, &super::super::theme::NORMAL, &value, 900.0);
        assert!(doc.finish().contains("reference"));
    }

    #[test]
    fn fidelity_names_sources_marks_borrowed_text_and_charts_windows() {
        let window = |kl: f64, top1: f64| serde_json::json!({"positions": 512, "kl": kl, "top1": top1});
        let verdict = serde_json::json!({"pass": false, "kl_max": 0.06, "top1_min": 0.9, "confident_top1_min": 0.98, "top3_min": 0.99,
            "generated": {"positions": 1024, "kl": 0.07, "top1": 0.91, "confident_top1": 0.985, "top3_contained": 0.995},
            "reasons": ["generated KL above expect/floor"]});
        let v = serde_json::json!({"dataset": {"config": "deepseek_v41-v2", "revision": "9fc394f9b7e8",
                "qualification": "reference-only qualified", "text_checkpoint": "deepseek-ai/DeepSeek-V4.1-Flash"},
            "reference_match": "name", "reference_root": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "text_checkpoint": "deepseek-ai/DeepSeek-V4.1-Flash",
            "decode": {"checkpoint": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "seconds": 160.0},
            "verdict": verdict, "per_window": {"a00": window(0.24, 0.86), "a01": window(0.03, 0.95)}});
        let mut doc = super::Doc::new(900.0, 900.0);
        assert!(super::fidelity(&mut doc, &super::super::theme::NORMAL, &v, 900.0) > 200.0);
        let svg = doc.finish();
        for label in ["Reference logits: DeepSeek-V4.1-Flash-1017 (reference-only)", "prompt text: shared DeepSeek-V4.1-Flash set",
            "provisional: window text was generated by DeepSeek-V4.1-Flash", "reference-only qualified:", "KL PER WINDOW",
            "TOP-1 PER WINDOW", "floor 0.060", "floor 90%", "decode window a00: KL 0.24000"] {
            assert!(svg.contains(label), "{label}");
        }
        assert!(!svg.contains("Window text from"), "no ambiguous source line");
    }

    #[test]
    fn retained_ticks_only_the_measured_contexts() {
        let point = |context: u64, rate: f64| serde_json::json!({"context": context, "prompt_tokens": context + 40, "decode_tok_s": rate});
        let v = serde_json::json!({"points": [point(0, 75.0), point(4096, 70.0), point(65536, 67.0), point(262144, 54.0)]});
        let mut doc = super::Doc::new(640.0, 400.0);
        super::retained(&mut doc, &super::super::theme::NORMAL, &v, 640.0);
        let svg = doc.finish();
        for label in [">0<", ">4K<", ">64K<", ">256K<"] { assert!(svg.contains(label), "{label}"); }
        for unmeasured in [">512<", ">1K<", ">2K<", ">16K<"] { assert!(!svg.contains(unmeasured), "{unmeasured}"); }
    }

    #[test]
    fn narrow_reasoning_stacks_the_tier_heatmap_under_the_ladder() {
        let rows: Vec<_> = ["off", "low", "high"].iter().flat_map(|level| ["medium", "hard"].iter().map(move |tier|
            serde_json::json!({"level": level, "tier": tier, "correct": true, "reasoning_tokens": 300, "solo_s": 2.0}))).collect();
        let v = serde_json::json!({"levels": ["off", "low", "high"], "rows": rows});
        let height = |w: f64| { let mut doc = super::Doc::new(w, 900.0); super::reasoning(&mut doc, &super::super::theme::NORMAL, &v, w) };
        assert!(height(360.0) > height(900.0) + 100.0, "the heatmap moves below the ladder on a phone");
    }

    #[test]
    fn draft_mix_charts_waves_and_rewrites() {
        let v = serde_json::json!({"waves": [{"c":16,"aggregate_tok_s":640.0,"per_request_tok_s":41.0},
            {"c":16,"aggregate_tok_s":610.0,"per_request_tok_s":39.5}],
            "rewrite": [{"tok_s":212.0,"tokens_per_round":5.4,"acceptance":0.91},{"tok_s":208.0,"tokens_per_round":5.2,"acceptance":0.9}]});
        let mut doc = super::Doc::new(700.0, 400.0);
        assert!(super::draft_mix(&mut doc, &super::super::theme::NORMAL, &v, 700.0) > 100.0);
        let svg = doc.finish();
        for label in ["C16 MIXED PROMPTS", "per-request median", "C1 COPY-HEAVY REWRITE", "5.4 tokens/round", "91% accepted"] {
            assert!(svg.contains(label), "{label}");
        }
    }

    #[test]
    fn context_table_renders_results_and_skips() {
        let v = serde_json::json!({"points": [
            {"context":32768,"c":8,"status":"ok","aggregate_tok_s":123.4,"per_request_tok_s":20.0,
             "ttft_s":0.01,"cached_tokens":32760,"cold_ttft_s":3.0,"restore_pages":"shared_device_pages"},
            {"context":131072,"c":16,"status":"skipped","reason":"shared estimate exceeds KV pool"}]});
        let mut doc = super::Doc::new(900.0, 300.0);
        assert!(super::context_load(&mut doc, &super::super::theme::NORMAL, &v, 900.0) > 0.0);
        let svg = doc.finish();
        for label in ["32K", "128K", "32760", "agg tps", "shared_device_pages", "shared estimate exceeds KV pool",
            "AGGREGATE DECODE AT SHARED CONTEXT", "32K · C8"] {
            assert!(svg.contains(label), "{label}");
        }
    }
}
