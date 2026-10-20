//! The 1200×675 share card: model, C1 and code decode scaling beside 8K
//! prefill, three content bars, one hardware line, the non-default option
//! chips, the build footer and the fidelity badge.
use super::bodies::{self, chips, content_color, LOGO, LOGO_ASPECT};
use super::svg::{fit, text_width, Anchor, Doc, Font};
use super::{grouped, rate, seconds, Theme};
use crate::context::DEPLOYMENT;
use crate::report::Report;

pub const WIDTH: f64 = 1200.0;
pub const HEIGHT: f64 = 675.0;

pub fn card_svg(report: &Report) -> String {
    let t = Theme::for_report(report.quality_failed());
    let r = report;
    let mut doc = Doc::new(WIDTH, HEIGHT);
    doc.defs(&t.defs());
    doc.rect(0.0, 0.0, WIDTH, HEIGHT, 24.0, t.bg);
    doc.rect_attrs(0.0, 0.0, WIDTH, HEIGHT, 24.0, r#"fill="url(#glow-a)""#);
    doc.rect_attrs(0.0, 0.0, WIDTH, HEIGHT, 24.0, r#"fill="url(#glow-b)""#);
    doc.rect_attrs(0.5, 0.5, WIDTH - 1.0, HEIGHT - 1.0, 24.0, &format!(r#"fill="none" stroke="{}""#, t.line2));
    let pad = 56.0;
    doc.nested(LOGO, pad, 36.0, 46.0 * LOGO_ASPECT, 46.0);
    doc.text(WIDTH - pad, 62.0, Font::new(13.0, t.muted).anchor(Anchor::End).spacing(1.5),
        &format!("{} · {}", crate::profiles::title_of(&r.profile).to_uppercase(), super::date(&r.created)));
    bodies::title(&mut doc, &t, pad, 148.0, 40.0, &r.server.checkpoint(), WIDTH - 2.0 * pad);
    let c = &r.server.configuration;
    let mut sub = Vec::new();
    if let Some(rev) = &r.server.revision {
        sub.push(format!("rev {}", &rev[..rev.len().min(10)]));
    }
    let quant: Vec<String> = c.quant.iter().filter(|q| q.group.contains("routed")).map(|q| q.formats.join("+")).collect();
    if !quant.is_empty() {
        sub.push(format!("experts {}", quant.join(" ")));
    }
    sub.push(format!("speculator {}", c.speculator_label()));
    if !c.supports_thinking_off() { sub.insert(0, c.thinking_label().into()); }
    doc.text(pad, 182.0, Font::new(15.0, t.ink2), &fit(&sub.join(" · "), 15.0, WIDTH - 2.0 * pad));
    let dim = if t.scary { 0.55 } else { 1.0 };
    let warn = if t.scary { "⚠" } else { "" };
    let baseline = r.baseline.as_ref();
    let code = baseline.and_then(|b| b.card.decode_of("code")).map_or(0.0, |d| d.tok_s);
    let prefill = baseline.and_then(|b| b.card.prefill.as_ref());
    let gap = 28.0;
    let column_width = (WIDTH - 2.0 * pad - gap * 2.0) / 3.0;
    let max_size: f64 = 88.0;
    let sub_size = 13.0;
    // Reserve the unit's width before sizing numerals, including commas and the failure badge.
    let huge = |doc: &mut Doc, x: f64, label: &str, value: &str, sub: &str, color: &str| {
        doc.text(x, 246.0, Font::new(15.0, t.muted).spacing(3.0).weight(600), label);
        let shown = format!("{warn}{value}");
        let count = shown.chars().count() as f64;
        let size = max_size.min((column_width - text_width("tps", 24.0) - 14.0 + 3.0 * count) / (0.6 * count));
        doc.text(x, 352.0, Font::new(size, color).bold().opacity(dim).spacing(-3.0), &shown);
        let width = text_width(&shown, size) - 3.0 * count;
        doc.text(x + width + 14.0, 352.0, Font::new(24.0, t.ink2), "tps");
        doc.text(x, 386.0, Font::new(sub_size, t.ink2), &fit(sub, sub_size, column_width));
    };
    if let Some(reason) = r.no_fit_reason() {
        doc.text(pad, 280.0, Font::new(42.0, t.ink).bold(), "Doesn't fit");
        doc.text(pad, 330.0, Font::new(18.0, t.ink2), &fit(reason, 18.0, WIDTH - 2.0 * pad));
        doc.text(pad, 386.0, Font::new(15.0, t.muted), "Planner-only qualification; no performance measured");
    } else {
        huge(&mut doc, pad, "C1 CODE DECODE", &rate(code), &format!("{} · one request", c.thinking_label()), t.series[0]);
        bodies::decode_scaling(&mut doc, &t, r, pad + column_width + gap, 246.0, column_width - 60.0, 1.0);
        let prefill_x = pad + 2.0 * (column_width + gap);
        match prefill {
            Some(p) => huge(&mut doc, prefill_x, "8K PREFILL", &rate(p.tok_s),
                &format!("TTFT {} for {} tokens", seconds(p.ttft_s), grouped(p.prompt_tokens as f64)), t.series[3]),
            None => huge(&mut doc, prefill_x, "8K PREFILL", "—", "not measured", t.series[3]),
        }
    }
    // Three content bars.
    let (bx, by, bw) = (pad, 424.0, WIDTH - 2.0 * pad);
    if let Some(b) = baseline {
        let max = b.card.decode.iter().map(|d| d.tok_s).fold(1.0f64, f64::max) * 1.08;
        for (i, d) in b.card.decode.iter().take(3).enumerate() {
            let y = by + 30.0 * i as f64;
            let color = content_color(&t, &d.content);
            doc.text(bx, y + 16.0, Font::new(14.0, t.ink2), &d.content);
            let track = bw - 200.0;
            doc.rect(bx + 90.0, y + 4.0, track, 16.0, 4.0, t.well);
            doc.rect_attrs(bx + 90.0, y + 4.0, track * (d.tok_s / max).clamp(0.0, 1.0), 16.0, 4.0,
                &format!(r#"fill="{color}" opacity="{dim}""#));
            doc.text(bx + bw, y + 18.0, Font::new(16.0, t.ink).anchor(Anchor::End).bold().opacity(dim),
                &format!("{} tps", rate(d.tok_s)));
        }
    }
    // Hardware, options, build and badge.
    let hw = &r.server.hardware;
    let hardware = r.server.hardware_line();
    let fabric = crate::fabric::header(&r.fabric);
    let rail_width = fabric.as_ref().map_or(0.0, |line| line.chars().count() as f64 * 8.5 + 24.0);
    doc.text(pad, 538.0, Font::new(16.0, t.ink).weight(600), &fit(&hardware, 16.0, WIDTH - 2.0 * pad - rail_width));
    if let Some(fabric) = fabric {
        doc.text(WIDTH - pad, 538.0, Font::new(13.0, t.ink2).anchor(Anchor::End), &fabric);
    }
    doc.text(pad, 560.0, Font::new(13.0, t.ink2), &fit(&r.capacity().line(), 13.0, WIDTH - 2.0 * pad));
    // Publication evidence belongs in the full report, not the fixed-height option strip.
    let mut options: Vec<String> = c.non_default().filter(|s| !DEPLOYMENT.contains(&s.name.as_str())
        && !s.name.starts_with("provenance.") && !s.name.starts_with("qualification.")
        && !(s.name == "pool-tokens" && matches!(s.value.as_deref(), Some("0" | "auto")))
        && s.name != "supersedes-checkpoint")
        .map(|s| s.chip()).collect();
    if let Some(power) = hw.power.as_ref().filter(|power| power.capped()) {
        // Report the longest per-GPU evidence, not a sum of GPU-seconds.
        let capped = power.gpus.iter().map(|gpu| gpu.cap_seconds).fold(0.0, f64::max);
        options.insert(0, format!("POWER-CAPPED {}", seconds(capped)));
    }
    let health = crate::fabric::health_header(&r.fabric);
    // Reserve the second option row for health evidence when available.
    let omitted = |items: &[String]| {
        let mut probe = Doc::new(WIDTH, HEIGHT);
        chips(&mut probe, &t, pad, 572.0, WIDTH - 2.0 * pad, items, t.series[2]) > if health.is_some() { 22.0 } else { 46.0 }
    };
    let total = options.len();
    while omitted(&options) {
        options.pop();
    }
    if options.len() < total {
        loop {
            let mut shown = options.clone();
            shown.push(format!("+{} more in report", total - options.len()));
            if !omitted(&shown) {
                options = shown;
                break;
            }
            options.pop();
        }
    }
    chips(&mut doc, &t, pad, 572.0, WIDTH - 2.0 * pad, &options, t.series[2]);
    if let Some(health) = health {
        doc.text(pad, 614.0, Font::new(12.0, t.ink2), &fit(&health, 12.0, WIDTH - 2.0 * pad));
    }
    let b = &r.server.build;
    let mut build = format!("build {}", b.label());
    if let Some(remote) = &b.remote {
        build.push_str(&format!(" · {}", remote.trim_start_matches("https://")));
    }
    let (badge, status) = match baseline {
        Some(b) => (b.quality.badge(), b.quality.status),
        None => ("quality not verified".to_string(), crate::report::CheckStatus::Pending),
    };
    if let Some(precision) = baseline.and_then(|b| b.quality.check("fidelity"))
        .and_then(|c| c.metrics.get("precision"))
        .and_then(|v| serde_json::from_value::<crate::fidelity_precision::Precision>(v.clone()).ok()) {
        doc.text(pad, 205.0, Font::new(12.0, t.ink2), &fit(&precision.label(), 12.0, WIDTH - 2.0 * pad));
    }
    let color = t.status(status);
    let width = (text_width(&badge, 14.0) + 36.0).min(520.0);
    doc.text(pad, HEIGHT - 28.0, Font::new(13.0, t.muted), &fit(&build, 13.0, WIDTH - 2.0 * pad - width - 24.0));
    doc.rect_attrs(WIDTH - pad - width, HEIGHT - 46.0, width, 34.0, 17.0,
        &format!(r#"fill="{}" stroke="{color}" stroke-width="1.5""#, t.panel2));
    doc.text(WIDTH - pad - width / 2.0, HEIGHT - 24.0, Font::new(14.0, color).anchor(Anchor::Middle).weight(600),
        &fit(&badge, 14.0, width - 20.0));
    bodies::watermark(&mut doc, &t);
    doc.finish()
}
