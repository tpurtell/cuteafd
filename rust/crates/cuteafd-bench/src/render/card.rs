//! The 1200×675 share card: model, C1 and concurrent code decode beside 8K
//! prefill (two headlines for old reports), three content bars, one hardware line, the non-default option
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
    doc.text(pad, 182.0, Font::new(15.0, t.ink2), &fit(&sub.join(" · "), 15.0, WIDTH - 2.0 * pad));
    let dim = if t.scary { 0.55 } else { 1.0 };
    let warn = if t.scary { "⚠" } else { "" };
    let baseline = r.baseline.as_ref();
    let code = baseline.and_then(|b| b.card.decode_of("code")).map_or(0.0, |d| d.tok_s);
    let prefill = baseline.and_then(|b| b.card.prefill.as_ref());
    let concurrent = baseline.and_then(|b| b.card.concurrent.as_ref());
    let columns = if concurrent.is_some() { 3.0 } else { 2.0 };
    let gap = if concurrent.is_some() { 28.0 } else { 20.0 };
    let column_width = (WIDTH - 2.0 * pad - gap * (columns - 1.0)) / columns;
    let max_size: f64 = if concurrent.is_some() { 88.0 } else { 112.0 };
    let sub_size = if concurrent.is_some() { 13.0 } else { 15.0 };
    // Reserve the unit's width before sizing numerals, including commas and the failure badge.
    let huge = |doc: &mut Doc, x: f64, label: &str, value: &str, sub: &str, color: &str| {
        doc.text(x, 246.0, Font::new(15.0, t.muted).spacing(3.0).weight(600), label);
        let shown = format!("{warn}{value}");
        let count = shown.chars().count() as f64;
        let size = max_size.min((column_width - text_width("tok/s", 24.0) - 14.0 + 3.0 * count) / (0.6 * count));
        doc.text(x, 352.0, Font::new(size, color).bold().opacity(dim).spacing(-3.0), &shown);
        let width = text_width(&shown, size) - 3.0 * count;
        doc.text(x + width + 14.0, 352.0, Font::new(24.0, t.ink2), "tok/s");
        doc.text(x, 386.0, Font::new(sub_size, t.ink2), &fit(sub, sub_size, column_width));
    };
    if let Some(reason) = r.no_fit_reason() {
        doc.text(pad, 280.0, Font::new(42.0, t.ink).bold(), "Doesn't fit");
        doc.text(pad, 330.0, Font::new(18.0, t.ink2), &fit(reason, 18.0, WIDTH - 2.0 * pad));
        doc.text(pad, 386.0, Font::new(15.0, t.muted), "Planner-only qualification; no performance measured");
    } else {
        huge(&mut doc, pad, "C1 CODE DECODE", &rate(code), "thinking off · one request", t.series[0]);
        if let Some(c) = concurrent {
            huge(&mut doc, pad + column_width + gap, &format!("C{} CODE DECODE", c.width), &rate(c.aggregate_tok_s),
                &format!("aggregate · {} tok/s per stream", rate(c.per_stream_median_tok_s)), t.concurrent);
        }
        let prefill_x = if concurrent.is_some() { pad + 2.0 * (column_width + gap) } else { 610.0 };
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
                &format!("{} tok/s", rate(d.tok_s)));
        }
    }
    // Hardware, options, build and badge.
    let hw = &r.server.hardware;
    let mut hardware = r.server.hardware_line();
    let links: Vec<f64> = hw.fabric.iter().filter(|p| p.active).map(|p| p.link_gbps).collect();
    if !hw.sparks.is_empty() && !links.is_empty() {
        hardware.push_str(&format!(" · RoCE {:.0} Gb/s", links.iter().cloned().fold(0.0, f64::max)));
    }
    doc.text(pad, 538.0, Font::new(16.0, t.ink).weight(600), &fit(&hardware, 16.0, WIDTH - 2.0 * pad));
    doc.text(pad, 560.0, Font::new(13.0, t.ink2), &fit(&r.capacity().line(), 13.0, WIDTH - 2.0 * pad));
    // Publication evidence belongs in the full report, not the fixed-height option strip.
    let options: Vec<String> = c.non_default().filter(|s| !DEPLOYMENT.contains(&s.name.as_str())
        && !s.name.starts_with("provenance.") && !s.name.starts_with("qualification.")
        && s.name != "supersedes-checkpoint")
        .map(|s| s.chip()).take(10).collect();
    if !options.is_empty() {
        chips(&mut doc, &t, pad, 572.0, WIDTH - 2.0 * pad, &options, t.series[2]);
    }
    let b = &r.server.build;
    let mut build = format!("build {}", b.label());
    if let Some(remote) = &b.remote {
        build.push_str(&format!(" · {}", remote.trim_start_matches("https://")));
    }
    doc.text(pad, HEIGHT - 34.0, Font::new(13.0, t.muted), &fit(&build, 13.0, 640.0));
    let (badge, status) = match baseline {
        Some(b) => (b.quality.badge(), b.quality.status),
        None => ("quality not verified".to_string(), crate::report::CheckStatus::Pending),
    };
    let color = t.status(status);
    let width = (text_width(&badge, 14.0) + 36.0).min(520.0);
    doc.rect_attrs(WIDTH - pad - width, HEIGHT - 58.0, width, 34.0, 17.0,
        &format!(r#"fill="{}" stroke="{color}" stroke-width="1.5""#, t.panel2));
    doc.text(WIDTH - pad - width / 2.0, HEIGHT - 36.0, Font::new(14.0, color).anchor(Anchor::Middle).weight(600),
        &fit(&badge, 14.0, width - 20.0));
    bodies::watermark(&mut doc, &t);
    doc.finish()
}
