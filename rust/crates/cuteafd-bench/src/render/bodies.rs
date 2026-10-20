//! Panel bodies and the pieces every document shares: frames, chips,
//! key/value rows, the logo and footers.
use super::svg::{fit, text_width, Anchor, Doc, Font};
use super::{rate, seconds, Theme};
use crate::context::DEPLOYMENT;
use crate::report::{Baseline, CheckStatus, PanelResult, Report};
use serde_json::Value;

/// The wordmark and the square mark for dark backgrounds (compact copies of
/// assets/brand/*-color-dark.svg made by scripts/bench/simplify-logo.py).
pub const LOGO: &str = include_str!("../../assets/logo-dark.svg");
pub const MARK: &str = include_str!("../../assets/mark-dark.svg");
pub const LOGO_ASPECT: f64 = 1770.0 / 486.0;

pub fn wrap(text: &str, size: f64, width: f64) -> Vec<String> {
    let max = (width / (size * 0.6)).floor().max(4.0) as usize;
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > max {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        // A word longer than a line (a URL, a hash) breaks at the edge instead of running past it.
        let mut rest: Vec<char> = word.chars().collect();
        while line.chars().count() + rest.len() > max {
            let take = max.saturating_sub(line.chars().count()).max(1);
            line.extend(rest.drain(..take.min(rest.len())));
            lines.push(std::mem::take(&mut line));
        }
        line.extend(rest);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// Rounded chips that wrap inside `width`; returns the height used.
pub fn chips(doc: &mut Doc, theme: &Theme, x: f64, y: f64, width: f64, items: &[String], color: &str) -> f64 {
    let (size, h, pad, gap) = (11.0, 20.0, 8.0, 6.0);
    let (mut cx, mut cy) = (x, y);
    for item in items {
        let label = fit(item, size, width - 2.0 * pad);
        let w = text_width(&label, size) + 2.0 * pad;
        if cx > x && cx + w > x + width {
            cx = x;
            cy += h + gap;
        }
        doc.rect_attrs(cx, cy, w, h, 10.0, &format!(r#"fill="{}" stroke="{color}" stroke-opacity="0.55""#, theme.panel2));
        doc.text(cx + pad, cy + 14.0, Font::new(size, theme.ink), &label);
        cx += w + gap;
    }
    if items.is_empty() { 0.0 } else { cy - y + h }
}

/// A label column and a value; returns the row height.
fn row(doc: &mut Doc, theme: &Theme, x: f64, y: f64, w: f64, label: &str, value: &str) -> f64 {
    doc.text(x, y + 14.0, Font::new(10.0, theme.muted).spacing(1.2), &label.to_uppercase());
    let lines = wrap(value, 12.0, w - 120.0);
    for (i, line) in lines.iter().enumerate() {
        doc.text(x + 120.0, y + 14.0 + 17.0 * i as f64, Font::new(12.0, theme.ink), line);
    }
    20.0 + 17.0 * lines.len().saturating_sub(1) as f64
}

/// Decode scaling chart, shared by the compact card and baseline panel.
/// Coordinates use the approved card's chart box; panels scale the same geometry.
pub fn decode_scaling(doc: &mut Doc, theme: &Theme, report: &Report, x: f64, y: f64, width: f64, scale: f64) {
    let Some(baseline) = &report.baseline else { return };
    // The axis is the sweep the report actually ran: a level with no valid
    // aggregate has no point, and is not labelled either.
    let points: Vec<_> = report.server.card_levels().iter().filter_map(|width| baseline.card.concurrent_at(*width))
        .filter(|point| point.aggregate_tok_s.is_finite() && point.aggregate_tok_s >= 0.0).collect();
    let measured: Vec<usize> = points.iter().map(|point| point.width).collect();
    let top = y + 36.0 * scale;
    let bottom = y + 106.0 * scale;
    doc.text(x, y, Font::new(15.0 * scale, theme.muted).spacing(3.0 * scale).weight(600), "DECODE SCALING");
    doc.line(x, bottom, x + width, bottom, theme.line2, 1.0);
    let max = points.iter().map(|point| point.aggregate_tok_s).fold(1.0f64, f64::max) * 1.08;
    let px = |level: usize| {
        let index = measured.iter().position(|width| *width == level).unwrap_or(0);
        if measured.len() > 1 {
            x + 16.0 * scale + (width - 32.0 * scale) * index as f64 / (measured.len() - 1) as f64
        } else {
            // One measured level (or none): the point stands alone, centred.
            x + width / 2.0
        }
    };
    let py = |value: f64| bottom - (bottom - top) * value / max;
    for level in &measured {
        doc.text(px(*level), bottom + 17.0 * scale,
            Font::new(12.0 * scale, if *level == 8 { theme.ink } else { theme.muted })
                .weight(if *level == 8 { 700 } else { 400 }).anchor(Anchor::Middle), &format!("C{level}"));
    }
    if let (Some(first), Some(last)) = (points.first(), points.last()) {
        let path = points.iter().enumerate().map(|(i, point)| format!("{}{},{}",
            if i == 0 { "M" } else { " L" }, px(point.width), py(point.aggregate_tok_s))).collect::<String>();
        if points.len() > 1 {
            doc.raw(&format!(r#"<path d="{path} L{},{} L{},{} Z" fill="{}" fill-opacity="0.10"/>"#,
                px(last.width), bottom, px(first.width), bottom, theme.concurrent));
            doc.raw(&format!(r#"<path d="{path}" fill="none" stroke="{}" stroke-width="{}" stroke-linejoin="round" stroke-linecap="round"/>"#,
                theme.concurrent, 3.0 * scale));
        }
        for point in &points {
            let highlight = point.width == 8;
            let (cx, cy) = (px(point.width), py(point.aggregate_tok_s));
            doc.group_attrs(&format!(r#"aria-label="{}""#, super::svg::escape(&format!(
                "C{}: {} aggregate tps, {} median tps per request", point.width,
                rate(point.aggregate_tok_s), rate(point.per_stream_median_tok_s)))));
            doc.raw(&format!("<title>{}</title>", super::svg::escape(&format!(
                "C{}: {} aggregate tps; {} per request (median)", point.width,
                rate(point.aggregate_tok_s), rate(point.per_stream_median_tok_s)))));
            doc.raw(&format!(r#"<circle cx="{cx}" cy="{cy}" r="{}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
                if highlight { 6.5 * scale } else { 4.5 * scale },
                if highlight { theme.concurrent } else { theme.bg }, theme.concurrent, 2.5 * scale));
            doc.text(cx, cy - 13.0 * scale, Font::new(if highlight { 19.0 * scale } else { 14.0 * scale }, theme.ink)
                .bold().anchor(Anchor::Middle), &format!("{:.0}", point.aggregate_tok_s));
            doc.end();
        }
    }
    let caption = baseline.card.concurrent_at(8).map_or_else(|| "aggregate tps · C8 not measured".into(),
        |point| format!("aggregate tps · {}/stream at C8", rate(point.per_stream_median_tok_s)));
    doc.text(x, y + 140.0 * scale, Font::new(13.0 * scale, theme.ink2), &fit(&caption, 13.0 * scale, width));
}

/// The rendering context of one report.
pub struct View<'a> {
    pub theme: Theme,
    pub report: &'a Report,
}

impl<'a> View<'a> {
    pub fn new(report: &'a Report) -> Self {
        Self { theme: Theme::for_report(report.quality_failed()), report }
    }

    /// A panel's title, hint and body as a fragment `width` wide: (defs, body, height).
    pub fn body(&self, id: &str, width: f64) -> (String, f64) {
        let mut doc = Doc::new(width, 0.0);
        let height = match id {
            "hardware" => self.hardware(&mut doc, width),
            "configuration" => self.configuration(&mut doc, width),
            "baseline" => self.baseline(&mut doc, width),
            _ => {
                // A failed or interrupted panel says why above whatever partial chart it has.
                let panel = self.report.panel(id);
                let banner = match panel.and_then(|p| p.error.as_deref().map(|e| (p, e))) {
                    Some((p, error)) if !p.passes.is_empty() || p.partial.is_some() => {
                        let t = &self.theme;
                        let label = if p.status == crate::report::PanelStatus::Failed { "failed" } else { "stopped" };
                        let lines = wrap(error, 11.5, width - 24.0);
                        let h = 14.0 + 16.0 * lines.len() as f64 + 8.0;
                        doc.rect_attrs(0.0, 0.0, width, h, 6.0, &format!(r#"fill="{}" stroke="{}""#, t.panel2, t.bad));
                        doc.text(12.0, 16.0, Font::new(10.0, t.bad).weight(700).spacing(1.2), &label.to_uppercase());
                        for (i, line) in lines.iter().enumerate() {
                            doc.text(12.0, 32.0 + 16.0 * i as f64, Font::new(11.5, t.ink), line);
                        }
                        h + 12.0
                    }
                    _ => 0.0,
                };
                if banner > 0.0 { doc.group(0.0, banner); }
                let height = match panel.and_then(|p| super::panels::body(&mut doc, &self.theme, id, p, width)) {
                    Some(height) => height,
                    None => self.generic(&mut doc, width, panel),
                };
                if banner > 0.0 { doc.end(); }
                banner + height
            }
        };
        (doc.into_parts().1, height)
    }

    /// A framed panel at x, y; returns its height.
    pub fn framed(&self, doc: &mut Doc, x: f64, y: f64, width: f64, id: &str, title: &str, hint: &str,
        failed: bool) -> f64 {
        self.framed_at(doc, x, y, width, None, id, title, hint, failed)
    }

    /// Height a framed panel needs at `width`.
    pub fn framed_height(&self, width: f64, id: &str) -> f64 {
        46.0 + self.body(id, width - 32.0).1 + 16.0
    }

    /// [`Self::framed`], stretched to `height` when given (panels sharing a row).
    #[allow(clippy::too_many_arguments)]
    pub fn framed_at(&self, doc: &mut Doc, x: f64, y: f64, width: f64, height: Option<f64>, id: &str, title: &str,
        hint: &str, failed: bool) -> f64 {
        let t = &self.theme;
        let (body, body_h) = self.body(id, width - 32.0);
        let height = height.unwrap_or(46.0 + body_h + 16.0);
        let hint = if width < 560.0 { "" } else { hint };
        doc.frame(x, y, width, height, 12.0, t.panel, if failed { t.bad } else { t.line });
        doc.rect_attrs(x + 12.0, y + 0.5, width - 24.0, 2.0, 1.0, r#"fill="url(#accent)" opacity="0.9""#);
        if failed {
            // Hazard stripes along the top and left edge of a failed panel.
            doc.rect_attrs(x + 1.0, y + 1.0, width - 2.0, 8.0, 0.0, r#"fill="url(#hazard)""#);
            doc.rect_attrs(x + 1.0, y + 1.0, 6.0, height - 2.0, 0.0, r#"fill="url(#hazard)""#);
        }
        doc.text(x + 16.0, y + 30.0, Font::new(12.0, t.ink).weight(600).spacing(1.7), &title.to_uppercase());
        let title_w = text_width(title, 12.0) + 1.7 * title.chars().count() as f64;
        if !hint.is_empty() {
            doc.text(x + 28.0 + title_w, y + 30.0, Font::new(11.0, t.muted),
                &fit(hint, 11.0, width - title_w - 60.0));
        }
        doc.group(x + 16.0, y + 46.0);
        doc.raw(&body);
        doc.end();
        height
    }

    /// The small footer of single-panel exports: mark, model, build, date.
    pub fn footer(&self, doc: &mut Doc, x: f64, y: f64, width: f64) {
        let t = &self.theme;
        doc.nested(MARK, x, y - 12.0, 16.0, 16.0);
        let r = self.report;
        let left = format!("cuteafd bench · {} · {}", r.server.checkpoint(), r.server.hardware.line());
        doc.text(x + 22.0, y, Font::new(10.0, t.muted), &fit(&left, 10.0, width * 0.62));
        let right = format!("build {} · {}", r.server.build.label(), super::date(&r.created));
        doc.text(x + width, y, Font::new(10.0, t.muted).anchor(Anchor::End), &right);
    }

    fn hardware(&self, doc: &mut Doc, w: f64) -> f64 {
        let t = &self.theme;
        let hw = &self.report.server.hardware;
        let mut y = 0.0;
        for gpu in &hw.gpus {
            let mut parts = vec![gpu.name.trim_start_matches("NVIDIA ").to_string()];
            if let Some(mib) = gpu.memory_mib {
                parts.push(format!("{:.0} GiB", mib as f64 / 1024.0));
            }
            if let Some(sms) = gpu.sm_count {
                parts.push(format!("{sms} SMs"));
            }
            match (gpu.power_limit_w, gpu.power_max_w) {
                (Some(cap), Some(max)) if (cap - max).abs() > 1.0 => parts.push(format!("capped {cap:.0} of {max:.0} W")),
                (Some(cap), _) => parts.push(format!("{cap:.0} W")),
                _ => {}
            }
            if let Some(pcie) = &gpu.pcie {
                parts.push(format!("PCIe {pcie}"));
            }
            if let Some(cc) = &gpu.compute_cap {
                parts.push(format!("SM{}", cc.replace('.', "")));
            }
            if !gpu.used {
                parts.push("not used".into());
            }
            y += row(doc, t, 0.0, y, w, &format!("GPU {}", gpu.index), &parts.join(" · "));
        }
        if hw.gpus.is_empty() {
            y += row(doc, t, 0.0, y, w, "GPU", "not visible to the server");
        }
        let driver = match (&hw.driver, &hw.cuda) {
            (Some(d), Some(c)) => format!("driver {d} · CUDA {c}"),
            (Some(d), None) => format!("driver {d}"),
            _ => "unknown".into(),
        };
        y += row(doc, t, 0.0, y, w, "Driver", &driver);
        let sparks = if hw.sparks.is_empty() {
            "none (experts on the coordinator)".to_string()
        } else {
            let names: Vec<String> = hw.sparks.iter().map(|s| match &s.name {
                Some(name) => format!("{name} {}", s.address),
                None => s.address.clone(),
            }).collect();
            format!("{}× DGX Spark (GB10) — {}", hw.sparks.len(), names.join(", "))
        };
        y += row(doc, t, 0.0, y, w, "Sparks", &sparks);
        let active: Vec<String> = hw.fabric.iter().filter(|p| p.active).map(|p| {
            let mut line = format!("{}/{} {:.0} Gb/s", p.device, p.port, p.link_gbps);
            if let Some(pcie) = &p.pcie {
                line.push_str(&format!(" · PCIe {pcie}"));
            }
            if !p.subnets.is_empty() {
                line.push_str(&format!(" · {}", p.subnets.join(" ")));
            }
            line
        }).collect();
        if active.is_empty() {
            y += row(doc, t, 0.0, y, w, "Fabric", "no active RDMA port");
        } else {
            for (i, line) in active.iter().enumerate() {
                y += row(doc, t, 0.0, y, w, if i == 0 { "Fabric" } else { "" }, line);
            }
        }
        if let Some(rails) = &hw.rails {
            y += row(doc, t, 0.0, y, w, "Rails", rails);
        }
        if let Some(ready) = self.report.server.readiness_s {
            let mut text = format!("{} from process start to serving", seconds(ready));
            if let Some(warmup) = self.report.baseline.as_ref().and_then(|b| b.card.warmup_s) {
                text.push_str(&format!(" · first requests (warm-up) {}", seconds(warmup)));
            }
            y += row(doc, t, 0.0, y, w, "Readiness", &text);
        }
        y
    }

    fn configuration(&self, doc: &mut Doc, w: f64) -> f64 {
        let t = &self.theme;
        let s = &self.report.server;
        let c = &s.configuration;
        let mut y = 0.0;
        y += row(doc, t, 0.0, y, w, "Model", &s.model);
        let mut checkpoint = s.revision.as_deref().map(|r| r.chars().take(12).collect::<String>())
            .unwrap_or_else(|| "unknown revision".into());
        if let Some(family) = &s.family {
            checkpoint.push_str(&format!(" · family {family}"));
        }
        y += row(doc, t, 0.0, y, w, "Checkpoint", &checkpoint);
        if !c.quant.is_empty() {
            doc.text(0.0, y + 14.0, Font::new(10.0, t.muted).spacing(1.2), "WEIGHTS");
            let items: Vec<String> = c.quant.iter().map(|q| format!("{} · {}", q.group, q.formats.join("+"))).collect();
            y += chips(doc, t, 120.0, y, w - 120.0, &items, t.series[0]) + 6.0;
        }
        y += row(doc, t, 0.0, y, w, "Speculator", &c.speculator_label());
        y += row(doc, t, 0.0, y, w, "Thinking", c.thinking_label());
        y += row(doc, t, 0.0, y, w, "Capacity", &self.report.capacity().line());
        if let Some(layout) = &c.layout {
            y += row(doc, t, 0.0, y, w, "Layout", layout);
        }
        let options: Vec<String> = c.non_default().filter(|s| !DEPLOYMENT.contains(&s.name.as_str()))
            .map(|s| s.chip()).collect();
        doc.text(0.0, y + 14.0, Font::new(10.0, t.muted).spacing(1.2), "OPTIONS");
        if options.is_empty() {
            doc.text(120.0, y + 14.0, Font::new(12.0, t.ink2), "all defaults");
            y += 20.0;
        } else {
            y += chips(doc, t, 120.0, y, w - 120.0, &options, t.series[2]) + 6.0;
        }
        let b = &s.build;
        let mut build = b.label();
        if let (Some(remote), Some(commit)) = (&b.remote, &b.commit) {
            build.push_str(&format!(" · {remote} @ {}", &commit[..commit.len().min(12)]));
        }
        if let Some(image) = &b.image {
            build.push_str(&format!(" · {image}"));
        }
        y += row(doc, t, 0.0, y, w, "Build", &build);
        y
    }

    /// The basic card and quick quality.
    fn baseline(&self, doc: &mut Doc, w: f64) -> f64 {
        let t = &self.theme;
        let Some(b) = &self.report.baseline else {
            doc.text(0.0, 16.0, Font::new(12.0, t.ink2), "Baseline pending: it runs first, once per server and configuration.");
            return 24.0;
        };
        let failed = t.scary;
        let dim = if failed { 0.55 } else { 1.0 };
        let warn = if failed { "⚠ " } else { "" };
        let code = b.card.decode_of("code").map_or(0.0, |d| d.tok_s);
        let prefill = b.card.prefill.as_ref();
        let headline_width = (w - 48.0) / 3.0;
        let big = |doc: &mut Doc, x: f64, label: &str, value: &str, unit: &str, sub: &str, color: &str| {
            doc.text(x, 12.0, Font::new(10.0, t.muted).spacing(1.4), label);
            let value = format!("{warn}{value}");
            let size = 42.0f64.min((headline_width - text_width(unit, 13.0) - 8.0) / (0.6 * value.chars().count() as f64));
            doc.text(x, 58.0, Font::new(size, color).bold().opacity(dim), &value);
            doc.text(x + text_width(&value, size) + 8.0, 58.0, Font::new(13.0, t.ink2), unit);
            doc.text(x, 80.0, Font::new(11.0, t.ink2), &fit(sub, 11.0, headline_width));
        };
        let code_tokens = b.card.decode_of("code").and_then(|d| d.runs.first()).map_or(0, |r| r.completion_tokens);
        big(doc, 0.0, "C1 CODE DECODE", &rate(code), "tps", &format!("{} · {code_tokens} tokens", self.report.server.configuration.thinking_label()),
            t.series[0]);
        decode_scaling(doc, t, self.report, headline_width + 24.0, 12.0, headline_width, 0.48);
        let prefill_x = 2.0 * (headline_width + 24.0);
        match prefill {
            Some(p) => big(doc, prefill_x, "8K PREFILL", &rate(p.tok_s), "tps",
                &format!("TTFT {} · {} tokens", seconds(p.ttft_s), super::grouped(p.prompt_tokens as f64)), t.series[3]),
            None => big(doc, prefill_x, "8K PREFILL", "—", "", "not measured", t.series[3]),
        }
        let (bx, by) = (0.0, 100.0);
        let bw = w - bx;
        doc.text(bx, by + 12.0, Font::new(10.0, t.muted).spacing(1.4), "C1 DECODE BY CONTENT");
        let max = b.card.decode.iter().map(|d| d.tok_s).fold(1.0f64, f64::max) * 1.12;
        for (i, d) in b.card.decode.iter().enumerate() {
            let y = by + 26.0 + 20.0 * i as f64;
            let color = content_color(t, &d.content);
            doc.text(bx, y + 11.0, Font::new(11.0, t.ink2), &d.content);
            let track = bw - 150.0;
            doc.rect(bx + 70.0, y + 2.0, track, 12.0, 3.0, t.well);
            doc.rect_attrs(bx + 70.0, y + 2.0, track * (d.tok_s / max).clamp(0.0, 1.0), 12.0, 3.0,
                &format!(r#"fill="{color}" opacity="{dim}""#));
            doc.text(bx + bw, y + 12.0, Font::new(12.0, t.ink).anchor(Anchor::End).opacity(dim),
                &format!("{warn}{}", rate(d.tok_s)));
        }
        let extra = by;
        doc.text(0.0, 102.0 + extra, Font::new(11.0, t.ink2), &fit(&self.report.capacity().line(), 11.0, w));
        // Quick quality.
        let mut y = 116.0 + extra;
        doc.line(0.0, y, w, y, t.line, 1.0);
        y += 8.0;
        let q = &b.quality;
        let verdict = match q.status {
            CheckStatus::Pass => "QUICK QUALITY · PASS",
            CheckStatus::Fail => "QUICK QUALITY · FIDELITY FAILURE",
            _ => "QUICK QUALITY · NOT GATED",
        };
        doc.text(0.0, y + 12.0, Font::new(10.0, t.status(q.status)).spacing(1.4).weight(600), verdict);
        doc.text(w, y + 12.0, Font::new(10.0, t.muted).anchor(Anchor::End),
            &format!("baseline {} · {}", seconds(b.seconds), &b.fingerprint));
        y += 22.0;
        for check in &q.checks {
            let color = t.status(check.status);
            if check.status == CheckStatus::Fail {
                doc.rect_attrs(-6.0, y - 2.0, w + 12.0, 22.0, 4.0, &format!(r#"fill="{}" opacity="0.18""#, t.bad));
                doc.rect_attrs(-12.0, y - 2.0, 5.0, 22.0, 0.0, r#"fill="url(#hazard)""#);
            }
            doc.text(0.0, y + 13.0, Font::new(13.0, color).bold(), check.status.symbol());
            doc.text(22.0, y + 13.0, Font::new(12.0, t.ink).weight(600), &check.title);
            doc.text(230.0, y + 13.0, Font::new(11.5, t.ink2), &fit(&check.detail(), 11.5, w - 300.0));
            if check.seconds > 0.0 {
                doc.text(w, y + 13.0, Font::new(10.0, t.muted).anchor(Anchor::End), &seconds(check.seconds));
            }
            y += 22.0;
            if check.id == "fidelity" {
                if let Some(precision) = check.metrics.get("precision")
                    .and_then(|v| serde_json::from_value::<crate::fidelity_precision::Precision>(v.clone()).ok()) {
                    for line in wrap(&precision.label(), 10.5, w - 22.0) {
                        doc.text(22.0, y + 11.0, Font::new(10.5, t.ink2), &line);
                        y += 16.0;
                    }
                    y += 4.0;
                }
            }
        }
        y
    }

    fn generic(&self, doc: &mut Doc, w: f64, panel: Option<&PanelResult>) -> f64 {
        let t = &self.theme;
        let Some(panel) = panel else {
            doc.text(0.0, 16.0, Font::new(12.0, t.ink2), "not run");
            return 24.0;
        };
        let mut y = 0.0;
        if let Some(error) = &panel.error {
            y += row(doc, t, 0.0, y, w, "Error", error);
        }
        for (i, pass) in panel.passes.iter().enumerate() {
            let summary = match pass {
                Value::Object(map) => map.iter().take(10).map(|(k, v)| format!("{k}={}", compact(v)))
                    .collect::<Vec<_>>().join(" · "),
                other => compact(other),
            };
            y += row(doc, t, 0.0, y, w, &format!("Pass {}", i + 1), &summary);
        }
        y.max(20.0)
    }
}

fn compact(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.as_f64().map_or_else(|| n.to_string(), |v| if v.fract() == 0.0 { format!("{v:.0}") } else { format!("{v:.3}") }),
        Value::Array(a) => format!("[{}]", a.len()),
        Value::Object(o) => format!("{{{}}}", o.len()),
        other => other.to_string(),
    }
}

pub fn content_color(theme: &Theme, content: &str) -> &'static str {
    match content {
        "code" => theme.series[0],
        "prose" => theme.series[1],
        "json" | "structured" => theme.series[2],
        "math" => theme.series[3],
        "chat" => theme.series[5],
        "translation" => theme.series[6],
        "summary" => theme.series[7],
        _ => theme.series[4],
    }
}

/// The watermark of a report whose quality gate failed.
pub fn watermark(doc: &mut Doc, theme: &Theme) {
    if !theme.scary {
        return;
    }
    let text = "UNVERIFIED — QUALITY GATE FAILED";
    let size = (doc.width / 22.0).clamp(22.0, 52.0);
    let step = size * 5.0;
    let mut y = size * 2.5;
    doc.group_attrs(r#"pointer-events="none""#);
    while y < doc.height + step {
        let x = doc.width / 2.0;
        doc.raw(&format!(r#"<text x="{x:.0}" y="{y:.0}" transform="rotate(-18 {x:.0} {y:.0})" text-anchor="middle" font-family="{}" font-size="{size:.0}" font-weight="700" fill="{}" opacity="0.13" letter-spacing="3">{}</text>"#,
            super::svg::MONO, theme.bad, super::svg::escape(text)));
        y += step;
    }
    doc.end();
}

/// The FIDELITY FAILURE banner (scary mode only); returns its height.
pub fn failure_banner(doc: &mut Doc, theme: &Theme, x: f64, y: f64, width: f64, detail: &str) -> f64 {
    if !theme.scary {
        return 0.0;
    }
    doc.rect_attrs(x, y, width, 44.0, 8.0, r#"fill="url(#hazard)""#);
    doc.rect(x + 10.0, y + 7.0, width - 20.0, 30.0, 5.0, "#140405");
    doc.text(x + 22.0, y + 27.0, Font::new(15.0, theme.bad).bold().spacing(3.0), "⚠ FIDELITY FAILURE");
    doc.text(x + width - 22.0, y + 27.0, Font::new(11.0, theme.warn).anchor(Anchor::End), &fit(detail, 11.0, width - 300.0));
    52.0
}

/// A glitching title: the text with offset red and cyan ghosts (scary mode).
pub fn title(doc: &mut Doc, theme: &Theme, x: f64, y: f64, size: f64, text: &str, max_width: f64) {
    // Long model ids shrink (to 70%) before they are cut.
    let size = size.min((max_width / (0.6 * text.chars().count().max(1) as f64)).max(size * 0.7));
    let text = fit(text, size, max_width);
    if theme.scary {
        doc.text(x - 2.5, y + 0.5, Font::new(size, "#00e5ff").bold().opacity(0.55), &text);
        doc.text(x + 2.5, y - 0.5, Font::new(size, theme.bad).bold().opacity(0.75), &text);
        doc.rect_attrs(x, y - size * 0.42, text_width(&text, size) * 0.7, 2.0, 0.0,
            &format!(r#"fill="{}" opacity="0.8""#, theme.bg));
    }
    doc.text(x, y, Font::new(size, theme.ink).bold(), &text);
}

/// Panel titles and hints by id (renderers outside the catalog: the baseline).
pub fn panel_title(id: &str) -> (&'static str, &'static str) {
    match id {
        "baseline" => ("Basic card", "C1 decode by content (thinking off), C1/C4/C8/C16 aggregate sweep, 8K prefill, quick quality"),
        _ => match crate::panels::find(id) {
            Some(panel) => (panel.title(), ""),
            None => ("Panel", ""),
        },
    }
}

/// Whether panel `id` failed its gate in this report.
pub fn failed(report: &Report, id: &str) -> bool {
    id == "baseline" && report.quality_failed()
}

/// The check list of a baseline as a short one-line summary.
pub fn quality_line(baseline: &Baseline) -> String {
    let failed: Vec<&str> = baseline.quality.checks.iter().filter(|c| c.status == CheckStatus::Fail)
        .map(|c| c.title.as_str()).collect();
    if failed.is_empty() { baseline.quality.badge() } else { format!("failed: {}", failed.join(", ")) }
}
