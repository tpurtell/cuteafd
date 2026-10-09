//! Single-panel exports and the whole-report page.
use super::bodies::{self, View, LOGO, LOGO_ASPECT};
use super::svg::{fit, Anchor, Doc, Font};
use crate::panels::ALWAYS;
use crate::report::Report;

/// Width of single-panel exports.
pub const PANEL_WIDTH: f64 = 960.0;
/// Width of the whole-report page.
pub const REPORT_WIDTH: f64 = 1000.0;

/// Panel ids a report shows, in order: hardware, configuration, the baseline,
/// then every panel that ran.
pub fn shown(report: &Report) -> Vec<String> {
    let mut ids: Vec<String> = ALWAYS.iter().map(|s| s.to_string()).collect();
    if report.no_fit_reason().is_none() {
        ids.push("baseline".into());
    }
    for panel in &report.panels {
        if !ids.contains(&panel.id) && !panel.passes.is_empty() {
            ids.push(panel.id.clone());
        }
    }
    ids
}

/// One panel as a standalone SVG (fixed width, its own footer).
pub fn panel_svg(report: &Report, id: &str) -> String {
    let view = View::new(report);
    let t = view.theme;
    let (title, hint) = bodies::panel_title(id);
    // Measure, then draw at the measured height.
    let mut probe = Doc::new(PANEL_WIDTH, 0.0);
    let height = view.framed(&mut probe, 16.0, 16.0, PANEL_WIDTH - 32.0, id, title, hint, bodies::failed(report, id));
    let total = height + 16.0 + 38.0;
    let mut doc = Doc::new(PANEL_WIDTH, total);
    doc.defs(&t.defs());
    doc.rect(0.0, 0.0, PANEL_WIDTH, total, 14.0, t.bg);
    view.framed(&mut doc, 16.0, 16.0, PANEL_WIDTH - 32.0, id, title, hint, bodies::failed(report, id));
    view.footer(&mut doc, 20.0, total - 14.0, PANEL_WIDTH - 40.0);
    bodies::watermark(&mut doc, &t);
    doc.finish()
}

/// A panel's body alone (no frame, title or footer) `width` wide on a
/// transparent background: the dashboard's chart view of the panel.
pub fn panel_body_svg(report: &Report, id: &str, width: f64) -> String {
    let view = View::new(report);
    let t = view.theme;
    let (body, height) = view.body(id, width);
    let mut doc = Doc::new(width, height + 4.0);
    doc.defs(&t.defs());
    doc.raw(&body);
    bodies::watermark(&mut doc, &t);
    doc.finish()
}

/// The whole report: header, failure banner, the panels that ran, footer.
pub fn report_svg(report: &Report) -> String {
    let view = View::new(report);
    let t = view.theme;
    let w = REPORT_WIDTH;
    let ids = shown(report);
    let gap = 14.0;
    let rows = super::layout::pack(&ids, w - 40.0, gap);
    let heights: Vec<f64> = rows.iter().map(|row| row.iter()
        .map(|cell| view.framed_height(cell.width, &cell.id)).fold(0.0, f64::max)).collect();
    let banner = if t.scary || report.no_fit_reason().is_some() { 52.0 } else { 0.0 };
    let header = 124.0;
    let footer = 64.0;
    let total = header + banner + heights.iter().map(|h| h + gap).sum::<f64>() + footer;
    let mut doc = Doc::new(w, total);
    doc.defs(&t.defs());
    doc.rect(0.0, 0.0, w, total, 16.0, t.bg);
    doc.rect_attrs(0.0, 0.0, w, total, 16.0, r#"fill="url(#glow-a)""#);
    doc.rect_attrs(0.0, 0.0, w, total, 16.0, r#"fill="url(#glow-b)""#);
    // Header: logo, model, profile/date/hardware, quality badge.
    let logo_h = 34.0;
    doc.nested(LOGO, 24.0, 20.0, logo_h * LOGO_ASPECT, logo_h);
    let r = report;
    let x = 24.0;
    bodies::title(&mut doc, &t, x, 88.0, 22.0, &r.server.checkpoint(), w - 360.0);
    let profile = crate::profiles::title_of(&r.profile);
    let sub = format!("{profile} · {} · {}", super::date(&r.created), r.server.hardware_line());
    doc.text(x, 108.0, Font::new(11.5, t.ink2), &fit(&sub, 11.5, w - 300.0));
    if let Some(baseline) = &r.baseline {
        let badge = baseline.quality.badge();
        let color = t.status(baseline.quality.status);
        let bw = (badge.chars().count() as f64 * 6.6 + 28.0).min(380.0);
        doc.rect_attrs(w - 24.0 - bw, 30.0, bw, 26.0, 13.0, &format!(r#"fill="{}" stroke="{color}""#, t.panel2));
        doc.text(w - 24.0 - bw / 2.0, 47.5, Font::new(11.0, color).anchor(Anchor::Middle).weight(600),
            &fit(&badge, 11.0, bw - 16.0));
    }
    doc.text(w - 24.0, 108.0, Font::new(10.5, t.muted).anchor(Anchor::End),
        &format!("fingerprint {}", r.fingerprint));
    let mut y = header;
    if let Some(reason) = r.no_fit_reason() {
        doc.rect(20.0, y, w - 40.0, 44.0, 8.0, t.panel2);
        doc.text(34.0, y + 27.0, Font::new(13.0, t.ink).weight(600),
            &fit(&format!("Doesn't fit: {reason}"), 13.0, w - 68.0));
        y += 52.0;
    } else if t.scary {
        let detail = r.baseline.as_ref().map(bodies::quality_line).unwrap_or_default();
        y += bodies::failure_banner(&mut doc, &t, 20.0, y, w - 40.0, &detail);
    }
    for (row, height) in rows.iter().zip(&heights) {
        for cell in row {
            let (title, hint) = bodies::panel_title(&cell.id);
            view.framed_at(&mut doc, 20.0 + cell.x, y, cell.width, Some(*height), &cell.id, title, hint,
                bodies::failed(report, &cell.id));
        }
        y += height + gap;
    }
    // Footer: build provenance and run identity.
    let b = &r.server.build;
    let mut provenance = format!("build {}", b.label());
    if let Some(remote) = &b.remote {
        provenance.push_str(&format!(" · {remote}"));
    }
    if let Some(commit) = &b.commit {
        provenance.push_str(&format!(" @ {}", &commit[..commit.len().min(12)]));
        if b.dirty == Some(true) {
            provenance.push_str(" (dirty)");
        }
    }
    doc.line(24.0, y + 8.0, w - 24.0, y + 8.0, t.line, 1.0);
    doc.text(24.0, y + 30.0, Font::new(10.5, t.ink2), &fit(&provenance, 10.5, w - 260.0));
    doc.text(24.0, y + 47.0, Font::new(10.0, t.muted),
        &format!("cuteafd bench · run {} · {}", &r.id[..r.id.len().min(8)], r.created));
    doc.text(w - 24.0, y + 47.0, Font::new(10.0, t.muted).anchor(Anchor::End), "github.com/tpurtell/cuteafd");
    bodies::watermark(&mut doc, &t);
    doc.finish()
}
