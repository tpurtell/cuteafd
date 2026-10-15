//! `cuteafd bench publish`: the root README's results table (the basic
//! profile per family, checkpoint and reference hardware) and the
//! `benchmarks/README.md` index, both rebuilt from the `report.json` files
//! under `benchmarks/<family>/<date>-<profile>-<hardware>/`.
use crate::render::{rate, seconds};
use crate::report::{CheckStatus, Report};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const RESULTS_BEGIN: &str = "<!-- results:begin -->";
pub const RESULTS_END: &str = "<!-- results:end -->";
pub const INDEX_BEGIN: &str = "<!-- reports:begin -->";
pub const INDEX_END: &str = "<!-- reports:end -->";

/// A report placed for publication.
#[derive(Debug, Clone)]
pub struct Placed {
    /// `benchmarks/<family>/<dir>` relative to the repository root.
    pub dir: PathBuf,
    pub report: Report,
}

pub fn family_title(family: &str) -> &str {
    match family {
        "deepseek_v41" => "DeepSeek V4.1",
        "deepseek_v4" => "DeepSeek V4",
        "glm5" => "GLM 5.3",
        "glm5_flash" => "GLM 5.3 Flash",
        "mimo_v2" => "MiMo V2",
        "qwen4" => "Qwen 3.8",
        other => other,
    }
}

/// Every `benchmarks/*/*/report.json` under `root`.
pub fn scan(root: &Path) -> Result<Vec<Placed>> {
    let mut placed = Vec::new();
    let base = root.join("benchmarks");
    let Ok(families) = std::fs::read_dir(&base) else { return Ok(placed) };
    for family in families.flatten().filter(|e| e.path().is_dir()) {
        for dir in std::fs::read_dir(family.path())?.flatten().filter(|e| e.path().is_dir()) {
            let path = dir.path().join("report.json");
            if !path.is_file() {
                continue;
            }
            let text = std::fs::read_to_string(&path)?;
            let mut report: Report = serde_json::from_str(&text).with_context(|| format!("{}", path.display()))?;
            // Publish under the rule in force now, not the one the report was
            // written under: a stored status is data, and the checkpoint's own
            // numerics decide whether absolute fidelity was a gate.
            crate::quality::reconcile(&mut report);
            let relative = dir.path().strip_prefix(root).unwrap_or(&dir.path()).to_path_buf();
            placed.push(Placed { dir: relative, report });
        }
    }
    placed.sort_by(|a, b| b.report.created.cmp(&a.report.created).then(a.dir.cmp(&b.dir)));
    Ok(placed)
}

fn coordinator_budget(report: &Report) -> Option<f64> {
    report.server.coordinator_budget()
}

fn short_hardware(report: &Report) -> String {
    let hw = &report.server.hardware;
    if report.server.simulated_5090() {
        return report.server.hardware_line();
    }
    let mut out = if report.server.column_5090() {
        "RTX 5090".into()
    } else {
        format!("{}× RTX", hw.used_gpus())
    };
    if !report.server.simulated_5090() {
        if let Some(gib) = coordinator_budget(report) {
            out.push_str(&format!(" ({gib} GiB budget)"));
        }
    }
    if !hw.sparks.is_empty() {
        out.push_str(&format!(" + {}× Spark", hw.sparks.len()));
    }
    out
}

/// `GLM-5.3-EXL3-K4-v1 (exl3)`: the checkpoint and its routed-expert format.
fn checkpoint(report: &Report) -> String {
    let checkpoint = report.server.checkpoint();
    let name = checkpoint.rsplit('/').next().unwrap_or(&checkpoint).to_string();
    let experts: Vec<String> = report.server.configuration.quant.iter().filter(|q| q.group.contains("routed"))
        .flat_map(|q| q.formats.clone()).collect();
    if experts.is_empty() { name } else { format!("{name} ({})", experts.join("+")) }
}

fn link(dir: &Path, file: &str) -> String {
    format!("{}/{file}", dir.display()).replace('\\', "/")
}

/// Family display order.
const FAMILIES: [&str; 6] = ["deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4"];

fn layout_label(p: &Placed) -> Option<&str> {
    p.dir.file_name()?.to_str()?.rsplit_once(&p.report.server.hardware.slug())
        .map(|(_, label)| without_release_candidate(label.trim_start_matches('-'))).filter(|label| !label.is_empty())
}

fn without_release_candidate(name: &str) -> &str {
    name.rsplit_once("-rc").filter(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        .map_or(name, |(name, _)| name)
}

// Smoke matrix entry names are appended by cli::labeled_dir, not stored in report.id.
fn reference_layout(p: &Placed) -> Option<u8> {
    let name = without_release_candidate(p.dir.file_name()?.to_str()?);
    if name.ends_with("-min") { Some(1) }
    else if name.ends_with("-max") { Some(2) }
    else { None }
}

fn basic_groups(placed: &[Placed]) -> BTreeMap<(usize, String, String), Vec<&Placed>> {
    // The newest basic-profile report per family, checkpoint and hardware (`placed`
    // is newest first), on the reference layouts (one or two RTX, Sparks or none).
    // A qualified replacement can retire a checkpoint from current cards without
    // deleting its historical reports or removing it from the benchmark index.
    let superseded: Vec<_> = placed.iter().filter(|p| {
        p.report.baseline.as_ref().is_some_and(|b| b.quality.status == CheckStatus::Pass)
            && matches!(p.report.profile.as_str(), "smoke" | "share")
    }).flat_map(|p| p.report.server.configuration.settings.iter()
        .filter(|s| s.name == "supersedes-checkpoint")
        .filter_map(|s| s.value.as_ref())
        .filter(|name| name.as_str() != p.report.server.checkpoint())
        .map(|name| (p.report.server.family.clone(), name.clone()))).collect();
    let mut newest: BTreeMap<(usize, String, String, String), &Placed> = BTreeMap::new();
    for p in placed {
        let r = &p.report;
        if (r.baseline.is_none() && !(r.no_fit_reason().is_some() && r.server.column_5090()))
            || !matches!(r.profile.as_str(), "smoke" | "share")
            || !(1..=2).contains(&r.server.hardware.used_gpus()) {
            continue;
        }
        if superseded.iter().any(|(family, name)| *family == r.server.family && *name == r.server.checkpoint()) {
            continue;
        }
        let family = r.server.family.clone().unwrap_or_else(|| "unknown".into());
        let order = FAMILIES.iter().position(|f| *f == family).unwrap_or(FAMILIES.len());
        let mut hardware = format!("{}-{}", if r.server.simulated_5090() { "sim5090" }
            else if r.server.column_5090() { "5090" } else { "pro" }, r.server.hardware.slug());
        if let Some(gib) = coordinator_budget(r) {
            hardware.push_str(&format!("-budget{gib}"));
        }
        // Keep labelled extra layouts even when they use the reference hardware.
        if let Some(label) = layout_label(p) {
            hardware.push('-');
            hardware.push_str(label);
        }
        newest.entry((order, family, checkpoint(r), hardware)).and_modify(|current| {
            if (p.report.created.as_str(), &p.dir) > (current.report.created.as_str(), &current.dir) { *current = p; }
        }).or_insert(p);
    }
    let mut groups: BTreeMap<(usize, String, String), Vec<&Placed>> = BTreeMap::new();
    for ((order, family, name, _), p) in newest {
        groups.entry((order, family, name)).or_default().push(p);
    }
    groups
}

/// The newest basic-profile report per family, checkpoint and reference hardware.
fn reference_rows(placed: &[Placed]) -> Vec<(String, String, u8, &Placed)> {
    let size = |p: &Placed| (p.report.server.hardware.used_gpus(), p.report.server.hardware.sparks.len());
    let mut rows = Vec::new();
    for ((_, family, name), reports) in basic_groups(placed) {
        // Real measurements always replace memory-only simulations, even older ones.
        fn newest_of(items: Vec<&Placed>) -> Option<&Placed> {
            items.into_iter().max_by(|a, b|
                a.report.created.cmp(&b.report.created).then(a.dir.cmp(&b.dir)))
        }
        let real = newest_of(reports.iter().copied().filter(|p|
            p.report.server.column_5090() && !p.report.server.simulated_5090()).collect());
        let simulated = newest_of(reports.iter().copied().filter(|p|
            p.report.server.simulated_5090()).collect());
        if let Some(p) = real.or(simulated) {
            rows.push((family.clone(), name.clone(), 0, p));
        }
        let pro: Vec<_> = reports.into_iter().filter(|p| !p.report.server.column_5090()).collect();
        let one: Vec<_> = pro.iter().copied().filter(|p| size(p).0 == 1).collect();
        let minimum = if one.iter().any(|p| reference_layout(p) == Some(1)) {
            newest_of(one.iter().copied().filter(|p| reference_layout(p) == Some(1)).collect())
        } else {
            one.iter().map(|p| size(p).1).min().and_then(|count|
                newest_of(one.iter().copied().filter(|p| size(p).1 == count).collect()))
        };
        if let Some(p) = minimum {
            rows.push((family.clone(), name.clone(), 1, p));
        }
        let two: Vec<_> = pro.iter().copied().filter(|p| size(p).0 == 2).collect();
        let maximum = if two.iter().any(|p| reference_layout(p) == Some(2)) {
            newest_of(two.iter().copied().filter(|p| reference_layout(p) == Some(2)).collect())
        } else {
            two.iter().map(|p| size(p).1).max().and_then(|count|
                newest_of(two.iter().copied().filter(|p| size(p).1 == count).collect()))
        };
        if let Some(p) = maximum {
            rows.push((family, name, 2, p));
        }
    }
    rows
}

fn column_name(class: u8) -> &'static str {
    match class { 0 => "5090", 1 => "1× RTX", _ => "2× RTX" }
}

fn quality_cell(report: &Report) -> String {
    let shown = crate::quality::shown(report, "quality");
    match shown.status {
        CheckStatus::Fail => format!("⚠ **FAILED** {}", if shown.fail_line.is_empty() { shown.verdict } else { shown.fail_line }),
        CheckStatus::Pass => format!("✓ {}", shown.verdict),
        _ => shown.verdict,
    }
}

// The caption is the policy's displayed caveat, not the stored qualification note.
// Escape text and restrict links to archived reports.
fn caveat(p: &Placed) -> String {
    let shown = crate::quality::shown(&p.report, "quality");
    if shown.caveat.is_empty() { return String::new(); }
    let escaped = crate::render::svg::escape(&shown.caveat);
    let mut text = match shown.caveat_lead.as_deref() {
        Some(lead) => {
            let lead = crate::render::svg::escape(lead);
            match escaped.strip_prefix(&lead) {
                Some(rest) => format!("<b>{lead}</b>{rest}"),
                None => escaped,
            }
        }
        None => escaped,
    };
    if let (Some(href), Some(label)) = (p.report.server.setting("qualification.caveat-link"),
        p.report.server.setting("qualification.caveat-link-label")) {
        if href.starts_with("benchmarks/") && !href.split('/').any(|part| part == "..") {
            text.push_str(&format!("<a href=\"{}\">{}</a>.", crate::render::svg::escape(href), crate::render::svg::escape(label)));
        }
    }
    format!("<br><sub>{text}</sub>")
}

/// The root README's results: one row per checkpoint (model and quant linking to
/// its family page, then its three reference-column cards, each linking to the
/// card SVG), then the same rows as a compact table.
pub fn results(placed: &[Placed]) -> String {
    let rows = reference_rows(placed);
    if rows.is_empty() {
        return "_Pending the first published run._\n".into();
    }
    // One row per checkpoint: the model and quant (linking to its family page),
    // then the 5090, one-RTX and two-RTX cards, each linking to the card itself.
    let mut out = String::from("<table>\n<tr><th>Model · quant</th><th>5090</th><th>1× RTX</th><th>2× RTX</th></tr>\n");
    let mut i = 0;
    while i < rows.len() {
        let (family, name, _, first) = &rows[i];
        let mut cells = [None, None, None];
        while i < rows.len() && rows[i].0 == *family && rows[i].1 == *name {
            cells[rows[i].2 as usize] = Some(rows[i].3);
            i += 1;
        }
        let card = |p: Option<&Placed>, column: u8| p.map_or(format!("<td width=\"27%\" align=\"center\">{}</td>",
            if column == 0 { "pending Hugh (TJ-T-3)" }
            else if column == 2 && family == "qwen4" { "n/a: fits one RTX (no two-GPU split for Qwen)" }
            else { "—" }), |p| {
            if let Some(reason) = p.report.no_fit_reason() {
                return format!("<td width=\"27%\" valign=\"top\"><a href=\"{}\">doesn't fit 1× 5090 (32 GB) + {} Sparks</a>\
                    <br><sub>{}</sub>{}</td>", link(&p.dir, "report.json"), p.report.server.hardware.sparks.len(),
                    crate::render::svg::escape(reason), caveat(p));
            }
            let svg = link(&p.dir, "card.svg");
            format!("<td width=\"27%\" valign=\"top\"><a href=\"{svg}\"><img src=\"{svg}\" alt=\"{} on {}\"></a>\
                <br><sub>{}</sub>{}</td>", p.report.server.checkpoint(), p.report.server.hardware_line(),
                short_hardware(&p.report), caveat(p))
        });
        out.push_str(&format!("<tr>\n<td width=\"19%\" valign=\"top\"><a href=\"docs/models/{family}.md\"><b>{}</b></a>\
            <br><sub>{}</sub><br><sub>{}</sub></td>\n{}\n{}\n{}\n</tr>\n", family_title(family),
            first.report.server.checkpoint(),
            name.rsplit_once(" (").map_or("", |(_, quant)| quant.trim_end_matches(')')),
            card(cells[0], 0), card(cells[1], 1), card(cells[2], 2)));
    }
    out.push_str("</table>\n\n");
    out.push_str("| Family | Checkpoint | Hardware | KV / req | C1 code | Concurrent code (aggregate) | prose | JSON | 8K prefill | TTFT | Quality | Report |\n");
    out.push_str("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n");
    let mut table_rows = rows.clone();
    for ((_, family, name), reports) in basic_groups(placed) {
        for p in reports {
            let current_build = rows.iter().filter(|row| row.0 == family && row.1 == name).any(|row| {
                let current = &row.3.report.server.build;
                let extra = &p.report.server.build;
                if current.release.is_some() { extra.release == current.release }
                else if current.commit.is_some() { extra.commit == current.commit }
                else { crate::render::date(&p.report.created) == crate::render::date(&row.3.report.created) }
            });
            // Keep current supplementary cards; older layouts remain in the archive index.
            if current_build && !p.report.server.column_5090() && !rows.iter().any(|row| row.3.dir == p.dir) {
                table_rows.push((family.clone(), name.clone(), 3, p));
            }
        }
    }
    table_rows.sort_by_key(|(family, name, class, p)|
        (FAMILIES.iter().position(|f| *f == family).unwrap_or(FAMILIES.len()), name.clone(), *class, p.dir.clone()));
    for (family, name, class, p) in &table_rows {
        let r = &p.report;
        let label = if *class == 3 { layout_label(p).unwrap_or("extra layout") }
            else { column_name(*class) };
        if let Some(reason) = r.no_fit_reason() {
            out.push_str(&format!("| [{}](docs/models/{family}.md) | {name} | {} ({}) | — | — | — | — | — | — | — | doesn't fit: {} | [planner report]({}) |\n",
                family_title(family), short_hardware(r), column_name(*class), reason.replace('|', "\\|"), link(&p.dir, "report.json")));
            continue;
        }
        let b = r.baseline.as_ref().expect("filtered");
        let decode = |content: &str| b.card.decode_of(content).map_or("—".into(), |d| rate(d.tok_s));
        let concurrent = r.server.card_levels().iter().filter_map(|width| b.card.concurrent_at(*width))
            .map(|c| format!("C{}: {}", c.width, rate(c.aggregate_tok_s))).collect::<Vec<_>>().join(" · ");
        let concurrent = if concurrent.is_empty() { "—".into() } else { concurrent };
        let (prefill, ttft) = b.card.prefill.as_ref()
            .map_or(("—".into(), "—".into()), |p| (rate(p.tok_s), seconds(p.ttft_s)));
        let hardware = r.server.setting("qualification.hardware-label").map(str::to_string)
            .unwrap_or_else(|| format!("{} ({label})", short_hardware(r)));
        out.push_str(&format!("| [{}](docs/models/{family}.md) | {name} | {hardware} | {} | {} | {concurrent} | {} | {} | {prefill} | {ttft} | {} | \
            [{} · {}]({}) |\n", family_title(family), r.capacity().compact(), decode("code"), decode("prose"),
            decode("json"), quality_cell(r), crate::render::date(&r.created), r.server.build.label(),
            link(&p.dir, "report.svg")));
    }
    out.push_str("\ntok/s; C1 decode and concurrent code aggregate with thinking off (C1/C4/C8/C16 on PRO 6000; through C8 at ≤32 GiB, within server admission), \
        8K prefill cold. Quality: logit fidelity against the family golden \
        reference, prefix-cache restore exactness, lossless speculation. Simulated 5090 reports cap RTX PRO 6000 memory only; \
        SM count (188 vs 170), L2, clocks and power are not emulated. Their speed is indicative and likely optimistic.\n");
    out
}

/// A family page's basic-eval cell: its cards (paths relative to `docs/models/`).
pub fn family_cards(placed: &[Placed], family: &str) -> String {
    reference_rows(placed).into_iter().filter(|(f, ..)| f == family).map(|(_, name, class, p)| {
        format!("<a href=\"../../{}\"><img src=\"../../{}\" width=\"360\" alt=\"{name} ({})\"></a>",
            link(&p.dir, "report.svg"), link(&p.dir, "card.svg"), column_name(class))
    }).collect::<Vec<_>>().join(" ")
}

/// The benchmarks/README.md index: per family, newest first.
pub fn index(placed: &[Placed]) -> String {
    let mut by_family: BTreeMap<String, Vec<&Placed>> = BTreeMap::new();
    for p in placed {
        by_family.entry(p.report.server.family.clone().unwrap_or_else(|| "unknown".into())).or_default().push(p);
    }
    let mut out = String::new();
    for (family, reports) in by_family {
        out.push_str(&format!("\n## {}\n\n", family_title(&family)));
        for p in reports {
            let r = &p.report;
            let relative = p.dir.strip_prefix("benchmarks").unwrap_or(&p.dir);
            let mut line = format!("- {} · {} · {} · {} · build {} · [report]({})",
                crate::render::date(&r.created), crate::profiles::title_of(&r.profile), r.server.checkpoint(),
                r.server.hardware_line(), r.server.build.label(), link(relative, "report.svg"));
            if let Some(reason) = r.no_fit_reason() {
                line.push_str(&format!(" · doesn't fit: {reason}"));
            } else if crate::quality::shown(r, "quality").status == CheckStatus::Fail {
                line.push_str(" · ⚠ quality gate failed");
            }
            out.push_str(&line);
            out.push('\n');
        }
    }
    if out.is_empty() {
        out.push_str("\n_No reports yet._\n");
    }
    out
}

/// Replaces the text between `begin` and `end` in `text`.
pub fn splice(text: &str, begin: &str, end: &str, body: &str) -> Result<String> {
    let start = text.find(begin).with_context(|| format!("marker {begin} missing"))? + begin.len();
    let stop = text[start..].find(end).with_context(|| format!("marker {end} missing"))? + start;
    Ok(format!("{}\n{}\n\n{}", &text[..start], body.trim_end_matches('\n'), &text[stop..]))
}

/// Release tags (v2.0.0-rc3) share a major-version changelog row (v2).
fn release_row(tag: &str) -> Option<String> {
    let major = tag.strip_prefix('v').unwrap_or(tag).split(['.', '-']).next()?;
    if major.is_empty() || !major.bytes().all(|c| c.is_ascii_digit()) { return None; }
    Some(format!("v{major}"))
}

fn report_release(p: &Placed) -> Option<String> {
    p.report.server.build.release.as_deref().and_then(release_row)
}

fn changelog_cards(text: &str, release: &str, date: &str, cards: &str) -> Result<String> {
    let row = format!("| {release} |");
    let mut lines: Vec<String> = text.split_inclusive('\n').map(str::to_string).collect();
    if let Some(line) = lines.iter_mut().find(|line| line.starts_with(&row)) {
        let cells: Vec<&str> = line.trim_end().trim_end_matches('|').split('|').collect();
        if cells.len() < 5 { bail!("malformed {release} changelog row"); }
        let newline = if line.ends_with("\r\n") { "\r\n" } else if line.ends_with('\n') { "\n" } else { "" };
        *line = format!("{}| {cards} |{newline}", cells[..cells.len() - 1].join("|"));
    } else {
        // Insert after the newest existing row; never borrow another release's cell.
        let newest = lines.iter().position(|line| {
            line.split('|').nth(1).is_some_and(|cell| {
                let cell = cell.trim();
                cell.starts_with('v') && release_row(cell).is_some()
            })
        });
        let after = newest.or_else(|| lines.iter().position(|line| line.starts_with("| ---")))
            .context("changelog table missing")?;
        if !lines[after].ends_with('\n') { lines[after].push('\n'); }
        lines.insert(after + 1, format!("| {release} | {date} | Published benchmarks | {cards} |\n"));
    }
    Ok(lines.concat())
}

/// Only the published release's cards update the family changelogs.
/// CUTEAFD_RELEASE_ROW remains an explicit override for manually placed reports.
fn family_pages(root: &Path, placed: &[Placed], release: Option<&str>) -> Result<()> {
    let override_row = std::env::var("CUTEAFD_RELEASE_ROW").ok();
    family_pages_with_row(root, placed, release, override_row.as_deref())
}

fn family_pages_with_row(root: &Path, placed: &[Placed], release: Option<&str>, override_row: Option<&str>) -> Result<()> {
    let Some(row) = override_row.or(release) else { return Ok(()) };
    let normalized_row = release_row(row);
    let selected = release.or(normalized_row.as_deref());
    let current: Vec<_> = placed.iter().filter(|p| {
        selected.is_some() && report_release(p).as_deref() == selected
            || (override_row.is_some() && report_release(p).is_none())
    }).cloned().collect();
    for family in FAMILIES {
        let path = root.join("docs/models").join(format!("{family}.md"));
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let newest = current.iter().filter(|p| p.report.server.family.as_deref() == Some(family)
            && matches!(p.report.profile.as_str(), "smoke" | "share")
            && (p.report.baseline.is_some() || p.report.no_fit_reason().is_some()))
            .max_by(|a, b| a.report.created.cmp(&b.report.created).then(a.dir.cmp(&b.dir)));
        let Some(newest) = newest else { continue };
        let cards = family_cards(&current, family);
        if cards.is_empty() { continue; }
        let out = changelog_cards(&text, row, &crate::render::date(&newest.report.created), &cards)?;
        crate::cli::write_if_changed(&path, &out)?;
    }
    Ok(())
}

const INDEX_HEADER: &str = "# Benchmarks\n\nReports from `cuteafd bench` (profiles other than the basic one run \
    when asked). Each directory holds `report.svg`, `report.json` and the share card; `cuteafd bench publish` \
    rebuilds this index and the root README's table.\n\n";

/// Rewrites card and report images already on disk when the displayed verdict
/// differs from the status stored in `report.json`. Missing images stay
/// missing, and the export itself is not edited.
fn refresh_changed_cards(root: &Path, placed: &[Placed]) -> Result<()> {
    for p in placed {
        let dir = root.join(&p.dir);
        let json_path = dir.join("report.json");
        let Ok(text) = std::fs::read_to_string(&json_path) else { continue };
        let raw: Report = serde_json::from_str(&text).with_context(|| format!("{}", json_path.display()))?;
        if !verdict_changed(&raw) { continue; }
        let report = &p.report;
        if dir.join("card.svg").is_file() {
            let svg = crate::render::card::card_svg(report);
            write_if_different(&dir.join("card.svg"), svg.as_bytes())?;
            if dir.join("card.png").is_file() {
                let png = crate::render::png::png(&svg, 1.0)?;
                write_if_different(&dir.join("card.png"), &png)?;
            }
        } else if dir.join("card.png").is_file() {
            let svg = crate::render::card::card_svg(report);
            let png = crate::render::png::png(&svg, 1.0)?;
            write_if_different(&dir.join("card.png"), &png)?;
        }
        if dir.join("report.svg").is_file() {
            let svg = crate::render::report::report_svg(report);
            write_if_different(&dir.join("report.svg"), svg.as_bytes())?;
        }
        let mut panels = Vec::new();
        for entry in std::fs::read_dir(&dir).with_context(|| format!("{}", dir.display()))? {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            if let Some(id) = name.strip_prefix("panel-").and_then(|rest| rest.strip_suffix(".svg")) {
                panels.push(id.to_string());
            }
        }
        for id in panels {
            let svg = crate::render::report::panel_svg(report, &id);
            write_if_different(&dir.join(format!("panel-{id}.svg")), svg.as_bytes())?;
        }
    }
    Ok(())
}

fn verdict_changed(raw: &Report) -> bool {
    let view = crate::quality::shown(raw, "quality");
    let Some(baseline) = &raw.baseline else {
        return raw.server.setting("qualification.caveat").unwrap_or("") != view.caveat;
    };
    baseline.quality.status != view.status
        || baseline.quality.badge() != view.verdict
        || raw.server.setting("qualification.caveat").unwrap_or("") != view.caveat
}

fn write_if_different(path: &Path, bytes: &[u8]) -> Result<()> {
    if std::fs::read(path).ok().as_deref() == Some(bytes) {
        return Ok(());
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

/// Checks `dirs` are placed for publication and rebuilds both documents.
pub fn publish(root: &Path, dirs: &[PathBuf]) -> Result<(PathBuf, PathBuf, usize)> {
    let mut releases = std::collections::BTreeSet::new();
    for dir in dirs {
        let absolute = if dir.is_absolute() { dir.clone() } else { std::env::current_dir()?.join(dir) };
        let base = root.canonicalize()?.join("benchmarks");
        let canonical = absolute.canonicalize().with_context(|| format!("{}", dir.display()))?;
        let Ok(relative) = canonical.strip_prefix(&base) else {
            bail!("{} is not under {}: published reports go in benchmarks/<family>/<date>-<profile>-<hardware>/",
                dir.display(), base.display());
        };
        if relative.components().count() != 2 || !canonical.join("report.json").is_file() {
            bail!("{} must be benchmarks/<family>/<date>-<profile>-<hardware>/ with a report.json", dir.display());
        }
        let report: Report = serde_json::from_str(&std::fs::read_to_string(canonical.join("report.json"))?)?;
        if let Some(release) = report.server.build.release.as_deref().and_then(release_row) {
            releases.insert(release);
        }
        let family = report.server.family.clone().unwrap_or_else(|| "unknown".into());
        let parent = relative.components().next().and_then(|c| c.as_os_str().to_str()).unwrap_or_default();
        if parent != family {
            bail!("{} holds a {family} report; move it under benchmarks/{family}/", dir.display());
        }
    }
    if releases.len() > 1 && std::env::var("CUTEAFD_RELEASE_ROW").is_err() {
        bail!("publish reports from one release at a time, or set CUTEAFD_RELEASE_ROW");
    }
    let placed = scan(root)?;
    refresh_changed_cards(root, &placed)?;
    let release = releases.into_iter().next().or_else(|| {
        dirs.is_empty().then(|| placed.iter().find_map(report_release)).flatten()
    });
    let readme = root.join("README.md");
    let text = std::fs::read_to_string(&readme).with_context(|| format!("{}", readme.display()))?;
    crate::cli::write_if_changed(&readme, &splice(&text, RESULTS_BEGIN, RESULTS_END, &results(&placed))?)?;
    let index_path = root.join("benchmarks/README.md");
    std::fs::create_dir_all(root.join("benchmarks"))?;
    let current = std::fs::read_to_string(&index_path)
        .unwrap_or_else(|_| format!("{INDEX_HEADER}{INDEX_BEGIN}\n{INDEX_END}\n"));
    crate::cli::write_if_changed(&index_path, &splice(&current, INDEX_BEGIN, INDEX_END, &index(&placed))?)?;
    family_pages(root, &placed, release.as_deref())?;
    Ok((readme, index_path, placed.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_local_minimum_is_not_displaced_by_legacy_budget_or_simulation() {
        let placed = |id: &str, sparks: usize, budget: Option<&str>| {
            let mut report = crate::sample::report(false);
            report.id = id.into();
            report.server.family = Some("qwen4".into());
            report.server.model = "org/Qwen3.8-EXL3".into();
            report.server.configuration.snapshot = None;
            report.server.configuration.quant[1].formats = vec!["exl3-k4".into(), "exl3-k5".into()];
            report.server.hardware.sparks.truncate(sparks);
            if let Some(value) = budget {
                report.server.configuration.settings.push(crate::report::Setting {
                    name: "coordinator-gpu-budget-gib".into(), value: Some(value.into()),
                    default: None, source: "cli".into(),
                });
            }
            Placed { dir: PathBuf::from(format!("benchmarks/qwen4/{id}")), report }
        };
        let legacy = placed("legacy", 4, None);
        let maximum = placed("maximum", 0, None);
        let unbudgeted = placed("unbudgeted", 1, None);
        let minimum = placed("minimum", 1, Some("32"));
        let mut simulated = placed("simulated", 1, Some("31.8"));
        simulated.report.server.configuration.settings.push(crate::report::Setting {
            name: "simulated".into(), value: Some("5090".into()), default: None, source: "publication".into(),
        });
        let reports = vec![unbudgeted, minimum, maximum, legacy];
        let with_simulation = [vec![simulated], reports.clone()].concat();
        assert_eq!(reference_rows(&with_simulation)[1].3.report.id, "maximum");
        let rows = reference_rows(&reports);
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].2, rows[0].3.report.id.as_str()), (1, "maximum"));
        assert!(short_hardware(&reports[1].report).contains("32 GiB budget"));
        assert!(crate::render::card::card_svg(&reports[1].report).contains("32 GiB budget"));
        let html = results(&reports);
        assert!(!html.contains("benchmarks/qwen4/minimum/card.svg"));
        assert!(html.contains("benchmarks/qwen4/maximum/card.svg"));
        assert!(html.contains("n/a: fits one RTX (no two-GPU split for Qwen)"));
        let v2 = vec![with_simulation[0].clone(), reports[2].clone()];
        let rows = reference_rows(&v2);
        assert_eq!(rows.iter().map(|r| (r.2, r.3.report.id.as_str())).collect::<Vec<_>>(),
            vec![(0, "simulated"), (1, "maximum")]);
        assert!(!html.contains("benchmarks/qwen4/legacy/card.svg"));
        // Historical one-GPU layouts never become a two-GPU maximum.
        let legacy_rows = reference_rows(&reports[2..]);
        assert_eq!((legacy_rows[0].2, legacy_rows[0].3.report.id.as_str()), (1, "maximum"));
        assert_eq!(legacy_rows.len(), 1, "one-RTX legacy layouts do not populate the two-RTX column");
    }

    fn marked(id: &str, gpus: usize, marker: Option<(&str, &str)>) -> Placed {
        let mut report = crate::sample::report(false);
        report.id = id.into();
        let gpu = report.server.hardware.gpus[0].clone();
        report.server.hardware.gpus = (0..gpus).map(|i| {
            let mut g = gpu.clone(); g.index = i as u32; g.used = true; g
        }).collect();
        if let Some((name, value)) = marker {
            report.server.configuration.settings.push(crate::report::Setting {
                name: name.into(), value: Some(value.into()), default: None, source: "publication".into(),
            });
        }
        Placed { dir: PathBuf::from(format!("benchmarks/deepseek_v41/{id}")), report }
    }

    #[test]
    fn fractional_budget_and_simulation_labels() {
        let mut p = marked("sim5090", 1, Some(("simulated", "5090")));
        p.report.server.hardware.gpus[0].sm_count = Some(188);
        p.report.server.configuration.settings.push(crate::report::Setting {
            name: "coordinator-gpu-budget-gib".into(), value: Some("31.8".into()),
            default: None, source: "cli".into(),
        });
        assert_eq!(coordinator_budget(&p.report), Some(31.8));
        assert!(short_hardware(&p.report).contains("simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB"));
        assert!(p.report.server.hardware_line().contains("RTX PRO 6000 (188 SMs) capped at 31.8 GiB"));
        assert!(crate::render::card::card_svg(&p.report).contains("simulated 5090"));
        p.report.server.configuration.settings.last_mut().unwrap().value = Some("NaN".into());
        assert_eq!(coordinator_budget(&p.report), None);
    }

    #[test]
    fn three_columns_use_markers_not_memory_layout() {
        let reports = vec![marked("sim", 1, Some(("simulated", "5090"))),
            marked("one", 1, None), marked("two", 2, None)];
        let rows = reference_rows(&reports);
        assert_eq!(rows.iter().map(|r| (r.2, r.3.report.id.as_str())).collect::<Vec<_>>(),
            vec![(0, "sim"), (1, "one"), (2, "two")]);
        let html = results(&reports);
        assert!(html.contains("<th>5090</th><th>1× RTX</th><th>2× RTX</th>"));
        assert_eq!(html.matches("<img src=").count(), 3);
        assert!(results(&reports[1..]).contains("pending Hugh (TJ-T-3)"));
    }

    #[test]
    fn real_5090_supersedes_even_newer_simulation_but_history_remains() {
        let mut sim = marked("sim", 1, Some(("simulated", "5090")));
        sim.report.created = "2026-10-09T00:00:00Z".into();
        let mut real = marked("real", 1, Some(("hardware.class", "5090")));
        real.report.created = "2026-10-08T00:00:00Z".into();
        let reports = vec![sim, real, marked("one", 1, None)];
        assert_eq!(reference_rows(&reports)[0].3.report.id, "real");
        assert!(!results(&reports).contains("/sim/card.svg"));
        assert!(index(&reports).contains("/sim/report.svg"));
    }

    #[test]
    fn no_fit_has_no_performance_and_real_5090_replaces_it() {
        let mut rejection = marked("no-fit-sim5090", 1, Some(("simulated", "5090")));
        rejection.report.status = crate::report::RunStatus::Failed;
        rejection.report.baseline = None;
        rejection.report.server.configuration.settings.push(crate::report::Setting {
            name: "qualification.no-fit".into(), value: Some("rtx0 weights < layout | over budget".into()),
            default: None, source: "planner".into(),
        });
        let html = results(&[rejection.clone()]);
        assert!(html.contains("doesn't fit 1× 5090 (32 GB) + 4 Sparks"));
        assert!(html.contains("weights &lt; layout"));
        assert!(html.contains("layout \\| over budget"));
        assert!(html.contains("no-fit-sim5090/report.json"));
        let card = crate::render::card::card_svg(&rejection.report);
        assert!(card.contains("Doesn&#39;t fit"));
        assert!(!card.contains("tok/s"));
        let report = crate::render::report::report_svg(&rejection.report);
        assert!(report.contains("Doesn&#39;t fit"));
        assert!(!crate::render::report::shown(&rejection.report).contains(&"baseline".into()));
        let mut real = marked("real", 1, Some(("hardware.class", "5090")));
        real.report.created = "2026-10-01T00:00:00Z".into();
        let reports = vec![rejection, real];
        assert_eq!(reference_rows(&reports)[0].3.report.id, "real");
        assert!(index(&reports).contains("no-fit-sim5090/report.svg"));
        let mut unmarked = reports[0].clone();
        unmarked.report.status = crate::report::RunStatus::Done;
        assert!(reference_rows(&[unmarked]).is_empty());
    }

    #[test]
    fn qualified_replacement_retires_cards_but_keeps_history() {
        let mut old = crate::sample::report(false);
        old.server.family = Some("mimo_v2".into());
        old.server.configuration.snapshot = None;
        old.server.model = "XiaomiMiMo/MiMo-V2-Flash".into();
        let mut replacement = old.clone();
        replacement.server.model = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD".into();
        replacement.server.configuration.settings.push(crate::report::Setting {
            name: "supersedes-checkpoint".into(), value: Some(old.server.checkpoint()),
            default: None, source: "publication".into(),
        });
        let mut other_family = old.clone();
        other_family.server.family = Some("qwen4".into());
        let mut placed = vec![
            Placed { dir: "benchmarks/mimo_v2/new".into(), report: replacement },
            Placed { dir: "benchmarks/mimo_v2/old".into(), report: old },
            Placed { dir: "benchmarks/qwen4/other".into(), report: other_family },
        ];
        let rows = reference_rows(&placed);
        assert_eq!(rows.len(), 2);
        assert!(!rows.iter().any(|(family, _, _, p)| family == "mimo_v2"
            && p.report.server.checkpoint() == "XiaomiMiMo/MiMo-V2-Flash"));
        assert!(family_cards(&placed, "mimo_v2").contains("/new/card.svg"));
        assert!(!family_cards(&placed, "mimo_v2").contains("/old/card.svg"));
        assert!(index(&placed).contains("MiMo-V2-Flash"));
        assert!(index(&placed).contains("MiMo-V2.6-Flash-MOPD"));
        // A failed or non-basic replacement never hides a previous qualified result.
        placed[0].report.baseline.as_mut().unwrap().quality.status = CheckStatus::Fail;
        assert_eq!(reference_rows(&placed).len(), 3);
        placed[0].report.baseline.as_mut().unwrap().quality.status = CheckStatus::Pass;
        placed[0].report.profile = "decode".into();
        assert_eq!(reference_rows(&placed).len(), 2);
        assert!(reference_rows(&placed).iter().any(|(family, _, _, p)| family == "mimo_v2"
            && p.report.server.checkpoint() == "XiaomiMiMo/MiMo-V2-Flash"));
    }

    #[test]
    fn matrix_minimum_beats_extra_layout_and_extras_keep_labelled_rows() {
        let mut minimum = marked("uuid-min", 1, None);
        minimum.report.server.family = Some("qwen4".into());
        minimum.report.server.hardware.sparks.truncate(2);
        minimum.dir = crate::cli::labeled_dir(Path::new("."), &minimum.report, Some("qwen38-exl3-min"));
        let mut extra = minimum.clone();
        extra.report.id = "uuid-extra".into();
        extra.report.created = "2026-10-10T00:00:00Z".into();
        extra.report.server.hardware.sparks.clear();
        extra.dir = crate::cli::labeled_dir(Path::new("."), &extra.report, Some("qwen38-exl3-nospark-vision"));
        let mut max = marked("uuid-max", 2, None);
        max.report.server.family = Some("qwen4".into());
        max.dir = crate::cli::labeled_dir(Path::new("."), &max.report, Some("qwen38-exl3-max"));
        let mut extra_two = max.clone();
        extra_two.report.server.hardware.sparks.push(extra_two.report.server.hardware.sparks[0].clone());
        extra_two.dir = crate::cli::labeled_dir(Path::new("."), &extra_two.report, Some("qwen38-exl3-extra"));
        let reports = vec![extra.clone(), minimum.clone(), extra_two.clone(), max.clone()];
        let rows = reference_rows(&reports);
        assert_eq!(rows.iter().map(|r| r.3.report.id.as_str()).collect::<Vec<_>>(), vec!["uuid-min", "uuid-max"]);
        let html = results(&reports);
        let (grid, table) = html.split_once("</table>").unwrap();
        assert!(grid.contains(&link(&minimum.dir, "card.svg")));
        assert!(grid.contains(&link(&max.dir, "card.svg")));
        assert!(!grid.contains(&link(&extra.dir, "card.svg")));
        assert!(!grid.contains(&link(&extra_two.dir, "card.svg")));
        assert!(table.contains("(qwen38-exl3-nospark-vision)"));
        assert!(table.contains(&link(&extra.dir, "report.svg")));
        assert!(table.contains(&link(&extra_two.dir, "report.svg")));
        let cards = family_cards(&reports, "qwen4");
        assert!(cards.contains(&link(&minimum.dir, "card.svg")));
        assert!(!cards.contains(&link(&extra.dir, "card.svg")));

        // An extra on identical hardware must not deduplicate away the reference.
        let mut same_hw = minimum.clone();
        same_hw.report.id = "uuid-same-hw".into();
        same_hw.dir = crate::cli::labeled_dir(Path::new("."), &same_hw.report, Some("qwen38-exl3-vision"));
        let html = results(&[same_hw.clone(), minimum]);
        assert!(html.contains("(qwen38-exl3-vision)"));
        assert!(html.contains(&link(&same_hw.dir, "report.svg")));
    }

    #[test]
    fn release_candidates_share_card_identity_and_newest_wins() {
        let mut old = marked("older", 1, None);
        old.dir = crate::cli::labeled_dir(Path::new("."), &old.report, Some("model-min"));
        let mut rc2 = old.clone();
        rc2.report.created = "2026-10-10T00:00:00Z".into();
        rc2.dir = crate::cli::labeled_dir(Path::new("."), &rc2.report, Some("model-min-rc2"));
        let mut rc3 = rc2.clone();
        rc3.report.id = "newest".into();
        rc3.report.created = "2026-10-11T00:00:00Z".into();
        rc3.dir = crate::cli::labeled_dir(Path::new("."), &rc3.report, Some("model-min-rc3"));
        assert_eq!(reference_layout(&rc3), Some(1));
        assert_eq!(layout_label(&rc3), Some("model-min"));
        let reports = [old, rc2, rc3.clone()];
        let rows = reference_rows(&reports);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].3.report.id, "newest");
        assert_eq!(basic_groups(&[rc3]).values().next().unwrap().len(), 1);
        assert_eq!(without_release_candidate("model-max-rc3"), "model-max");
        assert_eq!(without_release_candidate("model-min-rcfoo"), "model-min-rcfoo");
    }

    #[test]
    fn publication_caveats_are_escaped_report_data_for_fit_and_no_fit() {
        let mut p = marked("with-note-min", 1, None);
        let setting = |name: &str, value: &str| crate::report::Setting {
            name: name.into(), value: Some(value.into()), source: "publication".into(), ..Default::default()
        };
        p.report.server.configuration.settings.extend([
            setting("qualification.caveat", "FAIL: not recommended. KL > 0.06. With vision: "),
            setting("qualification.caveat-link", "benchmarks/qwen4/vision/report.svg"),
            setting("qualification.caveat-link-label", "vision <on>"),
        ]);
        assert_eq!(caveat(&p), "<br><sub><b>FAIL: not recommended.</b> KL &gt; 0.06. With vision: <a href=\"benchmarks/qwen4/vision/report.svg\">vision &lt;on&gt;</a>.</sub>");
        assert!(results(&[p.clone()]).contains(&caveat(&p)));
        p.report.server.configuration.settings.iter_mut().find(|s| s.name == "qualification.caveat-link").unwrap().value = Some("javascript:bad".into());
        assert!(!caveat(&p).contains("href="));
        p.report.baseline = None;
        p.report.status = crate::report::RunStatus::Failed;
        p.report.server.configuration.settings.push(setting("qualification.no-fit", "over budget"));
        p.report.server.configuration.settings.push(setting("simulated", "5090"));
        assert!(results(&[p.clone()]).contains(&caveat(&p)));
    }

    #[test]
    fn supplementary_cards_belong_to_the_current_release() {
        let mut current = marked("current", 1, None);
        current.report.server.build.release = Some("v2.0.0-rc3".into());
        current.dir = crate::cli::labeled_dir(Path::new("."), &current.report, Some("model-min-rc3"));
        let mut extra = current.clone();
        extra.dir = crate::cli::labeled_dir(Path::new("."), &extra.report, Some("model-vision-rc3"));
        let mut old = extra.clone();
        old.report.server.build.release = Some("v2.0.0-rc2".into());
        old.dir = crate::cli::labeled_dir(Path::new("."), &old.report, Some("model-vision-rc2"));
        let reports = [current, extra.clone(), old.clone()];
        let html = results(&reports);
        assert!(html.contains(&link(&extra.dir, "report.svg")));
        assert!(!html.contains(&link(&old.dir, "report.svg")));
        assert!(index(&reports).contains("model-vision-rc2/report.svg"));
        let mut labelled = extra;
        labelled.report.server.configuration.settings.push(crate::report::Setting {
            name: "qualification.hardware-label".into(), value: Some("vision on RTX0, fixed pool".into()),
            source: "publication".into(), ..Default::default()
        });
        assert!(results(&[labelled]).contains("| vision on RTX0, fixed pool |"));
    }

    #[test]
    fn reference_marker_fallback_is_per_gpu_column() {
        let mut one = marked("one", 1, None);
        let mut two = marked("two", 2, None);
        one.dir = crate::cli::labeled_dir(Path::new("."), &one.report, Some("model-min"));
        two.dir = crate::cli::labeled_dir(Path::new("."), &two.report, Some("model-rc2"));
        let check = |reports: &[Placed]| {
            assert_eq!(reference_rows(reports).iter().map(|r| (r.2, r.3.report.id.as_str()))
                .collect::<Vec<_>>(), vec![(1, "one"), (2, "two")]);
            assert_eq!(results(reports).matches("<img src=").count(), 2);
        };
        check(&[one.clone(), two.clone()]);
        one.dir = crate::cli::labeled_dir(Path::new("."), &one.report, Some("model-rc2"));
        two.dir = crate::cli::labeled_dir(Path::new("."), &two.report, Some("model-max"));
        check(&[one.clone(), two]);
        // Historical Qwen maxima used one GPU; they remain in the one-RTX column.
        one.dir = crate::cli::labeled_dir(Path::new("."), &one.report, Some("model-max"));
        assert_eq!(reference_rows(&[one])[0].2, 1);
    }

    #[test]
    fn publish_v2_preserves_previous_rows_and_adds_a_missing_row() {
        assert_eq!(release_row("v2.0.0-rc3"), Some("v2".into()));
        assert_eq!(release_row("2.0.0"), Some("v2".into()));
        assert_eq!(release_row("work/p0"), None);
        assert_eq!(release_row("v"), None);
        for existing in [false, true] {
            let root = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(root.path().join("docs/models")).unwrap();
            let page_path = root.path().join("docs/models/deepseek_v41.md");
            let v0 = "| v0 | 2026-10-02 | First release | <a href=\"v0.svg\">original v0</a> |\n";
            let v1 = "| v1 | 2026-10-05 | Second release | original v1 |\n";
            let v2 = if existing { "| v2 | 2026-10-10 | Third release | pending |\n" } else { "" };
            std::fs::write(&page_path, format!("## Changelog\n\n| Version | Date | Change | Basic eval |\n| --- | --- | --- | --- |\n{v2}{v1}{v0}\nOther prose\n")).unwrap();
            std::fs::write(root.path().join("README.md"), format!("{RESULTS_BEGIN}\n{RESULTS_END}\n")).unwrap();
            let mut old = marked("old", 2, None).report;
            old.server.build.release = Some("v1.0.0".into());
            old.created = "2026-10-05T00:00:00Z".into();
            crate::cli::write_exports(&old, &crate::cli::labeled_dir(root.path(), &old, Some("old-max")), &["json".into()]).unwrap();
            let mut new = marked("new", 1, None).report;
            new.server.build.release = Some("v2.0.0-rc3".into());
            new.created = "2026-10-10T00:00:00Z".into();
            let dir = crate::cli::labeled_dir(root.path(), &new, Some("new-min"));
            crate::cli::write_exports(&new, &dir, &["json".into()]).unwrap();
            publish(root.path(), &[dir]).unwrap();
            let page = std::fs::read_to_string(&page_path).unwrap();
            assert!(page.contains(v0), "{page}");
            assert!(page.contains(v1), "{page}");
            let row = page.lines().find(|line| line.starts_with("| v2 |")).unwrap();
            assert!(row.contains("new-min/card.svg"), "{row}");
            assert!(!row.contains("old-max"), "{row}");
            assert_eq!(page.matches("| v2 |").count(), 1);
            if !existing { assert!(page.find(v1).unwrap() < page.find(row).unwrap()); }
            publish(root.path(), &[]).unwrap();
            assert_eq!(std::fs::read_to_string(page_path).unwrap(), page);
        }
    }

    #[test]
    fn release_override_selects_destination_not_report_tag() {
        for row in ["v2-bringup", "v2.0.0-rc3", "custom-row"] {
            let root = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(root.path().join("docs/models")).unwrap();
            let path = root.path().join("docs/models/deepseek_v41.md");
            let old_row = "| v1 | date | change | old cards |\n";
            std::fs::write(&path, format!("| Version | Date | Change | Basic eval |\n| --- | --- | --- | --- |\n| {row} | date | change | pending |\n{old_row}")).unwrap();
            let mut current = marked("current-min", 1, None);
            current.report.server.build.release = Some("v2.0.0-rc3".into());
            let mut old = marked("old-max", 2, None);
            old.report.server.build.release = Some("v1.0.0".into());
            let reports = [current, old];
            family_pages_with_row(root.path(), &reports, Some("v2"), Some(row)).unwrap();
            let page = std::fs::read_to_string(&path).unwrap();
            let target = page.lines().find(|line| line.starts_with(&format!("| {row} |"))).unwrap();
            assert!(target.contains("current-min/card.svg"), "{target}");
            assert!(!target.contains("old-max"), "{target}");
            assert!(page.contains(old_row));
            if row != "custom-row" {
                family_pages_with_row(root.path(), &reports, None, Some(row)).unwrap();
                assert_eq!(std::fs::read_to_string(&path).unwrap(), page);
            }
        }
    }

    #[test]
    fn changelog_insertion_works_without_previous_releases() {
        let page = "## Changelog\n\n| Version | Date | Change | Basic eval |\n| --- | --- | --- | --- |\n\nTail\n";
        let out = changelog_cards(page, "v2", "2026-10-10", "new cards").unwrap();
        assert!(out.contains("| --- | --- | --- | --- |\n| v2 | 2026-10-10 | Published benchmarks | new cards |\n\nTail\n"));
        let crlf = "| v1 | old | old | old |\r\n| v2 | date | change | pending |\r\n";
        let out = changelog_cards(crlf, "v2", "date", "cards").unwrap();
        assert_eq!(out, "| v1 | old | old | old |\r\n| v2 | date | change | cards |\r\n");
    }

    #[test]
    fn publish_rebuilds_the_table_and_index() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("docs/models")).unwrap();
        std::fs::write(root.path().join("docs/models/deepseek_v41.md"),
            "## Changelog\n\n| Version | Date | Change | Basic eval |\n| --- | --- | --- | --- |\n| v0 | 2026-10-02 | First release | — |\n").unwrap();
        std::fs::write(root.path().join("README.md"),
            format!("# x\n\n{RESULTS_BEGIN}\n_Pending._\n{RESULTS_END}\n\nrest\n")).unwrap();
        let mut ok = crate::sample::report(false);
        ok.created = "2026-10-02T10:00:00Z".into();
        ok.server.build.release = Some("v0.1.0".into());
        let mut older = crate::sample::report(true);
        older.created = "2026-10-01T10:00:00Z".into();
        older.id = "older".into();
        for r in [&ok, &older] {
            let dir = crate::cli::default_dir(root.path(), r);
            let dir = if r.id == "older" { dir.with_file_name("2026-10-01-smoke-deepseek-v4-1-flash-1rtx-4spark") } else { dir };
            crate::cli::write_exports(r, &dir, &["json".into(), "svg".into()]).unwrap();
        }
        let dir = root.path().join("benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark");
        let (readme, index, count) = publish(root.path(), &[dir]).unwrap();
        assert_eq!(count, 2);
        let readme = std::fs::read_to_string(readme).unwrap();
        assert!(readme.contains("<img src=\"benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/card.svg\""), "{readme}");
        // One row per checkpoint × hardware: the newest wins.
        assert_eq!(readme.matches("| DeepSeek-V4.1-Flash (mxfp4) |").count(), 1, "{readme}");
        assert!(super::family_cards(&scan(root.path()).unwrap(), "deepseek_v41").contains("../../benchmarks/"));
        let page = std::fs::read_to_string(root.path().join("docs/models/deepseek_v41.md")).unwrap();
        assert!(page.contains("| v0 | 2026-10-02 | First release | <a href=\"../../benchmarks/"), "{page}");
        assert!(readme.contains("benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/report.svg"));
        assert!(readme.ends_with("rest\n"));
        let index = std::fs::read_to_string(index).unwrap();
        let first = index.find("2026-10-02").unwrap();
        assert!(first < index.find("2026-10-01").unwrap(), "newest first");
        assert!(index.contains("⚠ quality gate failed"));
        // Idempotent.
        let again = publish(root.path(), &[]).unwrap();
        assert_eq!(std::fs::read_to_string(again.0).unwrap(), readme);
        // Misplaced reports are refused.
        let stray = root.path().join("elsewhere");
        crate::cli::write_exports(&ok, &stray, &["json".into()]).unwrap();
        assert!(publish(root.path(), &[stray]).is_err());
    }

    /// A stored FAIL on a calibrated W4A4 export is rewritten into the images
    /// that already exist. Missing images stay missing, and report.json is
    /// left as measured.
    #[test]
    fn publish_rerenders_existing_images_only_when_the_displayed_verdict_changes() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("README.md"), format!("{RESULTS_BEGIN}\n{RESULTS_END}\n")).unwrap();
        let snap = root.path().join("models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/2061e0b0");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"),
            r#"{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16,"quantized_layers":{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4"}}}}"#)
            .unwrap();
        let mut report = crate::sample::report(true);
        report.server.model = "nvidia/Qwen3.8-Flash-Next-NVFP4".into();
        report.server.family = Some("qwen4".into());
        report.server.configuration.snapshot = Some(snap.display().to_string());
        report.server.configuration.settings.push(crate::report::Setting {
            name: "qualification.caveat".into(),
            value: Some("FAIL: not recommended. Fails the calibrated check.".into()),
            default: None, source: "publication".into(),
        });
        let dir = root.path().join("benchmarks/qwen4/2026-10-14-smoke-qwen-nvfp4-max");
        std::fs::create_dir_all(&dir).unwrap();
        let json = serde_json::to_vec_pretty(&report).unwrap();
        std::fs::write(dir.join("report.json"), &json).unwrap();
        std::fs::write(dir.join("card.svg"), "QUALITY GATE FAILED").unwrap();
        std::fs::write(dir.join("panel-baseline.svg"), "FIDELITY FAILURE").unwrap();
        publish(root.path(), &[dir.clone()]).unwrap();
        let card = std::fs::read_to_string(dir.join("card.svg")).unwrap();
        assert!(card.contains("info: calibrated W4A4"), "{card}");
        assert!(!card.contains("QUALITY GATE FAILED"), "{card}");
        let panel = std::fs::read_to_string(dir.join("panel-baseline.svg")).unwrap();
        assert!(!panel.contains("FIDELITY FAILURE"), "{panel}");
        assert!(!dir.join("card.png").exists());
        assert!(!dir.join("report.svg").exists());
        assert!(!dir.join("report.png").exists());
        assert_eq!(std::fs::read(dir.join("report.json")).unwrap(), json);
        let readme = std::fs::read_to_string(root.path().join("README.md")).unwrap();
        assert!(readme.contains("info: calibrated W4A4"), "{readme}");
        assert!(readme.contains("Informational"), "{readme}");
        assert!(!readme.contains("FAIL: not recommended"), "{readme}");
        let again = std::fs::read(dir.join("card.svg")).unwrap();
        publish(root.path(), &[]).unwrap();
        assert_eq!(std::fs::read(dir.join("card.svg")).unwrap(), again);
        assert_eq!(std::fs::read(dir.join("report.json")).unwrap(), json);
    }
}
