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
            let report: Report = serde_json::from_str(&text).with_context(|| format!("{}", path.display()))?;
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

/// The newest basic-profile report per family, checkpoint and reference hardware, in
/// family order, then checkpoint, 5090, one RTX, then two RTX.
fn reference_rows(placed: &[Placed]) -> Vec<(String, String, u8, &Placed)> {
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
        newest.entry((order, family, checkpoint(r), hardware)).or_insert(p);
    }
    let mut groups: BTreeMap<(usize, String, String), Vec<&Placed>> = BTreeMap::new();
    for ((order, family, name, _), p) in newest {
        groups.entry((order, family, name)).or_default().push(p);
    }
    let size = |p: &Placed| (p.report.server.hardware.used_gpus(), p.report.server.hardware.sparks.len());
    let mut rows = Vec::new();
    for ((_, family, name), reports) in groups {
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
        let minimum = one.iter().map(|p| size(p).1).min().and_then(|count|
            newest_of(one.iter().copied().filter(|p| size(p).1 == count).collect()));
        if let Some(p) = minimum {
            rows.push((family.clone(), name.clone(), 1, p));
        }
        let two: Vec<_> = pro.iter().copied().filter(|p| size(p).0 == 2).collect();
        let maximum = two.iter().map(|p| size(p).1).max().and_then(|count|
            newest_of(two.iter().copied().filter(|p| size(p).1 == count).collect()));
        if let Some(p) = maximum {
            rows.push((family, name, 2, p));
        }
    }
    rows
}

fn column_name(class: u8) -> &'static str {
    match class { 0 => "5090", 1 => "1× RTX", _ => "2× RTX" }
}

fn quality_cell(b: &crate::report::Baseline) -> String {
    match b.quality.status {
        CheckStatus::Fail => format!("⚠ **FAILED** {}", crate::render::bodies::quality_line(b)),
        CheckStatus::Pass => format!("✓ {}", b.quality.badge()),
        _ => b.quality.badge(),
    }
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
                    <br><sub>{}</sub></td>", link(&p.dir, "report.json"), p.report.server.hardware.sparks.len(),
                    crate::render::svg::escape(reason));
            }
            let svg = link(&p.dir, "card.svg");
            format!("<td width=\"27%\" valign=\"top\"><a href=\"{svg}\"><img src=\"{svg}\" alt=\"{} on {}\"></a>\
                <br><sub>{}</sub></td>", p.report.server.checkpoint(), p.report.server.hardware_line(),
                short_hardware(&p.report))
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
    for (family, name, class, p) in &rows {
        let r = &p.report;
        if let Some(reason) = r.no_fit_reason() {
            out.push_str(&format!("| [{}](docs/models/{family}.md) | {name} | {} ({}) | — | — | — | — | — | — | — | doesn't fit: {} | [planner report]({}) |\n",
                family_title(family), short_hardware(r), column_name(*class), reason.replace('|', "\\|"), link(&p.dir, "report.json")));
            continue;
        }
        let b = r.baseline.as_ref().expect("filtered");
        let decode = |content: &str| b.card.decode_of(content).map_or("—".into(), |d| rate(d.tok_s));
        let concurrent = b.card.concurrent.as_ref().map_or("—".into(),
            |c| format!("C{}: {}", c.width, rate(c.aggregate_tok_s)));
        let (prefill, ttft) = b.card.prefill.as_ref()
            .map_or(("—".into(), "—".into()), |p| (rate(p.tok_s), seconds(p.ttft_s)));
        let hardware = format!("{} ({})", short_hardware(r), column_name(*class));
        out.push_str(&format!("| [{}](docs/models/{family}.md) | {name} | {hardware} | {} | {} | {concurrent} | {} | {} | {prefill} | {ttft} | {} | \
            [{} · {}]({}) |\n", family_title(family), r.capacity().compact(), decode("code"), decode("prose"),
            decode("json"), quality_cell(b), crate::render::date(&r.created), r.server.build.label(),
            link(&p.dir, "report.svg")));
    }
    out.push_str("\ntok/s; C1 decode and concurrent code aggregate with thinking off (up to C8, clamped to server admission), \
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
            } else if r.quality_failed() {
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

/// The release whose changelog row (`| v0 | ... | Basic eval |`) gets the cards
/// (`CUTEAFD_RELEASE_ROW`, default `v0`).
fn release_row() -> String {
    std::env::var("CUTEAFD_RELEASE_ROW").unwrap_or_else(|_| "v0".into())
}

/// Fills the Basic eval cell of each `docs/models/<family>.md` changelog row of
/// this release with the family's Release smoke cards.
fn family_pages(root: &Path, placed: &[Placed]) -> Result<()> {
    let row = format!("| {} |", release_row());
    for family in FAMILIES {
        let path = root.join("docs/models").join(format!("{family}.md"));
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let cards = family_cards(placed, family);
        if cards.is_empty() {
            continue;
        }
        let mut changed = false;
        let lines: Vec<String> = text.lines().map(|line| {
            if !line.starts_with(&row) {
                return line.to_string();
            }
            let cells: Vec<&str> = line.trim_end().trim_end_matches('|').split('|').collect();
            if cells.len() < 3 {
                return line.to_string();
            }
            changed = true;
            format!("{}| {cards} |", cells[..cells.len() - 1].join("|"))
        }).collect();
        if changed {
            let mut out = lines.join("\n");
            if text.ends_with('\n') {
                out.push('\n');
            }
            crate::cli::write_if_changed(&path, &out)?;
        }
    }
    Ok(())
}

const INDEX_HEADER: &str = "# Benchmarks\n\nReports from `cuteafd bench` (profiles other than the basic one run \
    when asked). Each directory holds `report.svg`, `report.json` and the share card; `cuteafd bench publish` \
    rebuilds this index and the root README's table.\n\n";

/// Checks `dirs` are placed for publication and rebuilds both documents.
pub fn publish(root: &Path, dirs: &[PathBuf]) -> Result<(PathBuf, PathBuf, usize)> {
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
        let family = report.server.family.clone().unwrap_or_else(|| "unknown".into());
        let parent = relative.components().next().and_then(|c| c.as_os_str().to_str()).unwrap_or_default();
        if parent != family {
            bail!("{} holds a {family} report; move it under benchmarks/{family}/", dir.display());
        }
    }
    let placed = scan(root)?;
    let readme = root.join("README.md");
    let text = std::fs::read_to_string(&readme).with_context(|| format!("{}", readme.display()))?;
    crate::cli::write_if_changed(&readme, &splice(&text, RESULTS_BEGIN, RESULTS_END, &results(&placed))?)?;
    let index_path = root.join("benchmarks/README.md");
    std::fs::create_dir_all(root.join("benchmarks"))?;
    let current = std::fs::read_to_string(&index_path)
        .unwrap_or_else(|_| format!("{INDEX_HEADER}{INDEX_BEGIN}\n{INDEX_END}\n"));
    crate::cli::write_if_changed(&index_path, &splice(&current, INDEX_BEGIN, INDEX_END, &index(&placed))?)?;
    family_pages(root, &placed)?;
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
    fn publish_rebuilds_the_table_and_index() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("docs/models")).unwrap();
        std::fs::write(root.path().join("docs/models/deepseek_v41.md"),
            "## Changelog\n\n| Version | Date | Change | Basic eval |\n| --- | --- | --- | --- |\n| v0 | 2026-10-02 | First release | — |\n").unwrap();
        std::fs::write(root.path().join("README.md"),
            format!("# x\n\n{RESULTS_BEGIN}\n_Pending._\n{RESULTS_END}\n\nrest\n")).unwrap();
        let mut ok = crate::sample::report(false);
        ok.created = "2026-10-02T10:00:00Z".into();
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
}
