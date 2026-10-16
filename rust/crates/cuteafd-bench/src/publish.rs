//! `cuteafd bench publish`: the root README's results table (the published
//! matrix, each cell resolved to one accepted report) and the
//! `benchmarks/README.md` index, both rebuilt from the `report.json` files
//! under `benchmarks/<family>/<date>-<profile>-<hardware>/`.
use crate::render::{rate, seconds};
use crate::report::{CheckStatus, Report};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const RESULTS_BEGIN: &str = "<!-- results:begin -->";
pub const RESULTS_END: &str = "<!-- results:end -->";
pub const INDEX_BEGIN: &str = "<!-- reports:begin -->";
pub const INDEX_END: &str = "<!-- reports:end -->";

/// The published matrix, relative to the repository root (`--matrix` overrides).
pub const MATRIX_FILE: &str = "benchmarks/current-matrix.json";
pub const MATRIX_SCHEMA: &str = "cuteafd.bench.matrix/1";

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

/// The reference columns of the published table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    /// One RTX 5090 with the fewest Sparks the checkpoint fits.
    Rtx5090,
    /// One RTX PRO 6000 with the fewest Sparks the checkpoint fits.
    OneRtx,
    /// Two RTX PRO 6000 with four or six Sparks (or none, when the GPUs hold it).
    TwoRtx,
}

impl Column {
    pub const ALL: [Column; 3] = [Column::Rtx5090, Column::OneRtx, Column::TwoRtx];

    /// The matrix key.
    fn key(self) -> &'static str {
        match self { Column::Rtx5090 => "5090", Column::OneRtx => "1rtx", Column::TwoRtx => "2rtx" }
    }

    /// The README heading.
    fn title(self) -> &'static str {
        match self { Column::Rtx5090 => "5090", Column::OneRtx => "1× RTX", Column::TwoRtx => "2× RTX" }
    }

    /// The reference layout a labelled smoke directory carries for this column.
    fn reference(self) -> Option<u8> {
        match self { Column::Rtx5090 => None, Column::OneRtx => Some(1), Column::TwoRtx => Some(2) }
    }
}

/// What the matrix says about one cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The column's own reference hardware.
    Current,
    /// Measured, withheld from the table; the cell renders its reason.
    Held,
    /// Not measured yet; the cell renders its reason.
    Pending,
    /// No such layout for this checkpoint; the cell renders its reason.
    Na,
}

/// One cell of the matrix.
#[derive(Debug, Clone)]
pub struct Cell {
    pub status: Status,
    /// Rendered by `held`, `pending` and `na` cells; empty otherwise.
    pub reason: String,
    /// Exact accepted report directory, relative to the repository root.
    pub report: Option<PathBuf>,
}

impl Cell {
    pub fn current() -> Self {
        Cell { status: Status::Current, reason: String::new(), report: None }
    }

    pub fn held(reason: impl Into<String>) -> Self {
        Cell { status: Status::Held, reason: reason.into(), report: None }
    }

    pub fn pending(reason: impl Into<String>) -> Self {
        Cell { status: Status::Pending, reason: reason.into(), report: None }
    }

    pub fn na(reason: impl Into<String>) -> Self {
        Cell { status: Status::Na, reason: reason.into(), report: None }
    }
}

/// A family × checkpoint row with one cell per reference column.
#[derive(Debug, Clone)]
pub struct MatrixRow {
    pub family: String,
    pub checkpoint: String,
    pub cells: [Cell; 3],
}

impl MatrixRow {
    pub fn new(family: &str, checkpoint: &str, cells: [Cell; 3]) -> Self {
        MatrixRow { family: family.into(), checkpoint: checkpoint.into(), cells }
    }
}

/// `benchmarks/current-matrix.json`: which checkpoint the README publishes in
/// each reference column, and why a cell publishes nothing.
///
/// The table is this matrix resolved against the accepted reports on disk, not
/// "whatever is newest under `benchmarks/`": a newer report on an archived
/// layout, a card from an earlier release, or a simulation where no real card
/// exists cannot take a cell the matrix gives to another card. Every row keeps
/// at least one measured cell, because the checkpoint and quant that name the
/// row are report data.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: Vec<MatrixRow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMatrix {
    schema: String,
    cells: Vec<RawRow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRow {
    family: String,
    checkpoint: String,
    #[serde(rename = "5090")]
    rtx_5090: RawCell,
    #[serde(rename = "1rtx")]
    one_rtx: RawCell,
    #[serde(rename = "2rtx")]
    two_rtx: RawCell,
}

#[derive(Debug)]
struct RawCell {
    status: String,
    reason: Option<String>,
    report: Option<PathBuf>,
}

/// A cell is its status (`"current"`), or a status with the reason it renders.
impl<'de> Deserialize<'de> for RawCell {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Spec {
            status: String,
            #[serde(default)]
            reason: Option<String>,
            #[serde(default)]
            report: Option<PathBuf>,
        }
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(status) => Ok(RawCell { status, reason: None, report: None }),
            other => {
                let spec: Spec = serde_json::from_value(other).map_err(serde::de::Error::custom)?;
                Ok(RawCell { status: spec.status, reason: spec.reason, report: spec.report })
            }
        }
    }
}

impl Matrix {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("{} (the published matrix: \
            which checkpoint is current in each reference column)", path.display()))?;
        Self::parse(&text).with_context(|| format!("{}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let raw: RawMatrix = serde_json::from_str(text)?;
        if raw.schema != MATRIX_SCHEMA {
            bail!("matrix schema is {:?}, expected {:?}", raw.schema, MATRIX_SCHEMA);
        }
        let mut rows: Vec<MatrixRow> = Vec::new();
        for row in raw.cells {
            if row.family.trim().is_empty() || row.checkpoint.trim().is_empty() {
                bail!("every matrix cell needs a family and a checkpoint");
            }
            if rows.iter().any(|seen| seen.family == row.family && seen.checkpoint == row.checkpoint) {
                bail!("{}/{} is listed twice", row.family, row.checkpoint);
            }
            let raw = [row.rtx_5090, row.one_rtx, row.two_rtx];
            let mut cells = Vec::with_capacity(3);
            for (column, raw) in Column::ALL.iter().zip(raw) {
                cells.push(cell(*column, raw, &row.family, &row.checkpoint)?);
            }
            rows.push(MatrixRow::new(&row.family, &row.checkpoint,
                cells.try_into().map_err(|_| anyhow::anyhow!("three cells per row"))?));
        }
        Ok(Matrix { rows })
    }
}

fn cell(column: Column, raw: RawCell, family: &str, checkpoint: &str) -> Result<Cell> {
    let status = match raw.status.as_str() {
        "current" => Status::Current,
        "held" => Status::Held,
        "pending" => Status::Pending,
        "na" => Status::Na,
        other => bail!("{family}/{checkpoint} {}: unknown status {other:?} \
            (current, held, pending, na)", column.key()),
    };
    let reason = raw.reason.unwrap_or_default();
    let renders = matches!(status, Status::Held | Status::Pending | Status::Na);
    if renders && reason.trim().is_empty() {
        bail!("{family}/{checkpoint} {}: {:?} renders its reason, and it is empty", column.key(), raw.status);
    }
    if !renders && !reason.is_empty() {
        bail!("{family}/{checkpoint} {}: {:?} takes no reason", column.key(), raw.status);
    }
    if let Some(path) = &raw.report {
        if status != Status::Current {
            bail!("{family}/{checkpoint} {}: only current cells take a report", column.key());
        }
        if path.components().count() != 3 || !path.starts_with("benchmarks")
            || !path.components().all(|part| matches!(part, std::path::Component::Normal(_))) {
            bail!("report must be benchmarks/<family>/<directory>, relative to the repository");
        }
    }
    Ok(Cell { status, reason, report: raw.report })
}

/// A one-row matrix: the table and page renderers' test scaffolding.
#[cfg(test)]
pub fn one_row(family: &str, checkpoint: &str, cells: [Cell; 3]) -> Matrix {
    Matrix { rows: vec![MatrixRow::new(family, checkpoint, cells)] }
}

/// Re-renders the images a published run already carries from its own
/// `report.json`, so a card placed under an earlier fidelity rule shows the
/// verdict the rule in force now reaches.
///
/// Exports are never edited after publication, so the images are what goes
/// stale; the data they came from does not. The set re-rendered is the one the
/// run's exports are drawn from, and only files already present are written —
/// publishing never adds an export the run did not place — and only when the
/// bytes differ, which keeps a report whose card is already current (and a
/// repeated publish) from touching the tree at all.
///
/// A re-render that fails is the publisher's error, not a run without that
/// export: publishing stops rather than leave the stale image in place.
fn rerender(dir: &Path, report: &Report) -> Result<()> {
    let mut names: Vec<String> = ["card.svg", "card.png", "report.svg", "report.png"]
        .map(str::to_string).to_vec();
    names.extend(crate::render::report::shown(report).iter().map(|id| format!("panel-{id}.svg")));
    for name in names {
        let path = dir.join(&name);
        // Only files the run already has: publishing never adds an export the
        // run did not place. `report.json` is the data, not a rendering of it.
        if !path.is_file() {
            continue;
        }
        // A re-render that fails stops the publish; a placed file the current
        // rule has no export for is that same error, not a file to skip.
        let Some((_, bytes)) = crate::http::export(report, &name)
            .with_context(|| format!("re-rendering {name}"))? else {
            bail!("the placed {name} has no export under the current rule");
        };
        crate::cli::write_bytes_if_changed(&path, &bytes)?;
    }
    Ok(())
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
    if let Some(gib) = coordinator_budget(report) {
        out.push_str(&format!(" ({gib} GiB budget)"));
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

/// Whether a report may fill a cell: the basic profile, measured, or a planner
/// rejection of the 5090 card (which is a result in its own right).
fn accepted(report: &Report) -> bool {
    matches!(report.profile.as_str(), "smoke" | "share")
        && (report.baseline.is_some() || (report.no_fit_reason().is_some() && report.server.column_5090()))
}

fn newest<'a>(items: impl IntoIterator<Item = &'a Placed>) -> Option<&'a Placed> {
    items.into_iter().max_by(|a, b| a.report.created.cmp(&b.report.created).then(a.dir.cmp(&b.dir)))
}

/// The card a measured cell publishes: the newest accepted report on the
/// column's hardware, preferring the layout the column is defined by.
///
/// That layout (`-min` on one RTX, `-max` on two) is the cell; another layout of
/// the same checkpoint — an archived one, a vision variant, a legacy GPU count —
/// never displaces it. Simulations remain archived, never published as real cards.
fn card_for<'a>(placed: &'a [Placed], row: &MatrixRow, column: Column, pinned: Option<&Path>) -> Option<&'a Placed> {
    let candidates = || -> Vec<&'a Placed> {
        placed.iter().filter(|p| {
            let report = &p.report;
            let in_column = match column {
                Column::Rtx5090 => report.server.column_5090()
                    && report.server.hardware.used_gpus() == 1
                    && report.server.hardware.gpus.iter()
                        .filter(|g| g.used || !report.server.hardware.gpus.iter().any(|g| g.used))
                        .all(|g| g.name.contains("RTX 5090")),
                Column::OneRtx => !report.server.column_5090() && report.server.hardware.used_gpus() == 1,
                Column::TwoRtx => !report.server.column_5090() && report.server.hardware.used_gpus() == 2,
            };
            accepted(report) && in_column
                && report.server.family.as_deref().unwrap_or("unknown") == row.family
                && report.server.checkpoint() == row.checkpoint
                && !report.server.simulated_5090()
                && pinned.is_none_or(|dir| p.dir == dir)
        }).collect()
    };
    let reports = candidates();
    if pinned.is_some() || column == Column::Rtx5090 {
        return newest(reports);
    }
    if let Some(reference) = column.reference() {
        if reports.iter().any(|p| reference_layout(p) == Some(reference)) {
            return newest(reports.into_iter().filter(|p| reference_layout(p) == Some(reference)));
        }
    }
    // No reference marker: the column's extreme Spark count on its GPU count.
    let sparks = |p: &&Placed| p.report.server.hardware.sparks.len();
    let extreme = if column == Column::OneRtx {
        reports.iter().map(sparks).min()
    } else {
        reports.iter().map(sparks).max()
    }?;
    newest(reports.into_iter().filter(|p| p.report.server.hardware.sparks.len() == extreme))
}

/// A matrix row resolved against the reports on disk.
pub struct Row<'a> {
    pub family: String,
    /// `GLM-5.3-NVFP4 (nvfp4-g16)`: the checkpoint and its routed-expert format.
    pub label: String,
    pub cells: [Resolved<'a>; 3],
}

/// What one cell publishes.
pub enum Resolved<'a> {
    /// The card the matrix gives this cell.
    Card(&'a Placed),
    /// A measured cell with no accepted report: `results` refuses to publish it.
    Missing,
    Held(String),
    Pending(String),
    Na(String),
}

/// Resolves every matrix row against `placed`, in matrix order. Fails when a
/// measured cell has no accepted report, or a row has nothing to name it with.
pub fn select<'a>(placed: &'a [Placed], matrix: &Matrix) -> Result<Vec<Row<'a>>> {
    let mut rows = Vec::new();
    for row in &matrix.rows {
        let cells: [Resolved<'_>; 3] = std::array::from_fn(|i| {
            let (column, cell) = (Column::ALL[i], &row.cells[i]);
            match cell.status {
                Status::Current =>
                    card_for(placed, row, column, cell.report.as_deref())
                        .map_or(Resolved::Missing, Resolved::Card),
                Status::Held => Resolved::Held(cell.reason.clone()),
                Status::Pending => Resolved::Pending(cell.reason.clone()),
                Status::Na => Resolved::Na(cell.reason.clone()),
            }
        });
        for (column, cell) in Column::ALL.iter().zip(&cells) {
            if matches!(cell, Resolved::Missing) {
                bail!("{}/{}: the matrix publishes a {} card, but no accepted report for it is \
                    under benchmarks/", row.family, row.checkpoint, column.title());
            }
        }
        let Some(label) = cells.iter().find_map(|cell| match cell {
            Resolved::Card(p) => Some(checkpoint(&p.report)), _ => None }) else {
            bail!("{}/{}: no cell publishes a report; drop the row or measure one",
                row.family, row.checkpoint);
        };
        rows.push(Row { family: row.family.clone(), label, cells });
    }
    Ok(rows)
}

fn quality_cell(b: &crate::report::Baseline) -> String {
    match b.quality.status {
        CheckStatus::Fail => format!("⚠ **FAILED** {}", crate::render::bodies::quality_line(b)),
        CheckStatus::Pass => format!("✓ {}", b.quality.badge()),
        _ => b.quality.badge(),
    }
}

// Publication notes are report data; escape text and restrict links to archived reports.
//
// The note's stored verdict sentence leads a policy the publisher applies now:
// a calibrated W4A4 checkpoint's fidelity is informational, so its card must
// not open with "FAIL" just because the note was written while it failed. The
// verdict shown is the one the report's quality reaches under the rule in
// force now; the note keeps only its prose, its measured numbers and its links.
fn caveat(p: &Placed) -> String {
    let Some(note) = p.report.server.setting("qualification.caveat") else { return String::new() };
    let prose = crate::render::svg::escape(crate::quality::without_verdict(note));
    let mut text = match crate::quality::warning(&p.report) {
        Some(verdict) if prose.is_empty() => format!("<b>{verdict}</b>"),
        Some(verdict) => format!("<b>{verdict}</b> {prose}"),
        None => prose,
    };
    if let (Some(href), Some(label)) = (p.report.server.setting("qualification.caveat-link"),
        p.report.server.setting("qualification.caveat-link-label")) {
        if href.starts_with("benchmarks/") && !href.split('/').any(|part| part == "..") {
            text.push_str(&format!("<a href=\"{}\">{}</a>.", crate::render::svg::escape(href), crate::render::svg::escape(label)));
        }
    }
    if text.is_empty() { return String::new() }
    format!("<br><sub>{text}</sub>")
}

/// A cell that publishes a status instead of a card; the reason is escaped data.
fn status_cell(label: &str, reason: &str) -> String {
    format!("<td width=\"27%\" valign=\"top\">{label}<br><sub>{}</sub></td>", crate::render::svg::escape(reason))
}

fn grid_cell(cell: &Resolved) -> String {
    match cell {
        Resolved::Card(p) => {
            if let Some(reason) = p.report.no_fit_reason() {
                return format!("<td width=\"27%\" valign=\"top\"><a href=\"{}\">doesn't fit 1× 5090 (32 GB) + {} Sparks</a>\
                    <br><sub>{}</sub>{}</td>", link(&p.dir, "report.json"), p.report.server.hardware.sparks.len(),
                    crate::render::svg::escape(reason), caveat(p));
            }
            let svg = link(&p.dir, "card.svg");
            format!("<td width=\"27%\" valign=\"top\"><a href=\"{svg}\"><img src=\"{svg}\" alt=\"{} on {}\"></a>\
                <br><sub>{}</sub>{}</td>", p.report.server.checkpoint(), p.report.server.hardware_line(),
                short_hardware(&p.report), caveat(p))
        }
        Resolved::Held(reason) => status_cell("held", reason),
        Resolved::Pending(reason) => status_cell("pending", reason),
        Resolved::Na(reason) => status_cell("n/a", reason),
        Resolved::Missing => unreachable!("select() refuses a missing cell"),
    }
}

/// One compact-table row: the same data as the cell's card, or why it has none.
fn table_row(row: &Row, column: Column, cell: &Resolved) -> String {
    let head = format!("| [{}](docs/models/{}.md) | {} | ", family_title(&row.family), row.family, row.label);
    match cell {
        Resolved::Held(reason) => format!("{head}— ({}) | — | — | — | — | — | — | — | held: {} | — |\n",
            column.title(), reason.replace('|', "\\|")),
        Resolved::Card(p) => {
            let r = &p.report;
            if let Some(reason) = r.no_fit_reason() {
                return format!("{head}{} ({}) | — | — | — | — | — | — | — | doesn't fit: {} | [planner report]({}) |\n",
                    short_hardware(r), column.title(), reason.replace('|', "\\|"), link(&p.dir, "report.json"));
            }
            let b = r.baseline.as_ref().expect("accepted");
            let decode = |content: &str| b.card.decode_of(content).map_or("—".into(), |d| rate(d.tok_s));
            let concurrent = r.server.card_levels().iter().filter_map(|width| b.card.concurrent_at(*width))
                .map(|c| format!("C{}: {}", c.width, rate(c.aggregate_tok_s))).collect::<Vec<_>>().join(" · ");
            let concurrent = if concurrent.is_empty() { "—".into() } else { concurrent };
            let (prefill, ttft) = b.card.prefill.as_ref()
                .map_or(("—".into(), "—".into()), |p| (rate(p.tok_s), seconds(p.ttft_s)));
            let hardware = r.server.setting("qualification.hardware-label").map(str::to_string)
                .unwrap_or_else(|| format!("{} ({})", short_hardware(r), column.title()));
            format!("{head}{hardware} | {} | {} | {concurrent} | {} | {} | {prefill} | {ttft} | {} | \
                [{} · {}]({}) |\n", r.capacity().compact(), decode("code"), decode("prose"),
                decode("json"), quality_cell(b), crate::render::date(&r.created), r.server.build.label(),
                link(&p.dir, "report.svg"))
        }
        Resolved::Pending(_) | Resolved::Na(_) | Resolved::Missing => String::new(),
    }
}

/// The root README's results: one row per checkpoint of the published matrix
/// (the checkpoint and quant linking to its family page, then its three
/// reference-column cards, each linking to the card SVG), then the same rows as
/// a compact table.
pub fn results(placed: &[Placed], matrix: &Matrix) -> Result<String> {
    let rows = select(placed, matrix)?;
    if rows.is_empty() {
        return Ok("_Pending the first published run._\n".into());
    }
    let mut out = String::from("<table>\n<tr><th>Model · quant</th><th>5090</th><th>1× RTX</th><th>2× RTX</th></tr>\n");
    for row in &rows {
        // select() proved the row names a report; the checkpoint and quant that
        // name the row are that report's data, not the matrix's.
        let first = row.cells.iter().find_map(|cell| match cell {
            Resolved::Card(p) => Some(p), _ => None }).expect("select() names a report");
        let quant = row.label.rsplit_once(" (").map_or("", |(_, quant)| quant.trim_end_matches(')'));
        out.push_str(&format!("<tr>\n<td width=\"19%\" valign=\"top\"><a href=\"docs/models/{}.md\"><b>{}</b></a>\
            <br><sub>{}</sub><br><sub>{quant}</sub></td>\n", row.family, family_title(&row.family),
            first.report.server.checkpoint()));
        for cell in &row.cells {
            out.push_str(&grid_cell(cell));
            out.push('\n');
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</table>\n\n");
    out.push_str("| Family | Checkpoint | Hardware | KV / req | C1 code | Concurrent code (aggregate) | prose | JSON | 8K prefill | TTFT | Quality | Report |\n");
    out.push_str("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n");
    for row in &rows {
        for (column, cell) in Column::ALL.iter().zip(&row.cells) {
            out.push_str(&table_row(row, *column, cell));
        }
    }
    out.push_str("\ntok/s; C1 decode and concurrent code aggregate with thinking off (C1/C4/C8/C16 on PRO 6000; through C8 at ≤32 GiB, within server admission), \
        8K prefill cold. Quality: logit fidelity against the family golden \
        reference, prefix-cache restore exactness, lossless speculation. Every row is \
        `benchmarks/current-matrix.json` resolved against the reports under `benchmarks/`; a held or \
        pending cell renders why it has no card. The 5090 column publishes real hardware only.\n");
    Ok(out)
}

/// A family page's basic-eval cell: the cards of `family` the matrix publishes
/// among `published` (paths relative to `docs/models/`).
pub fn family_cards(rows: &[Row], published: &BTreeSet<PathBuf>, family: &str) -> String {
    rows.iter().filter(|row| row.family == family)
        .flat_map(|row| Column::ALL.iter().zip(&row.cells).map(move |(column, cell)| (row, column, cell)))
        .filter_map(|(row, column, cell)| match cell {
            Resolved::Card(p) if published.contains(&p.dir) => Some(format!(
                "<a href=\"../../{}\"><img src=\"../../{}\" width=\"360\" alt=\"{} ({})\"></a>",
                link(&p.dir, "report.svg"), link(&p.dir, "card.svg"), row.label, column.title())),
            _ => None,
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
fn family_pages(root: &Path, placed: &[Placed], matrix: &Matrix, release: Option<&str>) -> Result<()> {
    let override_row = std::env::var("CUTEAFD_RELEASE_ROW").ok();
    family_pages_with_row(root, placed, matrix, release, override_row.as_deref())
}

fn family_pages_with_row(root: &Path, placed: &[Placed], matrix: &Matrix, release: Option<&str>,
    override_row: Option<&str>) -> Result<()> {
    let Some(row) = override_row.or(release) else { return Ok(()) };
    let normalized_row = release_row(row);
    let selected = release.or(normalized_row.as_deref());
    let current: BTreeSet<PathBuf> = placed.iter().filter(|p| {
        selected.is_some() && report_release(p).as_deref() == selected
            || (override_row.is_some() && report_release(p).is_none())
    }).map(|p| p.dir.clone()).collect();
    // One resolution for every page: the matrix decides every card.
    let rows = select(placed, matrix)?;
    for family in FAMILIES {
        let path = root.join("docs/models").join(format!("{family}.md"));
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let newest = newest(placed.iter().filter(|p| current.contains(&p.dir)
            && p.report.server.family.as_deref() == Some(family)
            && matches!(p.report.profile.as_str(), "smoke" | "share")
            && (p.report.baseline.is_some() || p.report.no_fit_reason().is_some())));
        let Some(newest) = newest else { continue };
        let cards = family_cards(&rows, &current, family);
        if cards.is_empty() { continue; }
        let out = changelog_cards(&text, row, &crate::render::date(&newest.report.created), &cards)?;
        crate::cli::write_if_changed(&path, &out)?;
    }
    Ok(())
}

const INDEX_HEADER: &str = "# Benchmarks\n\nReports from `cuteafd bench` (profiles other than the basic one run \
    when asked). Each directory holds `report.svg`, `report.json` and the share card; `cuteafd bench publish` \
    rebuilds this index and the root README's table.\n\n";

/// Checks `dirs` are placed for publication and rebuilds both documents.
pub fn publish(root: &Path, dirs: &[PathBuf], matrix_path: &Path) -> Result<(PathBuf, PathBuf, usize)> {
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
    // The matrix is the published set, and it is read before anything is
    // written: a cell the tree cannot fill fails without touching a document.
    let matrix = Matrix::load(matrix_path)?;
    let placed = scan(root)?;
    // Resolve first: a missing pinned report must fail before any file is touched.
    // Superseded exports remain archived unchanged, including their images.
    let rows = select(&placed, &matrix)?;
    for row in &rows {
        for cell in &row.cells {
            if let Resolved::Card(p) = cell {
                rerender(&root.join(&p.dir), &p.report)?;
            }
        }
    }
    let release = releases.into_iter().next().or_else(|| {
        dirs.is_empty().then(|| placed.iter().find_map(report_release)).flatten()
    });
    let readme = root.join("README.md");
    let text = std::fs::read_to_string(&readme).with_context(|| format!("{}", readme.display()))?;
    crate::cli::write_if_changed(&readme, &splice(&text, RESULTS_BEGIN, RESULTS_END, &results(&placed, &matrix)?)?)?;
    let index_path = root.join("benchmarks/README.md");
    std::fs::create_dir_all(root.join("benchmarks"))?;
    let current = std::fs::read_to_string(&index_path)
        .unwrap_or_else(|_| format!("{INDEX_HEADER}{INDEX_BEGIN}\n{INDEX_END}\n"));
    crate::cli::write_if_changed(&index_path, &splice(&current, INDEX_BEGIN, INDEX_END, &index(&placed))?)?;
    family_pages(root, &placed, &matrix, release.as_deref())?;
    Ok((readme, index_path, placed.len()))
}

/// A one-row matrix for a single report: `current` in the column it fills,
/// `pending` elsewhere. Test scaffolding, so a renderer test need not restate a
/// whole matrix to see one cell.
#[cfg(test)]
pub fn one_row_of(placed: &Placed) -> Matrix {
    let report = &placed.report;
    let cell = |column: Column| {
        let in_column = match column {
            Column::Rtx5090 => report.server.column_5090(),
            Column::OneRtx => !report.server.column_5090() && report.server.hardware.used_gpus() == 1,
            Column::TwoRtx => !report.server.column_5090() && report.server.hardware.used_gpus() == 2,
        };
        if !in_column { return Cell::pending("not measured on this hardware") }
        Cell::current()
    };
    let cells = [cell(Column::Rtx5090), cell(Column::OneRtx), cell(Column::TwoRtx)];
    Matrix { rows: vec![MatrixRow::new(report.server.family.as_deref().unwrap_or("unknown"),
        &report.server.checkpoint(), cells)] }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A publication marker: cell selection reads report data, never a directory name.
    fn marker(p: &mut Placed, name: &str, value: &str) {
        if name == "hardware.class" && value == "5090" {
            for gpu in &mut p.report.server.hardware.gpus {
                gpu.name = "NVIDIA GeForce RTX 5090".into();
            }
        }
        p.report.server.configuration.settings.push(crate::report::Setting {
            name: name.into(), value: Some(value.into()), default: None, source: "publication".into(),
        });
    }

    /// A DeepSeek V4.1 card on `gpus` RTX with an optional marker.
    fn marked(id: &str, gpus: usize, mark: Option<(&str, &str)>) -> Placed {
        let mut report = crate::sample::report(false);
        report.id = id.into();
        let gpu = report.server.hardware.gpus[0].clone();
        report.server.hardware.gpus = (0..gpus).map(|i| {
            let mut g = gpu.clone(); g.index = i as u32; g.used = true; g
        }).collect();
        let mut placed = Placed { dir: PathBuf::from(format!("benchmarks/deepseek_v41/{id}")), report };
        if let Some((name, value)) = mark {
            marker(&mut placed, name, value);
        }
        placed
    }

    /// A Qwen card on `gpus` RTX and `sparks` Sparks, placed under its own directory.
    fn qwen(id: &str, gpus: usize, sparks: usize, budget: Option<&str>) -> Placed {
        let mut report = crate::sample::report(false);
        report.id = id.into();
        let gpu = report.server.hardware.gpus[0].clone();
        report.server.hardware.gpus = (0..gpus).map(|i| {
            let mut g = gpu.clone(); g.index = i as u32; g.used = true; g
        }).collect();
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
    }

    /// The same card under a labelled directory (the smoke matrix entry name).
    fn labelled(mut p: Placed, label: &str) -> Placed {
        p.dir = crate::cli::labeled_dir(Path::new(""), &p.report, Some(label));
        p
    }

    /// The card each cell publishes, as report ids (empty for a status cell).
    fn cards<'a>(row: &'a Row<'a>) -> Vec<Option<&'a str>> {
        row.cells.iter().map(|cell| match cell {
            Resolved::Card(p) => Some(p.report.id.as_str()), _ => None }).collect()
    }

    /// No real RTX 5090 exists yet, so the 5090 cell has no card in these fixtures.
    fn v41() -> [Cell; 3] {
        [Cell::pending("no 5090 card yet"), Cell::current(), Cell::pending("no two-RTX card")]
    }

    fn v41_row(cells: [Cell; 3]) -> Matrix {
        one_row("deepseek_v41", "deepseek-ai/DeepSeek-V4.1-Flash", cells)
    }

    #[test]
    fn the_matrix_decides_every_cell_and_the_newest_accepted_report_fills_it() {
        let mut older = marked("older", 1, None);
        older.report.created = "2026-10-01T00:00:00Z".into();
        let mut newer = marked("newer", 1, None);
        newer.report.created = "2026-10-10T00:00:00Z".into();
        let reports = vec![newer, older, marked("sim", 1, Some(("hardware.class", "5090"))), marked("two", 2, None)];
        let matrix = v41_row([Cell::current(), Cell::current(), Cell::current()]);
        let rows = select(&reports, &matrix).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(cards(&rows[0]), vec![Some("sim"), Some("newer"), Some("two")]);
        let html = results(&reports, &matrix).unwrap();
        assert!(html.contains("<th>5090</th><th>1× RTX</th><th>2× RTX</th>"));
        assert_eq!(html.matches("<img src=").count(), 3);
        // A cell the matrix gives to another card does not publish this one.
        assert!(!html.contains("older/card.svg"));
        assert!(index(&reports).contains("older/report.svg"));
        // An empty matrix publishes nothing, and says so.
        assert_eq!(results(&reports, &Matrix { rows: Vec::new() }).unwrap(), "_Pending the first published run._\n");
    }

    #[test]
    fn a_measured_cell_without_a_report_refuses_to_publish() {
        let reports = vec![marked("one", 1, None)];
        let matrix = v41_row([Cell::pending("no 5090 card yet"), Cell::current(), Cell::current()]);
        let error = results(&reports, &matrix).unwrap_err().to_string();
        assert!(error.contains("no accepted report"), "{error}");
        assert!(error.contains("2× RTX"), "{error}");
        // Pending and n/a are the ways to say a cell has no card.
        let matrix = v41_row([Cell::pending("no 5090 card yet"), Cell::current(), Cell::na("no two-GPU split")]);
        let html = results(&reports, &matrix).unwrap();
        assert!(html.contains("pending<br><sub>no 5090 card yet</sub>"), "{html}");
        assert!(html.contains("n/a<br><sub>no two-GPU split</sub>"), "{html}");
        assert!(!html.contains("pending Hugh"), "{html}");
        assert_eq!(html.matches("<img src=").count(), 1);
        // A row with no measured cell has nothing to name it.
        let matrix = v41_row([Cell::pending("a"), Cell::pending("b"), Cell::pending("c")]);
        assert!(results(&reports, &matrix).unwrap_err().to_string().contains("no cell publishes a report"));
    }

    #[test]
    fn held_cells_render_their_reason_and_keep_a_compact_row() {
        let reports = vec![marked("sim", 1, Some(("hardware.class", "5090"))), marked("one", 1, None), marked("two", 2, None)];
        let held = "W4A4 small-M path costs 13% per verify row vs the rc3 W4A16 card (#233)";
        let matrix = v41_row([Cell::current(), Cell::held(held), Cell::current()]);
        let html = results(&reports, &matrix).unwrap();
        let (grid, table) = html.split_once("</table>").unwrap();
        assert!(grid.contains(&format!("held<br><sub>{held}</sub>")), "{grid}");
        assert!(!grid.contains("one/card.svg"), "a held cell publishes no card");
        // The 5090 cell of the same row still publishes: holding is per cell.
        assert!(grid.contains("sim/card.svg"));
        assert_eq!(html.matches("<img src=").count(), 2);
        let row = table.lines().find(|line| line.contains("held: ")).unwrap();
        assert!(row.contains(&format!("| — (1× RTX) | — | — | — | — | — | — | — | held: {held} | — |")), "{row}");
        assert!(!table.contains("one/report.svg"), "{table}");
        // Escaped, and pipe-escaped for the table, like every other cell.
        let matrix = v41_row([Cell::current(), Cell::held("weights < layout | over budget"), Cell::current()]);
        let html = results(&reports, &matrix).unwrap();
        assert!(html.contains("weights &lt; layout | over budget"), "{html}");
        assert!(html.contains("held: weights < layout \\| over budget"), "{html}");
    }

    #[test]
    fn archived_layouts_never_leak_into_the_table() {
        // The reference-layout card is the cell; a newer report of another layout
        // stays in the index and out of the table.
        let reference = labelled(marked("reference", 1, None), "v41-mxfp4-min-rc3");
        let mut archived = labelled(marked("archived", 1, None), "v41-mxfp4-vision");
        archived.report.created = "2026-10-11T00:00:00Z".into();
        archived.report.server.hardware.sparks.push(archived.report.server.hardware.sparks[0].clone());
        let matrix = v41_row(v41());
        let reports = vec![archived.clone(), reference.clone()];
        let rows = select(&reports, &matrix).unwrap();
        assert_eq!(cards(&rows[0])[1], Some("reference"));
        let html = results(&reports, &matrix).unwrap();
        assert!(html.contains(&link(&reference.dir, "card.svg")), "{html}");
        assert!(!html.contains("vision"), "a supplementary layout never reaches the table");
        assert!(!html.contains(&link(&archived.dir, "report.svg")), "and no supplementary row does either");
        let published: BTreeSet<PathBuf> = reports.iter().map(|p| p.dir.clone()).collect();
        let cards = family_cards(&rows, &published, "deepseek_v41");
        assert!(cards.contains(&link(&reference.dir, "card.svg")));
        assert!(!cards.contains("vision"));
        // History stays in the index.
        assert!(index(&reports).contains("v41-mxfp4-vision/report.svg"));
    }

    #[test]
    fn the_reference_layout_wins_each_column() {
        let one = labelled(marked("one", 1, None), "model-min");
        let two = labelled(marked("two", 2, None), "model-rc2");
        let matrix = v41_row([Cell::pending("no 5090 card yet"), Cell::current(), Cell::current()]);
        assert_eq!(reference_layout(&one), Some(1));
        assert_eq!(reference_layout(&two), None);
        let placed = [one.clone(), two.clone()];
        let rows = select(&placed, &matrix).unwrap();
        assert_eq!(cards(&rows[0]), vec![None, Some("one"), Some("two")]);
        assert_eq!(results(&[one.clone(), two.clone()], &matrix).unwrap().matches("<img src=").count(), 2);
        // Without a reference marker the column's extreme Spark count decides.
        let fewer = labelled(marked("fewer", 1, None), "model");
        let mut more = labelled(marked("more", 2, None), "model");
        more.report.server.hardware.sparks.push(more.report.server.hardware.sparks[0].clone());
        more.report.created = "2026-10-11T00:00:00Z".into();
        let placed = [more.clone(), fewer.clone()];
        let rows = select(&placed, &matrix).unwrap();
        assert_eq!(cards(&rows[0]), vec![None, Some("fewer"), Some("more")]);
        // The marker is per column: a one-GPU layout marked -max fills the one-RTX cell.
        let historical = labelled(marked("historical", 1, None), "model-max");
        assert_eq!(reference_layout(&historical), Some(2));
        let one_column = v41_row([Cell::pending("no 5090 card yet"), Cell::current(), Cell::pending("no two-RTX card")]);
        assert_eq!(cards(&select(&[historical], &one_column).unwrap()[0])[1], Some("historical"));
    }

    #[test]
    fn simulated_cards_never_fill_a_current_cell() {
        let sim = marked("sim", 1, Some(("simulated", "5090")));
        let real = marked("real", 1, Some(("hardware.class", "5090")));
        let matrix = v41_row([Cell::current(), Cell::pending("no PRO card"), Cell::pending("no dual card")]);
        assert!(select(&[sim.clone()], &matrix).is_err());
        let mut mislabelled = real.clone();
        mislabelled.report.server.hardware.gpus[0].name = "NVIDIA RTX PRO 6000 Blackwell".into();
        assert!(select(&[mislabelled], &matrix).is_err());
        let mut dual = real.clone();
        dual.report.server.hardware.gpus.push(dual.report.server.hardware.gpus[0].clone());
        assert!(select(&[dual], &matrix).is_err());
        assert_eq!(cards(&select(&[sim.clone(), real], &matrix).unwrap()[0])[0], Some("real"));
        assert!(index(&[sim]).contains("sim/report.svg"));
    }

    #[test]
    fn pinned_current_report_never_falls_back_to_another_build_or_layout() {
        let old = marked("old-min", 1, None);
        let mut newer = marked("new-min", 1, None);
        newer.report.created = "2026-10-16T00:00:00Z".into();
        let mut matrix = v41_row(v41());
        matrix.rows[0].cells[1].report = Some(old.dir.clone());
        assert_eq!(cards(&select(&[old.clone(), newer.clone()], &matrix).unwrap()[0])[1], Some("old-min"));
        assert!(select(&[newer], &matrix).is_err());
        matrix.rows[0].cells[1].report = Some("benchmarks/deepseek_v41/wrong-gpu".into());
        let wrong = marked("wrong-gpu", 2, None);
        assert!(select(&[old, wrong], &matrix).is_err());
    }

    #[test]
    fn no_fit_cards_publish_their_rejection() {
        let mut rejection = marked("no-fit-5090", 1, Some(("hardware.class", "5090")));
        rejection.report.status = crate::report::RunStatus::Failed;
        rejection.report.baseline = None;
        marker(&mut rejection, "qualification.no-fit", "rtx0 weights < layout | over budget");
        let matrix = v41_row([Cell::current(), Cell::pending("no one-RTX card"), Cell::pending("no two-RTX card")]);
        let html = results(&[rejection.clone()], &matrix).unwrap();
        assert!(html.contains("doesn't fit 1× 5090 (32 GB) + 4 Sparks"));
        assert!(html.contains("weights &lt; layout"));
        assert!(html.contains("layout \\| over budget"));
        assert!(html.contains("no-fit-5090/report.json"));
        let card = crate::render::card::card_svg(&rejection.report);
        assert!(card.contains("Doesn&#39;t fit"));
        assert!(!card.contains("tok/s"));
        let report = crate::render::report::report_svg(&rejection.report);
        assert!(report.contains("Doesn&#39;t fit"));
        assert!(!crate::render::report::shown(&rejection.report).contains(&"baseline".into()));
        assert!(index(&[rejection.clone()]).contains("no-fit-5090/report.svg"));
        // A failed run with neither a measurement nor the planner's reason fills no cell.
        let mut unmarked = rejection;
        unmarked.report.server.configuration.settings.retain(|s| s.name != "qualification.no-fit");
        assert!(select(&[unmarked], &matrix).is_err());
    }

    #[test]
    fn a_retired_checkpoint_leaves_the_matrix_but_not_the_index() {
        // A qualified replacement retires a checkpoint by dropping it from the
        // matrix; its reports stay under benchmarks/ and in the index.
        let mut old = crate::sample::report(false);
        old.server.family = Some("mimo_v2".into());
        old.server.configuration.snapshot = None;
        old.server.model = "XiaomiMiMo/MiMo-V2-Flash".into();
        let mut replacement = old.clone();
        replacement.server.model = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD".into();
        let placed = vec![
            Placed { dir: "benchmarks/mimo_v2/new".into(), report: replacement },
            Placed { dir: "benchmarks/mimo_v2/old".into(), report: old },
            Placed { dir: "benchmarks/qwen4/other".into(), report: {
                let mut other = crate::sample::report(false);
                other.server.family = Some("qwen4".into());
                other
            } },
        ];
        let matrix = Matrix { rows: vec![MatrixRow::new("mimo_v2", "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
            [Cell::pending("no 5090 card yet"), Cell::current(), Cell::pending("no two-RTX card")])] };
        let rows = select(&placed, &matrix).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(!rows.iter().any(|row| row.label.contains("MiMo-V2-Flash")));
        let published: BTreeSet<PathBuf> = placed.iter().map(|p| p.dir.clone()).collect();
        let cards = family_cards(&rows, &published, "mimo_v2");
        assert!(cards.contains("/new/card.svg"));
        assert!(!cards.contains("/old/card.svg"));
        assert!(index(&placed).contains("MiMo-V2-Flash"));
        assert!(index(&placed).contains("MiMo-V2.6-Flash-MOPD"));
        assert!(index(&placed).contains("qwen4"));
    }

    #[test]
    fn matrix_minimum_beats_extra_layout_and_archived_extras_stay_in_the_index() {
        let minimum = labelled(qwen("uuid-min", 1, 2, None), "qwen38-exl3-min");
        let mut extra = labelled(qwen("uuid-extra", 1, 0, None), "qwen38-exl3-nospark-vision");
        extra.report.created = "2026-10-10T00:00:00Z".into();
        let matrix = one_row("qwen4", "org/Qwen3.8-EXL3",
            [Cell::pending("no 5090 card yet"), Cell::current(), Cell::na("fits one RTX; no two-GPU split")]);
        let reports = vec![extra.clone(), minimum.clone()];
        let rows = select(&reports, &matrix).unwrap();
        assert_eq!(cards(&rows[0])[1], Some("uuid-min"));
        let html = results(&reports, &matrix).unwrap();
        assert!(html.contains(&link(&minimum.dir, "card.svg")));
        assert!(!html.contains(&link(&extra.dir, "card.svg")), "an extra layout is not the cell");
        assert!(!html.contains("nospark-vision"), "and it is not a supplementary row either");
        assert!(index(&reports).contains("nospark-vision/report.svg"));
    }

    #[test]
    fn qwen_local_minimum_is_not_displaced_by_legacy_budget_or_other_hardware() {
        let minimum = qwen("minimum", 1, 1, Some("32"));
        let mut real_5090 = qwen("real_5090", 1, 1, Some("31.8"));
        marker(&mut real_5090, "hardware.class", "5090");
        let matrix = one_row("qwen4", "org/Qwen3.8-EXL3",
            [Cell::current(), Cell::current(), Cell::na("fits one RTX (no two-GPU split for Qwen)")]);
        let reports = vec![real_5090, minimum.clone(), qwen("maximum", 1, 0, None), qwen("legacy", 1, 4, None)];
        let rows = select(&reports, &matrix).unwrap();
        // Real 5090 hardware fills its own cell; a legacy 4-Spark layout never does.
        assert_eq!(cards(&rows[0]), vec![Some("real_5090"), Some("maximum"), None]);
        assert!(short_hardware(&minimum.report).contains("32 GiB budget"));
        assert!(crate::render::card::card_svg(&minimum.report).contains("32 GiB budget"));
        let html = results(&reports, &matrix).unwrap();
        assert!(html.contains("benchmarks/qwen4/maximum/card.svg"));
        assert!(html.contains("n/a<br><sub>fits one RTX (no two-GPU split for Qwen)</sub>"), "{html}");
        assert!(!html.contains("benchmarks/qwen4/legacy/card.svg"));
        // A two-GPU card is not a one-RTX cell, whatever the matrix says.
        let matrix = one_row("qwen4", "org/Qwen3.8-EXL3",
            [Cell::pending("no 5090 card yet"), Cell::na("no one-RTX split for Qwen"),
                Cell::current()]);
        let reports = vec![qwen("legacy-two", 2, 4, None), qwen("max-two", 2, 0, None)];
        assert_eq!(cards(&select(&reports, &matrix).unwrap()[0]), vec![None, None, Some("legacy-two")]);
    }

    #[test]
    fn matrix_files_are_validated_before_use() {
        let row = r#"{"family":"glm5","checkpoint":"nvidia/GLM-5.3-NVFP4","5090":"current","1rtx":"current","2rtx":"current"}"#;
        let held = r##"{"family":"glm5","checkpoint":"nvidia/GLM-5.3-NVFP4","5090":"current","1rtx":{"status":"held","reason":"#233: +13% per verify row"},"2rtx":"current"}"##;
        let held_cell = r##""1rtx":{"status":"held","reason":"#233: +13% per verify row"}"##;
        let doc = |cells: &str| format!(r#"{{"schema":"{MATRIX_SCHEMA}","cells":[{cells}]}}"#);
        let error = |text: String| Matrix::parse(&text).unwrap_err().to_string();
        let swap = |to: &str| doc(&held.replace(held_cell, to));
        assert_eq!(Matrix::parse(&doc(row)).unwrap().rows.len(), 1);
        let matrix = Matrix::parse(&doc(held)).unwrap();
        assert_eq!(matrix.rows[0].cells[1].status, Status::Held);
        assert_eq!(matrix.rows[0].cells[1].reason, "#233: +13% per verify row");
        assert_eq!(matrix.rows[0].cells[0].status, Status::Current);
        assert!(Matrix::parse(&doc("")).unwrap().rows.is_empty());
        assert!(error(format!(r#"{{"schema":"cuteafd.bench.matrix/2","cells":[{row}]}}"#)).contains("schema"));
        assert!(error(doc(&format!("{row},{row}"))).contains("twice"));
        assert!(error(format!(r#"{{"schema":"{MATRIX_SCHEMA}","cells":[{row}],"extra":1}}"#)).contains("unknown field"));
        assert!(error(swap(r#""1rtx":"held""#)).contains("renders its reason"));
        assert!(error(swap(r#""1rtx":"maybe""#)).contains("unknown status"));
        assert!(error(swap(r#""1rtx":{"status":"held"}"#)).contains("renders its reason"));
        assert!(error(swap(r#""1rtx":{"status":"held","reason":"   "}"#)).contains("renders its reason"));
        assert!(error(swap(r#""1rtx":{"status":"held","reason":"x","extra":1}"#)).contains("unknown field"));
        assert!(error(swap(r#""1rtx":{"status":"current","reason":"x"}"#)).contains("takes no reason"));
        assert!(error(doc(&held.replace(r#""family":"glm5""#, r#""family":"""#))).contains("family and a checkpoint"));
        assert!(error(doc(&row.replace(r#""5090":"current""#, r#""5090":"simulated-5090""#))).contains("unknown status"));
        assert!(error(swap(r#""1rtx":{"status":"current","report":"../archive"}"#)).contains("relative"));
        assert!(error(swap(r#""1rtx":{"status":"held","reason":"wait","report":"benchmarks/glm5/run"}"#)).contains("only current"));
        assert!(error(doc(&row.replace(r#""5090":"current""#, r#""5090":"current","reason":"x""#)))
            .contains("unknown field"));
        // A matrix that is not where the publisher looks is an error, not a default.
        let root = tempfile::tempdir().unwrap();
        assert!(Matrix::load(&root.path().join("absent.json")).is_err());
        assert!(Matrix::load(root.path()).is_err());
    }

    /// Writes `benchmarks/current-matrix.json` from rows: the real file, re-parsed by the real loader.
    fn write_matrix(root: &Path, rows: &[MatrixRow]) -> PathBuf {
        let path = root.join(MATRIX_FILE);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let raw = |cell: &Cell| match cell.status {
            Status::Current => serde_json::json!({"status": "current", "report": cell.report}),
            status => serde_json::json!({"status": match status {
                Status::Held => "held", Status::Pending => "pending", _ => "na" }, "reason": cell.reason}),
        };
        let cells: Vec<_> = rows.iter().map(|row| serde_json::json!({
            "family": row.family, "checkpoint": row.checkpoint,
            "5090": raw(&row.cells[0]), "1rtx": raw(&row.cells[1]), "2rtx": raw(&row.cells[2]),
        })).collect();
        std::fs::write(&path, serde_json::to_string_pretty(&serde_json::json!({
            "schema": MATRIX_SCHEMA, "cells": cells })).unwrap()).unwrap();
        path
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
    fn release_candidates_share_card_identity_and_newest_wins() {
        let old = labelled(marked("older", 1, None), "model-min");
        let mut rc2 = labelled(marked("rc2", 1, None), "model-min-rc2");
        rc2.report.created = "2026-10-10T00:00:00Z".into();
        let mut rc3 = labelled(marked("rc3", 1, None), "model-min-rc3");
        rc3.report.created = "2026-10-11T00:00:00Z".into();
        assert_eq!(reference_layout(&rc3), Some(1));
        assert_eq!(without_release_candidate("model-max-rc3"), "model-max");
        assert_eq!(without_release_candidate("model-min-rcfoo"), "model-min-rcfoo");
        let matches = without_release_candidate(rc3.dir.file_name().unwrap().to_str().unwrap());
        assert_eq!(matches, without_release_candidate(rc2.dir.file_name().unwrap().to_str().unwrap()));
        let placed = [old, rc2, rc3];
        let rows = select(&placed, &v41_row(v41())).unwrap();
        assert_eq!(cards(&rows[0])[1], Some("rc3"));
    }

    #[test]
    fn publication_caveats_are_escaped_report_data_for_fit_and_no_fit() {
        let mut p = marked("with-note-min", 1, None);
        // A card that failed its gates under the rule in force now keeps the
        // verdict; the note only supplies the prose around it.
        let baseline = p.report.baseline.as_mut().unwrap();
        baseline.quality.checks.iter_mut().find(|c| c.id == "fidelity").unwrap().status = crate::report::CheckStatus::Fail;
        baseline.quality.settle();
        let setting = |name: &str, value: &str| crate::report::Setting {
            name: name.into(), value: Some(value.into()), source: "publication".into(), ..Default::default()
        };
        p.report.server.configuration.settings.extend([
            setting("qualification.caveat", "FAIL: not recommended. KL > 0.06. With vision: "),
            setting("qualification.caveat-link", "benchmarks/qwen4/vision/report.svg"),
            setting("qualification.caveat-link-label", "vision <on>"),
        ]);
        assert_eq!(caveat(&p), "<br><sub><b>FAIL: not recommended.</b> KL &gt; 0.06. With vision: <a href=\"benchmarks/qwen4/vision/report.svg\">vision &lt;on&gt;</a>.</sub>");
        assert!(results(&[p.clone()], &v41_row(v41())).unwrap().contains(&caveat(&p)));
        p.report.server.configuration.settings.iter_mut().find(|s| s.name == "qualification.caveat-link").unwrap().value = Some("javascript:bad".into());
        assert!(!caveat(&p).contains("href="));
        p.report.baseline = None;
        p.report.status = crate::report::RunStatus::Failed;
        marker(&mut p, "qualification.no-fit", "over budget");
        marker(&mut p, "hardware.class", "5090");
        let matrix = v41_row([Cell::current(), Cell::pending("no one-RTX card"), Cell::pending("no two-RTX card")]);
        assert!(results(&[p.clone()], &matrix).unwrap().contains(&caveat(&p)));
        // A panel a later policy reclassifies: the same note, quality that now
        // passes, so the caption drops the verdict sentence and keeps its prose.
        let mut ok = marked("reclassified-min", 1, None);
        ok.report.server.configuration.settings.push(
            setting("qualification.caveat", "FAIL: not recommended. KL 0.039, inside the calibrated quant's band."));
        let reclassified = caveat(&ok);
        assert!(reclassified.contains("KL 0.039, inside the calibrated quant&#39;s band."), "{reclassified}");
        assert!(!reclassified.contains("FAIL"), "{reclassified}");
    }

    /// Every file under `dir`, by name: what a repeat publish must leave alone.
    fn tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
        std::fs::read_dir(dir).unwrap().map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name().to_string_lossy().into_owned(), std::fs::read(entry.path()).unwrap())
        }).collect()
    }

    /// A repository holding one published Qwen NVFP4 max run, with the images
    /// the rule in force at publication rendered: the failure the renderer used
    /// to bake in, and `failing` (a check id, or `""`) left failing beside it.
    /// Returns the run's directory and the file names the run placed.
    fn published_qwen_repo(root: &Path, algo: &str, failing: &str) -> (PathBuf, Vec<String>) {
        let report = crate::sample::qwen_nvfp4_max(&root.join("hub"), algo, failing);
        let dir = root.join("benchmarks/qwen4/2026-10-14-smoke-qwen38-nvfp4-max");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("report.json"), serde_json::to_string_pretty(&report).unwrap()).unwrap();
        let card = crate::render::card::card_svg(&report);
        let full = crate::render::report::report_svg(&report);
        assert!(card.contains("QUALITY GATE FAILED"), "the fixture card is the stale one");
        let mut placed = Vec::new();
        for (name, svg) in [("card.svg", &card), ("report.svg", &full)] {
            std::fs::write(dir.join(name), svg).unwrap();
            placed.push(name.to_string());
        }
        // The rasterized exports carry the same verdict as the SVG, and are the
        // files a re-render must not leave stale.
        for (name, svg) in [("card.png", &card), ("report.png", &full)] {
            std::fs::write(dir.join(name), crate::render::png::png(svg, 1.0).unwrap()).unwrap();
            placed.push(name.to_string());
        }
        std::fs::write(dir.join("panel-baseline.svg"),
            crate::render::report::panel_svg(&report, "baseline")).unwrap();
        placed.push("panel-baseline.svg".into());
        std::fs::write(root.join("README.md"), format!("# x\n\n{RESULTS_BEGIN}\n{RESULTS_END}\n")).unwrap();
        placed.push("report.json".into());
        (dir, placed)
    }

    /// The matrix for `published_qwen_repo`: its one run uses one RTX.
    fn qwen_matrix(root: &Path) -> PathBuf {
        write_matrix(root, &[MatrixRow::new("qwen4", "nvidia/Qwen3.8-Flash-Next-NVFP4",
            [Cell::pending("no 5090 card"), Cell::current(), Cell::pending("no two-RTX card")])])
    }

    #[test]
    fn a_calibrated_nvfp4_card_publishes_as_informational_not_as_the_stored_failure() {
        let root = tempfile::tempdir().unwrap();
        let (dir, placed) = published_qwen_repo(root.path(), "NVFP4", "");
        let fixture = tree(&dir);
        let (readme, _, count) = publish(root.path(), &[], &qwen_matrix(root.path())).unwrap();
        assert_eq!(count, 1);
        // The card image is re-rendered from report.json: the measured numbers
        // stay, the verdict the rule in force now reaches replaces the stored one.
        let card = std::fs::read_to_string(dir.join("card.svg")).unwrap();
        assert!(card.contains("top-1 93.7% (info: calibrated W4A4)"), "{card}");
        assert!(!card.contains("QUALITY GATE FAILED"), "{card}");
        assert!(!card.contains("✗ KL 0.039"), "the failed mark is gone: {card}");
        let panel = std::fs::read_to_string(dir.join("panel-baseline.svg")).unwrap();
        assert!(!panel.contains("FIDELITY FAILURE"), "{panel}");
        let report = std::fs::read_to_string(dir.join("report.svg")).unwrap();
        assert!(!report.contains("QUALITY GATE FAILED"), "{report}");
        // The exports are never edited after publication: the rasterized card
        // follows the SVG, and the data behind it is the file that was placed,
        // byte for byte.
        assert_eq!(std::fs::read(dir.join("report.json")).unwrap(), fixture["report.json"],
            "publishing rewrote the run's data");
        assert_ne!(std::fs::read(dir.join("card.png")).unwrap(), fixture["card.png"],
            "the stale card PNG was kept");
        // The README row and the caption under its image both follow.
        let published = std::fs::read_to_string(&readme).unwrap();
        assert!(published.contains("top-1 93.7% (info: calibrated W4A4)"), "{published}");
        assert!(published.contains("<img src=\"benchmarks/qwen4/2026-10-14-smoke-qwen38-nvfp4-max/card.svg\""), "{published}");
        assert!(!published.contains("FAIL: not recommended."), "{published}");
        assert!(!published.contains("FAILED"), "{published}");
        assert!(published.contains("KL &gt; 0.04 on a calibrated quant."), "the note keeps its prose: {published}");
        // Publishing again changes nothing, the data and the rasterized images
        // included: the exports this run placed are the ones it keeps, byte for
        // byte, `report.json` first among them.
        let before = tree(&dir);
        publish(root.path(), &[], &qwen_matrix(root.path())).unwrap();
        assert_eq!(tree(&dir), before, "a repeat publish rewrote the run");
        for name in placed {
            assert!(before.contains_key(&name), "the fixture never placed {name}");
            assert_eq!(std::fs::read(dir.join(&name)).unwrap(), before[&name], "{name} was rewritten");
        }
        assert!(before.contains_key("report.json") && before.contains_key("card.png"), "fixture files");
        assert_eq!(std::fs::read_to_string(&readme).unwrap(), published);
    }

    /// A card whose calibrated quantization the rule now treats as
    /// informational, beside a gate the bar still fails: the note loses its
    /// verdict, the aggregate does not, and no surface that shows one keeps
    /// showing the other.
    fn a_failing_gate_beside_a_reclassified_fidelity_check(failing: &str, title: &str, mark: &str) {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = published_qwen_repo(root.path(), "NVFP4", failing);
        let (readme, _, count) = publish(root.path(), &[], &qwen_matrix(root.path())).unwrap();
        assert_eq!(count, 1);
        // The aggregate: the fidelity check is informational, the gate is not.
        let stored: Report = serde_json::from_str(&std::fs::read_to_string(dir.join("report.json")).unwrap()).unwrap();
        assert!(stored.quality_failed(), "the fixture is a stored failure: {failing}");
        let served = crate::quality::under_current_rule(&stored);
        assert!(served.quality_failed(), "FAIL survives the reclassification: {failing}");
        let check = |id: &str| served.baseline.as_ref().unwrap().quality.checks.iter()
            .find(|check| check.id == id).unwrap().clone();
        assert_eq!(check("fidelity").status, CheckStatus::Info, "{failing}");
        assert_eq!(check("fidelity").note.as_deref(), Some(crate::quality::CALIBRATED_W4A4));
        assert_eq!(check(failing).status, CheckStatus::Fail, "{failing}");
        // The images: the watermark stands, the reclassified check takes the
        // informational note, and no failed mark is left on the reclassified
        // check. (A card's badge is width-fitted on a fixed 1200x675 card, so
        // the failing gate's name in the badge's tail is not asserted against
        // the pixels; the run image and the README cell name it in full.)
        let card = std::fs::read_to_string(dir.join("card.svg")).unwrap();
        assert!(card.contains("UNVERIFIED — QUALITY GATE FAILED"), "{card}");
        assert!(card.contains("top-1 93.7% (info: calibrated W4A4)"), "{card}");
        assert!(!card.contains("✗ KL 0.039"), "the reclassified check is not marked failed: {card}");
        // The badge the rule builds for this run does carry the failing mark.
        let badge = served.baseline.as_ref().unwrap().quality.badge();
        assert!(badge.contains(mark), "{badge}");
        assert!(!badge.contains("✗ KL 0.039"), "{badge}");
        let report = std::fs::read_to_string(dir.join("report.svg")).unwrap();
        assert!(report.contains(&format!("failed: {title}")), "{report}");
        assert!(report.contains("UNVERIFIED — QUALITY GATE FAILED"), "{report}");
        // The README: the cell names the gate that failed, never the one the
        // rule reclassified, and the caption keeps the failure sentence.
        let published = std::fs::read_to_string(&readme).unwrap();
        assert!(published.contains(&format!("⚠ **FAILED** failed: {title}")), "{published}");
        assert!(!published.contains("failed: Logit fidelity"), "{published}");
        // The caption keeps the failure sentence AND the note's prose, which
        // loses the verdict sentence it was stored with.
        assert!(published.contains("<b>FAIL: not recommended.</b> KL &gt; 0.04 on a calibrated quant."),
            "{published}");
        assert!(!published.contains("FAIL: not recommended. KL &gt; 0.04"), "{published}");
    }

    #[test]
    fn a_card_whose_cache_check_fails_keeps_its_failure_after_fidelity_is_reclassified() {
        a_failing_gate_beside_a_reclassified_fidelity_check("cache_exact", "Prefix-cache restore", "✗ cache mismatch");
    }

    #[test]
    fn a_card_whose_speculation_check_fails_keeps_its_failure_after_fidelity_is_reclassified() {
        a_failing_gate_beside_a_reclassified_fidelity_check("spec_lossless", "Speculation lossless", "✗ spec diverges");
    }

    /// A run whose card cannot be re-rendered stops the publish: an image that
    /// cannot be replaced is never kept in place of the one the rule in force
    /// calls for, and the failure names the file.
    ///
    /// Skipped for a user that can write anything (root): the write the test
    /// needs to fail cannot, and a skipped assertion is better than a test that
    /// passes without exercising the case.
    #[test]
    fn a_card_that_cannot_be_re_rendered_stops_the_publish() {
        use std::os::unix::fs::PermissionsExt;
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        if status.lines().any(|line| line.starts_with("Uid:") && line.contains("\t0\t0\t0\t0")) {
            eprintln!("a_card_that_cannot_be_re_rendered_stops_the_publish: running as root, skipped");
            return;
        }
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = published_qwen_repo(root.path(), "NVFP4", "");
        let card = dir.join("card.svg");
        let stale = std::fs::read(&card).unwrap();
        std::fs::set_permissions(&card, std::fs::Permissions::from_mode(0o444)).unwrap();
        let error = publish(root.path(), &[], &qwen_matrix(root.path())).unwrap_err();
        assert!(format!("{error:#}").contains("card.svg"), "{error:#}");
        assert_eq!(std::fs::read(&card).unwrap(), stale, "the stale card was left un-replaced");
    }

    #[test]
    fn a_stored_failure_the_bar_still_gates_keeps_its_verdict_everywhere() {
        let root = tempfile::tempdir().unwrap();
        // The same numbers from a checkpoint whose experts are weight-only: the
        // bar stands, so every surface keeps showing the failure it was
        // published with, byte for byte.
        let (dir, placed) = published_qwen_repo(root.path(), "W4A16_NVFP4", "");
        let before = tree(&dir);
        let card = std::fs::read_to_string(dir.join("card.svg")).unwrap();
        let (readme, _, _) = publish(root.path(), &[], &qwen_matrix(root.path())).unwrap();
        assert_eq!(tree(&dir), before, "a card the bar still fails is already current");
        for name in ["report.json", "card.png", "report.png"] {
            assert_eq!(std::fs::read(dir.join(name)).unwrap(), before[name], "{name} was rewritten");
        }
        assert!(placed.contains(&"card.png".to_string()));
        assert!(card.contains("QUALITY GATE FAILED"), "{card}");
        let readme = std::fs::read_to_string(readme).unwrap();
        assert!(readme.contains("<b>FAIL: not recommended.</b> KL &gt; 0.04 on a calibrated quant."), "{readme}");
        assert!(readme.contains("⚠ **FAILED** failed: Logit fidelity"), "{readme}");
    }

    #[test]
    fn publish_keeps_superseded_images_and_data_untouched() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("README.md"), format!("{RESULTS_BEGIN}\n{RESULTS_END}\n")).unwrap();
        let current = marked("current-min", 1, None);
        let old = marked("superseded-min", 1, None);
        for p in [&current, &old] {
            let dir = root.path().join(&p.dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("report.json"), serde_json::to_vec(&p.report).unwrap()).unwrap();
        }
        let archive = root.path().join(&old.dir);
        std::fs::write(archive.join("card.svg"), "archived image, not a rendering under today's rule").unwrap();
        let before = tree(&archive);
        let mut cells = v41();
        cells[1].report = Some(current.dir.clone());
        let matrix = write_matrix(root.path(), &[MatrixRow::new("deepseek_v41",
            "deepseek-ai/DeepSeek-V4.1-Flash", cells)]);
        let (readme, index, _) = publish(root.path(), &[], &matrix).unwrap();
        assert_eq!(tree(&archive), before);
        assert!(!std::fs::read_to_string(&readme).unwrap().contains("superseded-min"));
        assert!(std::fs::read_to_string(index).unwrap().contains("superseded-min"));
        // Refusing a missing pin must also leave both current and archived exports alone.
        let current_dir = root.path().join(&current.dir);
        std::fs::write(current_dir.join("card.svg"), "not yet rerendered").unwrap();
        let before_current = tree(&current_dir);
        let mut missing = v41();
        missing[1].report = Some("benchmarks/deepseek_v41/missing-min".into());
        let matrix = write_matrix(root.path(), &[MatrixRow::new("deepseek_v41",
            "deepseek-ai/DeepSeek-V4.1-Flash", missing)]);
        assert!(publish(root.path(), &[], &matrix).is_err());
        assert_eq!(tree(&current_dir), before_current);
        assert_eq!(tree(&archive), before);
    }

    #[test]
    fn checked_in_matrix_resolves_one_pinned_card_per_pro_cell() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let placed = scan(&root).unwrap();
        let matrix = Matrix::load(&root.join(MATRIX_FILE)).unwrap();
        let rows = select(&placed, &matrix).unwrap();
        assert_eq!(rows.len(), 14);
        for row in rows {
            assert!(matches!(row.cells[0], Resolved::Pending(_)));
            assert!(matches!(row.cells[1], Resolved::Card(_)));
            assert!(matches!(row.cells[2], Resolved::Card(_)));
        }
    }

    #[test]
    fn publication_rows_carry_a_hardware_label() {
        let mut labelled_row = marked("labelled-min", 1, None);
        marker(&mut labelled_row, "qualification.hardware-label", "vision on RTX0, fixed pool");
        assert!(results(&[labelled_row], &v41_row(v41())).unwrap().contains("| vision on RTX0, fixed pool |"));
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
            let matrix = write_matrix(root.path(), &[MatrixRow::new("deepseek_v41",
                "deepseek-ai/DeepSeek-V4.1-Flash", [Cell::pending("no 5090 card yet"), Cell::current(), Cell::current()])]);
            publish(root.path(), &[dir], &matrix).unwrap();
            let page = std::fs::read_to_string(&page_path).unwrap();
            assert!(page.contains(v0), "{page}");
            assert!(page.contains(v1), "{page}");
            let row = page.lines().find(|line| line.starts_with("| v2 |")).unwrap();
            assert!(row.contains("new-min/card.svg"), "{row}");
            assert!(!row.contains("old-max"), "{row}");
            assert_eq!(page.matches("| v2 |").count(), 1);
            if !existing { assert!(page.find(v1).unwrap() < page.find(row).unwrap()); }
            publish(root.path(), &[], &matrix).unwrap();
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
            let matrix = v41_row([Cell::pending("no 5090 card yet"), Cell::current(), Cell::current()]);
            family_pages_with_row(root.path(), &reports, &matrix, Some("v2"), Some(row)).unwrap();
            let page = std::fs::read_to_string(&path).unwrap();
            let target = page.lines().find(|line| line.starts_with(&format!("| {row} |"))).unwrap();
            assert!(target.contains("current-min/card.svg"), "{target}");
            assert!(!target.contains("old-max"), "{target}");
            assert!(page.contains(old_row));
            if row != "custom-row" {
                family_pages_with_row(root.path(), &reports, &matrix, None, Some(row)).unwrap();
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
        let matrix = write_matrix(root.path(), &[MatrixRow::new("deepseek_v41",
            "deepseek-ai/DeepSeek-V4.1-Flash", v41())]);
        let dir = root.path().join("benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark");
        let (readme, index, count) = publish(root.path(), &[dir], &matrix).unwrap();
        assert_eq!(count, 2);
        let readme = std::fs::read_to_string(readme).unwrap();
        assert!(readme.contains("<img src=\"benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/card.svg\""), "{readme}");
        // One row per checkpoint: the newest accepted report fills it.
        assert_eq!(readme.matches("| DeepSeek-V4.1-Flash (mxfp4) |").count(), 1, "{readme}");
        let placed = scan(root.path()).unwrap();
        let published: BTreeSet<PathBuf> = placed.iter().map(|p| p.dir.clone()).collect();
        let rows = select(&placed, &Matrix::load(&matrix).unwrap()).unwrap();
        assert!(family_cards(&rows, &published, "deepseek_v41").contains("../../benchmarks/"));
        let page = std::fs::read_to_string(root.path().join("docs/models/deepseek_v41.md")).unwrap();
        assert!(page.contains("| v0 | 2026-10-02 | First release | <a href=\"../../benchmarks/"), "{page}");
        assert!(readme.contains("benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/report.svg"));
        assert!(readme.ends_with("rest\n"));
        let index = std::fs::read_to_string(index).unwrap();
        let first = index.find("2026-10-02").unwrap();
        assert!(first < index.find("2026-10-01").unwrap(), "newest first");
        assert!(index.contains("⚠ quality gate failed"));
        // Idempotent.
        assert_eq!(std::fs::read_to_string(publish(root.path(), &[], &matrix).unwrap().0).unwrap(), readme);
        // Misplaced reports are refused.
        let stray = root.path().join("elsewhere");
        crate::cli::write_exports(&ok, &stray, &["json".into()]).unwrap();
        assert!(publish(root.path(), &[stray], &matrix).is_err());
        // And so is a cell the tree cannot fill, without writing a document.
        let before = std::fs::read_to_string(root.path().join("README.md")).unwrap();
        let missing = write_matrix(root.path(), &[MatrixRow::new("deepseek_v41",
            "deepseek-ai/DeepSeek-V4.1-Flash", [Cell::current(), Cell::current(), Cell::current()])]);
        assert!(publish(root.path(), &[], &missing).is_err());
        assert_eq!(std::fs::read_to_string(root.path().join("README.md")).unwrap(), before);
    }
}
