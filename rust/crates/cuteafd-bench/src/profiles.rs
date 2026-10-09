//! Run profiles: which panels a run ticks and how many passes each. The
//! baseline (basic card + quick quality) is implicit in every profile.
use crate::report::PlannedPanel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Profile {
    pub name: String,
    pub title: String,
    pub description: String,
    pub panels: Vec<PlannedPanel>,
    #[serde(default)]
    pub builtin: bool,
}

fn planned(ids: &[&str]) -> Vec<PlannedPanel> {
    ids.iter().map(|id| PlannedPanel { id: id.to_string(), passes: 1 }).collect()
}

const SPEED: [&str; 7] = ["decode_content", "concurrency", "prefill", "retained", "prefix_cache", "startup", "agentic"];
const QUALITY: [&str; 7] = ["tool_eval", "structured", "ifeval", "code", "math", "needle", "fidelity"];

/// The built-in profiles, in dialog order.
pub fn builtin() -> Vec<Profile> {
    let p = |name: &str, title: &str, description: &str, ids: &[&str]| Profile {
        name: name.into(), title: title.into(), description: description.into(), panels: planned(ids), builtin: true };
    vec![
        p("share", "Share card", "The baseline only: basic card and quick quality, exported as the share card.", &[]),
        p("smoke", "Release smoke",
            "Basic card + quick quality on a fresh server; with load, under five minutes per configuration.", &[]),
        p("speed", "Speed", "Decode by content, concurrency, prefill, retained context, prefix cache, startup.",
            &["decode_content", "concurrency", "prefill", "retained", "prefix_cache", "startup"]),
        p("daily", "Daily driver", "What a coding agent feels: decode, concurrency to C8, agentic session, tool eval, \
            prefix cache, structured output.",
            &["decode_content", "concurrency", "agentic", "tool_eval", "prefix_cache", "structured"]),
        p("quality", "Quality", "Tool eval, structured output, IFEval, code pass@1, math, long-context needle.",
            &QUALITY),
        p("fidelity", "Fidelity", "Standard-v2: balanced 32 decode / 32 prefill windows; all 64 decode when prefill is not admitted.", &["fidelity"]),
        p("quant", "Quant check", "Quality panels that move with quantization: tool eval, IFEval, math, code.",
            &["tool_eval", "ifeval", "math", "code"]),
        p("long", "Long context", "Prefill to the model maximum, decode vs retained context, needle heatmap, \
            prefix cache.", &["prefill", "retained", "needle", "prefix_cache"]),
        p("reasoning", "Reasoning effort", "Accuracy, tokens and time per official reasoning level.",
            &["reasoning_effort"]),
        p("full", "Full report", "Every panel.", &[SPEED.as_slice(), QUALITY.as_slice(), &["reasoning_effort"]].concat()),
    ]
}

/// Validate an explicit smoke panel selection before taking locks or launching.
pub fn validate_selection(profile: &str) -> Result<(), String> {
    if let Some(ids) = profile.strip_prefix("panels:") {
        for id in ids.split(',').map(str::trim).filter(|id| !id.is_empty()) {
            if id != "baseline" && crate::panels::find(id).is_none() {
                return Err(format!("unknown requested panel: {id}"));
            }
        }
    }
    Ok(())
}

/// A profile's display title (custom profiles show their name).
pub fn title_of(name: &str) -> String {
    builtin().into_iter().find(|p| p.name == name).map(|p| p.title)
        .unwrap_or_else(|| if name == "custom" { "Custom".into() } else { name.to_string() })
}

/// The profile's panels this build can run, with passes from `passes`
/// (`panel=N` overrides) and unknown ids dropped (reported back).
pub fn resolve(panels: &[PlannedPanel], passes: &[(String, u32)]) -> (Vec<PlannedPanel>, Vec<String>) {
    let mut out = Vec::new();
    let mut dropped = Vec::new();
    for planned in panels {
        if planned.id == "baseline" || crate::panels::ALWAYS.contains(&planned.id.as_str()) {
            continue;
        }
        if crate::panels::find(&planned.id).is_none() {
            dropped.push(planned.id.clone());
            continue;
        }
        let n = passes.iter().find(|(id, _)| id == &planned.id).map_or(planned.passes, |(_, n)| *n);
        if n > 0 && !out.iter().any(|p: &PlannedPanel| p.id == planned.id) {
            out.push(PlannedPanel { id: planned.id.clone(), passes: n.min(20) });
        }
    }
    // Hardware and configuration lead every report.
    let mut plan: Vec<PlannedPanel> = crate::panels::ALWAYS.iter()
        .map(|id| PlannedPanel { id: id.to_string(), passes: 1 }).collect();
    plan.extend(out);
    (plan, dropped)
}

/// `a=2,b=3` -> pairs.
pub fn parse_passes(text: &str) -> Result<Vec<(String, u32)>, String> {
    text.split(',').filter(|s| !s.trim().is_empty()).map(|pair| {
        let (id, n) = pair.split_once('=').ok_or_else(|| format!("{pair}: expected panel=N"))?;
        let n = n.trim().parse::<u32>().map_err(|_| format!("{pair}: N must be a whole number"))?;
        Ok((id.trim().to_string(), n))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_profiles_are_unique_and_resolve() {
        let profiles = builtin();
        let mut names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), profiles.len());
        let (plan, _) = resolve(&profiles[0].panels, &[]);
        assert_eq!(plan.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), vec!["hardware", "configuration"]);
        assert_eq!(title_of("smoke"), "Release smoke");
    }

    #[test]
    fn explicit_smoke_panels_reject_unknown_names() {
        assert!(validate_selection("panels:decode_content,fidelity").is_ok());
        assert!(validate_selection("smoke").is_ok());
        assert_eq!(validate_selection("panels:decode,fidelity").unwrap_err(), "unknown requested panel: decode");
    }

    #[test]
    fn passes_parse_and_override() {
        assert_eq!(parse_passes("tool_eval=3, x=1").unwrap(), vec![("tool_eval".into(), 3), ("x".into(), 1)]);
        assert!(parse_passes("tool_eval").is_err());
        let (plan, dropped) = resolve(&planned(&["hardware", "nope"]), &[("hardware".into(), 3)]);
        assert_eq!(plan.len(), 2);
        assert_eq!(dropped, vec!["nope".to_string()]);
    }
}
