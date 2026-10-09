//! Programs belonging to one coordinator, including unsplit draft/target work.
#[derive(Debug, Clone, Copy)]
pub struct CoordinatorPrograms<'a> {
    pub family: &'a str,
    pub split_family: Option<&'a str>,
}

impl CoordinatorPrograms<'_> {
    /// Namespace boundaries matter: `glmf` must not select `glmf2` or `glm`.
    pub fn contains(self, name: &str) -> bool {
        name.split_once('_').is_some_and(|(family, _)| {
            family == self.family || self.split_family == Some(family)
        })
    }

    /// Shared arenas exclude index top-k, which owns a separate persistent buffer.
    pub fn shared_scratch<'a>(self, sizes: impl IntoIterator<Item = (&'a str, u64)>) -> u64 {
        sizes.into_iter().filter(|(name, _)| self.contains(name) && !name.contains("index_topk"))
            .map(|(_, bytes)| bytes).max().unwrap_or(0)
    }

    pub fn validate_v4(self, prefill_rows: u64, decode_rows: u64,
        available: impl IntoIterator<Item = impl AsRef<str>>) -> Result<(), MissingCoordinatorProgram> {
        let available: std::collections::HashSet<String> = available.into_iter()
            .map(|name| name.as_ref().to_string()).collect();
        for name in self.v4_required(prefill_rows, decode_rows) {
            if !self.contains(&name) || !available.contains(&name) {
                return Err(MissingCoordinatorProgram(name));
            }
        }
        Ok(())
    }

    /// All V4 dispatch routes, including the unsplit dSpark target in split mode.
    /// Validate this list before weights, arenas or graph captures are allocated.
    pub fn v4_required(self, prefill_rows: u64, decode_rows: u64) -> Vec<String> {
        let mut names = Vec::new();
        let mut add = |family: &str, suffix: String| names.push(format!("{family}_{suffix}"));
        for suffix in ["mhc_pre", "mhc_post", "mhc_head", "router_scores", "expert_input_quant",
            "block_fp8_scale_prep"] {
            add(self.family, suffix.to_string());
        }
        for ratio in [4, 128] {
            for mode in ["decode", "prefill", "continuation"] {
                add(self.family, format!("compressor_{mode}_c{ratio}"));
            }
        }
        for (mode, rows) in [("decode", decode_rows), ("prefill", prefill_rows)] {
            for suffix in [format!("mhc_post_pre_m{rows}"), format!("index_producer_m{rows}"),
                format!("index_topk_{mode}_m{rows}")] {
                add(self.family, suffix);
            }
            for family in std::iter::once(self.family).chain(self.split_family) {
                for suffix in ["producer", "wo", "shared_ffn"] {
                    add(family, format!("{suffix}_m{rows}"));
                }
                for attention in ["win", "c4", "c128"] {
                    add(family, format!("sparse_mla_{mode}_{attention}_m{rows}"));
                }
            }
        }
        names
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingCoordinatorProgram(pub String);

impl std::fmt::Display for MissingCoordinatorProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V4 startup requires selected program {}", self.0)
    }
}

impl std::error::Error for MissingCoordinatorProgram {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_selection_keeps_only_serving_and_split_families() {
        let families = ["dsv4f", "dsv4p", "glm", "glmf", "mimo", "mimof", "mimop", "qwen4"];
        for family in families {
            let split_family = format!("{family}2");
            for split in [false, true] {
                let selected = CoordinatorPrograms { family, split_family: split.then_some(split_family.as_str()) };
                assert!(selected.contains(&format!("{family}_target")));
                assert_eq!(selected.contains(&format!("{family}2_target")), split);
                for other in families.into_iter().filter(|other| *other != family) {
                    assert!(!selected.contains(&format!("{other}_target")));
                    assert!(!selected.contains(&format!("{other}2_target")));
                }
                assert!(!selected.contains(family));
            }
        }
    }

    #[test]
    fn split_v4_keeps_unsplit_draft_programs() {
        for family in ["dsv4f", "dsv4p"] {
            let split_family = format!("{family}2");
            let selected = CoordinatorPrograms { family, split_family: Some(&split_family) };
            let names = selected.v4_required(4096, 64);
            assert!(names.iter().all(|name| selected.contains(name)));
            for suffix in ["producer_m64", "producer_m4096", "wo_m64", "sparse_mla_decode_win_m64"] {
                assert!(names.contains(&format!("{family}_{suffix}")));
                assert!(names.contains(&format!("{split_family}_{suffix}")));
            }
        }
    }
}
