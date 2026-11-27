//! One draft configuration for every family and speculator.
use cuteafd_core::DraftControl;

/// `DRAFT_POLICY`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DraftPolicyChoice {
    /// Keep the leading drafts while their calibrated product stays above 0.3, plus one explore.
    #[default]
    History,
    /// `history` with each round's cut set from the online ratio of non-row time per
    /// request to per-row time (`cuteafd_core::clamped`).
    Clamped,
    /// Verify no drafts.
    Off,
}

#[derive(Debug, Clone, Copy, Default, clap::Args)]
pub(crate) struct DraftLength {
    /// Verify this many drafts every round, capped by the drafter's native block and the
    /// step's row budget: a fixed width for measurement. Unset follows `--draft-policy`.
    #[arg(long, env = "DRAFT_LENGTH", global = true, value_parser = parse)]
    pub draft_length: Option<usize>,
    /// How many drafts each request verifies.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub draft_policy: DraftPolicyChoice,
    /// Zero preserves native attention; positive values bound shared draft context.
    #[arg(long, global = true, default_value_t = 0)]
    pub draft_context_window: usize,
}

fn parse(value: &str) -> Result<usize, String> {
    value.parse::<usize>().ok().filter(|&n| n > 0)
        .ok_or_else(|| "draft length must be a positive integer".into())
}

impl DraftLength {
    pub fn report_inactive(self, active: bool) {
        if !active {
            if let Some(length) = self.draft_length {
                tracing::info!("DRAFT_LENGTH={length} ignored: no speculator");
            }
            cuteafd_bench::context::set_resolved("draft-mode", "no speculation");
        }
    }

    /// A fixed width wins over the policy; it is a measurement override.
    pub fn control(self) -> DraftControl {
        match (self.draft_length, self.draft_policy) {
            (Some(length), _) => DraftControl::Fixed(length),
            (None, DraftPolicyChoice::History | DraftPolicyChoice::Clamped) => DraftControl::History,
            (None, DraftPolicyChoice::Off) => DraftControl::Off,
        }
    }

    /// The keep cut each `history` round uses.
    pub fn cut_rule(self) -> cuteafd_core::CutRule {
        match (self.draft_length, self.draft_policy) {
            (None, DraftPolicyChoice::Clamped) => cuteafd_core::CutRule::Clamped,
            _ => cuteafd_core::CutRule::Fixed,
        }
    }

    /// The most drafts a round verifies from a drafter whose native block is `native`.
    pub fn effective(self, native: usize) -> usize {
        match self.control() {
            DraftControl::History => native,
            DraftControl::Fixed(length) => length.min(native),
            DraftControl::Off => 0,
        }
    }

    /// The card's draft mode.
    pub fn label(self, native: usize) -> String {
        match self.control() {
            DraftControl::History if self.cut_rule() == cuteafd_core::CutRule::Clamped => "clamped".into(),
            DraftControl::History => "adaptive".into(),
            DraftControl::Fixed(_) => format!("fixed {}", self.effective(native)),
            DraftControl::Off => "no speculation".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Commands};
    use crate::shared::draft::history::Policy;
    use clap::Parser;
    use cuteafd_core::HeadScore;

    fn config(command: &str, extra: &[&str]) -> DraftLength {
        let mut argv = vec!["cuteafd", command, "--snapshot", "/model", "--native-lib", "/native.so"];
        if matches!(command, "serve-native" | "serve-step5") { argv.extend(["--peers", "127.0.0.1:19441"]); }
        argv.extend(extra);
        match Cli::try_parse_from(argv).unwrap().command {
            Commands::ServeNative(a) => a.draft_length,
            Commands::ServeDsv4(a) => a.draft_length,
            Commands::ServeGlm(a) => a.draft_length,
            Commands::ServeGlmf(a) => a.draft_length,
            Commands::ServeMimo(a) => a.draft_length,
            Commands::ServeKimi(a) => a.draft_length,
            Commands::ServeMuse(a) => a.draft_length,
            Commands::ServeMinimax(a) => a.draft_length,
            Commands::ServeInkling2(a) => a.draft_length,
            Commands::ServeInkling2Mini(a) => a.draft_length,
            Commands::ServeQwen4(a) => a.draft_length,
            Commands::ServeQwen35(a) => a.draft_length,
            Commands::ServeStep5(a) => a.draft_length,
            _ => panic!("not a family serve command"),
        }
    }

    fn contract(command: &str, native: usize) {
        let unset = config(command, &[]);
        assert_eq!((unset.control(), unset.effective(native), unset.label(native)),
            (DraftControl::History, native, "adaptive".to_string()));
        let off = config(command, &["--draft-policy", "off"]);
        assert_eq!((off.control(), off.effective(native), off.label(native)),
            (DraftControl::Off, 0, "no speculation".to_string()));
        // `clamped` is `history` with a per-round cut, on every family.
        let clamped = config(command, &["--draft-policy", "clamped"]);
        assert_eq!((clamped.control(), clamped.cut_rule(), clamped.effective(native), clamped.label(native)),
            (DraftControl::History, cuteafd_core::CutRule::Clamped, native, "clamped".to_string()));
        assert_eq!(unset.cut_rule(), cuteafd_core::CutRule::Fixed);
        // A fixed width still overrides it.
        assert_eq!(config(command, &["--draft-policy", "clamped", "--draft-length", "2"]).cut_rule(),
            cuteafd_core::CutRule::Fixed);
        for requested in [1, 4, native + 10] {
            let value = requested.to_string();
            // A fixed width overrides the policy either way.
            for extra in [&["--draft-length", &value][..], &["--draft-length", &value, "--draft-policy", "off"]] {
                let length = config(command, extra);
                let effective = requested.min(native);
                assert_eq!(length.effective(native), effective);
                let (mut policy, settings) = cuteafd_bench::context::capture_resolved(||
                    Policy::new(length, native, HeadScore::None));
                assert!(policy.fixed());
                assert!(settings.0.iter().any(|(k, v)| k == "draft-mode" && v == &format!("fixed {effective}")));
                let mut context = cuteafd_bench::context::ServerContext::default();
                settings.apply(&mut context);
                context.resolve("draft", "/drafter");
                assert_eq!(cuteafd_bench::server::configuration(&context, &Default::default()).draft,
                    format!("fixed {effective}"));
                assert_eq!(policy.plan(&[(1, 64)], 64), [effective]);
                let requests: Vec<_> = (0..16).map(|id| (id, 64)).collect();
                let capped = policy.plan(&requests, 24);
                assert!(capped.iter().all(|&n| n <= effective));
                assert_eq!(capped.iter().sum::<usize>() + requests.len(), 24.min(16 * (effective + 1)));
            }
        }
        assert!(parse("0").is_err());
        assert!(parse("-1").is_err());
        assert!(Cli::try_parse_from(["cuteafd", command, "--snapshot", "/model", "--native-lib", "/native.so",
            "--draft-policy", "simple"]).is_err());
    }

    #[test]
    fn inactive_override_reports_no_speculation() {
        let (_, settings) = cuteafd_bench::context::capture_resolved(||
            DraftLength { draft_length: Some(4), ..Default::default() }.report_inactive(false));
        let mut context = cuteafd_bench::context::ServerContext::default();
        settings.apply(&mut context);
        assert_eq!(context.draft_mode(), "no speculation");
        assert_eq!(cuteafd_bench::server::configuration(&context, &Default::default()).draft, "no speculation");
    }

    #[test] fn v41() { contract("serve-native", 7); }
    #[test] fn v4_markov_dspark() { contract("serve-dsv4", 5); }
    #[test] fn glm5_and_ml4() { contract("serve-glm", 7); contract("serve-glm", 5); }
    #[test] fn glm_flash() { contract("serve-glmf", 7); contract("serve-glmf", 8); }
    #[test] fn kimi() { contract("serve-kimi", 7); contract("serve-kimi", 16); }
    #[test] fn muse() { contract("serve-muse", 7); }
    #[test] fn minimax() { contract("serve-minimax", 7); }
    #[test] fn inkling() { contract("serve-inkling2", 8); }
    #[test] fn inkling_mini() { contract("serve-inkling2-mini", 8); }
    #[test] fn mimo() { contract("serve-mimo", 7); contract("serve-mimo", 3); }
    #[test] fn qwen35() { contract("serve-qwen35", 7); contract("serve-qwen35", 3); }
    #[test] fn qwen4() { contract("serve-qwen4", 3); }
    #[test] fn step5() { contract("serve-step5", 3); }
}
