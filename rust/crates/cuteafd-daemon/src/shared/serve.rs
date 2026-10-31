//! Serve-loop parts a family scheduler composes: independent decode lanes
//! ([`lanes`]) and the two-part lane commit ([`lanes::LaneCommit`]).
pub(crate) mod lanes;

#[cfg(test)]
mod readiness_tests {
    fn callback_follows_scheduler(source: &str) {
        let startup = source.split_once("\nfn serve_loop(").unwrap().1
            .split_once("\nfn schedule(").unwrap().0;
        let schedule = startup.find("        schedule(engine,").unwrap();
        let signal = startup.find("ready.send(Ok(").unwrap();
        assert!(schedule < signal, "readiness must be a scheduler callback, not an early signal");
    }

    #[test]
    fn every_family_publishes_kv_capacity_from_its_prefix_cache_path() {
        for (family, source, start, end) in [
            ("V4.1", include_str!("../families/deepseek_v41/v41_native_serve/scheduler.rs"), "pub(crate) fn publish_capacity(", "\n/// Occupancy"),
            ("V4", include_str!("../families/deepseek_v4/serve.rs"), "\nfn prefix_cache<", "\nfn schedule("),
            ("GLM", include_str!("../families/glm5/serve.rs"), "\nfn prefix_cache<", "\nfn schedule("),
            ("GLM Flash", include_str!("../families/glm5_flash/serve.rs"), "\nfn prefix_cache<", "\nfn schedule("),
            ("MiMo", include_str!("../families/mimo_v2/serve.rs"), "\nfn prefix_cache_with<", "\n/// Gives a finished"),
            ("Qwen4", include_str!("../families/qwen4/serve.rs"), "\nfn prefix_cache<", "\nstruct Qwen4Backend"),
            ("Kimi", include_str!("../families/kimi_k3/backend.rs"), "    fn prefix_cache(", "    fn prepare("),
            ("Step-5", include_str!("../families/step5/backend.rs"), "    fn prefix_cache(", "    fn prepare("),
            ("dense Qwen", include_str!("../families/qwen3_5/serve.rs"), "    fn prefix_cache(", "    fn prepare("),
        ] {
            let prefix = source.split_once(start).unwrap_or_else(|| panic!("{family} prefix path missing")).1
                .split_once(end).unwrap_or_else(|| panic!("{family} prefix path end missing")).0;
            assert!(prefix.contains("cuteafd_bench::context::set_kv("), "{family} must publish resolved KV capacity");
        }
    }

    #[test]
    fn mimo_capacity_and_all_fallible_startup_precede_readiness() {
        let source = include_str!("../families/mimo_v2/serve.rs");
        callback_follows_scheduler(source);
        let schedule = source.split_once("\nfn schedule(").unwrap().1;
        assert!(schedule.find("prefix_cache_with(").unwrap() < schedule.find("    schedule_inner(").unwrap());
        let inner = schedule.split_once("\nfn schedule_inner(").unwrap().1;
        let ready = inner.find("    ready();").unwrap();
        let loop_at = inner.find("loop {").unwrap();
        assert!(inner.find("prefix::DeferredAdmission").unwrap() < ready && ready < loop_at);
        assert!(inner[..ready].contains("publish(stats, requests, generated_total, 0, 0, cache, media, preparer);"));
        assert!(!inner[ready..loop_at].contains('?'), "no fallible startup work may follow readiness");
        let prefix = source.split_once("\nfn prefix_cache_with").unwrap().1
            .split_once("\n/// Gives a finished").unwrap().0;
        assert!(prefix.contains("cuteafd_bench::context::set_kv("));
    }

    #[test]
    fn glm_and_v4_capacity_and_all_fallible_startup_precede_readiness() {
        for source in [include_str!("../families/glm5/serve.rs"), include_str!("../families/deepseek_v4/serve.rs")] {
            callback_follows_scheduler(source);
            let schedule = source.split_once("\nfn schedule(").unwrap().1;
            let prefix = schedule.find("prefix_cache(").unwrap();
            let ready = schedule.find("    ready();").unwrap();
            let loop_at = schedule.find("    loop {").unwrap();
            assert!(prefix < ready && ready < loop_at);
            assert!(schedule[..ready].contains("decode_share.queue::<Prefill<'_>>()?"));
            assert!(!schedule[ready..loop_at].contains('?'), "no fallible startup work may follow readiness");
            assert!(source[..source.find("\nfn schedule(").unwrap()].contains("cuteafd_bench::context::set_kv("));
        }
    }
}
