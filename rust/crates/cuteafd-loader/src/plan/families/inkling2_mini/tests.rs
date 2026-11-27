use super::*;
use crate::families::inkling2_mini::config::tests::release_config;
use crate::families::inkling2_mini::weight_layout::{
    CONTROLLER_STREAMS, CROSS_PASS_KV_QUERIES, CROSS_PASS_QUERIES,
};
use crate::plan::{testing::*, ExpertPlacement, PlanInput, PlanOptions};

fn mini_snapshot_tensors() -> Vec<(&'static str, String, Vec<usize>)> {
    let mut out: Vec<(&'static str, String, Vec<usize>)> = Vec::new();
    let mut push = |dtype: &'static str, name: String, shape: Vec<usize>| {
        out.push((dtype, name, shape));
    };
    let (h, v, d, heads, kv_heads, d_rel, rel_ext, win, k, e, i) =
        (2304, 201024, 128, 24, 8, 16, 1024, 1024, 4, 128, 1152);
    push("BF16", format!("{TEXT}embed.weight"), vec![v, h]);
    push("BF16", format!("{TEXT}unembed.weight"), vec![v, h]);
    push("BF16", format!("{TEXT}embed_norm.weight"), vec![h]);
    push("BF16", format!("{TEXT}norm.weight"), vec![h]);
    push("F32", format!("{TEXT}attn_res.output_query"), vec![h]);
    push("F32", CROSS_PASS_QUERIES.into(), vec![24, h]);
    push("F32", CROSS_PASS_KV_QUERIES.into(), vec![24, h]);

    let globals = [5usize, 11, 17, 23];
    for layer in 0..24usize {
        let local = !globals.contains(&layer);
        let base = format!("{TEXT}layers.{layer}");
        let a = format!("{base}.attn");
        for (proj, n, cols) in [
            ("wq_du", heads * d, h),
            ("wr_du", heads * d_rel, h),
            ("wk_dv", kv_heads * d, h),
            ("wv_dv", kv_heads * d, h),
            ("gate_attn", heads * d, h),
            ("attn_sink_proj", heads, h),
            ("wo_ud", h, heads * d),
        ] {
            push("BF16", format!("{a}.{proj}.weight"), vec![n, cols]);
        }
        push("BF16", format!("{a}.q_l2_clip.scale"), vec![heads, d]);
        push("BF16", format!("{a}.rel_l2_clip.scale"), vec![heads, d_rel]);
        push("BF16", format!("{a}.rel_logits_proj.proj"), vec![d_rel, if local { win } else { rel_ext }]);
        for name in ["k", "v"] {
            if local {
                push("BF16", format!("{a}.{name}_sconv.weight"), vec![kv_heads * d, 1, k]);
            } else {
                push("BF16", format!("{a}.{name}_sconv.static_w"), vec![k, kv_heads * d]);
                push("BF16", format!("{a}.{name}_sconv.weight_proj"), vec![kv_heads * k, h]);
            }
        }
        for name in ["attn_sconv", "mlp_sconv"] {
            push("BF16", format!("{base}.{name}.weight"), vec![h, 1, k]);
        }
        let norms: &[&str] = if local {
            &["attn_norm"]
        } else {
            &["q_input_norm", "k_input_norm", "v_input_norm"]
        };
        for norm in norms.iter().chain(["mlp_norm"].iter()) {
            push("BF16", format!("{base}.{norm}.weight"), vec![h]);
        }
        push("BF16", format!("{base}.mlp.gate.weight"), vec![e, h]);
        push("F32", format!("{base}.mlp.gate.bias"), vec![e]);
        push("F32", format!("{TEXT}attn_res.mlp_queries.{layer}"), vec![h]);
        if layer > 0 {
            push(
                "F32",
                format!("{TEXT}attn_res.attn_queries.{}", layer - 1),
                if local { vec![h] } else { vec![3, h] },
            );
        }
        if (6..18).contains(&layer) {
            for stream in CONTROLLER_STREAMS {
                let c = format!("{base}.controllers.{stream}");
                push("BF16", format!("{c}.down_weight"), vec![64, 2 * h]);
                push("BF16", format!("{c}.forget_weight"), vec![h, 64]);
                push("BF16", format!("{c}.gain_weight"), vec![h, 64]);
            }
        }
        let w13 = expert_stem(layer, "w13");
        let w2 = expert_stem(layer, "w2");
        if layer < 20 {
            push("U8", format!("{w13}_weight"), vec![e, 2 * i, h / 2]);
            push("F8_E4M3", format!("{w13}_weight_scale"), vec![e, 2 * i, h / 16]);
            push("F32", format!("{w13}_weight_amax"), vec![e]);
            push("F32", format!("{w13}_weight_outer_scale_denominator"), vec![1]);
            push("U8", format!("{w2}_weight"), vec![e, h, i / 2]);
            push("F8_E4M3", format!("{w2}_weight_scale"), vec![e, h, i / 16]);
            push("F32", format!("{w2}_weight_amax"), vec![e]);
            push("F32", format!("{w2}_weight_outer_scale_denominator"), vec![1]);
        } else {
            push("F8_E4M3", format!("{w13}_weight"), vec![e, 2 * i, h]);
            push("U8", format!("{w13}_weight_scale"), vec![e, 2 * i, h / 32]);
            push("F8_E4M3", format!("{w2}_weight"), vec![e, h, i]);
            push("U8", format!("{w2}_weight_scale"), vec![e, h, i / 32]);
        }
    }
    out
}

pub(crate) fn mini_test_checkpoint() -> anyhow::Result<(crate::families::inkling2_mini::Inkling2MiniConfig, tempfile::TempDir, crate::plan::Checkpoint)> {
    let dir = tempfile::tempdir()?;
    let raw = mini_snapshot_tensors();
    let specs: Vec<_> = raw
        .iter()
        .map(|(dtype, name, shape)| t(name.clone(), dtype, shape))
        .collect();
    write_snapshot(dir.path(), &release_config(), &specs, None);
    let checkpoint = crate::plan::Checkpoint::open(dir.path())?;
    let cfg = crate::families::inkling2_mini::Inkling2MiniConfig::from_hf(&checkpoint.config)?;
    Ok((cfg, dir, checkpoint))
}

#[test]
fn mini_plan_detects_classifies_and_accepts_every_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let raw = mini_snapshot_tensors();
    let specs: Vec<_> = raw
        .iter()
        .map(|(dtype, name, shape)| t(name.clone(), dtype, shape))
        .collect();
    write_snapshot(dir.path(), &release_config(), &specs, None);
    let input = PlanInput::open(dir.path()).unwrap();
    let mut opts = PlanOptions::default();
    opts.placement = ExpertPlacement::Local;
    opts.vision = crate::plan::MediaMode::Off;
    opts.layout = Some(crate::plan::layout::LayoutOptions::default());
    let report = input.plan(&opts).unwrap();
    assert_eq!(report.family.as_deref(), Some("inkling2_mini"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Serving));
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    for comp in &report.components {
        assert_eq!(comp.rejected, 0, "{:?}: {:?}", comp.component, comp.rejections);
    }
    let layout = report.memory_layout.as_ref().expect("TP1 memory layout");
    assert_eq!(layout.devices.len(), 1, "hints={:?} notes={:?}", report.hints, layout.notes);
    assert!(layout.pool_tokens >= 262_144, "pool_tokens={}", layout.pool_tokens);

    // Explicit DECODE_ROWS=128 plans cleanly and charges decode scratch, head logits, and graph reserve once.
    let mut wide_opts = PlanOptions::default();
    wide_opts.placement = ExpertPlacement::Local;
    wide_opts.vision = crate::plan::MediaMode::Off;
    let mut wide_layout_opts = crate::plan::layout::LayoutOptions::default();
    wide_layout_opts.decode_rows = 128;
    wide_opts.layout = Some(wide_layout_opts);
    let wide_report = input.plan(&wide_opts).unwrap();
    assert!(wide_report.config_error.is_none(), "{:?}", wide_report.config_error);
    let wide_layout = wide_report.memory_layout.as_ref().expect("TP1 128-row layout");
    assert_eq!(wide_layout.devices.len(), 1, "hints={:?} notes={:?}", wide_report.hints, wide_layout.notes);
    assert!(wide_layout.pool_tokens >= 262_144, "pool_tokens={}", wide_layout.pool_tokens);
    let group_bytes = |layout: &cuteafd_core::memory_layout::MemoryLayout, group: &str| -> u64 {
        layout.devices[0]
            .items
            .iter()
            .filter(|i| i.group == group)
            .map(|i| i.bytes)
            .sum()
    };
    let base_head = group_bytes(layout, "Inkling 2 Mini bounded vocabulary head rows");
    let wide_head = group_bytes(wide_layout, "Inkling 2 Mini bounded vocabulary head rows");
    assert!(base_head > 0);
    assert_eq!(wide_head - base_head, 64 * (4 + 2304 * 2 + 200_058 * 4));
}

#[test]
fn mini_plan_charges_key_space_scaled_graph_reserve_once() {
    use crate::families::inkling2_mini::execution::DECODE_ROWS;
    use crate::serving_capacity::inkling2_mini_graphs::{self, KeySpace};

    let (_cfg, dir, _checkpoint) = mini_test_checkpoint().unwrap();
    let input = PlanInput::open(dir.path()).unwrap();
    let charge = |concurrency: u64, decode_rows: u32| -> (u64, usize) {
        let mut opts = PlanOptions::default();
        opts.placement = ExpertPlacement::Local;
        opts.vision = crate::plan::MediaMode::Off;
        let mut layout = crate::plan::layout::LayoutOptions::default();
        layout.concurrency = concurrency;
        layout.decode_rows = decode_rows;
        layout.inkling_dspark = false;
        opts.layout = Some(layout);
        let report = input.plan(&opts).unwrap();
        let placed = report.memory_layout.as_ref().expect("TP1 memory layout");
        let graph_items: Vec<_> = placed.devices[0]
            .items
            .iter()
            .filter(|item| item.group.contains("inkling2_mini.graphs.0"))
            .collect();
        (graph_items.iter().map(|i| i.bytes).sum(), graph_items.len())
    };
    for concurrency in [1u64, 16, 64] {
        let (bytes, count) = charge(concurrency, DECODE_ROWS as u32);
        let expected = inkling2_mini_graphs::policy(0, KeySpace::for_config(concurrency, DECODE_ROWS, false))
            .unwrap()
            .bytes;
        assert_eq!(count, 1, "graph reserve must be charged exactly once at C{concurrency}");
        assert_eq!(bytes, expected, "graph reserve mismatch at C{concurrency}");
    }
    assert_eq!(charge(64, 128), charge(64, DECODE_ROWS as u32));
    assert!(charge(64, 64).0 > charge(16, 64).0 && charge(16, 64).0 > charge(1, 64).0);
}
