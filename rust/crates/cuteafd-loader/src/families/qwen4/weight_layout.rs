//! Final coordinator operands, planned from headers before any device allocation.
use super::{resident, Qwen4Attention, Qwen4Config};
use crate::{plan::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

const PREFIX: &str = "model.language_model.";

pub fn rows_key(names: &[String]) -> String { names.join("+") }

/// Header-only scale recipe for the owning layer's 256-byte FP32 view.
/// BF16 tables need no checkpoint scalar; E4M3 tables carry exactly one.
pub fn ple_scale_dtype(checkpoint: &Checkpoint, p: &str) -> Result<Option<DType>> {
    let prefix = format!("{p}.ple.ple_embedding.ngram_embedding");
    let table = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{prefix}.shard_0.weight"))
        .with_context(|| format!("Qwen PLE lacks table headers under {prefix}"))?;
    ensure!(table.meta.shape.len() == 2, "{prefix}: PLE table must be a matrix");
    match table.meta.dtype {
        DType::Bf16 => Ok(None),
        DType::F8E4M3 => {
            let name = format!("{prefix}.weight_scale");
            let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == name)
                .with_context(|| format!("Qwen PLE lacks table scale {name}"))?;
            ensure!(tensor.meta.shape.iter().product::<usize>() == 1
                && matches!(tensor.meta.dtype, DType::Bf16 | DType::F32)
                && tensor.meta.byte_length == if tensor.meta.dtype == DType::Bf16 { 2 } else { 4 },
                "{name}: expected one BF16 or FP32 table scale");
            Ok(Some(tensor.meta.dtype.clone()))
        }
        ref dtype => anyhow::bail!("{prefix}: unsupported PLE table dtype {dtype:?}"),
    }
}

struct Requests<'a> {
    checkpoint: &'a Checkpoint,
    tensors: Vec<(String, usize)>,
}
impl Requests<'_> {
    fn tensor(&self, name: &str) -> Result<&crate::plan::checkpoint::CheckpointTensor> {
        self.checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("Qwen arena lacks checkpoint tensor {name}"))
    }
    fn one(&mut self, name: String, widen: bool) -> Result<()> {
        let tensor = self.tensor(&name)?;
        ensure!(tensor.meta.dtype == DType::Bf16 || widen && tensor.meta.dtype == DType::F32,
            "{name}: unsupported Qwen arena dtype {:?}", tensor.meta.dtype);
        let bytes = usize::try_from(tensor.meta.byte_length)?;
        let bytes = if widen && tensor.meta.dtype == DType::Bf16 {
            bytes.checked_mul(2).context("Qwen widened extent overflow")?
        } else { bytes };
        self.tensors.push((name, resident::native_operand_bytes(bytes)));
        Ok(())
    }
    fn rows(&mut self, names: Vec<String>, pad: usize, fp8: bool) -> Result<()> {
        let mut rows = pad;
        let mut cols = None;
        for name in &names {
            let tensor = self.tensor(name)?;
            let shape = &tensor.meta.shape;
            ensure!(tensor.meta.dtype == DType::Bf16 && shape.len() == 2 && shape[0] > 0
                && cols.is_none_or(|c| c == shape[1]), "invalid Qwen packed matrix {name}");
            rows = rows.checked_add(shape[0]).context("Qwen packed rows overflow")?;
            cols = Some(shape[1]);
        }
        let cols = cols.context("empty Qwen row concatenation")?;
        let elements = rows.checked_mul(cols).context("Qwen packed extent overflow")?;
        let key = rows_key(&names);
        if fp8 {
            self.tensors.push((format!("{key}.fp8"), elements));
            self.tensors.push((format!("{key}.scales"), resident::fp8_block_scale_bytes(rows, cols)));
        } else {
            self.tensors.push((key, elements.checked_mul(2).context("Qwen BF16 extent overflow")?));
        }
        Ok(())
    }
    fn hc(&mut self, p: &str, inject: bool) -> Result<()> {
        self.one(format!("{p}.hc_norm.weight"), false)?;
        let mut down = vec![format!("{p}.input_mix_weight_down.weight")];
        if inject { down.push(format!("{p}.block_inject_weight.weight")); }
        self.rows(down, 0, false)?;
        self.one(format!("{p}.input_mix_weight_up.weight"), false)
    }
    fn layer(&mut self, cfg: &Qwen4Config, p: &str, attention: Qwen4Attention, ple: bool,
        fp8: bool) -> Result<()> {
        for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
            self.hc(&format!("{p}.{site}"), true)?;
        }
        match attention {
            Qwen4Attention::Gdn => {
                let a = format!("{p}.linear_attn");
                for n in ["conv1d.weight", "A_log", "dt_bias"] { self.one(format!("{a}.{n}"), true)?; }
                self.one(format!("{a}.norm.weight"), false)?;
                self.rows(["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"].map(|n|
                    format!("{a}.{n}.weight")).to_vec(), 0, fp8)?;
                self.rows(vec![format!("{a}.out_proj.weight")], 0, fp8)?;
            }
            Qwen4Attention::Full => {
                let a = format!("{p}.self_attn");
                for n in ["q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"] {
                    self.one(format!("{a}.{n}.weight"), false)?;
                }
                self.rows(["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"].map(|n|
                    format!("{a}.{n}.weight")).to_vec(), 0, fp8)?;
                self.rows(vec![format!("{a}.o_proj.weight")], 0, fp8)?;
            }
        }
        self.one(format!("{p}.mlp.gate.weight"), false)?;
        let packed = 2 * cfg.shared_intermediate + 1;
        ensure!(packed <= 1296, "Qwen shared expert exceeds packed rows");
        self.rows(["shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert_gate"].map(|n|
            format!("{p}.mlp.{n}.weight")).to_vec(), 1296 - packed, false)?;
        self.one(format!("{p}.mlp.shared_expert.down_proj.weight"), false)?;
        if ple {
            self.rows(["key_proj", "value_proj"].map(|n| format!("{p}.ple.{n}.weight")).to_vec(), 0, false)?;
            for n in ["norm_key", "norm_query", "norm_conv"] { self.one(format!("{p}.ple.{n}.weight"), false)?; }
            self.one(format!("{p}.ple.conv1d.weight"), true)?;
            ple_scale_dtype(self.checkpoint, p)?;
            self.tensors.push((format!("{p}.ple.scale"), 256));
        }
        Ok(())
    }
}

/// Whole-layer placement is preserved. The optional native MTP is a separate
/// lifetime scope on rank zero and borrows the target head without a request.
pub fn coordinator_arenas(checkpoint: &Checkpoint, cfg: &Qwen4Config, owners: &[usize],
    ranks: usize, mtp: bool, selected: resident::Qwen4Representation, embedding_bytes: usize)
    -> Result<Vec<(usize, WeightArenaLayout)>> {
    ensure!(ranks > 0 && owners.len() <= cfg.layers && owners.iter().all(|&rank| rank < ranks),
        "invalid Qwen arena placement");
    let mut out = Vec::new();
    for rank in 0..ranks {
        let mut requests = Requests { checkpoint, tensors: Vec::new() };
        if rank == 0 {
            let head = requests.tensor("lm_head.weight")?;
            ensure!(head.meta.dtype == DType::Bf16 && head.meta.shape == [cfg.vocab_size, cfg.hidden],
                "Qwen head must be BF16 [vocab, hidden]");
            if selected.fp8_head {
                ensure!(cfg.hidden % 128 == 0, "Qwen FP8 head needs whole 128-K blocks");
                requests.tensors.push(("lm_head.weight.fp8".into(), (cfg.vocab_size * cfg.hidden).max(256)));
                requests.tensors.push(("lm_head.weight.scales".into(), (cfg.vocab_size * (cfg.hidden / 128) * 4).max(256)));
            } else { requests.one("lm_head.weight".into(), false)?; }
            requests.hc(&format!("{PREFIX}hyper_connection_mixer"), false)?;
            if embedding_bytes > 0 { requests.tensors.push(("embedding.table".into(), embedding_bytes)); }
        }
        for (layer, _) in owners.iter().enumerate().filter(|(_, owner)| **owner == rank) {
            requests.layer(cfg, &format!("{PREFIX}layers.{layer}"), cfg.attention[layer],
                cfg.ple_layers.contains(&layer), selected.fp8_projections)?;
        }
        if !requests.tensors.is_empty() {
            out.push((rank, WeightArenaLayout::new(format!("qwen4/target/rank{rank}"), requests.tensors)?));
        }
    }
    if mtp {
        ensure!(cfg.mtp_layers == 1, "Qwen arena requires one MTP layer");
        let mut requests = Requests { checkpoint, tensors: Vec::new() };
        requests.layer(cfg, "mtp.layers.0", Qwen4Attention::Full, false, selected.fp8_projections)?;
        for name in ["pre_fc_norm_hidden", "pre_fc_norm_embedding", "fc_hidden", "fc_embedding"] {
            requests.one(format!("mtp.{name}.weight"), false)?;
        }
        requests.hc("mtp.hyper_connection_mixer", false)?;
        out.push((0, WeightArenaLayout::new("qwen4/mtp/rank0", requests.tensors)?));
    }
    Ok(out)
}

/// Physical backing bytes of the single-owner arenas (target plus optional
/// MTP), `layout.bytes()` each: what TP1 declares once against its post-load
/// sampled baseline. `embedding_bytes` is the device table's header extent, 0
/// for a host embedding. Both the planner and the runtime call this.
pub fn single_arena_bytes(checkpoint: &Checkpoint, cfg: &Qwen4Config, layers: usize, mtp: bool,
    selected: resident::Qwen4Representation, embedding_bytes: usize) -> Result<u64> {
    let layers = layers.min(cfg.layers);
    coordinator_arenas(checkpoint, cfg, &vec![0; layers], 1, mtp, selected, embedding_bytes)?
        .iter().try_fold(0u64, |sum, (_, layout)| sum.checked_add(layout.bytes() as u64)
            .context("Qwen single arena bytes overflow"))
}

/// The dual route's arenas at its fixed contiguous cut (`dual_owners`): each
/// GPU's target arena and the entry rank's MTP arena, the exact layouts the
/// loader allocates. TP2 weights are future allocations, so the planner
/// charges each layout's physical bytes and tail once, from these.
pub fn dual_arenas(checkpoint: &Checkpoint, cfg: &Qwen4Config, owners: &[usize], mtp: bool,
    selected: resident::Qwen4Representation, embedding_bytes: usize) -> Result<Vec<(usize, WeightArenaLayout)>> {
    ensure!(owners.len() > 1 && owners.contains(&0) && owners.contains(&1),
        "Qwen dual arenas need both owners");
    coordinator_arenas(checkpoint, cfg, owners, 2, mtp, selected, embedding_bytes)
}

/// The device embedding table's header extent, as the loader allocates it.
pub fn embedding_table_bytes(checkpoint: &Checkpoint) -> Result<usize> {
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{PREFIX}embed_tokens.weight"))
        .context("Qwen arena lacks the token embedding")?;
    Ok(usize::try_from(tensor.meta.byte_length)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{qwen4_config, t, write_snapshot};

    fn checkpoint() -> (tempfile::TempDir, Checkpoint, Qwen4Config) {
        let dir = tempfile::tempdir().unwrap();
        let mut config = qwen4_config(2);
        config["text_config"]["layer_types"] = serde_json::json!(["linear_attention", "full_attention"]);
        config["text_config"]["ple_layer_ids"] = serde_json::json!([1]);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let cfg = Qwen4Config::from_hf(&config).unwrap();
        let mut tensors = vec![t("lm_head.weight", "BF16", &[cfg.vocab_size, cfg.hidden]),
            t(format!("{PREFIX}embed_tokens.weight"), "BF16", &[cfg.vocab_size, cfg.hidden])];
        let mut hc = |p: &str, inject: bool| {
            tensors.extend([t(format!("{p}.hc_norm.weight"), "BF16", &[cfg.hidden]),
                t(format!("{p}.input_mix_weight_down.weight"), "BF16", &[4, cfg.hidden]),
                t(format!("{p}.input_mix_weight_up.weight"), "BF16", &[cfg.hidden, 4])]);
            if inject { tensors.push(t(format!("{p}.block_inject_weight.weight"), "BF16", &[4, cfg.hidden])); }
        };
        hc(&format!("{PREFIX}hyper_connection_mixer"), false);
        hc("mtp.hyper_connection_mixer", false);
        for p in [format!("{PREFIX}layers.0"), format!("{PREFIX}layers.1"), "mtp.layers.0".into()] {
            hc(&format!("{p}.attn_hyper_connection"), true);
            hc(&format!("{p}.mlp_hyper_connection"), true);
        }
        for (p, gdn, ple) in [(format!("{PREFIX}layers.0"), true, true),
            (format!("{PREFIX}layers.1"), false, false), ("mtp.layers.0".into(), false, false)] {
            if gdn {
                for (n, rows) in [("in_proj_qkv", cfg.gdn_conv_width()), ("in_proj_z", cfg.gdn_value_width()),
                    ("in_proj_b", cfg.gdn_value_heads), ("in_proj_a", cfg.gdn_value_heads)] {
                    tensors.push(t(format!("{p}.linear_attn.{n}.weight"), "BF16", &[rows, cfg.hidden]));
                }
                tensors.extend([t(format!("{p}.linear_attn.out_proj.weight"), "BF16", &[cfg.hidden, cfg.gdn_value_width()]),
                    t(format!("{p}.linear_attn.conv1d.weight"), "BF16", &[cfg.gdn_conv_width(), 1, 4]),
                    t(format!("{p}.linear_attn.A_log"), "BF16", &[cfg.gdn_value_heads]),
                    t(format!("{p}.linear_attn.dt_bias"), "F32", &[cfg.gdn_value_heads]),
                    t(format!("{p}.linear_attn.norm.weight"), "BF16", &[128])]);
            } else {
                // Only the concatenated full-attention width matters to packing.
                for (n, rows) in [("q_proj", cfg.attn_in_width() - 3), ("k_proj", 1), ("v_proj", 1), ("indexer.index_qk_proj", 1)] {
                    tensors.push(t(format!("{p}.self_attn.{n}.weight"), "BF16", &[rows, cfg.hidden]));
                }
                tensors.push(t(format!("{p}.self_attn.o_proj.weight"), "BF16", &[cfg.hidden, cfg.heads * cfg.head_dim]));
                for n in ["q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"] {
                    tensors.push(t(format!("{p}.self_attn.{n}.weight"), "BF16", &[128]));
                }
            }
            for (n, shape) in [("gate", vec![cfg.experts, cfg.hidden]),
                ("shared_expert.gate_proj", vec![cfg.shared_intermediate, cfg.hidden]),
                ("shared_expert.up_proj", vec![cfg.shared_intermediate, cfg.hidden]),
                ("shared_expert_gate", vec![1, cfg.hidden]),
                ("shared_expert.down_proj", vec![cfg.hidden, cfg.shared_intermediate])] {
                tensors.push(t(format!("{p}.mlp.{n}.weight"), "BF16", &shape));
            }
            if ple {
                for n in ["key_proj", "value_proj"] { tensors.push(t(format!("{p}.ple.{n}.weight"), "BF16", &[cfg.hidden, cfg.hidden])); }
                for n in ["norm_key", "norm_query", "norm_conv"] { tensors.push(t(format!("{p}.ple.{n}.weight"), "BF16", &[cfg.hidden])); }
                tensors.push(t(format!("{p}.ple.conv1d.weight"), "BF16", &[cfg.hidden, 1, 4]));
                tensors.push(t(format!("{p}.ple.ple_embedding.ngram_embedding.shard_0.weight"), "BF16", &[100, 160]));
            }
        }
        for (n, shape) in [("pre_fc_norm_hidden", vec![4 * cfg.hidden]), ("pre_fc_norm_embedding", vec![cfg.hidden]),
            ("fc_hidden", vec![cfg.hidden, cfg.hidden]), ("fc_embedding", vec![cfg.hidden, cfg.hidden])] {
            tensors.push(t(format!("mtp.{n}.weight"), "BF16", &shape));
        }
        write_snapshot(dir.path(), &config, &tensors, None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        (dir, checkpoint, cfg)
    }

    /// TP1 declares each physical backing once: B = payload + inter-view
    /// padding per arena, never the per-tensor payload sum alone, and the
    /// device embedding adds exactly its header extent to the target arena.
    #[test]
    fn single_arena_bytes_are_the_physical_backings_once() -> Result<()> {
        let (_dir, checkpoint, cfg) = checkpoint();
        let embedding = embedding_table_bytes(&checkpoint)?;
        assert_eq!(embedding, cfg.vocab_size * cfg.hidden * 2);
        for fp8_projections in [false, true] { for fp8_head in [false, true] { for mtp in [false, true] {
            let selected = resident::Qwen4Representation { fp8_projections, fp8_head };
            for device_embedding in [0, embedding] {
                let layouts = coordinator_arenas(&checkpoint, &cfg, &vec![0; cfg.layers], 1, mtp, selected, device_embedding)?;
                assert_eq!(layouts.len(), 1 + usize::from(mtp));
                let backings = layouts.iter().map(|(_, layout)| layout.bytes() as u64).sum::<u64>();
                let payload = layouts.iter().map(|(_, layout)| layout.payload_bytes() as u64).sum::<u64>();
                let padding = layouts.iter().map(|(_, layout)| layout.padding_bytes() as u64).sum::<u64>();
                assert_eq!(backings, payload + padding);
                assert_eq!(single_arena_bytes(&checkpoint, &cfg, cfg.layers, mtp, selected, device_embedding)?, backings);
            }
            let host = single_arena_bytes(&checkpoint, &cfg, cfg.layers, mtp, selected, 0)?;
            let device = single_arena_bytes(&checkpoint, &cfg, cfg.layers, mtp, selected, embedding)?;
            // The table is one 256-aligned view appended to the target arena.
            assert!(device >= host + embedding as u64 && device < host + embedding as u64 + 256);
            // A shorter backbone prefix plans fewer layers, never more.
            assert!(single_arena_bytes(&checkpoint, &cfg, 1, mtp, selected, 0)? < host);
        }}}
        Ok(())
    }

    #[test]
    fn named_layout_preserves_placement_head_scope_formats_and_padding() -> Result<()> {
        let (_dir, checkpoint, cfg) = checkpoint();
        for fp8_projections in [false, true] { for fp8_head in [false, true] {
            let selected = resident::Qwen4Representation { fp8_projections, fp8_head };
            for owners in [vec![0, 0], vec![0, 1], vec![1, 0]] {
                let ranks = 1 + usize::from(owners.contains(&1));
                for mtp in [false, true] {
                    let layouts = coordinator_arenas(&checkpoint, &cfg, &owners, ranks, mtp, selected, 4096)?;
                    assert_eq!(layouts.len(), ranks + usize::from(mtp));
                    for (rank, layout) in &layouts {
                        let draft = layout.id().contains("/mtp/");
                        let mut regions = layout.views().values().collect::<Vec<_>>();
                        regions.sort_by_key(|r| r.offset());
                        assert!(regions.iter().all(|r| r.offset() % 256 == 0 && r.bytes() > 0));
                        assert!(regions.windows(2).all(|r| r[0].offset() + r[0].bytes() <= r[1].offset()));
                        assert_eq!(regions.last().unwrap().offset() + regions.last().unwrap().bytes(), layout.bytes());
                        assert_eq!(layout.payload_bytes() + layout.padding_bytes(), layout.bytes());
                        for name in layout.views().keys() {
                            assert_eq!(name.starts_with("mtp."), draft);
                            assert!(!name.contains("ngram_embedding") && !name.contains("experts."));
                            if draft { assert!(!name.contains("lm_head") && name != "embedding.table"); }
                            for (layer, &owner) in owners.iter().enumerate() {
                                if name.starts_with(&format!("{PREFIX}layers.{layer}.")) { assert_eq!(*rank, owner); }
                            }
                        }
                    }
                    let scale = format!("{PREFIX}layers.0.ple.scale");
                    let scale_owners = layouts.iter().filter(|(_, layout)| layout.view(&scale).is_some())
                        .collect::<Vec<_>>();
                    assert_eq!(scale_owners.len(), 1);
                    assert_eq!(scale_owners[0].0, owners[0]);
                    assert_eq!(scale_owners[0].1.view(&scale).unwrap().bytes(), 256);
                    let target = &layouts[0].1;
                    assert!(target.view("embedding.table").is_some());
                    assert_eq!(target.view("lm_head.weight.fp8").is_some(), fp8_head);
                    assert_eq!(target.view("lm_head.weight").is_some(), !fp8_head);
                }
            }
        }}
        Ok(())
    }

    /// #1519 charge table on the installed checkpoint: per shipped layout and
    /// owner GPU, the old payload-only admission charge, the new arena B plus
    /// its single SM120 tail, and the delta. Old Qwen admission charged per-
    /// tensor payload with no granularity owner, so it included no rounding.
    #[test]
    fn installed_qwen_arena_charge_table() -> Result<()> {
        let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_QWEN4") else { return Ok(()) };
        let checkpoint = Checkpoint::open(std::path::Path::new(&snapshot))?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(checkpoint.snapshot.join("config.json"))?)?;
        let cfg = Qwen4Config::from_hf(&config)?;
        let tail = |bytes: u64| bytes.div_ceil(2 << 20) * (2 << 20) - bytes;
        let embedding = embedding_table_bytes(&checkpoint)?;
        for fp8_projections in [false, true] { for fp8_head in [false, true] { for device_embedding in [false, true] {
            let selected = resident::Qwen4Representation { fp8_projections, fp8_head };
            let r = resident::checkpoint_resident_bytes(&checkpoint, &cfg, cfg.layers, true, selected)?;
            let table = if device_embedding { embedding } else { 0 };
            let embed = if device_embedding { r.embedding_bytes } else { 0 };
            // TP1: old = target + head + MTP + embedding payload (PLE scale unchanged, excluded).
            let old = r.target_bytes + r.head_bytes + r.mtp_bytes + embed;
            let arenas = coordinator_arenas(&checkpoint, &cfg, &vec![0; cfg.layers], 1, true, selected, table)?;
            let new = arenas.iter().map(|(_, l)| l.bytes() as u64 + tail(l.bytes() as u64)).sum::<u64>();
            assert_eq!(arenas.iter().map(|(_, l)| l.payload_bytes() as u64).sum::<u64>(), old, "TP1 payload");
            eprintln!("QWEN_CHARGE layout=TP1 proj={} head={} embed={} gpu=0 old={old} new={new} delta={}",
                if fp8_projections { "fp8" } else { "bf16" }, if fp8_head { "fp8" } else { "bf16" },
                if device_embedding { "gpu" } else { "host" }, new as i64 - old as i64);
            // TP2 at the fixed cut: old = owned layer payloads, plus owner0's
            // head, entry mixer, MTP and embedding.
            let owners = (0..cfg.layers).map(|l| usize::from(l >= cfg.layers / 2)).collect::<Vec<_>>();
            let arenas = dual_arenas(&checkpoint, &cfg, &owners, true, selected, table)?;
            for gpu in 0..2 {
                let layers = owners.iter().enumerate().filter(|(_, &o)| o == gpu).map(|(l, _)| r.layer_bytes[l]).sum::<u64>();
                let old = layers + if gpu == 0 { r.entry_bytes + r.head_bytes + r.mtp_bytes + embed } else { 0 };
                let mine = arenas.iter().filter(|(rank, _)| *rank == gpu).map(|(_, l)| l);
                let (payload, new) = mine.fold((0u64, 0u64), |(p, n), l|
                    (p + l.payload_bytes() as u64, n + l.bytes() as u64 + tail(l.bytes() as u64)));
                assert_eq!(payload, old, "TP2 payload gpu {gpu}");
                eprintln!("QWEN_CHARGE layout=TP2 proj={} head={} embed={} gpu={gpu} old={old} new={new} delta={}",
                    if fp8_projections { "fp8" } else { "bf16" }, if fp8_head { "fp8" } else { "bf16" },
                    if device_embedding { "gpu" } else { "host" }, new as i64 - old as i64);
            }
        }}}
        Ok(())
    }

    #[test]
    fn installed_qwen_arena_inventory() -> Result<()> {
        let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_QWEN4") else { return Ok(()) };
        let checkpoint = Checkpoint::open(std::path::Path::new(&snapshot))?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(checkpoint.snapshot.join("config.json"))?)?;
        let cfg = Qwen4Config::from_hf(&config)?;
        for fp8_projections in [false, true] { for fp8_head in [false, true] {
            for ranks in [1, 2] {
                let owners = (0..cfg.layers).map(|l| usize::from(ranks == 2 && l >= cfg.layers / 2)).collect::<Vec<_>>();
                let layouts = coordinator_arenas(&checkpoint, &cfg, &owners, ranks, true,
                    resident::Qwen4Representation { fp8_projections, fp8_head }, 0)?;
                for (rank, layout) in layouts {
                    let large = layout.views().values().filter(|v| v.bytes() > 1 << 20).collect::<Vec<_>>();
                    let slack = large.iter().map(|v| v.bytes().div_ceil(2 << 20) * (2 << 20) - v.bytes()).sum::<usize>();
                    eprintln!("QWEN_ARENA fp8={fp8_projections} head_fp8={fp8_head} ranks={ranks} rank={rank} scope={} n={} payload={} large_n={} large_bytes={} old_slack={} arena={} padding={} tail={}",
                        layout.id(), layout.views().len(), layout.payload_bytes(), large.len(), large.iter().map(|v| v.bytes()).sum::<usize>(),
                        slack, layout.bytes(), layout.padding_bytes(), layout.bytes().div_ceil(2 << 20) * (2 << 20) - layout.bytes());
                }
            }
        }}
        Ok(())
    }
}
