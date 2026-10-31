//! Shared short-prompt layout and segment-local consumers on ordinary prefill units.
use super::*;
use crate::shared::prefill_packing::{self, Limits, MemberOutcome, PackingMode, Segment};

impl<'a> GlmEngine<'a> {
    pub(crate) fn configure_packing(&self, mode: PackingMode) -> Result<()> {
        self.packed_fallback_reason.borrow_mut().clear();
        if mode == PackingMode::Unrestricted { self.packed_row_cap.set(None); return Ok(()); }
        let mut reasons = Vec::new();
        let inventory = coordinator_inventory(&self.cfg, self.programs.names(), self.peer.is_some())?;
        let mut cap = exact_program_cap(self.programs,
            inventory.iter().map(|(name, role)| (name.as_str(), *role)), self.prefill_rows, &mut reasons)?;
        for owner in self.native_dense.iter().flatten() {
            cap = native_owner_cap(exact_program_cap(self.programs,
                owner.packed_programs().map(|name| (name, prefill_packing::ProgramRows::PackedRows)), cap, &mut reasons)?);
        }
        for owner in self.core_programs.iter().flatten() {
            cap = exact_program_cap(self.programs,
                owner.packed_programs().map(|name| (name, prefill_packing::ProgramRows::PackedRows)), cap, &mut reasons)?;
        }
        // The native shared expert's bounded 64-row executor has no route
        // attestation yet; do not infer invariance from its staging capacity.
        if !self.native_experts.is_empty() {
            cap = 0;
            reasons.push("native shared EXLR8 expert has no arithmetic-route attestation".into());
        }
        // TP2's erased backend does not publish arithmetic-route bounds yet.
        // Exact validation fails closed; unrestricted serving still packs.
        if self.tp2.is_some() {
            cap = 0;
            reasons.push("TP2 backend has no arithmetic-route attestation".into());
        }
        let reports = self.prefill_link.borrow().as_ref().map(|link| link.installed_grouping()).transpose()?
            .unwrap_or_default();
        for (index, layer) in self.weights.layers.iter().enumerate() {
            if layer.dense || self.tp2_layer(index) || self.skip.is_some() { continue; }
            let mut found = false;
            for report in &reports {
                if report.layers().iter().any(|range| (range.first as usize..range.end as usize).contains(&index)) {
                    let expert_cap = report.packed_row_cap(self.spark_bf16);
                    if expert_cap == 0 {
                        reasons.push(format!("Spark expert layer {index} has no bounded arithmetic route"));
                    }
                    cap = cap.min(expert_cap);
                    found = true;
                }
            }
            if !found {
                cap = 0;
                reasons.push(format!("Spark expert layer {index} has no installed route report"));
            }
        }
        self.packed_row_cap.set(Some(cap));
        *self.packed_fallback_reason.borrow_mut() = reasons.join("; ");
        if cap == 0 {
            tracing::info!(reason = %self.packed_fallback_reason.borrow(),
                "exact GLM packed prefill disabled; requests run alone");
        } else { prefill_packing::log_mode(mode, cap); }
        Ok(())
    }

    pub(crate) fn packed_fallback_reason(&self) -> String { self.packed_fallback_reason.borrow().clone() }

    pub(crate) fn packed_limits(&self) -> Limits {
        Limits { rows: self.packed_row_cap.get().unwrap_or(self.prefill_rows), chunk_rows: self.prefill_rows,
            max_context: self.max_context,
            taps: self.drafter.as_ref().map(|_| crate::shared::speculation::dflash::TAP_ROWS) }
    }

    /// Consume every head before shared scratch is reused for the next member.
    /// All placements advance after device success, including cancelled clients.
    pub(crate) fn prefill_packed(&self, sequences: &mut [(&mut GlmPlacement, &[u32])],
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<(Vec<Segment>, Vec<Result<()>>)> {
        self.prefill_packed_selected(sequences, &mut |_| true, on_logits)
    }

    pub(crate) fn prefill_packed_selected(&self, sequences: &mut [(&mut GlmPlacement, &[u32])],
        needs_head: &mut dyn FnMut(usize) -> bool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<(Vec<Segment>, Vec<Result<()>>)> {
        let chunks = sequences.iter().map(|(p, tokens)| (p.len, tokens.len())).collect::<Vec<_>>();
        let segments = prefill_packing::plan(&chunks, &self.packed_limits())?;
        let tables = packed_tables(sequences, &segments, self.cfg.index_topk)?;
        let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
        if self.prefill_link.borrow().is_some() {
            let mut link = self.prefill_link.borrow_mut();
            let link = link.as_mut().context("packed prefill link")?;
            let runtime = self.prefill_runtime.as_ref().context("packed prefill runtime")?;
            // One retained unit keeps segment heads/taps in slot 0. Multi-wave
            // slicing needs explicit per-segment slot remapping (#847 follow-up).
            self.step_lanes(&[tables], &[&tokens], self.depth, 0, link, runtime)?;
        } else {
            self.step(&tables, &tokens, 0, None, None)?;
        }
        for (placement, tokens) in sequences.iter_mut() { placement.len += tokens.len(); }
        let slots = self.slot_workspaces(1)?;
        let w = &slots[0];
        let outcomes = prefill_packing::finish_members_selected(sequences.len(), needs_head, |i| {
            self.launch_head(w, segments[i].end_row(), 1)?;
            Ok(self.device_logits(w, 1, false))
        }, on_logits)?;
        Ok((segments, outcomes))
    }

    /// Golden-check consumer: score retained rows without widening serving's head storage.
    pub(crate) fn packed_check_logits(&self, segment: Segment,
        consume: &mut dyn FnMut(usize, DeviceLogits) -> Result<()>) -> Result<()> {
        let slots = self.prefill_slots.borrow();
        let w = slots.first().context("packed-check prefill slot")?;
        for first in (0..segment.rows).step_by(DECODE_ROWS) {
            let rows = (segment.rows - first).min(DECODE_ROWS);
            self.launch_head(w, segment.first + first + rows, rows)?;
            consume(first, self.device_logits(w, rows, false))?;
        }
        Ok(())
    }
}

// Program-local metadata cannot attest the owner's F16/BF16/A8 switch.
fn native_owner_cap(cap: usize) -> usize { cap.min(native_dense::NATIVE_DECODE_ROWS) }

fn exact_program_cap<'n>(programs: &Programs<'_>,
    inventory: impl IntoIterator<Item = (&'n str, prefill_packing::ProgramRows)>, capacity: usize,
    reasons: &mut Vec<String>) -> Result<usize> {
    let inventory = inventory.into_iter().collect::<Vec<_>>();
    for &(name, role) in &inventory {
        if role == prefill_packing::ProgramRows::PackedRows {
            if let Some(reason) = program_fallback_reason(name, &programs.spec(name)?.geometry) {
                reasons.push(reason);
            }
        }
    }
    prefill_packing::coordinator_row_cap(programs, inventory, capacity)
}

fn program_fallback_reason(name: &str, geometry: &serde_json::Value) -> Option<String> {
    match geometry.get("exact_row_cap").and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok()) {
        Some(cap) if cap > 0 => None,
        Some(_) => Some(format!("program {name} publishes exact_row_cap=0")),
        None => Some(format!("program {name} is missing valid exact_row_cap metadata")),
    }
}

/// Classify the startup catalogue explicitly; new prefill exports fail closed
/// until their row ownership is reviewed. Decode-only exports never see packs.
fn coordinator_inventory<'n>(cfg: &GlmDsaConfig, names: impl IntoIterator<Item = &'n str>, split: bool)
    -> Result<Vec<(String, prefill_packing::ProgramRows)>> {
    use prefill_packing::ProgramRows::{PackedRows, SequenceLocal};
    if !cfg.variant.is_glm() {
        return variant_inventory(cfg.program_family(), names);
    }
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
        family: "glm", split_family: split.then_some("glm2") };
    let required = names.into_iter().filter(|name| {
        (selected.contains(name)
            || cuteafd_loader::families::glm5::DensePrograms::Trellis.stage(name, split))
            && !name.ends_with("_m64") && !name.contains("_decode_")
    }).collect::<Vec<_>>();
    let mut inventory = Vec::new();
    for &name in &required {
        let suffix = name.strip_prefix("glm_").or_else(|| name.strip_prefix("glm2_"))
            .context("GLM packed catalogue namespace")?;
        let role = match suffix {
            "norm" | "router_scores" | "expert_input_quant" => PackedRows,
            "dsa_candidate_merge" | "lse_combine2" | "paged_staging_gather_kv"
                | "paged_staging_gather_index" => SequenceLocal,
            _ => {
                let stem = suffix.strip_suffix("_m4096").unwrap_or(suffix);
                if matches!(stem, "producer" | "producer_bf16" | "index_producer"
                    | "index_producer_bf16" | "o" | "o_bf16" | "index_topk_prefill"
                    | "sparse_mla_prefill" | "stage_rank_pack" | "stage_query_rope"
                    | "stage_index_post" | "stage_absorb" | "stage_absorb_bf16"
                    | "stage_values" | "stage_values_bf16" | "stage_query_rope_context"
                    | "stage_absorb_context" | "stage_absorb_bf16_context") {
                    SequenceLocal
                } else if stem.strip_prefix("ffn_i").is_some_and(|shape| {
                    let size = shape.strip_suffix("_bf16").or_else(|| shape.strip_suffix("_pt")).unwrap_or(shape);
                    size.parse::<usize>().is_ok_and(|n| n > 0)
                }) || stem.strip_prefix("stage_swiglu_i").is_some_and(|n| n.parse::<usize>().is_ok_and(|n| n > 0)) {
                    PackedRows
                } else { anyhow::bail!("GLM packed program {name} has no reviewed row role"); }
            }
        };
        inventory.push((name.to_owned(), role));
    }
    let classified = inventory.iter().map(|(name, role)| (name.as_str(), *role)).collect::<Vec<_>>();
    prefill_packing::validate_program_inventory(required, &classified)?;
    Ok(inventory)
}

/// A whole-attention variant's own namespace (`ml4_*`): per-row transforms
/// pack, the latent/index stages and attention stay sequence-local.
fn variant_inventory<'n>(family: &str, names: impl IntoIterator<Item = &'n str>)
    -> Result<Vec<(String, prefill_packing::ProgramRows)>> {
    use prefill_packing::ProgramRows::{PackedRows, SequenceLocal};
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family, split_family: None };
    let required = names.into_iter().filter(|name| selected.contains(name)
        && !name.ends_with("_m64") && !name.contains("_decode_")).collect::<Vec<_>>();
    let mut inventory = Vec::new();
    for &name in &required {
        let suffix = &name[family.len() + 1..];
        let stem = suffix.strip_suffix("_m4096").unwrap_or(suffix);
        let role = match stem {
            "norm" | "router_scores" | "expert_input_quant" | "index_kw" | "index_layer_norm"
                | "index_query_scale" | "query_scale" | "query_scale_nope" => PackedRows,
            "rank_pack" | "rank_pack_nope" | "absorb" | "absorb_nope" | "values" | "index_post"
                | "index_post_nope" | "index_topk_prefill" | "sparse_mla_prefill" | "sparse_mla_nope_prefill"
                => SequenceLocal,
            _ if stem.strip_prefix("swiglu_i").is_some_and(|n| n.parse::<usize>().is_ok_and(|n| n > 0)) => PackedRows,
            _ => anyhow::bail!("{family} packed program {name} has no reviewed row role"),
        };
        inventory.push((name.to_owned(), role));
    }
    let classified = inventory.iter().map(|(name, role)| (name.as_str(), *role)).collect::<Vec<_>>();
    prefill_packing::validate_program_inventory(required, &classified)?;
    Ok(inventory)
}

pub(super) fn sequence_tables(placement: &GlmPlacement, start: usize, rows: usize, topk: usize) -> Result<StepTables> {
    let used = (start + rows).div_ceil(PAGE_ROWS);
    ensure!(used <= placement.pages.len(), "prefill past sequence pages");
    Ok(StepTables { segments: Vec::new(), decode: false,
        positions: (start..start + rows).map(|p| p as i64).collect(),
        slots: (start..start + rows).map(|p| placement.slot(p)).collect::<Result<_>>()?,
        page_table: placement.table(used).collect(), table_width: used, table_stride: 0,
        cache_lengths: (start..start + rows).map(|p| (p + 1) as i32).collect(),
        lengths: (start..start + rows).map(|p| (p + 1).min(topk) as i32).collect(), exchange_rows: rows })
}

fn packed_tables(sequences: &[(&mut GlmPlacement, &[u32])], segments: &[Segment], topk: usize) -> Result<StepTables> {
    ensure!(sequences.len() == segments.len() && !segments.is_empty(), "packed layout members differ");
    let mut tables = sequence_tables(sequences[0].0, sequences[0].0.len, 0, topk)?;
    for ((placement, tokens), &segment) in sequences.iter().zip(segments) {
        let local = sequence_tables(placement, placement.len, tokens.len(), topk)?;
        tables.positions.extend_from_slice(&local.positions);
        tables.slots.extend_from_slice(&local.slots);
        tables.cache_lengths.extend_from_slice(&local.cache_lengths);
        tables.lengths.extend_from_slice(&local.lengths);
        tables.segments.push((segment, local));
    }
    // Segment attention uploads its own table in stream order. No combined
    // page table or per-row context matrix is allocated.
    tables.page_table.clear();
    tables.table_width = 0;
    tables.exchange_rows = tables.positions.len();
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glm_cfg() -> GlmDsaConfig {
        GlmDsaConfig::from_hf(&cuteafd_loader::plan::testing::glm5_config()).expect("GLM 5.3 test config")
    }

    #[test]
    fn mistral_large4_inventory_classifies_its_own_namespace() -> Result<()> {
        use prefill_packing::ProgramRows::{PackedRows, SequenceLocal};
        let cfg = GlmDsaConfig::from_hf(&serde_json::from_str(include_str!(
            "../../../../../cuteafd-loader/src/families/mistral_large4/official-config.json"))?)?;
        let names = ["ml4_norm", "ml4_index_kw", "ml4_query_scale_nope_m4096", "ml4_swiglu_i3072",
            "ml4_absorb_m4096", "ml4_sparse_mla_nope_prefill_m4096", "ml4_index_topk_prefill_m4096",
            "ml4_sparse_mla_decode_m64", "glm_norm"];
        let inventory = coordinator_inventory(&cfg, names, false)?;
        for name in ["ml4_norm", "ml4_index_kw", "ml4_query_scale_nope_m4096", "ml4_swiglu_i3072"] {
            assert!(inventory.contains(&(name.to_owned(), PackedRows)), "{name}");
        }
        for name in ["ml4_absorb_m4096", "ml4_sparse_mla_nope_prefill_m4096", "ml4_index_topk_prefill_m4096"] {
            assert!(inventory.contains(&(name.to_owned(), SequenceLocal)), "{name}");
        }
        assert!(!inventory.iter().any(|(n, _)| n.ends_with("_m64") || n.starts_with("glm_")));
        assert!(coordinator_inventory(&cfg, ["ml4_new_stage_m4096"], false).is_err());
        Ok(())
    }

    #[test]
    fn native_owner_dispatch_cap_preserves_smaller_program_caps() {
        for cap in [0, 8, 16, 64, 129, 4096] {
            assert_eq!(native_owner_cap(cap), cap.min(64));
        }
        let limits = Limits { rows: native_owner_cap(4096), chunk_rows: 4096, max_context: 4096, taps: None };
        let chunks = [(0, 29), (0, 7), (0, 64), (0, 65)];
        assert_eq!(prefill_packing::groups(4, |r| prefill_packing::plan(&chunks[r], &limits).is_ok()),
            [0..2, 2..3, 3..4]);
    }

    #[test]
    fn exact_fallback_names_program_and_metadata_reason() {
        use serde_json::json;
        assert!(program_fallback_reason("glm_norm", &json!({})).unwrap().contains("glm_norm is missing"));
        assert!(program_fallback_reason("glm_norm", &json!({"exact_row_cap": 0})).unwrap().contains("=0"));
        for value in [json!(-1), json!("64"), json!(null)] {
            assert!(program_fallback_reason("route", &json!({"exact_row_cap": value})).unwrap().contains("missing valid"));
        }
        for cap in [8, 16, 64, 129] {
            assert!(program_fallback_reason("route", &json!({"exact_row_cap": cap})).is_none());
        }
    }

    #[test]
    fn packed_tables_equal_singletons_and_ignore_partner_pages() -> Result<()> {
        let mut a = GlmPlacement { pages: vec![3, 4, 5], len: 60 };
        let mut b = GlmPlacement { pages: vec![11, 12], len: 1 };
        let tokens_a = [1; 12];
        let tokens_b = [2; 9];
        let limits = Limits { rows: 256, chunk_rows: 256, max_context: 4096, taps: Some(2048) };
        let segments = prefill_packing::plan(&[(60, 12), (1, 9)], &limits)?;
        let tables = packed_tables(&[(&mut a, &tokens_a), (&mut b, &tokens_b)], &segments, 64)?;
        for ((s, actual), reference) in tables.segments.iter().zip([
            sequence_tables(&a, a.len, tokens_a.len(), 64)?, sequence_tables(&b, b.len, tokens_b.len(), 64)?]) {
            assert_eq!(actual.positions, reference.positions);
            assert_eq!(actual.slots, reference.slots);
            assert_eq!(actual.page_table, reference.page_table);
            assert_eq!(actual.cache_lengths, reference.cache_lengths);
            assert_eq!(actual.lengths, reference.lengths);
            assert_eq!(tables.positions[s.first..s.end_row()], reference.positions);
        }
        b.pages = vec![18, 19];
        let changed = packed_tables(&[(&mut a, &tokens_a), (&mut b, &tokens_b)], &segments, 64)?;
        assert_eq!(tables.segments[0].1.page_table, changed.segments[0].1.page_table);
        assert_eq!(tables.segments[0].1.slots, changed.segments[0].1.slots);
        assert_ne!(tables.segments[1].1.slots, changed.segments[1].1.slots);
        assert_eq!(segments.iter().map(|s| (s.tap_offset, s.tap_rows)).collect::<Vec<_>>(), [(0, 12), (12, 9)]);
        Ok(())
    }

    #[test]
    fn exact_inventory_reviews_startup_programs_and_rejects_new_routes() -> Result<()> {
        use prefill_packing::ProgramRows::{PackedRows, SequenceLocal};
        let names = ["glm_norm", "glm_router_scores", "glm_expert_input_quant",
            "glm_ffn_i2048_m4096", "glm_ffn_i12288_pt_m4096", "glm_producer_m4096",
            "glm_index_topk_prefill_m4096", "glm_stage_swiglu_i2048",
            "glm2_ffn_i1024_bf16_m4096", "glm2_stage_query_rope_context_m4096",
            "glm_sparse_mla_decode_m64"];
        let inventory = coordinator_inventory(&glm_cfg(), names, true)?;
        for name in ["glm_norm", "glm_router_scores", "glm_expert_input_quant",
            "glm_ffn_i2048_m4096", "glm_ffn_i12288_pt_m4096", "glm_stage_swiglu_i2048",
            "glm2_ffn_i1024_bf16_m4096"] {
            assert!(inventory.contains(&(name.to_owned(), PackedRows)));
        }
        assert!(inventory.contains(&("glm_producer_m4096".into(), SequenceLocal)));
        assert!(inventory.contains(&("glm2_stage_query_rope_context_m4096".into(), SequenceLocal)));
        assert!(!inventory.iter().any(|(n, _)| n.ends_with("_m64")));
        assert!(coordinator_inventory(&glm_cfg(), ["glm_new_projection_m4096"], false).is_err());
        assert!(coordinator_inventory(&glm_cfg(), ["glm_stage_new_transform_m4096"], false).is_err());
        let limits = Limits { rows: 64, chunk_rows: 4096, max_context: 4096, taps: None };
        let chunks = [(0, 29), (0, 200), (100, 40), (0, 24)];
        assert_eq!(prefill_packing::groups(4, |r| prefill_packing::plan(&chunks[r], &limits).is_ok()),
            [0..1, 1..2, 2..4]);
        assert!(prefill_packing::plan(&chunks[..2], &Limits { rows: 4096, ..limits }).is_ok());
        Ok(())
    }

    #[test]
    fn packed_pass_retains_one_unit_for_segment_heads_and_taps() {
        let source = include_str!("packing.rs");
        let pass = source.split("pub(crate) fn prefill_packed_selected(").nth(1).unwrap()
            .split("pub(crate) fn packed_check_logits(").next().unwrap();
        assert!(pass.contains("self.step_lanes(&[tables], &[&tokens], self.depth, 0, link, runtime)?"));
        assert!(pass.contains("self.slot_workspaces(1)?"));
        assert!(!pass.contains("retained_logits_units("));
        assert!(!pass.contains(".chunks("));
    }

    #[test]
    fn exact_route_cap_splits_groups_but_never_rechunks_members() {
        let chunks = [(0, 64), (0, 64), (900, 129), (0, 12)];
        let limits = Limits { rows: 128, chunk_rows: 256, max_context: 4096, taps: Some(2048) };
        assert_eq!(prefill_packing::groups(4, |range| prefill_packing::plan(&chunks[range], &limits).is_ok()),
            [0..2, 2..3, 3..4]);
        assert_eq!(prefill_packing::groups(4, |range| prefill_packing::plan(&chunks[range], &Limits { rows: 0, ..limits }).is_ok()),
            [0..1, 1..2, 2..3, 3..4]);
    }

    #[test]
    fn cancelled_head_is_local_and_does_not_skip_other_members() -> Result<()> {
        let mut consumed = Vec::new();
        let outcomes = prefill_packing::finish_members(3, Ok, &mut |member, row| {
            consumed.push(row);
            Ok(if member == 1 { Err(anyhow::anyhow!("client went away")) } else { Ok(()) })
        })?;
        assert_eq!(consumed, [0, 1, 2]);
        let outcomes = prefill_packing::member_outcomes(3, Ok(outcomes));
        assert!(outcomes[0].is_ok() && outcomes[1].is_err() && outcomes[2].is_ok());
        Ok(())
    }
}
