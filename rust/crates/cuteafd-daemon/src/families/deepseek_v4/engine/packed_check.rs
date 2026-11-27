//! CPU contracts for V4 segmented metadata and the shared packed executor.
use super::*;
use crate::shared::prefill_packing::{self, Limits};
use crate::shared::prefill_pipeline::{self, FfnRoute, PrefillHooks, Unit};

#[test]
fn kv_arena_allocates_once_per_device_and_views_retain_the_generation() -> Result<()> {
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    fixture.strict_device_free(&library, true)?;
    let cfg = DeepseekV4Config::from_model_args(&serde_json::json!({
        "vocab_size":129280,"dim":4096,"moe_inter_dim":2048,"n_layers":3,"n_hash_layers":1,
        "n_heads":64,"n_routed_experts":256,"n_shared_experts":1,"n_activated_experts":6,
        "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,"q_lora_rank":1024,
        "head_dim":512,"rope_head_dim":64,"o_groups":8,"o_lora_rank":1024,"window_size":128,
        "original_seq_len":65536,"rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
        "index_n_heads":64,"index_head_dim":128,"index_topk":512,"hc_mult":4,"hc_sinkhorn_iters":20,
        "compress_rope_theta":160000,"compress_ratios":[0,4,128]
    }), 3)?;
    for rank in 0..2 {
        library.cuda_set_device(rank)?;
        let layout = cuteafd_loader::serving_capacity::V4KvLayout::new(&cfg, 1, 1, 2,
            true, if rank == 0 { 3 } else { 0 }, 128, 3, 64)?;
        let names = layout.views().keys().cloned().collect::<Vec<_>>();
        let arena = KvArena::new(&library, layout)?;
        let views = names.iter().map(|name| arena.view(name)).collect::<Result<Vec<_>>>()?;
        // All KV kernels use at most 16-byte vector/TMA bases. Each named view
        // preserves 64-byte alignment, including tiny tables and carry arrays.
        assert!(views.iter().all(|v| v.buffer.ptr as usize % 16 == 0 && v.buffer.ptr as usize % 64 == 0));
        for (i, view) in views.iter().enumerate() {
            let value = [i as u8; 32];
            library.copy_h2d(view.buffer, &value)?;
            let mut result = [0; 32];
            library.copy_d2h_quiescent(&mut result, view.buffer)?;
            assert_eq!(value, result);
        }
        drop(arena);
        assert_eq!(fixture.events()?.matches('d').count(), rank as usize);
        drop(views);
        assert_eq!(fixture.events()?.matches('D').count(), rank as usize + 1);
        assert_eq!(fixture.events()?.matches('d').count(), rank as usize + 1);
    }
    assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
    Ok(())
}

fn limits() -> Limits { Limits { rows: 256, chunk_rows: 256, max_context: 4096, taps: Some(TAP_ROWS) } }

#[test]
fn serving_keeps_one_terminal_head_row_without_diagnostic_callback() {
    for rows in [1, 29, 64, 127, 200, 2752] {
        assert_eq!(super::packing::serving_head_rows(rows), 1);
    }
    let source = include_str!("packing.rs");
    let serving = source.split("pub(crate) fn prefill_packed_check").next().unwrap();
    assert!(serving.contains("self.prefill_packed_inner(sequences, links, prompt_ends, on_logits, None)"));
    assert!(!include_str!("../serve.rs").contains("prefill_packed_check"));
    assert!(source.contains("if !prompt_ends[i] { return Ok(None); }"));
    assert!(source.contains("return Ok(None);"));
    let attention = source.split("pub(super) fn packed_attention").nth(1).unwrap();
    assert!(!attention.contains(".fill("));
    assert!(!attention.contains("put_context_tables"));
    assert!(include_str!("../mod.rs").contains("args.prefill_packing.unwrap_or_default()"));
}

#[test]
fn exact_inventory_caps_total_rows_and_preserves_large_singletons() -> Result<()> {
    use prefill_packing::ProgramRows::{PackedRows, SequenceLocal};
    for split in [None, Some("dsv4f2")] {
        for context in [false, true] {
            let inventory = super::packing::coordinator_inventory("dsv4f", split, 256, context)?;
            for suffix in ["router_scores", "expert_input_quant", "mhc_post_pre_m256", "shared_ffn_m256", "mhc_post"] {
                assert!(inventory.contains(&(format!("dsv4f_{suffix}"), PackedRows)));
            }
            assert!(inventory.contains(&("dsv4f_index_producer_m256".into(), SequenceLocal)));
            assert_eq!(inventory.iter().filter(|(_, role)| *role == PackedRows).count(), 5 + usize::from(split.is_some()));
        }
    }
    // Simulate a loaded router's attested first route ending at 64 rows. The
    // 200-row partner must stay whole and cannot pull a short member past it.
    let chunks = [(0, 29), (0, 200), (301, 40), (127, 24)];
    let limits = Limits { rows: 64, taps: None, ..limits() };
    let groups = prefill_packing::groups(chunks.len(), |r| prefill_packing::plan(&chunks[r], &limits).is_ok());
    assert_eq!(groups, vec![0..1, 1..2, 2..4]);
    assert_eq!(chunks[1].1, 200);
    assert!(prefill_packing::plan(&chunks[..2], &Limits { rows: 256, ..limits }).is_ok());
    Ok(())
}

#[test]
fn packed_members_keep_singleton_tables_and_disjoint_state() -> Result<()> {
    let shape = PoolShape::new(3, 256, 48);
    let chunks = [(0, 29), (301, 40), (127, 59)];
    let segments = prefill_packing::plan(&chunks, &limits())?;
    let placements: Vec<_> = chunks.iter().enumerate().map(|(i, &(start, _))| {
        let mut p = Placement::new(i, (i as u32 * 16..(i as u32 + 1) * 16).collect());
        p.len = start; p
    }).collect();
    for ((p, &(start, rows)), s) in placements.iter().zip(&chunks).zip(&segments) {
        let packed = metadata::prefill_step(p, &shape, s.start, s.rows, 16, 32)?;
        let alone = metadata::prefill_step(p, &shape, start, rows, 16, 32)?;
        assert_eq!(packed.positions, alone.positions);
        assert_eq!(packed.main_slots, alone.main_slots);
        assert_eq!(packed.swa_indices, alone.swa_indices);
        assert_eq!(packed.c4_tables, alone.c4_tables);
        assert_eq!(packed.c128_tables, alone.c128_tables);
        assert_eq!(packed.c4_page_table, alone.c4_page_table);
        assert_eq!(packed.c128_indices, alone.c128_indices);
        for slot in packed.swa_indices.into_iter().filter(|&slot| slot >= 0) {
            assert_eq!(slot as usize / (shape.ring_pages * 256), p.state);
        }
        for slot in packed.c128_indices.into_iter().filter(|&slot| slot >= 0) {
            assert!(p.units.contains(&(slot as u32 / 2)));
        }
    }
    assert_eq!(segments.iter().map(|s| (s.tap_offset, s.tap_rows)).collect::<Vec<_>>(), [(0, 29), (29, 40), (69, 59)]);
    assert!(prefill_packing::plan(&[(0, 29), (0, 100)], &limits()).is_err());
    Ok(())
}

#[test]
fn independent_draft_tails_allow_sixteen_short_prompts() -> Result<()> {
    let limits = Limits { rows: 4096, chunk_rows: 4096, max_context: 8192,
        taps: Some(cuteafd_loader::serving_capacity::deepseek_v4_tap_rows(4096, true)) };
    let segments = prefill_packing::plan_with_tap_rows(&[(300, 200); 16], &limits, TAP_ROWS)?;
    assert_eq!(segments.last().map(|s| (s.tap_offset, s.tap_rows)), Some((15 * TAP_ROWS, TAP_ROWS)));
    assert_eq!(segments.iter().map(|s| s.rows).sum::<usize>(), 3200);
    assert!(prefill_packing::plan_with_tap_rows(&[(300, 200); 17], &limits, TAP_ROWS).is_err());
    Ok(())
}

struct Model {
    segments: Vec<prefill_packing::Segment>,
    values: Vec<i64>,
    states: Vec<i64>,
    expert_calls: usize,
    spark: bool,
}
impl PrefillHooks for Model {
    type Ticket = ();
    fn route(&self, _: usize) -> FfnRoute { if self.spark { FfnRoute::Spark } else { FfnRoute::Local } }
    fn begin_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
    fn attention(&mut self, layer: usize, _: &Unit) -> Result<()> {
        for (member, s) in self.segments.iter().enumerate() {
            for row in s.first..s.end_row() {
                self.states[member] += self.values[row];
                self.values[row] += self.states[member] + (s.start + row - s.first + layer) as i64;
            }
        }
        Ok(())
    }
    fn local_ffn(&mut self, _: usize, _: &Unit) -> Result<()> {
        self.expert_calls += 1;
        for value in &mut self.values { *value = *value * 3 + 1; }
        Ok(())
    }
    fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
    fn land_experts(&mut self, layer: usize, unit: &Unit, _: ()) -> Result<()> { self.local_ffn(layer, unit) }
    fn end_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
    fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
}

#[test]
fn packed_executor_matches_singletons_and_partner_cannot_change_history() -> Result<()> {
    let chunks = [(0, 29), (301, 40), (127, 59)];
    let segments = prefill_packing::plan(&chunks, &limits())?;
    for spark in [false, true] {
        for changed_partner in [false, true] {
            let values: Vec<_> = (0..128).map(|i| if changed_partner && (29..69).contains(&i) { 1000 } else { i as i64 }).collect();
            let mut packed = Model { segments: segments.clone(), values: values.clone(), states: vec![3, 5, 7], expert_calls: 0, spark };
            let units = prefill_pipeline::sequence_units(128, 256, 1, 5)?;
            prefill_pipeline::run_prefill_units(&mut packed, &units, 1)?;
            assert_eq!(packed.expert_calls, 5);
            for (i, s) in segments.iter().enumerate() {
                let singleton = prefill_packing::plan(&[(s.start, s.rows)], &limits())?;
                let mut alone = Model { segments: singleton, values: values[s.first..s.end_row()].to_vec(),
                    states: vec![[3, 5, 7][i]], expert_calls: 0, spark };
                let units = prefill_pipeline::sequence_units(s.rows, 256, 1, 5)?;
                prefill_pipeline::run_prefill_units(&mut alone, &units, 1)?;
                assert_eq!(packed.values[s.first..s.end_row()], alone.values);
                assert_eq!(packed.states[i], alone.states[0]);
            }
        }
    }
    Ok(())
}

#[test]
fn cancellation_stays_member_local_and_exact_unknown_routes_run_alone() -> Result<()> {
    let mut seen = Vec::new();
    let outcomes = prefill_packing::finish_members(3, Ok, &mut |i, row| {
        seen.push(row);
        Ok(if i == 1 { Err(anyhow::anyhow!("cancelled")) } else { Ok(()) })
    })?;
    assert_eq!(seen, [0, 1, 2]);
    assert!(outcomes[0].is_ok() && outcomes[1].is_err() && outcomes[2].is_ok());
    let chunks = [(0, 29), (0, 40), (0, 59)];
    for (cap, expected) in [(0, vec![0..1, 1..2, 2..3]), (69, vec![0..2, 2..3]), (128, vec![0..3])] {
        assert_eq!(prefill_packing::groups(chunks.len(), |r|
            prefill_packing::plan(&chunks[r], &Limits { rows: cap, ..limits() }).is_ok()), expected);
    }
    Ok(())
}


#[test]
fn aligned_private_metadata_fits_admitted_arenas_for_sixteen_members() -> Result<()> {
    use super::packing::table_fields;
    use cuteafd_loader::serving_capacity::V4_PACKED_TABLE_EXTRA_ROWS;
    for width in [1, 29, 31, 64, 127, 200, 256] {
        let rows = 16 * width;
        let storage = rows + V4_PACKED_TABLE_EXTRA_ROWS;
        let units = 128;
        let shape = PoolShape::new(16, rows, units);
        let capacities = [storage * 8, storage * 8, storage * 128 * 4, storage * 4].into_iter()
            .chain([4 * (storage + 2); 18])
            .chain([16 * (units + 3) * 4, storage * 4, storage * 4, storage * 1024 * 4, storage * 4])
            .collect::<Vec<_>>();
        let mut planes = vec![Vec::new(); capacities.len()];
        for member in 0..16 {
            let start = if member % 2 == 0 { 0 } else { 301 };
            let p = Placement::new(member, (0..units as u32).collect());
            let tables = metadata::prefill_step(&p, &shape, start, width, 512, 1024)?;
            let fields = table_fields(&tables);
            assert_eq!(fields.len(), 27);
            for (field, (bytes, capacity)) in fields.iter().zip(&capacities).enumerate() {
                let plane = &mut planes[field];
                let offset = plane.len().next_multiple_of(16);
                plane.resize(offset, 0u8);
                plane.extend_from_slice(bytes);
                if bytes.is_empty() { plane.extend_from_slice(&[0; 4]); }
                assert_eq!(&plane[offset..offset + bytes.len()], *bytes);
                assert!(plane.len() <= *capacity, "width {width} member {member} field {field}");
            }
        }
    }
    Ok(())
}
