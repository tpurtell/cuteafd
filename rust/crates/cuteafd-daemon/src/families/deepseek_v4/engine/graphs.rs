//! Owner-mode target segments: exact bindings captured only before readiness.
use super::*;
use crate::shared::decode_graph::{GraphBank, GraphDecision, GraphOwner};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Segment { Front, Back, Router, Shared, Partial, Post, Tap, DraftKv, Head }

impl Segment {
    fn name(self) -> &'static str {
        match self {
            Self::Front => "front", Self::Back => "back", Self::Router => "router",
            Self::Shared => "shared", Self::Partial => "partial", Self::Post => "post",
            Self::Tap => "tap", Self::DraftKv => "draft-kv", Self::Head => "head",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct OwnerKey {
    rank: usize,
    device: i32,
    stream: usize,
    layer: usize,
    lane: usize,
    rows: usize,
    cap: usize,
    chunked: bool,
    attention: &'static str,
    width: usize,
    stride: usize,
    segment: Segment,
    output: usize,
    bindings: Vec<(usize, usize)>,
}

pub(super) type OwnerGraphs<'a> = GraphBank<OwnerKey, GraphOwner<'a, Vec<(usize, usize)>>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::decode_graph::GraphPolicy;

    #[test]
    fn startup_rows_cover_measured_c8_c16_lane_widths_without_padding() -> Result<()> {
        let measured = [17, 18, 19, 20, 21, 22, 23, 26, 28, 31, 33];
        assert!(STARTUP_ROWS.windows(2).all(|rows| rows[0] < rows[1]));
        assert_eq!(STARTUP_ROWS.len(), 33);
        let mut bank = GraphBank::with_policy(GraphPolicy::FixedStartup);
        bank.warm(&STARTUP_ROWS, |&rows| Ok((rows, None)))?;
        for rows in measured {
            let (decision, stored) = bank.enqueue(&rows);
            assert_eq!(decision, GraphDecision::Replay);
            assert_eq!(stored.copied(), Some(rows));
        }
        for rows in [25, 27, 29, 30, 34, 63] {
            assert_eq!(bank.enqueue(&rows).0, GraphDecision::Eager);
        }
        assert_eq!(bank.stats().captures, STARTUP_ROWS.len() as u64);
        Ok(())
    }

    #[test]
    fn fixed_owner_bank_never_replays_changed_pointers_extents_or_shapes() -> Result<()> {
        let key = OwnerKey { rank: 0, device: 0, stream: 11, layer: 42, lane: 1,
            rows: 8, cap: 64, chunked: true, attention: "c4", width: 4096, stride: 4096,
            segment: Segment::Front, output: 17, bindings: vec![(23, 4096)] };
        let mut bank = GraphBank::with_policy(GraphPolicy::FixedStartup);
        bank.warm(&[key.clone()], |_| Ok(((), None)))?;
        assert_eq!(bank.enqueue(&key).0, GraphDecision::Replay);
        let mut changes = Vec::new();
        macro_rules! changed { ($field:ident, $value:expr) => {{
            let mut changed = key.clone(); changed.$field = $value; changes.push(changed);
        }}; }
        changed!(rank, 1); changed!(device, 1); changed!(stream, 12);
        changed!(layer, 43); changed!(lane, 0); changed!(rows, 7); changed!(cap, 80);
        changed!(chunked, false); changed!(attention, "win"); changed!(width, 4095);
        changed!(stride, 4095); changed!(segment, Segment::Back); changed!(output, 18);
        changed!(bindings, vec![(24, 4096)]); changed!(bindings, vec![(23, 4097)]);
        for changed in changes { assert_eq!(bank.enqueue(&changed).0, GraphDecision::Eager); }
        assert_eq!(bank.stats().captures, 1);
        Ok(())
    }
}

// Exact rows preserve arithmetic routes. Add only lane widths observed in
// C8/C16 census; unlisted widths and mixed C128 ranges remain eager.
const STARTUP_ROWS: [usize; 33] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 31, 32, 33, 40, 48, 56, 64];

impl Engine<'_> {
    fn owner_key(&self, rank: usize, layer: usize, index: usize, tables: &StepTables,
        w: &Workspace<'_>, segment: Segment, output: *mut c_void) -> Result<OwnerKey> {
        let weights = if rank == 0 { &self.weights.layers[layer] } else { &self.peer()?.layers[layer] };
        let lane = &w.lanes[index];
        let m = &lane.tables;
        let mut bindings = weights.graph_bindings();
        let mut bind = |allocation: &Dev<'_>| bindings.push((allocation.buffer.ptr as usize, allocation.buffer.bytes));
        // Enumerate allocation addresses/extents, never a movable Workspace or
        // LayerWeights struct address. The containing engine retains them all.
        for allocation in [&lane.stream_a, &lane.stream_b, &lane.post, &lane.comb, &lane.y,
            &lane.route_ids, &lane.route_weights, &lane.wire, &lane.tokens, &lane.shared, &lane.taps,
            &w.query, &w.q_rank, &w.attn_out, &w.delta, &w.index_query, &w.index_weights,
            &w.selected, &w.topk_scratch, &w.logits, &w.scratch, &w.dummy, &w.sum,
            &m.positions, &m.main_slots, &m.swa_indices, &m.swa_lengths, &m.c4_page_table,
            &m.c4_visible, &m.c4_indexed_lengths, &m.c128_indices, &m.c128_lengths] { bind(allocation); }
        for allocation in lane.payload.iter().chain(m.c4.iter()).chain(m.c128.iter()) { bind(allocation); }
        if let Ok(cache) = self.cache_on(rank, layer) {
            bind(&cache.main);
            for allocation in cache.compressed.iter().chain(cache.index.iter()).chain(cache.states.iter()) { bind(allocation); }
        }
        bind(self.attention_rope(rank, weights.ratio));
        if matches!(segment, Segment::DraftKv | Segment::Head) {
            for allocation in [&w.vocab_logits, &w._head_workspace, &w.main_x, &w.main_work,
                &self.weights.head, &self.weights.head_fn, &self.weights.head_scale, &self.weights.head_base, &self.weights.norm] { bind(allocation); }
            if let Some(staging) = &w.staged_head { bind(staging); }
            bind(&self.rope_window);
            if let Some(draft) = &self.weights.dspark {
                for allocation in [&draft.main_proj, &draft.main_proj_scale, &draft.main_norm] { bind(allocation); }
                for stage in 0..draft.stages.len() {
                    let cache = self.cache_on(0, self.cfg.n_layers + stage)?;
                    bind(&cache.main);
                }
            }
        }
        if segment == Segment::DraftKv {
            if let Some(draft) = &self.weights.dspark {
                for weight in &draft.stages { bindings.extend(weight.graph_bindings()); }
            }
        }
        if segment == Segment::Partial { bindings.extend(self.tp2_graph_bindings(rank, layer, tables.rows)?); }
        Ok(OwnerKey { rank, device: if rank == 0 { self.device } else { self.peer()?.device },
            stream: self.stream_of(rank) as usize, layer, lane: index, rows: tables.rows,
            cap: self.decode_rows, chunked: segment == Segment::Front && tables.chunked,
            attention: if segment == Segment::Front { self.attention_kind(weights.ratio, tables) } else { "none" },
            width: if segment == Segment::Front { tables.c4_table_width } else { 0 },
            stride: if segment == Segment::Front { tables.c4_table_stride } else { 0 }, segment, output: output as usize, bindings })
    }

    pub(super) fn owner_segment(&self, rank: usize, layer: usize, index: usize, tables: &StepTables,
        w: &Workspace<'_>, cap: usize, segment: Segment, output: *mut c_void,
        run: impl FnOnce() -> Result<()>) -> Result<()> {
        self.require_live()?;
        if (segment == Segment::Tap && (!self.draft_ready() || !self.cfg.dspark_target_layer_ids.contains(&layer)))
            || (segment == Segment::DraftKv && !self.draft_ready()) { return run(); }
        let bypass = if !tables.decode || cap != self.decode_rows { Some("prefill") }
            else if segment == Segment::Front && self.mixed_attention(self.cfg.compress_ratios[layer], tables) { Some("mixed-c128") }
            else if self.tp2.as_ref().is_some_and(|tp2| tp2.checking.get()) { Some("route-check") }
            else { None };
        if let Some(reason) = bypass {
            self.coverage_launch(segment.name(), rank, index, tables.rows,
                crate::shared::decode_graph::census::Launch::Eager, reason);
            return run();
        }
        let binding = self.dev_layers.as_ref().context("owner graph binding")?;
        let key = self.owner_key(rank, layer, index, tables, w, segment, output)?;
        let raw = {
            let mut graphs = binding.graphs.borrow_mut();
            let (decision, owner) = graphs.enqueue(&key);
            match decision {
                GraphDecision::Replay => Some(owner.context("owner graph executable")?.raw),
                GraphDecision::Eager => None,
                GraphDecision::Capture => anyhow::bail!("owner bank permits live capture"),
            }
        };
        self.coverage_launch(segment.name(), rank, index, tables.rows,
            if raw.is_some() { crate::shared::decode_graph::census::Launch::Replay }
            else { crate::shared::decode_graph::census::Launch::Eager },
            if raw.is_some() { "hit" } else { "unknown-key" });
        match raw {
            Some(raw) => self.on(rank, || {
                // SAFETY: exact key matches every retained allocation and shape;
                // shutdown drains both streams before retiring the bank.
                unsafe { self.library.cuda_graph_launch(raw, self.stream_of(rank)) }
            }),
            None => run(),
        }
    }

    fn capture_owner(&self, key: &OwnerKey, run: impl FnOnce() -> Result<()>) -> Result<()> {
        let binding = self.dev_layers.as_ref().context("owner graph binding")?;
        ensure!(!binding.graphs.borrow().contains(key), "duplicate owner startup key");
        let stream = self.stream_of(key.rank);
        // SAFETY: startup records device-only segments with persistent operands,
        // without peer waits, host copies or any execution of dummy metadata.
        self.on(key.rank, || unsafe { self.library.cuda_graph_begin_capture(stream) })?;
        let result = run();
        // Always end capture, including a launch failure; any resulting executable
        // is owned before propagating the segment's error.
        let raw = self.on(key.rank, || unsafe { self.library.cuda_graph_end_capture(stream) })?;
        // SAFETY: the containing engine pins all key allocations until terminal
        // drain. The owner destroys on the capture's device, including errors.
        let owner = unsafe { GraphOwner::new(self.library, key.device, raw, key.bindings.clone())? };
        result?;
        binding.graphs.borrow_mut().insert(key.clone(), owner, None);
        if key.segment == Segment::Partial {
            self.expert_graph_captures.set(self.expert_graph_captures.get() + 1);
        }
        self.graph_captures.set(self.graph_captures.get() + 1);
        Ok(())
    }

    pub(super) fn warm_owner_graphs(&self) -> Result<()> {
        let binding = self.dev_layers.as_ref().context("owner startup")?;
        let w0 = self.decode_workspace.borrow();
        let w1 = self.peer()?.decode_workspace.borrow();
        let workspaces = [w0.as_ref().context("owner rank0 workspace")?, w1.as_ref().context("owner rank1 workspace")?];
        let mut keys = Vec::new();
        // Enumerate the expected inventory before any capture. The second pass
        // must capture that exact ordered set, not derive it from successes.
        for capturing in [false, true] {
            if capturing {
                let expected = keys.iter().collect::<std::collections::HashSet<_>>();
                ensure!(expected.len() == keys.len(), "duplicate expected owner startup key");
            }
            let mut captured = 0;
            let mut record = |key: OwnerKey, run: &dyn Fn() -> Result<()>| -> Result<()> {
                if capturing {
                    ensure!(keys.get(captured) == Some(&key), "owner capture differs from startup inventory");
                    self.capture_owner(&key, run)?;
                    captured += 1;
                } else { keys.push(key); }
                Ok(())
            };
            for rows in STARTUP_ROWS.into_iter().filter(|&rows| rows <= self.decode_rows) {
                for chunked in [false, true] {
                    let placement = Placement::new(0, vec![0]);
                    let tables = if chunked {
                        metadata::verify_step(&[(&placement, 0, rows)], &self.shape, self.cfg.index_topk, self.c128_width)?
                    } else {
                        metadata::decode_step(&vec![(&placement, 0); rows], &self.shape, self.cfg.index_topk, self.c128_width)?
                    };
                    let mut tables = vec![tables];
                    self.fixed_owner_tables(&mut tables)?;
                    let tables = &tables[0];
                    for (rank, w) in workspaces.iter().enumerate() {
                        for index in 0..self.decode_lanes() {
                            let lane = &w.lanes[index];
                            for layer in 0..self.cfg.n_layers {
                                let weights = if rank == 0 { &self.weights.layers[layer] } else { &self.peer()?.layers[layer] };
                                let resident = weights.ptr("w_q").is_ok();
                                if resident {
                                    for compressed in [false, true] {
                                        if compressed && weights.ratio == 0 { continue; }
                                        let mut variant = tables.clone();
                                        variant.c4_groups = usize::from(compressed);
                                        variant.c128_groups = usize::from(compressed);
                                        let key = self.owner_key(rank, layer, index, &variant, w, Segment::Front, std::ptr::null_mut())?;
                                        record(key, &|| self.attention_front(rank, layer, weights, &variant, lane, w,
                                            Scalar::I32(rows as i32), self.decode_rows))?;
                                    }
                                    if chunked { continue; }
                                    let output = if weights.attention_split { w.sum.buffer.ptr } else { w.delta.buffer.ptr };
                                    // Back/router/shared keys do not depend on which
                                    // attention route produced their retained input.
                                    let key = self.owner_key(rank, layer, index, tables, w, Segment::Back, output)?;
                                    record(key, &|| self.attention_back(rank, weights, lane, w,
                                        Scalar::I32(rows as i32), self.decode_rows, output))?;
                                }
                                if chunked { continue; }
                                if !self.skip_routed && (rank == 0 || self.tp2_layer(layer)) {
                                    let key = self.owner_key(rank, layer, index, tables, w, Segment::Router, std::ptr::null_mut())?;
                                    record(key, &|| self.route_input(rank, weights, lane, w, Scalar::I32(rows as i32)))?;
                                }
                                let key = self.owner_key(rank, layer, index, tables, w, Segment::Shared, std::ptr::null_mut())?;
                                record(key, &|| self.owner_shared(rank, weights, lane, w, rows, self.decode_rows))?;
                                if self.tp2_layer(layer) && !self.skip_routed {
                                    let key = self.owner_key(rank, layer, index, tables, w, Segment::Partial, std::ptr::null_mut())?;
                                    record(key, &|| self.tp2_compute(rank, layer, index, w, rows,
                                        crate::shared::experts::rtx::Routes { ids: lane.route_ids.buffer.ptr, weights: lane.route_weights.buffer.ptr }))?;
                                    if rank == 0 || layer + 1 < self.cfg.n_layers {
                                        let received = self.exchange()?.recv(rank, slot(layer, true, index))?;
                                        let key = self.owner_key(rank, layer, index, tables, w, Segment::Post, received)?;
                                        record(key, &|| self.tp2_combine_post(rank, index, w, rows, received))?;
                                    }
                                }
                                if rank == 0 && self.draft_ready() && self.cfg.dspark_target_layer_ids.contains(&layer) {
                                    let key = self.owner_key(rank, layer, index, tables, w, Segment::Tap, std::ptr::null_mut())?;
                                    record(key, &|| self.tap(w, lane, layer, rows))?;
                                }
                            }
                            if rank == 0 && !chunked {
                                let layer = self.cfg.n_layers - 1;
                                if self.draft_ready() {
                                    let key = self.owner_key(rank, layer, index, tables, w, Segment::DraftKv, std::ptr::null_mut())?;
                                    record(key, &|| self.write_draft_kv(w, lane, rows, self.decode_rows))?;
                                }
                                // Prime the cuBLAS handle/workspace outside capture;
                                // these initialized residuals never enter model state.
                                if capturing {
                                    self.library.cuda_zero_bytes(lane.stream_a.buffer, lane.stream_a.buffer.bytes)?;
                                    self.staged_head_launch(lane, rows, w)?;
                                    self.sync()?;
                                }
                                // Projection always uses the private origin. The live
                                // packed offset and real-row copy stay outside capture.
                                let key = self.owner_key(rank, layer, index, tables, w, Segment::Head, std::ptr::null_mut())?;
                                record(key, &|| self.staged_head_launch(lane, rows, w))?;
                            }
                        }
                    }
                }
            }
            drop(record);
            if capturing { ensure!(captured == keys.len(), "missing owner startup captures"); }
        }
        binding.graphs.borrow_mut().seal_startup(&keys)?;
        tracing::info!(graphs = keys.len(), rows = ?STARTUP_ROWS, "V4 owner FixedStartup inventory sealed");
        Ok(())
    }

    pub(super) fn owner_shared(&self, rank: usize, weights: &LayerWeights<'_>, lane: &Lane<'_>, w: &Workspace<'_>, rows: usize,
        cap: usize) -> Result<()> {
        self.run_on(rank, weights.ffn_split, &format!("shared_ffn_m{cap}"), &[
            ("x", lane.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
            ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?),
            ("out", lane.shared.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[Scalar::I32(rows as i32)])
    }
}
