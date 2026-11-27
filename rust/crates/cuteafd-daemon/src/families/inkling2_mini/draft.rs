//! Inkling 2 Mini TP1 binding of shared DFlash2/recurrent dSpark and FIR verification.
use super::{draft_taps::TapBuffer, verify::Verify};
use crate::shared::{
    memory::device::Stream,
    speculation::{
        recurrent::Drafter,
        dflash::{DflashDrafter, Draft, DraftSeq, TargetHead, TargetProjection},
        block_dspark::WindowSnapshots,
        recurrent_cache::{ContextBank, ContextState},
        recurrent_forward::{ProgramsForRank, Rank},
        recurrent_snapshots::Marks,
        recurrent_weights::Weights,
    },
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::{families::inkling2::draft::DraftLayout, placement::planner::ResolvedPlan};
use std::{cell::RefCell, rc::Rc};

pub(crate) struct Binding<'a> {
    pub drafter: Option<Rc<Drafter<'a>>>,
    dflash: Option<Gqa<'a>>,
    pub marks: Vec<Marks<'a>>,
    pub verify: Verify<'a>,
    pub prefill: Vec<TapBuffer<'a>>,
    pub decode: TapBuffer<'a>,
    pub sites: Vec<cuteafd_core::TapSite>,
    pub sequences: usize,
    pub block: usize,
    hidden: usize,
    context_rows: usize,
}

struct Gqa<'a> {
    body: crate::shared::memory::device::DeviceOwner<'a, DflashDrafter<'a>>,
    marks: Option<WindowSnapshots<'a>>,
    frontier: RefCell<Vec<usize>>,
    embed_eps: f32,
    stream: Rc<Stream<'a>>,
}

impl crate::shared::prefix::cuda_copy::SnapshotOwner for Binding<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<Buffer>> {
        let mut output = self.marks.first().map(|m| m.backing().buffer).into_iter().collect::<Vec<_>>();
        if let Some(gqa) = &self.dflash {
            return Ok(gqa.marks.iter().map(|m| m.backing().buffer).collect());
        }
        let drafter = self.drafter.as_ref().context("Mini recurrent body absent")?;
        for (layer, policy) in drafter.lead.config.context.iter().enumerate() {
            if matches!(policy, cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full) {
                for plane in ["k", "v"] {
                    output.push(
                        drafter.lead
                            .buffers
                            .get(&format!("cache.layer.{layer}.{plane}"))
                            .context("Mini snapshot full drafter KV owner missing")?
                            .buffer,
                    );
                }
            }
        }
        Ok(output)
    }
}

impl<'a> Binding<'a> {
    pub fn ranks(&self) -> impl Iterator<Item = &Rank<'a>> {
        self.drafter.iter().map(|d| &d.lead)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        programs: &'a Programs<'a>,
        stream: Rc<Stream<'a>>,
        weights: Weights<'a>,
        layout: DraftLayout,
        admitted: &ResolvedPlan,
        unembed_weight: Buffer,
        prefill_rows: usize,
        waves: usize,
        decode_rows: usize,
        mark_slots: usize,
    ) -> Result<Self> {
        ensure!(
            layout.weights.len() == 1 && layout.runtime.len() == 1,
            "Inkling 2 Mini drafter must be single-rank TP1"
        );
        let check = |id: &str, bytes: usize| -> Result<()> {
            let owner = admitted
                .facts
                .allocations
                .iter()
                .find(|owner| owner.id == id)
                .with_context(|| format!("unadmitted Mini drafter owner {id}"))?;
            ensure!(
                owner.bytes == bytes as u64
                    && owner.location == cuteafd_loader::placement::planner::Location::coordinator(0),
                "Mini drafter runtime owner differs from admission {id}"
            );
            Ok(())
        };
        let runtime = &layout.runtime[0];
        check(layout.weights[0].arena.id(), layout.weights[0].arena.bytes())?;
        for (name, bytes) in &runtime.buffers {
            check(&format!("inkling2_mini.draft.0.{name}"), *bytes)?;
        }
        if mark_slots > 0 {
            check("inkling2_mini.draft.0.marks", runtime.window_mark_bytes * mark_slots)?;
        }
        for (name, bytes) in &layout.target {
            check(&format!("inkling2_mini.draft.target.{name}"), *bytes)?;
        }

        let lead = stream.device.run(|| {
            let kernels = ProgramsForRank::load(programs, &weights, &layout.config, runtime, stream.device)?;
            let rank = Rank::allocate(weights, Rc::clone(&stream), layout.config.clone(), runtime.clone(), kernels)?;
            rank.initialize_derived()?;
            rank.fold_head(unembed_weight)?;
            stream.drain()?;
            Ok(rank)
        })?;
        let context = ContextBank {
            state: RefCell::new(ContextState::new(
                runtime.slots,
                usize::try_from(admitted.served_context())?,
                runtime.cache.target_tokens,
                0x494e4b4d494e49,
            )?),
        };
        let drafter = Rc::new(Drafter::whole(lead, context)?);
        let marks = if mark_slots == 0 {
            Vec::new()
        } else {
            vec![drafter.lead.stream.device.run(|| Marks::new(&drafter.lead, mark_slots))?]
        };
        let verify = stream
            .device
            .run(|| Verify::new(programs, Rc::clone(&stream), runtime.slots, 6656, decode_rows))?;
        let prefill = (0..waves)
            .map(|_| TapBuffer::new(Rc::clone(&stream), layout.config.transformer.hidden, &layout.config.taps, prefill_rows))
            .collect::<Result<Vec<_>>>()?;
        let decode = TapBuffer::new(Rc::clone(&stream), layout.config.transformer.hidden, &layout.config.taps, decode_rows)?;
        stream.drain()?;
        Ok(Self {
            drafter: Some(drafter),
            dflash: None,
            marks,
            verify,
            prefill,
            decode,
            sites: layout.config.taps.clone(), sequences: runtime.sequences,
            block: layout.config.transformer.block, hidden: layout.config.transformer.hidden,
            context_rows: runtime.context_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_dflash(programs: &'a Programs<'a>, stream: Rc<Stream<'a>>,
        layout: &cuteafd_loader::families::inkling2_mini::dflash::Layout, admitted: &ResolvedPlan,
        embed: Buffer, valid_vocab: usize, embed_eps: f32, prefill_rows: usize, waves: usize, decode_rows: usize, mark_slots: usize) -> Result<Self> {
        use crate::shared::{memory::weight_arena::DeviceWeightArena, fp8_linear::{Fp8Rows, Fp8Scales}};
        use cuteafd_core::memory_layout::ArenaTag;
        use cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation;
        let d = stream.device;
        let cfg = &layout.config;
        let slots = layout.capacity.context_slots;
        let sequences = layout.capacity.max_batch_sequences;
        let check = |id: &str, bytes: usize| -> Result<()> {
            let owner = admitted.facts.allocations.iter().find(|owner| owner.id == id)
                .with_context(|| format!("unadmitted Mini DFlash owner {id}"))?;
            ensure!(owner.bytes == bytes as u64, "Mini DFlash owner extent {id}");
            Ok(())
        };
        check(layout.arena.id(), layout.arena.bytes())?;
        let weights = layout.physical.weight_owners().map(|o| &o.id).collect::<std::collections::BTreeSet<_>>();
        for owner in layout.physical.owners().iter().filter(|o| !weights.contains(&o.id)) {
            check(&format!("inkling2_mini.dflash.{}", owner.id), usize::try_from(owner.bytes)?)?;
        }
        for (name, bytes) in &layout.target { check(&format!("inkling2_mini.draft.target.{name}"), *bytes)?; }
        let mask = d.run(|| {
            let extent = cfg.hidden * 2;
            let mut bytes = vec![0u8; extent];
            d.library.copy_d2h(&mut bytes, Buffer { ptr: embed.ptr.wrapping_byte_add(cfg.mask_token as usize * extent), bytes: extent, ..embed })?;
            Ok(bytes)
        })?;
        let file = crate::shared::speculation::dflash::prefetch(&layout.snapshot).join()
            .map_err(|_| anyhow::anyhow!("Mini DFlash prefetch panicked"))??;
        let arena = DeviceWeightArena::new(d, layout.arena.clone(), ArenaTag::drafter_weights("inkling2_mini/dflash"))?;
        let body = d.own(|| {
            let mut body = DflashDrafter::load_with(d.library, programs, TargetProjection::Bf16, &layout.snapshot,
                file, stream.raw, slots, sequences, mask, true, GlmDraftRepresentation::Bf16Only,
                Fp8Scales::Amax, Fp8Rows::W8a16, layout.kv_cache,
                |id, name, bytes| { ensure!(id == d.id, "Mini DFlash weight device"); arena.view(name, bytes) })?;
            body.set_valid_vocab(valid_vocab)?;
            body.prepare_workspace()?;
            Ok(body)
        })?;
        let physical_devices = [d.id];
        let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical_devices)?;
        let receipts = d.run(|| body.allocation_receipts(devices, "inkling2_mini.dflash"))?;
        let roots = receipts.iter().filter(|r| r.allocation.view.is_none()).collect::<Vec<_>>();
        ensure!(roots.iter().filter(|r| r.allocation.id == layout.arena.id()).count() == 1,
            "Mini DFlash requires one packed arena backing");
        for receipt in roots {
            let actual = &receipt.allocation;
            let planned = admitted.facts.allocations.iter().find(|o| o.id == actual.id)
                .with_context(|| format!("Mini DFlash unadmitted actual owner {}", actual.id))?;
            ensure!(planned.bytes == actual.bytes && planned.location == actual.location,
                "Mini DFlash actual owner differs {}", actual.id);
        }
        let marks = if mark_slots == 0 { None } else {
            check("inkling2_mini.dflash.marks", layout.mark_bytes * mark_slots)?;
            let planes = body.gqa_window_planes(0)?.iter().map(|p| p.bytes).collect();
            let marks = WindowSnapshots::new(&stream, planes, mark_slots)?;
            ensure!(marks.mark_bytes() == layout.mark_bytes, "Mini DFlash mark geometry");
            Some(marks)
        };
        let sites = cfg.taps.iter().map(|&id| if id == 48 { cuteafd_core::TapSite::FinalNorm }
            else { cuteafd_core::TapSite::LayerEntry(id) }).collect::<Vec<_>>();
        let verify = d.run(|| Verify::new(programs, stream.clone(), slots, 6656, decode_rows))?;
        let prefill = (0..waves).map(|_| TapBuffer::new(stream.clone(), cfg.hidden, &sites, prefill_rows)).collect::<Result<Vec<_>>>()?;
        let decode = TapBuffer::new(stream.clone(), cfg.hidden, &sites, decode_rows)?;
        stream.drain()?;
        Ok(Self { drafter: None, dflash: Some(Gqa { body, marks, frontier: RefCell::new(vec![0;slots]), embed_eps, stream }), marks: Vec::new(), verify, prefill, decode, sites,
            sequences, block: cfg.block, hidden: cfg.hidden, context_rows: 2048 })
    }

    pub fn frontier(&self, slot: usize) -> Result<(usize, usize)> {
        if let Some(d) = &self.drafter { d.frontier(slot) }
        else { Ok((0, self.dflash.as_ref().context("Mini GQA drafter missing")?.frontier.borrow()[slot])) }
    }
    pub fn bind_slot(&self, slot: usize, origin: usize, capacity: Option<usize>) -> Result<()> {
        if let Some(d) = &self.drafter { return d.bind_slot(slot, origin, capacity); }
        let gqa = self.dflash.as_ref().context("Mini GQA drafter missing")?;
        ensure!(origin == 0, "Mini GQA bind requires cold origin");
        gqa.stream.device.run(|| {
            for plane in gqa.body.gqa_window_planes(slot)? {
                // SAFETY: the slot is retired and all work shares the ordered stream.
                unsafe { gqa.stream.device.library.cuda_zero_bytes_async(plane, plane.bytes, gqa.stream.raw)?; }
            }
            Ok(())
        })?;
        gqa.frontier.borrow_mut()[slot] = 0;
        Ok(())
    }
    pub fn set_scores(&self, traced: bool, scored: bool) {
        if let Some(d) = &self.drafter { d.set_score_readback(scored); }
        if let Some(d) = &self.dflash { d.body.set_selector_readback(traced, scored); }
    }
    pub fn propose(&self, rows: &[DraftSeq], embed: Buffer, norm: Buffer, head: Buffer) -> Result<Vec<Draft>> {
        if let Some(d) = &self.drafter { return d.draft(rows, embed, norm, |n,a| self.select(n,a,head)); }
        let gqa = self.dflash.as_ref().context("Mini GQA drafter missing")?;
        for row in rows { ensure!(self.frontier(row.slot)?.1 == row.position, "Mini DFlash proposal frontier differs"); }
        gqa.stream.device.run(|| gqa.body.draft_normalized_table(rows, embed, norm, gqa.embed_eps, TargetHead::Bf16Buffer(&head)))
    }
    /// Compare real-weight block proposals with the independent reference generator.
    pub fn check_oracle(&self, path: &std::path::Path, embed: Buffer, norm: Buffer, head: Buffer) -> Result<()> {
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct Test { anchor: u32, tokens: Vec<u32>, features: Vec<[f32; 4]> }
        #[derive(Deserialize)]
        struct Fixture { rows: usize, results: Vec<Test> }
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(path.join("meta.json"))?)?;
        let features = std::fs::read(path.join("taps.bin"))?;
        let g = self.dflash.as_ref().context("Mini GQA oracle requires DFlash2")?;
        self.bind_slot(0, 0, None)?;
        let tap = &self.prefill[0];
        ensure!(features.len() == fixture.rows * self.sites.len() * self.hidden * 2 && features.len() <= tap.output.buffer.bytes,
            "Mini oracle features outside admitted tap owner");
        g.stream.drain()?;
        g.stream.device.run(|| g.stream.device.library.copy_h2d(tap.output.buffer, &features))?;
        self.update(tap, 0, 0, 0, fixture.rows)?;
        self.set_scores(true, true);
        let mut matched = 0;
        let mut total = 0;
        let mut first_matches = 0;
        let mut blocks = 0;
        let mut max_rel_l2 = 0.0f64;
        let mut max_top_rel_l2 = 0.0f64;
        let mut sum_rel_l2 = 0.0f64;
        let mut head_rows = 0;
        // Like GLM Flash's draft oracle, compare proposals and report numerical
        // drift. Strict target losslessness and delivered acceptance gate serving.
        let mut max_first_margin = 0.0f32;
        for test in fixture.results {
            let actual = self.propose(&[DraftSeq { slot: 0, anchor: test.anchor, position: fixture.rows, valid_from: 0 }], embed, norm, head)?;
            ensure!(actual[0].tokens.len() == self.block - 1 && test.tokens.len() == self.block - 1
                && test.features.len() == self.block - 1 && actual[0].features.len() == self.block - 1,
                "Mini oracle block length differs");
            ensure!(actual[0].features.iter().flatten().all(|v| v.is_finite()), "Mini oracle nonfinite selector features");
            let vocab = g.body.cfg.vocab;
            let valid_vocab = 200058;
            let expected = std::fs::read(path.join(format!("logits-{}.bin", test.anchor)))?;
            ensure!(expected.len() == (self.block-1)*valid_vocab*4, "Mini oracle logit extent differs");
            let logits = g.stream.device.run(|| g.body.last_logits(1))?;
            for (row, expected) in expected.chunks_exact(valid_vocab*4).enumerate() {
                let (mut error, mut norm) = (0.0f64, 0.0f64);
                for (a, b) in logits[(row+1)*vocab..(row+1)*vocab+valid_vocab].iter().zip(expected.chunks_exact(4)) {
                    let b = f32::from_le_bytes(b.try_into()?);
                    ensure!(a.is_finite() && b.is_finite(), "Mini oracle nonfinite logits");
                    error += (f64::from(*a)-f64::from(b)).powi(2);
                    norm += f64::from(b).powi(2);
                }
                let rel = (error/norm.max(f64::MIN_POSITIVE)).sqrt();
                let actual_row = &logits[(row+1)*vocab..(row+1)*vocab+valid_vocab];
                let reference = expected.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
                let mut ids = (0..valid_vocab).collect::<Vec<_>>();
                ids.select_nth_unstable_by(16, |&a,&b| reference[b].total_cmp(&reference[a]).then(a.cmp(&b)));
                let top_rel = |k: usize| {
                    let (e,n) = ids[..k].iter().fold((0.0f64,0.0f64), |(e,n),&id| {
                        let a = f64::from(actual_row[id]); let b = f64::from(reference[id]);
                        (e+(a-b).powi(2),n+b*b)
                    });
                    (e/n.max(f64::MIN_POSITIVE)).sqrt()
                };
                max_top_rel_l2 = max_top_rel_l2.max(top_rel(16));
                max_rel_l2 = max_rel_l2.max(rel);
                sum_rel_l2 += rel;
                head_rows += 1;
                println!("Mini DFlash2 anchor {} row {} head relL2 {rel:.6}", test.anchor, row+1);
            }
            blocks += 1;
            first_matches += usize::from(actual[0].tokens[0] == test.tokens[0]);
            if let Some(i) = actual[0].tokens.iter().zip(&test.tokens).position(|(a,b)| a != b) {
                ensure!(test.features[i][0].is_finite(), "Mini oracle nonfinite reference margin");
                max_first_margin = max_first_margin.max(test.features[i][0]);
            }
            matched += actual[0].tokens.iter().zip(&test.tokens).filter(|(a,b)| a == b).count();
            total += test.tokens.len();
            println!("Mini draft anchor {}: actual {:?}, expected {:?}", test.anchor, actual[0].tokens, test.tokens);
        }
        ensure!(blocks == 4 && total == 4 * (self.block - 1), "Mini oracle incomplete fixture");
        ensure!(head_rows > 0, "Mini oracle missing head rows");
        let mean_rel_l2 = sum_rel_l2 / head_rows as f64;
        println!("Mini DFlash2 real-weight oracle {matched}/{total}, first-position {first_matches}/{blocks}, max head relL2 {max_rel_l2:.6}, mean {mean_rel_l2:.6}, max top16 {max_top_rel_l2:.6}, max first-divergence margin {max_first_margin:.6}");
        Ok(())
    }

    pub fn mark_bytes(&self) -> usize { self.marks.iter().map(|m| m.bytes()).sum::<usize>() +
        self.dflash.as_ref().and_then(|g| g.marks.as_ref()).map_or(0, |m| m.mark_bytes()) }
    pub fn capture(&self, slot: usize, mark: usize, frontier: usize) -> Result<()> {
        let (origin, end) = self.frontier(slot)?;
        ensure!(end == frontier, "Mini drafter mark frontier differs");
        for (rank, marks) in self.ranks().zip(&self.marks) { rank.stream.device.run(|| marks.capture(rank,slot,mark,origin,end))?; }
        if let Some(g) = &self.dflash {
            // SAFETY: the retained exact ring planes share the ordered stream.
            unsafe { g.marks.as_ref().context("DFlash mark absent")?.capture(mark, &g.body.gqa_window_planes(slot)?, origin, end)?; }
        }
        Ok(())
    }
    pub fn restore(&self, slot: usize, mark: usize, frontier: usize) -> Result<()> {
        for (rank, marks) in self.ranks().zip(&self.marks) { rank.stream.device.run(|| marks.restore(rank,slot,mark,frontier))?; }
        if let Some(g) = &self.dflash {
            // SAFETY: retained exact mark and destination planes are ordered via the shared copy bank.
            unsafe { g.marks.as_ref().context("DFlash mark absent")?.restore(mark, &g.body.gqa_window_planes(slot)?, frontier)?; }
            g.frontier.borrow_mut()[slot] = frontier;
        }
        Ok(())
    }
    pub fn mark_buffers(&self, mark: usize) -> Vec<Buffer> {
        let mut buffers = self.marks.iter().filter_map(|m| m.mark(mark).ok()).collect::<Vec<_>>();
        if let Some(m) = self.dflash.as_ref().and_then(|g| g.marks.as_ref()) { buffers.extend(m.mark(mark).ok()); }
        buffers
    }
    pub fn drain(&self) -> Result<()> {
        for mark in &self.marks { mark.drain()?; }
        if let Some(d) = &self.drafter { d.drain()?; }
        if let Some(g) = &self.dflash { if let Some(m) = &g.marks { m.drain()?; } g.stream.drain()?; }
        Ok(())
    }

    pub fn select(&self, sequences: usize, anchors: Buffer, unembed_weight: Buffer) -> Result<Buffer> {
        self.drafter.as_ref().context("Mini recurrent body absent")?.lead.recurrent_tokens(sequences, anchors, unembed_weight)
    }

    pub fn taps(&self, decode: bool, unit: usize) -> Result<&TapBuffer<'a>> {
        if decode {
            Ok(&self.decode)
        } else {
            self.prefill
                .get(unit)
                .context("Mini drafter prefill tap unit outside admission")
        }
    }

    pub fn update(&self, taps: &TapBuffer<'a>, slot: usize, start: usize, first: usize, rows: usize) -> Result<()> {
        let pitch = self.sites.len() * self.hidden * 2;
        for offset in (0..rows).step_by(self.context_rows) {
            let count = (rows - offset).min(self.context_rows);
            let position = (first + offset) * pitch;
            let output = taps.output.buffer;
            ensure!(
                position.checked_add(count * pitch).is_some_and(|end| end <= output.bytes),
                "Mini drafter target taps outside owner"
            );
            let features = Buffer { ptr: output.ptr.wrapping_byte_add(position), bytes: count * pitch, ..output };
            if let Some(drafter) = &self.drafter {
                drafter.append_context(slot, start + offset, features, count)?;
            } else if let Some(gqa) = &self.dflash {
                ensure!(gqa.frontier.borrow()[slot] == start + offset, "Mini DFlash context gap");
                gqa.stream.device.run(|| gqa.body.append_gqa_features(slot, start + offset, features, count))?;
                gqa.frontier.borrow_mut()[slot] = start + offset + count;
            }
        }
        Ok(())
    }

    pub fn update_batch(
        &self,
        taps: &TapBuffer<'a>,
        spans: &[crate::shared::speculation::recurrent_cache::ContextSpan],
    ) -> Result<()> {
        if spans.is_empty() {
            return Ok(());
        }
        if let &[span] = spans {
            if span.rows == 0 {
                return Ok(());
            }
            return self.update(taps, span.slot, span.start, span.first_row, span.rows);
        }
        let max_rows = self.context_rows;
        let min_window = if let Some(drafter) = &self.drafter {
            drafter
                .lead
                .config
                .context
                .iter()
                .filter_map(|p| match p {
                    cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::BlockAnchoredWindow(w) => Some(*w),
                    cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full => None,
                })
                .min()
                .unwrap_or(usize::MAX)
        } else {
            cuteafd_loader::families::glm5::draft_physical::RING
        };
        let flush = |batch: &[crate::shared::speculation::recurrent_cache::ContextSpan]| -> Result<()> {
            if batch.is_empty() {
                return Ok(());
            }
            if let Some(drafter) = &self.drafter {
                drafter.append_context_batch(taps.output.buffer, batch)
            } else if let Some(gqa) = &self.dflash {
                {
                    let frontiers = gqa.frontier.borrow();
                    for span in batch {
                        ensure!(
                            frontiers.get(span.slot).copied() == Some(span.start),
                            "Mini DFlash batched context gap"
                        );
                    }
                }
                gqa.stream
                    .device
                    .run(|| gqa.body.append_gqa_batch(taps.output.buffer, batch))?;
                let mut frontiers = gqa.frontier.borrow_mut();
                for span in batch {
                    frontiers[span.slot] = span.start + span.rows;
                }
                Ok(())
            } else {
                Ok(())
            }
        };
        let mut batch = Vec::with_capacity(spans.len());
        let mut batch_rows = 0usize;
        let mut seen = std::collections::BTreeSet::new();
        for &span in spans {
            if span.rows == 0 {
                continue;
            }
            if span.rows > min_window || span.rows > max_rows {
                flush(&batch)?;
                batch.clear();
                batch_rows = 0;
                seen.clear();
                self.update(taps, span.slot, span.start, span.first_row, span.rows)?;
                continue;
            }
            if batch_rows + span.rows > max_rows || !seen.insert(span.slot) {
                flush(&batch)?;
                batch.clear();
                batch_rows = 0;
                seen.clear();
                seen.insert(span.slot);
            }
            batch.push(span);
            batch_rows += span.rows;
        }
        flush(&batch)?;
        Ok(())
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::shared::{
        memory::device::{Allocation, Device},
        speculation::{dflash::DraftSeq, recurrent_cache::ContextSpan},
    };
    use cuteafd_core::memory_layout::Tag;
    use cuteafd_loader::{
        families::qwen4::Qwen4KvCache,
        formats::block_dspark::recurrent::{loading::RecurrentSource, runtime::RecurrentRuntimeLayout},
        plan::{
            checkpoint::{Checkpoint, CheckpointView, CoreView},
        },
        serving_capacity::WeightArenaLayout,
    };
    use crate::shared::{
        experts::rtx::dense::{DenseLoad, Operand, Rect},
        memory::weight_arena::DeviceWeightArena,
    };

    /// Real Inkling 2 Mini bundled drafter GPU oracle (run via `golden-run.py --native-test`):
    /// 1. Proves batched context update (both contiguous C64 decode and gapped speculative verify)
    ///    produces bitwise-identical KV caches, `cache.last_features`, hidden states, logits,
    ///    logprobs and drafted tokens compared to per-request `append_context`.
    /// 2. Proves row-invariance across batch compositions (batch sizes 1, 2, 8, 11, 16, 32,
    ///    diverse context lengths 17..1500, and reversed/shuffled batch orderings).
    #[test]
    #[ignore = "requires real Inkling 2 Mini snapshot, native programs and GPU; golden-run only"]
    fn installed_inkling_mini_drafter_batch_and_context_invariance() -> Result<()> {
        let library = unsafe {
            cuteafd_ffi::NativeLibrary::load_for_experts(
                std::env::var("CUTEAFD_NATIVE_LIB")?,
                cuteafd_core::ExpertGeometry::INKLING2_MINI,
            )?
        };
        let programs = library
            .programs()?
            .with_manifest(std::path::Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
        let device = Device {
            library: &library,
            id: 0,
        };
        let stream = Rc::new(Stream::new(device)?);
        let view = CheckpointView::select(&snapshot, CoreView::Original)?;
        let source = RecurrentSource::open(&view)?;
        let config = source.config.clone();
        let target = Checkpoint::open_for_role(&snapshot, "mini-draft-batch-test", |name| {
            matches!(
                name,
                "model.llm.embed.weight" | "model.llm.embed_norm.weight" | "model.llm.unembed.weight"
            )
        })?;
        let names = [
            "model.llm.embed.weight",
            "model.llm.embed_norm.weight",
            "model.llm.unembed.weight",
        ];
        let target_arena = DeviceWeightArena::new(
            device,
            WeightArenaLayout::new(
                "mini-draft-test/target",
                names
                    .iter()
                    .map(|&name| {
                        Ok((
                            name.to_owned(),
                            usize::try_from(target.header(name).context("target header missing")?.byte_length)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
            )?,
            cuteafd_core::memory_layout::ArenaTag::weights("mini-draft-test/target"),
        )?;
        let dense = DenseLoad::mount(&library);
        let mut targets = std::collections::BTreeMap::new();
        for name in names {
            let rect = Rect::tensor(&target, name, None, None)?;
            let bytes = rect.bytes();
            let owner = target_arena.view(name, bytes)?;
            dense.operand(Operand {
                device: 0,
                buffer: owner.buffer,
                lands: vec![(rect, 0)],
                bytes,
            })?;
            targets.insert(name, owner);
        }
        dense.release();
        let embed = targets["model.llm.embed.weight"].buffer;
        let embed_norm = targets["model.llm.embed_norm.weight"].buffer;
        let unembed = targets["model.llm.unembed.weight"].buffer;

        let make_drafter = |kv: Qwen4KvCache| -> Result<Drafter<'_>> {
            let weights = Weights::load(device, &source, source.rank(0, 1, 0)?, stream.raw)?;
            let runtime = RecurrentRuntimeLayout::new(
                &config,
                64,
                64,
                2048,
                4096,
                64 * 1536,
                library.sm_count()? as usize,
                kv,
                0,
                1,
                0,
            )?;
            let kernels = ProgramsForRank::load(&programs, &weights, &config, &runtime, device)?;
            let rank = Rank::allocate(weights, Rc::clone(&stream), config.clone(), runtime.clone(), kernels)?;
            rank.initialize_derived()?;
            rank.fold_head(unembed)?;
            stream.drain()?;
            let context = ContextBank {
                state: RefCell::new(ContextState::new(
                    runtime.slots,
                    4096,
                    runtime.cache.target_tokens,
                    0x494e4b4d494e49,
                )?),
            };
            let drafter = Drafter::whole(rank, context)?;
            drafter.set_score_readback(true);
            Ok(drafter)
        };

        let hidden = config.transformer.hidden;
        let taps = config.taps.len();
        let pitch = taps * hidden * 2;
        let fill_taps = |rows: usize, seed: u64| -> Vec<u8> {
            let pattern = [0x3b80u16, 0xbb80, 0x3c00, 0xbc00, 0x3a80, 0xba80, 0x3c40, 0xbc40];
            let mut out = Vec::with_capacity(rows * pitch);
            for r in 0..rows {
                for c in 0..(taps * hidden) {
                    let h = ((r as u64).wrapping_mul(2654435761) ^ (c as u64).wrapping_mul(40503) ^ seed) as usize;
                    out.extend_from_slice(&pattern[h % pattern.len()].to_le_bytes());
                }
            }
            out
        };

        for kv in [Qwen4KvCache::Fp8, Qwen4KvCache::Bf16] {
            let serial = make_drafter(kv)?;
            let batched = make_drafter(kv)?;
            let seq_lens: Vec<usize> = (0..64)
                .map(|i| {
                    let base = [
                        17, 33, 64, 128, 192, 256, 583, 1024, 1500, 31, 95, 311, 777, 1024, 48, 600,
                        19, 45, 72, 130, 200, 300, 512, 900, 1100, 65, 193, 400, 800, 1025, 24, 1400,
                    ];
                    base[i % base.len()] + (i / base.len()) * 13
                })
                .collect();
            let tap_stage = Allocation::new(device, 2048 * pitch, Tag::drafter_workspace("test/taps"))?;
            for (slot, &len) in seq_lens.iter().enumerate() {
                serial.bind_slot(slot, 0, Some(len + 64))?;
                batched.bind_slot(slot, 0, Some(len + 64))?;
                let bytes = fill_taps(len, 1000 + slot as u64);
                library.copy_h2d(Buffer { bytes: bytes.len(), ..tap_stage.buffer }, &bytes)?;
                serial.append_context(slot, 0, Buffer { bytes: bytes.len(), ..tap_stage.buffer }, len)?;
                batched.append_context(slot, 0, Buffer { bytes: bytes.len(), ..tap_stage.buffer }, len)?;
            }

            // Part 1a: Batched contiguous decode update (64 slots x 1 row) vs serial per-slot update.
            let decode_bytes = fill_taps(64, 7777);
            library.copy_h2d(Buffer { bytes: decode_bytes.len(), ..tap_stage.buffer }, &decode_bytes)?;
            let decode_spans: Vec<ContextSpan> = (0..64)
                .map(|slot| ContextSpan {
                    slot,
                    start: seq_lens[slot],
                    first_row: slot,
                    rows: 1,
                })
                .collect();
            for span in &decode_spans {
                serial.append_context(
                    span.slot,
                    span.start,
                    Buffer {
                        ptr: tap_stage.buffer.ptr.wrapping_byte_add(span.first_row * pitch),
                        bytes: pitch,
                        ..tap_stage.buffer
                    },
                    1,
                )?;
            }
            batched.append_context_batch(tap_stage.buffer, &decode_spans)?;

            // Part 1b: Batched gapped speculative verify update (varying kept 1..4 with rejected gaps) vs serial.
            let mut gapped_spans = Vec::new();
            let mut cursor = 0usize;
            for slot in 0..16 {
                let verified = 1 + (slot % 4);
                let kept = 1 + (slot % verified);
                gapped_spans.push(ContextSpan {
                    slot,
                    start: seq_lens[slot] + 1,
                    first_row: cursor,
                    rows: kept,
                });
                cursor += verified; // leaves rejected rows as gaps between kept spans
            }
            let gap_bytes = fill_taps(cursor, 9999);
            library.copy_h2d(Buffer { bytes: gap_bytes.len(), ..tap_stage.buffer }, &gap_bytes)?;
            for span in &gapped_spans {
                serial.append_context(
                    span.slot,
                    span.start,
                    Buffer {
                        ptr: tap_stage.buffer.ptr.wrapping_byte_add(span.first_row * pitch),
                        bytes: span.rows * pitch,
                        ..tap_stage.buffer
                    },
                    span.rows,
                )?;
            }
            batched.append_context_batch(tap_stage.buffer, &gapped_spans)?;

            // Check bitwise equality of `cache.last_features` across all 64 slots.
            let lf_bytes = 64 * hidden * 2;
            let mut lf_serial = vec![0u8; lf_bytes];
            let mut lf_batched = vec![0u8; lf_bytes];
            library.copy_d2h(&mut lf_serial, Buffer { bytes: lf_bytes, ..serial.lead.buffer("cache.last_features")? })?;
            library.copy_d2h(&mut lf_batched, Buffer { bytes: lf_bytes, ..batched.lead.buffer("cache.last_features")? })?;
            ensure!(lf_serial == lf_batched, "kv={kv:?}: cache.last_features differs between serial and batched update");

            // Check bitwise equality of proposals, confidences, hidden states, and logits: serial vs batched.
            let draft_seq = |slot: usize, drafter: &Drafter<'_>| -> Result<DraftSeq> {
                Ok(DraftSeq {
                    slot,
                    anchor: 500 + (slot as u32) * 17,
                    position: drafter.frontier(slot)?.1,
                    valid_from: 0,
                })
            };
            let all_64_serial: Vec<_> = (0..64).map(|s| draft_seq(s, &serial)).collect::<Result<_>>()?;
            let all_64_batched: Vec<_> = (0..64).map(|s| draft_seq(s, &batched)).collect::<Result<_>>()?;
            let out_serial = serial.draft(&all_64_serial, embed, embed_norm, |n, a| {
                serial.lead.recurrent_tokens(n, a, unembed)
            })?;
            let out_batched = batched.draft(&all_64_batched, embed, embed_norm, |n, a| {
                batched.lead.recurrent_tokens(n, a, unembed)
            })?;
            let logits_bytes = 64 * config.valid_vocab * 4;
            let mut logits_serial = vec![0u8; logits_bytes];
            let mut logits_batched = vec![0u8; logits_bytes];
            library.copy_d2h(&mut logits_serial, Buffer { bytes: logits_bytes, ..serial.lead.buffer("head.logits")? })?;
            library.copy_d2h(&mut logits_batched, Buffer { bytes: logits_bytes, ..batched.lead.buffer("head.logits")? })?;
            ensure!(logits_serial == logits_batched, "kv={kv:?}: head.logits differ between serial and batched context update");
            for slot in 0..64 {
                ensure!(
                    out_serial[slot].tokens == out_batched[slot].tokens,
                    "kv={kv:?}: slot {slot} tokens differ between serial and batched context update"
                );
                let conf_s: Vec<u32> = out_serial[slot].confidence.iter().map(|v| v.to_bits()).collect();
                let conf_b: Vec<u32> = out_batched[slot].confidence.iter().map(|v| v.to_bits()).collect();
                ensure!(
                    conf_s == conf_b,
                    "kv={kv:?}: slot {slot} confidence bits differ between serial and batched context update"
                );
            }

            // Part 2: Row-invariance across batch compositions (1, 2, 8, 11, 16, 32, 48, 64 and reversed orders).
            let probes = [0usize, 1, 4, 6, 7, 8, 15, 31, 47, 63];
            let block_bytes = config.transformer.block * hidden * 2;
            let row_logit_bytes = config.valid_vocab * 4;
            let mut solo_ref = std::collections::BTreeMap::new();
            for &slot in &probes {
                let seq = [draft_seq(slot, &batched)?];
                let d = batched.draft(&seq, embed, embed_norm, |n, a| batched.lead.recurrent_tokens(n, a, unembed))?;
                let mut h = vec![0u8; block_bytes];
                let mut l = vec![0u8; row_logit_bytes];
                library.copy_d2h(&mut h, Buffer { bytes: block_bytes, ..batched.lead.buffer("step.normalized")? })?;
                library.copy_d2h(&mut l, Buffer { bytes: row_logit_bytes, ..batched.lead.buffer("head.logits")? })?;
                let conf: Vec<u32> = d[0].confidence.iter().map(|v| v.to_bits()).collect();
                solo_ref.insert(slot, (d[0].tokens.clone(), conf, h, l));
            }
            let compositions: Vec<Vec<usize>> = vec![
                vec![0, 8],
                vec![8, 0],
                vec![6, 63],
                vec![63, 6],
                vec![0, 1, 4, 6, 7, 8, 15, 31],
                vec![31, 15, 8, 7, 6, 4, 1, 0, 2, 10, 22], // 11 sparse drafting requests
                (0..16).collect(),
                (0..32).collect(),
                (0..48).collect(),
                (0..48).rev().collect(),
                (0..64).collect(),
                (0..64).rev().collect(),
            ];
            for comp in &compositions {
                let seqs: Vec<_> = comp.iter().map(|&s| draft_seq(s, &batched)).collect::<Result<_>>()?;
                let got = batched.draft(&seqs, embed, embed_norm, |n, a| batched.lead.recurrent_tokens(n, a, unembed))?;
                let all_h_bytes = comp.len() * block_bytes;
                let all_l_bytes = comp.len() * row_logit_bytes;
                let mut all_h = vec![0u8; all_h_bytes];
                let mut all_l = vec![0u8; all_l_bytes];
                library.copy_d2h(&mut all_h, Buffer { bytes: all_h_bytes, ..batched.lead.buffer("step.normalized")? })?;
                library.copy_d2h(&mut all_l, Buffer { bytes: all_l_bytes, ..batched.lead.buffer("head.logits")? })?;
                for (pos, &slot) in comp.iter().enumerate() {
                    let Some((exp_toks, exp_conf, exp_h, exp_l)) = solo_ref.get(&slot) else {
                        continue;
                    };
                    let got_conf: Vec<u32> = got[pos].confidence.iter().map(|v| v.to_bits()).collect();
                    ensure!(
                        &got[pos].tokens == exp_toks,
                        "kv={kv:?}: row-invariance tokens mismatch slot={slot} batch_len={} pos={pos}",
                        comp.len()
                    );
                    ensure!(
                        &got_conf == exp_conf,
                        "kv={kv:?}: row-invariance confidence mismatch slot={slot} batch_len={} pos={pos}",
                        comp.len()
                    );
                    ensure!(
                        &all_h[pos * block_bytes..(pos + 1) * block_bytes] == exp_h.as_slice(),
                        "kv={kv:?}: row-invariance hidden mismatch slot={slot} batch_len={} pos={pos}",
                        comp.len()
                    );
                    ensure!(
                        &all_l[pos * row_logit_bytes..(pos + 1) * row_logit_bytes] == exp_l.as_slice(),
                        "kv={kv:?}: row-invariance logits mismatch slot={slot} batch_len={} pos={pos}",
                        comp.len()
                    );
                }
            }
            eprintln!("MINI_DRAFTER_BATCH_INVARIANCE kv={kv:?} passed: serial==batched (contiguous C64 + gapped) and batch compositions [1,2,8,11,16,32,48,64] bitwise exact");
        }
        Ok(())
    }
}
