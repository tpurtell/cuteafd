//! Multi-slot implementation of the shared, slot-indexed block verification contract.
//! Padding is absent from native state tables and scattered into zeroed physical outputs.
use super::*;
use crate::shared::{prefix::view, qwen::{MarkLayout, REPLAY_ROWS},
    speculation::target::{AcceptedPrefix, BlockVerifyTarget, HiddenTap, VerifyOutput, VerifyRequest},
    token_io::DeviceLogits};
use std::{rc::Rc, sync::atomic::{AtomicU64, Ordering}};

static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);

pub(crate) struct DenseMark<'a> {
    owner: u64,
    generation: u64,
    slots: Vec<(usize, rows::Slot)>,
    storage: Rc<Allocation<'a>>,
}

pub(super) struct Verification<'a> {
    owner: u64,
    generation: u64,
    mark: Rc<Allocation<'a>>,
    layout: MarkLayout,
    logits: Allocation<'a>,
    hidden: Allocation<'a>,
    taps: Vec<HiddenTap>,
    pub(super) pending: Option<rows::Rows>,
}

impl<'a> Verification<'a> {
    fn layout(cfg: &Qwen35Config, slots: usize) -> Result<MarkLayout> {
        let gdn = GdnLayout::new(cfg.gdn)?;
        let regions = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn)
            .flat_map(|_| (0..slots).flat_map(|_| [(0, gdn.conv_slot_bytes), (0, gdn.state_slot_bytes)])).collect::<Vec<_>>();
        MarkLayout::new(&regions)
    }

    #[cfg(test)]
    pub(super) fn bytes(cfg: &Qwen35Config, slots: usize) -> Result<usize> {
        let mark = Self::layout(cfg, slots)?.bytes.max(256);
        let logits = REPLAY_ROWS.checked_mul(cfg.vocab).and_then(|n| n.checked_mul(4))
            .context("dense verify logits size overflow")?;
        let taps = REPLAY_ROWS.checked_mul(cfg.hidden).and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(cfg.attention.len())).context("dense verify taps size overflow")?;
        mark.checked_add(logits).and_then(|n| n.checked_add(taps)).context("dense verify admission overflow")
    }

    pub(super) fn new(device: Device<'a>, cfg: &Qwen35Config, slots: usize) -> Result<Self> {
        let mut layout = Self::layout(cfg, slots)?;
        for owner in &mut layout.owners { owner.0 = device.id; }
        let mark = Rc::new(Allocation::new(device, layout.bytes.max(256))?);
        let logits = Allocation::new(device, REPLAY_ROWS * cfg.vocab * 4)?;
        let hidden = Allocation::new(device, cfg.attention.len() * REPLAY_ROWS * cfg.hidden * 2)?;
        let owner = NEXT_ENGINE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow::anyhow!("dense target identity exhausted"))?;
        Ok(Self { owner, generation: 0, mark, layout, logits, hidden, taps: Vec::new(), pending: None })
    }

    pub(super) fn allocated_bytes(&self) -> [usize; 3] {
        [self.mark.buffer.bytes, self.logits.buffer.bytes, self.hidden.buffer.bytes]
    }

    pub(super) fn invalidate(&mut self) -> Result<()> {
        self.generation = self.generation.checked_add(1).context("dense mark generation exhausted")?;
        self.pending = None;
        self.taps.clear();
        Ok(())
    }

    fn validate(&self, mark: &DenseMark<'_>) -> Result<()> {
        ensure!(mark.owner == self.owner && mark.generation == self.generation
            && Rc::ptr_eq(&mark.storage, &self.mark), "stale or foreign dense checkpoint");
        Ok(())
    }
}

impl<'a> DenseEngine<'a> {
    /// Diagnostic on a fresh executor: every retained recurrence/conv byte must
    /// equal serial prefill, and padding/tap order must preserve physical rows.
    pub(crate) fn verify_check(&mut self, tokens: &[u32]) -> Result<()> {
        use crate::shared::speculation::target::VerifyRow;
        ensure!(self.position == 0 && tokens.len() >= 9, "verify check requires a fresh target and nine inputs");
        let library = self.stream.device.library;
        let initial = self.mark_regions().into_iter().map(|region| {
            let mut bytes = vec![0; region.bytes];
            library.copy_d2h(&mut bytes, region)?;
            Ok(bytes)
        }).collect::<Result<Vec<_>>>()?;
        let reset = |engine: &mut Self| -> Result<()> {
            engine.stream.drain()?;
            for (region, bytes) in engine.mark_regions().into_iter().zip(&initial) { engine.put(region, bytes)?; }
            engine.stream.drain()?;
            engine.position = 0;
            engine.slots[0].position = 0;
            engine.verification.invalidate()?;
            Ok(())
        };
        let original = self.checkpoint(&[0])?;
        let mut layers = vec![self.cfg.attention.len() - 1, 0];
        if self.cfg.attention.len() > 2 { layers.push(self.cfg.attention.len() / 2); }
        let mut rows = vec![VerifyRow { slot: usize::MAX, position: usize::MAX, token: None }];
        for (position, &token) in tokens[..7].iter().enumerate() {
            rows.push(VerifyRow { slot: 0, position, token: Some(token) });
            if position == 2 { rows.push(rows[0]); }
        }
        rows.push(rows[0]);
        let (logits, taps) = {
            let output = self.verify(&original, VerifyRequest { rows: &rows, tap_layers: &layers })?;
            (output.logits, output.taps.to_vec())
        };
        let actual = logits.to_host(self.stream.device.library)?;
        let mut tap_bytes = Vec::new();
        for (&layer, tap) in layers.iter().zip(&taps) {
            ensure!(tap.layer == layer && tap.rows == rows.len() && tap.hidden == self.cfg.hidden,
                "dense verify tap ordering/shape differs");
            let mut bytes = vec![0; rows.len() * self.cfg.hidden * 2];
            self.stream.device.library.copy_d2h(&mut bytes, CuteafdDeviceBuffer { ptr: tap.ptr.cast_mut(),
                bytes: rows.len() * self.cfg.hidden * 2, device_id: self.stream.device.id,
                ..CuteafdDeviceBuffer::default() })?;
            tap_bytes.push(bytes);
        }
        let mut active = 0;
        for (physical, row) in rows.iter().enumerate() {
            if row.token.is_none() {
                ensure!(actual[physical * self.cfg.vocab..(physical + 1) * self.cfg.vocab].iter().all(|&v| v == 0.0),
                    "dense padding logits are not zero");
                ensure!(tap_bytes.iter().all(|bytes| bytes[physical * self.cfg.hidden * 2..(physical + 1) * self.cfg.hidden * 2]
                    .iter().all(|&v| v == 0)), "dense padding taps are not zero");
            } else { active += 1; }
        }
        ensure!(active == 7, "dense verify fixture active rows changed");
        self.restore(&original)?;
        let library = self.stream.device.library;
        for (active, &token) in tokens[..7].iter().enumerate() {
            let mut expected_taps = vec![Vec::new(); layers.len()];
            let expected = self.evaluate(&[token], |layer, hidden| {
                if let Some(index) = layers.iter().position(|&id| id == layer) {
                    expected_taps[index].resize(hidden.bytes, 0);
                    library.copy_d2h(&mut expected_taps[index], hidden)?;
                }
                Ok(())
            })?;
            let physical = rows.iter().enumerate().filter(|(_, row)| row.token.is_some()).nth(active).unwrap().0;
            ensure!(expected.iter().map(|v| v.to_bits()).eq(actual[physical * self.cfg.vocab..(physical + 1) * self.cfg.vocab]
                .iter().map(|v| v.to_bits())), "dense serial/verify logits differ at active input {active}");
            for (index, bytes) in expected_taps.iter().enumerate() {
                ensure!(*bytes == tap_bytes[index][physical * self.cfg.hidden * 2..(physical + 1) * self.cfg.hidden * 2],
                    "dense serial/verify post-block tap {} differs at active input {active}", layers[index]);
            }
        }
        // Host-only diagnostic baseline: generation-invalidated marks are never reused.
        reset(self)?;
        for keep in 0..=7 {
            let mark = self.checkpoint(&[0])?;
            self.verify(&mark, VerifyRequest { rows: &rows, tap_layers: &[] })?;
            self.commit(&mark, &[AcceptedPrefix { slot: 0, rows: keep }])?;
            let mut replay = Vec::new();
            for region in self.mark_regions() {
                let mut bytes = vec![0; region.bytes];
                library.copy_d2h(&mut bytes, region)?;
                replay.push(bytes);
            }
            reset(self)?;
            for &token in &tokens[..keep] { self.evaluate(&[token], |_, _| Ok(()))?; }
            for (region, expected) in self.mark_regions().into_iter().zip(replay) {
                let mut bytes = vec![0; region.bytes];
                library.copy_d2h(&mut bytes, region)?;
                ensure!(bytes == expected, "dense accepted-input replay state differs for keep={keep}");
            }
            reset(self)?;
        }
        for &token in &tokens[..3] { self.evaluate(&[token], |_, _| Ok(()))?; }
        let continuation = self.evaluate(&tokens[3..7], |_, _| Ok(()))?;
        // A normal evaluate invalidates its old mark; rebuild the same prefix first.
        reset(self)?;
        for &token in &tokens[..3] { self.evaluate(&[token], |_, _| Ok(()))?; }
        let mark = self.checkpoint(&[0])?;
        let nonzero = tokens[3..7].iter().enumerate().map(|(i, &token)| VerifyRow {
            slot: 0, position: 3 + i, token: Some(token) }).collect::<Vec<_>>();
        self.verify(&mark, VerifyRequest { rows: &nonzero, tap_layers: &[] })?;
        self.restore(&mark)?;
        let restored = self.evaluate(&tokens[3..7], |_, _| Ok(()))?;
        ensure!(continuation.iter().map(|v| v.to_bits()).eq(restored.iter().map(|v| v.to_bits())),
            "dense nonzero-prefix restored continuation differs");
        reset(self)?;
        println!("dense block verify: serial logits/taps, sentinel padding, nonzero-prefix restore and every keep=0..7 state byte-exact");
        Ok(())
    }

    /// Exercise real disjoint KV pages and interleaved physical rows. All slot
    /// state, including untouched slots, must match independent serial prefill.
    pub(crate) fn multislot_verify_check(&mut self, tokens: &[u32]) -> Result<()> {
        use crate::shared::speculation::target::VerifyRow;
        ensure!(self.slots.len() >= 2 && tokens.len() >= 9 && self.pool_tokens >= 384,
            "multislot verify check requires two slots, six pages and nine inputs");
        let library = self.stream.device.library;
        let sequences = [&tokens[..7], &tokens[2..8]];
        let reset = |engine: &mut Self| -> Result<()> {
            for slot in 0..engine.slots.len() { engine.reset_slot(slot)?; }
            engine.bind_slot(0, &[3, 1])?;
            engine.bind_slot(1, &[5, 2])?;
            for (slot, sequence) in sequences.iter().enumerate() {
                engine.prefill_device(slot, &sequence[..3], 1)?;
            }
            engine.stream.drain()
        };
        let state = |engine: &Self| -> Result<Vec<Vec<u8>>> {
            engine.stream.drain()?;
            engine.mark_regions().into_iter().map(|region| {
                let mut bytes = vec![0; region.bytes];
                library.copy_d2h(&mut bytes, region)?;
                Ok(bytes)
            }).collect()
        };
        reset(self)?;
        let padding = VerifyRow { slot: usize::MAX, position: usize::MAX, token: None };
        let mut rows = vec![padding];
        for offset in 0..4 {
            for slot in [1, 0] {
                if let Some(&token) = sequences[slot].get(3 + offset) {
                    rows.push(VerifyRow { slot, position: 3 + offset, token: Some(token) });
                }
            }
            rows.push(padding);
        }
        let mark = self.checkpoint(&[1, 0])?;
        let actual = self.verify(&mark, VerifyRequest { rows: &rows, tap_layers: &[] })?
            .logits.to_host(library)?;
        self.restore(&mark)?;
        for (physical, row) in rows.iter().enumerate() {
            let actual = &actual[physical * self.cfg.vocab..(physical + 1) * self.cfg.vocab];
            if let Some(token) = row.token {
                let expected = self.prefill_device(row.slot, &[token], 1)?.to_host(library)?;
                ensure!(expected.iter().map(|v| v.to_bits()).eq(actual.iter().map(|v| v.to_bits())),
                    "dense multislot serial/verify logits differ at physical row {physical}");
            } else {
                ensure!(actual.iter().all(|&v| v == 0.0), "dense multislot padding logits are not zero");
            }
        }
        for kept in [[0, 3], [1, 2], [4, 0], [2, 3]] {
            reset(self)?;
            let mark = self.checkpoint(&[1, 0])?;
            self.verify(&mark, VerifyRequest { rows: &rows, tap_layers: &[] })?;
            self.commit(&mark, &[AcceptedPrefix { slot: 0, rows: kept[0] },
                AcceptedPrefix { slot: 1, rows: kept[1] }])?;
            let actual = state(self)?;
            ensure!(self.slots[0].position == 3 + kept[0] && self.slots[1].position == 3 + kept[1],
                "dense multislot commit changed the wrong frontier");
            reset(self)?;
            for (slot, sequence) in sequences.iter().enumerate() {
                for &token in &sequence[3..3 + kept[slot]] { self.prefill_device(slot, &[token], 1)?; }
            }
            ensure!(state(self)? == actual, "dense multislot accepted-input state differs for {kept:?}");
        }
        let kv = |engine: &Self| -> Result<Vec<Vec<u8>>> {
            let record = engine.kv_format.record_bytes(engine.cfg.kv_heads, engine.cfg.head_dim);
            let mut output = Vec::new();
            for slot in &engine.slots {
                for position in 0..slot.position {
                    let offset = (slot.pages[position / 64] as usize * 64 + position % 64) * record;
                    for pool in engine.kv.iter().flatten() {
                        let mut bytes = vec![0; record];
                        library.copy_d2h(&mut bytes, view(pool.buffer, offset, record)?)?;
                        output.push(bytes);
                    }
                }
            }
            Ok(output)
        };
        reset(self)?;
        let mark = self.checkpoint(&[1, 0])?;
        let expected_logits = self.verify(&mark, VerifyRequest { rows: &rows, tap_layers: &[] })?
            .logits.to_host(library)?;
        self.commit(&mark, &[AcceptedPrefix { slot: 0, rows: 4 }, AcceptedPrefix { slot: 1, rows: 3 }])?;
        let expected_state = state(self)?;
        let expected_kv = kv(self)?;
        reset(self)?;
        let stale = self.checkpoint(&[1, 0])?;
        let actual = self.decode_device(&[1, 0], VerifyRequest { rows: &rows, tap_layers: &[] })?.to_host(library)?;
        ensure!(actual.iter().map(|v| v.to_bits()).eq(expected_logits.iter().map(|v| v.to_bits()))
            && state(self)? == expected_state && kv(self)? == expected_kv,
            "dense direct multislot logits, KV or recurrent state differ from full speculative commit");
        ensure!(self.slots[0].position == 7 && self.slots[1].position == 6
            && self.verification.pending.is_none(), "dense direct decode did not retire committed inputs");
        ensure!(self.restore(&stale).is_err(), "dense direct decode retained a stale transactional mark");
        let invalid = [VerifyRow { slot: 0, position: 6, token: Some(tokens[0]) }];
        ensure!(self.decode_device(&[0], VerifyRequest { rows: &invalid, tap_layers: &[] }).is_err()
            && self.healthy && state(self)? == expected_state && kv(self)? == expected_kv,
            "dense invalid direct decode changed committed state");
        for slot in 0..self.slots.len() { self.reset_slot(slot)?; }
        println!("dense multislot verify: interleaved logits, sentinel padding, accepted-input replay and direct-commit logits/KV/state byte-exact");
        Ok(())
    }

    fn mark_regions(&self) -> Vec<CuteafdDeviceBuffer> {
        self.homes.layers.iter().filter_map(|home| home.gdn_ordinal)
            .flat_map(|ordinal| (0..self.slots.len()).flat_map(move |slot| self.bank.slot_regions(ordinal, slot))).collect()
    }

    fn move_mark(&self, mark: &DenseMark<'_>, capture: bool) -> Result<()> {
        let regions = self.mark_regions();
        ensure!(regions.len() == self.verification.layout.regions.len(), "dense mark geometry changed");
        for (index, (region, &(owner, offset, bytes))) in regions.into_iter().zip(&self.verification.layout.regions).enumerate() {
            let slot = (index / 2) % self.slots.len();
            if !mark.slots.iter().any(|(marked, _)| *marked == slot) { continue; }
            ensure!(owner == 0 && bytes == region.bytes && region.device_id == mark.storage.buffer.device_id,
                "dense mark region does not match its owner");
            let saved = view(mark.storage.buffer, offset, bytes)?;
            let (dst, src) = if capture { (saved, region) } else { (region, saved) };
            // SAFETY: mark storage and the live bank are retained until the enclosing drain.
            unsafe { self.stream.device.library.copy_d2d_async(dst, src, bytes, self.stream.raw)?; }
        }
        // Every reader and writer of the mark and the live bank runs on this stream.
        Ok(())
    }

    fn restore_mark(&mut self, mark: &DenseMark<'_>) -> Result<()> {
        self.verification.validate(mark)?;
        self.healthy = false;
        self.move_mark(mark, false)?;
        // Append-only records beyond each saved frontier are unreachable and overwritten.
        for (slot, state) in &mark.slots { self.slots[*slot] = state.clone(); }
        self.position = self.slots[0].position;
        self.verification.pending = None;
        self.verification.taps.clear();
        self.healthy = true;
        Ok(())
    }

    /// Plain decode commits every input in place; only speculative rows need a
    /// checkpoint and replay. Retained prefix snapshots own a separate arena.
    pub(crate) fn decode_device(&mut self, slots: &[usize], request: VerifyRequest<'_>) -> Result<DeviceLogits> {
        ensure!(self.healthy && self.verification.pending.is_none(), "dense decode requires committed healthy state");
        ensure!(!slots.is_empty(), "dense decode requires active slots");
        let marked = slots.iter().map(|&slot| {
            self.slots.get(slot).cloned().map(|state| (slot, state))
                .context("dense decode slot outside admission")
        }).collect::<Result<Vec<_>>>()?;
        let plan = rows::Rows::new(&request, self.cfg.attention.len(), self.cfg.vocab, &marked,
            self.cfg.max_context, self.pool_tokens)?;
        self.verification.invalidate()?;
        self.verify_rows(plan, request.rows.len(), request.tap_layers, false)?;
        Ok(DeviceLogits { ptr: self.verification.logits.buffer.ptr, rows: request.rows.len(),
            vocab: self.cfg.vocab, stride: self.cfg.vocab, stream: self.stream.raw, greedy: None })
    }

    pub(crate) fn verify_taps(&self) -> &[HiddenTap] { &self.verification.taps }

    fn verify_rows(&mut self, plan: rows::Rows, rows: usize, layers: &[usize], speculative: bool) -> Result<()> {
        self.healthy = false;
        let tokens = &plan.tokens;
        let physical = &plan.physical;
        let library = self.stream.device.library;
        // SAFETY: output arenas are admitted; their prior consumers precede this on the one stream.
        unsafe {
            library.cuda_zero_bytes_async(self.verification.logits.buffer, rows * self.cfg.vocab * 4, self.stream.raw)?;
            library.cuda_zero_bytes_async(self.verification.hidden.buffer, layers.len() * rows * self.cfg.hidden * 2,
                self.stream.raw)?;
        }
        self.verification.taps = layers.iter().enumerate().map(|(index, &layer)| HiddenTap {
            layer, ptr: self.verification.hidden.buffer.ptr.cast::<u8>().wrapping_add(index * rows * self.cfg.hidden * 2).cast(),
            rows, hidden: self.cfg.hidden, stride: self.cfg.hidden, stream: self.stream.raw }).collect();
        if !tokens.is_empty() {
            self.row_tables(&plan)?;
            self.embedding.embed(tokens, self.ids.buffer, 1, self.planes.residual, self.stream.raw)?;
            let step = Step { rows: tokens.len(), capacity: REPLAY_ROWS, table_stride: plan.stride, splits: 1, speculative };
            let blocks = Blocks { programs: self.programs, admitted: &self.admitted, cfg: &self.cfg,
                homes: &self.homes, bank: &self.bank, stream: self.stream.raw };
            for (index, layer) in self.weights.layers.iter().enumerate() {
                blocks.run(index, layer, self.planes, self.kv[index].as_ref().map(|v| v.buffer), &step)?;
                if let Some(tap) = layers.iter().position(|&layer| layer == index) {
                    let width = self.cfg.hidden * 2;
                    for (active, &row) in physical.iter().enumerate() {
                        let dst = view(self.verification.hidden.buffer, (tap * rows + row) * width, width)?;
                        let src = view(self.planes.residual, active * width, width)?;
                        // SAFETY: both admitted row slices outlive this ordered stream copy.
                        unsafe { library.copy_d2d_async(dst, src, width, self.stream.raw)?; }
                    }
                }
            }
            let norm = self.programs.program(&format!("{}_norm", self.admitted.prefix), &["x", "weight", "out"])?;
            // SAFETY: normalized active rows and head outputs match the admitted geometry.
            unsafe {
                norm.launch(&[self.planes.residual.ptr, self.weights.norm.buffer.ptr, self.planes.norm.ptr],
                    &[Scalar::I32(i32::try_from(tokens.len())?)], self.stream.raw)?;
                crate::shared::target_head::launch(TargetHead::configured(), &self.head, self.planes.norm.ptr,
                    self.weights.head.buffer.ptr, self.logits.buffer.ptr.cast(), tokens.len(), self.stream.raw)?;
            }
            let width = self.cfg.vocab * 4;
            for (active, &row) in physical.iter().enumerate() {
                let dst = view(self.verification.logits.buffer, row * width, width)?;
                let src = view(self.logits.buffer, active * width, width)?;
                // SAFETY: admitted source and physical output rows remain live through the drain.
                unsafe { library.copy_d2d_async(dst, src, width, self.stream.raw)?; }
            }
        }
        // The logits and taps carry this stream; selection and drafting wait on it.
        if speculative {
            self.verification.pending = Some(plan);
        } else {
            for group in &plan.groups { self.slots[group.slot].position += group.rows; }
            self.position = self.slots[0].position;
        }
        self.healthy = true;
        Ok(())
    }
}

impl<'a> BlockVerifyTarget for DenseEngine<'a> {
    type Mark = DenseMark<'a>;

    fn checkpoint(&mut self, slots: &[usize]) -> Result<Self::Mark> {
        ensure!(!slots.is_empty() && slots.len() <= self.slots.len(), "dense checkpoint slot count outside admission");
        for (index, &slot) in slots.iter().enumerate() {
            ensure!(slot < self.slots.len() && !slots[..index].contains(&slot), "duplicate or unadmitted dense checkpoint slot");
        }
        ensure!(self.healthy && self.verification.pending.is_none(), "dense checkpoint requires committed healthy state");
        self.verification.invalidate()?;
        let mark = DenseMark { owner: self.verification.owner, generation: self.verification.generation,
            slots: slots.iter().map(|&slot| (slot, self.slots[slot].clone())).collect(), storage: self.verification.mark.clone() };
        self.move_mark(&mark, true)?;
        Ok(mark)
    }

    fn verify<'e>(&'e mut self, mark: &Self::Mark, request: VerifyRequest<'_>) -> Result<VerifyOutput<'e>> {
        self.verification.validate(mark)?;
        let active = rows::Rows::new(&request, self.cfg.attention.len(), self.cfg.vocab, &mark.slots,
            self.cfg.max_context, self.pool_tokens);
        self.restore_mark(mark)?;
        let plan = active?;
        if let Err(error) = self.verify_rows(plan, request.rows.len(), request.tap_layers, true) {
            self.restore_mark(mark).context("restoring dense checkpoint after failed verify")?;
            return Err(error);
        }
        Ok(VerifyOutput { logits: DeviceLogits { ptr: self.verification.logits.buffer.ptr,
            rows: request.rows.len(), vocab: self.cfg.vocab, stride: self.cfg.vocab,
            stream: self.stream.raw, greedy: None }, taps: &self.verification.taps })
    }

    fn restore(&mut self, mark: &Self::Mark) -> Result<()> { self.restore_mark(mark) }

    fn commit(&mut self, mark: &Self::Mark, accepted: &[AcceptedPrefix]) -> Result<()> {
        self.verification.validate(mark)?;
        let verified = self.verification.pending.as_ref().context("dense commit requires a successful verify")?;
        let table = verified.accepted(accepted)?;
        self.restore_mark(mark)?;
        self.healthy = false;
        let result = self.bank.commit(&table, accepted.len());
        if let Err(error) = result {
            self.restore_mark(mark).context("restoring dense checkpoint after failed commit")?;
            return Err(error);
        }
        for entry in accepted { self.slots[entry.slot].position += entry.rows; }
        self.position = self.slots[0].position;
        self.verification.invalidate()?;
        self.healthy = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::speculation::target::VerifyRow;

    fn config() -> Qwen35Config {
        let value = serde_json::json!({"model_type":"qwen3_5", "text_config":{
            "model_type":"qwen3_5_text", "hidden_size":4096, "intermediate_size":12288,
            "num_hidden_layers":32, "num_attention_heads":16, "num_key_value_heads":4,
            "head_dim":256, "vocab_size":248320, "max_position_embeddings":262144,
            "full_attention_interval":4, "linear_num_key_heads":16, "linear_num_value_heads":32,
            "linear_key_head_dim":128, "linear_value_head_dim":128, "linear_conv_kernel_dim":4,
            "hidden_act":"silu", "attention_bias":false,
            "layer_types":(0..32).map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" }).collect::<Vec<_>>(),
            "rms_norm_eps":1e-6, "rope_parameters":{"rope_type":"default", "rope_theta":10000000.0,
                "partial_rotary_factor":0.25, "mrope_section":[11,11,10], "mrope_interleaved":true}}});
        Qwen35Config::from_hf(&value).unwrap()
    }

    #[test]
    fn padding_compaction_does_not_read_sentinel_metadata() {
        let cfg = config();
        let padding = VerifyRow { slot: usize::MAX, position: usize::MAX, token: None };
        let rows = [padding, VerifyRow { slot: 0, position: 17, token: Some(3) }, padding,
            VerifyRow { slot: 0, position: 18, token: Some(4) }];
        let starts = [(0, rows::Slot { position: 17, pages: vec![] })];
        let plan = rows::Rows::new(&VerifyRequest { rows: &rows, tap_layers: &[28, 4, 16] },
            cfg.attention.len(), cfg.vocab, &starts, cfg.max_context, 64).unwrap();
        assert_eq!((plan.tokens, plan.physical), (vec![3, 4], vec![1, 3]));
        assert!(rows::Rows::new(&VerifyRequest { rows: &rows, tap_layers: &[] },
            cfg.attention.len(), cfg.vocab, &starts, cfg.max_context, 18).is_err());
    }

    #[test]
    fn commit_counts_only_retained_active_inputs_and_requires_exact_slot_set() {
        let cfg = config();
        let inputs = (0..7).map(|position| VerifyRow { slot: 0, position, token: Some(3) }).collect::<Vec<_>>();
        let plan = rows::Rows::new(&VerifyRequest { rows: &inputs, tap_layers: &[] }, cfg.attention.len(), cfg.vocab,
            &[(0, rows::Slot { position: 0, pages: vec![] })], cfg.max_context, 64).unwrap();
        for keep in 0..=7 {
            assert_eq!(plan.accepted(&[AcceptedPrefix { slot: 0, rows: keep }]).unwrap(), [0, 0, keep as i32]);
        }
        for invalid in [vec![], vec![AcceptedPrefix { slot: 0, rows: 8 }],
            vec![AcceptedPrefix { slot: 1, rows: 1 }], vec![AcceptedPrefix { slot: 0, rows: 1 }; 2]] {
            assert!(plan.accepted(&invalid).is_err());
        }
    }

    #[test]
    fn verify_admission_reserves_marks_and_every_requested_tap() {
        let cfg = config();
        let layout = Verification::layout(&cfg, 1).unwrap();
        assert_eq!(layout.regions.len(), 48);
        assert_eq!(layout.bytes, 24 * cfg.gdn.state_bytes());
        assert_eq!(Verification::bytes(&cfg, 1).unwrap(), layout.bytes + 64 * cfg.vocab * 4
            + 32 * 64 * cfg.hidden * 2);
        assert_eq!(Verification::bytes(&cfg, 16).unwrap(), 16 * layout.bytes + 64 * cfg.vocab * 4
            + 32 * 64 * cfg.hidden * 2);
        for slots in [1, 4, 16, 64] {
            let native = placement::families::qwen3_5::NativeBuffers::new(&cfg, 1024, slots, 10, false).unwrap();
            assert_eq!(native.verify.iter().sum::<usize>(), Verification::bytes(&cfg, slots).unwrap());
            assert_eq!(native.verify[0], Verification::layout(&cfg, slots).unwrap().bytes);
            let gdn = GdnLayout::new(cfg.gdn).unwrap();
            assert_eq!(native.state_per_layer, gdn.conv_slot_bytes + gdn.state_slot_bytes);
            assert_eq!(native.replay_per_layer, gdn.replay_layer_bytes);
        }
    }
}
