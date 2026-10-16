//! Single-slot implementation of the shared, slot-indexed block verification contract.
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
    position: usize,
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
    pub(super) pending: Option<usize>,
}

impl<'a> Verification<'a> {
    fn layout(cfg: &Qwen35Config) -> Result<MarkLayout> {
        let gdn = GdnLayout::new(cfg.gdn)?;
        let regions = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn)
            .flat_map(|_| [(0, gdn.conv_slot_bytes), (0, gdn.state_slot_bytes)]).collect::<Vec<_>>();
        MarkLayout::new(&regions)
    }

    pub(super) fn bytes(cfg: &Qwen35Config) -> Result<usize> {
        let mark = Self::layout(cfg)?.bytes.max(256);
        let logits = REPLAY_ROWS.checked_mul(cfg.vocab).and_then(|n| n.checked_mul(4))
            .context("dense verify logits size overflow")?;
        let taps = REPLAY_ROWS.checked_mul(cfg.hidden).and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(cfg.attention.len())).context("dense verify taps size overflow")?;
        mark.checked_add(logits).and_then(|n| n.checked_add(taps)).context("dense verify admission overflow")
    }

    pub(super) fn new(device: Device<'a>, cfg: &Qwen35Config) -> Result<Self> {
        let mut layout = Self::layout(cfg)?;
        for owner in &mut layout.owners { owner.0 = device.id; }
        let mark = Rc::new(Allocation::new(device, layout.bytes.max(256))?);
        let logits = Allocation::new(device, REPLAY_ROWS * cfg.vocab * 4)?;
        let hidden = Allocation::new(device, cfg.attention.len() * REPLAY_ROWS * cfg.hidden * 2)?;
        let owner = NEXT_ENGINE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow::anyhow!("dense target identity exhausted"))?;
        Ok(Self { owner, generation: 0, mark, layout, logits, hidden, taps: Vec::new(), pending: None })
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

/// Only active tokens enter state, rotary and page tables. The mapping is the
/// entire scatter policy; sentinel padding slot/position fields are never read.
fn accepted_rows(accepted: &[AcceptedPrefix], verified: usize) -> Result<usize> {
    ensure!(accepted.len() == 1 && accepted[0].slot == 0 && accepted[0].rows <= verified,
        "dense accepted prefix outside verified slot/input rows");
    Ok(accepted[0].rows)
}

fn active_rows(request: &VerifyRequest<'_>, cfg: &Qwen35Config, position: usize, pool: usize)
    -> Result<(Vec<u32>, Vec<usize>)> {
    ensure!((1..=REPLAY_ROWS).contains(&request.rows.len()), "dense verify needs 1..64 physical rows");
    request.validate(cfg.attention.len(), cfg.vocab, &[(0, position)])?;
    let mut tokens = Vec::new();
    let mut physical = Vec::new();
    for (index, row) in request.rows.iter().enumerate() {
        if let Some(token) = row.token { tokens.push(token); physical.push(index); }
    }
    let end = position.checked_add(tokens.len()).context("dense verify position overflow")?;
    ensure!(end <= cfg.max_context && end <= pool, "dense verify exceeds admitted cache/context");
    Ok((tokens, physical))
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

    fn mark_regions(&self) -> Vec<CuteafdDeviceBuffer> {
        self.homes.layers.iter().filter_map(|home| home.gdn_ordinal)
            .flat_map(|ordinal| self.bank.slot_regions(ordinal, 0)).collect()
    }

    fn move_mark(&self, mark: &DenseMark<'_>, capture: bool) -> Result<()> {
        let regions = self.mark_regions();
        ensure!(regions.len() == self.verification.layout.regions.len(), "dense mark geometry changed");
        for (region, &(owner, offset, bytes)) in regions.into_iter().zip(&self.verification.layout.regions) {
            ensure!(owner == 0 && bytes == region.bytes && region.device_id == mark.storage.buffer.device_id,
                "dense mark region does not match its owner");
            let saved = view(mark.storage.buffer, offset, bytes)?;
            let (dst, src) = if capture { (saved, region) } else { (region, saved) };
            // SAFETY: mark storage and the live bank are retained until the enclosing drain.
            unsafe { self.stream.device.library.copy_d2d_async(dst, src, bytes, self.stream.raw)?; }
        }
        self.stream.drain()
    }

    fn restore_mark(&mut self, mark: &DenseMark<'_>) -> Result<()> {
        self.verification.validate(mark)?;
        self.stream.drain()?;
        self.healthy = false;
        self.move_mark(mark, false)?;
        // KV is append-only with an immutable identity page map. Rows at or past
        // this frontier are unreachable, and every subsequent input overwrites them.
        self.position = mark.position;
        self.verification.pending = None;
        self.verification.taps.clear();
        self.healthy = true;
        Ok(())
    }

    fn verify_rows(&mut self, tokens: &[u32], physical: &[usize], rows: usize, layers: &[usize]) -> Result<()> {
        self.stream.drain()?;
        self.healthy = false;
        let library = self.stream.device.library;
        // SAFETY: output arenas are admitted and no prior consumer remains after the drain.
        unsafe {
            library.cuda_zero_bytes_async(self.verification.logits.buffer, rows * self.cfg.vocab * 4, self.stream.raw)?;
            library.cuda_zero_bytes_async(self.verification.hidden.buffer, layers.len() * rows * self.cfg.hidden * 2,
                self.stream.raw)?;
        }
        self.verification.taps = layers.iter().enumerate().map(|(index, &layer)| HiddenTap {
            layer, ptr: self.verification.hidden.buffer.ptr.cast::<u8>().wrapping_add(index * rows * self.cfg.hidden * 2).cast(),
            rows, hidden: self.cfg.hidden, stride: self.cfg.hidden, stream: self.stream.raw }).collect();
        if !tokens.is_empty() {
            self.tables(tokens.len())?;
            self.embedding.embed(tokens, self.ids.buffer, 1, self.planes.residual, self.stream.raw)?;
            let step = Step { rows: tokens.len(), capacity: REPLAY_ROWS, table_stride: 0, splits: 1, speculative: true };
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
        self.stream.drain()?;
        self.verification.pending = Some(tokens.len());
        self.healthy = true;
        Ok(())
    }
}

impl<'a> BlockVerifyTarget for DenseEngine<'a> {
    type Mark = DenseMark<'a>;

    fn checkpoint(&mut self, slots: &[usize]) -> Result<Self::Mark> {
        ensure!(slots == [0], "dense target currently admits exactly slot 0");
        ensure!(self.healthy && self.verification.pending.is_none(), "dense checkpoint requires committed healthy state");
        self.stream.drain()?;
        self.verification.invalidate()?;
        let mark = DenseMark { owner: self.verification.owner, generation: self.verification.generation,
            position: self.position, storage: self.verification.mark.clone() };
        self.move_mark(&mark, true)?;
        Ok(mark)
    }

    fn verify<'e>(&'e mut self, mark: &Self::Mark, request: VerifyRequest<'_>) -> Result<VerifyOutput<'e>> {
        self.verification.validate(mark)?;
        let active = active_rows(&request, &self.cfg, mark.position, self.pool_tokens);
        self.restore_mark(mark)?;
        let (tokens, physical) = active?;
        if let Err(error) = self.verify_rows(&tokens, &physical, request.rows.len(), request.tap_layers) {
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
        let verified = self.verification.pending.context("dense commit requires a successful verify")?;
        let keep = accepted_rows(accepted, verified)?;
        self.restore_mark(mark)?;
        self.healthy = false;
        let result = self.bank.commit(&[0, 0, i32::try_from(keep)?], 1).and_then(|()| self.stream.drain());
        if let Err(error) = result {
            self.restore_mark(mark).context("restoring dense checkpoint after failed commit")?;
            return Err(error);
        }
        self.position = mark.position + keep;
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
        let rows = [VerifyRow { slot: usize::MAX, position: usize::MAX, token: None },
            VerifyRow { slot: 0, position: 17, token: Some(3) },
            VerifyRow { slot: usize::MAX, position: usize::MAX, token: None },
            VerifyRow { slot: 0, position: 18, token: Some(4) }];
        let request = VerifyRequest { rows: &rows, tap_layers: &[28, 4, 16] };
        assert_eq!(active_rows(&request, &cfg, 17, 64).unwrap(), (vec![3, 4], vec![1, 3]));
        assert!(active_rows(&request, &cfg, 17, 18).is_err());
        assert!(active_rows(&request, &cfg, 18, 64).is_err());
        let padding = [rows[0]; REPLAY_ROWS];
        assert_eq!(active_rows(&VerifyRequest { rows: &padding, tap_layers: &[] }, &cfg, 17, 64).unwrap(),
            (vec![], vec![]));
        assert!(active_rows(&VerifyRequest { rows: &[rows[0]; REPLAY_ROWS + 1], tap_layers: &[] }, &cfg, 17, 64).is_err());
    }

    #[test]
    fn commit_counts_only_retained_active_inputs_and_requires_exact_slot_set() {
        for keep in 0..=7 {
            assert_eq!(accepted_rows(&[AcceptedPrefix { slot: 0, rows: keep }], 7).unwrap(), keep);
        }
        for invalid in [vec![], vec![AcceptedPrefix { slot: 0, rows: 8 }],
            vec![AcceptedPrefix { slot: 1, rows: 1 }],
            vec![AcceptedPrefix { slot: 0, rows: 1 }; 2]] {
            assert!(accepted_rows(&invalid, 7).is_err());
        }
    }

    #[test]
    fn verify_admission_reserves_marks_and_every_requested_tap() {
        let cfg = config();
        let layout = Verification::layout(&cfg).unwrap();
        assert_eq!(layout.regions.len(), 48);
        assert_eq!(layout.bytes, 24 * cfg.gdn.state_bytes());
        assert_eq!(Verification::bytes(&cfg).unwrap(), layout.bytes + 64 * cfg.vocab * 4
            + 32 * 64 * cfg.hidden * 2);
    }
}
