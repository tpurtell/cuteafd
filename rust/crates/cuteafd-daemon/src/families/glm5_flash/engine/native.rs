//! Flash stage composition over the shared native EXLR8 owners and linears.
use super::*;
use crate::shared::{experts::exlr8::{self, dense_execution::{NativeOut, NativePrograms, ScratchOwner},
    dense_weights::key, NativeExperts}, memory::device::{Allocation, Device}};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, F32Bf16Narrow,
    glm_stage::{Buffers as KvBuffers, GlmStage}, projected_index::{self, ProjectedIndex, ProjectedKeys},
    projected_kda::{self, ProjectedKda}};
use cuteafd_loader::plan::exlr8::native::{FlashCharges, FlashRankLayout, FlashStageLayout};
use std::collections::{BTreeMap, BTreeSet};

// GlmfIndexPost and GlmfIndexPostRows in b12x/integration/cuteafd/_glmf_kernels.py
// consume [wk128 | weights32 | gate128], not [wk | gate | weights].
const INDEX_KEY_WIDTH: usize = 128;
const INDEX_WEIGHT_COLUMN: usize = INDEX_KEY_WIDTH;
const INDEX_GATE_COLUMN: usize = INDEX_WEIGHT_COLUMN + projected_index::HEADS;
const INDEX_PACKED_WIDTH: usize = INDEX_GATE_COLUMN + INDEX_KEY_WIDTH;
const INDEX_PACK: [(&str, usize, usize); 3] = [
    ("index_k", INDEX_KEY_WIDTH, 0),
    ("index_w", projected_index::HEADS, INDEX_WEIGHT_COLUMN),
    ("index_gate", INDEX_KEY_WIDTH, INDEX_GATE_COLUMN),
];

pub(super) fn is_native(layer: &GlmfLayer<'_>) -> bool { !layer.native_projections.is_empty() }
pub(super) fn dense_ffn(layer: &GlmfLayer<'_>) -> bool {
    layer.dense && layer.native_projections.contains_key("down")
}

struct Scratch<'a> {
    linear: ScratchOwner<'a>,
    stages: BTreeMap<&'static str, Allocation<'a>>,
}
impl<'a> Scratch<'a> {
    fn new(device: Device<'a>, linear: &cuteafd_loader::plan::exlr8::dense::scratch::LinearScratchLayout,
        stages: &FlashStageLayout) -> Result<Self> {
        Ok(Self { linear: ScratchOwner::new(device, linear)?, stages: stages.extents.iter()
            .map(|(&name, &bytes)| Ok((name, Allocation::new(device, bytes)?))).collect::<Result<_>>()? })
    }
    fn stage(&self, name: &str) -> Result<Buffer> {
        Ok(self.stages.get(name).with_context(|| format!("native Flash stage {name}"))?.buffer)
    }
}

/// Only two scratch owners per rank; every prefill lane borrows the second.
pub(super) struct Slot<'a> { scratch: Rc<Scratch<'a>> }
pub(super) struct Rank<'a> {
    programs: NativePrograms<'a>,
    kda: BTreeMap<String, ProjectedKda<'a>>,
    compact: BTreeMap<String, ProjectedIndex<'a>>,
    keys: BTreeMap<String, ProjectedKeys<'a>>,
    kv: BTreeMap<String, GlmStage<'a>>,
    scratch: [Rc<Scratch<'a>>; 2],
    narrow: F32Bf16Narrow<'a>,
    shared: NativeExperts<'a>,
    shared_execution: exlr8::Exlr8Execution<'a>,
}
impl<'a> Rank<'a> {
    fn new(engine: &GlmfEngine<'a>, rank: usize, manifest: &serde_json::Value,
        selected: &BTreeSet<String>, layout: &FlashRankLayout, shared: NativeExperts<'a>) -> Result<Self> {
        let device = Device { library: engine.library, id: engine.device_of(rank)? };
        let dense = selected.iter().filter(|n| n.starts_with("exlr8_dense_")).cloned().collect();
        let programs = NativePrograms::bind(engine.programs, manifest, &dense)?;
        let (mut kda, mut compact, mut keys, mut kv) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        for name in selected {
            if name.contains("_kda_projected_") { kda.insert(name.clone(), ProjectedKda::bind(engine.programs, name)?); }
            else if name.contains("_index_projected_c_") { compact.insert(name.clone(), ProjectedIndex::bind(engine.programs, name)?); }
            else if name.contains("_index_projected_keys_") { keys.insert(name.clone(), ProjectedKeys::bind(engine.programs, name)?); }
            else if name.contains("_absorb_fp8_") || name.contains("_values_fp8_") {
                kv.insert(name.clone(), GlmStage::bind(engine.programs, name)?);
            }
        }
        let shared_execution = shared.execution(engine.library, engine.programs)?;
        ensure!(layout.shared_workspace == exlr8::Exlr8Execution::shared_workspace_bytes(engine.programs,
            cuteafd_ffi::exlr8::KernelFamily::GlmFlash, cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS)? as u64,
            "native Flash shared workspace differs from admission");
        Ok(Self { programs, kda, compact, keys, kv, shared, shared_execution,
            scratch: [Rc::new(Scratch::new(device, &layout.linear[0], &layout.stages[0])?),
                Rc::new(Scratch::new(device, &layout.linear[1], &layout.stages[1])?)],
            narrow: engine.library.f32_bf16_narrow()? })
    }
}

fn rows(value: Scalar) -> Result<usize> {
    match value { Scalar::I32(n) if n > 0 => Ok(n as usize), _ => anyhow::bail!("native Flash positive row count") }
}
pub(super) fn sequence_count(slots: &[i32]) -> usize {
    // Masked graph rows have negative slots; an all-masked warm-up still validates one sequence.
    slots.iter().copied().filter(|slot| *slot >= 0).collect::<BTreeSet<_>>().len().max(1)
}
fn matrix(buffer: Buffer, first: usize, rows: usize, columns: usize, width: usize) -> Result<Buffer> {
    let stride = columns.checked_mul(width).context("native Flash row overflow")?;
    exlr8::view(buffer, first.checked_mul(stride).context("native Flash offset overflow")?,
        rows.checked_mul(stride).context("native Flash extent overflow")?)
}
fn stage_name(split: bool, stage: &str, cap: &str) -> String {
    format!("{}_stage_{stage}_{cap}", if split { "glmf2" } else { "glmf" })
}

impl<'a> GlmfEngine<'a> {
    pub(crate) fn bind_native(&mut self, manifest: &serde_json::Value, admission: &FlashCharges,
        shared: Vec<NativeExperts<'a>>) -> Result<()> {
        ensure!(self.native.is_empty() && admission.ranks.len() == self.ranks() && shared.len() == self.ranks(),
            "native Flash binding rank count or duplicate binding");
        ensure!(self.kda_state == KdaState::F32 && !self.kda_fp32_partials && !self.kda_output_shard,
            "native Flash uses F32 recurrent state and BF16 rank partials");
        for (rank, shared) in shared.into_iter().enumerate() {
            let layers = if rank == 0 { &self.weights.layers } else { &self.peer()?.layers };
            ensure!(layers.iter().all(is_native), "native Flash mixed layer representations");
            let value = self.on(rank, || Rank::new(self, rank, manifest, &admission.programs,
                &admission.ranks[rank], shared))?;
            self.native.push(value);
        }
        Ok(())
    }
    pub(super) fn native_slot(&self, rank: usize, w: &mut Workspace<'a>, decode: bool) -> Result<()> {
        if let Some(rank) = self.native.get(rank) {
            w.native = Some(Slot { scratch: rank.scratch[usize::from(!decode)].clone() });
        }
        Ok(())
    }
    fn device_of(&self, rank: usize) -> Result<i32> { Ok(if rank == 0 { self.device } else { self.peer()?.device }) }
    fn native_rank(&self, rank: usize) -> Result<&Rank<'a>> { self.native.get(rank).context("native Flash rank not bound") }
    fn native_scratch<'w, 'b>(&self, w: &'w Workspace<'b>) -> Result<&'w Scratch<'b>> {
        // Scratch owners are held by the workspace, never borrowed from a temporary binding.
        w.native.as_ref().map(|slot| slot.scratch.as_ref()).context("native Flash workspace not bound")
    }
    fn native_linear(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, projection: &str,
        input: Buffer, output: Buffer, count: usize) -> Result<()> {
        let execution = self.native_rank(rank)?;
        let scratch = self.native_scratch(w)?;
        let geometry = layer.native_projections.get(projection).context("native Flash projection geometry")?;
        self.on(rank, || {
            let out = execution.programs.linear(geometry, |component| layer.native(&key(projection, component)),
                scratch.linear.view(), input, count, output, false, self.stream_of(rank))?;
            if let NativeOut::F32 { n } = out {
                let acc = matrix(scratch.linear.view().acc, 0, count, n, 4)?;
                let output = matrix(output, 0, count, n, 2)?;
                // SAFETY: checked disjoint accumulator/output on this rank, retained by the lane.
                unsafe { execution.narrow.launch(acc.ptr, output.ptr, count * n, self.stream_of(rank))?; }
            }
            Ok(())
        })
    }
    fn native_stage(&self, rank: usize, split: bool, stage: &str, cap: &str,
        pointers: &[(&str, *mut c_void)], count: Scalar) -> Result<()> {
        // SAFETY: the admitted Flash stage ABI and retained stage/weight buffers match these operands.
        self.on(rank, || unsafe { self.programs.program(&stage_name(split, stage, cap),
            &pointers.iter().map(|p| p.0).collect::<Vec<_>>())?.launch(
                &pointers.iter().map(|p| p.1).collect::<Vec<_>>(), &[count], self.stream_of(rank)) })
    }
    fn native_pack(&self, rank: usize, w: &Workspace<'_>, output: Buffer,
        count: usize, width: usize, pitch: usize, column: usize) -> Result<()> {
        let scratch = self.native_scratch(w)?;
        let acc = matrix(scratch.linear.view().acc, 0, count, width, 4)?;
        let output = matrix(output, 0, count, pitch, 2)?;
        ensure!(column.checked_add(width).is_some_and(|end| end <= pitch), "native Flash pack columns");
        let name = format!("exlr8_dense_f32_narrow_pack_n{width}_p{pitch}");
        // SAFETY: the selected companion and checked views implement this packed row geometry.
        self.on(rank, || unsafe { self.native_rank(rank)?.programs.companion(&name)?.launch(
            &[acc.ptr, output.ptr], count.try_into()?, column.try_into()?, self.stream_of(rank))?; Ok(()) })
    }
    fn native_bf16(&self, rank: usize, w: &Workspace<'_>, input: Buffer, weights: Buffer,
        count: usize, outputs: usize) -> Result<()> {
        let input = matrix(input, 0, count, self.cfg.hidden, 2)?;
        let weights = matrix(weights, 0, outputs, self.cfg.hidden, 2)?;
        let acc = matrix(self.native_scratch(w)?.linear.view().acc, 0, count, outputs, 4)?;
        let name = format!("exlr8_dense_bf16_projection_k{}_n{outputs}", self.cfg.hidden);
        // SAFETY: native BF16 weights and caller-owned disjoint accumulator have checked extents.
        self.on(rank, || unsafe { self.native_rank(rank)?.programs.companion(&name)?.launch(
            &[input.ptr, weights.ptr, acc.ptr], count.try_into()?, 0, self.stream_of(rank))?; Ok(()) })
    }
    pub(super) fn native_ffn(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, cap: &str,
        output: *mut c_void, count: Scalar) -> Result<()> {
        let count = rows(count)?;
        let output = [w.delta.buffer, w.shared.buffer].into_iter().find(|b| b.ptr == output)
            .context("native Flash FFN output is not lane-owned")?;
        if !layer.dense {
            let index = if rank == 0 { &self.weights.layers } else { &self.peer()?.layers }
                .iter().position(|candidate| std::ptr::eq(candidate, layer)).context("native Flash shared layer")?;
            let execution = self.native_rank(rank)?;
            return self.on(rank, || execution.shared.shared(self.library, index, &execution.shared_execution,
                w.x.buffer, output, count, self.stream_of(rank)));
        }
        let scratch = self.native_scratch(w)?;
        let (gate, up, hidden, temp) = (scratch.stage("gate")?, scratch.stage("up")?, scratch.stage("hidden")?, scratch.stage("scratch")?);
        self.native_linear(rank, w, layer, "gate", w.x.buffer, gate, count)?;
        self.native_linear(rank, w, layer, "up", w.x.buffer, up, count)?;
        let inter = layer.native_projections["gate"].shape[0];
        self.native_stage(rank, layer.split, &format!("swiglu_i{inter}"), cap,
            &[("gate", gate.ptr), ("up", up.ptr), ("hidden", hidden.ptr), ("scratch", temp.ptr)], Scalar::I32(count.try_into()?))?;
        self.native_linear(rank, w, layer, "down", hidden, output, count)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn native_kda(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar,
        first: usize, cap: &str, speculative: bool, sequences: usize, [conv_state, state, replay]: [Buffer; 3]) -> Result<()> {
        let count = rows(count)?;
        let scratch = self.native_scratch(w)?;
        let input = matrix(w.x.buffer, first, count, self.cfg.hidden, 2)?;
        let (q, k, v, decay, gate, beta, aux, output) = (scratch.stage("q")?, scratch.stage("k")?, scratch.stage("v")?,
            scratch.stage("decay")?, scratch.stage("output_gate")?, scratch.stage("beta")?, scratch.stage("aux")?, scratch.stage("values")?);
        for (projection, output) in [("q", q), ("k", k), ("v", v), ("beta", beta)] {
            self.native_linear(rank, w, layer, projection, input, output, count)?;
        }
        for (a, b, output) in [("f_a", "f_b", decay), ("g_a", "g_b", gate)] {
            self.native_linear(rank, w, layer, a, input, aux, count)?;
            self.native_linear(rank, w, layer, b, aux, output, count)?;
        }
        let name = stage_name(layer.attention_share.split(), "kda_projected", cap);
        let stage = self.native_rank(rank)?.kda.get(&name).context("native Flash projected KDA")?;
        let weights = projected_kda::Weights { convolution: layer.buffer("conv_w")?, a_log: layer.buffer("a_log")?,
            dt_bias: layer.buffer("dt_bias")?, norm: layer.buffer("o_norm")? };
        let batch = projected_kda::Batch { rows: count, sequences,
            state_slots: self.slots, speculative, q, k, v, decay, gate, beta, conv_state, state,
            slots: matrix(w.kda_slots.buffer, first, count, 1, 4)?, first: matrix(w.seq_first.buffer, first, count, 1, 4)?,
            output, replay, scratch: scratch.stage("scratch")? };
        // SAFETY: the step's validated slot/sequence tables address the live caches; every view is retained.
        self.on(rank, || unsafe { stage.launch(weights, batch, self.stream_of(rank))?; Ok(()) })?;
        self.native_linear(rank, w, layer, "o", output, matrix(w.delta.buffer, first, count, self.cfg.hidden, 2)?, count)
    }
    pub(super) fn native_mla_producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar,
        cap: &str, cache: Buffer) -> Result<()> {
        let n = rows(count)?;
        let scratch = self.native_scratch(w)?;
        let (q, k, temp) = (scratch.stage("q")?, scratch.stage("k")?, scratch.stage("scratch")?);
        self.native_linear(rank, w, layer, "q_a", w.x.buffer, q, n)?;
        self.native_linear(rank, w, layer, "kv_a", w.x.buffer, k, n)?;
        self.native_stage(rank, layer.attention_share.split(), "rank_norm", cap, &[("q_a", q.ptr), ("kv_a", k.ptr),
            ("q_a_norm", layer.ptr("q_a_norm")?), ("kv_a_norm", layer.ptr("kv_a_norm")?),
            ("kv_slots", w.kv_slots.buffer.ptr), ("q_resid", w.q_resid.buffer.ptr), ("kv_cache", cache.ptr), ("scratch", temp.ptr)], count)?;
        self.native_linear(rank, w, layer, "q_b", w.q_resid.buffer, q, n)?;
        self.native_kv(rank, layer, "absorb", cap, q, w.query.buffer, n)
    }
    fn native_kv(&self, rank: usize, layer: &GlmfLayer<'_>, kind: &str, cap: &str,
        input: Buffer, output: Buffer, rows: usize) -> Result<()> {
        let name = stage_name(layer.attention_share.split(), &format!("{kind}_fp8"), cap);
        let stage = self.native_rank(rank)?.kv.get(&name).context("native Flash FP8 KV stage")?;
        let prefix = if kind == "absorb" { "w_uk" } else { "w_uv" };
        let weights = layer.buffer(&format!("{prefix}_fp8"))?;
        let scales = layer.buffer(&format!("{prefix}_scale"))?;
        // SAFETY: shared absorption owns whole head slices; their admitted stage inputs/outputs remain live.
        self.on(rank, || unsafe { stage.launch(KvBuffers::Kv { a: input.ptr, w: weights.ptr,
            scale: scales.ptr, out: output.ptr }, rows.try_into()?, self.stream_of(rank))?; Ok(()) })
    }
    pub(super) fn native_mla_output(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar,
        cap: &str) -> Result<()> {
        let count = rows(count)?;
        let values = self.native_scratch(w)?.stage("values")?;
        self.native_kv(rank, layer, "values", cap, w.latent.buffer, values, count)?;
        self.native_linear(rank, w, layer, "o", values, w.delta.buffer, count)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn native_index(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, count: Scalar,
        cap: &str, tables: &StepTables, first: usize, pools: Buffer) -> Result<()> {
        let count = rows(count)?;
        let scratch = self.native_scratch(w)?;
        let (query, packed) = (scratch.stage("index_query")?, scratch.stage("index_packed")?);
        let input = matrix(w.x.buffer, first, count, self.cfg.hidden, 2)?;
        let q_resid = matrix(w.q_resid.buffer, first, count, self.cfg.q_lora_rank, 2)?;
        self.native_linear(rank, w, layer, "index_q", q_resid, query, count)?;
        for (projection, width, column) in INDEX_PACK {
            if projection == "index_k" {
                let geometry = layer.native_projections.get(projection).context("native Flash index K geometry")?;
                self.on(rank, || {
                    ensure!(matches!(self.native_rank(rank)?.programs.linear(geometry,
                        |component| layer.native(&key(projection, component)), scratch.linear.view(), input, count,
                        packed, false, self.stream_of(rank))?, NativeOut::F32 { n } if n == width),
                        "native Flash index K must remain MXFP8");
                    Ok(())
                })?;
            } else {
                let weights = if projection == "index_gate" { layer.buffer("index_gate")? }
                    else { layer.native(&key(projection, "bf16"))? };
                self.native_bf16(rank, w, input, weights, count, width)?;
            }
            self.native_pack(rank, w, packed, count, width, INDEX_PACKED_WIDTH, column)?;
        }
        let weights = projected_index::Weights { norm: layer.buffer("k_norm_w")?, bias: layer.buffer("k_norm_b")?, ape: layer.buffer("ape")? };
        let caches = self.caches_of(rank);
        let cache = caches.index[index].as_ref().context("native Flash index cache")?;
        let query_fp8 = matrix(w.q_fp8.buffer, first, count, INDEX_HEADS * 128, 1)?;
        let head_weights = matrix(w.head_weights.buffer, first, count, INDEX_HEADS, 4)?;
        let pages = pools.bytes / 8448;
        let pool_slots = matrix(w.pool_slots.buffer, first, count, 1, 8)?;
        let prefix = stage_name(layer.attention_share.split(), "index_projected", cap);
        if let Some(keys) = &cache.keys {
            let name = prefix.replace("index_projected_", "index_projected_keys_");
            let stage = self.native_rank(rank)?.keys.get(&name).context("native Flash token-key stage")?;
            let batch = projected_index::KeysBatch { rows: count, record_slots: keys.buffer.bytes / KEY_BYTES,
                pool_pages: pages, query, packed, slots: matrix(w.kv_slots.buffer, first, count, 1, 8)?, pool_slots,
                token_keys: keys.buffer, cache: pools, query_fp8, head_weights };
            // SAFETY: admitted token slots/pool completion tables refer to initialized records in this cache.
            self.on(rank, || unsafe { stage.launch(weights, batch, self.stream_of(rank))?; Ok(()) })
        } else {
            let name = prefix.replace("index_projected_", "index_projected_c_");
            let stage = self.native_rank(rank)?.compact.get(&name).context("native Flash compact index stage")?;
            let ordinal = caches.mla_ordinal[index].context("native Flash index ordinal")?;
            let (tails, records) = caches.index_tails.as_ref().context("native Flash compact tails")?;
            let recorded = replay_rows_of(cap);
            let batch = projected_index::Batch { rows: count, state_slots: self.slots, pool_pages: pages,
                speculative: tables.spec, query, packed, pool_slots,
                positions: matrix(w.positions.buffer, first, count, 1, 8)?,
                kda_slots: matrix(w.kda_slots.buffer, first, count, 1, 4)?, first: matrix(w.seq_first.buffer, first, count, 1, 4)?,
                tails: exlr8::view(tails.buffer, ordinal * self.slots * TAIL_BYTES, self.slots * TAIL_BYTES)?,
                replay: exlr8::view(records.buffer, ordinal * recorded * KEY_BYTES, recorded * KEY_BYTES)?,
                cache: pools, query_fp8, head_weights, scratch: scratch.stage("scratch")? };
            // SAFETY: existing step tables address this layer's admitted tails/pages; replay is committed by the engine.
            self.on(rank, || unsafe { stage.launch(weights, batch, self.stream_of(rank))?; Ok(()) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_index_pack_matches_consumer_column_ranges() {
        let rows = 3;
        let mut packed = vec![u16::MAX; rows * INDEX_PACKED_WIDTH];
        for (projection, width, column) in INDEX_PACK {
            let base = match projection { "index_k" => 0, "index_w" => 1024, "index_gate" => 2048,
                _ => panic!("unknown index projection") };
            let source: Vec<u16> = (0..rows * width).map(|i| (base + i) as u16).collect();
            for row in 0..rows {
                let target = row * INDEX_PACKED_WIDTH + column;
                packed[target..target + width].copy_from_slice(&source[row * width..(row + 1) * width]);
            }
        }
        // Independently mirror both consumers' key, head-weight and gate reads.
        for row in 0..rows {
            let values = &packed[row * INDEX_PACKED_WIDTH..(row + 1) * INDEX_PACKED_WIDTH];
            assert_eq!(values.len(), 288);
            for channel in 0..128 {
                assert_eq!(values[channel], (row * 128 + channel) as u16);
                assert_eq!(values[128 + 32 + channel], (2048 + row * 128 + channel) as u16);
            }
            for head in 0..32 {
                assert_eq!(values[128 + head], (1024 + row * 32 + head) as u16);
            }
            assert!(!values.contains(&u16::MAX));
        }
    }

    #[test]
    fn projected_sequences_count_live_slots_not_verify_rows() {
        assert_eq!(sequence_count(&[3, 3, 3, 7, 7, -1, -1]), 2);
        assert_eq!(sequence_count(&[-1; 64]), 1);
        assert_eq!(sequence_count(&[0; 64]), 1);
    }

    #[test]
    fn projected_subrows_retain_real_allocation_bounds() {
        let allocation = Buffer { ptr: 0x10000usize as *mut c_void, bytes: 8 * 4096 * 2, device_id: 0, flags: 0 };
        let view = matrix(allocation, 3, 2, 4096, 2).unwrap();
        assert_eq!(view.ptr as usize, 0x10000 + 3 * 4096 * 2);
        assert_eq!(view.bytes, 2 * 4096 * 2);
        assert_eq!(view.device_id, allocation.device_id);
        assert!(matrix(allocation, 7, 2, 4096, 2).is_err());
        assert!(matrix(allocation, usize::MAX, 1, 4096, 2).is_err());
        assert!(matrix(allocation, 0, usize::MAX, 4096, 2).is_err());
    }
}
