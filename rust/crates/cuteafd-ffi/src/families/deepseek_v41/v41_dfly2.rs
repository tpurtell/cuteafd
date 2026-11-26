use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct V41Dfly2Request {
    pub window_values: *const u8,
    pub window_scales: *const u8,
    pub source_values: *const u8,
    pub source_scales: *const u8,
    pub source_pages: *const u32,
    pub selected: *const i32,
    pub window_begin: u64,
    pub committed_end: u64,
    pub source_capacity: u64,
    pub page_stride: u32,
    pub source_fp4: u32,
}
const _: [(); 80] = [(); std::mem::size_of::<V41Dfly2Request>()];

type AttentionInit = unsafe extern "C" fn() -> i32;
type AttentionLaunch = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const f32,
    *const V41Dfly2Request,
    *mut u16,
    *mut u8,
    *mut f32,
    u64,
    i32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type EmbedBlock = unsafe extern "C" fn(
    *const u16,
    *const u32,
    *mut u16,
    *mut u16,
    i32,
    i32,
    u32,
    *mut c_void,
) -> i32;
type EmbedTokens = unsafe extern "C" fn(*const u16, *const u32, *mut u16, i32, *mut c_void) -> i32;
type ConcatBf16 =
    unsafe extern "C" fn(*const u16, *const u16, *mut u16, i32, i32, i32, *mut c_void) -> i32;
type MtpFuseSeed = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const u16,
    *const u64,
    *mut u16,
    *mut f32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type SwiGlu =
    unsafe extern "C" fn(*const u16, *const u16, *mut u16, i32, i32, f32, *mut c_void) -> i32;
type Add2Bf16 = unsafe extern "C" fn(*const u16, *const u16, *mut u16, u64, *mut c_void) -> i32;
type Add3Bf16 =
    unsafe extern "C" fn(*const u16, *const u16, *const u16, *mut u16, u64, *mut c_void) -> i32;
type SliceSlot =
    unsafe extern "C" fn(*const u16, *mut u16, i32, i32, i32, i32, *mut c_void) -> i32;
type ScatterKvSlot =
    unsafe extern "C" fn(*const u16, *mut u16, i32, i32, i32, *mut c_void) -> i32;
type BlockGatedAdd = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const u16,
    *const u8,
    *const u16,
    *const f32,
    *const f32,
    *const f32,
    *mut u16,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type ShardStats = unsafe extern "C" fn(
    *const f32,
    *mut c_void,
    *mut u32,
    *mut f32,
    *mut f32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type FinalizeSlot = unsafe extern "C" fn(
    *const u32,
    *const f32,
    *mut u32,
    *mut u32,
    *mut f32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type MergeStats = unsafe extern "C" fn(
    *const u32,
    *const f32,
    *const f32,
    *const u32,
    *const f32,
    *const f32,
    *mut u32,
    *mut u32,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;

pub struct V41Dfly2Ops<'a> {
    library: &'a NativeLibrary,
    attention: AttentionLaunch,
    embed_block: EmbedBlock,
    embed_tokens: EmbedTokens,
    concat_bf16: ConcatBf16,
    mtp_fuse_and_seed_hc: MtpFuseSeed,
    swiglu: SwiGlu,
    add2_bf16: Add2Bf16,
    add3_bf16: Add3Bf16,
    slice_slot: SliceSlot,
    scatter_kv_slot: ScatterKvSlot,
    block_gated_add: BlockGatedAdd,
    shard_stats: ShardStats,
    finalize_slot: FinalizeSlot,
    merge_stats: MergeStats,
}

fn check_buf(buf: CuteafdDeviceBuffer, bytes: usize, name: &str) -> Result<()> {
    ensure!(!buf.ptr.is_null() && buf.bytes >= bytes, "invalid DFly2 buffer: {name}");
    Ok(())
}

impl NativeLibrary {
    pub fn v41_dfly2_ops(&self) -> Result<V41Dfly2Ops<'_>> {
        let init: AttentionInit =
            unsafe { *self.lib.get(b"cuteafd_v41_dfly2_attention_initialize")? };
        let status = unsafe { init() };
        self.status_to_result("DFly2 attention initialization", status)?;
        Ok(V41Dfly2Ops {
            library: self,
            attention: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_attention")? },
            embed_block: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_embed_block")? },
            embed_tokens: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_embed_tokens")? },
            concat_bf16: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_concat_bf16")? },
            mtp_fuse_and_seed_hc: unsafe {
                *self.lib.get(b"cuteafd_v41_dfly2_mtp_fuse_and_seed_hc")?
            },
            swiglu: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_swiglu")? },
            add2_bf16: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_add2_bf16")? },
            add3_bf16: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_add3_bf16")? },
            slice_slot: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_slice_slot")? },
            scatter_kv_slot: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_scatter_kv_slot")? },
            block_gated_add: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_block_gated_add")? },
            shard_stats: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_shard_stats_f32")? },
            finalize_slot: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_finalize_slot_f32")? },
            merge_stats: unsafe { *self.lib.get(b"cuteafd_v41_dfly2_merge_stats_f32")? },
        })
    }
}

impl V41Dfly2Ops<'_> {
    pub const VOCAB_TILE_SCRATCH_PER_ROW: usize = 64 * 16;

    /// # Safety
    /// All buffers live on the stream device through completion/replay.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention(
        &self,
        query: CuteafdDeviceBuffer,
        draft_kv: Option<CuteafdDeviceBuffer>,
        sink: CuteafdDeviceBuffer,
        requests_dev: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        alive_out: Option<CuteafdDeviceBuffer>,
        partial: Option<CuteafdDeviceBuffer>,
        requests: usize,
        block_size: usize,
        parts: usize,
        source_fp4: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        let rows = requests * block_size;
        check_buf(query, rows * 64 * 512 * 2, "query")?;
        check_buf(output, rows * 64 * 512 * 2, "output")?;
        check_buf(sink, 64 * 4, "sink")?;
        check_buf(
            requests_dev,
            requests * std::mem::size_of::<V41Dfly2Request>(),
            "requests_dev",
        )?;
        if let Some(d) = draft_kv {
            check_buf(d, rows * 512 * 2, "draft_kv")?;
        }
        if let Some(a) = alive_out {
            check_buf(a, rows, "alive_out")?;
        }
        let (partial_ptr, partial_bytes) = match partial {
            Some(p) => {
                let req = rows * parts * 64 * 514 * 4;
                check_buf(p, req, "partial")?;
                (p.ptr.cast(), p.bytes as u64)
            }
            None => (std::ptr::null_mut(), 0),
        };
        let status = unsafe {
            (self.attention)(
                query.ptr.cast(),
                draft_kv.map_or(std::ptr::null(), |b| b.ptr.cast()),
                sink.ptr.cast(),
                requests_dev.ptr.cast(),
                output.ptr.cast(),
                alive_out.map_or(std::ptr::null_mut(), |b| b.ptr.cast()),
                partial_ptr,
                partial_bytes,
                requests as i32,
                block_size as i32,
                parts as i32,
                i32::from(source_fp4),
                stream,
            )
        };
        self.library.status_to_result("DFly2 attention CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// `embed_table` may reside on a peer GPU with peer access enabled; all
    /// other buffers reside on the stream device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn embed_block(
        &self,
        embed_table: CuteafdDeviceBuffer,
        anchor_ids: CuteafdDeviceBuffer,
        block_out: CuteafdDeviceBuffer,
        anchor_out: Option<CuteafdDeviceBuffer>,
        requests: usize,
        block_size: usize,
        mask_token_id: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(embed_table, 129280 * 5120 * 2, "embed_table")?;
        check_buf(anchor_ids, requests * 4, "anchor_ids")?;
        check_buf(block_out, requests * block_size * 5120 * 2, "block_out")?;
        if let Some(a) = anchor_out {
            check_buf(a, requests * 5120 * 2, "anchor_out")?;
        }
        let status = unsafe {
            (self.embed_block)(
                embed_table.ptr.cast(),
                anchor_ids.ptr.cast(),
                block_out.ptr.cast(),
                anchor_out.map_or(std::ptr::null_mut(), |b| b.ptr.cast()),
                requests as i32,
                block_size as i32,
                mask_token_id,
                stream,
            )
        };
        self.library.status_to_result("DFly2 embed_block CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// `embed_table` may reside on a peer GPU with peer access enabled; other
    /// buffers reside on the stream device.
    pub unsafe fn embed_tokens(
        &self,
        embed_table: CuteafdDeviceBuffer,
        token_ids: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        requests: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(embed_table, 129280 * 5120 * 2, "embed_table")?;
        check_buf(token_ids, requests * 4, "token_ids")?;
        check_buf(out, requests * 5120 * 2, "embed_out")?;
        let status = unsafe {
            (self.embed_tokens)(
                embed_table.ptr.cast(),
                token_ids.ptr.cast(),
                out.ptr.cast(),
                requests as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 embed_tokens CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Disjoint BF16 buffers on the stream device.
    pub unsafe fn concat_bf16(
        &self,
        a: CuteafdDeviceBuffer,
        b: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        rows: usize,
        width_a: usize,
        width_b: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(a, rows * width_a * 2, "concat_a")?;
        check_buf(b, rows * width_b * 2, "concat_b")?;
        check_buf(out, rows * (width_a + width_b) * 2, "concat_out")?;
        let status = unsafe {
            (self.concat_bf16)(
                a.ptr.cast(),
                b.ptr.cast(),
                out.ptr.cast(),
                rows as i32,
                width_a as i32,
                width_b as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 concat_bf16 CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// All buffers reside on the stream device and stay live through completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mtp_fuse_and_seed_hc(
        &self,
        block_x: CuteafdDeviceBuffer,
        fused: CuteafdDeviceBuffer,
        gate: CuteafdDeviceBuffer,
        positions: CuteafdDeviceBuffer,
        residual: CuteafdDeviceBuffer,
        pre_mix: CuteafdDeviceBuffer,
        requests: usize,
        block_size: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let rows = requests * block_size;
        check_buf(block_x, rows * 5120 * 2, "block_x")?;
        check_buf(fused, requests * 5120 * 2, "fused")?;
        check_buf(gate, 5120 * 2, "gate")?;
        check_buf(positions, requests * 8, "positions")?;
        check_buf(residual, rows * 4 * 5120 * 2, "residual")?;
        check_buf(pre_mix, rows * 4 * 4, "pre_mix")?;
        let status = unsafe {
            (self.mtp_fuse_and_seed_hc)(
                block_x.ptr.cast(),
                fused.ptr.cast(),
                gate.ptr.cast(),
                positions.ptr.cast(),
                residual.ptr.cast(),
                pre_mix.ptr.cast(),
                requests as i32,
                block_size as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 mtp_fuse_and_seed_hc CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Buffers reside on the stream device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn swiglu(
        &self,
        gate: CuteafdDeviceBuffer,
        up: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        rows: usize,
        inter: usize,
        limit: f32,
        stream: *mut c_void,
    ) -> Result<()> {
        let bytes = rows * inter * 2;
        check_buf(gate, bytes, "swiglu_gate")?;
        check_buf(up, bytes, "swiglu_up")?;
        check_buf(out, bytes, "swiglu_out")?;
        let status = unsafe {
            (self.swiglu)(
                gate.ptr.cast(),
                up.ptr.cast(),
                out.ptr.cast(),
                rows as i32,
                inter as i32,
                limit,
                stream,
            )
        };
        self.library.status_to_result("DFly2 swiglu CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Buffers reside on the stream device.
    pub unsafe fn add2_bf16(
        &self,
        a: CuteafdDeviceBuffer,
        b: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        count: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let bytes = count * 2;
        check_buf(a, bytes, "add2_a")?;
        check_buf(b, bytes, "add2_b")?;
        check_buf(out, bytes, "add2_out")?;
        let status =
            unsafe { (self.add2_bf16)(a.ptr.cast(), b.ptr.cast(), out.ptr.cast(), count as u64, stream) };
        self.library.status_to_result("DFly2 add2_bf16 CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Buffers reside on the stream device.
    pub unsafe fn add3_bf16(
        &self,
        a: CuteafdDeviceBuffer,
        b: CuteafdDeviceBuffer,
        c: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        count: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let bytes = count * 2;
        check_buf(a, bytes, "add3_a")?;
        check_buf(b, bytes, "add3_b")?;
        check_buf(c, bytes, "add3_c")?;
        check_buf(out, bytes, "add3_out")?;
        let status = unsafe {
            (self.add3_bf16)(a.ptr.cast(), b.ptr.cast(), c.ptr.cast(), out.ptr.cast(), count as u64, stream)
        };
        self.library.status_to_result("DFly2 add3_bf16 CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Disjoint buffers on the stream device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn slice_slot(
        &self,
        block_hidden: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        requests: usize,
        block_size: usize,
        slot: usize,
        hidden_size: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(block_hidden, requests * block_size * hidden_size * 2, "slice_in")?;
        check_buf(out, requests * hidden_size * 2, "slice_out")?;
        let status = unsafe {
            (self.slice_slot)(
                block_hidden.ptr.cast(),
                out.ptr.cast(),
                requests as i32,
                block_size as i32,
                slot as i32,
                hidden_size as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 slice_slot CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Disjoint buffers on the stream device.
    pub unsafe fn scatter_kv_slot(
        &self,
        kv_row: CuteafdDeviceBuffer,
        block_keys: CuteafdDeviceBuffer,
        requests: usize,
        max_slots: usize,
        slot: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(kv_row, requests * 512 * 2, "kv_row")?;
        check_buf(block_keys, requests * max_slots * 512 * 2, "block_keys")?;
        let status = unsafe {
            (self.scatter_kv_slot)(
                kv_row.ptr.cast(),
                block_keys.ptr.cast(),
                requests as i32,
                max_slots as i32,
                slot as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 scatter_kv_slot CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// All buffers reside on the stream device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block_gated_add(
        &self,
        query: CuteafdDeviceBuffer,
        block_keys: CuteafdDeviceBuffer,
        ctx_read: CuteafdDeviceBuffer,
        ctx_alive: CuteafdDeviceBuffer,
        gate_logits: CuteafdDeviceBuffer,
        rel_bias: CuteafdDeviceBuffer,
        block_sink: CuteafdDeviceBuffer,
        gate_bias: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        requests: usize,
        max_slots: usize,
        slot: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(query, requests * 64 * 512 * 2, "bg_query")?;
        check_buf(block_keys, requests * max_slots * 512 * 2, "bg_keys")?;
        check_buf(ctx_read, requests * 64 * 512 * 2, "bg_ctx")?;
        check_buf(ctx_alive, requests, "bg_alive")?;
        check_buf(gate_logits, requests * 64 * 2, "bg_gate_logits")?;
        check_buf(rel_bias, 64 * 16 * 4, "bg_rel_bias")?;
        check_buf(block_sink, 64 * 4, "bg_sink")?;
        check_buf(gate_bias, 64 * 16 * 4, "bg_gate_bias")?;
        check_buf(out, requests * 64 * 512 * 2, "bg_out")?;
        let status = unsafe {
            (self.block_gated_add)(
                query.ptr.cast(),
                block_keys.ptr.cast(),
                ctx_read.ptr.cast(),
                ctx_alive.ptr.cast(),
                gate_logits.ptr.cast(),
                rel_bias.ptr.cast(),
                block_sink.ptr.cast(),
                gate_bias.ptr.cast(),
                out.ptr.cast(),
                requests as i32,
                max_slots as i32,
                slot as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 block_gated_add CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// All buffers reside on the stream device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn shard_stats_f32(
        &self,
        logits: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        out_ids: CuteafdDeviceBuffer,
        out_max: CuteafdDeviceBuffer,
        out_sum_exp: CuteafdDeviceBuffer,
        rows: usize,
        vocab_cols: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(logits, rows * vocab_cols * 4, "shard_logits")?;
        check_buf(scratch, rows * Self::VOCAB_TILE_SCRATCH_PER_ROW, "shard_scratch")?;
        check_buf(out_ids, rows * 4, "shard_ids")?;
        check_buf(out_max, rows * 4, "shard_max")?;
        check_buf(out_sum_exp, rows * 4, "shard_sum_exp")?;
        let status = unsafe {
            (self.shard_stats)(
                logits.ptr.cast(),
                scratch.ptr,
                out_ids.ptr.cast(),
                out_max.ptr.cast(),
                out_sum_exp.ptr.cast(),
                rows as i32,
                vocab_cols as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 shard_stats_f32 CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// All buffers reside on the stream device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn finalize_slot_f32(
        &self,
        ids: CuteafdDeviceBuffer,
        sum_exp: CuteafdDeviceBuffer,
        curr_tokens: CuteafdDeviceBuffer,
        all_tokens: CuteafdDeviceBuffer,
        all_probs: CuteafdDeviceBuffer,
        rows: usize,
        drafts: usize,
        slot_idx: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        check_buf(ids, rows * 4, "fin_ids")?;
        check_buf(sum_exp, rows * 4, "fin_sum_exp")?;
        check_buf(curr_tokens, rows * 4, "fin_curr")?;
        check_buf(all_tokens, rows * drafts * 4, "fin_all_tokens")?;
        check_buf(all_probs, rows * drafts * 4, "fin_all_probs")?;
        let status = unsafe {
            (self.finalize_slot)(
                ids.ptr.cast(),
                sum_exp.ptr.cast(),
                curr_tokens.ptr.cast(),
                all_tokens.ptr.cast(),
                all_probs.ptr.cast(),
                rows as i32,
                drafts as i32,
                slot_idx as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 finalize_slot_f32 CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// All buffers reside on the stream device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn merge_stats_f32(
        &self,
        ids0: CuteafdDeviceBuffer,
        max0: CuteafdDeviceBuffer,
        sum0: CuteafdDeviceBuffer,
        ids1: CuteafdDeviceBuffer,
        max1: CuteafdDeviceBuffer,
        sum1: CuteafdDeviceBuffer,
        curr_tokens: CuteafdDeviceBuffer,
        all_tokens: CuteafdDeviceBuffer,
        all_probs: CuteafdDeviceBuffer,
        rows: usize,
        split: usize,
        drafts: usize,
        slot_idx: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        for (b, name) in [
            (ids0, "ids0"),
            (max0, "max0"),
            (sum0, "sum0"),
            (ids1, "ids1"),
            (max1, "max1"),
            (sum1, "sum1"),
            (curr_tokens, "curr_tokens"),
        ] {
            check_buf(b, rows * 4, name)?;
        }
        check_buf(all_tokens, rows * drafts * 4, "merge_all_tokens")?;
        check_buf(all_probs, rows * drafts * 4, "merge_all_probs")?;
        let status = unsafe {
            (self.merge_stats)(
                ids0.ptr.cast(),
                max0.ptr.cast(),
                sum0.ptr.cast(),
                ids1.ptr.cast(),
                max1.ptr.cast(),
                sum1.ptr.cast(),
                curr_tokens.ptr.cast(),
                all_tokens.ptr.cast(),
                all_probs.ptr.cast(),
                rows as i32,
                split as i32,
                drafts as i32,
                slot_idx as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 merge_stats_f32 CUDA status", status)?;
        Ok(())
    }
}

type BlasCreateFn = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> i32;
type BlasDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type GemmBf16Fn = unsafe extern "C" fn(
    *mut c_void, *const u16, *const u16, *mut u16, i32, i32, i32, *mut c_void,
) -> i32;

/// Owned pre-Ready cuBLAS handle bound to a DFly2 workspace plane.
pub struct V41Dfly2Blas<'a> {
    library: &'a NativeLibrary,
    device: i32,
    raw: *mut c_void,
    destroy: BlasDestroyFn,
    gemm: GemmBf16Fn,
}

impl<'a> V41Dfly2Blas<'a> {
    pub const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

    /// # Safety
    /// `workspace` remains valid on `workspace.device_id` until `self` is dropped.
    pub unsafe fn create(library: &'a NativeLibrary, workspace: CuteafdDeviceBuffer) -> Result<Self> {
        ensure!(
            !workspace.ptr.is_null() && workspace.bytes >= Self::WORKSPACE_BYTES,
            "DFly2 BLAS workspace requires {} bytes, got {}",
            Self::WORKSPACE_BYTES,
            workspace.bytes
        );
        let create = *unsafe { library.lib.get::<BlasCreateFn>(b"cuteafd_v41_dfly2_blas_create") }?;
        let destroy = *unsafe { library.lib.get::<BlasDestroyFn>(b"cuteafd_v41_dfly2_blas_destroy") }?;
        let gemm = *unsafe { library.lib.get::<GemmBf16Fn>(b"cuteafd_v41_dfly2_gemm_bf16") }?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe { create(workspace.ptr, workspace.bytes as u64, &mut raw) };
        library.status_to_result("DFly2 BLAS create CUDA status", status)?;
        Ok(Self {
            library,
            device: workspace.device_id,
            raw,
            destroy,
            gemm,
        })
    }

    /// Computes `out[rows, n] = x[rows, k] @ w[n, k]^T` in BF16 with FP32 accumulation.
    ///
    /// # Safety
    /// Buffers reside on `self`'s device and stay alive until `stream` completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_bf16(
        &self,
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        rows: usize,
        k: usize,
        n: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let status = unsafe {
            (self.gemm)(
                self.raw,
                x.cast(),
                w.cast(),
                out.cast(),
                rows as i32,
                k as i32,
                n as i32,
                stream,
            )
        };
        self.library.status_to_result("DFly2 gemm_bf16 CUDA status", status)?;
        Ok(())
    }
}

impl Drop for V41Dfly2Blas<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = self.library.cuda_set_device(self.device);
            unsafe { (self.destroy)(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

impl NativeLibrary {
    /// # Safety
    /// `workspace` remains valid on the current device until the handle drops.
    pub unsafe fn v41_dfly2_blas(&self, workspace: CuteafdDeviceBuffer) -> Result<V41Dfly2Blas<'_>> {
        unsafe { V41Dfly2Blas::create(self, workspace) }
    }
}
