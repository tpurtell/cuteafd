//! One allocation description for MiMo startup admission and engine workspaces.
//! Keeping attention geometry explicit lets a separately qualified head-width
//! reduction change Q/attention/KV storage without changing hidden rows or MTP.
use super::{MimoAttention, MimoKvCache, MimoKvFormats, MimoV2Config};
use crate::serving_capacity::CacheGeometryError;
use cuteafd_core::serving_capacity::MemoryReservation;

/// Spark activation format; coordinator-native packages always consume BF16.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MimoExpertInput {
    #[default]
    Fp8,
    Bf16,
    Bf16Decode,
}

impl MimoExpertInput {
    pub fn bf16(self, decode: bool) -> bool {
        match self {
            Self::Fp8 => false,
            Self::Bf16 => true,
            Self::Bf16Decode => decode,
        }
    }
}

/// Immutable prefill output contract. Decode/verify always retain every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoPrefillOutput {
    LastRow,
    AllRows,
}

impl MimoPrefillOutput {
    pub fn logits_rows(self, rows: u64, decode: bool, with_head: bool) -> u64 {
        if !with_head { 0 } else if decode || self == Self::AllRows { rows } else { 1 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoAttentionWorkspace {
    /// Current engine geometry, including global-width peer/lane buffers.
    Global,
    PartitionedHeads {
        ranks: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct MimoWorkspaceOptions {
    pub rows: u64,
    /// Slot-owned rows; scratch remains one unit wide.
    pub waves: u64,
    pub decode: bool,
    pub lead: bool,
    pub with_head: bool,
    pub spark: bool,
    pub expert_input: MimoExpertInput,
    pub max_context: u64,
    pub pool_pages: u64,
    pub kv_cache: MimoKvFormats,
    pub attention: MimoAttentionWorkspace,
    pub prefill_output: MimoPrefillOutput,
    /// Maximum scratch of every applicable native program at this shape.
    pub native_scratch_bytes: u64,
    pub head_workspace_bytes: u64,
}

impl MimoWorkspaceOptions {
    /// Prefill widens one sequence's compact KV on its rank's compute stream.
    /// Both lanes may borrow the same arena: the next widen follows every
    /// earlier attention read on that stream. Decode/BF16 keep private dummies.
    pub fn shares_prefill_kv_wide(self) -> bool {
        !self.decode && self.kv_cache.full != MimoKvCache::Bf16
    }
}

/// One physical owner for the identical logical shadow extents of a rank's
/// prefill workspaces. Construct this independently for each physical GPU.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MimoPrefillKvShadowPlan {
    bytes: Option<u64>,
}

impl MimoPrefillKvShadowPlan {
    pub fn require_same_extent(owned: u64, requested: u64) -> Result<(), CacheGeometryError> {
        if owned < 256 || owned != requested {
            return Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "shared prefill KV shadow extents differ; published storage cannot be resized",
            });
        }
        Ok(())
    }

    pub fn workspace_reservations(&mut self, name: &str, layout: &MimoWorkspaceLayout,
        options: MimoWorkspaceOptions) -> Result<Vec<MemoryReservation>, CacheGeometryError> {
        let mut reservations = layout.reservations(name);
        if options.shares_prefill_kv_wide() {
            Self::require_same_extent(self.bytes.unwrap_or(layout.kv_wide), layout.kv_wide)?;
            self.bytes = Some(layout.kv_wide);
            let borrowed = format!("{name}.kv_wide");
            reservations.retain(|r| r.name != borrowed);
        }
        Ok(reservations)
    }

    pub fn reservation(self) -> Option<MemoryReservation> {
        self.bytes.map(|bytes| MemoryReservation { name: "state.prefill_kv_wide".into(), bytes })
    }
}

macro_rules! workspace_buffers {
    ($($field:ident),+ $(,)?) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct MimoWorkspaceLayout {
            $(pub $field: u64,)+
            pub router_host: u64,
            /// Shape of the logits allocation and vocabulary-head handle.
            pub logits_rows: u64,
        }

        impl MimoWorkspaceLayout {
            pub fn device_buffers(&self) -> impl Iterator<Item = (&'static str, u64)> {
                [$( (stringify!($field), self.$field), )+].into_iter()
            }

            pub fn device_bytes(&self) -> Result<u64, CacheGeometryError> {
                self.device_buffers().try_fold(0u64, |sum, (_, bytes)| sum.checked_add(bytes))
                    .ok_or(CacheGeometryError::Overflow("MiMo workspace allocation sum"))
            }

            /// Startup names each reservation so an admission failure can
            /// identify the shape and buffer instead of a nominal headroom.
            pub fn reservations(&self, shape: &str) -> Vec<MemoryReservation> {
                self.device_buffers().map(|(name, bytes)| MemoryReservation {
                    name: format!("{shape}.{name}"), bytes,
                }).collect()
            }
        }
    };
}

workspace_buffers! {
    h, x, query, attn, delta, kv_step, kv_wide, step_inputs, slots,
    step_slots, ring_slots, seq_first, page_table, scratch, logits,
    router_logits, route_ids, route_weights, wire, zero_plane,
    ids, select, head_workspace,
}

fn bytes(label: &'static str, dims: &[u64]) -> Result<u64, CacheGeometryError> {
    dims.iter()
        .try_fold(1u64, |value, &dim| value.checked_mul(dim))
        .map(|value| value.max(256))
        .ok_or(CacheGeometryError::Overflow(label))
}

impl MimoWorkspaceLayout {
    /// Base paths and slot zero share pointer keys. Nonzero slots have their
    /// own pinned source; the base extent must cover diagnostic whole-buffer use.
    pub fn ordered_host_bytes(&self, waves: u64) -> Result<u64, CacheGeometryError> {
        if waves == 0 { return Err(CacheGeometryError::Overflow("MiMo ordered slot count")); }
        [self.step_inputs, self.slots, self.ring_slots, self.seq_first, self.page_table, self.ids]
            .into_iter().try_fold(0u64, |sum, bytes| {
                bytes.checked_div(waves)?.checked_mul(waves - 1)?.checked_add(bytes)?.checked_add(sum)
            }).ok_or(CacheGeometryError::Overflow("MiMo ordered host tables"))
    }

    pub fn new(
        cfg: &MimoV2Config,
        options: MimoWorkspaceOptions,
    ) -> Result<Self, CacheGeometryError> {
        if options.rows == 0
            || options.waves == 0
            || options.decode && options.waves != 1
            || options.max_context == 0
            || options.with_head && !options.lead
            || options.with_head && options.head_workspace_bytes == 0
            || options.spark && !options.lead
            || cfg.head_dim != 192
            || cfg.v_head_dim != 128
            || cfg.program_family().is_err()
        {
            return Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "invalid workspace rows/context or lead-only work",
            });
        }
        let attention = match options.attention {
            MimoAttentionWorkspace::Global => cfg.clone(),
            MimoAttentionWorkspace::PartitionedHeads { ranks } => {
                cfg.head_split(ranks)
                    .map_err(|_| CacheGeometryError::Unsupported {
                        family: "mimo_v2",
                        what: "workspace heads do not partition over selected ranks",
                    })?
            }
        };
        let (t, h) = (options.rows, cfg.hidden as u64);
        let live = t.checked_mul(options.waves).ok_or(CacheGeometryError::Overflow("MiMo unit rows"))?;
        let record = attention
            .record_bytes(MimoAttention::Sliding, options.kv_cache)
            .max(attention.record_bytes(MimoAttention::Full, options.kv_cache))
            as u64;
        let page_table_rows = if options.decode { t } else { options.waves };
        let logits_rows = options.prefill_output.logits_rows(
            if options.decode { t } else if options.prefill_output == MimoPrefillOutput::AllRows { t.max(16384) } else { t },
            options.decode, options.with_head);
        let layout = Self {
            logits_rows,
            h: bytes("MiMo hidden rows", &[live, h, 2])?,
            x: bytes("MiMo normalized rows", &[live, h, 2])?,
            query: bytes(
                "MiMo query",
                &[t, attention.heads as u64, attention.head_dim as u64, 2],
            )?,
            attn: bytes(
                "MiMo attention",
                &[t, attention.heads as u64, attention.v_head_dim as u64, 2],
            )?,
            delta: bytes("MiMo layer delta", &[live, h, 2])?,
            kv_step: bytes("MiMo KV step", &[t, record])?,
            kv_wide: if options.decode || options.kv_cache.full == MimoKvCache::Bf16 {
                256
            } else {
                bytes(
                    "MiMo prefill BF16 KV shadow",
                    &[
                        options.max_context,
                        attention.record_bytes(MimoAttention::Full, MimoKvFormats { full: MimoKvCache::Bf16, sliding: MimoKvCache::Bf16 }) as u64,
                    ],
                )?
            },
            step_inputs: bytes("MiMo positions and full/SWA RoPE windows", &[live, 8 + 2 * cfg.rope_dim as u64 * 4])?,
            slots: bytes("MiMo KV slots", &[live, 8])?,
            step_slots: bytes("MiMo identity slots", &[live, 8])?,
            ring_slots: bytes("MiMo ring slots", &[live, 8])?,
            seq_first: bytes("MiMo sequence starts", &[live, 4])?,
            page_table: if options.decode { bytes("MiMo page table", &[page_table_rows, options.pool_pages, 4])? }
                else { bytes("MiMo unit page table", &[options.pool_pages, 4])?
                    .checked_mul(options.waves).ok_or(CacheGeometryError::Overflow("MiMo unit page tables"))? },
            scratch: options.native_scratch_bytes.max(256),
            logits: if options.with_head {
                bytes("MiMo logits", &[logits_rows, cfg.vocab_size as u64, 4])?
            } else {
                256
            },
            router_logits: bytes("MiMo router logits", &[t, cfg.experts as u64, 4])?,
            route_ids: bytes("MiMo route ids", &[t, cfg.topk as u64, 4])?,
            route_weights: bytes("MiMo route weights", &[t, cfg.topk as u64, 4])?,
            // BF16 Spark ingress borrows x, while still needing Spark's zero
            // plane and pinned route staging. Only FP8 ingress owns K32 wire.
            wire: if options.spark && !options.expert_input.bf16(options.decode) { bytes(
                "MiMo expert wire",
                &[
                    t,
                    h.checked_add(h / 32)
                        .ok_or(CacheGeometryError::Overflow("MiMo wire width"))?,
                ],
            )? } else { 256 },
            zero_plane: if options.spark {
                bytes("MiMo empty shared expert", &[t, h, 2])?
            } else {
                256
            },
            router_host: if options.spark {
                bytes(
                    "MiMo pinned route staging",
                    &[
                        t,
                        (cfg.topk as u64)
                            .checked_mul(8)
                            .and_then(|v| h.checked_mul(2)?.checked_add(v))
                            .ok_or(CacheGeometryError::Overflow("MiMo pinned route width"))?,
                    ],
                )?
            } else {
                256
            },
            ids: bytes("MiMo token ids", &[live, 4])?,
            select: bytes("MiMo greedy ids/statuses", &[t, 8])?,
            head_workspace: if options.with_head {
                options.head_workspace_bytes.max(256)
            } else {
                256
            },
        };
        layout.device_bytes()?;
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{mimo_flash_config, mimo_pro_config};

    fn options() -> MimoWorkspaceOptions {
        MimoWorkspaceOptions {
            rows: 4096,
            waves: 1,
            decode: false,
            lead: true,
            with_head: true,
            spark: true,
            expert_input: MimoExpertInput::Fp8,
            max_context: 1 << 20,
            pool_pages: 1 << 14,
            kv_cache: MimoKvFormats::default(),
            attention: MimoAttentionWorkspace::Global,
            prefill_output: MimoPrefillOutput::AllRows,
            native_scratch_bytes: 16 << 20,
            head_workspace_bytes: 32 << 20,
        }
    }

    #[test]
    fn unit_slots_multiply_only_live_planes_and_keep_page_tables_aligned() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        let single = MimoWorkspaceLayout::new(&cfg, options()).unwrap();
        for waves in [2, 3, 4] {
            let multi = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions { waves, ..options() }).unwrap();
            for ((name, one), (other, many)) in single.device_buffers().zip(multi.device_buffers()) {
                assert_eq!(name, other);
                let owned = matches!(name, "h" | "x" | "delta" | "step_inputs" | "slots" | "step_slots"
                    | "ring_slots" | "seq_first" | "page_table" | "ids");
                assert_eq!(many, one * if owned { waves } else { 1 }, "{name}");
            }
            let tiny = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions { waves, pool_pages: 1, ..options() }).unwrap();
            assert_eq!(tiny.page_table, waves * 256);
        }
    }

    #[test]
    fn ordered_host_staging_covers_base_and_nonzero_slot_pointer_keys() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        for waves in 1..=4 {
            let layout = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions { waves, ..options() }).unwrap();
            let base = layout.step_inputs + layout.slots + layout.ring_slots + layout.seq_first
                + layout.page_table + layout.ids;
            assert_eq!(layout.ordered_host_bytes(waves).unwrap(), base + base / waves * (waves - 1));
        }
        let one = super::super::admission::future_host_workspaces(&cfg, 4096, 2, 1, 1 << 20,
            1 << 20, 6, MimoExpertInput::Fp8, MimoKvFormats::default(), MimoPrefillOutput::LastRow,
            true, 256, 0).unwrap();
        let two = super::super::admission::future_host_workspaces(&cfg, 4096, 2, 2, 1 << 20,
            1 << 20, 6, MimoExpertInput::Fp8, MimoKvFormats::default(), MimoPrefillOutput::LastRow,
            true, 256, 0).unwrap();
        for kind in ["prefill/ordered-tables", "decode/ordered-tables"] {
            let sum = |demands: &[(String, u64)]| demands.iter().filter(|(name, _)| name.ends_with(kind))
                .map(|(_, bytes)| bytes).sum::<u64>();
            assert_eq!(sum(&two), 2 * sum(&one));
        }
        assert!(two.iter().any(|(name, bytes)| name == "mimo/owner-1/decode/ordered-tables" && *bytes > 0));
        assert!(two.iter().any(|(name, bytes)| name == "mimo/transport/rdma-rings" && *bytes > 0));
        assert!(super::super::admission::future_host_workspaces(&cfg, usize::MAX, 4, 2, u64::MAX,
            1 << 20, 6, MimoExpertInput::Fp8, MimoKvFormats::default(), MimoPrefillOutput::LastRow,
            true, 256, 0).is_err());
    }

    #[test]
    fn rope_windows_charge_rows_not_compiled_context() {
        let cfg = MimoV2Config::from_hf(&mimo_flash_config()).unwrap();
        let before = MimoWorkspaceOptions { max_context: 1 << 16, ..options() };
        let layout = MimoWorkspaceLayout::new(&cfg, before).unwrap();
        assert_eq!(layout.step_inputs, before.rows * (8 + 2 * 64 * 4));
        let large = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions { max_context: 1 << 20, ..before }).unwrap();
        assert_eq!(large.step_inputs, layout.step_inputs);
    }

    #[test]
    fn expert_ingress_owns_wire_only_for_fp8_spark_shapes() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        for spark in [false, true] {
            for decode in [false, true] {
                for expert_input in [MimoExpertInput::Fp8, MimoExpertInput::Bf16, MimoExpertInput::Bf16Decode] {
                    let opts = MimoWorkspaceOptions { spark, decode, expert_input, ..options() };
                    let actual = MimoWorkspaceLayout::new(&cfg, opts).unwrap();
                    let fp8 = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions {
                        expert_input: MimoExpertInput::Fp8, ..opts
                    }).unwrap();
                    let needs_wire = spark && !expert_input.bf16(decode);
                    assert_eq!(actual.wire, if needs_wire {
                        opts.rows * (cfg.hidden + cfg.hidden / 32) as u64
                    } else { 256 });
                    for ((name, bytes), (fp8_name, fp8_bytes)) in actual.device_buffers().zip(fp8.device_buffers()) {
                        assert_eq!(name, fp8_name);
                        if name != "wire" { assert_eq!(bytes, fp8_bytes, "{name}"); }
                    }
                    assert_eq!(actual.router_host, fp8.router_host);
                    assert_eq!(fp8.device_bytes().unwrap() - actual.device_bytes().unwrap(), fp8.wire - actual.wire);
                    assert_eq!(actual.zero_plane, if spark { opts.rows * cfg.hidden as u64 * 2 } else { 256 });
                }
            }
        }
    }

    #[test]
    fn serving_prefill_head_and_allocation_admit_exactly_one_row() {
        let mut raw = mimo_pro_config();
        raw["vocab_size"] = serde_json::json!(152576);
        let cfg = MimoV2Config::from_hf(&raw).unwrap();
        let full = MimoWorkspaceLayout::new(&cfg, options()).unwrap();
        let last = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions {
            prefill_output: MimoPrefillOutput::LastRow, ..options()
        }).unwrap();
        assert_eq!((full.logits_rows, last.logits_rows), (16384, 1));
        assert_eq!(last.logits, 152576 * 4);
        let saved = 16383 * 152576 * 4;
        assert_eq!(full.logits - last.logits, saved);
        assert_eq!(full.device_bytes().unwrap() - last.device_bytes().unwrap(), saved);
        for ((name, before), (after_name, after)) in full.device_buffers().zip(last.device_buffers()) {
            assert_eq!(name, after_name);
            if name != "logits" { assert_eq!(before, after, "{name}"); }
        }
        assert_eq!(last.reservations("prefill").iter().map(|r| r.bytes).sum::<u64>(), last.device_bytes().unwrap());
    }

    #[test]
    fn last_row_selection_preserves_decode_verify_and_headless_lanes() {
        for raw in [mimo_flash_config(), mimo_pro_config()] {
            let cfg = MimoV2Config::from_hf(&raw).unwrap();
            for rows in [1, 64, 1024, 4096] {
                for decode in [false, true] {
                    let full = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions { rows, decode, ..options() }).unwrap();
                    let last = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions {
                        rows, decode, prefill_output: MimoPrefillOutput::LastRow, ..options()
                    }).unwrap();
                    assert_eq!(last.logits_rows, if decode { rows } else { 1 });
                    if decode { assert_eq!(full, last); }
                    let headless = MimoWorkspaceLayout::new(&cfg, MimoWorkspaceOptions {
                        rows, decode, with_head: false, prefill_output: MimoPrefillOutput::LastRow, ..options()
                    }).unwrap();
                    assert_eq!((headless.logits_rows, headless.logits, headless.head_workspace), (0, 256, 256));
                }
            }
        }
    }

    #[test]
    fn independent_prefill_head_adds_exact_output_and_head_workspace_on_lead_only() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        for ranks in [1, 2] {
            for rank in 0..ranks {
                for kv_cache in [MimoKvFormats { full: MimoKvCache::Bf16, sliding: MimoKvCache::Bf16 }, MimoKvFormats::default()] {
                    let before_options = MimoWorkspaceOptions {
                        lead: rank == 0, with_head: false, spark: rank == 0,
                        kv_cache, prefill_output: MimoPrefillOutput::LastRow,
                        attention: if ranks == 2 { MimoAttentionWorkspace::PartitionedHeads { ranks } }
                            else { MimoAttentionWorkspace::Global }, ..options()
                    };
                    let before = MimoWorkspaceLayout::new(&cfg, before_options).unwrap();
                    let after_options = MimoWorkspaceOptions { with_head: rank == 0, ..before_options };
                    let after = MimoWorkspaceLayout::new(&cfg, after_options).unwrap();
                    let extra = if rank == 0 {
                        cfg.vocab_size as u64 * 4 - 256 + options().head_workspace_bytes - 256
                    } else { 0 };
                    assert_eq!(after.device_bytes().unwrap() - before.device_bytes().unwrap(), extra);
                    assert_eq!(after.logits_rows, u64::from(rank == 0));
                    for ((name, a), (_, b)) in before.device_buffers().zip(after.device_buffers()) {
                        if name != "logits" && name != "head_workspace" { assert_eq!(a, b, "{name}"); }
                    }
                    let mut shared = MimoPrefillKvShadowPlan::default();
                    let mut costs = shared.workspace_reservations("first", &after, after_options).unwrap();
                    costs.extend(shared.workspace_reservations("second", &after, after_options).unwrap());
                    costs.extend(shared.reservation());
                    if kv_cache.full != MimoKvCache::Bf16 {
                        assert_eq!(costs.iter().filter(|r| r.name == "state.prefill_kv_wide").count(), 1);
                        assert!(!costs.iter().any(|r| r.name == "first.kv_wide" || r.name == "second.kv_wide"));
                    }
                }
            }
        }
    }

    #[test]
    fn pro_global_geometry_requires_full_shadow_extent_and_peer_hidden_rows() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        let lead = MimoWorkspaceLayout::new(&cfg, options()).unwrap();
        let peer = MimoWorkspaceLayout::new(
            &cfg,
            MimoWorkspaceOptions {
                lead: false,
                with_head: false,
                spark: false,
                ..options()
            },
        )
        .unwrap();
        assert_eq!(lead.kv_wide, 5 << 30);
        assert_eq!(peer.kv_wide, lead.kv_wide);
        assert_eq!(peer.query, lead.query);
        assert_eq!(peer.attn, lead.attn);
        assert_eq!(peer.kv_step, lead.kv_step);
        assert_eq!((peer.h, peer.x, peer.delta), (lead.h, lead.x, lead.delta));
        assert_eq!(
            (
                peer.head_workspace,
                peer.logits,
                peer.zero_plane,
                peer.router_host
            ),
            (256, 256, 256, 256)
        );
        assert_eq!(lead.device_buffers().count(), 23);
        assert_eq!(
            lead.device_bytes().unwrap(),
            lead.reservations("prefill0")
                .iter()
                .map(|item| item.bytes)
                .sum::<u64>()
        );
    }

    #[test]
    fn attention_geometry_selection_isolated_from_hidden_head_and_router_storage() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        let global = MimoWorkspaceLayout::new(&cfg, options()).unwrap();
        let split = MimoWorkspaceLayout::new(
            &cfg,
            MimoWorkspaceOptions {
                attention: MimoAttentionWorkspace::PartitionedHeads { ranks: 2 },
                ..options()
            },
        )
        .unwrap();
        for (a, b) in [
            (global.query, split.query),
            (global.attn, split.attn),
            (global.kv_step, split.kv_step),
            (global.kv_wide, split.kv_wide),
        ] {
            assert_eq!(a, 2 * b);
        }
        assert_eq!(
            (global.h, global.x, global.delta, global.logits, global.wire),
            (split.h, split.x, split.delta, split.logits, split.wire)
        );
    }

    #[test]
    fn decode_and_bf16_skip_shadow_but_reserve_actual_table_stride_and_minimum_allocations() {
        let cfg = MimoV2Config::from_hf(&mimo_flash_config()).unwrap();
        let decode = MimoWorkspaceLayout::new(
            &cfg,
            MimoWorkspaceOptions {
                rows: 64,
                decode: true,
                ..options()
            },
        )
        .unwrap();
        let prefill = MimoWorkspaceLayout::new(
            &cfg,
            MimoWorkspaceOptions {
                kv_cache: MimoKvFormats { full: MimoKvCache::Bf16, sliding: MimoKvCache::Bf16 },
                ..options()
            },
        )
        .unwrap();
        assert_eq!((decode.kv_wide, prefill.kv_wide), (256, 256));
        assert_eq!(decode.page_table, 64 * (1 << 14) * 4);
        assert_eq!(prefill.page_table, (1 << 14) * 4);
        assert!(decode.device_buffers().all(|(_, bytes)| bytes >= 256));
        let huge = MimoWorkspaceLayout::new(
            &cfg,
            MimoWorkspaceOptions {
                rows: u64::MAX,
                ..options()
            },
        );
        assert!(matches!(huge, Err(CacheGeometryError::Overflow(_))));
    }

    #[test]
    fn prefill_lanes_share_one_rank_shadow_while_decode_keeps_its_dummy() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        for attention in [MimoAttentionWorkspace::Global,
            MimoAttentionWorkspace::PartitionedHeads { ranks: 2 }] {
            let opts = MimoWorkspaceOptions { attention, ..options() };
            let layout = MimoWorkspaceLayout::new(&cfg, opts).unwrap();
            let decode_opts = MimoWorkspaceOptions { decode: true, rows: 64, ..opts };
            let decode = MimoWorkspaceLayout::new(&cfg, decode_opts).unwrap();
            let mut owner = MimoPrefillKvShadowPlan::default();
            let mut physical = owner.workspace_reservations("prefill", &layout, opts).unwrap();
            physical.extend(owner.workspace_reservations("lane", &layout, opts).unwrap());
            physical.extend(owner.workspace_reservations("decode", &decode, decode_opts).unwrap());
            physical.extend(owner.reservation());
            let shadow = owner.reservation().unwrap();
            assert_eq!(shadow.bytes, if matches!(attention, MimoAttentionWorkspace::Global) { 5 << 30 } else { 5 << 29 });
            assert_eq!(physical.iter().filter(|r| r.name == "state.prefill_kv_wide").count(), 1);
            assert_eq!(physical.iter().find(|r| r.name == "decode.kv_wide").unwrap().bytes, 256);
            assert_eq!(physical.iter().map(|r| r.bytes).sum::<u64>(),
                layout.device_bytes().unwrap() * 2 + decode.device_bytes().unwrap() - shadow.bytes);
        }
    }

    #[test]
    fn shared_shadow_rejects_extent_changes_and_preserves_bf16_owned_dummies() {
        let cfg = MimoV2Config::from_hf(&mimo_pro_config()).unwrap();
        let opts = options();
        let layout = MimoWorkspaceLayout::new(&cfg, opts).unwrap();
        let changed_opts = MimoWorkspaceOptions { max_context: opts.max_context / 2, ..opts };
        let changed = MimoWorkspaceLayout::new(&cfg, changed_opts).unwrap();
        let mut owner = MimoPrefillKvShadowPlan::default();
        owner.workspace_reservations("prefill", &layout, opts).unwrap();
        assert!(owner.workspace_reservations("lane", &changed, changed_opts).is_err());
        assert_eq!(owner.reservation().unwrap().bytes, layout.kv_wide);
        let opts = MimoWorkspaceOptions { kv_cache: MimoKvFormats { full: MimoKvCache::Bf16, sliding: MimoKvCache::Bf16 }, ..opts };
        let bf16 = MimoWorkspaceLayout::new(&cfg, opts).unwrap();
        let mut owner = MimoPrefillKvShadowPlan::default();
        for name in ["prefill", "lane"] {
            let reservations = owner.workspace_reservations(name, &bf16, opts).unwrap();
            assert_eq!(reservations.iter().find(|r| r.name == format!("{name}.kv_wide")).unwrap().bytes, 256);
        }
        assert!(owner.reservation().is_none());
    }

}
