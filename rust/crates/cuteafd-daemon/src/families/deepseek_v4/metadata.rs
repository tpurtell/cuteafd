//! Host-built step metadata for DeepSeek V4: RoPE tables and the slot, window
//! and compressor tables one sequence prefilled from position 0 needs.
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;

pub(crate) const WINDOW: usize = 128;
pub(crate) const SOURCE_PAGE_TOKENS: usize = 256;
pub(crate) const MAIN_PAGE_BYTES: usize = 149_760;
pub(crate) const INDEX_PAGE_BYTES: usize = 8_448;

pub(crate) fn compressed_page_rows(ratio: usize) -> usize {
    SOURCE_PAGE_TOKENS / ratio
}

/// Compressed-MLA page: rows x 584 bytes rounded up to a 576-byte multiple.
pub(crate) fn compressed_page_bytes(ratio: usize) -> usize {
    (compressed_page_rows(ratio) * 584).div_ceil(576) * 576
}

/// b12x RoPE table, FP32 [positions, 64] = [cos(32) | sin(32)] (model.py
/// precompute_freqs_cis). Window-only layers use rope_theta without YaRN;
/// compressed layers use compress_rope_theta with YaRN.
pub(crate) fn rope_table(cfg: &DeepseekV4Config, compressed: bool, positions: usize) -> Vec<f32> {
    let dim = cfg.rope_head_dim;
    let (original, base) = if compressed {
        (cfg.original_seq_len, cfg.compress_rope_theta)
    } else {
        (0, cfg.rope_theta)
    };
    let half = dim / 2;
    let mut freqs: Vec<f64> = (0..half)
        .map(|i| {
            let exponent = (2 * i) as f32 / dim as f32;
            f64::from(1.0f32 / (base as f32).powf(exponent))
        })
        .collect();
    if original > 0 {
        let correction = |rotations: f64| {
            dim as f64 * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = correction(cfg.beta_fast).floor().max(0.0);
        let high = correction(cfg.beta_slow).ceil().min(dim as f64 - 1.0);
        let high = if low == high { high + 0.001 } else { high };
        for (i, freq) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / cfg.rope_factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    let mut table = vec![0f32; positions * dim];
    for t in 0..positions {
        for (i, freq) in freqs.iter().enumerate() {
            // torch.outer runs in FP32.
            let angle = f64::from(t as f32 * *freq as f32);
            table[t * dim + i] = angle.cos() as f32;
            table[t * dim + half + i] = angle.sin() as f32;
        }
    }
    table
}

/// Host tables for one step of one sequence: its rows start at `start`
/// (0 for the initial prefill) and the caches already hold `start` tokens.
/// Prefill steps must start at 0; decode steps are one row.
#[derive(Clone)]
pub(crate) struct StepTables {
    pub decode: bool,
    /// Decode-shaped step whose sequences contribute consecutive rows (a
    /// speculative verify): decode programs, continuation compressor.
    pub chunked: bool,
    /// First position of the rows (prefill chunks; 0 for decode).
    pub start: usize,
    pub rows: usize,
    pub positions: Vec<i64>,
    /// Physical window slots the producer writes, per row.
    pub main_slots: Vec<i64>,
    pub swa_indices: Vec<i32>,
    pub swa_lengths: Vec<i32>,
    /// Compressor metadata in program pointer order, per ratio.
    pub c4_tables: Vec<(&'static str, Vec<i32>)>,
    pub c128_tables: Vec<(&'static str, Vec<i32>)>,
    /// Completed groups after this step (grid bound for prefill programs).
    pub c4_groups: usize,
    pub c128_groups: usize,
    /// C4 index top-k: page table (shared row for prefill, [rows, stride] for decode).
    pub c4_page_table: Vec<i32>,
    pub c4_table_width: usize,
    pub c4_table_stride: usize,
    pub c4_visible: Vec<i32>,
    pub c4_indexed_lengths: Vec<i32>,
    pub c128_indices: Vec<i32>,
    pub c128_lengths: Vec<i32>,
}

/// Tables for a prefill chunk of one sequence: rows are positions
/// `start..start + tokens`; the caches already hold `start` tokens. The first
/// chunk uses the prefill compressor, later chunks its continuation.
pub(crate) fn prefill_step(
    placement: &super::pool::Placement,
    shape: &super::pool::PoolShape,
    start: usize,
    tokens: usize,
    index_topk: usize,
    c128_width: usize,
) -> anyhow::Result<StepTables> {
    let end = start + tokens;
    let positions: Vec<i64> = (start as i64..end as i64).collect();
    let slot = |p: usize| placement.window_slot(shape, p);
    let mut swa_indices = vec![-1i32; tokens * WINDOW];
    for (row, t) in (start..end).enumerate() {
        let first = t.saturating_sub(WINDOW - 1);
        for j in 0..WINDOW {
            if first + j <= t {
                swa_indices[row * WINDOW + j] = slot(first + j) as i32;
            }
        }
    }
    let nonempty = |v: Vec<i32>| if v.is_empty() { vec![0] } else { v };
    let compressor_tables = |ratio: usize| -> anyhow::Result<(Vec<(&'static str, Vec<i32>)>, usize)> {
        // Groups completed by a row of this chunk, by their first position.
        let group_starts: Vec<usize> = (start..end).filter(|p| p % ratio == ratio - 1).map(|p| p + 1 - ratio).collect();
        let groups = group_starts.len();
        let slots = group_starts.iter().map(|g| placement.group_slot(ratio, g / ratio)).collect::<anyhow::Result<Vec<_>>>()?;
        let absolute: Vec<i32> = group_starts.iter().map(|&g| g as i32).collect();
        let tables = if start == 0 {
            vec![
                ("active_groups", vec![groups as i32]),
                ("group_source_starts", nonempty(absolute.clone())),
                ("group_rope_positions", nonempty(absolute)),
                ("compressed_slots", nonempty(slots)),
                ("active_sequences", vec![1]),
                ("sequence_offsets", vec![0, tokens as i32]),
                ("state_sequence_ids", vec![placement.state as i32]),
            ]
        } else {
            vec![
                ("active_groups", vec![groups as i32]),
                ("group_sequence_slots", nonempty(vec![0; groups])),
                ("group_source_positions", nonempty(absolute.clone())),
                ("group_rope_positions", nonempty(absolute)),
                ("compressed_slots", nonempty(slots)),
                ("active_sequences", vec![1]),
                ("sequence_offsets", vec![0, tokens as i32]),
                ("sequence_start_positions", vec![start as i32]),
                ("state_sequence_ids", vec![placement.state as i32]),
            ]
        };
        Ok((tables, groups))
    };
    let (c4_tables, _) = compressor_tables(4)?;
    let (c128_tables, _) = compressor_tables(128)?;
    let visible4: Vec<i32> = (start..end).map(|t| ((t + 1) / 4) as i32).collect();
    let mut c128_indices = vec![-1i32; tokens * c128_width];
    for (row, t) in (start..end).enumerate() {
        for j in 0..((t + 1) / 128).min(c128_width) {
            c128_indices[row * c128_width + j] = placement.group_slot(128, j)?;
        }
    }
    let c4_groups = end / 4;
    Ok(StepTables {
        decode: false,
        chunked: false,
        start,
        rows: tokens,
        swa_indices,
        swa_lengths: (start..end).map(|t| (t + 1).min(WINDOW) as i32).collect(),
        main_slots: positions.iter().map(|&p| slot(p as usize)).collect(),
        positions,
        c4_tables,
        c128_tables,
        c4_groups,
        c128_groups: end / 128,
        c4_page_table: placement.pages.clone(),
        c4_table_width: c4_groups.div_ceil(compressed_page_rows(4)).max(1),
        c4_table_stride: 0,
        c4_indexed_lengths: visible4.iter().map(|&v| v.min(index_topk as i32)).collect(),
        c4_visible: visible4,
        c128_lengths: (start..end).map(|t| ((t + 1) / 128) as i32).collect(),
        c128_indices,
    })
}

/// Tables for one decode row per sequence: `(placement, position)`.
pub(crate) fn decode_step(
    rows: &[(&super::pool::Placement, usize)],
    shape: &super::pool::PoolShape,
    index_topk: usize,
    c128_width: usize,
) -> anyhow::Result<StepTables> {
    let n = rows.len();
    let mut tables = StepTables {
        decode: true,
        chunked: false,
        start: 0,
        rows: n,
        positions: Vec::with_capacity(n),
        main_slots: Vec::with_capacity(n),
        swa_indices: vec![-1; n * WINDOW],
        swa_lengths: Vec::with_capacity(n),
        c4_tables: Vec::new(),
        c128_tables: Vec::new(),
        c4_groups: 0,
        c128_groups: 0,
        c4_page_table: Vec::new(),
        c4_table_width: 1,
        c4_table_stride: rows.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1),
        c4_visible: Vec::with_capacity(n),
        c4_indexed_lengths: Vec::with_capacity(n),
        c128_indices: vec![-1; n * c128_width],
        c128_lengths: Vec::with_capacity(n),
    };
    let (mut positions4, mut states, mut slots4, mut slots128) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (row, (placement, position)) in rows.iter().enumerate() {
        let position = *position;
        tables.positions.push(position as i64);
        tables.main_slots.push(placement.window_slot(shape, position));
        let start = position.saturating_sub(WINDOW - 1);
        for j in 0..WINDOW {
            if start + j <= position {
                tables.swa_indices[row * WINDOW + j] = placement.window_slot(shape, start + j) as i32;
            }
        }
        tables.swa_lengths.push((position + 1).min(WINDOW) as i32);
        positions4.push(position as i32);
        states.push(placement.state as i32);
        // The compressor reads a row's slot only when the row completes its
        // group; a group that cannot complete within capacity has no page.
        let slot = |ratio: usize| -> anyhow::Result<i32> {
            if (position + 1) % ratio == 0 {
                placement.group_slot(ratio, position / ratio)
            } else {
                Ok(placement.group_slot(ratio, position / ratio).unwrap_or(0))
            }
        };
        slots4.push(slot(4)?);
        slots128.push(slot(128)?);
        let visible4 = (position + 1) / 4;
        let visible128 = (position + 1) / 128;
        tables.c4_groups = tables.c4_groups.max(visible4);
        tables.c128_groups = tables.c128_groups.max(visible128);
        tables.c4_visible.push(visible4 as i32);
        tables.c4_indexed_lengths.push(visible4.min(index_topk) as i32);
        tables.c4_table_width = tables.c4_table_width.max(visible4.div_ceil(compressed_page_rows(4)));
        let mut pages = placement.pages.clone();
        pages.resize(tables.c4_table_stride, pages.last().copied().unwrap_or(0));
        tables.c4_page_table.extend(pages);
        for j in 0..visible128.min(c128_width) {
            tables.c128_indices[row * c128_width + j] = placement.group_slot(128, j)?;
        }
        tables.c128_lengths.push(visible128 as i32);
    }
    tables.c4_tables = vec![("positions", positions4.clone()), ("sequence_ids", states.clone()), ("compressed_slots", slots4)];
    tables.c128_tables = vec![("positions", positions4), ("sequence_ids", states), ("compressed_slots", slots128)];
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::super::pool::{PoolAllocator, PoolShape};
    use super::*;

    #[test]
    fn c4_context_planar_commits_cover_owned_pages_byte_exactly() -> anyhow::Result<()> {
        use crate::shared::peer_split::context::{commit_blocks, CommitPlane};
        let units = [4, 9, 0, 3];
        let record_bytes = compressed_page_bytes(4);
        let record_planes = [
            (576, 0, 64),
            (8, 64 * 576, 64),
            (64, 64 * 584, 1),
        ];
        let mut planes = Vec::new();
        for (row_bytes, base, rows_per_unit) in record_planes {
            planes.push(CommitPlane { first: 0, rows: units.len() * rows_per_unit,
                rows_per_unit, row_bytes, base, unit_bytes: record_bytes });
        }
        for (row_bytes, base) in [(128, 0), (4, 64 * 128)] {
            planes.push(CommitPlane { first: 0, rows: units.len() * 64,
                rows_per_unit: 64, row_bytes, base, unit_bytes: INDEX_PAGE_BYTES });
        }
        for rank in 0..2 {
            let commits = commit_blocks(rank, &units, &planes, units.len())?;
            for (plane_index, plane) in planes.iter().enumerate() {
                let table = &commits.table[plane_index * 3 * units.len()..];
                for entry in 0..commits.counts[plane_index] {
                    let source = table[entry] as usize;
                    let destination = table[units.len() + entry] as usize;
                    let bytes = table[2 * units.len() + entry] as usize;
                    let logical = source / plane.unit_bytes;
                    assert_eq!(logical % 2, rank);
                    assert_eq!(destination / plane.unit_bytes, (units[logical] >> 1) as usize);
                    assert_eq!(source % plane.unit_bytes, plane.base);
                    assert_eq!(bytes, plane.rows_per_unit * plane.row_bytes);
                }
            }
            for (range, unit_bytes) in [(0..3, record_bytes), (3..5, INDEX_PAGE_BYTES)] {
                for logical in (rank..units.len()).step_by(2) {
                    let mut coverage = vec![0u8; unit_bytes];
                    for plane_index in range.clone() {
                        let table = &commits.table[plane_index * 3 * units.len()..];
                        for entry in 0..commits.counts[plane_index] {
                            let source = table[entry] as usize;
                            if source / unit_bytes != logical { continue; }
                            let begin = source % unit_bytes;
                            let bytes = table[2 * units.len() + entry] as usize;
                            for count in &mut coverage[begin..begin + bytes] { *count += 1; }
                        }
                    }
                    assert!(coverage.iter().all(|&count| count == 1));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn c4_context_ownership_frontiers_do_not_split_compression_groups() {
        use crate::shared::peer_split::context::{page_owner, shard_rows};
        for source_tokens in 0..=1025 {
            let completed = source_tokens / 4;
            let counts = [shard_rows(completed, 64, 0), shard_rows(completed, 64, 1)];
            assert_eq!(counts[0] + counts[1], completed);
            for rank in 0..2 {
                assert_eq!(counts[rank], (0..completed).filter(|&group|
                    page_owner(group / 64) == rank).count());
            }
            if source_tokens % 4 == 3 {
                assert_eq!(page_owner(source_tokens / SOURCE_PAGE_TOKENS),
                    page_owner(completed / 64));
            }
        }
    }

    #[test]
    fn c4_context_continuation_commits_preserve_prior_planar_rows() -> anyhow::Result<()> {
        let units = [4, 9, 0, 3];
        let planes = [(576, 0, compressed_page_bytes(4)), (8, 64 * 576, compressed_page_bytes(4)),
            (128, 0, INDEX_PAGE_BYTES), (4, 64 * 128, INDEX_PAGE_BYTES)];
        for first in [0, 1, 63, 64, 65, 127, 128, 191] {
            for groups in [0, 1, 4, 64] {
                if first + groups > units.len() * 64 { continue; }
                for rank in 0..2 {
                    let commits = c4_context_commits(rank, first, groups, &units, 4)?;
                    for (plane, &(row_bytes, base, unit_bytes)) in planes.iter().enumerate() {
                        let table = &commits.table[plane * 12..];
                        let mut coverage = vec![false; units.len() * 64];
                        for i in 0..commits.counts[plane] {
                            let source = table[i] as usize;
                            let logical = source / unit_bytes;
                            let start = (source % unit_bytes - base) / row_bytes;
                            let count = table[8 + i] as usize / row_bytes;
                            assert_eq!(logical % 2, rank);
                            assert_eq!(table[4 + i] as usize / unit_bytes, (units[logical] >> 1) as usize);
                            for group in logical * 64 + start..logical * 64 + start + count {
                                assert!(!coverage[group]);
                                coverage[group] = true;
                            }
                        }
                        for (group, &covered) in coverage.iter().enumerate() {
                            assert_eq!(covered, group >= first && group < first + groups && (group / 64) % 2 == rank);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn c4_context_decode_and_verify_tables_preserve_replicated_state() -> anyhow::Result<()> {
        let shape = PoolShape::new(2, 4096, 16);
        let placement = super::super::pool::Placement::new(0, vec![4, 9, 0, 3]);
        for tables in [decode_step(&[(&placement, 255), (&placement, 259), (&placement, 771)],
            &shape, 512, 8)?, verify_step(&[(&placement, 250, 16)], &shape, 512, 8)?] {
            for rank in 0..2 {
                let shard = c4_context_tables(&tables, rank, shape.units, 1)?;
                assert_eq!(shard.c128_tables, tables.c128_tables);
                assert_eq!(shard.c128_indices, tables.c128_indices);
                assert_eq!(shard.main_slots, tables.main_slots);
                assert_eq!(shard.swa_indices, tables.swa_indices);
                assert_eq!(shard.positions, tables.positions);
                for row in 0..tables.rows {
                    assert_eq!(shard.c4_visible[row] as usize,
                        crate::shared::peer_split::context::shard_rows(tables.c4_visible[row] as usize, 64, rank));
                    assert_eq!(shard.swa_lengths[row], if rank == 1 { tables.swa_lengths[row] } else { 0 });
                }
                let name = if tables.chunked { "group_source_positions" } else { "positions" };
                let positions = &tables.c4_tables.iter().find(|(n, _)| *n == name).unwrap().1;
                let slots = &shard.c4_tables.iter().find(|(n, _)| *n == "compressed_slots").unwrap().1;
                let active = if tables.chunked { tables.c4_tables.iter().find(|(n, _)| *n == "active_groups")
                    .unwrap().1[0] as usize } else { tables.rows };
                for (&position, &slot) in positions.iter().zip(slots).take(active) {
                    if (position as usize / 256) % 2 == rank {
                        let original = placement.group_slot(4, position as usize / 4)?;
                        assert_eq!(slot, (original / 64 / 2) * 64 + original % 64);
                    } else {
                        assert!(slot >= (shape.units / 2 * 64) as i32);
                        assert!(slot < ((shape.units / 2 * 64) + tables.rows) as i32);
                    }
                }
            }
        }
        let short = super::super::pool::Placement::new(0, vec![4]);
        let tables = decode_step(&[(&short, 256)], &shape, 512, 8)?;
        for rank in 0..2 {
            assert!(c4_context_tables(&tables, rank, shape.units, 0).is_ok());
        }
        let invalid = super::super::pool::Placement::new(0, vec![5]);
        let tables = decode_step(&[(&invalid, 3)], &shape, 512, 8)?;
        assert!(c4_context_tables(&tables, 0, shape.units, 0).is_err());
        Ok(())
    }

    #[test]
    fn c4_context_gathered_prefill_preserves_replicated_tables() -> anyhow::Result<()> {
        let shape = PoolShape::new(2, 4096, 16);
        let placement = super::super::pool::Placement::new(1, vec![4, 9, 0, 3]);
        for (start, rows) in [(0, 300), (1, 2), (250, 16), (300, 100), (509, 264)] {
            let tables = prefill_step(&placement, &shape, start, rows, 512, 8)?;
            let gathered = c4_context_prefill_tables(&tables)?;
            assert_eq!(gathered.c4_page_table, vec![0, 1, 2, 3]);
            assert_eq!(gathered.c4_visible, tables.c4_visible);
            assert_eq!(gathered.c4_indexed_lengths, tables.c4_indexed_lengths);
            assert_eq!(gathered.c4_table_width, tables.c4_table_width);
            assert_eq!(gathered.c128_tables, tables.c128_tables);
            assert_eq!(gathered.c128_indices, tables.c128_indices);
            assert_eq!(gathered.c128_lengths, tables.c128_lengths);
            assert_eq!(gathered.main_slots, tables.main_slots);
            assert_eq!(gathered.swa_indices, tables.swa_indices);
            assert_eq!(gathered.swa_lengths, tables.swa_lengths);
            for ((name, before), (_, after)) in tables.c4_tables.iter().zip(&gathered.c4_tables) {
                if *name == "compressed_slots" {
                    let groups = (start + rows) / 4 - start / 4;
                    let expected: Vec<_> = (start / 4..(start + rows) / 4).map(|g| g as i32).collect();
                    assert_eq!(&after[..groups], expected.as_slice());
                } else {
                    assert_eq!(before, after);
                }
            }
        }
        let tables = decode_step(&[(&placement, 3)], &shape, 512, 8)?;
        assert!(c4_context_prefill_tables(&tables).is_err());
        Ok(())
    }

    #[test]
    fn c4_context_discarded_groups_have_distinct_scratch_rows() -> anyhow::Result<()> {
        let shape = PoolShape::new(2, 4096, 16);
        let a = super::super::pool::Placement::new(0, vec![4, 9, 0, 3]);
        let b = super::super::pool::Placement::new(1, vec![6, 11, 2, 5]);
        let tables = verify_step(&[(&a, 0, 268), (&b, 0, 268)], &shape, 512, 8)?;
        let scratch = shape.units / 2 * 64;
        for rank in 0..2 {
            let shard = c4_context_tables(&tables, rank, shape.units, 0)?;
            let active = tables.c4_tables.iter().find(|(n, _)| *n == "active_groups").unwrap().1[0] as usize;
            let slots = &shard.c4_tables.iter().find(|(n, _)| *n == "compressed_slots").unwrap().1;
            let discarded: Vec<_> = slots.iter().take(active).copied().filter(|&s| s as usize >= scratch).collect();
            let unique: std::collections::HashSet<_> = discarded.iter().copied().collect();
            assert_eq!(unique.len(), discarded.len());
            assert!(!discarded.is_empty());
            assert!(discarded.iter().all(|&s| (s as usize) < scratch + tables.rows));
            if rank == 1 { assert!(discarded.len() > 64); }
        }
        Ok(())
    }

    #[test]
    fn prefill_and_decode_tables_use_physical_slots() -> anyhow::Result<()> {
        let shape = PoolShape::new(2, 4096, 64);
        let mut pool = PoolAllocator::new(shape);
        let _other = pool.admit(1000)?;
        let seq = pool.admit(1000)?;
        let base = (seq.state * shape.ring_pages * SOURCE_PAGE_TOKENS) as i32;
        let pre = prefill_step(&seq, &shape, 0, 300, 512, 1024)?;
        assert_eq!(&pre.swa_indices[..3], &[base, -1, -1]);
        assert_eq!(pre.swa_indices[200 * WINDOW], base + 73);
        assert_eq!((pre.c4_groups, pre.c128_groups), (75, 2));
        assert_eq!(pre.c4_table_width, 2);
        assert_eq!(pre.c4_tables[6].1, vec![seq.state as i32]);
        assert_eq!(&pre.c128_indices[255 * 1024..255 * 1024 + 3],
            &[seq.group_slot(128, 0)?, seq.group_slot(128, 1)?, -1]);
        let cont = prefill_step(&seq, &shape, 300, 100, 512, 1024)?;
        assert_eq!(cont.c4_tables.len(), 9);
        // Positions 303, 307, ... complete groups 75, 76, ...
        assert_eq!(cont.c4_tables[2].1[0], 300);
        assert_eq!(cont.c4_tables[4].1[0], seq.group_slot(4, 75)?);
        assert_eq!(cont.c4_tables[7].1, vec![300]);
        assert_eq!(cont.swa_indices[0], base + 300 - 127);
        let dec = decode_step(&[(&seq, 300)], &shape, 512, 1024)?;
        // Capacity 1000 has no C128 page for group 7; position 998 does not complete it.
        assert!(decode_step(&[(&seq, 998)], &shape, 512, 1024).is_ok());
        assert_eq!(dec.main_slots, vec![i64::from(base) + 300]);
        assert_eq!(dec.c4_visible, vec![75]);
        assert_eq!(dec.c4_tables[1].1, vec![seq.state as i32]);
        assert_eq!(compressed_page_bytes(4), 37_440);
        assert_eq!(compressed_page_bytes(128), 1_728);
        Ok(())
    }
}

/// Commit only newly completed C4 rows from a gathered prefill view. V4
/// records and index keys/scales are planar, not interleaved 584-byte rows.
fn c4_context_commits(rank: usize, first_group: usize, groups: usize, units: &[u32], limit: usize)
    -> anyhow::Result<crate::shared::peer_split::context::CommitBlocks> {
    use crate::shared::peer_split::context::{commit_blocks, CommitPlane};
    let rows_per_unit = compressed_page_rows(4);
    let planes = [(576, 0, compressed_page_bytes(4)),
        (8, rows_per_unit * 576, compressed_page_bytes(4)),
        (128, 0, INDEX_PAGE_BYTES), (4, rows_per_unit * 128, INDEX_PAGE_BYTES)]
        .map(|(row_bytes, base, unit_bytes)| CommitPlane {
            first: first_group, rows: groups, rows_per_unit, row_bytes, base, unit_bytes,
        });
    commit_blocks(rank, units, &planes, limit)
}

/// Head-split prefill over a logical, gathered C4 view. Only C4 output slots
/// and its index page table change; window rings, C128 and carry remain physical.
fn c4_context_prefill_tables(tables: &StepTables) -> anyhow::Result<StepTables> {
    anyhow::ensure!(!tables.decode && !tables.chunked, "C4 gathered tables need prefill");
    let mut out = tables.clone();
    let positions_name = if tables.start == 0 { "group_source_starts" } else { "group_source_positions" };
    let positions = tables.c4_tables.iter().find(|(name, _)| *name == positions_name)
        .ok_or_else(|| anyhow::anyhow!("C4 compressor lacks {positions_name}"))?.1.clone();
    let active = usize::try_from(*tables.c4_tables.iter().find(|(name, _)| *name == "active_groups")
        .and_then(|(_, values)| values.first()).ok_or_else(|| anyhow::anyhow!("C4 active groups missing"))?)?;
    let slots = &mut out.c4_tables.iter_mut().find(|(name, _)| *name == "compressed_slots")
        .ok_or_else(|| anyhow::anyhow!("C4 compressor lacks slots"))?.1;
    anyhow::ensure!(positions.len() >= active && slots.len() >= active && active <= tables.rows,
        "C4 prefill compressor table too short");
    let end = tables.start.checked_add(tables.rows)
        .ok_or_else(|| anyhow::anyhow!("C4 prefill range overflow"))?;
    for (&position, slot) in positions.iter().zip(slots).take(active) {
        let position = usize::try_from(position)?;
        anyhow::ensure!(position % 4 == 0 && position.checked_add(4).is_some_and(|p|
            p > tables.start && p <= end), "C4 group outside prefill range");
        *slot = i32::try_from(position / 4)?;
    }
    out.c4_page_table = (0..tables.c4_page_table.len()).map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(out)
}

/// Decode/verify metadata for one C4 token shard. C128, the window ring and
/// compressor carry stay replicated. Non-owner compressed writes land in a
/// scratch row per active compressor CTA; exactly one rank contributes the
/// window union. Both C4 caches need `ceil(rows / 64)` extra scratch pages.
fn c4_context_tables(tables: &StepTables, rank: usize, units: usize, window_rank: usize)
    -> anyhow::Result<StepTables> {
    use crate::shared::peer_split::context::{local_pages, page_owner, shard_rows};
    anyhow::ensure!(tables.decode && rank < 2 && window_rank < 2 && units > 0 && units % 2 == 0,
        "C4 context metadata needs decode/verify, two ranks and an even parity pool");
    let rows_per_unit = compressed_page_rows(4);
    let local_units = units / 2;
    let scratch_slot = local_units.checked_mul(rows_per_unit)
        .ok_or_else(|| anyhow::anyhow!("C4 scratch slot overflow"))?;
    let scratch_end = scratch_slot.checked_add(tables.rows)
        .ok_or_else(|| anyhow::anyhow!("C4 scratch range overflow"))?;
    i32::try_from(scratch_end)?;
    let pool_slots = units.checked_mul(rows_per_unit)
        .ok_or_else(|| anyhow::anyhow!("C4 pool slot overflow"))?;
    let mut out = tables.clone();
    out.c4_page_table.clear();
    out.c4_table_stride = tables.c4_table_stride.div_ceil(2).max(1);
    out.c4_table_width = tables.c4_table_width.div_ceil(2).max(1);
    anyhow::ensure!(tables.c4_visible.len() == tables.rows && tables.c4_table_stride > 0
        && tables.c4_page_table.len() == tables.rows * tables.c4_table_stride,
        "C4 decode page table does not cover its rows");
    for row in 0..tables.rows {
        let visible = usize::try_from(tables.c4_visible[row])?;
        let used = visible.div_ceil(rows_per_unit);
        anyhow::ensure!(used <= tables.c4_table_stride, "C4 visibility past row table");
        let ids = tables.c4_page_table[row * tables.c4_table_stride..][..used].iter()
            .map(|&id| {
                anyhow::ensure!(id >= 0 && (id as usize) < units, "C4 page outside parity pool");
                Ok(id as u32)
            }).collect::<anyhow::Result<Vec<_>>>()?;
        let local = local_pages(&ids, rank)?;
        out.c4_page_table.extend(local.into_iter().map(|id| id as i32));
        out.c4_page_table.resize((row + 1) * out.c4_table_stride, local_units as i32);
        out.c4_visible[row] = i32::try_from(shard_rows(visible, rows_per_unit, rank))?;
    }
    let positions_name = if tables.chunked { "group_source_positions" } else { "positions" };
    let positions = tables.c4_tables.iter().find(|(name, _)| *name == positions_name)
        .ok_or_else(|| anyhow::anyhow!("C4 compressor lacks {positions_name}"))?.1.clone();
    let active = if tables.chunked {
        usize::try_from(*tables.c4_tables.iter().find(|(name, _)| *name == "active_groups")
            .and_then(|(_, values)| values.first()).ok_or_else(|| anyhow::anyhow!("C4 active groups missing"))?)?
    } else { tables.rows };
    let slots = &mut out.c4_tables.iter_mut().find(|(name, _)| *name == "compressed_slots")
        .ok_or_else(|| anyhow::anyhow!("C4 compressor lacks slots"))?.1;
    anyhow::ensure!(active <= tables.rows && positions.len() >= active && slots.len() >= active,
        "C4 compressor table too short or too many active groups");
    for (group, (position, slot)) in positions.iter().zip(slots).take(active).enumerate() {
        let position = usize::try_from(*position)?;
        if !tables.chunked && position % 4 != 3 {
            // Decode updates carry on every row, but reads the output slot
            // only when a group completes. Capacity may end mid-group.
            *slot = i32::try_from(scratch_slot + group)?;
            continue;
        }
        anyhow::ensure!(*slot >= 0 && (*slot as usize) < pool_slots
            && (*slot as usize / rows_per_unit) % 2 == page_owner(position / SOURCE_PAGE_TOKENS),
            "C4 write on wrong parity page");
        if page_owner(position / SOURCE_PAGE_TOKENS) == rank {
            *slot = (*slot / rows_per_unit as i32 / 2) * rows_per_unit as i32
                + *slot % rows_per_unit as i32;
        } else {
            // Continuation launches one CTA per completed group, and decode
            // one per sequence: a shared scratch row would race across CTAs.
            *slot = i32::try_from(scratch_slot + group)?;
        }
    }
    if rank != window_rank { out.swa_lengths.fill(0); }
    Ok(out)
}

/// Tables for a decode-shaped step in which each sequence contributes
/// `count` consecutive rows from `start` (its current length): attention
/// tables per row as in [`decode_step`], compressor tables for the
/// continuation program (one ordered chunk per sequence). Grid bounds are the
/// row count, so a captured step depends only on it.
pub(crate) fn verify_step(
    sequences: &[(&super::pool::Placement, usize, usize)],
    shape: &super::pool::PoolShape,
    index_topk: usize,
    c128_width: usize,
) -> anyhow::Result<StepTables> {
    let rows: Vec<(&super::pool::Placement, usize)> = sequences.iter()
        .flat_map(|&(placement, start, count)| (start..start + count).map(move |position| (placement, position)))
        .collect();
    let mut tables = decode_step(&rows, shape, index_topk, c128_width)?;
    tables.chunked = true;
    let n = rows.len();
    let continuation = |ratio: usize| -> anyhow::Result<Vec<(&'static str, Vec<i32>)>> {
        let (mut group_sequences, mut sources, mut slots) = (Vec::new(), Vec::new(), Vec::new());
        let (mut offsets, mut starts, mut states) = (vec![0i32], Vec::new(), Vec::new());
        for (index, &(placement, start, count)) in sequences.iter().enumerate() {
            for position in (start..start + count).filter(|p| p % ratio == ratio - 1) {
                group_sequences.push(index as i32);
                sources.push((position + 1 - ratio) as i32);
                slots.push(placement.group_slot(ratio, position / ratio)?);
            }
            offsets.push(offsets.last().unwrap() + count as i32);
            starts.push(start as i32);
            states.push(placement.state as i32);
        }
        let groups = group_sequences.len() as i32;
        let pad = |mut v: Vec<i32>, len: usize| { v.resize(len.max(v.len()).max(1), 0); v };
        Ok(vec![
            ("active_groups", vec![groups]),
            ("group_sequence_slots", pad(group_sequences, n)),
            ("group_source_positions", pad(sources.clone(), n)),
            ("group_rope_positions", pad(sources, n)),
            ("compressed_slots", pad(slots, n)),
            ("active_sequences", vec![sequences.len() as i32]),
            ("sequence_offsets", pad(offsets, n + 1)),
            ("sequence_start_positions", pad(starts, n)),
            ("state_sequence_ids", pad(states, n)),
        ])
    };
    tables.c4_tables = continuation(4)?;
    tables.c128_tables = continuation(128)?;
    Ok(tables)
}
