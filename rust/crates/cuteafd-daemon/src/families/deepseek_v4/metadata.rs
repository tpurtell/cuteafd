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
