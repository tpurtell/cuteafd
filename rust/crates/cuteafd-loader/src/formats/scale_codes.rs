//! Lossless V4.1 expert scales: N256/K128 words with a compact final row tile.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScaleCodeError {
    #[error("{tensor}: invalid scale-code row geometry")]
    Geometry { tensor: String },
    #[error("{tensor}: row {row} UE8M0 span {span} exceeds 15; unsupported V4.1 scale codes")]
    Span { tensor: String, row: usize, span: u8 },
}

/// A source rectangle. The original row number is retained in refusal messages.
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub tensor: &'a str,
    pub bytes: &'a [u8],
    pub rows: usize,
    pub pitch: usize,
    pub columns: usize,
    pub first_row: usize,
}

impl Rows<'_> {
    /// Pack into an already admitted plane. Gate/up halves share one plane but
    /// retain their independently padded row positions and source tensor names.
    pub fn pack(self, output: &mut [u8], stored_rows: usize, start_row: usize) -> Result<(), ScaleCodeError> {
        let tiles = self.columns.div_ceil(4);
        let code_bytes = stored_rows.checked_mul(tiles).and_then(|n| n.checked_mul(2));
        if self.columns == 0 || self.rows == 0 || self.pitch < self.columns || stored_rows % 32 != 0
            || start_row.checked_add(self.rows).is_none_or(|n| n > stored_rows)
            || self.rows.checked_sub(1).and_then(|n| n.checked_mul(self.pitch))
                .and_then(|n| n.checked_add(self.columns)).is_none_or(|n| n > self.bytes.len())
            || code_bytes.and_then(|n| n.checked_add(stored_rows)) != Some(output.len()) {
            return Err(ScaleCodeError::Geometry { tensor: self.tensor.into() });
        }
        let code_bytes = code_bytes.unwrap();
        for row in 0..self.rows {
            let source = &self.bytes[row * self.pitch..row * self.pitch + self.columns];
            let low = *source.iter().min().unwrap();
            let high = *source.iter().max().unwrap();
            if high - low > 15 {
                return Err(ScaleCodeError::Span { tensor: self.tensor.into(), row: self.first_row + row, span: high - low });
            }
            let resident = start_row + row;
            output[code_bytes + resident] = low;
            for (tile, block) in source.chunks(4).enumerate() {
                let mut code = 0u16;
                for (byte, &value) in block.iter().enumerate() { code |= u16::from(value - low) << (byte * 4); }
                let tile_start = resident / 256 * 256;
                let tile_rows = (stored_rows - tile_start).min(256);
                let index = (tile_start * tiles + tile * tile_rows + resident % 256) * 2;
                output[index..index + 2].copy_from_slice(&code.to_le_bytes());
            }
        }
        Ok(())
    }
}

pub fn plane_bytes(rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(columns.div_ceil(4).checked_mul(2)?.checked_add(1)?)
}

/// Fixed resident layout of an MXFP4 package using coded UE8M0 scales.
/// Aligned payloads retain N256/K128; exact tails use N128. FC2 scales use logical K.
pub fn expert_planes(hidden: usize, intermediate: usize, stored: usize) -> Option<[usize; 4]> {
    if hidden == 0 || hidden % 256 != 0 || intermediate == 0 || intermediate % 32 != 0
        || Some(stored) != cuteafd_core::mxfp4_layouts::stored_width(hidden, intermediate) {
        return None;
    }
    let values = hidden.checked_mul(stored)?;
    Some([values, plane_bytes(stored.checked_mul(2)?, hidden / 32)?,
        values / 2, plane_bytes(hidden, intermediate / 32)?])
}

pub fn expert_bytes(experts: u64, hidden: u64, intermediate: u64) -> Option<u64> {
    let (hidden, intermediate) = (usize::try_from(hidden).ok()?, usize::try_from(intermediate).ok()?);
    let stored = cuteafd_core::mxfp4_layouts::stored_width(hidden, intermediate)?;
    let bytes = expert_planes(hidden, intermediate, stored)?.into_iter()
        .try_fold(0usize, |sum, n| sum.checked_add(n))?;
    u64::try_from(bytes).ok()?.checked_mul(experts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_offsets_and_zero_k_tail() {
        let source = [117, 118, 124, 119, 120, 118];
        let mut output = vec![0; plane_bytes(256, 6).unwrap()];
        Rows { tensor: "layers.3.ffn.experts.0.w2.scale", bytes: &source,
            rows: 1, pitch: 6, columns: 6, first_row: 0 }.pack(&mut output, 256, 0).unwrap();
        assert_eq!(&output[..2], &0x2710u16.to_le_bytes());
        assert_eq!(&output[512..514], &0x0013u16.to_le_bytes());
        assert_eq!(output[1024], 117);
        assert!(output[1025..].iter().all(|&v| v == 0));
    }
    #[test]
    fn refuse_span_16_names_original_tensor_and_row() {
        let mut output = vec![0; plane_bytes(256, 4).unwrap()];
        let error = Rows { tensor: "mtp.1.ffn.experts.7.w1.scale", bytes: &[111, 127, 111, 111],
            rows: 1, pitch: 4, columns: 4, first_row: 384 }.pack(&mut output, 256, 0).unwrap_err();
        assert_eq!(error.to_string(), "mtp.1.ffn.experts.7.w1.scale: row 384 UE8M0 span 16 exceeds 15; unsupported V4.1 scale codes");
    }
    #[test]
    fn compact_n_tail_reconstructs_every_scale_and_never_overlaps_bases() {
        for rows in [32usize, 64, 96, 288, 576, 640, 672, 704, 832, 1152, 1344] {
            for columns in [1usize, 3, 4, 5, 9, 18, 160, 224] {
                let source: Vec<u8> = (0..rows).flat_map(|row| (0..columns)
                    .map(move |col| 110 + (row % 8) as u8 + ((row + col) % 16) as u8)).collect();
                let mut output = vec![0; plane_bytes(rows, columns).unwrap()];
                Rows { tensor: "w.scale", bytes: &source, rows, pitch: columns,
                    columns, first_row: 0 }.pack(&mut output, rows, 0).unwrap();
                let tiles = columns.div_ceil(4);
                let code_bytes = rows * tiles * 2;
                let mut word = 0;
                // Independent tile-order decoder rather than the packer's row index.
                for start in (0..rows).step_by(256) {
                    for tile in 0..tiles {
                        for row in start..(start + 256).min(rows) {
                            let code = u16::from_le_bytes(output[word * 2..word * 2 + 2].try_into().unwrap());
                            for nibble in 0..4 {
                                let col = tile * 4 + nibble;
                                let delta = ((code >> (nibble * 4)) & 15) as u8;
                                if col < columns {
                                    assert_eq!(delta + output[code_bytes + row], source[row * columns + col]);
                                } else {
                                    assert_eq!(delta, 0);
                                }
                            }
                            word += 1;
                        }
                    }
                }
                assert_eq!(word * 2, code_bytes);
            }
        }
    }

    #[test]
    fn exact_gate_up_halves_share_a_compact_final_tile() {
        let width = 288;
        let rows = 2 * width;
        let mut output = vec![0; plane_bytes(rows, 5).unwrap()];
        for (start_row, base) in [(0, 111), (width, 117)] {
            Rows { tensor: "w13.scale", bytes: &vec![base; width * 5], rows: width,
                pitch: 5, columns: 5, first_row: 0 }.pack(&mut output, rows, start_row).unwrap();
        }
        assert!(output[..rows * 4].iter().all(|&byte| byte == 0));
        assert!(output[rows * 4..rows * 4 + width].iter().all(|&byte| byte == 111));
        assert!(output[rows * 4 + width..].iter().all(|&byte| byte == 117));
    }

    #[test]
    fn invalid_compact_geometry_is_refused_before_writing() {
        for stored in [0usize, 31, 33, 511] {
            let mut output = vec![0xa5; plane_bytes(stored, 4).unwrap()];
            assert!(Rows { tensor: "w.scale", bytes: &[111; 4], rows: 1,
                pitch: 4, columns: 4, first_row: 0 }.pack(&mut output, stored, 0).is_err());
            assert!(output.iter().all(|&byte| byte == 0xa5));
        }
    }

    #[test]
    fn independent_gate_up_padding() {
        let mut output = vec![0; plane_bytes(1280, 4).unwrap()];
        for (first, base) in [(0, 111), (640, 117)] {
            Rows { tensor: "w13.scale", bytes: &vec![base; 576 * 4], rows: 576, pitch: 4,
                columns: 4, first_row: 0 }.pack(&mut output, 1280, first).unwrap();
        }
        let bases = &output[2560..];
        assert!(bases[..576].iter().all(|&v| v == 111));
        assert!(bases[576..640].iter().all(|&v| v == 0));
        assert!(bases[640..1216].iter().all(|&v| v == 117));
        assert!(bases[1216..].iter().all(|&v| v == 0));
    }
}
