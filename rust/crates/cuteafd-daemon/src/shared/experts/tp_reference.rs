//! Independent checkpoint-source math for the real-worker rank-sum gate.
//! No candidate unit maps, packers, lookup tables or GPU kernels participate.
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, ExpertGeometry};
use cuteafd_loader::{OfficialV41Catalog, V41Exl3ProjectionKind};
use cuteafd_loader::formats::fp8_experts::{ExpertFormat, Fp8Projection};
use cuteafd_transport::ExpertV2Dtype;

pub(super) fn bf16(value: f64) -> f64 {
    let bits = (value as f32).to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000) as f64
}

fn f16(bits: u16) -> f64 {
    let (sign, exponent, fraction) = (bits >> 15, (bits >> 10) & 31, bits & 1023);
    let magnitude = match exponent {
        0 => f64::from(fraction) * 2f64.powi(-24),
        31 if fraction == 0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1. + f64::from(fraction) / 1024.) * 2f64.powi(i32::from(exponent) - 15),
    };
    if sign == 0 { magnitude } else { -magnitude }
}

fn e4m3(bits: u8) -> Result<f64> {
    ensure!(bits & 127 != 127, "FP8 reference encountered NaN");
    let (exponent, fraction) = ((bits >> 3) & 15, bits & 7);
    let magnitude = if exponent == 0 { f64::from(fraction) * 2f64.powi(-9) }
        else { (1. + f64::from(fraction) / 8.) * 2f64.powi(i32::from(exponent) - 7) };
    Ok(if bits & 128 == 0 { magnitude } else { -magnitude })
}

fn element(format: ExpertFormat, weights: &[u8], scales: &[u8], columns: usize,
    row: usize, col: usize) -> Result<f64> {
    match format {
        ExpertFormat::Mxfp4 => {
            let code = (weights[row * (columns / 2) + col / 2] >> (4 * (col % 2))) & 15;
            let exponent = scales[row * (columns / 32) + col / 32];
            ensure!(exponent != 255, "MXFP4 reference encountered a NaN scale");
            let value = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(code & 7) as usize];
            Ok(if code & 8 == 0 { value } else { -value } * 2f64.powi(i32::from(exponent) - 127))
        }
        ExpertFormat::Fp8Block128 => {
            let index = ((row / 128) * columns.div_ceil(128) + col / 128) * 4;
            let scale = f32::from_le_bytes(scales[index..index + 4].try_into()?);
            ensure!(scale.is_finite() && scale > 0., "invalid FP8 source block scale");
            // The serving BF16 MMA widens the scaled checkpoint value first.
            Ok(bf16(e4m3(weights[row * columns + col])? * f64::from(scale)))
        }
        _ => anyhow::bail!("rank-sum source decoder is not qualified for {format:?}"),
    }
}

pub(super) fn input(geometry: ExpertGeometry, row: usize) -> Vec<(usize, f64)> {
    (0..8).map(|slot| ((slot * 769 + 31) % geometry.hidden as usize,
        if (row + slot) % 2 == 0 { 0.5 } else { -0.25 })).collect()
}

// Scalar wire oracle: BF16 source rounding, power-of-two range scaling, then
// E4M3 nearest-even. This uses no candidate GPU quantizer or packed weights.
fn quantize_input(values: &[f64], dtype: ExpertV2Dtype) -> Result<Vec<u8>> {
    let pitch = dtype.row_bytes(values.len())?;
    let values: Vec<_> = values.iter().copied().map(bf16).collect();
    if dtype == ExpertV2Dtype::Bf16 {
        return Ok(values.iter().flat_map(|&value|
            (((value as f32).to_bits() >> 16) as u16).to_le_bytes()).collect());
    }
    ensure!(dtype == ExpertV2Dtype::Fp8E4m3Ue8m0K32, "unsupported gate input dtype");
    let mut wire = vec![0; pitch];
    for (group, block) in values.chunks_exact(32).enumerate() {
        let amax = block.iter().fold(1e-4f32, |amax, &v| amax.max((v as f32).abs()));
        let bits = (amax * (1f32 / 448.)).to_bits();
        let rounded = if bits & 0x7fffff == 0 { bits }
            else { bits.wrapping_add(0x800000) & 0x7f800000 };
        let exponent = ((rounded >> 23) & 255) as u8;
        ensure!((1..255).contains(&exponent), "nonfinite gate input scale");
        wire[values.len() + group] = exponent;
        let scale = 2f64.powi(i32::from(exponent) - 127);
        for (col, &value) in block.iter().enumerate() {
            let magnitude = (value / scale).abs().min(448.);
            let mut best = 0u8;
            let mut distance = f64::INFINITY;
            for code in 0..=126u8 {
                let delta = (e4m3(code)? - magnitude).abs();
                if delta < distance || (delta == distance && code & 1 == 0) {
                    best = code;
                    distance = delta;
                }
            }
            wire[group * 32 + col] = best | if value.is_sign_negative() { 128 } else { 0 };
        }
    }
    Ok(wire)
}

pub(super) fn wire_input(geometry: ExpertGeometry, row: usize, dtype: ExpertV2Dtype) -> Result<Vec<u8>> {
    let mut values = vec![0.; geometry.hidden as usize];
    for (column, value) in input(geometry, row) { values[column] = value; }
    quantize_input(&values, dtype)
}

fn decoded_input(geometry: ExpertGeometry, row: usize, dtype: ExpertV2Dtype) -> Result<Vec<(usize, f64)>> {
    let wire = wire_input(geometry, row, dtype)?;
    let mut values = Vec::new();
    for col in 0..geometry.hidden as usize {
        let value = match dtype {
            ExpertV2Dtype::Bf16 => f32::from_bits(u32::from(u16::from_le_bytes(
                [wire[col * 2], wire[col * 2 + 1]])) << 16) as f64,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32 => {
                let exponent = wire[geometry.hidden as usize + col / 32];
                ensure!(exponent != 255, "NaN gate wire scale");
                e4m3(wire[col])? * 2f64.powi(i32::from(exponent) - 127)
            }
            _ => anyhow::bail!("unsupported gate input dtype"),
        };
        if value != 0. { values.push((col, value)); }
    }
    Ok(values)
}

pub(super) fn expert(geometry: ExpertGeometry, layer: usize, row: usize, slot: usize) -> usize {
    (layer * 17 + (row % 2) * geometry.topk as usize + slot * 37) % geometry.experts as usize
}

pub(super) fn route_weight(geometry: ExpertGeometry, slot: usize) -> f32 {
    let k = geometry.topk as usize;
    (slot + 1) as f32 / (k * (k + 1) / 2) as f32
}

pub(super) fn outputs(geometry: ExpertGeometry) -> Vec<usize> {
    let h = geometry.hidden as usize;
    let mut columns = vec![0, 1, 31, 32, 127, 255, h - 2, h - 1];
    columns.extend((1..=8).map(|part| part * (h - 1) / 9));
    columns.sort_unstable();
    columns.dedup();
    columns
}

fn source(catalog: &OfficialV41Catalog, name: &str, dtype: DType, shape: &[usize]) -> Result<Vec<u8>> {
    let tensor = catalog.tensor(name)?;
    ensure!(tensor.metadata.dtype == dtype && tensor.metadata.shape == shape,
        "source oracle tensor header mismatch: {name}");
    let mut bytes = vec![0; usize::try_from(tensor.metadata.byte_length)?];
    catalog.read_projection_once(name, &mut bytes)?;
    Ok(bytes)
}

fn native_projections(catalog: &OfficialV41Catalog, layer: usize, id: usize)
    -> Result<(ExpertFormat, Vec<(Vec<u8>, Vec<u8>)>)> {
    if let Some(tensors) = catalog.fp8() {
        let tensors = tensors.for_layer(layer)?;
        let mut projections = Vec::new();
        for projection in Fp8Projection::ALL {
            let (w, s) = tensors.projection_bytes(projection);
            let (mut weights, mut scales) = (vec![0; w], vec![0; s]);
            tensors.read_projection_once(layer, id, projection, &mut weights, &mut scales)?;
            projections.push((weights, scales));
        }
        return Ok((tensors.format(), projections));
    }
    ensure!(catalog.exl3().is_none() && catalog.nvfp4().is_none(), "expected native MXFP4 source");
    let shape = catalog.routed_experts();
    let (h, i) = (shape.hidden, shape.intermediate);
    let mut projections = Vec::new();
    for (stem, rows, columns) in [("w1", i, h), ("w3", i, h), ("w2", h, i)] {
        let name = format!("layers.{layer}.ffn.experts.{id}.{stem}");
        projections.push((source(catalog, &format!("{name}.weight"), DType::I8, &[rows, columns / 2])?,
            source(catalog, &format!("{name}.scale"), DType::F8E8M0, &[rows, columns / 32])?));
    }
    Ok((ExpertFormat::Mxfp4, projections))
}

fn native_expert(catalog: &OfficialV41Catalog, geometry: ExpertGeometry, layer: usize,
    id: usize, input: &[(usize, f64)], columns: &[usize]) -> Result<Vec<f64>> {
    let (format, projections) = native_projections(catalog, layer, id)?;
    let (h, i) = (geometry.hidden as usize, geometry.intermediate as usize);
    let activation = catalog.fp8().and_then(|t| t.activation(layer)).unwrap_or(geometry.activation());
    let mut hidden = vec![0.; i];
    for (col, hidden) in hidden.iter_mut().enumerate() {
        let mut gu = [0.; 2];
        for projection in 0..2 {
            for &(index, value) in input {
                gu[projection] += element(format, &projections[projection].0,
                    &projections[projection].1, h, col, index)? * value;
            }
        }
        *hidden = f64::from(activation.evaluate(bf16(gu[0]) as f32, bf16(gu[1]) as f32));
    }
    columns.iter().map(|&column| {
        hidden.iter().enumerate().try_fold(0., |sum, (col, &value)|
            Ok(sum + element(format, &projections[2].0, &projections[2].1, i, column, col)? * value))
    }).collect()
}

fn hadamard(values: &mut [f64]) {
    for block in values.chunks_exact_mut(128) {
        let mut step = 1;
        while step < 128 {
            for pair in block.chunks_exact_mut(2 * step) {
                for j in 0..step {
                    let (a, b) = (pair[j], pair[j + step]);
                    pair[j] = a + b;
                    pair[j + step] = a - b;
                }
            }
            step *= 2;
        }
        for value in block { *value /= 128f64.sqrt(); }
    }
}

// Tail-biting MCG code and lane order, independently ported from the public
// GPTQModel reference (Apache-2.0), as in python/tools/lib/exl3_reference.py.
fn trellis_tile(bytes: &[u8], bits: usize) -> Result<[f64; 256]> {
    ensure!(matches!(bits, 2..=5) && bytes.len() == 32 * bits, "invalid EXL3 tile");
    let words: Vec<_> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    let mut symbols = [0u32; 256];
    for position in 0..16 {
        let (word, shift) = (position * bits / 16, position * bits % 16);
        for lane in 0..16 {
            let read = |word: usize| u32::from(words[(lane * bits + word) ^ 1]);
            let value = if shift + bits <= 16 { read(word) >> (16 - shift - bits) }
                else { (read(word) << (shift + bits - 16)) | (read(word + 1) >> (32 - shift - bits)) };
            symbols[lane * 16 + position] = value & ((1 << bits) - 1);
        }
    }
    let mut state = 0u32;
    for &symbol in &symbols[256 - (16usize.div_ceil(bits) - 1)..] {
        state = ((state << bits) | symbol) & 0xffff;
    }
    let mut decoded = [0.; 256];
    for (index, &symbol) in symbols.iter().enumerate() {
        state = ((state << bits) | symbol) & 0xffff;
        let raw = 0x3b603b60 ^ (state.wrapping_mul(0xcbac1fed) & 0x8fff8fff);
        let lane = index / 8;
        let (row, column) = ((lane % 4) * 2, lane / 4);
        let offsets = [row * 16 + column, (row + 1) * 16 + column,
            (row + 8) * 16 + column, (row + 9) * 16 + column,
            row * 16 + column + 8, (row + 1) * 16 + column + 8,
            (row + 8) * 16 + column + 8, (row + 9) * 16 + column + 8];
        decoded[offsets[index % 8]] = f16(raw as u16) + f16((raw >> 16) as u16);
    }
    Ok(decoded)
}

fn exl3_projection(catalog: &OfficialV41Catalog, name: &str, input: &[f64],
    columns: Option<&[usize]>) -> Result<Vec<f64>> {
    let projection = catalog.exl3().context("EXL3 oracle needs a source manifest")?
        .projections.get(name).context("missing source EXL3 projection")?;
    let (k, n) = (projection.input_features, projection.output_features);
    ensure!(input.len() == k && k % 128 == 0 && n % 128 == 0, "EXL3 oracle geometry");
    let trellis = source(catalog, &format!("{name}.trellis"), DType::I16, &projection.trellis_shape())?;
    let suh = source(catalog, &format!("{name}.suh"), DType::F16, &[k])?;
    let svh = source(catalog, &format!("{name}.svh"), DType::F16, &[n])?;
    let mcg = source(catalog, &format!("{name}.mcg"), DType::I32,
        &catalog.tensor(&format!("{name}.mcg"))?.metadata.shape)?;
    ensure!(mcg == 0xcbac1fedu32.to_le_bytes(), "unsupported EXL3 codebook");
    let scales = |bytes: &[u8]| bytes.chunks_exact(2).map(|v| f16(u16::from_le_bytes([v[0], v[1]]))).collect::<Vec<_>>();
    let (suh, svh) = (scales(&suh), scales(&svh));
    ensure!(suh.iter().chain(&svh).all(|v| v.is_finite()), "nonfinite EXL3 rotations");
    let mut x: Vec<_> = input.iter().zip(suh).map(|(x, scale)| x * scale).collect();
    hadamard(&mut x);
    let mut out = vec![0.; n];
    let mut wanted = vec![columns.is_none(); n / 128];
    if let Some(columns) = columns { for &column in columns { wanted[column / 128] = true; } }
    let tile_bytes = 32 * projection.bits;
    for kb in 0..k / 16 {
        let x = &x[kb * 16..kb * 16 + 16];
        if x.iter().all(|&v| v == 0.) { continue; }
        for nb in 0..n / 16 {
            if !wanted[nb / 8] { continue; }
            let offset = (kb * (n / 16) + nb) * tile_bytes;
            let tile = trellis_tile(&trellis[offset..offset + tile_bytes], projection.bits)?;
            for row in 0..16 { for col in 0..16 { out[nb * 16 + col] += x[row] * tile[row * 16 + col]; } }
        }
    }
    hadamard(&mut out);
    for (value, scale) in out.iter_mut().zip(svh) { *value *= scale; }
    Ok(out)
}

fn exl3_expert(catalog: &OfficialV41Catalog, geometry: ExpertGeometry, layer: usize,
    id: usize, input: &[(usize, f64)], columns: &[usize]) -> Result<Vec<f64>> {
    let manifest = catalog.exl3().context("EXL3 manifest")?;
    let name = |kind| manifest.naming.projection(false, layer, id, kind);
    let mut x = vec![0.; geometry.hidden as usize];
    for &(column, value) in input { x[column] += value; }
    let gate = exl3_projection(catalog, &name(V41Exl3ProjectionKind::Gate), &x, None)?;
    let up = exl3_projection(catalog, &name(V41Exl3ProjectionKind::Up), &x, None)?;
    // EXL3 is compared to the unrounded mathematical source oracle, including
    // normalized rotations, rather than the candidate's staged BF16 tensors.
    let hidden: Vec<_> = gate.into_iter().zip(up).map(|(gate, up)| {
        let (gate, up) = geometry.swiglu_limit().map_or((gate, up), |limit|
            (gate.min(f64::from(limit)), up.clamp(-f64::from(limit), f64::from(limit))));
        gate / (1. + (-gate).exp()) * up
    }).collect();
    let out = exl3_projection(catalog, &name(V41Exl3ProjectionKind::Down), &hidden, Some(columns))?;
    Ok(columns.iter().map(|&column| out[column]).collect())
}

pub(super) fn reference(catalog: &OfficialV41Catalog, geometry: ExpertGeometry,
    layer: usize, columns: &[usize], dtype: ExpertV2Dtype) -> Result<Vec<Vec<f64>>> {
    let mut result = vec![vec![0.; columns.len()]; 2];
    for (row, output) in result.iter_mut().enumerate() {
        let input = decoded_input(geometry, row, dtype)?;
        for slot in 0..geometry.topk as usize {
            let id = expert(geometry, layer, row, slot);
            let values = if catalog.exl3().is_some() {
                exl3_expert(catalog, geometry, layer, id, &input, columns)?
            } else { native_expert(catalog, geometry, layer, id, &input, columns)? };
            for (output, value) in output.iter_mut().zip(values) {
                *output += value * f64::from(route_weight(geometry, slot));
            }
        }
    }
    Ok(result)
}

pub(super) fn error(actual: &[f64], expected: &[f64]) -> Result<(f64, f64)> {
    ensure!(actual.len() == expected.len() && !actual.is_empty(), "reference shape");
    ensure!(actual.iter().chain(expected).all(|v| v.is_finite()), "nonfinite routed output");
    let norm = expected.iter().map(|v| v * v).sum::<f64>();
    let actual_norm = actual.iter().map(|v| v * v).sum::<f64>();
    ensure!(norm > 1e-20 && actual_norm > 1e-20, "degenerate routed output");
    let relative = (actual.iter().zip(expected).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / norm).sqrt();
    let cosine = actual.iter().zip(expected).map(|(a, b)| a * b).sum::<f64>() / (norm * actual_norm).sqrt();
    Ok((relative, cosine))
}

#[test]
fn k32_wire_roundtrip_preserves_sparse_oracle_and_uses_nearest_even() -> Result<()> {
    let dtype = ExpertV2Dtype::Fp8E4m3Ue8m0K32;
    let mut values = vec![0.; 64];
    values[..5].copy_from_slice(&[448., 1.0625, 1.1875, -1.0625, -1.1875]);
    let wire = quantize_input(&values, dtype)?;
    assert_eq!(&wire[..5], &[0x7e, 0x38, 0x3a, 0xb8, 0xba]);
    assert_eq!(&wire[64..], &[127, 105]);
    for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::GLM5,
        ExpertGeometry::MIMO_V2_FLASH, ExpertGeometry::MIMO_V26_PRO] {
        for row in 0..2 {
            for dtype in [ExpertV2Dtype::Bf16, dtype] {
                assert_eq!(wire_input(geometry, row, dtype)?.len(), dtype.row_bytes(geometry.hidden as usize)?);
                let decoded = decoded_input(geometry, row, dtype)?;
                let mut expected = input(geometry, row);
                expected.sort_by_key(|&(column, _)| column);
                assert_eq!(decoded, expected, "{geometry:?}, row={row}, dtype={dtype:?}");
            }
        }
    }
    Ok(())
}

#[test]
fn source_decoders_and_missing_sum_fail_closed() -> Result<()> {
    let weights = [0xf2; 16];
    assert_eq!(element(ExpertFormat::Mxfp4, &weights, &[128], 32, 0, 0)?, 2.);
    assert_eq!(element(ExpertFormat::Mxfp4, &weights, &[128], 32, 0, 1)?, -12.);
    assert!(element(ExpertFormat::Mxfp4, &weights, &[255], 32, 0, 0).is_err());
    for (code, expected) in [(0, 0.), (1, 1. / 512.), (0x38, 1.), (0x7e, 448.), (0xfe, -448.)] {
        assert_eq!(e4m3(code)?, expected);
    }
    assert!(e4m3(0x7f).is_err() && e4m3(0xff).is_err());
    assert_eq!(f16(0x3c00), 1.);
    assert_eq!(f16(0x8001), -2f64.powi(-24));
    assert_eq!(bf16(1.003), 1.);
    let (relative, cosine) = error(&[1., 2., 3.], &[1., 2., 3.])?;
    assert!(relative == 0. && (cosine - 1.).abs() < 1e-12);
    assert!(error(&[0.; 3], &[1., 2., 3.]).is_err());
    assert!(error(&[0.8, 1.6, 2.4], &[1., 2., 3.])?.0 > 0.05);
    Ok(())
}

#[test]
fn trellis_tiles_match_independent_numpy_source_vectors() -> Result<()> {
    use sha2::{Digest, Sha256};
    for (bits, expected) in [
        (2, "82d30f7ef468d0d25c2eb1953b76036333545aa9d73031733a7cb3877ac75a6c"),
        (3, "922cd479da12436505f391e552fc2c36d1cfd322049f9fbd8e44ef0bfe83311b"),
        (4, "d4dbb095ad1a5c44c0180bed8ea80a35b8eaf6271ef8a1c995604fe2aa6e3efb"),
        (5, "c74e46b71c95b343c64d73b05191598a955dd32a2afc7f4dcce2ef41f40b3e62"),
    ] {
        let bytes: Vec<_> = (0..16 * bits).flat_map(|j|
            ((j * 7919 + 12345) as u16).to_le_bytes()).collect();
        let tile = trellis_tile(&bytes, bits)?;
        let mut hash = Sha256::new();
        for value in tile { hash.update(value.to_le_bytes()); }
        assert_eq!(format!("{:x}", hash.finalize()), expected, "K{bits}");
    }
    assert!(trellis_tile(&[0; 32], 1).is_err());
    Ok(())
}

#[test]
fn fp8_block_scale_indexing_and_bf16_widening() -> Result<()> {
    let weights = vec![0x39; 256 * 256];
    let scales: Vec<_> = [0.5f32, 2., 4., 8.].into_iter().flat_map(f32::to_le_bytes).collect();
    for (row, col, scale) in [(0, 0, 0.5), (0, 255, 2.), (255, 0, 4.), (255, 255, 8.)] {
        assert_eq!(element(ExpertFormat::Fp8Block128, &weights, &scales, 256, row, col)?, bf16(1.125 * scale));
    }
    assert!(element(ExpertFormat::Fp8Block128, &weights, &0f32.to_le_bytes(), 256, 0, 0).is_err());
    Ok(())
}

#[test]
fn hadamard_is_normalized_and_self_inverse() {
    let original: Vec<_> = (0..256).map(|i| f64::from(i) / 64. - 2.).collect();
    let mut values = original.clone();
    hadamard(&mut values);
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    assert!((norm(&values) - norm(&original)).abs() < 1e-10);
    hadamard(&mut values);
    assert!(values.iter().zip(original).all(|(a, b)| (a - b).abs() < 1e-12));
}

#[test]
fn sparse_requests_cover_family_geometry_and_keep_routes_distinct() {
    for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::GLM5,
        ExpertGeometry::MIMO_V2_FLASH, ExpertGeometry::MIMO_V26_PRO] {
        for row in 0..2 {
            let ids: std::collections::BTreeSet<_> = (0..geometry.topk as usize)
                .map(|slot| expert(geometry, 3, row, slot)).collect();
            assert_eq!(ids.len(), geometry.topk as usize);
            assert!(input(geometry, row).iter().all(|&(col, _)| col < geometry.hidden as usize));
        }
        let sum: f32 = (0..geometry.topk as usize).map(|slot| route_weight(geometry, slot)).sum();
        assert!((sum - 1.).abs() < 1e-6);
        assert!(outputs(geometry).iter().all(|&col| col < geometry.hidden as usize));
    }
}
