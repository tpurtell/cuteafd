use super::*;
use cuteafd_loader::V41_IMAGE_TOKEN_ID;

#[test]
#[ignore = "requires CUDA and official embedding weights"]
fn native_image_replacement_and_reuse() -> Result<()> {
    let lib = unsafe {
        NativeLibrary::load(std::env::var_os("CUTEAFD_VISION_LIBRARY").context("vision library")?)?
    };
    let model =
        std::path::PathBuf::from(std::env::var_os("CUTEAFD_VISION_MODEL").context("vision model")?);
    let catalog =
        cuteafd_loader::read_official_v41_catalog(&model)?;
    let names = vec!["embed.weight".to_owned()];
    let table = NativeRtxTensors::load(
        &lib,
        &catalog,
        &names,
        NativeRtxTensors::plan(&catalog, &names)?,
        16 << 20,
    )?;
    let mut wave =
        TargetEmbeddingWave::new(&lib, &table, 4096, TargetEmbeddingWave::device_bytes(4096)?)?;
    let peer = TargetEmbeddingWave::new(&lib, &table, 4096, TargetEmbeddingWave::device_bytes(4096)?)?;
    let stream = LoadStream { library: &lib, raw: lib.cuda_stream_create()? };
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let copy = |value: TargetEmbedding<'_>| -> Result<(Vec<u8>, Vec<u8>)> {
        let mut residual = vec![0; value.residual.bytes];
        let mut pre = vec![0; value.pre.bytes];
        lib.copy_d2h(&mut residual, value.residual)?;
        lib.copy_d2h(&mut pre, value.pre)?;
        Ok((residual, pre))
    };
    for rows in [1, 35, 129, 4096, 35] {
        let positions: Vec<u64> = (0..rows).map(|i| (i % 128) as u64).collect();
        let selected: Vec<usize> = (0..rows)
            .filter(|i| rows == 4096 || i % 7 == 0 || *i == rows - 1)
            .collect();
        let mut tokens = vec![42; rows];
        for &row in &selected {
            tokens[row] = V41_IMAGE_TOKEN_ID;
        }
        let (baseline, pre) = copy(wave.execute(&tokens, &positions)?)?;
        unsafe { wave.enqueue_into(&tokens, stream.raw, [peer.residual.buffer, peer.pre.buffer])?; }
        runtime.block_on(stream.wait())?;
        let mut residual = peer.residual.buffer; residual.bytes = rows * 40960;
        let mut incoming = peer.pre.buffer; incoming.bytes = rows * 16;
        let queued = copy(TargetEmbedding { residual, pre: incoming, token_ids: &tokens,
            positions: &positions, _owner: PhantomData })?;
        assert_eq!(queued, (baseline.clone(), pre.clone()));
        let features: Vec<Vec<u8>> = selected
            .iter()
            .map(|&row| {
                (0..5120)
                    .flat_map(|col| (0x3f00u16 + ((row * 17 + col) % 128) as u16).to_le_bytes())
                    .collect()
            })
            .collect();
        let bindings: Vec<_> = selected
            .iter()
            .copied()
            .zip(features.iter().map(Vec::as_slice))
            .collect();
        let (actual, image_pre) = copy(wave.execute_with_images(&tokens, &positions, &bindings)?)?;
        assert_eq!(pre, image_pre);
        for row in 0..rows {
            if let Ok(i) = selected.binary_search(&row) {
                for hc in 0..4 {
                    assert_eq!(
                        &actual[(row * 4 + hc) * 10240..(row * 4 + hc + 1) * 10240],
                        features[i]
                    );
                }
            } else {
                assert_eq!(
                    &actual[row * 40960..(row + 1) * 40960],
                    &baseline[row * 40960..(row + 1) * 40960]
                );
            }
        }
        assert_eq!(copy(wave.execute(&tokens, &positions)?)?, (baseline, pre));
        eprintln!("PASS image replacement {rows} rows / {} image rows, all HC lanes, text preservation, repeated positions and text reuse",selected.len());
    }
    let features = vec![0; 10240];
    for (tokens, indices) in [
        (vec![V41_IMAGE_TOKEN_ID; 2], vec![0, 0]),
        (vec![V41_IMAGE_TOKEN_ID; 2], vec![1, 0]),
        (vec![V41_IMAGE_TOKEN_ID; 2], vec![2]),
        (vec![42; 2], vec![0]),
    ] {
        let bindings: Vec<_> = indices
            .iter()
            .map(|&row| (row, features.as_slice()))
            .collect();
        assert!(wave
            .execute_with_images(&tokens, &[0, 1], &bindings)
            .is_err());
        assert!(wave.output().is_err());
    }
    assert!(wave
        .execute_with_images(&[V41_IMAGE_TOKEN_ID], &[0], &[(0, &features[..10239])])
        .is_err());
    assert!(wave.output().is_err());
    wave.execute(&[42], &[0])?;
    eprintln!("PASS image replacement invalid-index/type/extent rejection and recovery");
    Ok(())
}

#[test]
#[ignore = "requires CUDA and official embedding weights"]
fn native_host_embedding_exact_512() -> Result<()> {
    use crate::shared::token_io::EmbedPlacement;
    // SAFETY: the test's native library remains alive after all of its owners.
    let lib = unsafe { NativeLibrary::load(std::env::var_os("CUTEAFD_VISION_LIBRARY").context("library")?)? };
    let model = std::path::PathBuf::from(std::env::var_os("CUTEAFD_VISION_MODEL").context("model")?);
    let catalog = cuteafd_loader::read_official_v41_catalog(&model)?;
    let tokens: Vec<u32> = (0..512).map(|i| ((i * 7919) % 129280) as u32).collect();
    let positions: Vec<u64> = (0..512).collect();
    let copy = |output: TargetEmbedding<'_>| -> Result<(Vec<u8>, Vec<u8>)> {
        let (mut residual, mut pre) = (vec![0; output.residual.bytes], vec![0; output.pre.bytes]);
        lib.copy_d2h(&mut residual, output.residual)?;
        lib.copy_d2h(&mut pre, output.pre)?;
        Ok((residual, pre))
    };
    let reader = catalog.coordinator_tensor_reader("embed.weight")?;
    let mut rows = std::collections::BTreeMap::new();
    for token in tokens.iter().copied().chain(std::iter::once(128799)) {
        let mut row = vec![0; 10240];
        reader.read_into(u64::from(token) * 10240, &mut row)?;
        rows.insert(token, row);
    }
    let mut reference = None;
    for placement in [EmbedPlacement::Gpu, EmbedPlacement::Host] {
        let before = lib.cuda_physical_memory_info()?.0;
        let table = NativeRtxTensors::load_embedding(&lib, &catalog, placement)?;
        let after = lib.cuda_physical_memory_info()?.0;
        assert_eq!(table.resident_bytes(), if placement == EmbedPlacement::Gpu { 129280 * 5120 * 2 } else { 0 });
        let mut wave = TargetEmbeddingWave::new(&lib, &table, 512, TargetEmbeddingWave::device_bytes(512)?)?;
        let output = copy(wave.execute(&tokens, &positions)?)?;
        assert_eq!(copy(wave.execute(&tokens, &positions)?)?, output);
        for (row, token) in tokens.iter().enumerate() {
            for hc in 0..4 {
                assert_eq!(&output.0[row * 40960 + hc * 10240..row * 40960 + (hc + 1) * 10240], rows[token]);
            }
        }
        for width in [5, 7] {
            let ids = DeviceAllocation::new(&lib, 16 * 4)?;
            let residual = DeviceAllocation::new(&lib, 16 * width * 40960)?;
            let pre = DeviceAllocation::new(&lib, 16 * width * 16)?;
            lib.copy_h2d(ids.buffer, &tokens[..16].iter().flat_map(|t| t.to_ne_bytes()).collect::<Vec<_>>())?;
            let ops = lib.v41_attention_ops_width(width)?;
            // SAFETY: immutable table and owned disjoint buffers live through both drains.
            unsafe { ops.embed(table.get("embed.weight")?, ids.buffer, residual.buffer, pre.buffer, 16, wave.stream.raw)?; }
            wave.synchronize()?;
            // SAFETY: the same fixed storage remains owned until replay drains.
            let graph = unsafe {
                lib.cuda_graph_begin_capture(wave.stream.raw)?;
                ops.embed(table.get("embed.weight")?, ids.buffer, residual.buffer, pre.buffer, 16, wave.stream.raw)?;
                lib.cuda_graph_end_capture(wave.stream.raw)?
            };
            // SAFETY: captured graph references these live buffers; destroy follows a drain.
            unsafe { lib.cuda_graph_launch(graph, wave.stream.raw)?; }
            wave.synchronize()?;
            // SAFETY: replay has completed and the graph has no queued users.
            unsafe { lib.cuda_graph_exec_destroy(graph)?; }
            let mut actual = vec![0; residual.buffer.bytes];
            lib.copy_d2h(&mut actual, residual.buffer)?;
            let mut actual_pre = vec![0; pre.buffer.bytes];
            lib.copy_d2h(&mut actual_pre, pre.buffer)?;
            for row in 0..16 * width {
                assert_eq!(&actual_pre[row * 16..row * 16 + 4], 1.0_f32.to_ne_bytes());
                assert_eq!(&actual_pre[row * 16 + 4..(row + 1) * 16], [0; 12]);
                let token = if row % width == 0 { tokens[row / width] } else { 128799 };
                for hc in 0..4 {
                    assert_eq!(&actual[row * 40960 + hc * 10240..row * 40960 + (hc + 1) * 10240], rows[&token]);
                }
            }
        }
        if let Some(reference) = &reference { assert_eq!(&output, reference); }
        else { reference = Some(output); }
        eprintln!("PASS embedding {placement:?}: 512 rows, graph replay exact, table device bytes {}, physical GPU delta {}",
            table.resident_bytes(), before.saturating_sub(after));
    }
    Ok(())
}
