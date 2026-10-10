//! Captured shapes per backbone layer, sharing the containing lane's buffers.
use anyhow::{ensure, Result};

const MAX_DECODE_ROWS: u32 = 8 * (cuteafd_core::MAX_DSPARK_PROPOSALS as u32 + 1);

pub(super) const SMALL_CARD_FIXED_ROWS: &str = "1,6,16,24,32,40,43,48";
static FIXED_SHAPES: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

pub(super) fn profile_fixed_shapes(total_bytes: usize, explicit: Option<&str>) -> Result<Option<Vec<u32>>> {
    if total_bytes > 32usize << 30 { return Ok(None); }
    let value = explicit.unwrap_or(SMALL_CARD_FIXED_ROWS);
    if value.is_empty() { return Ok(None); }
    Ok(Some(validate_fixed_shapes(value.split(',').map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()?)?))
}

/// Startup-only small-card policy. Exact shapes outside the set execute eagerly,
/// never evicting or recapturing a graph, and never changing causal/cache rows.
pub(super) fn set_fixed_shapes(shapes: Vec<u32>) -> Result<()> {
    FIXED_SHAPES.set(validate_fixed_shapes(shapes)?)
        .map_err(|_| anyhow::anyhow!("target graph policy already set"))
}

fn validate_fixed_shapes(mut shapes: Vec<u32>) -> Result<Vec<u32>> {
    ensure!(shapes.contains(&1) && shapes.contains(&6)
        && shapes.iter().all(|r| (1..=MAX_DECODE_ROWS).contains(r)),
        "fixed target graph set must include rows 1 and 6 and use decode rows only");
    shapes.sort_unstable();
    shapes.dedup();
    Ok(shapes)
}

pub(super) fn captures_shape(rows: u32) -> bool {
    FIXED_SHAPES.get().is_none_or(|shapes| shapes.contains(&rows))
}

pub(super) fn fixed_binding_limit() -> Option<usize> {
    FIXED_SHAPES.get().map(|shapes| shapes.len() * 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::decode_graph::{LayerGraphs, RowGraphs};
    use cuteafd_ffi::NativeLibrary;
    use std::collections::BTreeMap;
    use crate::shared::memory::{DeviceAllocation, LoadStream};

    #[test]
    fn small_card_profile_defaults_to_qualified_bank_and_preserves_overrides() {
        assert_eq!(profile_fixed_shapes(32usize << 30, None).unwrap(),
            Some(vec![1, 6, 16, 24, 32, 40, 43, 48]));
        assert_eq!(profile_fixed_shapes(32usize << 30, Some("")).unwrap(), None);
        assert_eq!(profile_fixed_shapes(32usize << 30, Some("1,6")).unwrap(), Some(vec![1, 6]));
        assert_eq!(profile_fixed_shapes(96usize << 30, None).unwrap(), None);
        assert_eq!(profile_fixed_shapes(96usize << 30, Some("1,6")).unwrap(), None);
        assert!(profile_fixed_shapes(32usize << 30, Some("1,80")).is_err());
    }

    #[test]
    fn fixed_shapes_preserve_c1_and_refuse_replacing_large_prefill_slots() {
        assert_eq!(validate_fixed_shapes(vec![48, 6, 1, 24, 32, 40, 6]).unwrap(),
            vec![1, 6, 24, 32, 40, 48]);
        for shapes in [vec![], vec![1], vec![6], vec![1, 6, 0], vec![1, 6, 80]] {
            assert!(validate_fixed_shapes(shapes).is_err());
        }
    }

    #[test]
    fn cuda_row_bank_preserves_decode_handles_over_prefill() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA row bank test: library unset");
            return Ok(());
        };
        // SAFETY: this fixture uses the task's native library and owns all buffers.
        let library = unsafe { NativeLibrary::load(path)? };
        let input = DeviceAllocation::new(&library, 128 * 4)?;
        let output = DeviceAllocation::new(&library, 128 * 4)?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut bank = RowGraphs::new(&library, "fixture", MAX_DECODE_ROWS as usize);
        let mut handles = BTreeMap::new();
        for (cycle, rows) in (1..=MAX_DECODE_ROWS as usize)
            .chain([80, 128, 2, 6, 3, 48, 64, 56, 63, 64]).enumerate() {
            let expected = vec![(cycle + 1) as u8; 128 * 4];
            library.copy_h2d(input.buffer, &expected)?;
            if bank.get(rows).is_none() {
                // SAFETY: the previous iteration drained all work before capture.
                unsafe {
                    library.cuda_graph_begin_capture(stream.raw)?;
                    library.copy_d2d_async(output.buffer, input.buffer, rows * 4, stream.raw)?;
                    let graph = library.cuda_graph_end_capture(stream.raw)?;
                    bank.insert(rows, graph)?;
                }
            }
            let graph = bank.get(rows).unwrap();
            if rows <= MAX_DECODE_ROWS as usize {
                assert_eq!(*handles.entry(rows).or_insert(graph), graph);
            }
            // SAFETY: buffers remain owned and exclusive until this launch drains.
            unsafe {
                library.cuda_graph_launch(graph, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; rows * 4];
            let mut view = output.buffer; view.bytes = actual.len();
            library.copy_d2h(&mut actual, view)?;
            assert_eq!(actual, expected[..actual.len()]);
        }
        assert!(bank.get(80).is_none());
        assert!(bank.get(128).is_some());
        assert_eq!(bank.len(), MAX_DECODE_ROWS as usize + 1);
        // SAFETY: every fixture launch was synchronized before destruction.
        unsafe { bank.clear()?; bank.clear()?; }
        assert_eq!(bank.len(), 0);
        Ok(())
    }

    #[test]
    fn cuda_small_shape_bank_replays_changes_and_bounds_large_shapes() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA shape bank test: library unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let weights = DeviceAllocation::new(&library, 128 * 4)?;
        let other = DeviceAllocation::new(&library, 128 * 4)?;
        let output = DeviceAllocation::new(&library, 128 * 4)?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut bank = LayerGraphs::new(&library, MAX_DECODE_ROWS);
        bank.enable_small_shapes();
        let mut handles = BTreeMap::new();
        for (cycle, rows) in (1..=MAX_DECODE_ROWS).chain([80, 128, 2, 6, 3, 2, 6, 48, 64, 56, 63, 64]).enumerate() {
            let expected = vec![(cycle + 1) as u8; 128 * 4];
            library.copy_h2d(weights.buffer, &expected)?;
            if bank.get_shape(0, &weights, rows).is_none() {
                unsafe {
                    library.cuda_graph_begin_capture(stream.raw)?;
                    library.copy_d2d_async(output.buffer, weights.buffer, rows as usize * 4, stream.raw)?;
                    let graph = library.cuda_graph_end_capture(stream.raw)?;
                    bank.insert(0, &weights, rows, graph)?;
                }
            }
            let (graph, _) = bank.get_shape(0, &weights, rows).unwrap();
            if rows <= MAX_DECODE_ROWS {
                assert_eq!(*handles.entry(rows).or_insert(graph), graph);
            }
            assert!(bank.get_shape(0, &other, rows).is_none());
            unsafe {
                library.cuda_graph_launch(graph, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; rows as usize * 4];
            let mut view = output.buffer;
            view.bytes = actual.len();
            library.copy_d2h(&mut actual, view)?;
            assert_eq!(actual, expected[..actual.len()]);
            assert!(bank.len() <= MAX_DECODE_ROWS as usize + 1);
        }
        assert!(bank.get_shape(0, &weights, 80).is_none());
        assert!(bank.get_shape(0, &weights, 128).is_some());
        unsafe {
            library.cuda_graph_begin_capture(stream.raw)?;
            library.copy_d2d_async(output.buffer, other.buffer, 8, stream.raw)?;
            let graph = library.cuda_graph_end_capture(stream.raw)?;
            bank.insert(0, &other, 2, graph)?;
        }
        assert!(bank.get_shape(0, &weights, 2).is_none());
        assert!(bank.get_shape(0, &weights, 6).is_none());
        assert!(bank.get_shape(0, &other, 2).is_some());
        unsafe { bank.clear()?; }
        assert_eq!(bank.len(), 0);
        assert_eq!(bank.len(), 0);
        Ok(())
    }

    #[test]
    fn cuda_layer_graphs_reuse_storage_and_retain_exact_weight_bindings() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA layer graph test: CUTEAFD_LAYER_GRAPH_TEST_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        // Real CUDA copies isolate graph ownership from SM120 FP8 arithmetic.
        let weights = (0..40)
            .map(|_| DeviceAllocation::new(&library, 4096 * 4))
            .collect::<Result<Vec<_>>>()?;
        let replacement = DeviceAllocation::new(&library, 4096 * 4)?;
        let streams = (0..2)
            .map(|_| {
                Ok(LoadStream {
                    library: &library,
                    raw: library.cuda_stream_create()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = (0..2)
            .map(|_| DeviceAllocation::new(&library, 4096 * 4))
            .collect::<Result<Vec<_>>>()?;
        let mut banks = [LayerGraphs::new(&library, MAX_DECODE_ROWS), LayerGraphs::new(&library, MAX_DECODE_ROWS)];
        let mut captures = 0;
        let mut replays = 0;
        for rows in [1u32, 80, 4096] {
            for cycle in 0..2u32 {
                for layer in 0..40 {
                    let expected: Vec<u8> = (0..rows)
                        .flat_map(|row| (row + 4096 * (layer as u32 + 40 * cycle)).to_ne_bytes())
                        .collect();
                    library.copy_h2d(weights[layer].buffer, &expected)?;
                    let mut handles = [std::ptr::null_mut(); 2];
                    for lane in 0..2 {
                        let bank = &mut banks[lane];
                        let current = bank.get(layer, &weights[layer]);
                        if current.is_none_or(|(_, count)| count != rows) {
                            unsafe {
                                bank.remove(layer)?;
                                library.cuda_graph_begin_capture(streams[lane].raw)?;
                                library.copy_d2d_async(
                                    outputs[lane].buffer,
                                    weights[layer].buffer,
                                    expected.len(),
                                    streams[lane].raw,
                                )?;
                                let graph = library.cuda_graph_end_capture(streams[lane].raw)?;
                                bank.insert(layer, &weights[layer], rows, graph)?;
                            }
                            captures += 1;
                        } else {
                            // A repeated same-shape pass must keep the graph handle.
                            assert_eq!(cycle, 1);
                        }
                        let (graph, count) = bank.get(layer, &weights[layer]).unwrap();
                        assert_eq!(count, rows);
                        assert!(bank.get(layer, &replacement).is_none());
                        handles[lane] = graph;
                        unsafe {
                            library.cuda_graph_launch(graph, streams[lane].raw)?;
                        }
                        replays += 1;
                    }
                    for lane in 0..2 {
                        unsafe {
                            library.cuda_stream_synchronize(streams[lane].raw)?;
                        }
                        let mut actual = vec![0; expected.len()];
                        library.copy_d2h(&mut actual, outputs[lane].buffer)?;
                        assert_eq!(actual, expected);
                        assert_eq!(
                            banks[lane].get(layer, &weights[layer]).unwrap().0,
                            handles[lane]
                        );
                    }
                }
                assert!(banks
                    .iter()
                    .all(|bank| bank.len() == 40));
            }
        }
        assert_eq!(captures, 240);
        assert_eq!(replays, 480);
        let bank = &mut banks[0];
        let original = bank.get(0, &weights[0]).unwrap();
        // Reject invalid metadata without touching the existing graph.
        unsafe {
            assert!(bank.insert(40, &replacement, 1, original.0).is_err());
            assert!(bank.insert(0, &replacement, 0, original.0).is_err());
            assert!(bank.insert(0, &replacement, 4097, original.0).is_err());
            assert!(bank
                .insert(0, &replacement, 1, std::ptr::null_mut())
                .is_err());
        }
        assert_eq!(bank.get(0, &weights[0]), Some(original));
        // Replacing an owner at the same layer evicts its old capture only.
        library.copy_h2d(replacement.buffer, &123456u32.to_ne_bytes())?;
        unsafe {
            library.cuda_graph_begin_capture(streams[0].raw)?;
            library.copy_d2d_async(outputs[0].buffer, replacement.buffer, 4, streams[0].raw)?;
            let graph = library.cuda_graph_end_capture(streams[0].raw)?;
            bank.insert(0, &replacement, 1, graph)?;
            assert!(bank.get(0, &weights[0]).is_none());
            assert!(bank.get(39, &weights[39]).is_some());
            library.cuda_graph_launch(bank.get(0, &replacement).unwrap().0, streams[0].raw)?;
            library.cuda_stream_synchronize(streams[0].raw)?;
        }
        let mut actual = [0u8; 4];
        library.copy_d2h(&mut actual, outputs[0].buffer)?;
        assert_eq!(u32::from_ne_bytes(actual), 123456);
        for bank in &mut banks {
            unsafe {
                bank.clear()?;
            }
            assert_eq!(bank.len(), 0);
            unsafe {
                bank.clear()?;
            }
        }
        eprintln!("PASS 240 captures, 480 replays, two independent lanes, 40 layers, rows 1/80/4096, changed payloads, owner replacement and invalid binding guards");
        Ok(())
    }
}
