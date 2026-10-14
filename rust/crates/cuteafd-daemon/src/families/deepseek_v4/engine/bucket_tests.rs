//! Implementation-local CUDA byte oracles; no checkpoint or serving driver.
use super::*;
use cuteafd_ffi::{CuteafdDeviceBuffer, V41ExpertKernel, V41ExpertLaunchArgs,
    V41ExpertOutputKind, V41_EXPERT_POINTER_COUNT, programs::VOCABULARY_HEAD_WORKSPACE};
use std::collections::BTreeMap;

const ROWS: usize = 64;
const VOCAB: usize = 129_280;

fn fixture_library() -> Result<NativeLibrary> {
    // NativeLibrary::load fixes the geometry used by native routed helpers.
    // Pro AOT programs carry their own shape; the routed oracle is Flash only.
    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH)
        .map_err(|g| anyhow::anyhow!("fixture geometry already fixed to {g:?}"))?;
    // SAFETY: matching sealed ABI; every fixture drains before library unload.
    unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?) }
}

fn validate_fp32_output(bytes: &[u8], label: &str, name: &str, rows: usize) -> Result<()> {
    let carry = if label.contains("compressor_continuation_c4") && name.ends_with("_score_state") {
        Some((16, if name == "index_score_state" { 256 } else { 1024 }, 3))
    } else if label.contains("compressor_continuation_c128") && name == "main_score_state" {
        Some((256, 512, 127))
    } else { None };
    // Each fixture sequence contributes one row starting at ratio-1. The
    // finalizer requires -inf exactly where no carry position exists yet.
    for (index, word) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_ne_bytes(word.try_into().unwrap());
        let sentinel = carry.is_some_and(|(state_rows, width, last)|
            index / (state_rows * width) < rows && (index / width) % state_rows > last);
        ensure!(if sentinel { value.to_bits() == f32::NEG_INFINITY.to_bits() } else { value.is_finite() },
            "{label} rows={rows} {name} FP32 element={index} invalid value {value}; sentinel={sentinel}");
    }
    Ok(())
}

#[test]
fn owner_bucket_fixture_score_sentinels_are_coordinate_exact() -> Result<()> {
    for (ratio, state_rows, width, name) in [(4, 16, 1024, "main_score_state"),
        (4, 16, 256, "index_score_state"), (128, 256, 512, "main_score_state")] {
        let label = format!("dsv4f_compressor_continuation_c{ratio}");
        let mut words = vec![0.25f32; 2 * state_rows * width];
        for row in ratio..state_rows { words[row * width..(row + 1) * width].fill(f32::NEG_INFINITY); }
        validate_fp32_output(&f32_bytes(words.iter().copied()), &label, name, 1)?;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            words[0] = invalid;
            assert!(validate_fp32_output(&f32_bytes(words.iter().copied()), &label, name, 1).is_err());
        }
        words[0] = 0.25;
        words[state_rows * width] = f32::NEG_INFINITY;
        assert!(validate_fp32_output(&f32_bytes(words.iter().copied()), &label, name, 1).is_err());
        words[state_rows * width] = 0.25;
        words[ratio * width] = 0.0;
        assert!(validate_fp32_output(&f32_bytes(words.iter().copied()), &label, name, 1).is_err());
    }
    Ok(())
}

struct Tensor<'a> {
    dev: Dev<'a>,
    initial: Vec<u8>,
    row_bytes: Option<usize>,
    role: String,
    dtype: Option<String>,
}

struct Fixture<'a> {
    library: &'a NativeLibrary,
    stream: *mut c_void,
    tensors: BTreeMap<String, Tensor<'a>>,
}

impl<'a> Fixture<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self { library, stream: library.cuda_stream_create()?, tensors: BTreeMap::new() })
    }

    fn add(&mut self, name: &str, initial: Vec<u8>, row_bytes: Option<usize>, role: &str) -> Result<()> {
        let dev = Dev::new(self.library, initial.len().max(16))?;
        self.library.cuda_zero_bytes(dev.buffer, dev.buffer.bytes)?;
        if !initial.is_empty() { self.library.copy_h2d(dev.buffer, &initial)?; }
        self.tensors.insert(name.to_owned(), Tensor { dev, initial, row_bytes, role: role.to_owned(), dtype: None });
        Ok(())
    }

    fn ptr(&self, name: &str) -> *mut c_void { self.tensors[name].dev.buffer.ptr }

    fn set(&mut self, name: &str, bytes: Vec<u8>) -> Result<()> {
        let tensor = self.tensors.get_mut(name).with_context(|| format!("missing fixture tensor {name}"))?;
        ensure!(bytes.len() <= tensor.dev.buffer.bytes, "fixture upload exceeds {name}");
        tensor.initial = bytes;
        self.library.copy_h2d(tensor.dev.buffer, &tensor.initial)
    }

    fn reset(&self) -> Result<()> {
        self.sync()?;
        for tensor in self.tensors.values() {
            if matches!(tensor.role.as_str(), "out" | "inout" | "scratch") {
                self.library.copy_h2d(tensor.dev.buffer, &tensor.initial)?;
            }
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        // SAFETY: the fixture owns this stream and retains all its allocations.
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
    }

    fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        let buffer = self.tensors[name].dev.buffer;
        ensure!(bytes <= buffer.bytes, "fixture read exceeds {name}");
        let mut output = vec![0; bytes];
        self.library.copy_d2h(&mut output, CuteafdDeviceBuffer { bytes, ..buffer })?;
        Ok(output)
    }

    fn compare(&self, reference: &Self, rows: usize, label: &str) -> Result<()> {
        self.sync()?;
        reference.sync()?;
        for (name, tensor) in &self.tensors {
            if !matches!(tensor.role.as_str(), "out" | "inout") { continue; }
            let bytes = if tensor.role == "inout" { tensor.initial.len() }
                else { tensor.row_bytes.context("output has no row stride")? * rows };
            let actual = self.read(name, bytes)?;
            let expected = reference.read(name, bytes)?;
            match tensor.dtype.as_deref() {
                Some("float32") => validate_fp32_output(&actual, label, name, rows)?,
                Some("bfloat16") => ensure!(actual.chunks_exact(2).all(|b|
                    u16::from_ne_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80), "{label} {name} nonfinite BF16 output"),
                Some("float8_e4m3fn") => ensure!(actual.iter().all(|b| b & 0x7f != 0x7f), "{label} {name} NaN FP8 output"),
                _ => {}
            }
            if let Some(at) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                anyhow::bail!("{label} rows={rows} bucket={} {name} byte={at}: {} != {}",
                    metadata::owner_row_bucket(rows), actual[at], expected[at]);
            }
            if tensor.role == "inout" {
                let other = &reference.tensors[name];
                // Comparing the whole allocation includes every unwritten state
                // sequence/page, not merely the slots named by real rows.
                ensure!(other.initial == tensor.initial, "{label} state seeds differ");
            }
        }
        Ok(())
    }
}

impl Drop for Fixture<'_> {
    fn drop(&mut self) {
        // SAFETY: drain before fields release device storage, even on a mismatch.
        unsafe {
            let _ = self.library.cuda_stream_synchronize(self.stream);
            let _ = self.library.cuda_stream_destroy(self.stream);
        }
    }
}

struct Captured<'a> { library: &'a NativeLibrary, stream: *mut c_void, raw: *mut c_void }
impl<'a> Captured<'a> {
    fn new(f: &Fixture<'a>, run: impl FnOnce() -> Result<()>) -> Result<Self> {
        f.sync()?;
        // SAFETY: only device launches on the retained fixture stream are captured.
        unsafe { f.library.cuda_graph_begin_capture(f.stream)?; }
        let result = run();
        // End capture on failure too; take ownership before propagating the error.
        let raw = unsafe { f.library.cuda_graph_end_capture(f.stream)? };
        let graph = Self { library: f.library, stream: f.stream, raw };
        result?;
        Ok(graph)
    }
    fn replay(&self) -> Result<()> {
        // SAFETY: the fixture outlives this executable and retains every binding.
        unsafe { self.library.cuda_graph_launch(self.raw, self.stream) }
    }
}
impl Drop for Captured<'_> {
    fn drop(&mut self) {
        // SAFETY: drain before retiring an executable, including error returns.
        unsafe {
            let _ = self.library.cuda_stream_synchronize(self.stream);
            let _ = self.library.cuda_graph_exec_destroy(self.raw);
        }
    }
}

fn i32_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_ne_bytes).collect()
}
fn i64_bytes(values: impl IntoIterator<Item = i64>) -> Vec<u8> {
    values.into_iter().flat_map(i64::to_ne_bytes).collect()
}
fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_ne_bytes).collect()
}

fn seeded(dtype: &str, elements: usize, name: &str) -> Result<Vec<u8>> {
    let hash = name.bytes().fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize));
    let sample = |i: usize| (((i.wrapping_mul(17).wrapping_add(hash) % 31) as f32) - 15.0) / 64.0;
    Ok(match dtype {
        "bfloat16" => (0..elements).flat_map(|i| {
            let v = if name.ends_with("norm") || name == "norm" { 1.0 + sample(i) } else { sample(i) };
            let bits = v.to_bits();
            ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_ne_bytes()
        }).collect(),
        "float32" => f32_bytes((0..elements).map(sample)),
        "float8_e4m3fn" => (0..elements).map(|i| [0x10, 0x18, 0x20, 0x90, 0x98, 0xa0][(i + hash) % 6]).collect(),
        "uint8" => vec![if name.ends_with("scale") { 120 } else { 0 }; elements],
        "int32" => i32_bytes(std::iter::repeat_n(0, elements)),
        "int64" => i64_bytes(std::iter::repeat_n(0, elements)),
        _ => anyhow::bail!("unknown fixture dtype {dtype}"),
    })
}

fn manifest_fixture<'a>(library: &'a NativeLibrary, entry: &serde_json::Value) -> Result<Fixture<'a>> {
    let mut f = Fixture::new(library)?;
    let ratio = entry["geometry"]["ratio"].as_u64().unwrap_or_else(||
        if entry["name"].as_str().unwrap_or_default().contains("c128") { 128 } else { 4 }) as usize;
    for operand in entry["pointers"].as_array().context("fixture pointers")? {
        let name = operand["name"].as_str().context("fixture operand name")?;
        let dtype = operand["dtype"].as_str().context("fixture operand dtype")?;
        let shape = operand["shape"].as_str().context("fixture operand shape")?;
        let role = operand["role"].as_str().context("fixture operand role")?;
        let element_bytes = match dtype { "bfloat16" => 2, "float32" | "int32" => 4,
            "int64" => 8, "uint8" | "float8_e4m3fn" => 1, _ => anyhow::bail!("unknown {dtype}") };
        let dimensions = shape.trim_matches(['[', ']']).split(',').map(str::trim).collect::<Vec<_>>();
        let row_major = dimensions.first() == Some(&"rows");
        let scratch = entry["scratch_bytes_at_capacity"][name].as_u64();
        let count = if let Some(bytes) = scratch { bytes as usize / element_bytes } else {
            dimensions.iter().try_fold(1usize, |n, dim| -> Result<usize> {
                let value = match *dim {
                    "rows" | "G" | "C" | "S" => ROWS,
                    "C+1" => ROWS + 1,
                    "P" => 512,
                    "pages" | "pages_c" | "pool_pages" => 65,
                    "table_stride" => 8,
                    other => other.parse().with_context(|| format!("unsupported fixture shape {name} {shape}"))?,
                };
                n.checked_mul(value).context("fixture shape overflow")
            })?
        };
        let mut initial = seeded(dtype, count, name)?;
        if role == "out" || role == "scratch" { initial.fill(0); }
        // Valid packed FP8 cache records: nonzero values, UE8M0 scale 1,
        // BF16 RoPE lanes, and untouched spare pages for the state-write oracle.
        if count > 0 && (name == "swa_cache" || name == "main_kv_cache" || name == "compressed_cache" || name == "indexed_cache") {
            let page_rows = if name == "swa_cache" || name == "main_kv_cache" { 256 }
                else { 256 / ratio };
            let page_bytes = count / 65;
            for page in initial.chunks_exact_mut(page_bytes) {
                for row in 0..page_rows {
                    page[row * 576..row * 576 + 448].fill(0x18);
                    for lane in page[row * 576 + 448..(row + 1) * 576].chunks_exact_mut(2) {
                        lane.copy_from_slice(&0x3d80u16.to_ne_bytes());
                    }
                    page[page_rows * 576 + row * 8..page_rows * 576 + row * 8 + 7].fill(127);
                }
            }
        }
        if matches!(name, "index_cache" | "index_k_cache") {
            for page in initial.chunks_exact_mut(metadata::INDEX_PAGE_BYTES) {
                page[..8192].fill(0x18);
                page[8192..].copy_from_slice(&f32_bytes(std::iter::repeat_n(1.0, 64)));
            }
        }
        if name == "page_table" {
            initial = i32_bytes((0..count).map(|i| (i % 8) as i32));
        }
        if name == "cache_lengths" {
            initial = i32_bytes(std::iter::repeat_n(512, ROWS));
        }
        if name == "cos_sin" {
            initial = f32_bytes((0..512 * 64).map(|i| if i % 64 < 32 { 1.0 } else { 0.0 }));
        }
        if name == "sequence_offsets" { initial = i32_bytes(0..=ROWS as i32); }
        if name == "active_groups" || name == "active_sequences" { initial = i32_bytes([ROWS as i32]); }
        if name == "positions" { initial = if dtype == "int64" { i64_bytes(std::iter::repeat_n(127, ROWS)) }
            else { i32_bytes(std::iter::repeat_n((ratio - 1) as i32, ROWS)) }; }
        if matches!(name, "main_slots" | "sequence_ids" | "state_sequence_ids" | "compressed_slots" | "group_sequence_slots") {
            initial = if dtype == "int64" { i64_bytes((0..ROWS).map(|i| i as i64 * 256 + 127)) }
                else if name == "compressed_slots" { i32_bytes((0..ROWS).map(|i| (i * (256 / ratio)) as i32)) }
                else { i32_bytes(0..ROWS as i32) };
        }
        if name == "sequence_start_positions" {
            initial = i32_bytes(std::iter::repeat_n((ratio - 1) as i32, ROWS));
        }
        if name == "group_source_positions" || name == "group_rope_positions" {
            initial = i32_bytes(std::iter::repeat_n(0, ROWS));
        }
        if name.ends_with("lengths") && name != "cache_lengths" { initial = i32_bytes(std::iter::repeat_n(1, ROWS)); }
        if count > 0 && name.ends_with("indices") {
            let stride = count / ROWS;
            initial = i32_bytes((0..count).map(|i| if i % stride == 0 { 0 } else { -1 }));
        }
        f.add(name, initial, row_major.then_some(count / ROWS * element_bytes), role)?;
        f.tensors.get_mut(name).unwrap().dtype = Some(dtype.to_owned());
    }
    Ok(f)
}

fn fixture_tables(name: &str, rows: usize) -> StepTables {
    let chunked = name.contains("continuation");
    let fields: &[&'static str] = if chunked {
        &["active_groups", "active_sequences", "group_sequence_slots", "compressed_slots",
            "group_source_positions", "group_rope_positions", "state_sequence_ids",
            "sequence_start_positions", "sequence_offsets"]
    } else { &["positions", "sequence_ids", "compressed_slots"] };
    let compressor = |ratio: usize| if name.contains("compressor") {
        fields.iter().map(|&field| {
            let values = match field {
                "active_groups" | "active_sequences" => vec![rows as i32],
                "sequence_offsets" => (0..=rows as i32).collect(),
                "positions" | "sequence_start_positions" => vec![(ratio - 1) as i32; rows],
                "group_source_positions" | "group_rope_positions" => vec![0; rows],
                "compressed_slots" => (0..rows).map(|i| (i * (256 / ratio)) as i32).collect(),
                _ => (0..rows as i32).collect(),
            };
            (field, values)
        }).collect()
    } else { Vec::new() };
    // Real steps carry metadata for both ratios, even in a single-ratio launch.
    StepTables {
        decode: true, chunked, start: 0, rows,
        positions: vec![127; rows],
        main_slots: (0..rows).map(|i| (i * 256 + 127) as i64).collect(),
        swa_indices: (0..rows * metadata::WINDOW).map(|i| if i % metadata::WINDOW == 0 { 0 } else { -1 }).collect(),
        swa_lengths: vec![1; rows],
        c4_tables: compressor(4),
        c128_tables: compressor(128),
        c4_groups: rows, c128_groups: rows,
        c4_page_table: (0..rows * 8).map(|i| (i % 8) as i32).collect(),
        c4_table_width: 8, c4_table_stride: 8,
        c4_visible: vec![512; rows], c4_indexed_lengths: vec![1; rows],
        c128_indices: vec![0; rows], c128_lengths: vec![1; rows],
    }
}

#[test]
fn owner_bucket_fixture_continuation_metadata_covers_both_ratios() -> Result<()> {
    for rows in 1..=ROWS {
        let tables = fixture_tables("dsv4f_compressor_continuation_c4", rows);
        let upload = tables.upload(ROWS, 64)?;
        for (ratio, entries) in [(4, &upload.c4_tables), (128, &upload.c128_tables)] {
            let field = |name| &entries.iter().find(|(key, _)| *key == name).unwrap().1;
            assert_eq!(field("active_groups"), &[rows as i32]);
            assert_eq!(field("active_sequences"), &[rows as i32]);
            assert_eq!(&field("sequence_start_positions")[..rows], &vec![ratio - 1; rows]);
            assert!(field("state_sequence_ids")[rows..].iter().all(|&id| id == -1));
            assert!(field("compressed_slots")[rows..].iter().all(|&slot| slot == -1));
            assert!(field("sequence_offsets")[rows..].iter().all(|&offset| offset == rows as i32));
        }
    }
    Ok(())
}

fn mask_metadata(f: &mut Fixture<'_>, entry: &serde_json::Value, rows: usize) -> Result<()> {
    let name = entry["name"].as_str().context("fixture name")?;
    let ratio = if name.contains("c128") { 128 } else { 4 };
    // Exercise the same canonical-to-upload conversion as Engine::fill. Real
    // requests stay narrow while every device metadata plane has masked tails.
    let tables = fixture_tables(name, rows);
    let upload = tables.upload(ROWS, 64)?;
    for operand in entry["pointers"].as_array().context("fixture pointers")? {
        let field = operand["name"].as_str().unwrap();
        let values = match field {
            "positions" if operand["dtype"] == "int64" => {
                f.set(field, i64_bytes(upload.positions.iter().copied()))?; continue;
            }
            "main_slots" => {
                f.set(field, i64_bytes(upload.main_slots.iter().copied()))?; continue;
            }
            "swa_indices" => Some(upload.swa_indices.clone()),
            "swa_lengths" => Some(upload.swa_lengths.clone()),
            "page_table" => Some(upload.c4_page_table.clone()),
            "cache_lengths" => Some(upload.c4_visible.clone()),
            "indexed_lengths" => Some(if ratio == 128 { upload.c128_lengths.clone() } else { upload.c4_indexed_lengths.clone() }),
            "indexed_indices" => {
                let stride = f.tensors[field].initial.len() / (ROWS * 4);
                Some((0..ROWS * stride).map(|i| if i / stride < rows && i % stride == 0 { 0 } else { -1 }).collect())
            }
            _ => (if ratio == 128 { &upload.c128_tables } else { &upload.c4_tables }).iter()
                .find(|(key, _)| *key == field).map(|(_, values)| values.clone()),
        };
        if let Some(values) = values { f.set(field, i32_bytes(values))?; }
    }
    Ok(())
}

fn launch_program(programs: &Programs<'_>, entry: &serde_json::Value, f: &Fixture<'_>, rows: usize) -> Result<()> {
    let name = entry["name"].as_str().context("fixture program name")?;
    let names = entry["pointers"].as_array().context("fixture pointers")?.iter()
        .map(|p| p["name"].as_str().unwrap()).collect::<Vec<_>>();
    let program = programs.program(name, &names)?;
    let pointers = names.iter().map(|n| f.ptr(n)).collect::<Vec<_>>();
    let scalars = entry["scalars"].as_array().context("fixture scalars")?.iter().map(|s| -> Result<Scalar> {
        Ok(Scalar::I32(match s["name"].as_str().context("scalar name")? {
            "rows" | "live_rows" | "groups" | "sequences" => rows as i32,
            "table_width" | "table_stride" => 8,
            other => anyhow::bail!("unknown fixture scalar {other}"),
        }))
    }).collect::<Result<Vec<_>>>()?;
    // SAFETY: tensor extents come from the attested manifest; all bindings are
    // retained on the current device and the scalar is bounded by 64 rows.
    unsafe { program.launch(&pointers, &scalars, f.stream) }
}

#[test]
#[ignore = "two RTX GPUs, sealed V4 AOT library and CUTEAFD_TEST_PROGRAMS manifest"]
fn owner_bucket_aot_segments_are_byte_exact_at_every_width() -> Result<()> {
    let path = std::env::var("CUTEAFD_TEST_PROGRAMS")?;
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    // SAFETY: matching sealed ABI; all fixture storage/graphs drain before unload.
    let library = fixture_library()?;
    for device in 0..2 {
        library.cuda_set_device(device)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(&path))?;
        for family in ["dsv4f", "dsv4p", "dsv4f2", "dsv4p2"] {
            let operations = ["mhc_pre", "producer_m64", "wo_m64", "mhc_post_pre_m64", "shared_ffn_m64", "mhc_post", "mhc_head",
                "sparse_mla_decode_win_m64", "sparse_mla_decode_c4_m64", "sparse_mla_decode_c128_m64"];
            let mut names = operations.iter().filter(|op| !family.ends_with('2') ||
                matches!(**op, "producer_m64" | "wo_m64" | "shared_ffn_m64" |
                    "sparse_mla_decode_win_m64" | "sparse_mla_decode_c4_m64" | "sparse_mla_decode_c128_m64"))
                .map(|op| format!("{family}_{op}")).collect::<Vec<_>>();
            if !family.ends_with('2') {
                for op in ["index_producer_m64", "index_topk_decode_m64", "router_scores"] {
                    names.push(format!("{family}_{op}"));
                }
                for ratio in [4, 128] { for mode in ["decode", "continuation"] {
                    names.push(format!("{family}_compressor_{mode}_c{ratio}"));
                }}
            }
            for name in names {
                let entry = manifest["programs"].as_array().context("program inventory")?.iter()
                    .find(|p| p["name"] == name).with_context(|| format!("missing program {name}"))?;
                let mut exact = manifest_fixture(&library, entry)?;
                let mut padded = manifest_fixture(&library, entry)?;
                // Preload outside capture. This is also a shape/finite-output warmup.
                launch_program(&programs, entry, &exact, ROWS)?;
                launch_program(&programs, entry, &padded, ROWS)?;
                exact.sync()?;
                padded.sync()?;
                for bucket in metadata::OWNER_ROWS {
                    let graph = Captured::new(&padded, || launch_program(&programs, entry, &padded, bucket))?;
                    for rows in 1..=ROWS {
                        if metadata::owner_row_bucket(rows) != bucket { continue; }
                        exact.reset()?;
                        padded.reset()?;
                        mask_metadata(&mut exact, entry, rows)?;
                        mask_metadata(&mut padded, entry, rows)?;
                        launch_program(&programs, entry, &exact, rows)?;
                        graph.replay()?;
                        padded.compare(&exact, rows, &name)?;
                        if name.contains("compressor") {
                            let width = entry["geometry"]["joint_width"].as_u64().context("compressor projection width")? as usize;
                            ensure!(padded.read("scratch", rows * width * 2)? == exact.read("scratch", rows * width * 2)?,
                                "{name} real projection rows={rows} differs");
                        }
                        // Replay the same graph again from identical retained state.
                        exact.reset()?;
                        padded.reset()?;
                        launch_program(&programs, entry, &exact, rows)?;
                        graph.replay()?;
                        padded.compare(&exact, rows, &name)?;
                    }
                }
                println!("device={device} program={name} widths=1..64 buckets=22 real outputs/full state byte-exact");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "two RTX GPUs, sealed V4 AOT library and CUTEAFD_TEST_PROGRAMS manifest"]
fn owner_bucket_head_tap_and_draft_kv_are_byte_exact_at_every_width() -> Result<()> {
    let path = std::env::var("CUTEAFD_TEST_PROGRAMS")?;
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    // SAFETY: the matching sealed ABI remains loaded until all fixtures drain.
    let library = fixture_library()?;
    for device in 0..2 {
        library.cuda_set_device(device)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(&path))?;
        for (family, h) in [("dsv4f", 4096usize), ("dsv4p", 7168)] {
            let entry = |op: &str| -> Result<&serde_json::Value> {
                let name = format!("{family}_{op}");
                manifest["programs"].as_array().context("program inventory")?.iter()
                    .find(|p| p["name"] == name).with_context(|| format!("missing {name}"))
            };
            let head_entry = entry("mhc_head")?;
            let producer_entry = entry("producer_m64")?;
            let mut exact = manifest_fixture(&library, head_entry)?;
            let mut padded = manifest_fixture(&library, head_entry)?;
            for f in [&mut exact, &mut padded] {
                f.add("head_workspace", vec![0; VOCABULARY_HEAD_WORKSPACE], None, "scratch")?;
                f.add("head_weight", seeded("bfloat16", VOCAB * h, "vocabulary_weight")?, None, "in")?;
                f.add("logits", vec![0; ROWS * VOCAB * 4], Some(VOCAB * 4), "out")?;
                f.tensors.get_mut("logits").unwrap().dtype = Some("float32".to_owned());
                f.add("packed", vec![0xa5; (ROWS + 2) * VOCAB * 4], None, "scratch")?;
            }
            // SAFETY: each workspace is retained beyond its head, with the full
            // production width/vocabulary and maximum row count.
            let exact_head = unsafe { library.vocabulary_head_rows(exact.ptr("head_workspace"), h as u32, ROWS as u32, VOCAB as u32)? };
            let padded_head = unsafe { library.vocabulary_head_rows(padded.ptr("head_workspace"), h as u32, ROWS as u32, VOCAB as u32)? };
            let project = |f: &Fixture<'_>, head: &cuteafd_ffi::programs::VocabularyHead<'_>, rows: usize| -> Result<()> {
                launch_program(&programs, head_entry, f, rows)?;
                // SAFETY: the mHC output and full BF16 vocabulary matrix are
                // retained on this stream; logits hold all execution rows.
                unsafe { crate::shared::target_head::launch(crate::shared::target_head::TargetHead::Tensor,
                    head, f.ptr("out"), f.ptr("head_weight"), f.ptr("logits").cast(), rows, f.stream) }
            };
            project(&exact, &exact_head, ROWS)?;
            project(&padded, &padded_head, ROWS)?;
            exact.sync()?;
            padded.sync()?;
            for bucket in metadata::OWNER_ROWS {
                let graph = Captured::new(&padded, || project(&padded, &padded_head, bucket))?;
                for rows in 1..=ROWS {
                    if metadata::owner_row_bucket(rows) != bucket { continue; }
                    exact.reset()?;
                    padded.reset()?;
                    project(&exact, &exact_head, rows)?;
                    graph.replay()?;
                    padded.compare(&exact, rows, &format!("{family} staged-head"))?;
                    let bytes = rows * VOCAB * 4;
                    let dst = padded.tensors["packed"].dev.buffer;
                    // SAFETY: copy exactly the real rows to a nonzero packed
                    // origin, with canaries before and after the landing.
                    unsafe { library.copy_d2d_async(CuteafdDeviceBuffer {
                        ptr: dst.ptr.cast::<u8>().add(VOCAB * 4).cast(), bytes: dst.bytes - VOCAB * 4, ..dst
                    }, padded.tensors["logits"].dev.buffer, bytes, padded.stream)?; }
                    padded.sync()?;
                    let landed = padded.read("packed", dst.bytes)?;
                    ensure!(landed[..VOCAB * 4].iter().all(|&b| b == 0xa5), "head prefix canary");
                    ensure!(landed[VOCAB * 4 + bytes..].iter().all(|&b| b == 0xa5), "head padded-row leak");
                    ensure!(landed[VOCAB * 4..VOCAB * 4 + bytes] == exact.read("logits", bytes)?, "head landing differs");
                }
            }
            // Release the production-sized vocabulary matrix before draft storage.
            drop(padded_head);
            drop(exact_head);
            drop(padded);
            drop(exact);
            let mut exact = manifest_fixture(&library, producer_entry)?;
            let mut padded = manifest_fixture(&library, producer_entry)?;
            for f in [&mut exact, &mut padded] {
                f.add("tap_residual", seeded("bfloat16", ROWS * 4 * h, "tap_residual")?, Some(4 * h * 2), "in")?;
                f.add("taps", vec![0; ROWS * 3 * h * 2], Some(3 * h * 2), "out")?;
                f.add("main_proj", seeded("float8_e4m3fn", h * 3 * h, "main_proj")?, None, "in")?;
                f.add("main_proj_scale", vec![120; h / 128 * (3 * h / 128)], None, "in")?;
                f.add("main_norm", seeded("bfloat16", h, "main_norm")?, None, "in")?;
                f.add("main_work", vec![0; ROWS * h * 4], Some(h * 4), "out")?;
                f.tensors.get_mut("hidden").context("draft projection output")?.role = "out".to_owned();
                // Stage zero uses the manifest's cache; every additional stage
                // has independent seeded storage and must preserve its padding.
                for stage in 1..3 {
                    let seed = f.tensors["main_kv_cache"].initial.clone();
                    f.add(&format!("draft_cache_{stage}"), seed, None, "inout")?;
                }
            }
            let run = |f: &Fixture<'_>, rows: usize| -> Result<()> {
                for tap in 0..3 {
                    // SAFETY: [rows,4,h] source and [rows,3*h] tap destination
                    // are retained; each launch touches only its own tap columns.
                    unsafe { library.dsv4_hc_mean(f.ptr("tap_residual"), f.ptr("taps"), rows, h, 3 * h, tap * h, f.stream)?; }
                }
                // SAFETY: production-shaped block-FP8 tap projection, raw UE8M0
                // block scales, BF16 output and FP32 workspace are retained.
                unsafe { library.dsv4_fp8_linear_rmsnorm(f.ptr("taps"), f.ptr("main_proj"), f.ptr("main_proj_scale"),
                    f.ptr("main_norm"), f.ptr("hidden"), f.ptr("main_work"), rows, h, 3 * h, 1e-6, f.stream)?; }
                let names = producer_entry["pointers"].as_array().unwrap().iter()
                    .map(|p| p["name"].as_str().unwrap()).collect::<Vec<_>>();
                let name = producer_entry["name"].as_str().unwrap();
                let program = programs.program(name, &names)?;
                for stage in 0..3 {
                    let cache_name = format!("draft_cache_{stage}");
                    let pointers = names.iter().map(|name| if *name == "main_kv_cache" && stage != 0 {
                        f.ptr(&cache_name)
                    } else { f.ptr(name) }).collect::<Vec<_>>();
                    // SAFETY: each stage writes a disjoint full KV allocation;
                    // dummy destinations are negative in retained device metadata.
                    unsafe { program.launch(&pointers, &[Scalar::I32(rows as i32)], f.stream)?; }
                }
                Ok(())
            };
            run(&exact, ROWS)?;
            run(&padded, ROWS)?;
            exact.sync()?;
            padded.sync()?;
            for bucket in metadata::OWNER_ROWS {
                let graph = Captured::new(&padded, || run(&padded, bucket))?;
                for rows in 1..=ROWS {
                    if metadata::owner_row_bucket(rows) != bucket { continue; }
                    exact.reset()?;
                    padded.reset()?;
                    mask_metadata(&mut exact, producer_entry, rows)?;
                    mask_metadata(&mut padded, producer_entry, rows)?;
                    run(&exact, rows)?;
                    graph.replay()?;
                    padded.compare(&exact, rows, &format!("{family} tap/draft-kv"))?;
                }
            }
            println!("device={device} family={family} head/tap/draft-kv widths=1..64 buckets=22 byte-exact, three full draft caches, real-only head landing");
        }
    }
    Ok(())
}

struct ExpertVariant<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; V41_EXPERT_POINTER_COUNT],
    fixture: Fixture<'a>,
}

impl<'a> ExpertVariant<'a> {
    fn new(library: &'a NativeLibrary, capacity: u32) -> Result<Self> {
        let mut fixture = Fixture::new(library)?;
        let kernel = library.v41_tp2_expert_kernel(capacity)?;
        let info = *kernel.info();
        ensure!(info.hidden_size == 4096 && info.input_dtype == 7,
            "Flash TP2 fixture requires the production FP8 K32 input geometry");
        fixture.add("scratch", vec![0; info.scratch_bytes as usize], None, "retained")?;
        fixture.add("dummy", vec![0; 16], None, "in")?;
        let mut slots = [fixture.ptr("dummy"); V41_EXPERT_POINTER_COUNT];
        // SAFETY: private retained scratch covers this compiled variant and is
        // initialized before warmup/capture; subsequent launches restore counters.
        unsafe {
            kernel.bind_scratch(fixture.ptr("scratch"), info.scratch_bytes, &mut slots)?;
            kernel.initialize_scratch(fixture.ptr("scratch"), info.scratch_bytes, fixture.stream)?;
        }
        let h = info.hidden_size as usize;
        let k = info.topk as usize;
        let mut wire = vec![0x18; ROWS * info.input_row_bytes()?];
        for row in wire.chunks_exact_mut(info.input_row_bytes()?) {
            row[h..].fill(127);
        }
        fixture.add("wire", wire, None, "in")?;
        fixture.add("ids", i32_bytes(std::iter::repeat_n(0, ROWS * k)), Some(k * 4), "out")?;
        fixture.add("routing", f32_bytes(std::iter::repeat_n(1.0 / k as f32, ROWS * k)), Some(k * 4), "out")?;
        fixture.tensors.get_mut("routing").unwrap().dtype = Some("float32".to_owned());
        fixture.add("logits", f32_bytes(std::iter::repeat_n(0.25, ROWS * info.experts as usize)), None, "in")?;
        fixture.add("hash", i32_bytes(std::iter::repeat_n(0, k)), None, "in")?;
        fixture.add("tokens", i32_bytes(std::iter::repeat_n(0, ROWS)), None, "in")?;
        fixture.add("main_slots", i64_bytes(std::iter::repeat_n(0, ROWS)), None, "in")?;
        // All real routes select expert zero. Its production packed strides
        // contain finite nonzero FP4 values and E8M0 scales; no full layer needed.
        let extent = info.kernel_intermediate as usize * h;
        for (slot, bytes, value) in [(30, extent, 0x22), (31, extent / 16, 120),
            (32, extent / 2, 0x22), (33, extent / 32, 120)] {
            let name = format!("weight_{slot}");
            fixture.add(&name, vec![value; bytes], None, "in")?;
            slots[slot] = fixture.ptr(&name);
        }
        slots[0] = fixture.ptr("wire");
        slots[1] = fixture.ptr("ids");
        slots[2] = fixture.ptr("routing");
        fixture.add("sums", vec![0; ROWS * h * 4], Some(h * 4), "out")?;
        fixture.tensors.get_mut("sums").unwrap().dtype = Some("float32".to_owned());
        fixture.sync()?;
        Ok(Self { kernel, slots, fixture })
    }

    fn mask(&mut self, rows: usize) -> Result<()> {
        self.fixture.set("main_slots", i64_bytes((0..ROWS).map(|i| if i < rows { 0 } else { -1 })))
    }

    fn launch(&self, rows: usize) -> Result<()> {
        let info = self.kernel.info();
        let args = V41ExpertLaunchArgs::new(info, self.slots, rows as u32, self.fixture.stream)?;
        let library = self.fixture.library;
        // SAFETY: the typed kernel, resident expert-zero packed operands and
        // route/scratch/output extents remain retained through graph retirement.
        unsafe {
            // One valid hash token routes every plane to expert zero. The
            // production fused mask invalidates dummy routes inside capture.
            library.dsv4_router_select_masked(self.fixture.ptr("logits"), std::ptr::null(),
                self.fixture.ptr("hash"), self.fixture.ptr("tokens"), self.fixture.ptr("main_slots").cast(),
                self.fixture.ptr("ids"), self.fixture.ptr("routing"), rows, info.experts as usize,
                info.topk as usize, 1.0, self.fixture.stream)?;
            self.kernel.launch(&args)?;
            ensure!(self.kernel.output_kind() == V41ExpertOutputKind::Fp32Routes,
                "Flash TP2 byte oracle requires canonical ordered FP32 route planes");
            library.v41_tp2_expert_reducer()?.sum_routes_geometry(CuteafdDeviceBuffer {
                ptr: self.slots[41], bytes: rows * info.hidden_size as usize * info.topk as usize * 4,
                device_id: library.cuda_get_device()?, flags: 0,
            }, self.fixture.tensors["sums"].dev.buffer, rows as u32,
                cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH, self.fixture.stream)
        }
    }
}

impl Drop for ExpertVariant<'_> {
    fn drop(&mut self) {
        // Drain before the kernel handle drops, including failed launches.
        let _ = self.fixture.sync();
    }
}

#[test]
#[ignore = "two RTX GPUs and sealed native Flash MXFP4 TP2 experts"]
fn owner_bucket_flash_routed_tp2_is_byte_exact_at_every_width() -> Result<()> {
    // SAFETY: the matching sealed ABI outlives every retained variant/graph.
    let library = fixture_library()?;
    for device in 0..2 {
        library.cuda_set_device(device)?;
        let mut exact = [1, 16, 80].into_iter().map(|c| ExpertVariant::new(&library, c)).collect::<Result<Vec<_>>>()?;
        let mut padded = [1, 16, 80].into_iter().map(|c| ExpertVariant::new(&library, c)).collect::<Result<Vec<_>>>()?;
        let plan = crate::shared::experts::ExpertReductionPlan::native(&library,
            exact.iter().map(|v| *v.kernel.info()))?;
        let capacities = exact.iter().map(|v| v.kernel.info().capacity_rows as usize).collect::<Vec<_>>();
        let select = |rows| plan.execution_index(rows, rows, capacities.iter().copied());
        for variant in exact.iter().chain(&padded) {
            variant.launch((variant.kernel.info().capacity_rows as usize).min(ROWS))?;
            variant.fixture.sync()?;
        }
        for bucket in metadata::OWNER_ROWS {
            let captured = select(bucket)?;
            let graph = Captured::new(&padded[captured].fixture, || padded[captured].launch(bucket))?;
            for rows in 1..=ROWS {
                if metadata::owner_row_bucket(rows) != bucket { continue; }
                let reference = select(rows)?;
                exact[reference].mask(rows)?;
                padded[captured].mask(rows)?;
                for _ in 0..2 {
                    exact[reference].launch(rows)?;
                    graph.replay()?;
                    padded[captured].fixture.compare(&exact[reference].fixture, rows, "Flash native TP2 routed")?;
                }
                let fixture = &padded[captured].fixture;
                let topk = padded[captured].kernel.info().topk as usize;
                let ids = fixture.read("ids", bucket * topk * 4)?;
                let routing = fixture.read("routing", bucket * topk * 4)?;
                ensure!(ids[rows * topk * 4..].iter().all(|&b| b == 0xff), "captured dummy expert IDs were not invalidated");
                ensure!(routing[rows * topk * 4..].iter().all(|&b| b == 0), "captured dummy route weights are not positive zero");
                let output = fixture.read("sums", rows * 4096 * 4)?;
                ensure!(output.chunks_exact(4).any(|b| f32::from_ne_bytes(b.try_into().unwrap()) != 0.0),
                    "Flash routed oracle is vacuous at rows={rows}");
            }
        }
        println!("device={device} Flash native MXFP4 TP2 routed widths=1..64 buckets=22 canonical capacity selection, nonzero FP32 sums byte-exact");
    }
    Ok(())
}
