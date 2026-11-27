//! Inkling biased-sink GQA over shared local rings and global paged records.
//! Producers supply clipped queries/relative bias and normalized KV; the
//! caller owns all storage and orders writes before these attention launches.
#[path = "inkling2_producers.rs"]
pub mod producers;
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

const LOCAL: &[&str] = &["q", "kv_step", "ring", "positions", "ring_slots", "seq_first", "tile_rows", "rel", "sinks", "out", "scratch"];
const GLOBAL: &[&str] = &["q", "kv_cache", "positions", "page_table", "table_indices", "seq_first", "tile_rows", "rel", "sinks", "out", "scratch"];
pub const RECORD_BYTES: usize = 4096;
pub const PAGE_ROWS: usize = 64;
pub const RING_ROWS: usize = 1088;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind { Local, Global }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route { Decode, Prefill }

impl Kind { fn name(self) -> &'static str { match self { Self::Local => "local", Self::Global => "global" } } }
impl Route {
    fn name(self) -> &'static str { match self { Self::Decode => "decode", Self::Prefill => "prefill" } }
    fn capacity(self) -> usize { match self { Self::Decode => 64, Self::Prefill => 4096 } }
}

pub struct Attention<'a> {
    program: Program<'a>,
    kind: Kind,
    route: Route,
    capacity: usize,
    scratch: usize,
    fp8: bool,
    head_parts: usize,
}

/// One stream-ordered step. Metadata is device resident and validated by its
/// owner: positions >=0, ring slots address their sequence's ring, first rows
/// partition the packed step, and page-table entries address allocated pages.
/// Global rows select tables through table_indices; tile_rows names each CTA
/// start and first prevents prefill tiles from crossing packed segments. No
/// producer may overwrite a retained history before use.
#[derive(Clone, Copy, Debug)]
pub struct Batch {
    pub rows: usize,
    pub tiles: usize,
    pub tile_rows: Buffer,
    pub q: Buffer,
    pub kv: Buffer,
    pub positions: Buffer,
    pub relative: Buffer,
    pub sinks: Buffer,
    pub output: Buffer,
    pub scratch: Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct Local {
    pub ring: Buffer,
    pub ring_stride_records: usize,
    pub ring_slots: Buffer,
    pub first: Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct Global {
    pub pages: usize,
    pub page_table: Buffer,
    pub table_width: usize,
    pub tables: usize,
    pub table_indices: Buffer,
    pub first: Buffer,
}

impl<'a> Attention<'a> {
    pub fn load(programs: &'a Programs<'a>, kind: Kind, route: Route, fp8: bool) -> Result<Self> {
        Self::load_rank(programs, kind, route, fp8, 1)
    }

    pub fn load_rank(programs: &'a Programs<'a>, kind: Kind, route: Route, fp8: bool,
        head_parts: usize) -> Result<Self> {
        Self::load_rank_rows(programs, kind, route, fp8, head_parts, route.capacity())
    }

    pub fn load_rank_rows(programs: &'a Programs<'a>, kind: Kind, route: Route, fp8: bool,
        head_parts: usize, capacity: usize) -> Result<Self> {
        ensure!(matches!(head_parts, 1 | 2), "Inkling attention head partition differs");
        ensure!(!fp8 || kind == Kind::Global, "Inkling FP8 local ring unsupported");
        let valid_capacity = match route {
            Route::Decode => matches!(capacity, 64 | 128),
            Route::Prefill => capacity == 4096,
        };
        ensure!(valid_capacity, "Inkling attention export capacity {capacity} unsupported for {route:?}");
        let name = format!("inkling2_{}_attention_{}_{}_v2_m{}{}", kind.name(), route.name(),
            if fp8 { "fp8" } else { "bf16" }, capacity, if head_parts == 2 { "_tp2" } else { "" });
        let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
        let program = programs.program(&name, pointers)?;
        let scratch = validate_manifest(program.spec(), kind, route, fp8, head_parts, capacity)?;
        Ok(Self { program, kind, route, capacity, scratch, fp8, head_parts })
    }

    pub fn scratch_bytes(&self) -> usize { self.scratch }

    pub fn local(&self, b: Batch, local: Local, stream: *mut c_void) -> Result<()> {
        ensure!(self.kind == Kind::Local, "Inkling local launch bound to global attention");
        let mut buffers = self.common(b)?;
        ensure!(local.ring_stride_records >= RING_ROWS, "Inkling ring stride is shorter than window");
        let ring_bytes = local.ring.bytes;
        let record_bytes = RECORD_BYTES / self.head_parts;
        ensure!(ring_bytes >= RING_ROWS*record_bytes, "Inkling ring bank is shorter than window");
        buffers.extend([(b.kv, b.rows * record_bytes, 16), (local.ring, ring_bytes, 16),
            (local.ring_slots, b.rows * 8, 8), (local.first, b.rows * 4, 4)]);
        validate_buffers(&buffers, &[b.output, b.scratch, local.ring])?;
        // SAFETY: checked extents, device, alignment and exclusive writers;
        // caller retains metadata, histories and stream through completion.
        unsafe { self.program.launch(&[b.q.ptr, b.kv.ptr, local.ring.ptr, b.positions.ptr,
            local.ring_slots.ptr, local.first.ptr, b.tile_rows.ptr, b.relative.ptr, b.sinks.ptr,
            b.output.ptr, b.scratch.ptr], &[Scalar::I32(b.rows as i32), Scalar::I32(b.tiles as i32)], stream)?; }
        Ok(())
    }

    pub fn global(&self, b: Batch, global: Global, stream: *mut c_void) -> Result<()> {
        ensure!(self.kind == Kind::Global, "Inkling global launch bound to local attention");
        ensure!(global.table_width > 0 && global.table_width <= i32::MAX as usize
            && global.pages > 0 && global.pages <= i32::MAX as usize,
            "Inkling page geometry differs");
        let mut buffers = self.common(b)?;
        let cache = global.pages.checked_mul(PAGE_ROWS).and_then(|n| n.checked_mul((if self.fp8 { 2112 } else { RECORD_BYTES }) / self.head_parts))
            .context("Inkling cache extent overflow")?;
        let stride = global.table_width;
        let tables = global.tables;
        ensure!(tables > 0, "Inkling page table bank empty");
        let bytes = tables.checked_mul(global.table_width).and_then(|n| n.checked_mul(4))
            .context("Inkling page-table extent overflow")?;
        buffers.extend([(b.kv, cache, 16), (global.page_table, bytes, 4),
            (global.table_indices, b.rows*4, 4), (global.first, b.rows*4, 4)]);
        validate_buffers(&buffers, &[b.output, b.scratch])?;
        let scalars = [Scalar::I32(b.rows as i32), Scalar::I32(stride as i32), Scalar::I32(b.tiles as i32)];
        // SAFETY: checked extents, device, alignment and writer aliases;
        // owner validates table contents and retains all operands until drain.
        // For global decode, dsv4_programs.cc injects device_sm_count from
        // cudaDevAttrMultiProcessorCount for the trailing 's' scalar slot.
        unsafe { self.program.launch(&[b.q.ptr, b.kv.ptr, b.positions.ptr, global.page_table.ptr, global.table_indices.ptr, global.first.ptr, b.tile_rows.ptr,
            b.relative.ptr, b.sinks.ptr, b.output.ptr, b.scratch.ptr], &scalars, stream)?; }
        Ok(())
    }

    fn common(&self, b: Batch) -> Result<Vec<(Buffer, usize, usize)>> {
        ensure!((1..=self.capacity).contains(&b.rows) && (1..=b.rows).contains(&b.tiles), "Inkling attention live rows exceed export");
        let heads = 48 / self.head_parts;
        Ok(vec![(b.tile_rows, b.tiles*4, 4), (b.q, b.rows * heads * 128 * 2, 16), (b.positions, b.rows * 8, 8),
            (b.relative, b.rows * heads * 1024 * 2, 16), (b.sinks, b.rows * heads * 4, 4),
            (b.output, b.rows * heads * 128 * 2, 16), (b.scratch, self.scratch, 16)])
    }
}

fn validate_manifest(spec: &ProgramSpec, kind: Kind, route: Route, fp8: bool, parts: usize, capacity: usize) -> Result<usize> {
    let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
    // Global decode exports kinds="iiis" (device_sm_count injected by dsv4_programs.cc);
    // cuteafd_dsv4_program_info exposes the 3 public scalars b"iii".
    let scalars: &[u8] = match kind { Kind::Local => b"ii", Kind::Global => b"iii" };
    ensure!(spec.pointers.iter().map(String::as_str).eq(pointers.iter().copied())
        && spec.scalar_kinds == scalars && spec.capacity_rows as usize == capacity,
        "Inkling attention export ABI/capacity differs");
    let g = &spec.geometry;
    let global_decode = kind == Kind::Global && route == Route::Decode;
    for (key, value) in [("heads", 48 / parts), ("kv_heads", 8 / parts), ("head_dim", 128), ("record_elems", 2048 / parts),
        ("record_bytes", (if fp8 { 2112 } else { RECORD_BYTES }) / parts), ("cache_abi", 2),
        ("scale_offset", if fp8 { 2048 / parts } else { 0 }), ("page_rows", PAGE_ROWS), ("ring_rows", RING_ROWS),
        ("extent", 1024), ("max_rows", capacity),
        ("window", if kind == Kind::Local { 1024 } else { 0 }),
        ("max_splits", if global_decode { 32 } else { 1 }),
        ("split_chunk", if global_decode { 512 } else { 0 }),
        ("rows_per_cta", if route == Route::Decode { 1 } else { 10 })] {
        ensure!(g[key].as_u64() == Some(value as u64), "Inkling attention export geometry {key} differs");
    }
    ensure!(g["runtime_sm_count"].as_bool() == Some(global_decode),
        "Inkling attention export runtime_sm_count differs");
    ensure!(g["kv_format"] == if fp8 { "fp8" } else { "bf16" }
        && g["kind"] == kind.name() && g["route"] == route.name() && g["sinks"] == "per_row_head",
        "Inkling attention export bias/sink convention differs");
    let expected = if global_decode {
        capacity * 32 * (48 / parts) * (128 * 4 + 4)
    } else { 1024 };
    ensure!(spec.scratch.len() == 1 && spec.scratch.get("scratch") == Some(&(expected as u64)),
        "Inkling attention scratch differs");
    Ok(expected)
}

fn validate_buffers(buffers: &[(Buffer, usize, usize)], writers: &[Buffer]) -> Result<()> {
    let device = buffers[0].0.device_id;
    ensure!(device >= 0, "Inkling attention requires device storage");
    let mut ranges = Vec::new();
    for &(buffer, bytes, align) in buffers {
        ensure!(buffer.device_id == device && buffer.flags == 0 && !buffer.ptr.is_null()
            && buffer.ptr as usize % align == 0 && buffer.bytes >= bytes,
            "Inkling attention operand device/alignment/extent differs");
        let start = buffer.ptr as usize;
        ranges.push((start, start.checked_add(bytes).context("Inkling attention pointer overflow")?));
    }
    for writer in writers {
        let index = buffers.iter().position(|(buffer, _, _)| buffer.ptr == writer.ptr)
            .context("Inkling writer not in operands")?;
        let (a, b) = ranges[index];
        for (other, &(c, d)) in ranges.iter().enumerate() {
            ensure!(index == other || b <= c || d <= a, "Inkling attention writer aliases operand");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec(kind: Kind, route: Route) -> ProgramSpec {
        let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
        let expected = if kind == Kind::Global && route == Route::Decode {
            route.capacity() * 32 * 48 * (128 * 4 + 4)
        } else { 1024 };
        let global_decode = kind == Kind::Global && route == Route::Decode;
        ProgramSpec { name: format!("inkling2_{}_attention_{}_m{}", kind.name(), route.name(), route.capacity()),
            pointers: pointers.iter().map(|s| s.to_string()).collect(),
            scalar_kinds: match kind { Kind::Local => b"ii".to_vec(), Kind::Global => b"iii".to_vec() },
            scratch: [("scratch".into(), expected as u64)].into(), capacity_rows: route.capacity() as u32,
            geometry: serde_json::json!({"kind": kind.name(), "route": route.name(), "heads": 48,
                "kv_heads": 8, "head_dim": 128, "record_elems": 2048, "record_bytes": RECORD_BYTES, "cache_abi": 2, "kv_format": "bf16", "scale_offset": 0,
                "max_rows": route.capacity(), "window": if kind == Kind::Local { 1024 } else { 0 },
                "extent": 1024, "ring_rows": RING_ROWS, "page_rows": PAGE_ROWS, "sinks": "per_row_head",
                "max_splits": if global_decode { 32 } else { 1 },
                "split_chunk": if global_decode { 512 } else { 0 },
                "runtime_sm_count": global_decode,
                "rows_per_cta": if route == Route::Decode { 1 } else { 10 } }) }
    }
    #[test]
    fn half_head_fp8_decode_checks_rank_local_record_and_scratch() {
        let route = Route::Decode;
        let kind = Kind::Global;
        let mut rank = spec(kind, route);
        for (key, value) in [("heads", 24), ("kv_heads", 4), ("record_elems", 1024),
            ("record_bytes", 1056), ("scale_offset", 1024)] {
            rank.geometry[key] = serde_json::json!(value);
        }
        rank.geometry["kv_format"] = serde_json::json!("fp8");
        let scratch = 64 * 32 * 24 * (128 * 4 + 4);
        rank.scratch.insert("scratch".into(), scratch as u64);
        assert_eq!(validate_manifest(&rank, kind, route, true, 2, 64).unwrap(), scratch);
        assert!(validate_manifest(&rank, kind, route, true, 1, 64).is_err());
        rank.geometry["record_bytes"] = serde_json::json!(2112);
        assert!(validate_manifest(&rank, kind, route, true, 2, 64).is_err());
        // Verify 128-row decode manifest geometry and scratch scaling.
        let mut wide = spec(kind, route);
        wide.capacity_rows = 128;
        wide.geometry["max_rows"] = serde_json::json!(128);
        let wide_scratch = 128 * 32 * 48 * (128 * 4 + 4);
        wide.scratch.insert("scratch".into(), wide_scratch as u64);
        assert_eq!(validate_manifest(&wide, kind, route, false, 1, 128).unwrap(), wide_scratch);
        assert!(validate_manifest(&wide, kind, route, false, 1, 64).is_err());
    }

    #[test]
    fn manifest_accepts_only_checkpoint_bias_sink_geometry() {
        for kind in [Kind::Local, Kind::Global] {
            for route in [Route::Decode, Route::Prefill] {
                let cap = route.capacity();
                let good = spec(kind, route);
                validate_manifest(&good, kind, route, false, 1, cap).unwrap();
                for key in ["heads", "kv_heads", "head_dim", "ring_rows", "page_rows", "extent", "window", "max_splits", "split_chunk"] {
                    let mut wrong = good.clone();
                    wrong.geometry[key] = serde_json::json!(999);
                    assert!(validate_manifest(&wrong, kind, route, false, 1, cap).is_err(), "{key}");
                }
                let mut wrong = good.clone();
                wrong.geometry["runtime_sm_count"] = serde_json::json!(!(kind == Kind::Global && route == Route::Decode));
                assert!(validate_manifest(&wrong, kind, route, false, 1, cap).is_err());
                let mut wrong = good.clone();
                wrong.geometry["sinks"] = serde_json::json!("per_head");
                assert!(validate_manifest(&wrong, kind, route, false, 1, cap).is_err());
                let mut wrong = good.clone();
                wrong.scratch.insert("scratch".into(), 0);
                assert!(validate_manifest(&wrong, kind, route, false, 1, cap).is_err());
                let mut wrong = good.clone();
                wrong.pointers.swap(0, 1);
                assert!(validate_manifest(&wrong, kind, route, false, 1, cap).is_err());
                let mut wrong = good.clone();
                wrong.scalar_kinds[0] = b'l';
                assert!(validate_manifest(&wrong, kind, route, false, 1, cap).is_err());
            }
        }
    }
    fn buffer(ptr: usize, bytes: usize) -> Buffer {
        Buffer { ptr: ptr as *mut c_void, bytes, device_id: 0, flags: 0 }
    }
    #[test]
    fn writer_aliases_misalignment_device_and_overflow_refused() {
        let input = buffer(4096, 1024);
        let output = buffer(8192, 1024);
        validate_buffers(&[(input, 1024, 16), (output, 1024, 16)], &[output]).unwrap();
        assert!(validate_buffers(&[(input, 1024, 16), (buffer(4096 + 512, 1024), 1024, 16)],
            &[buffer(4096 + 512, 1024)]).is_err());
        assert!(validate_buffers(&[(buffer(4097, 1024), 1024, 16)], &[]).is_err());
        assert!(validate_buffers(&[(input, 1025, 16)], &[]).is_err());
        let mut wrong = output;
        wrong.device_id = 1;
        assert!(validate_buffers(&[(input, 1024, 16), (wrong, 1024, 16)], &[wrong]).is_err());
        assert!(validate_buffers(&[(buffer(usize::MAX - 15, 32), 32, 16)], &[]).is_err());
    }
}
