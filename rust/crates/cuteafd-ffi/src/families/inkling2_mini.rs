//! Inkling 2 Mini biased-sink GQA (24 Q / 8 KV heads, head_dim 128, group 3, prefill_tokens 21).
#[path = "inkling2_mini_producers.rs"]
pub mod producers;
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

pub use crate::inkling2::{Batch, Global, Kind, Local, Route, PAGE_ROWS, RECORD_BYTES, RING_ROWS};

const LOCAL: &[&str] = &["q", "kv_step", "ring", "positions", "ring_slots", "seq_first", "tile_rows", "rel", "sinks", "out", "scratch"];
const GLOBAL: &[&str] = &["q", "kv_cache", "positions", "page_table", "table_indices", "seq_first", "tile_rows", "rel", "sinks", "out", "scratch"];
pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const PREFILL_TOKENS: usize = 21;
pub const FP8_RECORD_BYTES: usize = 2112;

fn kind_name(kind: Kind) -> &'static str { match kind { Kind::Local => "local", Kind::Global => "global" } }
fn route_name(route: Route) -> &'static str { match route { Route::Decode => "decode", Route::Prefill => "prefill" } }
fn route_capacity(route: Route) -> usize { match route { Route::Decode => 64, Route::Prefill => 4096 } }

pub struct Attention<'a> {
    program: Program<'a>,
    kind: Kind,
    route: Route,
    capacity: usize,
    scratch: usize,
    fp8: bool,
}

impl<'a> Attention<'a> {
    pub fn load(programs: &'a Programs<'a>, kind: Kind, route: Route, fp8: bool) -> Result<Self> {
        Self::load_rows(programs, kind, route, fp8, route_capacity(route))
    }

    pub fn load_rows(
        programs: &'a Programs<'a>,
        kind: Kind,
        route: Route,
        fp8: bool,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(!fp8 || kind == Kind::Global, "Inkling 2 Mini FP8 local ring unsupported");
        let valid_cap = match route {
            Route::Decode => matches!(capacity, 64 | 128),
            Route::Prefill => capacity == 4096,
        };
        ensure!(valid_cap, "Inkling 2 Mini attention capacity {capacity} unsupported for {route:?}");
        let name = format!(
            "inkling2_mini_{}_attention_{}_{}_v2_m{}",
            kind_name(kind),
            route_name(route),
            if fp8 { "fp8" } else { "bf16" },
            capacity,
        );
        let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
        let program = programs.program(&name, pointers)?;
        let scratch = validate_manifest(program.spec(), kind, route, fp8, capacity)?;
        Ok(Self { program, kind, route, capacity, scratch, fp8 })
    }

    pub fn scratch_bytes(&self) -> usize { self.scratch }

    pub fn local(&self, b: Batch, local: Local, stream: *mut c_void) -> Result<()> {
        ensure!(self.kind == Kind::Local, "Inkling 2 Mini local launch bound to global attention");
        let mut buffers = self.common(b)?;
        ensure!(local.ring_stride_records >= RING_ROWS, "Inkling 2 Mini ring stride shorter than window");
        ensure!(local.ring.bytes >= RING_ROWS * RECORD_BYTES, "Inkling 2 Mini ring bank shorter than window");
        buffers.extend([
            (b.kv, b.rows * RECORD_BYTES, 16),
            (local.ring, local.ring.bytes, 16),
            (local.ring_slots, b.rows * 8, 8),
            (local.first, b.rows * 4, 4),
        ]);
        validate_buffers(&buffers, &[b.output, b.scratch, local.ring])?;
        // SAFETY: checked extents, device, alignment and exclusive writers;
        // caller retains metadata, histories and stream through completion.
        unsafe {
            self.program.launch(
                &[b.q.ptr, b.kv.ptr, local.ring.ptr, b.positions.ptr, local.ring_slots.ptr,
                  local.first.ptr, b.tile_rows.ptr, b.relative.ptr, b.sinks.ptr, b.output.ptr, b.scratch.ptr],
                &[Scalar::I32(b.rows as i32), Scalar::I32(b.tiles as i32)],
                stream,
            )?;
        }
        Ok(())
    }

    pub fn global(&self, b: Batch, global: Global, stream: *mut c_void) -> Result<()> {
        ensure!(self.kind == Kind::Global, "Inkling 2 Mini global launch bound to local attention");
        ensure!(global.table_width > 0 && global.table_width <= i32::MAX as usize
            && global.pages > 0 && global.pages <= i32::MAX as usize,
            "Inkling 2 Mini page geometry differs");
        let mut buffers = self.common(b)?;
        let record_bytes = if self.fp8 { FP8_RECORD_BYTES } else { RECORD_BYTES };
        let cache = global.pages.checked_mul(PAGE_ROWS)
            .and_then(|n| n.checked_mul(record_bytes))
            .context("Inkling 2 Mini cache extent overflow")?;
        ensure!(global.tables > 0, "Inkling 2 Mini page table bank empty");
        let table_bytes = global.tables.checked_mul(global.table_width)
            .and_then(|n| n.checked_mul(4))
            .context("Inkling 2 Mini page-table extent overflow")?;
        buffers.extend([
            (b.kv, cache, 16),
            (global.page_table, table_bytes, 4),
            (global.table_indices, b.rows * 4, 4),
            (global.first, b.rows * 4, 4),
        ]);
        validate_buffers(&buffers, &[b.output, b.scratch])?;
        let scalars = [
            Scalar::I32(b.rows as i32),
            Scalar::I32(global.table_width as i32),
            Scalar::I32(b.tiles as i32),
        ];
        // SAFETY: checked extents, device, alignment and writer aliases;
        // owner validates table contents and retains all operands until drain.
        // For global decode, dsv4_programs.cc injects device_sm_count from
        // cudaDevAttrMultiProcessorCount for the trailing 's' scalar slot.
        unsafe {
            self.program.launch(
                &[b.q.ptr, b.kv.ptr, b.positions.ptr, global.page_table.ptr, global.table_indices.ptr,
                  global.first.ptr, b.tile_rows.ptr, b.relative.ptr, b.sinks.ptr, b.output.ptr, b.scratch.ptr],
                &scalars,
                stream,
            )?;
        }
        Ok(())
    }

    fn common(&self, b: Batch) -> Result<Vec<(Buffer, usize, usize)>> {
        ensure!((1..=self.capacity).contains(&b.rows) && (1..=b.rows).contains(&b.tiles),
            "Inkling 2 Mini attention live rows exceed export");
        Ok(vec![
            (b.tile_rows, b.tiles * 4, 4),
            (b.q, b.rows * HEADS * HEAD_DIM * 2, 16),
            (b.positions, b.rows * 8, 8),
            (b.relative, b.rows * HEADS * 1024 * 2, 16),
            (b.sinks, b.rows * HEADS * 4, 4),
            (b.output, b.rows * HEADS * HEAD_DIM * 2, 16),
            (b.scratch, self.scratch, 16),
        ])
    }
}

fn validate_manifest(spec: &ProgramSpec, kind: Kind, route: Route, fp8: bool, capacity: usize) -> Result<usize> {
    let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
    // Global decode exports kinds="iiis" (device_sm_count injected by dsv4_programs.cc);
    // cuteafd_dsv4_program_info exposes the 3 public scalars b"iii".
    let scalars: &[u8] = match kind {
        Kind::Local => b"ii",
        Kind::Global => b"iii",
    };
    ensure!(spec.pointers.iter().map(String::as_str).eq(pointers.iter().copied())
        && spec.scalar_kinds == scalars
        && spec.capacity_rows as usize == capacity,
        "Inkling 2 Mini attention export ABI/capacity differs");
    let g = &spec.geometry;
    let global_decode = kind == Kind::Global && route == Route::Decode;
    for (key, value) in [
        ("heads", HEADS),
        ("kv_heads", KV_HEADS),
        ("head_dim", HEAD_DIM),
        ("record_elems", 2048),
        ("record_bytes", if fp8 { FP8_RECORD_BYTES } else { RECORD_BYTES }),
        ("cache_abi", 2),
        ("scale_offset", if fp8 { 2048 } else { 0 }),
        ("page_rows", PAGE_ROWS),
        ("ring_rows", RING_ROWS),
        ("extent", 1024),
        ("max_rows", capacity),
        ("window", if kind == Kind::Local { 1024 } else { 0 }),
        ("max_splits", if global_decode { 32 } else { 1 }),
        ("split_chunk", if global_decode { 512 } else { 0 }),
        ("rows_per_cta", if route == Route::Decode { 1 } else { PREFILL_TOKENS }),
    ] {
        ensure!(g[key].as_u64() == Some(value as u64), "Inkling 2 Mini attention geometry {key} differs");
    }
    ensure!(g["runtime_sm_count"].as_bool() == Some(global_decode),
        "Inkling 2 Mini attention runtime_sm_count differs");
    ensure!(g["kv_format"] == if fp8 { "fp8" } else { "bf16" }
        && g["kind"] == kind_name(kind)
        && g["route"] == route_name(route)
        && g["sinks"] == "per_row_head",
        "Inkling 2 Mini attention bias/sink convention differs");
    let expected = if global_decode {
        capacity * 32 * HEADS * (HEAD_DIM * 4 + 4)
    } else {
        1024
    };
    ensure!(spec.scratch.len() == 1 && spec.scratch.get("scratch") == Some(&(expected as u64)),
        "Inkling 2 Mini attention scratch differs");
    Ok(expected)
}

fn validate_buffers(buffers: &[(Buffer, usize, usize)], writers: &[Buffer]) -> Result<()> {
    let device = buffers[0].0.device_id;
    ensure!(device >= 0, "Inkling 2 Mini attention requires device storage");
    let mut ranges = Vec::with_capacity(buffers.len());
    for &(buffer, bytes, align) in buffers {
        ensure!(buffer.device_id == device && buffer.flags == 0 && !buffer.ptr.is_null()
            && buffer.ptr as usize % align == 0 && buffer.bytes >= bytes,
            "Inkling 2 Mini attention operand device/alignment/extent differs");
        let start = buffer.ptr as usize;
        ranges.push((start, start.checked_add(bytes).context("Inkling 2 Mini attention pointer overflow")?));
    }
    for writer in writers {
        let index = buffers.iter().position(|(buffer, _, _)| buffer.ptr == writer.ptr)
            .context("Inkling 2 Mini writer not in operands")?;
        let (a, b) = ranges[index];
        for (other, &(c, d)) in ranges.iter().enumerate() {
            ensure!(index == other || b <= c || d <= a, "Inkling 2 Mini attention writer aliases operand");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: Kind, route: Route, fp8: bool, capacity: usize) -> ProgramSpec {
        let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
        let global_decode = kind == Kind::Global && route == Route::Decode;
        let scratch = if global_decode {
            capacity * 32 * HEADS * (HEAD_DIM * 4 + 4)
        } else {
            1024
        };
        ProgramSpec {
            name: format!(
                "inkling2_mini_{}_attention_{}_{}_v2_m{capacity}",
                kind_name(kind),
                route_name(route),
                if fp8 { "fp8" } else { "bf16" },
            ),
            pointers: pointers.iter().map(|s| (*s).into()).collect(),
            scalar_kinds: match kind {
                Kind::Local => b"ii".to_vec(),
                Kind::Global => b"iii".to_vec(),
            },
            capacity_rows: capacity as u32,
            scratch: [("scratch".into(), scratch as u64)].into(),
            geometry: serde_json::json!({
                "heads": HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "record_elems": 2048,
                "record_bytes": if fp8 { FP8_RECORD_BYTES } else { RECORD_BYTES },
                "cache_abi": 2,
                "kv_format": if fp8 { "fp8" } else { "bf16" },
                "scale_offset": if fp8 { 2048 } else { 0 },
                "page_rows": PAGE_ROWS,
                "ring_rows": RING_ROWS,
                "extent": 1024,
                "max_rows": capacity,
                "window": if kind == Kind::Local { 1024 } else { 0 },
                "max_splits": if global_decode { 32 } else { 1 },
                "split_chunk": if global_decode { 512 } else { 0 },
                "runtime_sm_count": global_decode,
                "rows_per_cta": if route == Route::Decode { 1 } else { PREFILL_TOKENS },
                "kind": kind_name(kind),
                "route": route_name(route),
                "sinks": "per_row_head",
            }),
        }
    }

    #[test]
    fn decode_manifest_validates_64_and_128_row_geometry_and_scratch() {
        for cap in [64usize, 128] {
            for (kind, fp8) in [(Kind::Local, false), (Kind::Global, false), (Kind::Global, true)] {
                let s = spec(kind, Route::Decode, fp8, cap);
                let expected = if kind == Kind::Global {
                    cap * 32 * HEADS * (HEAD_DIM * 4 + 4)
                } else {
                    1024
                };
                assert_eq!(validate_manifest(&s, kind, Route::Decode, fp8, cap).unwrap(), expected);
                let other = if cap == 64 { 128 } else { 64 };
                assert!(validate_manifest(&s, kind, Route::Decode, fp8, other).is_err());
            }
        }
    }

    #[test]
    fn manifest_validates_runtime_sm_count_and_split_chunk() {
        for (kind, fp8) in [(Kind::Local, false), (Kind::Global, false), (Kind::Global, true)] {
            for route in [Route::Decode, Route::Prefill] {
                let cap = route_capacity(route);
                let good = spec(kind, route, fp8, cap);
                validate_manifest(&good, kind, route, fp8, cap).unwrap();
                let mut bad_chunk = good.clone();
                bad_chunk.geometry["split_chunk"] = serde_json::json!(256);
                assert!(validate_manifest(&bad_chunk, kind, route, fp8, cap).is_err());
                let mut bad_runtime = good.clone();
                bad_runtime.geometry["runtime_sm_count"] =
                    serde_json::json!(!(kind == Kind::Global && route == Route::Decode));
                assert!(validate_manifest(&bad_runtime, kind, route, fp8, cap).is_err());
                let mut bad_scalars = good.clone();
                bad_scalars.scalar_kinds.push(b'i');
                assert!(validate_manifest(&bad_scalars, kind, route, fp8, cap).is_err());
            }
        }
    }
}
