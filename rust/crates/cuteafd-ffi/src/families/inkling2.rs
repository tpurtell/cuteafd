//! Inkling biased-sink GQA over shared local rings and global paged records.
//! Producers supply clipped queries/relative bias and normalized KV; the
//! caller owns all storage and orders writes before these attention launches.
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

const LOCAL: &[&str] = &["q", "kv_step", "ring", "positions", "ring_slots", "seq_first", "rel", "sinks", "out", "scratch"];
const GLOBAL: &[&str] = &["q", "kv_cache", "positions", "page_table", "rel", "sinks", "out", "scratch"];
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
    scratch: usize,
}

/// One stream-ordered step. Metadata is device resident and validated by its
/// owner: positions >=0, ring slots address their sequence's ring, first rows
/// partition the packed step, and page-table entries address allocated pages.
/// Prefill global attention accepts one sequence (table_stride=0); decode has
/// a table per row. No producer may overwrite a retained history before use.
#[derive(Clone, Copy, Debug)]
pub struct Batch {
    pub rows: usize,
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
    pub ring_count: usize,
    pub ring_slots: Buffer,
    pub first: Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct Global {
    pub pages: usize,
    pub page_table: Buffer,
    pub table_width: usize,
    pub splits: usize,
}

impl<'a> Attention<'a> {
    pub fn load(programs: &'a Programs<'a>, kind: Kind, route: Route) -> Result<Self> {
        let name = format!("inkling2_{}_attention_{}_m{}", kind.name(), route.name(), route.capacity());
        let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
        let program = programs.program(&name, pointers)?;
        let scratch = validate_manifest(program.spec(), kind, route)?;
        Ok(Self { program, kind, route, scratch })
    }
    pub fn scratch_bytes(&self) -> usize { self.scratch }

    pub fn local(&self, b: Batch, local: Local, stream: *mut c_void) -> Result<()> {
        ensure!(self.kind == Kind::Local, "Inkling local launch bound to global attention");
        let mut buffers = self.common(b)?;
        let ring_bytes = local.ring_count.checked_mul(RING_ROWS).and_then(|v| v.checked_mul(RECORD_BYTES))
            .filter(|&n| n > 0).context("Inkling ring extent overflow/empty")?;
        buffers.extend([(b.kv, b.rows * RECORD_BYTES, 16), (local.ring, ring_bytes, 16),
            (local.ring_slots, b.rows * 8, 8), (local.first, b.rows * 4, 4)]);
        validate_buffers(&buffers, &[b.output, b.scratch, local.ring])?;
        // SAFETY: checked extents, device, alignment and exclusive writers;
        // caller retains metadata, histories and stream through completion.
        unsafe { self.program.launch(&[b.q.ptr, b.kv.ptr, local.ring.ptr, b.positions.ptr,
            local.ring_slots.ptr, local.first.ptr, b.relative.ptr, b.sinks.ptr,
            b.output.ptr, b.scratch.ptr], &[Scalar::I32(b.rows as i32)], stream)?; }
        Ok(())
    }

    pub fn global(&self, b: Batch, global: Global, stream: *mut c_void) -> Result<()> {
        ensure!(self.kind == Kind::Global, "Inkling global launch bound to local attention");
        ensure!(global.table_width > 0 && global.table_width <= i32::MAX as usize
            && global.pages > 0 && global.pages <= i32::MAX as usize
            && (1..=32).contains(&global.splits)
            && (self.route == Route::Decode || global.splits == 1), "Inkling page/split geometry differs");
        let mut buffers = self.common(b)?;
        let cache = global.pages.checked_mul(PAGE_ROWS).and_then(|n| n.checked_mul(RECORD_BYTES))
            .context("Inkling cache extent overflow")?;
        let stride = if self.route == Route::Decode { global.table_width } else { 0 };
        let tables = if self.route == Route::Decode { b.rows } else { 1 };
        let bytes = tables.checked_mul(global.table_width).and_then(|n| n.checked_mul(4))
            .context("Inkling page-table extent overflow")?;
        buffers.extend([(b.kv, cache, 16), (global.page_table, bytes, 4)]);
        validate_buffers(&buffers, &[b.output, b.scratch])?;
        let mut scalars = vec![Scalar::I32(b.rows as i32), Scalar::I32(stride as i32)];
        if self.route == Route::Decode { scalars.push(Scalar::I32(global.splits as i32)); }
        // SAFETY: checked extents, device, alignment and writer aliases;
        // owner validates table contents and retains all operands until drain.
        unsafe { self.program.launch(&[b.q.ptr, b.kv.ptr, b.positions.ptr, global.page_table.ptr,
            b.relative.ptr, b.sinks.ptr, b.output.ptr, b.scratch.ptr], &scalars, stream)?; }
        Ok(())
    }

    fn common(&self, b: Batch) -> Result<Vec<(Buffer, usize, usize)>> {
        ensure!((1..=self.route.capacity()).contains(&b.rows), "Inkling attention live rows exceed export");
        Ok(vec![(b.q, b.rows * 48 * 128 * 2, 16), (b.positions, b.rows * 8, 8),
            (b.relative, b.rows * 48 * 1024 * 2, 16), (b.sinks, b.rows * 48 * 4, 4),
            (b.output, b.rows * 48 * 128 * 2, 16), (b.scratch, self.scratch, 16)])
    }
}

fn validate_manifest(spec: &ProgramSpec, kind: Kind, route: Route) -> Result<usize> {
    let pointers = if kind == Kind::Local { LOCAL } else { GLOBAL };
    let scalars: &[u8] = match (kind, route) { (Kind::Local, _) => b"i",
        (Kind::Global, Route::Prefill) => b"ii", (Kind::Global, Route::Decode) => b"iii" };
    ensure!(spec.pointers.iter().map(String::as_str).eq(pointers.iter().copied())
        && spec.scalar_kinds == scalars && spec.capacity_rows as usize == route.capacity(),
        "Inkling attention export ABI/capacity differs");
    let g = &spec.geometry;
    for (key, value) in [("heads", 48), ("kv_heads", 8), ("head_dim", 128), ("record_elems", 2048),
        ("record_bytes", RECORD_BYTES), ("page_rows", PAGE_ROWS), ("ring_rows", RING_ROWS),
        ("extent", 1024), ("max_rows", route.capacity()),
        ("window", if kind == Kind::Local { 1024 } else { 0 }),
        ("max_splits", if kind == Kind::Global && route == Route::Decode { 32 } else { 1 }),
        ("rows_per_cta", if route == Route::Decode { 1 } else { 10 })] {
        ensure!(g[key].as_u64() == Some(value as u64), "Inkling attention export geometry {key} differs");
    }
    ensure!(g["kind"] == kind.name() && g["route"] == route.name() && g["sinks"] == "per_row_head",
        "Inkling attention export bias/sink convention differs");
    let expected = if kind == Kind::Global && route == Route::Decode {
        route.capacity() * 32 * 48 * (128 * 4 + 4)
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
        ProgramSpec { name: format!("inkling2_{}_attention_{}_m{}", kind.name(), route.name(), route.capacity()),
            pointers: pointers.iter().map(|s| s.to_string()).collect(),
            scalar_kinds: match (kind, route) { (Kind::Local, _) => b"i".to_vec(),
                (Kind::Global, Route::Prefill) => b"ii".to_vec(), _ => b"iii".to_vec() },
            scratch: [("scratch".into(), expected as u64)].into(), capacity_rows: route.capacity() as u32,
            geometry: serde_json::json!({"kind": kind.name(), "route": route.name(), "heads": 48,
                "kv_heads": 8, "head_dim": 128, "record_elems": 2048, "record_bytes": RECORD_BYTES,
                "max_rows": route.capacity(), "window": if kind == Kind::Local { 1024 } else { 0 },
                "extent": 1024, "ring_rows": RING_ROWS, "page_rows": PAGE_ROWS, "sinks": "per_row_head",
                "max_splits": if kind == Kind::Global && route == Route::Decode { 32 } else { 1 },
                "rows_per_cta": if route == Route::Decode { 1 } else { 10 } }) }
    }
    #[test]
    fn manifest_accepts_only_checkpoint_bias_sink_geometry() {
        for kind in [Kind::Local, Kind::Global] {
            for route in [Route::Decode, Route::Prefill] {
                let good = spec(kind, route);
                validate_manifest(&good, kind, route).unwrap();
                for key in ["heads", "kv_heads", "head_dim", "ring_rows", "page_rows", "extent", "window", "max_splits"] {
                    let mut wrong = good.clone();
                    wrong.geometry[key] = serde_json::json!(999);
                    assert!(validate_manifest(&wrong, kind, route).is_err(), "{key}");
                }
                let mut wrong = good.clone();
                wrong.geometry["sinks"] = serde_json::json!("per_head");
                assert!(validate_manifest(&wrong, kind, route).is_err());
                let mut wrong = good.clone();
                wrong.scratch.insert("scratch".into(), 0);
                assert!(validate_manifest(&wrong, kind, route).is_err());
                let mut wrong = good.clone();
                wrong.pointers.swap(0, 1);
                assert!(validate_manifest(&wrong, kind, route).is_err());
                let mut wrong = good.clone();
                wrong.scalar_kinds[0] = b'l';
                assert!(validate_manifest(&wrong, kind, route).is_err());
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
