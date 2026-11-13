//! Byte-preserving, physically bounded paged index-cache staging.
use super::{programs::{Program, ProgramSpec, Programs, Scalar}, projected_buffers::validate_buffers};
use crate::CuteafdDeviceBuffer as Buffer;
use std::ffi::c_void;

#[derive(Debug, thiserror::Error)]
pub enum GatherError {
    #[error("invalid index cache gather: {0}")]
    Invalid(&'static str),
    #[error("index cache gather program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, GatherError>;
const POINTERS: &[&str] = &["cache", "pages", "lengths", "keys", "scales", "starts", "ends"];
const SCALARS: &[(&str, &str)] = &[("rows", "int32"), ("table_width", "int32"),
    ("page_begin", "int32"), ("pool_pages", "int64"), ("live_tokens", "int32"),
    ("output_tokens", "int32")];
const ALIGN: [usize; 7] = [16, 4, 4, 16, 4, 4, 4];
pub const PAGE_BYTES: usize = 8448;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout { pub rows: usize }
impl Layout {
    pub fn from_spec(spec: &ProgramSpec) -> Result<Self> {
        let g = &spec.geometry;
        if spec.capacity_rows == 0 || spec.capacity_rows > 4096
            || spec.scalar_kinds != b"iiilii" || !spec.scratch.is_empty()
            || spec.pointers.iter().map(String::as_str).ne(POINTERS.iter().copied())
            || g["page_rows"] != 64 || g["head_dim"] != 128
            || g["page_bytes"] != PAGE_BYTES || g["padding_multiple"] != 256 {
            return Err(GatherError::Invalid("geometry, capacity or ABI"));
        }
        Ok(Self { rows: spec.capacity_rows as usize })
    }
    fn buffers(self, b: Batch) -> Result<[Buffer; 7]> {
        // The kernel uses i32 grid/column/local-bound arithmetic. Device lengths
        // are nonnegative; bound the host scalars before computing their extents.
        let max = i32::MAX as usize;
        if b.rows == 0 || b.rows > self.rows || b.table_width > max
            || b.page_begin > max / 64 || b.output_tokens == 0
            || b.output_tokens > max - 255 || b.output_tokens % 256 != 0
            || b.live_tokens > b.output_tokens || b.pool_pages > max + 1 {
            return Err(GatherError::Invalid("live scalar capacities"));
        }
        let bytes = |count: usize, stride: usize| count.checked_mul(stride)
            .ok_or(GatherError::Invalid("byte extent overflow"));
        let items = [(b.cache, bytes(b.pool_pages, PAGE_BYTES)?), (b.pages, bytes(b.table_width, 4)?),
            (b.lengths, bytes(b.rows, 4)?), (b.keys, bytes(b.output_tokens, 128)?),
            (b.scales, bytes(b.output_tokens, 4)?), (b.starts, bytes(b.rows, 4)?),
            (b.ends, bytes(b.rows, 4)?)];
        validate_buffers(&items, &ALIGN, &[3, 4, 5, 6]).map_err(GatherError::Invalid)?;
        Ok(items.map(|(buffer, _)| buffer))
    }
}

pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<Layout> {
    let layout = Layout::from_spec(spec)?;
    let operands = [("uint8", "[pool_pages,8448]", "in"), ("int32", "[table_width]", "in"),
        ("int32", "[rows]", "in"), ("uint8", "[output_tokens,128]", "out"),
        ("float32", "[output_tokens]", "out"), ("int32", "[rows]", "out"),
        ("int32", "[rows]", "out")];
    let pointers = entry["pointers"].as_array().ok_or(GatherError::Invalid("pointer metadata"))?;
    let scalars = entry["scalars"].as_array().ok_or(GatherError::Invalid("scalar metadata"))?;
    if entry["op"] != "index_cache_gather" || pointers.len() != POINTERS.len()
        || pointers.iter().zip(POINTERS.iter().zip(operands)).any(|(p, (name, (dtype, shape, role)))|
            p["name"] != *name || p["dtype"] != dtype || p["shape"] != shape || p["role"] != role)
        || scalars.len() != SCALARS.len()
        || scalars.iter().zip(SCALARS).any(|(s, (name, kind))| s["name"] != *name || s["type"] != *kind) {
        return Err(GatherError::Invalid("exported operand or operation contract"));
    }
    Ok(layout)
}

#[derive(Clone, Copy)]
pub struct Batch {
    pub rows: usize,
    pub table_width: usize,
    pub page_begin: usize,
    pub pool_pages: usize,
    pub live_tokens: usize,
    pub output_tokens: usize,
    pub cache: Buffer,
    pub pages: Buffer,
    pub lengths: Buffer,
    pub keys: Buffer,
    pub scales: Buffer,
    pub starts: Buffer,
    pub ends: Buffer,
}

pub struct Gather<'a> { program: Program<'a>, pub layout: Layout }
impl<'a> Gather<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let layout = Layout::from_spec(programs.spec(name)?)?;
        Ok(Self { program: programs.program(name, POINTERS)?, layout })
    }
    /// # Safety
    /// Device lengths must be nonnegative global causal extents. Keep all
    /// storage and the stream live through queued work and graph replay. Page
    /// IDs may be invalid: the kernel checks both physical bounds and clears
    /// their output bytes, as it does for padding and out-of-table columns.
    pub unsafe fn launch(&self, b: Batch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.buffers(b)?;
        let scalars = [Scalar::I32(b.rows as i32), Scalar::I32(b.table_width as i32),
            Scalar::I32(b.page_begin as i32), Scalar::I64(b.pool_pages as i64),
            Scalar::I32(b.live_tokens as i32), Scalar::I32(b.output_tokens as i32)];
        // SAFETY: checked scalar/buffer contracts; caller retains device storage and stream.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &scalars, stream) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> ProgramSpec {
        ProgramSpec { name: "minimax_index_cache_gather".into(),
            pointers: POINTERS.iter().map(|s| (*s).into()).collect(), scalar_kinds: b"iiilii".to_vec(),
            scratch: Default::default(), capacity_rows: 4096,
            geometry: serde_json::json!({"page_rows":64,"head_dim":128,"page_bytes":8448,"padding_multiple":256}) }
    }
    fn batch() -> Batch {
        let buffer = |i: usize| Buffer { ptr: ((i + 1) * 0x100000000usize) as *mut c_void,
            bytes: 1 << 30, device_id: 0, flags: 0 };
        Batch { rows: 3, table_width: 8, page_begin: 1, pool_pages: 11,
            live_tokens: 321, output_tokens: 512, cache: buffer(0), pages: buffer(1), lengths: buffer(2),
            keys: buffer(3), scales: buffer(4), starts: buffer(5), ends: buffer(6) }
    }
    #[test]
    fn validates_geometry_abi_and_physical_output_owners() {
        let expected = spec();
        let layout = Layout::from_spec(&expected).unwrap();
        let mut wrong = expected.clone(); wrong.scalar_kinds = b"iiiiii".to_vec();
        assert!(Layout::from_spec(&wrong).is_err());
        let mut wrong = expected.clone(); wrong.geometry["page_bytes"] = 8192.into();
        assert!(Layout::from_spec(&wrong).is_err());
        let mut wrong = expected; wrong.pointers.swap(3, 4);
        assert!(Layout::from_spec(&wrong).is_err());
        let b = batch(); assert!(layout.buffers(b).is_ok());
        let mut bad = b; bad.keys.bytes = 512 * 128 - 1; assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.cache.bytes = 11 * PAGE_BYTES - 1; assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.scales = b.keys; assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.pages.device_id = 1; assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.starts.ptr = 3usize as *mut c_void; assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.ends.flags = 1; assert!(layout.buffers(bad).is_err());
    }
    #[test]
    fn checks_padding_live_extents_and_signed_scalar_arithmetic() {
        let layout = Layout::from_spec(&spec()).unwrap();
        let b = batch();
        for tokens in [0, 255, 513, i32::MAX as usize] {
            let mut bad = b; bad.output_tokens = tokens; assert!(layout.buffers(bad).is_err());
        }
        let mut bad = b; bad.live_tokens = 513; assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.rows = 4097; assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.page_begin = i32::MAX as usize / 64 + 1;
        assert!(layout.buffers(bad).is_err());
        let mut bad = b; bad.pool_pages = usize::MAX; assert!(layout.buffers(bad).is_err());
        let mut empty = b; empty.table_width = 0; empty.pool_pages = 0; empty.live_tokens = 0;
        assert!(layout.buffers(empty).is_ok());
    }
    #[test]
    fn rejects_manifest_operand_and_scalar_changes() {
        let spec = spec();
        let operands = [("uint8", "[pool_pages,8448]", "in"), ("int32", "[table_width]", "in"),
            ("int32", "[rows]", "in"), ("uint8", "[output_tokens,128]", "out"),
            ("float32", "[output_tokens]", "out"), ("int32", "[rows]", "out"), ("int32", "[rows]", "out")];
        let entry = serde_json::json!({"op":"index_cache_gather",
            "pointers":POINTERS.iter().zip(operands).map(|(n,(d,s,r))|
                serde_json::json!({"name":n,"dtype":d,"shape":s,"role":r})).collect::<Vec<_>>(),
            "scalars":SCALARS.iter().map(|(n,t)|serde_json::json!({"name":n,"type":t})).collect::<Vec<_>>()});
        assert!(validate_entry(&spec, &entry).is_ok());
        let fixture = crate::native_library_lifetime_fixture::Fixture::build_for_tests().unwrap();
        let library = fixture.load().unwrap();
        let mut installed = entry.clone();
        installed["name"] = spec.name.clone().into();
        installed["geometry"] = spec.geometry.clone();
        installed["capacity_rows"] = spec.capacity_rows.into();
        installed["scratch_bytes_at_capacity"] = serde_json::json!({});
        fixture.register_programs(&library, &[installed.clone()]).unwrap();
        let path = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"programs":[installed]})).unwrap()).unwrap();
        let programs = library.programs().unwrap().with_manifest(&path).unwrap();
        assert_eq!(Gather::bind(&programs, &spec.name).unwrap().layout, Layout { rows: 4096 });
        for (field, value) in [("dtype", "float32"), ("shape", "[rows,128]"), ("role", "out"), ("name", "wrong")] {
            let mut wrong = entry.clone(); wrong["pointers"][0][field] = value.into();
            assert!(validate_entry(&spec, &wrong).is_err());
        }
        let mut wrong = entry; wrong["scalars"][3]["type"] = "int32".into();
        assert!(validate_entry(&spec, &wrong).is_err());
    }
}
