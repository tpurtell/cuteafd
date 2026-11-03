//! Flash index post-stages over native projected rows. Checkpoint weight
//! formats belong to the caller; normalization, pooling and replay stay unchanged.
use super::programs::{Program, ProgramSpec, Programs, Scalar};
use super::projected_buffers::validate_buffers;
use crate::CuteafdDeviceBuffer as Buffer;
use std::ffi::c_void;

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("invalid projected index: {0}")]
    Invalid(&'static str),
    #[error("projected index program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, IndexError>;
const POINTERS: &[&str] = &[
    "iq",
    "kw",
    "pool_slots",
    "positions",
    "kda_slots",
    "seq_first",
    "k_norm_w",
    "k_norm_b",
    "ape",
    "tails",
    "replay",
    "index_cache",
    "q_fp8",
    "head_weights",
    "scratch",
];
const ALIGNMENTS: [usize; 15] = [16, 16, 8, 8, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16];
pub const HEADS: usize = 32;
pub const KEY_ROW_BYTES: usize = 512;
pub const TAIL_BYTES: usize = 1552;
pub const PAGE_BYTES: usize = 8448;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub capacity: usize,
    pub replay_rows: usize,
    pub scratch: usize,
}
impl Layout {
    pub fn query_bytes(self, rows: usize) -> Option<usize> {
        rows.checked_mul(HEADS * 128 * 2)
    }
    pub fn packed_bytes(self, rows: usize) -> Option<usize> {
        rows.checked_mul((256 + HEADS) * 2)
    }
    pub fn tail_bytes(self, slots: usize) -> Option<usize> {
        slots.checked_mul(TAIL_BYTES)
    }
    pub fn cache_bytes(self, pages: usize) -> Option<usize> {
        pages.checked_mul(PAGE_BYTES)
    }
    pub fn replay_bytes(self) -> usize {
        self.replay_rows * KEY_ROW_BYTES
    }
    fn pointer_contract(self) -> Vec<(&'static str, String, &'static str)> {
        vec![
            ("bfloat16", "[rows,32,128]".into(), "in"),
            ("bfloat16", "[rows,288]".into(), "in"),
            ("int64", "[rows]".into(), "in"),
            ("int64", "[rows]".into(), "in"),
            ("int32", "[rows]".into(), "in"),
            ("int32", "[rows]".into(), "in"),
            ("bfloat16", "[128]".into(), "in"),
            ("bfloat16", "[128]".into(), "in"),
            ("bfloat16", "[4,128]".into(), "in"),
            ("uint8", "[slots,1552]".into(), "inout"),
            ("bfloat16", format!("[{},256]", self.replay_rows), "out"),
            ("uint8", "[pool_pages,8448]".into(), "inout"),
            ("float8_e4m3fn", "[rows,32,128]".into(), "out"),
            ("float32", "[rows,32]".into(), "out"),
            ("uint8", "[index_keys_scratch]".into(), "scratch"),
        ]
    }
}
fn layout(spec: &ProgramSpec) -> Result<Layout> {
    let suffix = spec
        .name
        .strip_prefix("glmf_stage_index_projected_c_m")
        .or_else(|| spec.name.strip_prefix("glmf2_stage_index_projected_c_m"))
        .ok_or(IndexError::Invalid("unknown projected table"))?;
    let capacity: usize = suffix
        .parse()
        .map_err(|_| IndexError::Invalid("capacity"))?;
    if !matches!(capacity, 64 | 128 | 4096) || spec.capacity_rows as usize != capacity {
        return Err(IndexError::Invalid("capacity disagrees with table"));
    }
    let replay_rows = if capacity == 128 { 128 } else { 64 };
    let g = &spec.geometry;
    if [
        ("hidden", 4096),
        ("index_heads", HEADS),
        ("index_head_dim", 128),
        ("index_kpool", 4),
        ("page_rows", 64),
        ("max_rows", capacity),
        ("tail_bytes", TAIL_BYTES),
        ("replay_rows", replay_rows),
    ]
    .iter()
    .any(|&(k, v)| g[k].as_u64() != Some(v as u64))
        || g["index_norm_eps"].as_f64() != Some(1e-6)
        || g["projected_inputs"].as_bool() != Some(true)
        || spec.scalar_kinds != b"ii"
        || spec
            .pointers
            .iter()
            .map(String::as_str)
            .ne(POINTERS.iter().copied())
    {
        return Err(IndexError::Invalid("geometry, arithmetic or ABI"));
    }
    let scratch = (capacity * KEY_ROW_BYTES).div_ceil(1024) * 1024;
    if spec.scratch.len() != 1 || spec.scratch.get("scratch") != Some(&(scratch as u64)) {
        return Err(IndexError::Invalid("scratch layout"));
    }
    Ok(Layout {
        capacity,
        replay_rows,
        scratch,
    })
}

/// Validate the complete exported ABI before kernel binding or allocation.
pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<Layout> {
    let l = layout(spec)?;
    let scalars = entry["scalars"]
        .as_array()
        .ok_or(IndexError::Invalid("scalar metadata"))?;
    if entry["op"] != "index_producer"
        || scalars.len() != 2
        || scalars
            .iter()
            .zip(["rows", "spec"])
            .any(|(s, n)| s["name"] != n || s["type"] != "int32")
    {
        return Err(IndexError::Invalid("scalar ABI or operation"));
    }
    let pointers = entry["pointers"]
        .as_array()
        .ok_or(IndexError::Invalid("pointer metadata"))?;
    if pointers.len() != POINTERS.len()
        || pointers.iter().zip(POINTERS).zip(l.pointer_contract()).any(
            |((p, n), (dtype, shape, role))| {
                p["name"] != *n || p["dtype"] != dtype || p["shape"] != shape || p["role"] != role
            },
        )
    {
        return Err(IndexError::Invalid("pointer dtype, shape or access"));
    }
    Ok(l)
}

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub norm: Buffer,
    pub bias: Buffer,
    pub ape: Buffer,
}
#[derive(Debug, Clone, Copy)]
pub struct Batch {
    pub rows: usize,
    pub state_slots: usize,
    pub pool_pages: usize,
    pub speculative: bool,
    pub query: Buffer,
    pub packed: Buffer,
    pub pool_slots: Buffer,
    pub positions: Buffer,
    pub kda_slots: Buffer,
    pub first: Buffer,
    pub tails: Buffer,
    pub replay: Buffer,
    pub cache: Buffer,
    pub query_fp8: Buffer,
    pub head_weights: Buffer,
    pub scratch: Buffer,
}
pub struct ProjectedIndex<'a> {
    program: Program<'a>,
    pub layout: Layout,
}
impl<'a> ProjectedIndex<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let layout = layout(programs.spec(name)?)?;
        Ok(Self {
            program: programs.program(name, POINTERS)?,
            layout,
        })
    }
    /// # Safety
    /// Slots and positions must address admitted cache pages and tails. First-row
    /// offsets describe contiguous ordered sequences; retain storage and stream
    /// until queued work completes. Speculative tails are committed separately.
    pub unsafe fn launch(&self, w: Weights, b: Batch, stream: *mut c_void) -> Result<()> {
        let items = self.layout.operands(w, b)?;
        // SAFETY: extents/devices/disjoint writers checked below; metadata values
        // and graph/stream lifetimes belong to the caller's contract.
        unsafe {
            self.program.launch(
                &items.map(|(b, _)| b.ptr),
                &[
                    Scalar::I32(b.rows as i32),
                    Scalar::I32(i32::from(b.speculative)),
                ],
                stream,
            )?;
        }
        Ok(())
    }
}
impl Layout {
    fn operands(self, w: Weights, b: Batch) -> Result<[(Buffer, usize); 15]> {
        if b.rows == 0
            || b.rows > self.capacity
            || b.state_slots == 0
            || b.pool_pages == 0
            || b.state_slots > i32::MAX as usize
            || (b.speculative && b.rows > self.replay_rows)
        {
            return Err(IndexError::Invalid("batch/replay geometry"));
        }
        let items = [
            (
                b.query,
                self.query_bytes(b.rows)
                    .ok_or(IndexError::Invalid("query overflow"))?,
            ),
            (
                b.packed,
                self.packed_bytes(b.rows)
                    .ok_or(IndexError::Invalid("packed overflow"))?,
            ),
            (b.pool_slots, b.rows * 8),
            (b.positions, b.rows * 8),
            (b.kda_slots, b.rows * 4),
            (b.first, b.rows * 4),
            (w.norm, 256),
            (w.bias, 256),
            (w.ape, 1024),
            (
                b.tails,
                self.tail_bytes(b.state_slots)
                    .ok_or(IndexError::Invalid("tail overflow"))?,
            ),
            (b.replay, self.replay_bytes()),
            (
                b.cache,
                self.cache_bytes(b.pool_pages)
                    .ok_or(IndexError::Invalid("cache overflow"))?,
            ),
            (b.query_fp8, b.rows * HEADS * 128),
            (b.head_weights, b.rows * HEADS * 4),
            (b.scratch, self.scratch),
        ];
        validate_buffers(&items, &ALIGNMENTS, &[9, 10, 11, 12, 13, 14])
            .map_err(IndexError::Invalid)?;
        Ok(items)
    }
}

const KEY_POINTERS: &[&str] = &[
    "iq",
    "kw",
    "slots",
    "pool_slots",
    "k_norm_w",
    "k_norm_b",
    "ape",
    "token_keys",
    "index_cache",
    "q_fp8",
    "head_weights",
];
const KEY_ALIGNMENTS: [usize; 11] = [16, 16, 8, 8, 16, 16, 16, 16, 16, 16, 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeysLayout {
    pub capacity: usize,
}
impl KeysLayout {
    fn pointer_contract(self) -> [(&'static str, &'static str, &'static str); 11] {
        [
            ("bfloat16", "[rows,32,128]", "in"),
            ("bfloat16", "[rows,288]", "in"),
            ("int64", "[rows]", "in"),
            ("int64", "[rows]", "in"),
            ("bfloat16", "[128]", "in"),
            ("bfloat16", "[128]", "in"),
            ("bfloat16", "[4,128]", "in"),
            ("bfloat16", "[record_slots,256]", "inout"),
            ("uint8", "[pool_pages,8448]", "inout"),
            ("float8_e4m3fn", "[rows,32,128]", "out"),
            ("float32", "[rows,32]", "out"),
        ]
    }
    pub fn token_bytes(self, slots: usize) -> Option<usize> {
        slots.checked_mul(KEY_ROW_BYTES)
    }
    fn from_spec(spec: &ProgramSpec) -> Result<Self> {
        let suffix = spec
            .name
            .strip_prefix("glmf_stage_index_projected_keys_m")
            .or_else(|| spec.name.strip_prefix("glmf2_stage_index_projected_keys_m"))
            .ok_or(IndexError::Invalid("unknown token-key table"))?;
        let capacity = suffix
            .parse()
            .map_err(|_| IndexError::Invalid("capacity"))?;
        let g = &spec.geometry;
        if !matches!(capacity, 64 | 128 | 4096)
            || spec.capacity_rows as usize != capacity
            || [
                ("hidden", 4096),
                ("index_heads", HEADS),
                ("index_head_dim", 128),
                ("index_kpool", 4),
                ("page_rows", 64),
                ("max_rows", capacity),
                ("token_key_row_bytes", KEY_ROW_BYTES),
            ]
            .iter()
            .any(|&(key, value)| g[key].as_u64() != Some(value as u64))
            || g["index_norm_eps"].as_f64() != Some(1e-6)
            || g["projected_inputs"].as_bool() != Some(true)
            || spec.scalar_kinds != b"i"
            || !spec.scratch.is_empty()
            || spec
                .pointers
                .iter()
                .map(String::as_str)
                .ne(KEY_POINTERS.iter().copied())
        {
            return Err(IndexError::Invalid("token-key geometry, arithmetic or ABI"));
        }
        Ok(Self { capacity })
    }
    fn operands(self, w: Weights, b: KeysBatch) -> Result<[(Buffer, usize); 11]> {
        if b.rows == 0 || b.rows > self.capacity || b.record_slots == 0 || b.pool_pages == 0 {
            return Err(IndexError::Invalid("token-key batch geometry"));
        }
        let items = [
            (b.query, b.rows * HEADS * 128 * 2),
            (b.packed, b.rows * (256 + HEADS) * 2),
            (b.slots, b.rows * 8),
            (b.pool_slots, b.rows * 8),
            (w.norm, 256),
            (w.bias, 256),
            (w.ape, 1024),
            (
                b.token_keys,
                self.token_bytes(b.record_slots)
                    .ok_or(IndexError::Invalid("token-key overflow"))?,
            ),
            (
                b.cache,
                b.pool_pages
                    .checked_mul(PAGE_BYTES)
                    .ok_or(IndexError::Invalid("cache overflow"))?,
            ),
            (b.query_fp8, b.rows * HEADS * 128),
            (b.head_weights, b.rows * HEADS * 4),
        ];
        validate_buffers(&items, &KEY_ALIGNMENTS, &[7, 8, 9, 10]).map_err(IndexError::Invalid)?;
        Ok(items)
    }
}

pub fn validate_keys_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<KeysLayout> {
    let layout = KeysLayout::from_spec(spec)?;
    let scalars = entry["scalars"]
        .as_array()
        .ok_or(IndexError::Invalid("scalar metadata"))?;
    let pointers = entry["pointers"]
        .as_array()
        .ok_or(IndexError::Invalid("pointer metadata"))?;
    if entry["op"] != "index_producer"
        || scalars.len() != 1
        || scalars[0]["name"] != "rows"
        || scalars[0]["type"] != "int32"
        || pointers.len() != KEY_POINTERS.len()
        || pointers
            .iter()
            .zip(KEY_POINTERS)
            .zip(layout.pointer_contract())
            .any(|((p, name), (dtype, shape, role))| {
                p["name"] != *name
                    || p["dtype"] != dtype
                    || p["shape"] != shape
                    || p["role"] != role
            })
    {
        return Err(IndexError::Invalid("token-key exported ABI"));
    }
    Ok(layout)
}

#[derive(Debug, Clone, Copy)]
pub struct KeysBatch {
    pub rows: usize,
    pub record_slots: usize,
    pub pool_pages: usize,
    pub query: Buffer,
    pub packed: Buffer,
    pub slots: Buffer,
    pub pool_slots: Buffer,
    pub token_keys: Buffer,
    pub cache: Buffer,
    pub query_fp8: Buffer,
    pub head_weights: Buffer,
}
pub struct ProjectedKeys<'a> {
    program: Program<'a>,
    pub layout: KeysLayout,
}
impl<'a> ProjectedKeys<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let layout = KeysLayout::from_spec(programs.spec(name)?)?;
        Ok(Self {
            program: programs.program(name, KEY_POINTERS)?,
            layout,
        })
    }
    /// # Safety
    /// Nonnegative slots must address admitted token records. A completed pool
    /// reads four contiguous records ending at slots[row], all initialized before
    /// this launch. Pool slots address admitted pages; negative slots skip writes.
    /// Retain all buffers and the stream until queued work and graph use complete.
    pub unsafe fn launch(&self, w: Weights, b: KeysBatch, stream: *mut c_void) -> Result<()> {
        let items = self.layout.operands(w, b)?;
        // SAFETY: extents, device and writable aliases are checked; device slot
        // values and graph/stream lifetimes are the caller's contract.
        unsafe {
            self.program.launch(
                &items.map(|(buffer, _)| buffer.ptr),
                &[Scalar::I32(b.rows as i32)],
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn operand_data_contract_matches_runtime_tables() {
        let (spec, entry) = fixture("glmf_stage", 128);
        let layout = validate_entry(&spec, &entry).unwrap();
        super::super::projected_buffers::assert_operand_contract(
            "projected_index_compact",
            POINTERS,
            layout.pointer_contract(),
            &ALIGNMENTS,
        );
        super::super::projected_buffers::assert_operand_contract(
            "projected_index_keys",
            KEY_POINTERS,
            KeysLayout { capacity: 128 }.pointer_contract(),
            &KEY_ALIGNMENTS,
        );
    }
    fn fixture(prefix: &str, capacity: usize) -> (ProgramSpec, serde_json::Value) {
        let l = Layout {
            capacity,
            replay_rows: if capacity == 128 { 128 } else { 64 },
            scratch: (capacity * KEY_ROW_BYTES).div_ceil(1024) * 1024,
        };
        let spec = ProgramSpec {
            name: format!("{prefix}_index_projected_c_m{capacity}"),
            capacity_rows: capacity as u32,
            pointers: POINTERS.iter().map(|s| s.to_string()).collect(),
            scalar_kinds: b"ii".to_vec(),
            scratch: [("scratch".into(), l.scratch as u64)].into(),
            geometry: json!({"hidden":4096,"index_heads":32,"index_head_dim":128,"index_kpool":4,
                "page_rows":64,"max_rows":capacity,"tail_bytes":1552,"replay_rows":l.replay_rows,
                "index_norm_eps":1e-6,"projected_inputs":true}),
        };
        let pointers: Vec<_> = POINTERS.iter().zip(l.pointer_contract()).map(|(name, (dtype, shape, role))|
            json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect();
        (
            spec,
            json!({"op":"index_producer","pointers":pointers,"scalars":[
            {"name":"rows","type":"int32"},{"name":"spec","type":"int32"}]}),
        )
    }
    fn keys_fixture(prefix: &str, capacity: usize) -> (ProgramSpec, serde_json::Value) {
        let layout = KeysLayout { capacity };
        let spec = ProgramSpec {
            name: format!("{prefix}_index_projected_keys_m{capacity}"),
            capacity_rows: capacity as u32,
            pointers: KEY_POINTERS.iter().map(|name| name.to_string()).collect(),
            scalar_kinds: b"i".to_vec(),
            scratch: Default::default(),
            geometry: json!({"hidden":4096,"index_heads":32,"index_head_dim":128,"index_kpool":4,
                "page_rows":64,"max_rows":capacity,"token_key_row_bytes":512,
                "index_norm_eps":1e-6,"projected_inputs":true}),
        };
        let pointers: Vec<_> = KEY_POINTERS.iter().zip(layout.pointer_contract())
            .map(|(name, (dtype, shape, role))| json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect();
        (
            spec,
            json!({"op":"index_producer","pointers":pointers,"scalars":[{"name":"rows","type":"int32"}]}),
        )
    }
    #[test]
    fn token_keys_validate_replicated_geometry_and_reject_abi_drift() {
        for prefix in ["glmf_stage", "glmf2_stage"] {
            for capacity in [64, 128, 4096] {
                let (spec, entry) = keys_fixture(prefix, capacity);
                assert_eq!(
                    validate_keys_entry(&spec, &entry).unwrap().capacity,
                    capacity
                );
                for (field, value) in [
                    ("index_heads", json!(16)),
                    ("index_kpool", json!(8)),
                    ("token_key_row_bytes", json!(256)),
                    ("index_norm_eps", json!(1e-5)),
                    ("projected_inputs", json!(false)),
                ] {
                    let mut bad = spec.clone();
                    bad.geometry[field] = value;
                    assert!(validate_keys_entry(&bad, &entry).is_err(), "{field}");
                }
                let mut bad = spec.clone();
                bad.pointers.swap(2, 3);
                assert!(validate_keys_entry(&bad, &entry).is_err());
                let mut bad = spec.clone();
                bad.scratch.insert("scratch".into(), 1024);
                assert!(validate_keys_entry(&bad, &entry).is_err());
                let mut bad = spec.clone();
                bad.scalar_kinds.push(b'i');
                assert!(validate_keys_entry(&bad, &entry).is_err());
                for (field, value) in [
                    ("dtype", "float32"),
                    ("shape", "[record_slots,128]"),
                    ("role", "in"),
                ] {
                    let mut bad = entry.clone();
                    bad["pointers"][7][field] = json!(value);
                    assert!(validate_keys_entry(&spec, &bad).is_err(), "{field}");
                }
                let mut bad = entry.clone();
                bad["scalars"][0]["name"] = json!("spec");
                assert!(validate_keys_entry(&spec, &bad).is_err());
            }
        }
    }
    #[test]
    fn token_keys_check_extents_aliases_and_64bit_offsets() {
        let (weights, compact) = buffers();
        let mut batch = KeysBatch {
            rows: 17,
            record_slots: 128,
            pool_pages: 4,
            query: compact.query,
            packed: compact.packed,
            slots: compact.positions,
            pool_slots: compact.pool_slots,
            token_keys: compact.tails,
            cache: compact.cache,
            query_fp8: compact.query_fp8,
            head_weights: compact.head_weights,
        };
        let layout = KeysLayout { capacity: 128 };
        assert!(layout.operands(weights, batch).is_ok());
        let mut subrow = batch;
        subrow.slots.ptr = subrow.slots.ptr.wrapping_byte_add(8);
        subrow.pool_slots.ptr = subrow.pool_slots.ptr.wrapping_byte_add(8);
        assert!(layout.operands(weights, subrow).is_ok());
        let mut bad = batch;
        bad.token_keys.bytes = 1;
        assert!(layout.operands(weights, bad).is_err());
        let mut bad = batch;
        bad.record_slots = usize::MAX;
        assert!(layout.operands(weights, bad).is_err());
        let mut bad = batch;
        bad.pool_pages = usize::MAX;
        assert!(layout.operands(weights, bad).is_err());
        let mut bad = batch;
        bad.rows = 129;
        assert!(layout.operands(weights, bad).is_err());
        let mut bad = batch;
        bad.token_keys.ptr = batch.query.ptr;
        assert!(layout.operands(weights, bad).is_err());
        let mut bad = batch;
        bad.slots.ptr = (batch.slots.ptr as usize + 4) as *mut c_void;
        assert!(layout.operands(weights, bad).is_err());
        let mut bad = batch;
        bad.cache.device_id = 1;
        assert!(layout.operands(weights, bad).is_err());
        let mut bad = batch;
        bad.token_keys.flags = 1;
        assert!(layout.operands(weights, bad).is_err());
        batch.record_slots = (i32::MAX as usize / KEY_ROW_BYTES) + 1;
        batch.token_keys.ptr = 0x10_0000_0000usize as *mut c_void;
        batch.token_keys.bytes = layout.token_bytes(batch.record_slots).unwrap();
        assert!(batch.token_keys.bytes > i32::MAX as usize);
        assert!(layout.operands(weights, batch).is_ok());
        batch.token_keys.ptr = (usize::MAX - 15) as *mut c_void;
        assert!(layout.operands(weights, batch).is_err());
    }
    #[test]
    fn tp_capacities_keep_replicated_index_geometry() {
        for prefix in ["glmf_stage", "glmf2_stage"] {
            for capacity in [64, 128, 4096] {
                let (s, e) = fixture(prefix, capacity);
                let l = validate_entry(&s, &e).unwrap();
                assert_eq!(l.query_bytes(3), Some(3 * 8192));
                assert_eq!(l.packed_bytes(3), Some(3 * 576));
                assert_eq!(
                    l.replay_bytes(),
                    if capacity == 128 { 65536 } else { 32768 }
                );
                assert_eq!(l.scratch, capacity * 512);
            }
        }
    }
    #[test]
    fn rejects_metadata_drift() {
        let (s, e) = fixture("glmf2_stage", 128);
        for (field, value) in [
            ("index_heads", json!(16)),
            ("index_norm_eps", json!(1e-5)),
            ("index_kpool", json!(8)),
            ("tail_bytes", json!(1544)),
            ("replay_rows", json!(64)),
            ("projected_inputs", json!(false)),
        ] {
            let mut bad = s.clone();
            bad.geometry[field] = value;
            assert!(validate_entry(&bad, &e).is_err(), "{field}");
        }
        let mut bad = s.clone();
        bad.pointers.swap(2, 3);
        assert!(validate_entry(&bad, &e).is_err());
        let mut bad = s.clone();
        bad.scratch.insert("scratch".into(), 1);
        assert!(validate_entry(&bad, &e).is_err());
        let mut bad = e.clone();
        bad["op"] = json!("attention_stage");
        assert!(validate_entry(&s, &bad).is_err());
        for (field, value) in [
            ("dtype", "float32"),
            ("shape", "[rows,16,128]"),
            ("role", "in"),
        ] {
            let mut bad = e.clone();
            bad["pointers"][12][field] = json!(value);
            assert!(validate_entry(&s, &bad).is_err(), "{field}");
        }
    }
    #[test]
    #[ignore = "requires CUTEAFD_FLASH_STAGE_ABIS with actual TP1/TP2 exporter listings"]
    fn exported_compact_index_stages_validate() {
        let paths =
            std::env::var_os("CUTEAFD_FLASH_STAGE_ABIS").expect("actual stage ABI paths required");
        let mut checked = std::collections::BTreeSet::new();
        for path in std::env::split_paths(&paths) {
            let listing: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            for entry in listing["programs"].as_array().unwrap() {
                let name = entry["name"].as_str().unwrap();
                if !name.contains("_index_projected_c_m")
                    && !name.contains("_index_projected_keys_m")
                {
                    continue;
                }
                let spec = ProgramSpec {
                    name: name.into(),
                    capacity_rows: entry["capacity_rows"].as_u64().unwrap().try_into().unwrap(),
                    geometry: entry["geometry"].clone(),
                    pointers: entry["pointers"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| p["name"].as_str().unwrap().into())
                        .collect(),
                    scalar_kinds: entry["scalars"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|s| {
                            assert_eq!(s["type"], "int32");
                            b'i'
                        })
                        .collect(),
                    scratch: entry["scratch_bytes_at_capacity"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.as_u64().unwrap()))
                        .collect(),
                };
                if name.contains("_index_projected_keys_m") {
                    validate_keys_entry(&spec, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
                } else {
                    validate_entry(&spec, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
                }
                assert!(checked.insert(name.to_owned()), "duplicate stage {name}");
            }
        }
        let expected = ["glmf_stage", "glmf2_stage"]
            .into_iter()
            .flat_map(|prefix| {
                ["c", "keys"].into_iter().flat_map(move |kind| {
                    [64, 128, 4096]
                        .into_iter()
                        .map(move |rows| format!("{prefix}_index_projected_{kind}_m{rows}"))
                })
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(checked, expected);
    }
    fn buffers() -> (Weights, Batch) {
        let mut id = 0usize;
        let mut buffer = || {
            id += 1;
            Buffer {
                ptr: (id * 0x1000_0000) as *mut c_void,
                bytes: 0x1000_0000,
                device_id: 0,
                flags: 0,
            }
        };
        (
            Weights {
                norm: buffer(),
                bias: buffer(),
                ape: buffer(),
            },
            Batch {
                rows: 17,
                state_slots: 2,
                pool_pages: 4,
                speculative: false,
                query: buffer(),
                packed: buffer(),
                pool_slots: buffer(),
                positions: buffer(),
                kda_slots: buffer(),
                first: buffer(),
                tails: buffer(),
                replay: buffer(),
                cache: buffer(),
                query_fp8: buffer(),
                head_weights: buffer(),
                scratch: buffer(),
            },
        )
    }
    #[test]
    fn checks_extents_alignment_devices_aliases_and_replay_bounds() {
        let (s, e) = fixture("glmf_stage", 128);
        let l = validate_entry(&s, &e).unwrap();
        let (w, b) = buffers();
        assert!(l.operands(w, b).is_ok());
        let mut bad = b;
        bad.packed.bytes = 1;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.query_fp8.ptr = b.query.ptr;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.cache.device_id = 1;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.pool_pages = usize::MAX;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.pool_slots.ptr = (b.pool_slots.ptr as usize + 4) as *mut c_void;
        assert!(l.operands(w, bad).is_err());
        let mut aligned = b;
        aligned.kda_slots.ptr = (b.kda_slots.ptr as usize + 4) as *mut c_void;
        aligned.first.ptr = aligned.first.ptr.wrapping_byte_add(4);
        aligned.positions.ptr = aligned.positions.ptr.wrapping_byte_add(8);
        aligned.pool_slots.ptr = aligned.pool_slots.ptr.wrapping_byte_add(8);
        assert!(l.operands(w, aligned).is_ok());
        let (s, e) = fixture("glmf_stage", 4096);
        let l = validate_entry(&s, &e).unwrap();
        let mut b = b;
        b.rows = 65;
        assert!(l.operands(w, b).is_ok());
        b.speculative = true;
        assert!(l.operands(w, b).is_err());
        b.rows = 64;
        assert!(l.operands(w, b).is_ok());
    }
}
