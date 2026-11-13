//! Checked pointer contracts for the vendor MiniMax MSA index and paged GQA.
use crate::shared::projected_buffers::validate_buffers;
use crate::{
    programs::{Program, ProgramSpec, Programs, Scalar},
    CuteafdDeviceBuffer as Buffer,
};
use std::ffi::c_void;
pub mod producers;
pub mod forward;

#[derive(Debug, thiserror::Error)]
#[error("invalid MiniMax MSA program: {0}")]
pub struct MsaError(String);
type Result<T> = std::result::Result<T, MsaError>;
fn invalid(message: impl Into<String>) -> MsaError {
    MsaError(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    IndexDecode,
    IndexPrefill,
    AttentionDecode,
    AttentionExtend,
}
impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::IndexDecode => "minimax_index_decode",
            Self::IndexPrefill => "minimax_index_prefill",
            Self::AttentionDecode => "minimax_attention_decode",
            Self::AttentionExtend => "minimax_attention_extend",
        }
    }
    fn mode(self) -> &'static str {
        match self {
            Self::IndexDecode | Self::AttentionDecode => "decode",
            Self::IndexPrefill => "prefill",
            Self::AttentionExtend => "extend",
        }
    }
    fn index(self) -> bool {
        matches!(self, Self::IndexDecode | Self::IndexPrefill)
    }
    fn scalars(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::IndexDecode => &[
                ("rows", "int32"),
                ("table_width", "int32"),
                ("table_stride", "int32"),
                ("pool_pages", "int64"),
            ],
            Self::IndexPrefill => &[("rows", "int32"), ("k_rows", "int32")],
            _ => &[
                ("rows", "int32"),
                ("batch", "int32"),
                ("table_stride", "int32"),
                ("pool_pages", "int64"),
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Operand {
    name: &'static str,
    dtype: &'static str,
    shape: String,
    role: &'static str,
    align: usize,
}
fn operand(
    name: &'static str,
    dtype: &'static str,
    shape: impl Into<String>,
    role: &'static str,
) -> Operand {
    Operand {
        name,
        dtype,
        shape: shape.into(),
        role,
        align: if dtype == "int32" { 4 } else { 16 },
    }
}

/// Capacities and scratch are checked against the pinned fork's exact layout,
/// not trusted just because the manifest and native table agree with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub mode: Mode,
    pub rows: u32,
    pub extent: u32,
    pub scratch: u64,
}
impl Layout {
    pub fn new(mode: Mode, rows: u32, extent: u32) -> Result<Self> {
        if rows == 0 || extent == 0 || rows > i32::MAX as u32 || extent > i32::MAX as u32 {
            return Err(invalid("positive signed-int32 capacities required"));
        }
        let (r, e) = (u64::from(rows), u64::from(extent));
        let sizes = if mode.index() {
            let blocks = e.div_ceil(2);
            if 4 * r * blocks >= 1 << 31 {
                return Err(invalid("index score signed indexing extent"));
            }
            vec![
                4 * r * 4,
                4 * r * blocks * 4,
                if mode == Mode::IndexDecode {
                    4 * r * e * 4
                } else {
                    0
                },
                4 * r * 4,
                4 * r * 4,
                4 * r * 16 * 4,
                4 * r * 16 * 4,
                (r + 1) * 2 * 4,
                256,
            ]
        } else {
            let work = if mode == Mode::AttentionDecode {
                r
            } else {
                r.div_ceil(8) + e
            };
            let union = if mode == Mode::AttentionExtend {
                work * 4 * 128 * 4
            } else {
                0
            };
            vec![
                work * 4,
                work * 4,
                work.max(e + 1) * 4,
                work * 4,
                union,
                union,
                if mode == Mode::AttentionExtend {
                    work * 4 * 4
                } else {
                    0
                },
            ]
        };
        let scratch = sizes
            .into_iter()
            .try_fold(0u64, |sum, n| {
                n.checked_add(255)
                    .map(|n| n / 256 * 256)
                    .and_then(|n| sum.checked_add(n))
            })
            .ok_or_else(|| invalid("scratch overflow"))?;
        if scratch > isize::MAX as u64 {
            return Err(invalid("scratch exceeds addressable bytes"));
        }
        Ok(Self {
            mode,
            rows,
            extent,
            scratch,
        })
    }
    fn operands(self) -> Vec<Operand> {
        let mut out = if self.mode.index() {
            let mut out = vec![
                operand("q_fp8", "float8_e4m3fn", "[rows,4,128]", "in"),
                operand("q_scale", "float32", "[rows,4]", "in"),
            ];
            if self.mode == Mode::IndexDecode {
                out.extend([
                    operand("index_k_cache", "uint8", "[pool_pages,8448]", "in"),
                    operand("page_table", "int32", "[rows,table_stride]", "in"),
                    operand("cache_lengths", "int32", "[rows]", "in"),
                ]);
            } else {
                out.extend([
                    operand("k_fp8", "float8_e4m3fn", "[ceil(k_rows/256)*256,128]", "in"),
                    operand("k_scale", "float32", "[ceil(k_rows/256)*256]", "in"),
                    operand(
                        "k_start",
                        "int32",
                        "[rows] request base, multiple of 128",
                        "in",
                    ),
                    operand("k_end", "int32", "[rows] exclusive causal end", "in"),
                ]);
            }
            out.push(operand(
                "q2k_indices",
                "int32",
                format!("[4,{},16]", self.rows),
                "out",
            ));
            out
        } else {
            vec![
                operand("query", "bfloat16", "[rows,64,128]", "in"),
                operand(
                    "key_cache",
                    "uint8",
                    "[pool_pages,64,4,128] E4M3 bytes",
                    "in",
                ),
                operand(
                    "value_cache",
                    "uint8",
                    "[pool_pages,64,4,128] E4M3 bytes",
                    "in",
                ),
                operand("k_descale", "float32", "[batch,4]", "in"),
                operand("v_descale", "float32", "[batch,4]", "in"),
                operand("page_table", "int32", "[batch,table_stride]", "in"),
                operand(
                    "cache_lengths",
                    "int32",
                    "[batch] including appended queries",
                    "in",
                ),
                operand(
                    "cu_seqlens_q",
                    "int32",
                    "[batch+1] packed query offsets; decode arange",
                    "in",
                ),
                operand(
                    "q2k_indices",
                    "int32",
                    format!("[4,{},16]", self.rows),
                    "in",
                ),
                operand("output", "bfloat16", "[rows,64,128]", "out"),
                operand("lse", "float32", "[rows,64] base-2", "out"),
            ]
        };
        out.push(operand(
            "scratch",
            "uint8",
            format!("[{}]", self.scratch),
            "scratch",
        ));
        out
    }
    pub fn from_spec(spec: &ProgramSpec) -> Result<Self> {
        let mode = [
            Mode::IndexDecode,
            Mode::IndexPrefill,
            Mode::AttentionDecode,
            Mode::AttentionExtend,
        ]
        .into_iter()
        .find(|mode| spec.name == mode.name())
        .ok_or_else(|| invalid("unknown program name"))?;
        let g = &spec.geometry;
        let field = |name: &str| {
            g[name]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid(format!("geometry.{name}")))
        };
        let layout = Self::new(
            mode,
            field("max_rows")?,
            field(if mode.index() {
                "max_pages"
            } else {
                "max_batch"
            })?,
        )?;
        let fixed: &[(&str, u32)] = if mode.index() {
            &[
                ("heads", 4),
                ("head_dim", 128),
                ("block_tokens", 128),
                ("topk", 16),
            ]
        } else {
            &[
                ("page_rows", 64),
                ("query_heads", 64),
                ("kv_heads", 4),
                ("head_dim", 128),
                ("block_tokens", 128),
                ("topk", 16),
            ]
        };
        if fixed.iter().any(|&(key, n)| field(key).ok() != Some(n))
            || g["mode"] != mode.mode()
            || (!mode.index() && g["softmax_scale"].as_f64() != Some(0.08838834764831845))
            || spec.capacity_rows != layout.rows
            || spec.scratch.len() != 1
            || spec.scratch.get("scratch") != Some(&layout.scratch)
            || spec
                .pointers
                .iter()
                .map(String::as_str)
                .ne(layout.operands().iter().map(|o| o.name))
            || spec.scalar_kinds
                != if mode == Mode::IndexPrefill {
                    b"ii".as_slice()
                } else {
                    b"iiil".as_slice()
                }
        {
            return Err(invalid(
                "geometry, pointer, scalar, capacity or scratch contract",
            ));
        }
        Ok(layout)
    }
    fn validate<const N: usize>(self, buffers: [Buffer; N], sizes: [usize; N]) -> Result<()> {
        let operands = self.operands();
        if N != operands.len() {
            return Err(invalid("pointer count"));
        }
        let alignments: [usize; N] = std::array::from_fn(|i| operands[i].align);
        let writable = operands
            .iter()
            .enumerate()
            .filter_map(|(i, o)| (o.role != "in").then_some(i))
            .collect::<Vec<_>>();
        validate_buffers(
            &std::array::from_fn(|i| (buffers[i], sizes[i])),
            &alignments,
            &writable,
        )
        .map_err(invalid)
    }
    fn check_rows(self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.rows as usize {
            return Err(invalid("live row capacity"));
        }
        Ok(())
    }
}

pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<Layout> {
    let layout = Layout::from_spec(spec)?;
    let expected = layout.operands();
    let pointers = entry["pointers"]
        .as_array()
        .ok_or_else(|| invalid("pointer metadata"))?;
    let scalars = entry["scalars"]
        .as_array()
        .ok_or_else(|| invalid("scalar metadata"))?;
    if entry["op"]
        != if layout.mode.index() {
            "minimax_index"
        } else {
            "minimax_attention"
        }
        || pointers.len() != expected.len()
        || pointers.iter().zip(expected).any(|(actual, expected)| {
            actual["name"] != expected.name
                || actual["dtype"] != expected.dtype
                || actual["shape"] != expected.shape
                || actual["role"] != expected.role
        })
        || scalars.len() != layout.mode.scalars().len()
        || scalars
            .iter()
            .zip(layout.mode.scalars())
            .any(|(s, (name, kind))| s["name"] != *name || s["type"] != *kind)
    {
        return Err(invalid("exported operand or operation contract"));
    }
    Ok(layout)
}

#[derive(Clone, Copy)]
pub struct IndexDecodeBatch {
    pub rows: usize,
    pub table_width: usize,
    pub table_stride: usize,
    pub pool_pages: usize,
    pub query: Buffer,
    pub scales: Buffer,
    pub cache: Buffer,
    pub pages: Buffer,
    pub lengths: Buffer,
    pub selections: Buffer,
    pub scratch: Buffer,
}
#[derive(Clone, Copy)]
pub struct IndexPrefillBatch {
    pub rows: usize,
    pub k_rows: usize,
    pub query: Buffer,
    pub scales: Buffer,
    pub keys: Buffer,
    pub key_scales: Buffer,
    pub starts: Buffer,
    pub ends: Buffer,
    pub selections: Buffer,
    pub scratch: Buffer,
}
#[derive(Clone, Copy)]
pub struct AttentionBatch {
    pub rows: usize,
    pub batch: usize,
    pub table_stride: usize,
    pub pool_pages: usize,
    pub query: Buffer,
    pub keys: Buffer,
    pub values: Buffer,
    pub key_scales: Buffer,
    pub value_scales: Buffer,
    pub pages: Buffer,
    pub lengths: Buffer,
    pub query_offsets: Buffer,
    pub selections: Buffer,
    pub output: Buffer,
    pub lse: Buffer,
    pub scratch: Buffer,
}
fn bytes(count: usize, width: usize) -> Result<usize> {
    count
        .checked_mul(width)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("operand byte overflow"))
}
fn pages(pool: usize, stride: usize) -> Result<()> {
    if pool == 0 || pool > i32::MAX as usize || stride == 0 || stride > i32::MAX as usize {
        return Err(invalid("page pool or table stride"));
    }
    Ok(())
}
impl Layout {
    fn decode(self, b: IndexDecodeBatch) -> Result<[Buffer; 7]> {
        self.check_rows(b.rows)?;
        pages(b.pool_pages, b.table_stride)?;
        if self.mode != Mode::IndexDecode
            || b.table_width == 0
            || b.table_width > self.extent as usize
            || b.table_width > b.table_stride
        {
            return Err(invalid("decode index mode or table width"));
        }
        let buffers = [
            b.query,
            b.scales,
            b.cache,
            b.pages,
            b.lengths,
            b.selections,
            b.scratch,
        ];
        self.validate(
            buffers,
            [
                bytes(b.rows, 512)?,
                bytes(b.rows, 16)?,
                bytes(b.pool_pages, 8448)?,
                bytes(bytes(b.rows, b.table_stride)?, 4)?,
                bytes(b.rows, 4)?,
                bytes(self.rows as usize, 256)?,
                self.scratch as usize,
            ],
        )?;
        Ok(buffers)
    }
    fn prefill(self, b: IndexPrefillBatch) -> Result<[Buffer; 8]> {
        self.check_rows(b.rows)?;
        if self.mode != Mode::IndexPrefill
            || b.k_rows == 0
            || b.k_rows > i32::MAX as usize
            || b.k_rows.div_ceil(64) > self.extent as usize
        {
            return Err(invalid("prefill index mode or key capacity"));
        }
        let padded = b.k_rows.div_ceil(256) * 256;
        let buffers = [
            b.query,
            b.scales,
            b.keys,
            b.key_scales,
            b.starts,
            b.ends,
            b.selections,
            b.scratch,
        ];
        self.validate(
            buffers,
            [
                bytes(b.rows, 512)?,
                bytes(b.rows, 16)?,
                bytes(padded, 128)?,
                bytes(padded, 4)?,
                bytes(b.rows, 4)?,
                bytes(b.rows, 4)?,
                bytes(self.rows as usize, 256)?,
                self.scratch as usize,
            ],
        )?;
        Ok(buffers)
    }
    fn attention(self, b: AttentionBatch) -> Result<[Buffer; 12]> {
        self.check_rows(b.rows)?;
        pages(b.pool_pages, b.table_stride)?;
        if self.mode.index()
            || b.batch == 0
            || b.batch > self.extent as usize
            || b.batch > b.rows
            || (self.mode == Mode::AttentionDecode && b.batch != b.rows)
        {
            return Err(invalid("attention mode or batch capacity"));
        }
        let buffers = [
            b.query,
            b.keys,
            b.values,
            b.key_scales,
            b.value_scales,
            b.pages,
            b.lengths,
            b.query_offsets,
            b.selections,
            b.output,
            b.lse,
            b.scratch,
        ];
        self.validate(
            buffers,
            [
                bytes(b.rows, 16384)?,
                bytes(b.pool_pages, 32768)?,
                bytes(b.pool_pages, 32768)?,
                bytes(b.batch, 16)?,
                bytes(b.batch, 16)?,
                bytes(bytes(b.batch, b.table_stride)?, 4)?,
                bytes(b.batch, 4)?,
                bytes(b.batch + 1, 4)?,
                bytes(self.rows as usize, 256)?,
                bytes(b.rows, 16384)?,
                bytes(b.rows, 256)?,
                self.scratch as usize,
            ],
        )?;
        Ok(buffers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture(mode: Mode) -> (ProgramSpec, serde_json::Value) {
        let layout = Layout::new(mode, 64, if mode.index() { 33 } else { 16 }).unwrap();
        let mut geometry = json!({"max_rows":layout.rows,"mode":mode.mode(),
            "head_dim":128,"block_tokens":128,"topk":16});
        if mode.index() {
            geometry["max_pages"] = json!(layout.extent);
            geometry["heads"] = json!(4);
        } else {
            for (key, n) in [
                ("max_batch", 16),
                ("page_rows", 64),
                ("query_heads", 64),
                ("kv_heads", 4),
            ] {
                geometry[key] = json!(n);
            }
            geometry["softmax_scale"] = json!(0.08838834764831845);
        }
        let operands = layout.operands();
        let spec = ProgramSpec {
            name: mode.name().into(),
            pointers: operands.iter().map(|o| o.name.into()).collect(),
            scalar_kinds: mode
                .scalars()
                .iter()
                .map(|(_, kind)| if *kind == "int64" { b'l' } else { b'i' })
                .collect(),
            scratch: [("scratch".into(), layout.scratch)].into_iter().collect(),
            capacity_rows: layout.rows,
            geometry,
        };
        let entry = json!({"op":if mode.index() { "minimax_index" } else { "minimax_attention" },
            "pointers":operands.iter().map(|o| json!({"name":o.name,"dtype":o.dtype,"shape":o.shape,"role":o.role})).collect::<Vec<_>>(),
            "scalars":mode.scalars().iter().map(|(name,kind)| json!({"name":name,"type":kind})).collect::<Vec<_>>()});
        (spec, entry)
    }

    #[test]
    fn exact_modes_reject_mutated_geometry_and_abi() {
        for mode in [
            Mode::IndexDecode,
            Mode::IndexPrefill,
            Mode::AttentionDecode,
            Mode::AttentionExtend,
        ] {
            let (spec, entry) = fixture(mode);
            assert_eq!(validate_entry(&spec, &entry).unwrap().mode, mode);
            for field in ["max_rows", "head_dim", "block_tokens", "topk"] {
                let mut broken = spec.clone();
                broken.geometry[field] = json!(129);
                assert!(validate_entry(&broken, &entry).is_err(), "{mode:?} {field}");
            }
            for field in ["name", "dtype", "shape", "role"] {
                let mut broken = entry.clone();
                broken["pointers"][0][field] = json!("wrong");
                assert!(validate_entry(&spec, &broken).is_err(), "{mode:?} {field}");
            }
            let mut broken = spec.clone();
            broken
                .scratch
                .insert("scratch".into(), spec.scratch["scratch"] + 256);
            assert!(validate_entry(&broken, &entry).is_err());
            broken = spec.clone();
            broken.scalar_kinds[0] = b'l';
            assert!(validate_entry(&broken, &entry).is_err());
            let mut broken = entry.clone();
            broken["scalars"][0]["name"] = json!("wrong");
            assert!(validate_entry(&spec, &broken).is_err());
        }
    }

    #[test]
    fn pinned_fork_scratch_vectors() {
        // Independently checked against the pinned fork's _layout in the script suite.
        let cases = [
            (Mode::IndexDecode, 1, 1, 2304),
            (Mode::IndexPrefill, 1, 1, 2048),
            (Mode::AttentionDecode, 1, 1, 1024),
            (Mode::AttentionExtend, 1, 1, 9472),
            (Mode::IndexDecode, 64, 16384, 25202688),
            (Mode::IndexPrefill, 4096, 16384, 539197952),
            (Mode::AttentionDecode, 64, 64, 1280),
            (Mode::AttentionExtend, 4096, 64, 2377728),
        ];
        for (mode, rows, extent, expected) in cases {
            assert_eq!(
                Layout::new(mode, rows, extent).unwrap().scratch,
                expected,
                "{mode:?}"
            );
        }
        assert!(Layout::new(Mode::IndexPrefill, 4096, 1048576).is_err());
        assert!(Layout::new(Mode::AttentionDecode, 0, 1).is_err());
    }

    fn buffers<const N: usize>() -> [Buffer; N] {
        std::array::from_fn(|i| Buffer {
            ptr: ((i + 1) * 0x100000000usize) as *mut c_void,
            bytes: 1 << 30,
            device_id: 0,
            flags: 0,
        })
    }
    #[test]
    fn extent_device_alias_and_capacity_checks_precede_launch() {
        let layout = Layout::new(Mode::IndexDecode, 64, 33).unwrap();
        let [query, scales, cache, pages, lengths, selections, scratch] = buffers();
        let b = IndexDecodeBatch {
            rows: 1,
            table_width: 33,
            table_stride: 33,
            pool_pages: 2,
            query,
            scales,
            cache,
            pages,
            lengths,
            selections,
            scratch,
        };
        assert!(layout.decode(b).is_ok());
        assert!(layout
            .decode(IndexDecodeBatch {
                table_width: 34,
                ..b
            })
            .is_err());
        assert!(layout
            .decode(IndexDecodeBatch {
                table_stride: 32,
                ..b
            })
            .is_err());
        assert!(layout.decode(IndexDecodeBatch { rows: 65, ..b }).is_err());
        assert!(layout
            .decode(IndexDecodeBatch {
                selections: Buffer {
                    bytes: 256,
                    ..selections
                },
                ..b
            })
            .is_err());
        assert!(layout
            .decode(IndexDecodeBatch {
                cache: Buffer {
                    device_id: 1,
                    ..cache
                },
                ..b
            })
            .is_err());
        assert!(layout
            .decode(IndexDecodeBatch {
                selections: query,
                ..b
            })
            .is_err());
        assert!(layout
            .decode(IndexDecodeBatch {
                pool_pages: i32::MAX as usize + 1,
                ..b
            })
            .is_err());
        assert!(layout
            .decode(IndexDecodeBatch {
                query: Buffer {
                    ptr: 17usize as *mut c_void,
                    ..query
                },
                ..b
            })
            .is_err());

        let layout = Layout::new(Mode::IndexPrefill, 64, 33).unwrap();
        let [query, scales, keys, key_scales, starts, ends, selections, scratch] = buffers();
        let b = IndexPrefillBatch {
            rows: 1,
            k_rows: 2049,
            query,
            scales,
            keys,
            key_scales,
            starts,
            ends,
            selections,
            scratch,
        };
        assert!(layout.prefill(b).is_ok());
        assert!(layout
            .prefill(IndexPrefillBatch {
                keys: Buffer {
                    bytes: 2049 * 128,
                    ..keys
                },
                ..b
            })
            .is_err());
        assert!(layout
            .prefill(IndexPrefillBatch { k_rows: 2113, ..b })
            .is_err());
        for mode in [Mode::AttentionDecode, Mode::AttentionExtend] {
            let layout = Layout::new(mode, 64, 16).unwrap();
            let [query, keys, values, key_scales, value_scales, pages, lengths, query_offsets, selections, output, lse, scratch] =
                buffers();
            let b = AttentionBatch {
                rows: 2,
                batch: 2,
                table_stride: 33,
                pool_pages: 2,
                query,
                keys,
                values,
                key_scales,
                value_scales,
                pages,
                lengths,
                query_offsets,
                selections,
                output,
                lse,
                scratch,
            };
            assert!(layout.attention(b).is_ok());
            assert!(layout.attention(AttentionBatch { batch: 17, ..b }).is_err());
            assert!(layout
                .attention(AttentionBatch { output: query, ..b })
                .is_err());
            assert!(layout
                .attention(AttentionBatch {
                    query_offsets: Buffer {
                        bytes: 8,
                        ..query_offsets
                    },
                    ..b
                })
                .is_err());
            assert_eq!(
                layout.attention(AttentionBatch { batch: 1, ..b }).is_ok(),
                mode == Mode::AttentionExtend
            );
        }
    }
}

pub struct Msa<'a> {
    program: Program<'a>,
    pub layout: Layout,
}
impl<'a> Msa<'a> {
    pub fn bind(programs: &'a Programs<'a>, mode: Mode) -> Result<Self> {
        let spec = programs
            .spec(mode.name())
            .map_err(|e| invalid(e.to_string()))?;
        let layout = Layout::from_spec(spec)?;
        let operands = layout.operands();
        let names = operands.iter().map(|o| o.name).collect::<Vec<_>>();
        let program = programs
            .program(mode.name(), &names)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self { program, layout })
    }
    /// # Safety
    /// Page IDs and lengths must be in range, initialized and causal. All buffers
    /// and the stream must outlive queued work and every graph using their pointers.
    pub unsafe fn index_decode(&self, b: IndexDecodeBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.decode(b)?;
        // SAFETY: checked extents, devices, alignments and writable aliases;
        // device metadata and stream/graph lifetimes are the caller's contract.
        unsafe {
            self.program.launch(
                &buffers.map(|b| b.ptr),
                &[
                    Scalar::I32(b.rows as i32),
                    Scalar::I32(b.table_width as i32),
                    Scalar::I32(b.table_stride as i32),
                    Scalar::I64(b.pool_pages as i64),
                ],
                stream,
            )
        }
        .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Starts must be 128-aligned request bases, ends causal, 0 <= start < end <=
    /// k_rows. Key padding must be initialized. Retain storage through graph use.
    pub unsafe fn index_prefill(&self, b: IndexPrefillBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.prefill(b)?;
        // SAFETY: host extents and aliases checked; device indices/lifetimes are caller-owned.
        unsafe {
            self.program.launch(
                &buffers.map(|b| b.ptr),
                &[Scalar::I32(b.rows as i32), Scalar::I32(b.k_rows as i32)],
                stream,
            )
        }
        .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Page IDs, selections, lengths and packed query offsets must address valid
    /// causal initialized rows. Decode offsets are arange; extend requests are
    /// nonempty. Retain buffers and stream through queued work and graph use.
    pub unsafe fn attention(&self, b: AttentionBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.attention(b)?;
        // SAFETY: host capacities and writable aliases checked; metadata/lifetimes are caller-owned.
        unsafe {
            self.program.launch(
                &buffers.map(|b| b.ptr),
                &[
                    Scalar::I32(b.rows as i32),
                    Scalar::I32(b.batch as i32),
                    Scalar::I32(b.table_stride as i32),
                    Scalar::I64(b.pool_pages as i64),
                ],
                stream,
            )
        }
        .map_err(|e| invalid(e.to_string()))
    }
}
