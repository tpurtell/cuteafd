//! Checked pointer contracts for the vendor MiniMax MSA index and paged GQA.
use crate::shared::projected_buffers::validate_buffers;
use crate::{
    programs::{Program, ProgramSpec, Programs, Scalar},
    CuteafdDeviceBuffer as Buffer,
};
use std::ffi::c_void;
pub mod producers;
pub mod forward;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryValidator { Producer, Forward }

/// Producer stems are family-owned; unrecognized MiniMax entries retain the
/// strict MSA validator rather than being accepted by a prefix alone.
pub(crate) fn entry_validator(name: &str) -> Option<EntryValidator> {
    if producers::Kind::ALL.into_iter().chain(producers::Kind::HEADS)
        .any(|kind| kind.name() == name) {
        Some(EntryValidator::Producer)
    } else if forward::Kind::ALL.into_iter().chain(forward::Kind::HEADS)
        .any(|kind| kind.name() == name) {
        Some(EntryValidator::Forward)
    } else { None }
}

impl EntryValidator {
    pub(crate) fn validate(self, spec: &ProgramSpec, entry: &serde_json::Value) -> Result<()> {
        match self {
            Self::Producer => producers::validate_producer_entry(spec, entry).map(|_| ()),
            Self::Forward => forward::validate_entry(spec, entry).map(|_| ()),
        }
    }
}

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
    pub fn heads_name(self) -> &'static str {
        match self {
            Self::IndexDecode => "minimax_index_decode_h2",
            Self::IndexPrefill => "minimax_index_prefill_h2",
            Self::AttentionDecode => "minimax_attention_decode_h32",
            Self::AttentionExtend => "minimax_attention_extend_h32",
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
    kv_heads: u32,
}
impl Layout {
    pub fn new(mode: Mode, rows: u32, extent: u32) -> Result<Self> {
        Self::with_heads(mode, rows, extent, 4)
    }
    /// A rank-local 32Q/2KV ABI, never a strided slice of a whole-owner operand.
    pub fn for_heads(mode: Mode, rows: u32, extent: u32) -> Result<Self> {
        Self::with_heads(mode, rows, extent, 2)
    }
    pub fn kv_heads(self) -> u32 { self.kv_heads }
    pub fn query_heads(self) -> u32 { 16 * self.kv_heads }
    pub fn name(self) -> &'static str {
        if self.kv_heads == 2 { self.mode.heads_name() } else { self.mode.name() }
    }
    fn with_heads(mode: Mode, rows: u32, extent: u32, kv_heads: u32) -> Result<Self> {
        if rows == 0 || extent == 0 || rows > i32::MAX as u32 || extent > i32::MAX as u32 {
            return Err(invalid("positive signed-int32 capacities required"));
        }
        let (r, e, h) = (u64::from(rows), u64::from(extent), u64::from(kv_heads));
        let sizes = if mode.index() {
            let blocks = e.div_ceil(2);
            if h * r * blocks >= 1 << 31 {
                return Err(invalid("index score signed indexing extent"));
            }
            vec![
                h * r * 4,
                h * r * blocks * 4,
                if mode == Mode::IndexDecode {
                    h * r * e * 4
                } else {
                    0
                },
                h * r * 4,
                h * r * 4,
                h * r * 16 * 4,
                h * r * 16 * 4,
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
                work * h * 128 * 4
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
                    work * h * 4
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
            kv_heads,
        })
    }
    fn operands(self) -> Vec<Operand> {
        let h = self.kv_heads;
        let q = self.query_heads();
        let mut out = if self.mode.index() {
            let mut out = vec![
                operand("q_fp8", "float8_e4m3fn", format!("[rows,{h},128]"), "in"),
                operand("q_scale", "float32", format!("[rows,{h}]"), "in"),
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
                format!("[{h},{},16]", self.rows),
                "out",
            ));
            out
        } else {
            vec![
                operand("query", "bfloat16", format!("[rows,{q},128]"), "in"),
                operand(
                    "key_cache",
                    "uint8",
                    format!("[pool_pages,64,{h},128] E4M3 bytes"),
                    "in",
                ),
                operand(
                    "value_cache",
                    "uint8",
                    format!("[pool_pages,64,{h},128] E4M3 bytes"),
                    "in",
                ),
                operand("k_descale", "float32", format!("[batch,{h}]"), "in"),
                operand("v_descale", "float32", format!("[batch,{h}]"), "in"),
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
                    format!("[{h},{},16]", self.rows),
                    "in",
                ),
                operand("output", "bfloat16", format!("[rows,{q},128]"), "out"),
                operand("lse", "float32", format!("[rows,{q}] base-2"), "out"),
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
        .find(|mode| spec.name == mode.name() || spec.name == mode.heads_name())
        .ok_or_else(|| invalid("unknown program name"))?;
        let g = &spec.geometry;
        let field = |name: &str| {
            g[name]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid(format!("geometry.{name}")))
        };
        let kv_heads = if spec.name == mode.heads_name() { 2 } else { 4 };
        let layout = Self::with_heads(
            mode,
            field("max_rows")?,
            field(if mode.index() {
                "max_pages"
            } else {
                "max_batch"
            })?,
            kv_heads,
        )?;
        let fixed: &[(&str, u32)] = if mode.index() {
            &[
                ("heads", kv_heads),
                ("head_dim", 128),
                ("block_tokens", 128),
                ("topk", 16),
            ]
        } else {
            &[
                ("page_rows", 64),
                ("query_heads", 16 * kv_heads),
                ("kv_heads", kv_heads),
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
                bytes(b.rows, self.kv_heads as usize * 128)?,
                bytes(b.rows, self.kv_heads as usize * 4)?,
                bytes(b.pool_pages, 8448)?,
                bytes(bytes(b.rows, b.table_stride)?, 4)?,
                bytes(b.rows, 4)?,
                bytes(self.rows as usize, self.kv_heads as usize * 64)?,
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
                bytes(b.rows, self.kv_heads as usize * 128)?,
                bytes(b.rows, self.kv_heads as usize * 4)?,
                bytes(padded, 128)?,
                bytes(padded, 4)?,
                bytes(b.rows, 4)?,
                bytes(b.rows, 4)?,
                bytes(self.rows as usize, self.kv_heads as usize * 64)?,
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
                bytes(b.rows, self.query_heads() as usize * 256)?,
                bytes(b.pool_pages, self.kv_heads as usize * 8192)?,
                bytes(b.pool_pages, self.kv_heads as usize * 8192)?,
                bytes(b.batch, self.kv_heads as usize * 4)?,
                bytes(b.batch, self.kv_heads as usize * 4)?,
                bytes(bytes(b.batch, b.table_stride)?, 4)?,
                bytes(b.batch, 4)?,
                bytes(b.batch + 1, 4)?,
                bytes(self.rows as usize, self.kv_heads as usize * 64)?,
                bytes(b.rows, self.query_heads() as usize * 256)?,
                bytes(b.rows, self.query_heads() as usize * 4)?,
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

    #[test]
    #[ignore = "requires installed placement PROGRAMS.json; run explicitly by name"]
    fn installed_manifest_validates_every_stem_and_minimax_dispatch_partition() -> anyhow::Result<()> {
        let path = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
                .join(".cache/cuteafd/builds/placement-inputs/current/share/PROGRAMS.json"));
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let entries = manifest["programs"].as_array().expect("installed programs array");
        assert_eq!(entries.len(), 1746);
        let dense_slices = [
            "exlr8_dense_mxfp8_k6144_n4096_s1_a8",
            "exlr8_dense_mxfp8_k6144_n256_s1_a8",
            "exlr8_dense_mxfp8_k6144_n6144_s1_a8",
            "exlr8_dense_mxfp8_k6144_n1536_s1_a8",
            "exlr8_dense_mxfp8_k4096_n6144_s1_a8",
            "exlr8_dense_mxfp8_k1536_n6144_s1_a8",
            "exlr8_dense_quant_k4096_bf16_a8",
            "exlr8_dense_quant_k1536_bf16_a8",
        ];
        for name in dense_slices {
            let selected = entries.iter().filter(|entry| entry["name"] == name).collect::<Vec<_>>();
            assert_eq!(selected.len(), 1, "missing or duplicate rank-local dense stem {name}");
            assert_eq!(selected[0]["family"], "minimax");
            assert_eq!(selected[0]["op"], if name.contains("_quant_") { "dense_mxfp8_quant" } else { "dense_mxfp8" });
        }
        // Inkling is appended after every existing family. Pin its complete
        // producer inventory too, so a publication cannot silently add stems.
        let mut inkling2 = vec![
            "inkling2_local_attention_decode_m64".to_string(),
            "inkling2_local_attention_prefill_m4096".to_string(),
            "inkling2_global_attention_decode_m64".to_string(),
            "inkling2_global_attention_prefill_m4096".to_string(),
        ];
        for suffix in ["bf16", "f32"] {
            for name in ["conv_static_h4608", "conv_static_h1024", "conv_dynamic_h1024",
                "clip_local_q", "clip_local_k", "clip_local_r", "clip_global_q", "clip_global_r",
                "gate_h6144", "situ_h2304", "norm_h4608", "norm_v_h8_d128", "kv_pack"] {
                inkling2.push(format!("inkling2_{name}_{suffix}"));
            }
        }
        inkling2.extend(["residual_q1_keep", "residual_q1_add", "residual_q3_keep", "residual_q3_add", "router_e384_k8", "expert_input_quant"]
            .map(|name| format!("inkling2_{name}")));
        let installed = entries.iter().filter(|entry| entry["family"] == "inkling2"
            || entry["name"].as_str().is_some_and(|name| name.starts_with("inkling2")))
            .map(|entry| entry["name"].as_str().expect("installed program name")).collect::<Vec<_>>();
        assert_eq!(installed, inkling2);
        let first = entries.iter().position(|entry| entry["name"].as_str() == Some(inkling2[0].as_str())).unwrap();
        assert_eq!(entries[first..first + inkling2.len()].iter()
            .map(|entry| entry["name"].as_str().unwrap()).collect::<Vec<_>>(), inkling2);
        for entry in &entries[first..first + inkling2.len()] {
            assert_eq!(entry["family"], "inkling2");
            assert!(matches!(entry["op"].as_str(), Some("attention" | "convolution" | "l2_clip" | "gate"
                | "situ" | "rms_norm" | "kv_pack" | "attn_res" | "router" | "expert_input_quant")));
        }
        let mut partition = Vec::new();
        for entry in entries {
            let name = entry["name"].as_str().expect("installed program name");
            if !name.starts_with("minimax_") { continue; }
            let route = if entry["op"] == "index_cache_gather" {
                assert_eq!(entry_validator(name), None);
                "gather"
            } else {
                match entry_validator(name) {
                    Some(EntryValidator::Producer) => "producer",
                    Some(EntryValidator::Forward) => "forward",
                    None => "msa",
                }
            };
            partition.push((name, route));
        }
        partition.sort_unstable();
        assert_eq!(partition, vec![
            ("minimax_attention_decode", "msa"),
            ("minimax_attention_decode_h32", "msa"),
            ("minimax_attention_extend", "msa"),
            ("minimax_attention_extend_h32", "msa"),
            ("minimax_index_cache_gather", "gather"),
            ("minimax_index_decode", "msa"),
            ("minimax_index_decode_h2", "msa"),
            ("minimax_index_prefill", "msa"),
            ("minimax_index_prefill_h2", "msa"),
            ("minimax_index_producer", "producer"),
            ("minimax_index_producer_h2", "producer"),
            ("minimax_kv_pack", "forward"),
            ("minimax_kv_pack_h2", "forward"),
            ("minimax_norm_rope_h2", "producer"),
            ("minimax_norm_rope_h32", "producer"),
            ("minimax_norm_rope_h4", "producer"),
            ("minimax_norm_rope_h64", "producer"),
            ("minimax_oai_split_i12288", "producer"),
            ("minimax_oai_split_i1536", "producer"),
            ("minimax_oai_split_i3072", "producer"),
            ("minimax_oai_split_i6144", "producer"),
            ("minimax_residual_norm", "forward"),
            ("minimax_router_scores", "forward"),
        ]);
        let fixture = crate::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.register_programs(&library, entries)?;
        let programs = library.programs()?.with_manifest(&path)?;
        assert_eq!(programs.names().count(), 1746);
        for entry in entries {
            assert!(programs.contains(entry["name"].as_str().unwrap()));
        }
        println!("validated 1746 installed stems; MiniMax: 8 MSA, 10 Producer, 4 Forward, 1 gather");
        Ok(())
    }

    #[test]
    fn whole_and_narrow_producer_stems_never_dispatch_to_msa_validation() {
        for kind in producers::Kind::ALL.into_iter().chain(producers::Kind::HEADS) {
            assert_eq!(entry_validator(kind.name()), Some(EntryValidator::Producer), "{}", kind.name());
        }
        for kind in forward::Kind::ALL.into_iter().chain(forward::Kind::HEADS) {
            assert_eq!(entry_validator(kind.name()), Some(EntryValidator::Forward), "{}", kind.name());
        }
        for mode in [Mode::IndexDecode, Mode::IndexPrefill, Mode::AttentionDecode, Mode::AttentionExtend] {
            assert_eq!(entry_validator(mode.name()), None);
            assert_eq!(entry_validator(mode.heads_name()), None);
        }
        for unknown in ["minimax_norm_rope_h3", "minimax_index_producer_h3",
            "minimax_oai_split_i1", "minimax_kv_pack_h3"] {
            assert_eq!(entry_validator(unknown), None);
        }
    }

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
    fn whole_contract_is_identical_to_pinned_compiler_manifest() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/minimax_msa_programs.json")).unwrap();
        for entry in manifest["programs"].as_array().unwrap() {
            let spec = ProgramSpec {
                name: entry["name"].as_str().unwrap().into(),
                pointers: entry["pointers"].as_array().unwrap().iter()
                    .map(|p| p["name"].as_str().unwrap().into()).collect(),
                scalar_kinds: entry["scalars"].as_array().unwrap().iter()
                    .map(|s| if s["type"] == "int64" { b'l' } else { b'i' }).collect(),
                capacity_rows: entry["capacity_rows"].as_u64().unwrap() as u32,
                geometry: entry["geometry"].clone(),
                scratch: entry["scratch_bytes_at_capacity"].as_object().unwrap().iter()
                    .map(|(k,v)| (k.clone(), v.as_u64().unwrap())).collect(),
            };
            let layout = validate_entry(&spec, entry).unwrap();
            assert_eq!(layout, Layout::new(layout.mode, layout.rows, layout.extent).unwrap());
            assert_eq!(layout.name(), spec.name);
            assert_eq!(layout.kv_heads(), 4);
            assert_eq!(layout.query_heads(), 64);
        }
    }

    #[test]
    fn narrow_scratch_uses_each_compiled_region_not_half_the_whole_workspace() {
        // Independent values from the pinned fork's _layout(..., heads=2).
        for (mode, rows, extent, scratch) in [
            (Mode::IndexDecode, 1, 1, 2304),
            (Mode::IndexPrefill, 1, 1, 2048),
            (Mode::AttentionDecode, 1, 1, 1024),
            (Mode::AttentionExtend, 1, 1, 5376),
            (Mode::IndexDecode, 64, 16384, 12601856),
            (Mode::IndexPrefill, 4096, 16384, 269615616),
            (Mode::AttentionDecode, 64, 64, 1280),
            (Mode::AttentionExtend, 4096, 64, 1193472),
        ] {
            let narrow = Layout::for_heads(mode, rows, extent).unwrap();
            assert_eq!(narrow.scratch, scratch, "{mode:?}");
            assert_ne!(narrow.scratch, Layout::new(mode, rows, extent).unwrap().scratch / 2);
        }
        assert!(Layout::for_heads(Mode::IndexPrefill, 4096, 1048576).is_err());
        assert!(Layout::for_heads(Mode::AttentionDecode, 0, 1).is_err());
    }

    #[test]
    fn narrow_names_and_geometry_must_agree_and_match_the_full_operand_contract() {
        for mode in [Mode::IndexDecode, Mode::IndexPrefill, Mode::AttentionDecode, Mode::AttentionExtend] {
            let (whole, _) = fixture(mode);
            let layout = Layout::for_heads(mode, 64, if mode.index() { 33 } else { 16 }).unwrap();
            let mut spec = whole.clone();
            spec.name = layout.name().into();
            if mode.index() { spec.geometry["heads"] = json!(2); }
            else {
                spec.geometry["query_heads"] = json!(32);
                spec.geometry["kv_heads"] = json!(2);
            }
            spec.scratch.insert("scratch".into(), layout.scratch);
            let entry = json!({"op":if mode.index() { "minimax_index" } else { "minimax_attention" },
                "pointers":layout.operands().iter().map(|o| json!({"name":o.name,"dtype":o.dtype,
                    "shape":o.shape,"role":o.role})).collect::<Vec<_>>(),
                "scalars":mode.scalars().iter().map(|(name,kind)| json!({"name":name,"type":kind})).collect::<Vec<_>>()});
            assert_eq!(validate_entry(&spec, &entry).unwrap(), layout);
            let mut bad = spec.clone(); bad.name = mode.name().into();
            assert!(validate_entry(&bad, &entry).is_err());
            let mut bad = whole; bad.name = mode.heads_name().into();
            assert!(validate_entry(&bad, &entry).is_err());
            let mut bad = spec.clone(); bad.scratch.insert("scratch".into(), layout.scratch / 2);
            assert!(validate_entry(&bad, &entry).is_err());
            for field in ["name", "dtype", "shape", "role"] {
                let mut bad = entry.clone(); bad["pointers"][0][field] = json!("wrong");
                assert!(validate_entry(&spec, &bad).is_err());
            }
        }
    }

    #[test]
    fn narrow_attention_requires_packed_rank_local_descales_and_exact_extents() {
        for mode in [Mode::AttentionDecode, Mode::AttentionExtend] {
            for rows in [1, 2, 63, 64] {
                let layout = Layout::for_heads(mode, 64, 64).unwrap();
                let [query, keys, values, key_scales, value_scales, pages, lengths, query_offsets, selections, output, lse, scratch] = buffers();
                let exact = |b: Buffer, bytes| Buffer { bytes, ..b };
                let batch = rows;
                let b = AttentionBatch { rows, batch, table_stride: 33, pool_pages: 2,
                    query: exact(query, rows * 8192), keys: exact(keys, 2 * 16384),
                    values: exact(values, 2 * 16384), key_scales: exact(key_scales, batch * 8),
                    value_scales: exact(value_scales, batch * 8), pages: exact(pages, batch * 33 * 4),
                    lengths: exact(lengths, batch * 4), query_offsets: exact(query_offsets, (batch + 1) * 4),
                    selections: exact(selections, 2 * 64 * 16 * 4), output: exact(output, rows * 8192),
                    lse: exact(lse, rows * 128), scratch: exact(scratch, layout.scratch as usize) };
                assert!(layout.attention(b).is_ok());
                assert!(Layout::new(mode, 64, 64).unwrap().attention(b).is_err());
                for i in 0..12 {
                    let mut short = b;
                    let target = match i { 0 => &mut short.query, 1 => &mut short.keys,
                        2 => &mut short.values, 3 => &mut short.key_scales, 4 => &mut short.value_scales,
                        5 => &mut short.pages, 6 => &mut short.lengths, 7 => &mut short.query_offsets,
                        8 => &mut short.selections, 9 => &mut short.output, 10 => &mut short.lse,
                        _ => &mut short.scratch };
                    target.bytes -= 1;
                    assert!(layout.attention(short).is_err(), "{mode:?} operand {i}");
                }
                let offset = Buffer { ptr: (key_scales.ptr as usize + 8) as *mut c_void, ..b.key_scales };
                assert!(layout.attention(AttentionBatch { key_scales: offset, ..b }).is_err());
                assert!(layout.attention(AttentionBatch { value_scales: offset, ..b }).is_err());
                assert!(layout.attention(AttentionBatch { output: b.query, ..b }).is_err());
                assert!(layout.attention(AttentionBatch { keys: Buffer { device_id: 1, ..b.keys }, ..b }).is_err());
            }
        }
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
        Self::bind_name(programs, mode.name())
    }
    pub fn bind_heads(programs: &'a Programs<'a>, mode: Mode) -> Result<Self> {
        Self::bind_name(programs, mode.heads_name())
    }
    fn bind_name(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let spec = programs
            .spec(name)
            .map_err(|e| invalid(e.to_string()))?;
        let layout = Layout::from_spec(spec)?;
        let operands = layout.operands();
        let names = operands.iter().map(|o| o.name).collect::<Vec<_>>();
        let program = programs
            .program(name, &names)
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
