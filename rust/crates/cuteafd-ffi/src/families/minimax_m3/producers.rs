//! MiniMax-only Gemma/partial-NeoX and OAI producers. No epilogue math lives here.
use super::{bytes, invalid, operand, Operand, Result};
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    NormRope { heads: u32 },
    Index,
    IndexHeads,
    Oai { width: u32 },
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::NormRope { heads: 64 } => "minimax_norm_rope_h64",
            Self::NormRope { heads: 32 } => "minimax_norm_rope_h32",
            Self::NormRope { heads: 2 } => "minimax_norm_rope_h2",
            Self::NormRope { heads: 4 } => "minimax_norm_rope_h4",
            Self::Index => "minimax_index_producer",
            Self::IndexHeads => "minimax_index_producer_h2",
            Self::Oai { width: 1536 } => "minimax_oai_split_i1536",
            Self::Oai { width: 6144 } => "minimax_oai_split_i6144",
            Self::Oai { width: 3072 } => "minimax_oai_split_i3072",
            Self::Oai { width: 12288 } => "minimax_oai_split_i12288",
            _ => "invalid_minimax_producer",
        }
    }
    fn op(self) -> &'static str {
        match self {
            Self::NormRope { .. } => "minimax_norm_rope",
            Self::Index | Self::IndexHeads => "minimax_index_producer",
            Self::Oai { .. } => "minimax_oai_split",
        }
    }
    fn index_heads(self) -> Option<u32> {
        match self { Self::Index => Some(4), Self::IndexHeads => Some(2), _ => None }
    }
    pub const HEADS: [Self; 5] = [Self::NormRope { heads: 32 }, Self::NormRope { heads: 2 },
        Self::IndexHeads, Self::Oai { width: 1536 }, Self::Oai { width: 6144 }];
    pub const ALL: [Self; 5] = [Self::NormRope { heads: 64 }, Self::NormRope { heads: 4 },
        Self::Index, Self::Oai { width: 3072 }, Self::Oai { width: 12288 }];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProducerLayout {
    pub kind: Kind,
    pub rows: u32,
}
impl ProducerLayout {
    pub fn new(kind: Kind, rows: u32) -> Result<Self> {
        if !(Kind::ALL.contains(&kind) || Kind::HEADS.contains(&kind)) || rows == 0 || rows > 4096 {
            return Err(invalid("producer geometry or row capacity"));
        }
        Ok(Self { kind, rows })
    }
    pub fn scratch(self) -> u64 {
        u64::from(self.rows) * match self.kind {
            Kind::NormRope { heads } => u64::from(heads) * 256,
            Kind::Index | Kind::IndexHeads => u64::from(2 * self.kind.index_heads().unwrap() + 1) * 128 * 2,
            Kind::Oai { .. } => 0,
        }
    }
    fn operands(self) -> Vec<Operand> {
        let h = self.kind.index_heads().unwrap_or(0);
        match self.kind {
            Kind::NormRope { heads } => vec![
                operand("x", "bfloat16", format!("[rows,{heads},128]"), "in"),
                operand("weight", "bfloat16", "[128] Gemma offset", "in"),
                operand("positions", "int64", "[rows]", "in"),
                operand("cos_sin", "float32", "[positions,64] (cos32 then sin32)", "in"),
                operand("out", "bfloat16", format!("[rows,{heads},128]"), "out"),
                operand("scratch", "bfloat16", format!("[rows,{heads},128]"), "scratch"),
            ],
            Kind::Index | Kind::IndexHeads => vec![
                operand("iq", "bfloat16", format!("[rows,{h},128]"), "in"),
                operand("ik", "bfloat16", "[rows,128]", "in"),
                operand("iq_norm", "bfloat16", "[128] Gemma offset", "in"),
                operand("ik_norm", "bfloat16", "[128] Gemma offset", "in"),
                operand("positions", "int64", "[rows]", "in"),
                operand("slots", "int64", "[rows] negative skips paged write", "in"),
                operand("cos_sin", "float32", "[positions,64] (cos32 then sin32)", "in"),
                operand("q_fp8", "float8_e4m3fn", format!("[rows,{h},128]"), "out"),
                operand("q_scale", "float32", format!("[rows,{h}]"), "out"),
                operand("k_fp8", "float8_e4m3fn", "[rows,128]", "out"),
                operand("k_scale", "float32", "[rows]", "out"),
                operand("cache", "uint8", "[pool_pages,8448] keys then FP32 scales", "inout"),
                operand("scratch", "bfloat16", format!("[rows,{},128]", 2 * h + 1), "scratch"),
            ],
            Kind::Oai { width } => vec![
                operand("gate", "bfloat16", format!("[rows,{width}]"), "in"),
                operand("up", "bfloat16", format!("[rows,{width}]"), "in"),
                operand("out", "bfloat16", format!("[rows,{width}]"), "out"),
            ],
        }
    }
    pub fn from_spec(spec: &ProgramSpec) -> Result<Self> {
        let kind = Kind::ALL.into_iter().chain(Kind::HEADS).find(|kind| spec.name == kind.name())
            .ok_or_else(|| invalid("unknown producer name"))?;
        let layout = Self::new(kind, spec.capacity_rows)?;
        let g = &spec.geometry;
        let geometry = match kind {
            Kind::NormRope { heads } => g["heads"] == heads && g["head_dim"] == 128
                && g["rotary_dim"] == 64 && g["eps"].as_f64() == Some(1e-6),
            Kind::Index | Kind::IndexHeads => g["heads"] == kind.index_heads().unwrap() && g["head_dim"] == 128 && g["rotary_dim"] == 64
                && g["eps"].as_f64() == Some(1e-6) && g["page_rows"] == 64,
            Kind::Oai { width } => g["width"] == width && g["alpha"].as_f64() == Some(1.702)
                && g["beta"].as_f64() == Some(1.0) && g["limit"].as_f64() == Some(7.0),
        };
        let scratch = layout.scratch();
        if !geometry || spec.scalar_kinds != b"i"
            || spec.pointers.iter().map(String::as_str).ne(layout.operands().iter().map(|o| o.name))
            || (scratch == 0 && !spec.scratch.is_empty())
            || (scratch != 0 && (spec.scratch.len() != 1 || spec.scratch.get("scratch") != Some(&scratch))) {
            return Err(invalid("producer geometry, operands, scalars or scratch"));
        }
        Ok(layout)
    }
    fn validate<const N: usize>(self, rows: usize, buffers: [Buffer; N], sizes: [usize; N]) -> Result<[Buffer; N]> {
        if rows == 0 || rows > self.rows as usize { return Err(invalid("producer live row capacity")); }
        let operands = self.operands();
        if operands.len() != N { return Err(invalid("producer operand count")); }
        let align: [usize; N] = std::array::from_fn(|i| match operands[i].dtype {
            "int64" => 8, "float32" if self.kind != Kind::IndexHeads => 4, _ => 16,
        });
        let writable = operands.iter().enumerate().filter_map(|(i, o)| (o.role != "in").then_some(i)).collect::<Vec<_>>();
        super::validate_buffers(&std::array::from_fn(|i| (buffers[i], sizes[i])), &align, &writable).map_err(invalid)?;
        Ok(buffers)
    }
    fn norm(self, b: NormRopeBatch) -> Result<[Buffer; 6]> {
        let Kind::NormRope { heads } = self.kind else { return Err(invalid("not a norm/rope producer")); };
        if b.positions_capacity == 0 { return Err(invalid("empty RoPE table")); }
        let x = bytes(b.rows, heads as usize * 256)?;
        self.validate(b.rows, [b.input, b.weight, b.positions, b.cos_sin, b.output, b.scratch],
            [x, 256, bytes(b.rows, 8)?, bytes(b.positions_capacity, 256)?, x, x])
    }
    fn index(self, b: IndexProducerBatch) -> Result<[Buffer; 13]> {
        let h = self.kind.index_heads().ok_or_else(|| invalid("not an index producer"))? as usize;
        if b.positions_capacity == 0 || b.pool_pages == 0
            || b.pool_pages > i32::MAX as usize { return Err(invalid("index producer dimensions")); }
        self.validate(b.rows, [b.query, b.key, b.query_norm, b.key_norm, b.positions, b.slots,
            b.cos_sin, b.query_fp8, b.query_scale, b.key_fp8, b.key_scale, b.cache, b.scratch],
            [bytes(b.rows, h * 256)?, bytes(b.rows, 256)?, 256, 256, bytes(b.rows, 8)?, bytes(b.rows, 8)?,
                bytes(b.positions_capacity, 256)?, bytes(b.rows, h * 128)?, bytes(b.rows, h * 4)?,
                bytes(b.rows, 128)?, bytes(b.rows, 4)?, bytes(b.pool_pages, 8448)?, bytes(b.rows, (2 * h + 1) * 256)?])
    }
    fn oai(self, b: OaiBatch) -> Result<[Buffer; 3]> {
        let Kind::Oai { width } = self.kind else { return Err(invalid("not an OAI producer")); };
        let n = bytes(b.rows, width as usize * 2)?;
        self.validate(b.rows, [b.gate, b.up, b.output], [n; 3])
    }
}

pub fn validate_producer_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<ProducerLayout> {
    let layout = ProducerLayout::from_spec(spec)?;
    let operands = layout.operands();
    let pointers = entry["pointers"].as_array().ok_or_else(|| invalid("producer pointer metadata"))?;
    let scalars = entry["scalars"].as_array().ok_or_else(|| invalid("producer scalar metadata"))?;
    if entry["op"] != layout.kind.op() || pointers.len() != operands.len()
        || pointers.iter().zip(operands).any(|(a, e)| a["name"] != e.name || a["dtype"] != e.dtype
            || a["shape"] != e.shape || a["role"] != e.role)
        || scalars.len() != 1 || scalars[0]["name"] != "rows" || scalars[0]["type"] != "int32" {
        return Err(invalid("producer exported operand contract"));
    }
    Ok(layout)
}

#[derive(Clone, Copy)]
pub struct NormRopeBatch {
    pub rows: usize,
    pub positions_capacity: usize,
    pub input: Buffer,
    pub weight: Buffer,
    pub positions: Buffer,
    pub cos_sin: Buffer,
    pub output: Buffer,
    pub scratch: Buffer,
}
#[derive(Clone, Copy)]
pub struct IndexProducerBatch {
    pub rows: usize,
    pub positions_capacity: usize,
    pub pool_pages: usize,
    pub query: Buffer,
    pub key: Buffer,
    pub query_norm: Buffer,
    pub key_norm: Buffer,
    pub positions: Buffer,
    pub slots: Buffer,
    pub cos_sin: Buffer,
    pub query_fp8: Buffer,
    pub query_scale: Buffer,
    pub key_fp8: Buffer,
    pub key_scale: Buffer,
    pub cache: Buffer,
    pub scratch: Buffer,
}
#[derive(Clone, Copy)]
pub struct OaiBatch {
    pub rows: usize,
    pub gate: Buffer,
    pub up: Buffer,
    pub output: Buffer,
}

pub struct Producer<'a> {
    program: Program<'a>,
    pub layout: ProducerLayout,
}
impl<'a> Producer<'a> {
    pub fn bind(programs: &'a Programs<'a>, kind: Kind) -> Result<Self> {
        let spec = programs.spec(kind.name()).map_err(|e| invalid(e.to_string()))?;
        let layout = ProducerLayout::from_spec(spec)?;
        let names = layout.operands().iter().map(|o| o.name).collect::<Vec<_>>();
        let program = programs.program(layout.kind.name(), &names).map_err(|e| invalid(e.to_string()))?;
        Ok(Self { program, layout })
    }
    /// # Safety
    /// Device positions must address the cosine/sine table. Retain storage and
    /// stream through all queued work and graphs using these pointers.
    pub unsafe fn norm_rope(&self, batch: NormRopeBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.norm(batch)?;
        // SAFETY: checked host extents/devices/aliases; device indices and lifetimes are caller-owned.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Positions must address the table, nonnegative slots must address the cache
    /// and be unique within this batch; negative slots skip the cache write.
    /// Retain all storage through queued work and graph use.
    pub unsafe fn index(&self, batch: IndexProducerBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.index(batch)?;
        // SAFETY: checked host extents/devices/aliases; device indices and lifetimes are caller-owned.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Buffers and stream must remain live until queued work and graph use drain.
    pub unsafe fn oai(&self, batch: OaiBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.oai(batch)?;
        // SAFETY: checked host extents/devices/aliases; lifetimes are caller-owned.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn buffer(i: usize) -> Buffer {
        Buffer { ptr: ((i + 1) * 0x100000000usize) as *mut c_void, bytes: 1 << 30, device_id: 0, flags: 0 }
    }
    #[test]
    fn producers_match_all_pinned_export_contracts() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!("../../../tests/fixtures/minimax_producer_programs.json")).unwrap();
        for (kind, entry) in Kind::ALL.into_iter().zip(fixture["programs"].as_array().unwrap()) {
            let spec = ProgramSpec {
                name: entry["name"].as_str().unwrap().into(),
                pointers: entry["pointers"].as_array().unwrap().iter().map(|p| p["name"].as_str().unwrap().into()).collect(),
                scalar_kinds: b"i".to_vec(),
                capacity_rows: entry["capacity_rows"].as_u64().unwrap() as u32,
                geometry: entry["geometry"].clone(),
                scratch: entry["scratch_bytes_at_capacity"].as_object().unwrap().iter()
                    .map(|(k,v)| (k.clone(), v.as_u64().unwrap())).collect(),
            };
            let layout = validate_producer_entry(&spec, entry).unwrap();
            assert_eq!(layout.kind, kind);
            for field in ["name", "dtype", "shape", "role"] {
                let mut bad = entry.clone(); bad["pointers"][0][field] = serde_json::json!("wrong");
                assert!(validate_producer_entry(&spec, &bad).is_err());
            }
            let mut bad = spec.clone(); bad.scalar_kinds = b"l".to_vec();
            assert!(ProducerLayout::from_spec(&bad).is_err());
            let mut bad = spec.clone(); bad.geometry = serde_json::json!({});
            assert!(ProducerLayout::from_spec(&bad).is_err());
            let mut bad = spec.clone(); bad.scratch.insert("scratch".into(), layout.scratch() + 16);
            assert!(ProducerLayout::from_spec(&bad).is_err());
        }
    }
    #[test]
    fn narrow_producers_preserve_replicated_keys_and_validate_exact_compiled_contracts() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/minimax_producer_programs.json")).unwrap();
        for (kind, source) in Kind::HEADS.into_iter().zip(fixture["programs"].as_array().unwrap()) {
            let layout = ProducerLayout::new(kind, 4096).unwrap();
            let mut entry = source.clone();
            entry["name"] = serde_json::json!(kind.name());
            match kind {
                Kind::NormRope { heads } => entry["geometry"]["heads"] = heads.into(),
                Kind::IndexHeads => entry["geometry"]["heads"] = 2.into(),
                Kind::Oai { width } => entry["geometry"]["width"] = width.into(),
                _ => unreachable!(),
            }
            entry["pointers"] = serde_json::json!(layout.operands().iter().map(|o|
                serde_json::json!({"name":o.name,"dtype":o.dtype,"shape":o.shape,"role":o.role}))
                .collect::<Vec<_>>());
            let spec = ProgramSpec { name: kind.name().into(),
                pointers: layout.operands().iter().map(|o| o.name.into()).collect(), scalar_kinds: b"i".to_vec(),
                capacity_rows: 4096, geometry: entry["geometry"].clone(),
                scratch: if layout.scratch() == 0 { Default::default() }
                    else { [("scratch".into(), layout.scratch())].into_iter().collect() } };
            assert_eq!(validate_producer_entry(&spec, &entry).unwrap(), layout);
            let mut bad = spec.clone(); bad.name = source["name"].as_str().unwrap().into();
            assert!(validate_producer_entry(&bad, &entry).is_err());
            for field in ["dtype", "shape", "role"] {
                let mut bad = entry.clone(); bad["pointers"][0][field] = serde_json::json!("wrong");
                assert!(validate_producer_entry(&spec, &bad).is_err());
            }
        }
        assert_eq!(ProducerLayout::new(Kind::IndexHeads, 4096).unwrap().scratch(), 4096 * 1280);
        assert_ne!(1280, ProducerLayout::new(Kind::Index, 1).unwrap().scratch() / 2);
        for heads in [32, 2] {
            let layout = ProducerLayout::new(Kind::NormRope { heads }, 4096).unwrap();
            let n = 17 * heads as usize * 256;
            let b = NormRopeBatch { rows: 17, positions_capacity: 2049,
                input: Buffer { bytes: n, ..buffer(0) }, weight: buffer(1), positions: buffer(2),
                cos_sin: buffer(3), output: Buffer { bytes: n, ..buffer(4) },
                scratch: Buffer { bytes: n, ..buffer(5) } };
            assert!(layout.norm(b).is_ok());
            assert!(layout.norm(NormRopeBatch { output: Buffer { bytes: n - 1, ..b.output }, ..b }).is_err());
        }
        let layout = ProducerLayout::new(Kind::IndexHeads, 4096).unwrap();
        for rows in [1, 63, 64, 2049, 4096] {
            let b = IndexProducerBatch { rows, positions_capacity: 2049, pool_pages: 3,
                query: Buffer { bytes: rows * 512, ..buffer(0) }, key: Buffer { bytes: rows * 256, ..buffer(1) },
                query_norm: buffer(2), key_norm: buffer(3), positions: buffer(4), slots: buffer(5),
                cos_sin: buffer(6), query_fp8: Buffer { bytes: rows * 256, ..buffer(7) },
                query_scale: Buffer { bytes: rows * 8, ..buffer(8) },
                key_fp8: Buffer { bytes: rows * 128, ..buffer(9) },
                key_scale: Buffer { bytes: rows * 4, ..buffer(10) },
                cache: Buffer { bytes: 3 * 8448, ..buffer(11) },
                scratch: Buffer { bytes: rows * 1280, ..buffer(12) } };
            assert!(layout.index(b).is_ok());
            assert!(ProducerLayout::new(Kind::Index, 4096).unwrap().index(b).is_err());
            assert!(layout.index(IndexProducerBatch { query_scale: Buffer {
                ptr: (b.query_scale.ptr as usize + 8) as *mut c_void, ..b.query_scale }, ..b }).is_err());
            assert!(layout.index(IndexProducerBatch { scratch: Buffer { bytes: rows * 1280 - 1, ..b.scratch }, ..b }).is_err());
        }
        for width in [1536, 6144] {
            let layout = ProducerLayout::new(Kind::Oai { width }, 4096).unwrap();
            let n = 17 * width as usize * 2;
            let b = OaiBatch { rows: 17, gate: Buffer { bytes: n, ..buffer(0) },
                up: Buffer { bytes: n, ..buffer(1) }, output: Buffer { bytes: n, ..buffer(2) } };
            assert!(layout.oai(b).is_ok());
            assert!(layout.oai(OaiBatch { up: Buffer { bytes: n - 1, ..b.up }, ..b }).is_err());
        }
    }
    #[test]
    fn producer_buffers_reject_alias_device_extent_and_capacity_errors() {
        let layout = ProducerLayout::new(Kind::NormRope { heads: 64 }, 4096).unwrap();
        let b = NormRopeBatch { rows: 17, positions_capacity: 2049, input: buffer(0), weight: buffer(1),
            positions: buffer(2), cos_sin: buffer(3), output: buffer(4), scratch: buffer(5) };
        assert!(layout.norm(b).is_ok());
        assert!(layout.norm(NormRopeBatch { rows: 4097, ..b }).is_err());
        assert!(layout.norm(NormRopeBatch { positions_capacity: 0, ..b }).is_err());
        assert!(layout.norm(NormRopeBatch { output: b.input, ..b }).is_err());
        assert!(layout.norm(NormRopeBatch { scratch: Buffer { bytes: 1, ..b.scratch }, ..b }).is_err());
        assert!(layout.norm(NormRopeBatch { positions: Buffer { device_id: 1, ..b.positions }, ..b }).is_err());
        assert!(layout.norm(NormRopeBatch { positions_capacity: usize::MAX, ..b }).is_err());
        let layout = ProducerLayout::new(Kind::Index, 4096).unwrap();
        let b = IndexProducerBatch { rows: 17, positions_capacity: 2049, pool_pages: 300000,
            query: buffer(0), key: buffer(1), query_norm: buffer(2), key_norm: buffer(3), positions: buffer(4),
            slots: buffer(5), cos_sin: buffer(6), query_fp8: buffer(7), query_scale: buffer(8),
            key_fp8: buffer(9), key_scale: buffer(10), cache: Buffer { bytes: 300000 * 8448, ..buffer(11) }, scratch: buffer(12) };
        assert!(layout.index(b).is_ok());
        assert!(layout.index(IndexProducerBatch { cache: buffer(11), ..b }).is_err());
        assert!(layout.index(IndexProducerBatch { scratch: b.query, ..b }).is_err());
        assert!(layout.index(IndexProducerBatch { pool_pages: 0, ..b }).is_err());
        assert!(layout.index(IndexProducerBatch { slots: Buffer { ptr: 17usize as *mut c_void, ..b.slots }, ..b }).is_err());
        for width in [3072, 12288] {
            let layout = ProducerLayout::new(Kind::Oai { width }, 4096).unwrap();
            let b = OaiBatch { rows: 1, gate: buffer(0), up: buffer(1), output: buffer(2) };
            assert!(layout.oai(b).is_ok());
            assert!(layout.oai(OaiBatch { output: b.gate, ..b }).is_err());
            assert!(layout.oai(OaiBatch { rows: 0, ..b }).is_err());
        }
        assert!(ProducerLayout::new(Kind::NormRope { heads: 1 }, 64).is_err());
    }
}
