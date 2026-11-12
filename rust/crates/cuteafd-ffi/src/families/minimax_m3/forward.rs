//! Checked launch contracts for the native MiniMax residual/router/KV producers.
use super::{bytes, invalid, validate_buffers, Result};
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind { ResidualNorm, RouterScores, KvPack }
impl Kind {
    pub const ALL: [Self; 3] = [Self::ResidualNorm, Self::RouterScores, Self::KvPack];
    pub fn name(self) -> &'static str {
        match self {
            Self::ResidualNorm => "minimax_residual_norm",
            Self::RouterScores => "minimax_router_scores",
            Self::KvPack => "minimax_kv_pack",
        }
    }
    fn names(self) -> &'static [&'static str] {
        match self {
            Self::ResidualNorm => &["x", "weight", "bias", "cos", "sin", "positions", "out"],
            Self::RouterScores => &["x", "weight", "logits"],
            Self::KvPack => &["k", "v", "slots", "batch_ids", "k_descale", "v_descale", "key_cache", "value_cache"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardLayout { pub kind: Kind, pub rows: u32 }
impl ForwardLayout {
    pub fn from_spec(spec: &ProgramSpec) -> Result<Self> {
        let kind = Kind::ALL.into_iter().find(|k| spec.name == k.name())
            .ok_or_else(|| invalid("unknown forward producer"))?;
        let g = &spec.geometry;
        let geometry = match kind {
            Kind::ResidualNorm => g["width"] == 6144 && g["heads"] == 1
                && g["eps"].as_f64() == Some(1e-6) && g["zero_centered"] == true
                && g["subtract_mean"] == false && g["bias"] == false && g["rotary"] == 0,
            Kind::RouterScores => g["hidden"] == 6144 && g["experts"] == 128,
            Kind::KvPack => g["heads"] == 4 && g["head_dim"] == 128 && g["page_rows"] == 64,
        };
        if !geometry || spec.capacity_rows == 0 || spec.capacity_rows > 4096
            || spec.scalar_kinds != b"i" || !spec.scratch.is_empty()
            || spec.pointers.iter().map(String::as_str).ne(kind.names().iter().copied()) {
            return Err(invalid("forward producer geometry, capacity or ABI"));
        }
        Ok(Self { kind, rows: spec.capacity_rows })
    }
    fn validate<const N: usize>(self, rows: usize, items: [(Buffer, usize); N],
        align: [usize; N], writable: &[usize]) -> Result<[Buffer; N]> {
        if rows == 0 || rows > self.rows as usize { return Err(invalid("forward producer live rows")); }
        validate_buffers(&items, &align, writable).map_err(invalid)?;
        Ok(items.map(|(buffer, _)| buffer))
    }
    fn norm(self, b: NormBatch) -> Result<[Buffer; 7]> {
        if self.kind != Kind::ResidualNorm { return Err(invalid("not a residual norm")); }
        // The shared affine norm ABI retains unused operands. Reuse the weight
        // for them: the specialization never reads bias/tables/positions.
        self.validate(b.rows, [(b.input, bytes(b.rows, 6144 * 4)?), (b.weight, 6144 * 2),
            (b.weight, 0), (b.weight, 0), (b.weight, 0), (b.weight, 0),
            (b.output, bytes(b.rows, 6144 * 2)?)], [16; 7], &[6])
    }
    fn router(self, b: RouterBatch) -> Result<[Buffer; 3]> {
        if self.kind != Kind::RouterScores { return Err(invalid("not router scores")); }
        self.validate(b.rows, [(b.input, bytes(b.rows, 6144 * 2)?), (b.weight, 128 * 6144 * 4),
            (b.output, bytes(b.rows, 128 * 4)?)], [16; 3], &[2])
    }
    fn kv(self, b: KvBatch) -> Result<[Buffer; 8]> {
        if self.kind != Kind::KvPack || b.batch == 0 || b.pool_pages == 0 {
            return Err(invalid("KV producer geometry"));
        }
        self.validate(b.rows, [(b.key, bytes(b.rows, 1024)?), (b.value, bytes(b.rows, 1024)?),
            (b.slots, bytes(b.rows, 8)?), (b.batch_ids, bytes(b.rows, 4)?),
            (b.key_descale, bytes(b.batch, 16)?), (b.value_descale, bytes(b.batch, 16)?),
            (b.key_cache, bytes(b.pool_pages, 64 * 512)?),
            (b.value_cache, bytes(b.pool_pages, 64 * 512)?)], [16; 8], &[6, 7])
    }
}

pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<ForwardLayout> {
    let layout = ForwardLayout::from_spec(spec)?;
    let operands: &[(&str, &str, &str)] = match layout.kind {
        Kind::ResidualNorm => &[("float32", "[rows,1,6144]", "in"), ("bfloat16", "[6144]", "in"),
            ("bfloat16", "[6144] (ignored unless bias)", "in"),
            ("bfloat16", "[positions,0] (ignored unless rotary)", "in"),
            ("bfloat16", "[positions,0] (ignored unless rotary)", "in"),
            ("int64", "[rows] (ignored unless rotary)", "in"), ("bfloat16", "[rows,1,6144]", "out")],
        Kind::RouterScores => &[("bfloat16", "[rows,6144]", "in"),
            ("float32", "[128,6144]", "in"), ("float32", "[rows,128]", "out")],
        Kind::KvPack => &[("bfloat16", "[rows,4,128]", "in"), ("bfloat16", "[rows,4,128]", "in"),
            ("int64", "[rows] (physical slots; negative skips writes)", "in"), ("int32", "[rows]", "in"),
            ("float32", "[batch,4]", "in"), ("float32", "[batch,4]", "in"),
            ("uint8", "[pages,64,4,128]", "out"), ("uint8", "[pages,64,4,128]", "out")],
    };
    let pointers = entry["pointers"].as_array().ok_or_else(|| invalid("forward pointer metadata"))?;
    if entry["op"] != layout.kind.name() || pointers.len() != operands.len()
        || pointers.iter().zip(layout.kind.names().iter().zip(operands))
            .any(|(p, (name, (dtype, shape, role)))| p["name"] != *name || p["dtype"] != *dtype
                || p["shape"] != *shape || p["role"] != *role)
        || entry["scalars"] != serde_json::json!([{"name":"rows","type":"int32"}]) {
        return Err(invalid("forward exported operand or operation contract"));
    }
    Ok(layout)
}

#[derive(Clone, Copy)]
pub struct NormBatch { pub rows: usize, pub input: Buffer, pub weight: Buffer, pub output: Buffer }
#[derive(Clone, Copy)]
pub struct RouterBatch { pub rows: usize, pub input: Buffer, pub weight: Buffer, pub output: Buffer }
#[derive(Clone, Copy)]
pub struct KvBatch {
    pub rows: usize,
    pub batch: usize,
    pub pool_pages: usize,
    pub key: Buffer,
    pub value: Buffer,
    pub slots: Buffer,
    pub batch_ids: Buffer,
    pub key_descale: Buffer,
    pub value_descale: Buffer,
    pub key_cache: Buffer,
    pub value_cache: Buffer,
}

pub struct ForwardProducer<'a> { program: Program<'a>, pub layout: ForwardLayout }
impl<'a> ForwardProducer<'a> {
    pub fn bind(programs: &'a Programs<'a>, kind: Kind) -> Result<Self> {
        let spec = programs.spec(kind.name()).map_err(|e| invalid(e.to_string()))?;
        let layout = ForwardLayout::from_spec(spec)?;
        let program = programs.program(kind.name(), kind.names()).map_err(|e| invalid(e.to_string()))?;
        Ok(Self { program, layout })
    }
    /// # Safety
    /// Storage and stream must remain live through queued work and graph use.
    pub unsafe fn norm(&self, batch: NormBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.norm(batch)?;
        // SAFETY: host extents/devices/aliases checked; caller retains storage.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Storage and stream must remain live through queued work and graph use.
    pub unsafe fn router(&self, batch: RouterBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.router(batch)?;
        // SAFETY: host extents/devices/aliases checked; caller retains storage.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Nonnegative slots must be unique and address the cache; batch IDs must
    /// address positive finite descales matching attention. Retain all storage
    /// and stream through queued work and graph use. Negative slots skip writes.
    pub unsafe fn kv(&self, batch: KvBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.kv(batch)?;
        // SAFETY: host extents/devices/aliases checked; device indices and lifetimes are caller-owned.
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
    fn spec(kind: Kind) -> ProgramSpec {
        let geometry = match kind {
            Kind::ResidualNorm => serde_json::json!({"width":6144,"heads":1,"eps":1e-6,
                "zero_centered":true,"subtract_mean":false,"bias":false,"rotary":0}),
            Kind::RouterScores => serde_json::json!({"hidden":6144,"experts":128}),
            Kind::KvPack => serde_json::json!({"heads":4,"head_dim":128,"page_rows":64}),
        };
        ProgramSpec { name: kind.name().into(), pointers: kind.names().iter().map(|s| (*s).into()).collect(),
            scalar_kinds: b"i".to_vec(), scratch: Default::default(), capacity_rows: 4096, geometry }
    }
    #[test]
    fn accepts_exact_specs_rejects_wrong_abi_geometry_and_capacity() {
        for kind in Kind::ALL {
            let expected = spec(kind);
            assert_eq!(ForwardLayout::from_spec(&expected).unwrap(), ForwardLayout { kind, rows: 4096 });
            let mut wrong = expected.clone(); wrong.pointers.swap(0, 1);
            assert!(ForwardLayout::from_spec(&wrong).is_err());
            let mut wrong = expected.clone(); wrong.scalar_kinds = b"q".to_vec();
            assert!(ForwardLayout::from_spec(&wrong).is_err());
            let mut wrong = expected.clone(); wrong.capacity_rows = 4097;
            assert!(ForwardLayout::from_spec(&wrong).is_err());
            let mut wrong = expected; wrong.geometry = serde_json::json!({});
            assert!(ForwardLayout::from_spec(&wrong).is_err());
        }
    }
    #[test]
    fn validates_forward_manifest_operands_before_loading() {
        for kind in Kind::ALL {
            let spec = spec(kind);
            // The exporter ABI is independently pinned by the fork's GPU oracles.
            let operands = match kind {
                Kind::ResidualNorm => vec![("float32", "[rows,1,6144]"), ("bfloat16", "[6144]"),
                    ("bfloat16", "[6144] (ignored unless bias)"),
                    ("bfloat16", "[positions,0] (ignored unless rotary)"),
                    ("bfloat16", "[positions,0] (ignored unless rotary)"),
                    ("int64", "[rows] (ignored unless rotary)"), ("bfloat16", "[rows,1,6144]")],
                Kind::RouterScores => vec![("bfloat16", "[rows,6144]"),
                    ("float32", "[128,6144]"), ("float32", "[rows,128]")],
                Kind::KvPack => vec![("bfloat16", "[rows,4,128]"), ("bfloat16", "[rows,4,128]"),
                    ("int64", "[rows] (physical slots; negative skips writes)"), ("int32", "[rows]"),
                    ("float32", "[batch,4]"), ("float32", "[batch,4]"),
                    ("uint8", "[pages,64,4,128]"), ("uint8", "[pages,64,4,128]")],
            };
            let entry = serde_json::json!({"op":kind.name(),
                "scalars":[{"name":"rows","type":"int32"}],
                "pointers":kind.names().iter().zip(operands).enumerate().map(|(i,(n,(d,s)))|
                    serde_json::json!({"name":n,"dtype":d,"shape":s,
                        "role":if i + 1 == kind.names().len() || kind == Kind::KvPack && i == 6 {"out"} else {"in"}}))
                    .collect::<Vec<_>>()});
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
            assert_eq!(ForwardProducer::bind(&programs, kind).unwrap().layout, ForwardLayout { kind, rows: 4096 });
            for field in ["name", "dtype", "shape", "role"] {
                let mut wrong = entry.clone(); wrong["pointers"][0][field] = "wrong".into();
                assert!(validate_entry(&spec, &wrong).is_err());
            }
            let mut wrong = entry; wrong["scalars"][0]["type"] = "int64".into();
            assert!(validate_entry(&spec, &wrong).is_err());
        }
    }
    #[test]
    fn checks_native_operand_extents_devices_and_writable_aliases() {
        let norm = ForwardLayout::from_spec(&spec(Kind::ResidualNorm)).unwrap();
        let mut n = NormBatch { rows: 3, input: buffer(0), weight: buffer(1), output: buffer(2) };
        assert!(norm.norm(n).is_ok());
        n.output.bytes = 3 * 6144 * 2 - 1; assert!(norm.norm(n).is_err());
        n.output = n.input; assert!(norm.norm(n).is_err());
        let router = ForwardLayout::from_spec(&spec(Kind::RouterScores)).unwrap();
        let mut r = RouterBatch { rows: 1, input: buffer(0), weight: buffer(1), output: buffer(2) };
        assert!(router.router(r).is_ok());
        r.weight.bytes = 128 * 6144 * 2; assert!(router.router(r).is_err());
        let kv = ForwardLayout::from_spec(&spec(Kind::KvPack)).unwrap();
        let mut k = KvBatch { rows: 7, batch: 3, pool_pages: 11, key: buffer(0), value: buffer(1),
            slots: buffer(2), batch_ids: buffer(3), key_descale: buffer(4), value_descale: buffer(5),
            key_cache: buffer(6), value_cache: buffer(7) };
        assert!(kv.kv(k).is_ok());
        k.batch_ids.device_id = 1; assert!(kv.kv(k).is_err());
        k.batch_ids.device_id = 0; k.value_cache = k.key_cache; assert!(kv.kv(k).is_err());
        k.value_cache = buffer(7); k.pool_pages = usize::MAX; assert!(kv.kv(k).is_err());
        k.pool_pages = 11; k.rows = 4097; assert!(kv.kv(k).is_err());
    }
}
