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
            (b.output, bytes(b.rows, 6144 * 2)?)], [4, 2, 2, 2, 2, 8, 16], &[6])
    }
    fn router(self, b: RouterBatch) -> Result<[Buffer; 3]> {
        if self.kind != Kind::RouterScores { return Err(invalid("not router scores")); }
        self.validate(b.rows, [(b.input, bytes(b.rows, 6144 * 2)?), (b.weight, 128 * 6144 * 4),
            (b.output, bytes(b.rows, 128 * 4)?)], [16, 4, 4], &[2])
    }
    fn kv(self, b: KvBatch) -> Result<[Buffer; 8]> {
        if self.kind != Kind::KvPack || b.batch == 0 || b.pool_pages == 0 {
            return Err(invalid("KV producer geometry"));
        }
        self.validate(b.rows, [(b.key, bytes(b.rows, 1024)?), (b.value, bytes(b.rows, 1024)?),
            (b.slots, bytes(b.rows, 8)?), (b.batch_ids, bytes(b.rows, 4)?),
            (b.key_descale, bytes(b.batch, 16)?), (b.value_descale, bytes(b.batch, 16)?),
            (b.key_cache, bytes(b.pool_pages, 64 * 512)?),
            (b.value_cache, bytes(b.pool_pages, 64 * 512)?)], [16, 16, 8, 4, 4, 4, 16, 16], &[6, 7])
    }
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
