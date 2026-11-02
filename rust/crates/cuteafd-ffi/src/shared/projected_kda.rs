//! Fully projected KDA recurrence. Projection formats belong to the caller;
//! this seam owns the unchanged convolution, state, normalization and replay ABI.
use super::programs::{Program, ProgramSpec, Programs, Scalar};
use crate::CuteafdDeviceBuffer as Buffer;
use std::ffi::c_void;

#[derive(Debug, thiserror::Error)]
pub enum KdaError {
    #[error("invalid projected KDA: {0}")]
    Invalid(&'static str),
    #[error("projected KDA program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, KdaError>;
const POINTERS: &[&str] = &[
    "q",
    "k",
    "v",
    "decay",
    "gate",
    "beta",
    "conv_w",
    "a_log",
    "dt_bias",
    "o_norm",
    "conv_state",
    "state",
    "slots",
    "seq_first",
    "out",
    "replay",
    "scratch",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub heads: usize,
    pub capacity: usize,
    pub decode: bool,
    pub scratch: usize,
}
impl Layout {
    pub fn width(self) -> usize {
        self.heads * 128
    }
    pub fn conv_state_bytes(self, slots: usize) -> Option<usize> {
        slots.checked_mul(3 * 3 * self.width() * 2)
    }
    pub fn state_bytes(self, slots: usize) -> Option<usize> {
        slots.checked_mul(self.heads * 128 * 128 * 4)
    }
    pub fn replay_bytes(self) -> usize {
        let rows = if self.decode {
            self.capacity.max(64)
        } else {
            64
        };
        rows * self.heads * (3 * 128 * 4 + 4 + 3 * 128 * 2)
    }
    fn pointer_contract(self) -> Vec<(&'static str, String, &'static str)> {
        let h = self.heads;
        let d = self.width();
        let mut p = vec![("bfloat16", format!("[rows,{d}]"), "in"); 5];
        p.extend([
            ("bfloat16", format!("[rows,{h}]"), "in"),
            ("float32", format!("[{},4]", 3 * d), "in"),
            ("float32", format!("[{h}]"), "in"),
            ("float32", format!("[{d}]"), "in"),
            ("bfloat16", "[128]".into(), "in"),
            ("bfloat16", format!("[slots,3,{}]", 3 * d), "inout"),
            ("float32", format!("[slots,{h},128,128]"), "inout"),
            ("int32", "[rows]".into(), "in"),
            ("int32", "[rows]".into(), "in"),
            ("bfloat16", format!("[rows,{d}]"), "out"),
            ("float32", format!("[{}]", self.replay_bytes() / 4), "inout"),
            ("uint8", "[projected_kda_scratch_bytes]".into(), "scratch"),
        ]);
        p
    }
}

fn layout(spec: &ProgramSpec) -> Result<Layout> {
    let (suffix, heads) = if let Some(s) = spec.name.strip_prefix("glmf_stage_kda_projected_m") {
        (s, 64)
    } else if let Some(s) = spec.name.strip_prefix("glmf2_stage_kda_projected_m") {
        (s, 32)
    } else {
        return Err(KdaError::Invalid("unknown projected table"));
    };
    let capacity: usize = suffix.parse().map_err(|_| KdaError::Invalid("capacity"))?;
    if !matches!(capacity, 64 | 128 | 4096) || spec.capacity_rows as usize != capacity {
        return Err(KdaError::Invalid("capacity disagrees with table"));
    }
    let decode = capacity <= 128;
    let g = &spec.geometry;
    if [
        ("hidden", 4096),
        ("kda_heads", heads),
        ("kda_head_dim", 128),
        ("conv_kernel", 4),
        ("max_rows", capacity),
    ]
    .iter()
    .any(|&(k, v)| g[k].as_u64() != Some(v as u64))
        || g["gate_lower_bound"].as_f64() != Some(-5.)
        || g["norm_eps"].as_f64() != Some(1e-5)
        || g["output_kind"].as_str() != Some("norm")
        || g["projected_inputs"].as_bool() != Some(true)
        || g["decode"].as_bool() != Some(decode)
        || !g["state_dtype"].is_null()
        || !g["state_rounding"].is_null()
        || !g["output_gate"].is_null()
        || !g["o_norm_dtype"].is_null()
        || g["replay_rows"].as_u64() != if capacity == 128 { Some(128) } else { None }
        || spec.scalar_kinds != b"ii"
        || spec
            .pointers
            .iter()
            .map(String::as_str)
            .ne(POINTERS.iter().copied())
    {
        return Err(KdaError::Invalid("geometry, arithmetic or ABI"));
    }
    let scratch = spec
        .scratch
        .get("scratch")
        .and_then(|&v| usize::try_from(v).ok())
        .ok_or(KdaError::Invalid("missing scratch"))?;
    let align = |v: usize| v.div_ceil(1024) * 1024;
    let d = heads * 128;
    let p = 3 * d + 2 * 128 + heads;
    // The pack aliases proj/fg in the original body; it needs no second prefix.
    let base = align(capacity * p * 2)
        + align(capacity * 2 * d * 2)
        + align(capacity * 3 * d * 2)
        + 2 * align(capacity * d * 2);
    // Prefill appends the exported chunked recurrence workspace. Its size is a
    // kernel contract, not a host reimplementation of the recurrence planner.
    if spec.scratch.len() != 1
        || scratch % 1024 != 0
        || (decode && scratch != base)
        || (!decode && scratch <= base)
    {
        return Err(KdaError::Invalid("scratch layout"));
    }
    Ok(Layout {
        heads,
        capacity,
        decode,
        scratch,
    })
}

/// Validate the complete metadata before binding or allocating workspaces.
pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<Layout> {
    let l = layout(spec)?;
    let scalars = entry["scalars"]
        .as_array()
        .ok_or(KdaError::Invalid("scalar metadata"))?;
    if entry["op"] != "kda_core"
        || scalars.len() != 2
        || scalars
            .iter()
            .zip(["rows", "spec"])
            .any(|(s, n)| s["name"] != n || s["type"] != "int32")
    {
        return Err(KdaError::Invalid("scalar ABI or operation"));
    }
    let pointers = entry["pointers"]
        .as_array()
        .ok_or(KdaError::Invalid("pointer metadata"))?;
    if pointers.len() != POINTERS.len()
        || pointers.iter().zip(POINTERS).zip(l.pointer_contract()).any(
            |((p, n), (dtype, shape, role))| {
                p["name"] != *n || p["dtype"] != dtype || p["shape"] != shape || p["role"] != role
            },
        )
    {
        return Err(KdaError::Invalid("pointer dtype, shape or access"));
    }
    Ok(l)
}

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub convolution: Buffer,
    pub a_log: Buffer,
    pub dt_bias: Buffer,
    pub norm: Buffer,
}
#[derive(Debug, Clone, Copy)]
pub struct Batch {
    pub rows: usize,
    pub sequences: usize,
    pub state_slots: usize,
    pub speculative: bool,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub decay: Buffer,
    pub gate: Buffer,
    pub beta: Buffer,
    pub conv_state: Buffer,
    pub state: Buffer,
    pub slots: Buffer,
    pub first: Buffer,
    pub output: Buffer,
    pub replay: Buffer,
    pub scratch: Buffer,
}

pub struct ProjectedKda<'a> {
    program: Program<'a>,
    pub layout: Layout,
}
impl<'a> ProjectedKda<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let layout = layout(programs.spec(name)?)?;
        Ok(Self {
            program: programs.program(name, POINTERS)?,
            layout,
        })
    }

    /// # Safety
    /// Sequence slots/first offsets must describe valid, nonoverlapping sequences
    /// in the admitted states. Retain all allocations and stream until completion.
    pub unsafe fn launch(&self, w: Weights, b: Batch, stream: *mut c_void) -> Result<()> {
        let items = self.layout.operands(w, b)?;
        // SAFETY: extents/devices and disjoint writers are checked below; the
        // caller owns sequence metadata and queued-work/graph lifetimes.
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
    fn operands(self, w: Weights, b: Batch) -> Result<[(Buffer, usize); 17]> {
        if b.rows == 0
            || b.rows > self.capacity
            || b.sequences == 0
            || b.sequences > b.rows
            || b.sequences > b.state_slots
            || b.state_slots > i32::MAX as usize
            || (!self.decode && (b.speculative || (b.rows > 64 && b.sequences != 1)))
        {
            return Err(KdaError::Invalid("batch/replay geometry"));
        }
        let d = self.width();
        let items = [
            (b.q, b.rows * d * 2),
            (b.k, b.rows * d * 2),
            (b.v, b.rows * d * 2),
            (b.decay, b.rows * d * 2),
            (b.gate, b.rows * d * 2),
            (b.beta, b.rows * self.heads * 2),
            (w.convolution, 3 * d * 4 * 4),
            (w.a_log, self.heads * 4),
            (w.dt_bias, d * 4),
            (w.norm, 128 * 2),
            (
                b.conv_state,
                self.conv_state_bytes(b.state_slots)
                    .ok_or(KdaError::Invalid("conv slots overflow"))?,
            ),
            (
                b.state,
                self.state_bytes(b.state_slots)
                    .ok_or(KdaError::Invalid("state slots overflow"))?,
            ),
            (b.slots, b.rows * 4),
            (b.first, b.rows * 4),
            (b.output, b.rows * d * 2),
            (b.replay, self.replay_bytes()),
            (b.scratch, self.scratch),
        ];
        let device = items[0].0.device_id;
        if device < 0
            || items.iter().any(|(b, n)| {
                b.ptr.is_null()
                    || b.bytes < *n
                    || b.flags != 0
                    || b.device_id != device
                    || b.ptr as usize % 16 != 0
                    || (b.ptr as usize).checked_add(*n).is_none()
            })
        {
            return Err(KdaError::Invalid("buffer extent/alignment/device"));
        }
        for i in [10, 11, 14, 15, 16] {
            let (a, na) = items[i];
            if items.iter().enumerate().any(|(j, (b, nb))| {
                i != j
                    && (a.ptr as usize) < b.ptr as usize + nb
                    && (b.ptr as usize) < a.ptr as usize + na
            }) {
                return Err(KdaError::Invalid("overlapping writable operands"));
            }
        }
        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn fixture(heads: usize, capacity: usize) -> (ProgramSpec, serde_json::Value) {
        let decode = capacity <= 128;
        let align = |v: usize| v.div_ceil(1024) * 1024;
        let d = heads * 128;
        let scratch = align(capacity * (3 * d + 256 + heads) * 2)
            + align(capacity * 4 * d)
            + align(capacity * 6 * d)
            + 2 * align(capacity * 2 * d)
            + if decode { 0 } else { 1024 };
        let l = Layout {
            heads,
            capacity,
            decode,
            scratch,
        };
        let spec = ProgramSpec {
            name: format!(
                "{}_stage_kda_projected_m{capacity}",
                if heads == 64 { "glmf" } else { "glmf2" }
            ),
            capacity_rows: capacity as u32,
            scalar_kinds: b"ii".to_vec(),
            pointers: POINTERS.iter().map(|s| s.to_string()).collect(),
            geometry: json!({"hidden":4096,"kda_heads":heads,"kda_head_dim":128,"conv_kernel":4,
                "max_rows":capacity,"gate_lower_bound":-5.,"norm_eps":1e-5,"output_kind":"norm",
                "projected_inputs":true,"decode":decode,"replay_rows":if capacity == 128 {Some(128)} else {None}}),
            scratch: [("scratch".into(), scratch as u64)].into(),
        };
        let pointers: Vec<_> = POINTERS.iter().zip(l.pointer_contract()).map(|(name,(dtype,shape,role))|
            json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect();
        (
            spec,
            json!({"op":"kda_core","pointers":pointers,"scalars":[
            {"name":"rows","type":"int32"},{"name":"spec","type":"int32"}]}),
        )
    }
    #[test]
    fn tp_capacities_and_replay() {
        for h in [32, 64] {
            for c in [64, 128, 4096] {
                let (s, e) = fixture(h, c);
                let l = validate_entry(&s, &e).unwrap();
                assert_eq!(l.heads, h);
                assert_eq!(
                    l.replay_bytes(),
                    (if c == 128 { 128 } else { 64 }) * h * 2308
                );
            }
        }
    }
    #[test]
    fn rejects_abi_and_arithmetic_drift() {
        let (s, e) = fixture(32, 128);
        for (key, value) in [
            ("hidden", json!(7168)),
            ("kda_heads", json!(64)),
            ("decode", json!(false)),
            ("replay_rows", json!(64)),
            ("state_dtype", json!("bfloat16")),
            ("output_gate", json!("full_rank")),
            ("projected_inputs", json!(false)),
            ("norm_eps", json!(1e-6)),
        ] {
            let mut bad = s.clone();
            bad.geometry[key] = value;
            assert!(validate_entry(&bad, &e).is_err(), "{key}");
        }
        let mut bad = s.clone();
        bad.scratch
            .insert("scratch".into(), s.scratch["scratch"] + 1024);
        assert!(validate_entry(&bad, &e).is_err());
        let mut bad = s.clone();
        bad.pointers.swap(3, 4);
        assert!(validate_entry(&bad, &e).is_err());
        for (key, value) in [
            ("dtype", json!("float32")),
            ("shape", json!("[rows,8192]")),
            ("role", json!("out")),
        ] {
            let mut bad = e.clone();
            bad["pointers"][0][key] = value;
            assert!(validate_entry(&s, &bad).is_err(), "{key}");
        }
        let mut bad = e;
        bad["scalars"][1]["name"] = json!("mode");
        assert!(validate_entry(&s, &bad).is_err());
    }
    /// Validate real TP1/TP2 exporter output without loading a GPU library.
    #[test]
    #[ignore = "requires the two exported Flash stage ABI listings"]
    fn exported_flash_native_stages_validate() {
        let paths = std::env::var_os("CUTEAFD_FLASH_STAGE_ABIS")
            .expect("CUTEAFD_FLASH_STAGE_ABIS: TP1 and TP2 manifest paths");
        let mut checked = std::collections::BTreeSet::new();
        for path in std::env::split_paths(&paths) {
            let listing: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            for entry in listing["programs"].as_array().unwrap() {
                let name = entry["name"].as_str().unwrap();
                if !name.contains("_kda_projected_")
                    && !name.contains("_absorb_fp8_")
                    && !name.contains("_values_fp8_")
                {
                    continue;
                }
                let spec = ProgramSpec {
                    name: name.into(),
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
                    capacity_rows: entry["capacity_rows"].as_u64().unwrap().try_into().unwrap(),
                    geometry: entry["geometry"].clone(),
                };
                if name.contains("_kda_projected_") {
                    validate_entry(&spec, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
                } else {
                    super::super::glm_stage::validate_entry(&spec, entry)
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                }
                assert!(
                    checked.insert(name.to_owned()),
                    "duplicate exported stage {name}"
                );
            }
        }
        let expected: std::collections::BTreeSet<_> = ["glmf_stage", "glmf2_stage"]
            .into_iter()
            .flat_map(|prefix| {
                ["kda_projected", "absorb_fp8", "values_fp8"]
                    .into_iter()
                    .flat_map(move |op| {
                        [64, 128, 4096]
                            .into_iter()
                            .map(move |rows| format!("{prefix}_{op}_m{rows}"))
                    })
            })
            .collect();
        assert_eq!(checked, expected);
    }

    fn buffers(l: Layout) -> (Weights, Batch) {
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
        let w = Weights {
            convolution: buffer(),
            a_log: buffer(),
            dt_bias: buffer(),
            norm: buffer(),
        };
        let b = Batch {
            rows: 17,
            sequences: 1,
            state_slots: 2,
            speculative: false,
            q: buffer(),
            k: buffer(),
            v: buffer(),
            decay: buffer(),
            gate: buffer(),
            beta: buffer(),
            conv_state: buffer(),
            state: buffer(),
            slots: buffer(),
            first: buffer(),
            output: buffer(),
            replay: buffer(),
            scratch: Buffer {
                bytes: l.scratch,
                ..buffer()
            },
        };
        (w, b)
    }
    #[test]
    fn checks_extents_devices_aliases_and_prefill_sequences() {
        let (s, e) = fixture(64, 128);
        let l = validate_entry(&s, &e).unwrap();
        let (w, b) = buffers(l);
        assert!(l.operands(w, b).is_ok());
        let mut bad = b;
        bad.beta.bytes = 1;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.state.device_id = 1;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.output.ptr = b.q.ptr;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.replay.bytes = l.replay_bytes() / 2;
        assert!(l.operands(w, bad).is_err());
        let mut bad = b;
        bad.state_slots = usize::MAX;
        assert!(l.operands(w, bad).is_err());
        let (s, e) = fixture(32, 4096);
        let l = validate_entry(&s, &e).unwrap();
        let (w, mut b) = buffers(l);
        b.rows = 65;
        b.sequences = 2;
        assert!(l.operands(w, b).is_err());
        b.sequences = 1;
        assert!(l.operands(w, b).is_ok());
        b.speculative = true;
        assert!(l.operands(w, b).is_err());
    }
}
