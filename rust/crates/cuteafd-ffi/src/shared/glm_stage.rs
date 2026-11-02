//! Dormant GLM primitive boundaries; official fused program dispatch is unchanged.
use super::programs::{Program, ProgramSpec, Programs, Scalar};
use std::ffi::c_void;

type Result<T> = std::result::Result<T, StageError>;

#[derive(Debug, thiserror::Error)]
#[error("invalid GLM attention stage: {0}")]
pub struct StageError(String);

fn invalid(message: impl Into<String>) -> StageError {
    StageError(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    RankPack,
    QueryRope,
    IndexPost,
    Absorb,
    Values,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub kind: Kind,
    pub heads: u32,
    pub capacity: u32,
    pub query_row: u32,
    pub fp8: bool,
    pub nope_dim: u32,
    pub query_dim: u32,
}

impl Layout {
    fn pointers(self) -> &'static [&'static str] {
        match self.kind {
            Kind::RankPack => &[
                "qkv",
                "q_norm",
                "kv_norm",
                "positions",
                "slots",
                "cos_sin",
                "q_resid",
                "cache",
            ],
            Kind::QueryRope => &["q", "positions", "cos_sin", "query"],
            Kind::IndexPost => &[
                "iq",
                "kw",
                "positions",
                "slots",
                "cos_sin",
                "k_weight",
                "k_bias",
                "q_fp8",
                "head_weights",
                "cache",
            ],
            _ if self.fp8 => &["a", "w", "scale", "out"],
            _ => &["a", "w", "out"],
        }
    }

    fn pointer_contract(self) -> Vec<(&'static str, String, &'static str)> {
        let n = self.heads;
        let dim = self.query_dim;
        let nope = self.nope_dim;
        let query = if self.query_row == n * dim {
            format!("[rows,{n},{dim}]")
        } else {
            format!("[rows,{n},{dim}] (rows {} apart)", self.query_row)
        };
        let contract: Vec<(&str, String, &str)> = match self.kind {
            Kind::RankPack => vec![
                ("bfloat16", "[rows,2624]".into(), "in"),
                ("bfloat16", "[2048]".into(), "in"),
                ("bfloat16", "[512]".into(), "in"),
                ("int64", "[rows]".into(), "in"),
                ("int64", "[rows]".into(), "in"),
                ("float32", "[P,64]".into(), "in"),
                ("bfloat16", "[rows,2048]".into(), "out"),
                ("uint8", "[pages,41984]".into(), "inout"),
            ],
            Kind::QueryRope => vec![
                ("bfloat16", format!("[rows,{n},256]"), "in"),
                ("int64", "[rows]".into(), "in"),
                ("float32", "[P,64]".into(), "in"),
                ("bfloat16", query, "inout"),
            ],
            Kind::IndexPost => vec![
                ("bfloat16", "[rows,32,128]".into(), "in"),
                ("bfloat16", "[rows,160]".into(), "in"),
                ("int64", "[rows]".into(), "in"),
                ("int64", "[rows]".into(), "in"),
                ("float32", "[P,64]".into(), "in"),
                ("bfloat16", "[128]".into(), "in"),
                ("bfloat16", "[128]".into(), "in"),
                ("float8_e4m3fn", "[rows,32,128]".into(), "out"),
                ("float32", "[rows,32]".into(), "out"),
                ("uint8", "[pages,8448]".into(), "inout"),
            ],
            Kind::Absorb | Kind::Values => {
                let absorb = self.kind == Kind::Absorb;
                let mut v = vec![
                    (
                        "bfloat16",
                        format!("[rows,{n},{}]", if absorb { 256 } else { 512 }),
                        "in",
                    ),
                    (
                        if self.fp8 {
                            "float8_e4m3fn"
                        } else {
                            "bfloat16"
                        },
                        if absorb {
                            format!("[{n},512,{nope}]")
                        } else {
                            format!("[{n},256,512]")
                        },
                        "in",
                    ),
                ];
                if self.fp8 {
                    v.push((
                        "float32",
                        if absorb {
                            format!("[{n},512,{}]", nope / 64)
                        } else {
                            format!("[{n},256,8]")
                        },
                        "in",
                    ));
                }
                v.push((
                    "bfloat16",
                    if absorb {
                        query
                    } else {
                        format!("[rows,{n},256]")
                    },
                    "out",
                ));
                v
            }
        };
        contract
    }
}

fn layout(spec: &ProgramSpec) -> Result<Layout> {
    let (name, heads, flash) = if let Some(name) = spec.name.strip_prefix("glm_stage_") {
        (name, 64, false)
    } else if let Some(name) = spec.name.strip_prefix("glm2_stage_") {
        (name, 32, false)
    } else if let Some(name) = spec.name.strip_prefix("glmf_stage_") {
        (name, 64, true)
    } else if let Some(name) = spec.name.strip_prefix("glmf2_stage_") {
        (name, 32, true)
    } else {
        return Err(invalid("unknown stage table"));
    };
    let (name, cap) = name
        .rsplit_once("_m")
        .ok_or_else(|| invalid("missing capacity"))?;
    let capacity: u32 = cap.parse().map_err(|_| invalid("malformed capacity"))?;
    if !(matches!(capacity, 64 | 4096) || (flash && capacity == 128))
        || spec.capacity_rows != capacity
    {
        return Err(invalid("capacity disagrees with stage table"));
    }
    let name = if flash {
        name.strip_suffix("_fp8")
            .ok_or_else(|| invalid("Flash KV requires native FP8"))?
    } else {
        name
    };
    let (name, context) = name
        .strip_suffix("_context")
        .map_or((name, false), |s| (s, true));
    let (name, bf16) = name
        .strip_suffix("_bf16")
        .map_or((name, false), |s| (s, true));
    let kind = match name {
        "rank_pack" => Kind::RankPack,
        "query_rope" => Kind::QueryRope,
        "index_post" => Kind::IndexPost,
        "absorb" => Kind::Absorb,
        "values" => Kind::Values,
        _ => return Err(invalid("unknown primitive")),
    };
    let batched = matches!(kind, Kind::Absorb | Kind::Values);
    if (flash && (!batched || bf16 || context))
        || (bf16 && !batched)
        || (context && (heads != 32 || !matches!(kind, Kind::QueryRope | Kind::Absorb)))
    {
        return Err(invalid("unsupported weight/pitch specialization"));
    }
    let layout = Layout {
        kind,
        heads,
        capacity,
        query_row: heads
            * if flash {
                512
            } else {
                576 * if context { 2 } else { 1 }
            },
        fp8: batched && !bf16,
        nope_dim: if flash { 256 } else { 192 },
        query_dim: if flash { 512 } else { 576 },
    };
    let g = &spec.geometry;
    let stage = match kind {
        Kind::RankPack => "rank_pack",
        Kind::QueryRope => "query_rope",
        Kind::IndexPost => "index_post",
        Kind::Absorb => "absorb",
        Kind::Values => "values",
    };
    if g["stage"].as_str() != Some(stage)
        || [
            ("max_rows", capacity),
            ("heads", heads),
            ("index_heads", 32),
            ("q_lora_rank", if flash { 1536 } else { 2048 }),
            ("record_bytes", if flash { 528 } else { 656 }),
            ("page_rows", 64),
            ("query_row", layout.query_row),
        ]
        .iter()
        .any(|&(name, value)| g[name].as_u64() != Some(u64::from(value)))
        || g["norm_eps"].as_f64() != Some(1e-5)
        || g["index_norm_eps"].as_f64() != Some(1e-6)
        || g["fp8_weights"].as_bool() != Some(layout.fp8)
        || (layout.fp8
            && g["mode"].as_str() != Some(if capacity <= 128 { "decode" } else { "prefill" }))
        || (!layout.fp8 && !g["mode"].is_null())
        || !spec
            .pointers
            .iter()
            .map(String::as_str)
            .eq(layout.pointers().iter().copied())
        || spec.scalar_kinds != b"i"
        || !spec.scratch.is_empty()
    {
        return Err(invalid(
            "geometry, pointer order, scalar kind or scratch mismatch",
        ));
    }
    Ok(layout)
}

/// Validate before any kernel binding or allocation.
pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<Layout> {
    let layout = layout(spec)?;
    let scalars = entry["scalars"]
        .as_array()
        .ok_or_else(|| invalid("missing scalar ABI"))?;
    if scalars.len() != 1 || scalars[0]["name"] != "rows" || scalars[0]["type"] != "int32" {
        return Err(invalid("scalar ABI must be rows:int32"));
    }
    let pointers = entry["pointers"]
        .as_array()
        .ok_or_else(|| invalid("missing pointer metadata"))?;
    let contract = layout.pointer_contract();
    if pointers.len() != contract.len()
        || pointers.iter().zip(contract).zip(layout.pointers()).any(
            |((p, (dtype, shape, role)), name)| {
                p["name"].as_str() != Some(*name)
                    || p["dtype"].as_str() != Some(dtype)
                    || p["shape"].as_str() != Some(&shape)
                    || p["role"].as_str() != Some(role)
            },
        )
    {
        return Err(invalid("pointer dtype, shape or access mismatch"));
    }
    Ok(layout)
}

/// Caller-owned device pointers; the enum prevents mixing stage operand orders.
#[derive(Debug, Clone, Copy)]
pub enum Buffers {
    RankPack {
        qkv: *mut c_void,
        q_norm: *mut c_void,
        kv_norm: *mut c_void,
        positions: *mut c_void,
        slots: *mut c_void,
        cos_sin: *mut c_void,
        q_resid: *mut c_void,
        cache: *mut c_void,
    },
    QueryRope {
        q: *mut c_void,
        positions: *mut c_void,
        cos_sin: *mut c_void,
        query: *mut c_void,
    },
    IndexPost {
        iq: *mut c_void,
        kw: *mut c_void,
        positions: *mut c_void,
        slots: *mut c_void,
        cos_sin: *mut c_void,
        k_weight: *mut c_void,
        k_bias: *mut c_void,
        q_fp8: *mut c_void,
        head_weights: *mut c_void,
        cache: *mut c_void,
    },
    Kv {
        a: *mut c_void,
        w: *mut c_void,
        scale: *mut c_void,
        out: *mut c_void,
    },
}

pub struct GlmStage<'a> {
    program: Program<'a>,
    pub layout: Layout,
}

impl<'a> GlmStage<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let spec = programs.spec(name).map_err(|e| invalid(e.to_string()))?;
        let layout = layout(spec)?;
        let program = programs
            .program(name, layout.pointers())
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self { program, layout })
    }

    /// # Safety
    /// The owner admits every buffer extent and retains its device pointers and
    /// stream through completion. Paged slots must address admitted cache pages.
    pub unsafe fn launch(&self, buffers: Buffers, rows: u32, stream: *mut c_void) -> Result<()> {
        if rows == 0 || rows > self.layout.capacity {
            return Err(invalid("rows outside compiled capacity"));
        }
        let pointers = match (self.layout.kind, buffers) {
            (
                Kind::RankPack,
                Buffers::RankPack {
                    qkv,
                    q_norm,
                    kv_norm,
                    positions,
                    slots,
                    cos_sin,
                    q_resid,
                    cache,
                },
            ) => vec![
                qkv, q_norm, kv_norm, positions, slots, cos_sin, q_resid, cache,
            ],
            (
                Kind::QueryRope,
                Buffers::QueryRope {
                    q,
                    positions,
                    cos_sin,
                    query,
                },
            ) => vec![q, positions, cos_sin, query],
            (
                Kind::IndexPost,
                Buffers::IndexPost {
                    iq,
                    kw,
                    positions,
                    slots,
                    cos_sin,
                    k_weight,
                    k_bias,
                    q_fp8,
                    head_weights,
                    cache,
                },
            ) => vec![
                iq,
                kw,
                positions,
                slots,
                cos_sin,
                k_weight,
                k_bias,
                q_fp8,
                head_weights,
                cache,
            ],
            (Kind::Absorb | Kind::Values, Buffers::Kv { a, w, scale, out }) => {
                if self.layout.fp8 {
                    vec![a, w, scale, out]
                } else {
                    vec![a, w, out]
                }
            }
            _ => return Err(invalid("buffers do not match stage")),
        };
        if pointers.iter().any(|p| p.is_null()) {
            return Err(invalid("null stage operand"));
        }
        // SAFETY: the caller owns extents/lifetimes; the validated ABI fixes pointer order and scalar type.
        unsafe {
            self.program
                .launch(&pointers, &[Scalar::I32(rows as i32)], stream)
        }
        .map_err(|e| invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture(
        kind: Kind,
        heads: u32,
        capacity: u32,
        fp8: bool,
        context: bool,
    ) -> (ProgramSpec, serde_json::Value) {
        let stage = match kind {
            Kind::RankPack => "rank_pack",
            Kind::QueryRope => "query_rope",
            Kind::IndexPost => "index_post",
            Kind::Absorb => "absorb",
            Kind::Values => "values",
        };
        let l = Layout {
            kind,
            heads,
            capacity,
            query_row: heads * 576 * if context { 2 } else { 1 },
            fp8,
            nope_dim: 192,
            query_dim: 576,
        };
        let suffix = if context {
            "_context"
        } else if matches!(kind, Kind::Absorb | Kind::Values) && !fp8 {
            "_bf16"
        } else {
            ""
        };
        let name = format!(
            "{}_stage_{stage}{suffix}_m{capacity}",
            if heads == 64 { "glm" } else { "glm2" }
        );
        let spec = ProgramSpec {
            name,
            pointers: l.pointers().iter().map(|s| s.to_string()).collect(),
            scalar_kinds: b"i".to_vec(),
            scratch: Default::default(),
            capacity_rows: capacity,
            geometry: json!({"stage":stage,"max_rows":capacity,"heads":heads,"index_heads":32,
                "q_lora_rank":2048,"record_bytes":656,"page_rows":64,"query_row":l.query_row,
                "norm_eps":1e-5,"index_norm_eps":1e-6,"fp8_weights":fp8,
                "mode":if fp8 { Some(if capacity == 64 {"decode"} else {"prefill"}) } else {None}}),
        };
        let pointers: Vec<_> = l.pointers().iter().zip(l.pointer_contract()).map(|(name,(dtype,shape,role))|
            json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect();
        (
            spec,
            json!({"pointers":pointers,"scalars":[{"name":"rows","type":"int32"}]}),
        )
    }

    #[test]
    fn all_catalog_contracts() {
        let mut count = 0;
        for heads in [32, 64] {
            for capacity in [64, 4096] {
                for kind in [
                    Kind::RankPack,
                    Kind::QueryRope,
                    Kind::IndexPost,
                    Kind::Absorb,
                    Kind::Values,
                ] {
                    for fp8 in [false, true] {
                        if fp8 && !matches!(kind, Kind::Absorb | Kind::Values) {
                            continue;
                        }
                        let (spec, entry) = fixture(kind, heads, capacity, fp8, false);
                        assert_eq!(validate_entry(&spec, &entry).unwrap().kind, kind);
                        count += 1;
                    }
                }
                if heads == 32 {
                    for kind in [Kind::QueryRope, Kind::Absorb] {
                        let (spec, entry) =
                            fixture(kind, heads, capacity, kind == Kind::Absorb, true);
                        assert_eq!(validate_entry(&spec, &entry).unwrap().query_row, 36864);
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 32);
    }

    #[test]
    fn flash_native_kv_uses_its_own_geometry() {
        for heads in [32, 64] {
            for capacity in [64, 128, 4096] {
                for kind in [Kind::Absorb, Kind::Values] {
                    let (mut spec, _) = fixture(kind, heads, 64, true, false);
                    let stage = if kind == Kind::Absorb {
                        "absorb"
                    } else {
                        "values"
                    };
                    spec.name = format!(
                        "{}_stage_{stage}_fp8_m{capacity}",
                        if heads == 64 { "glmf" } else { "glmf2" }
                    );
                    spec.capacity_rows = capacity;
                    spec.geometry["max_rows"] = json!(capacity);
                    spec.geometry["q_lora_rank"] = json!(1536);
                    spec.geometry["record_bytes"] = json!(528);
                    spec.geometry["query_row"] = json!(heads * 512);
                    spec.geometry["mode"] =
                        json!(if capacity <= 128 { "decode" } else { "prefill" });
                    let l = Layout {
                        kind,
                        heads,
                        capacity,
                        query_row: heads * 512,
                        fp8: true,
                        nope_dim: 256,
                        query_dim: 512,
                    };
                    let pointers: Vec<_> = l.pointers().iter().zip(l.pointer_contract())
                        .map(|(name,(dtype,shape,role))| json!({"name":name,"dtype":dtype,"shape":shape,"role":role}))
                        .collect();
                    let entry =
                        json!({"pointers":pointers,"scalars":[{"name":"rows","type":"int32"}]});
                    assert_eq!(validate_entry(&spec, &entry).unwrap(), l);
                    for (key, value) in [
                        ("q_lora_rank", json!(2048)),
                        ("query_row", json!(heads * 576)),
                        ("record_bytes", json!(656)),
                        ("fp8_weights", json!(false)),
                    ] {
                        let mut bad = spec.clone();
                        bad.geometry[key] = value;
                        assert!(validate_entry(&bad, &entry).is_err(), "{key}");
                    }
                    if kind == Kind::Absorb {
                        let mut bad = entry.clone();
                        bad["pointers"][1]["shape"] = json!(format!("[{heads},512,192]"));
                        assert!(validate_entry(&spec, &bad).is_err());
                        let mut bad = entry.clone();
                        bad["pointers"][2]["shape"] = json!(format!("[{heads},512,3]"));
                        assert!(validate_entry(&spec, &bad).is_err());
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_manifest_mutations() {
        let (spec, entry) = fixture(Kind::Absorb, 32, 4096, true, true);
        for (key, value) in [
            ("heads", json!(64)),
            ("max_rows", json!(64)),
            ("query_row", json!(18432)),
            ("index_norm_eps", json!(1e-5)),
            ("mode", json!("decode")),
            ("fp8_weights", json!(false)),
        ] {
            let mut bad = spec.clone();
            bad.geometry[key] = value;
            assert!(validate_entry(&bad, &entry).is_err(), "{key}");
        }
        let mut bad = spec.clone();
        bad.pointers.swap(0, 1);
        assert!(validate_entry(&bad, &entry).is_err());
        let mut bad = spec.clone();
        bad.scalar_kinds = b"l".to_vec();
        assert!(validate_entry(&bad, &entry).is_err());
        let mut bad = spec.clone();
        bad.scratch.insert("scratch".into(), 0);
        assert!(validate_entry(&bad, &entry).is_err());
        for (key, value) in [
            ("dtype", json!("float16")),
            ("shape", json!("[rows,32,512]")),
            ("role", json!("out")),
        ] {
            let mut bad = entry.clone();
            bad["pointers"][0][key] = value;
            assert!(validate_entry(&spec, &bad).is_err(), "{key}");
        }
        let mut bad = entry;
        bad["scalars"][0]["name"] = json!("live_rows");
        assert!(validate_entry(&spec, &bad).is_err());
    }
}
