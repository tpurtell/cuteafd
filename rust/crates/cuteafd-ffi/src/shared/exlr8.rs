//! Fail-closed EXLR8 fused ingress ABI v2 and hidden-domain A16 ABI v3 contracts.
use super::programs::ProgramSpec;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
#[error("invalid EXLR8 program manifest: {0}")]
pub struct ManifestError(String);

fn invalid(message: impl Into<String>) -> ManifestError {
    ManifestError(message.into())
}

/// Checkpoint-calibrated expert kernel namespace and numerical contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelFamily { Glm, GlmFlash }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    Decode, Routed, Shared, GroupedDecode, GroupedA16, GroupedA8,
    Group, SharedGroup, GroupedGroup,
    RoutedCore, ReplyFp8, ReplyNvfp4, GroupedA16ReplyFp8, GroupedA8ReplyFp8,
    GroupedDecodeReplyFp8, GroupedA16ReplyNvfp4, GroupedA8ReplyNvfp4, GroupedDecodeReplyNvfp4,
}

impl KernelKind {
    /// Reply producers preserve their BF16 sibling's compute and scratch ABI.
    pub fn arithmetic_kind(self) -> Self {
        match self {
            Self::RoutedCore => Self::Routed,
            Self::GroupedA16ReplyFp8 | Self::GroupedA16ReplyNvfp4 => Self::GroupedA16,
            Self::GroupedA8ReplyFp8 | Self::GroupedA8ReplyNvfp4 => Self::GroupedA8,
            Self::GroupedDecodeReplyFp8 | Self::GroupedDecodeReplyNvfp4 => Self::GroupedDecode,
            _ => self,
        }
    }

    pub fn reply_format(self) -> Option<u64> {
        match self {
            Self::ReplyFp8 | Self::GroupedA16ReplyFp8 | Self::GroupedA8ReplyFp8 | Self::GroupedDecodeReplyFp8 => Some(1),
            Self::ReplyNvfp4 | Self::GroupedA16ReplyNvfp4 | Self::GroupedA8ReplyNvfp4 | Self::GroupedDecodeReplyNvfp4 => Some(2),
            _ => None,
        }
    }

    pub fn is_reply_epilogue(self) -> bool { matches!(self, Self::ReplyFp8 | Self::ReplyNvfp4) }

    pub fn is_grouped(self) -> bool {
        matches!(self.arithmetic_kind(), Self::GroupedDecode | Self::GroupedA16 | Self::GroupedA8)
    }

    pub fn is_glue(self) -> bool {
        matches!(self, Self::Group | Self::SharedGroup | Self::GroupedGroup)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelGeometry {
    pub hidden: usize,
    pub units: usize,
    pub top_k: usize,
    pub experts: usize,
    pub swiglu_limit: f64,
}

impl KernelFamily {
    pub const fn geometry(self) -> KernelGeometry {
        match self {
            Self::Glm => KernelGeometry { hidden: 6144, units: 16, top_k: 8, experts: 256, swiglu_limit: 0.0 },
            Self::GlmFlash => KernelGeometry { hidden: 4096, units: 16, top_k: 8, experts: 288, swiglu_limit: 10.0 },
        }
    }

    pub fn from_geometry(hidden: usize, units: usize, top_k: usize, experts: usize)
        -> Result<Self, ManifestError> {
        [Self::Glm, Self::GlmFlash].into_iter().find(|family| {
            let g = family.geometry();
            (hidden, units, top_k, experts) == (g.hidden, g.units, g.top_k, g.experts)
        }).ok_or_else(|| invalid("unsupported EXLR8 family geometry"))
    }

    pub fn program(self, kind: KernelKind) -> &'static str {
        match (self, kind) {
            (Self::Glm, KernelKind::RoutedCore) => "exlr8_a16_serving_core",
            (Self::Glm, KernelKind::ReplyFp8) => "exlr8_reply_1",
            (Self::Glm, KernelKind::ReplyNvfp4) => "exlr8_reply_2",
            (Self::Glm, KernelKind::GroupedA16ReplyFp8) => "exlr8_a16_grouped_reply1",
            (Self::Glm, KernelKind::GroupedA8ReplyFp8) => "exlr8_a8_grouped_reply1",
            (Self::Glm, KernelKind::GroupedDecodeReplyFp8) => "exlr8_a16_grouped64_reply1",
            (Self::Glm, KernelKind::GroupedA16ReplyNvfp4) => "exlr8_a16_grouped_reply2",
            (Self::Glm, KernelKind::GroupedA8ReplyNvfp4) => "exlr8_a8_grouped_reply2",
            (Self::Glm, KernelKind::GroupedDecodeReplyNvfp4) => "exlr8_a16_grouped64_reply2",
            (Self::GlmFlash, KernelKind::RoutedCore) => "exlr8_glmf_a16_serving_core",
            (Self::GlmFlash, KernelKind::ReplyFp8) => "exlr8_glmf_reply_1",
            (Self::GlmFlash, KernelKind::ReplyNvfp4) => "exlr8_glmf_reply_2",
            (Self::GlmFlash, KernelKind::GroupedA16ReplyFp8) => "exlr8_glmf_a16_grouped_reply1",
            (Self::GlmFlash, KernelKind::GroupedA8ReplyFp8) => "exlr8_glmf_a8_grouped_reply1",
            (Self::GlmFlash, KernelKind::GroupedDecodeReplyFp8) => "exlr8_glmf_a16_grouped64_reply1",
            (Self::GlmFlash, KernelKind::GroupedA16ReplyNvfp4) => "exlr8_glmf_a16_grouped_reply2",
            (Self::GlmFlash, KernelKind::GroupedA8ReplyNvfp4) => "exlr8_glmf_a8_grouped_reply2",
            (Self::GlmFlash, KernelKind::GroupedDecodeReplyNvfp4) => "exlr8_glmf_a16_grouped64_reply2",
            (Self::Glm, KernelKind::Decode) => "exlr8_a16_decode",
            (Self::Glm, KernelKind::Routed) => "exlr8_a16_serving",
            (Self::Glm, KernelKind::Shared) => "exlr8_shared_a16_serving",
            (Self::Glm, KernelKind::GroupedDecode) => "exlr8_a16_grouped64",
            (Self::Glm, KernelKind::GroupedA16) => "exlr8_a16_grouped",
            (Self::Glm, KernelKind::GroupedA8) => "exlr8_a8_grouped",
            (Self::Glm, KernelKind::Group) => "exlr8_group",
            (Self::Glm, KernelKind::SharedGroup) => "exlr8_shared_group",
            (Self::Glm, KernelKind::GroupedGroup) => "exlr8_grouped_group",
            (Self::GlmFlash, KernelKind::Decode) => "exlr8_glmf_a16_decode",
            (Self::GlmFlash, KernelKind::Routed) => "exlr8_glmf_a16_serving",
            (Self::GlmFlash, KernelKind::Shared) => "exlr8_glmf_shared_a16_serving",
            (Self::GlmFlash, KernelKind::GroupedDecode) => "exlr8_glmf_a16_grouped64",
            (Self::GlmFlash, KernelKind::GroupedA16) => "exlr8_glmf_a16_grouped",
            (Self::GlmFlash, KernelKind::GroupedA8) => "exlr8_glmf_a8_grouped",
            (Self::GlmFlash, KernelKind::Group) => "exlr8_glmf_group",
            (Self::GlmFlash, KernelKind::SharedGroup) => "exlr8_glmf_shared_group",
            (Self::GlmFlash, KernelKind::GroupedGroup) => "exlr8_glmf_grouped_group",
        }
    }

    /// Names identify the numerical contract; shape and clamp never select a fallback.
    pub fn of(spec: &ProgramSpec) -> Result<(Self, KernelKind), ManifestError> {
        let (family, suffix) = if let Some(suffix) = spec.name.strip_prefix("exlr8_glmf_") {
            (Self::GlmFlash, suffix)
        } else if let Some(suffix) = spec.name.strip_prefix("exlr8_") {
            (Self::Glm, suffix)
        } else { return Err(invalid("unknown EXLR8 family")); };
        let kind = match suffix {
            "a16_serving_core" => KernelKind::RoutedCore,
            "reply_1" => KernelKind::ReplyFp8,
            "reply_2" => KernelKind::ReplyNvfp4,
            "a16_grouped_reply1" => KernelKind::GroupedA16ReplyFp8,
            "a8_grouped_reply1" => KernelKind::GroupedA8ReplyFp8,
            "a16_grouped64_reply1" => KernelKind::GroupedDecodeReplyFp8,
            "a16_grouped_reply2" => KernelKind::GroupedA16ReplyNvfp4,
            "a8_grouped_reply2" => KernelKind::GroupedA8ReplyNvfp4,
            "a16_grouped64_reply2" => KernelKind::GroupedDecodeReplyNvfp4,
            "a16_decode" => KernelKind::Decode,
            "a16_serving" => KernelKind::Routed,
            "shared_a16_serving" => KernelKind::Shared,
            "a16_grouped64" => KernelKind::GroupedDecode,
            "a16_grouped" => KernelKind::GroupedA16,
            "a8_grouped" => KernelKind::GroupedA8,
            "group" => KernelKind::Group,
            "shared_group" => KernelKind::SharedGroup,
            "grouped_group" => KernelKind::GroupedGroup,
            _ => return Err(invalid("unknown EXLR8 program kind")),
        };
        if (kind == KernelKind::RoutedCore && spec.geometry["core_only"] != true)
            || (kind != KernelKind::RoutedCore && spec.geometry.get("core_only").is_some()) {
            return Err(invalid("core_only requires its named core export"));
        }
        if !kind.is_reply_epilogue() && spec.geometry.get("reply_format").and_then(|v| v.as_u64()) != kind.reply_format() {
            return Err(invalid("compressed producer name and reply_format disagree"));
        }
        if kind.reply_format().is_none() && spec.geometry.get("reply_format").is_some() {
            return Err(invalid("reply_format requires a named compressed producer"));
        }
        Ok((family, kind))
    }

    fn validate_compute(self, spec: &ProgramSpec, shared: bool) -> Result<(), ManifestError> {
        let g = self.geometry();
        let clamp = spec.geometry.get("swiglu_limit");
        let clamp_ok = match clamp {
            None => g.swiglu_limit == 0.0,
            Some(value) => value.as_f64() == Some(g.swiglu_limit),
        };
        if spec.geometry["hidden"].as_u64() != Some(g.hidden as u64)
            || spec.geometry["units"].as_u64() != Some(g.units as u64)
            || spec.geometry["top_k"].as_u64() != Some(if shared { 1 } else { g.top_k as u64 })
            || !clamp_ok {
            return Err(invalid("compute geometry or SwiGLU clamp disagrees with compiled family"));
        }
        Ok(())
    }
}

/// Compact compute and ingress ABIs have a fixed route-slot stride.
pub const COMPACT_GROUP_ROWS: usize = 16;

/// Read the route-slot stride from the named export ABI, never a missing-field default.
pub fn group_rows(spec: &ProgramSpec) -> Result<usize, ManifestError> {
    let (_, kind) = KernelFamily::of(spec)?;
    let (abi, capacity, grouped) = match kind.arithmetic_kind() {
        KernelKind::Decode | KernelKind::Routed | KernelKind::Shared => (3, 64, false),
        KernelKind::GroupedDecode => (3, 64, true),
        KernelKind::GroupedA16 | KernelKind::GroupedA8 => (3, 4096, true),
        KernelKind::Group | KernelKind::SharedGroup => (2, 64, false),
        KernelKind::GroupedGroup => (2, 4096, true),
        _ => return Err(invalid("reply epilogue has no grouping scratch ABI")),
    };
    if spec.geometry["abi_version"].as_u64() != Some(abi)
        || spec.geometry["max_rows"].as_u64() != Some(capacity)
        || u64::from(spec.capacity_rows) != capacity
    {
        return Err(invalid("grouping ABI version or capacity mismatch"));
    }
    if !grouped {
        if spec.geometry.get("group_rows").is_some_and(|r| r.as_u64() != Some(COMPACT_GROUP_ROWS as u64)) {
            return Err(invalid("compact ABI requires its fixed group row stride"));
        }
        return Ok(COMPACT_GROUP_ROWS);
    }
    let rows = spec.geometry["group_rows"].as_u64()
        .ok_or_else(|| invalid("grouped ABI requires explicit geometry.group_rows"))?;
    if !matches!(rows, 16 | 32 | 64) || (capacity == 64 && rows != COMPACT_GROUP_ROWS as u64) {
        return Err(invalid("grouped row tile must be 16, 32 or 64 (16 when decoding)"));
    }
    Ok(rows as usize)
}

pub fn validate(spec: &ProgramSpec) -> Result<(), ManifestError> {
    let field = |name: &str| {
        spec.geometry[name]
            .as_u64()
            .filter(|&v| v > 0 && v <= i32::MAX as u64)
            .ok_or_else(|| invalid(format!("missing or invalid geometry.{name}")))
    };
    let (family, kind) = KernelFamily::of(spec)?;
    let kind = kind.arithmetic_kind();
    let shared = kind == KernelKind::Shared;
    family.validate_compute(spec, shared)?;
    let grouped_a8 = kind == KernelKind::GroupedA8;
    // The 64-row grouped instance reuses the grouped ABI below the K1
    // threshold; its F32 FC2 partials must match the K1 serving kernel.
    let grouped_decode = kind == KernelKind::GroupedDecode;
    let grouped = kind.is_grouped();
    let serving = shared || grouped || kind == KernelKind::Routed;
    let activations = spec.geometry.get("activations").and_then(|v| v.as_str());
    if (grouped_a8 && activations != Some("a8"))
        || (grouped_decode && activations.is_some())
        || (!grouped_a8 && !grouped_decode && activations.is_some_and(|v| v != "a16"))
    {
        return Err(invalid("grouped activation mode mismatch"));
    }
    if grouped_decode
        && spec.geometry.get("fc2_partials").and_then(|v| v.as_str()) != Some("f32")
    {
        return Err(invalid("64-row grouped instance requires F32 FC2 partials"));
    }
    // The routed serving core is a compact-unit instance selected by name and
    // variant; its geometry must declare which served variant it is, and only
    // the compact-unit routed variants are accepted here.
    if kind == KernelKind::Routed {
        let variant = spec.geometry.get("serving_variant").and_then(|v| v.as_str());
        if !matches!(variant, Some("k1_k23_coalesced"))
        {
            return Err(invalid("routed serving requires a declared compact-unit variant"));
        }
    } else if spec.geometry.get("serving_variant").is_some() {
        return Err(invalid("serving_variant applies to the routed serving instance"));
    }
    let max_rows = if grouped && !grouped_decode { 4096 } else { 64 };
    let group_rows = group_rows(spec)? as u64;
    if (!serving && kind != KernelKind::Decode)
        || field("abi_version")? != 3
        || field("max_rows")? != max_rows
        || u64::from(spec.capacity_rows) != max_rows
        || field("split_k_blocks")? != 8
    {
        return Err(invalid(
            "name, ABI version, row capacity or fixed split order",
        ));
    }
    let h = field("hidden")?;
    let u = field("units")?;
    let t = field("top_k")?;
    let g = field("max_groups")?;
    if shared && (t != 1 || g != 4) {
        return Err(invalid("shared expert requires top_k=1 and max_groups=4"));
    }
    if h % 128 != 0 || field("fc1_splits")? != h.div_ceil(1024) {
        return Err(invalid("hidden alignment or FC1 split count"));
    }
    // Launch residency hint: the caller passes cta_limit = SM count (cudaDevAttr) * this.
    if field("ctas_per_sm")? > 32 {
        return Err(invalid("ctas_per_sm out of range"));
    }
    let mut expected = vec![
        "x",
        "pointers",
        "bits",
        "group_expert",
        "route_slots",
        "route_group_lane",
        "route_weights",
        "fc1",
        "intermediate",
        "fc2",
        "signs_out",
        "output",
    ];
    if serving {
        expected.push("group_count");
    }
    if grouped { expected.extend(["unit_count", "unit_map"]); }
    let scalars: &[u8] = if grouped { b"iiii" } else if serving { b"ii" } else { b"iii" };
    if !spec.pointers.iter().map(String::as_str).eq(expected) || spec.scalar_kinds != scalars {
        return Err(invalid("pointer order or scalar ABI"));
    }
    let product = |factors: &[u64]| {
        factors
            .iter()
            .try_fold(1u64, |a, &b| {
                a.checked_mul(b).filter(|&v| v <= i64::MAX as u64)
            })
            .ok_or_else(|| invalid("scratch size overflow"))
    };
    // Leave headroom for the final grid-stride increment before loop exit.
    let tasks = [
        product(&[g, u, 2, h.div_ceil(1024), if grouped { 4 } else { 1 }])?,
        product(&[g, u])?,
        product(&[g, h / 128, if grouped { 4 } else { 1 }])?,
        product(&[max_rows, h / 128])?,
    ];
    if tasks.iter().any(|&n| n > i32::MAX as u64 / 2)
        || product(&[max_rows, t])? > i32::MAX as u64
        || product(&[g, group_rows])? > i32::MAX as u64
    {
        return Err(invalid("task or route indexing exceeds signed int32"));
    }
    let sizes = HashMap::from([
        (
            "fc1".to_string(),
            product(&[g, u, 2, h.div_ceil(1024), group_rows, 128, 4])?,
        ),
        ("intermediate".to_string(), product(&[g, u, group_rows, 128, 2])?),
        ("fc2".to_string(), product(&[g, group_rows, h, 4])?),
    ]);
    if spec.scratch != sizes {
        return Err(invalid("scratch capacities differ from EXLR8 ABI"));
    }
    Ok(())
}

pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<(), ManifestError> {
    if spec.name.ends_with("_reply_1") || spec.name.ends_with("_reply_2") {
        return validate_reply_entry(spec, entry);
    }
    let (_, kind) = KernelFamily::of(spec)?;
    let kind = kind.arithmetic_kind();
    if kind.is_glue() {
        return validate_glue(spec, entry);
    }
    validate(spec)?;
    if entry["capacity_rows"].as_u64() != Some(u64::from(spec.capacity_rows)) || entry.get("card_plans").is_some() {
        return Err(invalid(
            "EXLR8 requires one SM-count-independent plan and exact capacity",
        ));
    }
    let h = spec.geometry["hidden"].as_u64().unwrap();
    let u = spec.geometry["units"].as_u64().unwrap();
    let t = spec.geometry["top_k"].as_u64().unwrap();
    let g = spec.geometry["max_groups"].as_u64().unwrap();
    let s = h.div_ceil(1024);
    let r = group_rows(spec)? as u64;
    let mut shapes = vec![
        format!("[rows,{h}]"),
        format!("[E,{u},3,3]"),
        format!("[E,{u},3]"),
        format!("[{g}]"),
        format!("[{g},{r}]"),
        format!("[rows,{t}]"),
        format!("[rows,{t}]"),
        format!("[{g},{u},2,{s},{r},128]"),
        format!("[{g},{u},{r},128]"),
        format!("[{g},{r},{h}]"),
        format!("[{h}]"),
        format!("[rows,{h}]"),
    ];
    let mut dtypes = vec![
        "float16", "int64", "int32", "int32", "int32", "int32", "float32", "float32", "float16",
        "float32", "int8", "bfloat16",
    ];
    if kind != KernelKind::Decode {
        shapes.push("[1]".into());
        dtypes.push("int32");
    }
    if kind.is_grouped() {
        shapes.extend(["[E]".into(), format!("[E,{u}]")]);
        dtypes.extend(["int32", "int32"]);
    }
    if let Some(format) = spec.geometry.get("reply_format") {
        let stride = match format.as_u64() {
            Some(1) => h + h / 32,
            Some(2) => h / 2 + h / 16,
            _ => return Err(invalid("unknown EXLR8 reply format")),
        };
        shapes[11] = format!("[rows,{stride}]");
        dtypes[11] = "uint8";
    }
    let pointers = entry["pointers"]
        .as_array()
        .ok_or_else(|| invalid("pointer descriptors missing"))?;
    if pointers.len() != shapes.len()
        || pointers.iter().enumerate().any(|(i, p)| {
            p["name"].as_str() != Some(spec.pointers[i].as_str())
                || p["dtype"].as_str() != Some(dtypes[i])
                || p["shape"].as_str() != Some(&shapes[i])
                || p["role"].as_str()
                    != Some(if (7..=9).contains(&i) {
                        "scratch"
                    } else if i == 11 {
                        "out"
                    } else {
                        "in"
                    })
        })
    {
        return Err(invalid("pointer shape, dtype or role"));
    }
    let scratch = entry["scratch_bytes_at_capacity"]
        .as_object()
        .ok_or_else(|| invalid("scratch missing"))?;
    if scratch.len() != 3
        || scratch
            .iter()
            .any(|(k, v)| v.as_u64() != spec.scratch.get(k).copied())
    {
        return Err(invalid("invalid scratch fields"));
    }
    Ok(())
}

fn validate_reply_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<(), ManifestError> {
    let (family, format) = match spec.name.as_str() {
        "exlr8_reply_1" => (KernelFamily::Glm, 1),
        "exlr8_reply_2" => (KernelFamily::Glm, 2),
        "exlr8_glmf_reply_1" => (KernelFamily::GlmFlash, 1),
        "exlr8_glmf_reply_2" => (KernelFamily::GlmFlash, 2),
        _ => return Err(invalid("unknown EXLR8 reply epilogue")),
    };
    let g = family.geometry();
    let stride = if format == 1 { g.hidden + g.hidden / 32 } else { g.hidden / 2 + g.hidden / 16 };
    let names = ["route_group_lane", "route_weights", "fc2", "signs_out", "output", "group_count"];
    let shapes = [format!("[rows,{}]", g.top_k), format!("[rows,{}]", g.top_k), "[fc2_scratch]".into(),
        format!("[{}]", g.hidden), format!("[rows,{stride}]"), "[1]".into()];
    let types = ["int32", "float32", "float32", "int8", "uint8", "int32"];
    let pointers = entry["pointers"].as_array().ok_or_else(|| invalid("reply pointer descriptors missing"))?;
    let scalars = entry["scalars"].as_array().ok_or_else(|| invalid("reply scalar descriptors missing"))?;
    if spec.capacity_rows != 64 || entry["capacity_rows"] != 64 || spec.scalar_kinds != b"ii"
        || !spec.scratch.is_empty() || entry.get("card_plans").is_some()
        || entry["scratch_bytes_at_capacity"].as_object().is_none_or(|s| !s.is_empty())
        || spec.geometry["hidden"].as_u64() != Some(g.hidden as u64)
        || spec.geometry["top_k"].as_u64() != Some(g.top_k as u64)
        || spec.geometry["format"] != format || spec.geometry["row_bytes"].as_u64() != Some(stride as u64)
        || spec.geometry["variant"] != "k1_k23_coalesced"
        || !spec.pointers.iter().map(String::as_str).eq(names)
        || pointers.len() != names.len() || pointers.iter().enumerate().any(|(i, p)|
            p["name"] != names[i] || p["shape"].as_str() != Some(shapes[i].as_str()) || p["dtype"] != types[i]
                || p["role"] != if i == 4 { "out" } else { "in" })
        || scalars.len() != 2 || scalars.iter().zip(["rows", "cta_limit"]).any(|(s, name)|
            s["name"] != name || s["type"] != "int32") {
        return Err(invalid("EXLR8 reply epilogue contract mismatch"));
    }
    Ok(())
}

fn validate_glue(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<(), ManifestError> {
    let field = |name: &str| {
        spec.geometry[name]
            .as_u64()
            .filter(|&v| v > 0 && v <= i32::MAX as u64)
            .ok_or_else(|| invalid(format!("invalid glue geometry.{name}")))
    };
    let rows = field("max_rows")?;
    if field("abi_version")? != 2
        || entry["capacity_rows"].as_u64() != Some(rows)
        || u64::from(spec.capacity_rows) != rows
        || entry.get("card_plans").is_some()
        || !spec.scratch.is_empty()
        || entry["scratch_bytes_at_capacity"]
            .as_object()
            .is_none_or(|s| !s.is_empty())
    {
        return Err(invalid("glue ABI, capacity, scratch or card plan"));
    }
    let (family, kind) = KernelFamily::of(spec)?;
    let descriptors: Vec<(&str, &str, String, &str)> = {
            let e = field("experts")?;
            let t = field("top_k")?;
            let g = field("max_groups")?;
            let h = field("hidden")?;
            let shared = kind == KernelKind::SharedGroup;
            let grouped = kind == KernelKind::GroupedGroup;
            let expected = family.geometry();
            if family == KernelFamily::GlmFlash
                && (h != expected.hidden as u64 || e != if shared { 1 } else { expected.experts as u64 }
                    || t != if shared { 1 } else { expected.top_k as u64 }) {
                return Err(invalid("Flash grouping family geometry mismatch"));
            }
            let r = group_rows(spec)? as u64;
            let (format, dtype, row, row_bytes) = if shared {
                ("bf16", "bfloat16", h, h * 2)
            } else {
                ("fp8_k32", "uint8", h + h / 32, h + h / 32)
            };
            if shared && (e != 1 || t != 1 || g != 4) {
                return Err(invalid(
                    "shared grouping requires experts=1, top_k=1, max_groups=4",
                ));
            }
            if rows != if grouped { 4096 } else { 64 }
                || e > 512
                || t > 32
                || g > i32::MAX as u64 / r
                || spec.scalar_kinds != b"ii"
                || h % 128 != 0
                || rows * h * 2 > i32::MAX as u64
                || field("ctas_per_sm")? != 4
                || field("ingress_warps")? != 8
                || spec.geometry["input_format"].as_str() != Some(format)
                || field("input_row_bytes")? != row_bytes
            {
                return Err(invalid("grouping geometry or scalar ABI"));
            }
            vec![
                ("ids", "int32", format!("[rows,{t}]"), "in"),
                ("x", dtype, format!("[rows,{row}]"), "in"),
                ("signs_in", "int8", format!("[{h}]"), "in"),
                ("group_expert", "int32", format!("[{g}]"), "out"),
                ("route_slots", "int32", format!("[{g},{r}]"), "out"),
                ("route_group_lane", "int32", format!("[rows,{t}]"), "out"),
                ("group_count", "int32", "[1]".into(), "out"),
                ("activation", "float16", format!("[rows,{h}]"), "out"),
            ]
        };
    let pointers = entry["pointers"]
        .as_array()
        .ok_or_else(|| invalid("glue pointers"))?;
    if pointers.len() != descriptors.len()
        || spec.pointers.len() != descriptors.len()
        || descriptors
            .iter()
            .enumerate()
            .any(|(i, (name, dtype, shape, role))| {
                spec.pointers[i] != *name
                    || pointers[i]["name"].as_str() != Some(*name)
                    || pointers[i]["dtype"].as_str() != Some(*dtype)
                    || pointers[i]["shape"].as_str() != Some(shape.as_str())
                    || pointers[i]["role"].as_str() != Some(*role)
            })
    {
        return Err(invalid("glue pointer name, shape, dtype or role"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> ProgramSpec {
        ProgramSpec {
            name: "exlr8_a16_decode".into(),
            pointers: [
                "x",
                "pointers",
                "bits",
                "group_expert",
                "route_slots",
                "route_group_lane",
                "route_weights",
                "fc1",
                "intermediate",
                "fc2",
                "signs_out",
                "output",
            ]
            .map(String::from)
            .to_vec(),
            scalar_kinds: b"iii".to_vec(),
            capacity_rows: 64,
            geometry: serde_json::json!({"abi_version":3,"hidden":6144,"units":16,"top_k":8,
                "max_groups":512,"max_rows":64,"split_k_blocks":8,"fc1_splits":6,
                "ctas_per_sm":4}),
            scratch: HashMap::from([
                ("fc1".into(), 805306368),
                ("intermediate".into(), 33554432),
                ("fc2".into(), 201326592),
            ]),
        }
    }
    #[test]
    fn grouping_stride_follows_the_export_abi_and_rejects_missing_grouped_metadata() {
        for (name, abi, capacity, explicit) in [
            ("exlr8_a16_serving", 3, 64, None),
            ("exlr8_shared_a16_serving", 3, 64, None),
            ("exlr8_group", 2, 64, None),
            ("exlr8_shared_group", 2, 64, None),
            ("exlr8_a16_grouped64", 3, 64, Some(16)),
            ("exlr8_a16_grouped", 3, 4096, Some(64)),
            ("exlr8_a8_grouped", 3, 4096, Some(64)),
            ("exlr8_grouped_group", 2, 4096, Some(64)),
        ] {
            let mut spec = spec();
            spec.name = name.into();
            spec.capacity_rows = capacity;
            spec.geometry["abi_version"] = serde_json::json!(abi);
            spec.geometry["max_rows"] = serde_json::json!(capacity);
            if let Some(rows) = explicit { spec.geometry["group_rows"] = serde_json::json!(rows); }
            assert_eq!(group_rows(&spec).unwrap(), explicit.unwrap_or(COMPACT_GROUP_ROWS), "{name}");
            spec.geometry.as_object_mut().unwrap().remove("group_rows");
            assert_eq!(group_rows(&spec).is_err(), explicit.is_some(), "{name}");
            spec.geometry["group_rows"] = serde_json::json!(32);
            assert_eq!(group_rows(&spec).is_ok(), capacity == 4096, "{name}");
            spec.geometry["group_rows"] = serde_json::json!("64");
            assert!(group_rows(&spec).is_err(), "{name}");
            spec.geometry["abi_version"] = serde_json::json!(99);
            assert!(group_rows(&spec).is_err(), "{name}");
        }
        let mut unknown = spec();
        unknown.name = "exlr8_unknown".into();
        assert!(group_rows(&unknown).is_err());
    }

    fn entry() -> serde_json::Value {
        let shapes = [
            "[rows,6144]",
            "[E,16,3,3]",
            "[E,16,3]",
            "[512]",
            "[512,16]",
            "[rows,8]",
            "[rows,8]",
            "[512,16,2,6,16,128]",
            "[512,16,16,128]",
            "[512,16,6144]",
            "[6144]",
            "[rows,6144]",
        ];
        let dtypes = [
            "float16", "int64", "int32", "int32", "int32", "int32", "float32", "float32",
            "float16", "float32", "int8", "bfloat16",
        ];
        let pointers: Vec<_> = shapes.iter().enumerate().map(|(i, shape)| serde_json::json!({
            "name": spec().pointers[i], "shape": shape, "dtype": dtypes[i],
            "role": if (7..=9).contains(&i) { "scratch" } else if i == 11 { "out" } else { "in" }
        })).collect();
        serde_json::json!({"capacity_rows":64,"pointers":pointers,
            "scratch_bytes_at_capacity": {"fc1":805306368,"intermediate":33554432,"fc2":201326592}})
    }
    #[test]
    fn glue_contracts_reject_pointer_and_geometry_drift() {
        for grouped in [false, true] {
        for (name, kinds, pointers) in [
            (
                "exlr8_group",
                b"ii".as_slice(),
                vec![
                    ("ids", "int32", "[rows,8]", "in"),
                    ("x", "uint8", "[rows,6336]", "in"),
                    ("signs_in", "int8", "[6144]", "in"),
                    ("group_expert", "int32", "[512]", "out"),
                    ("route_slots", "int32", "[512,16]", "out"),
                    ("route_group_lane", "int32", "[rows,8]", "out"),
                    ("group_count", "int32", "[1]", "out"),
                    ("activation", "float16", "[rows,6144]", "out"),
                ],
            ),
        ] {
            let rows = if grouped { 4096 } else { 64 };
            let tile = if grouped { 64 } else { 16 };
            let pointers: Vec<_> = pointers.into_iter().map(|(n, d, shape, r)|
                (n, d, if n == "route_slots" { format!("[512,{tile}]") } else { shape.into() }, r)).collect();
            let name = if grouped { "exlr8_grouped_group" } else { name };
            let s = ProgramSpec {
                name: name.into(),
                pointers: pointers.iter().map(|p| p.0.into()).collect(),
                scalar_kinds: kinds.to_vec(),
                capacity_rows: rows,
                scratch: HashMap::new(),
                geometry: serde_json::json!({"abi_version":1,"max_rows":rows,"hidden":6144,
                    "ctas_per_sm":4,"max_ranks":6,"experts":256,"top_k":8,"max_groups":512,
                    "input_format":"fp8_k32","input_row_bytes":6336,"ingress_warps":8}),
            };
            let e = serde_json::json!({"capacity_rows":rows,"scratch_bytes_at_capacity":{},
                "pointers": pointers.iter().map(|(n,d,s,r)| serde_json::json!({
                    "name":n,"dtype":d,"shape":s,"role":r})).collect::<Vec<_>>()});
            let mut s = s;
            s.geometry["abi_version"] = serde_json::json!(2);
            if grouped { s.geometry["group_rows"] = serde_json::json!(tile); }
            validate_entry(&s, &e).unwrap();
            for experts in [1, 256, 288, 512] {
                let mut wide = s.clone();
                wide.geometry["experts"] = serde_json::json!(experts);
                validate_entry(&wide, &e).unwrap();
            }
            let mut too_wide = s.clone();
            too_wide.geometry["experts"] = serde_json::json!(513);
            assert!(validate_entry(&too_wide, &e).is_err());
            let mut flash = s.clone();
            flash.name = KernelFamily::GlmFlash.program(if grouped { KernelKind::GroupedGroup } else { KernelKind::Group }).into();
            flash.geometry["experts"] = serde_json::json!(288);
            flash.geometry["hidden"] = serde_json::json!(4096);
            flash.geometry["input_row_bytes"] = serde_json::json!(4224);
            let mut flash_entry = e.clone();
            for pointer in flash_entry["pointers"].as_array_mut().unwrap() {
                if let Some(shape) = pointer["shape"].as_str() {
                    pointer["shape"] = serde_json::json!(shape.replace("6336", "4224").replace("6144", "4096"));
                }
            }
            validate_entry(&flash, &flash_entry).unwrap();
            flash.geometry["experts"] = serde_json::json!(256);
            assert!(validate_entry(&flash, &flash_entry).is_err());
            for key in ["name", "dtype", "shape", "role"] {
                let mut bad = e.clone();
                bad["pointers"][0][key] = serde_json::json!("wrong");
                assert!(validate_entry(&s, &bad).is_err(), "{name}:{key}");
            }
            let mut bad = s.clone();
            bad.geometry["max_rows"] = serde_json::json!(0);
            assert!(validate_entry(&bad, &e).is_err());
            let mut bad = e.clone();
            bad["card_plans"] = serde_json::json!([]);
            assert!(validate_entry(&s, &bad).is_err());
        }
    }
    }
    #[test]
    fn flash_compute_requires_calibrated_geometry_and_clamp() {
        for kind in [KernelKind::Decode, KernelKind::Routed, KernelKind::Shared,
            KernelKind::GroupedDecode, KernelKind::GroupedA16, KernelKind::GroupedA8] {
            let mut s = spec();
            s.name = KernelFamily::GlmFlash.program(kind).into();
            let grouped = kind.is_grouped();
            let shared = kind == KernelKind::Shared;
            let rows = if matches!(kind, KernelKind::GroupedA16 | KernelKind::GroupedA8) { 4096 } else { 64 };
            let tile = if rows == 4096 { 64u64 } else { 16 };
            let groups = if shared { 4u64 } else if rows == 4096 { 800 } else { 512 };
            s.capacity_rows = rows;
            s.geometry["max_rows"] = serde_json::json!(rows);
            s.geometry["hidden"] = serde_json::json!(4096);
            s.geometry["fc1_splits"] = serde_json::json!(4);
            s.geometry["top_k"] = serde_json::json!(if shared { 1 } else { 8 });
            s.geometry["max_groups"] = serde_json::json!(groups);
            s.geometry["swiglu_limit"] = serde_json::json!(10.0);
            if kind != KernelKind::Decode {
                s.pointers.push("group_count".into());
                s.scalar_kinds = b"ii".to_vec();
            }
            if kind == KernelKind::Routed { s.geometry["serving_variant"] = serde_json::json!("k1_k23_coalesced"); }
            if grouped {
                s.geometry["group_rows"] = serde_json::json!(tile);
                s.pointers.extend(["unit_count", "unit_map"].map(String::from));
                s.scalar_kinds = b"iiii".to_vec();
            }
            if kind == KernelKind::GroupedDecode { s.geometry["fc2_partials"] = serde_json::json!("f32"); }
            if kind == KernelKind::GroupedA8 { s.geometry["activations"] = serde_json::json!("a8"); }
            s.scratch = HashMap::from([("fc1".into(), groups * 16 * 2 * 4 * tile * 128 * 4),
                ("intermediate".into(), groups * 16 * tile * 128 * 2),
                ("fc2".into(), groups * tile * 4096 * 4)]);
            validate(&s).unwrap();
            for (key, value) in [("hidden", 6144), ("units", 8), ("top_k", 6), ("swiglu_limit", 0)] {
                let mut bad = s.clone();
                bad.geometry[key] = serde_json::json!(value);
                assert!(validate(&bad).is_err(), "{kind:?}:{key}");
                bad.geometry.as_object_mut().unwrap().remove(key);
                assert!(validate(&bad).is_err(), "{kind:?}:missing {key}");
            }
            let mut wrong_family = s.clone();
            wrong_family.name = KernelFamily::Glm.program(kind).into();
            assert!(validate(&wrong_family).is_err());
        }
        assert_eq!(KernelFamily::from_geometry(4096, 16, 8, 288).unwrap(), KernelFamily::GlmFlash);
        assert!(KernelFamily::from_geometry(4096, 16, 8, 256).is_err());
    }

    #[test]
    fn legacy_compute_allows_only_absent_or_zero_clamp() {
        let mut s = spec();
        validate(&s).unwrap();
        s.geometry["swiglu_limit"] = serde_json::json!(0.0);
        validate(&s).unwrap();
        for clamp in [serde_json::json!(10.0), serde_json::json!(-1.0),
            serde_json::json!("0"), serde_json::Value::Null] {
            s.geometry["swiglu_limit"] = clamp;
            assert!(validate(&s).is_err());
        }
        for family in [KernelFamily::Glm, KernelFamily::GlmFlash] {
            let g = family.geometry();
            assert_eq!(KernelFamily::from_geometry(g.hidden, g.units, g.top_k, g.experts).unwrap(), family);
            assert!(KernelFamily::from_geometry(g.hidden, g.units, 1, g.experts).is_err());
        }
    }

    #[test]
    fn reply_producer_names_require_matching_numerical_metadata() {
        for family in [KernelFamily::Glm, KernelFamily::GlmFlash] {
            let prefix = if family == KernelFamily::Glm { "exlr8" } else { "exlr8_glmf" };
            let mut s = spec();
            s.name = format!("{prefix}_a16_serving_core");
            assert!(KernelFamily::of(&s).is_err());
            s.geometry["core_only"] = serde_json::json!(true);
            assert_eq!(KernelFamily::of(&s).unwrap(), (family, KernelKind::RoutedCore));
            s.name = format!("{prefix}_a16_serving");
            assert!(KernelFamily::of(&s).is_err());
            s.geometry.as_object_mut().unwrap().remove("core_only");
            for (base, kind) in [("a16_grouped", KernelKind::GroupedA16),
                ("a8_grouped", KernelKind::GroupedA8), ("a16_grouped64", KernelKind::GroupedDecode)] {
                for format in [1, 2] {
                    s.name = format!("{prefix}_{base}_reply{format}");
                    s.geometry["reply_format"] = serde_json::json!(format);
                    let resolved = KernelFamily::of(&s).unwrap();
                    assert_eq!(resolved.0, family);
                    assert_eq!(resolved.1.arithmetic_kind(), kind);
                    assert_eq!(resolved.1.reply_format(), Some(format));
                    s.geometry["reply_format"] = serde_json::json!(3 - format);
                    assert!(KernelFamily::of(&s).is_err());
                    s.name = format!("{prefix}_{base}");
                    assert!(KernelFamily::of(&s).is_err());
                }
            }
        }
    }

    #[test]
    fn reply_epilogues_validate_both_families_and_wire_formats() {
        for (prefix, hidden) in [("exlr8", 6144), ("exlr8_glmf", 4096)] {
            for format in [1, 2] {
                let stride = if format == 1 { hidden * 33 / 32 } else { hidden * 9 / 16 };
                let descriptors = [
                    ("route_group_lane", "int32", "[rows,8]".into()),
                    ("route_weights", "float32", "[rows,8]".into()),
                    ("fc2", "float32", "[fc2_scratch]".into()),
                    ("signs_out", "int8", format!("[{hidden}]")),
                    ("output", "uint8", format!("[rows,{stride}]")),
                    ("group_count", "int32", "[1]".into()),
                ];
                let s = ProgramSpec { name: format!("{prefix}_reply_{format}"),
                    pointers: descriptors.iter().map(|d| d.0.into()).collect(),
                    scalar_kinds: b"ii".to_vec(), capacity_rows: 64, scratch: HashMap::new(),
                    geometry: serde_json::json!({"hidden":hidden,"top_k":8,"format":format,
                        "row_bytes":stride,"variant":"k1_k23_coalesced"}) };
                let e = serde_json::json!({"capacity_rows":64,"scratch_bytes_at_capacity":{},
                    "pointers":descriptors.iter().enumerate().map(|(i, (name, dtype, shape))|
                        serde_json::json!({"name":name,"dtype":dtype,"shape":shape,
                            "role":if i == 4 { "out" } else { "in" }})).collect::<Vec<_>>(),
                    "scalars":[{"name":"rows","type":"int32"},{"name":"cta_limit","type":"int32"}]});
                validate_entry(&s, &e).unwrap();
                for key in ["name", "dtype", "shape", "role"] {
                    let mut bad = e.clone();
                    bad["pointers"][4][key] = serde_json::json!("wrong");
                    assert!(validate_entry(&s, &bad).is_err(), "{prefix}:{format}:{key}");
                }
                for key in ["hidden", "top_k", "format", "row_bytes", "variant"] {
                    let mut bad = s.clone();
                    bad.geometry.as_object_mut().unwrap().remove(key);
                    assert!(validate_entry(&bad, &e).is_err(), "{prefix}:{format}:{key}");
                }
                let mut bad = e.clone();
                bad["scratch_bytes_at_capacity"] = serde_json::json!({"new":1024});
                assert!(validate_entry(&s, &bad).is_err());
                let mut bad = e.clone();
                bad["scalars"][1]["type"] = serde_json::json!("int64");
                assert!(validate_entry(&s, &bad).is_err());
            }
        }
    }

    #[test]
    fn accepts_decode_v3() {
        validate(&spec()).unwrap();
        validate_entry(&spec(), &entry()).unwrap();
    }
    #[test]
    fn accepts_serving_v3_and_rejects_mixed_contracts() {
        let mut s = spec();
        s.name = "exlr8_a16_serving".into();
        s.geometry["abi_version"] = serde_json::json!(3);
        s.geometry["serving_variant"] = serde_json::json!("k1_k23_coalesced");
        s.pointers.push("group_count".into());
        s.scalar_kinds = b"ii".to_vec();
        let mut e = entry();
        e["pointers"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "name":"group_count", "shape":"[1]", "dtype":"int32", "role":"in"
            }));
        validate_entry(&s, &e).unwrap();
        for version in [0, 1, 2, 4] {
            let mut invalid = s.clone();
            invalid.geometry["abi_version"] = serde_json::json!(version);
            assert!(validate(&invalid).is_err());
        }
        let mut invalid = s.clone();
        invalid.scalar_kinds = b"iii".to_vec();
        assert!(validate(&invalid).is_err());
        let mut invalid = s.clone();
        invalid.pointers.pop();
        assert!(validate(&invalid).is_err());
        e["pointers"][12]["dtype"] = serde_json::json!("int64");
        assert!(validate_entry(&s, &e).is_err());
    }
    #[test]
    fn grouped_prefill_requires_exact_capacity_tiles_scratch_and_index_bounds() {
        for tile in [16u64, 32, 64] {
            let mut s = spec();
            s.name = "exlr8_a16_grouped".into();
            s.capacity_rows = 4096;
            s.geometry["max_rows"] = serde_json::json!(4096);
            s.geometry["group_rows"] = serde_json::json!(tile);
            s.geometry["ctas_per_sm"] = serde_json::json!(2);
            s.pointers.extend(["group_count", "unit_count", "unit_map"].map(String::from));
            s.scalar_kinds = b"iiii".to_vec();
            for bytes in s.scratch.values_mut() { *bytes *= tile / 16; }
            let mut e = entry();
            e["capacity_rows"] = serde_json::json!(4096);
            e["pointers"][4]["shape"] = serde_json::json!(format!("[512,{tile}]"));
            e["pointers"][7]["shape"] = serde_json::json!(format!("[512,16,2,6,{tile},128]"));
            e["pointers"][8]["shape"] = serde_json::json!(format!("[512,16,{tile},128]"));
            e["pointers"][9]["shape"] = serde_json::json!(format!("[512,{tile},6144]"));
            e["pointers"].as_array_mut().unwrap().push(serde_json::json!({
                "name":"group_count", "shape":"[1]", "dtype":"int32", "role":"in"}));
            for (name, shape) in [("unit_count", "[E]"), ("unit_map", "[E,16]")] {
                e["pointers"].as_array_mut().unwrap().push(serde_json::json!({
                    "name":name, "shape":shape, "dtype":"int32", "role":"in"}));
            }
            e["scratch_bytes_at_capacity"] = serde_json::to_value(&s.scratch).unwrap();
            validate_entry(&s, &e).unwrap();
            let mut a8 = s.clone();
            a8.name = "exlr8_a8_grouped".into();
            assert!(validate(&a8).is_err());
            a8.geometry["activations"] = serde_json::json!("a8");
            validate_entry(&a8, &e).unwrap();
            let mut bad_a16 = s.clone();
            bad_a16.geometry["activations"] = serde_json::json!("a8");
            assert!(validate(&bad_a16).is_err());
            let mut legacy = s.clone();
            legacy.scalar_kinds = b"iii".to_vec();
            assert!(validate(&legacy).is_err());
            legacy.scalar_kinds = b"ii".to_vec();
            assert!(validate(&legacy).is_err());
            legacy = s.clone();
            legacy.pointers.truncate(13);
            assert!(validate(&legacy).is_err());
            for (key, value) in [("max_rows",64), ("group_rows",8), ("units",i32::MAX as u64),
                ("top_k",i32::MAX as u64), ("max_groups",i32::MAX as u64)] {
                let mut bad = s.clone();
                bad.geometry[key] = serde_json::json!(value);
                assert!(validate(&bad).is_err(), "{key}");
            }
            let mut bad = s.clone();
            bad.capacity_rows = 64;
            assert!(validate(&bad).is_err());
            bad = s.clone();
            *bad.scratch.get_mut("fc2").unwrap() -= 1;
            assert!(validate(&bad).is_err());
            e["pointers"][4]["shape"] = serde_json::json!("[512,8]");
            assert!(validate_entry(&s, &e).is_err());
        }
    }

    #[test]
    fn routed_serving_requires_a_declared_compact_unit_variant() {
        let mut s = spec();
        s.name = "exlr8_a16_serving".into();
        s.pointers.push("group_count".into());
        s.scalar_kinds = b"ii".to_vec();
        let mut e = entry();
        e["pointers"].as_array_mut().unwrap().push(serde_json::json!({
            "name":"group_count", "shape":"[1]", "dtype":"int32", "role":"in"}));
        // The manifest must name the compact-unit variant it was built from.
        assert!(validate(&s).is_err());
        s.geometry["serving_variant"] = serde_json::json!("k1_k23_coalesced");
        validate_entry(&s, &e).unwrap();
        for variant in ["baseline", "k1_compact_prefetch", "k1_paired_fc1", "k1_reg_pair_b2", "k1_unknown", ""] {
            let mut bad = s.clone();
            bad.geometry["serving_variant"] = serde_json::json!(variant);
            assert!(validate(&bad).is_err(), "{variant}");
        }
        // Only the routed serving instance carries a variant.
        let mut shared = s.clone();
        shared.name = "exlr8_shared_a16_serving".into();
        shared.geometry["top_k"] = serde_json::json!(1);
        shared.geometry["max_groups"] = serde_json::json!(4);
        for bytes in shared.scratch.values_mut() { *bytes /= 128; }
        assert!(validate(&shared).is_err());
        shared.geometry.as_object_mut().unwrap().remove("serving_variant");
        validate(&shared).unwrap();
    }

    #[test]
    fn shared_serving_requires_single_route_and_four_groups() {
        let mut s = spec();
        s.name = "exlr8_shared_a16_serving".into();
        s.geometry["abi_version"] = serde_json::json!(3);
        s.geometry["top_k"] = serde_json::json!(1);
        s.geometry["max_groups"] = serde_json::json!(4);
        s.pointers.push("group_count".into());
        s.scalar_kinds = b"ii".to_vec();
        for size in s.scratch.values_mut() {
            *size /= 128;
        }
        validate(&s).unwrap();
        for (key, value) in [("top_k", 8), ("max_groups", 512), ("abi_version", 1)] {
            let mut bad = s.clone();
            bad.geometry[key] = serde_json::json!(value);
            assert!(validate(&bad).is_err(), "{key}");
        }
    }
    #[test]
    fn shared_group_requires_bf16_single_expert_ingress() {
        let descriptors = [
            ("ids", "int32", "[rows,1]", "in"),
            ("x", "bfloat16", "[rows,6144]", "in"),
            ("signs_in", "int8", "[6144]", "in"),
            ("group_expert", "int32", "[4]", "out"),
            ("route_slots", "int32", "[4,16]", "out"),
            ("route_group_lane", "int32", "[rows,1]", "out"),
            ("group_count", "int32", "[1]", "out"),
            ("activation", "float16", "[rows,6144]", "out"),
        ];
        let s = ProgramSpec {
            name: "exlr8_shared_group".into(),
            pointers: descriptors.iter().map(|p| p.0.into()).collect(),
            scalar_kinds: b"ii".to_vec(),
            capacity_rows: 64,
            scratch: HashMap::new(),
            geometry: serde_json::json!({"abi_version":2,"max_rows":64,"experts":1,
                "top_k":1,"max_groups":4,"hidden":6144,"input_format":"bf16",
                "input_row_bytes":12288,"ctas_per_sm":4,"ingress_warps":8}),
        };
        let e = serde_json::json!({"capacity_rows":64,"scratch_bytes_at_capacity":{},
            "pointers": descriptors.iter().map(|(n,d,s,r)| serde_json::json!({
                "name":n,"dtype":d,"shape":s,"role":r})).collect::<Vec<_>>()});
        validate_entry(&s, &e).unwrap();
        for (key, value) in [("experts", 256), ("top_k", 8), ("max_groups", 512),
            ("abi_version", 1), ("hidden", 127), ("input_row_bytes", 6336),
            ("ctas_per_sm", 0), ("ingress_warps", 4)] {
            let mut bad = s.clone();
            bad.geometry[key] = serde_json::json!(value);
            assert!(validate_entry(&bad, &e).is_err(), "{key}");
        }
        let mut bad = s.clone();
        bad.geometry["input_format"] = serde_json::json!("fp8_k32");
        assert!(validate_entry(&bad, &e).is_err());
    }
    #[test]
    fn rejects_malformed_entry() {
        let mutations = [
            ("/capacity_rows", serde_json::json!(4294967360u64)),
            ("/pointers/0/dtype", serde_json::json!("bfloat16")),
            ("/pointers/10/dtype", serde_json::json!("int32")),
            ("/pointers/11/dtype", serde_json::json!("float32")),
            ("/pointers/11/role", serde_json::json!("in")),
            ("/pointers/0/name", serde_json::json!("activation")),
            ("/pointers/1/shape", serde_json::json!("[E,16,3,2]")),
            ("/pointers/7/role", serde_json::json!("out")),
            (
                "/scratch_bytes_at_capacity/fc1",
                serde_json::json!("805306368"),
            ),
        ];
        for (path, value) in mutations {
            let mut e = entry();
            *e.pointer_mut(path).unwrap() = value;
            assert!(validate_entry(&spec(), &e).is_err(), "{path}");
        }
        let mut e = entry();
        e["card_plans"] = serde_json::json!([]);
        assert!(validate_entry(&spec(), &e).is_err());
        let mut e = entry();
        e["scratch_bytes_at_capacity"]["extra"] = serde_json::json!(0);
        assert!(validate_entry(&spec(), &e).is_err());
    }
    #[test]
    fn rejects_stale_geometry() {
        for key in [
            "abi_version",
            "hidden",
            "units",
            "top_k",
            "max_groups",
            "max_rows",
            "split_k_blocks",
            "fc1_splits",
            "ctas_per_sm",
        ] {
            let mut s = spec();
            s.geometry[key] = serde_json::json!(0);
            assert!(validate(&s).is_err(), "{key}");
        }
    }
    #[test]
    fn rejects_int32_task_and_route_overflow() {
        for (key, value) in [
            ("top_k", 1u64 << 26),
            ("max_groups", 1 << 27),
            ("units", 1 << 24),
            ("hidden", 1 << 30),
        ] {
            let mut s = spec();
            s.geometry[key] = serde_json::json!(value);
            let h = s.geometry["hidden"].as_u64().unwrap();
            let u = s.geometry["units"].as_u64().unwrap();
            let g = s.geometry["max_groups"].as_u64().unwrap();
            s.geometry["fc1_splits"] = serde_json::json!(h.div_ceil(1024));
            s.scratch = HashMap::from([
                ("fc1".into(), g * u * 2 * h.div_ceil(1024) * 16 * 128 * 4),
                ("intermediate".into(), g * u * 16 * 128 * 2),
                ("fc2".into(), g * 16 * h * 4),
            ]);
            let refusal = if key == "max_groups" {
                "task or route indexing exceeds signed int32"
            } else {
                "compute geometry or SwiGLU clamp disagrees with compiled family"
            };
            assert_eq!(validate(&s).unwrap_err().to_string(),
                format!("invalid EXLR8 program manifest: {refusal}"), "{key}");
        }
    }
    #[test]
    fn rejects_abi_and_scratch_drift() {
        let mut s = spec();
        s.pointers.swap(0, 1);
        assert!(validate(&s).is_err());
        let mut s = spec();
        s.scalar_kinds = b"iil".to_vec();
        assert!(validate(&s).is_err());
        let mut s = spec();
        s.scratch.insert("fc1".into(), 1);
        assert!(validate(&s).is_err());
        let mut s = spec();
        s.scratch.insert("extra".into(), 1);
        assert!(validate(&s).is_err());
        let mut s = spec();
        s.capacity_rows = 63;
        assert!(validate(&s).is_err());
        let mut s = spec();
        s.geometry["max_groups"] = serde_json::json!(i32::MAX);
        s.geometry["units"] = serde_json::json!(i32::MAX);
        assert!(validate(&s).is_err());
    }
}
