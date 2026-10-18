//! Fail-closed EXLR8 A16 ABI v1 contract. No serving allocations or routing here.
use super::programs::ProgramSpec;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
#[error("invalid EXLR8 program manifest: {0}")]
pub struct ManifestError(String);

fn invalid(message: impl Into<String>) -> ManifestError { ManifestError(message.into()) }

pub fn validate(spec: &ProgramSpec) -> Result<(), ManifestError> {
    let field = |name: &str| spec.geometry[name].as_u64()
        .filter(|&v| v > 0 && v <= i32::MAX as u64)
        .ok_or_else(|| invalid(format!("missing or invalid geometry.{name}")));
    if spec.name != "exlr8_a16_decode" || field("abi_version")? != 1 ||
        field("max_rows")? != 64 || spec.capacity_rows != 64 || field("split_k_blocks")? != 8 {
        return Err(invalid("name, ABI version, row capacity or fixed split order"));
    }
    let h = field("hidden")?;
    let u = field("units")?;
    let t = field("top_k")?;
    let g = field("max_groups")?;
    if h % 128 != 0 || field("fc1_splits")? != h.div_ceil(1024) {
        return Err(invalid("hidden alignment or FC1 split count"));
    }
    let expected = ["x", "pointers", "bits", "group_expert", "route_slots",
        "route_group_lane", "route_weights", "fc1", "intermediate", "fc2", "output"];
    if !spec.pointers.iter().map(String::as_str).eq(expected) || spec.scalar_kinds != b"iii" {
        return Err(invalid("pointer order or scalar ABI"));
    }
    let product = |factors: &[u64]| factors.iter().try_fold(1u64, |a, &b|
        a.checked_mul(b).filter(|&v| v <= i64::MAX as u64))
        .ok_or_else(|| invalid("scratch size overflow"));
    // Validate route/index addressing too, even though these are caller-owned inputs.
    product(&[64, t])?;
    let sizes = HashMap::from([
        ("fc1".to_string(), product(&[g, u, 2, h.div_ceil(1024), 16, 128, 4])?),
        ("intermediate".to_string(), product(&[g, u, 16, 128, 2])?),
        ("fc2".to_string(), product(&[g, 16, h, 4])?),
    ]);
    if spec.scratch != sizes { return Err(invalid("scratch capacities differ from ABI v1")); }
    Ok(())
}

pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<(), ManifestError> {
    validate(spec)?;
    if entry["capacity_rows"].as_u64() != Some(64) || entry.get("card_plans").is_some() {
        return Err(invalid("EXLR8 requires one SM-count-independent plan and exact capacity"));
    }
    let h = spec.geometry["hidden"].as_u64().unwrap();
    let u = spec.geometry["units"].as_u64().unwrap();
    let t = spec.geometry["top_k"].as_u64().unwrap();
    let g = spec.geometry["max_groups"].as_u64().unwrap();
    let s = h.div_ceil(1024);
    let shapes = [format!("[rows,{h}]"), format!("[E,{u},3,3]"), format!("[E,{u},3]"),
        format!("[{g}]"), format!("[{g},16]"), format!("[rows,{t}]"), format!("[rows,{t}]"),
        format!("[{g},{u},2,{s},16,128]"), format!("[{g},{u},16,128]"),
        format!("[{g},16,{h}]"), format!("[rows,{h}]")];
    let dtypes = ["float16", "int64", "int32", "int32", "int32", "int32", "float32",
        "float32", "float16", "float32", "float32"];
    let pointers = entry["pointers"].as_array().ok_or_else(|| invalid("pointer descriptors missing"))?;
    if pointers.len() != 11 || pointers.iter().enumerate().any(|(i,p)|
        p["dtype"].as_str() != Some(dtypes[i]) || p["shape"].as_str() != Some(&shapes[i]) ||
        p["role"].as_str() != Some(if (7..=9).contains(&i) { "scratch" } else if i == 10 { "out" } else { "in" })) {
        return Err(invalid("pointer shape, dtype or role"));
    }
    let scratch = entry["scratch_bytes_at_capacity"].as_object().ok_or_else(|| invalid("scratch missing"))?;
    if scratch.len() != 3 || scratch.iter().any(|(k,v)| v.as_u64() != spec.scratch.get(k).copied()) {
        return Err(invalid("invalid scratch fields"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> ProgramSpec {
        ProgramSpec { name: "exlr8_a16_decode".into(), pointers:
            ["x", "pointers", "bits", "group_expert", "route_slots", "route_group_lane",
             "route_weights", "fc1", "intermediate", "fc2", "output"].map(String::from).to_vec(),
            scalar_kinds: b"iii".to_vec(), capacity_rows: 64,
            geometry: serde_json::json!({"abi_version":1,"hidden":6144,"units":16,"top_k":8,
                "max_groups":512,"max_rows":64,"split_k_blocks":8,"fc1_splits":6}),
            scratch: HashMap::from([("fc1".into(), 805306368),
                ("intermediate".into(),33554432),("fc2".into(),201326592)]) }
    }
    fn entry() -> serde_json::Value {
        let shapes = ["[rows,6144]", "[E,16,3,3]", "[E,16,3]", "[512]", "[512,16]",
            "[rows,8]", "[rows,8]", "[512,16,2,6,16,128]", "[512,16,16,128]",
            "[512,16,6144]", "[rows,6144]"];
        let dtypes = ["float16", "int64", "int32", "int32", "int32", "int32",
            "float32", "float32", "float16", "float32", "float32"];
        let pointers: Vec<_> = shapes.iter().enumerate().map(|(i, shape)| serde_json::json!({
            "shape": shape, "dtype": dtypes[i],
            "role": if (7..=9).contains(&i) { "scratch" } else if i == 10 { "out" } else { "in" }
        })).collect();
        serde_json::json!({"capacity_rows":64,"pointers":pointers,
            "scratch_bytes_at_capacity": {"fc1":805306368,"intermediate":33554432,"fc2":201326592}})
    }
    #[test]
    fn accepts_v1() { validate(&spec()).unwrap(); validate_entry(&spec(), &entry()).unwrap(); }
    #[test]
    fn rejects_malformed_entry() {
        let mutations = [
            ("/capacity_rows", serde_json::json!(4294967360u64)),
            ("/pointers/0/dtype", serde_json::json!("bfloat16")),
            ("/pointers/1/shape", serde_json::json!("[E,16,3,2]")),
            ("/pointers/7/role", serde_json::json!("out")),
            ("/scratch_bytes_at_capacity/fc1", serde_json::json!("805306368")),
        ];
        for (path, value) in mutations {
            let mut e = entry(); *e.pointer_mut(path).unwrap() = value;
            assert!(validate_entry(&spec(), &e).is_err(), "{path}");
        }
        let mut e = entry(); e["card_plans"] = serde_json::json!([]);
        assert!(validate_entry(&spec(), &e).is_err());
        let mut e = entry(); e["scratch_bytes_at_capacity"]["extra"] = serde_json::json!(0);
        assert!(validate_entry(&spec(), &e).is_err());
    }
    #[test]
    fn rejects_stale_geometry() {
        for key in ["abi_version", "hidden", "units", "top_k", "max_groups", "max_rows",
                    "split_k_blocks", "fc1_splits"] {
            let mut s = spec(); s.geometry[key] = serde_json::json!(0);
            assert!(validate(&s).is_err(), "{key}");
        }
    }
    #[test]
    fn rejects_abi_and_scratch_drift() {
        let mut s = spec(); s.pointers.swap(0,1); assert!(validate(&s).is_err());
        let mut s = spec(); s.scalar_kinds = b"iil".to_vec(); assert!(validate(&s).is_err());
        let mut s = spec(); s.scratch.insert("fc1".into(),1); assert!(validate(&s).is_err());
        let mut s = spec(); s.scratch.insert("extra".into(),1); assert!(validate(&s).is_err());
        let mut s = spec(); s.capacity_rows = 63; assert!(validate(&s).is_err());
        let mut s = spec(); s.geometry["max_groups"] = serde_json::json!(i32::MAX);
        s.geometry["units"] = serde_json::json!(i32::MAX);
        assert!(validate(&s).is_err());
    }
}
