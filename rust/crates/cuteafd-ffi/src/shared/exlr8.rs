//! Fail-closed EXLR8 fused ingress ABI v2 and hidden-domain A16 ABI v3 contracts.
use super::programs::ProgramSpec;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
#[error("invalid EXLR8 program manifest: {0}")]
pub struct ManifestError(String);

fn invalid(message: impl Into<String>) -> ManifestError {
    ManifestError(message.into())
}

pub fn validate(spec: &ProgramSpec) -> Result<(), ManifestError> {
    let field = |name: &str| {
        spec.geometry[name]
            .as_u64()
            .filter(|&v| v > 0 && v <= i32::MAX as u64)
            .ok_or_else(|| invalid(format!("missing or invalid geometry.{name}")))
    };
    let shared = spec.name == "exlr8_shared_a16_serving";
    let grouped_a8 = spec.name == "exlr8_a8_grouped";
    let grouped = spec.name == "exlr8_a16_grouped" || grouped_a8;
    let serving = shared || grouped || spec.name == "exlr8_a16_serving";
    let activations = spec.geometry.get("activations").and_then(|v| v.as_str());
    if (grouped_a8 && activations != Some("a8"))
        || (!grouped_a8 && activations.is_some_and(|v| v != "a16"))
    {
        return Err(invalid("grouped activation mode mismatch"));
    }
    let max_rows = if grouped { 4096 } else { 64 };
    let group_rows = if grouped {
        let rows = field("group_rows")?;
        if !matches!(rows, 16 | 32 | 64) {
            return Err(invalid("grouped row tile must be 16, 32 or 64"));
        }
        rows
    } else {
        if spec.geometry.get("group_rows").is_some_and(|r| r.as_u64() != Some(16)) {
            return Err(invalid("decode requires 16-row groups"));
        }
        16
    };
    if (!serving && spec.name != "exlr8_a16_decode")
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
    let scalars: &[u8] = if serving && !grouped { b"ii" } else { b"iii" };
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
    if matches!(
        spec.name.as_str(),
        "exlr8_group" | "exlr8_shared_group" | "exlr8_grouped_group"
    ) {
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
    let r = spec.geometry["group_rows"].as_u64().unwrap_or(16);
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
    if matches!(
        spec.name.as_str(),
        "exlr8_a16_serving"
            | "exlr8_shared_a16_serving"
            | "exlr8_a16_grouped"
            | "exlr8_a8_grouped"
    ) {
        shapes.push("[1]".into());
        dtypes.push("int32");
    }
    if matches!(spec.name.as_str(), "exlr8_a16_grouped" | "exlr8_a8_grouped") {
        shapes.extend(["[E]".into(), format!("[E,{u}]")]);
        dtypes.extend(["int32", "int32"]);
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
    let descriptors: Vec<(&str, &str, String, &str)> = {
            let e = field("experts")?;
            let t = field("top_k")?;
            let g = field("max_groups")?;
            let h = field("hidden")?;
            let shared = spec.name == "exlr8_shared_group";
            let grouped = spec.name == "exlr8_grouped_group";
            let r = if grouped { field("group_rows")? } else { 16 };
            if (grouped && !matches!(r, 16 | 32 | 64))
                || (!grouped && spec.geometry.get("group_rows").is_some_and(|r| r.as_u64() != Some(16))) {
                return Err(invalid("grouping row tile"));
            }
            let (format, dtype, row, row_bytes) = if shared {
                ("bf16", "bfloat16", h, h * 2)
            } else {
                ("fp8_k32", "uint8", h + h / 32, h + h / 32)
            };
            if spec.name == "exlr8_shared_group" && (e != 1 || t != 1 || g != 4) {
                return Err(invalid(
                    "shared grouping requires experts=1, top_k=1, max_groups=4",
                ));
            }
            if rows != if grouped { 4096 } else { 64 }
                || e > 256
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
    fn accepts_decode_v3() {
        validate(&spec()).unwrap();
        validate_entry(&spec(), &entry()).unwrap();
    }
    #[test]
    fn accepts_serving_v3_and_rejects_mixed_contracts() {
        let mut s = spec();
        s.name = "exlr8_a16_serving".into();
        s.geometry["abi_version"] = serde_json::json!(3);
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
            s.scalar_kinds = b"iii".to_vec();
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
            assert!(
                validate(&s).unwrap_err().to_string().contains("int32"),
                "{key}"
            );
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
