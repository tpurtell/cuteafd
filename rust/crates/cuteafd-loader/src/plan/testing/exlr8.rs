//! Small native mixed-tier EXLR8 checkpoint for loader and daemon tests.
use std::{fs::File, io::Write, path::Path};
use serde_json::{json, Value};

fn write(
    path: &Path,
    metadata: Value,
    tensors: Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
) {
    let mut header = serde_json::Map::from_iter([("__metadata__".into(), metadata)]);
    let mut offset = 0;
    for (name, dtype, shape, bytes) in &tensors {
        header.insert(
            name.clone(),
            json!({"dtype":dtype,"shape":shape,
            "data_offsets":[offset,offset+bytes.len()]}),
        );
        offset += bytes.len();
    }
    let header = serde_json::to_vec(&header).unwrap();
    assert!(header.len() < 65528);
    let mut out = File::create(path).unwrap();
    out.write_all(&65528u64.to_le_bytes()).unwrap();
    out.write_all(&header).unwrap();
    out.write_all(&vec![b' '; 65528 - header.len()]).unwrap();
    for (_, _, _, bytes) in tensors {
        out.write_all(&bytes).unwrap();
    }
}

pub fn fixture(root: &Path, padded: bool) {
    std::fs::create_dir(root.join("experts")).unwrap();
    std::fs::create_dir(root.join("manifests")).unwrap();
    std::fs::write(
        root.join("config.json"),
        json!({
        "architectures":["GlmMoeDsaForCausalLM"], "hidden_size":128,
        "moe_intermediate_size":384,"n_routed_experts":2,"num_hidden_layers":4,
        "first_k_dense_replace":3})
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        root.join("manifests/exlr8-manifest-mixed.json"),
        json!({
        "format":"exlr8-v2-manifest","name":"mixed","version":"test","set_id":"b".repeat(64),
        "layers":{"3":{"gate":[[2,3,4],[4,2,3]],"up":[[4,3,2],[2,3,4]],"down":[3,4],
        "shared":{"gate":[3,4,3],"up":[4,3,4],"down":4}}}})
        .to_string(),
    )
    .unwrap();
    let mut tensors = vec![
        ("signs.in".into(), "I8", vec![128], vec![1; 128]),
        ("signs.out".into(), "I8", vec![128], vec![255; 128]),
    ];
    for k in 2..=4 {
        let bytes: usize = 128 * 16 * k;
        let stride = if padded {
            bytes.div_ceil(65536) * 65536
        } else {
            bytes
        };
        let mut weights = Vec::new();
        let mut base = 65536usize;
        for (prefix, es) in [("", 2), ("shared.", 1)] {
            if k == 2 && es == 1 {
                continue;
            }
            let mut offsets = Vec::new();
            let mut checksums = Vec::new();
            for e in 0..es {
                for p in 0..3 {
                    for u in 0..3 {
                        offsets.extend_from_slice(
                            &((base + (p * es * 3 + e * 3 + u) * stride) as u64).to_le_bytes(),
                        );
                        offsets.extend_from_slice(&(bytes as u64).to_le_bytes());
                        checksums.extend_from_slice(&0u64.to_le_bytes());
                    }
                }
            }
            tensors.push((
                format!("{prefix}k{k}.index"),
                "I64",
                vec![es, 3, 3, 2],
                offsets,
            ));
            tensors.push((
                format!("{prefix}k{k}.checksum"),
                "U64",
                vec![es, 3, 3],
                checksums,
            ));
            for (p, (projection, channel, block)) in [
                ("gate", "svh", "su_block"),
                ("up", "svh", "su_block"),
                ("down", "suh", "sv_block"),
            ]
            .iter()
            .enumerate()
            {
                let mut payload = Vec::new();
                let mut channels = Vec::new();
                for e in 0..es {
                    for u in 0..3 {
                        payload.extend(vec![(k * 10 + e * 3 + p + u) as u8; bytes]);
                        payload.extend(vec![0; stride - bytes]);
                        channels.extend(vec![(e + p + u) as u8; 256]);
                    }
                }
                let len = payload.len();
                weights.push((
                    format!("{prefix}trellis.{projection}"),
                    "U8",
                    vec![len],
                    payload,
                ));
                tensors.push((
                    format!("{prefix}k{k}.{projection}.{channel}"),
                    "F16",
                    vec![es, 384],
                    channels,
                ));
                tensors.push((
                    format!("{prefix}k{k}.{projection}.{block}"),
                    "F16",
                    vec![es, 1],
                    vec![k as u8; es * 2],
                ));
            }
            base += 3 * es * 3 * stride;
        }
        write(
            &root.join(format!("experts/exlr8-layer-003.k{k}.safetensors")),
            json!({"format":"exlr8-v2","scheme":"exlr8-e4m3-v2","layer":"3","k":k.to_string(),
                "gate_up_order":"grouped","down_order":"item128","pad_halves":padded.to_string()}),
            weights,
        );
    }
    write(
        &root.join("experts/exlr8-layer-003.index.safetensors"),
        json!({"format":"exlr8-v2","scheme":"exlr8-e4m3-v2","layer":"3",
            "gate_up_order":"grouped","down_order":"item128","widths":"2,3,4","shared_widths":"3,4"}),
        tensors,
    );
}

/// Ordinary attention exports plus the native shared expert's validated ABI.
pub fn glm_admission_programs() -> Value {
    let mut manifest = super::glm5_native_programs();
    let pointers = ["x", "pointers", "bits", "group_expert", "route_slots", "route_group_lane",
        "route_weights", "fc1", "intermediate", "fc2", "signs_out", "output", "group_count"];
    manifest["programs"].as_array_mut().unwrap().extend([
        json!({"name":"exlr8_shared_a16_serving", "capacity_rows":64,
            "pointers":pointers.iter().map(|name| json!({"name":name})).collect::<Vec<_>>(),
            "scalars":[{"type":"int32"},{"type":"int32"}],
            "geometry":{"abi_version":3,"hidden":6144,"units":16,"top_k":1,
                "max_groups":4,"max_rows":64,"group_rows":16,"split_k_blocks":8,"fc1_splits":6,"ctas_per_sm":4},
            "scratch_bytes_at_capacity":{"fc1":4*16*2*6*16*128*4,
                "intermediate":4*16*16*128*2,"fc2":4*16*6144*4}}),
        json!({"name":"exlr8_shared_group", "capacity_rows":64,
            "geometry":{"abi_version":2,"experts":1,"top_k":1,"max_groups":4,"max_rows":64,
                "hidden":6144,"input_format":"bf16","input_row_bytes":12288,"ctas_per_sm":4,"ingress_warps":8},
            "pointers":([("ids","int32","[rows,1]","in"), ("x","bfloat16","[rows,6144]","in"),
                ("signs_in","int8","[6144]","in"), ("group_expert","int32","[4]","out"),
                ("route_slots","int32","[4,16]","out"), ("route_group_lane","int32","[rows,1]","out"),
                ("group_count","int32","[1]","out"), ("activation","float16","[rows,6144]","out")]
                .into_iter().map(|(name,dtype,shape,role)| json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect::<Vec<_>>()),
            "scalars":[{"type":"int32"},{"type":"int32"}],"scratch_bytes_at_capacity":{}}),
    ]);
    manifest
}

/// Selected shared-expert headers plus vocabulary for physical GLM admission.
pub fn glm_admission_snapshot(root: &Path) {
    glm_snapshot(root);
    std::fs::create_dir(root.join("manifests")).unwrap();
    std::fs::write(root.join("manifests/exlr8-manifest-k4.json"), json!({
        "format":"exlr8-v2-manifest", "name":"k4", "version":"test", "set_id":"b".repeat(64),
        "layers":{"1":{"gate":vec![vec![4;16];256], "up":vec![vec![4;16];256],
            "down":vec![4;256], "shared":{"gate":vec![4;16], "up":vec![4;16], "down":4}}}
    }).to_string()).unwrap();
    let metadata = json!({"format":"exlr8-v2", "scheme":"exlr8-e4m3-v2", "layer":"1",
        "gate_up_order":"grouped", "down_order":"item128", "widths":"4", "shared_widths":"4"});
    let unit = 6144 * 16 * 4;
    let mut index = Vec::new();
    for projection in 0..3 {
        for u in 0..16 {
            index.extend_from_slice(&((65536 + (projection * 16 + u) * unit) as i64).to_le_bytes());
            index.extend_from_slice(&(unit as i64).to_le_bytes());
        }
    }
    let mut tensors = vec![
        ("signs.in".into(), "I8", vec![6144], vec![1;6144]),
        ("signs.out".into(), "I8", vec![6144], vec![255;6144]),
        ("shared.k4.index".into(), "I64", vec![1,3,16,2], index),
        ("shared.k4.checksum".into(), "U64", vec![1,3,16], vec![0;3*16*8]),
    ];
    for (projection, channel, block) in [("gate","svh","su_block"), ("up","svh","su_block"), ("down","suh","sv_block")] {
        tensors.push((format!("shared.k4.{projection}.{channel}"), "F16", vec![1,2048], vec![0;4096]));
        tensors.push((format!("shared.k4.{projection}.{block}"), "F16", vec![1,48], vec![0;96]));
    }
    write(&root.join("experts/exlr8-layer-001.index.safetensors"), metadata.clone(), tensors);
    let mut weight_metadata = metadata;
    weight_metadata["k"] = json!("4");
    weight_metadata["pad_halves"] = json!("false");
    write(&root.join("experts/exlr8-layer-001.k4.safetensors"), weight_metadata,
        ["gate","up","down"].into_iter().map(|projection|
            (format!("shared.trellis.{projection}"), "U8", vec![16*unit], vec![0;16*unit])).collect());
    write(&root.join("dense/exlr8-vocabulary.safetensors"),
        json!({"format":"exlr8-v2", "scheme":"exlr8-e4m3-v2"}), vec![
            ("model.embed_tokens.weight".into(), "BF16", vec![64,6144], vec![0;64*6144*2]),
            ("lm_head.weight".into(), "BF16", vec![64,6144], vec![0;64*6144*2]),
        ]);
}

/// A GLM 5.3-geometry native snapshot, headers only: `glm5_config` (hidden 6144, 64 heads, 2
/// layers), one output-sharded trellis dense record (`model.layers.0.mlp.gate_proj`) and a
/// layer-1 expert index. Enough for `native_layout`, which reads no payload.
pub fn glm_snapshot(root: &Path) {
    std::fs::create_dir(root.join("dense")).unwrap();
    std::fs::create_dir(root.join("experts")).unwrap();
    std::fs::write(root.join("config.json"), super::glm5_config().to_string()).unwrap();
    let metadata = |layer: &str| json!({"format":"exlr8-v2","scheme":"exlr8-e4m3-v2","layer":layer});
    let record: Vec<u8> = [1i64, 256, 128, 4, 1, 128, 0, 1].into_iter().flat_map(i64::to_le_bytes).collect();
    write(&root.join("dense/exlr8-dense-000.safetensors"), metadata("0"), vec![
        ("mlp.gate_proj.record".into(), "I64", vec![8], record),
        ("mlp.gate_proj.trellis".into(), "U8", vec![2, 8192], vec![0; 16384]),
        ("mlp.gate_proj.suh".into(), "F16", vec![128], vec![0; 256]),
        ("mlp.gate_proj.svh".into(), "F16", vec![256], vec![0; 512]),
        ("mlp.gate_proj.checksum".into(), "U64", vec![2], vec![0; 16]),
    ]);
    write(&root.join("experts/exlr8-layer-001.index.safetensors"), metadata("1"),
        vec![("k4.index".into(), "I64", vec![1], vec![0; 8])]);
}
