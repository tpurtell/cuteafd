//! Routed-expert pieces every family uses: layer identity and native resident
//! weights, per-wave execution and the host exchange, the EXL3 package path,
//! the expertd-native service, the exact FP8 expert packages and the TP2 RTX
//! expert halves (`rtx`).

pub(crate) mod execution;
pub(crate) mod exl3;
pub(crate) mod exlr8;
#[cfg(test)]
mod exlr8_decode_gate;
pub(crate) mod fp8;
pub(crate) mod layer;
pub(crate) mod local;
pub(crate) mod paired_load;
pub(crate) mod rtx;
pub(crate) mod service;
pub(crate) mod staging;
#[cfg(test)]
mod tp_gate;
#[cfg(test)]
mod tp_reference;

/// Shuffle checkpoint traffic independently per rank, but retain the caller's
/// layer-indexed residency. Admission must finish before invoking this loader.
fn load_in_rank_order<T>(layers: &[usize], rank: usize,
    mut load: impl FnMut(usize, usize) -> anyhow::Result<T>) -> anyhow::Result<Vec<T>> {
    let mut order: Vec<_> = (0..layers.len()).collect();
    let mut state = 0x6a09_e667_f3bc_c909u64.wrapping_add(rank as u64);
    for end in (1..order.len()).rev() {
        // SplitMix64 and rejection sampling make a reproducible Fisher-Yates
        // permutation without a process-global RNG or modulo bias.
        let bound = (end + 1) as u64;
        let threshold = bound.wrapping_neg() % bound;
        let slot = loop {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            if value >= threshold { break (value % bound) as usize; }
        };
        order.swap(end, slot);
    }
    let mut weights: Vec<_> = (0..layers.len()).map(|_| None).collect();
    for index in order { weights[index] = Some(load(index, layers[index])?); }
    Ok(weights.into_iter().map(|weight| weight.expect("every layer loaded once")).collect())
}

#[cfg(test)]
mod load_order_tests {
    use super::load_in_rank_order;
    use anyhow::Result;
    use std::{collections::BTreeSet, fs::OpenOptions, os::unix::fs::FileExt};
    use cuteafd_loader::{formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection, Slicing},
        plan::testing as fixture, RoutedExpertShape};

    #[test]
    fn rank_permutations_are_reproducible_and_preserve_sparse_layer_slots() -> Result<()> {
        let layers = [3, 8, 13, 21, 34, 55, 89, 144];
        let mut orders = BTreeSet::new();
        for rank in 0..6 {
            let mut order = Vec::new();
            let loaded = load_in_rank_order(&layers, rank, |index, layer| {
                assert_eq!(layers[index], layer);
                order.push(layer);
                Ok(layer)
            })?;
            assert_eq!(loaded, layers);
            let mut repeated = Vec::new();
            load_in_rank_order(&layers, rank, |_, layer| { repeated.push(layer); Ok(()) })?;
            assert_eq!(order, repeated);
            assert_ne!(order, layers);
            assert!(orders.insert(order));
        }
        assert!(load_in_rank_order::<()>(&[], 0, |_, _| unreachable!())?.is_empty());
        assert_eq!(load_in_rank_order(&[19], usize::MAX, |_, layer| Ok(layer))?, [19]);
        assert!(load_in_rank_order::<()>(&layers, 0, |_, _| anyhow::bail!("read failure")).is_err());
        Ok(())
    }

    fn exlr8_fixture(root: &std::path::Path, layers: &[usize]) -> Result<()> {
        use serde_json::json;
        use std::io::Write;
        fn write(path: &std::path::Path, mut metadata: serde_json::Value,
            tensors: Vec<(String, &'static str, Vec<usize>, Vec<u8>)>) -> Result<Vec<usize>> {
            metadata["format"] = json!("exlr8-v2");
            metadata["scheme"] = json!("exlr8-e4m3-v2");
            let mut header = serde_json::Map::from_iter([("__metadata__".into(), metadata)]);
            let mut offsets = vec![];
            let mut end = 0;
            for (name, dtype, shape, bytes) in &tensors {
                offsets.push(end);
                header.insert(name.clone(), json!({"dtype":dtype,"shape":shape,
                    "data_offsets":[end,end+bytes.len()]}));
                end += bytes.len();
            }
            let header = serde_json::to_vec(&header)?;
            let mut file = std::fs::File::create(path)?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            for (_, _, _, bytes) in tensors { file.write_all(&bytes)?; }
            Ok(offsets.into_iter().map(|offset| offset + 8 + header.len()).collect())
        }
        std::fs::create_dir(root.join("experts"))?;
        std::fs::create_dir(root.join("manifests"))?;
        std::fs::write(root.join("config.json"), json!({
            "architectures":["GlmMoeDsaForCausalLM"],"hidden_size":128,
            "moe_intermediate_size":384,"n_routed_experts":2,
            "num_hidden_layers":layers.iter().max().unwrap()+1,"first_k_dense_replace":3
        }).to_string())?;
        let mut choices = serde_json::Map::new();
        for &layer in layers {
            choices.insert(layer.to_string(), json!({"gate":[[2,3,4],[4,2,3]],
                "up":[[4,3,2],[2,3,4]],"down":[3,4],
                "shared":{"gate":[3,4,3],"up":[4,3,4],"down":4}}));
            let mut index = vec![("signs.in".into(), "I8", vec![128], vec![1;128]),
                ("signs.out".into(), "I8", vec![128], vec![255;128])];
            // Independent XXH64 constants for payload + channel + block bytes.
            for (k, checksum) in [(2,0x39d89079a9c254bcu64), (3,0x6c2ec4eafd85e1ed),
                (4,0xfe1b9862dc9faa1f)] {
                let bytes = 128*16*k;
                let offsets = write(&root.join(format!("experts/exlr8-layer-{layer:03}.k{k}.safetensors")),
                    json!({"layer":layer.to_string(),"k":k.to_string(),"gate_up_order":"grouped",
                        "down_order":"item128","pad_halves":"false"}),
                    ["gate","up","down"].into_iter().map(|projection|
                        (format!("trellis.{projection}"),"U8",vec![6*bytes],vec![k as u8;6*bytes])).collect())?;
                let mut ranges = vec![];
                for e in 0..2 { for &base in &offsets { for u in 0..3 {
                    ranges.extend_from_slice(&((base+(e*3+u)*bytes) as u64).to_le_bytes());
                    ranges.extend_from_slice(&(bytes as u64).to_le_bytes());
                }}}
                index.push((format!("k{k}.index"),"I64",vec![2,3,3,2],ranges));
                index.push((format!("k{k}.checksum"),"U64",vec![2,3,3],checksum.to_le_bytes().repeat(18)));
                for (projection,channel,block) in [("gate","svh","su_block"),
                    ("up","svh","su_block"),("down","suh","sv_block")] {
                    index.push((format!("k{k}.{projection}.{channel}"),"F16",vec![2,384],vec![1;1536]));
                    index.push((format!("k{k}.{projection}.{block}"),"F16",vec![2,1],vec![k as u8;4]));
                }
            }
            write(&root.join(format!("experts/exlr8-layer-{layer:03}.index.safetensors")),
                json!({"layer":layer.to_string(),"gate_up_order":"grouped","down_order":"item128",
                    "widths":"2,3,4","shared_widths":"3,4"}),index)?;
        }
        std::fs::write(root.join("manifests/exlr8-manifest-mixed.json"), json!({
            "format":"exlr8-v2-manifest","name":"mixed","version":"test","set_id":"b".repeat(64),
            "layers":choices}).to_string())?;
        Ok(())
    }

    #[test]
    fn shuffled_exlr8_native_bytes_budgets_and_failure_are_exact() -> Result<()> {
        use crate::shared::experts::exlr8::Exlr8Weights;
        use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
        fn read(plan: Residency) -> Result<(usize, Vec<u8>, [usize;5])> {
            let budget = Exlr8Weights::plan(&plan)?;
            let mut arena = vec![0;plan.resident_bytes];
            let mut put = |offset: usize, bytes: &[u8]| { arena[offset..offset+bytes.len()].copy_from_slice(bytes); };
            for (offset, bytes) in plan.signs_offsets.iter().zip(&plan.signs) { put(*offset,bytes); }
            let (pointers,bits) = plan.descriptors(0x10000)?;
            put(plan.pointers_offset,&pointers.into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>());
            put(plan.bits_offset,&bits.into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>());
            let (counts,mapping,_) = plan.compact_descriptors()?;
            for (offset, values) in plan.compact_offsets.into_iter().zip([counts,mapping]) {
                put(offset,&values.into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>());
            }
            for unit in &plan.loads {
                let mut staging = vec![0;unit.staging_bytes()];
                unit.read(&mut staging)?;
                let mut at = 0;
                for (source,destination) in unit.source.iter().zip(unit.destination) {
                    put(destination,&staging[at..at+source.bytes]); at += source.bytes;
                }
            }
            Ok((plan.layer,arena,[budget.resident_bytes,budget.device_staging_bytes,
                budget.pinned_host_bytes,budget.read_scratch_bytes,plan.host_bytes()]))
        }
        let layers = [3,8,13,21,34,55,89,144];
        let root = tempfile::tempdir()?;
        exlr8_fixture(root.path(),&layers)?;
        for rank in 0..6 {
            let plans = || layers.iter().map(|&layer| Residency::plan(root.path(),"mixed",layer,6,rank,
                ExpertSet::Routed).map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>();
            let sequential = plans()?.into_iter().map(read).collect::<Result<Vec<_>>>()?;
            let mut owned: Vec<_> = plans()?.into_iter().map(Some).collect();
            let shuffled = load_in_rank_order(&layers,rank,|index,layer| {
                let plan = owned[index].take().unwrap(); assert_eq!(plan.layer,layer); read(plan)
            })?;
            assert_eq!(shuffled,sequential,"EXLR8 rank {rank}");
            assert!(owned.iter().all(Option::is_none));
            let mut visits = 0;
            assert!(load_in_rank_order(&layers,rank,|_,layer| {
                visits += 1;
                if visits == 3 { anyhow::bail!("injected publication failure"); }
                read(Residency::plan(root.path(),"mixed",layer,6,rank,ExpertSet::Routed)?)
            }).is_err());
            assert_eq!(visits,3);
        }
        Ok(())
    }

    // Distinct finite payloads expose layer/rank/offset mix-ups that zero-filled
    // fixture files would hide. MCG and ModelOpt scalars retain valid values.
    fn fill_payloads(root: &std::path::Path) -> Result<()> {
        let path = root.join("model-00001-of-00001.safetensors");
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut size = [0; 8];
        file.read_exact_at(&mut size, 0)?;
        let size = u64::from_le_bytes(size);
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, 8)?;
        let header: serde_json::Value = serde_json::from_slice(&bytes)?;
        for (name, entry) in header.as_object().unwrap() {
            let start = entry["data_offsets"][0].as_u64().unwrap();
            let end = entry["data_offsets"][1].as_u64().unwrap();
            let tag = name.bytes().fold(0u8, |v, b| v.wrapping_add(b));
            let mut payload: Vec<_> = (0..end-start).map(|i| tag.wrapping_add(i as u8) & 31).collect();
            if name.ends_with(".mcg") { payload.copy_from_slice(&0xcbac1fedu32.to_le_bytes()); }
            if name.ends_with(".input_scale") || name.ends_with(".weight_scale_2") {
                payload.copy_from_slice(&1.0f32.to_le_bytes());
            }
            file.write_all_at(&payload, 8 + size + start)?;
        }
        Ok(())
    }

    #[test]
    fn shuffled_native_checkpoint_weights_and_admission_are_byte_identical() -> Result<()> {
        let shape = RoutedExpertShape { layers: 5, first_layer: 1, experts: 2, topk: 1,
            hidden: 128, intermediate: 768, draft_stages: 0, draft_experts: 0 };
        let layers = [1, 3, 4];
        for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4, ExpertFormat::Nvfp4, ExpertFormat::Bf16] {
            let root = tempfile::tempdir()?;
            let mut tensors = Vec::new();
            for layer in 1..5 { for expert in 0..shape.experts {
                for (projection, n, k) in [("gate", 768, 128), ("up", 768, 128), ("down", 128, 768)] {
                    let name = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}_proj");
                    tensors.extend(match format {
                        ExpertFormat::Fp8Block128 => fixture::fp8(&name, n, k, None),
                        ExpertFormat::Mxfp4 => fixture::mxfp4(&name, n, k),
                        ExpertFormat::Nvfp4 => fixture::nvfp4(&name, n, k),
                        ExpertFormat::Bf16 => vec![fixture::t(format!("{name}.weight"), "BF16", &[n, k])],
                    });
                }
            }}
            fixture::write_snapshot(root.path(), &serde_json::json!({}), &tensors, None);
            fill_payloads(root.path())?;
            let tensors = Fp8ExpertTensors::read(root.path(), shape)?;
            for rank in 0..6 {
                let load = |layer| -> Result<(Vec<u8>, [usize; 3])> {
                    let budget = tensors.layer_load(layer, 6, rank, Slicing::Blocks(128))?;
                    let mut bytes = Vec::new();
                    for expert in 0..shape.experts { for projection in Fp8Projection::ALL {
                        let (w, s) = tensors.slice_bytes_with(projection, 6, rank, Slicing::Blocks(128))?;
                        let (mut w, mut s) = (vec![0; w], vec![0; s]);
                        tensors.read_slice_with(layer, expert, projection, 6, rank, Slicing::Blocks(128),
                            &mut w, &mut s, &mut Vec::new())?;
                        bytes.extend(w); bytes.extend(s);
                    }}
                    Ok((bytes, [budget.resident, budget.pinned, budget.host]))
                };
                let sequential = layers.iter().map(|&layer| load(layer)).collect::<Result<Vec<_>>>()?;
                let shuffled = load_in_rank_order(&layers, rank, |_, layer| load(layer))?;
                assert_eq!(sequential, shuffled, "{format:?} rank {rank}");
                let totals = |weights: &[(Vec<u8>, [usize; 3])]| weights.iter().fold([0; 3], |a, (_, b)|
                    [a[0] + b[0], a[1].max(b[1]), a[2].max(b[2])]);
                assert_eq!(totals(&sequential), totals(&shuffled));
            }
        }
        Ok(())
    }

    #[test]
    fn shuffled_exl3_residency_and_admission_are_byte_identical() -> Result<()> {
        let root = tempfile::tempdir()?;
        let mut config = fixture::glm5_config();
        for (key, value) in [("hidden_size", 128), ("moe_intermediate_size", 768),
            ("n_routed_experts", 2), ("num_experts_per_tok", 1), ("num_hidden_layers", 5)] {
            config[key] = value.into();
        }
        config["indexer_types"] = serde_json::json!(["full", "shared", "shared", "shared", "shared"]);
        config["mlp_layer_types"] = serde_json::json!(["dense", "sparse", "sparse", "sparse", "sparse"]);
        config["quantization_config"] = fixture::exl3_compact(4);
        let mut tensors = Vec::new();
        for layer in 1..5 { for expert in 0..2 {
            for (projection, n, k) in [("gate", 768, 128), ("up", 768, 128), ("down", 128, 768)] {
                tensors.extend(fixture::exl3(&format!("model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"), n, k, 4));
            }
        }}
        fixture::write_snapshot(root.path(), &config, &tensors, None);
        fill_payloads(root.path())?;
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let layers = [1, 3, 4];
        for rank in 0..6 {
            let load = |layer| -> Result<(Vec<Vec<u8>>, [usize; 4])> {
                let selection = super::layer::ExpertLayer::BackboneExl3Tp { layer, rank, world: 6 };
                let plan = super::exl3::planned_residency(&catalog, selection)?;
                let budget = super::exl3::Exl3Weights::plan(&catalog, selection)?;
                let mut buffers: Vec<_> = plan.buffers.iter().map(|b| {
                    let mut bytes = vec![0; b.bytes.max(16)];
                    for (dst, word) in bytes.chunks_exact_mut(4).zip(&b.initial_words) {
                        dst.copy_from_slice(&word.to_le_bytes());
                    }
                    bytes
                }).collect();
                let mut scratch = vec![0; budget.read_scratch_bytes];
                for (index, job) in plan.loads.iter().enumerate() {
                    let mut bytes = vec![0; job.bytes];
                    plan.read_into(&catalog, index, &mut bytes, &mut scratch)?;
                    for &(buffer, offset) in &job.destinations {
                        buffers[buffer][offset..offset + job.bytes].copy_from_slice(&bytes);
                    }
                }
                Ok((buffers, [budget.resident_bytes, budget.device_staging_bytes,
                    budget.pinned_host_bytes, budget.read_scratch_bytes]))
            };
            let sequential = layers.iter().map(|&layer| load(layer)).collect::<Result<Vec<_>>>()?;
            let shuffled = load_in_rank_order(&layers, rank, |_, layer| load(layer))?;
            assert_eq!(sequential, shuffled, "EXL3 rank {rank}");
        }
        Ok(())
    }
}

/// Frozen compatibility set shared by native and EXL3 expert execution.
/// Launch capacity may vary; every decode variant must match canonical m1 sums.
#[derive(Debug)]
pub(crate) struct ExpertReductionPlan {
    decode: Vec<(usize, usize)>,
}

impl ExpertReductionPlan {
    pub(crate) fn new<S: Eq + std::fmt::Debug>(variants: &[(usize, S)]) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let canonical = &variants
            .iter()
            .find(|(capacity, _)| *capacity == 1)
            .context("expert package needs canonical m1 reduction")?
            .1;
        ensure!(
            variants.iter().all(|(capacity, _)| *capacity > 0),
            "expert export capacity must be positive"
        );
        let maximum = variants
            .iter()
            .map(|(capacity, _)| *capacity)
            .max()
            .context("empty expert reduction package")?;
        let mut decode: Vec<_> = variants
            .iter()
            .enumerate()
            .filter(|(_, (capacity, signature))| *capacity <= 80 && signature == canonical)
            .map(|(index, (capacity, _))| (*capacity, index))
            .collect();
        decode.sort_unstable();
        ensure!(decode.last().is_some_and(|&(capacity, _)| capacity >= maximum.min(80)),
            "expert package has no export covering {} decode rows with canonical m1 reduction {canonical:?}; admitted capacity/recipes: {variants:?}; export a matching capacity",
            maximum.min(80));
        Ok(Self { decode })
    }

    /// One compiled-recipe check for all native families and routed-expert owners.
    pub(crate) fn native(
        library: &cuteafd_ffi::NativeLibrary,
        infos: impl IntoIterator<Item = cuteafd_ffi::V41ExpertInfo>,
    ) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let variants = infos.into_iter().map(|info| {
            let recipe = library.native_expert_reduction(&info).with_context(|| {
                format!("native MXFP4 role {} m{} reduction export", info.role, info.capacity_rows)
            })?;
            if info.capacity_rows <= 80 {
                ensure!(recipe.plane_fold == 1 && recipe.implementation == 1,
                    "native MXFP4 role {} m{} decode requires ordered FP32 slice planes, got {recipe:?}",
                    info.role, info.capacity_rows);
            }
            Ok((info.capacity_rows as usize, recipe))
        }).collect::<anyhow::Result<Vec<_>>>()?;
        Self::native_recipes(&variants)
            .context("native MXFP4 decode differs from canonical m1 slice/fold recipe")
    }

    fn native_recipes(variants: &[(usize, cuteafd_ffi::V41ExpertReduction)]) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let canonical = &variants.iter().find(|(capacity, _)| *capacity == 1)
            .context("native package needs canonical m1 reduction")?.1;
        ensure!(variants.iter().all(|(capacity, recipe)| *capacity > 80 || recipe == canonical),
            "native decode capacities must share one slice/fold recipe");
        Self::new(variants)
    }

    pub(crate) fn execution_index(
        &self,
        rows: usize,
        required: usize,
        capacities: impl Iterator<Item = usize>,
    ) -> anyhow::Result<usize> {
        use anyhow::{ensure, Context};
        ensure!(rows > 0, "expert live rows must be positive");
        if rows <= 80 {
            return self
                .decode
                .iter()
                .find(|&&(capacity, _)| capacity >= rows)
                .map(|&(_, index)| index)
                .context("decode exceeds fixed expert reduction plan");
        }
        capacities
            .enumerate()
            .find(|&(_, capacity)| capacity >= required)
            .map(|(index, _)| index)
            .context("no expert execution for admitted rows")
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::ExpertReductionPlan;

    #[test]
    fn native_plan_rejects_mixed_decode_tilings_below_80() {
        use cuteafd_ffi::V41ExpertReduction;
        let recipe = V41ExpertReduction { version: 1, slice_width: 128, fc2_k_steps: 4,
            plane_count: 3, plane_fold: 1, implementation: 1, mma_k: 32,
            block_threads: 128, fc1_k_stage: 128 };
        let variants = [(1, recipe), (16, recipe), (80, recipe),
            (4096, V41ExpertReduction { slice_width: 64, fc2_k_steps: 2, plane_count: 5, ..recipe })];
        ExpertReductionPlan::native_recipes(&variants).unwrap();
        for index in 0..3 {
            let mut bad = variants;
            bad[index].1 = V41ExpertReduction { slice_width: 64, fc2_k_steps: 2,
                plane_count: 5, ..recipe };
            assert!(ExpertReductionPlan::native_recipes(&bad).is_err(), "capacity {}", bad[index].0);
        }
    }

    #[test]
    fn native_owned_state_order_freezes_serial_and_grouped_decode() {
        // Execution owns main, m1, then m80 (not ascending capacities).
        let variants = [(4096, 192), (1, 64), (80, 64)];
        let plan = ExpertReductionPlan::new(&variants).unwrap();
        for rows in 1..=80 {
            assert_eq!(
                plan.execution_index(rows, rows, variants.iter().map(|v| v.0))
                    .unwrap(),
                if rows == 1 { 1 } else { 2 }
            );
        }
        assert_eq!(
            plan.execution_index(81, 81, variants.iter().map(|v| v.0))
                .unwrap(),
            0
        );
        let small = [(16, 64), (1, 64)];
        let plan = ExpertReductionPlan::new(&small).unwrap();
        for rows in 1..=16 {
            assert_eq!(
                plan.execution_index(rows, rows, small.iter().map(|v| v.0))
                    .unwrap(),
                if rows == 1 { 1 } else { 0 }
            );
        }
        assert!(plan
            .execution_index(17, 17, small.iter().map(|v| v.0))
            .is_err());
        let serial = ExpertReductionPlan::new(&[(1, 64)]).unwrap();
        assert_eq!(serial.execution_index(1, 1, [1].into_iter()).unwrap(), 0);
    }

    #[test]
    fn shared_decode_selector_is_width_invariant_and_ignores_prefill_numerics() {
        let variants = [(1, 64), (16, 192), (80, 64), (4096, 192)];
        let plan = ExpertReductionPlan::new(&variants).unwrap();
        for rows in 1..=80 {
            let index = plan
                .execution_index(rows, rows, variants.iter().map(|v| v.0))
                .unwrap();
            assert_eq!(variants[index].1, 64);
        }
        assert_eq!(
            plan.execution_index(81, 81, variants.iter().map(|v| v.0))
                .unwrap(),
            3
        );
        assert!(plan
            .execution_index(0, 1, variants.iter().map(|v| v.0))
            .is_err());
        assert!(ExpertReductionPlan::new(&[(1, 64), (80, 192), (4096, 192)]).is_err());
        assert!(ExpertReductionPlan::new(&[(80, 64)]).is_err());
        // Metadata changes after admission cannot mutate the owned compatibility set.
        let mut changed = variants;
        changed[2].1 = 192;
        assert_eq!(
            plan.execution_index(80, 80, changed.iter().map(|v| v.0))
                .unwrap(),
            2
        );
        assert!(ExpertReductionPlan::new(&changed).is_err());
    }
}
