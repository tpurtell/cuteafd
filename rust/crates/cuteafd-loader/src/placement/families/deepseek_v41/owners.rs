//! Allocation-owner geometry for the default native V4.1 executor.
//! Projection scratch is read from the same AOT manifest as the linked native
//! library. Lanes own separate arenas; immutable weights and cache banks do not.
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use serde_json::Value;
use crate::placement::{Demand, ProgramSet};

const BLAS: u64 = 4 << 20;
const VOCAB: u64 = 129280;

/// Sealed WIP artifacts sit beside the library; installed images keep their
/// manifests in share/, including when the library is a host-ISA variant.
pub fn native_manifest(library: &std::path::Path, name: &str) -> Result<Value> {
    let directory = library.parent().context("V4.1 native library has no parent")?;
    let candidates = [Some(directory.join(name)),
        directory.parent().map(|root| root.join("share").join(name)),
        directory.parent().and_then(std::path::Path::parent)
            .map(|root| root.join("share").join(name))];
    let path = candidates.into_iter().flatten().find(|path| path.is_file())
        .with_context(|| format!("missing V4.1 native manifest {name} beside the library or in its image share directory"))?;
    serde_json::from_slice(&std::fs::read(&path)
        .with_context(|| format!("reading V4.1 native manifest {}", path.display()))?)
        .with_context(|| format!("parsing V4.1 native manifest {}", path.display()))
}

/// Single-RTX cache execution retains a per-layer descriptor/destination table.
/// Placed two-RTX producers use their own publication storage instead.
pub fn window_batch_bytes(rows: u64) -> Result<u64> {
    ensure!((1..=4096).contains(&rows), "invalid V4.1 window batch rows");
    Ok(40 * (64 + rows * 8 + 16 * 16))
}

/// Coordinator-owned Spark wave planes. Pinned intake belongs to its own domain.
pub fn transport_bytes(rows: u64, ranks: usize) -> Result<u64> {
    ensure!((1..=4096).contains(&rows) && matches!(ranks, 2 | 3 | 4 | 6),
        "invalid V4.1 transport geometry");
    Ok(rows * (ranks as u64 * 5120 * 2 + 2 * 5120 * 2))
}

/// Full vocabulary packing peak above the pre-weight sample. The first lane's
/// backbone/cache/index/embedding/Engram owners already exist; heads, taps,
/// transports, the second lane, draft, vision and marks do not.
pub fn vocabulary_load_phase(weights: &[Demand], target: &[Demand], packing: u64) -> Result<Demand> {
    ensure!(weights.iter().chain(target).all(|d| d.gpu == 0),
        "vocabulary startup phase requires the single-RTX owners");
    let retained = weights.iter().filter(|d| !matches!(d.group.as_str(), "lm_head" | "norm"));
    let lane = target.iter().filter(|d| d.group.starts_with("lane 0 ")
        && !matches!(d.group.as_str(), "lane 0 target head" | "lane 0 target taps"));
    let bytes = retained.chain(lane).try_fold(packing, |sum, d|
        sum.checked_add(d.bytes).context("V4.1 vocabulary phase overflow"))?;
    Ok(Demand::new(0, Category::Staging, "vocabulary packing", bytes, Basis::Formula))
}

pub struct NativePrograms<'a> {
    manifest: &'a Value,
    sms: u32,
}

impl<'a> NativePrograms<'a> {
    pub fn new(manifest: &'a Value, sms: u32) -> Result<Self> {
        ensure!(sms > 0, "V4.1 native plans require physical SM count");
        Ok(Self { manifest, sms })
    }

    pub fn scratch(&self, rows: u64, k: u64, n: u64) -> Result<u64> {
        ProgramSet::v41_fp8(self.manifest, &[(k, n)], rows, self.sms)?.scratch
            .into_iter().try_fold(0u64, |sum, (_, bytes)| sum.checked_add(bytes)
                .context("V4.1 projection arena overflow"))
    }

    fn shared_ffn(&self, rows: u64) -> Result<u64> {
        Ok(self.scratch(rows, 5120, 2304)? + self.scratch(rows, 2304, 5120)?
            + rows * 2304 * 6 + 4)
    }

    fn projection(&self, rows: u64, k: u64, n: u64) -> Result<u64> {
        Ok(self.scratch(rows, k, n)? + rows * (k + n) * 2 + 4)
    }

    fn main_context(&self, rows: u64) -> Result<u64> {
        Ok(self.projection(rows, 15360, 5120)? + 1152
            + rows * (10240 + 256 + 8 + 3 * 1024)
            + 3 * (self.projection(rows, 5120, 512)? - rows * 10240))
    }

    /// `cuteafd_fp8_w8a16_workspace`: the row-64 base plus runtime-SM split-K.
    pub fn vocabulary_scratch(&self, rows: u64, n: u64) -> Result<u64> {
        ensure!((1..=4096).contains(&rows) && matches!(n, 64640 | VOCAB),
            "invalid V4.1 vocabulary shape");
        let k = 5120u64;
        let tiles = n / 16;
        let blocks = k / 128;
        let wanted = (u64::from(self.sms) * 16).div_ceil(tiles).clamp(1, blocks);
        let splits = blocks.div_ceil(blocks.div_ceil(wanted));
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        Ok(align(2 * 64 * 4) + align(k.div_ceil(256) * 64 * 4) + align(k * 64 * 2)
            + if splits > 1 { align(splits * rows.min(64) * n * 4) } else { 0 })
    }

    /// Target/prefill lanes, excluding conditional local expert/TP2 FFN arenas,
    /// weights, cache/prefix banks, vision, and decoder-owned Spark transports.
    /// Both physical lanes remain allocated even at concurrency one.
    pub fn target(&self, rows: u64, ranks: usize, fp8_head: bool,
        window_batch: bool) -> Result<Vec<Demand>> {
        ensure!((1..=4096).contains(&rows) && matches!(ranks, 1 | 2)
            && (!window_batch || ranks == 1), "invalid V4.1 target owner geometry");
        let mut out = Vec::new();
        for rank in 0..ranks {
            for lane in 0..2 {
                let add = |out: &mut Vec<Demand>, group: &str, bytes| out.push(Demand::new(
                    rank as u8, Category::Workspace, format!("lane {lane} {group}"), bytes, Basis::Formula));
                let query = 4 + rows * (10240 + 2560 + 2560 + 65536 + 8 + 256)
                    + self.scratch(rows, 5120, 1280)? + self.scratch(rows, 1280, 32768)?;
                let output = 4 + rows * (65536 + 16384 + 10240 + 8 + 256)
                    + self.scratch(rows, 32768, 8192)? + self.scratch(rows, 8192, 5120)?;
                let sparse = rows * (64 * 2048 + 88)
                    + rows.min(cuteafd_core::V41_VERIFY_ROWS as u64) * (10 * 64 * 514 * 4 + 120);
                let shared = if ranks == 1 { self.shared_ffn(rows)? + rows * 20480 } else { 0 };
                add(&mut out, "backbone", 2 * rows * 112752 + query + output + sparse
                    + rows * (11825 + 5280) + shared);
                let window = self.scratch(rows, 5120, 512)? + 4 + rows * 13088;
                let compressor = |ratio2| BLAS + rows * if ratio2 {
                    10240 + 2048 + 2048 + 8 + 1024 + 264 + 512 + 68 + 8 + 288
                } else { 10240 + 1024 + 1024 + 264 + 512 + 68 + 8 + 288 };
                add(&mut out, "cache producers", window * if ranks == 1 { 40 } else { 20 }
                    + if rank == 0 { 3 * compressor(true) } else { 0 }
                    + if rank == ranks - 1 { compressor(false) } else { 0 });
                if window_batch {
                    add(&mut out, "window batch table", window_batch_bytes(rows)?);
                }
                let index_rows = if ranks == 2 && rank == 1 && lane == 1 { 80 } else { rows };
                let index_query = BLAS + self.scratch(index_rows, 1280, 4096)? + 4 + index_rows * 23560;
                let topk = 16384u64.div_ceil(2 * 512) * 512 * 8 * 2;
                let blocks = 2048u64.div_ceil(2 * 2048) * 2048 * 8 * 2;
                let selection = index_rows * (48 + 8 + 8 + 16384 * 8 + 16384 * 4
                    + 2048 * 4 + 2048 * 8 + 4096 + topk + 2048 + 16384 + blocks + 8192);
                let follower = if rank == ranks - 1 { index_rows * (2048 + 8192) } else { 0 };
                add(&mut out, "index", index_query + selection + follower);
                if rank == 0 {
                    add(&mut out, "embedding", rows * (4 + 40960 + 16 + 10240 + 4));
                    add(&mut out, "engram", rows * (24 * (256 + 8 + 512) + 1)
                        + 2 * (rows * (40960 * 2 + 51200 + 24 * 512 + 1)
                            + self.scratch(rows, 6144, 25600)? + 4));
                }
                if rank == ranks - 1 { add(&mut out, "target taps", index_rows * 30720); }
                let head_rows = cuteafd_core::V41_VERIFY_ROWS as u64;
                let normalization = head_rows * (40960 + 16 + 10240 + 10240);
                let sampler = head_rows * (64 + 64 + 2 * VOCAB.div_ceil(32) * 4 + 2048 * 4 + 256 * 12 + 32);
                let head = if ranks == 1 {
                    BLAS + normalization + head_rows * (VOCAB * 4 + 8) + sampler
                } else {
                    BLAS + head_rows * (10240 + VOCAB / 2 * 4 + if rank == 0 { 8 } else { 24 })
                        + if rank == 1 { normalization + head_rows * VOCAB * 4 + sampler } else { 0 }
                };
                add(&mut out, "target head", head + if fp8_head {
                    self.vocabulary_scratch(head_rows, VOCAB / ranks as u64)?
                } else { 0 });
            }
        }
        Ok(out)
    }

    /// Decoder-local default dSpark: request chains are independent by lane;
    /// main-context producers and committed windows are shared across them.
    /// `experts` is the published coordinator (not RTX backbone) manifest.
    pub fn dspark(&self, experts: &Value, rows: u64, requests: u64, width: u64,
        ranks: usize, fp8_head: bool) -> Result<Vec<Demand>> {
        ensure!((1..=4096).contains(&rows) && (1..=16).contains(&requests)
            && matches!(width, 5 | 7) && matches!(ranks, 1 | 2), "invalid V4.1 draft owners");
        ensure!(experts["role"].as_str() == Some("coordinator"), "dSpark expert manifest role mismatch");
        ensure!(experts["input_format"].as_str() == Some("bf16"),
            "dSpark owner inventory requires the native BF16 coordinator input");
        for (key, expected) in [("hidden", 5120), ("intermediate", 2304),
            ("kernel_intermediate", 2304), ("experts", 128), ("topk", 3)] {
            ensure!(experts["geometry"][key].as_u64() == Some(expected), "dSpark expert geometry mismatch: {key}");
        }
        let lanes = if requests == 1 { 1 } else { 2 };
        let members = requests.div_ceil(lanes);
        let live = members * width;
        let capacity = if live <= 16 { 16 } else if live <= 40 { 40 } else if live <= 80 { 80 } else { 256 };
        let variants = experts["variants"].as_array().context("missing dSpark expert variants")?;
        let matching: Vec<_> = variants.iter().filter(|v|
            v.get("requested_rows").unwrap_or(&v["capacity_rows"]).as_u64() == Some(capacity)).collect();
        ensure!(matching.len() == 1, "missing or duplicate dSpark expert capacity {capacity}");
        let expert = matching[0]["core_scratch_nbytes"].as_u64().context("missing dSpark expert scratch")?
            + capacity * (10240 + 3 * 8 + 2 * 10240);
        let output = self.projection(capacity, 8192, 5120)?
            + self.scratch(capacity, 32768, 8192)? + 4 + capacity * (65536 + 256);
        let attention = live * (1024 + 8) + 128 + output
            + self.projection(capacity, 5120, 1280)?
            + self.projection(capacity, 1280, 32768)?
            + self.projection(capacity, 5120, 512)?;
        let chain = 3 * (expert + self.shared_ffn(capacity)? + capacity * 128 * 4
            + 2 * capacity * 112752 + attention) + members * 4;
        let terminal = members * (width * VOCAB * 8 + (width + 2) * 4 + 16
            + width * 10240 + width * (40960 + 16)) + BLAS
            + members * (4 + 512 + VOCAB * 4) + BLAS
            + live * (10240 + 512 + 4);
        let decoder = ranks - 1;
        let mut out = Vec::new();
        let mut add = |rank, group, bytes| out.push(Demand::new(rank as u8, Category::Drafter,
            group, bytes, Basis::Formula));
        for rank in 0..ranks {
            let terminal = if ranks == 1 { terminal } else {
                BLAS + live * (10240 + VOCAB / 2 * 4 + if rank == 0 { 8 } else { 24 })
                    + if rank == decoder { terminal - BLAS } else { 0 }
            };
            add(rank, "dSpark request lanes", lanes * (terminal
                + if rank == decoder { chain } else { 0 }
                + if fp8_head { self.vocabulary_scratch(live, VOCAB / ranks as u64)? } else { 0 }));
        }
        add(decoder, "dSpark main context", self.main_context(rows)?
            + if lanes > 1 { self.main_context(80)? } else { 0 });
        add(decoder, "dSpark windows", 3 * (requests * 128 * 528 + rows * 1024 + 384));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest() -> Value {
        let matrices = [(5120, 1280), (1280, 32768), (32768, 8192), (8192, 5120),
            (5120, 2304), (2304, 5120), (5120, 512), (1280, 4096),
            (6144, 25600), (15360, 5120)];
        let variants = matrices.into_iter().flat_map(|(k, n)| {
            [1, 16, 80, 256, 1024, 4096].map(move |rows| json!({
                "capacity": rows, "input_dim": k, "output_dim": n,
                "label": format!("matrix_{k}_{n}_{rows}"), "scratch_bytes": rows + 1,
                "expected_m": if rows <= 80 { 16 } else { rows },
                "split_k_slices": 1, "all_rows": rows <= 80,
            }))
        }).collect::<Vec<_>>();
        json!({"variants": variants})
    }

    fn experts() -> Value {
        json!({"role":"coordinator", "input_format":"bf16", "geometry":{
            "hidden":5120, "intermediate":2304, "kernel_intermediate":2304,
            "experts":128, "topk":3}, "variants": ([16, 40, 80, 256].map(|rows|
                json!({"capacity_rows":rows, "core_scratch_nbytes":rows * 7})))})
    }

    fn owner(demands: &[Demand], rank: u8, group: &str) -> u64 {
        demands.iter().find(|d| d.gpu == rank && d.group == group).unwrap().bytes
    }

    #[test]
    fn vocabulary_phase_charges_only_the_first_lanes_live_owners() -> Result<()> {
        let demand = |group: &str, bytes| Demand::new(0, Category::Workspace, group, bytes, Basis::Formula);
        let weights = vec![demand("attention", 100), demand("engram-projections", 20),
            demand("embedding", 50), demand("lm_head", 300), demand("norm", 10)];
        let target = vec![demand("lane 0 backbone", 40), demand("lane 0 engram", 30),
            demand("lane 0 window batch table", 5), demand("lane 0 target head", 60),
            demand("lane 0 target taps", 70), demand("lane 1 backbone", 80)];
        let phase = vocabulary_load_phase(&weights, &target, 400)?;
        assert_eq!(phase.bytes, 400 + 100 + 20 + 50 + 40 + 30 + 5);
        assert_eq!(phase.category, Category::Staging);
        assert!(vocabulary_load_phase(&[demand("attention", u64::MAX)], &target, 400).is_err());
        assert!(vocabulary_load_phase(&[demand("attention", 1).on(1)], &target, 400).is_err());
        for ranks in [2, 3, 4, 6] {
            assert_eq!(transport_bytes(1024, ranks)?, 1024 * (ranks as u64 + 2) * 10240);
        }
        assert!(transport_bytes(0, 4).is_err());
        assert!(transport_bytes(4097, 4).is_err());
        assert!(transport_bytes(1024, 5).is_err());
        Ok(())
    }

    #[test]
    fn native_manifest_uses_the_matching_artifact_layout_and_refuses_bad_json() -> Result<()> {
        let root = tempfile::tempdir()?;
        let lib = root.path().join("lib");
        let share = root.path().join("share");
        let variant = lib.join("x86-64-v4");
        std::fs::create_dir_all(&variant)?;
        std::fs::create_dir(&share)?;
        let name = "V41_FP8_AOT.json";
        assert!(native_manifest(&lib.join("native.so"), name).is_err());
        std::fs::write(share.join(name), br#"{"source":"installed"}"#)?;
        for directory in [&lib, &variant] {
            assert_eq!(native_manifest(&directory.join("native.so"), name)?["source"], "installed");
        }
        std::fs::write(lib.join(name), br#"{"source":"sealed"}"#)?;
        assert_eq!(native_manifest(&lib.join("native.so"), name)?["source"], "sealed");
        std::fs::write(lib.join(name), b"invalid")?;
        assert!(native_manifest(&lib.join("native.so"), name).is_err());
        Ok(())
    }

    #[test]
    fn target_retains_two_lanes_and_the_decoder_replay_extent() -> Result<()> {
        let manifest = manifest();
        let programs = NativePrograms::new(&manifest, 188)?;
        for ranks in [1, 2] {
            let narrow = programs.target(256, ranks, false, ranks == 1)?;
            let wide = programs.target(1024, ranks, false, ranks == 1)?;
            for rank in 0..ranks as u8 {
                assert_eq!(owner(&narrow, rank, "lane 0 target head"),
                    owner(&wide, rank, "lane 1 target head"));
                assert_eq!(owner(&wide, rank, "lane 0 backbone"),
                    owner(&wide, rank, "lane 1 backbone"));
                if ranks == 2 && rank == 1 {
                    assert_eq!(owner(&narrow, rank, "lane 1 index"),
                        owner(&wide, rank, "lane 1 index"));
                    assert_eq!(owner(&wide, rank, "lane 1 target taps"), 80 * 30720);
                    assert!(owner(&wide, rank, "lane 0 index") > owner(&wide, rank, "lane 1 index"));
                } else {
                    assert_eq!(owner(&wide, rank, "lane 0 index"),
                        owner(&wide, rank, "lane 1 index"));
                }
            }
            assert!(wide.iter().all(|d| d.category == Category::Workspace));
            assert!(!wide.iter().any(|d| d.gpu != 0 && d.group.ends_with("engram")));
            let fp8 = programs.target(1024, ranks, true, ranks == 1)?;
            for rank in 0..ranks as u8 {
                for lane in 0..2 {
                    let group = format!("lane {lane} target head");
                    assert_eq!(owner(&fp8, rank, &group) - owner(&wide, rank, &group), 660992);
                }
            }
        }
        assert!(programs.target(0, 1, true, false).is_err());
        assert!(programs.target(4097, 2, true, false).is_err());
        assert!(programs.target(256, 3, true, false).is_err());
        assert!(programs.target(256, 2, true, true).is_err());
        let off = programs.target(1024, 1, false, false)?;
        let on = programs.target(1024, 1, false, true)?;
        assert_eq!(on.iter().map(|d| d.bytes).sum::<u64>()
            - off.iter().map(|d| d.bytes).sum::<u64>(), 2 * 340480);
        for rows in [1, 80, 256, 1024, 4096] {
            assert_eq!(window_batch_bytes(rows)?, 40 * (320 + 8 * rows));
        }
        assert!(window_batch_bytes(0).is_err());
        assert!(window_batch_bytes(4097).is_err());
        Ok(())
    }

    #[test]
    fn draft_chains_are_lane_local_but_windows_and_producers_are_shared() -> Result<()> {
        let manifest = manifest();
        let programs = NativePrograms::new(&manifest, 188)?;
        let experts = experts();
        for ranks in [1, 2] {
            let one = programs.dspark(&experts, 1024, 1, 5, ranks, false)?;
            let two = programs.dspark(&experts, 1024, 2, 5, ranks, false)?;
            let decoder = (ranks - 1) as u8;
            for rank in 0..ranks as u8 {
                assert_eq!(owner(&two, rank, "dSpark request lanes"),
                    2 * owner(&one, rank, "dSpark request lanes"));
            }
            assert_eq!(owner(&two, decoder, "dSpark main context")
                - owner(&one, decoder, "dSpark main context"), programs.main_context(80)?);
            assert_eq!(owner(&two, decoder, "dSpark windows")
                - owner(&one, decoder, "dSpark windows"), 3 * 128 * 528);
            for requests in 1..=16 {
                for width in [5, 7] {
                    let demands = programs.dspark(&experts, 1024, requests, width, ranks, true)?;
                    assert!(demands.iter().all(|d| d.category == Category::Drafter));
                    assert_eq!(owner(&demands, decoder, "dSpark windows"),
                        3 * (requests * 128 * 528 + 1024 * 1024 + 384));
                }
            }
        }
        for (key, value) in [("input_format", json!("fp8")), ("role", json!("rtx_backbone"))] {
            let mut bad = experts.clone();
            bad[key] = value;
            assert!(programs.dspark(&bad, 256, 1, 5, 1, false).is_err());
        }
        let mut bad = experts.clone();
        bad["geometry"]["kernel_intermediate"] = 2560.into();
        assert!(programs.dspark(&bad, 256, 1, 5, 1, false).is_err());
        let mut duplicate = experts.clone();
        let variant = duplicate["variants"][0].clone();
        duplicate["variants"].as_array_mut().unwrap().push(variant);
        assert!(programs.dspark(&duplicate, 256, 1, 5, 1, false).is_err());
        Ok(())
    }

    #[test]
    fn vocabulary_workspace_matches_native_alignment_and_physical_sm_split() -> Result<()> {
        let manifest = manifest();
        for sms in [170, 188] {
            let programs = NativePrograms::new(&manifest, sms)?;
            for rows in [1, 16, 64, 80, 4096] {
                for n in [64640, VOCAB] {
                    assert_eq!(programs.vocabulary_scratch(rows, n)?, 512 + 5120 + 655360);
                }
            }
            assert!(programs.vocabulary_scratch(0, VOCAB).is_err());
            assert!(programs.vocabulary_scratch(4097, VOCAB).is_err());
            assert!(programs.vocabulary_scratch(64, u64::MAX).is_err());
        }
        let programs = NativePrograms::new(&manifest, 1024)?;
        assert_eq!(programs.vocabulary_scratch(64, 64640)?, 660992 + 5 * 64 * 64640 * 4);
        assert!(NativePrograms::new(&manifest, 0).is_err());
        Ok(())
    }
}
