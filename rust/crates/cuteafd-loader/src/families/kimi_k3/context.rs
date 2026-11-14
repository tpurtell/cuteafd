//! Exact context-only allocations shared by the Kimi planner and native owner.
use anyhow::{ensure, Context, Result};
use cuteafd_core::MlaCacheFormat;
use crate::placement::ContextBuffers;

/// Runtime split choices share one captured executable and fixed device scalar.
pub const SPLIT_BUCKETS: [usize; 4] = [2, 4, 8, 16];

/// Context-only padding inventory. Native route edges come from the package,
/// which exports each dense trellis program's actual arithmetic boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBuckets {
    thresholds: Vec<usize>,
    plain: Vec<usize>,
}

impl DecodeBuckets {
    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self> {
        let programs = manifest["programs"].as_array().context("Kimi context program inventory missing")?;
        let mut thresholds = Vec::new();
        for program in programs.iter().filter(|p| p["name"].as_str().is_some_and(|name|
            name.starts_with("dense_trellis_") && name.contains("_trellis_"))
            && matches!(p["geometry"]["bits"].as_u64(), Some(4 | 6 | 8))) {
            let geometry = &program["geometry"];
            let routes = geometry["decode_route_thresholds"].as_array()
                .context("Kimi context native decode route thresholds missing")?;
            let routes = routes.iter().map(|v| -> Result<usize> {
                usize::try_from(v.as_u64().context("Kimi context invalid native decode route threshold")?)
                    .context("Kimi context decode route threshold overflow")
            }).collect::<Result<Vec<_>>>()?;
            ensure!(routes.iter().all(|&edge| (1..64).contains(&edge))
                && routes.windows(2).all(|pair| pair[0] < pair[1]),
                "Kimi context native decode routes must be sorted unique edges below 64");
            thresholds.extend(routes);
        }
        thresholds.sort_unstable();
        thresholds.dedup();
        let mut plain = if thresholds.is_empty() { vec![1, 4, 16, 64] } else { vec![1, 4, 64] };
        plain.extend(&thresholds);
        plain.sort_unstable();
        plain.dedup();
        let buckets = Self { thresholds, plain };
        for rows in 1..=64 { buckets.bucket(rows)?; }
        Ok(buckets)
    }

    fn for_dsa(mut self) -> Self {
        // Index append/frontier metadata has no padded rows. Admission and
        // runtime use the same exact-shape graph inventory for this variant.
        self.plain = (1..=ROWS).collect();
        self
    }

    pub fn plain(&self) -> &[usize] { &self.plain }

    pub fn bucket(&self, rows: usize) -> Result<usize> {
        ensure!((1..=64).contains(&rows), "Kimi context decode rows");
        let bucket = *self.plain.iter().find(|&&bucket| rows <= bucket)
            .context("Kimi context decode bucket absent")?;
        ensure!(self.thresholds.iter().all(|&edge| (rows <= edge) == (bucket <= edge)),
            "Kimi context padding crosses a native arithmetic boundary");
        Ok(bucket)
    }

    pub fn dispatch_rows(&self, rows: usize, speculative: bool) -> Result<usize> {
        let padded = self.bucket(rows)?;
        ensure!(!speculative || rows >= 2, "Kimi context speculative batch needs a draft row");
        Ok(if speculative { rows } else { padded })
    }

    pub fn graph_rows(&self, verify: usize, slots: usize, capacity: usize) -> Vec<usize> {
        let mut rows = self.plain.iter().copied()
            .chain(super::allocation::verify_graph_rows(verify, slots, capacity))
            .filter(|&rows| rows <= capacity).collect::<Vec<_>>();
        rows.sort_unstable();
        rows.dedup();
        rows
    }
}

/// Fork-derived tile choice for each exact row width, including verify graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePolicy(Vec<usize>);

impl TilePolicy {
    pub fn from_geometry(geometry: &serde_json::Value) -> Result<Self> {
        let routes = geometry["context_heads_per_tile"].as_array()
            .context("Kimi context tile policy missing")?;
        ensure!(routes.len() == ROWS, "Kimi context tile policy must cover rows 1..64");
        let routes = routes.iter().map(|value| -> Result<usize> {
            let tile = value.as_u64().context("Kimi context tile policy value is not an integer")?;
            ensure!(matches!(tile, 8 | 16), "Kimi context tile policy requires an unexported tile");
            Ok(tile as usize)
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self(routes))
    }

    pub fn tile(&self, rows: usize) -> Result<usize> {
        ensure!((1..=ROWS).contains(&rows), "Kimi context tile row outside policy");
        Ok(self.0[rows - 1])
    }

    pub fn index(&self, rows: usize) -> Result<usize> {
        Ok(usize::from(self.tile(rows)? == 16))
    }
}

const ROWS: usize = 64;
const HEADS: usize = 48;
const METADATA: [&str; 7] = ["context slots", "compact page tables", "shard lengths", "context offsets", "gather tables", "commit blocks", "active splits"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextLayout {
    buffers: ContextBuffers,
    workspace: Vec<(&'static str, usize)>,
    metadata_banks: usize,
    metadata_rows: usize,
    decode: DecodeBuckets,
    tiles: Option<TilePolicy>,
    dsa: bool,
}

/// Required names are format-specific: a BF16 package cannot admit FP8 context.
pub fn program_names(format: MlaCacheFormat) -> Vec<String> {
    let tag = if format == MlaCacheFormat::Fp8 { "_fp8" } else { "" };
    let mut names = vec!["kimik3_lse_combine2".into(), format!("kimik3_paged_staging_gather{tag}")];
    for head in [0, HEADS] { for tile in [8, 16] {
        names.push(format!("kimik3_mla_partial{tag}_h{head}_t{tile}_m64"));
    }}
    names
}

pub fn program_names_for_config(format: MlaCacheFormat, cfg: &super::config::KimiK3Config) -> Vec<String> {
    variant_program_names(format, cfg.dsa.is_some())
}

fn variant_program_names(format: MlaCacheFormat, dsa: bool) -> Vec<String> {
    if !dsa { return program_names(format); }
    let tag = format.program_suffix();
    let mut names = vec!["kimik31_lse_combine2".into(), "kimik31_dsa_candidate_merge".into(),
        format!("kimik31_paged_staging_gather{tag}")];
    for begin in [0, HEADS] {
        names.push(format!("kimik31_sparse_mla_partial{tag}_h{begin}_m64"));
    }
    names
}

impl ContextLayout {
    pub fn from_manifest(manifest: &serde_json::Value, format: MlaCacheFormat,
        max_context: usize, unit_rows: usize, metadata_banks: usize) -> Result<Self> {
        Self::from_variant(manifest, format, max_context, unit_rows, metadata_banks, false)
    }

    pub fn from_manifest_for_config(manifest: &serde_json::Value, format: MlaCacheFormat,
        max_context: usize, unit_rows: usize, metadata_banks: usize,
        cfg: &super::config::KimiK3Config) -> Result<Self> {
        Self::from_variant(manifest, format, max_context, unit_rows, metadata_banks, cfg.dsa.is_some())
    }

    fn from_variant(manifest: &serde_json::Value, format: MlaCacheFormat,
        max_context: usize, unit_rows: usize, metadata_banks: usize, dsa: bool) -> Result<Self> {
        ensure!((1..=super::COMPILED_CONTEXT).contains(&max_context)
            && (1..=super::allocation::UNIT_ROWS_MAX).contains(&unit_rows)
            && (1..=super::allocation::LANE_ROWS_MAX).contains(&metadata_banks),
            "Kimi context allocation capacity outside supported geometry");
        let programs = manifest["programs"].as_array().context("Kimi context program inventory missing")?;
        let mut scratch = 0;
        let mut tiles = None;
        for name in variant_program_names(format, dsa) {
            let entries = programs.iter().filter(|p| p["name"].as_str() == Some(&name)).collect::<Vec<_>>();
            ensure!(entries.len() == 1, "Kimi context requires one {name} program");
            if dsa && name.ends_with("lse_combine2") {
                ensure!(entries[0]["geometry"]["heads"] == HEADS
                    && entries[0]["geometry"]["has_sink"] == false, "Kimi DSA context combine geometry differs");
            }
            if dsa && name.ends_with("dsa_candidate_merge") {
                ensure!(entries[0]["geometry"]["topk"] == 2048
                    && entries[0]["geometry"]["unit_rows"] == ROWS, "Kimi DSA context merge geometry differs");
            }
            if dsa && name.contains("paged_staging_gather") {
                let geometry = &entries[0]["geometry"];
                ensure!(geometry["row_bytes"] == format.record_bytes() && geometry["page_rows"] == ROWS
                    && geometry["page_bytes"] == ROWS * format.record_bytes(),
                    "Kimi DSA context gather geometry differs");
            }
            if name.contains("_partial") {
                let bytes = entries[0]["scratch_bytes_at_capacity"]["scratch"].as_u64()
                    .with_context(|| format!("Kimi context missing {name} scratch extent"))?;
                let bytes = usize::try_from(bytes).context("Kimi context scratch extent overflow")?;
                let geometry = &entries[0]["geometry"];
                if dsa {
                    let begin = if name.contains("_h0_") { 0 } else { HEADS };
                    ensure!(geometry["heads"] == 96 && geometry["head_begin"] == begin
                        && geometry["head_count"] == HEADS && geometry["max_rows"] == ROWS
                        && geometry["page_size"] == 1 && geometry["topk"] == 2048
                        && geometry["num_splits"] == 16 && geometry["lse_base"] == 2
                        && geometry["norm_eps"] == 1e-5 && geometry["kv_cache"] == format.label(),
                        "Kimi DSA context {name} selected partial contract differs");
                } else {
                    let tile = if name.contains("_t8_") { 8 } else { 16 };
                    ensure!(geometry["heads_per_tile"] == tile, "Kimi context {name} tile geometry differs");
                    let policy = TilePolicy::from_geometry(geometry)?;
                    ensure!(tiles.as_ref().is_none_or(|expected| *expected == policy),
                        "Kimi context tile policies differ between programs");
                    tiles = Some(policy);
                    ensure!(geometry["num_splits"] == 16 && geometry["runtime_splits"] == true
                        && geometry["device_splits"] == true && geometry["fixed_split_grid"] == 16
                        && geometry["lse_base"] == 2,
                        "Kimi context {name} requires runtime splits (capacity 16) and base-2 LSE");
                }
                let expected = (ROWS * HEADS * 16 * 512 * 2).next_multiple_of(1024)
                    + ROWS * HEADS * 16 * 4;
                ensure!(bytes == expected, "Kimi context {name} scratch extent {bytes} differs from native {expected}");
                scratch = scratch.max(bytes);
            }
        }
        let half_pages = max_context.div_ceil(ROWS).div_ceil(2);
        let partial = ROWS * HEADS * (512 * 2 + 4);
        let mut workspace = vec![("own-head partial", partial), ("peer-head partial", partial), ("partial scratch", scratch),
            ("decode input", ROWS * 7168 * 2), ("decode output", ROWS * 7168 * 2)];
        if dsa {
            for name in ["DSA global scores", "DSA global indices", "DSA merged scores", "DSA merged indices"] {
                workspace.push((name, unit_rows.max(ROWS) * 2048 * 4));
            }
            workspace.push(("DSA local slots", ROWS * 2048 * 4));
            workspace.push(("DSA local lengths", ROWS * 4));
            let stride = (unit_rows.max(ROWS) * 4).next_multiple_of(256);
            workspace.push(("DSA merged lengths", stride * metadata_banks));
        }
        let metadata_rows = if dsa { unit_rows.max(ROWS) } else { ROWS };
        for (name, bytes) in METADATA.into_iter().zip([unit_rows * 8,
            metadata_rows * half_pages * 4, metadata_rows * 4, (metadata_rows + 1) * 4, 2 * half_pages * 4,
            3 * (unit_rows.div_ceil(ROWS) + 1) * 8, 4]) {
            let stride = bytes.max(256).next_multiple_of(256);
            workspace.push((name, stride.checked_mul(metadata_banks).context("Kimi context metadata banks overflow")?));
        }
        let workspace_bytes = workspace.iter().try_fold(0u64, |sum, (_, bytes)|
            sum.checked_add(*bytes as u64).context("Kimi context workspace overflow"))?;
        let buffers = ContextBuffers { staging_unit_bytes: (ROWS * format.record_bytes()) as u64,
            staging_unit_rows: ROWS as u64, compiled_extent: max_context as u64,
            query_row_bytes: (HEADS * 576 * 2) as u64, partial_row_bytes: (HEADS * (512 * 2 + 4)) as u64,
            candidate_row_bytes: if dsa { 2048 * 8 } else { 0 },
            decode_rows: ROWS as u64, lanes: 1, workspace_bytes, ..Default::default() };
        buffers.demands()?;
        ensure!(dsa || tiles.is_some(), "Kimi context tile policy missing");
        let decode = DecodeBuckets::from_manifest(manifest)?;
        let decode = if dsa { decode.for_dsa() } else { decode };
        Ok(Self { buffers, workspace, metadata_banks, metadata_rows, decode, tiles, dsa })
    }

    pub fn decode(&self) -> &DecodeBuckets { &self.decode }
    pub fn tiles(&self) -> Option<&TilePolicy> { self.tiles.as_ref() }
    pub fn dsa(&self) -> bool { self.dsa }
    pub fn buffers(&self) -> ContextBuffers { self.buffers }
    pub fn workspace(&self) -> Vec<(&'static str, usize)> { self.workspace.clone() }

    /// Preserve validated programs and row banks, shrinking only history owners.
    pub fn with_served_context(&self, context: usize) -> Result<Self> {
        ensure!((1..=self.buffers.compiled_extent as usize).contains(&context),
            "Kimi context exceeds prepared extent");
        let mut layout = self.clone();
        let half_pages = context.div_ceil(ROWS).div_ceil(2);
        for (name, bytes) in &mut layout.workspace {
            let raw = match *name {
                "compact page tables" => self.metadata_rows * half_pages * 4,
                "gather tables" => 2 * half_pages * 4,
                _ => continue,
            };
            *bytes = raw.max(256).next_multiple_of(256) * self.metadata_banks;
        }
        layout.buffers.compiled_extent = context as u64;
        layout.buffers.workspace_bytes = layout.workspace.iter().map(|(_, bytes)| *bytes as u64).sum();
        Ok(layout)
    }

    /// Each unit keeps its compact/gather/commit metadata until the wave drains.
    pub fn metadata_extent(&self, name: &str, bank: usize) -> Result<(usize, usize)> {
        ensure!((METADATA.contains(&name) || (self.dsa && name == "DSA merged lengths"))
            && bank < self.metadata_banks, "Kimi context metadata bank outside admission");
        let bytes = self.workspace.iter().find(|(key, _)| *key == name).context("Kimi context metadata absent")?.1;
        let stride = bytes / self.metadata_banks;
        Ok((bank * stride, stride))
    }
}

#[cfg(test)]
pub(crate) fn add_manifest_programs(manifest: &mut serde_json::Value) {
    for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
        for name in program_names(format) {
            let splits = 16;
            let bytes = if name.contains("_partial") {
                (ROWS * HEADS * splits * 512 * 2).next_multiple_of(1024) + ROWS * HEADS * splits * 4
            } else { 0 };
            // Combine is shared by the two record formats.
            let programs = manifest["programs"].as_array_mut().unwrap();
            if !programs.iter().any(|p| p["name"] == name) {
                programs.push(serde_json::json!({"name":name,"scratch_bytes_at_capacity":{"scratch":bytes},
                    "geometry":{"num_splits":16,"runtime_splits":true,"device_splits":true,"fixed_split_grid":16,"lse_base":2,
                        "heads_per_tile":if name.contains("_t8_") {8} else {16},
                        "context_heads_per_tile":(1..=64).map(|rows| if rows < 4 {8} else {16}).collect::<Vec<_>>()}}));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_manifest(routes: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"programs": [{"name":"dense_trellis_k6_trellis_mla_q_a_mcg",
            "geometry":{"bits":6,"decode_route_thresholds":routes}}]})
    }

    #[test]
    fn context_buckets_derive_from_native_routes_without_spanning_a_boundary() -> Result<()> {
        // Vary the dispatcher geometry: the engine must not know its edges.
        for routes in [vec![8, 16, 32, 48], vec![3, 9, 25, 51]] {
            let manifest = native_manifest(serde_json::json!(routes));
            let decode = DecodeBuckets::from_manifest(&manifest)?;
            for rows in 1..=64 {
                let bucket = decode.bucket(rows)?;
                assert!(bucket >= rows);
                assert!(routes.iter().all(|&edge| (rows <= edge) == (bucket <= edge)));
            }
            for &edge in &routes { assert!(decode.plain().contains(&edge)); }
            assert!(decode.bucket(0).is_err());
            assert!(decode.bucket(65).is_err());
        }
        let decode = DecodeBuckets::from_manifest(&native_manifest(serde_json::json!([8, 16, 32, 48])))?;
        assert_eq!(decode.plain(), [1, 4, 8, 16, 32, 48, 64]);
        assert_eq!(decode.graph_rows(8, 1, 64), [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64]);
        assert_eq!(decode.graph_rows(8, 16, 64), (1..=64).collect::<Vec<_>>());
        assert_eq!(decode.graph_rows(0, 16, 64), decode.plain());
        assert_eq!(decode.dispatch_rows(13, true)?, 13);
        assert_eq!(decode.dispatch_rows(13, false)?, 16);
        assert!(decode.dispatch_rows(1, true).is_err());
        for routes in [serde_json::Value::Null, serde_json::json!([8, 8]),
            serde_json::json!([16, 8]), serde_json::json!([0]), serde_json::json!([64]), serde_json::json!(["8"])] {
            assert!(DecodeBuckets::from_manifest(&native_manifest(routes)).is_err());
        }
        let empty = DecodeBuckets::from_manifest(&native_manifest(serde_json::json!([])))?;
        assert_eq!(empty.plain(), [1, 4, 16, 64]);
        Ok(())
    }

    #[test]
    fn context_buckets_union_each_native_width_and_group_without_crossing_edges() -> Result<()> {
        let baseline = DecodeBuckets::from_manifest(&native_manifest(serde_json::json!([8, 16, 32, 48])))?;
        let mut mixed = native_manifest(serde_json::json!([8, 16, 32, 48]));
        for (name, bits, edges) in [
            ("dense_trellis_k6_trellis_mla_q_b_mcg_k8", 8, vec![8, 32, 48]),
            ("dense_trellis_k6_trellis_kda_grouped_mcg_k8", 8, vec![16, 32, 48]),
            ("dense_trellis_k4_trellis_mtp_mcg", 4, vec![]),
        ] {
            mixed["programs"].as_array_mut().unwrap().push(serde_json::json!({
                "name":name,"geometry":{"bits":bits,"decode_route_thresholds":edges}}));
        }
        let union = DecodeBuckets::from_manifest(&mixed)?;
        assert_eq!(union, baseline);
        assert!(union.plain().contains(&8) && union.plain().contains(&16));
        assert_eq!(union.graph_rows(8, 16, 64), baseline.graph_rows(8, 16, 64));
        for rows in 1..=64 {
            let bucket = union.bucket(rows)?;
            for edges in [[8, 16, 32, 48].as_slice(), &[8, 32, 48], &[16, 32, 48]] {
                assert!(edges.iter().all(|&edge| (rows <= edge) == (bucket <= edge)));
            }
        }
        // A genuinely new edge changes ordinary padding, even when verify rows already include it.
        let old = DecodeBuckets::from_manifest(&native_manifest(serde_json::json!([16, 32, 48])))?;
        assert_eq!(old.bucket(5)?, 16);
        assert_eq!(union.bucket(5)?, 8);
        assert_eq!(union.bucket(9)?, 16);
        assert_eq!(union.graph_rows(8, 16, 64), old.graph_rows(8, 16, 64));
        Ok(())
    }

    #[test]
    fn context_tiles_follow_complete_package_policy_without_changing_graphs_or_storage() -> Result<()> {
        let mut manifest = native_manifest(serde_json::json!([8, 16, 32, 48]));
        add_manifest_programs(&mut manifest);
        let baseline = ContextLayout::from_manifest(&manifest, MlaCacheFormat::Bf16, 8192, 128, 3)?;
        for (rows, tile) in [(1, 8), (2, 8), (3, 8), (4, 16), (8, 16), (16, 16), (32, 16), (48, 16), (64, 16)] {
            assert_eq!(baseline.tiles().unwrap().tile(rows)?, tile);
            assert_eq!(baseline.tiles().unwrap().index(rows)?, usize::from(tile == 16));
        }
        assert!(baseline.tiles().unwrap().tile(0).is_err());
        assert!(baseline.tiles().unwrap().tile(65).is_err());
        let routes = (1..=64).map(|rows| if rows % 3 == 0 {8} else {16}).collect::<Vec<_>>();
        for program in manifest["programs"].as_array_mut().unwrap().iter_mut()
            .filter(|program| program["name"].as_str().unwrap().contains("_partial")) {
            program["geometry"]["context_heads_per_tile"] = serde_json::json!(routes);
        }
        let changed = ContextLayout::from_manifest(&manifest, MlaCacheFormat::Bf16, 8192, 128, 3)?;
        for rows in 1..=64 { assert_eq!(changed.tiles().unwrap().tile(rows)?, routes[rows - 1]); }
        assert_eq!(baseline.decode(), changed.decode());
        assert_eq!(baseline.workspace(), changed.workspace());
        assert_eq!(baseline.buffers(), changed.buffers());
        let index = manifest["programs"].as_array().unwrap().iter()
            .position(|p| p["name"] == "kimik3_mla_partial_h0_t8_m64").unwrap();
        for bad in [serde_json::Value::Null, serde_json::json!([]), serde_json::json!(vec![8; 63]),
            serde_json::json!(vec![4; 64]), serde_json::json!(vec!["8"; 64]), serde_json::json!(vec![8; 64])] {
            let mut broken = manifest.clone();
            broken["programs"][index]["geometry"]["context_heads_per_tile"] = bad;
            assert!(ContextLayout::from_manifest(&broken, MlaCacheFormat::Bf16, 8192, 128, 3).is_err());
        }
        let mut broken = manifest;
        broken["programs"][index]["geometry"]["heads_per_tile"] = 16.into();
        assert!(ContextLayout::from_manifest(&broken, MlaCacheFormat::Bf16, 8192, 128, 3).is_err());
        Ok(())
    }

    #[test]
    fn named_runtime_allocations_equal_planner_context_charges() {
        let mut manifest = serde_json::json!({"programs":[]});
        add_manifest_programs(&mut manifest);
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            for extent in [262_144, 524_288, 1_048_576] {
                let layout = ContextLayout::from_manifest(&manifest, format, extent, 2752, 3).unwrap();
                let buffers = layout.buffers();
                let demands = buffers.demands().unwrap();
                for gpu in 0..2 {
                    let charge = |name: &str| demands.iter().find(|d| d.gpu == gpu && d.group == name).unwrap().bytes;
                    assert_eq!(charge("context workspace"), layout.workspace().iter().map(|(_, n)| *n as u64).sum::<u64>());
                    assert_eq!(charge("context staging"), 2 * extent as u64 * format.record_bytes() as u64);
                    assert_eq!(charge("context exchange"), crate::placement::ContextExchangeLayout::new(&buffers).bytes().unwrap());
                }
                for name in METADATA {
                    let bytes = layout.workspace().into_iter().find(|(key, _)| *key == name).unwrap().1;
                    for bank in 0..3 {
                        let (offset, stride) = layout.metadata_extent(name, bank).unwrap();
                        assert_eq!(offset, bank * stride);
                        assert_eq!(stride % 256, 0);
                        assert_eq!(3 * stride, bytes);
                    }
                    assert!(layout.metadata_extent(name, 3).is_err());
                }
            }
        }
    }

    #[test]
    fn k31_context_admits_selected_partials_and_exact_candidate_owners() -> Result<()> {
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            let scratch = ROWS * HEADS * 16 * (512 * 2 + 4);
            let entries = variant_program_names(format, true).into_iter().map(|name| {
                let begin = if name.contains("_h0_") { 0 } else { HEADS };
                serde_json::json!({"name":name,"scratch_bytes_at_capacity":{"scratch":scratch},
                    "geometry":{"heads":96,"head_begin":begin,"head_count":HEADS,"max_rows":ROWS,
                        "page_size":1,"topk":2048,"num_splits":16,"lse_base":2,"norm_eps":1e-5,
                        "kv_cache":format.label(),"unit_rows":ROWS}})
            }).collect::<Vec<_>>();
            let mut manifest = serde_json::json!({"programs":entries});
            manifest["programs"][0]["geometry"] = serde_json::json!({"heads":HEADS,"has_sink":false});
            manifest["programs"][2]["geometry"] = serde_json::json!({"row_bytes":format.record_bytes(),
                "page_rows":ROWS,"page_bytes":ROWS * format.record_bytes()});
            let layout = ContextLayout::from_variant(&manifest, format, 8192, 2752, 3, true)?;
            assert!(layout.dsa() && layout.tiles().is_none());
            assert_eq!(layout.decode().plain(), (1..=ROWS).collect::<Vec<_>>());
            assert_eq!(layout.decode().graph_rows(0, 1, ROWS), (1..=ROWS).collect::<Vec<_>>());
            for rows in 1..=ROWS { assert_eq!(layout.decode().dispatch_rows(rows, false)?, rows); }
            assert_eq!(layout.buffers().candidate_row_bytes, 2048 * 8);
            let candidates = layout.workspace().iter().filter(|(name, _)| name.starts_with("DSA "))
                .map(|(_, bytes)| *bytes).sum::<usize>();
            assert_eq!(candidates, 2752 * 4 * 2048 * 4 + ROWS * (2048 * 4 + 4) + 2752 * 4 * 3);
            for unit in [64, 2752, 4096] {
                let layout = ContextLayout::from_variant(&manifest, format, 8192, unit, 3, true)?;
                let owners = layout.workspace();
                for name in ["DSA global scores", "DSA global indices", "DSA merged scores", "DSA merged indices"] {
                    assert_eq!(owners.iter().filter(|(key, _)| *key == name).count(), 1);
                    assert_eq!(owners.iter().find(|(key, _)| *key == name).unwrap().1, unit * 2048 * 4);
                }
                assert_eq!(layout.metadata_extent("DSA merged lengths", 2)?, (2 * unit * 4, unit * 4));
                assert_eq!(layout.metadata_extent("compact page tables", 0)?.1, unit * 64 * 4);
            }
            let prepared = ContextLayout::from_variant(&manifest, format, 1_048_576, 2752, 3, true)?;
            for extent in [1, 65, 8192, 131_201, 997_376, 1_048_576] {
                let resolved = prepared.with_served_context(extent)?;
                let fresh = ContextLayout::from_variant(&manifest, format, extent, 2752, 3, true)?;
                assert_eq!(resolved, fresh, "resolved history must equal fresh layout at {extent}");
                assert_eq!(resolved.metadata_extent("compact page tables", 0)?.1,
                    (2752 * extent.div_ceil(64).div_ceil(2) * 4).max(256).next_multiple_of(256));
                assert_eq!(resolved.buffers().demands()?[0].bytes,
                    2 * extent.div_ceil(64) as u64 * 64 * format.record_bytes() as u64);
            }
            assert!(prepared.with_served_context(0).is_err());
            assert!(prepared.with_served_context(1_048_577).is_err());
            assert_eq!(layout.buffers().workspace_bytes,
                layout.workspace().iter().map(|(_, bytes)| *bytes as u64).sum::<u64>());
            assert!(ContextLayout::from_manifest(&manifest, format, 8192, 2752, 3).is_err());
            for field in ["norm_eps", "lse_base", "topk", "head_begin", "page_size", "kv_cache"] {
                let mut broken = manifest.clone();
                broken["programs"][3]["geometry"][field] = serde_json::Value::Null;
                assert!(ContextLayout::from_variant(&broken, format, 8192, 2752, 3, true).is_err(), "{field}");
            }
            for name in variant_program_names(format, true) {
                let mut broken = manifest.clone();
                broken["programs"].as_array_mut().unwrap().retain(|program| program["name"] != name);
                assert!(ContextLayout::from_variant(&broken, format, 8192, 2752, 3, true).is_err(), "{name}");
            }
        }
        Ok(())
    }

    #[test]
    fn context_manifest_is_format_specific_and_fails_closed() {
        let mut manifest = serde_json::json!({"programs":[]});
        add_manifest_programs(&mut manifest);
        for name in program_names(MlaCacheFormat::Fp8) {
            let mut bad = manifest.clone();
            bad["programs"].as_array_mut().unwrap().retain(|p| p["name"] != name);
            assert!(ContextLayout::from_manifest(&bad, MlaCacheFormat::Fp8, 1024, 64, 1)
                .unwrap_err().to_string().contains(&name));
        }
        let mut bad = manifest.clone();
        bad["programs"].as_array_mut().unwrap().iter_mut().find(|p| p["name"] == "kimik3_mla_partial_fp8_h0_t8_m64")
            .unwrap()["scratch_bytes_at_capacity"]["scratch"] = 1.into();
        assert!(ContextLayout::from_manifest(&bad, MlaCacheFormat::Fp8, 1024, 64, 1).is_err());
        assert!(ContextLayout::from_manifest(&manifest, MlaCacheFormat::Fp8, usize::MAX, 64, 1).is_err());
    }
}
