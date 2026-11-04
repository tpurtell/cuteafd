//! Checkpoint-native, unit-major EXLR8 residency and bounded range reads.
//! Arena offsets are independent of a CUDA address; the owner binds them only
//! after exact weight/workspace admission and retains the arena for graph life.
use super::{Header, Layer, number, owner};
use crate::plan::checkpoint::read_json;
use anyhow::{Context, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

type Result<T> = std::result::Result<T, ResidencyError>;

fn number_string(metadata: &serde_json::Value, key: &str) -> anyhow::Result<usize> {
    metadata[key].as_str().with_context(|| format!("EXLR8 slice {key} missing"))?
        .parse().with_context(|| format!("EXLR8 slice {key} invalid"))
}

#[derive(serde::Deserialize)]
struct SelectedManifest {
    format: String,
    name: String,
    #[serde(rename = "version")]
    _version: String,
    set_id: String,
    layers: BTreeMap<String, SelectedLayer>,
}
#[derive(serde::Deserialize)]
struct SelectedLayer {
    gate: Vec<Vec<usize>>,
    up: Vec<Vec<usize>>,
    down: Vec<usize>,
    shared: SelectedShared,
}
#[derive(serde::Deserialize)]
struct SelectedShared { gate: Vec<usize>, up: Vec<usize>, down: usize }

/// One invocation's immutable native metadata; payload/header checks still run per arena.
pub struct MetadataInput {
    snapshot: PathBuf,
    selected: String,
    pub(super) config: serde_json::Value,
    manifest: SelectedManifest,
    pub(crate) identity: String,
    census: Mutex<BTreeMap<usize, LayerCensus>>,
    #[cfg(test)]
    uncached_census: bool,
}

#[derive(Default)]
struct LayerSources {
    index: Option<(Header, Vec<u8>)>,
    weights: BTreeMap<usize, (PathBuf, Header)>,
}

struct LayerCensus {
    routed: Vec<Vec<usize>>,
    shared: Vec<Vec<usize>>,
}
impl MetadataInput {
    pub fn open(snapshot: &Path, selected: &str) -> anyhow::Result<Self> {
        use sha2::{Digest, Sha256};
        ensure!(!selected.is_empty() && selected.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
            "manifest must be a published name, not a path");
        let mut bytes = Vec::new();
        File::open(snapshot.join(format!("manifests/exlr8-manifest-{selected}.json")))?
            .take((64 << 20) + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 64 << 20, "oversized EXLR8 manifest");
        let manifest: SelectedManifest = serde_json::from_slice(&bytes)?;
        ensure!(manifest.format == "exlr8-v2-manifest" && manifest.name == selected,
            "invalid EXLR8 manifest identity");
        ensure!(manifest.set_id.len() == 64 && manifest.set_id.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid EXLR8 set_id");
        Ok(Self { snapshot: snapshot.into(), selected: selected.into(), config: read_json(&snapshot.join("config.json"))?,
            manifest, identity: format!("{:x}", Sha256::digest(&bytes)), census: Mutex::default(),
            #[cfg(test)] uncached_census: false })
    }

    pub fn plan(&self, layer: usize, world: usize, rank: usize, set: ExpertSet) -> Result<Residency> {
        Residency::plan_inner(self, layer, world, rank, set, &mut LayerSources::default()).map_err(Into::into)
    }

    /// Scalar admission facts live only for this selected manifest's plan. Runtime
    /// arena construction uses `plan`, which always revalidates its sources.
    pub(crate) fn resident_bytes(&self, layer: usize, world: usize, rank: usize, set: ExpertSet) -> Result<usize> {
        if !(1..=8).contains(&world) || rank >= world {
            return Err(anyhow::anyhow!("expected TP1..8 and rank below world").into());
        }
        #[cfg(test)]
        if self.uncached_census { return Ok(self.plan(layer, world, rank, set)?.resident_bytes); }
        let mut census = self.census.lock().map_err(|_| ResidencyError(anyhow::anyhow!("EXLR8 census lock poisoned")))?;
        if !census.contains_key(&layer) {
            let mut sources = LayerSources::default();
            let routed = Residency::plan_inner(self, layer, 1, 0, ExpertSet::Routed, &mut sources)?;
            let shared = Residency::plan_inner(self, layer, 1, 0, ExpertSet::Shared, &mut sources)?;
            census.insert(layer, LayerCensus { routed: routed.rank_bytes(routed.experts)?, shared: shared.rank_bytes(routed.experts)? });
        }
        let layer = &census[&layer];
        Ok(match set { ExpertSet::Routed => &layer.routed, ExpertSet::Shared => &layer.shared }[world - 1][rank])
    }

    #[cfg(test)]
    pub(super) fn without_census_cache(mut self) -> Self {
        self.uncached_census = true;
        self
    }

    /// Cross-check optional published rank slices against canonical residency.
    /// Slices never supply runtime ownership or become a loading prerequisite.
    pub fn cross_check_index_slices(&self) -> anyhow::Result<usize> {
        let mut checked = 0;
        for world in 1..=8 {
            let directory = self.snapshot.join(format!("index-slices/tp{world}/{}", self.selected));
            if !directory.exists() { continue; }
            for rank in 0..world {
                let path = directory.join(format!("rank{rank}.safetensors"));
                if !path.exists() { continue; }
                self.cross_check_index_slice(&path, world, rank)?;
                checked += 1;
            }
        }
        Ok(checked)
    }

    fn cross_check_index_slice(&self, path: &Path, world: usize, rank: usize) -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};
        let slice = Header::open(path)?;
        let metadata = &slice.value["__metadata__"];
        ensure!(metadata["format"] == "exlr8-v2-index-slice" && metadata["version"] == "1"
            && number_string(metadata, "tp")? == world && number_string(metadata, "rank")? == rank
            && metadata["set_id"] == self.manifest.set_id, "EXLR8 slice identity mismatch");
        let bits = number_string(metadata, "k")?;
        ensure!((2..=4).contains(&bits), "EXLR8 slice width unsupported");
        let config = self.config.get("text_config").unwrap_or(&self.config);
        let hidden = number(config, "hidden_size")?;
        let experts = number(config, "n_routed_experts")?;
        let units = number(config, "moe_intermediate_size")? / 128;
        ensure!(number_string(metadata, "hidden_size")? == hidden
            && number_string(metadata, "n_routed_experts")? == experts
            && number_string(metadata, "half_bytes")? == hidden * 16 * bits,
            "EXLR8 slice geometry mismatch");
        let layers = metadata["layers"].as_str().context("EXLR8 slice layers missing")?
            .split(',').map(|layer| layer.parse::<usize>().context("EXLR8 slice layer invalid"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let first = number(config, "first_k_dense_replace")?;
        let end = number(config, "num_hidden_layers")?.checked_add(config.get("num_nextn_predict_layers")
            .map(|_| number(config, "num_nextn_predict_layers")).transpose()?.unwrap_or(0))
            .context("EXLR8 slice layer overflow")?;
        ensure!(layers.iter().copied().eq(first..end), "EXLR8 slice layer extent mismatch");
        let hashes: BTreeMap<String, String> = serde_json::from_str(metadata["index_sha256"]
            .as_str().context("EXLR8 slice index hashes missing")?)?;
        ensure!(hashes.len() == layers.len(), "EXLR8 slice index hash extent mismatch");
        for layer in layers {
            let canonical_path = self.snapshot.join(format!("experts/exlr8-layer-{layer:03}.index.safetensors"));
            let canonical = Header::open(&canonical_path)?;
            let hash = format!("{:x}", Sha256::digest(std::fs::read(&canonical_path)?));
            ensure!(hashes.get(&layer.to_string()) == Some(&hash), "EXLR8 slice index hash mismatch");
            let count = (0..=experts).map(|expert| (0..units)
                .filter(|&unit| owner(units, expert, unit, layer, world) == rank).count()).sum::<usize>();
            let root = format!("L{layer:03}");
            let read = |suffix: &str, dtype: &str, shape: &[usize], bytes: usize| -> anyhow::Result<Vec<u8>> {
                let source = range(&slice, path, &format!("{root}.{suffix}"), dtype, shape, 0, bytes)?;
                let mut out = vec![0; bytes]; source.read(&mut out)?; Ok(out)
            };
            let owned = read("units", "I16", &[count, 2], count * 4)?;
            let offsets = read("offset", "I64", &[count, 3], count * 24)?;
            let checksums = read("checksum", "U64", &[count, 3], count * 24)?;
            let channels = read("chan", "F16", &[count, 3, 128], count * 3 * 256)?;
            let blocks = read("block", "F16", &[experts + 1, 3, hidden / 128], (experts + 1) * 3 * hidden / 128 * 2)?;
            let signs = read("signs", "I8", &[2, hidden], hidden * 2)?;
            let mut ordinal = 0;
            for set in [ExpertSet::Routed, ExpertSet::Shared] {
                let plan = self.plan(layer, world, rank, set)?;
                let prefix = if set == ExpertSet::Shared { "shared." } else { "" };
                let expected_checksums = canonical.read(&canonical_path, &format!("{prefix}k{bits}.checksum"))?;
                ensure!(signs == plan.signs.concat(), "EXLR8 slice signs mismatch");
                for group in plan.loads.chunks_exact(3) {
                    let expert = if set == ExpertSet::Shared { experts } else { group[0].expert };
                    ensure!(owned[ordinal * 4..ordinal * 4 + 2] == (expert as i16).to_le_bytes()
                        && owned[ordinal * 4 + 2..ordinal * 4 + 4] == (group[0].unit as i16).to_le_bytes(),
                        "EXLR8 slice ownership mismatch");
                    for load in group {
                        ensure!(load.bits == bits as i32, "EXLR8 slice disagrees with selected manifest width");
                        let slot = ordinal * 3 + load.projection;
                        ensure!(offsets[slot * 8..slot * 8 + 8] == load.source[0].offset.to_le_bytes(),
                            "EXLR8 slice payload offset mismatch");
                        let at = ((load.expert * 3 + load.projection) * units + load.unit) * 8;
                        ensure!(checksums[slot * 8..slot * 8 + 8] == expected_checksums[at..at + 8],
                            "EXLR8 slice checksum mismatch");
                        for (source, bytes, label) in [(&load.source[1], &channels[slot * 256..slot * 256 + 256], "channel"),
                            (&load.source[2], &blocks[(expert * 3 + load.projection) * hidden / 64..
                                (expert * 3 + load.projection + 1) * hidden / 64], "block")] {
                            let mut expected = vec![0; source.bytes]; source.read(&mut expected)?;
                            ensure!(bytes == expected, "EXLR8 slice {label} mismatch");
                        }
                    }
                    ordinal += 1;
                }
            }
            ensure!(ordinal == count, "EXLR8 slice owned unit count mismatch");
        }
        Ok(())
    }

    fn layer(&self, layer: &Layer, units: usize, experts: usize) -> anyhow::Result<&SelectedLayer> {
        let choices = self.manifest.layers.get(&layer.id.to_string()).context("missing manifest layer")?;
        let width = |k: usize, widths: &BTreeSet<usize>| -> anyhow::Result<()> {
            ensure!((2..=4).contains(&k) && widths.contains(&k), "manifest requests absent K{k}");
            Ok(())
        };
        for rows in [&choices.gate, &choices.up] {
            ensure!(rows.len() == experts, "manifest expert count mismatch");
            for row in rows {
                ensure!(row.len() == units, "manifest unit count mismatch");
                for &k in row { width(k, &layer.widths)?; }
            }
        }
        ensure!(choices.down.len() == experts, "manifest expert count mismatch");
        for &k in &choices.down { width(k, &layer.widths)?; }
        for row in [&choices.shared.gate, &choices.shared.up] {
            ensure!(row.len() == units, "manifest shared unit count mismatch");
            for &k in row { width(k, &layer.shared_widths)?; }
        }
        width(choices.shared.down, &layer.shared_widths)?;
        Ok(choices)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("EXLR8 residency: {0:#}")]
pub struct ResidencyError(#[from] anyhow::Error);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSet {
    Routed,
    Shared,
}

#[derive(Debug, Clone)]
pub struct Range {
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: usize,
    cached: Option<Arc<[u8]>>,
}
impl Range {
    fn read(&self, out: &mut [u8]) -> anyhow::Result<()> {
        ensure!(out.len() == self.bytes, "range destination length mismatch");
        if let Some(bytes) = &self.cached {
            out.copy_from_slice(bytes);
            return Ok(());
        }
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.offset))?;
        file.read_exact(out)
            .with_context(|| format!("reading {} at {}", self.path.display(), self.offset))
    }
}

#[derive(Debug, Clone)]
pub struct Unit {
    /// Descriptor expert index (zero for the shared expert).
    pub expert: usize,
    pub unit: usize,
    /// gate=0, up=1, down=2; each owns payload, channel and block pointers.
    pub projection: usize,
    pub bits: i32,
    pub source: [Range; 3],
    pub destination: [usize; 3],
}
impl Unit {
    pub fn staging_bytes(&self) -> usize {
        self.source.iter().map(|r| r.bytes).sum()
    }

    /// Read each component into its bounded staging range.
    pub fn read(&self, staging: &mut [u8]) -> Result<()> {
        self.read_inner(staging).map_err(Into::into)
    }
    fn read_inner(&self, staging: &mut [u8]) -> anyhow::Result<()> {
        ensure!(
            staging.len() == self.staging_bytes(),
            "unit staging length mismatch"
        );
        let mut at = 0;
        for source in &self.source {
            source.read(&mut staging[at..at + source.bytes])?;
            at += source.bytes;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Residency {
    pub layer: usize,
    pub world: usize,
    pub rank: usize,
    pub hidden: usize,
    pub experts: usize,
    pub units: usize,
    pub set: ExpertSet,
    pub set_id: String,
    pub manifest: String,
    pub loads: Vec<Unit>,
    pub signs: [Vec<u8>; 2],
    pub signs_offsets: [usize; 2],
    /// K1 [E,U,3,3] int64 pointers and [E,U,3] int32 widths.
    pub pointers_offset: usize,
    pub bits_offset: usize,
    /// Immutable routed compact counts [E] and ascending global unit map [E,U].
    pub compact_offsets: [usize; 2],
    pub resident_bytes: usize,
    pub payload_bytes: usize,
    pub staging_bytes: usize,
}

fn reserve(cursor: &mut usize, bytes: usize) -> anyhow::Result<usize> {
    let start = cursor
        .checked_add(255)
        .context("arena alignment overflow")?
        & !255;
    *cursor = start.checked_add(bytes).context("arena size overflow")?;
    Ok(start)
}
/// Exact arena shared by runtime native execution and offline admission.
pub fn execution_workspace_layout(
    rows: usize, max_rows: usize, group_rows: usize, hidden: usize, top_k: usize, max_groups: usize,
    fc1_scratch: usize, intermediate_scratch: usize, fc2_scratch: usize,
    set: ExpertSet,
) -> anyhow::Result<([(usize, usize); 11], usize)> {
    ensure!(matches!((max_rows, group_rows), (64, 16) | (4096, 16 | 32 | 64))
        && (max_rows == 64 || set == ExpertSet::Routed)
        && rows > 0 && rows <= max_rows && hidden > 0 && hidden % 128 == 0
        && top_k > 0 && max_groups > 0,
        "EXLR8 workspace requires admitted row/tile geometry, aligned hidden and nonzero routes");
    let mul = |a: usize, b: usize| a.checked_mul(b).context("EXLR8 workspace size overflow");
    let values = mul(rows, hidden)?;
    let routes = mul(rows, top_k)?;
    let input_row_bytes = match set {
        ExpertSet::Routed => hidden.checked_add(hidden / 32)
            .context("EXLR8 wire row overflow")?,
        ExpertSet::Shared => mul(hidden, 2)?,
    };
    // Input and rotated activation are separate: ingress runs once per row.
    let sizes = [mul(rows, input_row_bytes)?, mul(routes, 4)?, mul(routes, 4)?, mul(values, 2)?,
        mul(max_groups, 4)?, mul(mul(max_groups, group_rows)?, 4)?, mul(routes, 4)?, 4,
        fc1_scratch, intermediate_scratch.checked_add(fc2_scratch)
            .context("EXLR8 workspace scratch overflow")?, mul(values, 2)?];
    let mut cursor = 0;
    let mut ranges = [(0, 0); 11];
    for (range, bytes) in ranges.iter_mut().zip(sizes) {
        *range = (reserve(&mut cursor, bytes)?, bytes);
    }
    Ok((ranges, cursor))
}

fn range(
    header: &Header,
    path: &Path,
    name: &str,
    dtype: &str,
    shape: &[usize],
    start: usize,
    bytes: usize,
) -> anyhow::Result<Range> {
    let spec = &header.value[name];
    ensure!(
        spec["dtype"] == dtype && spec["shape"] == serde_json::json!(shape),
        "invalid tensor {name}: expected {dtype} {shape:?}"
    );
    let element = match dtype {
        "I8" | "U8" => 1,
        "F16" | "I16" => 2,
        "I64" | "U64" => 8,
        _ => unreachable!(),
    };
    let expected = shape
        .iter()
        .try_fold(element, |n: usize, &v| n.checked_mul(v))
        .context("tensor size overflow")?;
    let (base, length) = header.extent(name)?;
    ensure!(
        length == expected as u64 && start.checked_add(bytes).is_some_and(|n| n <= expected),
        "invalid tensor byte range {name}"
    );
    Ok(Range {
        path: path.into(),
        offset: base
            .checked_add(start as u64)
            .context("source offset overflow")?,
        bytes,
        cached: None,
    })
}
fn cache_range(mut source: Range, index: &[u8]) -> anyhow::Result<Range> {
    let start = usize::try_from(source.offset)?;
    source.cached = Some(Arc::from(
        index
            .get(start..start + source.bytes)
            .context("side data outside index")?,
    ));
    Ok(source)
}
fn read_u64(source: Range, index: &[u8]) -> anyhow::Result<u64> {
    let offset = usize::try_from(source.offset)?;
    let bytes = index
        .get(offset..offset + 8)
        .context("index range outside file")?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

impl Residency {
    pub fn plan(
        snapshot: &Path,
        selected: &str,
        layer: usize,
        world: usize,
        rank: usize,
        set: ExpertSet,
    ) -> Result<Self> {
        if !(1..=8).contains(&world) || rank >= world {
            return Err(anyhow::anyhow!("expected TP1..8 and rank below world").into());
        }
        MetadataInput::open(snapshot, selected)?.plan(layer, world, rank, set)
    }

    fn plan_inner(
        input: &MetadataInput,
        layer: usize,
        world: usize,
        rank: usize,
        set: ExpertSet,
        sources: &mut LayerSources,
    ) -> anyhow::Result<Self> {
        ensure!(
            (1..=8).contains(&world) && rank < world,
            "expected TP1..8 and rank below world"
        );
        let snapshot = &input.snapshot;
        let selected = &input.selected;
        let config = &input.config;
        super::architecture(config)?;
        let config = config.get("text_config").unwrap_or(config);
        let hidden = number(config, "hidden_size")?;
        let intermediate = number(config, "moe_intermediate_size")?;
        let routed = number(config, "n_routed_experts")?;
        let layers = number(config, "num_hidden_layers")?
            .checked_add(config.get("num_nextn_predict_layers")
                .map(|_| number(config, "num_nextn_predict_layers")).transpose()?.unwrap_or(0))
            .context("EXLR8 layer extent overflow")?;
        ensure!(
            hidden > 0
                && hidden % 128 == 0
                && intermediate > 0
                && intermediate % 128 == 0
                && routed > 0
                && layer >= number(config, "first_k_dense_replace")?
                && layer < layers,
            "invalid EXLR8 layer or geometry"
        );
        let units = intermediate / 128;
        let entries = routed
            .checked_mul(units)
            .and_then(|n| n.checked_mul(3))
            .context("descriptor size overflow")?;
        ensure!(
            entries <= i32::MAX as usize / 24 && hidden <= i32::MAX as usize / 64,
            "geometry exceeds K1 signed indexing"
        );
        let index_path = snapshot.join(format!("experts/exlr8-layer-{layer:03}.index.safetensors"));
        if sources.index.is_none() {
            let index = Header::open(&index_path)?;
            index.validate()?;
            ensure!(index.bytes <= 64 << 20, "oversized EXLR8 index");
            sources.index = Some((index, std::fs::read(&index_path)?));
        }
        let (index, index_bytes) = sources.index.as_ref().unwrap();
        let metadata = &index.value["__metadata__"];
        ensure!(
            metadata["layer"]
                .as_str()
                .and_then(|s| s.parse::<usize>().ok())
                == Some(layer)
                && metadata["gate_up_order"] == "grouped"
                && metadata["down_order"] == "item128",
            "unsupported EXLR8 index identity or tile order"
        );
        let widths = |key: &str| -> anyhow::Result<BTreeSet<usize>> {
            metadata[key]
                .as_str()
                .context("missing width list")?
                .split(',')
                .map(|s| s.parse().context("invalid width list"))
                .collect()
        };
        let layer_spec = Layer {
            id: layer,
            widths: widths("widths")?,
            shared_widths: widths("shared_widths")?,
        };
        let choices = input.layer(&layer_spec, units, routed)?;
        let experts = if set == ExpertSet::Shared { 1 } else { routed };
        let prefix = if set == ExpertSet::Shared {
            "shared."
        } else {
            ""
        };
        let mut signs = [Vec::new(), Vec::new()];
        let mut cursor = 0;
        let mut signs_offsets = [0; 2];
        for (i, name) in ["signs.in", "signs.out"].iter().enumerate() {
            signs[i] = vec![0; hidden];
            range(&index, &index_path, name, "I8", &[hidden], 0, hidden)?.read(&mut signs[i])?;
            ensure!(
                signs[i].iter().all(|&v| v == 1 || v == 255),
                "sign vector must contain +/-1"
            );
            signs_offsets[i] = reserve(&mut cursor, hidden)?;
        }
        let slots = experts * units * 3;
        let pointers_offset = reserve(
            &mut cursor,
            slots.checked_mul(24).context("pointer table overflow")?,
        )?;
        let bits_offset = reserve(
            &mut cursor,
            slots.checked_mul(4).context("bits table overflow")?,
        )?;
        let compact_offsets = if set == ExpertSet::Routed {
            [reserve(&mut cursor, experts.checked_mul(4).context("unit count overflow")?)?,
             reserve(&mut cursor, experts.checked_mul(units).and_then(|n| n.checked_mul(4))
                 .context("unit map overflow")?)?]
        } else { [0; 2] };
        let mut blocks = BTreeMap::new();
        let weight_headers = &mut sources.weights;
        let mut loads = Vec::new();
        let mut payload_bytes = 0usize;
        for e in 0..experts {
            let identity = if set == ExpertSet::Shared { routed } else { e };
            for u in 0..units {
                if owner(units, identity, u, layer, world) != rank {
                    continue;
                }
                for (p, projection) in ["gate", "up", "down"].iter().enumerate() {
                    let k = match (set, p) {
                        (ExpertSet::Shared, 0) => choices.shared.gate[u],
                        (ExpertSet::Shared, 1) => choices.shared.up[u],
                        (ExpertSet::Shared, _) => choices.shared.down,
                        (ExpertSet::Routed, 0) => choices.gate[e][u],
                        (ExpertSet::Routed, 1) => choices.up[e][u],
                        (ExpertSet::Routed, _) => choices.down[e],
                    };
                    let stem = format!("{prefix}k{k}");
                    let offset_at = ((e * 3 + p) * units + u) * 16;
                    let offset = read_u64(
                        range(
                            &index,
                            &index_path,
                            &format!("{stem}.index"),
                            "I64",
                            &[experts, 3, units, 2],
                            offset_at,
                            8,
                        )?,
                        &index_bytes,
                    )?;
                    let length = read_u64(
                        range(
                            &index,
                            &index_path,
                            &format!("{stem}.index"),
                            "I64",
                            &[experts, 3, units, 2],
                            offset_at + 8,
                            8,
                        )?,
                        &index_bytes,
                    )?;
                    let bytes = hidden
                        .checked_mul(16)
                        .and_then(|n| n.checked_mul(k))
                        .context("unit size overflow")?;
                    ensure!(length == bytes as u64, "invalid indexed unit length");
                    if !weight_headers.contains_key(&k) {
                        let path = snapshot
                            .join(format!("experts/exlr8-layer-{layer:03}.k{k}.safetensors"));
                        let h = Header::open(&path)?;
                        h.validate()?;
                        let m = &h.value["__metadata__"];
                        ensure!(
                            m["layer"].as_str().and_then(|s| s.parse::<usize>().ok())
                                == Some(layer)
                                && m["k"].as_str().and_then(|s| s.parse::<usize>().ok()) == Some(k)
                                && m["gate_up_order"] == "grouped"
                                && m["down_order"] == "item128",
                            "trellis file identity/order mismatch"
                        );
                        ensure!(
                            m["pad_halves"] == "true" || m["pad_halves"] == "false",
                            "invalid padding metadata"
                        );
                        weight_headers.insert(k, (path, h));
                    }
                    let (path, weight) = &weight_headers[&k];
                    let tensor = format!("{prefix}trellis.{projection}");
                    let (base, extent) = weight.extent(&tensor)?;
                    let stride = if weight.value["__metadata__"]["pad_halves"] == "true" {
                        bytes.div_ceil(65536) * 65536
                    } else {
                        bytes
                    };
                    let expected_extent = experts
                        .checked_mul(units)
                        .and_then(|n| n.checked_mul(stride))
                        .context("projection extent overflow")?;
                    ensure!(
                        weight.value[&tensor]["dtype"] == "U8"
                            && weight.value[&tensor]["shape"]
                                == serde_json::json!([expected_extent])
                            && extent == expected_extent as u64,
                        "invalid trellis projection shape or dtype"
                    );
                    ensure!(
                        offset
                            == base
                                .checked_add(((e * units + u) * stride) as u64)
                                .context("unit offset overflow")?
                            && offset
                                .checked_add(length)
                                .is_some_and(|end| end <= base + extent),
                        "index points outside its projection"
                    );
                    let (channel, block) = if p == 2 {
                        ("suh", "sv_block")
                    } else {
                        ("svh", "su_block")
                    };
                    let channel = range(
                        &index,
                        &index_path,
                        &format!("{stem}.{projection}.{channel}"),
                        "F16",
                        &[experts, intermediate],
                        (e * intermediate + u * 128) * 2,
                        256,
                    )?;
                    let block = range(
                        &index,
                        &index_path,
                        &format!("{stem}.{projection}.{block}"),
                        "F16",
                        &[experts, hidden / 128],
                        e * hidden / 128 * 2,
                        hidden / 128 * 2,
                    )?;
                    let channel = cache_range(channel, &index_bytes)?;
                    let block = cache_range(block, &index_bytes)?;
                    range(
                        &index,
                        &index_path,
                        &format!("{stem}.checksum"),
                        "U64",
                        &[experts, 3, units],
                        ((e * 3 + p) * units + u) * 8,
                        8,
                    )?;
                    let payload_dest = reserve(&mut cursor, bytes)?;
                    let channel_dest = reserve(&mut cursor, 256)?;
                    let block_dest = if let Some(&offset) = blocks.get(&(e, p, k)) {
                        offset
                    } else {
                        let offset = reserve(&mut cursor, block.bytes)?;
                        blocks.insert((e, p, k), offset);
                        offset
                    };
                    payload_bytes = payload_bytes
                        .checked_add(bytes)
                        .context("payload sum overflow")?;
                    loads.push(Unit {
                        expert: e,
                        unit: u,
                        projection: p,
                        bits: k as i32,
                        source: [
                            Range {
                                path: path.clone(),
                                offset,
                                bytes,
                                cached: None,
                            },
                            channel,
                            block,
                        ],
                        destination: [payload_dest, channel_dest, block_dest],
                    });
                }
            }
        }
        let staging_bytes = loads.iter().map(Unit::staging_bytes).max().unwrap_or(0);
        reserve(&mut cursor, 0)?;
        Ok(Self {
            layer,
            world,
            rank,
            hidden,
            experts,
            units,
            set,
            set_id: input.manifest.set_id.clone(),
            manifest: selected.clone(),
            loads,
            signs,
            signs_offsets,
            pointers_offset,
            bits_offset,
            compact_offsets,
            resident_bytes: cursor,
            payload_bytes,
            staging_bytes,
        })
    }

    /// Project the validated whole-layer unit order onto every supported TP
    /// world. Descriptor tables remain whole; only payload/scale ownership rotates.
    fn rank_bytes(&self, routed_experts: usize) -> anyhow::Result<Vec<Vec<usize>>> {
        ensure!(self.world == 1 && self.rank == 0, "EXLR8 census requires a whole layer");
        let tables = self.loads.first().context("EXLR8 census has no units")?.destination[0];
        (1..=8).map(|world| {
            let mut cursors = vec![tables; world];
            let mut blocks = BTreeSet::new();
            for unit in &self.loads {
                let identity = if self.set == ExpertSet::Shared { routed_experts } else { unit.expert };
                let rank = owner(self.units, identity, unit.unit, self.layer, world);
                let cursor = &mut cursors[rank];
                reserve(cursor, unit.source[0].bytes)?;
                reserve(cursor, unit.source[1].bytes)?;
                if blocks.insert((rank, unit.expert, unit.projection, unit.bits)) {
                    reserve(cursor, unit.source[2].bytes)?;
                }
            }
            for cursor in &mut cursors { reserve(cursor, 0)?; }
            Ok(cursors)
        }).collect()
    }

    /// Same rotated ownership used by payload selection, built before allocation.
    pub fn compact_descriptors(&self) -> anyhow::Result<(Vec<i32>, Vec<i32>, usize)> {
        use cuteafd_core::tp_units::{unit_owner, UnitRule};
        ensure!(self.set == ExpertSet::Routed && self.world > 0 && self.rank < self.world,
            "compact EXLR8 descriptors require a routed rank");
        let mut counts = vec![0i32; self.experts];
        let mut mapping = vec![0i32; self.experts.checked_mul(self.units).context("unit map overflow")?];
        let mut limit = 1;
        for (expert, count) in counts.iter_mut().enumerate() {
            for unit in 0..self.units {
                if unit_owner(UnitRule::Rotated, self.units, expert, unit, self.layer, self.world)? == self.rank {
                    mapping[expert * self.units + *count as usize] = i32::try_from(unit)?;
                    *count += 1;
                }
            }
            limit = limit.max(*count as usize);
        }
        Ok((counts, mapping, limit))
    }

    pub fn compact_unit_limit(&self) -> usize {
        self.units.div_ceil(self.world).max(1)
    }

    /// Peak temporary host images while relocating ordinary and compact descriptors.
    pub fn descriptor_host_bytes(&self) -> anyhow::Result<usize> {
        let descriptors = self.experts.checked_mul(self.units)
            .and_then(|n| n.checked_mul(3 * (3 * 8 + 4) * 2))
            .context("EXLR8 descriptor staging overflow")?;
        let compact = if self.set == ExpertSet::Routed {
            self.experts.checked_mul(self.units.checked_add(1).context("compact unit overflow")?)
                .and_then(|n| n.checked_mul(8)).context("compact descriptor overflow")?
        } else { 0 };
        descriptors.checked_add(compact).context("EXLR8 descriptor host overflow")
    }

    /// Host payload retained by this plan, excluding allocator bookkeeping.
    pub fn host_bytes(&self) -> usize {
        let sources = self.loads.iter().flat_map(|unit| &unit.source);
        self.loads.capacity() * std::mem::size_of::<Unit>()
            + self.signs.iter().map(Vec::capacity).sum::<usize>()
            + self.set_id.capacity()
            + self.manifest.capacity()
            + sources
                .map(|source| {
                    source.path.as_os_str().len()
                        + source.cached.as_ref().map_or(0, |bytes| bytes.len())
                })
                .sum::<usize>()
    }

    /// Noncontiguous unit ownership spans the full logical intermediate domain;
    /// counts describe this rank's actual native projection units by trellis tier.
    pub fn spark_slice(&self) -> Result<cuteafd_core::SparkSlice> {
        self.spark_slice_inner().map_err(Into::into)
    }
    fn spark_slice_inner(&self) -> anyhow::Result<cuteafd_core::SparkSlice> {
        let mut counts = [[0u32; 3]; 3];
        for unit in &self.loads {
            ensure!((2..=4).contains(&unit.bits) && unit.projection < 3,
                "invalid EXLR8 projection tier");
            let count = &mut counts[(unit.bits - 2) as usize][unit.projection];
            *count = count.checked_add(1).context("EXLR8 projection count overflow")?;
        }
        let width = u32::try_from(self.units.checked_mul(128).context("EXLR8 slice width overflow")?)?;
        let map = cuteafd_core::tp_units::RankUnits::new(cuteafd_core::tp_units::UnitRule::Rotated,
            self.experts, self.units * 128, 128, self.layer, self.world, self.rank)?;
        Ok(cuteafd_core::SparkSlice { unit_map: Some(map.fingerprint()), layer: u32::try_from(self.layer)?, first: 0,
            width, stored_width: width, tiers: vec![2, 3, 4], projection_counts: counts.to_vec() })
    }

    /// Includes alignment and both descriptor tables, unlike weight-only plan.
    /// Callers supply workspace, graph, staging and other device reservations.
    pub fn check_budget(&self, ceiling: usize, reservations: &[usize]) -> Result<usize> {
        let required = reservations
            .iter()
            .try_fold(self.resident_bytes, |n, &r| n.checked_add(r))
            .context("admission overflow")?;
        if required > ceiling {
            return Err(ResidencyError(anyhow::anyhow!(
                "weights plus reservations require {required} bytes, ceiling {ceiling}"
            )));
        }
        Ok(required)
    }

    pub fn descriptors(&self, arena_address: u64) -> Result<(Vec<u64>, Vec<i32>)> {
        ensure_inner(self, arena_address).map_err(Into::into)
    }
}
fn ensure_inner(plan: &Residency, address: u64) -> anyhow::Result<(Vec<u64>, Vec<i32>)> {
    ensure!(
        address != 0
            && address % 256 == 0
            && address.checked_add(plan.resident_bytes as u64).is_some(),
        "invalid arena address"
    );
    let mut pointers = vec![0; plan.experts * plan.units * 9];
    let mut bits = vec![0; plan.experts * plan.units * 3];
    for unit in &plan.loads {
        let slot = (unit.expert * plan.units + unit.unit) * 3 + unit.projection;
        bits[slot] = unit.bits;
        for (j, &offset) in unit.destination.iter().enumerate() {
            pointers[slot * 3 + j] = address + offset as u64;
        }
    }
    Ok((pointers, bits))
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn execution_workspace_is_exact_aligned_and_checked() {
        let (ranges, total) = execution_workspace_layout(64, 64, 16, 6144, 8, 24, 17, 19, 23, ExpertSet::Routed).unwrap();
        let sizes = [64 * 6336, 64 * 8 * 4, 64 * 8 * 4, 64 * 6144 * 2,
            24 * 4, 24 * 16 * 4, 64 * 8 * 4, 4, 17, 42, 64 * 6144 * 2];
        let mut end: usize = 0;
        for ((offset, bytes), expected) in ranges.into_iter().zip(sizes) {
            assert_eq!(offset, end.div_ceil(256) * 256);
            assert_eq!(bytes, expected);
            end = offset + bytes;
        }
        assert_eq!(total, end);
        for (rows, hidden, top_k, groups, fc1, intermediate, fc2) in [
            (0, 6144, 8, 24, 0, 0, 0), (64, 0, 8, 24, 0, 0, 0),
            (65, 6144, 8, 24, 0, 0, 0), (64, 127, 8, 24, 0, 0, 0),
            (64, 6144, 0, 24, 0, 0, 0), (64, 6144, 8, 0, 0, 0, 0),
            (usize::MAX, 6144, 8, 24, 0, 0, 0),
            (64, 6144, 8, usize::MAX, 0, 0, 0),
            (64, 6144, 8, 24, 0, usize::MAX, 1),
            (64, 6144, 8, 24, usize::MAX, 0, 0),
        ] {
            assert!(execution_workspace_layout(rows, 64, 16, hidden, top_k, groups, fc1, intermediate, fc2, ExpertSet::Routed).is_err());
        }
    }

    use crate::plan::testing::exlr8::fixture;

    #[test]
    fn optional_rank_slices_cross_check_canonical_residency_and_reject_corruption() {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), false);
        let manifest_path = root.path().join("manifests/exlr8-manifest-mixed.json");
        let mut manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["name"] = json!("k3");
        manifest["layers"]["3"] = json!({"gate":vec![vec![3;3];2], "up":vec![vec![3;3];2], "down":vec![3;2],
            "shared":{"gate":vec![3;3],"up":vec![3;3],"down":3}});
        std::fs::write(root.path().join("manifests/exlr8-manifest-k3.json"), manifest.to_string()).unwrap();
        let input = MetadataInput::open(root.path(), "k3").unwrap();
        assert_eq!(input.cross_check_index_slices().unwrap(), 0);
        let index_hash = format!("{:x}", Sha256::digest(std::fs::read(root.path()
            .join("experts/exlr8-layer-003.index.safetensors")).unwrap()));
        let mut last = None;
        for world in 1..=4 {
            for rank in 0..world {
                let mut owned = Vec::new();
                let mut offsets = Vec::new();
                let mut channels = Vec::new();
                let mut blocks = vec![0; 3 * 3 * 2];
                let mut signs = Vec::new();
                for set in [ExpertSet::Routed, ExpertSet::Shared] {
                    let plan = input.plan(3, world, rank, set).unwrap();
                    signs = plan.signs.concat();
                    for group in plan.loads.chunks_exact(3) {
                        let expert = if set == ExpertSet::Shared { 2 } else { group[0].expert };
                        owned.extend_from_slice(&(expert as i16).to_le_bytes());
                        owned.extend_from_slice(&(group[0].unit as i16).to_le_bytes());
                        for unit in group {
                            offsets.extend_from_slice(&unit.source[0].offset.to_le_bytes());
                            let mut bytes = vec![0; 256]; unit.source[1].read(&mut bytes).unwrap();
                            channels.extend(bytes);
                            unit.source[2].read(&mut blocks[(expert * 3 + unit.projection) * 2..
                                (expert * 3 + unit.projection + 1) * 2]).unwrap();
                        }
                    }
                }
                let count = owned.len() / 4;
                let tensors = vec![("units", "I16", vec![count,2], owned),
                    ("offset", "I64", vec![count,3], offsets),
                    ("checksum", "U64", vec![count,3], vec![0;count*24]),
                    ("chan", "F16", vec![count,3,128], channels),
                    ("block", "F16", vec![3,3,1], blocks),
                    ("signs", "I8", vec![2,128], signs)];
                let metadata = json!({"format":"exlr8-v2-index-slice","version":"1",
                    "tp":world.to_string(),"rank":rank.to_string(),"k":"3","set_id":"b".repeat(64),
                    "hidden_size":"128","n_routed_experts":"2","half_bytes":"6144","layers":"3",
                    "index_sha256":json!({"3":index_hash}).to_string()});
                let mut header = serde_json::Map::from_iter([("__metadata__".into(),metadata)]);
                let mut payload = Vec::new();
                for (name,dtype,shape,bytes) in tensors {
                    header.insert(format!("L003.{name}"), json!({"dtype":dtype,"shape":shape,
                        "data_offsets":[payload.len(),payload.len()+bytes.len()]}));
                    payload.extend(bytes);
                }
                let header_bytes = serde_json::to_vec(&header).unwrap();
                let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
                bytes.extend(&header_bytes); bytes.extend(payload);
                let directory = root.path().join(format!("index-slices/tp{world}/k3"));
                std::fs::create_dir_all(&directory).unwrap();
                let path = directory.join(format!("rank{rank}.safetensors"));
                std::fs::write(&path, &bytes).unwrap();
                last = Some((path, bytes, header, 8 + header_bytes.len()));
            }
        }
        assert_eq!(input.cross_check_index_slices().unwrap(), 10);
        let (path, original, header, base) = last.unwrap();
        for tensor in ["units", "offset", "checksum", "chan", "block", "signs"] {
            let mut corrupt = original.clone();
            let offset = header[&format!("L003.{tensor}")]["data_offsets"][0].as_u64().unwrap() as usize;
            corrupt[base + offset] ^= 1;
            std::fs::write(&path, corrupt).unwrap();
            assert!(input.cross_check_index_slices().is_err(), "{tensor}");
        }
        std::fs::write(&path, original).unwrap();
        let index_path = root.path().join("experts/exlr8-layer-003.index.safetensors");
        std::fs::OpenOptions::new().append(true).open(index_path).unwrap().write_all(&[0]).unwrap();
        assert!(input.cross_check_index_slices().unwrap_err().to_string().contains("hash mismatch"));
    }

    #[test]
    fn published_index_slices_when_explicitly_supplied() {
        let Some(root) = std::env::var_os("CUTEAFD_EXLR8_SLICE_SNAPSHOT") else { return; };
        for manifest in super::super::manifest_names(Path::new(&root)).unwrap() {
            let input = MetadataInput::open(Path::new(&root), &manifest).unwrap();
            let count = input.cross_check_index_slices().unwrap();
            assert!(count > 0, "expected published slices for {manifest}");
            println!("{manifest}: {count} published rank slices agree with canonical residency");
        }
    }

    #[test]
    fn scalar_census_matches_every_fresh_arena_and_keeps_validation_local() {
        for padded in [false, true] {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), padded);
            let input = MetadataInput::open(root.path(), "mixed").unwrap();
            for world in 1..=8 {
                for rank in 0..world {
                    for set in [ExpertSet::Routed, ExpertSet::Shared] {
                        let fresh = input.plan(3, world, rank, set).unwrap();
                        assert_eq!(input.resident_bytes(3, world, rank, set).unwrap(), fresh.resident_bytes,
                            "padded={padded}, world={world}, rank={rank}, set={set:?}");
                    }
                }
            }
            assert_eq!(input.census.lock().unwrap().len(), 1);
            for (world, rank) in [(0, 0), (9, 0), (2, 2)] {
                assert!(input.resident_bytes(3, world, rank, ExpertSet::Routed).is_err());
            }
            assert!(input.resident_bytes(2, 1, 0, ExpertSet::Routed).is_err());
            let before = input.resident_bytes(3, 2, 1, ExpertSet::Routed).unwrap();
            let path = root.path().join("experts/exlr8-layer-003.k2.safetensors");
            std::fs::OpenOptions::new().write(true).open(path).unwrap().set_len(8).unwrap();
            assert_eq!(input.resident_bytes(3, 2, 1, ExpertSet::Routed).unwrap(), before);
            assert!(input.plan(3, 2, 1, ExpertSet::Routed).is_err());
            let fresh = MetadataInput::open(root.path(), "mixed").unwrap();
            assert!(fresh.resident_bytes(3, 2, 1, ExpertSet::Routed).is_err());
            assert!(fresh.census.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn metadata_input_reuses_selection_but_rechecks_payload_headers() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), false);
        let input = MetadataInput::open(root.path(), "mixed").unwrap();
        for world in 1..=8 {
            for rank in 0..world {
                for set in [ExpertSet::Routed, ExpertSet::Shared] {
                    let cached = input.plan(3, world, rank, set).unwrap();
                    let fresh = Residency::plan(root.path(), "mixed", 3, world, rank, set).unwrap();
                    assert_eq!((cached.resident_bytes, cached.payload_bytes, cached.staging_bytes),
                        (fresh.resident_bytes, fresh.payload_bytes, fresh.staging_bytes));
                    assert_eq!(format!("{cached:?}"), format!("{fresh:?}"));
                }
            }
        }
        let path = root.path().join("experts/exlr8-layer-003.k2.safetensors");
        std::fs::OpenOptions::new().write(true).open(path).unwrap().set_len(8).unwrap();
        assert!(input.plan(3, 1, 0, ExpertSet::Routed).is_err());
    }

    #[test]
    fn residency_ownership_mixed_widths_shared_and_padded_ranges() {
        for padded in [false, true] {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), padded);
            for world in 1..=8 {
                for set in [ExpertSet::Routed, ExpertSet::Shared] {
                    let mut owners = BTreeSet::new();
                    for rank in 0..world {
                        let plan =
                            Residency::plan(root.path(), "mixed", 3, world, rank, set).unwrap();
                        let slice = plan.spark_slice().unwrap();
                        assert_eq!(slice.layer, 3);
                        assert_eq!((slice.first, slice.width, slice.stored_width), (0, 384, 384));
                        assert_eq!(slice.tiers, [2, 3, 4]);
                        assert_eq!(slice.projection_counts.iter().flatten().sum::<u32>() as usize, plan.loads.len());
                        for (tier, counts) in slice.tiers.iter().zip(&slice.projection_counts) {
                            for (projection, &count) in counts.iter().enumerate() {
                                assert_eq!(count as usize, plan.loads.iter().filter(|unit|
                                    unit.bits as u32 == *tier && unit.projection == projection).count());
                            }
                        }
                        if set == ExpertSet::Routed {
                            let (counts, mapping, limit) = plan.compact_descriptors().unwrap();
                            assert!(limit <= plan.compact_unit_limit());
                            assert_eq!(counts.len(), plan.experts);
                            assert_eq!(mapping.len(), plan.experts * plan.units);
                            assert_eq!(counts.iter().sum::<i32>() as usize * 3, plan.loads.len());
                            assert_eq!(plan.compact_offsets[0] % 256, 0);
                            assert_eq!(plan.compact_offsets[1] % 256, 0);
                            assert!(plan.compact_offsets[0] >= plan.bits_offset + plan.experts * plan.units * 3 * 4);
                            assert!(plan.compact_offsets[1] >= plan.compact_offsets[0] + plan.experts * 4);
                            assert!(plan.compact_offsets[1] + plan.experts * plan.units * 4 <= plan.resident_bytes);
                            for (expert, &count) in counts.iter().enumerate() {
                                let slots = &mapping[expert * plan.units..(expert + 1) * plan.units];
                                let expected = (0..plan.units).filter(|&unit|
                                    owner(plan.units, expert, unit, plan.layer, plan.world) == plan.rank)
                                    .map(|unit| unit as i32).collect::<Vec<_>>();
                                assert_eq!(&slots[..count as usize], expected);
                                assert!(slots[count as usize..].iter().all(|&unit| unit == 0));
                            }
                        } else {
                            assert_eq!(plan.compact_offsets, [0; 2]);
                            assert!(plan.compact_descriptors().is_err());
                        }
                        let (ptrs, bits) = plan.descriptors(0x10000).unwrap();
                        assert_eq!(plan.resident_bytes % 256, 0);
                        assert_eq!(
                            plan.check_budget(plan.resident_bytes + 100, &[100])
                                .unwrap(),
                            plan.resident_bytes + 100
                        );
                        assert!(plan.check_budget(plan.resident_bytes + 99, &[100]).is_err());
                        assert!(plan.check_budget(usize::MAX, &[usize::MAX]).is_err());
                        assert!(plan.descriptors(1).is_err());
                        for unit in &plan.loads {
                            assert!(owners.insert((unit.expert, unit.unit, unit.projection)));
                            let source_e = if set == ExpertSet::Shared {
                                2
                            } else {
                                unit.expert
                            };
                            assert_eq!(owner(3, source_e, unit.unit, 3, world), rank);
                            let slot = (unit.expert * 3 + unit.unit) * 3 + unit.projection;
                            assert_eq!(bits[slot], unit.bits);
                            for (i, &dest) in unit.destination.iter().enumerate() {
                                assert_eq!(dest % 256, 0);
                                assert!(dest + unit.source[i].bytes <= plan.resident_bytes);
                                assert_eq!(ptrs[slot * 3 + i], 0x10000 + dest as u64);
                            }
                            let mut staging = vec![0; unit.staging_bytes()];
                            unit.read(&mut staging).unwrap();
                            assert!(unit.read(&mut staging[1..]).is_err());
                        }
                        for (slot, &width) in bits.iter().enumerate() {
                            if width == 0 {
                                assert_eq!(&ptrs[slot * 3..slot * 3 + 3], &[0; 3]);
                            }
                        }
                        let mut block_offsets = BTreeMap::new();
                        for unit in &plan.loads {
                            if let Some(previous) = block_offsets.insert(
                                (unit.expert, unit.projection, unit.bits),
                                unit.destination[2],
                            ) {
                                assert_eq!(previous, unit.destination[2]);
                            }
                        }
                    }
                    assert_eq!(owners.len(), if set == ExpertSet::Routed { 18 } else { 9 });
                }
            }
        }
    }

    #[test]
    fn truncated_native_source_is_refused() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), false);
        let plan = Residency::plan(root.path(), "mixed", 3, 1, 0, ExpertSet::Routed).unwrap();
        let unit = &plan.loads[0];
        let source = &unit.source[0];
        std::fs::OpenOptions::new().write(true).open(&source.path).unwrap()
            .set_len(source.offset + source.bytes as u64 - 1).unwrap();
        assert!(unit.read(&mut vec![0; unit.staging_bytes()]).is_err());
    }

    #[test]
    fn corrupt_index_is_refused() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), false);
        let path = root
            .path()
            .join("experts/exlr8-layer-003.index.safetensors");
        let header = Header::open(&path).unwrap();
        let (offset, _) = header.extent("k2.index").unwrap();
        let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        assert!(
            Residency::plan(root.path(), "mixed", 3, 1, 0, ExpertSet::Routed)
                .unwrap_err()
                .to_string()
                .contains("outside its projection")
        );
        assert!(Residency::plan(root.path(), "../mixed", 3, 1, 0, ExpertSet::Routed).is_err());
        assert!(Residency::plan(root.path(), "mixed", 3, 9, 0, ExpertSet::Routed).is_err());
    }

    #[test]
    fn malformed_native_projection_is_refused() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), false);
        let path = root.path().join("experts/exlr8-layer-003.k2.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let original: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        for replacement in [json!({"dtype":"I8"}), json!({"shape":[1]})] {
            let mut header = original.clone();
            for (key, value) in replacement.as_object().unwrap() {
                header["trellis.gate"][key] = value.clone();
            }
            let encoded = serde_json::to_vec(&header).unwrap();
            let mut changed = bytes.clone();
            changed[8..8 + header_len].fill(b' ');
            changed[8..8 + encoded.len()].copy_from_slice(&encoded);
            std::fs::write(&path, changed).unwrap();
            assert!(
                Residency::plan(root.path(), "mixed", 3, 1, 0, ExpertSet::Routed)
                    .unwrap_err()
                    .to_string()
                    .contains("shape or dtype")
            );
        }
    }

    #[test]
    fn real_checkpoint_residency_when_supplied() {
        let Some(root) = std::env::var_os("CUTEAFD_EXLR8_SNAPSHOT") else {
            return;
        };
        for selected in ["up1", "bit325", "home"] {
            for (world, rank, set) in [
                (4, 0, ExpertSet::Routed),
                (6, 5, ExpertSet::Routed),
                (2, 1, ExpertSet::Shared),
            ] {
                let plan =
                    Residency::plan(Path::new(&root), selected, 40, world, rank, set).unwrap();
                assert!(plan.resident_bytes > plan.payload_bytes);
                assert_eq!(plan.loads.len() % 3, 0);
                plan.descriptors(0x10000).unwrap();
                for unit in plan.loads.iter().take(3) {
                    unit.read(&mut vec![0; unit.staging_bytes()])
                        .unwrap();
                }
            }
        }
    }
}
