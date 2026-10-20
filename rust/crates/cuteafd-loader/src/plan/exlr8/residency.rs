//! Checkpoint-native, unit-major EXLR8 residency and verified range reads.
//! Arena offsets are independent of a CUDA address; the owner binds them only
//! after exact weight/workspace admission and retains the arena for graph life.
use super::{Header, Layer, manifest, number, owner};
use crate::plan::checkpoint::read_json;
use anyhow::{Context, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Result<T> = std::result::Result<T, ResidencyError>;

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
    pub checksum: u64,
}
impl Unit {
    pub fn staging_bytes(&self) -> usize {
        self.source.iter().map(|r| r.bytes).sum()
    }

    /// One bounded range read per component; checksum order is payload, channel,
    /// blocks. A checksum failure must never publish this unit to a live arena.
    pub fn read_verified(&self, staging: &mut [u8]) -> Result<()> {
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
        let actual = xxhash64(staging);
        ensure!(
            actual == self.checksum,
            "checksum mismatch e{} u{} p{} K{}: {actual:016x} != {:016x}",
            self.expert,
            self.unit,
            self.projection,
            self.bits,
            self.checksum
        );
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
        "F16" => 2,
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
        Self::plan_inner(snapshot, selected, layer, world, rank, set).map_err(Into::into)
    }

    fn plan_inner(
        snapshot: &Path,
        selected: &str,
        layer: usize,
        world: usize,
        rank: usize,
        set: ExpertSet,
    ) -> anyhow::Result<Self> {
        ensure!(
            (1..=6).contains(&world) && rank < world,
            "expected TP1..6 and rank below world"
        );
        let config = read_json(&snapshot.join("config.json"))?;
        ensure!(
            config["architectures"] == serde_json::json!(["GlmMoeDsaForCausalLM"]),
            "unsupported architecture"
        );
        let hidden = number(&config, "hidden_size")?;
        let intermediate = number(&config, "moe_intermediate_size")?;
        let routed = number(&config, "n_routed_experts")?;
        ensure!(
            hidden > 0
                && hidden % 128 == 0
                && intermediate > 0
                && intermediate % 128 == 0
                && routed > 0
                && layer >= number(&config, "first_k_dense_replace")?
                && layer < number(&config, "num_hidden_layers")?,
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
        let index = Header::open(&index_path)?;
        index.validate()?;
        ensure!(index.bytes <= 64 << 20, "oversized EXLR8 index");
        // Read index tables once, not one open/pread per selected descriptor.
        let index_bytes = std::fs::read(&index_path)?;
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
        let checked = manifest(snapshot, selected, hidden, units, routed, &[layer_spec])?;
        let value = read_json(&snapshot.join(format!("manifests/exlr8-manifest-{selected}.json")))?;
        let choices = &value["layers"][layer.to_string()];
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
        let mut weight_headers = BTreeMap::new();
        let mut loads = Vec::new();
        let mut payload_bytes = 0usize;
        for e in 0..experts {
            let identity = if set == ExpertSet::Shared { routed } else { e };
            for u in 0..units {
                if owner(units, identity, u, layer, world) != rank {
                    continue;
                }
                for (p, projection) in ["gate", "up", "down"].iter().enumerate() {
                    let chosen = if set == ExpertSet::Shared {
                        &choices["shared"][projection]
                    } else {
                        &choices[projection][e]
                    };
                    let k = (if p == 2 { chosen } else { &chosen[u] })
                        .as_u64()
                        .context("missing manifest width")? as usize;
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
                    let checksum = read_u64(
                        range(
                            &index,
                            &index_path,
                            &format!("{stem}.checksum"),
                            "U64",
                            &[experts, 3, units],
                            ((e * 3 + p) * units + u) * 8,
                            8,
                        )?,
                        &index_bytes,
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
                        checksum,
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
            set_id: checked.set_id,
            manifest: selected.into(),
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

// XXH64, seed zero, canonical little-endian arithmetic from the published
// algorithm. Keeping it here avoids depending on an optional native library.
fn xxhash64(data: &[u8]) -> u64 {
    const P1: u64 = 11400714785074694791;
    const P2: u64 = 14029467366897019727;
    const P3: u64 = 1609587929392839161;
    const P4: u64 = 9650029242287828579;
    const P5: u64 = 2870177450012600261;
    fn lane(a: u64, v: u64) -> u64 {
        a.wrapping_add(v.wrapping_mul(P2))
            .rotate_left(31)
            .wrapping_mul(P1)
    }
    let word = |at: usize| u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
    let mut at = 0;
    let mut h;
    if data.len() >= 32 {
        let mut v = [P1.wrapping_add(P2), P2, 0, 0u64.wrapping_sub(P1)];
        while at + 32 <= data.len() {
            for (i, value) in v.iter_mut().enumerate() {
                *value = lane(*value, word(at + i * 8));
            }
            at += 32;
        }
        h = v[0]
            .rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18));
        for value in v {
            h = (h ^ lane(0, value)).wrapping_mul(P1).wrapping_add(P4);
        }
    } else {
        h = P5;
    }
    h = h.wrapping_add(data.len() as u64);
    while at + 8 <= data.len() {
        h = (h ^ lane(0, word(at)))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
        at += 8;
    }
    if at + 4 <= data.len() {
        h = (h
            ^ (u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as u64).wrapping_mul(P1))
        .rotate_left(23)
        .wrapping_mul(P2)
        .wrapping_add(P3);
        at += 4;
    }
    for &byte in &data[at..] {
        h = (h ^ (byte as u64).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::io::Write;

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

    fn fixture(root: &Path, padded: bool) {
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
                            let mut data = vec![(k * 10 + e * 3 + p + u) as u8; bytes];
                            data.extend(vec![(e + p + u) as u8; 256]);
                            data.extend(vec![k as u8; 2]);
                            checksums.extend_from_slice(&xxhash64(&data).to_le_bytes());
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

    #[test]
    fn checksum_matches_independent_e1_vectors() {
        for (n, expected) in [
            (0, 0xef46db3751d8e999),
            (1, 0xe934a84adb052768),
            (4, 0xffced8604453cc1e),
            (8, 0x884a173614b81b8d),
            (31, 0xc346d2b59b4d8ee1),
            (32, 0xcbf59c5116ff32b4),
            (33, 0x0c535d1acafb8ead),
            (1000, 0xf306f04aa88b54d3),
        ] {
            assert_eq!(
                xxhash64(&(0..n).map(|i| (i % 251) as u8).collect::<Vec<_>>()),
                expected
            );
        }
    }

    #[test]
    fn residency_ownership_mixed_widths_shared_and_padded_ranges() {
        for padded in [false, true] {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), padded);
            for world in 1..=6 {
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
                            unit.read_verified(&mut staging).unwrap();
                            assert!(unit.read_verified(&mut staging[1..]).is_err());
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
    fn corrupt_checksum_and_index_are_refused() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), false);
        let plan = Residency::plan(root.path(), "mixed", 3, 1, 0, ExpertSet::Routed).unwrap();
        let unit = &plan.loads[0];
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&unit.source[0].path)
            .unwrap();
        file.seek(SeekFrom::Start(unit.source[0].offset)).unwrap();
        file.write_all(&[0]).unwrap();
        assert!(
            unit.read_verified(&mut vec![0; unit.staging_bytes()])
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );
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
        assert!(Residency::plan(root.path(), "mixed", 3, 7, 0, ExpertSet::Routed).is_err());
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
                    unit.read_verified(&mut vec![0; unit.staging_bytes()])
                        .unwrap();
                }
            }
        }
    }
}
