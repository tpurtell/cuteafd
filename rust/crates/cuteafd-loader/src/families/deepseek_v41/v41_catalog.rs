//! Official checkpoint storage and fixed RTX/Spark-TP4 placement contracts.
use crate::{
    read_official_v41_config, read_safetensors_metadata, OfficialV41Config,
    SafetensorsTensorMetadata,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

thread_local! {
    static PROJECTION_READ_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn projection_read(file: &File, destination: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(destination, offset)?;
    PROJECTION_READ_BYTES.with(|counter| counter.set(counter.get().saturating_add(destination.len())));
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V41TensorPlacement {
    CoordinatorRtx,
    HostMappedEngram,
    /// Every Spark holds one intermediate-dimension quarter of every backbone expert.
    BackboneExpertTp4 {
        layer: usize,
        expert: usize,
        axis: usize,
    },
    /// Packed EXL3 projection; rotations and unequal aligned TP slices require
    /// the projection descriptor rather than the native FP4 slicing rule.
    BackboneExl3,
    /// Per-tensor NVFP4 metadata replicated to every TP rank (global weight
    /// scale and activation scale). Small enough to read whole.
    BackboneExpertReplicated {
        layer: usize,
        expert: usize,
    },
}

#[derive(Debug, Clone)]
pub struct V41Tensor {
    pub shard: String,
    pub metadata: SafetensorsTensorMetadata,
    pub placement: V41TensorPlacement,
}

/// Routed-expert extents every expert path reads, independent of the model
/// family's full configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedExpertShape {
    /// Layer id bound; routed experts live in `first_layer..layers` (earlier
    /// layers are dense).
    pub layers: usize,
    pub first_layer: usize,
    pub experts: usize,
    pub topk: usize,
    pub hidden: usize,
    pub intermediate: usize,
    /// MTP draft stages and their routed experts (0 when the draft is dense).
    pub draft_stages: usize,
    pub draft_experts: usize,
}

impl RoutedExpertShape {
    pub(crate) fn of_v41(config: &OfficialV41Config) -> Self {
        let text = config.text();
        Self {
            layers: text.num_hidden_layers,
            first_layer: 0,
            experts: text.n_routed_experts,
            topk: text.num_experts_per_tok,
            hidden: text.hidden_size,
            intermediate: text.moe_intermediate_size,
            draft_stages: text.num_nextn_predict_layers,
            draft_experts: text.dspark_n_routed_experts,
        }
    }

    pub fn geometry(&self) -> Result<cuteafd_core::ExpertGeometry> {
        let narrow = |value: usize| u32::try_from(value).context("routed expert extent overflow");
        Ok(cuteafd_core::ExpertGeometry {
            hidden: narrow(self.hidden)?,
            experts: narrow(self.experts)?,
            topk: narrow(self.topk)?,
            intermediate: narrow(self.intermediate)?,
            layers: narrow(self.layers)?,
        })
    }
}

#[derive(Debug)]
pub struct OfficialV41Catalog {
    /// The strict V4.1 contract; `None` for other families, which only expose
    /// their routed experts through this catalog.
    config: Option<OfficialV41Config>,
    experts: RoutedExpertShape,
    snapshot: PathBuf,
    tensors: Vec<V41Tensor>,
    exl3: Option<crate::V41Exl3Manifest>,
    nvfp4: Option<crate::V41Nvfp4Contract>,
    fp8: Option<crate::formats::fp8_experts::Fp8ExpertTensors>,
}

/// Bounded range reads for a validated, unsharded coordinator tensor.
pub struct V41CoordinatorTensorReader {
    file: File,
    offset: u64,
    bytes: u64,
}
impl V41CoordinatorTensorReader {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn read_into(&self, offset: u64, destination: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let end = offset
            .checked_add(u64::try_from(destination.len())?)
            .context("coordinator tensor read extent overflow")?;
        ensure!(
            end <= self.bytes,
            "coordinator tensor range exceeds payload"
        );
        self.file.read_exact_at(
            destination,
            self.offset
                .checked_add(offset)
                .context("coordinator tensor file offset overflow")?,
        )?;
        Ok(())
    }
}

impl OfficialV41Catalog {
    pub fn exl3(&self) -> Option<&crate::V41Exl3Manifest> {
        self.exl3.as_ref()
    }
    /// The checkpoint's own FP8 routed experts (E4M3 + FP32 128x128 block
    /// scales), served without re-quantization by the `fp8` family.
    pub fn fp8(&self) -> Option<&crate::formats::fp8_experts::Fp8ExpertTensors> {
        self.fp8.as_ref()
    }
    /// Validated ModelOpt NVFP4 expert contract, when the checkpoint has one.
    pub fn nvfp4(&self) -> Option<&crate::V41Nvfp4Contract> {
        self.nvfp4.as_ref()
    }
    /// True when the draft (MTP) routed experts load as native FP4 weights:
    /// either the checkpoint is not EXL3 at all, or it is a raw EXL3
    /// publication that kept its draft experts at source precision.
    pub fn native_dspark_experts(&self) -> bool {
        self.exl3
            .as_ref()
            .is_none_or(|manifest| manifest.mtp_experts_are_source())
    }
    pub fn coordinator_tensor_reader(&self, name: &str) -> Result<V41CoordinatorTensorReader> {
        let tensor = self.tensor(name)?;
        ensure!(
            matches!(tensor.placement, V41TensorPlacement::CoordinatorRtx),
            "only unsharded RTX tensors support coordinator range reads"
        );
        Ok(V41CoordinatorTensorReader {
            file: File::open(self.snapshot.join(&tensor.shard))?,
            offset: tensor.metadata.byte_offset,
            bytes: tensor.metadata.byte_length,
        })
    }
    /// The V4.1 configuration. Catalogs of other families carry only their
    /// routed experts; V4.1-only callers never receive one.
    pub fn config(&self) -> &OfficialV41Config {
        self.config
            .as_ref()
            .expect("only DeepSeek V4.1 catalogs carry the official configuration")
    }
    pub fn routed_experts(&self) -> &RoutedExpertShape {
        &self.experts
    }
    pub fn snapshot(&self) -> &Path {
        &self.snapshot
    }
    pub fn tensors(&self) -> &[V41Tensor] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&V41Tensor> {
        let index = self
            .tensors
            .binary_search_by(|tensor| tensor.metadata.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("unknown official tensor {name}"))?;
        Ok(&self.tensors[index])
    }

    /// Count bytes delivered by positioned projection reads on this thread.
    /// This measures requested storage I/O even when the OS serves a cached page,
    /// not block-device traffic. Reader threads must each enter their own scope.
    pub fn count_storage_reads<T>(read: impl FnOnce() -> Result<T>) -> Result<(T, usize)> {
        struct Restore(usize);
        impl Drop for Restore {
            fn drop(&mut self) { PROJECTION_READ_BYTES.with(|counter| counter.set(self.0)); }
        }
        let _restore = Restore(PROJECTION_READ_BYTES.with(|counter| counter.replace(0)));
        let value = read()?;
        Ok((value, PROJECTION_READ_BYTES.with(|counter| counter.get())))
    }

    /// One complete physical projection read, counted at the I/O boundary.
    /// TP2 callers validate that the union of their slices covers this extent.
    pub fn read_projection_once(&self, name: &str, destination: &mut [u8]) -> Result<usize> {
        use std::os::unix::fs::FileExt;
        let tensor = self.tensor(name)?;
        let bytes = usize::try_from(tensor.metadata.byte_length)?;
        ensure!(destination.len() == bytes, "projection staging size mismatch for {name}");
        projection_read(&File::open(self.snapshot.join(&tensor.shard))?, destination, tensor.metadata.byte_offset)
            .with_context(|| format!("shared TP2 projection read {name}"))?;
        Ok(bytes)
    }

    /// Shape of one native TP2 slice, suitable for a pitched H2D copy from a
    /// complete physical projection without a second host-side packed copy.
    pub fn backbone_tp2_slice(&self, name: &str, rank: usize) -> Result<crate::V41Exl3TensorSlice> {
        let tensor = self.tensor(name)?;
        ensure!(rank < 2, "invalid TP2 rank");
        let V41TensorPlacement::BackboneExpertTp4 { axis, .. } = tensor.placement else {
            anyhow::bail!("TP2 slice requires a native backbone projection");
        };
        let bytes = usize::try_from(tensor.metadata.byte_length)?;
        ensure!(bytes % 2 == 0, "uneven TP2 projection bytes");
        let slice = if axis == 0 {
            crate::V41Exl3TensorSlice { rows: 1, source_row_bytes: bytes,
                column_start_bytes: rank * (bytes / 2), selected_row_bytes: bytes / 2 }
        } else {
            ensure!(axis == 1 && tensor.metadata.shape.len() == 2, "invalid TP2 projection axis");
            let rows = tensor.metadata.shape[0];
            ensure!(rows > 0 && bytes % rows == 0 && (bytes / rows) % 2 == 0,
                "uneven TP2 projection rows");
            crate::V41Exl3TensorSlice { rows, source_row_bytes: bytes / rows,
                column_start_bytes: rank * (bytes / rows / 2), selected_row_bytes: bytes / rows / 2 }
        };
        Ok(slice)
    }

    pub fn device_tensor_bytes(&self, name: &str, spark_rank: Option<usize>) -> Result<u64> {
        let tensor = self.tensor(name)?;
        match tensor.placement {
            V41TensorPlacement::BackboneExl3 => {
                let rank = spark_rank.context("EXL3 backbone requires a Spark rank")?;
                self.exl3_tensor_bytes(name, 4, rank)
            }
            V41TensorPlacement::HostMappedEngram => {
                anyhow::bail!("engram tables must be mapped, not eagerly loaded")
            }
            V41TensorPlacement::CoordinatorRtx => {
                ensure!(
                    spark_rank.is_none(),
                    "RTX tensor {name} cannot be loaded as a Spark shard"
                );
                Ok(tensor.metadata.byte_length)
            }
            V41TensorPlacement::BackboneExpertTp4 { .. } => {
                ensure!(
                    spark_rank.is_some_and(|rank| rank < 4),
                    "backbone expert {name} requires a Spark TP rank in 0..4"
                );
                Ok(tensor.metadata.byte_length / 4)
            }
            V41TensorPlacement::BackboneExpertReplicated { .. } => Ok(tensor.metadata.byte_length),
        }
    }

    /// Read the selected physical tensor/shard into caller-owned staging memory.
    /// Column-sharded W2 uses bounded scratch to coalesce file reads, then packs rows.
    pub fn read_device_tensor_into(
        &self,
        name: &str,
        spark_rank: Option<usize>,
        dst: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<usize> {
        if matches!(self.tensor(name)?.placement, V41TensorPlacement::BackboneExl3) {
            return self.read_exl3_tensor_into(name, 4,
                spark_rank.context("EXL3 backbone requires a Spark rank")?, dst, scratch);
        }
        if let V41TensorPlacement::BackboneExpertReplicated { .. } = self.tensor(name)?.placement {
            let bytes = usize::try_from(self.tensor(name)?.metadata.byte_length)?;
            ensure!(
                dst.len() >= bytes,
                "replicated expert scalar {name} needs {bytes} bytes"
            );
            use std::os::unix::fs::FileExt;
            let tensor = self.tensor(name)?;
            File::open(self.snapshot.join(&tensor.shard))?
                .read_exact_at(&mut dst[..bytes], tensor.metadata.byte_offset)
                .with_context(|| format!("staging replicated expert scalar {name}"))?;
            return Ok(bytes);
        }
        let bytes = usize::try_from(self.device_tensor_bytes(name, spark_rank)?)?;
        let axis = match self.tensor(name)?.placement {
            V41TensorPlacement::BackboneExpertTp4 { axis, .. } => Some(axis),
            _ => None,
        };
        self.read_tensor_partition(name, spark_rank, 4, axis, bytes, dst, scratch)
    }

    pub(crate) fn read_backbone_tp2_into(
        &self, name: &str, rank: usize, dst: &mut [u8], scratch: &mut [u8],
    ) -> Result<usize> {
        let tensor = self.tensor(name)?;
        ensure!(rank < 2 && matches!(tensor.placement, V41TensorPlacement::BackboneExpertTp4 { .. }),
            "TP2 read requires a backbone expert and rank in 0..2");
        let bytes = usize::try_from(tensor.metadata.byte_length / 2)?;
        let V41TensorPlacement::BackboneExpertTp4 { axis, .. } = tensor.placement else { unreachable!() };
        self.read_tensor_partition(name, Some(rank), 2, Some(axis), bytes, dst, scratch)
    }

    /// Read a row or column half of an ordinary coordinator matrix. Shapes and
    /// dtype remain native; column reads use caller-owned full-row scratch.
    pub fn read_coordinator_tp2_into(&self, name: &str, axis: usize, rank: usize,
        dst: &mut [u8], scratch: &mut [u8]) -> Result<usize> {
        let tensor = self.tensor(name)?;
        ensure!(matches!(tensor.placement, V41TensorPlacement::CoordinatorRtx)
            && tensor.metadata.shape.len() == 2 && axis < 2 && rank < 2,
            "TP2 coordinator reads require a matrix, axis 0/1, and rank 0/1");
        ensure!(tensor.metadata.shape[axis] % 2 == 0 && tensor.metadata.byte_length % 2 == 0,
            "coordinator matrix cannot be divided in half");
        let bytes = usize::try_from(tensor.metadata.byte_length / 2)?;
        self.read_tensor_partition(name, Some(rank), 2, Some(axis), bytes, dst, scratch)
    }

    fn read_tensor_partition(
        &self, name: &str, spark_rank: Option<usize>, partitions: usize, axis: Option<usize>, bytes: usize,
        dst: &mut [u8], scratch: &mut [u8],
    ) -> Result<usize> {
        use std::os::unix::fs::FileExt;
        ensure!(
            dst.len() >= bytes,
            "tensor staging buffer for {name} needs {bytes} bytes"
        );
        let tensor = self.tensor(name)?;
        let metadata = &tensor.metadata;
        let file = File::open(self.snapshot.join(&tensor.shard))?;
        match axis {
            Some(0) => {
                let offset = metadata
                    .byte_offset
                    .checked_add(
                        (bytes as u64)
                            .checked_mul(spark_rank.unwrap() as u64)
                            .context("TP row offset overflow")?,
                    )
                    .context("TP file offset overflow")?;
                projection_read(&file, &mut dst[..bytes], offset)?;
            }
            Some(1) => {
                let rows = metadata.shape[0];
                let row_bytes = usize::try_from(metadata.byte_length / rows as u64)?;
                ensure!(
                    scratch.len() >= row_bytes,
                    "W2 scratch needs at least {row_bytes} bytes"
                );
                let shard_bytes = row_bytes / partitions;
                let column = spark_rank.unwrap() * shard_bytes;
                let rows_per_read = scratch.len() / row_bytes;
                for start in (0..rows).step_by(rows_per_read) {
                    let count = rows_per_read.min(rows - start);
                    let offset = metadata
                        .byte_offset
                        .checked_add(
                            (start as u64)
                                .checked_mul(row_bytes as u64)
                                .context("TP column row offset overflow")?,
                        )
                        .context("TP column file offset overflow")?;
                    projection_read(&file, &mut scratch[..count * row_bytes], offset)?;
                    for row in 0..count {
                        dst[(start + row) * shard_bytes..(start + row + 1) * shard_bytes].copy_from_slice(
                            &scratch[row * row_bytes + column..row * row_bytes + column + shard_bytes],
                        );
                    }
                }
            }
            None => {
                projection_read(&file, &mut dst[..bytes], metadata.byte_offset)?
            }
            _ => anyhow::bail!("unsupported device tensor placement"),
        }
        Ok(bytes)
    }

    /// # Safety
    /// Checkpoint files must remain immutable for the lifetime of the returned maps.
    pub unsafe fn map_engram(&self, layer: usize) -> Result<crate::EngramTable> {
        ensure!(
            self.config().text().engram_layer_ids.contains(&layer),
            "no engram table at layer {layer}"
        );
        let map = |suffix: &str| -> Result<crate::MappedTable> {
            let tensor = self.tensor(&format!("layers.{layer}.engram.embed.{suffix}"))?;
            let metadata = &tensor.metadata;
            ensure!(metadata.shape.len() == 2, "engram tensor {suffix} is not 2-D");
            // SAFETY: forwarded from this function's contract.
            let table = unsafe {
                crate::MappedTable::single(
                    &self.snapshot.join(&tensor.shard),
                    metadata.byte_offset,
                    metadata.shape[0] as u64,
                    // One byte per element for every engram encoding (FP8, UE8M0, packed NVFP4 bytes).
                    crate::RowFormat { dtype: metadata.dtype.clone(), width: metadata.shape[1], row_bytes: metadata.shape[1] },
                )?
            };
            table.name_stats(format!("engram.{layer}.{suffix}"));
            Ok(table)
        };
        if let Some(ple) = self.exl3.as_ref().and_then(|m| m.ple_quantization.as_ref()) {
            use std::os::unix::fs::FileExt;
            let prefix = format!("layers.{layer}.engram.embed");
            let tensor = self.tensor(&format!("{prefix}.weight_scale_2"))?;
            let mut bytes = [0; 4];
            File::open(self.snapshot.join(&tensor.shard))?.read_exact_at(&mut bytes, tensor.metadata.byte_offset)?;
            let global = f32::from_le_bytes(bytes);
            let declared = ple["tensors"][&prefix]["global_scale"].as_f64().context("missing NVFP4 global scale")? as f32;
            ensure!(global.to_bits() == declared.to_bits(), "NVFP4 global scale differs from metadata");
            crate::EngramTable::new_nvfp4(map("weight")?, map("weight_scale")?, global)
        } else {
            crate::EngramTable::new(map("weight")?, map("scale")?)
        }
    }

    /// Physical checkpoint bytes only; packing, caches and execution scratch are additional.
    pub fn storage_budget(&self) -> Result<V41StorageBudget> {
        let mut budget = V41StorageBudget::default();
        for tensor in &self.tensors {
            let bytes = tensor.metadata.byte_length;
            match tensor.placement {
                V41TensorPlacement::CoordinatorRtx => {
                    budget.coordinator_bytes = budget
                        .coordinator_bytes
                        .checked_add(bytes)
                        .context("RTX weight byte overflow")?;
                    if tensor.metadata.name.starts_with("mtp.") {
                        budget.dspark_bytes = budget
                            .dspark_bytes
                            .checked_add(bytes)
                            .context("dSpark weight byte overflow")?;
                    }
                }
                V41TensorPlacement::HostMappedEngram => {
                    budget.host_mapped_bytes = budget
                        .host_mapped_bytes
                        .checked_add(bytes)
                        .context("engram byte overflow")?;
                }
                V41TensorPlacement::BackboneExpertTp4 { .. } => {
                    ensure!(
                        bytes % 4 == 0,
                        "TP4 tensor byte count is not divisible by four"
                    );
                    budget.per_spark_bytes = budget
                        .per_spark_bytes
                        .checked_add(bytes / 4)
                        .context("Spark weight byte overflow")?;
                    for rank_bytes in &mut budget.spark_rank_bytes {
                        *rank_bytes = rank_bytes.checked_add(bytes / 4).context("Spark rank budget overflow")?;
                    }
                }
                V41TensorPlacement::BackboneExl3 => {
                    for rank in 0..4 {
                        budget.spark_rank_bytes[rank] = budget.spark_rank_bytes[rank]
                            .checked_add(self.exl3_tensor_bytes(&tensor.metadata.name, 4, rank)?)
                            .context("EXL3 Spark rank budget overflow")?;
                    }
                }
                V41TensorPlacement::BackboneExpertReplicated { .. } => {
                    for rank_bytes in &mut budget.spark_rank_bytes {
                        *rank_bytes = rank_bytes
                            .checked_add(bytes)
                            .context("replicated expert scalar budget overflow")?;
                    }
                }
            }
        }
        budget.per_spark_bytes = *budget.spark_rank_bytes.iter().max().unwrap();
        Ok(budget)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct V41StorageBudget {
    pub coordinator_bytes: u64,
    /// Included in coordinator_bytes; shared embedding/output tensors are counted only once.
    pub dspark_bytes: u64,
    pub per_spark_bytes: u64,
    /// Exact per-rank bytes; per_spark_bytes is their maximum for legacy callers.
    pub spark_rank_bytes: [u64; 4],
    pub host_mapped_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Template {
    pattern: String,
    axes: Vec<Vec<usize>>,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ExpectedTensor {
    dtype: DType,
    shape: Vec<usize>,
    bytes: u64,
}

fn expected_tensors() -> Result<BTreeMap<String, ExpectedTensor>> {
    let templates: Vec<Template> = serde_json::from_str(include_str!("official-v41-tensors.json"))?;
    let mut expected = BTreeMap::new();
    for template in templates {
        let dtype = DType::from_safetensors(&template.dtype);
        let width = match dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
            DType::F8E4M3 | DType::F8E8M0 | DType::I8 => 1,
            _ => anyhow::bail!("unsupported official tensor dtype {}", template.dtype),
        };
        let bytes = template
            .shape
            .iter()
            .try_fold(width, |n: u64, &dim| n.checked_mul(dim as u64))
            .context("tensor size overflow")?;
        let mut names = vec![template.pattern];
        for axis in template.axes {
            names = names
                .into_iter()
                .flat_map(|name| {
                    axis.iter()
                        .map(move |i| name.replacen("{}", &i.to_string(), 1))
                })
                .collect();
        }
        for name in names {
            ensure!(!name.contains("{}"), "incomplete tensor template {name}");
            ensure!(
                expected
                    .insert(
                        name.clone(),
                        ExpectedTensor {
                            dtype: dtype.clone(),
                            shape: template.shape.clone(),
                            bytes
                        }
                    )
                    .is_none(),
                "duplicate tensor template {name}"
            );
        }
    }
    ensure!(
        expected.len() == 96085,
        "incomplete official tensor contract"
    );
    Ok(expected)
}

/// NVFP4 backbone experts shard like native FP4 weights, except the
/// per-tensor FP32 global weight and activation scales, which replicate.
fn nvfp4_placement(name: &str) -> V41TensorPlacement {
    if name.ends_with(".weight_scale_2") || name.ends_with(".input_scale") {
        let parts: Vec<_> = name.split('.').collect();
        return V41TensorPlacement::BackboneExpertReplicated {
            layer: parts[1].parse().expect("validated NVFP4 layer"),
            expert: parts[4].parse().expect("validated NVFP4 expert"),
        };
    }
    placement(name)
}

fn placement(name: &str) -> V41TensorPlacement {
    // Keep the complete draft model local, including its distinct routed
    // experts. Decide this before applying any backbone offload rules.
    if name.starts_with("mtp.") {
        return V41TensorPlacement::CoordinatorRtx;
    }
    if name.contains(".engram.embed.") {
        return V41TensorPlacement::HostMappedEngram;
    }
    let parts: Vec<_> = name.split('.').collect();
    if parts.len() == 7 && parts[0] == "layers" && parts[2] == "ffn" && parts[3] == "experts" {
        V41TensorPlacement::BackboneExpertTp4 {
            layer: parts[1].parse().expect("validated official layer"),
            expert: parts[4].parse().expect("validated official expert"),
            axis: usize::from(parts[5] == "w2"),
        }
    } else {
        V41TensorPlacement::CoordinatorRtx
    }
}

/// Header-only inspection: never reads or eagerly allocates checkpoint tensor payloads.
pub fn read_official_v41_catalog(model_id: &str, snapshot: &Path) -> Result<OfficialV41Catalog> {
    read_official_v41_catalog_filtered(model_id, snapshot, None)
}

/// Official-format worker inventory, without opening coordinator-only shards.
pub fn read_official_v41_spark_catalog(snapshot: &Path, rank: usize, world: usize) -> Result<OfficialV41Catalog> {
    ensure!(rank < world, "Spark rank {rank} outside TP{world}");
    read_official_v41_catalog_filtered(crate::OFFICIAL_V41_MODEL_ID, snapshot,
        Some(format!("spark{rank} (TP{world})")))
}

fn read_official_v41_catalog_filtered(model_id: &str, snapshot: &Path, spark_role: Option<String>) -> Result<OfficialV41Catalog> {
    let raw_config: serde_json::Value = crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    let exl3 = if raw_config["quantization_config"]["quant_method"] == "exl3" {
        Some(crate::read_v41_exl3_manifest(snapshot)?)
    } else { None };
    let nvfp4 = if exl3.is_none() {
        crate::read_v41_nvfp4_contract(snapshot)?
    } else { None };
    let config = match (&exl3, &nvfp4) {
        (Some(manifest), _) => manifest
            .config
            .clone()
            .context("V4.1 EXL3 manifest lacks the official configuration")?,
        (None, Some(contract)) => contract.config.clone(),
        (None, None) => read_official_v41_config(model_id, snapshot)?,
    };
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let mut bytes = Vec::new();
    File::open(snapshot.join("model.safetensors.index.json"))?
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 64 * 1024 * 1024,
        "checkpoint index exceeds sixty-four MiB"
    );
    let index: Index = serde_json::from_slice(&bytes)?;
    let mut expected = expected_tensors()?;
    if let Some(manifest) = &exl3 {
        // Raw publications keep their MTP draft experts at the native FP4
        // source precision, so only backbone expert tensors leave the
        // official contract; staged snapshots quantize drafts too.
        let keep_native_mtp = manifest.mtp_experts_are_source();
        expected.retain(|name, _| {
            !name.contains(".ffn.experts.") || (keep_native_mtp && name.starts_with("mtp."))
        });
        for projection in manifest.projections.values() {
            for (suffix, dtype, shape, bytes) in [
                ("trellis", DType::I16, projection.trellis_shape().to_vec(), projection.trellis_bytes()),
                ("suh", DType::F16, vec![projection.input_features], projection.input_features * 2),
                ("svh", DType::F16, vec![projection.output_features], projection.output_features * 2),
                ("mcg", DType::I32, vec![], 4),
            ] {
                expected.insert(format!("{}.{suffix}", projection.name),
                    ExpectedTensor { dtype, shape, bytes: bytes as u64 });
            }
        }
        if let Some(ple) = &manifest.ple_quantization {
            apply_nvfp4_ple_contract(&mut expected, ple, config.text().engram_layer_ids.as_slice())?;
        }
    }
    if let Some(contract) = &nvfp4 {
        // ModelOpt NVFP4 replaces the two native routed-expert tensors with a
        // packed E2M1 payload, an E4M3 per-16 scale plane, a global weight
        // scale and an activation scale for W4A4. Draft experts keep the
        // native FP4 contract.
        apply_nvfp4_expert_contract(&mut expected, contract, config.text())?;
    }
    ensure!(
        index.weight_map.len() == expected.len(),
        "official tensor inventory count mismatch: expected {}, got {}",
        expected.len(),
        index.weight_map.len()
    );
    let shard_count = if exl3.is_some() {
        index.weight_map.values().collect::<BTreeSet<_>>().len()
    } else { 48 };
    let allowed_shards: BTreeSet<_> = (1..=shard_count)
        .map(|i| format!("model-{i:05}-of-{shard_count:05}.safetensors"))
        .collect();
    let mut shards: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, shard) in &index.weight_map {
        ensure!(
            expected.contains_key(name),
            "unexpected checkpoint tensor {name}"
        );
        ensure!(
            allowed_shards.contains(shard),
            "unsupported checkpoint shard {shard}"
        );
        shards
            .entry(shard.clone())
            .or_default()
            .insert(name.clone());
    }
    ensure!(
        shards.len() == shard_count,
        "checkpoint requires all {shard_count} shards"
    );
    let mut tensors = Vec::with_capacity(expected.len());
    let needed = |name: &str| spark_role.is_none() || name.starts_with("layers.") && name.contains(".ffn.experts.");
    for (shard, names) in shards {
        let Some(first_needed) = names.iter().find(|name| needed(name)) else { continue };
        let path = snapshot.join(&shard);
        let metadata = read_safetensors_metadata(&path).with_context(|| format!(
            "role {} needs tensor {first_needed} in shard {shard} at snapshot {}",
            spark_role.as_deref().unwrap_or("full catalog"), snapshot.display()))?;
        ensure!(
            metadata.len() == names.len(),
            "index/header tensor count mismatch in {shard}"
        );
        let mut intervals = Vec::with_capacity(metadata.len());
        for tensor in metadata {
            ensure!(
                names.contains(&tensor.name),
                "tensor {} appears in unexpected shard {shard}",
                tensor.name
            );
            let spec = &expected[&tensor.name];
            validate_tensor(&tensor, spec)?;
            let target = if exl3.is_some() && tensor.name.starts_with("layers.")
                && tensor.name.contains(".ffn.experts.") {
                V41TensorPlacement::BackboneExl3
            } else if nvfp4.is_some() && tensor.name.starts_with("layers.")
                && tensor.name.contains(".ffn.experts.") {
                nvfp4_placement(&tensor.name)
            } else { placement(&tensor.name) };
            if let V41TensorPlacement::BackboneExpertTp4 { axis, .. } = target {
                ensure!(
                    tensor.shape[axis] % 4 == 0,
                    "TP4 axis cannot be divided for {}",
                    tensor.name
                );
            }
            intervals.push((
                tensor.byte_offset,
                tensor
                    .byte_offset
                    .checked_add(tensor.byte_length)
                    .context("tensor end overflow")?,
            ));
            tensors.push(V41Tensor {
                shard: shard.clone(),
                metadata: tensor,
                placement: target,
            });
        }
        intervals.sort_unstable();
        let mut header_len = [0u8; 8];
        File::open(&path)?.read_exact(&mut header_len)?;
        let mut cursor = u64::from_le_bytes(header_len)
            .checked_add(8)
            .context("header offset overflow")?;
        for (start, end) in intervals {
            ensure!(
                start == cursor,
                "overlap or gap in {shard} at byte {cursor}"
            );
            cursor = end;
        }
        ensure!(
            cursor == path.metadata()?.len(),
            "unindexed trailing bytes in {shard}"
        );
    }
    tensors.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    Ok(OfficialV41Catalog {
        experts: RoutedExpertShape::of_v41(&config),
        config: Some(config),
        snapshot: snapshot.to_path_buf(),
        tensors,
        exl3,
        nvfp4,
        fp8: None,
    })
}

/// Opens the routed experts of any supported checkpoint: the strict V4.1
/// contract, or a DeepSeek V4 (Flash/Pro) checkpoint whose experts share the
/// V4.1 storage (packed FP4 in I8, E8M0 scales per 32 values).
pub fn read_expert_catalog(snapshot: &Path) -> Result<OfficialV41Catalog> {
    let config: serde_json::Value =
        crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    if config.get("model_type").and_then(serde_json::Value::as_str) == Some("glm5_next") {
        return read_glm_dsa_expert_catalog(snapshot, &config);
    }
    if config.get("model_type").and_then(serde_json::Value::as_str) == Some("qwen4_exp") {
        return read_qwen4_expert_catalog(snapshot, &config);
    }
    if config.get("text_config").is_some() {
        return read_official_v41_catalog(crate::OFFICIAL_V41_MODEL_ID, snapshot);
    }
    match config.get("model_type").and_then(serde_json::Value::as_str) {
        Some("deepseek_v4") => read_deepseek_v4_expert_catalog(snapshot),
        Some("glm_moe_dsa") => read_glm_dsa_expert_catalog(snapshot, &config),
        Some("mimo_v2_flash" | "mimo_v2") => read_mimo_v2_expert_catalog(snapshot, &config),
        other => anyhow::bail!(
            "the Spark expert service does not know model_type {other:?}; add a family \
             reader next to read_deepseek_v4_expert_catalog that maps its routed expert \
             tensors onto the six-region W1,W3,W2,S1,S3,S2 staging plan"
        ),
    }
}

/// GLM 5.x (glm_moe_dsa) and GLM 5.3 Flash (glm5_next) routed experts. Only EXL3 publications serve from
/// the Sparks today (the official FP8 experts need ~675 GiB); dense layers
/// come first, and the MTP layer after the backbone keeps its experts on the
/// coordinator.
fn read_glm_dsa_expert_catalog(snapshot: &Path, config: &serde_json::Value) -> Result<OfficialV41Catalog> {
    // The geometry from the coordinator's own config readers, so the Spark
    // ranks and serve-glm / serve-glmf agree on the routed layers.
    let (dense, mtp, shape) = if config.get("model_type").and_then(serde_json::Value::as_str) == Some("glm5_next") {
        let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(config)?;
        let text = config.get("text_config").unwrap_or(config);
        let mtp = text["num_nextn_predict_layers"].as_u64().unwrap_or(0) as usize;
        (cfg.dense.clone(), mtp, (cfg.layers, cfg.experts, cfg.topk, cfg.hidden, cfg.moe_intermediate))
    } else {
        let cfg = crate::families::glm5::GlmDsaConfig::from_hf(config)?;
        let dense = (0..cfg.layers).map(|layer| layer < cfg.first_moe_layer).collect::<Vec<_>>();
        (dense, cfg.mtp_layers, (cfg.layers, cfg.experts, cfg.topk, cfg.hidden, cfg.moe_intermediate))
    };
    let (layers, experts, topk, hidden, intermediate) = shape;
    let first_layer = dense.iter().position(|dense| !dense).context("GLM config has no MoE layer")?;
    ensure!(dense[first_layer..].iter().all(|dense| !dense), "GLM MoE layers must follow the dense ones");
    let shape = RoutedExpertShape {
        layers,
        first_layer,
        experts,
        topk,
        hidden,
        intermediate,
        draft_stages: 0,
        draft_experts: 0,
    };
    if let Some(catalog) = nvfp4_catalog(snapshot, config, shape)? {
        return Ok(catalog);
    }
    if config["quantization_config"]["quant_method"] == "fp8" {
        // The official FP8 experts (~675 GiB) do not fit the Spark pool; the
        // catalog serves selected layers, above all the MTP layer, whose ids
        // follow the backbone (num_hidden_layers..): the id bound includes them.
        return fp8_catalog(snapshot, RoutedExpertShape { layers: layers + mtp, ..shape });
    }
    ensure!(
        config["quantization_config"]["quant_method"] == "exl3",
        "GLM routed experts serve from EXL3 publications (e.g. wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1), \
         ModelOpt NVFP4 releases or, coordinator-local, from the official FP8 checkpoint"
    );
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let index: Index = crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("model.safetensors.index.json"), 64 * 1024 * 1024)
        .and_then(|value| Ok(serde_json::from_value(value)?))?;
    // The tier pair of the family's expert packages (glmf:exl3-k34, glm:exl3-k45).
    let tiers: &[usize] = if config.get("model_type").and_then(serde_json::Value::as_str) == Some("glm5_next") {
        &[3, 4]
    } else {
        &[4, 5]
    };
    deepseek_v4_exl3_catalog(snapshot, &index.weight_map, shape, tiers)
}

/// Qwen 3.8 Flash Next (qwen4_exp) routed experts: every layer is MoE (512
/// experts, top-10). The official FP8 checkpoint stores E4M3 experts with BF16
/// 128x128 block scales; the EXL3 publications (wrldsuksgo2mars
/// Qwen3.8-Flash-Next-EXL3-K4.25-*) mixed K4/K5 Trellis projections with the
/// MTP experts under `mtp.layers.0`. The BF16 release fuses each layer's
/// experts into `[512, ...]` tensors and NVIDIA's release is NVFP4: neither
/// has an expert package.
/// Official-format coordinator-local Qwen MTP experts. EXL3 remains on the
/// complete catalog path until its per-projection metadata is role-filtered.
pub fn read_qwen4_mtp_expert_catalog(snapshot: &Path, stages: usize) -> Result<OfficialV41Catalog> {
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    if config["quantization_config"]["quant_method"] == "exl3" {
        return read_qwen4_expert_catalog(snapshot, &config);
    }
    let cfg = crate::families::qwen4::Qwen4Config::from_hf(&config)?;
    ensure!(stages > 0 && stages <= cfg.mtp_layers, "Qwen MTP role requests {stages} stages, config provides {}", cfg.mtp_layers);
    let shape = RoutedExpertShape { layers: cfg.layers, first_layer: 0, experts: cfg.experts,
        topk: cfg.topk, hidden: cfg.hidden, intermediate: cfg.moe_intermediate, draft_stages: 0, draft_experts: 0 };
    let layers = (cfg.layers..cfg.layers + stages).collect();
    let fp8 = crate::formats::fp8_experts::Fp8ExpertTensors::read_layers(snapshot, shape, &layers)?;
    for layer in layers { fp8.for_layer(layer)?; }
    Ok(OfficialV41Catalog { config: None, experts: shape, snapshot: snapshot.to_path_buf(), tensors: Vec::new(),
        exl3: None, nvfp4: None, fp8: Some(fp8) })
}

fn read_qwen4_expert_catalog(snapshot: &Path, config: &serde_json::Value) -> Result<OfficialV41Catalog> {
    let cfg = crate::families::qwen4::Qwen4Config::from_hf(config)?;
    let shape = RoutedExpertShape {
        layers: cfg.layers,
        first_layer: 0,
        experts: cfg.experts,
        topk: cfg.topk,
        hidden: cfg.hidden,
        intermediate: cfg.moe_intermediate,
        draft_stages: 0,
        draft_experts: 0,
    };
    if let Some(catalog) = nvfp4_catalog(snapshot, config, shape)? {
        return Ok(catalog);
    }
    let quant = &config["quantization_config"];
    if quant["quant_method"] == "fp8" {
        ensure!(quant["weight_block_size"] == serde_json::json!([128, 128]),
            "Qwen FP8 routed experts must use 128x128 weight blocks");
        return fp8_catalog(snapshot, shape);
    }
    if quant["quant_method"] == "exl3" {
        #[derive(Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let index: Index =
            crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("model.safetensors.index.json"), 64 * 1024 * 1024)
                .and_then(|value| Ok(serde_json::from_value(value)?))?;
        // qwen4:exl3-k45.
        return deepseek_v4_exl3_catalog(snapshot, &index.weight_map, shape, &[4, 5]);
    }
    if quant.get("config_groups").is_some() || quant["quant_method"] == "modelopt" {
        anyhow::bail!("Qwen ModelOpt routed experts other than NVFP4 have no expert package: serve the \
            NVFP4, FP8 (Qwen/Qwen3.8-Flash-Next-FP8) or EXL3 (wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-*) release");
    }
    anyhow::bail!("Qwen BF16 routed experts are fused [512, ...] tensors with no expert package: serve the \
        FP8 (Qwen/Qwen3.8-Flash-Next-FP8) or EXL3 (wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-*) release")
}

/// MiMo V2 (mimo_v2_flash) routed experts: the checkpoint's FP8 E4M3 weights
/// with FP32 128x128 block scales (`model.layers.{l}.mlp.experts.{e}.
/// {gate,up,down}_proj.weight[_scale_inv]`); layer 0 is dense. V2.6 Pro
/// (mimo_v2) stores them MXFP4 (`weight` + UE8M0 `weight_scale`, quantization
/// `store_dtype: mxfp4`); the same catalog detects the format.
fn read_mimo_v2_expert_catalog(snapshot: &Path, config: &serde_json::Value) -> Result<OfficialV41Catalog> {
    let cfg = crate::families::mimo_v2::MimoV2Config::from_hf(config)?;
    let first_layer = cfg.dense.iter().position(|dense| !dense).context("MiMo config has no MoE layer")?;
    ensure!(cfg.dense[first_layer..].iter().all(|dense| !dense), "MiMo MoE layers must follow the dense ones");
    ensure!(cfg.routed_scale == 1.0, "MiMo routed scaling must be 1");
    ensure!(config["quantization_config"]["quant_method"] == "fp8"
        && config["quantization_config"]["weight_block_size"] == serde_json::json!([128, 128]),
        "MiMo routed experts serve from the checkpoint's FP8 128x128 block weights");
    fp8_catalog(snapshot, RoutedExpertShape {
        layers: cfg.layers,
        first_layer,
        experts: cfg.experts,
        topk: cfg.topk,
        hidden: cfg.hidden,
        intermediate: cfg.moe_intermediate,
        draft_stages: 0,
        draft_experts: 0,
    })
}

/// The routed experts of a ModelOpt export whose metadata (hf_quant_config.json,
/// config.json) declares them NVFP4: the backbone's routed layers through the
/// `fp8` family's NVFP4 packages. `None` for other checkpoints.
fn nvfp4_catalog(snapshot: &Path, config: &serde_json::Value, shape: RoutedExpertShape)
    -> Result<Option<OfficialV41Catalog>> {
    let Some(modelopt) = crate::formats::modelopt::ModelOpt::read(snapshot, config)? else {
        return Ok(None);
    };
    if !modelopt.experts_nvfp4() {
        return Ok(None);
    }
    let mut catalog = fp8_catalog(snapshot, shape)?;
    let tensors = catalog.fp8().context("NVFP4 catalog")?;
    ensure!(tensors.format() == crate::formats::fp8_experts::ExpertFormat::Nvfp4,
        "{} declares NVFP4 routed experts, but layer {} stores {:?}", modelopt.source, shape.first_layer,
        tensors.format());
    // The coordinator's tensors (serve-glm reads its weights through the catalog):
    // everything but the backbone's routed experts, which Fp8ExpertTensors serves.
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let index: Index = crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("model.safetensors.index.json"),
        64 * 1024 * 1024).and_then(|value| Ok(serde_json::from_value(value)?))?;
    let mut coordinator: Vec<V41Tensor> = read_index_headers(snapshot, &index.weight_map)?.into_iter()
        .filter(|(_, tensor)| !(tensor.name.contains(".mlp.experts.")
            && hf_layer(&tensor.name).is_some_and(|layer| layer < shape.layers)))
        .map(|(shard, metadata)| V41Tensor { shard, metadata, placement: V41TensorPlacement::CoordinatorRtx })
        .collect();
    coordinator.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    catalog.tensors = coordinator;
    Ok(Some(catalog))
}

fn fp8_catalog(snapshot: &Path, shape: RoutedExpertShape) -> Result<OfficialV41Catalog> {
    let fp8 = crate::formats::fp8_experts::Fp8ExpertTensors::read(snapshot, shape)?;
    Ok(OfficialV41Catalog {
        config: None,
        experts: shape,
        snapshot: snapshot.to_path_buf(),
        tensors: Vec::new(),
        exl3: None,
        nvfp4: None,
        fp8: Some(fp8),
    })
}

/// DeepSeek V4 routed experts; every other tensor stays with the coordinator
/// and is not validated here.
fn read_deepseek_v4_expert_catalog(snapshot: &Path) -> Result<OfficialV41Catalog> {
    let v4 = crate::families::deepseek_v4::DeepseekV4Config::read(snapshot, 0)?;
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let index: Index = crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("model.safetensors.index.json"), 64 * 1024 * 1024)
        .and_then(|value| Ok(serde_json::from_value(value)?))?;
    let (hidden, intermediate) = (v4.dim, v4.moe_inter_dim);
    ensure!(
        hidden % 32 == 0 && intermediate % 32 == 0,
        "DeepSeek V4 expert extents {hidden}x{intermediate} are not K32-aligned"
    );
    let backbone = RoutedExpertShape {
        layers: v4.n_layers,
        first_layer: 0,
        experts: v4.n_routed_experts,
        topk: v4.n_activated_experts,
        hidden,
        intermediate,
        draft_stages: 0,
        draft_experts: 0,
    };
    let raw_config = crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    if raw_config["quantization_config"]["quant_method"] == "exl3" {
        // dsv4p:exl3-k23.
        return deepseek_v4_exl3_catalog(snapshot, &index.weight_map, backbone, &[2, 3]);
    }
    let draft_stages = index.weight_map.keys()
        .filter_map(|name| name.strip_prefix("mtp.")?.split('.').next()?.parse::<usize>().ok())
        .max()
        .map_or(0, |stage| stage + 1);
    let draft_experts = index.weight_map.keys()
        .filter(|name| name.starts_with("mtp.0.ffn.experts.") && name.ends_with(".w1.weight"))
        .count();
    let expect = |name: &str| -> Option<(DType, [usize; 2])> {
        let parts: Vec<_> = name.split('.').collect();
        let routed = parts.len() == 7 && parts[0] == "layers" && parts[2] == "ffn" && parts[3] == "experts";
        if !routed {
            return None;
        }
        let (rows, cols) = if parts[5] == "w2" { (hidden, intermediate) } else { (intermediate, hidden) };
        match parts[6] {
            "weight" => Some((DType::I8, [rows, cols / 2])),
            "scale" => Some((DType::F8E8M0, [rows, cols / 32])),
            _ => None,
        }
    };
    let mut shards: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (name, shard) in &index.weight_map {
        shards.entry(shard).or_default().insert(name);
    }
    let mut tensors = Vec::with_capacity(index.weight_map.len());
    let mut routed = 0usize;
    for (shard, names) in shards {
        let metadata = read_safetensors_metadata(&snapshot.join(shard))
            .with_context(|| format!("reading {shard}"))?;
        ensure!(metadata.len() == names.len(), "index/header tensor count mismatch in {shard}");
        for tensor in metadata {
            ensure!(names.contains(tensor.name.as_str()), "tensor {} appears in unexpected shard {shard}", tensor.name);
            let placement = match expect(&tensor.name) {
                Some((dtype, shape)) => {
                    ensure!(
                        tensor.dtype == dtype && tensor.shape == shape,
                        "routed expert {} is {:?} {:?}, expected {dtype:?} {shape:?} (packed FP4 with E8M0 K32 scales)",
                        tensor.name, tensor.dtype, tensor.shape
                    );
                    routed += 1;
                    placement(&tensor.name)
                }
                None => V41TensorPlacement::CoordinatorRtx,
            };
            tensors.push(V41Tensor { shard: shard.to_string(), metadata: tensor, placement });
        }
    }
    ensure!(
        routed == v4.n_layers * v4.n_routed_experts * 6,
        "checkpoint has {routed} routed expert tensors, expected {} layers x {} experts x 6",
        v4.n_layers,
        v4.n_routed_experts
    );
    tensors.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    Ok(OfficialV41Catalog {
        config: None,
        experts: RoutedExpertShape {
            layers: v4.n_layers,
            first_layer: 0,
            experts: v4.n_routed_experts,
            topk: v4.n_activated_experts,
            hidden,
            intermediate,
            draft_stages,
            draft_experts,
        },
        snapshot: snapshot.to_path_buf(),
        tensors,
        exl3: None,
        nvfp4: None,
        fp8: None,
    })
}

/// The decoder layer of a Hugging Face tensor name (`model.layers.{L}.` or
/// `model.language_model.layers.{L}.`).
fn hf_layer(name: &str) -> Option<usize> {
    name.strip_prefix("model.layers.").or_else(|| name.strip_prefix("model.language_model.layers."))?
        .split('.').next()?.parse().ok()
}

/// Every tensor header of the shards `weight_map` names, each in its shard.
fn read_index_headers(
    snapshot: &Path,
    weight_map: &BTreeMap<String, String>,
) -> Result<Vec<(String, SafetensorsTensorMetadata)>> {
    let mut shards: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (name, shard) in weight_map {
        shards.entry(shard).or_default().insert(name);
    }
    let mut headers = Vec::with_capacity(weight_map.len());
    for (shard, names) in shards {
        let metadata = read_safetensors_metadata(&snapshot.join(shard))
            .with_context(|| format!("reading {shard}"))?;
        ensure!(metadata.len() == names.len(), "index/header tensor count mismatch in {shard}");
        for tensor in metadata {
            ensure!(names.contains(tensor.name.as_str()), "tensor {} appears in unexpected shard {shard}", tensor.name);
            headers.push((shard.to_string(), tensor));
        }
    }
    Ok(headers)
}

/// DeepSeek V4, GLM 5.x and Qwen 3.8 EXL3 routed experts: the storage layout
/// comes from the headers (a storage map, when present, must agree), every
/// projection's trellis/suh/svh/mcg is checked against its header; backbone
/// projections are sliced onto the Sparks, draft projections stay with the
/// coordinator. `tiers`: the family's package tier pair.
fn deepseek_v4_exl3_catalog(
    snapshot: &Path,
    weight_map: &BTreeMap<String, String>,
    backbone: RoutedExpertShape,
    tiers: &[usize],
) -> Result<OfficialV41Catalog> {
    let headers = read_index_headers(snapshot, weight_map)?;
    let manifest = crate::families::deepseek_v41::v41_exl3::read_deepseek_v4_exl3_manifest(
        snapshot,
        backbone,
        headers.iter().map(|(_, tensor)| tensor),
        tiers,
    )?;
    let mut tensors = Vec::with_capacity(headers.len());
    let mut routed = 0usize;
    for (shard, tensor) in headers {
        let projection = tensor
            .name
            .rsplit_once('.')
            .and_then(|(prefix, _)| manifest.projections.get(prefix));
        let placement = match projection {
            Some(projection) => {
                projection.validate_tensor(&tensor)?;
                routed += 1;
                let backbone = hf_layer(&tensor.name).is_some_and(|layer| layer < manifest.experts.layers);
                if backbone {
                    V41TensorPlacement::BackboneExl3
                } else {
                    V41TensorPlacement::CoordinatorRtx
                }
            }
            None => {
                // Experts past the backbone (a native MTP layer) stay on the coordinator.
                let past_backbone = hf_layer(&tensor.name).is_some_and(|layer| layer >= manifest.experts.layers);
                ensure!(
                    past_backbone
                        || (!tensor.name.contains(".mlp.experts.") && !tensor.name.contains(".ffn.experts.")),
                    "routed expert tensor {} is not in the EXL3 storage map",
                    tensor.name
                );
                V41TensorPlacement::CoordinatorRtx
            }
        };
        tensors.push(V41Tensor { shard, metadata: tensor, placement });
    }
    ensure!(
        routed == 4 * manifest.projections.len(),
        "checkpoint has {routed} EXL3 expert tensors, the storage map declares {} projections x 4",
        manifest.projections.len()
    );
    tensors.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    Ok(OfficialV41Catalog {
        config: None,
        experts: manifest.experts,
        snapshot: snapshot.to_path_buf(),
        tensors,
        exl3: Some(manifest),
        nvfp4: None,
        fp8: None,
    })
}

fn validate_tensor(tensor: &SafetensorsTensorMetadata, spec: &ExpectedTensor) -> Result<()> {
    ensure!(
        tensor.dtype == spec.dtype,
        "{} dtype mismatch: expected {:?}, got {:?}",
        tensor.name,
        spec.dtype,
        tensor.dtype
    );
    ensure!(
        tensor.shape == spec.shape,
        "{} shape mismatch: expected {:?}, got {:?}",
        tensor.name,
        spec.shape,
        tensor.shape
    );
    ensure!(
        tensor.byte_length == spec.bytes,
        "{} storage length mismatch: expected {}, got {}",
        tensor.name,
        spec.bytes,
        tensor.byte_length
    );
    Ok(())
}

fn apply_nvfp4_ple_contract(expected: &mut BTreeMap<String, ExpectedTensor>,
    ple: &serde_json::Value, layers: &[usize]) -> Result<()> {
    ensure!(ple["schema"] == "cuteafd.nvfp4-ple.v1" && ple["format"] == "nvfp4"
        && ple["block_size"] == 16 && ple["packing"] == "even-element-low-nibble"
        && ple["scale_layout"] == "row-major"
        && ple["reconstruction"] == "E2M1(weight) * FP8_E4M3(weight_scale) * FP32(weight_scale_2)",
        "unsupported NVFP4 PLE storage contract");
    let tables = ple["tensors"].as_object().context("missing NVFP4 PLE tables")?;
    ensure!(tables.len() == layers.len(), "NVFP4 PLE must cover exactly the original tables");
    for &layer in layers {
        let prefix = format!("layers.{layer}.engram.embed");
        let table = tables.get(&prefix).with_context(|| format!("missing {prefix} PLE metadata"))?;
        let original = expected.get(&format!("{prefix}.weight")).context("missing original PLE weight")?;
        let shape = original.shape.clone();
        ensure!(shape.len() == 2 && shape[1] % 16 == 0
            && table["logical_shape"] == serde_json::to_value(&shape)?
            && table["weight_dtype"] == "uint8" && table["weight_scale_dtype"] == "float8_e4m3fn"
            && table["weight_scale_2_dtype"] == "float32"
            && table["global_scale"].as_f64().is_some_and(|s| s.is_finite() && s > 0.0),
            "invalid NVFP4 PLE geometry or scale for {prefix}");
        let rows = shape[0]; let columns = shape[1];
        ensure!(expected.remove(&format!("{prefix}.scale")).is_some(), "missing original PLE scales");
        for (suffix, dtype, shape, bytes) in [
            ("weight", DType::U8, vec![rows, columns / 2], rows as u64 * columns as u64 / 2),
            ("weight_scale", DType::F8E4M3, vec![rows, columns / 16], rows as u64 * columns as u64 / 16),
            ("weight_scale_2", DType::F32, vec![], 4),
        ] {
            expected.insert(format!("{prefix}.{suffix}"), ExpectedTensor { dtype, shape, bytes });
        }
    }
    Ok(())
}

/// Replace the official native routed-expert tensors with the ModelOpt NVFP4
/// contract. Draft (`mtp.*`) experts keep the native FP4 layout.
fn apply_nvfp4_expert_contract(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    contract: &crate::V41Nvfp4Contract,
    text: &crate::V41TextConfig,
) -> Result<()> {
    let group = contract.group_size;
    ensure!(group == 16, "NVFP4 expert contract requires 16-wide groups");
    for layer in 0..text.num_hidden_layers {
        for expert in 0..text.n_routed_experts {
            for (stem, down) in [("w1", false), ("w3", false), ("w2", true)] {
                let prefix = format!("layers.{layer}.ffn.experts.{expert}.{stem}");
                let (rows, columns) = if down {
                    (text.hidden_size, text.moe_intermediate_size)
                } else {
                    (text.moe_intermediate_size, text.hidden_size)
                };
                ensure!(
                    columns % 16 == 0,
                    "NVFP4 expert column extent is not a multiple of 16"
                );
                expected.remove(&format!("{prefix}.weight"));
                expected.remove(&format!("{prefix}.scale"));
                for (suffix, dtype, shape, bytes) in [
                    (
                        "weight",
                        DType::U8,
                        vec![rows, columns / 2],
                        (rows as u64) * (columns as u64) / 2,
                    ),
                    (
                        "weight_scale",
                        DType::F8E4M3,
                        vec![rows, columns / group],
                        (rows as u64) * (columns as u64) / group as u64,
                    ),
                    ("weight_scale_2", DType::F32, vec![], 4),
                    ("input_scale", DType::F32, vec![], 4),
                ] {
                    expected.insert(
                        format!("{prefix}.{suffix}"),
                        ExpectedTensor {
                            dtype: dtype.clone(),
                            shape,
                            bytes,
                        },
                    );
                }
            }
        }
    }
    let backbone = expected
        .keys()
        .filter(|name| name.starts_with("layers.") && name.contains(".ffn.experts."))
        .count();
    ensure!(
        backbone == text.num_hidden_layers * text.n_routed_experts * 3 * 4,
        "NVFP4 routed-expert tensor count mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn official_config() -> OfficialV41Config {
        OfficialV41Config::from_json(crate::OFFICIAL_V41_MODEL_ID, include_bytes!("official-v41-config.json"))
            .unwrap()
    }
    #[test]
    fn checkpoint_contract_keeps_native_types_and_local_draft_experts() {
        let tensors = expected_tensors().unwrap();
        assert_eq!(tensors["head.weight"].dtype, DType::Bf16);
        assert_eq!(tensors["layers.0.attn.wo_a.weight"].dtype, DType::F8E4M3);
        assert_eq!(tensors["layers.0.attn.wo_a.scale"].shape, [256, 128]);
        assert_eq!(
            tensors["mtp.0.ffn.experts.127.w1.weight"].shape,
            [2304, 2560]
        );
        assert!(!tensors.contains_key("mtp.0.ffn.experts.128.w1.weight"));
        assert_eq!(
            placement("mtp.0.ffn.experts.127.w1.weight"),
            V41TensorPlacement::CoordinatorRtx
        );
        assert_eq!(
            placement("layers.39.ffn.experts.383.w2.scale"),
            V41TensorPlacement::BackboneExpertTp4 {
                layer: 39,
                expert: 383,
                axis: 1
            }
        );
        assert_eq!(
            placement("layers.14.engram.embed.weight"),
            V41TensorPlacement::HostMappedEngram
        );
    }
    #[test]
    fn rejects_same_byte_count_with_wrong_representation() {
        let spec = ExpectedTensor {
            dtype: DType::I8,
            shape: vec![2304, 2560],
            bytes: 5898240,
        };
        let mut tensor = SafetensorsTensorMetadata {
            name: "expert".into(),
            dtype: DType::I8,
            shape: vec![2304, 2560],
            byte_offset: 8,
            byte_length: 5898240,
        };
        validate_tensor(&tensor, &spec).unwrap();
        tensor.shape = vec![2560, 2304];
        assert!(validate_tensor(&tensor, &spec).is_err());
        tensor.shape = spec.shape.clone();
        tensor.dtype = DType::F8E4M3;
        assert!(validate_tensor(&tensor, &spec).is_err());
        tensor.dtype = DType::I8;
        tensor.byte_length -= 1;
        assert!(validate_tensor(&tensor, &spec).is_err());
    }
    #[test]
    fn tp4_staging_reads_columns_and_rows_beyond_two_gib() {
        use std::os::unix::fs::FileExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture");
        let offset = (1u64 << 31) + 64;
        let file = File::create(&path).unwrap();
        file.set_len(offset + 32).unwrap();
        file.write_all_at(&(0u8..32).collect::<Vec<_>>(), offset)
            .unwrap();
        let name = "layers.0.ffn.experts.0.w2.weight";
        let mut catalog = OfficialV41Catalog {
            config: Some(official_config()),
            experts: RoutedExpertShape::of_v41(&official_config()),
            exl3: None,
            nvfp4: None,
            fp8: None,
            snapshot: dir.path().into(),
            tensors: vec![V41Tensor {
                shard: "fixture".into(),
                metadata: SafetensorsTensorMetadata {
                    name: name.into(),
                    dtype: DType::I8,
                    shape: vec![4, 8],
                    byte_offset: offset,
                    byte_length: 32,
                },
                placement: V41TensorPlacement::BackboneExpertTp4 {
                    layer: 0,
                    expert: 0,
                    axis: 1,
                },
            }],
        };
        for rank in 0..4 {
            let mut out = [255u8; 8];
            catalog
                .read_device_tensor_into(name, Some(rank), &mut out, &mut [0; 16])
                .unwrap();
            let expected: Vec<_> = (0..4)
                .flat_map(|r| [r * 8 + rank * 2, r * 8 + rank * 2 + 1])
                .map(|v| v as u8)
                .collect();
            assert_eq!(out.as_slice(), expected);
        }
        assert!(catalog.device_tensor_bytes(name, None).is_err());
        assert!(catalog.device_tensor_bytes(name, Some(4)).is_err());
        catalog.tensors[0].placement = V41TensorPlacement::BackboneExpertTp4 {
            layer: 0,
            expert: 0,
            axis: 0,
        };
        let mut out = [0; 8];
        catalog
            .read_device_tensor_into(name, Some(3), &mut out, &mut [])
            .unwrap();
        assert_eq!(out, [24, 25, 26, 27, 28, 29, 30, 31]);
        for axis in 0..2 {
            catalog.tensors[0].placement = V41TensorPlacement::BackboneExpertTp4 { layer: 0, expert: 0, axis };
            let pair = [catalog.backbone_tp2_slice(name, 0).unwrap(), catalog.backbone_tp2_slice(name, 1).unwrap()];
            crate::V41Exl3TensorSlice::validate_pair(pair, 32).unwrap();
            let mut shared = [0; 32];
            let (bytes, storage) = OfficialV41Catalog::count_storage_reads(||
                catalog.read_projection_once(name, &mut shared)).unwrap();
            assert_eq!((bytes, storage), (32, 32));
            for (rank, slice) in pair.into_iter().enumerate() {
                let mut legacy = [0; 16];
                let (_, storage) = OfficialV41Catalog::count_storage_reads(||
                    catalog.read_backbone_tp2_into(name, rank, &mut legacy, &mut [0; 16])).unwrap();
                assert_eq!(storage, if axis == 0 { 16 } else { 32 });
                let pitched: Vec<_> = (0..slice.rows).flat_map(|row| {
                    let start = row * slice.source_row_bytes + slice.column_start_bytes;
                    shared[start..start + slice.selected_row_bytes].iter().copied()
                }).collect();
                assert_eq!(pitched, legacy);
            }
        }
        catalog.tensors[0].placement = V41TensorPlacement::CoordinatorRtx;
        for axis in 0..2 {
            for rank in 0..2 {
                let mut out = [255u8; 20];
                let bytes = catalog.read_coordinator_tp2_into(name, axis, rank, &mut out, &mut [0; 17]).unwrap();
                assert_eq!(bytes, 16);
                let expected: Vec<u8> = if axis == 0 {
                    (rank*16..(rank+1)*16).map(|v| v as u8).collect()
                } else {
                    (0..4).flat_map(|r| (r*8+rank*4..r*8+(rank+1)*4).map(|v| v as u8)).collect()
                };
                assert_eq!(&out[..16], expected);
                assert_eq!(&out[16..], &[255; 4]);
            }
        }
        assert!(catalog.read_coordinator_tp2_into(name, 1, 0, &mut [0; 16], &mut [0; 7]).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 0, 2, &mut [0; 16], &mut []).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 2, 0, &mut [0; 16], &mut []).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 0, 0, &mut [0; 15], &mut []).is_err());
        catalog.tensors[0].placement = V41TensorPlacement::HostMappedEngram;
        assert!(catalog.device_tensor_bytes(name, None).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 0, 0, &mut [0; 16], &mut []).is_err());
    }

    #[test]
    #[ignore = "requires CUTEAFD_NVFP4_SNAPSHOT pointing to a local ModelOpt NVFP4 publication"]
    fn nvfp4_catalog_keeps_native_draft_experts() {
        let path = std::env::var_os("CUTEAFD_NVFP4_SNAPSHOT").expect("CUTEAFD_NVFP4_SNAPSHOT");
        let catalog =
            read_official_v41_catalog("nvidia/DeepSeek-V4.1-Flash-NVFP4", Path::new(&path))
                .unwrap();
        let contract = catalog.nvfp4().expect("NVFP4 contract");
        assert_eq!(contract.group_size, 16);
        let mut tp4 = 0usize;
        let mut replicated = 0usize;
        let mut native_mtp = 0usize;
        let mut nvfp4_scalars = 0usize;
        for tensor in &catalog.tensors {
            let name = &tensor.metadata.name;
            if name.starts_with("layers.") && name.contains(".ffn.experts.") {
                match &tensor.placement {
                    V41TensorPlacement::BackboneExpertTp4 { .. } => tp4 += 1,
                    V41TensorPlacement::BackboneExpertReplicated { .. } => {
                        replicated += 1;
                        if tensor.metadata.dtype == DType::F32 {
                            nvfp4_scalars += 1;
                        }
                    }
                    other => panic!("unexpected NVFP4 placement {other:?} for {name}"),
                }
            } else if name.starts_with("mtp.") && name.contains(".ffn.experts.") {
                native_mtp += 1;
            }
        }
        // 384 experts x 3 projections x (weight, weight_scale) per layer.
        assert_eq!(tp4, 40 * 384 * 3 * 2);
        assert_eq!(replicated, 40 * 384 * 3 * 2);
        assert_eq!(nvfp4_scalars, 40 * 384 * 3 * 2);
        assert_eq!(native_mtp, 3 * 128 * 6);
        println!(
            "NVFP4 catalog: {} tensors, {tp4} TP4 payload/scale, {replicated} replicated, {native_mtp} native draft",
            catalog.tensors.len()
        );
    }

    #[test]
    #[ignore = "requires CUTEAFD_EXL3_RAW_SNAPSHOT pointing to a raw local publication"]
    fn raw_publication_catalog_keeps_native_draft_experts() {
        let path = std::env::var_os("CUTEAFD_EXL3_RAW_SNAPSHOT").expect("CUTEAFD_EXL3_RAW_SNAPSHOT");
        let catalog =
            read_official_v41_catalog("diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000", Path::new(&path))
                .unwrap();
        assert!(catalog.exl3().is_some());
        assert!(catalog.native_dspark_experts());
        let mut backbone_exl3 = 0usize;
        let mut native_mtp = 0usize;
        for tensor in &catalog.tensors {
            match tensor.placement {
                V41TensorPlacement::BackboneExl3 => backbone_exl3 += 1,
                V41TensorPlacement::CoordinatorRtx
                    if tensor.metadata.name.starts_with("mtp.") && tensor.metadata.name.contains(".ffn.experts.") =>
                {
                    native_mtp += 1
                }
                _ => {}
            }
        }
        assert_eq!(backbone_exl3, 4 * 46_080);
        assert_eq!(native_mtp, 6 * 3 * 128);
        println!(
            "raw catalog: {} tensors, {backbone_exl3} EXL3 backbone, {native_mtp} native draft",
            catalog.tensors.len()
        );
    }
}

#[cfg(test)]
mod expert_staging_tests {
    use super::tests::official_config;
    use super::*;
    use crate::V41ExpertSelection;
    use std::os::unix::fs::FileExt;

    #[test]
    fn native_expert_staging_covers_all_tp_ranks_and_full_experts() {
        let dir = tempfile::tempdir().unwrap();
        let file = File::create(dir.path().join("fixture")).unwrap();
        let mut offset = (1u64 << 31) + 128;
        let specs = expected_tensors().unwrap();
        let mut tensors = Vec::new();
        let suffixes = [
            "w1.weight",
            "w3.weight",
            "w2.weight",
            "w1.scale",
            "w3.scale",
            "w2.scale",
        ];
        let mut payloads = Vec::new();
        for (slot, suffix) in suffixes.iter().enumerate() {
            let name = format!("layers.39.ffn.experts.383.{suffix}");
            let spec = &specs[&name];
            let cols = spec.shape[1];
            let payload: Vec<u8> = (0..spec.bytes as usize)
                .map(|index| ((index / cols * 7 + index % cols * 13 + slot * 19) & 255) as u8)
                .collect();
            file.write_all_at(&payload, offset).unwrap();
            for prefix in ["layers.39.ffn.experts.383", "mtp.2.ffn.experts.127"] {
                let name = format!("{prefix}.{suffix}");
                tensors.push(V41Tensor {
                    shard: "fixture".into(),
                    placement: placement(&name),
                    metadata: SafetensorsTensorMetadata {
                        name,
                        dtype: spec.dtype.clone(),
                        shape: spec.shape.clone(),
                        byte_offset: offset,
                        byte_length: spec.bytes,
                    },
                });
            }
            payloads.push(payload);
            offset += spec.bytes;
        }
        tensors.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
        let catalog = OfficialV41Catalog {
            config: Some(official_config()),
            experts: RoutedExpertShape::of_v41(&official_config()),
            exl3: None,
            nvfp4: None,
            fp8: None,
            snapshot: dir.path().into(),
            tensors,
        };
        let select = |rank| V41ExpertSelection::Backbone {
            layer: 39,
            expert: 383,
            rank,
        };
        for rank in 0..4 {
            let plan = catalog.expert_staging(select(rank)).unwrap();
            assert_eq!(plan.staging_bytes(), 4_700_160);
            assert_eq!(plan.intermediate_size(), 576);
            assert_eq!(plan.minimum_read_scratch_bytes(), 1152);
            // An odd number of physical rows exercises the final partial read batch.
            let mut scratch = vec![0; 1152 * 7 + 3];
            let mut staging = vec![205; plan.staging_bytes() + 32];
            assert!(plan
                .read_into(&mut staging[..plan.staging_bytes() - 1], &mut scratch)
                .is_err());
            assert!(plan.read_into(&mut staging, &mut scratch[..1151]).is_err());
            assert!(staging.iter().all(|&byte| byte == 205));
            plan.prefetch().unwrap();
            plan.read_into(&mut staging, &mut scratch).unwrap();
            for (slot, range) in plan.tensor_ranges().iter().enumerate() {
                assert_eq!(range.start % 16, 0);
                let source = &payloads[slot];
                let expected = if slot == 2 || slot == 5 {
                    let row = source.len() / 5120;
                    source
                        .chunks_exact(row)
                        .flat_map(|r| r[rank * row / 4..(rank + 1) * row / 4].iter().copied())
                        .collect::<Vec<_>>()
                } else {
                    source[rank * source.len() / 4..(rank + 1) * source.len() / 4].to_vec()
                };
                assert_eq!(&staging[range.clone()], expected.as_slice());
            }
            assert!(staging[plan.staging_bytes()..]
                .iter()
                .all(|&byte| byte == 205));
        }
        // Two RTX halves reconstruct the official row and column partitions.
        for rank in 0..2 {
            let plan = catalog.expert_staging(V41ExpertSelection::BackboneTp2 {
                layer: 39, expert: 383, rank,
            }).unwrap();
            assert_eq!(plan.intermediate_size(), 1152);
            assert_eq!(plan.staging_bytes(), 9_400_320);
            assert_eq!(plan.minimum_read_scratch_bytes(), 1152);
            let mut staging = vec![205; plan.staging_bytes() + 32];
            let mut scratch = vec![0; 1152 * 7 + 3];
            assert!(plan.read_into(&mut staging[..plan.staging_bytes()-1], &mut scratch).is_err());
            assert!(plan.read_into(&mut staging, &mut scratch[..1151]).is_err());
            assert!(staging.iter().all(|&v| v == 205));
            plan.prefetch().unwrap();
            plan.read_into(&mut staging, &mut scratch).unwrap();
            for (slot, range) in plan.tensor_ranges().iter().enumerate() {
                let source = &payloads[slot];
                let expected = if slot == 2 || slot == 5 {
                    let row = source.len() / 5120;
                    source.chunks_exact(row).flat_map(|r|
                        r[rank*row/2..(rank+1)*row/2].iter().copied()).collect::<Vec<_>>()
                } else {
                    source[rank*source.len()/2..(rank+1)*source.len()/2].to_vec()
                };
                assert_eq!(&staging[range.clone()], expected.as_slice());
            }
            assert!(staging[plan.staging_bytes()..].iter().all(|&v| v == 205));
        }
        // The generic explicit-TP shard path (used by the replicated TP2/TP3
        // groups and the pure unreplicated TP6 layout) slices W1/W3 output rows
        // and W2 input columns; every slice must stay byte- and scale-aligned.
        for (world, intermediate, staging_bytes, column_bytes, scale_column_bytes) in
            [(3usize, 768usize, 6_266_880usize, 384usize, 24usize),
             (6, 384, 3_133_440, 192, 12)]
        {
            for rank in 0..world {
                let plan = catalog
                    .expert_staging(V41ExpertSelection::BackboneTp { layer: 39, expert: 383, rank, world })
                    .unwrap();
                assert_eq!(plan.intermediate_size(), intermediate, "TP{world} rank {rank}");
                assert_eq!(plan.staging_bytes(), staging_bytes, "TP{world} rank {rank}");
                assert_eq!(plan.minimum_read_scratch_bytes(), 1152);
                let mut staging = vec![205; plan.staging_bytes() + 32];
                let mut scratch = vec![0; 1152 * 7 + 3];
                assert!(plan
                    .read_into(&mut staging[..plan.staging_bytes() - 1], &mut scratch)
                    .is_err());
                assert!(plan.read_into(&mut staging, &mut scratch[..1151]).is_err());
                assert!(staging.iter().all(|&byte| byte == 205));
                plan.prefetch().unwrap();
                plan.read_into(&mut staging, &mut scratch).unwrap();
                for (slot, range) in plan.tensor_ranges().iter().enumerate() {
                    assert_eq!(range.start % 16, 0);
                    let source = &payloads[slot];
                    let expected = if slot == 2 {
                        let row = source.len() / 5120;
                        source
                            .chunks_exact(row)
                            .flat_map(|r| {
                                r[rank * column_bytes..(rank + 1) * column_bytes].iter().copied()
                            })
                            .collect::<Vec<_>>()
                    } else if slot == 5 {
                        // Every rank owns whole 32-value scale groups, so the W2
                        // scale column slice is exactly shard_intermediate/32.
                        let row = source.len() / 5120;
                        source
                            .chunks_exact(row)
                            .flat_map(|r| {
                                r[rank * scale_column_bytes..(rank + 1) * scale_column_bytes]
                                    .iter()
                                    .copied()
                            })
                            .collect::<Vec<_>>()
                    } else {
                        source[rank * source.len() / world..(rank + 1) * source.len() / world].to_vec()
                    };
                    assert_eq!(&staging[range.clone()], expected.as_slice(), "TP{world} rank {rank} slot {slot}");
                }
                assert!(staging[plan.staging_bytes()..].iter().all(|&byte| byte == 205));
            }
        }
        // Full backbone reads must preserve every official byte, including W2
        // columns that the TP4 path normally slices into separate ranks.
        let full = catalog.expert_staging(V41ExpertSelection::BackboneFull {
            layer: 39, expert: 383,
        }).unwrap();
        assert_eq!(full.staging_bytes(), 18_800_640);
        assert_eq!(full.intermediate_size(), 2304);
        assert_eq!(full.minimum_read_scratch_bytes(), 0);
        let mut full_staging = vec![205; full.staging_bytes() + 32];
        assert!(full.read_into(&mut full_staging[..full.staging_bytes() - 1], &mut []).is_err());
        assert!(full_staging.iter().all(|&byte| byte == 205));
        full.prefetch().unwrap();
        full.read_into(&mut full_staging, &mut []).unwrap();
        for (range, expected) in full.tensor_ranges().iter().zip(&payloads) {
            assert_eq!(&full_staging[range.clone()], expected.as_slice());
        }
        assert!(full_staging[full.staging_bytes()..].iter().all(|&byte| byte == 205));
        let plan = catalog
            .expert_staging(V41ExpertSelection::Dspark {
                stage: 2,
                expert: 127,
            })
            .unwrap();
        assert_eq!(plan.staging_bytes(), 18_800_640);
        assert_eq!(plan.minimum_read_scratch_bytes(), 0);
        assert_eq!(plan.intermediate_size(), 2304);
        let mut staging = vec![0; plan.staging_bytes()];
        plan.prefetch().unwrap();
        plan.read_into(&mut staging, &mut []).unwrap();
        for (range, expected) in plan.tensor_ranges().iter().zip(&payloads) {
            assert_eq!(&staging[range.clone()], expected.as_slice());
        }
        // Draft halves use the same packed row/column split, while preserving
        // all 128 expert IDs independently on each RTX rank.
        for rank in 0..2 {
            let plan=catalog.expert_staging(V41ExpertSelection::DsparkTp2 {stage:2,expert:127,rank}).unwrap();
            assert_eq!(plan.intermediate_size(),1152);
            assert_eq!(plan.staging_bytes(),9_400_320);
            assert_eq!(plan.minimum_read_scratch_bytes(),1152);
            let mut staging=vec![205;plan.staging_bytes()+32];
            let mut scratch=vec![0;1152*7+3];
            assert!(plan.read_into(&mut staging[..plan.staging_bytes()-1],&mut scratch).is_err());
            assert!(plan.read_into(&mut staging,&mut scratch[..1151]).is_err());
            assert!(staging.iter().all(|&v|v==205));
            plan.prefetch().unwrap();
            plan.read_into(&mut staging,&mut scratch).unwrap();
            for (slot,range) in plan.tensor_ranges().iter().enumerate() {
                let source=&payloads[slot];
                let expected=if slot==2 || slot==5 {
                    let row=source.len()/5120;
                    source.chunks_exact(row).flat_map(|r|
                        r[rank*row/2..(rank+1)*row/2].iter().copied()).collect::<Vec<_>>()
                } else {source[rank*source.len()/2..(rank+1)*source.len()/2].to_vec()};
                assert_eq!(&staging[range.clone()],expected.as_slice());
            }
            assert!(staging[plan.staging_bytes()..].iter().all(|&v|v==205));
        }
        for selection in [
            select(4),
            V41ExpertSelection::DsparkTp2 {stage:3,expert:0,rank:0},
            V41ExpertSelection::DsparkTp2 {stage:0,expert:128,rank:0},
            V41ExpertSelection::DsparkTp2 {stage:0,expert:0,rank:2},
            V41ExpertSelection::BackboneTp2 { layer: 0, expert: 0, rank: 2 },
            V41ExpertSelection::BackboneTp2 { layer: 40, expert: 0, rank: 0 },
            V41ExpertSelection::BackboneTp2 { layer: 0, expert: 384, rank: 0 },
            V41ExpertSelection::BackboneFull { layer: 40, expert: 0 },
            V41ExpertSelection::BackboneFull { layer: 0, expert: 384 },
            V41ExpertSelection::Backbone {
                layer: 40,
                expert: 0,
                rank: 0,
            },
            V41ExpertSelection::Backbone {
                layer: 0,
                expert: 384,
                rank: 0,
            },
            V41ExpertSelection::Dspark {
                stage: 3,
                expert: 0,
            },
            V41ExpertSelection::Dspark {
                stage: 0,
                expert: 128,
            },
        ] {
            assert!(catalog.expert_staging(selection).is_err());
        }
        // Catalog metadata alone is insufficient after a file changes: propagate I/O failure.
        file.set_len(0).unwrap();
        assert!(full.read_into(&mut full_staging, &mut []).is_err());
        assert!(plan.read_into(&mut staging, &mut []).is_err());
    }
}
