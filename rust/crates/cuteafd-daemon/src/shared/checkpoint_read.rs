//! Coordinator tensor reads through the standard checkpoint inventory: one
//! header-located positional read per tensor, checked against the index.
//! Routed expert payloads stay on the Spark loading path.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::LoadedTensor;
use std::os::unix::fs::FileExt;

pub(crate) fn read_tensor(checkpoint: &cuteafd_loader::plan::Checkpoint, name: &str) -> Result<LoadedTensor> {
    let at = checkpoint.tensors.binary_search_by(|tensor| tensor.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("coordinator checkpoint lacks {name}"))?;
    let tensor = &checkpoint.tensors[at];
    ensure!(checkpoint.weight_map.get(name) == Some(&tensor.shard),
        "tensor header differs from checkpoint index: {name}");
    let metadata = &tensor.meta;
    let bytes = usize::try_from(metadata.byte_length).context("tensor extent exceeds host address space")?;
    let source_path = checkpoint.snapshot.join(&tensor.shard);
    cuteafd_ffi::memory_ledger::tensor(name);
    let mut bytes = vec![0; bytes];
    let started = std::time::Instant::now();
    std::fs::File::open(&source_path)?.read_exact_at(&mut bytes, metadata.byte_offset)
        .with_context(|| format!("reading coordinator tensor {name}"))?;
    Ok(LoadedTensor { info: cuteafd_core::TensorInfo { name: name.into(), file: tensor.shard.clone(),
        dtype: metadata.dtype.clone(), shape: metadata.shape.clone(), byte_offset: metadata.byte_offset,
        byte_length: metadata.byte_length, role: cuteafd_core::TensorRole::Other, layer_id: None,
        expert_id: None, is_quantization_metadata: false }, source_path, bytes,
        elapsed_micros: started.elapsed().as_micros(), sha256: String::new() })
}
