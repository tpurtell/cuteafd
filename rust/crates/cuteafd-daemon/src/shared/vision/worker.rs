//! expertd tower startup: reserve the resident ledger before expert allocation.
use super::{normalization_lut, remote::{EncoderHandshake, EncoderServer}, EncoderService, TowerSpec};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::vision::NativeVision;
use cuteafd_loader::media::ProcessorConfig;
use std::{path::{Path, PathBuf}, time::Duration};

#[derive(Clone, Debug)]
pub struct EncoderWorkerConfig {
    pub listen: String,
    pub plan_hash: [u8; 32],
    pub revision: String,
    pub max_tokens: usize,
}
pub fn parse_plan_hash(value: &str) -> Result<[u8; 32]> {
    ensure!(value.len() == 64 && value.is_ascii(), "encoder plan hash must be 64 hexadecimal digits");
    let mut hash = [0; 32];
    for (i, byte) in hash.iter_mut().enumerate() { *byte = u8::from_str_radix(&value[i*2..i*2+2], 16).context("encoder plan hash")?; }
    Ok(hash)
}
fn architecture(library: &Path) -> Result<u32> {
    // SAFETY: this is the user-selected, architecture-matching native library.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(library) }?;
    let info = library.cuda_device_info(0)?;
    Ok((info.compute_capability_major * 10 + info.compute_capability_minor) as u32)
}
/// Returns a server only after the tower owner has reported ready. The caller
/// subtracts this exact ledger before any expert weight/workspace admission.
pub fn start(config: &EncoderWorkerConfig, snapshot: &Path, library: PathBuf, budget: u64) -> Result<(EncoderServer, u64)> {
    let spec = TowerSpec::from_snapshot(snapshot, config.max_tokens)?;
    let sm = architecture(&library)?;
    let id = spec.encoder_id(&config.revision, sm);
    let ledger = NativeVision::required(&library, &spec.native)?;
    ensure!(ledger.total_bytes() <= budget, "vision tower admission needs {} bytes, budget {budget}", ledger.total_bytes());
    let processor = ProcessorConfig::from_snapshot(snapshot, spec.image_family())?;
    let handshake = EncoderHandshake { encoder_id: id, max_patches: (config.max_tokens * 4) as u32,
        output_width: spec.native.output_width, patch_size: processor.patch, merge_size: processor.merge, plan_hash: config.plan_hash };
    let service = EncoderService::start(spec, library, 0, ledger.total_bytes())?;
    let server = EncoderServer::start(config.listen.parse().context("encoder listen address")?, handshake, service, normalization_lut(&processor), Duration::from_secs(60))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    tracing::info!(address = %server.address, sm, admitted_bytes = ledger.total_bytes(), "vision encoder ready");
    Ok((server, ledger.total_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hash_parser_fails_closed() {
        assert_eq!(parse_plan_hash(&"ab".repeat(32)).unwrap(), [0xab; 32]);
        for bad in ["f".repeat(63), "z".repeat(64), "é".repeat(32)] { assert!(parse_plan_hash(&bad).is_err()); }
    }
}
