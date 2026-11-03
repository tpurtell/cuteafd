//! Bundled PCM audio preparation and placement on the shared media owner.
use crate::shared::vision::{
    audio::AudioOwnerConfig,
    pcm_audio::PcmTowerSpec,
    remote::{AudioHandshake, RemoteEncoder},
};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{
    media::audio::MAX_CLIP_SAMPLES,
    plan::encoder::{EncoderKind, EncoderPlacement},
};
use std::{net::SocketAddr, path::Path, sync::Arc, time::Duration};

/// Local configuration joins the image owner's single queue. Remote clients use
/// the existing tagged audio protocol; no family-specific request binding.
pub(super) struct ReadyAudio {
    pub preparer: Arc<cuteafd_api::openai::media::audio::AudioPreparer>,
    pub local: Option<AudioOwnerConfig>,
    pub remote: Option<RemoteEncoder>,
    pub admitted_bytes: u64,
}
impl ReadyAudio {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        snapshot: &Path,
        placement: &EncoderPlacement,
        library: &Path,
        device: i32,
        revision: &str,
        plan_hash: [u8; 32],
        peers: Vec<SocketAddr>,
        slots: Arc<tokio::sync::Semaphore>,
    ) -> Result<Option<Self>> {
        if placement.kind == EncoderKind::Off {
            return Ok(None);
        }
        let spec = PcmTowerSpec::from_snapshot(snapshot, MAX_CLIP_SAMPLES)?;
        let bytes = spec.required_bytes()?;
        let geometry = cuteafd_core::media::AudioTokenGeometry::PcmPatches {
            samples_per_token: 1920,
        };
        let (local, remote, id) = match placement.kind {
            EncoderKind::Rtx { .. } => {
                ensure!(
                    peers.is_empty(),
                    "RTX Muse audio cannot use remote encoder peers"
                );
                ensure!(
                    placement.admitted_bytes() == bytes,
                    "Muse PCM plan differs from native reservation"
                );
                // SAFETY: architecture-matching native library selected by the launcher.
                let native = unsafe { cuteafd_ffi::NativeLibrary::load(library) }?;
                let info = native.cuda_device_info(device)?;
                let sm = u32::try_from(
                    info.compute_capability_major * 10 + info.compute_capability_minor,
                )?;
                ensure!([120, 121].contains(&sm), "Muse PCM requires SM120/SM121");
                let id = spec.encoder_id(revision, sm);
                (
                    Some(AudioOwnerConfig::Pcm {
                        spec,
                        admitted_bytes: bytes,
                    }),
                    None,
                    id,
                )
            }
            EncoderKind::Spark { .. } | EncoderKind::SparkIdle { .. } => {
                ensure!(
                    !peers.is_empty(),
                    "Spark Muse audio requires planner-resolved encoder peers"
                );
                ensure!(
                    placement.admitted_bytes() >= bytes,
                    "Muse PCM reservation below native ledger"
                );
                let id = spec.encoder_id(revision, 121);
                let handshake = AudioHandshake {
                    encoder_id: id,
                    plan_hash,
                    max_samples: MAX_CLIP_SAMPLES as u32,
                    output_width: 8192,
                    geometry,
                };
                (
                    None,
                    Some(
                        RemoteEncoder::connect_audio(peers, handshake, Duration::from_secs(60))
                            .context("Muse PCM encoder handshake")?,
                    ),
                    id,
                )
            }
            EncoderKind::Off => unreachable!(),
        };
        Ok(Some(Self {
            preparer: Arc::new(
                cuteafd_api::openai::media::audio::AudioPreparer::with_geometry(
                    id, slots, geometry,
                ),
            ),
            local,
            remote,
            admitted_bytes: bytes,
        }))
    }
}
