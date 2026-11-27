//! Cold fidelity feature replacement for still towers that are not production-qualified.
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{ModelProfile, NativeRequest};
use cuteafd_core::MediaSpan;
use cuteafd_engine::media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia};
use cuteafd_loader::media::{ImageFamily, ProcessorConfig};
use std::{path::Path, sync::Arc};

pub(crate) fn install(profile: ModelProfile, snapshot: &Path, family: ImageFamily) -> Result<ModelProfile> {
    let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
    if config.get("vision_config").is_none_or(serde_json::Value::is_null) { return Ok(profile); }
    let processor = ProcessorConfig::from_snapshot(snapshot, family)?;
    let spec = crate::shared::vision::TowerSpec::from_snapshot(snapshot, processor.max_image_tokens)?;
    let revision = snapshot.file_name().and_then(|s| s.to_str()).context("probe snapshot revision")?;
    // No GPU tower exists on this route. The zero architecture identifies the
    // processor-only source; feature overrides have their own cache identity.
    let preparer = cuteafd_api::openai::media::MediaPreparer::for_loaded_encoder(
        processor, spec.encoder_id(revision, 0), 2)?;
    Ok(profile.with_probe_media(Arc::new(preparer)))
}

pub(crate) fn admission<T>() -> cuteafd_engine::media::MediaAdmission<T, crate::shared::qwen::Encoder> {
    cuteafd_engine::media::MediaAdmission::new(EmbeddingCache::new(8 << 30), crate::shared::qwen::Encoder::Off, 16)
}

fn marker(snapshot: &Path, name: &str, vocabulary: u32) -> Result<u32> {
    let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("tokenizer_config.json"))?;
    let mut ids = config["added_tokens_decoder"].as_object().context("still probe tokenizer added tokens")?
        .iter().filter(|(_, token)| token["content"].as_str() == Some(name));
    let id: u32 = ids.next().with_context(|| format!("still probe tokenizer lacks {name}"))?.0.parse()?;
    ensure!(ids.next().is_none() && id < vocabulary, "still probe marker duplicate/outside vocabulary");
    Ok(id)
}

pub(crate) fn spans(job: &NativeRequest, tokens: &[u32], vocabulary: u32, context: usize,
    snapshot: &Path, family: ImageFamily) -> Result<Vec<MediaSpan>> {
    ensure!(job.audio.is_empty(), "still probe does not accept audio");
    if job.media.is_empty() { return Ok(Vec::new()); }
    let probe = job.probe.as_ref().filter(|p| p.spec.feature_scoring())
        .context("bundled vision is not production-qualified; images require a cold fidelity feature override")?;
    ensure!(job.audio.is_empty() && tokens.len() <= context && tokens.iter().all(|&id| id < vocabulary),
        "still probe token/context/audio extent");
    ensure!(probe.spec.prompt_ids.as_deref() == Some(tokens) && probe.spec.media.len() == job.media.len(),
        "still probe expanded ids/image count differs");
    probe.spec.validate_media_geometry(family.token_geometry())?;
    let placeholder = marker(snapshot, if family == ImageFamily::Kimi { "<|media_pad|>" } else { "]<]image[>[" }, vocabulary)?;
    let tokenizer = cuteafd_loader::cached_tokenizer(snapshot)?;
    let mut covered = std::collections::BTreeSet::new();
    let mut spans = Vec::new();
    for (span, image) in probe.spec.media.iter().zip(&job.media) {
        span.verify_image(image)?;
        let end = span.start.checked_add(span.len).context("still probe span overflow")?;
        ensure!(end <= tokens.len() && tokens[span.start..end].iter().all(|&id| id == placeholder),
            "still probe feature placeholders differ");
        if family == ImageFamily::Kimi {
            let [width, height] = image.pixels.source;
            let prefix = tokenizer.encode_ids(&format!("<|media_begin|>image {width}x{height}<|media_content|>"))?;
            let suffix = marker(snapshot, "<|media_end|>", vocabulary)?;
            ensure!(span.start >= prefix.len() && tokens[span.start-prefix.len()..span.start] == prefix
                && tokens.get(end) == Some(&suffix), "Kimi probe source-dimension markers differ");
        }
        else {
            ensure!(span.start > 0 && tokens.get(span.start - 1) == Some(&marker(snapshot, "]<]start of image[>[", vocabulary)?)
                && tokens.get(end) == Some(&marker(snapshot, "]<]end of image[>[", vocabulary)?),
                "MiniMax probe image boundaries differ");
        }
        covered.extend(span.start..end);
        spans.push(MediaSpan { start: span.start, len: span.len, key: image.key.into() });
    }
    ensure!(tokens.iter().enumerate().all(|(i, &id)| id != placeholder || covered.contains(&i)),
        "unbound still probe placeholders");
    Ok(spans)
}

pub(crate) fn prepare(job: NativeRequest, tokens: Vec<u32>, vocabulary: u32, hidden: usize,
    context: usize, snapshot: &Path, family: ImageFamily)
    -> Result<(crate::shared::qwen::scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
    let spans = spans(&job, &tokens, vocabulary, context, snapshot, family)?;
    let keys = MediaKeys::new(&tokens, vocabulary, &spans)?;
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let jobs = job.media.iter().map(|image| EncodeJob::image_with_pixels(image.key,
        [image.grid.t, image.grid.h, image.grid.w], image.rgb8.clone(), image.pixels, image.tokens, hidden)).collect();
    crate::shared::probe::image_echo(&job, &spans)?;
    Ok((crate::shared::qwen::scheduler::Prompt { job, tokens, keys }, media, jobs))
}

/// The pinned serving reference whose tower produced fidelity features.
fn source_revision(family: ImageFamily) -> &'static str {
    if matches!(family, ImageFamily::Hy4Flash | ImageFamily::Hy4) { "sglang@ccb8d9cc" } else { "sglang@a9bf8c76" }
}

pub(crate) fn snapshot_identity(snapshot: &Path, family: ImageFamily) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let mut identity = serde_json::json!({"snapshot_revision": snapshot.file_name().and_then(|s| s.to_str())
        .context("probe snapshot revision")?, "family": format!("{family:?}"), "source": source_revision(family)});
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json", "tiktoken.model", "model.safetensors.index.json"] {
        let path = snapshot.join(name);
        if path.is_file() { identity[format!("{name}_sha256")] = format!("{:x}", Sha256::digest(std::fs::read(path)?)).into(); }
    }
    Ok(identity)
}

pub(crate) fn features(prompt: &crate::shared::qwen::scheduler::Prompt, media: &mut RequestMedia,
    cache: &mut EmbeddingCache, snapshot: &Path, family: ImageFamily) -> Result<()> {
    if media.spans().is_empty() { return Ok(()); }
    ensure!(prompt.job.probe.as_ref().is_some_and(|p| p.spec.feature_scoring()), "still media is fidelity-only");
    let root = std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR").context("fidelity image feature payload directory absent")?;
    super::features::apply_probe_features(&prompt.job, media, cache, &std::path::PathBuf::from(root),
        &snapshot_identity(snapshot, family)?, &source_revision(family)["sglang@".len()..])
}

/// Diagnostic path only: reuse admitted embedding storage, never allocate GPU
/// scratch or introduce a captured pointer. Normal text calls return immediately.
pub(crate) fn inject(stream: &crate::shared::memory::device::Stream<'_>,
    output: cuteafd_ffi::CuteafdDeviceBuffer, media: &RequestMedia, start: usize, rows: usize,
    width: usize, fp32: bool) -> Result<()> {
    let end = start.checked_add(rows).context("still probe injection extent")?;
    if media.needed(start, end).next().is_none() { return Ok(()); }
    let mut chunk = cuteafd_engine::media::MediaChunk::default();
    media.write_chunk(start, end, &mut chunk)?;
    ensure!(media.row_bytes() == width * 2 && output.bytes >= rows * width * if fp32 { 4 } else { 2 },
        "still probe injection output extent");
    stream.drain()?;
    for (&row, bytes) in chunk.indices.iter().zip(chunk.features.chunks_exact(width * 2)) {
        let converted;
        let data = if fp32 {
            converted = bytes.chunks_exact(2).flat_map(|b|
                (u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes()).collect::<Vec<_>>();
            &converted[..]
        } else { bytes };
        let offset = row as usize * data.len();
        let target = crate::shared::prefix::view(output, offset, data.len())?;
        stream.device.library.copy_h2d(target, data)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::openai::probe::{Probe, ProbeFixture, ProbeImageUrl, ProbeMedia, ProbeSpec};
    use cuteafd_loader::media::{EncoderId, ImageGrid, ImageKey, PreparedImage, PreprocessId};

    #[test]
    fn kimi_full_span_dimensions_reach_media_ready_and_exact_kv_lease() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_kimi_tokenizer(root.path());
        let mut config: serde_json::Value = cuteafd_loader::read_json_file(&root.path().join("tokenizer_config.json"))?;
        for (id, token) in [(258, "<|media_begin|>"), (259, "<|media_content|>"),
            (260, "<|media_pad|>"), (261, "<|media_end|>")] {
            config["added_tokens_decoder"][id.to_string()] = serde_json::json!({"content":token});
        }
        std::fs::write(root.path().join("tokenizer_config.json"), serde_json::to_vec(&config)?)?;
        let tokenizer = cuteafd_loader::cached_tokenizer(root.path())?;
        let prefix = tokenizer.encode_ids("<|media_begin|>image 7069x1833<|media_content|>")?;
        let mut tokens = vec![1]; tokens.extend(prefix);
        let start = tokens.len();
        tokens.extend(std::iter::repeat_n(260, 16698)); tokens.extend([261, 2, 3]);
        let image = Arc::new(PreparedImage { key: ImageKey([7;32]), encoder: EncoderId([1;32]),
            processor: PreprocessId([2;32]), grid: ImageGrid { t:1, h:132, w:506 }, tokens:16698,
            rgb8: vec![0; 7084 * 1848 * 3].into(),
            pixels: cuteafd_core::media::ImagePixels { source:[7069,1833], content:[0,0,7069,1833],
                canvas:[7084,1848] } });
        let probe = Probe::new(ProbeSpec { cold:true, no_speculation:true, score_from:Some(tokens.len()-1),
            prompt_ids:Some(tokens.clone()), media:vec![ProbeMedia { start,len:16698,kind:"image".into(),
                key:"07".repeat(32),grid:[1,132,506],identity:None,input_key:None,
                fixture:Some(ProbeFixture {path:"image.png".into(),sha256:"ab".repeat(32)}),
                image_url:Some(ProbeImageUrl {url:"fixture".into(),detail:None}) }], ..Default::default() });
        let mut job = NativeRequest { image_groups: Vec::new(), prefix_cache_source:Default::default(),finish_detail:Default::default(),
            literal_spans:vec![],prompt:String::new(),constraint:None,images:vec![],media:vec![image],audio:vec![],
            max_tokens:64,sampling:Default::default(),stop_token_ids:vec![],
            events:tokio::sync::mpsc::unbounded_channel().0,usage:None,probe:None };
        assert!(spans(&job, &tokens,512,32768,root.path(),ImageFamily::Kimi).is_err());
        job.probe=Some(probe);
        let (prompt,mut media,jobs)=prepare(job,tokens.clone(),512,1,32768,root.path(),ImageFamily::Kimi)?;
        assert_eq!(media.spans()[0].len,16698);
        let mut queue=admission();
        let key=ImageKey([8;32]); let pin=queue.cache.reserve(key,16698*2)?;
        let lease=queue.cache.complete(key,vec![0;16698*2].into())?;
        media.attach_probe_override(ImageKey([7;32]),lease)?; drop(pin);
        queue.enqueue(cuteafd_engine::media::MediaWaiter::new(prompt,media,jobs,0)?).map_err(|(_,e)|e)?;
        let ready=match queue.poll(|_|false) {cuteafd_engine::media::MediaPoll::Ready(ready)=>ready,
            _=>anyhow::bail!("not media-ready")};
        let reserve=cuteafd_engine::reserve::KvReserve::new(
            cuteafd_engine::reserve::ReserveConfig::new(16384,0,-1,131072)?,64);
        let ready_tokens=ready.job().tokens.len();
        let lease=reserve.lease(ready_tokens,64,8,32768)?;
        assert_eq!(lease.lifetime(),ready_tokens+64+8);
        assert_eq!(reserve.admission_rows(&lease),ready_tokens+64+8);
        Ok(())
    }
}

#[cfg(test)]
mod injection_tests {
    use super::*;
    use crate::shared::memory::device::{Allocation, Device, Stream};
    use cuteafd_core::memory_layout::Tag;
    use cuteafd_loader::media::ImageKey;

    #[test]
    fn feature_injection_preserves_text_rows_and_converts_bf16_to_f32_exactly() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let stream = Stream::new(device)?;
        let source = [0x3f80u16,0xc000,0x3f00,0x4000];
        let payload = source.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let mut cache = EmbeddingCache::new(payload.len());
        let key = ImageKey([1;32]); let pin = cache.reserve(key,payload.len())?;
        let lease = cache.complete(key,payload.into())?;
        let mut media = RequestMedia::new(vec![MediaSpan {start:2,len:2,key:key.into()}],2,6)?;
        media.attach(lease)?;drop(pin);
        for fp32 in [false,true] {
            let row_bytes = if fp32 {8} else {4};
            let output = Allocation::new(device,6*row_bytes,Tag::workspace("probe-test"))?;
            let original=vec![0x55;6*row_bytes];
            device.run(|| {
                library.copy_h2d(output.buffer,&original)?;
                inject(&stream,output.buffer,&media,0,6,2,fp32)
            })?;
            let mut actual=vec![0;original.len()];
            stream.copy_d2h(&mut actual,output.buffer)?;
            assert_eq!(&actual[..2*row_bytes],&original[..2*row_bytes]);
            assert_eq!(&actual[4*row_bytes..],&original[4*row_bytes..]);
            let expected=if fp32 { source.into_iter().flat_map(|v|(u32::from(v)<<16).to_le_bytes()).collect::<Vec<_>>() }
                else {source.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>()};
            assert_eq!(&actual[2*row_bytes..4*row_bytes],expected);
        }
        Ok(())
    }
}
