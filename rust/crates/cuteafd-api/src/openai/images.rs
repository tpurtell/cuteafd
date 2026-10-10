//! Bounded CPU preparation; CUDA ownership remains on the serving worker.
use anyhow::{ensure, Context, Result};
use base64::{
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
    Engine,
};
use deepseek_recipe_core::multimodal::ImageSource;
use cuteafd_loader::{V41Image, V41_MAX_IMAGES};
use std::{
    io::Read,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

const IMAGE_BYTES: usize = 32 << 20;
const TOTAL_BYTES: usize = 64 << 20;
pub(super) const BODY_BYTES: usize = 96 << 20;

pub(super) fn guard_content(body: &serde_json::Value, loaded: super::MediaCapabilities) -> Result<(), &'static str> {
    for message in body.get("messages").and_then(serde_json::Value::as_array).into_iter().flatten() {
        for part in message.get("content").and_then(serde_json::Value::as_array).into_iter().flatten() {
            match part.get("type").and_then(serde_json::Value::as_str) {
                Some("image_url" | "image" | "input_image") if !loaded.vision =>
                    return Err("this deployment has no image input; launch with VISION=auto"),
                Some("input_audio" | "audio") if !loaded.audio =>
                    return Err("this deployment has no audio input; launch with AUDIO=auto"),
                Some("video" | "video_url" | "input_video") => return Err("video input is not supported"),
                _ => {}
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
pub(super) struct ImageDecoder {
    agent: ureq::Agent,
    pub slots: Arc<Semaphore>,
}
impl ImageDecoder {
    pub fn new(slots: usize) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .redirects(3)
                .timeout_connect(Duration::from_secs(10))
                .build(),
            slots: Arc::new(Semaphore::new(slots.clamp(1, 4))),
        }
    }
    pub fn decode(&self, sources: Vec<ImageSource>) -> Result<Vec<V41Image>> {
        ensure!(
            sources.len() <= V41_MAX_IMAGES,
            "at most 16 images are supported"
        );
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut remaining = TOTAL_BYTES;
        sources
            .into_iter()
            .map(|source| {
                let timeout = deadline
                    .checked_duration_since(Instant::now())
                    .context("image preparation deadline exceeded")?
                    .min(Duration::from_secs(30));
                let limit = remaining.min(IMAGE_BYTES);
                let bytes = match source {
                    ImageSource::Bytes { data, .. } => data,
                    ImageSource::DataUrl { data_url, .. } => data_url_bytes(&data_url, limit)?,
                    ImageSource::Url { url, .. } => {
                        let parsed = url::Url::parse(&url).context("invalid image URL")?;
                        ensure!(
                            matches!(parsed.scheme(), "http" | "https"),
                            "image URL must use HTTP or HTTPS"
                        );
                        ensure!(
                            parsed.username().is_empty() && parsed.password().is_none(),
                            "image URL must not contain credentials"
                        );
                        let response = self
                            .agent
                            .get(parsed.as_str())
                            .timeout(timeout)
                            .call()
                            .context("image download failed")?;
                        if let Some(length) = response.header("Content-Length") {
                            ensure!(
                                length
                                    .parse::<u64>()
                                    .context("invalid image Content-Length")?
                                    <= limit as u64,
                                "encoded image exceeds byte limit"
                            );
                        }
                        let mut bytes = Vec::new();
                        response
                            .into_reader()
                            .take(limit as u64 + 1)
                            .read_to_end(&mut bytes)
                            .context("image download body failed")?;
                        bytes
                    }
                };
                ensure!(
                    bytes.len() <= limit,
                    "encoded image exceeds byte limit (32 MiB each, 64 MiB total)"
                );
                remaining -= bytes.len();
                let image = V41Image::decode(&bytes)?;
                ensure!(
                    Instant::now() < deadline,
                    "image preparation deadline exceeded"
                );
                Ok(image)
            })
            .collect()
    }
}

pub(super) fn data_url_bytes(url: &str, limit: usize) -> Result<Vec<u8>> {
    let (header, body) = url
        .strip_prefix("data:")
        .context("invalid image data URL")?
        .split_once(',')
        .context("image data URL has no payload")?;
    ensure!(
        header.split(';').next().unwrap_or("").starts_with("image/"),
        "data URL must contain an image media type"
    );
    // Percent encoding can triple the base64 payload size. Bound before decoding.
    ensure!(
        body.len() <= limit.saturating_add(2).div_ceil(3).saturating_mul(12),
        "image data URL exceeds byte limit"
    );
    let decoded = percent_encoding::percent_decode_str(body).collect::<Vec<_>>();
    let bytes = if header
        .split(';')
        .any(|part| part.eq_ignore_ascii_case("base64"))
    {
        ensure!(
            decoded.len() <= limit.div_ceil(3) * 4,
            "base64 image exceeds byte limit"
        );
        STANDARD
            .decode(&decoded)
            .or_else(|_| STANDARD_NO_PAD.decode(&decoded))
            .context("invalid image base64")?
    } else {
        decoded
    };
    ensure!(bytes.len() <= limit, "encoded image exceeds byte limit");
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_recipe_core::multimodal::ImageDetail;
    const PNG: &[u8] = include_bytes!("fixtures/black.png");

    #[test]
    fn data_formats_limits_and_recovery() {
        let decoder = ImageDecoder::new(1);
        let expected = V41Image::decode(PNG).unwrap();
        for url in [
            format!("data:image/png;base64,{}", STANDARD.encode(PNG)),
            format!("data:image/png;base64,{}", STANDARD_NO_PAD.encode(PNG)),
            format!(
                "data:image/png,{}",
                PNG.iter().map(|b| format!("%{b:02X}")).collect::<String>()
            ),
        ] {
            assert_eq!(data_url_bytes(&url, PNG.len()).unwrap(), PNG);
            assert!(data_url_bytes(&url, PNG.len() - 1).is_err());
            let images = decoder
                .decode(vec![ImageSource::DataUrl {
                    data_url: url,
                    detail: ImageDetail::High,
                }])
                .unwrap();
            assert_eq!(images[0].identity(), expected.identity());
        }
        for url in [
            "data:text/plain;base64,AAAA",
            "data:image/png;base64,!",
            "data:image/png",
        ] {
            assert!(data_url_bytes(url, 100).is_err());
        }
        for url in ["file:///etc/hosts", "http://user:password@localhost/image"] {
            assert!(decoder
                .decode(vec![ImageSource::Url {
                    url: url.into(),
                    detail: ImageDetail::High
                }])
                .is_err());
        }
        let source = ImageSource::Bytes {
            data: PNG.to_vec(),
            detail: ImageDetail::High,
        };
        assert!(decoder.decode(vec![source.clone(); 17]).is_err());
        assert_eq!(decoder.decode(vec![source; 16]).unwrap().len(), 16);
    }

    #[test]
    fn http_download_and_declared_and_streamed_bounds() {
        use std::{io::Write, net::TcpListener};
        for (headers, body, succeeds) in [
            (
                format!("Content-Length: {}\r\n", PNG.len()),
                PNG.to_vec(),
                true,
            ),
            (
                format!("Content-Length: {}\r\n", IMAGE_BYTES + 1),
                vec![],
                false,
            ),
            (String::new(), vec![0; IMAGE_BYTES + 1], false),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}/image", listener.local_addr().unwrap());
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = [0; 4096];
                stream.read(&mut request).unwrap();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nConnection: close\r\n{headers}\r\n"
                );
                let _ = stream.write_all(&body);
            });
            let result = ImageDecoder::new(1).decode(vec![ImageSource::Url {
                url,
                detail: ImageDetail::High,
            }]);
            assert_eq!(result.is_ok(), succeeds, "{:?}", result.err());
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn cancelled_download_holds_decoder_slot_until_reader_finishes() {
        use super::super::*;
        use std::{io::Write, net::TcpListener};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/image", listener.local_addr().unwrap());
        let (started, waiting) = tokio::sync::oneshot::channel();
        let (release, released) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = [0; 4096];
            stream.read(&mut request).unwrap();
            started.send(()).unwrap();
            released.recv_timeout(Duration::from_secs(5)).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                PNG.len()
            )
            .unwrap();
            stream.write_all(PNG).unwrap();
        });
        let (queue, mut receive) = mpsc::channel(1);
        let state = NativeState {
            queue,
            limits: NativeLimits::default(),
            images: ImageDecoder::new(1),
            tables: cuteafd_loader::MappedTableStatsReader::default(),
            admission: crate::openai::admission::Admission::new(1, Duration::from_millis(1)),
            stats: std::sync::Arc::new(std::sync::Mutex::new(serde_json::Value::Null)),
            profile: std::sync::Arc::new(crate::openai::ModelProfile::default()),
        };
        let body = |url: String| {
            json!({"model":MODEL,"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":url}}]}],"max_tokens":1})
        };
        let copied = state.clone();
        let slow = body(url);
        let handler = tokio::spawn(async move { chat(State(copied), Default::default(), None, Json(slow)).await });
        tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();
        handler.abort();
        assert!(handler.await.unwrap_err().is_cancelled());
        assert_eq!(state.queue.capacity(), 1);
        assert_eq!(state.images.slots.available_permits(), 0);
        let immediate = body(format!("data:image/png;base64,{}", STANDARD.encode(PNG)));
        let copied = state.clone();
        let queued = tokio::spawn(async move { chat(State(copied), Default::default(), None, Json(immediate)).await });
        tokio::task::yield_now().await;
        assert_eq!(state.queue.capacity(), 0);
        assert!(!queued.is_finished());
        queued.abort();
        assert!(queued.await.unwrap_err().is_cancelled());
        assert_eq!(state.queue.capacity(), 1);
        release.send(()).unwrap();
        let permit = tokio::time::timeout(Duration::from_secs(5), state.images.slots.acquire())
            .await
            .unwrap()
            .unwrap();
        drop(permit);
        server.join().unwrap();
        assert!(receive.try_recv().is_err());
        assert_eq!(state.queue.capacity(), 1);
        assert_eq!(state.images.slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn api_image_count_bad_input_and_saturation_recover() {
        use super::super::*;
        use tower::ServiceExt;
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_admission(tx.clone(), NativeLimits::default(), Arc::new(Mutex::new(Value::Null)), Duration::from_millis(1));
        let request = |count: usize, url: String| {
            let mut content = vec![json!({"type":"text","text":"Describe these images."})];
            content.extend((0..count).map(|_| json!({"type":"image_url","image_url":{"url":url}})));
            axum::http::Request::post("/v1/chat/completions").header("content-type", "application/json")
                .body(Body::from(json!({"model":MODEL,"messages":[{"role":"user","content":content}],"max_tokens":1}).to_string())).unwrap()
        };
        let url = format!("data:image/png;base64,{}", STANDARD.encode(PNG));
        for (count, value) in [(17, url.clone()), (1, "data:image/png;base64,AAAA".into())] {
            assert_eq!(
                app.clone()
                    .oneshot(request(count, value))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::BAD_REQUEST
            );
            assert!(rx.try_recv().is_err());
            assert_eq!(tx.capacity(), 1);
        }
        let permit = tx.clone().try_reserve_owned().unwrap();
        assert_eq!(
            app.clone()
                .oneshot(request(1, url.clone()))
                .await
                .unwrap()
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(permit);
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            assert_eq!(job.images.len(), 16);
            assert_eq!(job.prompt.matches("<｜deepseek_image｜>").count(), 16);
            assert!(!job.prompt.contains("<｜image｜>"));
            job.events
                .send(Ok(InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage {
                        prompt_tokens: 3000,
                        prompt_cache_hit_tokens: 0,
                    },
                }))
                
                .unwrap();
            job.events
                .send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Length,
                }))
                
                .unwrap();
        });
        assert_eq!(
            app.oneshot(request(16, url)).await.unwrap().status(),
            StatusCode::OK
        );
        worker.await.unwrap();
    }
}

#[cfg(test)]
mod guard_tests {
    use super::*;
    use crate::openai::*;
    use axum::{body::{to_bytes, Body}, http::Request};
    use serde_json::json;
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    #[tokio::test]
    async fn unavailable_media_is_rejected_before_rendering_for_every_family() {
        let profiles = [
            ModelProfile::new("deepseek-v4", ModelEncoding::DeepseekV4),
            ModelProfile::new("glm5", ModelEncoding::Glm(Arc::new(chat::glm5::fixtures::encoding()))),
            ModelProfile::new("glm5_flash", ModelEncoding::Glm(Arc::new(chat::glm5::fixtures::encoding()))),
            ModelProfile::new("qwen4", ModelEncoding::Qwen(Arc::new(chat::qwen4::fixtures::encoding()))),
            ModelProfile::new("mimo_v2", ModelEncoding::Qwen(Arc::new(chat::qwen4::fixtures::encoding()))),
        ];
        for profile in profiles {
            let (queue, mut receive) = mpsc::channel(1);
            let app = router_for_model(queue, NativeLimits::default(), Arc::new(std::sync::Mutex::new(serde_json::Value::Null)),
                Duration::from_secs(1), ConsoleHub::disabled(), profile.clone());
            for kind in ["image_url", "input_audio", "video_url"] {
                let body = json!({"model":profile.id,"messages":[{"role":"user","content":[{"type":kind}]}]});
                let response = app.clone().oneshot(Request::post("/v1/chat/completions")
                    .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
                assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST, "{} {kind}", profile.id);
                let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
                let message = String::from_utf8(bytes.to_vec()).unwrap();
                assert!(message.contains(if kind == "image_url" { "VISION=auto" } else if kind == "input_audio" { "AUDIO=auto" } else { "video" }));
                assert!(receive.try_recv().is_err());
            }
            let response = app.oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
            assert_eq!(body["data"][0]["capabilities"], json!({"vision":false,"audio":false}));
        }
        let v41 = ModelProfile::default();
        assert!(v41.capabilities.vision);
        assert!(!v41.capabilities.audio);
        assert!(guard_content(&json!({"messages":[{"content":[{"type":"image_url"}]}]}), v41.capabilities).is_ok());
        assert!(guard_content(&json!({"messages":[{"content":[{"type":"input_audio"}]}]}), v41.capabilities).is_err());
    }
}
