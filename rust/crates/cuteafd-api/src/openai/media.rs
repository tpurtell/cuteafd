//! Family-neutral, bounded preparation. CUDA work stays on the encoder owner.
pub mod audio;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::media::{
    decode_bounded, EncoderId, ImageProcessor, PreparedImage, PreprocessId, ProcessorConfig,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    io::Read,
    net::{IpAddr, ToSocketAddrs},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageUrlFetch {
    Off,
    Public,
    Any,
}
impl std::str::FromStr for ImageUrlFetch {
    type Err = &'static str;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "public" => Ok(Self::Public),
            "any" => Ok(Self::Any),
            _ => Err("image URL fetch must be off, public or any"),
        }
    }
}
#[derive(Debug, Clone)]
pub struct MediaSource {
    pub url: String,
    pub low: bool,
}
#[derive(Debug, Clone)]
pub struct MediaLimits {
    pub images: usize,
    pub decode_misses: usize,
    pub encoded_bytes: usize,
    pub decoded_bytes: usize,
    pub memo_entries: usize,
}
impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            images: 128,
            decode_misses: 16,
            encoded_bytes: 32 << 20,
            decoded_bytes: 64 << 20,
            memo_entries: 512,
        }
    }
}
/// Template order is message order then content-part order; no sorting by URL.
pub fn extract_image_sources(body: &Value, limit: usize) -> Result<Vec<MediaSource>> {
    let mut sources = Vec::new();
    for message in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for part in message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match part.get("type").and_then(Value::as_str) {
                Some("image_url") => {
                    ensure!(
                        sources.len() < limit,
                        "at most {limit} images are supported including history"
                    );
                    let image = part.get("image_url").context("image_url is required")?;
                    let url = image
                        .as_str()
                        .or_else(|| image.get("url").and_then(Value::as_str))
                        .context("image_url.url must be a string")?;
                    let detail = match image.get("detail") {
                        None => "auto",
                        Some(value) => value.as_str().context("image detail must be a string")?,
                    };
                    ensure!(
                        matches!(detail, "auto" | "high" | "low"),
                        "image detail must be auto, high or low"
                    );
                    sources.push(MediaSource {
                        url: url.into(),
                        low: detail == "low",
                    });
                }
                Some("image" | "input_image") => {
                    anyhow::bail!("send images as OpenAI image_url content parts")
                }
                Some("video" | "video_url" | "input_video") => {
                    anyhow::bail!("video input is not supported")
                }
                _ => {}
            }
        }
    }
    Ok(sources)
}

#[derive(Debug)]
struct MemoEntry {
    source: [u8; 32],
    processor: PreprocessId,
    image: Arc<PreparedImage>,
}
#[derive(Debug)]
pub struct MediaPreparer {
    config: ProcessorConfig,
    encoder: EncoderId,
    pub limits: MediaLimits,
    fetch: ImageUrlFetch,
    agent: ureq::Agent,
    memo: Mutex<VecDeque<MemoEntry>>,
    memo_hits: std::sync::atomic::AtomicU64,
    /// Bound both running CPU tasks and request preparation waiters at API admission.
    pub slots: Arc<tokio::sync::Semaphore>,
}
#[derive(Debug)]
pub struct PreparedMedia {
    pub images: Vec<Arc<PreparedImage>>,
    pub memo_hits: usize,
    pub decode_misses: usize,
    pub image_tokens: usize,
}
static PREPARATION_POLICY: std::sync::OnceLock<(usize, ImageUrlFetch)> = std::sync::OnceLock::new();
/// Startup-only generic-family policy; V4.1 keeps its established limits.
pub fn set_preparation_policy(max_image_tokens: usize, fetch: ImageUrlFetch) -> Result<()> {
    ensure!(
        (1..=16_384).contains(&max_image_tokens),
        "max image tokens must be 1 through 16384"
    );
    PREPARATION_POLICY
        .set((max_image_tokens, fetch))
        .map_err(|_| anyhow::anyhow!("media preparation policy already installed"))
}
impl MediaPreparer {
    pub fn memo_hits(&self) -> u64 {
        self.memo_hits.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }
    pub fn encoder(&self) -> EncoderId {
        self.encoder
    }
    pub fn fetch_policy(&self) -> ImageUrlFetch {
        self.fetch
    }
    /// Serving integrations use startup CLI policy; explicit `new` remains
    /// available to tests and applications with independently configured routers.
    pub fn for_loaded_encoder(
        mut config: ProcessorConfig,
        encoder: EncoderId,
        slots: usize,
    ) -> Result<Self> {
        let (cap, fetch) = PREPARATION_POLICY
            .get()
            .copied()
            .unwrap_or((4096, ImageUrlFetch::Public));
        config.max_image_tokens = match config.family {
            cuteafd_loader::media::ImageFamily::Qwen => cap.min(cuteafd_loader::media::QWEN_MAX_IMAGE_TOKENS),
            // Mistral's span cap follows the checkpoint's longest edge; the policy only lowers it.
            cuteafd_loader::media::ImageFamily::Mistral => cap.min(config.max_image_tokens),
            _ => cap,
        };
        Self::new(config, encoder, fetch, slots)
    }
    pub fn new(
        config: ProcessorConfig,
        encoder: EncoderId,
        fetch: ImageUrlFetch,
        slots: usize,
    ) -> Result<Self> {
        config.validate()?;
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .try_proxy_from_env(false)
            .timeout_connect(Duration::from_secs(10))
            .resolver(move |netloc: &str| {
                let addresses = resolve_bounded(netloc)?;
                if addresses.is_empty()
                    || (fetch == ImageUrlFetch::Public
                        && addresses.iter().any(|a| !public_ip(a.ip())))
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "image URL resolves to a non-public address",
                    ));
                }
                // The exact checked addresses go to the connection, avoiding DNS rebinding.
                Ok(addresses)
            })
            .build();
        Ok(Self {
            config,
            encoder,
            limits: MediaLimits::default(),
            fetch,
            agent,
            memo: Mutex::new(VecDeque::new()),
            memo_hits: std::sync::atomic::AtomicU64::new(0),
            slots: Arc::new(tokio::sync::Semaphore::new(slots.clamp(1, 4))),
        })
    }
    pub fn prepare(&self, sources: &[MediaSource]) -> Result<PreparedMedia> {
        self.prepare_verified(sources, &[])
    }
    /// Probe fixture identities bind the same fetched bytes used for preprocessing.
    pub fn prepare_verified(&self, sources: &[MediaSource], hashes: &[Option<String>]) -> Result<PreparedMedia> {
        ensure!(hashes.is_empty() || hashes.len() == sources.len(), "fixture count differs from image sources");
        ensure!(
            sources.len() <= self.limits.images,
            "at most {} images are supported including history",
            self.limits.images
        );
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut result = PreparedMedia {
            images: Vec::with_capacity(sources.len()),
            memo_hits: 0,
            decode_misses: 0,
            image_tokens: 0,
        };
        let mut decoded_bytes = 0usize;
        for (index, source) in sources.iter().enumerate() {
            ensure!(
                Instant::now() < deadline,
                "image preparation deadline exceeded"
            );
            let bytes = self.source_bytes(&source.url, deadline)?;
            let digest: [u8; 32] = Sha256::digest(&bytes).into();
            if let Some(Some(expected)) = hashes.get(index) {
                ensure!(super::probe::sha256_hex(expected) && format!("{:x}", Sha256::digest(&bytes)) == *expected,
                    "probe fixture source hash differs");
            }
            let config = self.config.with_detail(source.low);
            let processor = config.id();
            let cached = {
                let mut memo = self
                    .memo
                    .lock()
                    .map_err(|_| anyhow::anyhow!("image source memo unavailable"))?;
                memo.iter()
                    .position(|e| e.source == digest && e.processor == processor)
                    .map(|index| {
                        let entry = memo.remove(index).expect("located memo entry");
                        let image = entry.image.clone();
                        memo.push_back(entry);
                        image
                    })
            };
            let image = if let Some(image) = cached {
                result.memo_hits += 1;
                self.memo_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                image
            } else {
                ensure!(
                    result.decode_misses < self.limits.decode_misses,
                    "at most {} new images may need decode",
                    self.limits.decode_misses
                );
                let rgb = decode_bounded(
                    &bytes,
                    config.decode,
                    self.limits.decoded_bytes.saturating_sub(decoded_bytes),
                )?;
                decoded_bytes = decoded_bytes
                    .checked_add(rgb.data.len())
                    .context("decoded image byte count overflow")?;
                ensure!(
                    decoded_bytes <= self.limits.decoded_bytes,
                    "decoded images exceed 64 MiB request limit"
                );
                let image = Arc::new(config.prepare_rgb(rgb, self.encoder)?);
                result.decode_misses += 1;
                let mut memo = self
                    .memo
                    .lock()
                    .map_err(|_| anyhow::anyhow!("image source memo unavailable"))?;
                if self.limits.memo_entries > 0 {
                    // Concurrent misses may duplicate work, but never duplicate LRU entries.
                    if let Some(index) = memo
                        .iter()
                        .position(|e| e.source == digest && e.processor == processor)
                    {
                        memo.remove(index);
                    }
                    memo.push_back(MemoEntry {
                        source: digest,
                        processor,
                        image: image.clone(),
                    });
                    while memo.len() > self.limits.memo_entries {
                        memo.pop_front();
                    }
                }
                image
            };
            result.image_tokens = result
                .image_tokens
                .checked_add(image.tokens)
                .context("image token count overflow")?;
            result.images.push(image);
        }
        ensure!(
            Instant::now() < deadline,
            "image preparation deadline exceeded"
        );
        Ok(result)
    }
    fn source_bytes(&self, source: &str, deadline: Instant) -> Result<Vec<u8>> {
        if source.starts_with("data:") {
            return super::images::data_url_bytes(source, self.limits.encoded_bytes);
        }
        ensure!(
            self.fetch != ImageUrlFetch::Off,
            "remote image URL fetching is disabled"
        );
        let mut url = url::Url::parse(source).context("invalid image URL")?;
        for redirect in 0..=3 {
            ensure!(
                matches!(url.scheme(), "http" | "https"),
                "image URL must use HTTP or HTTPS"
            );
            ensure!(
                url.username().is_empty() && url.password().is_none(),
                "image URL must not contain credentials"
            );
            let host = url.host_str().context("image URL needs a host")?;
            if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
                ensure!(
                    self.fetch != ImageUrlFetch::Public || public_ip(ip),
                    "image URL is not a public address"
                );
            }
            let timeout = deadline
                .checked_duration_since(Instant::now())
                .context("image preparation deadline exceeded")?
                .min(Duration::from_secs(30));
            let response = self
                .agent
                .get(url.as_str())
                .timeout(timeout)
                .call()
                .context("image download failed")?;
            if (300..400).contains(&response.status()) {
                ensure!(redirect < 3, "too many image URL redirects");
                url = url.join(
                    response
                        .header("Location")
                        .context("image redirect has no location")?,
                )?;
                continue;
            }
            ensure!(
                (200..300).contains(&response.status()),
                "image download returned unexpected status"
            );
            if let Some(length) = response.header("Content-Length") {
                ensure!(
                    length.parse::<u64>()? <= self.limits.encoded_bytes as u64,
                    "encoded image exceeds byte limit"
                );
            }
            let mut bytes = Vec::new();
            response
                .into_reader()
                .take(self.limits.encoded_bytes as u64 + 1)
                .read_to_end(&mut bytes)?;
            ensure!(
                bytes.len() <= self.limits.encoded_bytes,
                "encoded image exceeds byte limit"
            );
            return Ok(bytes);
        }
        anyhow::bail!("too many image URL redirects")
    }
}

fn resolve_bounded(netloc: &str) -> std::io::Result<Vec<std::net::SocketAddr>> {
    let netloc = netloc.to_owned();
    resolve_with_timeout(
        move || netloc.to_socket_addrs().map(|addresses| addresses.collect()),
        Duration::from_secs(10),
    )
}

fn resolve_with_timeout(
    resolve: impl FnOnce() -> std::io::Result<Vec<std::net::SocketAddr>> + Send + 'static,
    timeout: Duration,
) -> std::io::Result<Vec<std::net::SocketAddr>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    struct Slot;
    impl Drop for Slot {
        fn drop(&mut self) {
            ACTIVE.fetch_sub(1, Ordering::Release);
        }
    }
    // libc DNS is not cancellable. Timed-out workers keep their slot until
    // they return, bounding both API waits and outstanding resolver threads.
    ACTIVE.fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
        (active < 4).then_some(active + 1)
    }).map_err(|_| std::io::Error::new(std::io::ErrorKind::WouldBlock, "image DNS resolver is busy"))?;
    let slot = Slot;
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new().name("image-dns".into()).spawn(move || {
        let _slot = slot;
        let _ = send.send(resolve());
    })?;
    receive.recv_timeout(timeout).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::TimedOut, "image DNS resolution timed out")
    })?
}

fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(a == 0
                || a == 10
                || a == 127
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && (b == 168 || (b == 0 && matches!(c, 0 | 2))))
                || (a == 198 && matches!(b, 18 | 19))
                || (a == 198 && b == 51 && c == 100)
                || (a == 203 && b == 0 && c == 113)
                || a >= 224)
        }
        IpAddr::V6(ip) => {
            if let Some(v4) = ip.to_ipv4_mapped() {
                return public_ip(IpAddr::V4(v4));
            }
            let s = ip.segments();
            // Only global unicast; reject documentation and special-use transition ranges.
            (s[0] & 0xe000) == 0x2000
                && !(s[0] == 0x2001 && (s[1] < 0x0200 || s[1] == 0x0db8))
                && s[0] != 0x2002
                && !(s[0] == 0x3fff && (s[1] & 0xf000) == 0)
        }
    }
}

#[cfg(test)]
mod tests;
