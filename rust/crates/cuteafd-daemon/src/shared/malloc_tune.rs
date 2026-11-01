//! glibc malloc tuning for the serving process (2026-10-25 coordinator OOM).
//!
//! The daemon has no custom global allocator, so every request's host transients — the JSON
//! body (megabytes at long contexts), its parsed tree, the rendered prompt, the token vectors —
//! come from glibc. glibc's mmap threshold is *dynamic*: it starts at 128 KiB, and every free
//! of an mmapped chunk raises it to that chunk's size (up to 32 MiB), so after the first few
//! long requests the multi-megabyte transients are carved from the arenas instead. Freed arena
//! chunks sit between the long-lived small objects of the next requests and are never returned
//! to the kernel: the process's anonymous RSS ratchets up with request volume although nothing
//! is leaked in the Rust sense. Pinning the threshold (`mallopt(M_MMAP_THRESHOLD)`) switches the
//! dynamic behaviour off, so every allocation at or above it is its own mapping and goes back
//! to the kernel on free; `M_TRIM_THRESHOLD` keeps the arena tops trimmed as well.
//!
//! `CUTEAFD_MALLOC_TUNE` selects the policy: unset or `on` = pin at the defaults below;
//! `off` = leave glibc alone; `<bytes>` = pin the mmap threshold at that size (the trim
//! threshold follows at four times it). Applied once, before any worker thread exists.

use std::fmt;

/// glibc caps `M_MMAP_THRESHOLD` at `HEAP_MAX_SIZE / 2` = 32 MiB on 64-bit; larger is refused.
pub const MAX_MMAP_THRESHOLD: usize = 32 << 20;
/// Default pinned mmap threshold: a 512-token page of ids is 2 KiB, a V4.1 logits row 517 KB,
/// a long prompt's body megabytes — the per-request transients that matter sit above 256 KiB.
pub const DEFAULT_MMAP_THRESHOLD: usize = 256 << 10;

/// What the environment asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Leave glibc's dynamic threshold in place.
    Off,
    /// Pin the mmap threshold at `mmap` bytes and the trim threshold at `trim` bytes.
    Pin { mmap: usize, trim: usize },
}

impl Policy {
    /// The default: pinned at [`DEFAULT_MMAP_THRESHOLD`], trim at four times it.
    pub const fn default_pin() -> Self {
        Policy::Pin { mmap: DEFAULT_MMAP_THRESHOLD, trim: 4 * DEFAULT_MMAP_THRESHOLD }
    }

    /// Parse `CUTEAFD_MALLOC_TUNE`: `None`/empty/`on`/`default` = the default pin, `off` = off,
    /// a byte count (optionally `k`/`m` suffixed) = pin at that size, clamped to glibc's maximum.
    /// Anything else is an error naming the value.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        let value = value.map(str::trim).unwrap_or("");
        match value.to_ascii_lowercase().as_str() {
            "" | "on" | "default" | "1" => Ok(Self::default_pin()),
            "off" | "0" | "none" => Ok(Policy::Off),
            text => {
                let (digits, unit) = match text.chars().last() {
                    Some('k') => (&text[..text.len() - 1], 1usize << 10),
                    Some('m') => (&text[..text.len() - 1], 1usize << 20),
                    _ => (text, 1usize),
                };
                let bytes = digits.parse::<usize>().ok().and_then(|n| n.checked_mul(unit))
                    .filter(|&bytes| bytes > 0)
                    .ok_or_else(|| format!("CUTEAFD_MALLOC_TUNE={value:?}: expected on, off or a byte count"))?;
                let mmap = bytes.min(MAX_MMAP_THRESHOLD);
                Ok(Policy::Pin { mmap, trim: mmap.saturating_mul(4) })
            }
        }
    }
}

/// What was applied, for the boot log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    pub policy: Policy,
    /// `mallopt` results in call order (1 = accepted); empty when the policy is off.
    pub results: Vec<i32>,
}

impl fmt::Display for Applied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.policy {
            Policy::Off => write!(f, "malloc tune: off (glibc dynamic mmap threshold)"),
            Policy::Pin { mmap, trim } => write!(f,
                "malloc tune: mmap threshold pinned at {mmap} bytes, trim threshold {trim} bytes (mallopt {:?})",
                self.results),
        }
    }
}

/// Apply `policy` to this process's glibc malloc.
pub fn apply(policy: Policy) -> Applied {
    let results = match policy {
        Policy::Off => Vec::new(),
        Policy::Pin { mmap, trim } => {
            let clamp = |bytes: usize| i32::try_from(bytes.min(i32::MAX as usize)).unwrap_or(i32::MAX);
            // SAFETY: mallopt only sets allocator parameters; it is documented safe to call at
            // any time and the daemon calls it before spawning its worker threads.
            unsafe {
                vec![
                    libc::mallopt(libc::M_MMAP_THRESHOLD, clamp(mmap)),
                    libc::mallopt(libc::M_TRIM_THRESHOLD, clamp(trim)),
                ]
            }
        }
    };
    Applied { policy, results }
}

/// Parse `CUTEAFD_MALLOC_TUNE` and apply it; an unparsable value is reported on stderr and
/// treated as the default pin (a typo must not silently reopen the OOM).
pub fn apply_from_env() -> Applied {
    let value = std::env::var("CUTEAFD_MALLOC_TUNE").ok();
    let policy = Policy::parse(value.as_deref()).unwrap_or_else(|error| {
        eprintln!("warning: {error}; using the default malloc tune");
        Policy::default_pin()
    });
    apply(policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_parses_on_off_and_sizes() {
        assert_eq!(Policy::parse(None), Ok(Policy::default_pin()));
        assert_eq!(Policy::parse(Some("")), Ok(Policy::default_pin()));
        assert_eq!(Policy::parse(Some(" ON ")), Ok(Policy::default_pin()));
        assert_eq!(Policy::parse(Some("off")), Ok(Policy::Off));
        assert_eq!(Policy::parse(Some("0")), Ok(Policy::Off));
        assert_eq!(Policy::parse(Some("1m")), Ok(Policy::Pin { mmap: 1 << 20, trim: 4 << 20 }));
        assert_eq!(Policy::parse(Some("512k")), Ok(Policy::Pin { mmap: 512 << 10, trim: 2 << 20 }));
        assert_eq!(Policy::parse(Some("65536")), Ok(Policy::Pin { mmap: 65536, trim: 4 * 65536 }));
        // Above glibc's ceiling clamps rather than fails.
        assert_eq!(Policy::parse(Some("1024m")), Ok(Policy::Pin { mmap: MAX_MMAP_THRESHOLD, trim: 4 * MAX_MMAP_THRESHOLD }));
        assert!(Policy::parse(Some("lots")).is_err());
        assert!(Policy::parse(Some("-1")).is_err());
    }

    #[test]
    fn default_pin_is_inside_glibc_limits() {
        let Policy::Pin { mmap, trim } = Policy::default_pin() else { panic!("default must pin") };
        assert!(mmap >= 128 << 10 && mmap <= MAX_MMAP_THRESHOLD);
        assert!(trim >= mmap);
    }

    /// glibc accepts the default pin (mallopt returns 1 for both parameters); `off` touches
    /// nothing. Deleting the pin would fail this: the results would be empty.
    #[test]
    fn apply_pins_glibc_thresholds() {
        let applied = apply(Policy::default_pin());
        assert_eq!(applied.results, vec![1, 1], "{applied}");
        assert_eq!(apply(Policy::Off).results, Vec::<i32>::new());
        assert!(applied.to_string().contains("pinned at 262144 bytes"));
        // Restore glibc's dynamic behaviour for the rest of the test binary: mallopt with
        // M_MMAP_THRESHOLD disables it, so re-pin at the default start value.
        let _ = apply(Policy::Pin { mmap: 128 << 10, trim: 128 << 10 });
    }
}
