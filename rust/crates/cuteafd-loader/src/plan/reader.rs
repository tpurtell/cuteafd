//! Shard read handles: one open `File` per shard, reused across tensors.
//!
//! Opening a shard per tensor is a metadata round trip on FUSE mounts
//! (sparknest): a dense/lead weight read over 45 layers pays it thousands of
//! times and lands at a fraction of the fabric rate the expert reader reaches
//! with one handle per shard. [`ShardReader`] owns the handles so a loader
//! opens each shard once.
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Open shard handles keyed by resolved path, opened lazily and kept for the
/// reader's lifetime. One reader serves one load; not `Send`/`Sync` (each
/// loading thread keeps its own, so no lock is needed on the read path).
#[derive(Default)]
pub struct ShardReader {
    files: RefCell<HashMap<PathBuf, File>>,
}

impl ShardReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads exactly `bytes` at `offset` of `path`, opening (and then reusing)
    /// that shard's handle. `path` is the shard's full path (`snapshot/shard`).
    pub fn read_exact_at(&self, path: &Path, offset: u64, bytes: &mut [u8]) -> io::Result<()> {
        let mut files = self.files.borrow_mut();
        let file = match files.entry(path.to_path_buf()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(File::open(path)?),
        };
        file.read_exact_at(bytes, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::ShardReader;
    use std::fs;
    use std::os::unix::fs::FileExt;

    /// Reading through one handle per shard is byte-identical to opening the
    /// shard per tensor, across repeated and interleaved shards.
    #[test]
    fn reads_match_a_fresh_open_per_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a.safetensors"), dir.path().join("b.safetensors"));
        let (abytes, bbytes): (Vec<u8>, Vec<u8>) =
            ((0..4096u32).map(|i| i as u8).collect(), (0..4096u32).map(|i| (i * 7 + 3) as u8).collect());
        fs::write(&a, &abytes).unwrap();
        fs::write(&b, &bbytes).unwrap();
        // Same shard twice (the handle is reused), then the other, then back.
        let reads = [(&a, &abytes, 0usize, 100usize), (&a, &abytes, 1000, 128), (&a, &abytes, 1000, 128),
            (&b, &bbytes, 17, 200), (&a, &abytes, 4096 - 8, 8), (&b, &bbytes, 0, 1), (&b, &bbytes, 3000, 64)];
        let reader = ShardReader::new();
        for (path, source, offset, len) in reads {
            let mut got = vec![0u8; len];
            reader.read_exact_at(path, offset as u64, &mut got).unwrap();
            let mut want = vec![0u8; len];
            fs::File::open(path).unwrap().read_exact_at(&mut want, offset as u64).unwrap();
            assert_eq!(got, want, "{path:?} at {offset} for {len}");
            assert_eq!(got, &source[offset..offset + len]);
        }
    }

    #[test]
    fn a_missing_shard_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let mut got = [0u8; 4];
        assert!(ShardReader::new().read_exact_at(&dir.path().join("absent.safetensors"), 0, &mut got).is_err());
    }
}
