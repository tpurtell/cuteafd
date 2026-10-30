//! Where a table file's copies live, from its own metadata: SparkNest answers
//! `user.sparknest.where` on every file it serves (`{size, generation, sealed,
//! copies: [{store, kind}]}`, kind `local`, `fabric`, `offgrid` or `archive`);
//! any other filesystem has no such attribute (ENODATA / ENOTSUP), and its
//! file is read from this host.
//!
//! Reading ahead pays where the rows come from this host (1-2 ms a read) and
//! not where every copy is remote: remote reads share one capped path on the
//! reading host (about 60K rows/s at 13 ms a read, whatever the number of
//! source hosts), so read-ahead only competes with the demand gathers there.
use std::fs::File;
use std::os::fd::AsRawFd;

const WHERE_XATTR: &std::ffi::CStr = c"user.sparknest.where";
/// Kernel FUSE connections, one directory per mounted connection, named by the
/// minor number of the mount's device.
pub(super) const FUSE_CONNECTIONS: &str = "/sys/fs/fuse/connections";

/// A FUSE mount's background request limit (`max_background`), for the file on
/// device `dev`, under `connections`. None when the file is missing or unreadable
/// (no cap): reading more requests at once than the mount takes only queues them
/// in the kernel, where a demand read waits behind read-ahead.
pub(super) fn fuse_max_background(connections: &std::path::Path, dev: u64) -> Option<u32> {
    let minor = libc::minor(dev);
    std::fs::read_to_string(connections.join(minor.to_string()).join("max_background")).ok()?
        .trim().parse::<u32>().ok().filter(|&limit| limit > 0)
}

/// The reactor's in-flight cap for an open file: the FUSE connection and its
/// `max_background` when every copy is remote and the file is on FUSE; local
/// and non-FUSE files keep the reactor's full depth.
pub(super) fn remote_read_cap(file: &File, placement: Placement, fuse: bool) -> Option<(u32, u32)> {
    if placement != Placement::Remote || !fuse { return None; }
    use std::os::unix::fs::MetadataExt;
    let dev = file.metadata().ok()?.dev();
    fuse_max_background(std::path::Path::new(FUSE_CONNECTIONS), dev).map(|limit| (libc::minor(dev), limit))
}

/// Every mapped table file's placement so far: whether any is remote, and a
/// generation that moves when that changes (the reader's CPUs follow it).
static REMOTE_FILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Record a mapped file's placement.
pub(super) fn note(placement: Placement) {
    if placement == Placement::Remote { REMOTE_FILES.fetch_add(1, std::sync::atomic::Ordering::Release); }
}

/// Mapped remote files so far.
pub(super) fn remote_files() -> u64 { REMOTE_FILES.load(std::sync::atomic::Ordering::Acquire) }

/// Whether a file's rows are read from this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// A copy on this host (or a plain local file).
    Local,
    /// Every copy on other hosts or archive stores.
    Remote,
}

/// The placement a `user.sparknest.where` value describes: local when any copy
/// has kind `local`. A value that does not parse counts as local (the file
/// opened, and nothing says its reads leave this host).
pub fn parse_where(value: &[u8]) -> Placement {
    #[derive(serde::Deserialize)]
    struct Copy { kind: String }
    #[derive(serde::Deserialize)]
    struct Where { copies: Vec<Copy> }
    match serde_json::from_slice::<Where>(value) {
        Ok(at) if !at.copies.iter().any(|copy| copy.kind == "local") => Placement::Remote,
        _ => Placement::Local,
    }
}

/// An open file's placement: its `user.sparknest.where` when it has one, else local.
pub fn file_placement(file: &File) -> Placement {
    let mut value = vec![0u8; 4096];
    // SAFETY: the fd is live for the call and `value` is a writable buffer of its length.
    let read = unsafe {
        libc::fgetxattr(file.as_raw_fd(), WHERE_XATTR.as_ptr(), value.as_mut_ptr().cast(), value.len())
    };
    if read < 0 { return Placement::Local; }
    value.truncate(read as usize);
    parse_where(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn where_values_select_local_or_remote() {
        let local = br#"{"copies":[{"kind":"local","store":"raptor"},{"kind":"fabric","store":"kiwi"}],"generation":1,"sealed":true,"size":101535150936}"#;
        let fabric = br#"{"copies":[{"kind":"fabric","store":"dodo"}],"generation":1,"sealed":true,"size":6813262858}"#;
        let replicated = br#"{"copies":[{"kind":"fabric","store":"dodo"},{"kind":"fabric","store":"moa"},{"kind":"fabric","store":"rhea"}],"generation":1,"sealed":true,"size":4000020872}"#;
        let archive = br#"{"copies":[{"kind":"archive","store":"models"}],"generation":1,"sealed":true,"size":1}"#;
        assert_eq!(parse_where(local), Placement::Local);
        assert_eq!(parse_where(fabric), Placement::Remote);
        assert_eq!(parse_where(replicated), Placement::Remote);
        assert_eq!(parse_where(archive), Placement::Remote);
        // Unparseable or unexpected values never disable read-ahead.
        assert_eq!(parse_where(b"not json"), Placement::Local);
        assert_eq!(parse_where(br#"{"size":1}"#), Placement::Local);
    }

    #[test]
    fn fuse_max_background_reads_the_connection_of_the_device_minor() -> std::io::Result<()> {
        let root = tempfile::tempdir()?;
        let dev = libc::makedev(0, 776);
        assert_eq!(fuse_max_background(root.path(), dev), None, "missing connection: no cap");
        std::fs::create_dir(root.path().join("776"))?;
        assert_eq!(fuse_max_background(root.path(), dev), None, "missing file: no cap");
        std::fs::write(root.path().join("776/max_background"), "128\n")?;
        assert_eq!(fuse_max_background(root.path(), dev), Some(128));
        assert_eq!(fuse_max_background(root.path(), libc::makedev(0, 78)), None, "another connection");
        std::fs::write(root.path().join("776/max_background"), "garbage")?;
        assert_eq!(fuse_max_background(root.path(), dev), None);
        std::fs::write(root.path().join("776/max_background"), "0")?;
        assert_eq!(fuse_max_background(root.path(), dev), None, "zero never caps reads to nothing");
        Ok(())
    }

    #[test]
    fn only_remote_fuse_files_are_capped() -> std::io::Result<()> {
        let file = tempfile::tempfile()?;
        assert_eq!(remote_read_cap(&file, Placement::Local, true), None);
        assert_eq!(remote_read_cap(&file, Placement::Remote, false), None);
        Ok(())
    }

    #[test]
    fn a_plain_file_without_the_attribute_is_local() -> std::io::Result<()> {
        let file = tempfile::tempfile()?;
        assert_eq!(file_placement(&file), Placement::Local);
        Ok(())
    }
}
