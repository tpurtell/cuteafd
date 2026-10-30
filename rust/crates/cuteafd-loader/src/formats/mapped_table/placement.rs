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
    fn a_plain_file_without_the_attribute_is_local() -> std::io::Result<()> {
        let file = tempfile::tempfile()?;
        assert_eq!(file_placement(&file), Placement::Local);
        Ok(())
    }
}
