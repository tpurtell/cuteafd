//! Startup routing is decided once from namespace identity, never from a failed load.
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PushPolicy {
    Mount,
    Required,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum PushStartupError {
    #[error("cannot discover SparkNest namespace for {snapshot:?}: {source}")]
    Namespace { snapshot: PathBuf, #[source] source: std::io::Error },
    #[error("required push planning failed: {0:#}")]
    Planning(#[source] anyhow::Error),
    #[error("required coordinator push delivery failed: {0:#}")]
    Delivery(#[source] anyhow::Error),
    #[error("required coordinator push did not commit; mount loading is forbidden in a SparkNest namespace")]
    Uncommitted,
}

impl PushPolicy {
    pub(crate) fn for_snapshot(snapshot: &Path) -> Result<Self, PushStartupError> {
        Self::from_namespace(snapshot, cuteafd_loader::push_load::namespace(snapshot))
    }

    pub(crate) fn from_namespace(snapshot: &Path,
        namespace: std::io::Result<(PathBuf, String)>) -> Result<Self, PushStartupError> {
        match namespace {
            Ok(_) => Ok(Self::Required),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound
                && error.raw_os_error().is_none() && snapshot.exists() => Ok(Self::Mount),
            Err(source) => Err(PushStartupError::Namespace { snapshot: snapshot.into(), source }),
        }
    }

    /// Planning is not attempted for a mount snapshot; errors never change the selected route.
    pub(crate) fn plan<T>(self, plan: impl FnOnce() -> anyhow::Result<T>) -> Result<Option<T>, PushStartupError> {
        match self {
            Self::Mount => Ok(None),
            Self::Required => plan().map(Some).map_err(PushStartupError::Planning),
        }
    }

    /// Call after the receive protocol has committed or drained every unpublished owner.
    pub(crate) fn committed<T>(self, delivered: anyhow::Result<Option<T>>) -> Result<Option<T>, PushStartupError> {
        match delivered.map_err(PushStartupError::Delivery)? {
            Some(value) => Ok(Some(value)),
            None if self == Self::Mount => Ok(None),
            None => Err(PushStartupError::Uncommitted),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonnamespace_uses_mount_without_attempting_push() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let policy = PushPolicy::for_snapshot(root.path())?;
        assert_eq!(policy, PushPolicy::Mount);
        assert!(policy.plan::<()>(|| panic!("mount must not prepare a push group"))?.is_none());
        assert!(policy.committed::<()>(Ok(None))?.is_none());
        Ok(())
    }

    #[test]
    fn engaged_push_never_turns_planning_or_delivery_failure_into_mount() {
        assert!(matches!(PushPolicy::Required.plan::<()>(|| anyhow::bail!("bad headers")),
            Err(PushStartupError::Planning(_))));
        assert!(matches!(PushPolicy::Required.committed::<()>(Ok(None)), Err(PushStartupError::Uncommitted)));
        assert!(matches!(PushPolicy::Required.committed::<()>(Err(anyhow::anyhow!("drained failure"))),
            Err(PushStartupError::Delivery(_))));
        assert_eq!(PushPolicy::Required.committed(Ok(Some(7))).unwrap(), Some(7));
    }

    #[test]
    fn namespace_identity_requires_push_and_malformed_identity_is_an_error() -> anyhow::Result<()> {
        use std::os::unix::ffi::OsStrExt;
        let root = tempfile::tempdir()?;
        let child = root.path().join("snapshot");
        std::fs::create_dir(&child)?;
        let path = std::ffi::CString::new(root.path().as_os_str().as_bytes())?;
        let name = c"user.sparknest.cluster";
        for (value, valid) in [(br#"{"me":"raptor"}"#.as_slice(), true), (b"broken".as_slice(), false)] {
            // SAFETY: both C strings and the immutable attribute buffer live through setxattr.
            let result = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
            anyhow::ensure!(result == 0, "setxattr failed: {}", std::io::Error::last_os_error());
            let policy = PushPolicy::for_snapshot(&child);
            if valid { assert_eq!(policy?, PushPolicy::Required); }
            else { assert!(matches!(policy, Err(PushStartupError::Namespace { .. }))); }
        }
        Ok(())
    }

    #[test]
    fn namespace_read_errors_fail_coordinator_startup_with_the_original_errno() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        for code in [libc::EACCES, libc::EIO, libc::ENOENT] {
            let result = PushPolicy::from_namespace(root.path(), Err(std::io::Error::from_raw_os_error(code)));
            match result {
                Err(PushStartupError::Namespace { snapshot, source }) => {
                    assert_eq!(snapshot, root.path());
                    assert_eq!(source.raw_os_error(), Some(code));
                }
                other => panic!("namespace read failure became {other:?}"),
            }
        }
        Ok(())
    }

    #[test]
    fn missing_snapshot_is_a_discovery_error_not_a_mount_fallback() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        assert!(matches!(PushPolicy::for_snapshot(&root.path().join("missing")),
            Err(PushStartupError::Namespace { .. })));
        Ok(())
    }
}
