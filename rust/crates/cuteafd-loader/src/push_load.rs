//! Optional SparkNest push loading. The mount remains the authoritative fallback.
//!
//! A caller stages an entire group's load inside `push`, including its rank
//! completion barrier, and publishes only the returned value. On any error it
//! must drain outstanding copies and close all group sockets before returning;
//! `load_group` then invokes the existing mount loader once for the whole group.
//! Arena/ring integration is deliberately separate from this client boundary.

use std::{io, path::Path, time::Duration};

use nest_push::{
    client::{Coordinator, Rank},
    plan::{Entry, Pattern},
    proto::{Event, Forecast, RankSpec, SOCKET},
};

pub use nest_push::{client::{JoinArgs, Poll, Conn, Listener}, plan, proto, slot};
pub mod builder;
pub mod native;

// Applied by nest-push as SO_RCVTIMEO/SO_SNDTIMEO for each blocking I/O,
// not a wall-clock load limit: every received progress event renews the wait.
pub const PUSH_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// A joined rank's wait for its group's first event: the hub runs nothing (and
/// sends no Progress) until every rank has joined, and a peer rank may first
/// allocate a large arena. The idle deadline applies once the group runs.
pub const PUSH_START_TIMEOUT: Duration = Duration::from_secs(1200);
const HOSTS_XATTR: &str = "user.sparknest.hosts";

/// A SparkNest file a push source can stream: the hosts attribute is present.
/// Since SparkNest 0.6.5 (ADR-069) an archive-only file (empty hosts) is
/// pushable too: the hub streams it from the archive. Missing/unsupported
/// attributes, access errors and oversized values mean the mount path.
/// Follows HF snapshot symlinks like the tensor reader.
pub fn is_push_shard(shard: &Path) -> bool {
    let mut value = [0u8; 65536];
    rustix::fs::getxattr(shard, HOSTS_XATTR, &mut value[..]).is_ok()
}

/// Discover the namespace root and this host from ADR-064, never from a
/// container hostname or a coordinator-side address-to-host guess.
pub fn namespace(path: &Path) -> io::Result<(std::path::PathBuf, String)> {
    namespace_with(path, |root, bytes| {
        rustix::fs::getxattr(root, "user.sparknest.cluster", bytes).map_err(Into::into)
    })
}

fn namespace_with(path: &Path, mut read: impl FnMut(&Path, &mut [u8]) -> io::Result<usize>)
    -> io::Result<(std::path::PathBuf, String)> {
    let path = path.canonicalize()?;
    let mut bytes = vec![0; 65536];
    for root in path.ancestors() {
        let size = match read(root, &mut bytes) {
            Ok(size) => size,
            Err(error) if matches!(error.raw_os_error(), Some(code) if
                code == rustix::io::Errno::NODATA.raw_os_error()
                    || code == rustix::io::Errno::NOTSUP.raw_os_error()) => continue,
            Err(error) => return Err(error),
        };
        let value: serde_json::Value = serde_json::from_slice(&bytes[..size])?;
        let host = value["me"].as_str().filter(|host| !host.is_empty())
            .ok_or_else(|| io::Error::other("SparkNest cluster has no local host"))?;
        return Ok((root.into(), host.into()));
    }
    Err(io::Error::new(io::ErrorKind::NotFound, "not a SparkNest namespace"))
}

#[derive(Debug)]
pub enum LoadRoute {
    Push,
    Mount { reason: MountReason },
}

#[derive(Debug)]
pub enum MountReason {
    NoPushShard,
    PushFailed(io::Error),
}

#[derive(Debug)]
pub struct GroupLoad<T> {
    pub value: T,
    pub route: LoadRoute,
}

/// One transaction, never a mix of pushed ranks and pulled ranks. In particular,
/// a failure after partial delivery restarts the *entire* mount loader, not just
/// the missing entries. The mount loader's own error is preserved unchanged.
pub fn load_group<T, E>(
    shards: &[impl AsRef<Path>],
    push: impl FnOnce(&PushClient) -> io::Result<T>,
    mount: impl FnOnce() -> Result<T, E>,
) -> Result<GroupLoad<T>, E> {
    load_group_with_client(shards, &PushClient::default(), push, mount)
}

fn load_group_with_client<T, E>(
    shards: &[impl AsRef<Path>],
    client: &PushClient,
    push: impl FnOnce(&PushClient) -> io::Result<T>,
    mount: impl FnOnce() -> Result<T, E>,
) -> Result<GroupLoad<T>, E> {
    let reason = if shards.is_empty() || !shards.iter().all(|p| is_push_shard(p.as_ref())) {
        MountReason::NoPushShard
    } else {
        match push(client) {
            Ok(value) => {
                return Ok(GroupLoad {
                    value,
                    route: LoadRoute::Push,
                })
            }
            Err(error) => MountReason::PushFailed(error),
        }
    };
    tracing::debug!(?reason, "loading complete weight group through mount");
    mount().map(|value| GroupLoad {
        value,
        route: LoadRoute::Mount { reason },
    })
}

/// Uses the shared protocol implementation, including deadlines from the first
/// Create/Join send. No optional accelerator may hang normal model loading.
pub struct PushClient {
    socket: String,
    timeout: Duration,
}

impl Default for PushClient {
    fn default() -> Self {
        Self::configured(std::env::var("CUTEAFD_PUSH_SOCKET").ok())
    }
}

impl PushClient {
    fn configured(socket: Option<String>) -> Self {
        Self { socket: socket.filter(|name| !name.is_empty()).unwrap_or_else(|| SOCKET.into()),
            timeout: PUSH_IDLE_TIMEOUT }
    }

    /// Explicit endpoint for embedding and isolated startup protocol fixtures.
    pub fn at(socket: String, timeout: Duration) -> io::Result<Self> {
        if socket.is_empty() || timeout.is_zero() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid push endpoint or timeout"));
        }
        Ok(Self { socket, timeout })
    }

    pub fn create(&self, ranks: Vec<RankSpec>, files: Vec<String>) -> io::Result<PushGroup> {
        Coordinator::create(&self.socket, ranks, files, Some(self.timeout))
            .map(|coordinator| PushGroup { coordinator })
    }

    pub fn join(&self, args: JoinArgs) -> io::Result<PushRank> {
        Rank::join(&self.socket, args, Some(self.timeout)).map(|rank| PushRank { rank })
    }
}

pub struct PushGroup {
    coordinator: Coordinator,
}

impl PushGroup {
    pub fn credentials(&self) -> (u64, u64) {
        (self.coordinator.group, self.coordinator.token)
    }

    pub fn patterns(&self, patterns: Vec<Pattern>) -> io::Result<()> {
        self.coordinator.patterns(patterns)
    }

    /// The shared client batches entries below the socket message limit.
    pub fn entries(&self, entries: Vec<Entry>) -> io::Result<()> {
        self.coordinator.entries(entries)
    }

    pub fn start(&self) -> io::Result<Forecast> {
        self.coordinator.start()
    }

    pub fn next(&self) -> io::Result<Event> {
        match checked_event(self.coordinator.next()?)? {
            event @ (Event::Progress { .. } | Event::Done) => Ok(event),
            other => Err(unexpected(other)),
        }
    }
}

pub struct PushRank {
    rank: Rank,
}

impl PushRank {
    /// Borrow the landing-event socket for a shared blocking completion wake.
    pub fn fd(&self) -> std::os::fd::BorrowedFd<'_> { self.rank.fd() }

    pub fn try_next(&self) -> io::Result<Poll> {
        match self.rank.try_next()? {
            Poll::Ready(event) => {
                let event = checked_event(Some(event))?;
                match event {
                    event @ (Event::Filled { .. } | Event::Progress { .. } | Event::Done) => Ok(Poll::Ready(event)),
                    other => Err(unexpected(other)),
                }
            }
            other => Ok(other),
        }
    }
    pub fn next(&self) -> io::Result<Event> {
        match checked_event(self.rank.next()?)? {
            event @ (Event::Filled { .. } | Event::Progress { .. } | Event::Done) => Ok(event),
            other => Err(unexpected(other)),
        }
    }

    /// The group's first event, waited for up to `start` (every rank joining,
    /// then planning); afterwards `next` keeps the idle deadline `idle`.
    pub fn first(&self, start: Duration, idle: Duration) -> io::Result<Event> {
        self.rank.set_timeout(Some(start))?;
        let event = self.next();
        self.rank.set_timeout(Some(idle))?;
        event
    }

    /// Only after the slot's copy event completes (ring consumer owns this).
    pub fn release(&self, slot: u32) -> io::Result<()> {
        self.rank.release(slot)
    }
}

fn checked_event(event: Option<Event>) -> io::Result<Event> {
    match event {
        Some(Event::Failed { reason, .. } | Event::Refused(reason)) => {
            Err(io::Error::other(reason))
        }
        Some(event) => Ok(event),
        None => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "push socket closed before Done",
        )),
    }
}

fn unexpected(event: Event) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("unexpected push event: {event:?}"),
    )
}

#[cfg(test)]
mod tests;
