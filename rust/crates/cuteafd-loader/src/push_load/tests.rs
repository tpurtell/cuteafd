use super::*;
use nest_push::{client::Conn, proto::Request};
use socket2::{Domain, SockAddr, Socket, Type};
use std::{cell::Cell, fs::File, os::fd::AsRawFd, sync::mpsc, thread};
use tempfile::{NamedTempFile, TempDir};

struct Server {
    _directory: TempDir,
    client: PushClient,
    thread: thread::JoinHandle<()>,
}

impl Server {
    fn new(script: impl FnOnce(Conn) + Send + 'static) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("push.sock");
        let socket = Socket::new(Domain::UNIX, Type::SEQPACKET, None).unwrap();
        socket.bind(&SockAddr::unix(&path).unwrap()).unwrap();
        socket.listen(1).unwrap();
        let thread = thread::spawn(move || {
            let (peer, _) = socket.accept().unwrap();
            let conn = Conn::from_fd(peer.into());
            conn.set_timeout(Some(Duration::from_secs(2))).unwrap();
            script(conn);
        });
        Self {
            client: PushClient {
                socket: path.to_str().unwrap().into(),
                timeout: Duration::from_millis(50),
                setting: "push socket",
            },
            _directory: directory,
            thread,
        }
    }

    fn finish(self) {
        self.thread.join().unwrap();
    }
}

fn request(conn: &Conn) -> Request {
    let mut bytes = Vec::new();
    assert!(conn.recv(&mut bytes).unwrap().is_some());
    proto::decode(&bytes).unwrap()
}

fn event(conn: &Conn, event: Event) {
    conn.send(&proto::encode(&event), None).unwrap();
}

fn created(conn: &Conn) {
    assert!(matches!(
        request(conn),
        Request::Create {
            version: proto::VERSION,
            ..
        }
    ));
    event(
        conn,
        Event::Created {
            group: 7,
            token: 11,
        },
    );
}

fn push_shard() -> NamedTempFile {
    let shard = NamedTempFile::new().unwrap();
    rustix::fs::setxattr(
        shard.path(),
        HOSTS_XATTR,
        b"rank-a,rank-b",
        rustix::fs::XattrFlags::empty(),
    )
    .unwrap();
    shard
}

fn assert_fallback(client: &PushClient, push: impl FnOnce(&PushClient) -> io::Result<()>) {
    let shards = [push_shard(), push_shard()];
    let calls = Cell::new(0);
    let loaded = load_group_with_client(
        &shards.iter().map(|p| p.path()).collect::<Vec<_>>(),
        client,
        push,
        || {
            calls.set(calls.get() + 1);
            Ok::<_, ()>(())
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 1, "whole mount group runs exactly once");
    assert!(matches!(
        loaded.route,
        LoadRoute::Mount {
            reason: MountReason::PushFailed(_)
        }
    ));
}

#[test]
fn absent_attribute_missing_file_and_empty_group_use_mount_without_push() {
    let shard = NamedTempFile::new().unwrap();
    let push_shard = push_shard();
    let missing = shard.path().with_extension("missing");
    for shards in [
        vec![shard.path()],
        vec![&missing],
        vec![],
        vec![push_shard.path(), shard.path()],
    ] {
        let loaded = load_group(
            &shards,
            |_| -> io::Result<u32> { panic!("push must not run") },
            || Ok::<_, ()>(42),
        )
        .unwrap();
        assert_eq!(loaded.value, 42);
        assert!(matches!(
            loaded.route,
            LoadRoute::Mount {
                reason: MountReason::NoPushShard
            }
        ));
    }
}

#[test]
fn archive_only_files_are_pushed_and_snapshot_symlinks_are_followed() {
    let shard = push_shard();
    let directory = tempfile::tempdir().unwrap();
    let link = directory.path().join("shard");
    std::os::unix::fs::symlink(shard.path(), &link).unwrap();
    assert!(is_push_shard(&link));
    rustix::fs::setxattr(
        shard.path(),
        HOSTS_XATTR,
        b"",
        rustix::fs::XattrFlags::empty(),
    )
    .unwrap();
    // SparkNest 0.6.5 (ADR-069) streams archive-only files: empty hosts still push.
    assert!(is_push_shard(shard.path()), "archive-only files are pushable");
    assert!(is_push_shard(&link));
    for value in [&b"[]"[..], b" , "] {
        rustix::fs::setxattr(shard.path(), HOSTS_XATTR, value, rustix::fs::XattrFlags::empty()).unwrap();
        assert!(is_push_shard(shard.path()), "{value:?}");
    }
}

#[test]
fn namespace_uses_mount_identity_and_follows_snapshot_symlinks() {
    let directory = tempfile::tempdir().unwrap();
    let snapshot = directory.path().join("snapshot");
    std::fs::create_dir(&snapshot).unwrap();
    rustix::fs::setxattr(directory.path(), "user.sparknest.cluster", br#"{"me":"namespace-rank"}"#,
        rustix::fs::XattrFlags::empty()).unwrap();
    assert_eq!(namespace(&snapshot).unwrap(), (directory.path().into(), "namespace-rank".into()));
    let aliases = tempfile::tempdir().unwrap();
    let alias = aliases.path().join("snapshot-link");
    std::os::unix::fs::symlink(&snapshot, &alias).unwrap();
    assert_eq!(namespace(&alias).unwrap(), (directory.path().into(), "namespace-rank".into()));
    rustix::fs::setxattr(directory.path(), "user.sparknest.cluster", b"{}",
        rustix::fs::XattrFlags::empty()).unwrap();
    assert!(namespace(&snapshot).is_err());
    assert!(PushClient::at(String::new(), Duration::from_secs(1)).is_err());
    assert!(PushClient::at("fixture".into(), Duration::ZERO).is_err());
}

#[test]
fn namespace_skips_only_absent_or_unsupported_attributes() {
    let root = tempfile::tempdir().unwrap();
    let child = root.path().join("snapshot");
    std::fs::create_dir(&child).unwrap();
    for errno in [rustix::io::Errno::NODATA, rustix::io::Errno::NOTSUP] {
        let mut reads = 0;
        let found = namespace_with(&child, |_, bytes| {
            reads += 1;
            if reads == 1 { return Err(errno.into()); }
            let identity = br#"{"me":"ancestor"}"#;
            bytes[..identity.len()].copy_from_slice(identity);
            Ok(identity.len())
        }).unwrap();
        assert_eq!(found, (root.path().into(), "ancestor".into()));
        assert_eq!(reads, 2);
    }
    for errno in [rustix::io::Errno::ACCESS, rustix::io::Errno::IO, rustix::io::Errno::NOENT,
        rustix::io::Errno::RANGE] {
        let mut reads = 0;
        let error = namespace_with(&child, |_, _| {
            reads += 1;
            Err(errno.into())
        }).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(errno.raw_os_error()));
        assert_eq!(reads, 1, "a failed ancestor must not be hidden by a higher namespace");
    }
}

#[test]
fn container_subtree_bind_does_not_guess_the_hidden_namespace_root() {
    let directory = tempfile::tempdir().unwrap();
    let namespace_root = directory.path().join("namespace");
    let original = namespace_root.join("hf-home/hub/model/snapshots/revision");
    std::fs::create_dir_all(&original).unwrap();
    rustix::fs::setxattr(&namespace_root, "user.sparknest.cluster", br#"{"me":"namespace-rank"}"#,
        rustix::fs::XattrFlags::empty()).unwrap();
    assert_eq!(namespace(&original).unwrap().0, namespace_root);
    // A subtree bind exposes the hub inode and descendants, not the FUSE root.
    // A separate directory models that ancestor visibility without mount privileges.
    let bind_root = directory.path().join("container/hf/hub");
    let exposed = bind_root.join("model/snapshots/revision");
    std::fs::create_dir_all(&exposed).unwrap();
    assert!(namespace(&exposed).is_err(), "subtree-only serving bind must remain on mount");
    // If the actual namespace root is bound at this path, its root attribute
    // must be visible there; paths are then relative to precisely that root.
    rustix::fs::setxattr(&bind_root, "user.sparknest.cluster", br#"{"me":"namespace-rank"}"#,
        rustix::fs::XattrFlags::empty()).unwrap();
    let (root, host) = namespace(&exposed).unwrap();
    assert_eq!(root, bind_root);
    assert_eq!(host, "namespace-rank");
    assert_eq!(exposed.strip_prefix(root).unwrap(), Path::new("model/snapshots/revision"));
}

#[test]
fn absent_socket_restarts_whole_group() {
    let directory = tempfile::tempdir().unwrap();
    let client = PushClient {
        socket: directory.path().join("absent").to_str().unwrap().into(),
        timeout: Duration::from_millis(50),
        setting: "push socket",
    };
    assert_fallback(&client, |c| c.create(vec![], vec![]).map(|_| ()));
}

#[test]
fn create_refusal_failure_disconnect_and_bad_wire_restart_whole_group() {
    for response in [
        Some(Event::Refused("old server".into())),
        Some(Event::Failed {
            reason: "source gone".into(),
            missing: vec![],
        }),
        None,
    ] {
        let server = Server::new(move |conn| {
            assert!(matches!(request(&conn), Request::Create { .. }));
            if let Some(response) = response {
                event(&conn, response);
            }
        });
        assert_fallback(&server.client, |c| c.create(vec![], vec![]).map(|_| ()));
        server.finish();
    }
    let server = Server::new(|conn| {
        request(&conn);
        conn.send(&[255], None).unwrap();
    });
    assert_fallback(&server.client, |c| c.create(vec![], vec![]).map(|_| ()));
    server.finish();
}

#[test]
fn start_refusal_and_stream_failure_restart_whole_group() {
    for fail_at_start in [true, false] {
        let server = Server::new(move |conn| {
            created(&conn);
            assert!(matches!(request(&conn), Request::Patterns(_)));
            assert!(matches!(request(&conn), Request::Entries(_)));
            assert_eq!(request(&conn), Request::Start);
            if fail_at_start {
                event(&conn, Event::Refused("plan rejected".into()));
            } else {
                event(&conn, Event::Planned(Forecast::default()));
                event(
                    &conn,
                    Event::Progress {
                        done: 16,
                        total: 32,
                    },
                );
                event(
                    &conn,
                    Event::Failed {
                        reason: "partial load".into(),
                        missing: vec![],
                    },
                );
            }
        });
        assert_fallback(&server.client, |c| {
            let group = c.create(vec![], vec![])?;
            group.patterns(vec![])?;
            group.entries(vec![])?;
            group.start()?;
            loop {
                if group.next()? == Event::Done {
                    return Ok(());
                }
            }
        });
        server.finish();
    }
}

#[test]
fn join_refusal_restarts_whole_group_and_passes_a_real_fd() {
    let server = Server::new(|conn| {
        let mut bytes = Vec::new();
        let received_fd = conn.recv(&mut bytes).unwrap().unwrap().unwrap();
        let join: Request = proto::decode(&bytes).unwrap();
        assert!(matches!(
            join,
            Request::Join {
                group: 7,
                rank: 1,
                ..
            }
        ));
        assert_eq!(File::from(received_fd).metadata().unwrap().len(), 4096);
        event(&conn, Event::Refused("registration failed".into()));
    });
    let ring = tempfile::tempfile().unwrap();
    ring.set_len(4096).unwrap();
    assert_fallback(&server.client, |c| {
        c.join(JoinArgs {
            group: 7,
            token: 11,
            rank: 1,
            ring_fd: ring.as_raw_fd(),
            slots: 1,
            slot_bytes: 4096,
        })
        .map(|_| ())
    });
    server.finish();
}

#[test]
fn idle_create_is_bounded_and_cancels_before_mount() {
    let (unblock, blocked) = mpsc::channel();
    let server = Server::new(move |conn| {
        request(&conn);
        blocked.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            conn.event().unwrap().is_none(),
            "group socket closed before fallback"
        );
    });
    assert_fallback(&server.client, |c| {
        let error = match c.create(vec![], vec![]) {
            Ok(_) => panic!("hung server succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        Err(error)
    });
    unblock.send(()).unwrap();
    server.finish();
}

#[test]
fn success_skips_mount_and_mount_error_is_preserved() {
    let shard = push_shard();
    let success = load_group(
        &[shard.path()],
        |_| Ok(7),
        || -> Result<_, ()> { panic!("mount must not run") },
    )
    .unwrap();
    assert_eq!(success.value, 7);
    assert!(matches!(success.route, LoadRoute::Push));
    let result = load_group(
        &[shard.path()],
        |_| -> io::Result<()> { Err(io::Error::other("push failed")) },
        || Err("mount error"),
    );
    assert_eq!(result.unwrap_err(), "mount error");
}

#[test]
fn done_progress_rank_release_and_group_cancel_use_shared_protocol() {
    let server = Server::new(|conn| {
        created(&conn);
        assert_eq!(request(&conn), Request::Start);
        event(&conn, Event::Planned(Forecast::default()));
        event(
            &conn,
            Event::Progress {
                done: 32,
                total: 32,
            },
        );
        event(&conn, Event::Done);
        assert!(conn.event().unwrap().is_none());
    });
    {
        let group = server.client.create(vec![], vec![]).unwrap();
        assert_eq!(group.credentials(), (7, 11));
        group.start().unwrap();
        assert!(matches!(group.next().unwrap(), Event::Progress { .. }));
        assert_eq!(group.next().unwrap(), Event::Done);
    }
    server.finish();
    let server = Server::new(|conn| {
        assert!(matches!(request(&conn), Request::Join { .. }));
        event(&conn, Event::Joined);
        event(&conn, Event::Filled { slot: 0, len: 4096 });
        assert_eq!(request(&conn), Request::Release { slot: 0 });
        event(&conn, Event::Done);
    });
    let ring = tempfile::tempfile().unwrap();
    let rank = server
        .client
        .join(JoinArgs {
            group: 7,
            token: 11,
            rank: 1,
            ring_fd: ring.as_raw_fd(),
            slots: 1,
            slot_bytes: 4096,
        })
        .unwrap();
    assert_eq!(rank.next().unwrap(), Event::Filled { slot: 0, len: 4096 });
    rank.release(0).unwrap();
    assert_eq!(rank.next().unwrap(), Event::Done);
    server.finish();
}

#[test]
fn rank_nonblocking_poll_preserves_events_failures_and_eof() {
    let (step, commands) = mpsc::channel();
    let (sent, ready) = mpsc::channel();
    let server = Server::new(move |conn| {
        assert!(matches!(request(&conn), Request::Join { .. }));
        event(&conn, Event::Joined);
        commands.recv_timeout(Duration::from_secs(2)).unwrap();
        event(&conn, Event::Filled { slot: 0, len: 4096 });
        sent.send(()).unwrap();
        commands.recv_timeout(Duration::from_secs(2)).unwrap();
        event(
            &conn,
            Event::Failed {
                reason: "injected".into(),
                missing: vec![],
            },
        );
        sent.send(()).unwrap();
        commands.recv_timeout(Duration::from_secs(2)).unwrap();
    });
    let ring = tempfile::tempfile().unwrap();
    let rank = server
        .client
        .join(JoinArgs {
            group: 7,
            token: 11,
            rank: 0,
            ring_fd: ring.as_raw_fd(),
            slots: 1,
            slot_bytes: 4096,
        })
        .unwrap();
    assert!(matches!(rank.try_next().unwrap(), Poll::Empty));
    step.send(()).unwrap();
    ready.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(
        rank.try_next().unwrap(),
        Poll::Ready(Event::Filled { slot: 0, len: 4096 })
    ));
    step.send(()).unwrap();
    ready.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(rank
        .try_next()
        .unwrap_err()
        .to_string()
        .contains("injected"));
    step.send(()).unwrap();
    server.finish();
    assert!(matches!(rank.try_next().unwrap(), Poll::Closed));
}

#[test]
fn join_start_and_stream_idle_deadlines_fall_back() {
    for phase in ["join", "start", "next"] {
        let (unblock, blocked) = mpsc::channel();
        let server = Server::new(move |conn| {
            if phase == "join" {
                assert!(matches!(request(&conn), Request::Join { .. }));
            } else {
                created(&conn);
                assert_eq!(request(&conn), Request::Start);
                if phase == "next" {
                    event(&conn, Event::Planned(Forecast::default()));
                }
            }
            blocked.recv_timeout(Duration::from_secs(2)).unwrap();
            assert!(conn.event().unwrap().is_none());
        });
        let ring = tempfile::tempfile().unwrap();
        assert_fallback(&server.client, |client| {
            let attempt = || -> io::Result<()> {
                if phase == "join" {
                    client.join(JoinArgs {
                        group: 7,
                        token: 11,
                        rank: 1,
                        ring_fd: ring.as_raw_fd(),
                        slots: 1,
                        slot_bytes: 4096,
                    })?;
                } else {
                    let group = client.create(vec![], vec![])?;
                    group.start()?;
                    group.next()?;
                }
                Ok(())
            };
            let error = attempt().unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::TimedOut, "{phase}");
            Err(error)
        });
        unblock.send(()).unwrap();
        server.finish();
    }
}

#[test]
fn hub_progress_resets_receive_idle_timeout_instead_of_total_load_deadline() {
    let server = Server::new(|conn| {
        created(&conn);
        assert_eq!(request(&conn), Request::Start);
        event(&conn, Event::Planned(Forecast::default()));
        for done in 0..20 {
            thread::sleep(Duration::from_millis(100));
            event(&conn, Event::Progress { done, total: 20 });
        }
        event(&conn, Event::Done);
    });
    let client = PushClient::at(server.client.socket.clone(), Duration::from_secs(1)).unwrap();
    let group = client.create(vec![], vec![]).unwrap();
    group.start().unwrap();
    let started = std::time::Instant::now();
    for expected in 0..20 {
        assert!(matches!(group.next().unwrap(), Event::Progress { done, .. } if done == expected));
    }
    assert!(started.elapsed() > Duration::from_secs(1));
    assert!(matches!(group.next().unwrap(), Event::Done));
    drop(group);
    server.finish();
}

#[test]
fn full_socket_backlog_cannot_hang_model_loading() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("full.sock");
    let listener = Socket::new(Domain::UNIX, Type::SEQPACKET, None).unwrap();
    listener.bind(&SockAddr::unix(&path).unwrap()).unwrap();
    listener.listen(0).unwrap();
    // Linux admits backlog + 1 connections. Leave the first unaccepted.
    let _occupant = Conn::connect(path.to_str().unwrap()).unwrap();
    let client = PushClient {
        socket: path.to_str().unwrap().into(),
        timeout: Duration::from_millis(50),
        setting: "push socket",
    };
    assert_fallback(&client, |client| {
        let error = match client.create(vec![], vec![]) {
            Ok(_) => panic!("full listener backlog connected"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        Err(error)
    });
}

#[test]
fn checked_plan_upload_batches_messages_without_changing_ids() {
    use super::builder::CheckedPlan;
    use nest_push::plan::{Dim, Entry, Pattern, Plan, Steps, Totals};
    let patterns = (0..2)
        .map(|_| Pattern {
            run: 1,
            dims: vec![Dim {
                steps: Steps::Indices((0..40_000).collect()),
                src_stride: 1,
                dst_stride: 1,
            }],
        })
        .collect::<Vec<_>>();
    let entries = (0..9000)
        .map(|i| Entry {
            file: 0,
            pattern: i % 2,
            src_base: 0,
            dst_base: u64::from(i) * 40_000,
            rank: 0,
        })
        .collect::<Vec<_>>();
    let want_patterns = patterns.clone();
    let want_entries = entries.clone();
    let server = Server::new(move |conn| {
        created(&conn);
        let mut patterns = Vec::new();
        let mut entries = Vec::new();
        while patterns.len() < want_patterns.len() {
            let Request::Patterns(batch) = request(&conn) else {
                panic!("expected patterns")
            };
            assert!(proto::encode(&Request::Patterns(batch.clone())).len() <= proto::MAX_MSG);
            patterns.extend(batch);
        }
        while entries.len() < want_entries.len() {
            let Request::Entries(batch) = request(&conn) else {
                panic!("expected entries")
            };
            assert!(proto::encode(&Request::Entries(batch.clone())).len() <= proto::MAX_MSG);
            entries.extend(batch);
        }
        assert_eq!(patterns, want_patterns);
        assert_eq!(entries, want_entries);
    });
    let group = server.client.create(vec![], vec![]).unwrap();
    CheckedPlan {
        plan: Plan {
            files: vec!["weights".into()],
            patterns,
            entries,
        },
        file_sizes: vec![],
        arenas: vec![],
        totals: Totals::default(),
    }
    .upload(&group)
    .unwrap();
    drop(group);
    server.finish();
}

#[test]
fn eof_refusal_failure_and_wrong_phase_events_are_errors() {
    for response in [
        None,
        Some(Event::Refused("refused".into())),
        Some(Event::Failed {
            reason: "failed".into(),
            missing: vec![],
        }),
        Some(Event::Joined),
    ] {
        let server = Server::new(move |conn| {
            created(&conn);
            if let Some(e) = response {
                event(&conn, e);
            }
        });
        let group = server.client.create(vec![], vec![]).unwrap();
        assert!(group.next().is_err());
        server.finish();
    }
}

#[test]
fn socket_setting_defaults_and_accepts_an_explicit_endpoint() {
    for setting in [None, Some(String::new())] {
        assert_eq!(PushClient::configured(setting).socket, SOCKET);
    }
    assert_eq!(PushClient::configured(Some("@isolated-push".into())).socket, "@isolated-push");
}

#[test]
fn socket_endpoint_limits_count_bytes_and_distinguish_abstract_names() {
    for socket in ["x".repeat(107), format!("@{}", "x".repeat(107)), "\u{00e9}".repeat(53)] {
        assert!(PushClient::at(socket, Duration::from_secs(1)).is_ok());
    }
    for (socket, bytes) in [("x".repeat(108), 108),
        (format!("@{}", "x".repeat(108)), 108), ("\u{00e9}".repeat(54), 108)] {
        let error = validate_socket_endpoint(&socket).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        let cause = error.get_ref().unwrap().downcast_ref::<SocketEndpointError>().unwrap();
        assert_eq!((cause.bytes, cause.limit), (bytes, 107));
        assert!(error.to_string().contains("108 bytes > 107 bytes"));
        assert!(PushClient::at(socket, Duration::from_secs(1)).is_err());
    }
}

#[test]
fn overlong_env_endpoint_fails_before_create_or_join_connects() {
    let client = PushClient::configured(Some("x".repeat(108)));
    let create = match client.create(vec![], vec![]) {
        Ok(_) => panic!("overlong endpoint accepted"),
        Err(error) => error,
    };
    let join = match client.join(JoinArgs {
        group: 1, token: 2, rank: 0, ring_fd: -1, slots: 1, slot_bytes: 4096,
    }) {
        Ok(_) => panic!("overlong endpoint accepted"),
        Err(error) => error,
    };
    for error in [create, join] {
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        let cause = error.get_ref().unwrap().downcast_ref::<SocketEndpointError>().unwrap();
        assert_eq!(cause.setting, "CUTEAFD_PUSH_SOCKET");
        assert!(error.to_string().starts_with("CUTEAFD_PUSH_SOCKET: socket path"));
    }
}

/// A joined rank waits through the group's start (its peers still joining,
/// with no hub events at all) beyond the stream idle deadline; once the group
/// runs, the idle deadline applies again.
#[test]
fn joined_rank_waits_for_the_group_start_then_keeps_the_idle_deadline() {
    let server = Server::new(|conn| {
        assert!(matches!(request(&conn), Request::Join { .. }));
        event(&conn, Event::Joined);
        // Slow peers: nothing for several idle deadlines.
        thread::sleep(Duration::from_millis(900));
        event(&conn, Event::Progress { done: 0, total: 1 });
        // Then the group stalls: the idle deadline must fire.
        thread::sleep(Duration::from_millis(900));
    });
    let client = PushClient::at(server.client.socket.clone(), Duration::from_millis(200)).unwrap();
    let ring = tempfile::tempfile().unwrap();
    let rank = client
        .join(JoinArgs { group: 7, token: 11, rank: 0, ring_fd: ring.as_raw_fd(), slots: 1, slot_bytes: 4096 })
        .unwrap();
    let first = rank.first(Duration::from_secs(5), Duration::from_millis(200)).unwrap();
    assert!(matches!(first, Event::Progress { .. }));
    let started = std::time::Instant::now();
    assert_eq!(rank.next().unwrap_err().kind(), io::ErrorKind::TimedOut);
    assert!(started.elapsed() < Duration::from_millis(800), "the idle deadline applies after the start");
    drop(rank);
    server.finish();
}

#[test]
fn host_envelope_validates_geometry_capacity_and_physical_lifetimes() {
    use crate::placement::planner::{Lifetime, Location, Phase, Refusal};
    assert!(PushHostEnvelope::new(vec![], vec![], 1, 1).is_err());
    assert!(PushHostEnvelope::new(vec![1], vec![], 1, 1).is_err());
    assert!(PushHostEnvelope::new(vec![1], vec![1], 0, 1).is_err());
    assert!(PushHostEnvelope::new(vec![1], vec![1], u64::MAX, 2).is_err());
    assert!(PushHostEnvelope::new(vec![u64::MAX], vec![1], 1, 1).is_err());
    assert!(PushHostEnvelope::with_retained(vec![1], vec![1], vec![], 1, 1).is_err());
    let admitted = PushHostEnvelope::with_retained(vec![40], vec![80], vec![20], 4, 16).unwrap();
    let owners = admitted.allocations("push").unwrap();
    assert_eq!(owners.iter().map(|owner| owner.bytes).sum::<u64>(), admitted.bytes().unwrap());
    let retained = owners.iter().find(|owner| owner.id == "push.0.retained").unwrap();
    assert_eq!(retained.location, Location::HOST);
    assert_eq!(retained.lifetime, Lifetime::RESIDENT);
    assert!(owners.iter().filter(|owner| owner.id != retained.id)
        .all(|owner| owner.location == Location::PINNED
            && owner.lifetime == Lifetime { first: Phase::Load, last: Phase::Load }));
    let actual = PushHostEnvelope::with_retained(vec![41], vec![81], vec![21], 4, 16).unwrap();
    let refusal = admitted.validate_usage(&actual).unwrap_err();
    assert!(matches!(&refusal, Refusal::Capacity { budget, required: 41, available: 40, .. }
        if budget == "coordinator push expert host rank 0"));
    assert_eq!(refusal, admitted.validate_usage(&actual).unwrap_err());
    assert!(admitted.allocations("").is_err());
}
