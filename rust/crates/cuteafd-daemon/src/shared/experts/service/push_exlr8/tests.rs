use super::*;
use anyhow::Context;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::{
    push_load::{
        proto::{Event, Request},
        slot, Listener,
    },
};
use std::{fs::File, os::unix::fs::FileExt, time::Duration};

fn read_control(stream: &TcpStream) -> Result<serde_json::Value> {
    use std::io::{BufRead, BufReader};
    let mut line = String::new();
    BufReader::with_capacity(1, stream).read_line(&mut line)?;
    Ok(serde_json::from_str(&line)?)
}
fn write_control(stream: &mut TcpStream, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    serde_json::to_writer(&mut *stream, value)?;
    stream.write_all(b"\n")?;
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn exlr8_ring_commit_corruption_and_failure_drain_before_mount() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    std::fs::write(fixture.directory().join("events"), b"")?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    cuteafd_loader::plan::testing::exlr8::fixture(root.path(), false);
    // Success, all-rank commit rejection, source failure, invalid sign structure.
    for outcome in 0..4 {
        let plan = RankPlan::new(root.path(), "mixed", &[3], 4, 3, usize::MAX)?;
        let budgets = plan.residencies.iter().map(Exlr8Weights::plan).collect::<Result<Vec<_>>>()?;
        let checked = plan.checked(root.path(), root.path(), 4, 3)?;
        let corrupt_destination = plan.residencies[0].signs_offsets[0] as u64;
        let sources = checked
            .plan
            .files
            .iter()
            .map(|name| std::fs::read(root.path().join(name)))
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut packed = Vec::new();
        let mut packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]);
        for entry in checked.plan.entries.iter().rev() {
            let pattern = &checked.plan.patterns[entry.pattern as usize];
            for (src, dst) in pattern.runs().into_iter().rev() {
                let mut bytes = sources[entry.file as usize][(entry.src_base + src) as usize
                    ..(entry.src_base + src + u64::from(pattern.run)) as usize]
                    .to_vec();
                if outcome == 3 && entry.dst_base + dst == corrupt_destination {
                    bytes[0] = 0;
                }
                let mut bytes = bytes.as_slice();
                let mut destination = entry.dst_base + dst;
                while !bytes.is_empty() {
                    let n = packer.push(destination, bytes);
                    if n == 0 {
                        packed.push(packer.finish().0);
                        packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]);
                    }
                    destination += n as u64;
                    bytes = &bytes[n..];
                }
            }
        }
        if !packer.is_empty() {
            packed.push(packer.finish().0);
        }
        let socket = root
            .path()
            .join(format!("native-{outcome}.sock"))
            .to_string_lossy()
            .into_owned();
        let listener = Listener::bind(&socket)?;
        let control = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut worker = TcpStream::connect(control.local_addr()?)?;
        let (mut coordinator, _) = control.accept()?;
        for stream in [&worker, &coordinator] {
            stream.set_read_timeout(Some(Duration::from_secs(10)))?;
            stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        }
        let events_path = fixture.directory().join("events");
        let before = fixture.events()?.len();
        let server_socket = socket.clone();
        let server = std::thread::spawn(move || -> Result<()> {
            let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket, Duration::from_secs(120))?;
            conn.set_timeout(Some(Duration::from_secs(10)))?;
            let (request, fd) = conn.next_request()?.context("native rank did not join")?;
            let Request::Join {
                slots,
                slot_bytes,
                rank,
                ..
            } = request
            else {
                anyhow::bail!("expected Join")
            };
            assert_eq!(rank, 3);
            let file = File::from(fd.context("native rank ring fd absent")?);
            conn.send_event(&Event::Joined)?;
            assert_eq!(read_control(&coordinator)?["message"], "joined");
            for (index, bytes) in packed.into_iter().enumerate() {
                let index = index % slots as usize;
                file.write_all_at(&bytes, index as u64 * u64::from(slot_bytes))?;
                conn.send_event(&Event::Filled {
                    slot: index as u32,
                    len: bytes.len() as u32,
                })?;
                let (release, _) = conn.next_request()?.context("missing native release")?;
                assert!(matches!(release, Request::Release { slot } if slot == index as u32));
                if outcome == 2 {
                    break;
                }
            }
            if outcome == 2 {
                conn.send_event(&Event::Failed {
                    reason: "injected source failure".into(),
                    missing: vec![],
                })?;
            } else {
                conn.send_event(&Event::Done)?;
            }
            let completed = read_control(&coordinator)?;
            assert_eq!(completed["message"], "completed");
            assert_eq!(completed["success"], outcome < 2);
            write_control(
                &mut coordinator,
                &serde_json::json!({"message":"commit", "success":outcome == 0}),
            )?;
            if outcome != 0 {
                assert_eq!(read_control(&coordinator)?["message"], "drained");
                // The retry allocation is still forbidden while the barrier is held.
                let events = std::fs::read_to_string(&events_path)?;
                assert_eq!(
                    events[before..].matches('d').count(),
                    1,
                    "Drained retained native arena"
                );
                write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
            }
            Ok(())
        });
        let client = PushClient::at(socket, Duration::from_secs(10))?;
        let layers = receive(&library, plan, &checked, &budgets, 3, &client, &mut worker, 7, 11)?;
        server
            .join()
            .map_err(|_| anyhow::anyhow!("fake native push server panicked"))??;
        if outcome == 0 {
            let layers = layers.context("successful EXLR8 push fell back")?;
            assert_eq!(layers.len(), 1);
            let pushed = &layers[0];
            let mount = Exlr8Weights::load(&library, pushed.layout.clone(), usize::MAX)?;
            // Relocated descriptor addresses differ, all source data and bits agree.
            for unit in &pushed.layout.loads {
                for (&destination, source) in unit.destination.iter().zip(&unit.source) {
                    let mut a = vec![0; source.bytes];
                    let mut b = vec![0; source.bytes];
                    library.copy_d2h_quiescent(&mut a, crate::shared::experts::exlr8::view(
                        pushed.image(), destination, source.bytes)?)?;
                    library.copy_d2h_quiescent(&mut b, crate::shared::experts::exlr8::view(
                        mount.image(), destination, source.bytes)?)?;
                    assert_eq!(a, b);
                }
            }
            let mut actual = vec![0; pushed.pointers()?.bytes];
            library.copy_d2h_quiescent(&mut actual, pushed.pointers()?)?;
            let expected = pushed.layout.descriptors(pushed.image().ptr as u64)?.0
                .into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>();
            assert_eq!(actual, expected);
            drop(mount);
            let after_mount = fixture.events()?.len();
            drop(layers);
            assert_eq!(fixture.events()?[after_mount..].matches('d').count(), 1);
        } else {
            assert!(layers.is_none());
        }
    }
    Ok(())
}
