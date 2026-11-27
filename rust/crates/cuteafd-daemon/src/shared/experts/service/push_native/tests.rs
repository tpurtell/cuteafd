use super::*;
use crate::shared::memory::{push_native::LayerPlan, weight_arena::ArenaLayout};
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::push_load::{builder::{Builder, Destination, Source}, native::NativePlanes,
    proto::{Event, Request}, slot, Listener};
use std::{fs::File, os::unix::fs::FileExt, time::Duration};

#[test]
fn native_admission_includes_ring_host_and_transform_scratch() {
    let ring = DEFAULT_SLOTS * DEFAULT_SLOT_BYTES;
    let expected = ring + 1000 + 200 + 300 + 400;
    assert_eq!(admit_peak(1000, 200, 300, 400, expected).unwrap(), expected);
    assert!(admit_peak(1000, 200, 300, 400, expected - 1).is_err());
    assert!(admit_peak(usize::MAX, 200, 300, 400, usize::MAX).is_err());
}

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

// Forward tile/lane oracle, independent of the native fixture's inverse pack map.
fn packed_plane(raw: &[u8], plane: usize, layout: &NativePlanes) -> Vec<u8> {
    if layout.nvfp4 && plane % 2 == 0 { return raw.to_vec(); }
    let mut output = vec![0; raw.len()];
    if layout.nvfp4 {
        let (rows, cols) = if plane == 1 { (2 * layout.kernel_intermediate, layout.hidden / 16) }
            else { (layout.hidden, layout.kernel_intermediate / 16) };
        for row in 0..rows { for col in 0..cols {
            let target = row / 128 * cols * 128 + col / 4 * 512 + row % 32 * 16
                + (row / 32) % 4 * 4 + col % 4;
            output[target] = raw[row * cols + col];
        }}
    } else {
        let scales = plane % 2 == 1;
        let element = if scales { 1 } else { 4 };
        let divisor = if scales { 32 } else { 8 };
        let rows = if plane < 2 { 2 * layout.kernel_intermediate } else { layout.hidden };
        let k = if plane < 2 { layout.hidden } else { layout.intermediate };
        let cols = k / divisor;
        for row in 0..rows { for col in 0..cols {
            let tile = row / 256 * k.div_ceil(128) + col / (128 / divisor);
            let r = row % 256;
            let c = col % (128 / divisor);
            let lane = if scales { r * 4 + c } else {
                (r % 32) / 8 + ((r % 8 * 4 + c % 4) << 2) + ((r / 32) << 7) + ((c / 4) << 10)
            };
            let target = (tile * if scales { 1024 } else { 4096 } + lane) * element;
            let source = (row * cols + col) * element;
            output[target..target + element].copy_from_slice(&raw[source..source + element]);
        }}
    }
    output
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn native_planes_ring_exact_commit_and_failed_transactions_drain_before_mount() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    std::fs::write(fixture.directory().join("events"), b"")?;
    for nvfp4 in [false, true] {
        // Commit, group rejection, source failure, and transform failure.
        for outcome in 0..4 {
            let root = tempfile::tempdir_in(fixture.directory())?;
            let planes = NativePlanes::new(5120, 384, 384, 1, nvfp4)?;
            let layout = ArenaLayout::new([planes.bytes, planes.bytes], planes.bytes * 2 + 255)?
                .with_validation(planes.scalar_bytes() * 2, planes.scalar_bytes() * 2)?;
            let mut builder = Builder::new(root.path(), vec![layout.logical_bytes() as u64]);
            let mut images = Vec::new();
            for region in 0..2 {
                let mut image = (0..planes.bytes).map(|i| (i.wrapping_mul(37) % 251 + 1) as u8).collect::<Vec<_>>();
                if nvfp4 { for (offset, bytes) in planes.vectors { image[offset..offset + bytes].fill(0); } }
                let path = root.path().join(format!("plane{region}"));
                std::fs::write(&path, &image)?;
                builder.contiguous(&Source { path, offset: 0, bytes: image.len() as u64 }, 0, image.len() as u64,
                    Destination { rank: 0, offset: layout.region(region)?.start as u64, bytes: image.len() as u64 })?;
                images.push(image);
                if nvfp4 {
                    let scalars = [0.5f32, 0.5, 0.25, 2., 2., 4.].into_iter()
                        .flat_map(f32::to_le_bytes).collect::<Vec<_>>();
                    let path = root.path().join(format!("scalar{region}"));
                    std::fs::write(&path, &scalars)?;
                    builder.contiguous(&Source { path, offset: 0, bytes: scalars.len() as u64 }, 0, scalars.len() as u64,
                        Destination { rank: 0, offset: (layout.bytes() + region * planes.scalar_bytes()) as u64,
                            bytes: scalars.len() as u64 })?;
                }
            }
            let checked = builder.finish()?;
            let layers = (0..2).map(|layer| LayerPlan { slice: cuteafd_core::SparkSlice {
                unit_map: None, layer: layer as u32, first: 0, width: 384, stored_width: 384,
                tiers: vec![], projection_counts: vec![] }, planes: planes.clone(),
                scalars: layer * planes.scalar_bytes()..(layer + 1) * planes.scalar_bytes() }).collect();
            let plan = RankPlan { layers, layout, placement: vec![], scratch: planes.scratch_bytes(), host_scratch: 256, scales: Default::default() };
            let budgets = [ExpertLoadBudget { resident_bytes: planes.bytes, device_staging_bytes: 0,
                pinned_host_bytes: 0, read_scratch_bytes: 0 }; 2];
            let selections = [ExpertLayer::BackboneFull { layer: 0 }, ExpertLayer::BackboneFull { layer: 1 }];
            let sources = checked.plan.files.iter().map(|name| std::fs::read(root.path().join(name)))
                .collect::<std::io::Result<Vec<_>>>()?;
            let mut packed = Vec::new();
            let mut packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]);
            for entry in checked.plan.entries.iter().rev() {
                let pattern = &checked.plan.patterns[entry.pattern as usize];
                for (src, dst) in pattern.runs().into_iter().rev() {
                    let bytes = &sources[entry.file as usize][(entry.src_base + src) as usize
                        ..(entry.src_base + src + u64::from(pattern.run)) as usize];
                    let mut remaining = bytes;
                    let mut destination = entry.dst_base + dst;
                    while !remaining.is_empty() {
                        let n = packer.push(destination, remaining);
                        if n == 0 { packed.push(packer.finish().0); packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]); }
                        destination += n as u64;
                        remaining = &remaining[n..];
                    }
                }
            }
            if !packer.is_empty() { packed.push(packer.finish().0); }
            let socket = root.path().join("n.sock").to_string_lossy().into_owned();
            let listener = Listener::bind(&socket)?;
            let control = std::net::TcpListener::bind("127.0.0.1:0")?;
            let mut worker = TcpStream::connect(control.local_addr()?)?;
            let (mut coordinator, _) = control.accept()?;
            for stream in [&worker, &coordinator] {
                stream.set_read_timeout(Some(Duration::from_secs(20)))?;
                stream.set_write_timeout(Some(Duration::from_secs(20)))?;
            }
            fixture.configure_pack(&library, i32::from(outcome == 3), 0)?;
            let events_path = fixture.directory().join("events");
            let before = fixture.events()?.len();
            let server_socket = socket.clone();
            let server = std::thread::spawn(move || -> Result<()> {
                let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket, Duration::from_secs(120))?;
                conn.set_timeout(Some(Duration::from_secs(20)))?;
                let (request, fd) = conn.next_request()?.context("native rank did not join")?;
                let Request::Join { slots, slot_bytes, rank, .. } = request else { anyhow::bail!("expected Join") };
                assert_eq!(rank, 0);
                let file = File::from(fd.context("ring fd absent")?);
                conn.send_event(&Event::Joined)?;
                assert_eq!(read_control(&coordinator)?["message"], "joined");
                for (index, bytes) in packed.into_iter().enumerate() {
                    let index = index % slots as usize;
                    file.write_all_at(&bytes, index as u64 * u64::from(slot_bytes))?;
                    conn.send_event(&Event::Filled { slot: index as u32, len: bytes.len() as u32 })?;
                    let (release, _) = conn.next_request()?.context("missing Release")?;
                    assert!(matches!(release, Request::Release { slot } if slot == index as u32));
                    assert!(!std::fs::read_to_string(&events_path)?[before..].contains('P'), "transform before Done");
                    if outcome == 2 { break; }
                }
                if outcome == 2 { conn.send_event(&Event::Failed { reason: "injected failure".into(), missing: vec![] })?; }
                else { conn.send_event(&Event::Done)?; }
                let completed = read_control(&coordinator)?;
                assert_eq!(completed["message"], "completed");
                assert_eq!(completed["success"], outcome < 2);
                write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success":outcome == 0}))?;
                if outcome != 0 {
                    assert_eq!(read_control(&coordinator)?["message"], "drained");
                    let events = std::fs::read_to_string(&events_path)?;
                    assert_eq!(events[before..].matches('D').count(), events[before..].matches('d').count(),
                        "failed candidate still owns native memory at mount barrier");
                    write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
                }
                Ok(())
            });
            let client = PushClient::at(socket, Duration::from_secs(20))?;
            let weights = receive(&library, plan, &checked, &selections, &budgets, 0, &client, &mut worker, 7, 11)?;
            server.join().map_err(|_| anyhow::anyhow!("native fake server panicked"))??;
            if outcome == 0 {
                let mut weights = weights.context("successful native push fell back")?;
                assert_eq!(weights.len(), 2);
                for (weight, image) in weights.iter().zip(&images) {
                    for plane in 0..4 {
                        let (offset, bytes) = planes.planes[plane];
                        let expected = packed_plane(&image[offset..offset + bytes], plane, &planes);
                        let mut actual = vec![0; bytes];
                        library.copy_d2h_quiescent(&mut actual, weight.buffers[plane].buffer)?;
                        assert_eq!(actual, expected, "nvfp4={nvfp4} plane={plane}");
                    }
                    // One-expert vectors leave alignment gaps; production vectors are aligned.
                    let payload = planes.planes.iter().map(|(_, bytes)| bytes).sum::<usize>()
                        + if nvfp4 { planes.vectors.iter().map(|(_, bytes)| bytes).sum::<usize>() } else { 0 };
                    assert_eq!(weight.resident_bytes()?, payload);
                    assert_eq!(weight.budget().resident_bytes, planes.bytes);
                    if nvfp4 {
                        assert_eq!(planes.planes[0].0, 0);
                        let mut actual = vec![0; planes.bytes];
                        library.copy_d2h_quiescent(&mut actual, cuteafd_ffi::CuteafdDeviceBuffer {
                            bytes: planes.bytes, ..weight.buffers[0].buffer
                        })?;
                        for ((offset, bytes), expected) in planes.vectors.into_iter().zip([1.0f32, 1.0, 0.5, 0.25]) {
                            assert_eq!(bytes, 4);
                            assert_eq!(&actual[offset..offset + bytes], &expected.to_le_bytes());
                        }
                    }
                }
                let before_drop = fixture.events()?.len();
                drop(weights.pop());
                assert!(!fixture.events()?[before_drop..].contains('d'), "layer released shared rank arena");
                drop(weights);
                assert_eq!(fixture.events()?[before_drop..].matches('d').count(), 1);
            } else { assert!(weights.is_none()); }
            fixture.configure_pack(&library, 0, 0)?;
        }
    }
    Ok(())
}
