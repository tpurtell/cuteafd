use super::*;
use anyhow::Context;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::{plan::testing::{qwen4_config, qwen4_exl3, write_snapshot},
    push_load::{builder::Builder, proto::{Event, Request}, slot, Listener}};
use std::{fs::File, os::unix::fs::FileExt, time::Duration};

#[test]
fn independent_ring_peak_rejects_short_memory_validation_and_overflow() -> Result<()> {
    let ring = DEFAULT_SLOTS * DEFAULT_SLOT_BYTES;
    let peak = 4096 + ring + 128 + 1024;
    assert_eq!(admit_peak(4096, 128, 128, 1024, peak)?, peak);
    assert!(admit_peak(4096, 128, 128, 1024, peak - 1).is_err());
    assert!(admit_peak(4096, 128, 127, 1024, peak).is_err());
    assert!(admit_peak(usize::MAX, 0, 0, 0, usize::MAX).is_err());
    assert!(admit_peak(0, usize::MAX, usize::MAX, 0, usize::MAX).is_err());
    assert!(admit_peak(0, 0, 0, usize::MAX, usize::MAX).is_err());
    Ok(())
}

#[test]
fn kimi_k2_and_glm_k4_tp6_admit_ring_independently_from_mount_staging() -> Result<()> {
    use crate::shared::experts::layer::ExpertLayer;
    use cuteafd_loader::plan::testing::{exl3, glm5_config, write_kimi_snapshot};
    let kimi = tempfile::tempdir()?;
    write_kimi_snapshot(kimi.path(), true);
    let glm = tempfile::tempdir()?;
    let tensors = (0..256).flat_map(|expert| [
        ("gate_proj", 2048, 6144), ("up_proj", 2048, 6144), ("down_proj", 6144, 2048),
    ].into_iter().flat_map(move |(projection, n, k)|
        exl3(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k, 4))).collect::<Vec<_>>();
    let mut config = glm5_config();
    config["quantization_config"] = serde_json::json!({"quant_method": "exl3", "bits": 4.0,
        "checkpoint_format": "exl3", "format": "exl3"});
    write_snapshot(glm.path(), &config, &tensors, None);
    for (snapshot, pinned) in [(kimi.path(), 44_826_624), (glm.path(), 114_499_584)] {
        let catalog = cuteafd_loader::read_expert_catalog(snapshot)?;
        for rank in 0..6 {
            let budget = Exl3Weights::plan(&catalog, ExpertLayer::BackboneExl3Tp { layer: 1, rank, world: 6 })?;
            // GLM's balanced TP6 has both 256- and 384-row ranks.
            if rank == 0 { assert_eq!(budget.pinned_host_bytes, pinned); }
            assert_eq!(budget.read_scratch_bytes, 268_435_456);
            let ring = DEFAULT_SLOTS * DEFAULT_SLOT_BYTES;
            assert_eq!(ring, 134_217_728);
            assert!(ring > budget.pinned_host_bytes);
            let plan = RankPlan::new(&catalog, &[1], 6, rank, budget.resident_bytes)?;
            let headroom = usize::try_from(cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES)?;
            let peak = plan.layout.bytes() + ring + plan.validation.bytes() + headroom;
            assert_eq!(admit_peak(plan.layout.bytes(), plan.validation.bytes(), budget.read_scratch_bytes,
                headroom, peak)?, peak);
            assert!(admit_peak(plan.layout.bytes(), plan.validation.bytes(), budget.read_scratch_bytes,
                headroom, peak - 1).is_err());
        }
    }
    Ok(())
}

fn installed_tp6_admission() -> Result<usize> {
    let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_PUSH_EXL3")
        .context("set CUTEAFD_PUSH_EXL3 to the installed EXL3 snapshot")?);
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let shape = catalog.routed_experts();
    let layers: Vec<_> = (shape.first_layer..shape.layers).collect();
    let mut largest_pinned = 0;
    for rank in 0..6 {
        let budgets = layers.iter().map(|&layer| Exl3Weights::plan(&catalog,
            crate::shared::experts::layer::ExpertLayer::BackboneExl3Tp { layer, rank, world: 6 }))
            .collect::<Result<Vec<_>>>()?;
        let resident = budgets.iter().map(|budget| budget.resident_bytes).sum();
        let pinned = budgets.iter().map(|budget| budget.pinned_host_bytes).max().unwrap();
        let scratch = budgets.iter().map(|budget| budget.read_scratch_bytes).max().unwrap();
        let plan = RankPlan::new(&catalog, &layers, 6, rank, resident)?;
        let headroom = usize::try_from(cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES)?;
        let peak = plan.layout.bytes() + DEFAULT_SLOTS * DEFAULT_SLOT_BYTES + plan.validation.bytes() + headroom;
        assert_eq!(admit_peak(plan.layout.bytes(), plan.validation.bytes(), scratch, headroom, peak)?, peak);
        assert!(admit_peak(plan.layout.bytes(), plan.validation.bytes(), scratch, headroom, peak - 1).is_err());
        eprintln!("EXL3 TP6 rank {rank}: resident={resident}, mount_pinned={pinned}, push_ring={}, validation={}, push_peak={peak}",
            DEFAULT_SLOTS * DEFAULT_SLOT_BYTES, plan.validation.bytes());
        largest_pinned = largest_pinned.max(pinned);
    }
    Ok(largest_pinned)
}

#[test]
#[ignore = "requires an installed EXL3 checkpoint selected by CUTEAFD_PUSH_EXL3"]
fn installed_exl3_tp6_admits_ring_independently_from_mount_staging() -> Result<()> {
    installed_tp6_admission()?;
    Ok(())
}

#[test]
#[ignore = "requires installed GLM K4 headers selected by CUTEAFD_PUSH_EXL3"]
fn installed_glm_k4_tp6_admits_ring_larger_than_mount_staging() -> Result<()> {
    let pinned = installed_tp6_admission()?;
    assert_eq!(pinned, 114_499_584, "select the GLM K4 checkpoint headers");
    assert!(pinned < DEFAULT_SLOTS * DEFAULT_SLOT_BYTES);
    Ok(())
}

fn control_pair() -> Result<(TcpStream, TcpStream)> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let client = TcpStream::connect(listener.local_addr()?)?;
    let (worker, _) = listener.accept()?;
    for stream in [&client, &worker] {
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    }
    Ok((client, worker))
}

#[test]
fn namespace_access_failure_declines_all_spark_ranks_before_typed_startup_error() -> Result<()> {
    use crate::shared::experts::rtx::push::PushStartupError;
    let root = tempfile::tempdir()?;
    let (first_driver, mut first) = control_pair()?;
    let (second_driver, mut second) = control_pair()?;
    let driver = std::thread::spawn(move || {
        cuteafd_transport::worker_selection::push::coordinate(&mut [first_driver, second_driver], |_| {
            anyhow::bail!("a declined namespace must not open a push group")
        })
    });
    let peer = std::thread::spawn(move || -> Result<()> {
        let planned = Ok((std::path::PathBuf::from("/fixture"), "spark".into()));
        let error = plan_offer_namespace(std::path::Path::new("/fixture"), &mut second, planned,
            |_, host| Ok(((), Offer { rank: 1, host: host.into(), arena_bytes: 1, placement: vec![] })))
            .unwrap_err();
        assert!(matches!(error.downcast_ref::<PushStartupError>(), Some(PushStartupError::Uncommitted)));
        Ok(())
    });
    let error = plan_offer_namespace::<()>(root.path(), &mut first,
        Err(std::io::Error::from_raw_os_error(libc::EACCES)),
        |_, _| panic!("namespace discovery must fail before planning"))
        .unwrap_err();
    match error.downcast_ref::<PushStartupError>() {
        Some(PushStartupError::Namespace { source, .. }) => assert_eq!(source.raw_os_error(), Some(libc::EACCES)),
        other => panic!("expected typed namespace error, got {other:?}"),
    }
    peer.join().map_err(|_| anyhow::anyhow!("peer panicked"))??;
    driver.join().map_err(|_| anyhow::anyhow!("driver panicked"))??;
    Ok(())
}

#[test]
fn required_receive_reports_failure_only_after_the_peer_drain_barrier() -> Result<()> {
    use crate::shared::experts::rtx::push::PushStartupError;
    let (mut driver, mut worker) = control_pair()?;
    let rank = std::thread::spawn(move || -> Result<()> {
        assert!(!push::completed_with(&mut worker, Err("injected delivery failure".into()))?);
        push::drained(&mut worker)?;
        let error = required_receive::<()>(Ok(None)).unwrap_err();
        assert!(matches!(error.downcast_ref::<PushStartupError>(), Some(PushStartupError::Uncommitted)));
        Ok(())
    });
    let completed: serde_json::Value = read_control(&driver)?;
    assert_eq!(completed["message"], "completed");
    assert_eq!(completed["success"], false);
    write_control(&mut driver, &serde_json::json!({"message":"commit","success":false}))?;
    let drained: serde_json::Value = read_control(&driver)?;
    assert_eq!(drained["message"], "drained");
    write_control(&mut driver, &serde_json::json!({"message":"mount"}))?;
    rank.join().map_err(|_| anyhow::anyhow!("rank panicked"))??;
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn spark_exl3_ring_to_shared_owner_is_mount_exact_and_drops_before_retry() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    std::fs::write(fixture.directory().join("events"), b"")?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    // Nonzero checkpoint data makes a misdirected pitch/rank visible. Rotations
    // remain finite and each projection carries its calibrated MCG prefix.
    for tensor in catalog.tensors().iter().filter(|t| t.metadata.name.contains(".experts.")) {
        let source = Builder::v41_source(&catalog, &tensor.metadata.name)?;
        let image = if tensor.metadata.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
            else if tensor.metadata.name.ends_with(".trellis") {
                (0..source.bytes).map(|i| (i.wrapping_mul(17) >> 3) as u8).collect()
            } else { [0u8, 0x38].into_iter().cycle().take(source.bytes as usize).collect() };
        File::options().write(true).open(source.path)?.write_all_at(&image, source.offset)?;
    }
    for fail in [false, true] {
        let plan = RankPlan::new(&catalog, &[0, 1], 2, 1, usize::MAX)?;
        let budgets = (0..2).map(|layer| Exl3Weights::plan(&catalog,
            crate::shared::experts::layer::ExpertLayer::BackboneExl3Tp { layer, rank: 1, world: 2 }))
            .collect::<Result<Vec<_>>>()?;
        let checked = plan.checked(&catalog, root.path(), 2, 1)?;
        // Independent mount staging reference, not another Pattern walker.
        let mut expected = Vec::new();
        for residency in &plan.residencies {
            let mut image: Vec<Vec<u8>> = residency.buffers.iter().map(|b| vec![0; b.bytes.max(16)]).collect();
            for (index, load) in residency.loads.iter().enumerate() {
                let mut bytes = vec![0; load.bytes];
                let mut scratch = vec![0; residency.scratch_bytes(&catalog)?];
                residency.read_into(&catalog, index, &mut bytes, &mut scratch)?;
                for &(buffer, offset) in &load.destinations {
                    image[buffer][offset..offset + load.bytes].copy_from_slice(&bytes);
                }
            }
            for (buffer, spec) in image.iter_mut().zip(&residency.buffers) {
                for (dst, word) in buffer.chunks_exact_mut(4).zip(&spec.initial_words) {
                    dst.copy_from_slice(&word.to_le_bytes());
                }
            }
            expected.push(image);
        }
        let socket = root.path().join(if fail { "failed.sock" } else { "good.sock" }).to_string_lossy().into_owned();
        let listener = Listener::bind(&socket)?;
        let sources = checked.plan.files.iter().map(|name| std::fs::read(root.path().join(name)))
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut packed = Vec::new();
        let mut packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]);
        for entry in &checked.plan.entries {
            let pattern = &checked.plan.patterns[entry.pattern as usize];
            for (src, dst) in pattern.runs() {
                let mut bytes = &sources[entry.file as usize][(entry.src_base + src) as usize
                    ..(entry.src_base + src + u64::from(pattern.run)) as usize];
                let mut destination = entry.dst_base + dst;
                while !bytes.is_empty() {
                    let n = packer.push(destination, bytes);
                    if n == 0 { packed.push(packer.finish().0); packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]); }
                    destination += n as u64;
                    bytes = &bytes[n..];
                }
            }
        }
        if !packer.is_empty() { packed.push(packer.finish().0); }
        let (mut coordinator, mut worker) = control_pair()?;
        let server_socket = socket.clone();
        let server = std::thread::spawn(move || -> Result<()> {
            let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket, Duration::from_secs(120))?;
            conn.set_timeout(Some(Duration::from_secs(10)))?;
            let (request, fd) = conn.next_request()?.context("rank did not join")?;
            let Request::Join { slots, slot_bytes, rank, .. } = request else { anyhow::bail!("expected rank Join") };
            assert_eq!(rank, 1);
            let file = File::from(fd.context("rank ring fd absent")?);
            conn.send_event(&Event::Joined)?;
            let joined: serde_json::Value = read_control(&coordinator)?;
            assert_eq!(joined["message"], "joined");
            for (index, bytes) in packed.into_iter().enumerate() {
                let index = index % slots as usize;
                file.write_all_at(&bytes, index as u64 * u64::from(slot_bytes))?;
                conn.send_event(&Event::Filled { slot: index as u32, len: bytes.len() as u32 })?;
                let (release, _) = conn.next_request()?.context("missing slot release")?;
                assert!(matches!(release, Request::Release { slot } if slot == index as u32));
            }
            conn.send_event(&Event::Done)?;
            let completed: serde_json::Value = read_control(&coordinator)?;
            assert_eq!(completed["success"], true);
            write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success":!fail}))?;
            if fail {
                let drained: serde_json::Value = read_control(&coordinator)?;
                assert_eq!(drained["message"], "drained");
                write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
            }
            Ok(())
        });
        let client = PushClient::at(socket, Duration::from_secs(10))?;
        let before = fixture.events()?;
        let weights = receive(&library, plan, &checked, &budgets, 1, &client, &mut worker, 7, 11)?;
        server.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked"))??;
        if fail {
            assert!(weights.is_none());
            assert!(fixture.events()?[before.len()..].contains('d'), "retry retained failed arena");
        } else {
            let mut weights = weights.context("successful push fell back")?;
            assert_eq!(weights.len(), 2);
            for (weight, image) in weights.iter().zip(expected) {
                for (spec, expected) in weight.layout.buffers.iter().zip(image) {
                    let mut actual = vec![0; expected.len()];
                    library.copy_d2h(&mut actual, weight.buffer(&spec.name)?)?;
                    assert_eq!(actual, expected, "{}", spec.name);
                }
            }
            assert!(!fixture.events()?[before.len()..].contains('d'), "published views lost rank owner");
            drop(weights.pop());
            assert!(!fixture.events()?[before.len()..].contains('d'), "first layer drop freed shared arena");
            drop(weights);
            assert_eq!(fixture.events()?[before.len()..].matches('d').count(), 1);
        }
    }
    Ok(())
}

fn read_control<T: serde::de::DeserializeOwned>(stream: &TcpStream) -> Result<T> {
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
