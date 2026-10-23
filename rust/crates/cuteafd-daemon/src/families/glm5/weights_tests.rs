//! GLM 5.3 coordinator operands on a patterned synthetic checkpoint, through
//! the fake native DSO with a two-GPU head split. Every operand's bytes are
//! pinned to a golden digest taken from the loader before the landing
//! refactor, so the refactor (and push landings built on it) must reproduce
//! the mount load bit for bit. `w_ik` is excluded: the fake DSO does not
//! implement the FP8 dequantization that fills it.
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::plan::testing as fixture;
use sha2::{Digest, Sha256};
use std::os::unix::fs::FileExt;

/// Deterministic, finite payloads: FP32 grids are positive normals, BF16 and
/// E4M3 bytes vary with the tensor and position.
fn fill(root: &std::path::Path) -> Result<()> {
    let catalog = cuteafd_loader::read_expert_catalog(root)?;
    for tensor in catalog.tensors() {
        let meta = &tensor.metadata;
        let seed = Sha256::digest(meta.name.as_bytes());
        let bytes: Vec<u8> = match meta.dtype {
            DType::F32 => (0..meta.byte_length / 4).flat_map(|i| {
                let v = 0.5 + ((i as u32).wrapping_mul(2654435761) ^ u32::from(seed[0])) as f32 / u32::MAX as f32;
                v.to_le_bytes()
            }).collect(),
            DType::Bf16 => (0..meta.byte_length / 2).flat_map(|i| {
                let v = (((i as u32).wrapping_mul(40503) ^ u32::from(seed[1])) % 2000) as f32 / 1000.0 - 1.0;
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            }).collect(),
            _ => (0..meta.byte_length).map(|i| (i as u8).wrapping_mul(31) ^ seed[(i % 32) as usize] & 0x77).collect(),
        };
        std::fs::File::options().write(true).open(root.join(&tensor.shard))?
            .write_all_at(&bytes, meta.byte_offset)?;
    }
    Ok(())
}

fn digests(library: &NativeLibrary, layers: &[Vec<GlmLayer<'_>>], norm: &DeviceAllocation<'_>,
    head: &DeviceAllocation<'_>) -> Result<std::collections::BTreeMap<String, String>> {
    let mut out = std::collections::BTreeMap::new();
    let mut put = |key: String, buffer: CuteafdDeviceBuffer| -> Result<()> {
        let mut bytes = vec![0u8; buffer.bytes];
        library.copy_d2h(&mut bytes, buffer)?;
        out.insert(key, format!("{:x}", Sha256::digest(&bytes)));
        Ok(())
    };
    for (rank, layers) in layers.iter().enumerate() {
        for (index, layer) in layers.iter().enumerate() {
            for (name, allocation) in &layer.operands {
                if *name == "w_ik" { continue; }
                put(format!("rank{rank}.layer{index}.{name}"), allocation.buffer)?;
            }
        }
    }
    put("norm".into(), norm.buffer)?;
    put("head".into(), head.buffer)?;
    Ok(out)
}

/// Set CUTEAFD_WRITE_GLM5_OPERAND_GOLDEN=1 to regenerate (only from a loader
/// already known to be correct).
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn head_split_operands_match_the_mount_golden() -> Result<()> {
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture_dso.directory())?;
    let tensors = fixture::glm5_tensors(|name, n, k| (0..256).flat_map(|expert|
        fixture::fp8(&name.replace(".experts.0.", &format!(".experts.{expert}.")), n, k, None)).collect());
    fixture::write_snapshot(root.path(), &fixture::glm5_config(), &tensors, None);
    fill(root.path())?;
    let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let stream = std::ptr::null_mut();
    let loader = GlmLoader { library: &library, catalog: &catalog, stream, device: 0,
        peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }], push: None };
    let (model, shares) = loader.model(&cfg, cfg.layers, None, false)?;
    let layers: Vec<Vec<GlmLayer<'_>>> = std::iter::once(model.layers).chain(shares).collect();
    let actual = digests(&library, &layers, &model.norm, model.head.bf16()?)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5/weights_golden.json");
    if std::env::var_os("CUTEAFD_WRITE_GLM5_OPERAND_GOLDEN").is_some() {
        std::fs::write(&path, serde_json::to_vec_pretty(&actual)?)?;
    }
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    assert_eq!(actual.keys().collect::<Vec<_>>(), golden.keys().collect::<Vec<_>>(), "operand inventory changed");
    for (key, digest) in &golden {
        assert_eq!(&actual[key], digest, "{key} differs from the mount golden");
    }
    Ok(())
}

/// The same operands pushed: the loader allocates and records class-2
/// operands, each GPU's plan replays through a ring into them, and the
/// deferred transforms run after delivery. Every operand matches the mount golden.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn pushed_head_split_operands_match_the_mount_golden() -> Result<()> {
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture_dso.directory())?;
    let tensors = fixture::glm5_tensors(|name, n, k| (0..256).flat_map(|expert|
        fixture::fp8(&name.replace(".experts.0.", &format!(".experts.{expert}.")), n, k, None)).collect());
    fixture::write_snapshot(root.path(), &fixture::glm5_config(), &tensors, None);
    fill(root.path())?;
    let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let stream = std::ptr::null_mut();
    let sink = std::rc::Rc::new(super::landing::PushSink::default());
    let loader = GlmLoader { library: &library, catalog: &catalog, stream, device: 0,
        peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }], push: Some(sink.clone()) };
    let (model, shares) = loader.model(&cfg, cfg.layers, None, false)?;
    let (records, deferred) = sink.take();
    let (sparks, world) = (6usize, 8usize);
    for gpu in 0..2 {
        use crate::shared::experts::rtx::dense::{DenseArena, RankLayout};
        let device = crate::shared::memory::device::Device { library: &library, id: gpu };
        let layout = RankLayout::new(records.clone(), Vec::new(), gpu)?;
        let checked = layout.checked(root.path(), world, sparks + gpu as usize)?;
        let expected = crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(&checked, (sparks + gpu as usize) as u32)?;
        device.run(|| -> Result<()> {
            let ring = cuteafd_ffi::push_ring::RegisteredRing::new(&library, 8, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES)?;
            let fake = crate::shared::experts::rtx::dense::testing::FakeRank::new(&checked, root.path(), (sparks + gpu as usize) as u32, &ring)?;
            let arena = DenseArena::new(device, layout, expected)?;
            crate::shared::memory::push_consumer::consume(device, arena, ring, fake, 8)
        })?;
    }
    for job in &deferred {
        crate::shared::memory::device::Device { library: &library, id: job.device }.run(|| job.run(&library, stream))?;
    }
    let layers: Vec<Vec<GlmLayer<'_>>> = std::iter::once(model.layers).chain(shares).collect();
    let actual = digests(&library, &layers, &model.norm, model.head.bf16()?)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5/weights_golden.json");
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    for (key, digest) in &golden {
        assert_eq!(&actual[key], digest, "pushed {key} differs from the mount golden");
    }
    Ok(())
}
