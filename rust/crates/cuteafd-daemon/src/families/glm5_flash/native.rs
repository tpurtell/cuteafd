//! Flash adapter facts; execution and storage remain in the shared native backend.
use super::{engine, fp8, EngineArgs, GlmNextConfig};
use crate::shared::experts::exlr8::dense_weights::NativeDenseRecords;
use crate::shared::token_io::native::VocabularyPlan;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::plan::checkpoint::CheckpointFormat;
use cuteafd_loader::plan::exlr8::{self, dense::{DensePlan, DenseRecord, DenseSelection},
    native::{DenseFacts, FlashCharges}, residency::MetadataInput};
use std::collections::BTreeMap;

#[derive(Debug, thiserror::Error)]
#[error("native Flash EXLR8 prefill is A16 (requested {0:?})")]
pub(super) struct PrefillActivationsError(String);

pub(super) fn prefill_activations(value: &str)
    -> Result<exlr8::PrefillActivations, PrefillActivationsError> {
    match value {
        "a16" => Ok(exlr8::PrefillActivations::A16),
        _ => Err(PrefillActivationsError(value.into())),
    }
}

pub(super) fn normalize(args: &mut EngineArgs) {
    args.kda_fp8 = fp8::KdaFp8::Off;
    args.fp8_head = false;
    args.kda_fp32_partials = false;
    args.kda_output_shard = false;
    args.kda_prefill_expanded = false;
    args.kda_nvfp4_gate = None;
    args.fp8_prefill.clear();
    args.fp8_prefill.push(super::Fp8PrefillGroup::None);
}

pub(super) fn validate_headers(format: CheckpointFormat, legacy: impl FnOnce() -> Result<()>) -> Result<()> {
    if format == CheckpointFormat::Exlr8 { Ok(()) } else { legacy() }
}

pub(super) fn legacy_embedding_source(native: bool,
    legacy: impl FnOnce() -> Result<crate::shared::token_io::EmbedSource>)
    -> Result<Option<crate::shared::token_io::EmbedSource>> {
    if native { Ok(None) } else { legacy().map(Some) }
}

fn embedding_source(records: &[DenseRecord], selection: &DenseSelection, hidden: usize, vocab: usize)
    -> Result<crate::shared::token_io::EmbedSource> {
    let name = selection.vocabulary_names()[0].clone();
    let record = records.iter().find(|record| record.tensor == name).context("native Flash embedding")?;
    ensure!(record.encoding == exlr8::dense::DenseEncoding::RowFp8 && record.shape == [vocab, hidden],
        "native Flash embedding requires the manifest row-FP8 vocabulary");
    let payload = record.components.iter().find(|c| c.suffix == "fp8").context("native Flash embedding payload")?;
    ensure!(payload.source.bytes == vocab.checked_mul(hidden).context("native Flash embedding size overflow")?,
        "native Flash embedding payload extent");
    Ok(crate::shared::token_io::EmbedSource { path: payload.source.path.clone(), offset: payload.source.offset,
        vocab, hidden, snapshot: None, tensor_name: name })
}

pub(super) fn whole_mla(cfg: &GlmNextConfig, layers: usize, attention: engine::GlmfAttention)
    -> BTreeMap<usize, usize> {
    let mut owner = attention.first_owner as usize;
    cfg.attention[..layers].iter().enumerate().filter_map(|(layer, kind)| {
        if attention.mode != cuteafd_loader::placement::AttentionPlacement::Layers
            || *kind != cuteafd_loader::families::glm5_flash::GlmNextAttention::Mla { return None; }
        let assigned = owner; owner ^= 1; Some((layer, assigned))
    }).collect()
}

pub(super) struct Admission {
    pub facts: DenseFacts,
    pub metadata: MetadataInput,
    pub dense: Vec<NativeDenseRecords>,
    pub charges: FlashCharges,
    pub resident: Vec<cuteafd_loader::families::glm5_flash::resident::GlmfResidentRank>,
}

impl Admission {
    pub(super) fn new(args: &EngineArgs, cfg: &GlmNextConfig, layers: usize, ranks: usize,
        attention: engine::GlmfAttention, index: engine::IndexCache, manifest: &serde_json::Value) -> Result<Self> {
        ensure!(layers == cfg.layers, "native Flash startup requires the complete backbone");
        ensure!(args.kda_state == engine::KdaState::F32, "native Flash requires F32 KDA state programs");
        let metadata = MetadataInput::open(&args.snapshot, args.exlr8_manifest.as_deref()
            .context("native Flash requires --exlr8-manifest k2, k3 or k4")?)?;
        let first = cfg.dense.iter().position(|dense| !dense).context("Flash routed suffix")?;
        let shared = exlr8::shared_expert_resident(&metadata, &(first..layers).collect::<Vec<_>>(), ranks)?;
        let records = DenseRecord::inventory(&args.snapshot)?;
        let config = serde_json::from_slice(&std::fs::read(args.snapshot.join("config.json"))?)?;
        let selection = DenseSelection::from_hf(&config)?;
        let owners = whole_mla(cfg, layers, attention);
        let dense = (0..ranks).map(|rank| NativeDenseRecords::new(&args.snapshot, layers, ranks, rank, &owners))
            .collect::<Result<Vec<_>>>()?;
        let facts = DenseFacts::new(records, selection, layers, shared)?;
        let resident = owners_resident(&facts, &dense.iter().map(|r| r.plan.clone()).collect::<Vec<_>>(), cfg)?;
        let canonical = cuteafd_loader::families::glm5_flash::resident::native_resident(
            &facts.selection, layers, &facts.plans(&owners)?, &facts.shared_bytes)?;
        ensure!(resident.iter().zip(&canonical).all(|(a, b)| a.weights == b.weights && a.embedding == b.embedding),
            "Flash native runtime owners differ from shared planner residency");
        let charges = FlashCharges::from_manifest(&facts, cfg, &owners, args.prefill_rows,
            args.prefill_waves, args.decode_rows, index.into(), manifest)?;
        Ok(Self { facts, metadata, dense, charges, resident })
    }

    pub(super) fn load_shared<'a>(&self, args: &EngineArgs, cfg: &GlmNextConfig,
        library: &'a cuteafd_ffi::NativeLibrary, programs: &cuteafd_ffi::programs::Programs<'_>)
        -> Result<Vec<crate::shared::experts::exlr8::NativeExperts<'a>>> {
        use crate::shared::{experts::exlr8::{NativeExperts, NativePlan}, memory::device::Device};
        let first = cfg.dense.iter().position(|dense| !dense).context("Flash routed suffix")?;
        let devices = std::iter::once(args.device).chain(args.split_device).collect::<Vec<_>>();
        ensure!(devices.len() == self.dense.len(), "native Flash shared owner count");
        devices.into_iter().enumerate().map(|(rank, id)| Device { library, id }.run(|| {
            let plan = NativePlan::new(&args.snapshot, args.exlr8_manifest.as_deref().context("native Flash manifest")?,
                first, cfg.layers, programs, cuteafd_ffi::exlr8::KernelFamily::GlmFlash, self.dense.len(), rank)?;
            ensure!(plan.resident == self.facts.shared_bytes[rank]
                && plan.workspace as u64 == self.charges.ranks[rank].shared_workspace,
                "native Flash shared weights/workspace differ from admission");
            let expected = plan.resident;
            let shared = NativeExperts::load(library, plan)?;
            ensure!(shared.resident_bytes() as u64 == expected, "native Flash uploaded shared bytes differ from admission");
            Ok(shared)
        })).collect()
    }

    pub(super) fn load<'a>(&self, library: &'a cuteafd_ffi::NativeLibrary,
        programs: &'a cuteafd_ffi::programs::Programs<'a>, loader: &super::weights::GlmfLoader<'a>,
        cfg: &GlmNextConfig, execution: &cuteafd_loader::placement::LayerExecutionPlan)
        -> Result<(crate::shared::token_io::TokenEmbedding<'a>,
            (super::weights::GlmfWeights<'a>, Vec<Vec<super::weights::GlmfLayer<'a>>>))> {
        use crate::shared::token_io::{native::NativeVocabulary, TokenEmbedding};
        let (_, available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
        let available = usize::try_from(available)?;
        let peak = self.dense.iter().try_fold(0usize, |sum, rank|
            sum.checked_add(rank.plan.read_staging_bytes).context("native Flash loading peak overflow"))?;
        ensure!(peak <= available, "native Flash dense loading peak {peak} exceeds host availability {available}");
        let plan = VocabularyPlan::new(&self.facts.records, &self.facts.selection, self.dense.len(), 0,
            cfg.hidden, cfg.vocab_size)?;
        let vocabulary = std::rc::Rc::new(NativeVocabulary::load(library, programs, plan, available, 0, peak)?
            .context("native Flash rank0 owns vocabulary")?);
        let source = embedding_source(&self.facts.records, &self.facts.selection, cfg.hidden, cfg.vocab_size)?;
        let embedding = TokenEmbedding::row_fp8(library, source, vocabulary.clone())?;
        let model = loader.native_model(cfg, execution, &self.dense, vocabulary)?;
        Ok((embedding, model))
    }
}

/// Disjoint owners: body/norm, shared arenas, and the one vocabulary owner.
fn owners_resident(facts: &DenseFacts, body: &[DensePlan], cfg: &GlmNextConfig)
    -> Result<Vec<cuteafd_loader::families::glm5_flash::resident::GlmfResidentRank>> {
    use cuteafd_loader::families::glm5_flash::resident::GlmfResidentRank;
    ensure!(body.len() == facts.shared_bytes.len(), "native resident rank count");
    body.iter().enumerate().map(|(rank, plan)| {
        ensure!(plan.rank == rank && plan.world == body.len(), "native resident body rank/world");
        ensure!(plan.tensors.iter().all(|t| !facts.selection.vocabulary_names().contains(&t.tensor)),
            "native body must not own vocabulary");
        let vocabulary = VocabularyPlan::new(&facts.records, &facts.selection, body.len(), rank,
            cfg.hidden, cfg.vocab_size)?;
        let mut resident = GlmfResidentRank { embedding: 0, weights: facts.shared_bytes[rank] };
        for (group, _, bytes) in exlr8::dense::resident_native(&facts.selection, facts.layers, plan)?
            .into_iter().chain(vocabulary.resident_groups(&facts.selection, facts.layers)?) {
            let total = if group == "embedding" { &mut resident.embedding } else { &mut resident.weights };
            *total = total.checked_add(bytes).context("native resident owner sum overflow")?;
        }
        Ok(resident)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_headers_never_enter_bf16_or_k6_validation() {
        validate_headers(CheckpointFormat::Exlr8, || panic!("legacy header validator called")).unwrap();
        for format in [CheckpointFormat::HfIndexed, CheckpointFormat::HfSingle] {
            let called = std::cell::Cell::new(false);
            assert!(validate_headers(format, || { called.set(true); anyhow::bail!("legacy validation") }).is_err());
            assert!(called.get());
        }
    }

    #[test]
    fn vocabulary_source_uses_native_manifest_or_legacy_bf16_without_crossing() -> Result<()> {
        use crate::shared::token_io::EmbedSource;
        use exlr8::dense::{DenseComponent, DenseEncoding, DenseExecution, DenseOwnership, DenseRange, ShardSide};
        let root = crate::shared::experts::exlr8::dense_weights::tests::fixture("flash");
        let config = serde_json::from_slice(&std::fs::read(root.path().join("config.json"))?)?;
        let selection = DenseSelection::from_hf(&config)?;
        let name = selection.vocabulary_names()[0].clone();
        let (vocab, hidden) = (256, 8);
        let path = root.path().join("native-vocabulary.bin");
        std::fs::write(&path, vec![17u8; vocab * hidden + 32])?;
        let mut records = vec![DenseRecord { tensor: name.clone(), encoding: DenseEncoding::RowFp8,
            shape: vec![vocab, hidden], side: ShardSide::Output,
            ownership: DenseOwnership::new(DenseExecution::RankZero), channels_per_shard: 128,
            rotation_id: 0, rotation: None, hessian_solved: false, native_blocks: 2,
            components: vec![DenseComponent { suffix: "fp8".into(), dtype: "U8".into(),
                shape: vec![vocab, hidden], source: DenseRange { path: path.clone(), offset: 32,
                    bytes: vocab * hidden }, window: None }] }];
        let source = embedding_source(&records, &selection, hidden, vocab)?;
        assert_eq!((source.path, source.offset, source.snapshot), (path, 32, None));
        assert!(legacy_embedding_source(true, || panic!("native startup attempted BF16 source"))?.is_none());
        let mut meta = cuteafd_loader::SafetensorsTensorMetadata { name, dtype: cuteafd_core::DType::Bf16,
            shape: vec![vocab, hidden], byte_offset: 64, byte_length: (vocab * hidden * 2) as u64 };
        let legacy = || EmbedSource::new(root.path(), "legacy.safetensors", &meta, hidden);
        let source = legacy_embedding_source(false, legacy)?.context("legacy embedding")?;
        assert_eq!((source.path, source.offset, source.snapshot),
            (root.path().join("legacy.safetensors"), 64, Some(root.path().to_owned())));
        meta.dtype = cuteafd_core::DType::U8;
        assert!(legacy_embedding_source(false, ||
            EmbedSource::new(root.path(), "legacy.safetensors", &meta, hidden)).is_err());
        records[0].encoding = DenseEncoding::Mxfp8;
        assert!(embedding_source(&records, &selection, hidden, vocab).is_err());
        assert!(embedding_source(&[], &selection, hidden, vocab).is_err());
        records[0].encoding = DenseEncoding::RowFp8;
        records[0].components.clear();
        assert!(embedding_source(&records, &selection, hidden, vocab).is_err());
        Ok(())
    }

    #[test]
    fn native_resident_owners_count_body_shared_and_vocabulary_once() -> Result<()> {
        let root = crate::shared::experts::exlr8::dense_weights::tests::fixture("flash");
        let cfg = GlmNextConfig::read(root.path())?;
        let records = DenseRecord::inventory(root.path())?;
        let config = serde_json::from_slice(&std::fs::read(root.path().join("config.json"))?)?;
        let selection = DenseSelection::from_hf(&config)?;
        // Use real row-FP8 layouts without payload: this is an ownership test.
        let mut records = records;
        for name in selection.vocabulary_names() {
            use exlr8::dense::{DenseComponent, DenseEncoding, DenseOwnership, DenseExecution, DenseRange, ShardSide};
            records.push(DenseRecord { tensor: name.clone(), encoding: DenseEncoding::RowFp8,
                shape: vec![cfg.vocab_size, cfg.hidden], side: ShardSide::Output,
                ownership: DenseOwnership::new(DenseExecution::RankZero), channels_per_shard: 128,
                rotation_id: 0, rotation: None, hessian_solved: false, native_blocks: cfg.vocab_size.div_ceil(128),
                components: vec![
                    DenseComponent { suffix: "fp8".into(), dtype: "U8".into(), shape: vec![cfg.vocab_size, cfg.hidden],
                        source: DenseRange { path: "not-read".into(), offset: 0, bytes: cfg.vocab_size * cfg.hidden }, window: None },
                    DenseComponent { suffix: "scale".into(), dtype: "F32".into(), shape: vec![cfg.vocab_size],
                        source: DenseRange { path: "not-read".into(), offset: 0, bytes: cfg.vocab_size * 4 }, window: None }],
            });
        }
        for world in [1, 2] {
            let shared = (0..world).map(|rank| 123456 + rank as u64 * 256).collect::<Vec<_>>();
            let facts = DenseFacts::new(records.clone(), selection.clone(), cfg.layers, shared.clone())?;
            let body = (0..world).map(|rank| NativeDenseRecords::new(root.path(), cfg.layers, world, rank, &BTreeMap::new())
                .map(|r| r.plan)).collect::<Result<Vec<_>>>()?;
            let actual = owners_resident(&facts, &body, &cfg)?;
            let canonical = cuteafd_loader::families::glm5_flash::resident::native_resident(
                &selection, cfg.layers, &facts.plans(&BTreeMap::new())?, &shared)?;
            for rank in 0..world {
                let vocabulary = VocabularyPlan::new(&facts.records, &selection, world, rank, cfg.hidden, cfg.vocab_size)?;
                assert_eq!(actual[rank].weights + actual[rank].embedding,
                    body[rank].resident_bytes as u64 + shared[rank] + vocabulary.resident_bytes() as u64);
                assert_eq!((actual[rank].weights, actual[rank].embedding), (canonical[rank].weights, canonical[rank].embedding));
                assert_eq!(vocabulary.resident_bytes() > 0, rank == 0);
            }
            let full = facts.plans(&BTreeMap::new())?;
            assert!(owners_resident(&facts, &full, &cfg).is_err(), "vocabulary in body would be double counted");
        }
        Ok(())
    }
}
