"""Native family hooks must retain shared engine ownership boundaries."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT/'rust/crates/cuteafd-daemon/src/families/inkling2'


def test_native_golden_uses_shared_executor_and_native_expert_dispatch():
    prefill = (FAMILY/'prefill.rs').read_text()
    assert 'impl PrefillHooks for Hooks' in prefill
    assert 'run_prefill_units(' in prefill
    assert 'FfnRoute::Local' in prefill
    layer = (FAMILY/'layers.rs').read_text()
    assert 'experts: &dyn Dispatcher' in layer
    assert 'DraftProjections::load_with_output' in layer
    attention = layer.split('pub(crate) fn attention(')[1].split('/// Complete layer')[0]
    assert 'cuda_stream_synchronize' not in attention and '.drain(' not in attention
    golden = (FAMILY/'golden.rs').read_text()
    assert 'super::prefill::layer(' in golden
    assert 'Mxfp8::load(' in golden
    assert 'rms<=0.02 && cosine>=0.999' in golden
    assert '(nll-reference_nll).abs() <= 0.03' in golden
    assert 'python' not in golden.lower()


def test_arenas_are_single_backings_and_head_is_shared():
    weights = (FAMILY/'weights.rs').read_text()
    experts = (FAMILY/'experts.rs').read_text()
    assert 'DeviceWeightArena::new' in weights and 'DenseLoad::mount' in weights
    assert 'DeviceWeightArena::new' in experts and 'DenseLoad::mount' in experts
    assert 'dequant' not in experts.lower()
    assert 'transpose_relative' in weights
    assert 'padding_zero_exact = true' in weights
    head = (FAMILY/'head.rs').read_text()
    assert 'target_head::launch(' in head
    assert 'cfg.unpadded_vocab_size as u32' in head
    assert 'cuda_logits_softcap_f32_async' in head
    assert 'drain_or_abort' in head
    assert 'drain_or_abort' in (FAMILY/'owners.rs').read_text()


def test_owner_reconciliation_is_per_id_and_stream_slabs_are_allowances():
    admission = (FAMILY/'admission.rs').read_text()
    assert 'devices.remap_plan_weight_arenas(plan)' in admission
    assert 'placement::planner::reconcile(&projected' in admission
    assert 'ComponentKind::Allowance(Allowance' in admission
    assert 'driver.stream-slabs: one ordered cold golden stream' in admission
    golden = (FAMILY/'golden.rs').read_text()
    assert 'admission::reconcile_native(&plan, devices, &receipts)' in golden
    assert 'measured == admitted' in golden
    assert 'measured <= admitted' not in golden
