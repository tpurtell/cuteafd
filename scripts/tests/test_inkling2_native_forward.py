"""Native family hooks must retain shared engine ownership boundaries."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT/'rust/crates/cuteafd-daemon/src/families/inkling2'


def test_native_golden_uses_shared_executor_and_native_expert_dispatch():
    prefill = (FAMILY/'prefill.rs').read_text()
    assert "impl<'a, O: UnitBuffers<'a>> PrefillHooks for Hooks" in prefill
    assert 'run_prefill_units(' in prefill
    assert 'FfnRoute::Local' in prefill
    layer = (FAMILY/'layers.rs').read_text()
    assert 'experts: &dyn Dispatcher' in layer
    assert 'DraftProjections::load_with_output' in layer
    attention = layer.split("pub(crate) fn attention<'a>(")[1].split('/// Complete layer')[0]
    assert 'cuda_stream_synchronize' not in attention and '.drain(' not in attention
    golden = (FAMILY/'golden.rs').read_text()
    assert 'super::prefill::layer(' in golden
    assert 'Mxfp8::load(' in golden
    assert golden.index('set_expert_geometry(geometry)') < golden.index('experts::exl3::regions(')
    assert 'NativeLibrary::load_for_experts(&args.native_lib, geometry)' in golden
    assert 'rms<=0.02 && cosine>=0.999' in golden
    assert '(nll-reference_nll).abs() <= 0.03' in golden
    assert 'python' not in golden.lower()


def test_persistent_layer_state_is_borrowed_and_metadata_uses_retained_uploads():
    owners = (FAMILY/'owners.rs').read_text()
    cache = (FAMILY/'cache.rs').read_text()
    assert "impl<'a> UnitBuffers<'a> for LayerUnit" in owners
    assert 'self.cache.field(self.sequence, self.layer, name)' in owners
    assert 'self.cache.global(self.layer)' in owners
    assert 'cache.require_reader(&self.stream)' in owners
    assert 'RetainedUploads' in owners and 'uploads.upload(' in owners
    assert 'copy_h2d(' not in owners
    assert 'reader.raw == stream.raw' in cache
    assert 'self.copies.copy_d2d(' in cache
    assert 'fatal_drain(self.copies.drain()' in cache
    layer = (FAMILY/'layers.rs').read_text()
    assert "owner: &impl UnitBuffers<'a>" in layer
    # Binding may allocate admitted scratch; per-request math may not.
    arithmetic = layer.split('    pub fn project_target(')[1].split('#[cfg(test)]')[0]
    assert 'Allocation::new' not in arithmetic
    head = (FAMILY/'head.rs').read_text()
    assert 'self.rows <= 128' in head
    assert 'probe::head_spans(rows, self.rows)' in head
    assert 'scoring.record(' in head
    assert 'ScoreLogits::Recorded' in head
    embedding = head.split('pub fn embed_unit(')[1].split('pub fn project(')[0]
    assert 'uploads.upload(' in embedding
    assert 'owner.buffer("first")?' in embedding
    assert 'owner.buffer("projection_input")?' in embedding
    assert '.drain(' not in embedding and 'copy_h2d(' not in embedding
    prefill = (FAMILY/'prefill.rs').read_text()
    assert 'for member in members(unit)' in prefill
    assert 'owner.packed_metadata(' in prefill
    assert 'layers::begin_layer(self.kernels, &state, self.target, self.cfg, layer, rows)' in prefill
    assert 'layers::join_experts(self.kernels, &state, self.target, self.cfg, layer, rows)' in prefill
    assert 'unit.reply_origins()' in prefill
    local = prefill.split('fn local_ffn(')[1].split('fn send_experts(')[0]
    assert local.index('self.prepare_routes(layer, unit)?') < local.index('layers::dispatch_experts(owner,')
    assert local.index('layers::dispatch_experts(owner,') < local.index('self.join(layer, unit)')
    assert 'impl<\'a> UnitIo<\'a> for TokenIo' in prefill
    assert 'self.head.embed_unit(' in prefill
    assert 'probe::head_spans(rows, 64)' in prefill
    assert 'projected_spans(unit.rows.len(), &members, self.head_rows)' in prefill
    assert '(self.on_logits)(member.sequence,' in prefill


def test_arenas_are_single_backings_and_head_is_shared():
    weights = (FAMILY/'weights.rs').read_text()
    experts = (FAMILY/'experts.rs').read_text()
    assert 'DeviceWeightArena::new' in weights and 'DenseLoad::mount' in weights
    assert 'DeviceWeightArena::new' in experts and 'DenseLoad::mount' in experts
    assert 'dequant' not in experts.lower()
    resident = experts.split('pub fn from_pushed(')[1].split('/// All admitted layers')[0]
    assert 'Whole::from_pushed(' in resident and 'Exl3Tp2::pushed_whole_weights(' in resident
    assert 'backend.workspace_bytes() == workspace' in resident
    assert 'Whole::load(' not in resident and 'load_whole_weights(' not in resident
    push = weights.split('pub fn record_push(')[1].split('pub fn operand(')[0]
    assert 'Finish::PaddedTranspose' in push
    assert 'CoordinatorPush::prepare_view(' in weights
    assert 'PushPolicy::Required.committed(received)' in weights
    assert 'load_while_admitting(Some(pending)' in weights
    assert 'dense.operand(' in push and 'dense.finish(' in push
    assert 'native_rect(checkpoint, name, layout.rank)?' in push
    rectangle = weights.split('fn native_rect(')[1].split("impl<'a> TargetWeights")[0]
    assert 'NativeSource::new(checkpoint, view, rank)?' in rectangle
    assert 'Rect::tensor(checkpoint, source.name, Some(source.rows), Some(source.columns))?' in rectangle
    assert 'transpose_relative(' not in push and '.read(' not in push and 'copy_h2d(' not in push
    assert 'transpose_relative' in weights
    assert 'padding_zero_exact = true' in weights
    head = (FAMILY/'head.rs').read_text()
    assert 'target_head::launch(' in head
    assert 'Self::with_local(stream, cfg, rows, stored_head, cfg.unpadded_vocab_size)' in head
    assert 'split.handoff.copy_columns(' in head
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
    assert 'admission::reconcile_native(' in golden
    assert 'devices, &receipts)' in golden
    assert 'measured == admitted' in golden
    assert 'measured <= admitted' not in golden


def test_reference_full_depth_metadata_and_no_stage_hooks_in_arithmetic():
    import ast
    path=ROOT/'scripts/diagnostics/inkling2-prefix-reference.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='is_partial')
    scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    assert scope['is_partial'](0,48)
    assert scope['is_partial'](35,48)
    assert not scope['is_partial'](47,48)
    assert 'partial=is_partial(args.last_layer,cfg.num_hidden_layers)' in path.read_text()
    assert 'owner.tap(' not in (FAMILY/'layers.rs').read_text()
    assert 'diagnostics' not in (FAMILY/'owners.rs').read_text()
