"""Forecast arithmetic, timezone and phase separation with tiny receipt fixtures."""
from datetime import datetime, timezone
import json

import pytest

from inkling2.estimate import estimate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_partial_expert_measurements_supply_provisional_calendar_forecast(tmp_path):
    start = datetime(2026,11,13,tzinfo=timezone.utc)
    events = [dict(event='attention_input_capture', time=start.isoformat())]
    for expert in range(7):
        events.append(dict(bits=4, module=f'model.llm.layers.0.mlp.experts.{expert}.down_proj',
            time=datetime.fromtimestamp(start.timestamp()+300+expert*20,timezone.utc).isoformat()))
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    now = datetime.fromtimestamp(start.timestamp()+450, timezone.utc)
    value = estimate(tmp_path,now)
    assert value['seconds_per_expert'] == 20
    assert value['sample_expert_cycles'] == 6
    assert value['seconds_per_layer'] == 384*20+280+180
    assert value['provisional'] is True
    assert all(t.endswith('+08:00') for t in value['completion_time_range'])
    for i in (0,1):
        assert value['remaining_seconds_range'][i] == value['layer_remaining_seconds_range'][i]+sum(
            v[i] for v in value['tail_allowance_seconds'].values())


def test_whole_layer_receipts_replace_partial_rate_model(tmp_path):
    for layer, seconds in enumerate((6000,7200,6600)):
        write(tmp_path/f'layers/layer-{layer:03d}/complete.json',dict(layer=layer,seconds=seconds))
    value = estimate(tmp_path,datetime(2026,11,13,tzinfo=timezone.utc))
    assert value['basis'] == 'mean of recent completed layers'
    assert value['seconds_per_layer'] == 6600
    assert value['layer_remaining_seconds_range'] == pytest.approx([45*6600*.9,45*6600*1.15])


def test_completed_numerics_remove_layer_head_mtp_time_but_keep_handoff(tmp_path):
    write(tmp_path/'quantization-complete.json',dict(status='passed'))
    value=estimate(tmp_path)
    assert value['layer_remaining_seconds_range'] == [0,0]
    assert value['tail_allowance_seconds']['head_mtp'] == [0,0]
    assert value['remaining_seconds_range'][1] > 0


def test_finished_cleanup_forecast_is_zero(tmp_path):
    for name in ('quantization','publication','handoff','handoff-cleanup'):
        write(tmp_path/f'{name}-complete.json',dict(status='passed'))
    value=estimate(tmp_path)
    assert value['remaining_seconds_range'] == [0,0]
    assert not value['provisional']


def test_missing_measurements_and_blockers_are_explicit(tmp_path):
    write(tmp_path/'failure.json',dict(status='failed'))
    value=estimate(tmp_path)
    assert value['remaining_seconds_range'] is None
    assert value['blocked']


def test_io_recovery_uses_new_cycles_and_excludes_partial_restart_layer(tmp_path):
    start=datetime(2026,11,13,tzinfo=timezone.utc)
    events=[dict(event='attention_input_capture',time=start.isoformat())]
    for expert in range(6):
        events.append(dict(bits=4,module=f'model.llm.layers.0.mlp.experts.{expert}.down_proj',
            time=datetime.fromtimestamp(start.timestamp()+300+expert*100,timezone.utc).isoformat()))
    epoch=datetime.fromtimestamp(start.timestamp()+1200,timezone.utc)
    write(tmp_path/'io-recovery.json',dict(resumed_at=epoch.isoformat(),resumed_layer=0))
    for expert in range(6,13):
        events.append(dict(bits=4,module=f'model.llm.layers.0.mlp.experts.{expert}.down_proj',
            time=datetime.fromtimestamp(epoch.timestamp()+(expert-6)*10,timezone.utc).isoformat()))
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    write(tmp_path/'layers/layer-000/complete.json',dict(layer=0,seconds=100))
    value=estimate(tmp_path,epoch)
    assert value['seconds_per_expert']==10
    assert value['mixed_layer_excluded']==0
    assert value['unfinished_expert_sets']==48*384-13
    assert value['layer_remaining_seconds_estimate']==(48*384-13)*10+46*290+47*180
    write(tmp_path/'layers/layer-001/complete.json',dict(layer=1,seconds=7200))
    value=estimate(tmp_path,epoch)
    assert value['basis']=='mean of recent completed layers'
    assert value['sample_layers']==1
    assert value['seconds_per_layer']==7200
