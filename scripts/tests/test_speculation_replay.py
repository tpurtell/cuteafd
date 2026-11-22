import importlib.util
import math
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'speculation_replay', Path(__file__).resolve().parents[1] / 'bench' / 'speculation-replay.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def row(outcome, score=None, history=None, rnd=0, request=1, context=100, proposed=None, block=7,
        round_us=10_000, c=1, batch_rows=None, batch_requests=1, family='f', content='code'):
    proposed = len(outcome) if proposed is None else proposed
    return dict(family=family, card='card', round=rnd, lane=0, request=request, request_id=None, panel='x',
                content=content, c=c, context=context, batch_requests=batch_requests,
                batch_rows=batch_rows if batch_rows is not None else proposed + 1, block=block, policy='fixed 7',
                room=63, proposed=proposed, accepted=sum(1 for _ in iter(lambda: None, 0)) if False else 0,
                censored=False, score=score or [None] * len(outcome), history=history or [0.5] * len(outcome),
                outcome=outcome, round_us=round_us, verify_us=None, draft_us=None)


def test_accepted_run_stops_at_the_first_rejection():
    assert replay.accepted_run(row([True, True, False, None, None])) == 2
    assert replay.accepted_run(row([False, None])) == 0
    assert replay.accepted_run(row([True] * 7)) == 7


def test_fixed_rules_truncate_and_commit_min_of_length_and_run():
    r = row([True, True, True, False, None, None, None])
    assert replay.rule_fixed(2)(r) == 2 and replay.rule_fixed(7)(r) == 7
    fit = dict(coef=[1000., 100., 0., 0., 0.])
    one = replay.replay([r], fit, replay.rule_fixed(2))
    # 2 drafts accepted + 1 target token over 1000 + 100*3 us
    assert one['tokens'] == 3 and one['tok_s'] == pytest.approx(3 / 1300e-6)
    full = replay.replay([r], fit, replay.rule_fixed(7))
    assert full['tokens'] == 4 and full['tok_s'] == pytest.approx(4 / 1800e-6)


def calibrated(*rows):
    replay.attach_calibration(list(rows))
    return rows


def test_history_rule_keeps_scores_above_the_cut_plus_one_explore():
    r, token_only, head = calibrated(
        dict(row([True] * 7, score=[0.9, 0.8, 0.4, 0.9, 0.9, 0.9, 0.9]), score_source='raw candidate p (cycle features)'),
        dict(row([True] * 3, history=[0.6, 0.2, 0.9], block=3), score_source='history p (position acceptance)'),
        dict(row([True] * 7, score=[0.5, 0.35, 0.2, 0.9, 0.9, 0.9, 0.9], request=2), score_source='head (dSpark/trained, calibrated)'))
    assert replay.rule_history()(r) == 3  # raw 0.5 cut: two kept + explore
    assert replay.rule_history()(token_only) == 2  # history cut 0.3: one kept + explore
    assert replay.rule_history()(head) == 3  # engine-calibrated head, 0.3 cut
    # The calibrated sweep keys on `prob`; a cold Platt is the identity, so 0.85 keeps one.
    assert replay.rule_history(cut=0.85, calibrated=True)(r) == 2


def test_product_rule_uses_the_prefix_product():
    (r,) = calibrated(dict(row([True] * 7, score=[0.9] * 7), score_source='head (dSpark/trained, calibrated)'))
    # 0.9^k > 0.5 for k <= 6 (0.531), fails at 7 (0.478): six kept, explore capped by proposed
    assert replay.rule_product(0.5)(r) == 7
    assert replay.rule_product(0.75)(r) == 3  # 0.81 > 0.75, 0.729 fails: 2 kept + explore


def test_cycle_fit_recovers_a_linear_cost_model():
    rows = []
    for k in range(40):
        n = 1 + k % 8
        drafts = k % 7
        ctx = (0, 8000, 32000, 128000)[k % 4]
        ctx_k = replay.BAND_K[replay.context_band(ctx)]
        t = 5000 + 300 * n * (drafts + 1) + 2 * n * (drafts + 1) * ctx_k + 50 * n
        for i in range(n):
            rows.append(row([True] * drafts + [False], rnd=k, request=i, context=ctx, proposed=drafts,
                            round_us=t, batch_rows=n * (drafts + 1), batch_requests=n))
    fit = replay.fit_linear(rows)
    assert fit['r2'] > 0.999
    assert replay.predict(fit, 16, 2, 32.0) == pytest.approx(5000 + 300 * 16 + 2 * 16 * 32 + 100, rel=1e-3)


def test_measured_table_interpolates_rows_within_a_context_band():
    rows = []
    for k, (n, t) in enumerate([(8, 70_000), (8, 70_000), (8, 72_000), (32, 150_000), (32, 152_000),
                                 (32, 154_000), (64, 230_000), (64, 230_000), (64, 236_000)]):
        rows.append(row([False], rnd=k, context=500, proposed=n - 1, round_us=t, batch_rows=n))
    replay.MIN_POINT_ROUNDS, saved = 3, replay.MIN_POINT_ROUNDS
    try:
        table = replay.fit_table(rows)
    finally:
        replay.MIN_POINT_ROUNDS = saved
    assert table['0'] == {8: 70_000, 32: 152_000, 64: 230_000}
    fit = {'coef': [0.] * 5, 'table': table}
    assert replay.predict(fit, 20, 1, 0.) == pytest.approx(70_000 + 82_000 * 12 / 24)
    assert replay.predict(fit, 64, 8, 0.) == 230_000


class FixedCost:
    def __init__(self, base, per_row):
        self.base, self.per_row = base, per_row

    def predict(self, band, rows):
        return self.base + self.per_row * rows


def test_marginal_rule_stops_when_rows_cost_more_than_they_return():
    (r,) = calibrated(dict(row([True] * 7, score=[0.99] * 7), score_source='head (dSpark/trained, calibrated)'))
    assert replay.rule_marginal([r], FixedCost(10_000., 1.), '0') == [7]
    assert replay.rule_marginal([r], FixedCost(1_000., 1_000_000.), '0') == [0]
    # Scale-free: the ratio model's 1 + r*rows makes the same decisions as base + per_row*rows.
    assert replay.rule_marginal([r], FixedCost(1., 1 / 10_000.), '0') == [7]
    # No cost observed yet: verify the trace's own full block.
    assert replay.rule_marginal([r], replay.OnlineCost(), '0') == [7]


def test_acceptance_table_is_conditional_and_cumulative():
    rows = [row([True, False, None]), row([True, True, False]), row([False, None, None])]
    table = {t['position']: t for t in replay.acceptance_table(rows)}
    assert table[1]['conditional'] == pytest.approx(2 / 3)
    assert table[2]['conditional'] == pytest.approx(1 / 2)
    assert table[2]['cumulative'] == pytest.approx(1 / 3)


def test_traffic_mix_requires_half_the_weight_at_agentic_context():
    with pytest.raises(AssertionError):
        replay.traffic_mix([], {'0': 0.6, '32K': 0.4})


def test_tidy_reads_a_timed_round_record_and_its_context(tmp_path):
    job = tmp_path / 'job'
    (job / 'reports' / 'r').mkdir(parents=True)
    (job / 'job.json').write_text(json.dumps({'card': 'glmf', 'entry': {'family': 'glm5_flash', 'set': {}}}))
    (job / 'reports' / 'r' / 'report.json').write_text(json.dumps({'panels': [
        {'id': 'decode_content', 'content': 'code', 'runs': [
            {'speculation': {'request_id': 'h1'}, 'prompt_tokens': 63}]}]}))
    record = {'type': 'draft_round', 'round': 0, 'policy': 'fixed 7', 'block': 7, 'lane': 0,
              'round_us': 9000, 'draft_us': 1500, 'verify_us': 7000, 'context': [67],
              'requests': [{'id': 1, 'request_id': 'h1', 'room': 63, 'keep': 7, 'neural': 7, 'copy': 0,
                            'proposed': 7, 'accepted_drafts': 2, 'censored': False,
                            'positions': [{'track': 'neural', 'position': k, 'p': 1.0, 'score': 0.9,
                                           'outcome': [True, True, False, None, None, None, None][k]}
                                          for k in range(7)]}]}
    (job / 'speculation-trace.jsonl').write_text(json.dumps(record) + '\n{"truncated":')
    rows = replay.tidy_job(job)
    assert len(rows) == 1
    r = rows[0]
    assert (r['content'], r['context'], r['round_us'], r['verify_us'], r['batch_rows']) == ('code', 67, 9000, 7000, 8)
    assert replay.accepted_run(r) == 2 and replay.full_block(r)


def test_tidy_pairs_shared_scheduler_cycles_with_their_rounds(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'job.json').write_text(json.dumps({'card': 'kimi', 'entry': {'family': 'kimi_k3', 'set': {}}}))
    round_record = {'type': 'draft_round', 'round': 0, 'policy': 'fixed 7', 'block': 7,
                    'requests': [{'id': 1, 'room': 63, 'keep': 7, 'neural': 7, 'copy': 0, 'proposed': 3,
                                  'accepted_drafts': 1, 'censored': False,
                                  'positions': [{'p': 1.0, 'head': 0.7, 'calibrated': 0.7, 'outcome': o}
                                                for o in (True, False, None)]}]}
    cycle = {'ids': [1], 'rows': 4, 'seqs': 1, 'contexts': [[33000, 33001, 33002, 33003]], 'verify_ms': 80.0,
             'draft_ms': 2.0, 'cycle_ms': 90.5}
    (job / 'speculation-trace.jsonl').write_text(json.dumps(round_record) + '\n' + json.dumps(cycle) + '\n')
    r = replay.tidy_job(job)[0]
    assert (r['context'], r['round_us'], r['verify_us'], r['draft_us'], r['score']) == (32999, 90500, 80000, 2000, [0.7, 0.7, 0.7])


def test_settle_grants_the_row_budget_by_next_score_keeping_prefixes():
    a = row([True] * 7, score=[0.9] * 7, request=1)
    b = row([True] * 7, score=[0.99, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], request=2)
    # budget 6 drafts (8 rows - 2 anchors): b's 0.99 first, then a's 0.9 run beats b's 0.2s
    calibrated(dict(a, score_source='head (dSpark/trained, calibrated)'), dict(b, score_source='head (dSpark/trained, calibrated)'))
    a['prob'], b['prob'] = a['score'], b['score']
    assert replay.settle([7, 7], [a, b], max_rows=8) == [5, 1]
    assert replay.settle([2, 3], [a, b], max_rows=64) == [2, 3]


def test_tidy_joins_family_round_features_as_raw_candidate_p(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'job.json').write_text(json.dumps({'card': 'glmf', 'entry': {'family': 'glm5_flash', 'set': {}}}))
    record = {'type': 'draft_round', 'round': 0, 'policy': 'fixed 7', 'block': 7, 'lane': 0, 'round_us': 30000,
              'draft_us': 2000, 'verify_us': 26000, 'context': [67],
              'requests': [{'id': 1, 'room': 63, 'keep': 7, 'neural': 7, 'copy': 0, 'proposed': 2, 'accepted_drafts': 1,
                            'censored': False, 'positions': [{'p': 1.0, 'score': None, 'outcome': o} for o in (True, False)]}]}
    family = {'round': 1, 'request_ids': [1], 'neural_features': [[[1.6, 0.79, 0.68, 0.0], [2.9, 0.94, 0.24, 2.0]]]}
    (job / 'speculation-trace.jsonl').write_text(json.dumps(record) + '\n' + json.dumps(family) + '\n')
    r = replay.tidy_job(job)[0]
    assert (r['score'], r['score_source'], r['round_us']) == ([0.79, 0.94], 'raw candidate p (cycle features)', 30000)


def test_rows_below_a_long_band_take_the_short_curve_shape_scaled_at_its_anchor():
    table = {'0': {1: 10_000, 8: 20_000, 64: 100_000}, '128K': {8: 40_000, 64: 160_000}}
    fit = {'coef': [0.] * 5, 'table': table}
    assert replay.predict(fit, 1, 1, 128.) == pytest.approx(20_000)  # 10k * 40k/20k
    assert replay.predict(fit, 8, 1, 128.) == 40_000
    assert replay.predict(fit, 36, 1, 128.) == pytest.approx(100_000)


def test_platt_mirror_moves_toward_observed_acceptance_and_is_causal():
    platt = replay.Platt()
    assert platt.apply(0.) == pytest.approx(0.5) and platt.apply(2.) == pytest.approx(1 / (1 + math.exp(-2)))
    for _ in range(400):
        platt.observe(replay.logit_of(0.9), False)  # an overconfident raw score
    assert platt.apply(replay.logit_of(0.9)) < 0.2
    rows = [dict(row([False], score=[0.9], rnd=k), score_source='raw candidate p (cycle features)') for k in range(300)]
    replay.attach_calibration(rows)
    assert rows[0]['prob'][0] == pytest.approx(0.9)  # first round: cold, identity, never sees its own outcome
    assert rows[-1]['prob'][0] < 0.2 and rows[-1]['calibration'] == 'online platt (replay)'
    head = dict(row([True], score=[0.42]), score_source='head (dSpark/trained, calibrated)')
    replay.attach_calibration([head])
    assert head['prob'] == [0.42] and head['calibration'] == 'engine'  # never calibrated twice


def test_online_ratio_recovers_per_row_over_base_and_reports_convergence():
    model = replay.OnlineRatio()
    for k in range(40):
        rows = 8 * (1 + k % 8)
        model.observe('32K', rows, 50_000 + 2_000 * rows)
    assert model.ratio('32K') == pytest.approx(0.04)
    conv = model.convergence()['32K']
    assert conv['final_ratio'] == pytest.approx(0.04) and conv['rounds_to_10pct'] <= 3
    assert model.predict('32K', 10) == pytest.approx(1.4)


def test_two_term_cost_separates_per_request_from_per_row_and_ratio_is_scale_free():
    model = replay.OnlineTwoTerm()
    for k in range(60):
        n = 1 + k % 8
        rows = n * (1 + (k // 8) % 8)  # rows vary at fixed requests
        model.observe('32K', rows, 40_000 + 9_000 * n + 1_500 * rows, n)
    a, b, c = model.coef('32K')
    assert (a, b, c) == pytest.approx((40_000, 9_000, 1_500), rel=1e-3)
    assert model.predict('32K', 10, 2) == pytest.approx(58_000 + 15_000)
    ratio = replay.OnlineTwoTerm(ratio=True)
    for k in range(60):
        n = 1 + k % 8
        ratio.observe('32K', n * (1 + (k // 8) % 8), 40_000 + 9_000 * n + 1_500 * n * (1 + (k // 8) % 8), n)
    assert ratio.predict('32K', 10, 2) == pytest.approx(1 + 1_500 / 58_000 * 10)
    (r,) = calibrated(dict(row([True] * 7, score=[0.99] * 7), score_source='head (dSpark/trained, calibrated)'))
    assert replay.rule_marginal([r], model, '32K') == replay.rule_marginal([r], ratio, '32K')
