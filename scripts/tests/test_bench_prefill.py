"""The shared prefill panel primes each shape outside the measured interval."""
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "rust/crates/cuteafd-bench/src/panels/speed.rs").read_text()
PREFILL = SOURCE.split("impl Panel for Prefill {", 1)[1].split("impl Retained {", 1)[0]


def test_each_prefill_shape_has_one_fresh_cold_warmup_before_measurement() -> None:
    loop = PREFILL.split("for (i, &target) in lengths.iter().enumerate() {", 1)[1]
    warm = loop.index("let warm = sized_prompt(ctx, 2000 + i as u64, target,")
    timed = loop.index("let text = sized_prompt(ctx, 1000 + i as u64, target,")
    mark = loop.index("let fabric_mark = ctx.client.fabric_mark();")
    cold = loop.index("let cold = ctx.client.chat(")
    assert warm < timed < mark < cold
    request = loop[warm:timed]
    assert request.count("ctx.client.chat(") == 1
    assert "plain(ctx.client, &warm, 1)" in request
    assert "Some(ProbeSpec { cold: true, ..ProbeSpec::default() })" in request
    assert '.with_context(|| format!("{target}-token prefill warm-up"))?;' in request


def test_timed_prefills_bypass_a_cached_shared_template_prefix() -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = (root / "rust/crates/cuteafd-bench/src/baseline.rs").read_text()
    sweep = (root / "rust/crates/cuteafd-bench/src/sweep.rs").read_text()
    for source, start, end in [
        (baseline, 'let chat = client.chat(plain(client, &text.text, 1),', 'let rate = PrefillRate'),
        (sweep, 'let chat = client.chat(plain(client, &timed_text.text, 1),', 'points.push(PrefillPoint'),
    ]:
        timed = source.split(start, 1)[1].split(end, 1)[0]
        assert "ProbeSpec { cold: true," in timed
        assert "t.cached_tokens == 0" in timed

    # Nonces inside user messages cannot distinguish an earlier template preamble.
    template = tuple(range(128))
    warm = template + (1000, 1001)
    measured = template + (2000, 2001)
    common = next(i for i, (a, b) in enumerate(zip(warm, measured)) if a != b)
    assert common == 128
    # The existing engine switch bypasses lookup and retention, rather than
    # disabling the cache or trusting those nonces to make a cold request.
    probe = (root / "rust/crates/cuteafd-daemon/src/shared/probe.rs").read_text()
    cold = probe.split("pub(crate) fn cold(", 1)[1].split("\n}", 1)[0]
    assert "p.spec.cold" in cold
    scheduler = (root / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
    assert "!probe::cold(&" in scheduler


def test_prefill_estimate_includes_the_extra_full_length_request() -> None:
    estimate = PREFILL.split("fn estimate_s", 1)[1].split("fn run", 1)[0]
    assert "rates.seconds(l as f64 * 2.15, 3.0)" in estimate


def test_sized_prompts_always_start_with_a_fresh_nonce() -> None:
    common = (Path(__file__).resolve().parents[2] / "rust/crates/cuteafd-bench/src/panels/common.rs").read_text()
    sized = common.split("pub fn sized_prompt(", 1)[1].split("/// One request", 1)[0]
    assert 'format!("[{}] Read the notes below.' in sized
    assert "nonce(), corpus(seed, words)" in sized
