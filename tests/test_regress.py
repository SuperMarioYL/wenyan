"""m1 kill-gate (b): 助词 particle-strip prototype net-token positive per model.

Asserts that stripping low-load Chinese 助词 saves tokens (net of retry cost) on
each model — the comprehension-degradation falsifier (mvp_plan §5 m1, §8). m1
uses retry_cost=0 (the low-risk prototype assumption); m3 will replace it with
measured retry tokens from a real task-success run.
"""

from __future__ import annotations

import pytest

from wenyan.profiler import DEFAULT_PROMPT_SUITE, count_tokens, load_model_specs, load_tokenizer
from wenyan.regress import (
    NetTokenResult,
    RECORD_META,
    RECORDED_COUNTS,
    REGRESS_PROMPT_SUITE,
    net_tokens,
    record_live_counts,
    regress,
    run_harness,
)
from wenyan.strategies import (
    ALL_STRATEGIES,
    IMPLEMENTED_STRATEGIES,
    PARTICLES,
    apply,
    particle_strip,
)


def _tokenizers_or_skip():
    specs = load_model_specs()
    toks = {}
    for spec in specs:
        try:
            toks[spec.name] = load_tokenizer(spec.repo)
        except Exception as exc:
            pytest.skip(
                f"tokenizer {spec.name} unavailable offline — gate needs real tokenizers "
                f"({type(exc).__name__})"
            )
    return specs, toks


# --- pure-logic unit tests (no network) ---------------------------------------

def test_particle_strip_removes_particles_only():
    text = "请帮我写一个详细的函数"
    out = particle_strip(text)
    for ch in PARTICLES:
        assert ch not in out
    # content characters survive
    assert "请帮我写" in out
    assert "一个" in out
    assert "详细" in out
    assert "函数" in out


def test_particle_strip_is_a_subset_of_chars():
    text = "请把这段中文文档翻译成英文"
    out = particle_strip(text)
    assert len(out) <= len(text)
    assert set(out) <= set(text)


def test_only_zhuci_strip_is_implemented_in_m1():
    assert IMPLEMENTED_STRATEGIES == ("助词_strip",)
    assert set(ALL_STRATEGIES) == {"文言文", "助词_strip", "成语_sub"}


def test_m2_strategies_raise_not_implemented():
    for s in ("文言文", "成语_sub"):
        with pytest.raises(NotImplementedError):
            apply(s, "测试")  # type: ignore[arg-type]


def test_net_tokens_arithmetic():
    assert net_tokens(100, 80, retry_cost=0) == 20
    assert net_tokens(100, 80, retry_cost=25) == -5  # retries cost more than saved
    assert net_tokens(50, 50) == 0


def test_regress_aggregates_per_model():
    res = regress("qwen", baseline_tokens=[10, 20, 30],
                  compressed_tokens=[8, 18, 27], retry_cost_tokens=0)
    assert isinstance(res, NetTokenResult)
    assert res.baseline_tokens == 60
    assert res.compressed_tokens == 53
    assert res.retry_cost_tokens == 0
    assert res.net_saved_tokens == 7
    assert res.net_positive


def test_regress_with_per_prompt_retry_cost():
    res = regress("glm", [10, 20], [8, 18], retry_cost_tokens=[1, 2])
    assert res.baseline_tokens == 30
    assert res.compressed_tokens == 26
    assert res.retry_cost_tokens == 3
    assert res.net_saved_tokens == 1  # 30 - 26 - 3
    assert res.net_positive


# --- the real kill-gate (network; skipped offline) ---------------------------

def test_zhuci_net_token_positive_per_model():
    specs, toks = _tokenizers_or_skip()
    for spec in specs:
        tok = toks[spec.name]
        base = [count_tokens(tok, p) for p in DEFAULT_PROMPT_SUITE]
        comp = [count_tokens(tok, particle_strip(p)) for p in DEFAULT_PROMPT_SUITE]
        res = regress(spec.name, base, comp, retry_cost_tokens=0)
        # mvp_plan §8: 助词 prototype must be net-token positive per model.
        assert res.net_positive, (
            f"{spec.name} net-saved {res.net_saved_tokens} tokens — comprehension "
            f"regression would cost more than saved (kill)"
        )


def test_zhuci_never_inflates_token_count():
    """Stripping a particle can only remove chars ⇒ never increases tokens."""
    specs, toks = _tokenizers_or_skip()
    for spec in specs:
        tok = toks[spec.name]
        for p in DEFAULT_PROMPT_SUITE:
            before = count_tokens(tok, p)
            after = count_tokens(tok, particle_strip(p))
            assert after <= before, f"{spec.name}: stripping inflated tokens ({before}→{after})"


# =========================================================================== #
# m4_harness_ci — net-token regression harness as a reproducible artifact      #
# (mvp_plan §5 m3/m4, §8 post-ship kill-gate net-savings falsifier)            #
#                                                                             #
# Two layers, both committed:                                                 #
#   * OFFLINE deterministic — replays the committed RECORDED_COUNTS fixture   #
#     (real-tokenizer counts recorded once, cached for replay, NOT a heuristic)#
#     so `wenyan harness` / `run_harness` is one-command, no network.          #
#   * NETWORK-gated — re-runs live tokenizers and asserts the committed       #
#     fixture still matches (catches drift; skipped offline, never flakes CI). #
# =========================================================================== #


def test_regress_suite_has_20_chinese_prompts():
    assert len(REGRESS_PROMPT_SUITE) == 20
    for p in REGRESS_PROMPT_SUITE:
        assert len(p) > 10
        assert any("\u4e00" <= ch <= "\u9fff" for ch in p)
    # first 10 are the m1 kill-gate corpus (a strict superset keeps the gate comparable)
    from wenyan.profiler import DEFAULT_PROMPT_SUITE

    assert REGRESS_PROMPT_SUITE[:10] == DEFAULT_PROMPT_SUITE


def test_recorded_counts_fixture_shape():
    """Committed fixture covers all three pinned models × 20 prompts × {base, compressed}."""
    assert set(RECORDED_COUNTS) == {"deepseek", "qwen", "glm"}
    for model, entry in RECORDED_COUNTS.items():
        assert entry["repo"].startswith(("deepseek-ai/", "Qwen/", "zai-org/"))
        assert len(entry["baseline"]) == 20
        assert len(entry["compressed"]) == 20
        # compressed never exceeds baseline (助词 strip only removes chars)
        for b, c in zip(entry["baseline"], entry["compressed"]):
            assert c <= b, f"{model}: compressed {c} > baseline {b}"


def test_recorded_counts_meta_is_set():
    assert RECORD_META["recorded_with"].startswith("transformers==")
    assert RECORD_META["recorded_at"]
    assert RECORD_META["suite_size"] == "20"


def test_harness_offline_falsifier_survives():
    """§8 net-savings falsifier survives against the committed fixture (no network).

    This is the reproducible-artifact assertion: every pinned model nets positive
    under the m1 助词 comprehension-safe retry=0 assumption, mechanically
    re-evaluable offline with `wenyan harness`.
    """
    report = run_harness()
    assert report.suite_size == 20
    assert len(report.results) == 3
    assert report.falsifier_survives
    for r in report.results:
        assert r.available
        assert r.net_positive, f"{r.model} net-saved {r.net_saved_tokens} — falsifier trips"
        assert r.task_success_rate == 1.0  # m1 comprehension-safe assumption
        # break-even retry budget = gross saved = the deterministic falsification threshold
        assert r.break_even_retry_budget == r.baseline_tokens - r.compressed_tokens
        assert r.break_even_retry_budget > 0


def test_harness_break_even_budget_is_gross_saved():
    r = run_harness().results[0]
    # budget == baseline - compressed (retry not yet counted) == gross saved
    assert r.break_even_retry_budget == r.baseline_tokens - r.compressed_tokens
    # and equals net when retry=0
    assert r.break_even_retry_budget == r.net_saved_tokens


def test_harness_trips_when_retry_exceeds_budget():
    """The falsifier is genuine: if retries cost more than saved, it trips."""
    report = run_harness(retry_cost_tokens=25)  # 25 tokens/prompt × 20 = 500 per model
    assert report.falsifier_survives is False
    for r in report.results:
        assert not r.net_positive, f"{r.model} unexpectedly net-positive under retry=25/prompt"


def test_harness_trips_at_exact_break_even():
    """At retry == break-even budget, net == 0 (not strictly positive) → falsifier trips."""
    report = run_harness(retry_cost_tokens=1)  # 20 retry/model; deepseek gross=20 → net 0
    deepseek = [r for r in report.results if r.model == "deepseek"][0]
    assert deepseek.break_even_retry_budget == 20
    assert deepseek.net_saved_tokens == 0  # exactly break-even
    assert deepseek.net_positive is False    # net_positive requires > 0
    assert report.falsifier_survives is False


def test_harness_per_prompt_retry_list_shape():
    """m3-style per-prompt retry lists are accepted and aggregated correctly."""
    report = run_harness(retry_cost_tokens=[1] * 20, task_success=[True] * 20)
    assert len(report.results) == 3
    for r in report.results:
        assert r.retry_cost_tokens == 20  # 1 × 20 prompts
        assert r.net_saved_tokens == r.break_even_retry_budget - 20


def test_harness_partial_task_success_lowers_rate():
    report = run_harness(task_success=[True] * 10 + [False] * 10)
    for r in report.results:
        assert r.task_success_rate == 0.5


def test_harness_rejects_mismatched_fixture_suite():
    with pytest.raises(ValueError):
        run_harness(
            recorded_counts={"qwen": {"baseline": [1, 2], "compressed": [1, 2], "repo": "x"}},
            suite=["a", "b", "c"],
        )


def test_harness_rejects_mismatched_retry_length():
    with pytest.raises(ValueError):
        run_harness(retry_cost_tokens=[1, 2, 3])  # 3 != 20-prompt suite


def test_harness_custom_fixture_falsifies_independently():
    """A hand-rolled fixture demonstrates the harness is a real falsifier, not a constant."""
    fixture = {
        "x": {"repo": "x/x", "baseline": [10, 10], "compressed": [9, 9]},
    }
    report = run_harness(recorded_counts=fixture, suite=["a", "b"], retry_cost_tokens=0)
    assert report.falsifier_survives  # net = 2 > 0
    # retry of 2/prompt → net 0 → trips
    report2 = run_harness(recorded_counts=fixture, suite=["a", "b"], retry_cost_tokens=2)
    assert report2.falsifier_survives is False


# --- network-gated: re-verify the committed fixture against live tokenizers ---


def test_live_counts_match_recorded_fixture():
    """The committed RECORDED_COUNTS must still match live tokenizers.

    Catches fixture drift (a pinned tokenizer repo updates its vocab). Skipped
    offline — this is the live re-verification; the offline replay above is the
    deterministic CI gate.
    """
    specs, _toks = _tokenizers_or_skip()
    live = record_live_counts(REGRESS_PROMPT_SUITE, specs)
    for model, entry in RECORDED_COUNTS.items():
        assert model in live, f"{model} missing from live record_live_counts output"
        assert live[model]["baseline"] == entry["baseline"], (
            f"{model}.baseline drifted — re-record the fixture (see docs/regress.md)"
        )
        assert live[model]["compressed"] == entry["compressed"], (
            f"{model}.compressed drifted — re-record the fixture (see docs/regress.md)"
        )


# =========================================================================== #
# fix-regress-crash-on-unavailable-tokenizer (v0.3.0)                          #
#                                                                             #
# `wenyan regress` (cli.py) must route tokenizers through `_load_tokenizers`   #
# (per-spec try/except → None on failure) so an unavailable pinned tokenizer   #
# (offline CI / air-gapped / repo removed / HF outage) degrades to the         #
# 'unavailable' row instead of crashing the §8 net-savings falsifier's         #
# re-eval path with a traceback. `load_tokenizer` RAISES on failure (never     #
# returns None, profiler.py), so the prior unguarded inline dict comprehension #
# propagated the exception — leaving the `if tok is None: ... unavailable`     #
# branch dead. This test exercises the now-live graceful-degrade branch.      #
# =========================================================================== #


def test_regress_degrades_on_unavailable_tokenizer(monkeypatch):
    """`wenyan regress` exits 0 with an 'unavailable' row, no traceback.

    Patches ``load_tokenizer`` to raise (simulated HF outage / offline CI) and
    asserts the command degrades gracefully rather than crashing — proving the
    cli.py:191 fix (``_load_tokenizers`` instead of the unguarded comprehension)
    holds. This is the offline-CI / air-gapped path that falsifier re-eval runs
    through, so a crash here would defeat the §8 post-ship kill-gate.
    """
    import io
    from unittest import mock

    from click.testing import CliRunner
    from rich.console import Console

    import wenyan.cli as cli_mod

    buf = io.StringIO()
    # drive the command's rich output into a buffer we can assert on; plain text
    # (non-tty) is fine — we only check the graceful-degrade row + no traceback.
    monkeypatch.setattr(cli_mod, "console", Console(file=buf, width=200))

    runner = CliRunner()
    with mock.patch.object(
        cli_mod,
        "load_tokenizer",
        side_effect=RuntimeError("simulated HF outage (repo unreachable / offline CI)"),
    ):
        result = runner.invoke(cli_mod.cli, ["regress", "--suite", "--retry-cost", "0"])

    assert result.exit_code == 0, (
        f"regress should degrade gracefully (exit 0), got exit={result.exit_code}; "
        f"exception={result.exception!r}; output:\n{result.output}"
    )
    assert result.exception is None, f"regress raised: {result.exception!r}"
    out = buf.getvalue()
    # every pinned model hits the graceful-degrade 'unavailable' row (3 warnings +
    # 3 table rows), proving _load_tokenizers (not the crashing comprehension) ran.
    assert "unavailable" in out, f"expected an 'unavailable' row; got:\n{out}"
    assert out.count("unavailable") >= 3, f"expected >=3 'unavailable' marks; got:\n{out}"
    assert "Traceback" not in out, f"regress leaked a traceback:\n{out}"
