"""m1 kill-gate (a): per-tokenizer 中文 subword variance across DeepSeek/Qwen/GLM.

Asserts that the same 中文 prompt costs >5% different tokens across the three
国产模型 tokenizers — the falsifiable signal that the per-tokenizer layer is
worth building (mvp_plan §8). Uses real ``transformers`` tokenizers (no heuristic
fallback). If the tokenizers cannot be fetched (offline CI), the gate is skipped
rather than fabricated — a real tokenizer is required for the count.
"""

from __future__ import annotations

import pytest

from wenyan.profiler import (
    DEFAULT_PROMPT_SUITE,
    load_model_specs,
    load_tokenizer,
    mean_variance,
    per_prompt_variance,
    profile,
    variance_pct,
)


def _tokenizers_or_skip():
    """Load all three real tokenizers; skip the whole suite if any is unreachable."""
    specs = load_model_specs()
    toks = {}
    for spec in specs:
        try:
            toks[spec.name] = load_tokenizer(spec.repo)
        except Exception as exc:
            pytest.skip(
                f"tokenizer {spec.name} ({spec.repo}) unavailable offline — "
                f"kill-gate needs real tokenizers ({type(exc).__name__})"
            )
    return specs, toks


def test_suite_has_ten_chinese_prompts():
    assert len(DEFAULT_PROMPT_SUITE) == 10
    for p in DEFAULT_PROMPT_SUITE:
        # every prompt is non-empty Chinese coding-agent prose
        assert len(p) > 10
        assert any("\u4e00" <= ch <= "\u9fff" for ch in p)


def test_model_registry_lists_deepseek_qwen_glm():
    specs = {s.name: s for s in load_model_specs()}
    assert set(specs) == {"deepseek", "qwen", "glm"}
    for s in specs.values():
        assert s.repo and s.repo.startswith(("deepseek-ai/", "Qwen/", "zai-org/"))


def test_variance_pct_metric():
    # max 6 vs min 4 → (6-4)/4 = 50%
    assert variance_pct([4, 6, 5]) == pytest.approx(50.0)
    assert variance_pct([5, 5, 5]) == 0.0
    assert variance_pct([0, 0]) == 0.0
    assert variance_pct([7]) == 0.0


def test_variance_kill_gate_above_5pct():
    specs, _toks = _tokenizers_or_skip()
    results = profile(DEFAULT_PROMPT_SUITE, specs)
    available = [r for r in results if r.available]
    assert len(available) == 3, "all three tokenizers must load for the gate"

    spreads = per_prompt_variance(results, len(DEFAULT_PROMPT_SUITE))
    mean_var = mean_variance(results, len(DEFAULT_PROMPT_SUITE))
    min_var = min(spreads)

    # mvp_plan §8: variance >5% across DeepSeek/Qwen/GLM ⇒ per-tokenizer thesis holds.
    assert mean_var > 5.0, f"mean variance {mean_var:.1f}% < 5% — per-tokenizer layer collapses"
    assert min_var > 5.0, f"min per-prompt variance {min_var:.1f}% < 5%"


def test_deepseek_costs_most_tokens_on_chinese():
    """DeepSeek's 32K BBPE splits Chinese into more subwords than Qwen/GLM's 151K vocabs."""
    specs, toks = _tokenizers_or_skip()
    sample = DEFAULT_PROMPT_SUITE[0]
    counts = {name: len(tok(sample, add_special_tokens=False)["input_ids"])
              for name, tok in toks.items()}
    assert counts["deepseek"] > counts["qwen"]
    assert counts["deepseek"] > counts["glm"]


# =========================================================================== #
# fix-profile-false-kill-variance-one-model (v0.4.0)                            #
#                                                                             #
# `wenyan profile` computes per-prompt variance and derives                     #
# `gate_passed = min_var > 5.0` with no check that ≥2 models loaded.           #
# `variance_pct` returns 0.0 whenever fewer than 2 counts are available         #
# (profiler.py), so with exactly 1 of 3 tokenizers available every per-prompt  #
# spread is 0.0 and the command used to print a false                          #
# "FAIL ❌ variance <5% — per-tokenizer layer collapses" §8 KILL and exit 0 —   #
# the thesis is not falsified, there is just insufficient data to measure it.  #
# The `len(available) < 2` guard now marks the gate INDETERMINATE instead of   #
# running the 0%-variance FAIL path. The 0-available case is already handled   #
# above the new branch.                                                         #
# =========================================================================== #


def test_profile_variance_indeterminate_with_one_model(monkeypatch):
    """`wenyan profile` with 1 of 3 tokenizers available marks the variance
    gate INDETERMINATE (not a false FAIL).

    Patches ``run_profile`` to return 1 available + 2 unavailable results and
    asserts the indeterminate path runs instead of the 0%-variance FAIL path.
    This is mutation-genuine: without the ``len(available) < 2`` guard,
    ``per_prompt_variance`` returns ``[0.0]*n`` (variance_pct needs ≥2 counts),
    so ``min_var == 0.0`` → ``gate_passed is False`` → the false
    "FAIL ❌ variance <5%" KILL fires.
    """
    import io
    from unittest import mock
    from click.testing import CliRunner
    from rich.console import Console

    import wenyan.cli as cli_mod
    from wenyan.profiler import ProfileResult

    buf = io.StringIO()
    monkeypatch.setattr(cli_mod, "console", Console(file=buf, width=200))

    # 1 of 3 tokenizers available — variance_pct needs ≥2 counts to measure.
    def fake_profile(prompts, specs):
        n = len(prompts)
        return [
            ProfileResult(
                model="deepseek", repo="deepseek-ai/deepseek-coder-1.3b-base",
                family="DeepSeek BBPE (32K)",
                per_prompt_tokens=[10] * n, baseline_tokens=10 * n, available=True,
            ),
            ProfileResult(
                model="qwen", repo="Qwen/Qwen2.5-0.5B", family="Qwen2 BBPE (151K)",
                available=False, error="RuntimeError: simulated offline",
            ),
            ProfileResult(
                model="glm", repo="zai-org/GLM-4-9B-0414", family="GLM-4 BBPE (151K)",
                available=False, error="RuntimeError: simulated offline",
            ),
        ]

    # Mock tokenizer so the pre-fix table-build reload (cli.py: `toks = {r.model:
    # load_tokenizer(r.repo) for r in available}`) wouldn't crash if reached —
    # keeps the mutation-genuine check focused on the variance-gate path.
    class _MockTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    runner = CliRunner()
    with mock.patch.object(cli_mod, "run_profile", side_effect=fake_profile), \
         mock.patch.object(cli_mod, "load_tokenizer", return_value=_MockTok()):
        result = runner.invoke(cli_mod.cli, ["profile", "--suite"])

    out = buf.getvalue()
    assert result.exit_code == 2, (
        f"profile should exit 2 (indeterminate) with 1 tokenizer, got "
        f"exit={result.exit_code}; exception={result.exception!r}; output:\n{out}"
    )
    assert "Traceback" not in out, f"profile leaked a traceback:\n{out}"
    assert "INDETERMINATE" in out, f"expected an INDETERMINATE verdict; got:\n{out}"
    assert "only 1 available" in out, f"expected the 'only N available' message; got:\n{out}"
    # the false §8 KILL must NOT fire (0.0 variance with <2 counts is not a kill)
    assert "variance <5%" not in out, f"false FAIL path ran (the bug); got:\n{out}"


# =========================================================================== #
# fix-profile-variance-gate-exits-zero-on-fail (v0.5.0)                        #
#                                                                             #
# `wenyan profile` derives `gate_passed = min_var > 5.0` and prints the       #
# FAIL line on failure but never raised SystemExit, so the command exited 0    #
# whether the §8 variance kill-gate passed OR failed — a mechanical exit-code  #
# check treated a falsified per-tokenizer thesis as PASS. The measured-and-    #
# failed path exited 0 while the v0.4.0 cannot-measure path exited non-zero   #
# (SystemExit(2)). The fix raises SystemExit(1) when gate_passed is False,     #
# mirroring the indeterminate SystemExit(2) and the harness SystemExit(1).    #
# =========================================================================== #


def test_profile_variance_gate_exits_nonzero_on_fail(monkeypatch):
    """`wenyan profile` with ≥2 available tokenizers whose counts collapse to
    <5% variance exits non-zero (1), not 0 — a falsified §8 variance kill-gate
    must kill mechanically.

    Patches ``run_profile`` to return 2 available models with IDENTICAL
    per-prompt token counts (so every per-prompt spread is 0.0% and
    ``min_var == 0.0 < 5.0`` → ``gate_passed is False``) and asserts the
    command exits 1 with the FAIL line and no traceback. This is
    mutation-genuine: pre-fix the FAIL line printed but the command exited 0.
    """
    import io
    from unittest import mock
    from click.testing import CliRunner
    from rich.console import Console

    import wenyan.cli as cli_mod
    from wenyan.profiler import ProfileResult

    buf = io.StringIO()
    monkeypatch.setattr(cli_mod, "console", Console(file=buf, width=200))

    # 2 of 3 tokenizers available with IDENTICAL counts → variance_pct 0.0%.
    def fake_profile(prompts, specs):
        n = len(prompts)
        return [
            ProfileResult(
                model="deepseek", repo="deepseek-ai/deepseek-coder-1.3b-base",
                family="DeepSeek BBPE (32K)",
                per_prompt_tokens=[10] * n, baseline_tokens=10 * n, available=True,
            ),
            ProfileResult(
                model="qwen", repo="Qwen/Qwen2.5-0.5B", family="Qwen2 BBPE (151K)",
                per_prompt_tokens=[10] * n, baseline_tokens=10 * n, available=True,
            ),
            ProfileResult(
                model="glm", repo="zai-org/GLM-4-9B-0414", family="GLM-4 BBPE (151K)",
                available=False, error="RuntimeError: simulated offline",
            ),
        ]

    # Mock the table-build tokenizer reload (now guarded by the v0.5.0 fix) so
    # the run reaches the variance gate instead of crashing on the reload.
    class _MockTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    runner = CliRunner()
    with mock.patch.object(cli_mod, "run_profile", side_effect=fake_profile), \
         mock.patch.object(cli_mod, "load_tokenizer", return_value=_MockTok()):
        result = runner.invoke(cli_mod.cli, ["profile", "--suite"])

    out = buf.getvalue()
    assert result.exit_code == 1, (
        f"profile should exit 1 (falsified variance gate), got "
        f"exit={result.exit_code}; exception={result.exception!r}; output:\n{out}"
    )
    assert "Traceback" not in out, f"profile leaked a traceback:\n{out}"
    assert "variance <5%" in out, f"expected the FAIL line; got:\n{out}"
    # the variance gate kills mechanically at the gate — the net-token gate
    # panel below must NOT render on a variance FAIL.
    assert "m1 kill-gate · net-token" not in out, (
        f"variance FAIL should kill before the net-token gate runs; got:\n{out}"
    )


# =========================================================================== #
# fix-profile-unguarded-tokenizer-reload (v0.5.0)                              #
#                                                                             #
# After the ≥2-availability guard passes, `profile` reloaded tokenizers via   #
# `toks = {r.model: load_tokenizer(r.repo) for r in available}` with NO       #
# try/except. `load_tokenizer` RAISES on failure (never returns None), so a   #
# transient re-load failure of a repo that loaded in run_profile crashed      #
# `profile` with an unhandled traceback right when building the table —       #
# AFTER the graceful-degrade guard already ran. Every other load site guards   #
# failure (`_load_tokenizers`, `profiler.profile`, `compress`). The fix       #
# wraps the reload in the same try/except→None pattern and skips None rows in  #
# the table + any_net loop, mirroring the `regress` command's graceful-        #
# degrade. The shipped test file used to mock load_tokenizer on this exact     #
# line specifically to keep the reload from crashing.                          #
# =========================================================================== #


def test_profile_degrades_on_reload_failure(monkeypatch):
    """`wenyan profile` degrades to an 'unavailable' row (no traceback) when a
    tokenizer that loaded in run_profile fails on the table-build reload.

    Patches ``run_profile`` to return 3 available models with differing counts
    (so the variance gate passes and the any_net loop also runs) and patches
    ``load_tokenizer`` to raise on the qwen repo's reload — simulating a
    transient HF online etag-check timeout / cache corruption between the two
    loads. Asserts the command degrades gracefully (an 'unavailable' row +
    warning, no traceback, exit 0) instead of crashing the table build.
    Mutation-genuine: pre-fix the unguarded comprehension propagated the
    reload exception; the any_net loop also crashed on `count_tokens(None, ...)`.
    """
    import io
    from unittest import mock
    from click.testing import CliRunner
    from rich.console import Console

    import wenyan.cli as cli_mod
    from wenyan.profiler import ProfileResult

    buf = io.StringIO()
    monkeypatch.setattr(cli_mod, "console", Console(file=buf, width=200))

    # 3 available models with DIFFERING counts → variance_pct > 5% (gate passes,
    # so the any_net loop also runs and must skip the None reload).
    def fake_profile(prompts, specs):
        n = len(prompts)
        return [
            ProfileResult(
                model="deepseek", repo="deepseek-ai/deepseek-coder-1.3b-base",
                family="DeepSeek BBPE (32K)",
                per_prompt_tokens=[20] * n, baseline_tokens=20 * n, available=True,
            ),
            ProfileResult(
                model="qwen", repo="Qwen/Qwen2.5-0.5B", family="Qwen2 BBPE (151K)",
                per_prompt_tokens=[15] * n, baseline_tokens=15 * n, available=True,
            ),
            ProfileResult(
                model="glm", repo="zai-org/GLM-4-9B-0414", family="GLM-4 BBPE (151K)",
                per_prompt_tokens=[10] * n, baseline_tokens=10 * n, available=True,
            ),
        ]

    class _MockTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    # The reload (the only call site once run_profile is mocked) raises for qwen
    # and returns a mock tok for the others — proving the reload is guarded.
    def fake_load(repo):
        if "Qwen" in repo:
            raise RuntimeError("simulated transient re-load failure (HF etag timeout)")
        return _MockTok()

    runner = CliRunner()
    with mock.patch.object(cli_mod, "run_profile", side_effect=fake_profile), \
         mock.patch.object(cli_mod, "load_tokenizer", side_effect=fake_load):
        result = runner.invoke(cli_mod.cli, ["profile", "--suite"])

    out = buf.getvalue()
    assert result.exit_code == 0, (
        f"profile should degrade gracefully (exit 0) when a reload fails but "
        f"the variance gate passes, got exit={result.exit_code}; "
        f"exception={result.exception!r}; output:\n{out}"
    )
    assert result.exception is None, f"profile raised: {result.exception!r}"
    assert "Traceback" not in out, f"profile leaked a traceback:\n{out}"
    # the qwen reload failed → a warning + an 'unavailable' table row.
    assert "unavailable" in out, f"expected an 'unavailable' row; got:\n{out}"
    assert "qwen tokenizer unavailable" in out, (
        f"expected the qwen reload-failure warning; got:\n{out}"
    )


# =========================================================================== #
# fix-profile-nettoken-gate-false-pass-on-all-reload-fail (v0.6.0)            #
#                                                                             #
# After the v0.5.0 reload guard, `profile` rebuilds tokenizers into `toks`  #
# and skips None rows in the table + the `any_net` loop:                      #
# `any_net = all(... for r in available if toks[r.model] is not None)`.      #
# When EVERY available tokenizer's table-build reload failed (all None —    #
# the degenerate boundary the v0.5.0 reload fix made reachable), the filter  #
# yielded nothing and `all([])` returned True, so `any_net = True` and the   #
# command printed a false "PASS ... net-token positive on every model" — a   #
# FALSE PASS on the §8 net-token kill-gate when NO model was measured. The     #
# v0.6.0 fix guards the empty-measured set: print INDETERMINATE + exit       #
# non-zero (SystemExit(2)), mirroring the variance gate's indeterminate exit. #
# =========================================================================== #


def test_profile_nettoken_gate_indeterminate_when_all_reloads_fail(monkeypatch):
    """`wenyan profile` with ≥2 available tokenizers whose table-build reloads
    ALL fail marks the net-token gate INDETERMINATE (not a false PASS).

    Patches ``run_profile`` to return 3 available models with DIFFERING
    per-prompt counts (so the variance gate PASSES and the run reaches the
    net-token gate) and patches ``load_tokenizer`` to raise on EVERY repo's
    reload — simulating all three table-build reloads failing (the degenerate
    boundary of the v0.5.0 reload fix). Asserts the command exits 2 with an
    INDETERMINATE net-token verdict and no false 'net-token positive on every
    model' PASS and no traceback. Mutation-genuine: pre-fix `all([]) == True`
    made ``any_net`` True → the false PASS.
    """
    import io
    from unittest import mock
    from click.testing import CliRunner
    from rich.console import Console

    import wenyan.cli as cli_mod
    from wenyan.profiler import ProfileResult

    buf = io.StringIO()
    monkeypatch.setattr(cli_mod, "console", Console(file=buf, width=200))

    # 3 available models with DIFFERING counts → variance_pct 100% (>5%, gate
    # passes) so the run reaches the net-token gate instead of the variance FAIL.
    def fake_profile(prompts, specs):
        n = len(prompts)
        return [
            ProfileResult(
                model="deepseek", repo="deepseek-ai/deepseek-coder-1.3b-base",
                family="DeepSeek BBPE (32K)",
                per_prompt_tokens=[20] * n, baseline_tokens=20 * n, available=True,
            ),
            ProfileResult(
                model="qwen", repo="Qwen/Qwen2.5-0.5B", family="Qwen2 BBPE (151K)",
                per_prompt_tokens=[15] * n, baseline_tokens=15 * n, available=True,
            ),
            ProfileResult(
                model="glm", repo="zai-org/GLM-4-9B-0414", family="GLM-4 BBPE (151K)",
                per_prompt_tokens=[10] * n, baseline_tokens=10 * n, available=True,
            ),
        ]

    # Every table-build reload fails — the degenerate all-None case. With no
    # measured model the pre-fix `all([]) == True` false-PASSed the net-token gate.
    runner = CliRunner()
    with mock.patch.object(cli_mod, "run_profile", side_effect=fake_profile), \
         mock.patch.object(
             cli_mod, "load_tokenizer",
             side_effect=RuntimeError("simulated all-reloads-fail (HF outage)")):
        result = runner.invoke(cli_mod.cli, ["profile", "--suite"])

    out = buf.getvalue()
    assert result.exit_code == 2, (
        f"profile should exit 2 (net-token INDETERMINATE) when all reloads fail, "
        f"got exit={result.exit_code}; exception={result.exception!r}; output:\n{out}"
    )
    assert "Traceback" not in out, f"profile leaked a traceback:\n{out}"
    assert "INDETERMINATE" in out, f"expected an INDETERMINATE net-token verdict; got:\n{out}"
    # the false §8 net-token PASS must NOT fire (all([])==True is not a real PASS)
    assert "net-token positive on every model" not in out, (
        f"false net-token PASS fired (the bug); got:\n{out}"
    )
    # all 3 reloads failed → 'unavailable' table rows
    assert "unavailable" in out, f"expected unavailable table rows; got:\n{out}"
