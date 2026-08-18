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
