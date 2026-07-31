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
