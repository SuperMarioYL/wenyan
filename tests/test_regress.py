"""m1 kill-gate (b): 助词 particle-strip prototype net-token positive per model.

Asserts that stripping low-load Chinese 助词 saves tokens (net of retry cost) on
each model — the comprehension-degradation falsifier (mvp_plan §5 m1, §8). m1
uses retry_cost=0 (the low-risk prototype assumption); m3 will replace it with
measured retry tokens from a real task-success run.
"""

from __future__ import annotations

import pytest

from wenyan.profiler import DEFAULT_PROMPT_SUITE, count_tokens, load_model_specs, load_tokenizer
from wenyan.regress import NetTokenResult, net_tokens, regress
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
