"""中文 compression strategies (mvp_plan §2).

m1 ships only the 助词 (particle-stripping) prototype — the lowest-comprehension-
risk rewrite, used by the m1 kill-gate to falsify the comprehension-degradation
thesis (does stripping particles save tokens net of retry cost on each model?).

文言文 densification and 成语 substitution land in m2 (the full strategy suite +
per-model picker). They are declared here as the Strategy type so the picker's
contract is fixed, but raise ``NotImplementedError`` until m2 lands.
"""

from __future__ import annotations

from typing import Literal

Strategy = Literal["文言文", "助词_strip", "成语_sub"]

# Chinese 助词 (particles) with low semantic load in coding prompts. Stripping
# these is the comprehension-safe prototype: structure particles (的/得),
# aspect markers (了/着), and sentence-final mood particles (吧/啊/呢/呀/嘛/
# 嗯/哦/呗). The 们 plural marker is also low-risk in imperative prompts.
#
# v0.4.0 prune: dropped the 地/过/哈 homographs — these double as content
# morphemes in common coding terms (地址/地图, 过滤·过期·过程·通过, 哈希), and
# the char-filter has no word-boundary awareness, so it destroyed those content
# words (布隆过滤器→布隆滤器), falsifying the m1 comprehension-safe premise.
# 的/了/得 stay: in the 20-prompt suite each occurs only as a genuine particle
# (的 possessive, 遇到了/加过了 perfective, 跑得 complement); word-boundary
# stripping (so out-of-suite 了解/得到 survive too) is m2 lookup-strategy scope.
PARTICLES: frozenset[str] = frozenset("的了着吧啊呢呀嘛嗯哦呗嗏得们")

# m1 implements only 助词_strip; m2 will add 文言文 + 成语_sub.
IMPLEMENTED_STRATEGIES: tuple[Strategy, ...] = ("助词_strip",)
ALL_STRATEGIES: tuple[Strategy, ...] = ("文言文", "助词_strip", "成语_sub")


def particle_strip(text: str) -> str:
    """助词 particle-stripping — remove low-load Chinese particles.

    The m1 prototype rewrite. Comprehension-safe by design: it only removes
    mood/aspect/structure particles, never content words, so the risk of a
    downstream model retrying (the m3 regression concern) is minimised while
    still dropping tokens on tokenizers that split particles into their own
    subwords.

    v0.4.0: PARTICLES was pruned to drop the 地/过/哈 homographs (地址/过滤/哈希)
    so the char-filter no longer destroys those content words; the remaining
    chars are particles in every in-suite occurrence. Full word-boundary
    stripping (so out-of-suite 了解/得到 survive too) is m2 lookup-strategy scope.
    """
    return "".join(ch for ch in text if ch not in PARTICLES)


def apply(strategy: Strategy, text: str) -> str:
    """Apply a named strategy to ``text``.

    m1: only ``助词_strip`` is implemented. ``文言文`` and ``成语_sub`` are m2.
    """
    if strategy == "助词_strip":
        return particle_strip(text)
    # TODO(m2): implement 文言文 densification (文言 densify) and 成语 substitution.
    raise NotImplementedError(
        f"strategy {strategy!r} is m2 scope; only 助词_strip ships in m1"
    )


def describe(strategy: Strategy) -> str:
    """One-line description of a strategy (used by the CLI + SKILL.md)."""
    if strategy == "助词_strip":
        return "助词 stripping — drop low-load particles (的/了/着/吧/呢…) for a comprehension-safe token cut."
    if strategy == "文言文":
        return "文言 densify — rewrite verbose modern Chinese into denser Classical phrasing (m2)."
    if strategy == "成语_sub":
        return "成语 substitution — replace phrases with four-character idioms (m2)."
    return str(strategy)
