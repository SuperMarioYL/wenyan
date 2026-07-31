# Strategy · 成语 substitution (m2 stub)

<!-- TODO(m2): implement 成语 substitution strategy + per-model picker fragment. -->

**Status**: stub — m2 scope (mvp_plan §5 m2_strategy_picker).

## Intended mechanism

Replace verbose multi-character Chinese phrases with the matching four-character
成语 (idiom), where the idiom encodes the same meaning in exactly four
characters. Because four-character idioms often tokenize as fewer subwords than
the phrases they replace on 国产模型 BBPE tokenizers, this can cut tokens — but
only on tokenizers that fuse the idiom densely; the per-model picker decides.

## Why it is m2, not m1

成语 substitution requires a curated phrase→idiom lookup table and (like 文言)
is meaning-adjacent enough to carry comprehension risk. It is gated behind the m3
regression guardrail alongside 文言 densification. m1 ships the safe 助词
prototype only.

## Intended m2 contract

- Input: modern Chinese prose.
- Output: phrases replaced with idioms from the curated lookup where a gloss
  is preserved.
- Per-model: chosen only where measured `strategy_net_tokens` is the minimum.
