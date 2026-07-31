# Strategy · 文言 densify (m2 stub)

<!-- TODO(m2): implement 文言文 densification strategy + per-model picker fragment. -->

**Status**: stub — m2 scope (mvp_plan §5 m2_strategy_picker).

## Intended mechanism

Rewrite verbose modern Chinese into denser Classical-Chinese (文言) phrasing.
Classical Chinese encodes the same meaning in fewer characters, and fewer
characters tend to cost fewer subword tokens — the highest-ceiling compression
primitive, but also the highest comprehension-degradation risk (the m3
regression harness exists precisely to measure whether 文言 rewriting costs more
retries than it saves).

## Why it is m2, not m1

文言 densification is meaning-changing in a way 助词 stripping is not, so it
cannot ship before the comprehension guardrail (m3 net-token regression with real
task-success + retry tokens) is in place. m1 deliberately runs the safe 助词
prototype first to falsify the thesis on day one; 文言 lands once the guardrail
passes.

## Intended m2 contract

- Input: modern Chinese prose (code blocks / identifiers excluded).
- Output: denser 文言 phrasing preserving instruction intent.
- Per-model: only chosen where its measured `strategy_net_tokens` is the
  minimum for that model (the picker reads the profiler's per-model table).
