# Strategy · 助词 particle-stripping (m1, implemented)

**Status**: implemented in m1 — the comprehension-safe prototype the kill-gate runs on.

## Mechanism

Strip Chinese 助词 (particles) that carry low semantic load in coding prompts.
These are function/mood words, never content words, so removing them rarely
changes instruction meaning while dropping tokens on tokenizers that split
particles into their own subwords.

## Particle set

| Class            | Particles                       |
|------------------|---------------------------------|
| structure        | 的 地 得                        |
| aspect           | 了 着 过                         |
| mood (final)     | 吧 啊 呢 呀 嘛 哈 嗯 哦 呗       |
| plural           | 们                              |

`PARTICLES = frozenset("的了着过吧啊呢呀嘛哈嗯哦呗嗏的地得们")`

## Rules

1. Apply only to Chinese prose — never inside fenced code blocks, identifiers,
   or ASCII/English spans (particles don't exist there).
2. Drop the particle wholesale; do not substitute. e.g. "详细的中文注释" →
   "详细中文注释".
3. Leave content words, numbers, punctuation, and Latin/code untouched.

## Why it is the comprehension-safe prototype

caveman's English mechanism (space-delimited function-word truncation) has
nothing to bite in Chinese — Chinese has no spaces and fewer strip-able function
words. 助词 stripping is the closest Chinese analog: it removes *real* low-load
particles that the 国产模型 tokenizers often split into their own subwords, so
it saves tokens without rewriting meaning (the risk 文言文/成语 rewriting carries).
The m1 gate therefore runs on 助词 first — if even this safe prototype nets
negative, comprehension degradation is falsified and the project is killed
(mvp_plan §8).

## Measured m1 net-token outcome

Net-token positive on every model (retry_cost = 0 for the low-risk prototype; m3
replaces with measured retry tokens):

| Model    | Baseline | Compressed | Net saved |
|----------|---------:|-----------:|----------:|
| DeepSeek |      303 |        293 |       +10 |
| Qwen     |      250 |        241 |        +9 |
| GLM      |      243 |        238 |        +5 |
