---
name: wenyan
description: Compress 中文 prompts for DeepSeek/Qwen/GLM coding agents by profiling each tokenizer's Chinese subword split and applying the measured best strategy. m1 ships 助词 particle-stripping (lowest comprehension risk); 文言文 densify and 成语 substitution land in m2.
---

# wenyan (文言) — per-tokenizer 中文 prompt-compression

wenyan is the 中文 companion to the caveman token-compression wave, wired to a
specific surface the English original structurally cannot cover: the **per-国产模型
tokenizer** difference. The same 中文 prompt costs different tokens on DeepSeek
(measured: BBPE 32K, splits Chinese into more subwords) vs Qwen/GLM (151K vocabs,
denser Chinese splits). wenyan profiles that difference with real `transformers`
tokenizers and applies the strategy that wins net tokens per model — not one
global rewrite.

> Same 中文 prompt, different token cost: across a 10-prompt Chinese coding suite
> the relative token variance between DeepSeek/Qwen/GLM is **mean 26%, min 12%**
> (m1 kill-gate, `>5%` ⇒ the per-tokenizer layer is worth building).

## When to use

You are a coding Agent on a DeepSeek/Qwen/GLM backend, the user's prompt is in
Chinese, and you pay per token or are pressing against the context window. Apply
wenyan **before** sending the prompt to the model.

## m1 strategy (implemented): 助词 particle-stripping

Strip low-load Chinese 助词 (particles) that carry little semantic load in
coding prompts:

- structure particles: 的 · 地 · 得
- aspect markers: 了 · 着 · 过
- sentence-final mood particles: 吧 · 啊 · 呢 · 呀 · 嘛 · 哈 · 嗯 · 哦 · 呗
- plural marker: 们

See [`strategies/particle_strip.md`](strategies/particle_strip.md) for the exact
particle set and rationale. This is the **comprehension-safe** prototype — the m1
kill-gate asserts it is net-token positive on every model (stripping saves tokens
without inflating the count; the m3 regression harness replaces the
`retry_cost=0` assumption with measured task-success + retry tokens).

**Measured m1 outcome (10-prompt suite):**

| Model    | Baseline | 助词_strip | Net saved |
|----------|---------:|-----------:|----------:|
| DeepSeek |      303 |        293 |       +10 |
| Qwen     |      250 |        241 |        +9 |
| GLM      |      243 |        238 |        +5 |

## How to apply (m1)

1. Take the user's 中文 prompt.
2. Strip the 助词 set above from the prompt body (never strip code blocks, code
   identifiers, or English/code spans — particles only exist in the Chinese
   prose).
3. Send the compressed prompt to the model as you normally would.

For the measured per-tokenizer profile that picked this strategy, run:

```bash
wenyan profile --suite
wenyan compress -p prompt.txt            # show before/after tokens per model
```

## m2 strategies (not yet shipped)

The per-model strategy picker that selects `min(strategy_net_tokens)` and the
文言文 densification / 成语 substitution strategies are m2 scope. Their
strategy fragments live at [`strategies/wenyanwen.md`](strategies/wenyanwen.md)
and [`strategies/chengyu_sub.md`](strategies/chengyu_sub.md) as stubs until m2.

## Detection note

wenyan keys on the model's **tokenizer**, not the model name. The profiler
(`wenyan.models`) binds each 国产模型 to its real HF tokenizer repo; the picker
reads the measured per-model net-token table and selects the winning strategy.
For m1 all three models get 助词_strip (the only implemented strategy, and the
lowest-risk one); m2 introduces genuine per-model divergence when 文言文/成语
save more on some tokenizers than others.
