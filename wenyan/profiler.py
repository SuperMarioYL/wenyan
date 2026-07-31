"""Per-tokenizer 中文 subword profiler (m1 core primitive, mvp_plan §2/§4).

Loads a real ``transformers`` tokenizer for each 国产模型 (DeepSeek/Qwen/GLM) and
measures token counts on a fixed Chinese prompt suite. The relative spread across
the three tokenizers is the m1 variance kill-gate: if the same 中文 prompt costs
near-identical tokens on every model, the per-tokenizer layer has nothing to key
on and the project is killed (mvp_plan §8).

No heuristic fallback: a real tokenizer is required for every count. If a
tokenizer cannot be loaded (e.g. offline CI), the caller is expected to skip the
gate rather than fabricate a count (out_of_scope: estimation without a real
tokenizer).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

try:  # transformers is a hard dependency; import lazily so module import never hard-fails
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - transformers missing is an install error
    AutoTokenizer = None  # type: ignore[assignment]


# Bundled 10-prompt Chinese coding-agent suite (m1 kill-gate corpus).
# These span code-gen, debugging, infra, infra-as-code, ML, scripting and
# translation prompts — the realistic shape of prompts that hit a 国产模型
# coding agent and whose token cost the per-tokenizer thesis keys on.
DEFAULT_PROMPT_SUITE: list[str] = [
    "请帮我写一个 Python 函数，输入一个列表，返回其中所有偶数的平方和，并加上详细的中文注释。",
    "我正在做一个 React 项目，遇到了状态管理的性能问题，请帮我分析一下并给出优化建议。",
    "请用 Go 语言实现一个并发的爬虫，要求能够控制并发数量，并且对失败的请求进行重试。",
    "帮我把这段 SQL 查询优化一下，它现在在百万级的数据上跑得很慢，索引也加过了。",
    "请解释一下 Kubernetes 中的 Service 和 Ingress 有什么区别，以及各自的适用场景。",
    "我想要实现一个中文文本分类的模型，数据集大概有十万条，请给我一个完整的训练流程。",
    "请帮我写一个 Shell 脚本，监控某个目录下的日志文件，一旦出现错误关键字就发邮件报警。",
    "请用 TypeScript 写一个深拷贝函数，要处理循环引用、日期、正则、Map 和 Set 等特殊情况。",
    "帮我设计一个高可用的 Redis 集群方案，要求支持自动故障转移和读写分离，并说明原因。",
    "请把这段中文需求文档翻译成英文，同时保留其中的代码示例和 Markdown 格式不要改动。",
]


@dataclass
class ModelSpec:
    """A 国产模型 → tokenizer repo binding."""

    name: str
    repo: str
    family: str = ""


@dataclass
class ProfileResult:
    """Measured token profile for one model across the prompt suite."""

    model: str
    repo: str
    family: str
    per_prompt_tokens: list[int] = field(default_factory=list)
    baseline_tokens: int = 0
    available: bool = False
    error: str | None = None


def load_model_specs(models_toml: Path | str | None = None) -> list[ModelSpec]:
    """Read the model registry (wenyan/models.toml by default)."""
    path = Path(models_toml) if models_toml else Path(__file__).parent / "models.toml"
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return [
        ModelSpec(name=m["name"], repo=m["repo"], family=m.get("family", ""))
        for m in data["models"]
    ]


def load_tokenizer(repo: str) -> "PreTrainedTokenizerBase":
    """Load a real tokenizer via transformers (no heuristic fallback)."""
    if AutoTokenizer is None:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for token counting; install with `pip install transformers`."
        )
    return AutoTokenizer.from_pretrained(repo)


def count_tokens(tokenizer: Any, text: str) -> int:
    """Token count of ``text`` under ``tokenizer`` (input ids, no special tokens)."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def profile(prompts: list[str], specs: list[ModelSpec]) -> list[ProfileResult]:
    """Profile each model's tokenizer across the prompt suite."""
    results: list[ProfileResult] = []
    for spec in specs:
        try:
            tokenizer = load_tokenizer(spec.repo)
            per_prompt = [count_tokens(tokenizer, p) for p in prompts]
            results.append(
                ProfileResult(
                    model=spec.name,
                    repo=spec.repo,
                    family=spec.family,
                    per_prompt_tokens=per_prompt,
                    baseline_tokens=sum(per_prompt),
                    available=True,
                )
            )
        except Exception as exc:  # network / repo / backend issues → mark unavailable
            results.append(
                ProfileResult(
                    model=spec.name,
                    repo=spec.repo,
                    family=spec.family,
                    available=False,
                    error=f"{type(exc).__name__}: {exc}"[:200],
                )
            )
    return results


def variance_pct(per_model_counts: list[int]) -> float:
    """Relative range across models: (max - min) / min * 100.

    The most expensive tokenizer costs this many percent more tokens than the
    cheapest on the same text. This is the m1 kill-gate metric (>5% ⇒ the
    per-tokenizer layer is worth building).
    """
    available = [c for c in per_model_counts if c and c > 0]
    if len(available) < 2:
        return 0.0
    return (max(available) - min(available)) / min(available) * 100.0


def per_prompt_variance(results: list[ProfileResult], n_prompts: int) -> list[float]:
    """Relative spread per prompt index across all available models."""
    out: list[float] = []
    for i in range(n_prompts):
        counts = [
            r.per_prompt_tokens[i]
            for r in results
            if r.available and len(r.per_prompt_tokens) > i
        ]
        out.append(variance_pct(counts))
    return out


def mean_variance(results: list[ProfileResult], n_prompts: int) -> float:
    """Mean per-prompt relative spread across the suite (the gate aggregate)."""
    spreads = per_prompt_variance(results, n_prompts)
    return sum(spreads) / len(spreads) if spreads else 0.0
