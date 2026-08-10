"""Net-token regression harness (mvp_plan §5, m1/m3/m4).

The m1 gate measures ``net = baseline − compressed − retry_cost`` per model for
the 助词 prototype. ``retry_cost`` defaults to 0 for the m1 low-risk prototype
(the safe rewrite is assumed not to trigger retries); the m3 harness replaces
that assumption with *measured* retry tokens from a real task-success run
(mvp_plan §5 m3). Until then the net-token number here is a real measurement of
gross compression savings on the actual tokenizers — the falsifiable signal that
the kill-gate hinges on.

m4_harness_ci (v0.2.0) commits the m3 net-token regression as a *reproducible
artifact*: a 20-prompt Chinese task suite, a committed recording of real
transformers tokenizer counts (RECORDED_COUNTS), and an offline ``run_harness``
that replays the recording so the §8 post-ship kill-gate's net-savings falsifier
can be re-evaluated mechanically with one command (``wenyan harness``) — no
network, no flaky live task-success API. The recording is a *cached real
measurement*, not a heuristic (out_of_scope bans estimating tokens without a real
tokenizer; this records a real-tokenizer run and replays it). A separate
``record_live_counts`` + the network-gated ``test_live_counts_match_recorded``
re-verify the recording still matches live tokenizers, catching fixture drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# m3/m4 net-token regression harness                                           #
# --------------------------------------------------------------------------- #

# The 20-prompt Chinese coding-agent task suite (mvp_plan §5 m3). The first 10
# prompts are the m1 kill-gate corpus (profiler.DEFAULT_PROMPT_SUITE); prompts
# 11-20 extend it into a fuller task-success/regression corpus spanning systems
# code, async, data structures, DB internals, observability and product work.
REGRESS_PROMPT_SUITE: list[str] = [
    # --- m1 kill-gate corpus (prompts 1-10) ---
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
    # --- m3 extension (prompts 11-20) ---
    "请用 Rust 实现一个内存安全的 LRU 缓存，要求支持过期时间和容量淘汰，并给出单元测试。",
    "帮我把这个 Python 脚本改造成异步的，用 asyncio 和 aiohttp 并发抓取多个接口并合并结果。",
    "请解释一下 B+ 树和 LSM 树在数据库索引上的区别，以及它们各自适合的写入负载。",
    "我有一个 Docker 镜像体积太大，请帮我分析如何用多阶段构建和 scratch 基础镜像把它瘦身到 20MB 以内。",
    "请用 SQL 写一个窗口查询，找出每个部门薪资排名前三的员工，并处理并列的情况。",
    "帮我写一个 Python 装饰器，记录函数的调用次数、平均耗时和异常率，并把指标暴露成 Prometheus 格式。",
    "请用 Java 实现一个线程安全的单例模式，要兼顾懒加载和高并发性能，并说明为什么不用简单的双重检查锁。",
    "我有一个微服务架构，服务间调用链路太长导致延迟高，请帮我设计一个基于消息队列的异步化解耦方案。",
    "请用 Python 实现一个简易的布隆过滤器，支持自定义误判率和预期元素数量，并说明它的假阳性原理。",
    "帮我把这段中文产品需求文档梳理成结构化的用户故事和验收标准，方便开发排期。",
]
assert len(REGRESS_PROMPT_SUITE) == 20, "m3 regression suite must be 20 prompts"


# Recorded real-tokenizer counts for the 20-prompt suite across the three pinned
# 国产模型 tokenizers (wenyan/models.toml). Recorded once via `record_live_counts`
# against live `transformers` tokenizers and committed here so the §8 net-savings
# falsifier is re-evaluable offline (no network, no flaky live API) with
# `wenyan harness` / `run_harness`. This is a CACHED REAL MEASUREMENT — every
# count below was produced by a real tokenizer (transformers 5.14.1) on the pinned
# repo; it is not a heuristic estimate (out_of_scope bans estimation without a real
# tokenizer). Re-verify against live tokenizers with `wenyan harness --verify` or
# the network-gated `test_live_counts_match_recorded_fixture` test; if a pinned
# tokenizer repo drifts, that test fails and the fixture must be re-recorded.
RECORD_META: dict[str, str] = {
    "suite_size": "20",
    "recorded_with": "transformers==5.14.1",
    "recorded_at": "2026-08-10",
    "models_toml": "wenyan/models.toml",
    "strategy": "助词_strip (m1 prototype; retry_cost=0 — comprehension-safe assumption)",
}

RECORDED_COUNTS: dict[str, dict[str, list[int]]] = {
    "deepseek": {
        "repo": "deepseek-ai/deepseek-coder-1.3b-base",
        "baseline": [
            34, 27, 28, 31, 23, 29, 32, 36, 30, 33,
            37, 35, 30, 40, 29, 39, 36, 38, 36, 29,
        ],
        "compressed": [
            33, 25, 27, 29, 22, 27, 32, 36, 30, 32,
            35, 34, 29, 40, 28, 38, 35, 37, 34, 29,
        ],
    },
    "qwen": {
        "repo": "Qwen/Qwen2.5-0.5B",
        "baseline": [
            27, 22, 27, 25, 19, 23, 27, 29, 26, 25,
            27, 29, 27, 35, 23, 33, 31, 32, 30, 22,
        ],
        "compressed": [
            26, 21, 25, 24, 18, 22, 27, 29, 25, 24,
            25, 28, 26, 35, 22, 32, 30, 31, 29, 22,
        ],
    },
    "glm": {
        "repo": "zai-org/GLM-4-9B-0414",
        "baseline": [
            26, 21, 25, 26, 19, 24, 26, 28, 26, 22,
            27, 28, 25, 35, 22, 30, 30, 30, 29, 22,
        ],
        "compressed": [
            26, 21, 24, 25, 18, 23, 26, 28, 25, 22,
            27, 27, 24, 35, 21, 29, 30, 29, 29, 22,
        ],
    },
}


@dataclass
class PromptOutcome:
    """Per-prompt net-token outcome of the 助词 prototype for one model."""

    prompt_index: int
    baseline_tokens: int
    compressed_tokens: int
    retry_cost_tokens: int
    task_success: bool
    net_saved_tokens: int  # baseline − compressed − retry_cost

    @property
    def net_positive(self) -> bool:
        return self.net_saved_tokens > 0


@dataclass
class ModelHarnessResult:
    """Aggregated m3 net-token regression result for one model (offline replay)."""

    model: str
    repo: str
    available: bool
    per_prompt: list[PromptOutcome] = field(default_factory=list)
    baseline_tokens: int = 0
    compressed_tokens: int = 0
    retry_cost_tokens: int = 0
    net_saved_tokens: int = 0
    task_success_rate: float = 0.0  # 0..1
    error: str | None = None

    @property
    def net_positive(self) -> bool:
        """§8 net-savings falsifier per model: net tokens saved > 0."""
        return self.net_saved_tokens > 0

    @property
    def break_even_retry_budget(self) -> int:
        """Max retry-token overhead the prototype absorbs before net goes negative.

        ``net = baseline − compressed − retry``; net > 0 iff retry < baseline −
        compressed. So the per-model break-even budget = gross saved tokens. The
        §8 falsifier trips the moment a measured retry cost exceeds this budget on
        any model — a real, deterministic threshold derived from real token counts.
        """
        return self.baseline_tokens - self.compressed_tokens


@dataclass
class HarnessReport:
    """Full m3 harness report across all available models (offline replay)."""

    results: list[ModelHarnessResult] = field(default_factory=list)
    suite_size: int = 0
    recorded_with: str = ""
    recorded_at: str = ""

    @property
    def falsifier_survives(self) -> bool:
        """§8 post-ship kill-gate: net-savings falsifier survives iff every
        available model nets positive (retries did NOT cost more than saved)."""
        if not self.results:
            return False
        return all(r.net_positive for r in self.results if r.available)


def _coerce_retry(
    retry_cost_tokens: int | list[int] | None, n_prompts: int
) -> list[int]:
    """Normalise retry_cost into a per-prompt list of length n_prompts."""
    if retry_cost_tokens is None:
        return [0] * n_prompts
    if isinstance(retry_cost_tokens, int):
        return [retry_cost_tokens] * n_prompts
    if len(retry_cost_tokens) != n_prompts:
        raise ValueError(
            f"retry_cost_tokens length {len(retry_cost_tokens)} != suite {n_prompts}"
        )
    return list(retry_cost_tokens)


def _coerce_task_success(
    task_success: bool | list[bool] | None, n_prompts: int
) -> list[bool]:
    """Normalise task_success into a per-prompt list of length n_prompts."""
    if task_success is None:
        return [True] * n_prompts
    if isinstance(task_success, bool):
        return [task_success] * n_prompts
    if len(task_success) != n_prompts:
        raise ValueError(
            f"task_success length {len(task_success)} != suite {n_prompts}"
        )
    return list(task_success)


def run_harness(
    recorded_counts: dict[str, dict[str, list[int]]] | None = None,
    retry_cost_tokens: int | list[int] | None = None,
    task_success: bool | list[bool] | None = None,
    suite: list[str] | None = None,
) -> HarnessReport:
    """Replay the m3 net-token regression offline against committed counts.

    Defaults: RECORDED_COUNTS, retry_cost=0, task_success=True (the m1 助词
    comprehension-safe assumption, now committed as the harness default). All
    counts come from the committed real-tokenizer recording — no network, no
    flaky live API, fully deterministic. Returns a HarnessReport whose
    ``falsifier_survives`` is the §8 net-savings falsifier verdict.
    """
    counts = recorded_counts if recorded_counts is not None else RECORDED_COUNTS
    prompts = suite if suite is not None else REGRESS_PROMPT_SUITE
    n = len(prompts)
    report = HarnessReport(
        suite_size=n,
        recorded_with=RECORD_META.get("recorded_with", ""),
        recorded_at=RECORD_META.get("recorded_at", ""),
    )
    retry_list = _coerce_retry(retry_cost_tokens, n)
    ts_list = _coerce_task_success(task_success, n)
    for model, entry in counts.items():
        baseline = entry["baseline"]
        compressed = entry["compressed"]
        if len(baseline) != n or len(compressed) != n:
            raise ValueError(
                f"recorded counts for {model} do not match suite size {n} "
                f"(baseline={len(baseline)}, compressed={len(compressed)})"
            )
        per_prompt: list[PromptOutcome] = []
        for i in range(n):
            net = net_tokens(baseline[i], compressed[i], retry_list[i])
            per_prompt.append(
                PromptOutcome(
                    prompt_index=i,
                    baseline_tokens=baseline[i],
                    compressed_tokens=compressed[i],
                    retry_cost_tokens=retry_list[i],
                    task_success=ts_list[i],
                    net_saved_tokens=net,
                )
            )
        base_sum = sum(baseline)
        comp_sum = sum(compressed)
        retry_sum = sum(retry_list)
        ts_rate = sum(1 for t in ts_list if t) / n
        report.results.append(
            ModelHarnessResult(
                model=model,
                repo=entry.get("repo", ""),
                available=True,
                per_prompt=per_prompt,
                baseline_tokens=base_sum,
                compressed_tokens=comp_sum,
                retry_cost_tokens=retry_sum,
                net_saved_tokens=net_tokens(base_sum, comp_sum, retry_sum),
                task_success_rate=ts_rate,
            )
        )
    return report


def break_even_retry_budget(result: ModelHarnessResult) -> int:
    """Convenience accessor — max retry tokens before net goes negative."""
    return result.break_even_retry_budget


def verdict(report: HarnessReport) -> str:
    """Human-readable §8 net-savings falsifier verdict for the harness."""
    if not report.results:
        return "FAIL: no models in harness report"
    survives = report.falsifier_survives
    head = (
        "PASS: §8 net-savings falsifier survives — 助词 prototype nets positive on "
        "every available model"
        if survives
        else "FAIL: §8 net-savings falsifier TRIPPED — a model nets negative "
        "(retries cost more than saved) → kill"
    )
    lines = [head, ""]
    for r in report.results:
        if not r.available:
            lines.append(f"  {r.model}: unavailable ({r.error})")
            continue
        lines.append(
            f"  {r.model}: baseline={r.baseline_tokens} compressed={r.compressed_tokens} "
            f"retry={r.retry_cost_tokens} net={'+' if r.net_saved_tokens > 0 else ''}"
            f"{r.net_saved_tokens}  break-even retry budget={r.break_even_retry_budget}"
        )
    lines.append("")
    lines.append(
        f"  recorded_with={report.recorded_with}  recorded_at={report.recorded_at}  "
        f"suite={report.suite_size} prompts"
    )
    return "\n".join(lines)


def record_live_counts(
    prompts: list[str] | None = None,
    specs: list[Any] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Re-run the real tokenizers and return a fresh counts dict.

    Used by `wenyan harness --verify` and the network-gated drift test to check
    the committed RECORDED_COUNTS still match live tokenizers. Imports the
    profiler lazily so the offline harness (`run_harness`) never needs network.
    """
    from .profiler import DEFAULT_PROMPT_SUITE, count_tokens, load_tokenizer  # noqa: F401
    from .strategies import particle_strip

    if prompts is None:
        prompts = REGRESS_PROMPT_SUITE
    if specs is None:
        from .profiler import load_model_specs
        specs = load_model_specs()
    out: dict[str, dict[str, list[int]]] = {}
    for spec in specs:
        tok = load_tokenizer(spec.repo)
        out[spec.name] = {
            "repo": spec.repo,
            "baseline": [count_tokens(tok, p) for p in prompts],
            "compressed": [count_tokens(tok, particle_strip(p)) for p in prompts],
        }
    return out


def dump_recorded_counts_json() -> str:
    """Serialize RECORDED_COUNTS + RECORD_META to JSON (for re-recording)."""
    return json.dumps(
        {"_meta": RECORD_META, "models": RECORDED_COUNTS},
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------------------------------- #
# m1 net-token arithmetic (gate primitive)                                     #
# --------------------------------------------------------------------------- #


@dataclass
class NetTokenResult:
    """Per-model net-token outcome of a strategy across the prompt suite."""

    model: str
    baseline_tokens: int
    compressed_tokens: int
    retry_cost_tokens: int
    net_saved_tokens: int

    @property
    def net_positive(self) -> bool:
        return self.net_saved_tokens > 0


def net_tokens(baseline: int, compressed: int, retry_cost: int = 0) -> int:
    """Net tokens saved = baseline − compressed − retry cost (mvp_plan §5 m1)."""
    return baseline - compressed - retry_cost


def regress(
    model: str,
    baseline_tokens: list[int],
    compressed_tokens: list[int],
    retry_cost_tokens: int | list[int] = 0,
) -> NetTokenResult:
    """Aggregate a per-prompt run into one NetTokenResult for a model.

    ``retry_cost_tokens`` may be a single int (applied once, m1 assumption) or a
    per-prompt list (m3 harness). m1 passes 0 (low-risk prototype); m3 will pass
    measured per-prompt retry costs.
    """
    base_sum = sum(baseline_tokens)
    comp_sum = sum(compressed_tokens)
    if isinstance(retry_cost_tokens, int):
        retry_sum = retry_cost_tokens
    else:
        retry_sum = sum(retry_cost_tokens)
    return NetTokenResult(
        model=model,
        baseline_tokens=base_sum,
        compressed_tokens=comp_sum,
        retry_cost_tokens=retry_sum,
        net_saved_tokens=net_tokens(base_sum, comp_sum, retry_sum),
    )
