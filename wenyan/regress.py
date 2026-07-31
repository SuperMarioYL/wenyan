"""Net-token regression harness (mvp_plan §5, m1/m3).

The m1 gate measures ``net = baseline − compressed − retry_cost`` per model for
the 助词 prototype. ``retry_cost`` defaults to 0 for the m1 low-risk prototype
(the safe rewrite is assumed not to trigger retries); the m3 harness replaces
that assumption with *measured* retry tokens from a real task-success run
(mvp_plan §5 m3). Until then the net-token number here is a real measurement of
gross compression savings on the actual tokenizers — the falsifiable signal that
the kill-gate hinges on.
"""

from __future__ import annotations

from dataclasses import dataclass


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
