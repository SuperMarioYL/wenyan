"""wenyan CLI — the per-tokenizer 中文 prompt-compression surface.

m1 surface (implemented):
  * ``wenyan profile``  — per-model token counts + variance % + 助词 net per model.
  * ``wenyan compress``  — apply 助词_strip to a prompt, show before/after tokens.
  * ``wenyan regress``   — net-token gate (baseline − compressed − retry cost) per model.

m4 surface (v0.2.0 — committed reproducible net-token regression harness):
  * ``wenyan harness``          — offline replay of the §8 net-savings falsifier against
                                  the committed RECORDED_COUNTS fixture (deterministic, no network).
  * ``wenyan harness --verify`` — re-run live tokenizers and check the committed fixture still matches.

m2/m3 surface (stubbed with ``# TODO(m2)`` / ``# TODO(m3)``):
  * ``wenyan picker``    — per-model strategy picker reading profiler output.
  * ``wenyan suite``     — full 20-prompt regression suite + task-success run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .profiler import (
    DEFAULT_PROMPT_SUITE,
    count_tokens,
    load_model_specs,
    load_tokenizer,
    mean_variance,
    per_prompt_variance,
    profile as run_profile,
)
from .regress import regress as compute_regress
from .regress import (
    REGRESS_PROMPT_SUITE,
    RECORD_META,
    RECORDED_COUNTS,
    record_live_counts,
    run_harness,
    verdict as harness_verdict,
)
from .strategies import ALL_STRATEGIES, IMPLEMENTED_STRATEGIES, apply, describe, particle_strip

console = Console()

VARIANCE_GATE_PCT = 5.0  # mvp_plan §8 kill-gate


def _read_prompts(prompt_path: str | None, use_suite: bool) -> list[str]:
    if use_suite and prompt_path:
        raise click.UsageError("pass either --suite or -p, not both")
    if use_suite:
        return list(DEFAULT_PROMPT_SUITE)
    if prompt_path:
        text = Path(prompt_path).read_text(encoding="utf-8")
        prompts = [ln for ln in text.splitlines() if ln.strip()]
        return prompts or [text.strip()]
    raise click.UsageError("provide -p <file> or --suite")


def _load_tokenizers(specs):
    """Load tokenizers once (cached by transformers after first fetch)."""
    toks = {}
    for spec in specs:
        try:
            toks[spec.name] = load_tokenizer(spec.repo)
        except Exception as exc:  # network/backend — surface but keep going
            console.print(f"[yellow]warning:[/yellow] {spec.name} tokenizer unavailable: {exc}")
            toks[spec.name] = None
    return toks


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """wenyan (文言) — per-tokenizer 中文 prompt-compression Skill for DeepSeek/Qwen/GLM."""


@cli.command()
@click.option("-p", "--prompt", "prompt_path", type=click.Path(exists=True),
              help="Path to a prompt file (one prompt per line).")
@click.option("--suite", is_flag=True, help="Use the bundled 10-prompt Chinese suite.")
def profile(prompt_path: str | None, suite: bool) -> None:
    """Profile per-tokenizer token counts + variance % + 助词 net per model."""
    prompts = _read_prompts(prompt_path, suite)
    specs = load_model_specs()
    results = run_profile(prompts, specs)
    available = [r for r in results if r.available]

    if not available:
        console.print("[red]no tokenizer could be loaded[/red] "
                      "(network/HF unreachable). wenyan needs real tokenizers.")
        raise SystemExit(2)

    if len(available) < 2:
        # v0.4.0 fix: variance_pct returns 0.0 with <2 counts (profiler), so a
        # single available tokenizer makes every per-prompt spread 0.0 and would
        # print a false "FAIL ❌ variance <5%" §8 KILL. The thesis is not
        # falsified — there is just insufficient data to measure a cross-model
        # spread. Mark the gate INDETERMINATE and exit instead of running the
        # 0%-variance FAIL path. (The 0-available case is handled above.)
        console.print(Panel.fit(
            f"need ≥2 tokenizers to measure variance; only {len(available)} "
            f"available ({available[0].model}). variance gate: INDETERMINATE — "
            "§8 thesis not falsified, just unmeasurable (no cross-model spread).",
            title="m1 kill-gate · variance",
            border_style="yellow",
        ))
        raise SystemExit(2)

    table = Table(title="wenyan · per-tokenizer 中文 profile", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Family", style="dim")
    table.add_column("Baseline tokens", justify="right")
    table.add_column("助词_strip tokens", justify="right")
    table.add_column("Net saved", justify="right", style="green")

    # v0.5.0 fix: re-load tokenizers through the same per-spec try/except→None
    # pattern `_load_tokenizers` uses, so a transient re-load failure of a repo
    # that loaded successfully in run_profile degrades to an 'unavailable' row
    # instead of crashing the table build with an unhandled traceback.
    # `load_tokenizer` raises (never returns None) on failure (profiler.py), so
    # the prior unguarded comprehension propagated the exception. None rows are
    # skipped when building the table and the any_net loop below.
    toks = {}
    for r in available:
        try:
            toks[r.model] = load_tokenizer(r.repo)
        except Exception as exc:  # transient re-load failure — surface but keep going
            console.print(f"[yellow]warning:[/yellow] {r.model} tokenizer unavailable: {exc}")
            toks[r.model] = None
    for r in available:
        if toks[r.model] is None:
            table.add_row(r.model, r.family, "-", "-", "unavailable")
            continue
        comp = [count_tokens(toks[r.model], particle_strip(p)) for p in prompts]
        net = compute_regress(r.model, r.per_prompt_tokens, comp, retry_cost_tokens=0)
        net_str = f"+{net.net_saved_tokens}" if net.net_positive else str(net.net_saved_tokens)
        style = "green" if net.net_positive else "red"
        table.add_row(r.model, r.family, str(r.baseline_tokens), str(sum(comp)),
                      Text(net_str, style=style))

    console.print(table)

    spreads = per_prompt_variance(results, len(prompts))
    mean_var = mean_variance(results, len(prompts))
    min_var = min(spreads) if spreads else 0.0

    gate_passed = min_var > VARIANCE_GATE_PCT
    verdict = ("PASS ✅ per-tokenizer thesis holds — variance >5%"
               if gate_passed else "FAIL ❌ variance <5% — per-tokenizer layer collapses")
    console.print(Panel.fit(
        f"variance: mean [bold]{mean_var:.1f}%[/bold]  ·  min [bold]{min_var:.1f}%[/bold]"
        f"  (gate > {VARIANCE_GATE_PCT:.0f}%)\n{verdict}",
        title="m1 kill-gate · variance",
        border_style="magenta" if gate_passed else "red",
    ))

    # v0.5.0 fix: a falsified §8 variance kill-gate must exit non-zero, not 0.
    # Pre-fix the command printed the FAIL line but exited 0, so a mechanical
    # exit-code check treated a falsified per-tokenizer thesis as PASS — the
    # measured-and-failed path exited 0 while the cannot-measure path above
    # exits non-zero. Mirrors the indeterminate SystemExit(2) above and the
    # harness SystemExit(1) (cli.py harness). One guarded exit, no metric change.
    if not gate_passed:
        raise SystemExit(1)

    # v0.6.0 fix: guard the §8 net-token kill-gate against the all-reloads-fail
    # degenerate case the v0.5.0 reload fix made reachable. When every available
    # tokenizer's table-build reload failed (all `toks[r.model] is None`), the
    # `any_net = all(... for r in available if toks[r.model] is not None)` filter
    # yielded nothing, so `all([])` returned True and the command printed a false
    # "PASS ✅ ... net-token positive on every model" — a FALSE PASS on the §8
    # net-token kill-gate when NO model was actually measured. The variance gate
    # above measured real variance from run_profile's first load, but this gate
    # has nothing to measure on the reload. Mark INDETERMINATE and exit non-zero,
    # mirroring the variance gate's indeterminate SystemExit(2) above and the
    # v0.5.0 variance-FAIL SystemExit(1). (Reachable via the v0.5.0 reload-failure
    # path the test test_profile_degrades_on_reload_failure exercises for 1-of-3;
    # the all-reloads-fail case is the degenerate boundary it missed.)
    measured = [r for r in available if toks[r.model] is not None]
    if not measured:
        console.print(Panel.fit(
            f"need ≥1 tokenizer to measure net tokens; all {len(available)} "
            "available reload(s) failed. net-token gate: INDETERMINATE — §8 "
            "thesis not falsified, just unmeasurable (no model survived the "
            "table-build reload).",
            title="m1 kill-gate · net-token",
            border_style="yellow",
        ))
        raise SystemExit(2)
    any_net = all(
        compute_regress(r.model, r.per_prompt_tokens,
                [count_tokens(toks[r.model], particle_strip(p)) for p in prompts],
                retry_cost_tokens=0).net_positive
        for r in measured
    )
    net_verdict = ("PASS ✅ 助词 prototype net-token positive on every model"
                   if any_net else "FAIL ❌ a model nets negative — kill")
    console.print(Panel.fit(net_verdict, title="m1 kill-gate · net-token",
                            border_style="green" if any_net else "red"))


@cli.command()
@click.option("-p", "--prompt", "prompt_path", required=True, type=click.Path(exists=True))
@click.option("-s", "--strategy", type=click.Choice(list(ALL_STRATEGIES)), default="助词_strip",
              show_default=True, help="Compression strategy.")
def compress(prompt_path: str, strategy: str) -> None:
    """Apply a strategy and show before/after tokens per model."""
    text = Path(prompt_path).read_text(encoding="utf-8").strip()
    if strategy not in IMPLEMENTED_STRATEGIES:
        raise click.ClickException(
            f"{strategy!r} is m2 scope; only 助词_strip ships in m1")
    compressed = apply(strategy, text)  # type: ignore[arg-type]
    specs = load_model_specs()
    console.print(Panel.fit(describe(strategy), title=f"strategy · {strategy}"))
    console.print("[bold]原文:[/bold]")
    console.print(text)
    console.print("[bold]压缩后:[/bold]")
    console.print(compressed)

    table = Table(title="token deltas")
    table.add_column("Model")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Saved", justify="right")
    for spec in specs:
        try:
            tok = load_tokenizer(spec.repo)
            before = count_tokens(tok, text)
            after = count_tokens(tok, compressed)
            saved = before - after
            table.add_row(spec.name, str(before), str(after),
                          Text(f"+{saved}" if saved > 0 else str(saved),
                               style="green" if saved > 0 else "red"))
        except Exception as exc:
            table.add_row(spec.name, "-", "-", f"unavailable: {type(exc).__name__}")
    console.print(table)


@cli.command()
@click.option("-p", "--prompt", "prompt_path", type=click.Path(exists=True))
@click.option("--suite", is_flag=True, help="Use the bundled 10-prompt Chinese suite.")
@click.option("--retry-cost", type=int, default=0, show_default=True,
              help="Per-prompt retry cost in tokens (m1 default 0; m3 measures real).")
def regress(prompt_path: str | None, suite: bool, retry_cost: int) -> None:
    """Net-token gate (baseline − compressed − retry cost) per model."""
    prompts = _read_prompts(prompt_path, suite)
    specs = load_model_specs()
    # v0.3.0 fix: use _load_tokenizers (per-spec try/except → None on failure) so an
    # unavailable pinned tokenizer (offline CI / air-gapped / repo removed / HF outage)
    # degrades to the 'unavailable' row below instead of crashing the §8 net-savings
    # falsifier re-eval path with a traceback. `load_tokenizer` raises (never returns
    # None) on failure, so the prior inline comprehension was unguarded.
    toks = _load_tokenizers(specs)
    table = Table(title="wenyan · 助词 net-token regression", show_lines=True)
    table.add_column("Model")
    table.add_column("Baseline", justify="right")
    table.add_column("Compressed", justify="right")
    table.add_column("Retry cost", justify="right")
    table.add_column("Net saved", justify="right")
    for spec in specs:
        tok = toks.get(spec.name)
        if tok is None:
            table.add_row(spec.name, "-", "-", "-", "unavailable")
            continue
        base = [count_tokens(tok, p) for p in prompts]
        comp = [count_tokens(tok, particle_strip(p)) for p in prompts]
        res = compute_regress(spec.name, base, comp, retry_cost_tokens=retry_cost * len(prompts))
        table.add_row(spec.name, str(res.baseline_tokens), str(res.compressed_tokens),
                      str(res.retry_cost_tokens),
                      Text(f"+{res.net_saved_tokens}" if res.net_positive else str(res.net_saved_tokens),
                           style="green" if res.net_positive else "red"))
    console.print(table)


@cli.command()
@click.option("--verify", is_flag=True,
              help="Re-run live tokenizers and check the committed RECORDED_COUNTS still match "
                   "(network; the default offline replay needs no network).")
@click.option("--retry-cost", type=int, default=0, show_default=True,
              help="Override the per-prompt retry-cost assumption (tokens/prompt) to stress the falsifier.")
def harness(verify: bool, retry_cost: int) -> None:
    """m4: replay the §8 net-savings falsifier against the committed counts fixture.

    Default (offline, deterministic): replays the committed RECORDED_COUNTS — the
    §8 post-ship kill-gate's net-savings falsifier, mechanically re-evaluable with
    one command, no network. Exits 0 if every model nets positive, 1 if the
    falsifier trips. ``--verify`` re-runs the live tokenizers and fails on drift.
    """
    if verify:
        console.print("[bold]verifying[/bold] committed RECORDED_COUNTS against live tokenizers …")
        try:
            live = record_live_counts(REGRESS_PROMPT_SUITE)
        except Exception as exc:
            console.print(f"[red]could not load live tokenizers:[/red] {type(exc).__name__}: {exc}")
            raise SystemExit(2)
        drift = []
        for model, entry in RECORDED_COUNTS.items():
            for key in ("baseline", "compressed"):
                if live[model][key] != entry[key]:
                    drift.append(
                        f"{model}.{key}: recorded={entry[key][:8]}… live={live[model][key][:8]}…"
                    )
        if drift:
            console.print("[red]FAIL:[/red] RECORDED_COUNTS drifted from live tokenizers — "
                          "re-record the fixture (see docs/regress.md):")
            for d in drift:
                console.print(f"  - {d}")
            raise SystemExit(1)
        console.print("[green]PASS:[/green] RECORDED_COUNTS matches live tokenizers — "
                      "fixture is current.")
        # fall through to the offline replay on the (now-verified) fixture

    report = run_harness(retry_cost_tokens=retry_cost)
    table = Table(title="wenyan · m4 net-token regression harness (committed fixture)", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Compressed", justify="right")
    table.add_column("Retry cost", justify="right")
    table.add_column("Net saved", justify="right")
    table.add_column("Break-even retry budget", justify="right")
    table.add_column("Task success", justify="right")
    for r in report.results:
        if not r.available:
            table.add_row(r.model, "-", "-", "-", "-", "-", f"unavailable: {r.error}")
            continue
        net_str = f"+{r.net_saved_tokens}" if r.net_positive else str(r.net_saved_tokens)
        table.add_row(
            r.model,
            str(r.baseline_tokens),
            str(r.compressed_tokens),
            str(r.retry_cost_tokens),
            Text(net_str, style="green" if r.net_positive else "red"),
            str(r.break_even_retry_budget),
            f"{r.task_success_rate * 100:.0f}%",
        )
    console.print(table)
    console.print(Panel.fit(
        harness_verdict(report),
        title="§8 post-ship kill-gate · net-savings falsifier",
        border_style="green" if report.falsifier_survives else "red",
    ))
    console.print(f"[dim]recorded_with={RECORD_META.get('recorded_with')}  "
                  f"recorded_at={RECORD_META.get('recorded_at')}  "
                  f"suite={report.suite_size} prompts  "
                  f"(wenyan/models.toml)[/dim]")
    if not report.falsifier_survives:
        raise SystemExit(1)


@cli.command()
def picker() -> None:
    """Per-model strategy picker (m2)."""
    # TODO(m2): implement the full 文言文/助词_strip/成语_sub suite + picker that
    # selects min(strategy_net_tokens) per model from profiler output.
    console.print("[yellow]m2 scope:[/yellow] per-model strategy picker not yet implemented. "
                  "m1 ships only the 助词 prototype + variance profiler.")


def main() -> None:  # entry point: `wenyan`
    try:
        cli()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
