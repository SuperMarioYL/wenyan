# Net-token regression harness — reproducible re-run

m4_harness_ci (v0.2.0) commits the m3 net-token regression harness as a
**reproducible artifact** so the §8 post-ship kill-gate's net-savings falsifier
can be re-evaluated mechanically instead of by eyeballing uncommitted demo
output.

## One-command re-run (offline, deterministic)

```bash
wenyan harness
```

Replays the committed `RECORDED_COUNTS` fixture (real `transformers` tokenizer
counts, recorded once and cached for replay — **not** a heuristic estimate) and
prints the §8 verdict per model: baseline / compressed / retry / net-saved / the
**break-even retry budget** (the max retry-token overhead the 助词 prototype
absorbs on that model before net goes negative — the deterministic
falsification threshold).

Exits `0` if every model nets positive (falsifier survives), `1` if any model
nets negative (falsifier trips → kill). No network, no flaky live task-success
API.

## Re-run under a different retry assumption

```bash
# stress the falsifier: assume 2 retry tokens / prompt and re-evaluate
wenyan harness --retry-cost 2
```

## Re-verify the committed fixture against live tokenizers

The committed counts can drift if a pinned tokenizer repo (`wenyan/models.toml`)
updates its vocab. Re-verify:

```bash
wenyan harness --verify
```

This re-runs the live tokenizers and fails (`exit 1`) on drift, with the
mismatching model/field. The network-gated `test_live_counts_match_recorded_fixture`
pytest test does the same check under CI (skipped offline).

## Re-recording the fixture on drift

If `--verify` or the drift test reports a mismatch, re-record the fixture:

1. Edit `wenyan/regress.py` — replace the `baseline` / `compressed` lists in
   `RECORDED_COUNTS` with the live values reported by `wenyan harness --verify`
   (or regenerate them with the recorder snippet below).
2. Bump `RECORD_META["recorded_at"]` to today's date and `recorded_with` to the
   installed `transformers` version.
3. Re-run `wenyan harness --verify` → should now pass — and commit.

Recorder snippet (run in the repo venv with network):

```python
from wenyan.regress import REGRESS_PROMPT_SUITE, record_live_counts
import json
print(json.dumps(record_live_counts(), ensure_ascii=False, indent=2))
```

## How the harness respects "no heuristic fallback"

`out_of_scope` bans token-count estimation without a real tokenizer. This harness
does **not** estimate — every count in `RECORDED_COUNTS` was produced by a real
`transformers` tokenizer against the pinned `wenyan/models.toml` repos. The
offline replay reuses that real measurement (cached for reproducibility); the
`--verify` path and the `test_live_counts_match_recorded_fixture` test re-run the
real tokenizers to confirm the cache is still valid. There is no code path that
fabricates a token count.

## Under CI

`.github/workflows/ci.yml` runs, on every push/PR:

- `pytest -v` — the full suite, including the offline replay assertions
  (`test_harness_offline_falsifier_survives` etc.) and the network-gated live
  re-verify (`test_live_counts_match_recorded_fixture`, runs in CI with the
  HuggingFace cache).
- `wenyan harness` — the documented one-command re-run, pinned as its own
  deterministic CI step (the committed-artifact gate).
- `python -m build` — smoke-builds the wheel + sdist the release ships.

The §8 net-savings falsifier is therefore re-evaluable mechanically from the
committed tree, with no uncommitted demo output in the loop.
