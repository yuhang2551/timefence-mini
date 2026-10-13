# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## What this is

An A-share factor-mining system with three LLM roles (planner, researcher, reviewer) in a
LangGraph loop, a whitelisted factor language, and a backtest with A-share rules. The core
promise is that nothing can use data from after the decision time. Read `docs/architecture.md`
and the ADRs in `docs/adr/` before changing structure.

## Commands

```bash
uv sync --group dev                 # install
uv run pytest                       # all tests (about 30 s)
uv run ruff check src tests         # lint
uv run ruff format src tests        # format
uv run lint-imports                 # layer contract
uv run timefence --help             # CLI
```

Run all four checks before saying a change is done.

## Layers

`cli -> experiments -> agents -> strategy -> (backtest | factors) -> data -> core`

A module may import only from layers below it. `lint-imports` enforces this. If you need
something from a higher layer, the design is wrong: pass it in as an argument or a Protocol
(see `backtest.engine.Strategy`).

## Invariants (do not break; each has tests)

1. Strategies receive a `FencedData`, never a `MarketStore`. `FencedData` cuts every query at
   `as_of` and raises `LeakError` if anything later slips through.
2. Every operator in `factors/ops.py` looks only backwards. A new operator needs an entry in
   `dsl.OPS` and must pass `tests/test_ops.py` (changing the future must not change the past).
   Never add an operator that shifts data backwards in time (a lead, a negative delay).
3. Factor values must be identical when all data after a day is deleted
   (`tests/test_fence.py::test_factors_are_the_same_with_the_future_deleted`, and
   `timefence audit` on real data).
4. Labels (`forward_open_returns`) are used only by `factors/evaluate.py`.
5. The research agents work through `FactorLab`, built on a context from
   `store.slice_until(train.end)`. `FactorLab` refuses a context that reaches past the window.
   Validation and test data are touched only in `experiments/runner.py`, after mining ends.
6. Acceptance rules live in code (`AcceptanceRules`), not in prompts.
7. Cross-sectional operators (`rank`, `zscore`) compare only universe stocks of that day.
8. The stock table a strategy sees has no names and shows a delisting only after it happened.
9. API keys come from environment variables only (the CLI also reads a local, gitignored `.env`).
   Never write a key into code, committed config or logs; `.env.example` holds a placeholder.

## Conventions

- Panels are wide DataFrames: rows are trading days, columns are 6-digit codes.
- Prices in the store are raw; `panel("close")` is adjusted (raw x adj_factor),
  `panel("raw_close")` is raw. The broker trades raw prices; factors use adjusted ones.
- Dates are midnight `pd.Timestamp`s; `available_at` and `as_of` carry the time of day.
- Docstrings and comments in English; user-facing docs and reports in Chinese.
- Results from `data synth` or `--offline` runs are not research results. Never present them
  as such in docs or reports; the report generator labels them.
