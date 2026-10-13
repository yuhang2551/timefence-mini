"""One experiment: mine factors on the training window, check them on the validation window,
then test the frozen result once on data nobody has looked at.

    train  -> the agent (or a baseline) proposes factors; the lab scores them here only
    valid  -> keep factors whose sign holds up out of sample
    test   -> factor statistics and a top-K backtest with A-share rules, run once
    holdout-> optional, later data for a final check after everything is frozen
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from timefence.agents.graph import ResearchAgent
from timefence.agents.llm import Budget, OpenAICompatibleLLM
from timefence.agents.memory import ResearchMemory
from timefence.agents.offline import offline_llm
from timefence.agents.tools import FactorLab
from timefence.backtest.engine import BacktestConfig, run_backtest
from timefence.backtest.metrics import equal_weight_returns
from timefence.core.types import Split, Window
from timefence.data.store import MarketStore
from timefence.data.universe import UniverseConfig
from timefence.experiments.baselines import run_handcrafted, run_random
from timefence.factors.evaluate import AcceptanceRules, ResearchContext, evaluate_factor
from timefence.strategy.model import FactorModel, FactorSpec
from timefence.strategy.topk import TopKStrategy

DEFAULT_SPLIT = (
    ("2018-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
)


@dataclass
class RunConfig:
    method: str = "agent"  # agent | random | handcrafted
    offline: bool = False  # agent only: use the template stand-in instead of a model
    rounds: int = 10
    proposals_per_round: int = 4
    random_trials: int = 40
    seed: int = 0
    horizon: int = 5
    top_n: int = 500  # universe size
    min_listed_days: int = 365
    top_k: int = 50  # stocks held
    buffer: float = 0.5
    rebalance_every: int = 5
    valid_min_icir: float = 0.1
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    max_calls: int = 300
    llm_cache: str = ".cache/llm"
    holdout: bool = False
    holdout_start: str = "2025-01-01"  # after the model's knowledge cutoff; check the model's docs

    @property
    def universe(self) -> UniverseConfig:
        return UniverseConfig(top_n=self.top_n, min_listed_days=self.min_listed_days)


def choose_split(calendar: pd.DatetimeIndex) -> Split:
    """The fixed 2018-2024 split when the data covers it, otherwise proportional windows."""
    if calendar[0] <= pd.Timestamp("2017-06-30") and calendar[-1] >= pd.Timestamp("2024-12-31"):
        return Split(*(Window(*w) for w in DEFAULT_SPLIT))
    warmup = min(260, len(calendar) // 4)  # history the factors need before the first score
    days = calendar[warmup:]
    a, b = int(len(days) * 0.5), int(len(days) * 0.7)
    return Split(
        Window(days[0], days[a - 1]), Window(days[a], days[b - 1]), Window(days[b], days[-1])
    )


def run_experiment(
    store: MarketStore, cfg: RunConfig, out: str | Path, data_info: dict | None = None
) -> dict:
    if cfg.method == "agent" and not cfg.offline and not os.environ.get(cfg.api_key_env):
        # Fail before the slow data preparation, not halfway through mining.
        raise RuntimeError(
            f"set the environment variable {cfg.api_key_env} to your API key, or use --offline"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Train, valid and test all end before the holdout, so the holdout stays untouched.
    split = choose_split(store.calendar[store.calendar < pd.Timestamp(cfg.holdout_start)])
    run_id = out.name

    mining, lab_reports = _mine(store, cfg, split, out, run_id)
    factors = _validate(store, cfg, split, lab_reports)
    backtest = _test(store, cfg, split, factors, out)
    holdout = _holdout(store, cfg, factors) if cfg.holdout else None

    summary = {
        "run_id": run_id,
        "config": asdict(cfg),
        "data": data_info or {},
        "split": {k: str(getattr(split, k)) for k in ("train", "valid", "test")},
        "mining": mining,
        "factors": factors,
        "backtest": backtest,
        "holdout": holdout,
    }
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2, ensure_ascii=False))
    from timefence.experiments.report import write_report

    write_report(summary, out / "report.md")
    return summary


def _mine(store: MarketStore, cfg: RunConfig, split: Split, out: Path, run_id: str):
    """Everything here sees data up to the end of the training window and nothing later."""
    train_store = store.slice_until(split.train.end)
    ctx = ResearchContext(train_store, cfg.universe.build(train_store), cfg.horizon)
    lab = FactorLab(ctx, split.train, AcceptanceRules())
    memory = ResearchMemory(out / "memory.sqlite")
    if cfg.method == "agent":
        if cfg.offline:
            llm = offline_llm(cfg.seed)
        else:
            llm = OpenAICompatibleLLM(
                model=cfg.model, base_url=cfg.base_url, api_key_env=cfg.api_key_env,
                cache_dir=Path(cfg.llm_cache), budget=Budget(max_calls=cfg.max_calls),
            )  # fmt: skip
        agent = ResearchAgent(
            llm, lab, memory, run_id, rounds=cfg.rounds, proposals_per_round=cfg.proposals_per_round,
            horizon=cfg.horizon, transcript=out / "transcript.jsonl",
        )  # fmt: skip
        mining = agent.run()
        mining["model"] = llm.name
    elif cfg.method == "random":
        mining = run_random(lab, memory, run_id, cfg.random_trials, cfg.seed)
    elif cfg.method == "handcrafted":
        mining = run_handcrafted(lab, memory, run_id)
    else:
        raise ValueError(f"unknown method {cfg.method!r}")
    mining["scored"] = mining["counts"]["accepted"] + mining["counts"]["rejected"]
    memory.close()
    return mining, dict(lab.reports)


def _validate(store: MarketStore, cfg: RunConfig, split: Split, reports: dict) -> list[dict]:
    """Score accepted factors on the validation window; keep those whose sign holds."""
    if not reports:
        return []
    valid_store = store.slice_until(split.valid.end)
    ctx = ResearchContext(valid_store, cfg.universe.build(valid_store), cfg.horizon)
    rows = []
    for expression, train in reports.items():
        valid = evaluate_factor(ctx, expression, split.valid)
        same_sign = valid.rank_ic * train.direction > 0
        rows.append(
            {
                "expression": expression,
                "direction": train.direction,
                "train": train.to_dict(),
                "valid": valid.to_dict(),
                "kept": bool(same_sign and abs(valid.rank_icir) >= cfg.valid_min_icir),
            }
        )
    return rows


def _test(
    store: MarketStore, cfg: RunConfig, split: Split, factors: list[dict], out: Path
) -> dict | None:
    """Run once, on the frozen factor list, over the untouched test window."""
    if not factors:
        return None
    universe = cfg.universe.build(store)
    ctx = ResearchContext(store, universe, cfg.horizon)
    for row in factors:
        row["test"] = evaluate_factor(ctx, row["expression"], split.test).to_dict()
    kept = [FactorSpec(r["expression"], r["direction"]) for r in factors if r["kept"]]
    if not kept:
        return None
    strategy = TopKStrategy(FactorModel(kept), cfg.top_k, cfg.buffer, cfg.universe)
    config = BacktestConfig(rebalance_every=cfg.rebalance_every)
    result = run_backtest(
        store, strategy, split.test, config, equal_weight_returns(store, universe)
    )
    pd.DataFrame({"nav": result.nav, "benchmark": result.benchmark}).to_csv(out / "nav.csv")
    result.fills.to_csv(out / "fills.csv", index=False)
    return {"window": str(split.test), "n_factors": len(kept), **result.summary()}


def _holdout(store: MarketStore, cfg: RunConfig, factors: list[dict]) -> dict | None:
    """The final check on data after the test window. Run it once, after everything is frozen."""
    days = store.calendar[store.calendar >= pd.Timestamp(cfg.holdout_start)]
    kept = [r for r in factors if r["kept"]]
    if len(days) < 60 or not kept:
        return None
    window = Window(days[0], days[-1])
    universe = cfg.universe.build(store)
    ctx = ResearchContext(store, universe, cfg.horizon)
    stats = {r["expression"]: evaluate_factor(ctx, r["expression"], window).to_dict() for r in kept}
    model = FactorModel([FactorSpec(r["expression"], r["direction"]) for r in kept])
    strategy = TopKStrategy(model, cfg.top_k, cfg.buffer, cfg.universe)
    result = run_backtest(store, strategy, window, BacktestConfig(rebalance_every=cfg.rebalance_every),
                          equal_weight_returns(store, universe))  # fmt: skip
    return {"window": str(window), "factors": stats, "backtest": result.summary()}


def _clean(value):
    """Make a structure JSON-safe: NaN and infinities become None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
