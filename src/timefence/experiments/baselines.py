"""Baselines the agent is compared with, judged by exactly the same lab and rules."""

from __future__ import annotations

import random

from timefence.agents.memory import ResearchMemory
from timefence.agents.tools import FactorLab
from timefence.factors.generate import random_expression

# Well-known daily factors written in the factor language. Signs follow the usual A-share
# findings, but the lab learns the direction on training data anyway.
HANDCRAFTED = {
    "5-day reversal": "-ts_sum(returns, 5)",
    "20-day reversal": "-pct_change(close, 20)",
    "momentum, 60 days skipping the last 5": "delay(pct_change(close, 55), 5)",
    "low volatility": "-ts_std(returns, 20)",
    "small and illiquid": "-log(ts_mean(amount, 20))",
    "volume surge": "-ts_mean(volume, 5) / ts_mean(volume, 60)",
    "price-volume correlation": "-ts_corr(close, volume, 20)",
    "close location in range": "ts_mean((close - low) / (high - low), 5)",
    "lottery (max daily return)": "-ts_max(returns, 20)",
    "turnover stability": "-ts_std(amount, 20) / ts_mean(amount, 20)",
    "close above vwap": "-(close / vwap - 1)",
    "high-low range": "-ts_mean((high - low) / close, 20)",
}


def run_handcrafted(lab: FactorLab, memory: ResearchMemory, run_id: str) -> dict:
    for name, expression in HANDCRAFTED.items():
        _record(lab, memory, run_id, 1, expression, theme="handcrafted", hypothesis=name)
    return {"counts": memory.counts(run_id), "accepted": list(lab.library)}


def run_random(
    lab: FactorLab, memory: ResearchMemory, run_id: str, trials: int, seed: int = 0
) -> dict:
    """Random expressions until `trials` of them have been scored (duplicates do not count)."""
    rng = random.Random(seed)
    scored = attempts = 0
    while scored < trials and attempts < trials * 20:
        attempts += 1
        status = _record(
            lab, memory, run_id, 1, random_expression(rng, max_depth=3), theme="random"
        )
        scored += status in ("accepted", "rejected")
    return {"counts": memory.counts(run_id), "accepted": list(lab.library)}


def _record(lab, memory, run_id, round_, expression, theme, hypothesis=None) -> str:
    v = lab.judge(expression, lambda c: memory.status_of(run_id, c))
    memory.add_trial(
        run_id, round_, v.expression, v.status, canonical=v.canonical, theme=theme,
        hypothesis=hypothesis, reason=v.reason, metrics=v.report.to_dict() if v.report else None,
    )  # fmt: skip
    return v.status
