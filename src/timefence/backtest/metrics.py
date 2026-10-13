"""Performance statistics and the equal-weight benchmark."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from timefence.data.store import MarketStore

TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def performance(nav: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    """Return, risk and (with a benchmark NAV) excess-return statistics of a daily NAV."""
    rets = nav.pct_change().dropna()
    years = len(rets) / TRADING_DAYS
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    vol = float(rets.std() * math.sqrt(TRADING_DAYS))
    out = {
        "total_return": total,
        "annual_return": (1 + total) ** (1 / years) - 1 if years > 0 else math.nan,
        "annual_vol": vol,
        "sharpe": float(rets.mean() * TRADING_DAYS / vol) if vol > 0 else math.nan,
        "max_drawdown": max_drawdown(nav),
    }
    out["calmar"] = (
        out["annual_return"] / -out["max_drawdown"] if out["max_drawdown"] < 0 else math.nan
    )
    if benchmark is None:
        return out

    bench = benchmark.reindex(nav.index)
    b_rets = bench.pct_change().dropna()
    excess = (rets - b_rets).dropna()
    excess_nav = pd.concat([pd.Series([1.0]), (1 + excess).cumprod()], ignore_index=True)
    b_total = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
    te = float(excess.std() * math.sqrt(TRADING_DAYS))
    monthly = (1 + rets).groupby(rets.index.to_period("M")).prod() - (1 + b_rets).groupby(
        b_rets.index.to_period("M")
    ).prod()
    out.update(
        {
            "benchmark_total_return": b_total,
            "benchmark_annual_return": (1 + b_total) ** (1 / years) - 1 if years > 0 else math.nan,
            "excess_annual_return": float(excess_nav.iloc[-1] ** (1 / years) - 1)
            if years > 0
            else math.nan,
            "tracking_error": te,
            "information_ratio": float(excess.mean() * TRADING_DAYS / te) if te > 0 else math.nan,
            "excess_max_drawdown": max_drawdown(excess_nav),
            "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else math.nan,
        }
    )
    return out


def equal_weight_returns(store: MarketStore, universe: pd.DataFrame) -> pd.Series:
    """Daily return of holding yesterday's universe in equal weights (close to close, no costs)."""
    rets = store.panel("returns")[universe.columns]
    members = universe.shift(1, fill_value=False).astype(bool)
    with np.errstate(all="ignore"):
        return rets.where(members).mean(axis=1).fillna(0.0)
