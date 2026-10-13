"""The event loop: decide at the close, trade at the next open, value at the close."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from timefence.backtest.broker import Broker, MarketTape
from timefence.backtest.costs import CostModel
from timefence.backtest.metrics import TRADING_DAYS, performance
from timefence.core.clock import SimClock
from timefence.core.types import Window
from timefence.data.fenced import FencedData
from timefence.data.store import MarketStore


class Strategy(Protocol):
    """What the engine needs from a strategy."""

    lookback: int  # trading days of history the strategy reads

    def target_weights(self, data: FencedData, holdings: Mapping[str, float]) -> pd.Series | None:
        """Target portfolio weights by code, or None to keep the current portfolio."""
        ...


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000_000.0
    rebalance_every: int = 5  # trading days between decisions
    costs: CostModel = field(default_factory=CostModel)
    tolerance: float = 0.2


@dataclass
class BacktestResult:
    nav: pd.Series  # daily, after the close, starting at 1.0
    fills: pd.DataFrame
    blocked: pd.DataFrame
    n_holdings: pd.Series
    initial_cash: float
    benchmark: pd.Series | None = None  # daily NAV starting at 1.0

    def summary(self) -> dict[str, float]:
        out = performance(self.nav, self.benchmark)
        years = max(len(self.nav) - 1, 1) / TRADING_DAYS
        traded = self.fills["value"].sum() if len(self.fills) else 0.0
        fees = self.fills["fee"].sum() if len(self.fills) else 0.0
        capital = self.nav.mean() * self.initial_cash
        out["annual_turnover"] = float(traded / 2 / capital / years)  # one-sided
        out["annual_cost"] = float(fees / capital / years)
        out["avg_holdings"] = float(self.n_holdings.mean())
        out["blocked_orders"] = float(len(self.blocked))
        delisted = self.fills["reason"].eq("delisted").sum() if len(self.fills) else 0
        out["delisted_exits"] = float(delisted)
        return out


def run_backtest(
    store: MarketStore,
    strategy: Strategy,
    window: Window,
    config: BacktestConfig | None = None,
    benchmark_returns: pd.Series | None = None,
) -> BacktestResult:
    """Simulate `strategy` over `window`.

    On every rebalance day t the strategy gets a FencedData view cut at t's close and
    returns target weights; the broker trades them at t+1's open. Holdings are valued at
    each close.
    """
    config = config or BacktestConfig()
    days = window.clip(store.calendar)
    if len(days) < 2:
        raise ValueError(f"window {window} has fewer than two trading days")
    tape = MarketTape(store, days)
    broker = Broker(config.initial_cash, config.costs, config.tolerance)
    last_trading_day = store.stocks["delist_date"]

    clock = SimClock(days)
    nav, n_holdings = [], []
    pending: pd.Series | None = None
    step = 0
    while True:
        day = clock.today
        q = tape.quotes(day)
        gone = [
            c
            for c in broker.shares
            if pd.notna(last_trading_day.get(c)) and day > last_trading_day[c]
        ]
        broker.start_day(q, delisted=gone)
        if pending is not None:
            broker.rebalance(q, pending)
            pending = None
        nav.append(broker.nav(q))
        n_holdings.append(len(broker.shares))
        if not clock.has_next():
            break
        if step % config.rebalance_every == 0:
            data = FencedData(store, clock.as_of, lookback=strategy.lookback)
            pending = strategy.target_weights(data, dict(broker.shares))
        clock.advance()
        step += 1

    nav_series = pd.Series(nav, index=days, name="nav") / config.initial_cash
    bench = None
    if benchmark_returns is not None:
        r = benchmark_returns.reindex(days).fillna(0.0)
        r.iloc[0] = 0.0
        bench = (1 + r).cumprod().rename("benchmark")
    return BacktestResult(
        nav=nav_series,
        fills=pd.DataFrame(
            broker.fills,
            columns=["date", "code", "side", "shares", "price", "value", "fee", "reason"],
        ),
        blocked=pd.DataFrame(broker.blocked, columns=["date", "code", "side", "reason"]),
        n_holdings=pd.Series(n_holdings, index=days, name="n_holdings"),
        initial_cash=config.initial_cash,
        benchmark=bench,
    )
