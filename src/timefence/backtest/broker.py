"""Simulated broker: holds cash and shares and fills orders at the open under A-share rules.

The broker plays the market, so it may read the day it trades on (that day's open). It
never decides anything: decisions come from a strategy that only saw data up to the
previous close.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from timefence.backtest.costs import CostModel
from timefence.backtest.rules import LOT, limit_pct_panel, round_price
from timefence.data.store import MarketStore

TICK = 1e-6


@dataclass
class DayQuotes:
    """One day's raw open, valuation price, limits and corporate-action ratio per stock."""

    day: pd.Timestamp
    index: Mapping[str, int]
    open: np.ndarray
    close: np.ndarray  # last known close (carried through suspensions), for valuation
    tradable: np.ndarray
    up: np.ndarray  # limit-up price
    down: np.ndarray  # limit-down price
    ratio: np.ndarray  # share multiplier from dividends and splits; 1 on normal days


class MarketTape:
    """Everything the broker needs for each day of a backtest, precomputed as arrays."""

    def __init__(self, store: MarketStore, days: pd.DatetimeIndex):
        raw_open = store.panel("raw_open")
        close = store.panel("raw_close").ffill()
        adj = store.panel("adj_factor")
        adj_ff = adj.ffill()
        # Reference price for the limits: yesterday's close, adjusted on ex-dates the way
        # the exchange does it (prev close * adj_prev / adj_today).
        ref = close.shift(1) * adj_ff.shift(1) / adj
        pct = limit_pct_panel(store.stocks["board"].reindex(raw_open.columns), raw_open.index)

        def rows(frame: pd.DataFrame) -> np.ndarray:
            return frame.loc[days].to_numpy(dtype=float)

        self.days = days
        self._row = {d: i for i, d in enumerate(days)}
        self._index = {c: i for i, c in enumerate(raw_open.columns)}
        self._open = rows(raw_open)
        self._close = rows(close)
        self._tradable = store.tradable().loc[days].to_numpy()
        self._up = round_price(rows(ref * (1 + pct)))
        self._down = round_price(rows(ref * (1 - pct)))
        self._ratio = rows((adj_ff / adj_ff.shift(1)).fillna(1.0))

    def quotes(self, day: pd.Timestamp) -> DayQuotes:
        i = self._row[day]
        return DayQuotes(
            day=day,
            index=self._index,
            open=self._open[i],
            close=self._close[i],
            tradable=self._tradable[i],
            up=self._up[i],
            down=self._down[i],
            ratio=self._ratio[i],
        )


class Broker:
    """Cash, positions in shares, and the fill log."""

    def __init__(self, cash: float, costs: CostModel | None = None, tolerance: float = 0.2):
        self.cash = float(cash)
        self.costs = costs or CostModel()
        self.tolerance = tolerance  # skip resizing a position by less than this share of target
        self.shares: dict[str, float] = {}
        self._bought_today: dict[str, float] = {}
        self.fills: list[dict] = []
        self.blocked: list[dict] = []

    # ---- daily bookkeeping --------------------------------------------------------------

    def start_day(self, q: DayQuotes, delisted: Iterable[str] = ()) -> None:
        """Apply dividends and splits, unlock yesterday's buys, exit delisted stocks."""
        self._bought_today.clear()
        for code in list(self.shares):
            ratio = q.ratio[q.index[code]]
            if np.isfinite(ratio) and ratio != 1.0:
                self.shares[code] *= ratio  # dividends reinvested, splits add shares
        for code in delisted:
            if code in self.shares:
                # Gone from the market: sold at its last close. Optimistic, and rare in a
                # liquidity universe; every case is listed in the fill log.
                self._fill(q, code, "sell", self.shares[code], q.close[q.index[code]], "delisted")

    def nav(self, q: DayQuotes, at_open: bool = False) -> float:
        total = self.cash
        for code, n in self.shares.items():
            i = q.index[code]
            price = q.open[i] if at_open and q.tradable[i] else q.close[i]
            total += n * price
        return total

    # ---- trading --------------------------------------------------------------------------

    def rebalance(self, q: DayQuotes, weights: pd.Series) -> None:
        """Trade toward target weights at today's open. Sells go first to free cash."""
        nav = self.nav(q, at_open=True)
        target = {c: float(w) * nav for c, w in weights.items() if w > 0}
        orders = []
        for code in set(self.shares) | set(target):
            i = q.index[code]
            price = q.open[i] if q.tradable[i] else q.close[i]
            current = self.shares[code] * price if code in self.shares else 0.0
            want = target.get(code, 0.0)
            if code in self.shares and want > 0 and abs(want - current) < self.tolerance * want:
                continue
            orders.append((want - current, code))
        for delta, code in sorted(orders):
            if delta < 0:
                self.sell(q, code, -delta, everything=code not in target)
            elif delta > 0:
                self.buy(q, code, delta)

    def sell(self, q: DayQuotes, code: str, value: float, everything: bool = False) -> None:
        i = q.index[code]
        if not q.tradable[i]:
            return self._block(q, code, "sell", "suspended")
        if q.open[i] <= q.down[i] + TICK:
            return self._block(q, code, "sell", "limit_down")
        sellable = self.shares.get(code, 0.0) - self._bought_today.get(code, 0.0)
        if sellable <= 0:
            return self._block(q, code, "sell", "t_plus_1")
        price = self.costs.fill_price(q.open[i], "sell")
        n = sellable if everything else min(sellable, math.floor(value / price / LOT) * LOT)
        if n > 0:
            self._fill(q, code, "sell", n, price, "rebalance")

    def buy(self, q: DayQuotes, code: str, value: float) -> None:
        i = q.index[code]
        if not q.tradable[i]:
            return self._block(q, code, "buy", "suspended")
        if q.open[i] >= q.up[i] - TICK:
            return self._block(q, code, "buy", "limit_up")
        price = self.costs.fill_price(q.open[i], "buy")
        n = math.floor(min(value, self.cash) / price / LOT) * LOT
        while n > 0 and n * price + self.costs.fees(n * price, "buy", q.day) > self.cash:
            n -= LOT
        if n > 0:
            self._fill(q, code, "buy", n, price, "rebalance")
            self._bought_today[code] = self._bought_today.get(code, 0.0) + n

    def _fill(self, q: DayQuotes, code: str, side: str, n: float, price: float, why: str) -> None:
        value = n * price
        fee = self.costs.fees(value, side, q.day)
        if side == "buy":
            self.cash -= value + fee
            self.shares[code] = self.shares.get(code, 0.0) + n
        else:
            self.cash += value - fee
            left = self.shares[code] - n
            if left > 1e-9:
                self.shares[code] = left
            else:
                del self.shares[code]
        self.fills.append(
            {"date": q.day, "code": code, "side": side, "shares": n, "price": price,
             "value": value, "fee": fee, "reason": why}
        )  # fmt: skip

    def _block(self, q: DayQuotes, code: str, side: str, reason: str) -> None:
        self.blocked.append({"date": q.day, "code": code, "side": side, "reason": reason})
