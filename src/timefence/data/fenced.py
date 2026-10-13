"""A read-only view of the store that cannot return anything published after `as_of`."""

from __future__ import annotations

import pandas as pd

from timefence.core.errors import LeakError
from timefence.core.types import BAR_AVAILABLE_TIME, DECISION_TIME, to_ts
from timefence.data.store import MarketStore


class FencedData:
    """Bind a store to a decision time. Every query is cut at `as_of` and then checked.

    The cut does the work; the check is a second line of defence that turns a bug in the
    cut (or in the store) into a loud LeakError instead of a quietly optimistic backtest.
    Strategies receive one of these, never the store itself.
    """

    def __init__(self, store: MarketStore, as_of: pd.Timestamp, lookback: int | None = None):
        self._store = store
        self._as_of = pd.Timestamp(as_of)
        self._lookback = lookback

    @classmethod
    def at_close_of(
        cls, store: MarketStore, day: str | pd.Timestamp, lookback: int | None = None
    ) -> FencedData:
        return cls(store, to_ts(day) + DECISION_TIME, lookback)

    @property
    def as_of(self) -> pd.Timestamp:
        return self._as_of

    @property
    def calendar(self) -> pd.DatetimeIndex:
        cal = self._store.calendar
        return cal[cal + BAR_AVAILABLE_TIME <= self._as_of]

    @property
    def stocks(self) -> pd.DataFrame:
        """Stocks listed by `as_of`. A delisting shows only once it has happened.

        Names are dropped: we only have today's names, and a name like "*ST..." would tell
        a 2019 strategy that the company gets into trouble later.
        """
        day = self._as_of.normalize()
        stocks = self._store.stocks
        stocks = stocks[stocks["list_date"] <= day].drop(columns=["name"])
        delisted = stocks["delist_date"].where(stocks["delist_date"] < day)
        return stocks.assign(delist_date=delisted)

    def panel(self, field: str, lookback: int | None = None) -> pd.DataFrame:
        """Rows for the last `lookback` trading days up to `as_of` (all of them if None)."""
        days = self.calendar
        lookback = lookback or self._lookback
        if lookback is not None:
            days = days[-lookback:]
        self._check(days)
        return self._store.panel(field).loc[days]

    def _check(self, days: pd.DatetimeIndex) -> None:
        if len(days) == 0:
            return
        latest = self._store.latest_available.loc[days].max()
        if pd.notna(latest) and latest > self._as_of:
            raise LeakError(f"query returned data available at {latest}, after as_of {self._as_of}")
