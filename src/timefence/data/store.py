"""Parquet-backed market data store. This is the only way the rest of the code reads data."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from timefence.data.schema import BAR_COLUMNS, STOCK_COLUMNS, validate_bars

DERIVED_FIELDS = ("vwap", "returns")


class MarketStore:
    """Holds daily bars and the stock list, and serves them as wide panels (dates x codes)."""

    def __init__(self, bars: pd.DataFrame, stocks: pd.DataFrame, *, _checked: bool = False):
        if not _checked:  # slices of a checked store skip the (slow) sort and checks
            bars = bars[BAR_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
            validate_bars(bars)
        self._bars = bars
        self._stocks = stocks[STOCK_COLUMNS].set_index("code").sort_index()
        self._panels: dict[str, pd.DataFrame] = {}

    # ---- loading and saving -------------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path) -> MarketStore:
        root = Path(root)
        bars = pd.read_parquet(root / "bars.parquet")
        stocks = pd.read_parquet(root / "stocks.parquet")
        return cls(bars, stocks)

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self._bars.to_parquet(root / "bars.parquet", index=False)
        self._stocks.reset_index().to_parquet(root / "stocks.parquet", index=False)

    # ---- basic views --------------------------------------------------------------------

    @cached_property
    def calendar(self) -> pd.DatetimeIndex:
        dates = self._bars["date"].to_numpy()  # bars are sorted by date
        first = np.r_[True, dates[1:] != dates[:-1]] if len(dates) else np.zeros(0, bool)
        return pd.DatetimeIndex(dates[first], name="date")

    @cached_property
    def _grid(self) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        """Row and column position of every bar in the wide panels, computed once."""
        cols, codes = pd.factorize(self._bars["code"], sort=True)
        rows = np.searchsorted(self.calendar.to_numpy(), self._bars["date"].to_numpy())
        return rows, cols, pd.Index(codes, name="code")

    @property
    def stocks(self) -> pd.DataFrame:
        return self._stocks

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars

    def panel(self, field: str) -> pd.DataFrame:
        """A wide frame (dates x codes) for one field. Missing bars (suspensions) are NaN.

        Price fields (`open`, `high`, `low`, `close`, `vwap`) are backward-adjusted so that
        dividends and splits do not show up as jumps. `raw_<field>` gives unadjusted prices.
        """
        if field not in self._panels:
            self._panels[field] = self._build_panel(field)
        return self._panels[field]

    def _pivot(self, column: str) -> pd.DataFrame:
        rows, cols, codes = self._grid
        values = self._bars[column].to_numpy()
        if np.issubdtype(values.dtype, np.datetime64):
            wide = np.full(
                (len(self.calendar), len(codes)), np.datetime64("NaT"), dtype=values.dtype
            )
        else:
            wide = np.full((len(self.calendar), len(codes)), np.nan)
            values = values.astype(float)
        wide[rows, cols] = values
        return pd.DataFrame(wide, index=self.calendar, columns=codes, copy=False)

    def _build_panel(self, field: str) -> pd.DataFrame:
        if field.startswith("raw_"):
            return self._pivot(field.removeprefix("raw_"))
        if field in ("open", "high", "low", "close"):
            return self._pivot(field) * self._pivot("adj_factor")
        if field == "volume":
            volume = self._pivot("volume")
            return volume.where(volume > 0)  # zero volume means no trading
        if field in ("amount", "adj_factor", "available_at"):
            return self._pivot(field)
        if field == "vwap":
            raw_vwap = self._pivot("amount") / self.panel("volume")
            return raw_vwap * self._pivot("adj_factor")
        if field == "returns":
            # After a suspension, the return runs from the last close before it.
            close = self.panel("close")
            return close / close.ffill().shift(1) - 1.0
        raise KeyError(f"unknown field: {field}")

    def tradable(self) -> pd.DataFrame:
        """True where the stock traded that day (has a bar with positive volume)."""
        return self.panel("volume").notna()

    @cached_property
    def latest_available(self) -> pd.Series:
        """For each trading day, the latest `available_at` among its bars."""
        return self._bars.groupby("date")["available_at"].max().reindex(self.calendar)

    def slice_until(self, last_day: pd.Timestamp) -> MarketStore:
        """A new store containing only bars on or before `last_day`. Used by leak tests."""
        end = int(
            np.searchsorted(self._bars["date"].to_numpy(), pd.Timestamp(last_day), side="right")
        )
        return MarketStore(self._bars.iloc[:end], self._stocks.reset_index(), _checked=True)


def forward_open_returns(store: MarketStore, horizon: int) -> pd.DataFrame:
    """Return earned by a signal formed at day t's close: buy at t+1 open, sell at t+1+h open.

    This is the label used to score factors. It looks into the future on purpose and must
    never be visible to factor expressions; only the evaluator touches it.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    open_adj = store.panel("open")
    entry = open_adj.shift(-1)
    exit_ = open_adj.shift(-(1 + horizon))
    out = exit_ / entry - 1.0
    return out.replace([np.inf, -np.inf], np.nan)
