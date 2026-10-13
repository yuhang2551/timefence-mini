"""Table layouts. Every market record carries `available_at`: the moment it became public."""

from __future__ import annotations

import pandas as pd

from timefence.core.types import BAR_AVAILABLE_TIME

# Long table of daily bars, one row per (date, code).
BAR_COLUMNS = [
    "date",  # trading day
    "code",  # 6-digit stock code, e.g. "600000"
    "open",  # raw (unadjusted) prices in yuan
    "high",
    "low",
    "close",
    "volume",  # shares
    "amount",  # yuan
    "adj_factor",  # backward adjustment factor: adjusted price = raw price * adj_factor
    "available_at",  # date + 15:00, when the bar became public
]

STOCK_COLUMNS = ["code", "name", "board", "list_date", "delist_date"]

PRICE_FIELDS = ("open", "high", "low", "close")


def board_of(code: str) -> str:
    """Which board a code trades on. The board decides the daily price limit."""
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("8", "4", "92")):
        return "bse"
    return "main"


def with_available_at(bars: pd.DataFrame) -> pd.DataFrame:
    """Stamp each daily bar as available at 15:00 on its own trading day."""
    out = bars.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["available_at"] = out["date"] + BAR_AVAILABLE_TIME
    return out


def validate_bars(bars: pd.DataFrame) -> None:
    """Fail fast on structural problems before anything is written to disk."""
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    if bars.duplicated(["date", "code"]).any():
        raise ValueError("bars contain duplicate (date, code) rows")
    late = bars["available_at"] < bars["date"]
    if late.any():
        raise ValueError("available_at earlier than the trading day itself")
    bad_adj = (bars["adj_factor"] <= 0) | bars["adj_factor"].isna()
    if bad_adj.any():
        raise ValueError(f"{int(bad_adj.sum())} rows have a non-positive or missing adj_factor")
