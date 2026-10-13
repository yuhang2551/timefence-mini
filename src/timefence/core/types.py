"""Small value types passed between layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

# Daily bars become available at the close. Decisions are made at 15:30, after the close,
# and executed at the next trading day's open.
BAR_AVAILABLE_TIME = pd.Timedelta(hours=15)
DECISION_TIME = pd.Timedelta(hours=15, minutes=30)


def to_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalise a date-like value to a midnight Timestamp."""
    return pd.Timestamp(value).normalize()


class PanelSource(Protocol):
    """Anything that serves wide panels and the stock table: a MarketStore or a FencedData."""

    def panel(self, field: str) -> pd.DataFrame: ...

    @property
    def stocks(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class Window:
    """A closed date range [start, end] of trading days."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", to_ts(self.start))
        object.__setattr__(self, "end", to_ts(self.end))
        if self.start > self.end:
            raise ValueError(f"window start {self.start.date()} is after end {self.end.date()}")

    def contains(self, day: pd.Timestamp) -> bool:
        return self.start <= to_ts(day) <= self.end

    def clip(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return index[(index >= self.start) & (index <= self.end)]

    def __str__(self) -> str:
        return f"{self.start.date()}~{self.end.date()}"


@dataclass(frozen=True)
class Split:
    """Train / valid / test windows. The mining agent only ever sees `train`."""

    train: Window
    valid: Window
    test: Window

    def __post_init__(self) -> None:
        if not (self.train.end < self.valid.start and self.valid.end < self.test.start):
            raise ValueError("train, valid and test windows must be in order and not overlap")
