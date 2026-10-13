"""The simulation clock. Only the framework moves it; agents and tools never set the date."""

from __future__ import annotations

import pandas as pd

from timefence.core.types import DECISION_TIME, to_ts


class SimClock:
    """Walks a trading calendar one decision point at a time."""

    def __init__(self, calendar: pd.DatetimeIndex, start: str | pd.Timestamp | None = None):
        if len(calendar) == 0:
            raise ValueError("calendar is empty")
        self._calendar = pd.DatetimeIndex(sorted(calendar))
        first = self._calendar[0] if start is None else to_ts(start)
        pos = self._calendar.searchsorted(first)
        if pos >= len(self._calendar):
            raise ValueError(f"start {first.date()} is after the last trading day")
        self._pos = int(pos)

    @property
    def today(self) -> pd.Timestamp:
        """The current trading day."""
        return self._calendar[self._pos]

    @property
    def as_of(self) -> pd.Timestamp:
        """The decision timestamp: today's close plus 30 minutes."""
        return self.today + DECISION_TIME

    def has_next(self) -> bool:
        return self._pos + 1 < len(self._calendar)

    def advance(self) -> pd.Timestamp:
        if not self.has_next():
            raise StopIteration("no more trading days")
        self._pos += 1
        return self.today
