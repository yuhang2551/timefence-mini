import pandas as pd
import pytest

from timefence.core.clock import SimClock
from timefence.core.types import DECISION_TIME, Split, Window


def test_window_normalises_and_validates():
    w = Window("2020-01-02 13:00", "2020-03-31")
    assert w.start == pd.Timestamp("2020-01-02")
    assert w.contains("2020-02-01") and not w.contains("2020-04-01")
    assert str(w) == "2020-01-02~2020-03-31"
    with pytest.raises(ValueError):
        Window("2020-02-01", "2020-01-01")


def test_split_must_be_ordered():
    a, b, c = (
        Window("2018-01-01", "2018-12-31"),
        Window("2019-01-01", "2019-12-31"),
        Window("2020-01-01", "2020-12-31"),
    )
    Split(a, b, c)
    with pytest.raises(ValueError):
        Split(a, a, c)
    with pytest.raises(ValueError):
        Split(b, a, c)


def test_clock_walks_the_calendar():
    cal = pd.bdate_range("2020-01-01", periods=5)
    clock = SimClock(cal, start="2020-01-02")
    assert clock.today == cal[1]
    assert clock.as_of == cal[1] + DECISION_TIME
    seen = [clock.today]
    while clock.has_next():
        seen.append(clock.advance())
    assert seen == list(cal[1:])
    with pytest.raises(StopIteration):
        clock.advance()
