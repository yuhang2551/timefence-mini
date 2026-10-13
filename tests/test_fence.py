"""The fence: nothing published after the decision time may reach a strategy or an agent."""

import random

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from timefence.core.errors import LeakError
from timefence.core.types import DECISION_TIME
from timefence.data.fenced import FencedData
from timefence.data.store import MarketStore
from timefence.data.universe import liquidity_universe
from timefence.factors.evaluate import ResearchContext
from timefence.factors.generate import random_expression


def test_calendar_and_panels_stop_at_as_of(store):
    day = store.calendar[100]
    data = FencedData.at_close_of(store, day)
    assert data.as_of == day + DECISION_TIME
    assert data.calendar[-1] == day
    assert data.panel("close").index[-1] == day
    assert len(data.panel("close", lookback=5)) == 5


def test_before_the_close_today_is_not_visible(store):
    day = store.calendar[100]
    data = FencedData(store, day + pd.Timedelta(hours=10))  # 10:00, market still open
    assert data.calendar[-1] == store.calendar[99]


def test_late_published_data_raises(store):
    """If a record says it became public later than its date, the check must catch it."""
    bars = store.bars.copy()
    day = store.calendar[100]
    late = bars["date"] == day
    bars.loc[late, "available_at"] = day + pd.Timedelta(days=1, hours=9)  # published next morning
    bad_store = MarketStore(bars, store.stocks.reset_index())
    with pytest.raises(LeakError):
        FencedData.at_close_of(bad_store, day).panel("close")
    FencedData.at_close_of(bad_store, store.calendar[99]).panel("close")  # earlier is fine


def test_stock_table_hides_the_future(store):
    stocks = store.stocks
    delisted = stocks.index[stocks["delist_date"].notna()][0]
    last_day = stocks.loc[delisted, "delist_date"]
    before = FencedData.at_close_of(store, store.calendar[50]).stocks
    after = FencedData.at_close_of(store, store.calendar[-1]).stocks
    assert "name" not in before.columns
    assert pd.isna(before.loc[delisted, "delist_date"])
    assert after.loc[delisted, "delist_date"] == last_day
    ipo = stocks["list_date"].idxmax()
    assert ipo not in before.index and ipo in after.index


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 100_000), cut=st.integers(150, 560))
def test_factors_are_the_same_with_the_future_deleted(store, seed, cut):
    """Compute a random factor on the full history and on history truncated at `cut`.

    If any operator, the universe or the evaluator read even one day ahead, the values on
    or before the cut would differ.
    """
    expression = random_expression(random.Random(seed))
    last = store.calendar[cut]
    short = store.slice_until(last)

    full_uni = liquidity_universe(store, top_n=40)
    short_uni = liquidity_universe(short, top_n=40)
    common = short_uni.columns  # stocks listed later have no column in the short store
    pd.testing.assert_frame_equal(full_uni.loc[:last, common], short_uni)
    assert not full_uni.loc[:last].drop(columns=common).any().any()

    full = ResearchContext(store, full_uni).compute(expression)[1]
    part = ResearchContext(short, short_uni).compute(expression)[1]
    full = full.loc[:last, part.columns]
    pd.testing.assert_frame_equal(full, part, check_exact=False, rtol=1e-9, atol=1e-12)


def test_fenced_window_gives_the_same_signal_as_full_history(store):
    """A strategy sees only a short fenced window; its last row must match research values."""
    from timefence.factors import dsl

    expression = "rank(ts_mean(returns, 10)) - zscore(ts_std(close, 20))"
    parsed = dsl.parse(expression)
    full_uni = liquidity_universe(store, top_n=40)
    full = ResearchContext(store, full_uni).compute(expression)[1]
    for day in store.calendar[[200, 333, 599]]:
        data = FencedData.at_close_of(store, day, lookback=parsed.lookback + 30)
        uni = liquidity_universe(data, top_n=40)
        values = dsl.evaluate(parsed, data.panel, uni)
        row = values.iloc[-1][uni.iloc[-1]]
        expected = full.loc[day, row.index]
        pd.testing.assert_series_equal(row, expected, check_names=False)
