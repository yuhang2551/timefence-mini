import pandas as pd

from timefence.data.synthetic import synthetic_store
from timefence.data.universe import liquidity_universe


def test_top_n_and_tradable(store, universe):
    assert universe.sum(axis=1).max() <= 40
    assert not (universe & ~store.tradable()).any().any()


def test_fresh_ipos_wait_a_year(store):
    uni = liquidity_universe(store, top_n=60, min_listed_days=365)
    listed = store.stocks["list_date"]
    for code in listed[listed > store.calendar[0]].index:
        first_ok = listed[code] + pd.Timedelta(days=365)
        assert not uni.loc[uni.index < first_ok, code].any()


def test_delisted_stock_leaves_and_stays_in_history(store, universe):
    stocks = store.stocks
    code = stocks.index[stocks["delist_date"].notna()][0]
    last_day = stocks.loc[code, "delist_date"]
    assert universe.loc[:last_day, code].any()  # it was in the universe before
    assert not universe.loc[universe.index > last_day, code].any()


def test_excluded_boards():
    store = synthetic_store(n_stocks=30, n_days=120)
    uni = liquidity_universe(store, top_n=30, min_listed_days=0, exclude_boards=("chinext",))
    chinext = store.stocks.index[store.stocks["board"] == "chinext"]
    assert not uni[chinext].any().any()
