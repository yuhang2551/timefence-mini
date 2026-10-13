import numpy as np
import pandas as pd
import pytest

from timefence.data.schema import board_of, validate_bars, with_available_at
from timefence.data.store import MarketStore, forward_open_returns
from timefence.data.synthetic import synthetic_store


def _tiny():
    days = pd.bdate_range("2021-01-04", periods=4)
    bars = pd.DataFrame(
        {
            "date": list(days) * 2,
            "code": ["600000"] * 4 + ["300001"] * 4,
            "open": [10, 11, 5.5, 6, 20, 21, 22, 23],
            "high": [10.5, 11.5, 6, 6.5, 21, 22, 23, 24],
            "low": [9.5, 10.5, 5, 5.5, 19, 20, 21, 22],
            "close": [10, 11, 5.6, 6.1, 20.5, 21.5, 22.5, 23.5],
            "volume": [1000, 2000, 4000, 0, 500, 600, 700, 800],
            "amount": [1e4, 2.2e4, 2.2e4, 0, 1e4, 1.3e4, 1.6e4, 1.9e4],
            # 600000 splits 2-for-1 on day 3: raw prices halve, the factor doubles.
            "adj_factor": [1, 1, 2, 2, 1, 1, 1, 1],
        }
    )
    stocks = pd.DataFrame(
        {
            "code": ["600000", "300001"],
            "name": ["A", "B"],
            "board": ["main", "chinext"],
            "list_date": pd.Timestamp("2015-01-05"),
            "delist_date": pd.NaT,
        }
    )
    return MarketStore(with_available_at(bars), stocks)


def test_adjusted_prices_remove_the_split():
    store = _tiny()
    close = store.panel("close")["600000"]
    assert list(close) == [10, 11, 11.2, 12.2]
    assert list(store.panel("raw_close")["600000"]) == [10, 11, 5.6, 6.1]
    assert store.panel("returns")["600000"].iloc[2] == pytest.approx(11.2 / 11 - 1)


def test_zero_volume_counts_as_not_traded():
    store = _tiny()
    assert np.isnan(store.panel("volume")["600000"].iloc[3])
    assert not store.tradable()["600000"].iloc[3]
    assert np.isnan(store.panel("vwap")["600000"].iloc[3])


def test_vwap_is_adjusted():
    store = _tiny()
    assert store.panel("vwap")["600000"].iloc[2] == pytest.approx(2.2e4 / 4000 * 2)


def test_forward_open_returns_buy_next_open_sell_h_days_later():
    store = _tiny()
    labels = forward_open_returns(store, horizon=1)
    # Signal at day 0 close -> buy day 1 open (11), sell day 2 open (adjusted 5.5 * 2 = 11).
    assert labels["600000"].iloc[0] == pytest.approx(0.0)
    assert labels["300001"].iloc[0] == pytest.approx(22 / 21 - 1)
    assert labels.iloc[-2:].isna().all().all()  # no future to look at
    with pytest.raises(ValueError):
        forward_open_returns(store, horizon=0)


def test_save_and_load_round_trip(tmp_path):
    store = synthetic_store(n_stocks=12, n_days=50)
    store.save(tmp_path)
    again = MarketStore.load(tmp_path)
    pd.testing.assert_frame_equal(store.panel("close"), again.panel("close"))
    pd.testing.assert_frame_equal(store.stocks, again.stocks)


def test_slice_until_drops_later_bars():
    store = synthetic_store(n_stocks=12, n_days=50)
    cut = store.calendar[20]
    sliced = store.slice_until(cut)
    assert sliced.calendar[-1] == cut
    assert (sliced.bars["date"] <= cut).all()


def test_validate_bars_catches_problems():
    bars = _tiny().bars
    with pytest.raises(ValueError, match="duplicate"):
        validate_bars(pd.concat([bars, bars.iloc[:1]]))
    bad = bars.copy()
    bad.loc[0, "adj_factor"] = 0
    with pytest.raises(ValueError, match="adj_factor"):
        validate_bars(bad)
    early = bars.copy()
    early.loc[0, "available_at"] = early.loc[0, "date"] - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="available_at"):
        validate_bars(early)
    with pytest.raises(ValueError, match="missing"):
        validate_bars(bars.drop(columns=["amount"]))


@pytest.mark.parametrize(
    "code, board",
    [
        ("600519", "main"),
        ("000001", "main"),
        ("002594", "main"),
        ("300750", "chinext"),
        ("301269", "chinext"),
        ("688981", "star"),
        ("830799", "bse"),
        ("920001", "bse"),
    ],
)
def test_board_of(code, board):
    assert board_of(code) == board
