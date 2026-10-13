"""The download code is tested with a fake fetch function: no network needed."""

import numpy as np
import pandas as pd

from timefence.data.ingest import clean_adj_factor, fetch_one


def test_clean_adj_factor_removes_rounding_noise_and_keeps_real_steps():
    rng = np.random.default_rng(0)
    true = np.r_[np.full(30, 2.0), np.full(30, 2.1)]
    noisy = pd.Series(true * (1 + rng.uniform(-1.5e-3, 1.5e-3, size=60)))
    clean = clean_adj_factor(noisy)
    assert clean.nunique() == 2
    assert clean.iloc[:29].round(2).eq(2.0).all() and clean.iloc[31:].round(2).eq(2.1).all()


def _fake_fetch(symbol, period, start_date, end_date, adjust):
    days = pd.bdate_range("2021-01-04", periods=6).strftime("%Y-%m-%d")
    close = np.array([10.0, 10.2, 10.1, 9.6, 9.7, 9.8])  # 0.5 yuan dividend before day 3
    if adjust == "hfq":
        factor = np.array([3.0, 3.0, 3.0, 3.0 * 10.1 / 9.6, 3.0 * 10.1 / 9.6, 3.0 * 10.1 / 9.6])
        close = np.round(close * factor, 2)
    return pd.DataFrame(
        {
            "日期": days,
            "股票代码": symbol,
            "开盘": close,
            "收盘": close,
            "最高": close,
            "最低": close,
            "成交量": [100, 120, 110, 300, 90, 80],
            "成交额": [1e5] * 6,
            "振幅": 0.0,
            "涨跌幅": 0.0,
            "涨跌额": 0.0,
            "换手率": 0.0,
        }
    )


def test_fetch_one_builds_our_schema():
    bars = fetch_one("600000", "20210101", "20210131", fetch=_fake_fetch)
    assert list(bars.columns) == [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "adj_factor",
        "code",
    ]
    assert bars["volume"].iloc[0] == 100 * 100  # lots converted to shares
    adj_close = bars["close"] * bars["adj_factor"]
    assert adj_close.iloc[3] / adj_close.iloc[2] - 1 < 0.002  # the dividend is not a crash
    assert bars["adj_factor"].nunique() == 2
