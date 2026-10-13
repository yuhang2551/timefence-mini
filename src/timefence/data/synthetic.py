"""A small synthetic A-share market for tests and offline demos.

Nothing produced from this data is a research result. It exists so that every layer can be
exercised end to end without network access: suspensions, a delisting, dividends, price
limits, and one planted pattern (short-term reversal) that the evaluator should detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from timefence.data.schema import board_of, with_available_at
from timefence.data.store import MarketStore

LIMITS = {"main": 0.10, "chinext": 0.20, "star": 0.20}


def make_codes(n: int) -> list[str]:
    prefixes = ["600", "000", "300", "688", "601", "002"]
    return [f"{prefixes[i % len(prefixes)]}{i:03d}" for i in range(n)]


def synthetic_store(
    n_stocks: int = 60,
    start: str = "2018-01-02",
    n_days: int = 600,
    reversal: float = 0.08,
    seed: int = 7,
) -> MarketStore:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    codes = make_codes(n_stocks)

    market = rng.normal(0.0003, 0.012, n_days)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    idio_vol = rng.uniform(0.012, 0.03, n_stocks)

    rets = np.zeros((n_days, n_stocks))
    for t in range(n_days):
        past = rets[max(0, t - 5) : t].sum(axis=0)
        noise = rng.normal(0.0, idio_vol)
        rets[t] = beta * market[t] + noise - reversal * past / 5.0

    limits = np.array([LIMITS[board_of(c)] for c in codes])
    rets = np.clip(rets, -limits, limits)

    close = 10.0 * np.exp(np.cumsum(np.log1p(rets), axis=0)) * rng.uniform(0.5, 5.0, n_stocks)
    prev_close = np.vstack([close[0] / (1 + rets[0]), close[:-1]])
    gap = rng.normal(0.0, 0.004, (n_days, n_stocks))
    open_ = np.clip(prev_close * (1 + gap), prev_close * (1 - limits), prev_close * (1 + limits))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, (n_days, n_stocks))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, (n_days, n_stocks))))
    high = np.minimum(high, prev_close * (1 + limits))
    low = np.maximum(low, prev_close * (1 - limits))

    base_volume = rng.uniform(2e6, 5e7, n_stocks)
    volume = base_volume * np.exp(rng.normal(0, 0.35, (n_days, n_stocks)) + 4 * np.abs(rets))
    volume = np.round(volume / 100) * 100
    amount = volume * (open_ + close + high + low) / 4

    # Dividends: raw prices drop on the ex-date, the adjustment factor absorbs the drop.
    adj = np.ones((n_days, n_stocks))
    for j in range(n_stocks):
        for ex_day in rng.choice(np.arange(min(50, n_days // 3), n_days), size=2, replace=False):
            ratio = 1 - rng.uniform(0.01, 0.04)
            close[ex_day:, j] *= ratio
            open_[ex_day:, j] *= ratio
            high[ex_day:, j] *= ratio
            low[ex_day:, j] *= ratio
            adj[ex_day:, j] /= ratio

    frame = pd.DataFrame(
        {
            "date": np.repeat(days, n_stocks),
            "code": np.tile(codes, n_days),
            "open": open_.ravel().round(2),
            "high": high.ravel().round(2),
            "low": low.ravel().round(2),
            "close": close.ravel().round(2),
            "volume": volume.ravel(),
            "amount": amount.ravel(),
            "adj_factor": adj.ravel(),
        }
    )

    # Suspensions: drop a few multi-day stretches of rows.
    drop = np.zeros(len(frame), dtype=bool)
    for j in rng.choice(n_stocks, size=max(1, n_stocks // 6), replace=False):
        s = int(rng.integers(n_days // 10, n_days - n_days // 10))
        length = int(rng.integers(2, 15))
        rows = np.arange(s, min(s + length, n_days)) * n_stocks + j
        drop[rows] = True

    # Most stocks listed years before the data starts; two list (IPO) during the sample.
    list_dates = {c: days[0] - pd.Timedelta(days=3 * 365) for c in codes}
    for c, start_pos in ((codes[-2], n_days // 4), (codes[-3], n_days // 2)):
        list_dates[c] = days[start_pos]
        drop |= (frame["code"] == c).to_numpy() & (frame["date"] < days[start_pos]).to_numpy()

    # One stock is delisted two thirds of the way through: its last trading day is
    # `last_day` and it has no bars after that.
    delisted = codes[-1]
    last_day = days[int(n_days * 2 / 3) - 1]
    drop |= (frame["code"] == delisted).to_numpy() & (frame["date"] > last_day).to_numpy()
    frame = frame[~drop].reset_index(drop=True)

    stocks = pd.DataFrame(
        {
            "code": codes,
            "name": [f"SYN{c}" for c in codes],
            "board": [board_of(c) for c in codes],
            "list_date": [list_dates[c] for c in codes],
            "delist_date": [last_day if c == delisted else pd.NaT for c in codes],
        }
    )
    return MarketStore(with_available_at(frame), stocks)
