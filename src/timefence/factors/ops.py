"""Factor operators on wide panels (rows = trading days, columns = stocks).

Every time-series operator looks only backwards: the value on day t depends on days
t-d+1 .. t. That property is what makes any expression built from them safe to compute
on the full history; tests/test_fence.py checks it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Panel = pd.DataFrame


def _clean(x: Panel) -> Panel:
    return x.replace([np.inf, -np.inf], np.nan)


# ---- time series: per stock, along the date axis --------------------------------------


def delay(x: Panel, d: int) -> Panel:
    return x.shift(d)


def delta(x: Panel, d: int) -> Panel:
    return x - x.shift(d)


def pct_change(x: Panel, d: int) -> Panel:
    return _clean(x / x.shift(d) - 1.0)


def ts_mean(x: Panel, d: int) -> Panel:
    return x.rolling(d, min_periods=d).mean()


def ts_sum(x: Panel, d: int) -> Panel:
    return x.rolling(d, min_periods=d).sum()


def ts_std(x: Panel, d: int) -> Panel:
    return x.rolling(d, min_periods=d).std()


def ts_max(x: Panel, d: int) -> Panel:
    return x.rolling(d, min_periods=d).max()


def ts_min(x: Panel, d: int) -> Panel:
    return x.rolling(d, min_periods=d).min()


def ts_rank(x: Panel, d: int) -> Panel:
    """Percentile of today's value within the last d days."""
    return x.rolling(d, min_periods=d).rank(pct=True)


def ts_zscore(x: Panel, d: int) -> Panel:
    roll = x.rolling(d, min_periods=d)
    return _clean((x - roll.mean()) / roll.std())


def ts_corr(x: Panel, y: Panel, d: int) -> Panel:
    return _clean(x.rolling(d, min_periods=d).corr(y))


def ts_cov(x: Panel, y: Panel, d: int) -> Panel:
    return x.rolling(d, min_periods=d).cov(y)


def decay_linear(x: Panel, d: int) -> Panel:
    """Weighted mean of the last d days with weights 1..d (today weighs most)."""
    values = x.to_numpy(dtype=float)
    out = np.full_like(values, np.nan)
    if len(values) >= d:
        windows = np.lib.stride_tricks.sliding_window_view(values, d, axis=0)
        weights = np.arange(1, d + 1, dtype=float)
        out[d - 1 :] = (windows * weights).sum(axis=-1) / weights.sum()
    return pd.DataFrame(out, index=x.index, columns=x.columns)


# ---- cross section: per day, across stocks --------------------------------------------


def rank(x: Panel) -> Panel:
    return x.rank(axis=1, pct=True)


def zscore(x: Panel) -> Panel:
    return _clean(x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0))


# ---- element-wise ---------------------------------------------------------------------


def abs_(x: Panel) -> Panel:
    return x.abs()


def sign(x: Panel) -> Panel:
    return np.sign(x)


def log(x: Panel) -> Panel:
    """Signed log, defined for every real number: sign(x) * log(1 + |x|)."""
    return np.sign(x) * np.log1p(x.abs())


def sqrt(x: Panel) -> Panel:
    """Signed square root."""
    return np.sign(x) * np.sqrt(x.abs())


def max_(x: Panel, y: Panel) -> Panel:
    return np.maximum(x, y)


def min_(x: Panel, y: Panel) -> Panel:
    return np.minimum(x, y)
