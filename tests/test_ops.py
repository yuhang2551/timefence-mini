"""Every time-series operator must look only backwards in time."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from timefence.factors import dsl

WINDOWED = [n for n, s in dsl.OPS.items() if s.windowed]


def _panel(seed: int, n: int = 60, m: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.lognormal(size=(n, m))
    values[rng.random(size=values.shape) < 0.05] = np.nan  # suspensions
    return pd.DataFrame(
        values, index=pd.bdate_range("2021-01-01", periods=n), columns=list("ABCDE")
    )


@pytest.mark.parametrize("name", WINDOWED)
@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), cut=st.integers(10, 50), d=st.integers(1, 12))
def test_changing_the_future_does_not_change_the_past(name, seed, cut, d):
    spec = dsl.OPS[name]
    x, y = _panel(seed), _panel(seed + 1)
    x2, y2 = x.copy(), y.copy()
    x2.iloc[cut:] *= 3.7  # rewrite everything from row `cut` on
    y2.iloc[cut:] += 1.3
    args = [x, y][: spec.n_series]
    args2 = [x2, y2][: spec.n_series]
    before = spec.func(*args, d).iloc[:cut]
    after = spec.func(*args2, d).iloc[:cut]
    pd.testing.assert_frame_equal(before, after)


def test_decay_linear_weights_recent_days_more():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2021-01-01", periods=3))
    out = dsl.OPS["decay_linear"].func(x, 3)
    assert np.isnan(out.iloc[0, 0]) and np.isnan(out.iloc[1, 0])
    assert out.iloc[2, 0] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)


def test_signed_log_and_sqrt_keep_the_sign():
    x = pd.DataFrame({"A": [-4.0, 0.0, 4.0]})
    assert list(dsl.OPS["sqrt"].func(x)["A"]) == [-2.0, 0.0, 2.0]
    assert list(np.sign(dsl.OPS["log"].func(x)["A"])) == [-1.0, 0.0, 1.0]


ROLLING = {
    "ts_mean": lambda x, y, d: x.rolling(d, min_periods=d).mean(),
    "ts_sum": lambda x, y, d: x.rolling(d, min_periods=d).sum(),
    "ts_std": lambda x, y, d: x.rolling(d, min_periods=d).std(),
    "ts_max": lambda x, y, d: x.rolling(d, min_periods=d).max(),
    "ts_min": lambda x, y, d: x.rolling(d, min_periods=d).min(),
    "ts_rank": lambda x, y, d: x.rolling(d, min_periods=d).rank(pct=True),
    "ts_cov": lambda x, y, d: x.rolling(d, min_periods=d).cov(y),
    "ts_corr": lambda x, y, d: x.rolling(d, min_periods=d).corr(y),
}


@pytest.mark.parametrize("name", list(ROLLING))
@settings(max_examples=10, deadline=None)
@given(
    seed=st.integers(0, 10_000), d=st.integers(1, 15), scale=st.sampled_from([1e-3, 1.0, 1e4, 1e8])
)
def test_fast_rolling_matches_pandas(name, seed, d, scale):
    """Our whole-array rolling functions must give pandas' answers, NaN for NaN."""
    x, y = _panel(seed) * scale, _panel(seed + 7)
    spec = dsl.OPS[name]
    got = spec.func(*([x, y][: spec.n_series]), d)
    want = ROLLING[name](x, y, d)
    if name == "ts_corr":
        want = want.clip(-1, 1)  # pandas can give values like 1.0000000000000002
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-7, atol=1e-9 * scale)


def test_rolling_rank_handles_ties():
    x = pd.DataFrame({"A": [1.0, 1.0, 1.0, 2.0, 1.0]})
    got = dsl.OPS["ts_rank"].func(x, 3)
    want = x.rolling(3, min_periods=3).rank(pct=True)
    pd.testing.assert_frame_equal(got, want)
