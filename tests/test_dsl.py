import numpy as np
import pandas as pd
import pytest

from timefence.core.errors import ExpressionError
from timefence.factors import dsl, ops


@pytest.mark.parametrize(
    "text, lookback",
    [
        ("close", 0),
        ("returns", 1),
        ("ts_mean(close, 20)", 20),
        ("rank(ts_mean(volume, 5) / ts_mean(volume, 60))", 60),
        ("ts_corr(close, volume, 10) * -1", 10),
        ("ts_mean(delta(close, 5), 20)", 25),
        ("max(close - open, 0)", 0),
        ("ts_std(returns, 20)", 21),
    ],
)
def test_valid_expressions(text, lookback):
    parsed = dsl.parse(text)
    assert parsed.lookback == lookback
    assert dsl.parse(parsed.canonical).canonical == parsed.canonical


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "empty"),
        ("close +", "syntax"),
        ("1 + 2", "data field"),
        ("label", "unknown field"),
        ("future_return", "unknown field"),
        ("delay(close, -1)", "window"),
        ("delay(close, 0)", "window"),
        ("ts_mean(close, 2.5)", "literal integer"),
        ("ts_mean(close, volume)", "literal integer"),
        ("ts_mean(close, 251)", "outside"),
        ("ts_mean(close)", "takes 2 arguments"),
        ("rank(close, 5)", "takes 1 arguments"),
        ("ts_mean(x=close, d=5)", "keyword"),
        ("close.shift(-1)", "unknown operator"),
        ("__import__('os')", "unknown operator"),
        ("close[1]", "not allowed"),
        ("lambda: close", "not allowed"),
        ("close if close > 0 else open", "not allowed"),
        ("close ** 2", "not allowed"),
        ("ts_mean(5, 10)", "data fields"),
        ("close / 0", "division by zero"),
        ("rank(3)", "data fields"),
        ("close + True", "not a finite number"),
        ("close + 1e999", "not a finite number"),
        ("ts_mean(ts_mean(ts_mean(close, 200), 50), 60)", "history"),
    ],
)
def test_invalid_expressions(text, message):
    with pytest.raises(ExpressionError, match=message):
        dsl.parse(text)


def test_size_limits():
    deep = "close"
    for _ in range(dsl.MAX_DEPTH):
        deep = f"abs({deep})"
    with pytest.raises(ExpressionError, match="depth"):
        dsl.parse(deep)
    wide = " + ".join(["close"] * (dsl.MAX_NODES + 1))
    with pytest.raises(ExpressionError, match="nodes"):
        dsl.parse(wide)


def test_language_reference_lists_every_operator():
    text = dsl.describe_language()
    for name in dsl.OPS:
        assert f"{name}(" in text
    for f in dsl.FIELDS:
        assert f in text


def _fields(seed=0, n=80, m=6):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"S{i}" for i in range(m)]
    data = {
        f: pd.DataFrame(rng.lognormal(size=(n, m)), index=idx, columns=cols) for f in dsl.FIELDS
    }
    return data


def test_evaluate_matches_operators():
    data = _fields()
    got = dsl.evaluate(dsl.parse("rank(ts_mean(close, 5) - open)"), data.__getitem__)
    want = ops.rank(ops.ts_mean(data["close"], 5) - data["open"])
    pd.testing.assert_frame_equal(got, want)


def test_cross_sectional_ops_only_see_the_universe():
    data = _fields()
    universe = pd.DataFrame(True, index=data["close"].index, columns=data["close"].columns)
    universe.iloc[:, 0] = False
    parsed = dsl.parse("rank(close)")
    masked = dsl.evaluate(parsed, data.__getitem__, universe)
    assert masked.iloc[:, 0].isna().all()
    # Changing a stock outside the universe must not move anyone else's rank.
    data2 = dict(data)
    data2["close"] = data["close"].copy()
    data2["close"].iloc[:, 0] *= 1000
    masked2 = dsl.evaluate(parsed, data2.__getitem__, universe)
    pd.testing.assert_frame_equal(masked, masked2)


def test_division_by_zero_constant_is_rejected_and_inf_removed():
    data = _fields()
    with pytest.raises(ExpressionError, match="zero constant"):
        dsl.evaluate(dsl.parse("close + 1 / (2 - 2)"), data.__getitem__)
    out = dsl.evaluate(dsl.parse("close / (open - open)"), data.__getitem__)
    assert not np.isinf(out.to_numpy()).any()
