"""Rolling-window statistics on whole 2-D arrays (rows = days, columns = stocks).

pandas computes rolling windows one column at a time, which is slow with thousands of
stocks. These functions handle every column at once and match pandas' results:
a window's value is NaN unless it holds at least `min_periods` valid numbers (by default
the whole window must be valid). Each value depends only on its own row and earlier rows.
"""

from __future__ import annotations

import numpy as np

CHUNK_ELEMENTS = 20_000_000  # bounds temporary memory for the sliding-window functions


def _windowed_sum(a: np.ndarray, d: int) -> np.ndarray:
    """Sum of the last d rows (fewer at the top). `a` must not contain NaN."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    out[d:] -= c[:-d]
    return out


def _first_valid(a: np.ndarray) -> np.ndarray:
    """Each column's first non-NaN value (0 if none). Used to centre data for precision;
    it only looks at the earliest rows, so it never brings in later information."""
    valid = ~np.isnan(a)
    idx = valid.argmax(axis=0)
    ref = a[idx, np.arange(a.shape[1])]
    return np.where(valid.any(axis=0), ref, 0.0)


def _prepare(a: np.ndarray, d: int, min_periods: int | None):
    valid = ~np.isnan(a)
    n = _windowed_sum(valid.astype(float), d)
    need = d if min_periods is None else min_periods
    return valid, n, n >= need


def rolling_sum(a: np.ndarray, d: int, min_periods: int | None = None) -> np.ndarray:
    _, _, ok = _prepare(a, d, min_periods)
    s = _windowed_sum(np.nan_to_num(a, nan=0.0), d)
    return np.where(ok, s, np.nan)


def rolling_mean(a: np.ndarray, d: int, min_periods: int | None = None) -> np.ndarray:
    _, n, ok = _prepare(a, d, min_periods)
    ref = _first_valid(a)
    s = _windowed_sum(np.nan_to_num(a - ref, nan=0.0), d)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, s / n + ref, np.nan)


def rolling_cov(x: np.ndarray, y: np.ndarray, d: int, ddof: int = 1) -> np.ndarray:
    both = ~np.isnan(x) & ~np.isnan(y)
    n = _windowed_sum(both.astype(float), d)
    xc = np.where(both, x - _first_valid(np.where(both, x, np.nan)), 0.0)
    yc = np.where(both, y - _first_valid(np.where(both, y, np.nan)), 0.0)
    sx, sy = _windowed_sum(xc, d), _windowed_sum(yc, d)
    sxy = _windowed_sum(xc * yc, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (sxy - sx * sy / n) / (n - ddof)
    return np.where((n >= d) & (n > ddof), cov, np.nan)


def rolling_std(a: np.ndarray, d: int, ddof: int = 1) -> np.ndarray:
    var = rolling_cov(a, a, d, ddof)
    return np.sqrt(np.clip(var, 0.0, None))


def rolling_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    both = ~np.isnan(x) & ~np.isnan(y)
    xm, ym = np.where(both, x, np.nan), np.where(both, y, np.nan)
    cov = rolling_cov(xm, ym, d)
    sx, sy = rolling_std(xm, d), rolling_std(ym, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (sx * sy)
    # Flat windows have no correlation; tiny float noise must not turn them into +-1.
    flat = (sx <= 1e-12 * (1 + np.abs(rolling_mean(xm, d)))) | (
        sy <= 1e-12 * (1 + np.abs(rolling_mean(ym, d)))
    )
    corr = np.where(flat, np.nan, corr)
    return np.clip(corr, -1.0, 1.0)


def _sliding(a: np.ndarray, d: int, reduce) -> np.ndarray:
    """Apply `reduce(windows) -> (rows, cols)` to every full window, chunk by chunk."""
    out = np.full(a.shape, np.nan)
    if a.shape[0] < d:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(a, d, axis=0)  # (rows-d+1, cols, d)
    step = max(1, CHUNK_ELEMENTS // max(1, a.shape[1] * d))
    for start in range(0, windows.shape[0], step):
        chunk = windows[start : start + step]
        out[d - 1 + start : d - 1 + start + len(chunk)] = reduce(chunk)
    return out


def rolling_max(a: np.ndarray, d: int) -> np.ndarray:
    return _sliding(a, d, lambda w: w.max(axis=-1))  # NaN anywhere in the window -> NaN


def rolling_min(a: np.ndarray, d: int) -> np.ndarray:
    return _sliding(a, d, lambda w: w.min(axis=-1))


def rolling_rank_pct(a: np.ndarray, d: int) -> np.ndarray:
    """Percentile rank of each row's value within its window (ties get the average rank)."""

    def reduce(w: np.ndarray) -> np.ndarray:
        last = w[..., -1:]
        less = (w < last).sum(axis=-1)
        equal = (w == last).sum(axis=-1)
        rank = (less + (equal + 1) / 2.0) / d
        return np.where(np.isnan(w).any(axis=-1), np.nan, rank)

    return _sliding(a, d, reduce)
