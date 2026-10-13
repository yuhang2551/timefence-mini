"""Score factors: information coefficient, quantile returns, turnover, overlap with others.

Conventions
- A factor value on day t is formed at t's close. The label is the return from buying at
  t+1's open and selling h days later at the open (see `forward_open_returns`).
- Scores are taken every h days so the label periods do not overlap.
- The last h+1 days of a window are dropped (purged): their labels would use prices from
  after the window, which for the training window means peeking into validation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from timefence.core.types import Window
from timefence.data.store import MarketStore, forward_open_returns
from timefence.factors import dsl

TRADING_DAYS = 252
MIN_STOCKS = 20


@dataclass
class FactorReport:
    expression: str
    window: str
    n_dates: int
    coverage: float  # share of universe stocks with a factor value
    ic: float  # mean Pearson correlation with the label
    rank_ic: float  # mean Spearman correlation with the label
    rank_ic_std: float
    rank_icir: float  # rank_ic / rank_ic_std, per period
    rank_ic_tstat: float
    hit_rate: float  # share of periods where rank IC has the factor's sign
    quantile_returns: list[float]  # annualised, quintile 1 (lowest value) .. 5 (highest)
    long_short: float  # annualised Q5 - Q1 after applying the direction
    turnover: float  # share of the top quintile replaced each period
    direction: int  # +1 if high values predict high returns, -1 otherwise
    max_corr: float = 0.0  # largest |rank correlation| with a reference factor
    max_corr_with: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def brief(self) -> str:
        q = ", ".join(f"{v:+.1%}" for v in self.quantile_returns)
        corr = f"{self.max_corr:.2f} with `{self.max_corr_with}`" if self.max_corr_with else "n/a"
        return (
            f"window {self.window}, {self.n_dates} periods, coverage {self.coverage:.0%}\n"
            f"RankIC {self.rank_ic:+.4f} (IC {self.ic:+.4f}), RankICIR {self.rank_icir:+.3f}, "
            f"t {self.rank_ic_tstat:+.2f}, hit rate {self.hit_rate:.0%}, direction {self.direction:+d}\n"
            f"quintile returns (annualised, low to high): {q}; long-short {self.long_short:+.1%}\n"
            f"top-quintile turnover {self.turnover:.0%} per period; max overlap {corr}"
        )


class ResearchContext:
    """Everything needed to compute and score factors on one store.

    Columns are limited to stocks that ever enter the universe, which keeps panels small.
    Build it from `store.slice_until(train_end)` to give an agent a context that physically
    contains no data after the training window.
    """

    def __init__(self, store: MarketStore, universe: pd.DataFrame, horizon: int = 5):
        self.store = store
        self.horizon = horizon
        self.codes = universe.columns[universe.any(axis=0)]
        self.universe = universe[self.codes]
        self.labels = forward_open_returns(store, horizon)[self.codes]
        self._fields: dict[str, pd.DataFrame] = {}
        self._factors: dict[str, pd.DataFrame] = {}

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.store.calendar

    @property
    def last_day(self) -> pd.Timestamp:
        return self.store.calendar[-1]

    def field(self, name: str) -> pd.DataFrame:
        if name not in self._fields:
            self._fields[name] = self.store.panel(name)[self.codes]
        return self._fields[name]

    def compute(self, expression: str | dsl.Parsed) -> tuple[dsl.Parsed, pd.DataFrame]:
        parsed = expression if isinstance(expression, dsl.Parsed) else dsl.parse(expression)
        if parsed.canonical not in self._factors:
            self._factors[parsed.canonical] = dsl.evaluate(parsed, self.field, self.universe)
        return parsed, self._factors[parsed.canonical]

    def sample_dates(self, window: Window) -> pd.DatetimeIndex:
        """Every h-th trading day of the window, minus the last h+1 days (purged)."""
        days = window.clip(self.calendar)
        days = days[: max(0, len(days) - (self.horizon + 1))]
        return days[:: self.horizon]


def _masked(values: pd.DataFrame, universe: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    return values.loc[dates].where(universe.loc[dates]).to_numpy(dtype=float)


def _row_rank(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank(axis=1).to_numpy()


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation of each row of `a` with the same row of `b`, ignoring NaN pairs."""
    valid = ~np.isnan(a) & ~np.isnan(b)
    a = np.where(valid, a, np.nan)
    b = np.where(valid, b, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows with no valid pairs
        am = a - np.nanmean(a, axis=1, keepdims=True)
        bm = b - np.nanmean(b, axis=1, keepdims=True)
        num = np.nansum(am * bm, axis=1)
        den = np.sqrt(np.nansum(am**2, axis=1) * np.nansum(bm**2, axis=1))
        out = num / den
    out[(valid.sum(axis=1) < MIN_STOCKS) | (den == 0)] = np.nan
    return out


def _rank_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(a) & ~np.isnan(b)
    return _row_corr(_row_rank(np.where(valid, a, np.nan)), _row_rank(np.where(valid, b, np.nan)))


def evaluate_factor(
    ctx: ResearchContext,
    expression: str,
    window: Window,
    reference: Mapping[str, pd.DataFrame] | None = None,
) -> FactorReport:
    parsed, values = ctx.compute(expression)
    dates = ctx.sample_dates(window)
    if len(dates) < 10:
        raise ValueError(f"window {window} has only {len(dates)} scoring periods")

    x = _masked(values, ctx.universe, dates)
    y = _masked(ctx.labels, ctx.universe, dates)
    in_universe = ctx.universe.loc[dates].to_numpy()
    coverage = float((~np.isnan(x) & in_universe).sum() / max(1, in_universe.sum()))

    ic = _row_corr(x, y)
    rank_ic = _rank_corr(x, y)
    ric = rank_ic[~np.isnan(rank_ic)]
    mean_ric = float(ric.mean()) if len(ric) else 0.0
    std_ric = float(ric.std(ddof=1)) if len(ric) > 1 else 0.0
    direction = 1 if mean_ric >= 0 else -1
    icir = mean_ric / std_ric if std_ric > 0 else 0.0

    quintiles, turnover = _quintiles(x, y)
    periods_per_year = TRADING_DAYS / ctx.horizon
    q_annual = [float(q * periods_per_year) for q in quintiles]
    long_short = direction * (q_annual[-1] - q_annual[0])

    report = FactorReport(
        expression=parsed.canonical,
        window=str(window),
        n_dates=int(len(ric)),
        coverage=coverage,
        ic=float(np.nanmean(ic)) if np.isfinite(ic).any() else 0.0,
        rank_ic=mean_ric,
        rank_ic_std=std_ric,
        rank_icir=icir,
        rank_ic_tstat=icir * np.sqrt(len(ric)) if len(ric) else 0.0,
        hit_rate=float((np.sign(ric) == direction).mean()) if len(ric) else 0.0,
        quantile_returns=q_annual,
        long_short=long_short,
        turnover=turnover,
        direction=direction,
    )
    if not len(ric):
        report.notes.append("no scoring period had enough stocks with distinct values")
    if reference:
        report.max_corr, report.max_corr_with = _max_overlap(x, reference, ctx, dates)
    return report


def _quintiles(x: np.ndarray, y: np.ndarray) -> tuple[list[float], float]:
    sums = np.zeros(5)
    counts = np.zeros(5)
    tops: list[set[int]] = []
    for row_x, row_y in zip(x, y, strict=True):
        valid = ~np.isnan(row_x) & ~np.isnan(row_y)
        if valid.sum() < MIN_STOCKS:
            continue
        pct = pd.Series(row_x[valid]).rank(pct=True).to_numpy()
        bucket = np.minimum((pct * 5).astype(int), 4)  # 0..4
        for b in range(5):
            sel = bucket == b
            if sel.any():
                sums[b] += row_y[valid][sel].mean()
                counts[b] += 1
        tops.append(set(np.flatnonzero(valid)[bucket == 4]))
    quint = [float(s / c) if c else 0.0 for s, c in zip(sums, counts, strict=True)]
    changes = [1 - len(a & b) / len(b) for a, b in zip(tops, tops[1:], strict=False) if b]
    return quint, float(np.mean(changes)) if changes else 0.0


def _max_overlap(
    x: np.ndarray,
    reference: Mapping[str, pd.DataFrame],
    ctx: ResearchContext,
    dates: pd.DatetimeIndex,
) -> tuple[float, str | None]:
    best, best_name = 0.0, None
    for name, panel in reference.items():
        other = _masked(panel.reindex(index=dates, columns=ctx.codes), ctx.universe, dates)
        corrs = np.abs(_rank_corr(x, other))
        corr = float(np.nanmean(corrs)) if np.isfinite(corrs).any() else 0.0
        if np.isfinite(corr) and corr > best:
            best, best_name = corr, name
    return best, best_name


@dataclass(frozen=True)
class AcceptanceRules:
    """Hard rules a factor must pass before it can join the library. Written in code, so
    no prompt can talk its way around them."""

    min_coverage: float = 0.6
    min_abs_rank_ic: float = 0.02
    min_abs_icir: float = 0.15
    min_hit_rate: float = 0.55
    max_corr: float = 0.7
    max_turnover: float = 0.9  # a random top quintile is replaced ~80% of the time

    def failures(self, r: FactorReport) -> list[str]:
        out = []
        if r.coverage < self.min_coverage:
            out.append(f"coverage {r.coverage:.0%} < {self.min_coverage:.0%}")
        if abs(r.rank_ic) < self.min_abs_rank_ic:
            out.append(f"|RankIC| {abs(r.rank_ic):.4f} < {self.min_abs_rank_ic}")
        if abs(r.rank_icir) < self.min_abs_icir:
            out.append(f"|RankICIR| {abs(r.rank_icir):.3f} < {self.min_abs_icir}")
        if r.hit_rate < self.min_hit_rate:
            out.append(f"hit rate {r.hit_rate:.0%} < {self.min_hit_rate:.0%}")
        if r.max_corr > self.max_corr:
            out.append(f"overlaps `{r.max_corr_with}` (|corr| {r.max_corr:.2f} > {self.max_corr})")
        if r.turnover > self.max_turnover:
            out.append(f"turnover {r.turnover:.0%} > {self.max_turnover:.0%}")
        return out
