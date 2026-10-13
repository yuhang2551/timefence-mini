import numpy as np
import pytest

from timefence.core.types import Window
from timefence.data.synthetic import synthetic_store
from timefence.data.universe import liquidity_universe
from timefence.factors.evaluate import (
    AcceptanceRules,
    FactorReport,
    ResearchContext,
    evaluate_factor,
)


def test_planted_reversal_is_found(ctx, window):
    report = evaluate_factor(ctx, "-ts_sum(returns, 5)", window)
    assert report.direction == 1
    assert report.rank_ic > 0.08
    assert report.rank_ic_tstat > 5
    assert report.quantile_returns[-1] > report.quantile_returns[0]
    assert AcceptanceRules().failures(report) == []
    flipped = evaluate_factor(ctx, "ts_sum(returns, 5)", window)
    assert flipped.direction == -1
    assert flipped.rank_ic == pytest.approx(-report.rank_ic)
    assert flipped.long_short == pytest.approx(
        report.long_short, rel=0.05
    )  # bucket edges differ slightly


def test_no_signal_without_a_planted_pattern():
    store = synthetic_store(n_stocks=60, n_days=600, reversal=0.0, seed=11)
    ctx = ResearchContext(store, liquidity_universe(store, top_n=40), horizon=5)
    window = Window(store.calendar[260], store.calendar[-1])
    report = evaluate_factor(ctx, "-ts_sum(returns, 5)", window)
    assert abs(report.rank_ic_tstat) < 3


def test_scoring_dates_are_spaced_and_purged(ctx, window):
    dates = ctx.sample_dates(window)
    cal = ctx.calendar
    positions = cal.get_indexer(dates)
    assert (np.diff(positions) == ctx.horizon).all()
    # The label of the last scoring date ends inside the window.
    assert positions[-1] + 1 + ctx.horizon <= cal.get_loc(window.end)
    longer = ctx.sample_dates(Window(window.start, window.end + np.timedelta64(30, "D")))
    assert longer.equals(dates)  # a window running past the data is cut at the data's end


def test_overlap_with_the_library_is_measured(ctx, window):
    _, existing = ctx.compute("-ts_sum(returns, 5)")
    report = evaluate_factor(ctx, "-ts_mean(returns, 5) * 3", window, reference={"rev5": existing})
    assert report.max_corr == pytest.approx(1.0)
    assert report.max_corr_with == "rev5"
    assert any("overlaps" in f for f in AcceptanceRules().failures(report))


def test_unscorable_factor_gets_a_note(ctx, window):
    report = evaluate_factor(ctx, "close - close", window)
    assert report.n_dates == 0
    assert report.notes
    assert AcceptanceRules().failures(report)


def test_short_window_is_rejected(ctx, store):
    with pytest.raises(ValueError, match="scoring periods"):
        evaluate_factor(ctx, "close", Window(store.calendar[100], store.calendar[120]))


def test_report_brief_is_readable(ctx, window):
    report = evaluate_factor(ctx, "-ts_sum(returns, 5)", window)
    text = report.brief()
    assert "RankIC" in text and "turnover" in text
    assert FactorReport(**report.to_dict()) == report
