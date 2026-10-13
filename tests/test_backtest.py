import pandas as pd
import pytest

from tests.conftest import make_store
from timefence.backtest.broker import Broker, MarketTape
from timefence.backtest.costs import ZERO_COSTS, CostModel
from timefence.backtest.engine import BacktestConfig, run_backtest
from timefence.backtest.metrics import equal_weight_returns, max_drawdown, performance
from timefence.backtest.rules import limit_pct, round_price, stamp_duty_rate
from timefence.core.types import Window
from timefence.data.universe import UniverseConfig
from timefence.strategy.model import FactorModel, FactorSpec
from timefence.strategy.topk import TopKStrategy


class Scripted:
    """Returns pre-set weights on given decision days and records what it was shown."""

    lookback = 5

    def __init__(self, plan):
        self.plan = {pd.Timestamp(k): v for k, v in plan.items()}
        self.seen = []

    def target_weights(self, data, holdings):
        today = data.calendar[-1]
        self.seen.append((today, data.as_of, data.panel("close").index.max()))
        w = self.plan.get(today)
        return None if w is None else pd.Series(w, dtype=float)


def _flat(codes, days, price=10.0, volume=1e6):
    return [(d, c, price, price, volume, 1.0) for d in days for c in codes]


def test_decide_at_close_trade_next_open_with_costs():
    days = pd.bdate_range("2021-03-01", periods=4)
    store = make_store(_flat(["600000", "600001"], days))
    strat = Scripted({days[0]: {"600000": 1.0}})
    res = run_backtest(
        store,
        strat,
        Window(days[0], days[-1]),
        BacktestConfig(initial_cash=1e6, rebalance_every=10),
    )
    fill = res.fills.iloc[0]
    assert fill["date"] == days[1] and fill["side"] == "buy"
    assert fill["price"] == pytest.approx(10.005)  # 5 bp slippage
    assert fill["shares"] == 99_900  # whole lots that fit in the cash
    assert fill["fee"] == pytest.approx(fill["value"] * 0.00025 + fill["value"] * 0.00001)
    assert strat.seen[0][0] == days[0] and strat.seen[0][2] == days[0]  # nothing after the close
    assert res.nav.iloc[0] == 1.0 and res.nav.iloc[-1] < 1.0  # paid costs, prices flat


def test_strategy_never_sees_the_day_it_trades_on(store):
    days = store.calendar[300:340]
    strat = Scripted({})
    run_backtest(store, strat, Window(days[0], days[-1]), BacktestConfig(rebalance_every=3))
    for today, as_of, latest in strat.seen:
        assert latest == today
        assert as_of.normalize() == today
    assert [s[0] for s in strat.seen] == list(days[:-1:3])


def _two_day_store(open1, board="main", day0="2021-03-01", **kw):
    d0 = pd.Timestamp(day0)
    days = pd.bdate_range(d0, periods=3)
    code = "300001" if board == "chinext" else "600000"
    rows = [
        (days[0], code, 10.0, 10.0, 1e6, 1.0),
        (days[1], code, open1, open1, 1e6, 1.0),
        (days[2], code, open1, open1, 1e6, 1.0),
    ]
    return make_store(rows, **kw), days, code


def _try_buy(store, days, code):
    tape = MarketTape(store, days)
    broker = Broker(1e6, ZERO_COSTS)
    broker.buy(tape.quotes(days[1]), code, 5e5)
    return broker


def test_limit_up_open_blocks_buying():
    store, days, code = _two_day_store(11.0)
    broker = _try_buy(store, days, code)
    assert not broker.fills and broker.blocked[0]["reason"] == "limit_up"
    store, days, code = _two_day_store(10.99)
    assert _try_buy(store, days, code).fills


def test_chinext_limit_widened_in_august_2020():
    store, days, code = _two_day_store(11.0, board="chinext", day0="2020-08-03")
    assert _try_buy(store, days, code).blocked  # 10% limit then
    store, days, code = _two_day_store(11.0, board="chinext", day0="2020-09-01")
    assert _try_buy(store, days, code).fills  # 20% limit now
    assert limit_pct("chinext", pd.Timestamp("2020-08-21")) == 0.10
    assert limit_pct("chinext", pd.Timestamp("2020-08-24")) == 0.20
    assert limit_pct("star", pd.Timestamp("2019-08-01")) == 0.20


def test_limit_down_open_blocks_selling():
    store, days, code = _two_day_store(9.0)
    tape = MarketTape(store, days)
    broker = Broker(1e6, ZERO_COSTS)
    broker.shares[code] = 1000
    broker.sell(tape.quotes(days[1]), code, 1e9, everything=True)
    assert broker.shares[code] == 1000 and broker.blocked[0]["reason"] == "limit_down"


def test_suspended_stock_cannot_trade():
    days = pd.bdate_range("2021-03-01", periods=3)
    rows = [
        (days[0], "600000", 10.0, 10.0, 1e6, 1.0),
        (days[2], "600000", 10.0, 10.0, 1e6, 1.0),
        (days[0], "600001", 10.0, 10.0, 1e6, 1.0),
        (days[1], "600001", 10.0, 10.0, 1e6, 1.0),
        (days[2], "600001", 10.0, 10.0, 1e6, 1.0),
    ]
    store = make_store(rows)
    broker = _try_buy(store, days, "600000")
    assert broker.blocked[0]["reason"] == "suspended"


def test_t_plus_one():
    days = pd.bdate_range("2021-03-01", periods=3)
    store = make_store(_flat(["600000"], days))
    tape = MarketTape(store, days)
    broker = Broker(1e6, ZERO_COSTS)
    q = tape.quotes(days[1])
    broker.buy(q, "600000", 5e5)
    broker.sell(q, "600000", 1e9, everything=True)
    assert broker.blocked[-1]["reason"] == "t_plus_1"
    q2 = tape.quotes(days[2])
    broker.start_day(q2)
    broker.sell(q2, "600000", 1e9, everything=True)
    assert "600000" not in broker.shares


def test_split_changes_shares_not_value():
    days = pd.bdate_range("2021-03-01", periods=3)
    rows = [
        (days[0], "600000", 10.0, 10.0, 1e6, 1.0),
        (days[1], "600000", 5.0, 5.0, 1e6, 2.0),
        (days[2], "600000", 5.0, 5.0, 1e6, 2.0),
    ]
    store = make_store(rows)
    tape = MarketTape(store, days)
    broker = Broker(0.0, ZERO_COSTS)
    broker.shares["600000"] = 1000
    before = broker.nav(tape.quotes(days[0]))
    q1 = tape.quotes(days[1])
    broker.start_day(q1)
    assert broker.shares["600000"] == 2000
    assert broker.nav(q1) == pytest.approx(before)
    # Ex-date limits use the adjusted reference price: 10 * 1 / 2 = 5, so 5.5 is the cap.
    assert q1.up[0] == pytest.approx(5.5)


def test_delisted_holding_is_sold_at_its_last_close():
    days = pd.bdate_range("2021-03-01", periods=5)
    rows = _flat(["600000"], days) + [(d, "600001", 8.0, 8.0, 1e6, 1.0) for d in days[:2]]
    store = make_store(rows, delist={"600001": days[1]})
    strat = Scripted({days[0]: {"600001": 0.5, "600000": 0.5}})
    res = run_backtest(
        store,
        strat,
        Window(days[0], days[-1]),
        BacktestConfig(initial_cash=1e6, rebalance_every=10, costs=ZERO_COSTS),
    )
    exits = res.fills[res.fills["reason"] == "delisted"]
    assert list(exits["code"]) == ["600001"] and exits.iloc[0]["date"] == days[2]
    assert exits.iloc[0]["price"] == 8.0
    assert res.summary()["delisted_exits"] == 1


def test_costs():
    costs = CostModel()
    assert costs.fees(10_000, "buy", pd.Timestamp("2022-01-04")) == pytest.approx(
        5 + 0.1
    )  # minimum commission
    sell_2022 = costs.fees(1e6, "sell", pd.Timestamp("2022-01-04"))
    sell_2024 = costs.fees(1e6, "sell", pd.Timestamp("2024-01-04"))
    assert sell_2022 - sell_2024 == pytest.approx(1e6 * 0.0005)
    assert stamp_duty_rate(pd.Timestamp("2023-08-25")) == 0.001
    assert stamp_duty_rate(pd.Timestamp("2023-08-28")) == 0.0005
    assert ZERO_COSTS.fees(1e6, "sell", pd.Timestamp("2022-01-04")) == 0


def test_round_price_half_up():
    assert round_price(10.005) == pytest.approx(10.01)
    assert round_price(11.0) == pytest.approx(11.0)
    assert round_price(5.444) == pytest.approx(5.44)


def test_performance_numbers():
    nav = pd.Series([1.0, 1.1, 0.99, 1.2], index=pd.bdate_range("2021-01-04", periods=4))
    assert max_drawdown(nav) == pytest.approx(0.99 / 1.1 - 1)
    stats = performance(nav, benchmark=nav)
    assert stats["total_return"] == pytest.approx(0.2)
    assert stats["tracking_error"] == pytest.approx(0.0)


def test_equal_weight_benchmark_uses_yesterdays_universe(store, universe):
    bench = equal_weight_returns(store, universe)
    day = store.calendar[400]
    members = universe.loc[store.calendar[399]]
    expected = store.panel("returns").loc[day, members[members].index].mean()
    assert bench.loc[day] == pytest.approx(expected)


def test_topk_trades_the_planted_signal(store, universe):
    cfg = UniverseConfig(top_n=40)
    model = FactorModel([FactorSpec("-ts_sum(returns, 5)", 1)])
    window = Window(store.calendar[300], store.calendar[-1])
    bench = equal_weight_returns(store, universe)
    res = run_backtest(
        store,
        TopKStrategy(model, top_k=8, universe=cfg),
        window,
        BacktestConfig(costs=ZERO_COSTS),
        bench,
    )
    stats = res.summary()
    assert stats["excess_annual_return"] > 0.1
    assert stats["avg_holdings"] == pytest.approx(8, abs=1)
    tight = run_backtest(
        store,
        TopKStrategy(model, top_k=8, buffer=0.0, universe=cfg),
        window,
        BacktestConfig(costs=ZERO_COSTS),
    )
    loose = run_backtest(
        store,
        TopKStrategy(model, top_k=8, buffer=1.0, universe=cfg),
        window,
        BacktestConfig(costs=ZERO_COSTS),
    )
    assert loose.summary()["annual_turnover"] < tight.summary()["annual_turnover"]


def test_factor_model_direction_and_coverage(ctx):
    universe = ctx.universe
    up = FactorModel([FactorSpec("-ts_sum(returns, 5)", 1)]).score(ctx.field, universe)
    down = FactorModel([FactorSpec("-ts_sum(returns, 5)", -1)]).score(ctx.field, universe)
    pd.testing.assert_frame_equal(up, -down)
    assert up.where(~universe).isna().all().all()  # only universe stocks get a score
    assert FactorModel([FactorSpec("ts_mean(close, 60)"), FactorSpec("returns")]).lookback == 60
    with pytest.raises(ValueError):
        FactorModel([])
