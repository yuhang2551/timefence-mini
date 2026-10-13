import json

import pandas as pd

from timefence.cli import main
from timefence.data.synthetic import synthetic_store
from timefence.experiments.report import write_comparison
from timefence.experiments.runner import RunConfig, choose_split, run_experiment


def test_split_uses_fixed_dates_when_the_data_covers_them():
    cal = pd.bdate_range("2016-01-04", "2025-06-30")
    split = choose_split(cal)
    assert str(split.train) == "2018-01-01~2021-12-31"
    assert str(split.test) == "2023-01-01~2024-12-31"
    short = pd.bdate_range("2018-01-02", periods=1000)
    auto = choose_split(short)
    assert auto.train.start == short[250] and auto.test.end == short[-1]


def test_end_to_end_handcrafted(tmp_path):
    store = synthetic_store(n_stocks=80, n_days=700, reversal=0.3)
    cfg = RunConfig(method="handcrafted", top_n=40, top_k=8)
    summary = run_experiment(store, cfg, tmp_path / "hc", {"source": "synthetic"})
    assert summary["mining"]["counts"]["accepted"] >= 1
    assert any(f["kept"] for f in summary["factors"])
    assert summary["backtest"]["avg_holdings"] > 5
    report = (tmp_path / "hc" / "report.md").read_text()
    assert "模拟行情" in report and "组合回测" in report
    saved = json.loads((tmp_path / "hc" / "summary.json").read_text())
    assert saved["split"]["train"] == summary["split"]["train"]
    nav = pd.read_csv(tmp_path / "hc" / "nav.csv", index_col=0, parse_dates=True)
    assert nav.index[0] >= pd.Timestamp(summary["split"]["test"].split("~")[0])


def test_offline_agent_and_comparison(tmp_path):
    store = synthetic_store(n_stocks=60, n_days=600, reversal=0.3)
    run_experiment(
        store, RunConfig(method="agent", offline=True, rounds=2, top_n=40, top_k=8), tmp_path / "a"
    )
    run_experiment(
        store, RunConfig(method="random", random_trials=6, top_n=40, top_k=8), tmp_path / "r"
    )
    report = (tmp_path / "a" / "report.md").read_text()
    assert "不是大模型" in report
    text = write_comparison([tmp_path / "a", tmp_path / "r"], tmp_path / "cmp.md")
    assert "多智能体挖掘" in text and "随机表达式基线" in text


def test_cli(tmp_path, capsys):
    data, run = tmp_path / "data", tmp_path / "run"
    main(
        [
            "data",
            "synth",
            "--out",
            str(data),
            "--stocks",
            "60",
            "--days",
            "600",
            "--reversal",
            "0.3",
        ]
    )
    main(["run", "--data", str(data), "--out", str(run), "--method", "random", "--trials", "5",
          "--top-n", "40", "--top-k", "8"])  # fmt: skip
    out = capsys.readouterr().out
    assert "synthetic market saved" in out and "report:" in out
    assert json.loads((data / "meta.json").read_text())["source"] == "synthetic"
