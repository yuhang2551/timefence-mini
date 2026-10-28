import json

import pandas as pd
import pytest

from timefence.agents.llm import PLACEHOLDER_KEY
from timefence.cli import _load_dotenv, main
from timefence.data.synthetic import synthetic_store
from timefence.experiments.report import write_comparison
from timefence.experiments.runner import RunConfig, choose_split, holdout_boundary, run_experiment


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


def test_holdout_comes_after_everything_else(tmp_path):
    store = synthetic_store(n_stocks=60, n_days=800, reversal=0.3)
    start = store.calendar[-120]
    cfg = RunConfig(
        method="handcrafted", top_n=40, top_k=8, holdout=True, holdout_start=str(start.date())
    )
    summary = run_experiment(store, cfg, tmp_path / "h")
    test_end = pd.Timestamp(summary["split"]["test"].split("~")[1])
    assert test_end < start
    assert summary["holdout"]["window"].startswith(str(start.date()))
    report = (tmp_path / "h" / "report.md").read_text()
    assert "最终留出期" in report and f"留出期：{start.date()} 起（运行时指定）" in report


def test_holdout_starts_where_the_provider_stops_knowing(monkeypatch):
    monkeypatch.delenv("K2_MODEL", raising=False)
    k2 = holdout_boundary(RunConfig())
    assert (k2["start"], k2["basis"]) == ("2026-09-03", "release")
    monkeypatch.setenv("K2_MODEL", "k2-horizon")  # the same model under its host's name
    assert holdout_boundary(RunConfig())["model"] == "k2-horizon"
    with pytest.raises(ValueError, match="--holdout-start"):
        holdout_boundary(RunConfig(model="k2-horizon-32b"))
    claude = holdout_boundary(RunConfig(provider="claude"))
    assert (claude["start"], claude["basis"]) == ("2026-07-01", "cutoff")
    early = holdout_boundary(RunConfig(provider="claude", holdout_start="2026-01-01"))
    assert early["basis"] == "manual" and early["before_bound"]
    with pytest.raises(ValueError, match="--holdout-start"):
        holdout_boundary(RunConfig(provider="deepseek", model="deepseek-pro"))
    other = holdout_boundary(
        RunConfig(provider="deepseek", model="deepseek-pro", holdout_start="2026-11-01")
    )
    assert other["knowledge_bound"] is None and not other["before_bound"]


def test_comparison_flags_runs_with_different_splits(tmp_path):
    for name, test in [("a", "2023-01-01~2024-12-31"), ("b", "2023-01-01~2025-06-30")]:
        (tmp_path / name).mkdir()
        summary = {
            "run_id": name, "config": {"method": "random"},
            "split": {"train": "t", "valid": "v", "test": test},
            "mining": {"scored": 0, "counts": {"accepted": 0}}, "factors": [], "backtest": None,
        }  # fmt: skip
        (tmp_path / name / "summary.json").write_text(json.dumps(summary))
    assert "不能直接比较" in write_comparison([tmp_path / "a", tmp_path / "b"], tmp_path / "c.md")
    assert "不能直接比较" not in write_comparison([tmp_path / "a"], tmp_path / "c.md")


def test_agent_run_without_a_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("K2_BASE_URL", raising=False)
    monkeypatch.delenv("K2_API_KEY", raising=False)
    store = synthetic_store(n_stocks=30, n_days=300)
    with pytest.raises(RuntimeError, match="K2_BASE_URL.*--offline"):
        run_experiment(store, RunConfig(method="agent"), tmp_path / "x")
    monkeypatch.setenv("K2_BASE_URL", "https://k2.example.com/v1")
    with pytest.raises(RuntimeError, match="K2_API_KEY"):
        run_experiment(store, RunConfig(method="agent"), tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_agent_run_with_the_placeholder_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("K2_BASE_URL", "https://k2.example.com/v1")
    monkeypatch.setenv("K2_API_KEY", PLACEHOLDER_KEY)
    store = synthetic_store(n_stocks=30, n_days=300)
    with pytest.raises(RuntimeError, match="placeholder.*--offline"):
        run_experiment(store, RunConfig(method="agent"), tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_dotenv_fills_only_unset_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nTF_TEST_A=from-file\nexport TF_TEST_B='quoted'\nTF_TEST_C=from-file\n"
    )
    monkeypatch.delenv("TF_TEST_A", raising=False)
    monkeypatch.delenv("TF_TEST_B", raising=False)
    monkeypatch.setenv("TF_TEST_C", "from-shell")
    _load_dotenv(env)
    import os

    assert os.environ["TF_TEST_A"] == "from-file"
    assert os.environ["TF_TEST_B"] == "quoted"
    assert os.environ["TF_TEST_C"] == "from-shell"
    _load_dotenv(tmp_path / "missing.env")  # no file is fine
