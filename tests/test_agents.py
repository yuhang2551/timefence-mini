import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from timefence.agents.graph import ResearchAgent
from timefence.agents.llm import (
    PLACEHOLDER_KEY,
    Budget,
    OpenAICompatibleLLM,
    ScriptedLLM,
    parse_json,
)
from timefence.agents.memory import ResearchMemory
from timefence.agents.tools import FactorLab
from timefence.core.errors import BudgetExceeded, LeakError
from timefence.core.types import Window
from timefence.data.universe import UniverseConfig
from timefence.factors.evaluate import ResearchContext


@pytest.fixture(scope="module")
def train(store):
    cal = store.calendar
    window = Window(cal[260], cal[450])
    short = store.slice_until(window.end)
    return ResearchContext(short, UniverseConfig(top_n=40).build(short)), window


def _lab(train):
    ctx, window = train
    return FactorLab(ctx, window)


def test_lab_refuses_a_context_with_future_data(ctx, train):
    _, window = train
    with pytest.raises(LeakError, match="past the training window"):
        FactorLab(ctx, window)  # ctx holds the full history


def test_lab_verdicts(train):
    lab = _lab(train)
    assert lab.judge("close +").status == "invalid"
    first = lab.judge("-ts_sum(returns,5)")
    assert first.status == "accepted" and first.canonical == "-ts_sum(returns, 5)"
    assert lab.judge("-ts_sum(returns, 5)").status == "duplicate"
    near_copy = lab.judge("-ts_mean(returns, 5) * 2")
    assert near_copy.status == "rejected" and "overlaps" in near_copy.reason
    weak = lab.judge("ts_mean(close, 3) / ts_mean(close, 3)")
    assert weak.status in ("rejected", "invalid")


def test_memory(tmp_path):
    mem = ResearchMemory(tmp_path / "m.sqlite")
    metrics = {"rank_ic": 0.05, "rank_icir": 0.4, "turnover": 0.5}
    mem.add_trial("r1", 1, "a", "accepted", canonical="a", metrics=metrics)
    mem.add_trial(
        "r1", 1, "b", "rejected", canonical="b", reason="|RankIC| too small", metrics=metrics
    )
    mem.add_trial("r1", 1, "c(", "invalid", reason="syntax error")
    mem.add_trial("r2", 1, "a", "rejected", canonical="a", metrics=metrics)
    mem.add_note("r1", 1, "reviewer", "short windows work better")
    assert mem.status_of("r1", "a") == "accepted"
    assert mem.status_of("r2", "a") == "rejected"  # runs do not see each other
    assert mem.status_of("r1", "zzz") is None
    assert mem.counts("r1") == {"invalid": 1, "duplicate": 0, "rejected": 1, "accepted": 1}
    digest = mem.digest("r1")
    assert "a  RankIC +0.0500" in digest and "too small" in digest and "short windows" in digest
    mem.close()
    again = ResearchMemory(tmp_path / "m.sqlite")
    assert again.trials("r1", "accepted")[0]["metrics"] == metrics
    with pytest.raises(ValueError):
        again.add_trial("r1", 1, "x", "maybe")


class FakeTeam:
    """Plays the three roles. The first researcher reply is unusable, to exercise the retry."""

    def __init__(self):
        self.researcher_calls = 0

    def __call__(self, messages):
        system, user = messages[0]["content"], messages[1]["content"]
        if system.startswith("You lead"):
            return json.dumps({"theme": "short-term reversal", "brief": "Look at recent losers."})
        if system.startswith("You are a quantitative researcher"):
            self.researcher_calls += 1
            if self.researcher_calls == 1:
                assert "could not be used" not in user
                return json.dumps(
                    {"proposals": [{"expression": "future_return(5)", "hypothesis": "cheat"}]}
                )
            if self.researcher_calls == 2:
                assert "could not be used" in user and "unknown operator" in user
            windows = (5, 3, 10, 20)
            exprs = [
                f"-ts_sum(returns, {windows[(self.researcher_calls + i) % 4]})" for i in range(2)
            ]
            return json.dumps(
                {"proposals": [{"expression": e, "hypothesis": "reversal"} for e in exprs]}
            )
        return json.dumps({"lessons": ["reversal works on this data"], "next": "try volume"})


def test_agent_loop_retries_records_and_never_shows_the_future(train, tmp_path):
    lab = _lab(train)
    mem = ResearchMemory()
    llm = ScriptedLLM(FakeTeam())
    agent = ResearchAgent(
        llm, lab, mem, "t", rounds=3, proposals_per_round=2, transcript=tmp_path / "t.jsonl"
    )
    result = agent.run()
    assert result["stop_reason"] == "finished all rounds"
    assert result["counts"]["invalid"] == 1 and result["counts"]["accepted"] >= 1
    assert len(mem.notes("t")) == 6  # one lesson plus one "next" per round
    roles = [json.loads(line)["role"] for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert roles[:5] == ["planner", "researcher", "researcher", "reviewer", "planner"]
    # Nothing the model was shown mentions a date after the training window.
    _, window = train
    for messages in llm.log:
        for m in messages:
            for d in re.findall(r"\d{4}-\d{2}-\d{2}", m["content"]):
                assert pd.Timestamp(d) <= window.end


def test_budget_stops_the_run_cleanly(train):
    llm = ScriptedLLM(FakeTeam(), budget=Budget(max_calls=5))
    agent = ResearchAgent(llm, _lab(train), ResearchMemory(), "t", rounds=10)
    result = agent.run()
    assert "limit of 5 model calls" in result["stop_reason"]
    assert result["model_calls"] == 5


def test_garbage_replies_do_not_crash_the_loop(train):
    llm = ScriptedLLM(lambda m: "I think you should buy low and sell high.")
    result = ResearchAgent(llm, _lab(train), ResearchMemory(), "t", rounds=2).run()
    assert result["counts"]["accepted"] == 0
    assert result["stop_reason"] == "finished all rounds"


def test_parse_json():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("[1, 2]")


def _fake_client(calls):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_http_client_caches_and_counts(tmp_path, monkeypatch):
    calls = []
    llm = OpenAICompatibleLLM(cache_dir=tmp_path, budget=Budget(max_calls=1))
    monkeypatch.setattr(llm, "_connect", lambda: _fake_client(calls))
    messages = [{"role": "user", "content": "hi, reply in json"}]
    first = llm.complete(messages)
    second = llm.complete(messages)  # the budget is spent, but this one is cached
    assert first.text == second.text == '{"ok": true}'
    assert not first.cached and second.cached
    assert len(calls) == 1 and calls[0]["response_format"] == {"type": "json_object"}
    assert llm.budget.calls == 1 and llm.budget.tokens == 120
    with pytest.raises(BudgetExceeded):
        llm.complete([{"role": "user", "content": "something new"}])


def test_missing_key_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    llm = OpenAICompatibleLLM(cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is not set"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_placeholder_key_is_refused_before_any_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", PLACEHOLDER_KEY)
    llm = OpenAICompatibleLLM(cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="placeholder"):
        llm.complete([{"role": "user", "content": "hi"}])
    assert llm.budget.calls == 0


def test_env_example_ships_the_placeholder():
    example = Path(__file__).parents[1] / ".env.example"
    assert f"DEEPSEEK_API_KEY={PLACEHOLDER_KEY}" in example.read_text().splitlines()
