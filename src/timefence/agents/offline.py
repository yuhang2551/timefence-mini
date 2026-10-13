"""An offline stand-in for the model, so the whole loop runs without an API key.

It fills factor templates for each theme with random windows. It is not a language model and
its output says nothing about what an LLM would do; it exists for tests and demos only.
"""

from __future__ import annotations

import json
import random
import re

from timefence.agents.llm import Messages, ScriptedLLM
from timefence.agents.prompts import THEMES

TEMPLATES = {
    "short-term reversal": [
        "-ts_sum(returns, {s})",
        "-pct_change(close, {s})",
        "-delta(close, {s}) / ts_std(close, {m})",
        "-rank(ts_mean(returns, {s}))",
    ],
    "medium-term momentum": [
        "delay(pct_change(close, {m}), {s})",
        "ts_mean(returns, {l})",
        "pct_change(close, {l}) - pct_change(close, {s})",
    ],
    "volatility and downside risk": [
        "-ts_std(returns, {m})",
        "-ts_max(abs(returns), {m})",
        "-ts_std(returns, {s}) / ts_std(returns, {l})",
    ],
    "liquidity and trading activity": [
        "-log(ts_mean(amount, {m}))",
        "-ts_mean(volume, {s}) / ts_mean(volume, {l})",
        "-ts_std(amount, {m}) / ts_mean(amount, {m})",
    ],
    "price-volume correlation": [
        "-ts_corr(close, volume, {m})",
        "-ts_corr(returns, volume, {m})",
        "-ts_corr(rank(close), rank(volume), {m})",
    ],
    "where the close sits within the day's range": [
        "ts_mean((close - low) / (high - low), {s})",
        "-(close / vwap - 1)",
        "-ts_mean((high - low) / close, {m})",
    ],
    "volume shocks and attention": [
        "-ts_zscore(volume, {m})",
        "-ts_max(volume, {s}) / ts_mean(volume, {l})",
    ],
    "trend smoothness and quality": [
        "pct_change(close, {m}) / ts_std(returns, {m})",
        "ts_corr(close, ts_sum(returns, {s}), {m})",
    ],
}
SHORT, MEDIUM, LONG = (3, 5, 10), (10, 20, 40), (60, 120)


def _respond(messages: Messages, rng: random.Random) -> str:
    system, user = messages[0]["content"], messages[-1]["content"]
    if system.startswith("You lead"):
        n = int(re.search(r"Round (\d+)", user).group(1))
        theme = THEMES[(n - 1) % len(THEMES)]
        return json.dumps({"theme": theme, "brief": f"Try simple {theme} signals."})
    if system.startswith("You are a quantitative researcher"):
        theme = re.search(r"Theme: (.*)", user).group(1).strip()
        n = int(re.search(r"Propose (\d+)", user).group(1))
        templates = TEMPLATES.get(theme, sum(TEMPLATES.values(), []))
        proposals = []
        for _ in range(n):
            text = rng.choice(templates).format(
                s=rng.choice(SHORT), m=rng.choice(MEDIUM), l=rng.choice(LONG)
            )
            proposals.append({"expression": text, "hypothesis": f"template for {theme}"})
        return json.dumps({"proposals": proposals})
    accepted = user.count(": ACCEPTED")
    rejected = user.count(": REJECTED")
    return json.dumps(
        {
            "lessons": [f"{accepted} accepted and {rejected} rejected this round."],
            "next": "Try another theme.",
        }
    )


def offline_llm(seed: int = 0) -> ScriptedLLM:
    rng = random.Random(seed)
    return ScriptedLLM(
        respond=lambda m: _respond(m, rng),
        name="offline template stand-in (not a language model)",
    )
