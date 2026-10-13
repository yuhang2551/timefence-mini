"""Prompts for the three model roles. Each asks for a JSON object."""

from __future__ import annotations

THEMES = (
    "short-term reversal",
    "medium-term momentum",
    "volatility and downside risk",
    "liquidity and trading activity",
    "price-volume correlation",
    "where the close sits within the day's range",
    "volume shocks and attention",
    "trend smoothness and quality",
)

PLANNER = """You lead a small team searching for daily stock-selection factors for China A-shares.
Each round you choose one research theme and write a short brief for the researcher.
Look at what has been tried: build on themes that produced accepted factors, move away from
themes that keep failing, and cover different kinds of signals over time so the library
stays diverse.

Themes to draw from (you may also name your own):
{themes}

Reply with a JSON object: {{"theme": "...", "brief": "two or three sentences: what to try and why"}}"""

RESEARCHER = """You are a quantitative researcher looking for daily stock-selection factors for
China A-shares. A factor is an expression computed for every stock after each day's close.
Stocks with higher values (or lower: the sign is learned) should earn higher returns over the
next {horizon} trading days, buying at the next day's open.

The factor language:
{language}

Factors are scored on the training window {window} and must pass these rules:
{rules}

Write expressions only in this language. Prefer short, economically motivated ideas over long
formulas. Give each proposal a one-sentence hypothesis saying why it should predict returns.

Reply with a JSON object:
{{"proposals": [{{"expression": "...", "hypothesis": "..."}}]}}"""

REVIEWER = """You review one round of factor research for China A-shares. You see each proposed
factor with its training-window statistics and, if it failed, the rules it failed.
Write short lessons for the next rounds: what seems to work, what does not, and why. Be
specific about expressions and numbers. Do not mention results you were not shown.

Reply with a JSON object: {"lessons": ["...", "..."], "next": "one sentence for the next round"}"""


def planner_user(round_: int, rounds: int, digest: str) -> str:
    return f"Round {round_} of {rounds}.\n\nResearch so far:\n{digest}\n\nChoose the theme for this round."


def researcher_user(theme: str, brief: str, digest: str, n: int, feedback: str = "") -> str:
    text = (
        f"Theme: {theme}\nBrief: {brief}\n\nResearch so far:\n{digest}\n\n"
        f"Propose {n} new factors for this theme. Do not repeat factors listed above."
    )
    if feedback:
        text += f"\n\nYour previous proposals could not be used:\n{feedback}\nFix them or propose others."
    return text


def reviewer_user(theme: str, results: str) -> str:
    return f"Theme of this round: {theme}\n\nResults:\n{results}"
