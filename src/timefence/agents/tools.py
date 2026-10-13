"""The deterministic tools the research agents work with, bound to the training window."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from timefence.core.errors import ExpressionError, LeakError
from timefence.core.types import Window
from timefence.factors import dsl
from timefence.factors.evaluate import (
    AcceptanceRules,
    FactorReport,
    ResearchContext,
    evaluate_factor,
)


@dataclass
class Verdict:
    expression: str  # as proposed
    canonical: str | None
    status: str  # invalid | duplicate | rejected | accepted
    reason: str | None = None
    report: FactorReport | None = None

    def describe(self) -> str:
        head = f"`{self.canonical or self.expression}`: {self.status.upper()}"
        if self.reason:
            head += f" ({self.reason})"
        return head + (f"\n{self.report.brief()}" if self.report else "")


class FactorLab:
    """Checks, scores and accepts factors using training data only.

    The research context must not reach past the training window: it should be built
    from `store.slice_until(train.end)`. The constructor refuses one that does, so an
    agent cannot be handed future data by mistake.
    """

    def __init__(self, ctx: ResearchContext, window: Window, rules: AcceptanceRules | None = None):
        if ctx.last_day > window.end:
            raise LeakError(
                f"research data runs to {ctx.last_day.date()}, past the training window "
                f"end {window.end.date()}; build the context from store.slice_until(train.end)"
            )
        self.ctx = ctx
        self.window = window
        self.rules = rules or AcceptanceRules()
        self.library: dict[str, pd.DataFrame] = {}  # accepted factors by canonical form
        self.reports: dict[str, FactorReport] = {}

    def judge(self, expression: str, seen: Callable[[str], str | None] = lambda c: None) -> Verdict:
        """Parse, de-duplicate, score and apply the acceptance rules to one expression."""
        try:
            parsed = dsl.parse(expression)
        except ExpressionError as exc:
            return Verdict(expression, None, "invalid", str(exc))
        canonical = parsed.canonical
        prior = "accepted" if canonical in self.library else seen(canonical)
        if prior is not None:
            return Verdict(expression, canonical, "duplicate", f"already tried, was {prior}")
        try:
            report = evaluate_factor(self.ctx, canonical, self.window, reference=self.library)
        except (ExpressionError, ValueError) as exc:
            return Verdict(expression, canonical, "invalid", str(exc))
        failures = self.rules.failures(report)
        if failures:
            return Verdict(expression, canonical, "rejected", "; ".join(failures), report)
        self.library[canonical] = self.ctx.compute(parsed)[1]
        self.reports[canonical] = report
        return Verdict(expression, canonical, "accepted", None, report)

    def describe_rules(self) -> str:
        r = self.rules
        return (
            f"- coverage of universe stocks >= {r.min_coverage:.0%}\n"
            f"- |mean RankIC| >= {r.min_abs_rank_ic} and |RankICIR| >= {r.min_abs_icir}"
            " (negative is fine: the sign is applied for you)\n"
            f"- RankIC has the factor's sign in >= {r.min_hit_rate:.0%} of periods\n"
            f"- rank correlation with every accepted factor <= {r.max_corr}\n"
            f"- top-quintile turnover <= {r.max_turnover:.0%} per period"
        )
