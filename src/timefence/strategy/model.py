"""Combine accepted factors into one score per stock and day."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import pandas as pd

from timefence.factors import dsl, ops


@dataclass(frozen=True)
class FactorSpec:
    expression: str
    direction: int = 1  # +1 if high values predict high returns (decided on training data)
    weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class FactorModel:
    """Score = weighted mean of the factors' cross-sectional z-scores, signed by direction.

    Z-scores are taken among universe stocks on each day and clipped at +-3 so one extreme
    value cannot dominate. A stock needs values for at least half the total weight.
    """

    def __init__(self, factors: Sequence[FactorSpec]):
        if not factors:
            raise ValueError("a factor model needs at least one factor")
        self.factors = list(factors)
        self.parsed = [dsl.parse(f.expression) for f in self.factors]

    @property
    def lookback(self) -> int:
        return max(p.lookback for p in self.parsed)

    def score(self, field: Callable[[str], pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
        total = pd.DataFrame(0.0, index=universe.index, columns=universe.columns)
        weight = total.copy()
        for spec, parsed in zip(self.factors, self.parsed, strict=True):
            values = dsl.evaluate(parsed, field, universe).where(universe)
            z = ops.zscore(values).clip(-3, 3) * spec.direction
            total += z.fillna(0.0) * spec.weight
            weight += z.notna() * spec.weight
        full = sum(f.weight for f in self.factors)
        return (total / weight).where(weight >= 0.5 * full)
