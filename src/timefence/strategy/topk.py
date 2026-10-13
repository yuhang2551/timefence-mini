"""Hold the top-K stocks by model score in equal weights."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from timefence.data.fenced import FencedData
from timefence.data.universe import UniverseConfig
from timefence.strategy.model import FactorModel


class TopKStrategy:
    """Equal-weight the K best-scoring universe stocks.

    A holding is kept while it ranks within K * (1 + buffer), which cuts turnover: stocks
    hovering around rank K are not sold and bought back every rebalance.
    """

    def __init__(
        self,
        model: FactorModel,
        top_k: int = 50,
        buffer: float = 0.5,
        universe: UniverseConfig | None = None,
    ):
        self.model = model
        self.top_k = top_k
        self.buffer = buffer
        self.universe = universe or UniverseConfig()
        # Enough history for the deepest factor plus the universe's own window.
        self.lookback = model.lookback + self.universe.lookback + 30

    def target_weights(self, data: FencedData, holdings: Mapping[str, float]) -> pd.Series:
        universe = self.universe.build(data)
        score = self.model.score(data.panel, universe).iloc[-1].dropna()
        ranked = score.sort_values(ascending=False, kind="stable").index
        keep_zone = set(ranked[: int(self.top_k * (1 + self.buffer))])
        kept = [c for c in ranked if c in holdings and c in keep_zone][: self.top_k]
        fresh = [c for c in ranked if c not in kept][: self.top_k - len(kept)]
        picks = kept + fresh
        if not picks:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(picks), index=picks)
