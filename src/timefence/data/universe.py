"""Point-in-time stock universe built only from information known on each day."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from timefence.core import rolling
from timefence.core.types import PanelSource


def liquidity_universe(
    source: PanelSource,
    top_n: int = 500,
    lookback: int = 20,
    min_listed_days: int = 365,
    exclude_boards: tuple[str, ...] = ("bse",),
) -> pd.DataFrame:
    """Boolean mask (dates x codes): the `top_n` most traded stocks on each day.

    A stock qualifies on day t if it traded on t, listed at least `min_listed_days`
    calendar days before t (drops fresh IPOs), and is not on an excluded board. Stocks are
    then ranked by average turnover over the last `lookback` days. Every input is known by
    t's close and delisted stocks stay in the data, so there is no survivorship bias.

    `source` is a MarketStore or a FencedData view. The mask on day t depends only on the
    last `lookback` days, so a fenced window longer than that gives the same answer on its
    last day as the full history does.
    """
    amount = source.panel("amount")
    traded = source.panel("volume").notna().to_numpy()
    stocks = source.stocks.reindex(amount.columns)

    list_date = pd.to_datetime(stocks["list_date"]).to_numpy()
    seasoned = amount.index.to_numpy()[:, None] - list_date[None, :] >= np.timedelta64(
        min_listed_days, "D"
    )
    allowed_board = ~stocks["board"].isin(exclude_boards).to_numpy()

    avg_amount = rolling.rolling_mean(amount.to_numpy(dtype=float), lookback, max(1, lookback // 2))
    score = np.where(traded & seasoned & allowed_board[None, :], avg_amount, np.nan)
    return pd.DataFrame(_top_n(score, top_n), index=amount.index, columns=amount.columns)


def _top_n(score: np.ndarray, n: int) -> np.ndarray:
    """True for the n largest values in each row; NaN never counts."""
    valid = ~np.isnan(score)
    if n >= score.shape[1]:
        return valid
    filled = np.where(valid, score, -np.inf)
    top = np.argpartition(-filled, n - 1, axis=1)[:, :n]
    mask = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(mask, top, True, axis=1)
    return mask & valid


@dataclass(frozen=True)
class UniverseConfig:
    """Universe settings shared by research and trading, so both see the same stocks."""

    top_n: int = 500
    lookback: int = 20
    min_listed_days: int = 365
    exclude_boards: tuple[str, ...] = ("bse",)

    def build(self, source: PanelSource) -> pd.DataFrame:
        return liquidity_universe(
            source, self.top_n, self.lookback, self.min_listed_days, self.exclude_boards
        )
