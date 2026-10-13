"""Point-in-time stock universe built only from information known on each day."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    traded = source.panel("volume").notna()
    stocks = source.stocks.reindex(amount.columns)

    list_date = pd.to_datetime(stocks["list_date"]).to_numpy()
    age = amount.index.to_numpy()[:, None] - list_date[None, :]
    seasoned = pd.DataFrame(
        age >= np.timedelta64(min_listed_days, "D"), index=amount.index, columns=amount.columns
    )
    allowed_board = ~stocks["board"].isin(exclude_boards)

    avg_amount = amount.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    eligible = traded & seasoned & allowed_board
    rank = avg_amount.where(eligible).rank(axis=1, ascending=False, method="first")
    return rank <= top_n


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
