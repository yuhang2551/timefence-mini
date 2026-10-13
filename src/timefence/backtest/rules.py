"""A-share trading rules the simulated broker enforces.

Simplifications, all documented in docs/adr/0004-backtest-rules.md:
- ST stocks have a 5% limit in reality; we do not have historical ST flags, so they get
  their board's limit. The liquidity universe rarely contains them.
- Every board trades in 100-share lots (the STAR board really allows odd sizes above 200).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOT = 100
CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")  # ChiNext registration reform
STAMP_DUTY_CUT = pd.Timestamp("2023-08-28")  # sell-side stamp duty 0.1% -> 0.05%


def limit_pct(board: str, day: pd.Timestamp) -> float:
    """Daily price limit as a fraction of the reference price."""
    if board == "star":
        return 0.20
    if board == "chinext":
        return 0.20 if day >= CHINEXT_20PCT_FROM else 0.10
    if board == "bse":
        return 0.30
    return 0.10


def limit_pct_panel(boards: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """`limit_pct` for every (day, code). `boards` is indexed by code."""
    out = pd.DataFrame(0.10, index=index, columns=boards.index)
    out.loc[:, (boards == "star").to_numpy()] = 0.20
    out.loc[:, (boards == "bse").to_numpy()] = 0.30
    chinext = (boards == "chinext").to_numpy()
    out.loc[index >= CHINEXT_20PCT_FROM, chinext] = 0.20
    return out


def round_price(x):
    """Exchange rounding to the cent: half up."""
    return np.floor(np.asarray(x, dtype=float) * 100 + 0.5 + 1e-9) / 100


def stamp_duty_rate(day: pd.Timestamp) -> float:
    """Sell-side stamp duty."""
    return 0.0005 if day >= STAMP_DUTY_CUT else 0.001
