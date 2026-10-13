"""Trading costs for A-shares."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from timefence.backtest.rules import stamp_duty_rate


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.00025  # broker commission, both sides
    min_commission: float = 5.0  # yuan per order
    transfer_fee: float = 0.00001  # exchange transfer fee, both sides (simplified to one rate)
    slippage: float = 0.0005  # fill price moves 5 bp against us
    stamp_duty: bool = True  # sell-side tax; the rate depends on the date

    def fees(self, value: float, side: str, day: pd.Timestamp) -> float:
        """Commission, transfer fee and (for sells) stamp duty on a fill worth `value` yuan."""
        fee = max(value * self.commission, self.min_commission) + value * self.transfer_fee
        if side == "sell" and self.stamp_duty:
            fee += value * stamp_duty_rate(day)
        return fee

    def fill_price(self, price: float, side: str) -> float:
        return price * (1 + self.slippage) if side == "buy" else price * (1 - self.slippage)


ZERO_COSTS = CostModel(
    commission=0.0, min_commission=0.0, transfer_fee=0.0, slippage=0.0, stamp_duty=False
)
