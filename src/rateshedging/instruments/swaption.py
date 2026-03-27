from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rateshedging.pricing.curve import CurveSnapshot

from .instrument import Cashflow, InterestRateInstrument
from .swap import Swap


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Swaption(InterestRateInstrument):
    underlying_swap: Swap
    expiry: float
    is_long: bool = True

    def __post_init__(self) -> None:
        if self.expiry < 0.0:
            raise ValueError("expiry must be non-negative.")
        if self.expiry > float(self.underlying_swap.payment_times[-1]):
            raise ValueError("expiry must not exceed the final underlying payment time.")

    @property
    def direction(self) -> float:
        return 1.0 if self.is_long else -1.0

    def required_model_times(self, valuation_time: float = 0.0) -> FloatArray:
        if self.expiry <= valuation_time + 1.0e-12:
            return np.zeros(0, dtype=np.float64)
        return np.asarray([self.expiry - valuation_time], dtype=np.float64)

    def exercise_value_from_zero_rates(
        self,
        zero_rates: ArrayLike,
        curve_tenors: ArrayLike,
    ) -> FloatArray:
        swap_value = self.underlying_swap.present_value_from_zero_rates(
            zero_rates,
            curve_tenors,
            valuation_time=self.expiry,
        )
        return np.maximum(swap_value, 0.0)

    def exercised_cashflows(self, curve: CurveSnapshot) -> tuple[Cashflow, ...]:
        intrinsic_value = self.underlying_swap.present_value_from_curve(curve, valuation_time=self.expiry)
        if intrinsic_value <= 0.0:
            return ()
        return (
            Cashflow(
                payment_time=float(self.expiry),
                amount=float(self.direction * intrinsic_value),
                leg="option",
            ),
        )

    def cashflows_at_time(self, curve: CurveSnapshot, payment_time: float) -> tuple[Cashflow, ...]:
        if not np.isclose(payment_time, self.expiry, atol=1.0e-12, rtol=0.0):
            return ()
        return self.exercised_cashflows(curve)

    def cashflows(self, curve: CurveSnapshot, valuation_time: float = 0.0) -> tuple[Cashflow, ...]:
        return self.cashflows_at_time(curve, valuation_time)


__all__ = ["Swaption"]
