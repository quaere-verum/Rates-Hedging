from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rateshedging.models._curve_utils import as_1d_float_array
from rateshedging.pricing.curve import CurveSnapshot

from .instrument import Cashflow, InterestRateInstrument
from .swap import Swap


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BermudanSwaption(InterestRateInstrument):
    underlying_swap: Swap
    exercise_times: ArrayLike
    is_long: bool = True

    def __post_init__(self) -> None:
        exercise_times = as_1d_float_array(self.exercise_times, "exercise_times")
        if np.any(exercise_times < 0.0):
            raise ValueError("exercise_times must be non-negative.")
        if np.any(np.diff(exercise_times) <= 0.0):
            raise ValueError("exercise_times must be strictly increasing.")
        if exercise_times[-1] > float(self.underlying_swap.payment_times[-1]):
            raise ValueError("The final exercise time must not exceed the final swap payment time.")
        object.__setattr__(self, "exercise_times", exercise_times)

    @property
    def direction(self) -> float:
        return 1.0 if self.is_long else -1.0

    def remaining_exercise_times(self, valuation_time: float = 0.0) -> FloatArray:
        return self.exercise_times[self.exercise_times > valuation_time + 1.0e-12]

    def required_model_times(self, valuation_time: float = 0.0) -> FloatArray:
        remaining = self.remaining_exercise_times(valuation_time)
        if remaining.size == 0:
            return np.zeros(0, dtype=np.float64)
        return remaining - valuation_time

    def exercise_value_from_zero_rates(
        self,
        zero_rates: ArrayLike,
        curve_tenors: ArrayLike,
        exercise_time: float,
    ) -> FloatArray:
        swap_value = self.underlying_swap.present_value_from_zero_rates(
            zero_rates,
            curve_tenors,
            valuation_time=exercise_time,
        )
        return np.maximum(swap_value, 0.0)

    def exercised_cashflows(self, curve: CurveSnapshot, exercise_time: float) -> tuple[Cashflow, ...]:
        intrinsic_value = self.underlying_swap.present_value_from_curve(curve, valuation_time=exercise_time)
        if intrinsic_value <= 0.0:
            return ()
        return (
            Cashflow(
                payment_time=float(exercise_time),
                amount=float(self.direction * intrinsic_value),
                leg="option",
            ),
        )

    def cashflows_at_time(self, curve: CurveSnapshot, payment_time: float) -> tuple[Cashflow, ...]:
        matches = np.flatnonzero(np.isclose(self.exercise_times, payment_time, atol=1.0e-12, rtol=0.0))
        if matches.size == 0:
            return ()
        return self.exercised_cashflows(curve, float(payment_time))

    def cashflows(self, curve: CurveSnapshot, valuation_time: float = 0.0) -> tuple[Cashflow, ...]:
        return self.cashflows_at_time(curve, valuation_time)


__all__ = ["BermudanSwaption"]
