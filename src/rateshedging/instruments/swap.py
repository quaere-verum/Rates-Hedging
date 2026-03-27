from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rateshedging.models._curve_utils import as_1d_float_array
from rateshedging.pricing.curve import CurveSnapshot, discount_from_zero_rates, forward_rates_from_zero_rates

from .instrument import Cashflow, InterestRateInstrument

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class Swap(InterestRateInstrument):
    start_time: float
    payment_times: ArrayLike
    fixed_rate: float
    notional: float = 1.0
    pay_fixed: bool = True
    is_long: bool = True
    floating_rate_fixings: Mapping[float, float] | None = None

    def __post_init__(self) -> None:
        payment_times = as_1d_float_array(self.payment_times, "payment_times")
        if np.any(payment_times <= self.start_time):
            raise ValueError("payment_times must all be strictly greater than start_time.")
        if np.any(np.diff(payment_times) <= 0.0):
            raise ValueError("payment_times must be strictly increasing.")
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")

        object.__setattr__(self, "payment_times", payment_times)
        accrual_start_times = np.concatenate(([self.start_time], payment_times[:-1]))
        year_fractions = payment_times - accrual_start_times
        object.__setattr__(self, "_accrual_start_times", accrual_start_times)
        object.__setattr__(self, "_year_fractions", year_fractions)

    @property
    def accrual_start_times(self) -> FloatArray:
        return self._accrual_start_times.copy()

    @property
    def accrual_end_times(self) -> FloatArray:
        return self.payment_times.copy()

    @property
    def year_fractions(self) -> FloatArray:
        return self._year_fractions.copy()

    @property
    def direction(self) -> float:
        payer_direction = 1.0 if self.pay_fixed else -1.0
        long_direction = 1.0 if self.is_long else -1.0
        return payer_direction * long_direction

    def _remaining_schedule(self, valuation_time: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        start_index = int(np.searchsorted(self.payment_times, valuation_time + 1.0e-12, side="right"))
        return (
            self._accrual_start_times[start_index:],
            self.payment_times[start_index:],
            self.payment_times[start_index:],
            self._year_fractions[start_index:],
        )

    def _floating_leg_pv_from_discount_factors(
        self,
        payment_discounts: FloatArray,
        zero_rates: ArrayLike,
        curve_tenors: ArrayLike,
        valuation_time: float,
        accrual_starts: FloatArray,
        accrual_ends: FloatArray,
        year_fractions: FloatArray,
    ) -> FloatArray:
        if payment_discounts.shape[-1] == 0:
            return np.zeros(np.asarray(zero_rates, dtype=np.float64).shape[:-1], dtype=np.float64)

        # Most swap valuations in the engine occur without explicit floating fixings,
        # so we use the equivalent telescoping discount-factor identity in the hot path.
        if not self.floating_rate_fixings:
            start_maturity = max(float(accrual_starts[0] - valuation_time), 0.0)
            start_discount = discount_from_zero_rates(
                zero_rates,
                curve_tenors,
                np.asarray([start_maturity], dtype=np.float64),
            )[..., 0]
            return self.notional * (start_discount - payment_discounts[..., -1])

        coupon_pv = np.zeros(np.asarray(zero_rates, dtype=np.float64).shape[:-1], dtype=np.float64)
        continuation_index = 0
        if accrual_starts.size > 0:
            first_start = float(accrual_starts[0])
            first_end = float(accrual_ends[0])
            if first_start <= valuation_time < first_end:
                fixing = self.floating_rate_fixings.get(first_start)
                if fixing is not None:
                    coupon_pv = self.notional * fixing * year_fractions[0] * payment_discounts[..., 0]
                    continuation_index = 1

        if continuation_index >= accrual_starts.size:
            return coupon_pv

        start_maturity = max(float(accrual_starts[continuation_index] - valuation_time), 0.0)
        start_discount = discount_from_zero_rates(
            zero_rates,
            curve_tenors,
            np.asarray([start_maturity], dtype=np.float64),
        )[..., 0]
        return coupon_pv + self.notional * (start_discount - payment_discounts[..., -1])

    def fixed_leg_cashflows(self, valuation_time: float = 0.0) -> tuple[Cashflow, ...]:
        accrual_starts, accrual_ends, payment_times, year_fractions = self._remaining_schedule(valuation_time)
        fixed_sign = -self.direction

        return tuple(
            Cashflow(
                payment_time=float(payment_time),
                amount=float(self.notional * self.fixed_rate * year_fraction * fixed_sign),
                leg="fixed",
                accrual_start=float(accrual_start),
                accrual_end=float(accrual_end),
                year_fraction=float(year_fraction),
            )
            for accrual_start, accrual_end, payment_time, year_fraction in zip(
                accrual_starts,
                accrual_ends,
                payment_times,
                year_fractions,
                strict=True,
            )
        )

    def floating_leg_cashflows(self, curve: CurveSnapshot, valuation_time: float = 0.0) -> tuple[Cashflow, ...]:
        accrual_starts, accrual_ends, payment_times, year_fractions = self._remaining_schedule(valuation_time)
        if payment_times.size == 0:
            return ()

        start_times = np.maximum(accrual_starts - valuation_time, 0.0)
        end_times = accrual_ends - valuation_time
        forward_rates = forward_rates_from_zero_rates(curve.zero_rates, curve.tenors, start_times, end_times)

        if self.floating_rate_fixings:
            for index, accrual_start in enumerate(accrual_starts):
                if accrual_start <= valuation_time < accrual_ends[index]:
                    fixing = self.floating_rate_fixings.get(float(accrual_start))
                    if fixing is not None:
                        forward_rates[index] = fixing

        floating_sign = self.direction
        return tuple(
            Cashflow(
                payment_time=float(payment_time),
                amount=float(self.notional * forward_rate * year_fraction * floating_sign),
                leg="floating",
                accrual_start=float(accrual_start),
                accrual_end=float(accrual_end),
                year_fraction=float(year_fraction),
            )
            for accrual_start, accrual_end, payment_time, year_fraction, forward_rate in zip(
                accrual_starts,
                accrual_ends,
                payment_times,
                year_fractions,
                forward_rates,
                strict=True,
            )
        )

    def cashflows(self, curve: CurveSnapshot, valuation_time: float = 0.0) -> tuple[Cashflow, ...]:
        return self.fixed_leg_cashflows(valuation_time) + self.floating_leg_cashflows(curve, valuation_time)

    def annuity_from_zero_rates(self, zero_rates: ArrayLike, curve_tenors: ArrayLike, valuation_time: float = 0.0) -> FloatArray:
        _, _, payment_times, year_fractions = self._remaining_schedule(valuation_time)
        if payment_times.size == 0:
            return np.zeros(np.asarray(zero_rates, dtype=np.float64).shape[:-1], dtype=np.float64)

        payment_maturities = payment_times - valuation_time
        payment_discounts = discount_from_zero_rates(zero_rates, curve_tenors, payment_maturities)
        return self.notional * np.sum(payment_discounts * year_fractions, axis=-1)

    def fair_rate_from_zero_rates(self, zero_rates: ArrayLike, curve_tenors: ArrayLike, valuation_time: float = 0.0) -> FloatArray:
        floating_pv = self.projected_floating_leg_pv_from_zero_rates(zero_rates, curve_tenors, valuation_time)
        annuity = self.annuity_from_zero_rates(zero_rates, curve_tenors, valuation_time)
        return np.divide(floating_pv, annuity, out=np.zeros_like(floating_pv), where=annuity > 0.0)

    def projected_floating_leg_pv_from_zero_rates(
        self,
        zero_rates: ArrayLike,
        curve_tenors: ArrayLike,
        valuation_time: float = 0.0,
    ) -> FloatArray:
        accrual_starts, accrual_ends, payment_times, year_fractions = self._remaining_schedule(valuation_time)
        if payment_times.size == 0:
            return np.zeros(np.asarray(zero_rates, dtype=np.float64).shape[:-1], dtype=np.float64)

        payment_maturities = payment_times - valuation_time
        payment_discounts = discount_from_zero_rates(zero_rates, curve_tenors, payment_maturities)
        return self._floating_leg_pv_from_discount_factors(
            payment_discounts,
            zero_rates,
            curve_tenors,
            valuation_time,
            accrual_starts,
            accrual_ends,
            year_fractions,
        )

    def present_value_from_zero_rates(
        self,
        zero_rates: ArrayLike,
        curve_tenors: ArrayLike,
        valuation_time: float = 0.0,
    ) -> FloatArray:
        accrual_starts, accrual_ends, payment_times, year_fractions = self._remaining_schedule(valuation_time)
        if payment_times.size == 0:
            return np.zeros(np.asarray(zero_rates, dtype=np.float64).shape[:-1], dtype=np.float64)

        payment_maturities = payment_times - valuation_time
        payment_discounts = discount_from_zero_rates(zero_rates, curve_tenors, payment_maturities)
        fixed_leg_pv = self.fixed_rate * self.notional * np.sum(payment_discounts * year_fractions, axis=-1)
        floating_leg_pv = self._floating_leg_pv_from_discount_factors(
            payment_discounts,
            zero_rates,
            curve_tenors,
            valuation_time,
            accrual_starts,
            accrual_ends,
            year_fractions,
        )
        return self.direction * (floating_leg_pv - fixed_leg_pv)

    def present_value_from_curve(self, curve: CurveSnapshot, valuation_time: float = 0.0) -> float:
        return float(self.present_value_from_zero_rates(curve.zero_rates, curve.tenors, valuation_time))

    def required_model_times(self, valuation_time: float = 0.0) -> FloatArray:
        return np.zeros(0, dtype=np.float64)


__all__ = ["Cashflow", "Swap"]
