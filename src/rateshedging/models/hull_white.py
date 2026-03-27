from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._curve_utils import (
    LogLinearDiscountCurve,
    continuous_zero_rates,
    stable_cholesky,
    validate_tenors,
    validate_time_grid,
)
from .model import InterestRateModel, RatePaths


FloatArray = NDArray[np.float64]


def _b(mean_reversion: float, maturity: ArrayLike) -> FloatArray:
    maturity_array = np.asarray(maturity, dtype=np.float64)
    return -np.expm1(-mean_reversion * maturity_array) / mean_reversion


def _integrated_ou_variance(mean_reversion: float, volatility: float, maturity: ArrayLike) -> FloatArray:
    maturity_array = np.asarray(maturity, dtype=np.float64)
    one_minus_exp = -np.expm1(-mean_reversion * maturity_array)
    one_minus_exp_2 = -np.expm1(-2.0 * mean_reversion * maturity_array)
    b_term = one_minus_exp / mean_reversion
    variance = (volatility * volatility / (mean_reversion * mean_reversion)) * (
        maturity_array - 2.0 * b_term + one_minus_exp_2 / (2.0 * mean_reversion)
    )
    return np.maximum(variance, 0.0)


def _step_covariance(mean_reversion: float, volatility: float, dt: float) -> FloatArray:
    one_minus_exp = -np.expm1(-mean_reversion * dt)
    one_minus_exp_2 = -np.expm1(-2.0 * mean_reversion * dt)

    variance_x = volatility * volatility * one_minus_exp_2 / (2.0 * mean_reversion)
    variance_integral = _integrated_ou_variance(mean_reversion, volatility, dt)
    covariance = volatility * volatility * one_minus_exp * one_minus_exp / (2.0 * mean_reversion * mean_reversion)

    return np.array(
        [
            [variance_x, covariance],
            [covariance, variance_integral],
        ],
        dtype=np.float64,
    )


class HullWhiteModel(InterestRateModel):
    """One-factor Hull-White model fitted to an initial discount curve.

    Parameters
    ----------
    mean_reversion:
        Positive mean reversion speed.
    volatility:
        Positive short-rate volatility.
    time_grid:
        Simulation times in years, starting at 0.
    curve_times:
        Maturities of the input discount curve in years.
    discount_factors:
        Discount factors corresponding to ``curve_times``.
    yield_curve_tenors:
        Positive time-to-maturity tenors at which continuously compounded zero
        rates are returned for every simulation time.
    seed:
        Optional seed for reproducible path generation.
    """

    def __init__(
        self,
        mean_reversion: float,
        volatility: float,
        time_grid: ArrayLike,
        curve_times: ArrayLike,
        discount_factors: ArrayLike,
        yield_curve_tenors: ArrayLike,
        seed: int | None = None,
    ) -> None:
        if mean_reversion <= 0.0:
            raise ValueError("mean_reversion must be strictly positive.")
        if volatility <= 0.0:
            raise ValueError("volatility must be strictly positive.")

        self.mean_reversion = float(mean_reversion)
        self.volatility = float(volatility)
        self.time_grid = validate_time_grid(time_grid)
        self.yield_curve_tenors = validate_tenors(yield_curve_tenors)
        self._curve = LogLinearDiscountCurve(curve_times, discount_factors)
        self._rng = np.random.default_rng(seed)

        self._dt = np.diff(self.time_grid)
        self._decay = np.exp(-self.mean_reversion * self._dt)
        self._step_b = _b(self.mean_reversion, self._dt)
        self._step_cholesky = tuple(
            stable_cholesky(_step_covariance(self.mean_reversion, self.volatility, float(dt)))
            for dt in self._dt
        )

        self._discounts_on_time_grid = self._curve.discount(self.time_grid)
        self._integrated_variance_on_time_grid = _integrated_ou_variance(
            self.mean_reversion,
            self.volatility,
            self.time_grid,
        )

        self._maturity_grid = self.time_grid[:, None] + self.yield_curve_tenors[None, :]
        self._discounts_on_maturity_grid = self._curve.discount(self._maturity_grid)
        self._integrated_variance_on_maturity_grid = _integrated_ou_variance(
            self.mean_reversion,
            self.volatility,
            self._maturity_grid,
        )
        self._tenor_b = _b(self.mean_reversion, self.yield_curve_tenors)
        self._future_integrated_variance = _integrated_ou_variance(
            self.mean_reversion,
            self.volatility,
            self.yield_curve_tenors,
        )
        self._bond_price_adjustment = (
            self._discounts_on_maturity_grid / self._discounts_on_time_grid[:, None]
        ) * np.exp(
            0.5 * self._future_integrated_variance[None, :]
            - 0.5
            * (
                self._integrated_variance_on_maturity_grid
                - self._integrated_variance_on_time_grid[:, None]
            )
        )

    def generate_paths(self, n_paths: int) -> RatePaths:
        if int(n_paths) != n_paths or n_paths <= 0:
            raise ValueError("n_paths must be a strictly positive integer.")

        n_paths = int(n_paths)
        n_times = self.time_grid.size

        factors = np.empty((n_paths, n_times, 1), dtype=np.float64)
        stochastic_discount_factors = np.empty((n_paths, n_times), dtype=np.float64)

        x = np.zeros(n_paths, dtype=np.float64)
        integrated_x = np.zeros(n_paths, dtype=np.float64)

        factors[:, 0, 0] = 0.0
        stochastic_discount_factors[:, 0] = 1.0

        for time_index, (decay, b_step, step_cholesky) in enumerate(
            zip(self._decay, self._step_b, self._step_cholesky, strict=True),
            start=1,
        ):
            shocks = self._rng.standard_normal((n_paths, 2), dtype=np.float64) @ step_cholesky.T
            x_previous = x
            x = decay * x_previous + shocks[:, 0]
            integrated_x = integrated_x + b_step * x_previous + shocks[:, 1]

            factors[:, time_index, 0] = x
            stochastic_discount_factors[:, time_index] = self._discounts_on_time_grid[time_index] * np.exp(
                -integrated_x - 0.5 * self._integrated_variance_on_time_grid[time_index]
            )

        bond_prices = self._bond_price_adjustment[None, :, :] * np.exp(
            -factors[:, :, [0]] * self._tenor_b[None, None, :]
        )
        yield_curve_paths = continuous_zero_rates(
            bond_prices,
            self.yield_curve_tenors[None, None, :],
        )

        return RatePaths(
            yield_curve_paths=yield_curve_paths,
            stochastic_discount_factors=stochastic_discount_factors,
            yield_curve_factors=factors,
            yield_curve_tenors=self.yield_curve_tenors.copy(),
            time=self.time_grid.copy(),
            n_paths=n_paths,
        )


__all__ = ["HullWhiteModel"]
