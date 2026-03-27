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


def _integrated_cross_variance(
    mean_reversion_x: float,
    mean_reversion_y: float,
    volatility_x: float,
    volatility_y: float,
    correlation: float,
    maturity: ArrayLike,
) -> FloatArray:
    maturity_array = np.asarray(maturity, dtype=np.float64)
    b_x = _b(mean_reversion_x, maturity_array)
    b_y = _b(mean_reversion_y, maturity_array)
    b_xy = _b(mean_reversion_x + mean_reversion_y, maturity_array)
    variance = (
        correlation
        * volatility_x
        * volatility_y
        / (mean_reversion_x * mean_reversion_y)
        * (maturity_array - b_x - b_y + b_xy)
    )
    return variance


def _integrated_short_rate_variance(
    mean_reversion_x: float,
    mean_reversion_y: float,
    volatility_x: float,
    volatility_y: float,
    correlation: float,
    maturity: ArrayLike,
) -> FloatArray:
    variance_x = _integrated_ou_variance(mean_reversion_x, volatility_x, maturity)
    variance_y = _integrated_ou_variance(mean_reversion_y, volatility_y, maturity)
    cross_variance = _integrated_cross_variance(
        mean_reversion_x,
        mean_reversion_y,
        volatility_x,
        volatility_y,
        correlation,
        maturity,
    )
    return np.maximum(variance_x + variance_y + 2.0 * cross_variance, 0.0)


def _step_covariance(
    mean_reversion_x: float,
    mean_reversion_y: float,
    volatility_x: float,
    volatility_y: float,
    correlation: float,
    dt: float,
) -> FloatArray:
    one_minus_exp_x = -np.expm1(-mean_reversion_x * dt)
    one_minus_exp_y = -np.expm1(-mean_reversion_y * dt)
    one_minus_exp_xy = -np.expm1(-(mean_reversion_x + mean_reversion_y) * dt)

    variance_x = volatility_x * volatility_x * (-np.expm1(-2.0 * mean_reversion_x * dt)) / (2.0 * mean_reversion_x)
    variance_y = volatility_y * volatility_y * (-np.expm1(-2.0 * mean_reversion_y * dt)) / (2.0 * mean_reversion_y)
    covariance_xy = (
        correlation * volatility_x * volatility_y * one_minus_exp_xy / (mean_reversion_x + mean_reversion_y)
    )

    covariance_x_integral_x = (
        volatility_x
        * volatility_x
        * one_minus_exp_x
        * one_minus_exp_x
        / (2.0 * mean_reversion_x * mean_reversion_x)
    )
    covariance_y_integral_y = (
        volatility_y
        * volatility_y
        * one_minus_exp_y
        * one_minus_exp_y
        / (2.0 * mean_reversion_y * mean_reversion_y)
    )
    covariance_x_integral_y = correlation * volatility_x * volatility_y / mean_reversion_y * (
        one_minus_exp_x / mean_reversion_x - one_minus_exp_xy / (mean_reversion_x + mean_reversion_y)
    )
    covariance_y_integral_x = correlation * volatility_x * volatility_y / mean_reversion_x * (
        one_minus_exp_y / mean_reversion_y - one_minus_exp_xy / (mean_reversion_x + mean_reversion_y)
    )

    variance_integral = _integrated_short_rate_variance(
        mean_reversion_x,
        mean_reversion_y,
        volatility_x,
        volatility_y,
        correlation,
        dt,
    )

    covariance_x_integral = covariance_x_integral_x + covariance_x_integral_y
    covariance_y_integral = covariance_y_integral_y + covariance_y_integral_x

    return np.array(
        [
            [variance_x, covariance_xy, covariance_x_integral],
            [covariance_xy, variance_y, covariance_y_integral],
            [covariance_x_integral, covariance_y_integral, variance_integral],
        ],
        dtype=np.float64,
    )


class G2PPModel(InterestRateModel):
    """Two-factor G2++ model fitted to an initial discount curve."""

    def __init__(
        self,
        mean_reversion_x: float,
        mean_reversion_y: float,
        volatility_x: float,
        volatility_y: float,
        correlation: float,
        time_grid: ArrayLike,
        curve_times: ArrayLike,
        discount_factors: ArrayLike,
        yield_curve_tenors: ArrayLike,
        seed: int | None = None,
    ) -> None:
        if mean_reversion_x <= 0.0 or mean_reversion_y <= 0.0:
            raise ValueError("Both mean reversion parameters must be strictly positive.")
        if volatility_x <= 0.0 or volatility_y <= 0.0:
            raise ValueError("Both volatilities must be strictly positive.")
        if not -1.0 < correlation < 1.0:
            raise ValueError("correlation must lie strictly between -1 and 1.")

        self.mean_reversion_x = float(mean_reversion_x)
        self.mean_reversion_y = float(mean_reversion_y)
        self.volatility_x = float(volatility_x)
        self.volatility_y = float(volatility_y)
        self.correlation = float(correlation)
        self.time_grid = validate_time_grid(time_grid)
        self.yield_curve_tenors = validate_tenors(yield_curve_tenors)
        self._curve = LogLinearDiscountCurve(curve_times, discount_factors)
        self._rng = np.random.default_rng(seed)

        self._dt = np.diff(self.time_grid)
        self._decay_x = np.exp(-self.mean_reversion_x * self._dt)
        self._decay_y = np.exp(-self.mean_reversion_y * self._dt)
        self._step_b_x = _b(self.mean_reversion_x, self._dt)
        self._step_b_y = _b(self.mean_reversion_y, self._dt)
        self._step_cholesky = tuple(
            stable_cholesky(
                _step_covariance(
                    self.mean_reversion_x,
                    self.mean_reversion_y,
                    self.volatility_x,
                    self.volatility_y,
                    self.correlation,
                    float(dt),
                )
            )
            for dt in self._dt
        )

        self._discounts_on_time_grid = self._curve.discount(self.time_grid)
        self._integrated_variance_on_time_grid = _integrated_short_rate_variance(
            self.mean_reversion_x,
            self.mean_reversion_y,
            self.volatility_x,
            self.volatility_y,
            self.correlation,
            self.time_grid,
        )

        self._maturity_grid = self.time_grid[:, None] + self.yield_curve_tenors[None, :]
        self._discounts_on_maturity_grid = self._curve.discount(self._maturity_grid)
        self._integrated_variance_on_maturity_grid = _integrated_short_rate_variance(
            self.mean_reversion_x,
            self.mean_reversion_y,
            self.volatility_x,
            self.volatility_y,
            self.correlation,
            self._maturity_grid,
        )
        self._tenor_b_x = _b(self.mean_reversion_x, self.yield_curve_tenors)
        self._tenor_b_y = _b(self.mean_reversion_y, self.yield_curve_tenors)
        self._future_integrated_variance = _integrated_short_rate_variance(
            self.mean_reversion_x,
            self.mean_reversion_y,
            self.volatility_x,
            self.volatility_y,
            self.correlation,
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

        factors = np.empty((n_paths, n_times, 2), dtype=np.float64)
        stochastic_discount_factors = np.empty((n_paths, n_times), dtype=np.float64)

        x = np.zeros(n_paths, dtype=np.float64)
        y = np.zeros(n_paths, dtype=np.float64)
        integrated_short_rate = np.zeros(n_paths, dtype=np.float64)

        factors[:, 0, 0] = 0.0
        factors[:, 0, 1] = 0.0
        stochastic_discount_factors[:, 0] = 1.0

        for time_index, (decay_x, decay_y, b_x, b_y, step_cholesky) in enumerate(
            zip(
                self._decay_x,
                self._decay_y,
                self._step_b_x,
                self._step_b_y,
                self._step_cholesky,
                strict=True,
            ),
            start=1,
        ):
            shocks = self._rng.standard_normal((n_paths, 3), dtype=np.float64) @ step_cholesky.T
            x_previous = x
            y_previous = y

            x = decay_x * x_previous + shocks[:, 0]
            y = decay_y * y_previous + shocks[:, 1]
            integrated_short_rate = integrated_short_rate + b_x * x_previous + b_y * y_previous + shocks[:, 2]

            factors[:, time_index, 0] = x
            factors[:, time_index, 1] = y
            stochastic_discount_factors[:, time_index] = self._discounts_on_time_grid[time_index] * np.exp(
                -integrated_short_rate - 0.5 * self._integrated_variance_on_time_grid[time_index]
            )

        bond_prices = self._bond_price_adjustment[None, :, :] * np.exp(
            -factors[:, :, [0]] * self._tenor_b_x[None, None, :]
            - factors[:, :, [1]] * self._tenor_b_y[None, None, :]
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


__all__ = ["G2PPModel"]
