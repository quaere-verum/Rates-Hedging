from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._curve_utils import LogLinearDiscountCurve, as_1d_float_array, validate_time_grid, validate_tenors
from .model import InterestRateModel, RatePaths


FloatArray = NDArray[np.float64]


class LIBORMarketModel(InterestRateModel):
    """Simple lognormal LIBOR Market Model on a uniform accrual tenor grid."""

    def __init__(
        self,
        *,
        tenor_spacing: float,
        factor_loading_matrix: ArrayLike | None = None,
        factor_volatilities: ArrayLike | None = None,
        factor_decay_rates: ArrayLike | None = None,
        time_grid: ArrayLike,
        curve_times: ArrayLike,
        discount_factors: ArrayLike,
        yield_curve_tenors: ArrayLike,
        seed: int | None = None,
    ) -> None:
        if tenor_spacing <= 0.0:
            raise ValueError("tenor_spacing must be strictly positive.")

        self.tenor_spacing = float(tenor_spacing)
        self.time_grid = validate_time_grid(time_grid)
        self.yield_curve_tenors = validate_tenors(yield_curve_tenors)
        self._curve = LogLinearDiscountCurve(curve_times, discount_factors)
        self._rng = np.random.default_rng(seed)

        terminal_horizon = float(self.time_grid[-1] + self.yield_curve_tenors[-1])
        n_periods = int(round(terminal_horizon / self.tenor_spacing))
        if not np.isclose(n_periods * self.tenor_spacing, terminal_horizon, atol=1.0e-10, rtol=0.0):
            raise ValueError("time_grid[-1] + yield_curve_tenors[-1] must be a multiple of tenor_spacing.")
        self._tenor_dates = np.arange(
            self.tenor_spacing,
            terminal_horizon + 1.0e-12,
            self.tenor_spacing,
            dtype=np.float64,
        )
        self._tenor_grid_with_zero = np.concatenate(([0.0], self._tenor_dates))
        self._taus = np.diff(self._tenor_grid_with_zero)

        time_indices = np.rint(self.time_grid / self.tenor_spacing).astype(np.int64)
        if not np.allclose(time_indices * self.tenor_spacing, self.time_grid, atol=1.0e-10, rtol=0.0):
            raise ValueError("time_grid entries must lie on the accrual tenor grid.")
        self._time_indices = time_indices

        yield_indices = np.rint(self.yield_curve_tenors / self.tenor_spacing).astype(np.int64)
        if not np.allclose(yield_indices * self.tenor_spacing, self.yield_curve_tenors, atol=1.0e-10, rtol=0.0):
            raise ValueError("yield_curve_tenors must lie on the accrual tenor grid.")

        initial_discounts = self._curve.discount(self._tenor_dates)
        discount_starts = np.concatenate(([1.0], initial_discounts[:-1]))
        self._initial_forwards = (discount_starts / initial_discounts - 1.0) / self._taus
        if np.any(self._initial_forwards <= 0.0):
            raise ValueError("Initial forwards implied by the curve must be strictly positive.")

        if factor_loading_matrix is not None:
            loading_matrix = np.asarray(factor_loading_matrix, dtype=np.float64)
            if loading_matrix.ndim != 2:
                raise ValueError("factor_loading_matrix must be a two-dimensional array.")
            if loading_matrix.shape[0] != self._initial_forwards.size:
                raise ValueError("factor_loading_matrix must have one row per forward tenor.")
            if not np.all(np.isfinite(loading_matrix)):
                raise ValueError("factor_loading_matrix must contain only finite values.")
            self._factor_loading_matrix = loading_matrix
        else:
            if factor_volatilities is None or factor_decay_rates is None:
                raise ValueError(
                    "Provide either factor_loading_matrix or both factor_volatilities and factor_decay_rates."
                )
            factor_volatility_array = as_1d_float_array(factor_volatilities, "factor_volatilities")
            factor_decay_array = as_1d_float_array(factor_decay_rates, "factor_decay_rates")
            if factor_volatility_array.shape != factor_decay_array.shape:
                raise ValueError("factor_volatilities and factor_decay_rates must have the same shape.")
            if np.any(factor_volatility_array <= 0.0):
                raise ValueError("factor_volatilities must be strictly positive.")
            if np.any(factor_decay_array < 0.0):
                raise ValueError("factor_decay_rates must be non-negative.")

            remaining_maturities = self._tenor_dates[:, None]
            self._factor_loading_matrix = factor_volatility_array[None, :] * np.exp(
                -remaining_maturities * factor_decay_array[None, :]
            )

        self.n_factors = int(self._factor_loading_matrix.shape[1])

    def _curve_from_forwards(
        self,
        forwards: FloatArray,
        active_index: int,
        current_time: float,
    ) -> FloatArray:
        remaining_dates = self._tenor_dates[active_index:] - current_time
        remaining_discounts = np.cumprod(
            1.0 / (1.0 + self._taus[active_index:][None, :] * forwards[:, active_index:]),
            axis=1,
        )

        zero_rates = np.empty((forwards.shape[0], self.yield_curve_tenors.size), dtype=np.float64)
        for path_index in range(forwards.shape[0]):
            node_times = np.concatenate(([0.0], remaining_dates))
            log_discount_nodes = np.concatenate(([0.0], np.log(remaining_discounts[path_index])))
            interpolated = np.interp(self.yield_curve_tenors, node_times, log_discount_nodes)
            zero_rates[path_index] = -interpolated / self.yield_curve_tenors
        return zero_rates

    def generate_paths(self, n_paths: int) -> RatePaths:
        if int(n_paths) != n_paths or n_paths <= 0:
            raise ValueError("n_paths must be a strictly positive integer.")

        n_paths = int(n_paths)
        n_times = self.time_grid.size
        n_factors = self.n_factors

        forwards = np.repeat(self._initial_forwards[None, :], n_paths, axis=0)
        factor_states = np.zeros((n_paths, n_factors), dtype=np.float64)

        yield_curve_paths = np.empty((n_paths, n_times, self.yield_curve_tenors.size), dtype=np.float64)
        stochastic_discount_factors = np.empty((n_paths, n_times), dtype=np.float64)
        yield_curve_factors = np.empty((n_paths, n_times, n_factors), dtype=np.float64)

        yield_curve_paths[:, 0, :] = self._curve_from_forwards(forwards, 0, 0.0)
        stochastic_discount_factors[:, 0] = 1.0
        yield_curve_factors[:, 0, :] = 0.0

        for time_position in range(1, n_times):
            previous_index = int(self._time_indices[time_position - 1])
            current_index = int(self._time_indices[time_position])
            current_discount = stochastic_discount_factors[:, time_position - 1].copy()

            for tenor_index in range(previous_index, current_index):
                current_time = float(self._tenor_grid_with_zero[tenor_index])
                active_slice = slice(tenor_index, forwards.shape[1])
                active_forwards = forwards[:, active_slice]
                forward_for_discount = active_forwards[:, 0].copy()

                loadings = self._factor_loading_matrix[active_slice]
                covariance_lower = np.tril(loadings @ loadings.T)
                weights = self._taus[active_slice][None, :] * active_forwards / (
                    1.0 + self._taus[active_slice][None, :] * active_forwards
                )
                drift = weights @ covariance_lower.T

                dt = self.tenor_spacing
                brownian_shocks = self._rng.standard_normal((n_paths, n_factors), dtype=np.float64) * np.sqrt(dt)
                variance = np.sum(loadings * loadings, axis=1)
                diffusion = brownian_shocks @ loadings.T
                active_forwards *= np.exp((drift - 0.5 * variance[None, :]) * dt + diffusion)

                forwards[:, active_slice] = active_forwards
                factor_states += brownian_shocks
                current_discount = current_discount / (1.0 + self._taus[tenor_index] * forward_for_discount)

            stochastic_discount_factors[:, time_position] = current_discount
            current_time = float(self.time_grid[time_position])
            yield_curve_paths[:, time_position, :] = self._curve_from_forwards(forwards, current_index, current_time)
            yield_curve_factors[:, time_position, :] = factor_states

        return RatePaths(
            yield_curve_paths=yield_curve_paths,
            stochastic_discount_factors=stochastic_discount_factors,
            yield_curve_factors=yield_curve_factors,
            yield_curve_tenors=self.yield_curve_tenors.copy(),
            time=self.time_grid.copy(),
            n_paths=n_paths,
        )


__all__ = ["LIBORMarketModel"]
