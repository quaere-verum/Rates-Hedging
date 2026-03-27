from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d


FloatArray = NDArray[np.float64]


def as_1d_float_array(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def validate_time_grid(values: ArrayLike) -> FloatArray:
    time_grid = as_1d_float_array(values, "time_grid")
    if not np.isclose(time_grid[0], 0.0):
        raise ValueError("time_grid must start at 0.0.")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")
    return time_grid


def validate_tenors(values: ArrayLike) -> FloatArray:
    tenors = as_1d_float_array(values, "yield_curve_tenors")
    if np.any(tenors <= 0.0):
        raise ValueError("yield_curve_tenors must contain strictly positive values.")
    if np.any(np.diff(tenors) <= 0.0):
        raise ValueError("yield_curve_tenors must be strictly increasing.")
    return tenors


class LogLinearDiscountCurve:
    def __init__(self, curve_times: ArrayLike, discount_factors: ArrayLike) -> None:
        times = as_1d_float_array(curve_times, "curve_times")
        discounts = as_1d_float_array(discount_factors, "discount_factors")

        if times.shape != discounts.shape:
            raise ValueError("curve_times and discount_factors must have the same shape.")
        if np.any(times < 0.0):
            raise ValueError("curve_times must be non-negative.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("curve_times must be strictly increasing.")
        if np.any(discounts <= 0.0):
            raise ValueError("discount_factors must be strictly positive.")

        if times[0] > 0.0:
            times = np.concatenate(([0.0], times))
            discounts = np.concatenate(([1.0], discounts))
        elif not np.isclose(discounts[0], 1.0):
            raise ValueError("discount_factors[0] must be 1.0 when curve_times[0] is 0.0.")

        self.times = times
        self.discount_factors = discounts
        self._log_interp = interp1d(
            times,
            np.log(discounts),
            kind="linear",
            assume_sorted=True,
            bounds_error=False,
            fill_value="extrapolate",
        )

    def discount(self, query_times: ArrayLike) -> FloatArray:
        query = np.asarray(query_times, dtype=np.float64)
        if np.any(query < 0.0):
            raise ValueError("Discount factors are only defined for non-negative maturities.")
        return np.exp(self._log_interp(query))


def continuous_zero_rates(discount_factors: ArrayLike, tenors: ArrayLike) -> FloatArray:
    discounts = np.asarray(discount_factors, dtype=np.float64)
    maturity_tenors = np.asarray(tenors, dtype=np.float64)
    return -np.log(discounts) / maturity_tenors


def stable_cholesky(covariance: ArrayLike) -> FloatArray:
    matrix = np.asarray(covariance, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    identity = np.eye(matrix.shape[0], dtype=np.float64)

    for jitter in (0.0, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8):
        try:
            return np.linalg.cholesky(matrix + jitter * identity)
        except np.linalg.LinAlgError:
            continue

    raise np.linalg.LinAlgError("Covariance matrix is not numerically positive definite.")
