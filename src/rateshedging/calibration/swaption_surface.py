from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rateshedging.models._curve_utils import as_1d_float_array, validate_time_grid
from rateshedging.pricing.curve import CurveSnapshot


FloatArray = NDArray[np.float64]


def _surface_weights(expiries: FloatArray, swap_tenors: FloatArray) -> FloatArray:
    return np.ones((expiries.size, swap_tenors.size), dtype=np.float64)


@dataclass(frozen=True)
class SwaptionSurfaceSnapshot:
    expiries: ArrayLike
    swap_tenors: ArrayLike
    normal_volatilities: ArrayLike
    weights: ArrayLike | None = None

    def __post_init__(self) -> None:
        expiries = as_1d_float_array(self.expiries, "expiries")
        swap_tenors = as_1d_float_array(self.swap_tenors, "swap_tenors")
        if np.any(expiries <= 0.0):
            raise ValueError("expiries must be strictly positive.")
        if np.any(np.diff(expiries) <= 0.0):
            raise ValueError("expiries must be strictly increasing.")
        if np.any(swap_tenors <= 0.0):
            raise ValueError("swap_tenors must be strictly positive.")
        if np.any(np.diff(swap_tenors) <= 0.0):
            raise ValueError("swap_tenors must be strictly increasing.")

        normal_volatilities = np.asarray(self.normal_volatilities, dtype=np.float64)
        if normal_volatilities.shape != (expiries.size, swap_tenors.size):
            raise ValueError("normal_volatilities must have shape (n_expiries, n_swap_tenors).")
        if not np.all(np.isfinite(normal_volatilities)) or np.any(normal_volatilities <= 0.0):
            raise ValueError("normal_volatilities must contain strictly positive finite values.")

        if self.weights is None:
            weights = _surface_weights(expiries, swap_tenors)
        else:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape != normal_volatilities.shape:
                raise ValueError("weights must have the same shape as normal_volatilities.")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("weights must contain strictly positive finite values.")

        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "swap_tenors", swap_tenors)
        object.__setattr__(self, "normal_volatilities", normal_volatilities)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class SwaptionSurfaceTrajectory:
    time_grid: ArrayLike
    expiries: ArrayLike
    swap_tenors: ArrayLike
    normal_volatility_path: ArrayLike
    weights: ArrayLike | None = None

    def __post_init__(self) -> None:
        time_grid = validate_time_grid(self.time_grid)
        expiries = as_1d_float_array(self.expiries, "expiries")
        swap_tenors = as_1d_float_array(self.swap_tenors, "swap_tenors")
        normal_volatility_path = np.asarray(self.normal_volatility_path, dtype=np.float64)
        expected_shape = (time_grid.size, expiries.size, swap_tenors.size)
        if normal_volatility_path.shape != expected_shape:
            raise ValueError("normal_volatility_path must have shape (n_times, n_expiries, n_swap_tenors).")
        if not np.all(np.isfinite(normal_volatility_path)) or np.any(normal_volatility_path <= 0.0):
            raise ValueError("normal_volatility_path must contain strictly positive finite values.")

        if self.weights is None:
            weights = _surface_weights(expiries, swap_tenors)
        else:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape != (expiries.size, swap_tenors.size):
                raise ValueError("weights must have shape (n_expiries, n_swap_tenors).")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("weights must contain strictly positive finite values.")

        object.__setattr__(self, "time_grid", time_grid)
        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "swap_tenors", swap_tenors)
        object.__setattr__(self, "normal_volatility_path", normal_volatility_path)
        object.__setattr__(self, "weights", weights)

    def snapshot(self, time_index: int) -> SwaptionSurfaceSnapshot:
        return SwaptionSurfaceSnapshot(
            expiries=self.expiries,
            swap_tenors=self.swap_tenors,
            normal_volatilities=self.normal_volatility_path[time_index],
            weights=self.weights,
        )


@dataclass(frozen=True)
class SwaptionSurfacePaths:
    time_grid: ArrayLike
    expiries: ArrayLike
    swap_tenors: ArrayLike
    normal_volatility_paths: ArrayLike
    weights: ArrayLike | None = None

    def __post_init__(self) -> None:
        time_grid = validate_time_grid(self.time_grid)
        expiries = as_1d_float_array(self.expiries, "expiries")
        swap_tenors = as_1d_float_array(self.swap_tenors, "swap_tenors")
        normal_volatility_paths = np.asarray(self.normal_volatility_paths, dtype=np.float64)
        if normal_volatility_paths.ndim != 4:
            raise ValueError("normal_volatility_paths must be a four-dimensional array.")
        expected_shape = (normal_volatility_paths.shape[0], time_grid.size, expiries.size, swap_tenors.size)
        if normal_volatility_paths.shape != expected_shape:
            raise ValueError("normal_volatility_paths must have shape (n_paths, n_times, n_expiries, n_swap_tenors).")
        if not np.all(np.isfinite(normal_volatility_paths)) or np.any(normal_volatility_paths <= 0.0):
            raise ValueError("normal_volatility_paths must contain strictly positive finite values.")

        if self.weights is None:
            weights = _surface_weights(expiries, swap_tenors)
        else:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape != (expiries.size, swap_tenors.size):
                raise ValueError("weights must have shape (n_expiries, n_swap_tenors).")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("weights must contain strictly positive finite values.")

        object.__setattr__(self, "time_grid", time_grid)
        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "swap_tenors", swap_tenors)
        object.__setattr__(self, "normal_volatility_paths", normal_volatility_paths)
        object.__setattr__(self, "weights", weights)

    @property
    def n_paths(self) -> int:
        return int(self.normal_volatility_paths.shape[0])

    def path(self, path_index: int) -> SwaptionSurfaceTrajectory:
        return SwaptionSurfaceTrajectory(
            time_grid=self.time_grid,
            expiries=self.expiries,
            swap_tenors=self.swap_tenors,
            normal_volatility_path=self.normal_volatility_paths[path_index],
            weights=self.weights,
        )


def _mean_reversion_loading(mean_reversion: float, expiries: ArrayLike, swap_tenors: ArrayLike) -> FloatArray:
    expiry_array = np.asarray(expiries, dtype=np.float64)[:, None]
    tenor_array = np.asarray(swap_tenors, dtype=np.float64)[None, :]
    expiry_variance = (-np.expm1(-2.0 * mean_reversion * expiry_array)) / (2.0 * mean_reversion)
    expiry_scale = np.sqrt(expiry_variance / expiry_array)
    tenor_scale = -np.expm1(-mean_reversion * tenor_array) / (mean_reversion * tenor_array)
    return expiry_scale * tenor_scale


def hull_white_atm_normal_volatilities(
    mean_reversion: float,
    volatility: float,
    expiries: ArrayLike,
    swap_tenors: ArrayLike,
) -> FloatArray:
    if mean_reversion <= 0.0 or volatility <= 0.0:
        raise ValueError("Hull-White calibration inputs must be strictly positive.")
    return volatility * _mean_reversion_loading(mean_reversion, expiries, swap_tenors)


def g2pp_atm_normal_volatilities(
    mean_reversion_x: float,
    mean_reversion_y: float,
    volatility_x: float,
    volatility_y: float,
    correlation: float,
    expiries: ArrayLike,
    swap_tenors: ArrayLike,
) -> FloatArray:
    if mean_reversion_x <= 0.0 or mean_reversion_y <= 0.0:
        raise ValueError("G2++ mean reversions must be strictly positive.")
    if volatility_x <= 0.0 or volatility_y <= 0.0:
        raise ValueError("G2++ volatilities must be strictly positive.")
    if not -1.0 < correlation < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1.")

    loading_x = _mean_reversion_loading(mean_reversion_x, expiries, swap_tenors)
    loading_y = _mean_reversion_loading(mean_reversion_y, expiries, swap_tenors)
    total_variance = (
        (volatility_x * loading_x) ** 2
        + (volatility_y * loading_y) ** 2
        + 2.0 * correlation * volatility_x * volatility_y * loading_x * loading_y
    )
    return np.sqrt(np.maximum(total_variance, 1.0e-16))


def apply_curve_surface_adjustment(
    base_normal_volatilities: ArrayLike,
    curve_snapshot: CurveSnapshot,
    reference_curve: CurveSnapshot,
    expiries: ArrayLike,
    swap_tenors: ArrayLike,
) -> FloatArray:
    base_vols = np.asarray(base_normal_volatilities, dtype=np.float64)
    expiry_profile = np.asarray(expiries, dtype=np.float64)
    tenor_profile = np.asarray(swap_tenors, dtype=np.float64)
    if base_vols.shape != (expiry_profile.size, tenor_profile.size):
        raise ValueError("base_normal_volatilities shape must match expiries and swap_tenors.")

    level_shift = float(np.mean(curve_snapshot.zero_rates) - np.mean(reference_curve.zero_rates))
    slope_shift = float(
        (curve_snapshot.zero_rates[-1] - curve_snapshot.zero_rates[0])
        - (reference_curve.zero_rates[-1] - reference_curve.zero_rates[0])
    )
    level_scale = np.clip(1.0 + 6.0 * level_shift, 0.70, 1.30)

    expiry_centered = expiry_profile / np.mean(expiry_profile) - 1.0
    tenor_centered = tenor_profile / np.mean(tenor_profile) - 1.0
    twist = 1.0 + 1.5 * slope_shift * (expiry_centered[:, None] - tenor_centered[None, :])

    return base_vols * np.clip(level_scale * twist, 0.60, 1.40)


def build_surface_paths(
    *,
    time_grid: ArrayLike,
    curve_paths: ArrayLike,
    curve_tenors: ArrayLike,
    expiries: ArrayLike,
    swap_tenors: ArrayLike,
    base_normal_volatilities: ArrayLike,
) -> SwaptionSurfacePaths:
    time_array = validate_time_grid(time_grid)
    tenor_array = as_1d_float_array(curve_tenors, "curve_tenors")
    curve_paths_array = np.asarray(curve_paths, dtype=np.float64)
    if curve_paths_array.ndim != 3:
        raise ValueError("curve_paths must be a three-dimensional array.")
    expected_shape = (curve_paths_array.shape[0], time_array.size, tenor_array.size)
    if curve_paths_array.shape != expected_shape:
        raise ValueError("curve_paths must have shape (n_paths, n_times, n_curve_tenors).")

    expiries_array = as_1d_float_array(expiries, "expiries")
    swap_tenors_array = as_1d_float_array(swap_tenors, "swap_tenors")
    base_vols = np.asarray(base_normal_volatilities, dtype=np.float64)
    if base_vols.shape != (expiries_array.size, swap_tenors_array.size):
        raise ValueError("base_normal_volatilities shape must match expiries and swap_tenors.")

    reference_curve = CurveSnapshot.from_zero_rates(tenor_array, curve_paths_array[0, 0])
    surface_paths = np.empty(
        (curve_paths_array.shape[0], time_array.size, expiries_array.size, swap_tenors_array.size),
        dtype=np.float64,
    )
    for path_index in range(curve_paths_array.shape[0]):
        for time_index in range(time_array.size):
            current_curve = CurveSnapshot.from_zero_rates(tenor_array, curve_paths_array[path_index, time_index])
            surface_paths[path_index, time_index] = apply_curve_surface_adjustment(
                base_vols,
                current_curve,
                reference_curve,
                expiries_array,
                swap_tenors_array,
            )

    return SwaptionSurfacePaths(
        time_grid=time_array,
        expiries=expiries_array,
        swap_tenors=swap_tenors_array,
        normal_volatility_paths=surface_paths,
    )


__all__ = [
    "SwaptionSurfacePaths",
    "SwaptionSurfaceSnapshot",
    "SwaptionSurfaceTrajectory",
    "apply_curve_surface_adjustment",
    "build_surface_paths",
    "g2pp_atm_normal_volatilities",
    "hull_white_atm_normal_volatilities",
]
