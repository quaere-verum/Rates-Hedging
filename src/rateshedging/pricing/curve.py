from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rateshedging.models._curve_utils import as_1d_float_array, validate_tenors


FloatArray = NDArray[np.float64]


def zero_rates_to_discount_factors(zero_rates: ArrayLike, tenors: ArrayLike) -> FloatArray:
    rates = np.asarray(zero_rates, dtype=np.float64)
    tenor_array = np.asarray(tenors, dtype=np.float64)
    return np.exp(-rates * tenor_array)


def _curve_nodes_from_zero_rates(zero_rates: ArrayLike, tenors: ArrayLike) -> tuple[FloatArray, FloatArray]:
    tenor_array = validate_tenors(tenors)
    rates = np.asarray(zero_rates, dtype=np.float64)
    if rates.shape[-1] != tenor_array.size:
        raise ValueError("The last dimension of zero_rates must match the size of tenors.")

    log_discount_nodes = np.concatenate(
        [
            np.zeros(rates.shape[:-1] + (1,), dtype=np.float64),
            -rates * tenor_array,
        ],
        axis=-1,
    )
    node_times = np.concatenate(([0.0], tenor_array))
    return node_times, log_discount_nodes


def discount_from_zero_rates(zero_rates: ArrayLike, tenors: ArrayLike, query_times: ArrayLike) -> FloatArray:
    node_times, log_discount_nodes = _curve_nodes_from_zero_rates(zero_rates, tenors)
    query = np.asarray(query_times, dtype=np.float64)
    if np.any(query < 0.0):
        raise ValueError("Query times must be non-negative.")

    right_index = np.searchsorted(node_times, query, side="right")
    right_index = np.clip(right_index, 1, node_times.size - 1)
    left_index = right_index - 1

    left_time = node_times[left_index]
    right_time = node_times[right_index]
    weight = (query - left_time) / (right_time - left_time)

    left_values = np.take(log_discount_nodes, left_index, axis=-1)
    right_values = np.take(log_discount_nodes, right_index, axis=-1)
    interpolated = left_values + weight * (right_values - left_values)
    return np.exp(interpolated)


def forward_rates_from_zero_rates(
    zero_rates: ArrayLike,
    tenors: ArrayLike,
    accrual_start_times: ArrayLike,
    accrual_end_times: ArrayLike,
) -> FloatArray:
    start_times = np.asarray(accrual_start_times, dtype=np.float64)
    end_times = np.asarray(accrual_end_times, dtype=np.float64)
    if start_times.shape != end_times.shape:
        raise ValueError("accrual_start_times and accrual_end_times must have the same shape.")
    if np.any(end_times <= start_times):
        raise ValueError("Each accrual_end_time must exceed the corresponding accrual_start_time.")

    discount_start = discount_from_zero_rates(zero_rates, tenors, start_times)
    discount_end = discount_from_zero_rates(zero_rates, tenors, end_times)
    return (discount_start / discount_end - 1.0) / (end_times - start_times)


@dataclass(frozen=True)
class CurveSnapshot:
    tenors: FloatArray
    zero_rates: FloatArray

    def __post_init__(self) -> None:
        tenor_array = validate_tenors(self.tenors)
        zero_rate_array = as_1d_float_array(self.zero_rates, "zero_rates")
        if tenor_array.shape != zero_rate_array.shape:
            raise ValueError("tenors and zero_rates must have the same shape.")
        object.__setattr__(self, "tenors", tenor_array)
        object.__setattr__(self, "zero_rates", zero_rate_array)

    @classmethod
    def from_zero_rates(cls, tenors: ArrayLike, zero_rates: ArrayLike) -> "CurveSnapshot":
        return cls(np.asarray(tenors, dtype=np.float64), np.asarray(zero_rates, dtype=np.float64))

    @classmethod
    def from_discount_factors(cls, tenors: ArrayLike, discount_factors: ArrayLike) -> "CurveSnapshot":
        tenor_array = validate_tenors(tenors)
        discounts = as_1d_float_array(discount_factors, "discount_factors")
        if tenor_array.shape != discounts.shape:
            raise ValueError("tenors and discount_factors must have the same shape.")
        if np.any(discounts <= 0.0):
            raise ValueError("discount_factors must be strictly positive.")
        zero_rates = -np.log(discounts) / tenor_array
        return cls(tenor_array, zero_rates)

    def discount(self, query_times: ArrayLike) -> FloatArray:
        return discount_from_zero_rates(self.zero_rates, self.tenors, query_times)

    def forward_rate(self, accrual_start: float, accrual_end: float) -> float:
        rate = forward_rates_from_zero_rates(
            self.zero_rates,
            self.tenors,
            np.asarray([accrual_start], dtype=np.float64),
            np.asarray([accrual_end], dtype=np.float64),
        )
        return float(rate[0])

    @property
    def discount_factors(self) -> FloatArray:
        return zero_rates_to_discount_factors(self.zero_rates, self.tenors)
