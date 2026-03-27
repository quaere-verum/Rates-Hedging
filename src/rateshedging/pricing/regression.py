from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rateshedging.pricing.exercise_strategy import ExerciseStrategy, ExerciseStrategyResult


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BasisFunction = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class BermudanRegressionResult(ExerciseStrategyResult):
    regression_coefficients: tuple[FloatArray | None, ...]


def default_basis_function(intrinsic_values: FloatArray, factors: FloatArray) -> FloatArray:
    columns = [
        np.ones_like(intrinsic_values),
        intrinsic_values,
        intrinsic_values * intrinsic_values,
    ]

    for factor_index in range(factors.shape[1]):
        factor = factors[:, factor_index]
        columns.append(factor)
        columns.append(factor * factor)

    if factors.shape[1] > 1:
        columns.append(np.prod(factors, axis=1))

    return np.column_stack(columns)


class LongstaffSchwartzExerciseStrategy(ExerciseStrategy):
    def __init__(
        self,
        basis_function: BasisFunction | None = None,
        min_regression_paths: int = 32,
    ) -> None:
        if min_regression_paths <= 0:
            raise ValueError("min_regression_paths must be strictly positive.")
        self.basis_function = basis_function or default_basis_function
        self.min_regression_paths = int(min_regression_paths)

    def fit(
        self,
        intrinsic_values: ArrayLike,
        discount_factors: ArrayLike,
        state_factors: ArrayLike,
    ) -> BermudanRegressionResult:
        intrinsic = np.asarray(intrinsic_values, dtype=np.float64)
        discounts = np.asarray(discount_factors, dtype=np.float64)
        factors = np.asarray(state_factors, dtype=np.float64)

        if intrinsic.ndim != 2:
            raise ValueError("intrinsic_values must be a two-dimensional array.")
        if discounts.shape != intrinsic.shape:
            raise ValueError("discount_factors must have the same shape as intrinsic_values.")
        if factors.ndim != 3 or factors.shape[:2] != intrinsic.shape:
            raise ValueError("state_factors must have shape (n_paths, n_exercise_dates, n_factors).")

        n_paths, n_steps = intrinsic.shape
        continuation_values = np.zeros_like(intrinsic)
        candidate_exercise = np.zeros_like(intrinsic, dtype=bool)
        regression_coefficients: list[FloatArray | None] = [None] * n_steps

        candidate_exercise[:, -1] = intrinsic[:, -1] > 0.0
        continuation_value = intrinsic[:, -1].copy()
        continuation_values[:, -1] = continuation_value

        for step in range(n_steps - 2, -1, -1):
            realized_continuation = (discounts[:, step + 1] / discounts[:, step]) * continuation_value
            immediate_exercise = intrinsic[:, step]
            in_the_money = immediate_exercise > 0.0

            basis = self.basis_function(immediate_exercise, factors[:, step, :])
            if np.count_nonzero(in_the_money) >= max(self.min_regression_paths, basis.shape[1]):
                coefficients, *_ = np.linalg.lstsq(
                    basis[in_the_money],
                    realized_continuation[in_the_money],
                    rcond=None,
                )
                estimated_continuation = basis @ coefficients
                regression_coefficients[step] = coefficients
            elif np.any(in_the_money):
                estimated_continuation = np.full(n_paths, np.mean(realized_continuation[in_the_money]))
            else:
                estimated_continuation = np.zeros(n_paths, dtype=np.float64)

            candidate_exercise[:, step] = in_the_money & (immediate_exercise >= estimated_continuation)
            continuation_value = np.where(candidate_exercise[:, step], immediate_exercise, realized_continuation)
            continuation_values[:, step] = continuation_value

        exercise_indicators = np.zeros_like(candidate_exercise)
        alive = np.ones(n_paths, dtype=bool)
        for step in range(n_steps):
            exercised = alive & candidate_exercise[:, step]
            exercise_indicators[:, step] = exercised
            alive &= ~exercised

        discounted_payoffs = np.sum(exercise_indicators * intrinsic * discounts, axis=1)
        exercise_probabilities = np.mean(exercise_indicators, axis=0)

        return BermudanRegressionResult(
            regression_coefficients=tuple(regression_coefficients),
            continuation_values=continuation_values,
            candidate_exercise=candidate_exercise,
            exercise_indicators=exercise_indicators,
            discounted_payoffs=discounted_payoffs,
            exercise_probabilities=exercise_probabilities,
        )


__all__ = [
    "BermudanRegressionResult",
    "LongstaffSchwartzExerciseStrategy",
    "default_basis_function",
]
