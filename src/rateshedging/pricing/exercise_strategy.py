from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ExerciseStrategyResult:
    continuation_values: FloatArray
    candidate_exercise: BoolArray
    exercise_indicators: BoolArray
    discounted_payoffs: FloatArray
    exercise_probabilities: FloatArray


class ExerciseStrategy(ABC):
    @abstractmethod
    def fit(
        self,
        intrinsic_values: ArrayLike,
        discount_factors: ArrayLike,
        state_factors: ArrayLike,
    ) -> ExerciseStrategyResult:
        raise NotImplementedError


__all__ = ["ExerciseStrategy", "ExerciseStrategyResult"]
