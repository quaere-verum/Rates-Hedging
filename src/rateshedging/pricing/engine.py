from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np
from numpy.typing import NDArray

from rateshedging.instruments.instrument import InterestRateInstrument
from rateshedging.instruments.bermudan_swaption import BermudanSwaption
from rateshedging.instruments.swap import Swap
from rateshedging.instruments.swaption import Swaption
from rateshedging.models.model import InterestRateModel
from rateshedging.pricing.exercise_strategy import ExerciseStrategy
from rateshedging.pricing.regression import LongstaffSchwartzExerciseStrategy


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PricingResult:
    price: float
    standard_error: float
    n_paths: int
    exercise_probabilities: FloatArray | None = None


def _time_index(time_grid: FloatArray, target_time: float) -> int:
    candidate = int(np.searchsorted(time_grid, target_time))
    if candidate < time_grid.size and abs(float(time_grid[candidate]) - target_time) <= 1.0e-12:
        return candidate
    if candidate > 0 and abs(float(time_grid[candidate - 1]) - target_time) <= 1.0e-12:
        return candidate - 1
    raise ValueError(f"Model time grid must contain {target_time}.")

class MonteCarloPricingEngine:
    def __init__(
        self,
        n_paths: int = 50_000,
        exercise_strategy: ExerciseStrategy | None = None,
    ) -> None:
        if int(n_paths) != n_paths or n_paths <= 0:
            raise ValueError("n_paths must be a strictly positive integer.")
        self.n_paths = int(n_paths)
        self.exercise_strategy = exercise_strategy or LongstaffSchwartzExerciseStrategy()
        self._path_cache: WeakKeyDictionary[InterestRateModel, object] = WeakKeyDictionary()

    def _paths_for_model(self, model: InterestRateModel):
        cached = self._path_cache.get(model)
        if cached is None:
            cached = model.generate_paths(self.n_paths)
            self._path_cache[model] = cached
        return cached

    def price(
        self,
        instrument: InterestRateInstrument,
        model: InterestRateModel,
        valuation_time: float = 0.0,
    ) -> PricingResult:
        if isinstance(instrument, Swap):
            return self._price_swap(instrument, model, valuation_time)
        if isinstance(instrument, Swaption):
            return self._price_swaption(instrument, model, valuation_time)
        if isinstance(instrument, BermudanSwaption):
            return self._price_bermudan_swaption(instrument, model, valuation_time)
        raise TypeError(f"Unsupported instrument type: {type(instrument)!r}")

    def _price_swap(self, instrument: Swap, model: InterestRateModel, valuation_time: float) -> PricingResult:
        paths = model.generate_paths(1)
        zero_rates = paths.yield_curve_paths[0, 0]
        price = instrument.present_value_from_zero_rates(
            zero_rates,
            paths.yield_curve_tenors,
            valuation_time=valuation_time,
        )
        return PricingResult(price=float(price), standard_error=0.0, n_paths=1)

    def _price_swaption(self, instrument: Swaption, model: InterestRateModel, valuation_time: float) -> PricingResult:
        paths = self._paths_for_model(model)
        expiry_index = _time_index(paths.time, instrument.expiry - valuation_time)

        exercise_value = instrument.exercise_value_from_zero_rates(
            paths.yield_curve_paths[:, expiry_index, :],
            paths.yield_curve_tenors,
        )
        discounted_payoff = instrument.direction * paths.stochastic_discount_factors[:, expiry_index] * exercise_value

        return PricingResult(
            price=float(np.mean(discounted_payoff)),
            standard_error=float(np.std(discounted_payoff, ddof=1) / np.sqrt(self.n_paths)),
            n_paths=self.n_paths,
            exercise_probabilities=np.asarray([np.mean(exercise_value > 0.0)], dtype=np.float64),
        )

    def _price_bermudan_swaption(
        self,
        instrument: BermudanSwaption,
        model: InterestRateModel,
        valuation_time: float,
    ) -> PricingResult:
        paths = self._paths_for_model(model)
        remaining_exercise_times = instrument.exercise_times[
            instrument.exercise_times >= valuation_time - 1.0e-12
        ]
        if remaining_exercise_times.size == 0:
            return PricingResult(price=0.0, standard_error=0.0, n_paths=self.n_paths)

        exercise_indices = np.asarray(
            [_time_index(paths.time, float(time - valuation_time)) for time in remaining_exercise_times],
            dtype=np.int64,
        )

        intrinsic_values = np.column_stack(
            [
                instrument.exercise_value_from_zero_rates(
                    paths.yield_curve_paths[:, exercise_index, :],
                    paths.yield_curve_tenors,
                    float(exercise_time),
                )
                for exercise_index, exercise_time in zip(exercise_indices, remaining_exercise_times, strict=True)
            ]
        )

        path_discounts = paths.stochastic_discount_factors[:, exercise_indices]
        state_factors = paths.yield_curve_factors[:, exercise_indices, :]
        regression_result = self.exercise_strategy.fit(
            intrinsic_values=intrinsic_values,
            discount_factors=path_discounts,
            state_factors=state_factors,
        )
        discounted_payoff = instrument.direction * regression_result.discounted_payoffs

        return PricingResult(
            price=float(np.mean(discounted_payoff)),
            standard_error=float(np.std(discounted_payoff, ddof=1) / np.sqrt(self.n_paths)),
            n_paths=self.n_paths,
            exercise_probabilities=regression_result.exercise_probabilities,
        )


__all__ = ["MonteCarloPricingEngine", "PricingResult"]
