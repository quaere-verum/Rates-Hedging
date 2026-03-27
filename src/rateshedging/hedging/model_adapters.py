from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rateshedging.models.g2pp import G2PPModel
from rateshedging.models.hull_white import HullWhiteModel
from rateshedging.models.model import InterestRateModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ModelParameterBump:
    name: str
    bump_size: float

    def __post_init__(self) -> None:
        if self.bump_size <= 0.0:
            raise ValueError("bump_size must be strictly positive.")


class InterestRateModelAdapter(ABC):
    @abstractmethod
    def build(
        self,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_times: FloatArray,
        discount_factors: FloatArray,
        yield_curve_tenors: FloatArray,
    ) -> InterestRateModel:
        raise NotImplementedError

    @property
    def vega_parameters(self) -> tuple[ModelParameterBump, ...]:
        return ()

    def build_with_parameter_shift(
        self,
        parameter_name: str,
        parameter_shift: float,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_times: FloatArray,
        discount_factors: FloatArray,
        yield_curve_tenors: FloatArray,
    ) -> InterestRateModel:
        raise ValueError(f"Unsupported model parameter bump: {parameter_name}.")


@dataclass(frozen=True)
class HullWhiteModelAdapter(InterestRateModelAdapter):
    mean_reversion: float
    volatility: float
    seed: int | None = None
    volatility_bump: float = 1.0e-4

    @property
    def vega_parameters(self) -> tuple[ModelParameterBump, ...]:
        return (ModelParameterBump("volatility", self.volatility_bump),)

    def build(
        self,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_times: FloatArray,
        discount_factors: FloatArray,
        yield_curve_tenors: FloatArray,
    ) -> HullWhiteModel:
        del valuation_time
        return HullWhiteModel(
            mean_reversion=self.mean_reversion,
            volatility=self.volatility,
            time_grid=local_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=yield_curve_tenors,
            seed=self.seed,
        )

    def build_with_parameter_shift(
        self,
        parameter_name: str,
        parameter_shift: float,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_times: FloatArray,
        discount_factors: FloatArray,
        yield_curve_tenors: FloatArray,
    ) -> HullWhiteModel:
        del valuation_time
        if parameter_name != "volatility":
            raise ValueError(f"Unsupported Hull-White parameter bump: {parameter_name}.")
        bumped_volatility = self.volatility + parameter_shift
        if bumped_volatility <= 0.0:
            raise ValueError("Bumped Hull-White volatility must remain strictly positive.")
        return HullWhiteModel(
            mean_reversion=self.mean_reversion,
            volatility=bumped_volatility,
            time_grid=local_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=yield_curve_tenors,
            seed=self.seed,
        )


@dataclass(frozen=True)
class G2PPModelAdapter(InterestRateModelAdapter):
    mean_reversion_x: float
    mean_reversion_y: float
    volatility_x: float
    volatility_y: float
    correlation: float
    seed: int | None = None
    volatility_x_bump: float = 1.0e-4
    volatility_y_bump: float = 1.0e-4

    @property
    def vega_parameters(self) -> tuple[ModelParameterBump, ...]:
        return (
            ModelParameterBump("volatility_x", self.volatility_x_bump),
            ModelParameterBump("volatility_y", self.volatility_y_bump),
        )

    def build(
        self,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_times: FloatArray,
        discount_factors: FloatArray,
        yield_curve_tenors: FloatArray,
    ) -> G2PPModel:
        del valuation_time
        return G2PPModel(
            mean_reversion_x=self.mean_reversion_x,
            mean_reversion_y=self.mean_reversion_y,
            volatility_x=self.volatility_x,
            volatility_y=self.volatility_y,
            correlation=self.correlation,
            time_grid=local_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=yield_curve_tenors,
            seed=self.seed,
        )

    def build_with_parameter_shift(
        self,
        parameter_name: str,
        parameter_shift: float,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_times: FloatArray,
        discount_factors: FloatArray,
        yield_curve_tenors: FloatArray,
    ) -> G2PPModel:
        del valuation_time
        volatility_x = self.volatility_x
        volatility_y = self.volatility_y
        if parameter_name == "volatility_x":
            volatility_x += parameter_shift
        elif parameter_name == "volatility_y":
            volatility_y += parameter_shift
        else:
            raise ValueError(f"Unsupported G2++ parameter bump: {parameter_name}.")
        if volatility_x <= 0.0 or volatility_y <= 0.0:
            raise ValueError("Bumped G2++ volatilities must remain strictly positive.")
        return G2PPModel(
            mean_reversion_x=self.mean_reversion_x,
            mean_reversion_y=self.mean_reversion_y,
            volatility_x=volatility_x,
            volatility_y=volatility_y,
            correlation=self.correlation,
            time_grid=local_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=yield_curve_tenors,
            seed=self.seed,
        )


__all__ = [
    "G2PPModelAdapter",
    "HullWhiteModelAdapter",
    "InterestRateModelAdapter",
    "ModelParameterBump",
]
