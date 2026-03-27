from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from rateshedging.calibration.swaption_surface import (
    SwaptionSurfaceSnapshot,
    g2pp_atm_normal_volatilities,
    hull_white_atm_normal_volatilities,
)
from rateshedging.pricing.curve import CurveSnapshot

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


@dataclass(frozen=True)
class AdapterCalibration:
    adapter: "InterestRateModelAdapter"
    parameter_names: tuple[str, ...]
    parameter_values: FloatArray
    rmse: float
    max_abs_error: float


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

    @property
    def calibration_parameter_names(self) -> tuple[str, ...]:
        return ()

    @property
    def calibration_parameter_values(self) -> FloatArray:
        return np.zeros(0, dtype=np.float64)

    def calibrate(
        self,
        valuation_time: float,
        curve_snapshot: CurveSnapshot,
        swaption_surface: SwaptionSurfaceSnapshot,
    ) -> AdapterCalibration:
        del valuation_time, curve_snapshot, swaption_surface
        return AdapterCalibration(
            adapter=self,
            parameter_names=self.calibration_parameter_names,
            parameter_values=self.calibration_parameter_values.copy(),
            rmse=0.0,
            max_abs_error=0.0,
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
    ) -> InterestRateModel:
        raise ValueError(f"Unsupported model parameter bump: {parameter_name}.")

    @abstractmethod
    def surface_normal_volatilities(self, swaption_surface: SwaptionSurfaceSnapshot) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True)
class HullWhiteModelAdapter(InterestRateModelAdapter):
    mean_reversion: float
    volatility: float
    seed: int | None = None
    volatility_bump: float = 1.0e-4

    @property
    def vega_parameters(self) -> tuple[ModelParameterBump, ...]:
        return (ModelParameterBump("volatility", self.volatility_bump),)

    @property
    def calibration_parameter_names(self) -> tuple[str, ...]:
        return ("volatility",)

    @property
    def calibration_parameter_values(self) -> FloatArray:
        return np.asarray([self.volatility], dtype=np.float64)

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

    def surface_normal_volatilities(self, swaption_surface: SwaptionSurfaceSnapshot) -> FloatArray:
        return hull_white_atm_normal_volatilities(
            self.mean_reversion,
            self.volatility,
            swaption_surface.expiries,
            swaption_surface.swap_tenors,
        )

    def calibrate(
        self,
        valuation_time: float,
        curve_snapshot: CurveSnapshot,
        swaption_surface: SwaptionSurfaceSnapshot,
    ) -> AdapterCalibration:
        del valuation_time, curve_snapshot
        loadings = hull_white_atm_normal_volatilities(
            self.mean_reversion,
            1.0,
            swaption_surface.expiries,
            swaption_surface.swap_tenors,
        )
        numerator = float(np.sum(swaption_surface.weights * loadings * swaption_surface.normal_volatilities))
        denominator = float(np.sum(swaption_surface.weights * loadings * loadings))
        calibrated_volatility = max(numerator / denominator, 1.0e-8)
        calibrated_adapter = replace(self, volatility=calibrated_volatility)
        calibration_errors = calibrated_adapter.surface_normal_volatilities(swaption_surface) - swaption_surface.normal_volatilities
        weighted_errors = np.sqrt(swaption_surface.weights) * calibration_errors
        return AdapterCalibration(
            adapter=calibrated_adapter,
            parameter_names=calibrated_adapter.calibration_parameter_names,
            parameter_values=calibrated_adapter.calibration_parameter_values,
            rmse=float(np.sqrt(np.mean(weighted_errors * weighted_errors))),
            max_abs_error=float(np.max(np.abs(calibration_errors))),
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

    @property
    def calibration_parameter_names(self) -> tuple[str, ...]:
        return ("volatility_x", "volatility_y")

    @property
    def calibration_parameter_values(self) -> FloatArray:
        return np.asarray([self.volatility_x, self.volatility_y], dtype=np.float64)

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

    def surface_normal_volatilities(self, swaption_surface: SwaptionSurfaceSnapshot) -> FloatArray:
        return g2pp_atm_normal_volatilities(
            self.mean_reversion_x,
            self.mean_reversion_y,
            self.volatility_x,
            self.volatility_y,
            self.correlation,
            swaption_surface.expiries,
            swaption_surface.swap_tenors,
        )

    def calibrate(
        self,
        valuation_time: float,
        curve_snapshot: CurveSnapshot,
        swaption_surface: SwaptionSurfaceSnapshot,
    ) -> AdapterCalibration:
        del valuation_time, curve_snapshot

        target_vols = swaption_surface.normal_volatilities
        sqrt_weights = np.sqrt(swaption_surface.weights)

        def objective(parameters: FloatArray) -> FloatArray:
            model_vols = g2pp_atm_normal_volatilities(
                self.mean_reversion_x,
                self.mean_reversion_y,
                float(parameters[0]),
                float(parameters[1]),
                self.correlation,
                swaption_surface.expiries,
                swaption_surface.swap_tenors,
            )
            return (sqrt_weights * (model_vols - target_vols)).ravel()

        optimization = least_squares(
            objective,
            x0=self.calibration_parameter_values,
            bounds=(np.full(2, 1.0e-8, dtype=np.float64), np.full(2, np.inf, dtype=np.float64)),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        calibrated_adapter = replace(
            self,
            volatility_x=float(optimization.x[0]),
            volatility_y=float(optimization.x[1]),
        )
        calibration_errors = calibrated_adapter.surface_normal_volatilities(swaption_surface) - target_vols
        weighted_errors = np.sqrt(swaption_surface.weights) * calibration_errors
        return AdapterCalibration(
            adapter=calibrated_adapter,
            parameter_names=calibrated_adapter.calibration_parameter_names,
            parameter_values=calibrated_adapter.calibration_parameter_values,
            rmse=float(np.sqrt(np.mean(weighted_errors * weighted_errors))),
            max_abs_error=float(np.max(np.abs(calibration_errors))),
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
    "AdapterCalibration",
    "G2PPModelAdapter",
    "HullWhiteModelAdapter",
    "InterestRateModelAdapter",
    "ModelParameterBump",
]
