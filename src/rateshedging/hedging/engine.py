from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rateshedging.calibration.swaption_surface import SwaptionSurfaceTrajectory
from rateshedging.hedging.model_adapters import InterestRateModelAdapter
from rateshedging.instruments.bermudan_swaption import BermudanSwaption
from rateshedging.instruments.instrument import InterestRateInstrument
from rateshedging.instruments.swap import Swap
from rateshedging.instruments.swaption import Swaption
from rateshedging.models.model import InterestRateModel
from rateshedging.pricing.curve import CurveSnapshot, zero_rates_to_discount_factors
from rateshedging.pricing.engine import MonteCarloPricingEngine


FloatArray = NDArray[np.float64]
ModelFactory = Callable[[float, FloatArray, FloatArray, FloatArray, FloatArray], InterestRateModel]


@dataclass(frozen=True)
class HedgingResult:
    time: FloatArray
    key_rate_tenors: FloatArray
    vega_labels: tuple[str, ...]
    vega_bump_sizes: FloatArray
    calibrated_parameter_names: tuple[str, ...]
    calibrated_parameters: FloatArray
    calibration_rmse: FloatArray
    calibration_max_abs_error: FloatArray
    target_values: FloatArray
    hedge_values: FloatArray
    hedge_weights: FloatArray
    cash_account: FloatArray
    portfolio_value: FloatArray
    incremental_pnl: FloatArray
    cumulative_pnl: FloatArray
    cash_carry_pnl: FloatArray
    target_cashflow_pnl: FloatArray
    hedge_cashflow_pnl: FloatArray
    target_revaluation_pnl: FloatArray
    hedge_revaluation_pnl: FloatArray
    rebalancing_cashflow: FloatArray
    target_key_rate_sensitivities: FloatArray
    hedge_key_rate_sensitivities: FloatArray
    residual_key_rate_sensitivities: FloatArray
    target_vega_sensitivities: FloatArray
    hedge_vega_sensitivities: FloatArray
    residual_vega_sensitivities: FloatArray


class DynamicHedgingEngine:
    def __init__(
        self,
        pricing_engine: MonteCarloPricingEngine,
        model_factory: ModelFactory | None = None,
        model_adapter: InterestRateModelAdapter | None = None,
        curve_bump_size: float = 1.0e-4,
        include_vega: bool = False,
        exercise_tolerance: float = 1.0e-8,
        ridge_penalty: float = 0.0,
        pnl_identity_tolerance: float = 1.0e-6,
    ) -> None:
        if curve_bump_size <= 0.0:
            raise ValueError("curve_bump_size must be strictly positive.")
        if ridge_penalty < 0.0:
            raise ValueError("ridge_penalty must be non-negative.")
        if pnl_identity_tolerance <= 0.0:
            raise ValueError("pnl_identity_tolerance must be strictly positive.")

        self.pricing_engine = pricing_engine
        self.model_factory = model_factory
        self.model_adapter = model_adapter
        self.curve_bump_size = float(curve_bump_size)
        self.include_vega = bool(include_vega)
        self.exercise_tolerance = float(exercise_tolerance)
        self.ridge_penalty = float(ridge_penalty)
        self.pnl_identity_tolerance = float(pnl_identity_tolerance)

    def run(
        self,
        yield_curve_trajectory: ArrayLike,
        time_grid: ArrayLike,
        yield_curve_tenors: ArrayLike,
        target_instrument: InterestRateInstrument,
        hedging_instruments: Sequence[InterestRateInstrument],
        swaption_surface_trajectory: SwaptionSurfaceTrajectory | None = None,
    ) -> HedgingResult:
        curve_path = np.asarray(yield_curve_trajectory, dtype=np.float64)
        times = np.asarray(time_grid, dtype=np.float64)
        tenors = np.asarray(yield_curve_tenors, dtype=np.float64)

        if curve_path.ndim != 2:
            raise ValueError("yield_curve_trajectory must be a two-dimensional array.")
        if curve_path.shape != (times.size, tenors.size):
            raise ValueError("yield_curve_trajectory shape must match time_grid and yield_curve_tenors.")
        if swaption_surface_trajectory is not None and not np.allclose(
            swaption_surface_trajectory.time_grid,
            times,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("swaption_surface_trajectory time_grid must match time_grid.")

        vega_parameters = self.model_adapter.vega_parameters if self.include_vega and self.model_adapter else ()
        if self.include_vega and self.model_adapter is None:
            raise ValueError("A model_adapter is required when include_vega=True.")

        vega_labels = tuple(parameter.name for parameter in vega_parameters)
        vega_bump_sizes = np.asarray([parameter.bump_size for parameter in vega_parameters], dtype=np.float64)
        calibrated_parameter_names = self.model_adapter.calibration_parameter_names if self.model_adapter else ()

        n_times = times.size
        n_hedges = len(hedging_instruments)
        n_vega = len(vega_parameters)
        n_calibration_parameters = len(calibrated_parameter_names)
        hedging_instruments = list(hedging_instruments)

        calibrated_parameters = np.zeros((n_times, n_calibration_parameters), dtype=np.float64)
        calibration_rmse = np.zeros(n_times, dtype=np.float64)
        calibration_max_abs_error = np.zeros(n_times, dtype=np.float64)
        target_values = np.zeros(n_times, dtype=np.float64)
        hedge_values = np.zeros((n_times, n_hedges), dtype=np.float64)
        hedge_weights = np.zeros((n_times, n_hedges), dtype=np.float64)
        cash_account = np.zeros(n_times, dtype=np.float64)
        portfolio_value = np.zeros(n_times, dtype=np.float64)
        incremental_pnl = np.zeros(n_times, dtype=np.float64)
        cumulative_pnl = np.zeros(n_times, dtype=np.float64)
        cash_carry_pnl = np.zeros(n_times, dtype=np.float64)
        target_cashflow_pnl = np.zeros(n_times, dtype=np.float64)
        hedge_cashflow_pnl = np.zeros(n_times, dtype=np.float64)
        target_revaluation_pnl = np.zeros(n_times, dtype=np.float64)
        hedge_revaluation_pnl = np.zeros(n_times, dtype=np.float64)
        rebalancing_cashflow = np.zeros(n_times, dtype=np.float64)
        target_key_rate_sensitivities = np.zeros((n_times, tenors.size), dtype=np.float64)
        hedge_key_rate_sensitivities = np.zeros((n_times, n_hedges, tenors.size), dtype=np.float64)
        residual_key_rate_sensitivities = np.zeros((n_times, tenors.size), dtype=np.float64)
        target_vega_sensitivities = np.zeros((n_times, n_vega), dtype=np.float64)
        hedge_vega_sensitivities = np.zeros((n_times, n_hedges, n_vega), dtype=np.float64)
        residual_vega_sensitivities = np.zeros((n_times, n_vega), dtype=np.float64)

        target_active = True
        hedge_active = np.ones(n_hedges, dtype=bool)
        previous_weights = np.zeros(n_hedges, dtype=np.float64)
        previous_target_value = 0.0
        previous_hedge_values = np.zeros(n_hedges, dtype=np.float64)
        cash = 0.0
        previous_portfolio_value = 0.0
        previous_curve: CurveSnapshot | None = None

        for time_index, valuation_time in enumerate(times):
            current_curve = CurveSnapshot.from_zero_rates(tenors, curve_path[time_index])
            active_model_adapter = self.model_adapter
            if self.model_adapter is not None and swaption_surface_trajectory is not None:
                calibration = self.model_adapter.calibrate(
                    valuation_time,
                    current_curve,
                    swaption_surface_trajectory.snapshot(time_index),
                )
                active_model_adapter = calibration.adapter
                calibrated_parameters[time_index] = calibration.parameter_values
                calibration_rmse[time_index] = calibration.rmse
                calibration_max_abs_error[time_index] = calibration.max_abs_error
            elif active_model_adapter is not None and n_calibration_parameters > 0:
                calibrated_parameters[time_index] = active_model_adapter.calibration_parameter_values

            target_swap_cashflow = 0.0
            hedge_swap_cashflows = np.zeros(n_hedges, dtype=np.float64)
            if time_index > 0:
                previous_time = times[time_index - 1]
                dt = valuation_time - previous_time
                accrued_cash = cash / float(previous_curve.discount(np.asarray([dt], dtype=np.float64))[0])
                cash_carry_pnl[time_index] = accrued_cash - cash
                cash = accrued_cash

                target_swap_cashflow = self._realized_swap_cashflow(
                    target_instrument,
                    previous_curve,
                    previous_time,
                    valuation_time,
                )
                cash += target_swap_cashflow
                target_cashflow_pnl[time_index] = target_swap_cashflow

                hedge_swap_cashflows = np.asarray(
                    [
                        self._realized_swap_cashflow(
                            hedge_instrument,
                            previous_curve,
                            previous_time,
                            valuation_time,
                        )
                        for hedge_instrument in hedging_instruments
                    ],
                    dtype=np.float64,
                )
                hedge_cashflow_pnl[time_index] = -float(previous_weights @ hedge_swap_cashflows)
                cash += hedge_cashflow_pnl[time_index]

            provisional_target_value = self._mark_to_market(
                target_instrument,
                current_curve,
                curve_path[time_index],
                tenors,
                valuation_time,
                target_active,
                active_model_adapter,
            )
            if n_hedges == 0:
                provisional_hedge_values = np.zeros(0, dtype=np.float64)
            else:
                provisional_hedge_values = np.asarray(
                    [
                        self._mark_to_market(
                            hedge_instrument,
                            current_curve,
                            curve_path[time_index],
                            tenors,
                            valuation_time,
                            active,
                            active_model_adapter,
                        )
                        for hedge_instrument, active in zip(hedging_instruments, hedge_active, strict=True)
                    ],
                    dtype=np.float64,
                )

            target_option_settlement = 0.0
            if target_active:
                target_option_settlement, target_active = self._option_settlement(
                    target_instrument,
                    current_curve,
                    valuation_time,
                    provisional_target_value,
                    target_active,
                )
                cash += target_option_settlement
                target_cashflow_pnl[time_index] += target_option_settlement
                if not target_active:
                    provisional_target_value = 0.0

            hedge_option_settlements = np.zeros(n_hedges, dtype=np.float64)
            for hedge_index, hedge_instrument in enumerate(hedging_instruments):
                settlement, hedge_active[hedge_index] = self._option_settlement(
                    hedge_instrument,
                    current_curve,
                    valuation_time,
                    provisional_hedge_values[hedge_index],
                    bool(hedge_active[hedge_index]),
                )
                hedge_option_settlements[hedge_index] = settlement
                cash -= previous_weights[hedge_index] * settlement
                if not hedge_active[hedge_index]:
                    provisional_hedge_values[hedge_index] = 0.0

            hedge_cashflow_pnl[time_index] -= float(previous_weights @ hedge_option_settlements)
            target_revaluation_pnl[time_index] = provisional_target_value - previous_target_value
            hedge_revaluation_pnl[time_index] = -float(
                previous_weights @ (provisional_hedge_values - previous_hedge_values)
            )

            pre_rebalance_portfolio = provisional_target_value - previous_weights @ provisional_hedge_values + cash
            incremental_pnl[time_index] = pre_rebalance_portfolio - previous_portfolio_value
            cumulative_pnl[time_index] = pre_rebalance_portfolio

            explained_pnl = (
                cash_carry_pnl[time_index]
                + target_cashflow_pnl[time_index]
                + hedge_cashflow_pnl[time_index]
                + target_revaluation_pnl[time_index]
                + hedge_revaluation_pnl[time_index]
            )
            if not np.isclose(
                explained_pnl,
                incremental_pnl[time_index],
                atol=self.pnl_identity_tolerance,
                rtol=1.0e-9,
            ):
                raise RuntimeError("PnL decomposition identity failed during hedging.")

            target_sensitivities = self._instrument_curve_sensitivities(
                target_instrument,
                curve_path[time_index],
                tenors,
                valuation_time,
                target_active,
                active_model_adapter,
            )
            if n_hedges == 0:
                hedge_sensitivities = np.zeros((0, tenors.size), dtype=np.float64)
            else:
                hedge_sensitivities = np.asarray(
                    [
                        self._instrument_curve_sensitivities(
                            hedge_instrument,
                            curve_path[time_index],
                            tenors,
                            valuation_time,
                            bool(active),
                            active_model_adapter,
                        )
                        for hedge_instrument, active in zip(hedging_instruments, hedge_active, strict=True)
                    ],
                    dtype=np.float64,
                )

            target_vegas = self._instrument_vega_sensitivities(
                target_instrument,
                curve_path[time_index],
                tenors,
                valuation_time,
                target_active,
                vega_parameters,
                active_model_adapter,
            )
            if n_hedges == 0:
                hedge_vegas = np.zeros((0, n_vega), dtype=np.float64)
            else:
                hedge_vegas = np.asarray(
                    [
                        self._instrument_vega_sensitivities(
                            hedge_instrument,
                            curve_path[time_index],
                            tenors,
                            valuation_time,
                            bool(active),
                            vega_parameters,
                            active_model_adapter,
                        )
                        for hedge_instrument, active in zip(hedging_instruments, hedge_active, strict=True)
                    ],
                    dtype=np.float64,
                )

            new_weights = self._solve_hedge_weights(
                target_curve_sensitivities=target_sensitivities,
                hedge_curve_sensitivities=hedge_sensitivities,
                target_vega_sensitivities=target_vegas,
                hedge_vega_sensitivities=hedge_vegas,
                curve_bump_size=self.curve_bump_size,
                vega_bump_sizes=vega_bump_sizes,
                ridge_penalty=self.ridge_penalty,
            )

            if time_index == 0:
                rebalancing_cashflow[time_index] = -provisional_target_value + new_weights @ provisional_hedge_values
                cash = rebalancing_cashflow[time_index]
                pre_rebalance_portfolio = 0.0
                incremental_pnl[time_index] = 0.0
                cumulative_pnl[time_index] = 0.0
                cash_carry_pnl[time_index] = 0.0
                target_cashflow_pnl[time_index] = 0.0
                hedge_cashflow_pnl[time_index] = 0.0
                target_revaluation_pnl[time_index] = 0.0
                hedge_revaluation_pnl[time_index] = 0.0
            else:
                rebalancing_cashflow[time_index] = float((new_weights - previous_weights) @ provisional_hedge_values)
                cash += rebalancing_cashflow[time_index]

            current_portfolio_value = provisional_target_value - new_weights @ provisional_hedge_values + cash
            if time_index > 0 and not np.isclose(
                current_portfolio_value,
                pre_rebalance_portfolio,
                atol=self.pnl_identity_tolerance,
                rtol=1.0e-9,
            ):
                raise RuntimeError("Rebalancing changed portfolio value unexpectedly.")

            target_values[time_index] = provisional_target_value
            hedge_values[time_index] = provisional_hedge_values
            hedge_weights[time_index] = new_weights
            cash_account[time_index] = cash
            portfolio_value[time_index] = current_portfolio_value
            target_key_rate_sensitivities[time_index] = target_sensitivities
            hedge_key_rate_sensitivities[time_index] = hedge_sensitivities
            residual_key_rate_sensitivities[time_index] = target_sensitivities - hedge_sensitivities.T @ new_weights
            target_vega_sensitivities[time_index] = target_vegas
            hedge_vega_sensitivities[time_index] = hedge_vegas
            residual_vega_sensitivities[time_index] = target_vegas - hedge_vegas.T @ new_weights

            previous_weights = new_weights
            previous_target_value = provisional_target_value
            previous_hedge_values = provisional_hedge_values
            previous_portfolio_value = current_portfolio_value
            previous_curve = current_curve

        return HedgingResult(
            time=times.copy(),
            key_rate_tenors=tenors.copy(),
            vega_labels=vega_labels,
            vega_bump_sizes=vega_bump_sizes.copy(),
            calibrated_parameter_names=calibrated_parameter_names,
            calibrated_parameters=calibrated_parameters,
            calibration_rmse=calibration_rmse,
            calibration_max_abs_error=calibration_max_abs_error,
            target_values=target_values,
            hedge_values=hedge_values,
            hedge_weights=hedge_weights,
            cash_account=cash_account,
            portfolio_value=portfolio_value,
            incremental_pnl=incremental_pnl,
            cumulative_pnl=cumulative_pnl,
            cash_carry_pnl=cash_carry_pnl,
            target_cashflow_pnl=target_cashflow_pnl,
            hedge_cashflow_pnl=hedge_cashflow_pnl,
            target_revaluation_pnl=target_revaluation_pnl,
            hedge_revaluation_pnl=hedge_revaluation_pnl,
            rebalancing_cashflow=rebalancing_cashflow,
            target_key_rate_sensitivities=target_key_rate_sensitivities,
            hedge_key_rate_sensitivities=hedge_key_rate_sensitivities,
            residual_key_rate_sensitivities=residual_key_rate_sensitivities,
            target_vega_sensitivities=target_vega_sensitivities,
            hedge_vega_sensitivities=hedge_vega_sensitivities,
            residual_vega_sensitivities=residual_vega_sensitivities,
        )

    def _mark_to_market(
        self,
        instrument: InterestRateInstrument,
        curve: CurveSnapshot,
        zero_rates: FloatArray,
        curve_tenors: FloatArray,
        valuation_time: float,
        is_active: bool,
        model_adapter: InterestRateModelAdapter | None,
    ) -> float:
        if not is_active:
            return 0.0
        local_time_grid = self._required_time_grid([instrument], valuation_time)
        model = None
        if local_time_grid is not None:
            model = self._build_model_from_inputs(
                model_adapter,
                valuation_time,
                local_time_grid,
                curve_tenors,
                zero_rates_to_discount_factors(zero_rates, curve_tenors),
            )
        return self._instrument_value(instrument, curve, valuation_time, model, True)

    def _required_time_grid(
        self,
        instruments: Sequence[InterestRateInstrument],
        valuation_time: float,
    ) -> FloatArray | None:
        required_times = [np.asarray([0.0], dtype=np.float64)]
        for instrument in instruments:
            instrument_times = instrument.required_model_times(valuation_time)
            if instrument_times.size > 0:
                required_times.append(instrument_times)
        if len(required_times) == 1:
            return None
        return np.unique(np.concatenate(required_times))

    def _build_model_from_inputs(
        self,
        model_adapter: InterestRateModelAdapter | None,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_tenors: FloatArray,
        discount_factors: FloatArray,
    ) -> InterestRateModel:
        if model_adapter is not None:
            return model_adapter.build(
                valuation_time,
                local_time_grid,
                curve_tenors,
                discount_factors,
                curve_tenors,
            )
        if self.model_factory is None:
            raise ValueError("A model_factory or model_adapter is required to hedge optional instruments.")
        return self.model_factory(
            valuation_time,
            local_time_grid,
            curve_tenors,
            discount_factors,
            curve_tenors,
        )

    def _build_bumped_model(
        self,
        model_adapter: InterestRateModelAdapter | None,
        parameter_name: str,
        parameter_shift: float,
        valuation_time: float,
        local_time_grid: FloatArray,
        curve_tenors: FloatArray,
        discount_factors: FloatArray,
    ) -> InterestRateModel:
        if model_adapter is None:
            raise ValueError("A model_adapter is required to compute vega sensitivities.")
        return model_adapter.build_with_parameter_shift(
            parameter_name,
            parameter_shift,
            valuation_time,
            local_time_grid,
            curve_tenors,
            discount_factors,
            curve_tenors,
        )

    def _instrument_value(
        self,
        instrument: InterestRateInstrument,
        curve: CurveSnapshot,
        valuation_time: float,
        model: InterestRateModel | None,
        is_active: bool,
    ) -> float:
        if not is_active:
            return 0.0
        if isinstance(instrument, Swap):
            return instrument.present_value_from_curve(curve, valuation_time)
        if isinstance(instrument, Swaption):
            if valuation_time > instrument.expiry + 1.0e-12:
                return 0.0
            if np.isclose(valuation_time, instrument.expiry, atol=1.0e-12, rtol=0.0):
                return float(
                    instrument.direction
                    * instrument.exercise_value_from_zero_rates(curve.zero_rates, curve.tenors)
                )
        if isinstance(instrument, BermudanSwaption) and valuation_time > instrument.exercise_times[-1] + 1.0e-12:
            return 0.0
        if model is None:
            if isinstance(instrument, BermudanSwaption):
                exercise_today = np.any(np.isclose(instrument.exercise_times, valuation_time, atol=1.0e-12, rtol=0.0))
                if exercise_today:
                    return float(
                        instrument.direction
                        * instrument.exercise_value_from_zero_rates(curve.zero_rates, curve.tenors, valuation_time)
                    )
            raise ValueError("A model is required to value optional instruments before maturity.")
        return self.pricing_engine.price(instrument, model, valuation_time).price

    def _option_settlement(
        self,
        instrument: InterestRateInstrument,
        curve: CurveSnapshot,
        valuation_time: float,
        current_value: float,
        is_active: bool,
    ) -> tuple[float, bool]:
        if not is_active or isinstance(instrument, Swap):
            return 0.0, is_active

        if isinstance(instrument, Swaption):
            if valuation_time < instrument.expiry - 1.0e-12:
                return 0.0, True
            if np.isclose(valuation_time, instrument.expiry, atol=1.0e-12, rtol=0.0):
                intrinsic = float(
                    instrument.exercise_value_from_zero_rates(curve.zero_rates, curve.tenors)
                )
                return instrument.direction * intrinsic, False
            return 0.0, False

        if valuation_time > instrument.exercise_times[-1] + 1.0e-12:
            return 0.0, False
        if not np.any(np.isclose(instrument.exercise_times, valuation_time, atol=1.0e-12, rtol=0.0)):
            return 0.0, True

        intrinsic = float(
            instrument.exercise_value_from_zero_rates(curve.zero_rates, curve.tenors, valuation_time)
        )
        holder_value = instrument.direction * current_value
        should_exercise = intrinsic > 0.0 and holder_value <= intrinsic + self.exercise_tolerance
        if should_exercise:
            return instrument.direction * intrinsic, False
        return 0.0, True

    def _instrument_curve_sensitivities(
        self,
        instrument: InterestRateInstrument,
        zero_rates: FloatArray,
        curve_tenors: FloatArray,
        valuation_time: float,
        is_active: bool,
        model_adapter: InterestRateModelAdapter | None,
    ) -> FloatArray:
        if not is_active:
            return np.zeros(curve_tenors.size, dtype=np.float64)

        bump_matrix = np.eye(curve_tenors.size, dtype=np.float64) * self.curve_bump_size
        if isinstance(instrument, Swap):
            value_up = instrument.present_value_from_zero_rates(
                zero_rates[None, :] + bump_matrix,
                curve_tenors,
                valuation_time,
            )
            value_down = instrument.present_value_from_zero_rates(
                zero_rates[None, :] - bump_matrix,
                curve_tenors,
                valuation_time,
            )
            return (value_up - value_down) / (2.0 * self.curve_bump_size)

        local_time_grid = self._required_time_grid([instrument], valuation_time)
        if local_time_grid is None:
            return np.zeros(curve_tenors.size, dtype=np.float64)

        sensitivities = np.empty(curve_tenors.size, dtype=np.float64)
        for tenor_index in range(curve_tenors.size):
            bumped_up = zero_rates.copy()
            bumped_down = zero_rates.copy()
            bumped_up[tenor_index] += self.curve_bump_size
            bumped_down[tenor_index] -= self.curve_bump_size

            curve_up = CurveSnapshot.from_zero_rates(curve_tenors, bumped_up)
            curve_down = CurveSnapshot.from_zero_rates(curve_tenors, bumped_down)
            discount_factors_up = zero_rates_to_discount_factors(bumped_up, curve_tenors)
            discount_factors_down = zero_rates_to_discount_factors(bumped_down, curve_tenors)
            model_up = self._build_model_from_inputs(
                model_adapter,
                valuation_time,
                local_time_grid,
                curve_tenors,
                discount_factors_up,
            )
            model_down = self._build_model_from_inputs(
                model_adapter,
                valuation_time,
                local_time_grid,
                curve_tenors,
                discount_factors_down,
            )
            value_up = self._instrument_value(instrument, curve_up, valuation_time, model_up, True)
            value_down = self._instrument_value(instrument, curve_down, valuation_time, model_down, True)
            sensitivities[tenor_index] = (value_up - value_down) / (2.0 * self.curve_bump_size)

        return sensitivities

    def _instrument_vega_sensitivities(
        self,
        instrument: InterestRateInstrument,
        zero_rates: FloatArray,
        curve_tenors: FloatArray,
        valuation_time: float,
        is_active: bool,
        vega_parameters: tuple,
        model_adapter: InterestRateModelAdapter | None,
    ) -> FloatArray:
        if not is_active or len(vega_parameters) == 0:
            return np.zeros(len(vega_parameters), dtype=np.float64)
        if isinstance(instrument, Swap):
            return np.zeros(len(vega_parameters), dtype=np.float64)

        local_time_grid = self._required_time_grid([instrument], valuation_time)
        if local_time_grid is None:
            return np.zeros(len(vega_parameters), dtype=np.float64)

        curve = CurveSnapshot.from_zero_rates(curve_tenors, zero_rates)
        discount_factors = zero_rates_to_discount_factors(zero_rates, curve_tenors)
        vegas = np.empty(len(vega_parameters), dtype=np.float64)
        for parameter_index, parameter in enumerate(vega_parameters):
            model_up = self._build_bumped_model(
                model_adapter,
                parameter.name,
                parameter.bump_size,
                valuation_time,
                local_time_grid,
                curve_tenors,
                discount_factors,
            )
            model_down = self._build_bumped_model(
                model_adapter,
                parameter.name,
                -parameter.bump_size,
                valuation_time,
                local_time_grid,
                curve_tenors,
                discount_factors,
            )
            value_up = self._instrument_value(instrument, curve, valuation_time, model_up, True)
            value_down = self._instrument_value(instrument, curve, valuation_time, model_down, True)
            vegas[parameter_index] = (value_up - value_down) / (2.0 * parameter.bump_size)
        return vegas

    @staticmethod
    def _solve_hedge_weights(
        target_curve_sensitivities: FloatArray,
        hedge_curve_sensitivities: FloatArray,
        target_vega_sensitivities: FloatArray,
        hedge_vega_sensitivities: FloatArray,
        curve_bump_size: float,
        vega_bump_sizes: FloatArray,
        ridge_penalty: float,
    ) -> FloatArray:
        if hedge_curve_sensitivities.size == 0 and hedge_vega_sensitivities.size == 0:
            return np.zeros(0, dtype=np.float64)

        risk_blocks: list[FloatArray] = [target_curve_sensitivities * curve_bump_size]
        hedge_blocks: list[FloatArray] = [hedge_curve_sensitivities * curve_bump_size]
        if target_vega_sensitivities.size > 0:
            risk_blocks.append(target_vega_sensitivities * vega_bump_sizes)
            hedge_blocks.append(hedge_vega_sensitivities * vega_bump_sizes[None, :])

        target_risk_vector = np.concatenate(risk_blocks)
        hedge_risk_matrix = np.concatenate(hedge_blocks, axis=1)
        sensitivity_matrix = hedge_risk_matrix.T

        if ridge_penalty == 0.0:
            weights, *_ = np.linalg.lstsq(sensitivity_matrix, target_risk_vector, rcond=None)
            return weights

        gram_matrix = sensitivity_matrix.T @ sensitivity_matrix
        regularized = gram_matrix + ridge_penalty * np.eye(gram_matrix.shape[0], dtype=np.float64)
        right_hand_side = sensitivity_matrix.T @ target_risk_vector
        return np.linalg.solve(regularized, right_hand_side)

    @staticmethod
    def _realized_swap_cashflow(
        instrument: InterestRateInstrument,
        previous_curve: CurveSnapshot,
        previous_time: float,
        current_time: float,
    ) -> float:
        if not isinstance(instrument, Swap):
            return 0.0

        total_cashflow = 0.0
        for accrual_start, payment_time, year_fraction in zip(
            instrument.accrual_start_times,
            instrument.payment_times,
            instrument.year_fractions,
            strict=True,
        ):
            if not (
                np.isclose(accrual_start, previous_time, atol=1.0e-12, rtol=0.0)
                and np.isclose(payment_time, current_time, atol=1.0e-12, rtol=0.0)
            ):
                continue

            fixed_cashflow = instrument.notional * instrument.fixed_rate * year_fraction * (-instrument.direction)
            if instrument.floating_rate_fixings and float(accrual_start) in instrument.floating_rate_fixings:
                floating_rate = instrument.floating_rate_fixings[float(accrual_start)]
            else:
                floating_rate = previous_curve.forward_rate(0.0, current_time - previous_time)
            floating_cashflow = instrument.notional * floating_rate * year_fraction * instrument.direction
            total_cashflow += fixed_cashflow + floating_cashflow

        return float(total_cashflow)


__all__ = ["DynamicHedgingEngine", "HedgingResult", "ModelFactory"]
