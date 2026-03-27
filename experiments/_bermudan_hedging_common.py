from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rateshedging.calibration.swaption_surface import (
    SwaptionSurfacePaths,
    SwaptionSurfaceSnapshot,
    build_lmm_atm_surface_paths,
)
from rateshedging.hedging.engine import DynamicHedgingEngine, HedgingResult
from rateshedging.hedging.model_adapters import G2PPModelAdapter, HullWhiteModelAdapter, InterestRateModelAdapter
from rateshedging.instruments.bermudan_swaption import BermudanSwaption
from rateshedging.instruments.instrument import InterestRateInstrument
from rateshedging.instruments.swap import Swap
from rateshedging.instruments.swaption import Swaption
from rateshedging.models.libor_market_model import LIBORMarketModel
from rateshedging.models.model import InterestRateModel, RatePaths
from rateshedging.pricing.curve import CurveSnapshot
from rateshedging.pricing.engine import MonteCarloPricingEngine


FloatArray = np.ndarray
FINAL_COMPONENT_COLUMNS = {
    "final_cash_carry_pnl": "Cash Carry",
    "final_target_cashflow_pnl": "Target Cashflow",
    "final_hedge_cashflow_pnl": "Hedge Cashflow",
    "final_target_revaluation_pnl": "Target Revaluation",
    "final_hedge_revaluation_pnl": "Hedge Revaluation",
}
RISK_QUANTITIES = ("final_pnl", *FINAL_COMPONENT_COLUMNS.keys())


@dataclass(frozen=True)
class ExperimentConfig:
    time_grid: FloatArray
    yield_curve_tenors: FloatArray
    curve_times: FloatArray
    discount_factors: FloatArray
    surface_expiries: FloatArray
    surface_swap_tenors: FloatArray
    n_outer_paths: int = 100
    n_inner_paths: int = 10_000
    curve_bump_size: float = 1.0e-4
    ridge_penalty: float = 1.0e6
    lmm_tenor_spacing: float = 0.5
    lmm_seed: int = 303
    lmm_factor_volatilities: FloatArray = field(
        default_factory=lambda: np.array([0.18, 0.10, 0.05], dtype=np.float64)
    )
    lmm_factor_decay_rates: FloatArray = field(
        default_factory=lambda: np.array([0.0, 0.35, 1.10], dtype=np.float64)
    )
    hw_seed: int = 101
    g2_seed: int = 202
    pricing_seed_hw: int = 17
    pricing_seed_g2: int = 29
    parallel_outer_paths: bool = True
    min_parallel_outer_paths: int = 4
    outer_workers: int | None = None


@dataclass(frozen=True)
class Scenario:
    label: str
    model_adapter: InterestRateModelAdapter
    outer_model: InterestRateModel | None = None
    outer_paths: RatePaths | None = None
    outer_surface_paths: SwaptionSurfacePaths | None = None
    pricing_model: str | None = None
    real_dynamics: str | None = None

    def __post_init__(self) -> None:
        if self.outer_model is None and self.outer_paths is None:
            raise ValueError("A scenario requires either an outer_model or pre-generated outer_paths.")


@dataclass(frozen=True)
class _WorkerResult:
    path_rows: list[dict[str, float | str | int]]
    time_rows: list[dict[str, float | str | int]]


def default_config() -> ExperimentConfig:
    curve_times = np.linspace(0.0, 20.0, 401, dtype=np.float64)
    base_zero_rates = 0.015 + 0.006 * (1.0 - np.exp(-0.45 * curve_times)) + 0.0004 * curve_times
    discount_factors = np.exp(-base_zero_rates * curve_times)
    return ExperimentConfig(
        time_grid=np.arange(0.0, 5.0 + 0.5, 0.5, dtype=np.float64),
        yield_curve_tenors=np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0], dtype=np.float64),
        curve_times=curve_times,
        discount_factors=discount_factors,
        surface_expiries=np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float64),
        surface_swap_tenors=np.array([2.0, 5.0, 10.0, 15.0], dtype=np.float64),
    )


def _payment_grid(start_time: float, maturity_time: float, frequency: float = 0.5) -> FloatArray:
    return np.arange(start_time + frequency, maturity_time + 1.0e-12, frequency, dtype=np.float64)


def _base_zero_rates(config: ExperimentConfig) -> FloatArray:
    interpolated_discounts = np.interp(
        config.yield_curve_tenors,
        config.curve_times[1:],
        config.discount_factors[1:],
    )
    return -np.log(interpolated_discounts) / config.yield_curve_tenors


def build_target_and_hedges(
    config: ExperimentConfig,
) -> tuple[BermudanSwaption, list[Swap], list[Swaption]]:
    base_zero_rates = _base_zero_rates(config)

    target_template = Swap(
        start_time=2.0,
        payment_times=_payment_grid(2.0, 15.0),
        fixed_rate=0.0,
        notional=1_000_000.0,
        pay_fixed=True,
    )
    target_fixed_rate = float(
        target_template.fair_rate_from_zero_rates(base_zero_rates, config.yield_curve_tenors, valuation_time=0.0)
    )
    bermudan = BermudanSwaption(
        underlying_swap=Swap(
            start_time=2.0,
            payment_times=_payment_grid(2.0, 15.0),
            fixed_rate=target_fixed_rate + 0.0025,
            notional=1_000_000.0,
            pay_fixed=True,
        ),
        exercise_times=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
    )

    delta_swap_maturities = (2.0, 4.0, 6.0, 8.0, 12.0, 15.0)
    delta_hedges: list[Swap] = []
    for maturity_time in delta_swap_maturities:
        template = Swap(
            start_time=0.0,
            payment_times=_payment_grid(0.0, maturity_time),
            fixed_rate=0.0,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        fair_rate = float(
            template.fair_rate_from_zero_rates(base_zero_rates, config.yield_curve_tenors, valuation_time=0.0)
        )
        delta_hedges.append(
            Swap(
                start_time=0.0,
                payment_times=_payment_grid(0.0, maturity_time),
                fixed_rate=fair_rate,
                notional=1_000_000.0,
                pay_fixed=True,
            )
        )

    vega_specifications = (
        (1.0, 4.0),
        (2.0, 5.0),
        (3.0, 7.0),
        (5.0, 10.0),
    )
    vega_hedges: list[Swaption] = []
    for expiry, swap_tenor in vega_specifications:
        template_swap = Swap(
            start_time=expiry,
            payment_times=_payment_grid(expiry, expiry + swap_tenor),
            fixed_rate=0.0,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        fair_rate = float(
            template_swap.fair_rate_from_zero_rates(base_zero_rates, config.yield_curve_tenors, valuation_time=0.0)
        )
        vega_hedges.append(
            Swaption(
                underlying_swap=Swap(
                    start_time=expiry,
                    payment_times=_payment_grid(expiry, expiry + swap_tenor),
                    fixed_rate=fair_rate,
                    notional=1_000_000.0,
                    pay_fixed=True,
                ),
                expiry=expiry,
                is_long=True,
            )
        )

    return bermudan, delta_hedges, vega_hedges


def _initial_curve_snapshot(config: ExperimentConfig) -> CurveSnapshot:
    return CurveSnapshot.from_zero_rates(config.yield_curve_tenors, _base_zero_rates(config))


def _surface_template(config: ExperimentConfig) -> SwaptionSurfaceSnapshot:
    return SwaptionSurfaceSnapshot(
        expiries=config.surface_expiries,
        swap_tenors=config.surface_swap_tenors,
        normal_volatilities=np.ones(
            (config.surface_expiries.size, config.surface_swap_tenors.size),
            dtype=np.float64,
        ),
    )


def _lmm_terminal_horizon(config: ExperimentConfig) -> float:
    return float(
        max(
            config.time_grid[-1] + config.yield_curve_tenors[-1],
            config.time_grid[-1] + config.surface_expiries[-1] + config.surface_swap_tenors[-1],
        )
    )


def _build_lmm_loading_matrix(config: ExperimentConfig) -> FloatArray:
    terminal_horizon = _lmm_terminal_horizon(config)
    tenor_dates = np.arange(
        config.lmm_tenor_spacing,
        terminal_horizon + 1.0e-12,
        config.lmm_tenor_spacing,
        dtype=np.float64,
    )
    maturity_profile = tenor_dates / terminal_horizon
    level_factor = 0.18 * np.exp(-0.08 * tenor_dates)
    slope_factor = 0.30 * (maturity_profile - 0.42) * np.exp(-0.03 * tenor_dates)
    return np.column_stack([level_factor, slope_factor])


def _hull_white_initial_guess_adapter(config: ExperimentConfig) -> HullWhiteModelAdapter:
    return HullWhiteModelAdapter(
        mean_reversion=0.08,
        volatility=0.0085,
        seed=config.pricing_seed_hw,
        volatility_bump=5.0e-4,
    )


def _g2pp_initial_guess_adapter(config: ExperimentConfig) -> G2PPModelAdapter:
    return G2PPModelAdapter(
        mean_reversion_x=0.15,
        mean_reversion_y=0.03,
        volatility_x=0.0085,
        volatility_y=0.0050,
        correlation=-0.70,
        seed=config.pricing_seed_g2,
        volatility_x_bump=5.0e-4,
        volatility_y_bump=5.0e-4,
    )


def _calibrated_hull_white_adapter(
    config: ExperimentConfig,
    market_surface: SwaptionSurfaceSnapshot,
) -> HullWhiteModelAdapter:
    calibration = _hull_white_initial_guess_adapter(config).calibrate(
        0.0,
        _initial_curve_snapshot(config),
        market_surface,
    )
    return calibration.adapter


def _calibrated_g2pp_adapter(
    config: ExperimentConfig,
    market_surface: SwaptionSurfaceSnapshot,
) -> G2PPModelAdapter:
    calibration = _g2pp_initial_guess_adapter(config).calibrate(
        0.0,
        _initial_curve_snapshot(config),
        market_surface,
    )
    return calibration.adapter


def _surface_paths_from_market(
    config: ExperimentConfig,
    outer_model: LIBORMarketModel,
    outer_paths: RatePaths,
) -> SwaptionSurfacePaths:
    return build_lmm_atm_surface_paths(
        model=outer_model,
        outer_paths=outer_paths,
        expiries=config.surface_expiries,
        swap_tenors=config.surface_swap_tenors,
    )


def build_lmm_market_scenarios(config: ExperimentConfig) -> tuple[Scenario, ...]:
    outer_model = LIBORMarketModel(
        tenor_spacing=config.lmm_tenor_spacing,
        factor_loading_matrix=_build_lmm_loading_matrix(config),
        time_grid=config.time_grid,
        curve_times=config.curve_times,
        discount_factors=config.discount_factors,
        yield_curve_tenors=config.yield_curve_tenors,
        terminal_horizon=_lmm_terminal_horizon(config),
        seed=config.lmm_seed,
    )
    shared_outer_paths = outer_model.generate_paths(config.n_outer_paths)
    shared_surface_paths = _surface_paths_from_market(config, outer_model, shared_outer_paths)
    market_surface = shared_surface_paths.path(0).snapshot(0)
    hull_white_market_adapter = _calibrated_hull_white_adapter(config, market_surface=market_surface)
    g2pp_market_adapter = _calibrated_g2pp_adapter(config, market_surface=market_surface)

    return (
        Scenario(
            label="Hull-White",
            model_adapter=hull_white_market_adapter,
            outer_paths=shared_outer_paths,
            outer_surface_paths=shared_surface_paths,
            pricing_model="Hull-White",
            real_dynamics="LMM",
        ),
        Scenario(
            label="G2++",
            model_adapter=g2pp_market_adapter,
            outer_paths=shared_outer_paths,
            outer_surface_paths=shared_surface_paths,
            pricing_model="G2++",
            real_dynamics="LMM",
        ),
    )


def _loss_var(values: FloatArray, confidence: float) -> float:
    return float(-np.quantile(values, 1.0 - confidence))


def _loss_cvar(values: FloatArray, confidence: float) -> float:
    cutoff = np.quantile(values, 1.0 - confidence)
    tail = values[values <= cutoff + 1.0e-12]
    return float(-np.mean(tail))


def summarize_risk_metrics(path_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    group_columns = [column for column in ("strategy", "model", "pricing_model", "real_dynamics") if column in path_summary.columns]
    for group_key, group_rows in path_summary.groupby(group_columns, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metadata = {column: value for column, value in zip(group_columns, group_key, strict=True)}
        for quantity in RISK_QUANTITIES:
            values = group_rows[quantity].to_numpy(dtype=np.float64)
            rows.append(
                {
                    **metadata,
                    "quantity": quantity,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "var_95_loss": _loss_var(values, 0.95),
                    "cvar_95_loss": _loss_cvar(values, 0.95),
                    "var_99_loss": _loss_var(values, 0.99),
                    "cvar_99_loss": _loss_cvar(values, 0.99),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _cumulative_component_profiles(result: HedgingResult) -> dict[str, FloatArray]:
    return {
        "Cash Carry": np.cumsum(result.cash_carry_pnl),
        "Target Cashflow": np.cumsum(result.target_cashflow_pnl),
        "Hedge Cashflow": np.cumsum(result.hedge_cashflow_pnl),
        "Target Revaluation": np.cumsum(result.target_revaluation_pnl),
        "Hedge Revaluation": np.cumsum(result.hedge_revaluation_pnl),
    }


def _path_summary_row(
    scenario: Scenario,
    strategy_label: str,
    path_index: int,
    result: HedgingResult,
    curve_bump_size: float,
) -> dict[str, float | str | int]:
    residual_curve_impacts = np.sum(
        np.abs(result.residual_key_rate_sensitivities) * curve_bump_size,
        axis=1,
    )
    if result.residual_vega_sensitivities.shape[1] == 0:
        residual_vega_impacts = np.zeros_like(result.time)
    else:
        residual_vega_impacts = np.sum(
            np.abs(result.residual_vega_sensitivities) * result.vega_bump_sizes[None, :],
            axis=1,
        )

    row: dict[str, float | str | int] = {
        "model": scenario.label,
        "pricing_model": scenario.pricing_model or scenario.label,
        "real_dynamics": scenario.real_dynamics or scenario.label,
        "strategy": strategy_label,
        "path": path_index,
        "final_pnl": float(result.portfolio_value[-1]),
        "final_cash_carry_pnl": float(np.sum(result.cash_carry_pnl)),
        "final_target_cashflow_pnl": float(np.sum(result.target_cashflow_pnl)),
        "final_hedge_cashflow_pnl": float(np.sum(result.hedge_cashflow_pnl)),
        "final_target_revaluation_pnl": float(np.sum(result.target_revaluation_pnl)),
        "final_hedge_revaluation_pnl": float(np.sum(result.hedge_revaluation_pnl)),
        "max_abs_portfolio": float(np.max(np.abs(result.portfolio_value))),
        "mean_abs_residual_curve_impact": float(np.mean(residual_curve_impacts)),
        "mean_abs_residual_vega_impact": float(np.mean(residual_vega_impacts)),
        "mean_calibration_rmse": float(np.mean(result.calibration_rmse)),
        "max_calibration_rmse": float(np.max(result.calibration_rmse)),
        "mean_calibration_max_abs_error": float(np.mean(result.calibration_max_abs_error)),
    }
    return row


def _time_profile_rows(
    scenario: Scenario,
    strategy_label: str,
    path_index: int,
    result: HedgingResult,
) -> list[dict[str, float | str | int]]:
    cumulative_profiles = _cumulative_component_profiles(result)
    rows: list[dict[str, float | str | int]] = []
    for time_index, time_value in enumerate(result.time):
        row: dict[str, float | str | int] = {
            "model": scenario.label,
            "pricing_model": scenario.pricing_model or scenario.label,
            "real_dynamics": scenario.real_dynamics or scenario.label,
            "strategy": strategy_label,
            "path": path_index,
            "time": float(time_value),
            "portfolio_value": float(result.portfolio_value[time_index]),
            "calibration_rmse": float(result.calibration_rmse[time_index]),
            "calibration_max_abs_error": float(result.calibration_max_abs_error[time_index]),
        }
        for label, values in cumulative_profiles.items():
            row[label] = float(values[time_index])
        for parameter_index, parameter_name in enumerate(result.calibrated_parameter_names):
            row[f"calibrated_{parameter_name}"] = float(result.calibrated_parameters[time_index, parameter_index])
        rows.append(row)
    return rows


def _resolve_outer_worker_count(config: ExperimentConfig) -> int:
    available_cores = os.cpu_count() or 1
    requested_workers = config.outer_workers
    if requested_workers is None:
        requested_workers = max(1, available_cores - 1)
    return max(1, min(int(requested_workers), config.n_outer_paths))


def _should_parallelize_outer_loop(config: ExperimentConfig) -> bool:
    return (
        config.parallel_outer_paths
        and config.n_outer_paths >= config.min_parallel_outer_paths
        and _resolve_outer_worker_count(config) > 1
    )


def _run_experiment_chunk(
    *,
    config: ExperimentConfig,
    scenario: Scenario,
    strategy_label: str,
    include_vega: bool,
    path_indices: tuple[int, ...],
) -> _WorkerResult:
    target_instrument, delta_hedges, vega_hedges = build_target_and_hedges(config)
    hedging_instruments: list[InterestRateInstrument] = list(delta_hedges)
    if include_vega:
        hedging_instruments.extend(vega_hedges)

    engine = DynamicHedgingEngine(
        pricing_engine=MonteCarloPricingEngine(n_paths=config.n_inner_paths),
        model_adapter=scenario.model_adapter,
        curve_bump_size=config.curve_bump_size,
        include_vega=include_vega,
        ridge_penalty=config.ridge_penalty,
    )
    outer_paths = scenario.outer_paths
    if outer_paths is None:
        if scenario.outer_model is None:
            raise ValueError(f"Scenario {scenario.label!r} has neither outer_paths nor outer_model.")
        outer_paths = scenario.outer_model.generate_paths(config.n_outer_paths)

    path_rows: list[dict[str, float | str | int]] = []
    time_rows: list[dict[str, float | str | int]] = []
    for path_index in path_indices:
        result = engine.run(
            yield_curve_trajectory=outer_paths.yield_curve_paths[path_index],
            time_grid=config.time_grid,
            yield_curve_tenors=config.yield_curve_tenors,
            target_instrument=target_instrument,
            hedging_instruments=hedging_instruments,
            swaption_surface_trajectory=(
                scenario.outer_surface_paths.path(path_index)
                if scenario.outer_surface_paths is not None
                else None
            ),
        )

        explained_pnl = (
            np.sum(result.cash_carry_pnl)
            + np.sum(result.target_cashflow_pnl)
            + np.sum(result.hedge_cashflow_pnl)
            + np.sum(result.target_revaluation_pnl)
            + np.sum(result.hedge_revaluation_pnl)
        )
        if not np.isclose(explained_pnl, result.portfolio_value[-1], atol=1.0e-4, rtol=1.0e-8):
            raise RuntimeError("Final PnL does not match the cumulative PnL breakdown.")

        path_rows.append(
            _path_summary_row(
                scenario,
                strategy_label,
                path_index,
                result,
                config.curve_bump_size,
            )
        )
        time_rows.extend(_time_profile_rows(scenario, strategy_label, path_index, result))

    return _WorkerResult(path_rows=path_rows, time_rows=time_rows)


def _run_experiment_chunk_from_payload(payload: dict) -> _WorkerResult:
    return _run_experiment_chunk(**payload)


def run_experiment(
    *,
    config: ExperimentConfig,
    strategy_label: str,
    include_vega: bool,
    scenarios: tuple[Scenario, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path_rows: list[dict[str, float | str | int]] = []
    time_rows: list[dict[str, float | str | int]] = []

    for scenario in scenarios or build_lmm_market_scenarios(config):
        outer_paths = scenario.outer_paths
        if outer_paths is None:
            if scenario.outer_model is None:
                raise ValueError(f"Scenario {scenario.label!r} has neither outer_paths nor outer_model.")
            outer_paths = scenario.outer_model.generate_paths(config.n_outer_paths)
        if outer_paths.n_paths != config.n_outer_paths:
            raise ValueError("Outer-path count must match config.n_outer_paths.")
        if not np.allclose(outer_paths.time, config.time_grid, atol=1.0e-12, rtol=0.0):
            raise ValueError("Outer paths must use the experiment time grid.")
        if not np.allclose(outer_paths.yield_curve_tenors, config.yield_curve_tenors, atol=1.0e-12, rtol=0.0):
            raise ValueError("Outer paths must use the experiment yield-curve tenors.")
        if scenario.outer_surface_paths is not None:
            if scenario.outer_surface_paths.n_paths != config.n_outer_paths:
                raise ValueError("Swaption surface path count must match config.n_outer_paths.")
            if not np.allclose(scenario.outer_surface_paths.time_grid, config.time_grid, atol=1.0e-12, rtol=0.0):
                raise ValueError("Swaption surface paths must use the experiment time grid.")

        if _should_parallelize_outer_loop(config):
            worker_count = _resolve_outer_worker_count(config)
            index_chunks = [
                tuple(int(path_index) for path_index in chunk.tolist())
                for chunk in np.array_split(np.arange(config.n_outer_paths, dtype=np.int64), worker_count)
                if chunk.size > 0
            ]
            worker_payloads = [
                {
                    "config": config,
                    "scenario": scenario,
                    "strategy_label": strategy_label,
                    "include_vega": include_vega,
                    "path_indices": chunk,
                }
                for chunk in index_chunks
            ]
            try:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    worker_results = list(
                        executor.map(
                            _run_experiment_chunk_from_payload,
                            worker_payloads,
                        )
                    )
                for worker_result in worker_results:
                    path_rows.extend(worker_result.path_rows)
                    time_rows.extend(worker_result.time_rows)
            except (OSError, PermissionError):
                for payload in worker_payloads:
                    worker_result = _run_experiment_chunk_from_payload(payload)
                    path_rows.extend(worker_result.path_rows)
                    time_rows.extend(worker_result.time_rows)
        else:
            worker_result = _run_experiment_chunk(
                config=config,
                scenario=scenario,
                strategy_label=strategy_label,
                include_vega=include_vega,
                path_indices=tuple(range(config.n_outer_paths)),
            )
            path_rows.extend(worker_result.path_rows)
            time_rows.extend(worker_result.time_rows)

    path_summary = pd.DataFrame(path_rows)
    time_summary = pd.DataFrame(time_rows)
    if not path_summary.empty:
        path_summary = path_summary.sort_values(["strategy", "model", "path"], kind="stable").reset_index(drop=True)
    if not time_summary.empty:
        time_summary = time_summary.sort_values(["strategy", "model", "path", "time"], kind="stable").reset_index(drop=True)
    risk_summary = summarize_risk_metrics(path_summary)
    return path_summary, time_summary, risk_summary


def make_plot(
    *,
    path_summary: pd.DataFrame,
    time_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    model_labels = sorted(path_summary["model"].unique())
    if len(model_labels) != 2:
        raise ValueError("make_plot expects exactly two model labels.")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    colors = {label: palette[index % len(palette)] for index, label in enumerate(model_labels)}

    for model_label, model_rows in path_summary.groupby("model"):
        axes[0, 0].hist(
            model_rows["final_pnl"],
            bins=12,
            alpha=0.60,
            label=model_label,
            color=colors[model_label],
        )
    axes[0, 0].set_title("Final PnL Distribution")
    axes[0, 0].set_xlabel("PnL")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    final_pnl_risks = risk_summary[risk_summary["quantity"] == "final_pnl"].copy()
    metric_names = ["std", "var_95_loss", "cvar_95_loss", "cvar_99_loss"]
    x_positions = np.arange(len(metric_names), dtype=np.float64)
    width = 0.35
    for offset, model_label in enumerate(model_labels):
        model_row = final_pnl_risks[final_pnl_risks["model"] == model_label].iloc[0]
        axes[0, 1].bar(
            x_positions + (offset - 0.5) * width,
            [model_row[metric_name] for metric_name in metric_names],
            width=width,
            label=model_label,
            color=colors[model_label],
        )
    axes[0, 1].set_xticks(x_positions, ["Std", "VaR 95%", "CVaR 95%", "CVaR 99%"])
    axes[0, 1].set_title("Final PnL Risk Metrics")
    axes[0, 1].set_ylabel("Loss / Dispersion")
    axes[0, 1].legend()

    for axis_index, model_label in enumerate(model_labels):
        axis = axes[1, axis_index]
        model_rows = time_summary[time_summary["model"] == model_label]
        averaged = model_rows.groupby("time")[list(FINAL_COMPONENT_COLUMNS.values())].mean(numeric_only=True)
        for component_label in FINAL_COMPONENT_COLUMNS.values():
            axis.plot(
                averaged.index.to_numpy(dtype=np.float64),
                averaged[component_label].to_numpy(dtype=np.float64),
                linewidth=2.0,
                label=component_label,
            )
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_title(f"{model_label}: Mean Cumulative PnL Breakdown")
        axis.set_xlabel("Time (years)")
        axis.set_ylabel("Cumulative PnL")
        axis.legend(fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

def run_strategy_suite(
    *,
    config: ExperimentConfig,
    scenarios: tuple[Scenario, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared_scenarios = scenarios or build_lmm_market_scenarios(config)
    all_path_rows: list[pd.DataFrame] = []
    all_time_rows: list[pd.DataFrame] = []
    for strategy_label, include_vega in (("Delta", False), ("Delta + Vega", True)):
        path_summary, time_summary, _ = run_experiment(
            config=config,
            strategy_label=strategy_label,
            include_vega=include_vega,
            scenarios=shared_scenarios,
        )
        all_path_rows.append(path_summary)
        all_time_rows.append(time_summary)
    combined_paths = pd.concat(all_path_rows, ignore_index=True)
    combined_time = pd.concat(all_time_rows, ignore_index=True)
    combined_risk = summarize_risk_metrics(combined_paths)
    return combined_paths, combined_time, combined_risk


__all__ = [
    "ExperimentConfig",
    "Scenario",
    "build_lmm_market_scenarios",
    "build_target_and_hedges",
    "default_config",
    "make_plot",
    "run_experiment",
    "run_strategy_suite",
    "summarize_risk_metrics",
]
