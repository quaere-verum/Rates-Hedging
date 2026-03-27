from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rateshedging.hedging.engine import DynamicHedgingEngine, HedgingResult
from rateshedging.hedging.model_adapters import G2PPModelAdapter, HullWhiteModelAdapter, InterestRateModelAdapter
from rateshedging.instruments.bermudan_swaption import BermudanSwaption
from rateshedging.instruments.instrument import InterestRateInstrument
from rateshedging.instruments.swap import Swap
from rateshedging.instruments.swaption import Swaption
from rateshedging.models.g2pp import G2PPModel
from rateshedging.models.hull_white import HullWhiteModel
from rateshedging.models.model import InterestRateModel, RatePaths
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
    n_outer_paths: int = 100
    n_inner_paths: int = 10_000
    curve_bump_size: float = 1.0e-4
    ridge_penalty: float = 1.0e6
    hw_seed: int = 101
    g2_seed: int = 202
    pricing_seed_hw: int = 17
    pricing_seed_g2: int = 29


@dataclass(frozen=True)
class Scenario:
    label: str
    model_adapter: InterestRateModelAdapter
    outer_model: InterestRateModel | None = None
    outer_paths: RatePaths | None = None
    pricing_model: str | None = None
    real_dynamics: str | None = None

    def __post_init__(self) -> None:
        if self.outer_model is None and self.outer_paths is None:
            raise ValueError("A scenario requires either an outer_model or pre-generated outer_paths.")


def default_config() -> ExperimentConfig:
    curve_times = np.linspace(0.0, 10.0, 201, dtype=np.float64)
    base_zero_rates = 0.015 + 0.006 * (1.0 - np.exp(-0.45 * curve_times)) + 0.0004 * curve_times
    discount_factors = np.exp(-base_zero_rates * curve_times)
    return ExperimentConfig(
        time_grid=np.arange(0.0, 4.0 + 0.5, 0.5, dtype=np.float64),
        yield_curve_tenors=np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64),
        curve_times=curve_times,
        discount_factors=discount_factors,
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
        start_time=1.0,
        payment_times=_payment_grid(1.0, 7.0),
        fixed_rate=0.0,
        notional=1_000_000.0,
        pay_fixed=True,
    )
    target_fixed_rate = float(
        target_template.fair_rate_from_zero_rates(base_zero_rates, config.yield_curve_tenors, valuation_time=0.0)
    )
    bermudan = BermudanSwaption(
        underlying_swap=Swap(
            start_time=1.0,
            payment_times=_payment_grid(1.0, 7.0),
            fixed_rate=target_fixed_rate + 0.0025,
            notional=1_000_000.0,
            pay_fixed=True,
        ),
        exercise_times=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

    delta_swap_maturities = (2.0, 4.0, 6.0, 8.0)
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

    vega_expiries = (1.0, 2.0, 3.0)
    vega_hedges: list[Swaption] = []
    for expiry in vega_expiries:
        template_swap = Swap(
            start_time=expiry,
            payment_times=_payment_grid(expiry, expiry + 4.0),
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
                    payment_times=_payment_grid(expiry, expiry + 4.0),
                    fixed_rate=fair_rate,
                    notional=1_000_000.0,
                    pay_fixed=True,
                ),
                expiry=expiry,
                is_long=True,
            )
        )

    return bermudan, delta_hedges, vega_hedges


def build_scenarios(config: ExperimentConfig) -> tuple[Scenario, ...]:
    hull_white_outer = HullWhiteModel(
        mean_reversion=0.08,
        volatility=0.01,
        time_grid=config.time_grid,
        curve_times=config.curve_times,
        discount_factors=config.discount_factors,
        yield_curve_tenors=config.yield_curve_tenors,
        seed=config.hw_seed,
    )
    g2pp_outer = G2PPModel(
        mean_reversion_x=0.15,
        mean_reversion_y=0.03,
        volatility_x=0.010,
        volatility_y=0.006,
        correlation=-0.70,
        time_grid=config.time_grid,
        curve_times=config.curve_times,
        discount_factors=config.discount_factors,
        yield_curve_tenors=config.yield_curve_tenors,
        seed=config.g2_seed,
    )
    return (
        Scenario(
            label="Hull-White",
            model_adapter=HullWhiteModelAdapter(
                mean_reversion=0.08,
                volatility=0.01,
                seed=config.pricing_seed_hw,
                volatility_bump=5.0e-4,
            ),
            outer_model=hull_white_outer,
            pricing_model="Hull-White",
            real_dynamics="Hull-White",
        ),
        Scenario(
            label="G2++",
            model_adapter=G2PPModelAdapter(
                mean_reversion_x=0.15,
                mean_reversion_y=0.03,
                volatility_x=0.010,
                volatility_y=0.006,
                correlation=-0.70,
                seed=config.pricing_seed_g2,
                volatility_x_bump=5.0e-4,
                volatility_y_bump=5.0e-4,
            ),
            outer_model=g2pp_outer,
            pricing_model="G2++",
            real_dynamics="G2++",
        ),
    )


def build_g2pp_misspecification_scenarios(config: ExperimentConfig) -> tuple[Scenario, ...]:
    g2pp_outer = G2PPModel(
        mean_reversion_x=0.15,
        mean_reversion_y=0.03,
        volatility_x=0.010,
        volatility_y=0.006,
        correlation=-0.70,
        time_grid=config.time_grid,
        curve_times=config.curve_times,
        discount_factors=config.discount_factors,
        yield_curve_tenors=config.yield_curve_tenors,
        seed=config.g2_seed,
    )
    shared_outer_paths = g2pp_outer.generate_paths(config.n_outer_paths)

    return (
        Scenario(
            label="HW Pricing / G2++ Dynamics",
            model_adapter=HullWhiteModelAdapter(
                mean_reversion=0.08,
                volatility=0.01,
                seed=config.pricing_seed_hw,
                volatility_bump=5.0e-4,
            ),
            outer_paths=shared_outer_paths,
            pricing_model="Hull-White",
            real_dynamics="G2++",
        ),
        Scenario(
            label="G2++ Pricing / G2++ Dynamics",
            model_adapter=G2PPModelAdapter(
                mean_reversion_x=0.15,
                mean_reversion_y=0.03,
                volatility_x=0.010,
                volatility_y=0.006,
                correlation=-0.70,
                seed=config.pricing_seed_g2,
                volatility_x_bump=5.0e-4,
                volatility_y_bump=5.0e-4,
            ),
            outer_paths=shared_outer_paths,
            pricing_model="G2++",
            real_dynamics="G2++",
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
    metadata_columns = [column for column in ("pricing_model", "real_dynamics") if column in path_summary.columns]
    for model_label in sorted(path_summary["model"].unique()):
        model_rows = path_summary[path_summary["model"] == model_label]
        metadata = {column: str(model_rows[column].iloc[0]) for column in metadata_columns}
        for quantity in RISK_QUANTITIES:
            values = model_rows[quantity].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "model": model_label,
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
        }
        for label, values in cumulative_profiles.items():
            row[label] = float(values[time_index])
        rows.append(row)
    return rows


def run_experiment(
    *,
    config: ExperimentConfig,
    strategy_label: str,
    include_vega: bool,
    scenarios: tuple[Scenario, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_instrument, delta_hedges, vega_hedges = build_target_and_hedges(config)
    hedging_instruments: list[InterestRateInstrument] = list(delta_hedges)
    if include_vega:
        hedging_instruments.extend(vega_hedges)

    path_rows: list[dict[str, float | str | int]] = []
    time_rows: list[dict[str, float | str | int]] = []

    for scenario in scenarios or build_scenarios(config):
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
        if outer_paths.n_paths != config.n_outer_paths:
            raise ValueError("Outer-path count must match config.n_outer_paths.")
        if not np.allclose(outer_paths.time, config.time_grid, atol=1.0e-12, rtol=0.0):
            raise ValueError("Outer paths must use the experiment time grid.")
        if not np.allclose(outer_paths.yield_curve_tenors, config.yield_curve_tenors, atol=1.0e-12, rtol=0.0):
            raise ValueError("Outer paths must use the experiment yield-curve tenors.")

        for path_index in range(config.n_outer_paths):
            result = engine.run(
                yield_curve_trajectory=outer_paths.yield_curve_paths[path_index],
                time_grid=config.time_grid,
                yield_curve_tenors=config.yield_curve_tenors,
                target_instrument=target_instrument,
                hedging_instruments=hedging_instruments,
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

    path_summary = pd.DataFrame(path_rows)
    time_summary = pd.DataFrame(time_rows)
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

__all__ = [
    "ExperimentConfig",
    "Scenario",
    "build_target_and_hedges",
    "build_g2pp_misspecification_scenarios",
    "default_config",
    "make_plot",
    "run_experiment",
]
