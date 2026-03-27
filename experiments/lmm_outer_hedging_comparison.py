from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bermudan_hedging_common import default_config, run_strategy_suite


FINAL_COMPONENT_COLUMNS = [
    "Cash Carry",
    "Target Cashflow",
    "Hedge Cashflow",
    "Target Revaluation",
    "Hedge Revaluation",
]


def _make_overview_plot(path_summary, risk_summary, output_path: Path) -> None:
    strategies = ["Delta", "Delta + Vega"]
    models = ["Hull-White", "G2++"]
    colors = {"Hull-White": "#1f77b4", "G2++": "#d62728"}

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row_index, strategy in enumerate(strategies):
        axis_hist = axes[row_index, 0]
        axis_risk = axes[row_index, 1]
        strategy_paths = path_summary[path_summary["strategy"] == strategy]
        strategy_risk = risk_summary[
            (risk_summary["strategy"] == strategy) & (risk_summary["quantity"] == "final_pnl")
        ].copy()

        for model in models:
            model_rows = strategy_paths[strategy_paths["model"] == model]
            axis_hist.hist(
                model_rows["final_pnl"],
                bins=12,
                alpha=0.60,
                color=colors[model],
                label=model,
            )
        axis_hist.set_title(f"{strategy}: Final PnL Distribution")
        axis_hist.set_xlabel("PnL")
        axis_hist.set_ylabel("Frequency")
        axis_hist.legend()

        metric_names = ["std", "var_95_loss", "cvar_95_loss", "cvar_99_loss"]
        x_positions = np.arange(len(metric_names), dtype=np.float64)
        width = 0.35
        for offset, model in enumerate(models):
            row = strategy_risk[strategy_risk["model"] == model].iloc[0]
            axis_risk.bar(
                x_positions + (offset - 0.5) * width,
                [row[metric] for metric in metric_names],
                width=width,
                color=colors[model],
                label=model,
            )
        axis_risk.set_xticks(x_positions, ["Std", "VaR 95%", "CVaR 95%", "CVaR 99%"])
        axis_risk.set_title(f"{strategy}: Risk Metrics")
        axis_risk.set_ylabel("Loss / Dispersion")
        axis_risk.legend()

    figure.suptitle("LMM Outer Market: HW vs G2++ Hedging Comparison")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _make_breakdown_plot(time_summary, output_path: Path) -> None:
    strategies = ["Delta", "Delta + Vega"]
    models = ["Hull-White", "G2++"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row_index, strategy in enumerate(strategies):
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            rows = time_summary[(time_summary["strategy"] == strategy) & (time_summary["model"] == model)]
            averaged = rows.groupby("time")[FINAL_COMPONENT_COLUMNS].mean(numeric_only=True)
            for component in FINAL_COMPONENT_COLUMNS:
                axis.plot(
                    averaged.index.to_numpy(dtype=np.float64),
                    averaged[component].to_numpy(dtype=np.float64),
                    linewidth=2.0,
                    label=component,
                )
            axis.axhline(0.0, color="#333333", linewidth=0.8)
            axis.set_title(f"{strategy} / {model}")
            axis.set_xlabel("Time (years)")
            axis.set_ylabel("Mean Cumulative PnL")
            axis.legend(fontsize=8)

    figure.suptitle("LMM Outer Market: Mean Cumulative PnL Breakdown")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    config = default_config()
    path_summary, time_summary, risk_summary = run_strategy_suite(config=config)

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path_summary_path = artifact_dir / "lmm_outer_hedging_comparison_paths.csv"
    time_summary_path = artifact_dir / "lmm_outer_hedging_comparison_time.csv"
    risk_summary_path = artifact_dir / "lmm_outer_hedging_comparison_risk.csv"
    overview_plot_path = artifact_dir / "lmm_outer_hedging_comparison_overview.png"
    breakdown_plot_path = artifact_dir / "lmm_outer_hedging_comparison_breakdown.png"

    path_summary.to_csv(path_summary_path, index=False)
    time_summary.to_csv(time_summary_path, index=False)
    risk_summary.to_csv(risk_summary_path, index=False)
    _make_overview_plot(path_summary, risk_summary, overview_plot_path)
    _make_breakdown_plot(time_summary, breakdown_plot_path)

    final_pnl_summary = risk_summary[risk_summary["quantity"] == "final_pnl"].copy()
    calibration_summary = (
        path_summary.groupby(["strategy", "model"])[
            ["mean_calibration_rmse", "mean_calibration_max_abs_error"]
        ]
        .mean()
        .reset_index()
    )

    print("Saved overview plot to:", overview_plot_path.resolve())
    print("Saved breakdown plot to:", breakdown_plot_path.resolve())
    print("Saved path summary to:", path_summary_path.resolve())
    print("Saved risk summary to:", risk_summary_path.resolve())
    print()
    print(
        final_pnl_summary[
            ["strategy", "model", "mean", "std", "var_95_loss", "cvar_95_loss", "cvar_99_loss"]
        ]
        .round(2)
        .to_string(index=False)
    )
    print()
    print(calibration_summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
