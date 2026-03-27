from __future__ import annotations

from pathlib import Path

from _bermudan_hedging_common import default_config, make_plot, run_experiment


def main() -> None:
    config = default_config()
    path_summary, time_summary, risk_summary = run_experiment(
        config=config,
        strategy_label="Delta + Vega",
        include_vega=True,
    )

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path_summary_path = artifact_dir / "bermudan_delta_vega_pnl_breakdown_paths.csv"
    time_summary_path = artifact_dir / "bermudan_delta_vega_pnl_breakdown_time.csv"
    risk_summary_path = artifact_dir / "bermudan_delta_vega_pnl_breakdown_risk.csv"
    plot_path = artifact_dir / "bermudan_delta_vega_pnl_breakdown.png"

    path_summary.to_csv(path_summary_path, index=False)
    time_summary.to_csv(time_summary_path, index=False)
    risk_summary.to_csv(risk_summary_path, index=False)
    make_plot(
        path_summary=path_summary,
        time_summary=time_summary,
        risk_summary=risk_summary,
        title="Bermudan Swaption Delta + Vega Hedging PnL Breakdown",
        output_path=plot_path,
    )

    final_pnl_summary = risk_summary[risk_summary["quantity"] == "final_pnl"].copy()
    print("Saved plot to:", plot_path.resolve())
    print("Saved path summary to:", path_summary_path.resolve())
    print("Saved risk summary to:", risk_summary_path.resolve())
    print()
    print(
        final_pnl_summary[["model", "mean", "std", "var_95_loss", "cvar_95_loss", "cvar_99_loss"]]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
