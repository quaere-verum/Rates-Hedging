from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _bermudan_hedging_common import build_lmm_market_scenarios, default_config, run_experiment, run_strategy_suite


class ExperimentScenarioTests(unittest.TestCase):
    def test_lmm_scenarios_share_outer_market_paths(self) -> None:
        config = replace(default_config(), n_outer_paths=3)
        scenarios = build_lmm_market_scenarios(config)

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0].pricing_model, "Hull-White")
        self.assertEqual(scenarios[1].pricing_model, "G2++")
        self.assertEqual(scenarios[0].real_dynamics, "LMM")
        self.assertEqual(scenarios[1].real_dynamics, "LMM")
        self.assertIsNotNone(scenarios[0].outer_paths)
        self.assertIsNotNone(scenarios[0].outer_surface_paths)
        self.assertIs(scenarios[0].outer_paths, scenarios[1].outer_paths)
        self.assertIs(scenarios[0].outer_surface_paths, scenarios[1].outer_surface_paths)
        np.testing.assert_allclose(scenarios[0].outer_paths.time, config.time_grid)
        np.testing.assert_allclose(scenarios[0].outer_paths.yield_curve_tenors, config.yield_curve_tenors)
        initial_surface = scenarios[0].outer_surface_paths.normal_volatility_paths[:, 0, :, :]
        np.testing.assert_allclose(initial_surface[0], initial_surface[1], atol=1.0e-12)
        self.assertTrue(np.all(initial_surface > 0.0))

    def test_run_experiment_accepts_shared_outer_paths(self) -> None:
        config = replace(default_config(), n_outer_paths=2, n_inner_paths=200)
        scenarios = build_lmm_market_scenarios(config)

        path_summary, time_summary, risk_summary = run_experiment(
            config=config,
            strategy_label="Delta",
            include_vega=False,
            scenarios=scenarios,
        )

        self.assertEqual(set(path_summary["pricing_model"]), {"Hull-White", "G2++"})
        self.assertEqual(set(path_summary["real_dynamics"]), {"LMM"})
        self.assertEqual(path_summary.shape[0], 4)
        self.assertEqual(set(time_summary["model"]), {scenario.label for scenario in scenarios})
        self.assertEqual(set(risk_summary["pricing_model"]), {"Hull-White", "G2++"})
        self.assertIn("mean_calibration_rmse", path_summary.columns)
        self.assertIn("calibration_rmse", time_summary.columns)
        self.assertEqual(set(risk_summary["strategy"]), {"Delta"})

    def test_run_strategy_suite_reports_both_strategies(self) -> None:
        config = replace(default_config(), n_outer_paths=2, n_inner_paths=150)
        scenarios = build_lmm_market_scenarios(config)

        path_summary, time_summary, risk_summary = run_strategy_suite(config=config, scenarios=scenarios)

        self.assertEqual(set(path_summary["strategy"]), {"Delta", "Delta + Vega"})
        self.assertEqual(set(time_summary["strategy"]), {"Delta", "Delta + Vega"})
        self.assertEqual(set(risk_summary["strategy"]), {"Delta", "Delta + Vega"})


if __name__ == "__main__":
    unittest.main()
