from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rateshedging.calibration.swaption_surface import (
    SwaptionSurfaceSnapshot,
    SwaptionSurfaceTrajectory,
    g2pp_atm_normal_volatilities,
    hull_white_atm_normal_volatilities,
)
from rateshedging.hedging.engine import DynamicHedgingEngine
from rateshedging.hedging.model_adapters import G2PPModelAdapter, HullWhiteModelAdapter
from rateshedging.instruments.swap import Swap
from rateshedging.instruments.swaption import Swaption
from rateshedging.pricing.curve import CurveSnapshot
from rateshedging.pricing.engine import MonteCarloPricingEngine


class CalibrationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expiries = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.swap_tenors = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        self.surface_weights = np.ones((self.expiries.size, self.swap_tenors.size), dtype=np.float64)
        self.curve_tenors = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
        self.zero_rates = np.array([0.017, 0.018, 0.019, 0.021, 0.023], dtype=np.float64)

    def test_hull_white_surface_calibration_recovers_volatility(self) -> None:
        true_volatility = 0.010
        surface = SwaptionSurfaceSnapshot(
            expiries=self.expiries,
            swap_tenors=self.swap_tenors,
            normal_volatilities=hull_white_atm_normal_volatilities(
                mean_reversion=0.08,
                volatility=true_volatility,
                expiries=self.expiries,
                swap_tenors=self.swap_tenors,
            ),
            weights=self.surface_weights,
        )
        curve_snapshot = CurveSnapshot.from_zero_rates(self.curve_tenors, self.zero_rates)

        guess_adapter = HullWhiteModelAdapter(mean_reversion=0.08, volatility=0.0075)
        calibration = guess_adapter.calibrate(
            valuation_time=0.0,
            curve_snapshot=curve_snapshot,
            swaption_surface=surface,
        )

        self.assertEqual(calibration.parameter_names, ("volatility",))
        self.assertAlmostEqual(calibration.parameter_values[0], true_volatility, places=10)
        self.assertLess(calibration.rmse, 1.0e-12)

    def test_g2pp_surface_calibration_recovers_volatilities(self) -> None:
        true_volatility_x = 0.010
        true_volatility_y = 0.006
        surface = SwaptionSurfaceSnapshot(
            expiries=self.expiries,
            swap_tenors=self.swap_tenors,
            normal_volatilities=g2pp_atm_normal_volatilities(
                mean_reversion_x=0.15,
                mean_reversion_y=0.03,
                volatility_x=true_volatility_x,
                volatility_y=true_volatility_y,
                correlation=-0.70,
                expiries=self.expiries,
                swap_tenors=self.swap_tenors,
            ),
            weights=self.surface_weights,
        )
        curve_snapshot = CurveSnapshot.from_zero_rates(self.curve_tenors, self.zero_rates)

        guess_adapter = G2PPModelAdapter(
            mean_reversion_x=0.15,
            mean_reversion_y=0.03,
            volatility_x=0.0080,
            volatility_y=0.0045,
            correlation=-0.70,
        )
        calibration = guess_adapter.calibrate(
            valuation_time=0.0,
            curve_snapshot=curve_snapshot,
            swaption_surface=surface,
        )

        self.assertEqual(calibration.parameter_names, ("volatility_x", "volatility_y"))
        self.assertAlmostEqual(calibration.parameter_values[0], true_volatility_x, places=8)
        self.assertAlmostEqual(calibration.parameter_values[1], true_volatility_y, places=8)
        self.assertLess(calibration.rmse, 1.0e-10)

    def test_dynamic_hedging_uses_surface_trajectory_for_calibration(self) -> None:
        time_grid = np.array([0.0, 0.5], dtype=np.float64)
        curve_path = np.vstack([self.zero_rates, self.zero_rates + 0.0005])
        underlying_swap = Swap(
            start_time=0.5,
            payment_times=np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64),
            fixed_rate=0.021,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        target_swaption = Swaption(underlying_swap=underlying_swap, expiry=0.5)
        hedge_swaption = Swaption(underlying_swap=underlying_swap, expiry=0.5)

        surface_path = np.stack(
            [
                hull_white_atm_normal_volatilities(0.08, 0.010, self.expiries, self.swap_tenors),
                hull_white_atm_normal_volatilities(0.08, 0.011, self.expiries, self.swap_tenors),
            ],
            axis=0,
        )
        trajectory = SwaptionSurfaceTrajectory(
            time_grid=time_grid,
            expiries=self.expiries,
            swap_tenors=self.swap_tenors,
            normal_volatility_path=surface_path,
        )

        hedging_engine = DynamicHedgingEngine(
            pricing_engine=MonteCarloPricingEngine(n_paths=4_000),
            model_adapter=HullWhiteModelAdapter(
                mean_reversion=0.08,
                volatility=0.008,
                seed=123,
            ),
            include_vega=True,
            ridge_penalty=1.0e-12,
        )
        result = hedging_engine.run(
            yield_curve_trajectory=curve_path,
            time_grid=time_grid,
            yield_curve_tenors=self.curve_tenors,
            target_instrument=target_swaption,
            hedging_instruments=[hedge_swaption],
            swaption_surface_trajectory=trajectory,
        )

        self.assertEqual(result.calibrated_parameter_names, ("volatility",))
        np.testing.assert_allclose(result.calibrated_parameters[:, 0], np.array([0.010, 0.011]), atol=1.0e-10)
        np.testing.assert_allclose(result.calibration_rmse, np.zeros(time_grid.size), atol=1.0e-12)
        np.testing.assert_allclose(result.portfolio_value, np.zeros(time_grid.size), atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
