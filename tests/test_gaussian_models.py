from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rateshedging.models.g2pp import G2PPModel
from rateshedging.models.hull_white import HullWhiteModel
from rateshedging.models.libor_market_model import LIBORMarketModel


class GaussianModelPathGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curve_times = np.linspace(0.0, 30.0, 121)
        self.discount_factors = np.exp(-0.02 * self.curve_times)
        self.time_grid = np.linspace(0.0, 5.0, 41)
        self.tenors = np.array([0.25, 1.0, 5.0, 10.0], dtype=np.float64)
        self.expected_discount_curve = np.exp(-0.02 * self.time_grid)
        self.expected_initial_yields = np.full_like(self.tenors, 0.02)

    def test_hull_white_paths_fit_initial_curve(self) -> None:
        model = HullWhiteModel(
            mean_reversion=0.08,
            volatility=0.01,
            time_grid=self.time_grid,
            curve_times=self.curve_times,
            discount_factors=self.discount_factors,
            yield_curve_tenors=self.tenors,
            seed=123,
        )

        paths = model.generate_paths(20_000)

        self.assertEqual(paths.yield_curve_paths.shape, (20_000, self.time_grid.size, self.tenors.size))
        self.assertEqual(paths.stochastic_discount_factors.shape, (20_000, self.time_grid.size))
        self.assertEqual(paths.yield_curve_factors.shape, (20_000, self.time_grid.size, 1))
        np.testing.assert_allclose(paths.yield_curve_paths[0, 0], self.expected_initial_yields, atol=1.0e-12)
        np.testing.assert_allclose(
            paths.stochastic_discount_factors.mean(axis=0),
            self.expected_discount_curve,
            atol=6.0e-4,
        )

    def test_g2pp_paths_fit_initial_curve(self) -> None:
        model = G2PPModel(
            mean_reversion_x=0.12,
            mean_reversion_y=0.03,
            volatility_x=0.009,
            volatility_y=0.006,
            correlation=-0.75,
            time_grid=self.time_grid,
            curve_times=self.curve_times,
            discount_factors=self.discount_factors,
            yield_curve_tenors=self.tenors,
            seed=456,
        )

        paths = model.generate_paths(20_000)

        self.assertEqual(paths.yield_curve_paths.shape, (20_000, self.time_grid.size, self.tenors.size))
        self.assertEqual(paths.stochastic_discount_factors.shape, (20_000, self.time_grid.size))
        self.assertEqual(paths.yield_curve_factors.shape, (20_000, self.time_grid.size, 2))
        np.testing.assert_allclose(paths.yield_curve_paths[0, 0], self.expected_initial_yields, atol=1.0e-12)
        np.testing.assert_allclose(
            paths.stochastic_discount_factors.mean(axis=0),
            self.expected_discount_curve,
            atol=6.0e-4,
        )

    def test_lmm_paths_fit_initial_curve(self) -> None:
        model = LIBORMarketModel(
            tenor_spacing=0.125,
            factor_volatilities=np.array([0.16, 0.08, 0.04], dtype=np.float64),
            factor_decay_rates=np.array([0.0, 0.35, 1.10], dtype=np.float64),
            time_grid=self.time_grid,
            curve_times=self.curve_times,
            discount_factors=self.discount_factors,
            yield_curve_tenors=self.tenors,
            seed=789,
        )

        paths = model.generate_paths(5_000)

        self.assertEqual(paths.yield_curve_paths.shape, (5_000, self.time_grid.size, self.tenors.size))
        self.assertEqual(paths.stochastic_discount_factors.shape, (5_000, self.time_grid.size))
        self.assertEqual(paths.yield_curve_factors.shape, (5_000, self.time_grid.size, 3))
        self.assertIsNotNone(paths.forward_rate_paths)
        self.assertIsNotNone(paths.forward_rate_tenor_dates)
        self.assertEqual(paths.forward_rate_paths.shape[:2], (5_000, self.time_grid.size))
        np.testing.assert_allclose(paths.yield_curve_paths[0, 0], self.expected_initial_yields, atol=1.0e-10)
        np.testing.assert_allclose(
            paths.stochastic_discount_factors.mean(axis=0),
            self.expected_discount_curve,
            atol=2.0e-3,
        )


if __name__ == "__main__":
    unittest.main()
