from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rateshedging.hedging.engine import DynamicHedgingEngine
from rateshedging.hedging.model_adapters import HullWhiteModelAdapter
from rateshedging.instruments.bermudan_swaption import BermudanSwaption
from rateshedging.instruments.swap import Swap
from rateshedging.instruments.swaption import Swaption
from rateshedging.models.g2pp import G2PPModel
from rateshedging.models.hull_white import HullWhiteModel
from rateshedging.pricing.curve import CurveSnapshot
from rateshedging.pricing.engine import MonteCarloPricingEngine
from rateshedging.pricing.regression import LongstaffSchwartzExerciseStrategy


class InstrumentsPricingAndHedgingTests(unittest.TestCase):
    def test_swap_cashflows_and_par_value(self) -> None:
        tenors = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        zero_rates = np.full_like(tenors, 0.02)
        curve = CurveSnapshot.from_zero_rates(tenors, zero_rates)

        template_swap = Swap(
            start_time=0.0,
            payment_times=np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
            fixed_rate=0.0,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        fair_rate = float(template_swap.fair_rate_from_zero_rates(zero_rates, tenors))

        par_swap = Swap(
            start_time=0.0,
            payment_times=np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
            fixed_rate=fair_rate,
            notional=1_000_000.0,
            pay_fixed=True,
        )

        fixed_leg = par_swap.fixed_leg_cashflows()
        floating_leg = par_swap.floating_leg_cashflows(curve)

        self.assertEqual(len(fixed_leg), 4)
        self.assertEqual(len(floating_leg), 4)
        self.assertAlmostEqual(par_swap.present_value_from_curve(curve), 0.0, places=8)

    def test_bermudan_prices_at_least_european(self) -> None:
        curve_times = np.linspace(0.0, 5.0, 41)
        discount_factors = np.exp(-0.02 * curve_times)
        model_time_grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        curve_tenors = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        underlying = Swap(
            start_time=1.0,
            payment_times=np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            fixed_rate=0.02,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        european = Swaption(underlying_swap=underlying, expiry=1.0)
        bermudan = BermudanSwaption(underlying_swap=underlying, exercise_times=np.array([1.0, 2.0], dtype=np.float64))

        model = HullWhiteModel(
            mean_reversion=0.08,
            volatility=0.01,
            time_grid=model_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=curve_tenors,
            seed=123,
        )

        engine = MonteCarloPricingEngine(n_paths=20_000)
        european_price = engine.price(european, model).price

        model = HullWhiteModel(
            mean_reversion=0.08,
            volatility=0.01,
            time_grid=model_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=curve_tenors,
            seed=123,
        )
        bermudan_price = engine.price(bermudan, model).price

        self.assertGreaterEqual(bermudan_price + 5.0e-4, european_price)

    def test_g2pp_bermudan_prices_at_least_european(self) -> None:
        curve_times = np.linspace(0.0, 5.0, 41)
        discount_factors = np.exp(-0.02 * curve_times)
        model_time_grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        curve_tenors = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        underlying = Swap(
            start_time=1.0,
            payment_times=np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            fixed_rate=0.02,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        european = Swaption(underlying_swap=underlying, expiry=1.0)
        bermudan = BermudanSwaption(underlying_swap=underlying, exercise_times=np.array([1.0, 2.0], dtype=np.float64))

        model = G2PPModel(
            mean_reversion_x=0.15,
            mean_reversion_y=0.03,
            volatility_x=0.010,
            volatility_y=0.006,
            correlation=-0.70,
            time_grid=model_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=curve_tenors,
            seed=321,
        )

        engine = MonteCarloPricingEngine(n_paths=20_000)
        european_price = engine.price(european, model).price

        model = G2PPModel(
            mean_reversion_x=0.15,
            mean_reversion_y=0.03,
            volatility_x=0.010,
            volatility_y=0.006,
            correlation=-0.70,
            time_grid=model_time_grid,
            curve_times=curve_times,
            discount_factors=discount_factors,
            yield_curve_tenors=curve_tenors,
            seed=321,
        )
        bermudan_price = engine.price(bermudan, model).price

        self.assertGreaterEqual(bermudan_price + 5.0e-4, european_price)

    def test_option_cashflows_only_exist_on_exercise_dates(self) -> None:
        tenors = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        high_rate_curve = CurveSnapshot.from_zero_rates(tenors, np.full_like(tenors, 0.05))

        underlying = Swap(
            start_time=1.0,
            payment_times=np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            fixed_rate=0.02,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        european = Swaption(underlying_swap=underlying, expiry=1.0)
        bermudan = BermudanSwaption(underlying_swap=underlying, exercise_times=np.array([1.0, 2.0], dtype=np.float64))

        self.assertEqual(european.cashflows(high_rate_curve, valuation_time=0.0), ())
        self.assertEqual(bermudan.cashflows(high_rate_curve, valuation_time=0.5), ())
        self.assertTrue(european.cashflows(high_rate_curve, valuation_time=1.0))
        self.assertTrue(bermudan.cashflows(high_rate_curve, valuation_time=1.0))
        self.assertEqual(european.cashflows(high_rate_curve, valuation_time=1.5), ())
        self.assertEqual(bermudan.cashflows(high_rate_curve, valuation_time=1.5), ())

    def test_dynamic_hedging_replicates_identical_swap(self) -> None:
        tenors = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        time_grid = np.array([0.0, 0.25, 0.5], dtype=np.float64)
        curve_path = np.full((time_grid.size, tenors.size), 0.02, dtype=np.float64)

        target_swap = Swap(
            start_time=1.0,
            payment_times=np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            fixed_rate=0.021,
            notional=2_000_000.0,
            pay_fixed=True,
        )
        hedge_swap = Swap(
            start_time=1.0,
            payment_times=np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            fixed_rate=0.021,
            notional=2_000_000.0,
            pay_fixed=True,
        )

        hedging_engine = DynamicHedgingEngine(pricing_engine=MonteCarloPricingEngine(n_paths=2_000))
        result = hedging_engine.run(
            yield_curve_trajectory=curve_path,
            time_grid=time_grid,
            yield_curve_tenors=tenors,
            target_instrument=target_swap,
            hedging_instruments=[hedge_swap],
        )

        np.testing.assert_allclose(result.hedge_weights[:, 0], np.ones(time_grid.size), atol=1.0e-10)
        np.testing.assert_allclose(result.portfolio_value, np.zeros(time_grid.size), atol=1.0e-8)
        np.testing.assert_allclose(result.residual_key_rate_sensitivities, 0.0, atol=1.0e-8)
        np.testing.assert_allclose(
            result.incremental_pnl,
            result.cash_carry_pnl
            + result.target_cashflow_pnl
            + result.hedge_cashflow_pnl
            + result.target_revaluation_pnl
            + result.hedge_revaluation_pnl,
            atol=1.0e-10,
        )

    def test_delta_vega_hedging_replicates_identical_swaption(self) -> None:
        tenors = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        time_grid = np.array([0.0, 0.5], dtype=np.float64)
        curve_path = np.full((time_grid.size, tenors.size), 0.02, dtype=np.float64)

        underlying_swap = Swap(
            start_time=0.5,
            payment_times=np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64),
            fixed_rate=0.021,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        target_swaption = Swaption(underlying_swap=underlying_swap, expiry=0.5)
        hedge_swaption = Swaption(underlying_swap=underlying_swap, expiry=0.5)

        hedging_engine = DynamicHedgingEngine(
            pricing_engine=MonteCarloPricingEngine(n_paths=4_000),
            model_adapter=HullWhiteModelAdapter(
                mean_reversion=0.08,
                volatility=0.01,
                seed=123,
            ),
            include_vega=True,
            ridge_penalty=1.0e-12,
        )
        result = hedging_engine.run(
            yield_curve_trajectory=curve_path,
            time_grid=time_grid,
            yield_curve_tenors=tenors,
            target_instrument=target_swaption,
            hedging_instruments=[hedge_swaption],
        )

        self.assertEqual(result.vega_labels, ("volatility",))
        np.testing.assert_allclose(result.hedge_weights[0, 0], 1.0, atol=1.0e-8)
        np.testing.assert_allclose(result.residual_key_rate_sensitivities[0], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(result.residual_vega_sensitivities[0], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(result.portfolio_value, np.zeros(time_grid.size), atol=1.0e-6)

    def test_unhedged_swap_pnl_breakdown_matches_incremental_pnl(self) -> None:
        tenors = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        time_grid = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        curve_path = np.array(
            [
                [0.0200, 0.0200, 0.0200, 0.0200],
                [0.0210, 0.0215, 0.0220, 0.0225],
                [0.0190, 0.0195, 0.0200, 0.0205],
            ],
            dtype=np.float64,
        )

        target_swap = Swap(
            start_time=0.5,
            payment_times=np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64),
            fixed_rate=0.021,
            notional=1_000_000.0,
            pay_fixed=True,
        )

        result = DynamicHedgingEngine(pricing_engine=MonteCarloPricingEngine(n_paths=2_000)).run(
            yield_curve_trajectory=curve_path,
            time_grid=time_grid,
            yield_curve_tenors=tenors,
            target_instrument=target_swap,
            hedging_instruments=[],
        )

        np.testing.assert_allclose(
            result.incremental_pnl,
            result.cash_carry_pnl
            + result.target_cashflow_pnl
            + result.hedge_cashflow_pnl
            + result.target_revaluation_pnl
            + result.hedge_revaluation_pnl,
            atol=1.0e-10,
        )
        self.assertAlmostEqual(result.rebalancing_cashflow[0], -result.target_values[0], places=8)
        np.testing.assert_allclose(result.target_vega_sensitivities, np.zeros((time_grid.size, 0)))

    def test_rebalancing_cashflow_preserves_portfolio_value(self) -> None:
        tenors = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        time_grid = np.array([0.0, 0.25, 0.5], dtype=np.float64)
        curve_path = np.array(
            [
                [0.0200, 0.0205, 0.0210, 0.0215],
                [0.0185, 0.0190, 0.0195, 0.0200],
                [0.0220, 0.0225, 0.0230, 0.0235],
            ],
            dtype=np.float64,
        )

        target_swap = Swap(
            start_time=1.0,
            payment_times=np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            fixed_rate=0.021,
            notional=1_000_000.0,
            pay_fixed=True,
        )
        hedge_swaps = [
            Swap(
                start_time=0.0,
                payment_times=np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
                fixed_rate=0.02,
                notional=1_000_000.0,
                pay_fixed=True,
            ),
            Swap(
                start_time=0.0,
                payment_times=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64),
                fixed_rate=0.021,
                notional=1_000_000.0,
                pay_fixed=True,
            ),
        ]

        result = DynamicHedgingEngine(pricing_engine=MonteCarloPricingEngine(n_paths=2_000)).run(
            yield_curve_trajectory=curve_path,
            time_grid=time_grid,
            yield_curve_tenors=tenors,
            target_instrument=target_swap,
            hedging_instruments=hedge_swaps,
        )

        np.testing.assert_allclose(
            result.portfolio_value[1:],
            result.portfolio_value[:-1] + result.incremental_pnl[1:],
            atol=1.0e-8,
        )

    def test_regression_framework_returns_single_exercise_per_path(self) -> None:
        intrinsic = np.array(
            [
                [0.10, 0.00, 0.00],
                [0.00, 0.20, 0.00],
                [0.00, 0.00, 0.30],
            ],
            dtype=np.float64,
        )
        discounts = np.ones_like(intrinsic)
        factors = np.stack([intrinsic], axis=-1)

        strategy = LongstaffSchwartzExerciseStrategy(min_regression_paths=1)
        result = strategy.fit(intrinsic, discounts, factors)

        self.assertEqual(result.exercise_indicators.shape, intrinsic.shape)
        np.testing.assert_array_less(np.sum(result.exercise_indicators, axis=1), np.full(3, 2))
        np.testing.assert_allclose(result.discounted_payoffs, np.array([0.10, 0.20, 0.30], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
