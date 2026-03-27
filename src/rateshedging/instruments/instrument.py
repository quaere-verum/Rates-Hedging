from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from rateshedging.pricing.curve import CurveSnapshot


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Cashflow:
    payment_time: float
    amount: float
    leg: str
    accrual_start: float | None = None
    accrual_end: float | None = None
    year_fraction: float | None = None


class InterestRateInstrument(ABC):
    @property
    @abstractmethod
    def direction(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def cashflows(self, curve: CurveSnapshot, valuation_time: float = 0.0) -> tuple[Cashflow, ...]:
        raise NotImplementedError

    @abstractmethod
    def required_model_times(self, valuation_time: float = 0.0) -> FloatArray:
        raise NotImplementedError


__all__ = ["Cashflow", "InterestRateInstrument"]
