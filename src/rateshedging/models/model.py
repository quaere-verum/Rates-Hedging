from dataclasses import dataclass
import numpy as np
import abc

@dataclass(frozen=True)
class RatePaths:
    yield_curve_paths: np.ndarray
    stochastic_discount_factors: np.ndarray
    yield_curve_factors: np.ndarray
    yield_curve_tenors: np.ndarray
    time: np.ndarray
    n_paths: int
    forward_rate_paths: np.ndarray | None = None
    forward_rate_tenor_dates: np.ndarray | None = None

class InterestRateModel(abc.ABC):
    @abc.abstractmethod
    def generate_paths(self, n_paths: int) -> RatePaths:
        raise NotImplementedError
