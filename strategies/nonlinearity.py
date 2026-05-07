"""Nonlinearity models — microphone nonlinear response simulation.

Microphone output: y = A1*x + A2*x² + A3*x³ + ...
Per the paper, terms >= A3 can be ignored due to low power (Sec.2).
"""

from abc import ABC, abstractmethod

import numpy as np


class INonlinearityModel(ABC):
    @abstractmethod
    def apply(self, signal: np.ndarray) -> np.ndarray:
        pass


class PolynomialNonlinearity(INonlinearityModel):
    """Polynomial microphone nonlinearity: y = sum(coeff[i] * x^(i+1))."""

    def __init__(self, coeff: list = None):
        self.coeff = coeff or [1.0, 0.1, 0.0]

    def apply(self, signal: np.ndarray) -> np.ndarray:
        y = np.zeros_like(signal, dtype=np.float64)
        x = signal.astype(np.float64)
        for i, c in enumerate(self.coeff):
            if c != 0.0:
                y += c * x ** (i + 1)
        return y.astype(np.float32)


class PassThroughNonlinearity(INonlinearityModel):
    """Identity mapping — for debugging linear-only mode."""

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return signal.astype(np.float32)


def create_nonlinearity(name: str, params: dict) -> INonlinearityModel:
    if name == "polynomial":
        return PolynomialNonlinearity(coeff=params.get("coeff", [1.0, 0.1, 0.0]))
    elif name == "passthrough":
        return PassThroughNonlinearity()
    raise ValueError(f"Unknown nonlinearity model: {name}")
