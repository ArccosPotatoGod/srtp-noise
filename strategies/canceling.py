"""Canceling strategies — anti-speech signal generation.

Implements:
- PhaseInversionCanceling (with precompensation per Eq.9)
- AdaptiveFilterCanceling (NLMS per Eq.13)
"""

import numpy as np
from src.jammer import ICancelingStrategy


class PhaseInversionCanceling(ICancelingStrategy):
    """Simple phase-inversion with optional precompensation."""

    def __init__(self, gain: float = 1.0, precompensate: bool = True):
        self.gain = gain
        self.precompensate = precompensate

    def compute(self, ref_signal: np.ndarray) -> np.ndarray:
        if self.precompensate:
            # Eq.9: n(t) = -ŝ(t) - 0.5 * ŝ²(t)
            return (-self.gain * ref_signal
                    - 0.5 * self.gain ** 2 * ref_signal ** 2).astype(np.float32)
        return (-self.gain * ref_signal).astype(np.float32)


class AdaptiveFilterCanceling(ICancelingStrategy):
    """NLMS adaptive filter for inverse-channel estimation (Sec.5.3)."""

    def __init__(self, n_taps: int = 64, mu: float = 0.01, delta: float = 1e-6):
        self.n_taps = n_taps
        self.mu = mu
        self.delta = delta

    def compute(self, ref_signal: np.ndarray) -> np.ndarray:
        T = len(ref_signal)
        w = np.zeros(self.n_taps, dtype=np.float32)
        s_cancel = np.zeros(T, dtype=np.float32)

        for n in range(self.n_taps, T):
            x = ref_signal[n - self.n_taps + 1:n + 1][::-1]
            y = np.dot(w, x)
            s_cancel[n] = y
            e = -y  # desired = 0 (perfect cancellation)
            norm = np.dot(x, x) + self.delta
            w += self.mu / norm * e * x

        return s_cancel


class PassthroughCanceling(ICancelingStrategy):
    """Jammer-off — returns silence (no cancel signal)."""

    def compute(self, ref_signal: np.ndarray) -> np.ndarray:
        return np.zeros(len(ref_signal), dtype=np.float32)


def create_canceling(strategy_name: str, params: dict) -> ICancelingStrategy:
    if strategy_name == "phase_inversion":
        return PhaseInversionCanceling(
            gain=params.get("gain", 1.0),
            precompensate=params.get("precompensate", True),
        )
    elif strategy_name == "adaptive_filter":
        return AdaptiveFilterCanceling(
            n_taps=params.get("n_taps", 64),
            mu=params.get("mu", 0.01),
            delta=params.get("delta", 1e-6),
        )
    elif strategy_name == "off":
        return PassthroughCanceling()
    raise ValueError(f"Unknown canceling strategy: {strategy_name}")
