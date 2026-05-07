"""Coherent noise and baseline jamming strategies.

Implements:
- FixedWeightCoherentNoise (strictly per paper Eq.17)
- BaselineGaussianNoise (4 kHz Gaussian, SOTA UMJ baseline)
- BaselineSweepingNoise (0-4 kHz sweep)
- BaselineHoppingNoise (4 kHz band hopping)
"""

import numpy as np
from src.jammer import ICoherentNoiseStrategy


class FixedWeightCoherentNoise(ICoherentNoiseStrategy):
    """Coherent noise per Eq.17 — three layers of inseparability.

    n_co(t) = n_T(t) * [ A_{1×3} · sgm( s(t), n_M(t), n_F(t)·s(t) )^T ]

    Layer 1 — time: convolution with n_T makes noise coupled at any delay.
    Layer 2 — frequency: n_F·s makes frequency filters distort speech.
    Layer 3 — spatial: jammer placed near source (via config positions).
    """

    def __init__(self, time_coupling: bool = True, freq_coupling: bool = True,
                 mixing_dim: int = 3, rng: np.random.Generator = None):
        self.time_coupling = time_coupling
        self.freq_coupling = freq_coupling
        self.mixing_dim = mixing_dim
        self.rng = rng or np.random.default_rng()

    def compute(self, speech: np.ndarray) -> np.ndarray:
        T = len(speech)
        # Random mixing matrix A_{1×M}
        A = self.rng.normal(0.2, 0.8, size=(1, self.mixing_dim))

        # Build mixture inputs
        sources = [speech]
        if self.mixing_dim >= 2:
            n_M = self.rng.normal(0.0, 0.5, size=T)
            sources.append(n_M)
        if self.mixing_dim >= 3:
            n_F = self.rng.normal(0.0, 0.3, size=T)
            sources.append(n_F * speech)

        # Linear mixture
        mixed = np.zeros(T, dtype=np.float64)
        for i, src in enumerate(sources):
            mixed += A[0, i] * src

        # Sigmoid activation (Eq.15)
        sgm = 1.0 - np.exp(-mixed) / (1.0 + np.exp(-mixed))

        # Time-domain convolution coupling
        if self.time_coupling:
            n_t_len = int(0.3 * 16000)  # 300 ms
            n_T = self.rng.normal(0.0, 0.3, size=n_t_len)
            sgm = np.convolve(sgm, n_T, mode="same")
        return sgm.astype(np.float32)


class BaselineGaussianNoise(ICoherentNoiseStrategy):
    """4 kHz bandwidth Gaussian noise — SOTA UMJ baseline."""

    def __init__(self, bandwidth: float = 4000.0, fs: int = 16000,
                 rng: np.random.Generator = None):
        self.bandwidth = bandwidth
        self.fs = fs
        self.rng = rng or np.random.default_rng()

    def compute(self, speech: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, lfilter
        T = len(speech)
        noise = self.rng.normal(0.0, 1.0, size=T)
        nyq = 0.5 * self.fs
        low = min(self.bandwidth / nyq, 0.99)
        b, a = butter(4, low, btype="low")
        return lfilter(b, a, noise).astype(np.float32)


class BaselineSweepingNoise(ICoherentNoiseStrategy):
    """0–4 kHz sweeping frequency noise — baseline from [48]."""

    def __init__(self, f_low: float = 0.0, f_high: float = 4000.0,
                 fs: int = 16000, rng: np.random.Generator = None):
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.rng = rng or np.random.default_rng()

    def compute(self, speech: np.ndarray) -> np.ndarray:
        T = len(speech)
        t = np.arange(T) / self.fs
        freq = self.f_low + (self.f_high - self.f_low) * (t / t[-1] if T > 1 else 0)
        phase = 2.0 * np.pi * np.cumsum(freq) / self.fs
        return np.sin(phase).astype(np.float32)


class BaselineHoppingNoise(ICoherentNoiseStrategy):
    """Frequency-hopping noise within 4 kHz band — baseline from [15]."""

    def __init__(self, hop_range: tuple = (0.0, 4000.0), hop_rate: float = 100.0,
                 fs: int = 16000, rng: np.random.Generator = None):
        self.hop_range = hop_range
        self.hop_rate = hop_rate
        self.fs = fs
        self.rng = rng or np.random.default_rng()

    def compute(self, speech: np.ndarray) -> np.ndarray:
        T = len(speech)
        hop_samples = int(self.fs / self.hop_rate)
        signal = np.zeros(T, dtype=np.float64)
        t = np.arange(T) / self.fs
        for start in range(0, T, hop_samples):
            end = min(start + hop_samples, T)
            freq = self.rng.uniform(*self.hop_range)
            segment = np.sin(2.0 * np.pi * freq * t[:end - start])
            signal[start:end] = segment
        return signal.astype(np.float32)


def create_coherent(strategy_name: str, params: dict,
                    rng: np.random.Generator = None) -> ICoherentNoiseStrategy:
    rng = rng or np.random.default_rng()
    if strategy_name == "fixed_weight":
        return FixedWeightCoherentNoise(
            time_coupling=params.get("time_coupling", True),
            freq_coupling=params.get("freq_coupling", True),
            mixing_dim=params.get("mixing_dim", 3),
            rng=rng,
        )
    elif strategy_name == "gaussian_baseline":
        return BaselineGaussianNoise(
            bandwidth=params.get("bandwidth", 4000.0),
            fs=params.get("fs", 16000),
            rng=rng,
        )
    elif strategy_name == "sweeping_baseline":
        return BaselineSweepingNoise(
            f_low=params.get("f_low", 0.0),
            f_high=params.get("f_high", 4000.0),
            fs=params.get("fs", 16000),
            rng=rng,
        )
    elif strategy_name == "hopping_baseline":
        return BaselineHoppingNoise(
            hop_range=tuple(params.get("hop_range", (0.0, 4000.0))),
            hop_rate=params.get("hop_rate", 100.0),
            fs=params.get("fs", 16000),
            rng=rng,
        )
    raise ValueError(f"Unknown coherent strategy: {strategy_name}")
