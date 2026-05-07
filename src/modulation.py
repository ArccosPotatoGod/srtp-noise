"""Ultrasonic modulation/demodulation helper functions.

Provides the equivalent-baseband simulation of ultrasonic carrier modulation
and microphone nonlinear demodulation, per the paper's acoustic model.
"""

import numpy as np


def ultrasonic_modulate(baseband: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """Simulate ultrasonic amplitude modulation.

    Returns a pseudo-ultrasonic signal at sample rate fs with carrier fc.
    In equivalent-baseband mode this is a pass-through; callers should use
    the nonlinearity model at the receiver side instead.
    """
    t = np.arange(len(baseband)) / fs
    carrier = np.cos(2.0 * np.pi * fc * t)
    return baseband * carrier


def precompensate_cancel(ref_signal: np.ndarray) -> np.ndarray:
    """Apply pre-compensation per Eq.9 of the paper.

    n(t) = -ŝ(t) - 0.5 * ŝ²(t)

    When this signal passes through microphone nonlinearity
    y = A1*x + A2*x², the baseband output is -ŝ(t) + higher-order residues.
    """
    return -ref_signal - 0.5 * ref_signal ** 2
