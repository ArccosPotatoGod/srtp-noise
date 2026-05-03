"""Coherent noise and Gaussian noise generation.

Implements:
- Coherent noise coupling (simplified Eq.17 from Gao et al., 2023)
- Adaptive-weight coherent noise (improvement, Section 6)
- Band-limited Gaussian noise baseline (traditional UMJ)

The coherent noise uses a nonlinear sigmoid activation on a linear
mixture of speech, random noise, and frequency-modulated components,
making it resistant to blind source separation attacks.
"""

import numpy as np
from scipy.signal import butter, lfilter
from .demodulation import nonlinear_demod


def generate_coherent_noise(s, fs=16000, rng=None):
    """Generate coherent noise tightly coupled to speech signal.

    Implements simplified Eq.(17):
      mixed = A0*s + A1*n_M + A2*(n_F * s)
      noise_baseband = sigmoid(mixed)
      N = nonlinear_demod(noise_baseband)

    Parameters
    ----------
    s : np.ndarray
        Speech signal.
    fs : int
        Sample rate (unused; kept for interface consistency).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    N : np.ndarray
        Coherent noise (same length as s).
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(s)
    n_M = rng.normal(0, 0.5, T)
    n_F = rng.normal(0, 0.3, T)
    freq_component = n_F * s

    A = rng.normal(0.2, 0.8, 3)
    mixed = A[0] * s + A[1] * n_M + A[2] * freq_component

    noise_baseband = 1.0 - np.exp(-mixed) / (1.0 + np.exp(-mixed))
    N = nonlinear_demod(noise_baseband)
    return N


def adaptive_coherent_noise(s, estimated_distance, fs=16000, rng=None):
    """Generate coherent noise with adaptive coupling weight.

    Improvement (Section 6): adjusts the speech-component weight alpha
    based on estimated eavesdropper distance:
    - Close range (<2m): alpha=1.2 — high coupling resists denoising.
    - Medium range (2-4m): alpha=0.8 — moderate coupling.
    - Far range (>4m): alpha=0.4 — low coupling, maximize jamming.

    Parameters
    ----------
    s : np.ndarray
        Speech signal.
    estimated_distance : float
        Estimated distance to eavesdropper in meters.
    fs : int
        Sample rate.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    N : np.ndarray
        Adaptive coherent noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    if estimated_distance < 2.0:
        alpha = 1.2
    elif estimated_distance < 4.0:
        alpha = 0.8
    else:
        alpha = 0.4

    T = len(s)
    n_M = rng.normal(0, 0.5, T)
    n_F = rng.normal(0, 0.3, T)
    freq_component = n_F * s

    A = np.array([alpha * 1.0, 0.8, 0.5])
    mixed = A[0] * s + A[1] * n_M + A[2] * freq_component

    noise_baseband = 1.0 - np.exp(-mixed) / (1.0 + np.exp(-mixed))
    N = nonlinear_demod(noise_baseband)
    return N


def generate_gaussian_noise(T, fs=16000, bandwidth=4000, rng=None):
    """Generate band-limited Gaussian noise (traditional UMJ baseline).

    Parameters
    ----------
    T : int
        Number of samples.
    fs : int
        Sample rate.
    bandwidth : float
        Cutoff frequency for low-pass filter.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    noise : np.ndarray
        Band-limited Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    b, a = butter(4, bandwidth / (fs / 2), btype='low')
    noise = rng.normal(0, 1, T)
    gaussian_noise = lfilter(b, a, noise)
    return gaussian_noise * 0.5
