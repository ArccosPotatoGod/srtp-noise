"""Coherent noise and Gaussian noise generation.

Implements:
- Coherent noise coupling (simplified Eq.17 from paper)
- Adaptive-weight coherent noise (improvement)
- Band-limited Gaussian noise baseline
"""

import numpy as np
from scipy.signal import butter, lfilter
from .demodulation import nonlinear_demod


def generate_coherent_noise(s, fs=16000):
    """Generate coherent noise tightly coupled to speech signal.

    Implements a simplified version of Eq.(17):
      n_co(t) = sigma(A0*s + A1*n_M + A2*n_F*s)

    The sigmoid activation creates nonlinear coupling that is
    resistant to blind source separation.

    Parameters
    ----------
    s : np.ndarray
        Speech signal.
    fs : int
        Sample rate (unused; kept for interface consistency).

    Returns
    -------
    N : np.ndarray
        Coherent noise (same length as s).
    """
    T = len(s)
    n_M = np.random.randn(T) * 0.5
    n_F = np.random.randn(T) * 0.3
    freq_component = n_F * s

    A = np.random.randn(3) * 0.8 + 0.2
    mixed = A[0] * s + A[1] * n_M + A[2] * freq_component

    noise_baseband = 1.0 - np.exp(-mixed) / (1.0 + np.exp(-mixed))
    N = nonlinear_demod(noise_baseband)
    return N


def adaptive_coherent_noise(s, estimated_distance, fs=16000):
    """Generate coherent noise with adaptive coupling weight.

    Improvement: adjusts the speech-component weight alpha based
    on estimated eavesdropper distance.
    - Close range (<2m): high coupling (alpha=1.2) to resist denoising.
    - Medium range (2-4m): moderate coupling (alpha=0.8).
    - Far range (>4m): low coupling (alpha=0.4) to maximize jamming.

    Parameters
    ----------
    s : np.ndarray
        Speech signal.
    estimated_distance : float
        Estimated distance to eavesdropper in meters.
    fs : int
        Sample rate.

    Returns
    -------
    N : np.ndarray
        Adaptive coherent noise.
    """
    if estimated_distance < 2.0:
        alpha = 1.2
    elif estimated_distance < 4.0:
        alpha = 0.8
    else:
        alpha = 0.4

    T = len(s)
    n_M = np.random.randn(T) * 0.5
    n_F = np.random.randn(T) * 0.3
    freq_component = n_F * s

    A = np.array([alpha * 1.0, 0.8, 0.5])
    mixed = A[0] * s + A[1] * n_M + A[2] * freq_component

    noise_baseband = 1.0 - np.exp(-mixed) / (1.0 + np.exp(-mixed))
    N = nonlinear_demod(noise_baseband)
    return N


def generate_gaussian_noise(T, fs=16000, bandwidth=4000):
    """Generate band-limited Gaussian noise.

    Parameters
    ----------
    T : int
        Number of samples.
    fs : int
        Sample rate.
    bandwidth : float
        Cutoff frequency for low-pass filter.

    Returns
    -------
    noise : np.ndarray
        Band-limited Gaussian noise.
    """
    b, a = butter(4, bandwidth / (fs / 2), btype='low')
    noise = np.random.randn(T)
    gaussian_noise = lfilter(b, a, noise)
    return gaussian_noise * 0.5
