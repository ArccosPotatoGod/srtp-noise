"""Acoustic propagation and attenuation model.

Simulates sound propagation in air:
- Distance-dependent attenuation (inverse-square law approximation)
- Ultrasonic extra attenuation (exponential decay, alpha = 2.0)
- Time-delay based on speed of sound (340 m/s)

Reference: Gao et al., MobiCom 2023, Section 4.1
"""

import numpy as np

SPEED_OF_SOUND = 340.0
MIN_DIST = 0.1
ULTRA_ALPHA = 2.0


def propagate_signal(signal, src_pos, mic_pos, fs,
                     v=SPEED_OF_SOUND, is_ultrasound=False):
    """Propagate a signal from source to microphone with delay & attenuation.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Source signal (float32 per spec).
    src_pos : tuple (x, y)
        Source position in meters.
    mic_pos : tuple (x, y)
        Microphone position in meters.
    fs : int
        Sample rate in Hz.
    v : float
        Speed of sound in m/s (default 340).
    is_ultrasound : bool
        If True, apply extra ultrasonic attenuation exp(-alpha * dist).

    Returns
    -------
    propagated : np.ndarray, shape (n_samples,)
    """
    dist = compute_distance(src_pos, mic_pos)
    delay_samples = int(round(dist / v * fs))

    # Audible attenuation: 1 / (dist + 0.1)
    atten = 1.0 / (dist + MIN_DIST)
    if is_ultrasound:
        atten *= np.exp(-ULTRA_ALPHA * dist)

    propagated = np.zeros_like(signal)
    if delay_samples < len(signal):
        propagated[delay_samples:] = signal[:len(signal) - delay_samples] * atten
    # If delay exceeds signal length, signal arrives after recording ends -> all zeros
    return propagated


def compute_distance(pos_a, pos_b):
    """Euclidean distance between two 2D positions."""
    return np.linalg.norm(np.array(pos_a) - np.array(pos_b))
