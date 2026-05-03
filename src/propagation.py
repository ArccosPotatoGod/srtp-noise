"""Acoustic propagation and attenuation model.

Simulates sound propagation in air:
- Distance-dependent attenuation (inverse-square law approximation)
- Ultrasonic extra attenuation (exponential decay)
- Time-delay based on speed of sound (340 m/s)
"""

import numpy as np

SPEED_OF_SOUND = 340.0


def propagate_signal(signal, src_pos, mic_pos, fs,
                     v=SPEED_OF_SOUND, is_ultrasound=False):
    """Propagate a signal from source to microphone with delay & attenuation.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Source signal (float32).
    src_pos : tuple (x, y)
        Source position in meters.
    mic_pos : tuple (x, y)
        Microphone position in meters.
    fs : int
        Sample rate in Hz.
    v : float
        Speed of sound in m/s.
    is_ultrasound : bool
        Apply extra ultrasonic attenuation if True.

    Returns
    -------
    propagated : np.ndarray, shape (n_samples,)
    """
    dist = np.linalg.norm(np.array(mic_pos) - np.array(src_pos))
    delay_samples = int(dist / v * fs)

    atten = 1.0 / (dist + 0.1)
    if is_ultrasound:
        atten *= np.exp(-2.0 * dist)

    propagated = np.zeros_like(signal, dtype=np.float64)
    if delay_samples < len(signal):
        propagated[delay_samples:] = signal[:len(signal) - delay_samples] * atten
    return propagated


def compute_distance(pos_a, pos_b):
    """Euclidean distance between two 2D positions."""
    return np.linalg.norm(np.array(pos_a) - np.array(pos_b))
