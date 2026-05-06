"""Acoustic propagation and attenuation model.

Simulates sound propagation in air:
- Distance-dependent attenuation (inverse-square law approximation)
- Ultrasonic extra attenuation (exponential decay, alpha = 2.0)
- Fractional-sample time delay (linear interpolation)
- Optional multipath reflections
- Optional microphone self-noise
- Hardware latency constant for jammer processing chain

Reference: Gao et al., MobiCom 2023, Section 4.1
"""

import numpy as np

SPEED_OF_SOUND = 340.0
MIN_DIST = 0.1
ULTRA_ALPHA = 2.0

# Hardware processing latency in the jammer: AD conversion → DSP → DA conversion.
# Typical low-latency ANC systems: 0.1–0.5 ms. We use 0.25 ms.
# Residual timing jitter after latency compensation (seconds).
# Modern ANC systems compensate for their known hardware latency, but
# residual clock jitter and timing uncertainty remain (~5 μs RMS).
TIMING_JITTER_STD = 5e-6  # 5 μs

# Microphone self-noise floor relative to a nominal speech level (RMS).
# Consumer MEMS mics: SNR ~60–65 dB. We use 60 dB.
MIC_NOISE_FLOOR_DB = -60.0

# Ground reflection coefficient (0 = no reflection, 0.3 = moderate).
REFLECTION_COEFF = 0.3


def compute_distance(pos_a, pos_b):
    """Euclidean distance between two 2D positions."""
    return np.linalg.norm(np.array(pos_a) - np.array(pos_b))


def fractional_delay(signal, delay_seconds, fs):
    """Apply a fractional-sample time delay via linear interpolation.

    Unlike integer-sample delay (which forces delays to be exact sample
    multiples), this properly models sub-sample propagation delay where
    the signal arrival does not align precisely to sample boundaries.
    This sub-sample misalignment is a key physical mechanism limiting
    ANC cancellation depth.

    Uses np.interp with linear interpolation. Samples arriving from
    before time zero are zero (signal hasn't started yet).

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    delay_seconds : float
        Desired delay in seconds (can be fractional).
    fs : int
        Sample rate in Hz.

    Returns
    -------
    delayed : np.ndarray, shape (n_samples,)
    """
    n = len(signal)
    delay_samples = delay_seconds * fs

    if delay_samples >= n:
        return np.zeros_like(signal)

    # Output sample i comes from input time (i - delay_samples).
    # Query the original signal at shifted positions: query positions before
    # index 0 return left=0 (signal hasn't arrived yet = causal delay).
    t_in = np.arange(n, dtype=np.float64)
    t_query = t_in - delay_samples
    delayed = np.interp(t_query, t_in, signal, left=0.0, right=0.0)
    return delayed.astype(signal.dtype)


def add_mic_noise(signal, noise_floor_db=MIC_NOISE_FLOOR_DB, rng=None):
    """Add microphone self-noise to a captured signal.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal captured at microphone.
    noise_floor_db : float
        Noise floor in dB relative to nominal speech level.
    rng : np.random.Generator or None

    Returns
    -------
    noisy : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    sig_rms = np.sqrt(np.mean(signal ** 2)) + 1e-12
    noise_rms = sig_rms * (10 ** (noise_floor_db / 20.0))
    noise = rng.normal(0, noise_rms, len(signal))
    return signal + noise


def propagate_signal(signal, src_pos, mic_pos, fs,
                     v=SPEED_OF_SOUND, is_ultrasound=False,
                     timing_jitter=0.0, add_noise=False, rng=None):
    """Propagate a signal from source to microphone with sub-sample delay.

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
        Speed of sound in m/s (default 340).
    is_ultrasound : bool
        If True, apply extra ultrasonic attenuation.
    timing_jitter : float
        Std dev of residual timing jitter (seconds) after latency compensation.
    add_noise : bool
        If True, add microphone self-noise to the received signal.
    rng : np.random.Generator or None

    Returns
    -------
    propagated : np.ndarray, shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    dist = compute_distance(src_pos, mic_pos)
    total_delay_s = dist / v

    # Add residual timing jitter (uncompensated random error)
    if timing_jitter > 0:
        total_delay_s += rng.normal(0, timing_jitter)

    # Fractional-sample delay (sub-sample precision → realistic misalignment)
    delayed = fractional_delay(signal, total_delay_s, fs)

    # Audible attenuation: 1 / (dist + 0.1)
    atten = 1.0 / (dist + MIN_DIST)
    if is_ultrasound:
        atten *= np.exp(-ULTRA_ALPHA * dist)

    propagated = delayed * atten

    # Add microphone self-noise if requested
    if add_noise:
        propagated = add_mic_noise(propagated, rng=rng)

    return propagated
