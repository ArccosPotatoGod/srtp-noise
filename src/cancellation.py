"""Inverse-channel speech signal cancellation with realistic imperfections.

Simulates the MicFrozen core mechanism with physically plausible limitations:
1. Reference mic captures delayed speech + self-noise + timing jitter.
2. Jammer estimates spy position with error → gain mismatch.
3. Nominal hardware latency is compensated; residual timing jitter remains.
4. Speaker nonlinearity distorts the emitted cancel signal.
5. Sub-sample delay misalignment via fractional-delay interpolation.
6. Maximum cancellation bounded by acoustic physics (~25 dB ceiling).

Without these, the "perfect" cancellation is a mathematical artefact
achieving >100 dB reduction — physically impossible for acoustic ANC.
"""

import numpy as np
from .propagation import (propagate_signal, compute_distance,
                           TIMING_JITTER_STD, MIC_NOISE_FLOOR_DB)

# RMS error in jammer's estimate of spy position (meters).
# Acoustic localization from a single device is inherently imprecise.
POSITION_EST_ERROR_STD = 0.06  # meters (per axis)

# Maximum cancellation ratio (linear). Real-world ANC achieves 15–25 dB
# reduction. We model a ceiling of ~25 dB (= factor ~316 in amplitude).
MAX_CANCELLATION_DB = 25.0

# Speaker soft-clip threshold: cancelling signal is clipped at this fraction
# of the original speech peak to model speaker saturation.
SPEAKER_CLIP_FRAC = 0.85


def _soft_clip(x, threshold):
    """Soft-clip using tanh to model speaker saturation."""
    return threshold * np.tanh(x / threshold)


def _estimate_spy_position(spy_pos, rng):
    """Simulate imperfect estimation of spy position.

    The jammer uses acoustic ranging/localization to estimate where the
    eavesdropper is. Each coordinate has independent Gaussian error.

    Returns estimated (x, y) position.
    """
    x_err = rng.normal(0, POSITION_EST_ERROR_STD)
    y_err = rng.normal(0, POSITION_EST_ERROR_STD)
    return (spy_pos[0] + x_err, spy_pos[1] + y_err)


def apply_cancellation(s, src_pos, jammer_pos, spy_pos, fs,
                        max_cancel_db=MAX_CANCELLATION_DB, seed=None):
    """Apply MicFrozen speech cancellation with realistic imperfections.

    Signal chain:
    1. Reference mic at jammer captures speech + self-noise
    2. Jammer estimates spy distance with error
    3. Gain calibrated with the estimated (imperfect) distance
    4. Inverted replica is distorted by speaker nonlinearity
    5. Both signals propagate to spy with fractional-sample delays
    6. Cancellation ratio is bounded by a physical ceiling

    Parameters
    ----------
    s : np.ndarray, shape (n_samples,)
        Source speech signal.
    src_pos : tuple (x, y)
        Position of the speaker (sound source).
    jammer_pos : tuple (x, y)
        Position of MicFrozen device / jammer.
    spy_pos : tuple (x, y)
        Actual position of eavesdropping microphone.
    fs : int
        Sample rate in Hz.
    max_cancel_db : float
        Maximum cancellation ratio in dB (physical ceiling).
    seed : int or None
        Seed for reproducible randomness.

    Returns
    -------
    residual : np.ndarray
        Residual speech after cancellation.
    s_direct : np.ndarray
        Direct-path signal at spy (for comparison).
    """
    rng = np.random.default_rng(seed)

    # ---- 1. Reference microphone at jammer ----
    # Captures delayed speech with mic self-noise + timing jitter
    s_ref = propagate_signal(s, src_pos, jammer_pos, fs,
                             timing_jitter=TIMING_JITTER_STD,
                             add_noise=True, rng=rng)

    # ---- 2. Channel estimation error ----
    # Jammer does not know spy's exact position. It estimates with error
    # in each coordinate, leading to imperfect distance/gain calibration.
    d_ref = compute_distance(src_pos, jammer_pos)
    d_src_to_spy = compute_distance(src_pos, spy_pos)
    d_jam_to_spy = compute_distance(jammer_pos, spy_pos)

    spy_est = _estimate_spy_position(spy_pos, rng)
    d_src_estimated = compute_distance(src_pos, spy_est)
    d_jam_estimated = compute_distance(jammer_pos, spy_est)

    # ---- 3. Gain calibration with imperfect estimates ----
    atten_ref = 1.0 / (d_ref + 0.1)
    atten_direct = 1.0 / (d_src_to_spy + 0.1)
    atten_cancel_path = 1.0 / (d_jam_to_spy + 0.1)

    # Imperfect estimates of path attenuations
    atten_cancel_est = 1.0 / (d_jam_estimated + 0.1)
    atten_direct_est = 1.0 / (d_src_estimated + 0.1)

    # Gain based on IMPERFECT estimates (both distance errors compound)
    gain = atten_direct_est / (atten_ref * atten_cancel_est + 1e-10)

    anti_signal = -s_ref * gain

    # ---- 4. Speaker nonlinearity ----
    # The jammer's speaker introduces soft saturation
    peak = np.max(np.abs(s)) + 1e-12
    clip_threshold = SPEAKER_CLIP_FRAC * peak * gain * atten_ref
    anti_signal = _soft_clip(anti_signal, clip_threshold)

    # ---- 5. Jammer emits cancelling signal (latency-compensated) ----
    # Nominal hardware latency is compensated by advancing the emit.
    # Only residual timing jitter remains as imperfection.
    s_cancel = propagate_signal(anti_signal, jammer_pos, spy_pos, fs,
                                timing_jitter=TIMING_JITTER_STD, rng=rng)

    # Direct path from source to spy (also with jitter)
    s_direct = propagate_signal(s, src_pos, spy_pos, fs,
                                timing_jitter=TIMING_JITTER_STD, rng=rng)

    # ---- 6. Superposition with bounded cancellation ----
    # Even in the best case (zero delay mismatch, perfect calibration),
    # real ANC cannot exceed PHYSICAL cancellation ceiling.
    # We model this as a leakage floor: a fraction of the direct signal
    # always survives, representing uncancellable multipath, diffraction,
    # and other physical effects.

    residual = s_direct + s_cancel

    # Ensure residual energy doesn't fall below the cancellation floor.
    # Leakage floor: the fraction of direct signal that survives cancellation.
    leak_frac = 10 ** (-max_cancel_db / 20.0)  # e.g., 25dB → 0.056
    residual_min_rms = np.sqrt(np.mean(s_direct ** 2)) * leak_frac
    residual_rms = np.sqrt(np.mean(residual ** 2)) + 1e-12

    if residual_rms < residual_min_rms:
        # Add uncorrelated residual to reach the cancellation floor
        needed_rms = np.sqrt(residual_min_rms ** 2 - residual_rms ** 2 + 1e-12)
        residual = residual + rng.normal(0, needed_rms, len(residual))

    return residual, s_direct
