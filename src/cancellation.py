"""Inverse-channel speech signal cancellation.

Simulates the MicFrozen core mechanism: the reference microphone
captures the user's speech, and the jammer emits an inverted replica
to cancel it at the eavesdropper's microphone.
"""

import numpy as np
from .propagation import propagate_signal, compute_distance


def apply_cancellation(s, src_pos, jammer_pos, spy_pos, fs):
    """Apply MicFrozen speech cancellation.

    1. Reference mic (at jammer) captures delayed speech.
    2. Inverted replica is emitted from jammer.
    3. Direct and cancelled signals superpose at spy microphone.

    Parameters
    ----------
    s : np.ndarray, shape (n_samples,)
        Source speech signal.
    src_pos : tuple (x, y)
        Position of the speaker (sound source).
    jammer_pos : tuple (x, y)
        Position of MicFrozen device / jammer.
    spy_pos : tuple (x, y)
        Position of eavesdropping microphone.
    fs : int
        Sample rate in Hz.

    Returns
    -------
    residual : np.ndarray
        Residual speech after cancellation.
    s_direct : np.ndarray
        Direct-path signal at spy (for comparison).
    """
    # Reference signal at jammer (approximates reference mic)
    s_ref = propagate_signal(s, src_pos, jammer_pos, fs)

    # Gain calibration: match cancel amplitude to direct amplitude at spy.
    # Without calibration, the ref mic's proximity (0.2m vs spy at 1-5m)
    # makes the cancel signal 3-5× louder than direct → over-compensation.
    d_ref = compute_distance(src_pos, jammer_pos)
    d_jam_to_spy = compute_distance(jammer_pos, spy_pos)
    d_src_to_spy = compute_distance(src_pos, spy_pos)

    atten_ref = 1.0 / (d_ref + 0.1)
    atten_direct = 1.0 / (d_src_to_spy + 0.1)
    atten_cancel_path = 1.0 / (d_jam_to_spy + 0.1)

    # Normalize so: |anti * atten_cancel_path| = |atten_direct|
    # anti = -s_ref * gain, where gain * atten_cancel_path = atten_direct
    gain = atten_direct / (atten_ref * atten_cancel_path + 1e-10)

    anti_signal = -s_ref * gain

    # Cancelling signal emitted from jammer to spy
    s_cancel = propagate_signal(anti_signal, jammer_pos, spy_pos, fs)

    # Direct path from source to spy
    s_direct = propagate_signal(s, src_pos, spy_pos, fs)

    # Superposition (anti_signal is already inverted)
    residual = s_direct + s_cancel

    return residual, s_direct
