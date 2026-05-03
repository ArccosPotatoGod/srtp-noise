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

    # Inverted replica
    anti_signal = -s_ref

    # Cancelling signal emitted from jammer to spy
    s_cancel = propagate_signal(anti_signal, jammer_pos, spy_pos, fs)

    # Direct path from source to spy
    s_direct = propagate_signal(s, src_pos, spy_pos, fs)

    # Superposition (anti_signal is already inverted)
    residual = s_direct + s_cancel

    return residual, s_direct
