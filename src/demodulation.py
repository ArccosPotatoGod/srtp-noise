"""Ultrasonic nonlinear demodulation.

Simulates the nonlinear response of microphones to ultrasound,
which produces audible-frequency interference components.
Based on Equation (1) from the paper (Gao et al., 2023).
"""

import numpy as np


def nonlinear_demod(noise_baseband):
    """Apply nonlinear demodulation to baseband noise.

    Implements N(t) = A2 * (n(t) + 0.5 * n(t)^2) with A2 = 1.
    This models the microphone's nonlinearity converting
    ultrasonic signals into audible-band interference.

    Parameters
    ----------
    noise_baseband : np.ndarray
        Baseband noise signal.

    Returns
    -------
    demodulated : np.ndarray
        Demodulated audible interference signal.
    """
    return noise_baseband + 0.5 * noise_baseband ** 2
