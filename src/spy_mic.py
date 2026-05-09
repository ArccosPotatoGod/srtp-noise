"""SpyMicrophoneModule — simulates the eavesdropper's microphone.

Splits signal paths: audible speech arrives linearly; ultrasonic jamming
signals are demodulated via the nonlinearity model before summation.
"""

from typing import Dict, List

import numpy as np
from scipy.signal import fftconvolve


def build_sniffer_reference(s_cancel: np.ndarray, n_coherent: np.ndarray,
                            ultrasonic_rir: np.ndarray, signal_len: int,
                            nonlinearity: "INonlinearityModel") -> np.ndarray:
    """Construct the sniffer reference signal for adaptive noise filtering.

    Simulates an ultrasonic sniffer co-located with the spy mic:
    jammer baseband → ultrasonic RIR → nonlinear demodulation.

    Returns the demodulated baseband as the noise reference for NLMS ANF.
    """
    jammer_baseband = s_cancel + n_coherent
    ref = fftconvolve(jammer_baseband, ultrasonic_rir)[:signal_len]
    return nonlinearity.apply(ref)


class SpyMicrophoneModule:
    """Simulates a spy microphone with separated audible/ultrasonic paths."""

    def __init__(self, audible_rirs: Dict, ultrasonic_rirs: Dict,
                 nonlinearity: "INonlinearityModel"):
        self.rir_src_to_spy: List[np.ndarray] = audible_rirs["src_to_spy"]
        self.rir_jammer_to_spy: List[np.ndarray] = ultrasonic_rirs["jammer_to_spy"]
        self.nonlinearity = nonlinearity
        self.n_channels = len(self.rir_src_to_spy)

    def capture(self, s_src: np.ndarray, s_cancel: np.ndarray,
                n_coherent: np.ndarray) -> np.ndarray:
        """Produce the spy microphone recording(s).

        Returns
        -------
        recording : np.ndarray, shape=(n_channels, samples) or (samples,)
        """
        T = len(s_src)
        jammer_baseband = s_cancel + n_coherent  # single transducer
        recordings = []

        for ch in range(self.n_channels):
            # Audible path — linear
            spy_speech = fftconvolve(s_src, self.rir_src_to_spy[ch])[:T]
            # Ultrasonic path — nonlinear demodulation
            ultrasonic_arrival = fftconvolve(jammer_baseband, self.rir_jammer_to_spy[ch])[:T]
            # jammer_demod = self.nonlinearity.apply(ultrasonic_arrival)
            jammer_demod = ultrasonic_arrival
            recordings.append(spy_speech + jammer_demod)

        out = np.stack(recordings).astype(np.float32)
        if self.n_channels == 1:
            out = out.squeeze(0)
        return out
