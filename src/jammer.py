"""JammerModule — generates anti-speech cancel signal and coherent noise."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from scipy.signal import fftconvolve


class JammerModule:
    """Produces two baseband jamming signals: s_cancel and n_coherent."""

    def __init__(self, config, cancel_strategy: "ICancelingStrategy",
                 coherent_strategy: "ICoherentNoiseStrategy"):
        self.config = config
        self.cancel_strategy = cancel_strategy
        self.coherent_strategy = coherent_strategy

    def generate(self, s_src: np.ndarray, ref_rir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ref = fftconvolve(s_src, ref_rir)[:len(s_src)]
        # Normalize ref to same RMS as s_src (compensates near-field RIR gain)
        rms_src = float(np.sqrt(np.mean(s_src ** 2))) or 1.0
        rms_ref = float(np.sqrt(np.mean(ref ** 2))) or 1.0
        ref = ref * (rms_src / rms_ref)
        s_cancel = self.cancel_strategy.compute(ref)
        n_coherent = self.coherent_strategy.compute(s_src)
        gain_linear = 10.0 ** (self.config.jammer.system_gain_db / 20.0)
        s_cancel *= gain_linear
        n_coherent *= gain_linear
        return s_cancel.astype(np.float32), n_coherent.astype(np.float32)


class ICancelingStrategy(ABC):
    @abstractmethod
    def compute(self, ref_signal: np.ndarray) -> np.ndarray:
        pass


class ICoherentNoiseStrategy(ABC):
    @abstractmethod
    def compute(self, speech: np.ndarray) -> np.ndarray:
        pass
