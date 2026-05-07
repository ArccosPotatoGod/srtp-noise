"""AttackerModule — denoising and speech recognition on spy recordings."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AttackerModule:
    """Receives spy recording, applies denoising, transcribes."""

    def __init__(self, denoiser: "IDenoiser", asr: "IASREngine"):
        self.denoiser = denoiser
        self.asr = asr

    def attack(self, recording: np.ndarray,
               noise_ref: np.ndarray = None) -> Tuple[np.ndarray, str]:
        enhanced = self.denoiser.denoise(recording, noise_ref=noise_ref)
        transcription = self.asr.transcribe(enhanced)
        return enhanced.astype(np.float32), transcription


class IDenoiser(ABC):
    @abstractmethod
    def denoise(self, audio: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        pass


class IASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        pass
