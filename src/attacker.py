"""AttackerModule — denoising and speech recognition on spy recordings."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AttackerModule:
    """Receives spy recording, applies denoising, transcribes."""

    def __init__(self, denoiser: "IDenoiser", asr: "IASREngine"):
        self.denoiser = denoiser
        self.asr = asr

    def attack(self, recording: np.ndarray) -> Tuple[np.ndarray, str]:
        enhanced = self.denoiser.denoise(recording)
        transcription = self.asr.transcribe(enhanced)
        return enhanced.astype(np.float32), transcription


class IDenoiser(ABC):
    @abstractmethod
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        pass


class IASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        pass
