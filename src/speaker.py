"""SpeakerModule — loads speech signal and reference text."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class SpeakerModule:
    """Loads a speech file and provides normalized signal plus reference text."""

    def __init__(self, audio_path: str, fs: int = 16000, ref_text: Optional[str] = None):
        self.audio_path = Path(audio_path)
        self.fs = fs
        self._ref_text = ref_text
        self._signal: Optional[np.ndarray] = None

    def load(self) -> None:
        """Read the audio file into memory and normalize to [-1, 1]."""
        import soundfile as sf
        data, sr = sf.read(self.audio_path)
        if sr != self.fs:
            from scipy.signal import resample
            data = resample(data, int(len(data) * self.fs / sr))
        if data.ndim > 1:
            data = data[:, 0]
        peak = np.max(np.abs(data)) or 1.0
        self._signal = data.astype(np.float32) / peak

    def get_signal(self) -> np.ndarray:
        """Return the float32 speech signal, shape=(samples,)."""
        if self._signal is None:
            self.load()
        return self._signal

    def get_reference_text(self) -> str:
        """Return the ground-truth transcription for WER/CWER evaluation."""
        if self._ref_text is not None:
            return self._ref_text
        txt_path = self.audio_path.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text().strip()
        return ""


def load_speaker(audio_path: str, fs: int = 16000,
                 ref_text: Optional[str] = None) -> SpeakerModule:
    spk = SpeakerModule(audio_path, fs, ref_text)
    return spk
