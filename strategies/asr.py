"""ASR engine strategies — speech-to-text transcription."""

import numpy as np
from src.attacker import IASREngine


class WhisperASR(IASREngine):
    """OpenAI Whisper model (local, offline)."""

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        import whisper
        self._model = whisper.load_model(self.model_size)

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            self._load_model()
        audio_fp32 = audio.astype(np.float32)
        if audio_fp32.ndim > 1:
            audio_fp32 = audio_fp32[0]
        result = self._model.transcribe(audio_fp32)
        return result["text"].strip()


class GoogleSTT(IASREngine):
    """Google Cloud Speech-to-Text API (requires network & credentials)."""

    def __init__(self, language_code: str = "en-US"):
        self.language_code = language_code

    def transcribe(self, audio: np.ndarray) -> str:
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)
        try:
            from google.cloud import speech
        except ImportError:
            print("GoogleSTT: google-cloud-speech not installed. "
                  "Install with: pip install google-cloud-speech")
            return ""
        try:
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.language_code,
            )
            audio_obj = speech.RecognitionAudio(content=buf.read())
            response = client.recognize(config=config, audio=audio_obj)
            if not response.results:
                return ""
            return " ".join(
                result.alternatives[0].transcript for result in response.results
            ).strip()
        except Exception as exc:
            print(f"GoogleSTT: transcription failed — {exc}")
            return ""


class DummyASR(IASREngine):
    """For pipeline testing: reads audio energy to simulate ASR behavior.

    If audio has significant energy (SNR > threshold), returns the
    reference text. Otherwise returns empty string, simulating
    failed recognition.
    """

    def __init__(self, ref_text: str = ""):
        self.ref_text = ref_text
        self._energy_threshold = 0.01

    def transcribe(self, audio: np.ndarray) -> str:
        # Simulate ASR: if signal is strong enough, "recognize" reference
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms > self._energy_threshold:
            return self.ref_text
        return ""


class QualityASR(IASREngine):
    """SNR-driven simulated ASR — maps signal quality to recognition accuracy.

    Uses short-time energy variance (speech has clear onsets/offsets, noise
    is steady) as a proxy for speech quality, then maps to CWER via a
    logistic curve calibrated to typical ASR performance.
    """

    def __init__(self, ref_text: str = "", fs: int = 16000,
                 center_snr: float = 0.0, width: float = 6.0):
        self.ref_text = ref_text
        self.fs = fs
        self.center_snr = center_snr
        self.width = width

    def _estimate_quality(self, audio: np.ndarray) -> float:
        """Returns a pseudo-SNR estimate from unsupervised features.

        High value = likely intelligible speech; low value = likely noise.
        Uses the variance of short-time energy (speech has high variance).
        """
        sig = audio.astype(np.float64)
        if sig.ndim > 1:
            sig = sig[0]
        frame_len = int(0.025 * self.fs)
        n_frames = max(len(sig) // frame_len, 4)
        frame_len = len(sig) // n_frames
        rms = np.array([
            np.sqrt(np.mean(sig[i * frame_len:(i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ])
        mean_rms = np.mean(rms) + 1e-12
        # Normalized variance + spectral flatness proxy
        energy_var = float(np.std(rms) / mean_rms)
        # Map to pseudo-SNR: typical speech has var/mean ~0.3-1.0
        pseudo_snr = 20.0 * np.log10(energy_var + 0.01) + 15.0
        return float(np.clip(pseudo_snr, -30.0, 40.0))

    def _pseudo_snr_to_cwer(self, pseudo_snr: float) -> float:
        """Logistic mapping from pseudo-SNR to CWER (0–100%)."""
        return 100.0 / (1.0 + np.exp((pseudo_snr - self.center_snr) / self.width))

    def _corrupt_text(self, text: str, cwer: float, rng: np.random.Generator) -> str:
        """Randomly drop/replace words to achieve target CWER."""
        if not text.strip():
            return ""
        words = text.strip().split()
        n_err = max(1, int(round(cwer / 100.0 * len(words))))
        indices = rng.choice(len(words), size=min(n_err, len(words)), replace=False)
        result = []
        for i, w in enumerate(words):
            if i in indices:
                result.append("[???]")
            else:
                result.append(w)
        return " ".join(result)

    def transcribe(self, audio: np.ndarray) -> str:
        quality = self._estimate_quality(audio)
        cwer = self._pseudo_snr_to_cwer(quality)
        rng = np.random.default_rng(
            int(np.sum(audio.astype(np.float64))) & 0x7FFFFFFF
        )
        return self._corrupt_text(self.ref_text, cwer, rng)


def create_asr(name: str, params: dict) -> IASREngine:
    if name == "whisper_tiny":
        return WhisperASR(model_size="tiny")
    elif name == "whisper_base":
        return WhisperASR(model_size="base")
    elif name == "google_stt":
        return GoogleSTT(language_code=params.get("language_code", "en-US"))
    elif name == "dummy":
        return DummyASR(ref_text=params.get("ref_text", ""))
    elif name == "quality":
        return QualityASR(
            ref_text=params.get("ref_text", ""),
            fs=params.get("fs", 16000),
            center_snr=params.get("center_snr", 0.0),
            width=params.get("width", 6.0),
        )
    raise ValueError(f"Unknown ASR engine: {name}")
