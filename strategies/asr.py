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
        # Placeholder — requires google-cloud-speech and credentials
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)
        try:
            from google.cloud import speech
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.language_code,
            )
            audio_obj = speech.RecognitionAudio(content=buf.read())
            response = client.recognize(config=config, audio=audio_obj)
            return " ".join(
                result.alternatives[0].transcript for result in response.results
            ).strip()
        except Exception:
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


def create_asr(name: str, params: dict) -> IASREngine:
    if name == "whisper_tiny":
        return WhisperASR(model_size="tiny")
    elif name == "whisper_base":
        return WhisperASR(model_size="base")
    elif name == "google_stt":
        return GoogleSTT(language_code=params.get("language_code", "en-US"))
    elif name == "dummy":
        return DummyASR(ref_text=params.get("ref_text", ""))
    raise ValueError(f"Unknown ASR engine: {name}")
