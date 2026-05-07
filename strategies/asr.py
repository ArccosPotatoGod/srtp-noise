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
    """For pipeline testing: uses frame energy variance to simulate ASR.

    Speech has alternating loud/quiet frames → high energy variance.
    Steady noise (jamming) flattens the envelope → low energy variance.
    When variance ratio drops below threshold, recognition is treated
    as failed.
    """

    def __init__(self, ref_text: str = "", fs: int = 16000,
                 variance_threshold: float = 0.15):
        self.ref_text = ref_text
        self.fs = fs
        self._variance_threshold = variance_threshold

    def transcribe(self, audio: np.ndarray) -> str:
        sig = audio.astype(np.float64)
        if sig.ndim > 1:
            sig = sig[0]
        frame_len = int(0.025 * self.fs)
        n_frames = max(len(sig) // frame_len, 4)
        frame_len = len(sig) // n_frames
        frame_rms = np.array([
            np.sqrt(np.mean(sig[i * frame_len:(i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ])
        mean_rms = np.mean(frame_rms) + 1e-12
        normalized_variance = float(np.var(frame_rms) / mean_rms)
        if normalized_variance > self._variance_threshold:
            return self.ref_text
        return ""


class QualityASR(IASREngine):
    """SNR-driven simulated ASR — maps signal quality to recognition accuracy.

    Uses envelope dynamics (frame-to-frame energy variation) as the primary
    quality indicator.  Speech has alternating loud/quiet frames (dyn ≈ 0.3–0.7);
    steady noise flattens the envelope (dyn ≈ 0.01–0.1).  A secondary RMS
    elevation penalty handles cases where additive noise inflates overall power.
    """

    def __init__(self, ref_text: str = "", fs: int = 16000,
                 center_snr: float = 3.0, width: float = 4.0):
        self.ref_text = ref_text
        self.fs = fs
        self.center_snr = center_snr
        self.width = width

    def _estimate_quality(self, audio: np.ndarray) -> float:
        sig = audio.astype(np.float64)
        if sig.ndim > 1:
            sig = sig[0]

        # Frame-level envelope dynamics — primary quality indicator
        frame_len = int(0.025 * self.fs)
        n_frames = max(len(sig) // frame_len, 4)
        frame_len = len(sig) // n_frames
        frame_rms = np.array([
            np.sqrt(np.mean(sig[i * frame_len:(i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ])
        mean_frame_rms = np.mean(frame_rms) + 1e-12
        envelope_dynamics = float(np.std(frame_rms) / mean_frame_rms)

        # Dynamics-based quality: dyn ≈ 0.5 = clean, dyn ≈ 0.05 = noise-dominated
        quality_dyn = envelope_dynamics * 30.0 - 2.5

        # RMS elevation penalty (secondary): excessive power suggests additive noise
        rms_total = float(np.sqrt(np.mean(sig ** 2))) + 1e-12
        rms_elevation_db = 20.0 * np.log10(rms_total / 0.5 + 1e-12)
        quality_rms = -0.3 * max(0.0, rms_elevation_db - 3.0)

        pseudo_snr = quality_dyn + quality_rms
        return float(np.clip(pseudo_snr, -20.0, 25.0))

    def _pseudo_snr_to_cwer(self, pseudo_snr: float) -> float:
        """Logistic mapping from pseudo-SNR to CWER (0–100%)."""
        return 100.0 / (1.0 + np.exp((pseudo_snr - self.center_snr) / self.width))

    def _corrupt_text(self, text: str, cwer: float, rng: np.random.Generator) -> str:
        """Probabilistically corrupt each word to approximate target CWER.

        Each word is independently corrupted with probability cwer/100,
        avoiding the coarse quantization of a fixed error count.
        """
        if not text.strip():
            return ""
        words = text.strip().split()
        prob = cwer / 100.0
        result = []
        for w in words:
            if rng.random() < prob:
                result.append("[???]")
            else:
                result.append(w)
        # If cwer > 0 but nothing got corrupted, force at least one error
        if cwer > 0.0 and all(r == w for r, w in zip(result, words)):
            idx = rng.integers(0, len(words))
            result[idx] = "[???]"
        return " ".join(result)

    def transcribe(self, audio: np.ndarray, seed_offset: int = 0) -> str:
        quality = self._estimate_quality(audio)
        cwer = self._pseudo_snr_to_cwer(quality)
        # Deterministic seed from frame-level energy statistics (robust to DC offset)
        frame_len = int(0.025 * self.fs)
        n_frames = max(len(audio) // frame_len, 4)
        frame_len_actual = len(audio) // n_frames
        frame_energies = np.array([
            np.mean(audio[i * frame_len_actual:(i + 1) * frame_len_actual] ** 2)
            for i in range(n_frames)
        ])
        seed = (int(np.sum(frame_energies * 1e6)) + seed_offset) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)
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
            center_snr=params.get("center_snr", 5.0),
            width=params.get("width", 2.5),
        )
    raise ValueError(f"Unknown ASR engine: {name}")
