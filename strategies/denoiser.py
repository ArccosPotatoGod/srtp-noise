"""Denoiser strategies — adversarial countermeasures (Sec.3).

Implements the four paper-evaluated methods plus a no-op pass-through:
1. FastICA (BSS in time domain)
2. SpectralSubtraction (frequency domain)
3. BandstopFilter (frequency domain)
4. DelaySumBeamformer (spatial domain)
5. AdaptiveNoiseFilter (sniffer-assisted NLMS)
6. NoDenoiser (raw pass-through)
"""

import numpy as np
from src.attacker import IDenoiser


class ICADenoiser(IDenoiser):
    """FastICA-based blind source separation (multi-channel).

    Uses speech-band energy ratio (300-3400 Hz) to identify the speech
    component among ICA sources.  For independent noise (Gaussian baseline)
    ICA separates speech from noise; for coherent noise (MicFrozen) the
    noise is coupled to speech and ICA cannot separate — matching the
    paper's key claim.
    """

    def __init__(self, n_components: int = None, fs: int = 16000):
        self.n_components = n_components
        self.fs = fs

    def _speech_score(self, component: np.ndarray) -> float:
        """Short-time energy variance — speech has alternating loud/quiet frames,
        while steady noise (Gaussian, etc.) has uniform energy across time."""
        frame_len = int(0.025 * self.fs)  # 25 ms
        n_frames = max(len(component) // frame_len, 4)
        frame_len = len(component) // n_frames
        frame_rms = np.array([
            np.sqrt(np.mean(component[i * frame_len:(i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ])
        mean_rms = np.mean(frame_rms) + 1e-12
        return float(np.var(frame_rms) / mean_rms)

    def denoise(self, audio: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        from sklearn.decomposition import FastICA
        if audio.ndim < 2 or audio.shape[0] < 2:
            return audio  # need >= 2 channels
        n_comp = self.n_components or audio.shape[0]
        ica = FastICA(n_components=n_comp, random_state=0, max_iter=2000)
        S = ica.fit_transform(audio.T)  # (samples, components)
        n_actual = S.shape[1]  # FastICA may reduce n_components to n_channels
        scores = [self._speech_score(S[:, i]) for i in range(n_actual)]
        speech_idx = int(np.argmax(scores))
        return S[:, speech_idx].astype(np.float32)


class SpectralSubtraction(IDenoiser):
    """Spectral subtraction — estimate noise spectrum from sniffer reference.

    When a noise_ref is provided (sniffer signal), its spectrum is used as the
    noise estimate.  Otherwise falls back to the first noise_frames of the input.
    """

    def __init__(self, noise_frames: int = 10, alpha: float = 1.2):
        self.noise_frames = noise_frames
        self.alpha = alpha

    def denoise(self, audio: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio[0]
        n_fft = 512
        hop = n_fft // 2
        pad = n_fft  # eliminate edge artifacts from overlap-add boundaries

        audio_orig_len = len(audio)
        audio_fp = audio.astype(np.float64)
        audio_pad = np.pad(audio_fp, (pad, pad + hop), mode="reflect")

        # Use sniffer reference for noise estimate when available
        if noise_ref is not None:
            if noise_ref.ndim > 1:
                noise_ref = noise_ref[0]
            noise_ref_fp = noise_ref.astype(np.float64)
            noise_ref_pad = np.pad(noise_ref_fp, (pad, pad + hop), mode="reflect")
            min_len = min(len(audio_pad), len(noise_ref_pad))
            noise_stft = np.array([np.fft.rfft(noise_ref_pad[i:i + n_fft] * np.hanning(n_fft))
                                   for i in range(0, min_len - n_fft, hop)])
            noise_mag = np.mean(np.abs(noise_stft), axis=0)
        else:
            stft_fallback = np.array([np.fft.rfft(audio_pad[i:i + n_fft] * np.hanning(n_fft))
                                      for i in range(0, len(audio_pad) - n_fft, hop)])
            noise_mag = np.mean(np.abs(stft_fallback[:self.noise_frames]), axis=0)

        stft = np.array([np.fft.rfft(audio_pad[i:i + n_fft] * np.hanning(n_fft))
                         for i in range(0, len(audio_pad) - n_fft, hop)])
        mag = np.abs(stft) - self.alpha * noise_mag
        mag = np.maximum(mag, 0.0)
        phase = np.angle(stft)
        enhanced_stft = mag * np.exp(1j * phase)
        # Overlap-add reconstruction on padded signal
        out = np.zeros(len(audio_pad), dtype=np.float64)
        win = np.hanning(n_fft)
        for idx, frame in enumerate(enhanced_stft):
            i = idx * hop
            out[i:i + n_fft] += np.fft.irfft(frame) * win
        norm = np.zeros_like(out)
        for idx in range(len(enhanced_stft)):
            i = idx * hop
            norm[i:i + n_fft] += win ** 2
        out = np.divide(out, norm, where=norm > 1e-12)
        # Trim padding to recover original-length signal
        out = out[pad:pad + audio_orig_len]
        return out.astype(np.float32)


class BandstopFilter(IDenoiser):
    """Adaptive bandstop / multi-notch filter for jamming suppression.

    Uses a narrow notch (default 1000-2000 Hz) to suppress jammer energy
    while preserving most of the speech band.  In the equivalent-baseband
    simulation there is no ultrasonic carrier, so wide bandstop would
    also remove speech.  A narrow notch provides a meaningful trade-off.
    """

    def __init__(self, f_low: float = 1000.0, f_high: float = 2000.0,
                 fs: int = 16000, order: int = 4):
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.order = order

    def denoise(self, audio: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        if audio.ndim > 1:
            audio = audio[0]
        nyq = 0.5 * self.fs
        low = self.f_low / nyq
        high = self.f_high / nyq
        b, a = butter(self.order, [low, high], btype="bandstop")
        return filtfilt(b, a, audio).astype(np.float32)


class DelaySumBeamformer(IDenoiser):
    """Delay-and-sum beamforming (multi-channel, TDOA-based)."""

    def __init__(self, target_angle: float = 0.0, fs: int = 16000,
                 mic_spacing: float = 0.15, speed_of_sound: float = 343.0):
        self.target_angle = target_angle
        self.fs = fs
        self.mic_spacing = mic_spacing
        self.speed_of_sound = speed_of_sound

    def denoise(self, audio: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        if audio.ndim < 2 or audio.shape[0] < 2:
            return audio
        n_channels, n_samples = audio.shape
        delays = np.arange(n_channels) * (self.mic_spacing
                     * np.sin(np.radians(self.target_angle))
                     / self.speed_of_sound * self.fs)
        out = np.zeros(n_samples)
        for i, delay in enumerate(delays):
            shift = int(round(delay))
            if shift >= 0:
                out += np.pad(audio[i], (shift, 0))[:n_samples]
            else:
                out += np.pad(audio[i], (0, -shift))[-n_samples:]
        return (out / n_channels).astype(np.float32)


class AdaptiveNoiseFilter(IDenoiser):
    """Sniffer-assisted NLMS adaptive noise filter (paper [32])."""

    def __init__(self, n_taps: int = 256, mu: float = 0.001, fs: int = 16000):
        self.n_taps = n_taps
        self.mu = mu
        self.fs = fs

    def denoise(self, audio: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio[0]
        T = len(audio)
        w = np.zeros(self.n_taps, dtype=np.float32)
        out = np.zeros(T, dtype=np.float32)
        delta = 1e-4
        # Use sniffer reference if available; otherwise fall back to self-reference.
        ref = noise_ref if noise_ref is not None else audio
        if ref.ndim > 1:
            ref = ref[0]
        for n in range(self.n_taps, T):
            x = ref[n - self.n_taps:n][::-1]
            y = np.dot(w, x)
            e = audio[n] - y
            norm = np.dot(x, x) + delta
            w += self.mu / norm * e * x
            out[n] = e
        return out


class NoDenoiser(IDenoiser):
    """Pass-through — returns raw recording unchanged."""

    def denoise(self, audio: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        if audio.ndim > 1:
            return audio[0].astype(np.float32)
        return audio.astype(np.float32)


def create_denoiser(name: str, params: dict) -> IDenoiser:
    if name == "ica":
        return ICADenoiser(n_components=params.get("n_channels"),
                          fs=params.get("fs", 16000))
    elif name == "spectral_subtraction":
        return SpectralSubtraction(
            noise_frames=params.get("noise_frames", 10),
            alpha=params.get("alpha", 2.0),
        )
    elif name == "bandstop":
        return BandstopFilter(
            f_low=params.get("f_low", 300.0),
            f_high=params.get("f_high", 3500.0),
            fs=params.get("fs", 16000),
        )
    elif name == "beamforming":
        return DelaySumBeamformer(
            target_angle=params.get("target_angle", 0.0),
            fs=params.get("fs", 16000),
        )
    elif name == "adaptive_noise_filter":
        return AdaptiveNoiseFilter(
            n_taps=params.get("n_taps", 64),
            mu=params.get("mu", 0.005),
        )
    elif name == "none":
        return NoDenoiser()
    raise ValueError(f"Unknown denoiser: {name}")
