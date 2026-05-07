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
    """FastICA-based blind source separation (multi-channel)."""

    def __init__(self, n_components: int = None):
        self.n_components = n_components

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import FastICA
        if audio.ndim < 2 or audio.shape[0] < 2:
            return audio  # need >= 2 channels
        n_comp = self.n_components or audio.shape[0]
        ica = FastICA(n_components=n_comp, random_state=0, max_iter=2000)
        S = ica.fit_transform(audio.T)  # (samples, components)
        # Pick component with lowest energy as "noise" — crude heuristic
        energies = np.sum(S ** 2, axis=0)
        speech_idx = np.argmax(energies)
        return S[:, speech_idx].astype(np.float32)


class SpectralSubtraction(IDenoiser):
    """Spectral subtraction — estimate noise spectrum and subtract."""

    def __init__(self, noise_frames: int = 10, alpha: float = 2.0):
        self.noise_frames = noise_frames
        self.alpha = alpha

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio[0]
        n_fft = 512
        hop = n_fft // 2
        stft = np.array([np.fft.rfft(audio[i:i + n_fft] * np.hanning(n_fft))
                         for i in range(0, len(audio) - n_fft, hop)])
        noise_mag = np.mean(np.abs(stft[:self.noise_frames]), axis=0)
        mag = np.abs(stft) - self.alpha * noise_mag
        mag = np.maximum(mag, 0.0)
        phase = np.angle(stft)
        enhanced_stft = mag * np.exp(1j * phase)
        # Overlap-add reconstruction
        out = np.zeros(len(audio))
        win = np.hanning(n_fft)
        for idx, frame in enumerate(enhanced_stft):
            i = idx * hop
            out[i:i + n_fft] += np.fft.irfft(frame) * win
        norm = np.zeros_like(out)
        for idx in range(len(enhanced_stft)):
            i = idx * hop
            norm[i:i + n_fft] += win ** 2
        out = np.divide(out, norm, where=norm > 1e-12)
        return out.astype(np.float32)


class BandstopFilter(IDenoiser):
    """Butterworth bandstop filter to suppress known jamming band."""

    def __init__(self, f_low: float = 300.0, f_high: float = 3500.0,
                 fs: int = 16000, order: int = 4):
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.order = order

    def denoise(self, audio: np.ndarray) -> np.ndarray:
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
                 mic_spacing: float = 0.05, speed_of_sound: float = 343.0):
        self.target_angle = target_angle
        self.fs = fs
        self.mic_spacing = mic_spacing
        self.speed_of_sound = speed_of_sound

    def denoise(self, audio: np.ndarray) -> np.ndarray:
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

    def __init__(self, n_taps: int = 64, mu: float = 0.005, fs: int = 16000):
        self.n_taps = n_taps
        self.mu = mu
        self.fs = fs

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio[0]
        T = len(audio)
        w = np.zeros(self.n_taps, dtype=np.float32)
        out = np.zeros(T, dtype=np.float32)
        delta = 1e-4
        # In practice the sniffer reference would be available as a second input.
        # Here we approximate via self-reference (will be improved with proper ref).
        for n in range(self.n_taps, T):
            x = audio[n - self.n_taps:n][::-1]
            y = np.dot(w, x)
            e = audio[n] - y
            norm = np.dot(x, x) + delta
            w += self.mu / norm * e * x
            out[n] = e
        return out


class NoDenoiser(IDenoiser):
    """Pass-through — returns raw recording unchanged."""

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            return audio[0].astype(np.float32)
        return audio.astype(np.float32)


def create_denoiser(name: str, params: dict) -> IDenoiser:
    if name == "ica":
        return ICADenoiser(n_components=params.get("n_components"))
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
