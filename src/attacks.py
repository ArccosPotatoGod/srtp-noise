"""Adversarial denoising attack methods.

Three eavesdropper countermeasure strategies (Section 5):
1. Bandstop/bandpass filtering — frequency-domain suppression.
2. FastICA — blind source separation with dual-channel input.
3. Delay-and-sum beamforming — spatial filtering with mic array.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def apply_bandstop(y, fs, f_low=300, f_high=3500):
    """Apply Butterworth bandstop filter to suppress known jamming band.

    Parameters
    ----------
    y : np.ndarray
        Noisy signal.
    fs : int
        Sample rate.
    f_low : float
        Lower cutoff frequency (Hz).
    f_high : float
        Upper cutoff frequency (Hz).

    Returns
    -------
    filtered : np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    low = f_low / nyq
    high = f_high / nyq
    b, a = butter(4, [low, high], btype='bandstop')
    return filtfilt(b, a, y)


def apply_bandpass(y, fs, f_low=300, f_high=3400):
    """Apply Butterworth bandpass filter to keep speech band only.

    Useful when noise is concentrated outside speech frequencies.

    Parameters
    ----------
    y : np.ndarray
        Noisy signal.
    fs : int
        Sample rate.
    f_low : float
        Lower cutoff frequency (Hz).
    f_high : float
        Upper cutoff frequency (Hz).

    Returns
    -------
    filtered : np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    low = f_low / nyq
    high = f_high / nyq
    b, a = butter(4, [low, high], btype='bandpass')
    return filtfilt(b, a, y)


def ica_denoise(mix1, mix2):
    """Blind source separation via FastICA.

    Requires dual-channel input (two microphone positions or
    two recordings with different spatial signatures).

    Parameters
    ----------
    mix1 : np.ndarray
        First mixed channel.
    mix2 : np.ndarray
        Second mixed channel.

    Returns
    -------
    S_ : np.ndarray, shape (n_samples, 2)
        Estimated source components (columns 0 and 1).
    """
    from sklearn.decomposition import FastICA

    X = np.c_[mix1, mix2]
    ica = FastICA(n_components=2, max_iter=2000, random_state=0, tol=1e-4)
    S_ = ica.fit_transform(X)
    return S_


def select_speech_component(S_, s_original):
    """Select the ICA component most correlated with original speech.

    Parameters
    ----------
    S_ : np.ndarray, shape (n_samples, 2)
        ICA-separated components.
    s_original : np.ndarray
        Original clean speech for correlation reference.

    Returns
    -------
    best : np.ndarray
        Component with highest absolute correlation to s_original.
    """
    min_len = min(len(S_), len(s_original))
    corr0 = np.corrcoef(S_[:min_len, 0], s_original[:min_len])[0, 1]
    corr1 = np.corrcoef(S_[:min_len, 1], s_original[:min_len])[0, 1]
    idx = 0 if abs(corr0) >= abs(corr1) else 1
    return S_[:min_len, idx]


def delay_and_sum(multi_channel_signals, delays):
    """Delay-and-sum beamforming for a linear microphone array.

    Aligns signals from each mic by applying compensating delays,
    then averages. Suppresses off-axis interference while preserving
    on-axis speech.

    Simulates 4-mic linear array with 0.05m spacing (per spec).

    Parameters
    ----------
    multi_channel_signals : np.ndarray, shape (n_mics, n_samples)
        Input from each microphone.
    delays : list of int
        Sample delays to align each channel (negative = advance).

    Returns
    -------
    output : np.ndarray, shape (n_samples,)
        Beamformed output.
    """
    n_mics = multi_channel_signals.shape[0]
    aligned = np.zeros_like(multi_channel_signals)
    for i in range(n_mics):
        aligned[i] = np.roll(multi_channel_signals[i], delays[i])
    return np.mean(aligned, axis=0)
