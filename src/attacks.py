"""Adversarial denoising attack methods.

Three eavesdropper countermeasure strategies:
1. Bandstop filtering — remove energy in known jamming bands.
2. FastICA — blind source separation (needs 2-channel input).
3. Delay-and-sum beamforming — spatial filtering with mic array.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def apply_bandstop(y, fs, f_low=300, f_high=3500):
    """Apply Butterworth bandstop filter to remove narrowband interference.

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


def ica_denoise(mix1, mix2, s_original=None):
    """Blind source separation via FastICA.

    Requires dual-channel input (two microphone positions).
    Returns the component with highest correlation to original speech.

    Parameters
    ----------
    mix1 : np.ndarray
        First mixed channel.
    mix2 : np.ndarray
        Second mixed channel.
    s_original : np.ndarray or None
        Original clean speech for component selection.
        If None, returns both components.

    Returns
    -------
    S_ : np.ndarray, shape (n_samples, 2)
        Estimated source components (column 0 and column 1).
    """
    from sklearn.decomposition import FastICA

    X = np.c_[mix1, mix2]
    ica = FastICA(n_components=2, max_iter=1000, random_state=0)
    S_ = ica.fit_transform(X)
    return S_


def select_speech_component(S_, s_original):
    """Select the ICA component most correlated with original speech.

    Parameters
    ----------
    S_ : np.ndarray, shape (n_samples, 2)
        ICA-separated components.
    s_original : np.ndarray
        Original clean speech.

    Returns
    -------
    best : np.ndarray
        Component with highest correlation to s_original.
    """
    min_len = min(len(S_), len(s_original))
    corr0 = np.corrcoef(S_[:min_len, 0], s_original[:min_len])[0, 1]
    corr1 = np.corrcoef(S_[:min_len, 1], s_original[:min_len])[0, 1]
    return S_[:min_len, 0] if abs(corr0) > abs(corr1) else S_[:min_len, 1]


def delay_and_sum(multi_channel_signals, delays):
    """Delay-and-sum beamforming.

    Aligns signals from a linear microphone array by applying
    compensating delays, then averages them. Suppresses off-axis
    interference while preserving on-axis speech.

    Parameters
    ----------
    multi_channel_signals : np.ndarray, shape (n_mics, n_samples)
        Input from each microphone.
    delays : list of int
        Sample delays to align each channel.

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
