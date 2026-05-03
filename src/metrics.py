"""Evaluation metrics for speech privacy protection.

Computes:
- SNR (Signal-to-Noise Ratio)
- WER approximation via MFCC distance
- CER-like character-level metrics
"""

import numpy as np


def compute_snr(signal_clean, signal_noisy):
    """Compute Signal-to-Noise Ratio in dB.

    SNR = 10 * log10(||s||^2 / ||s_noisy - s||^2)

    Parameters
    ----------
    signal_clean : np.ndarray
        Original clean speech.
    signal_noisy : np.ndarray
        Noisy/corrupted speech (same length).

    Returns
    -------
    snr_db : float
        SNR value in decibels.
    """
    min_len = min(len(signal_clean), len(signal_noisy))
    s = signal_clean[:min_len]
    n = signal_noisy[:min_len]

    signal_power = np.sum(s ** 2)
    noise_power = np.sum((n - s) ** 2)

    if noise_power == 0:
        return np.inf
    return 10.0 * np.log10(signal_power / noise_power)


def compute_mfcc_distance(s_clean, s_noisy, fs=16000, n_mfcc=13):
    """Approximate speech quality via MFCC Euclidean distance.

    Lower distance -> better preservation of speech features.

    Parameters
    ----------
    s_clean : np.ndarray
        Clean speech signal.
    s_noisy : np.ndarray
        Noisy/processed signal.
    fs : int
        Sample rate.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    distance : float
        Average MFCC Euclidean distance.
    """
    import librosa

    min_len = min(len(s_clean), len(s_noisy))
    s_c = s_clean[:min_len]
    s_n = s_noisy[:min_len]

    mfcc_clean = librosa.feature.mfcc(y=s_c.astype(np.float64), sr=fs, n_mfcc=n_mfcc)
    mfcc_noisy = librosa.feature.mfcc(y=s_n.astype(np.float64), sr=fs, n_mfcc=n_mfcc)

    diff = mfcc_clean - mfcc_noisy
    return np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
