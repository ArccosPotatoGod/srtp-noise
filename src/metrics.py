"""Evaluation metrics for speech privacy protection.

Computes:
- SNR (Signal-to-Noise Ratio)
- WER (Word Error Rate) via jiwer
- MFCC distance (approximate speech quality without ASR)
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
        Noisy/corrupted speech.

    Returns
    -------
    snr_db : float
        SNR in decibels (inf if noise_power == 0).
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
    Used as a light-weight proxy for WER when ASR is unavailable.

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
    s_c = s_clean[:min_len].astype(np.float64)
    s_n = s_noisy[:min_len].astype(np.float64)

    mfcc_clean = librosa.feature.mfcc(y=s_c, sr=fs, n_mfcc=n_mfcc)
    mfcc_noisy = librosa.feature.mfcc(y=s_n, sr=fs, n_mfcc=n_mfcc)

    diff = mfcc_clean - mfcc_noisy
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


def compute_wer(reference_text, hypothesis_text):
    """Compute Word Error Rate using jiwer.

    Parameters
    ----------
    reference_text : str
        Ground-truth transcription.
    hypothesis_text : str
        ASR output / recovered text.

    Returns
    -------
    wer : float
        Word Error Rate (0.0 = perfect, higher = worse).
    """
    from jiwer import wer
    return wer(reference_text, hypothesis_text)


def compute_cer(reference_text, hypothesis_text):
    """Compute Character Error Rate using jiwer.

    Parameters
    ----------
    reference_text : str
        Ground-truth transcription.
    hypothesis_text : str
        ASR output / recovered text.

    Returns
    -------
    cer : float
        Character Error Rate.
    """
    from jiwer import cer
    return cer(reference_text, hypothesis_text)


def compute_segment_snr(s, residual, frame_length=512, hop_length=256):
    """Compute segmental SNR averaged over short frames.

    More perceptually relevant than full-signal SNR.

    Parameters
    ----------
    s : np.ndarray
        Clean speech.
    residual : np.ndarray
        Processed signal.
    frame_length : int
        Frame length in samples.
    hop_length : int
        Hop length in samples.

    Returns
    -------
    seg_snr : float
        Average segmental SNR (dB), frames below -10 dB clipped to -10.
    """
    min_len = min(len(s), len(residual))
    s = s[:min_len]
    r = residual[:min_len]

    snr_frames = []
    for start in range(0, min_len - frame_length, hop_length):
        s_frame = s[start:start + frame_length]
        r_frame = r[start:start + frame_length]
        sig_pow = np.sum(s_frame ** 2)
        noise_pow = np.sum((r_frame - s_frame) ** 2)
        if noise_pow == 0:
            snr = 35.0
        else:
            snr = 10.0 * np.log10(sig_pow / noise_pow)
        snr_frames.append(max(snr, -10.0))

    return float(np.mean(snr_frames))
