"""Evaluator — computes SNR, CWER, and generates comparison plots."""

from typing import Dict

import numpy as np


class Evaluator:
    """Computes evaluation metrics and generates comparison figures."""

    def __init__(self, fs: int = 16000):
        self.fs = fs

    def evaluate(self, ref_signal: np.ndarray, spy_signal: np.ndarray,
                 enhanced_signal: np.ndarray, ref_text: str,
                 hyp_text: str) -> Dict:
        snr_raw = _compute_snr(ref_signal, spy_signal)
        snr_enhanced = _compute_snr(ref_signal, enhanced_signal)
        cwer_raw = _compute_cwer(ref_text, hyp_text)  # TODO: ASR on raw recording
        cwer_enhanced = _compute_cwer(ref_text, hyp_text)
        return {
            "snr_raw": snr_raw,
            "snr_enhanced": snr_enhanced,
            "cwer_raw": cwer_raw,
            "cwer_enhanced": cwer_enhanced,
        }


def _compute_snr(ref: np.ndarray, est: np.ndarray) -> float:
    noise = ref[:len(est)] - est[:len(ref)]
    p_signal = np.sum(ref[:len(est)] ** 2)
    p_noise = np.sum(noise ** 2)
    if p_noise < 1e-12:
        return 100.0
    return float(10.0 * np.log10(p_signal / p_noise))


def _compute_cwer(ref_text: str, hyp_text: str) -> float:
    """Cooperative Word Error Rate as percentage (0–100)."""
    if not ref_text.strip():
        return 100.0
    ref_words = ref_text.strip().split()
    hyp_words = hyp_text.strip().split()
    # Levenshtein distance at word level
    m, n = len(ref_words), len(hyp_words)
    d = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    cwer = (float(d[m][n]) / m) * 100.0
    return min(cwer, 100.0)
