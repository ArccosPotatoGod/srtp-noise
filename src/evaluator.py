"""Evaluator — computes SNR, CWER, and generates comparison plots."""

from pathlib import Path
from typing import Dict, List

import numpy as np


class Evaluator:
    """Computes evaluation metrics aligned with the paper (Sec.8.1)."""

    def __init__(self, fs: int = 16000):
        self.fs = fs

    def compute_snr(self, speech_at_spy: np.ndarray,
                    recording: np.ndarray) -> float:
        """Signal-to-jamming ratio at spy position.

        SNR = 10 * log10(||speech_at_spy||² / ||recording - speech_at_spy||²)
        """
        return _compute_snr(speech_at_spy, recording)

    def compute_cwer(self, ref_text: str, hyp_text: str) -> float:
        """Cooperative Word Error Rate as percentage."""
        return _compute_cwer(ref_text, hyp_text)

    def evaluate(self, speech_at_spy: np.ndarray,
                 spy_rec: np.ndarray, enhanced: np.ndarray,
                 ref_text: str, raw_hyp: str, enhanced_hyp: str) -> Dict:
        """Run full metric suite for one experiment."""
        return {
            "snr_raw": self.compute_snr(speech_at_spy, spy_rec),
            "snr_enhanced": self.compute_snr(speech_at_spy, enhanced),
            "cwer_raw": self.compute_cwer(ref_text, raw_hyp),
            "cwer_enhanced": self.compute_cwer(ref_text, enhanced_hyp),
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
    m, n = len(ref_words), len(hyp_words)
    d = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                         d[i - 1][j - 1] + cost)
    cwer = (float(d[m][n]) / m) * 100.0
    return min(cwer, 100.0)


def format_text_report(results: List[Dict], title: str = "MicFrozen Experiment Report") -> str:
    """Generate a comprehensive text report from experiment results."""
    if not results:
        return "No results."

    lines = []
    lines.append("=" * 72)
    lines.append(f"  {title}")
    lines.append("=" * 72)

    keys = list(results[0].keys())
    non_metric_keys = [k for k in keys
                       if not k.startswith("snr") and not k.startswith("cwer")]

    # Header
    header_cols = ["snr_raw", "snr_enhanced", "cwer_raw", "cwer_enhanced"]
    header_cols += non_metric_keys
    header_cols = [k for k in header_cols if k in keys]

    header = " | ".join(f"{k:<24}" for k in header_cols)
    sep = "-+-".join("-" * 24 for _ in header_cols)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for r in results:
        row_parts = []
        for k in header_cols:
            v = r.get(k, "")
            if isinstance(v, float):
                row_parts.append(f"{v:<24.4f}")
            else:
                row_parts.append(f"{str(v):<24}")
        lines.append(" | ".join(row_parts))

    lines.append(sep)

    # Summary statistics
    lines.append("")
    lines.append("--- Summary Statistics ---")
    for metric in ["snr_raw", "snr_enhanced", "cwer_raw", "cwer_enhanced"]:
        if metric in keys:
            vals = [r[metric] for r in results if isinstance(r.get(metric), (int, float))]
            if vals:
                lines.append(f"  {metric}: min={min(vals):.2f}  max={max(vals):.2f}  "
                           f"mean={np.mean(vals):.2f}  median={np.median(vals):.2f}")

    # Per-strategy grouping
    strategy_keys = [k for k in non_metric_keys if "strategy" in k.lower() or "denoiser" in k.lower()]
    if strategy_keys:
        lines.append("")
        lines.append("--- Per-Strategy Breakdown ---")
        for sk in strategy_keys:
            by_val: Dict[str, List[float]] = {}
            # Use snr_enhanced for denoiser grouping, snr_raw for jammer strategy
            metric_key = "snr_enhanced" if "denoiser" in sk.lower() else "snr_raw"
            for r in results:
                val = str(r.get(sk, "?"))
                by_val.setdefault(val, []).append(r.get(metric_key, 0))
            lines.append(f"  Grouped by {sk}:")
            for val, snrs in sorted(by_val.items()):
                lines.append(f"    {val:<24}  mean {metric_key}={np.mean(snrs):.2f} dB  "
                           f"(n={len(snrs)})")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def save_text_report(results: List[Dict], path: str,
                     title: str = "MicFrozen Experiment Report") -> None:
    """Write a comprehensive text report to a file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report = format_text_report(results, title)
    Path(path).write_text(report)
