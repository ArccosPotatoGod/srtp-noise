"""Exporter — plots, audio, text reports, CSV from experiment results."""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

from src.evaluator import save_text_report as _save_text_report


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv(results: List[Dict], path: str) -> None:
    """Write results list to a CSV file."""
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = results[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def export_text_report(results: List[Dict], path: str,
                       title: str = "MicFrozen Experiment Report") -> None:
    """Write a comprehensive text report to a file."""
    _save_text_report(results, path, title)


# ---------------------------------------------------------------------------
# WAV audio export
# ---------------------------------------------------------------------------

def export_wav_stages(signals: Dict, output_dir: str, fs: int) -> None:
    """Export four audio stages as WAV files for listening comparison.

    signsals dict must contain:
        speech_at_spy, spy_noise_only, spy_full, enhanced
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = [
        ("01_source_speech.wav", signals.get("speech_at_spy")),
        ("02_noise_only.wav", signals.get("spy_noise_only")),
        ("03_noise_plus_cancel.wav", signals.get("spy_full")),
        ("04_enhanced.wav", signals.get("enhanced")),
    ]
    for fname, sig in items:
        if sig is not None:
            _save_wav(out / fname, sig, fs)


def _save_wav(path: Path, signal: np.ndarray, fs: int) -> None:
    """Save a float32 signal as 16-bit WAV, peak-normalized."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.95
    wavfile.write(str(path), fs, signal.astype(np.float32))


# ---------------------------------------------------------------------------
# Pipeline audio-stage plots (waveform + spectrogram)
# ---------------------------------------------------------------------------

def save_pipeline_audio_plots(signals: Dict, output_dir: str,
                              fs: int, denoiser_name: str = "unknown") -> None:
    """Save waveform/spectrogram plots for every pipeline stage.

    signals dict keys:
        s_src, ref_sig, s_cancel, n_coherent, audible_arrival,
        ultrasonic_arrival, ultrasonic_demod, spy_rec, enhanced,
        jammer_baseband, noise_ref
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stages = [
        ("01_source_speech", signals.get("s_src"), "Source speech s(t)"),
        ("02_reference_mic", signals.get("ref_sig"), "Reference mic (RIR convolved)"),
        ("03_cancel_signal", signals.get("s_cancel"), "Cancel signal s_cancel(t)"),
        ("04_coherent_noise", signals.get("n_coherent"), "Coherent noise n_coherent(t)"),
        ("05_audible_arrival", signals.get("audible_arrival"), "Audible arrival at spy (linear)"),
        ("06_ultrasonic_arrival", signals.get("ultrasonic_arrival"), "Ultrasonic arrival at spy"),
        ("07_ultrasonic_demod", signals.get("ultrasonic_demod"), "Ultrasonic arrival after nonlinear demod"),
        ("08_spy_recording", signals.get("spy_rec"), "Spy recording (audible + demod ultrasonic)"),
        ("09_enhanced", signals.get("enhanced"), f"After denoising ({denoiser_name})"),
        ("10_jammer_baseband", signals.get("jammer_baseband"), "Jammer baseband (cancel + noise)"),
        ("11_noise_reference", signals.get("noise_ref"), "Sniffer noise reference"),
    ]

    for fname, sig, title in stages:
        if sig is not None:
            _save_signal_plot(out, fname, sig, fs, title)

    _save_overview_plot(out,
                        signals.get("s_src"), signals.get("s_cancel"),
                        signals.get("n_coherent"), signals.get("spy_rec"),
                        signals.get("enhanced"), fs)
    _save_spectrogram_plot(out,
                           signals.get("s_src"), signals.get("spy_rec"),
                           signals.get("enhanced"), fs)
    print(f"Pipeline audio plots saved to {out}/")


def _save_signal_plot(out_dir, fname, signal, fs, title):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, signal, linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{fname}.png", dpi=150)
    plt.close(fig)


def _save_overview_plot(out_dir, s_src, s_cancel, n_coherent, spy_rec, enhanced, fs):
    if s_src is None:
        return
    t = np.arange(len(s_src)) / fs
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, s_src, linewidth=0.5, color="tab:green", label="Source speech")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Source Speech s(t)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    if s_cancel is not None and n_coherent is not None:
        axes[1].plot(t, s_cancel, linewidth=0.5, color="tab:red", alpha=0.7, label="s_cancel")
        axes[1].plot(t, n_coherent, linewidth=0.5, color="tab:purple", alpha=0.7, label="n_coherent")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Jammer Signals")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    if spy_rec is not None:
        axes[2].plot(t, spy_rec, linewidth=0.5, color="tab:orange", alpha=0.7, label="Spy recording")
    if enhanced is not None:
        axes[2].plot(t[:len(enhanced)], enhanced[:len(t)], linewidth=0.8, color="tab:blue",
                     alpha=0.8, label="Enhanced")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("Spy Recording vs Enhanced")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=150)
    plt.close(fig)


def _save_spectrogram_plot(out_dir, s_src, spy_rec, enhanced, fs):
    if s_src is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for ax, sig, title in [
        (axes[0], s_src, "Source Speech"),
        (axes[1], spy_rec, "Spy Recording (jammed)"),
        (axes[2], enhanced, "After Denoising"),
    ]:
        if sig is None:
            continue
        f, t_spec, Sxx = spectrogram(sig.astype(np.float64), fs, nperseg=512, noverlap=256)
        im = ax.pcolormesh(t_spec, f[:80], 10 * np.log10(Sxx[:80] + 1e-12),
                           shading="auto", cmap="inferno")
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="dB")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "spectrograms.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: build strategy+cancel label
# ---------------------------------------------------------------------------

def _strategy_cancel_label(r: Dict) -> str:
    strat = r.get("jammer.coherent_strategy", "?")
    cancel = r.get("jammer.canceling_strategy", "phase_inversion")
    if cancel == "off":
        return f"{strat} (no cancel)"
    return strat


# ---------------------------------------------------------------------------
# Plot: SNR vs Distance (raw recording)
# ---------------------------------------------------------------------------

def plot_snr_vs_distance(results: List[Dict], path: str) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in results:
        groups[_strategy_cancel_label(r)].append(
            (r.get("spy_mic_distance", 0), r.get("snr_raw", 0)))

    agg = defaultdict(list)
    for label, pts in groups.items():
        by_dist = defaultdict(list)
        for d, v in pts:
            by_dist[d].append(v)
        for d, vs in by_dist.items():
            agg[label].append((d, float(np.mean(vs))))

    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "D", "^", "v", "p", "h", "*"]
    for i, (label, pts) in enumerate(sorted(agg.items())):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        x = [p[0] for p in pts_sorted]
        y = [p[1] for p in pts_sorted]
        style = "--" if "no cancel" in label else "-"
        ax.plot(x, y, marker=markers[i % len(markers)], linestyle=style, label=label)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR vs Distance (raw recording)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: CWER vs Distance (raw recording)
# ---------------------------------------------------------------------------

def plot_cwer_vs_distance(results: List[Dict], path: str) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in results:
        groups[_strategy_cancel_label(r)].append(
            (r.get("spy_mic_distance", 0), r.get("cwer_raw", 0)))

    agg = defaultdict(list)
    for label, pts in groups.items():
        by_dist = defaultdict(list)
        for d, v in pts:
            by_dist[d].append(v)
        for d, vs in by_dist.items():
            agg[label].append((d, float(np.mean(vs))))

    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "D", "^", "v", "p", "h", "*"]
    for i, (label, pts) in enumerate(sorted(agg.items())):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        x = [p[0] for p in pts_sorted]
        y = [p[1] for p in pts_sorted]
        style = "--" if "no cancel" in label else "-"
        ax.plot(x, y, marker=markers[i % len(markers)], linestyle=style, label=label)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("CWER (%)")
    ax.set_title("CWER vs Distance (raw recording)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Enhanced SNR vs Distance (per strategy)
# ---------------------------------------------------------------------------

def plot_snr_enhanced_vs_distance(results: List[Dict], path: str,
                                  strategy: str = "fixed_weight") -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    subset = [r for r in results if r.get("jammer.coherent_strategy") == strategy]
    groups = defaultdict(list)
    for r in subset:
        den = r.get("attacker.denoiser", "?")
        cancel = r.get("jammer.canceling_strategy", "?")
        label = f"{den}" if cancel == "phase_inversion" else f"{den} (no cancel)"
        groups[label].append((r.get("spy_mic_distance", 0), r.get("snr_enhanced", 0)))

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v", "p", "h", "*"]
    for i, (label, pts) in enumerate(sorted(groups.items())):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        x = [p[0] for p in pts_sorted]
        y = [p[1] for p in pts_sorted]
        style = "--" if "no cancel" in label else "-"
        ax.plot(x, y, marker=markers[i % len(markers)], linestyle=style, label=label)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(f"SNR vs Distance — after denoising ({strategy})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Enhanced CWER vs Distance (per strategy)
# ---------------------------------------------------------------------------

def plot_cwer_enhanced_vs_distance(results: List[Dict], path: str,
                                   strategy: str = "fixed_weight") -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    subset = [r for r in results if r.get("jammer.coherent_strategy") == strategy]
    groups = defaultdict(list)
    for r in subset:
        den = r.get("attacker.denoiser", "?")
        cancel = r.get("jammer.canceling_strategy", "?")
        label = f"{den}" if cancel == "phase_inversion" else f"{den} (no cancel)"
        groups[label].append((r.get("spy_mic_distance", 0), r.get("cwer_enhanced", 0)))

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v", "p", "h", "*"]
    for i, (label, pts) in enumerate(sorted(groups.items())):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        x = [p[0] for p in pts_sorted]
        y = [p[1] for p in pts_sorted]
        style = "--" if "no cancel" in label else "-"
        ax.plot(x, y, marker=markers[i % len(markers)], linestyle=style, label=label)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("CWER (%)")
    ax.set_title(f"CWER vs Distance — after denoising ({strategy})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Denoiser comparison bar chart
# ---------------------------------------------------------------------------

def plot_denoiser_comparison(results: List[Dict], path: str,
                             distance: float = 1.0) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    subset = [r for r in results
              if abs(float(r.get("spy_mic_distance", 0)) - distance) < 0.01]
    if not subset:
        return

    strat_labels = sorted(set(
        f"{r['jammer.coherent_strategy']}"
        + ("" if r.get("jammer.canceling_strategy") == "phase_inversion"
           else " (no cancel)")
        for r in subset
    ))
    denoisers = sorted(set(r["attacker.denoiser"] for r in subset))

    x = np.arange(len(denoisers))
    width = 0.8 / max(len(strat_labels), 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, sl in enumerate(strat_labels):
        sl_parts = sl.split(" (no cancel)")
        strat = sl_parts[0]
        no_cancel = len(sl_parts) > 1
        strat_rows = [r for r in subset
                      if r["jammer.coherent_strategy"] == strat
                      and (no_cancel == (r.get("jammer.canceling_strategy") != "phase_inversion"))]
        den_map = {r["attacker.denoiser"]: r.get("cwer_enhanced", 0) for r in strat_rows}
        y_vals = [den_map.get(d, 0) for d in denoisers]
        bars = ax.bar(x + i * width, y_vals, width, label=sl)
        for bar, val in zip(bars, y_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6)

    ax.set_xlabel("Denoiser")
    ax.set_ylabel("CWER (%)")
    ax.set_title(f"Denoiser Comparison — cwer_enhanced at {distance:.0f} m")
    ax.set_xticks(x + width * (len(strat_labels) - 1) / 2)
    ax.set_xticklabels(denoisers, rotation=30, ha="right")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: CWER scatter (raw vs enhanced)
# ---------------------------------------------------------------------------

def plot_cwer_scatter(results: List[Dict], path: str) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    strategies = sorted(set(_strategy_cancel_label(r) for r in results))
    denoisers = sorted(set(r["attacker.denoiser"] for r in results))
    colors = plt.cm.tab20(np.linspace(0, 1, len(strategies)))
    markers = ["o", "s", "D", "^", "v", "p"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for si, strat in enumerate(strategies):
        for di, den in enumerate(denoisers):
            pts = [(r["cwer_raw"], r["cwer_enhanced"])
                   for r in results
                   if _strategy_cancel_label(r) == strat
                   and r.get("attacker.denoiser") == den]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=[colors[si]], marker=markers[di % len(markers)],
                       alpha=0.7, s=40)

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y = x (no change)")
    ax.set_xlabel("CWER raw (%)")
    ax.set_ylabel("CWER enhanced (%)")
    ax.set_title("CWER: Raw vs Enhanced")
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)

    from matplotlib.lines import Line2D
    strategy_handles = [Line2D([0], [0], color=colors[i], lw=2, label=s)
                        for i, s in enumerate(strategies)]
    denoiser_handles = [Line2D([0], [0], color="gray", marker=markers[i % len(markers)],
                               linestyle="none", label=d)
                        for i, d in enumerate(denoisers)]
    leg1 = ax.legend(handles=strategy_handles, title="Strategy", fontsize=7,
                     loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=denoiser_handles, title="Denoiser", fontsize=7,
              loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Scenario comparison (jammer-off / noise-only / noise+cancel)
# ---------------------------------------------------------------------------

def plot_scenario_comparison(results: List[Dict], path: str) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _scenario(r):
        if r.get("jammer.coherent_strategy") == "off":
            return "1. jammer off"
        if r.get("jammer.canceling_strategy") == "off":
            return "2. noise only (no cancel)"
        return "3. noise + cancel"

    scenarios = ["1. jammer off", "2. noise only (no cancel)", "3. noise + cancel"]
    colors = {"1. jammer off": "tab:green",
              "2. noise only (no cancel)": "tab:orange",
              "3. noise + cancel": "tab:red"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Raw SNR vs distance
    ax = axes[0][0]
    for sc in scenarios:
        by_dist = defaultdict(list)
        for r in results:
            if _scenario(r) == sc:
                by_dist[r["spy_mic_distance"]].append(r["snr_raw"])
        if not by_dist:
            continue
        dists = sorted(by_dist.keys())
        vals = [np.mean(by_dist[d]) for d in dists]
        ax.plot(dists, vals, "o-", color=colors[sc], label=sc, markersize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("SNR raw (dB)")
    ax.set_title("Raw SNR: Three Scenarios")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Raw CWER vs distance
    ax = axes[0][1]
    for sc in scenarios:
        by_dist = defaultdict(list)
        for r in results:
            if _scenario(r) == sc:
                by_dist[r["spy_mic_distance"]].append(r["cwer_raw"])
        if not by_dist:
            continue
        dists = sorted(by_dist.keys())
        vals = [np.mean(by_dist[d]) for d in dists]
        ax.plot(dists, vals, "s-", color=colors[sc], label=sc, markersize=8)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("CWER raw (%)")
    ax.set_title("Raw CWER: Three Scenarios")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Enhanced CWER for spectral_subtraction
    ax = axes[1][0]
    for sc in scenarios:
        by_dist = defaultdict(list)
        for r in results:
            if (_scenario(r) == sc
                    and r.get("attacker.denoiser") == "spectral_subtraction"):
                by_dist[r["spy_mic_distance"]].append(r["cwer_enhanced"])
        if not by_dist:
            continue
        dists = sorted(by_dist.keys())
        vals = [np.mean(by_dist[d]) for d in dists]
        ax.plot(dists, vals, "D-", color=colors[sc], label=sc, markersize=8)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("CWER enhanced (%)")
    ax.set_title("Enhanced CWER: spectral_subtraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: CWER raw vs enhanced scatter
    ax = axes[1][1]
    for sc in scenarios:
        xs, ys = [], []
        for r in results:
            if _scenario(r) == sc:
                xs.append(r["cwer_raw"])
                ys.append(r["cwer_enhanced"])
        if xs:
            ax.scatter(xs, ys, c=colors[sc], label=sc, alpha=0.6, s=30)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
    ax.set_xlabel("CWER raw (%)")
    ax.set_ylabel("CWER enhanced (%)")
    ax.set_title("CWER: Raw vs Enhanced (all denoisers)")
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("MicFrozen Scenario Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
