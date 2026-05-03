"""Visualization utilities for experiment results.

Generates:
- Waveform comparison plots (Section 7.1)
- Spectrogram subplots (Section 7.1)
- SNR-vs-distance line charts (Section 7.4)
- WER bar charts (Section 7.4)
- Heatmaps over distance-angle grid (Section 7.4)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Default style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def plot_waveforms(signals, labels, fs=16000, title="Waveform Comparison"):
    """Plot multiple waveforms in vertical subplots.

    Parameters
    ----------
    signals : list of np.ndarray
        Signals to plot.
    labels : list of str
        Labels for each subplot.
    fs : int
        Sample rate (for time axis).
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    time = np.arange(len(signals[0])) / fs
    for ax, sig, label in zip(axes, signals, labels):
        ax.plot(time, sig, linewidth=0.4, color="steelblue")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_spectrogram(signal, fs=16000, title="Spectrogram", ax=None):
    """Plot a single spectrogram.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : int
        Sample rate.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(signal, Fs=fs, NFFT=512, noverlap=256, cmap="inferno")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return ax


def plot_spectrogram_comparison(signals, labels, fs=16000,
                                title="Spectrogram Comparison",
                                output_path=None):
    """Plot spectrograms of multiple signals side-by-side.

    Used for Section 7.1 comparisons:
    - Original speech
    - Gaussian noise only
    - MicFrozen cancellation + coherent noise
    - After ICA denoising

    Parameters
    ----------
    signals : list of np.ndarray
        Signals to plot.
    labels : list of str
        Labels for each subplot.
    fs : int
        Sample rate.
    title : str
        Figure title.
    output_path : str or None
        If set, save figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(signals)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (sig, label) in enumerate(zip(signals, labels)):
        ax = axes[i]
        ax.specgram(sig, Fs=fs, NFFT=512, noverlap=256, cmap="inferno")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(label, fontsize=9)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_snr_vs_distance(results_df, output_path=None):
    """Line chart: SNR vs distance, grouped by jamming method.

    Parameters
    ----------
    results_df : pd.DataFrame
        Required columns: distance, method, attack, snr.
    output_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=results_df, x="distance", y="snr",
                 hue="method", style="attack", markers=True,
                 markersize=6, ax=ax)
    ax.set_xlabel("Eavesdropper Distance (m)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR vs Distance by Jamming Method")
    ax.invert_yaxis()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_wer_bars(results_df, output_path=None):
    """Bar chart: WER/MFCC distance comparison.

    Parameters
    ----------
    results_df : pd.DataFrame
        Required columns: method, attack, wer (or mfcc_distance).
    output_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_col = "wer" if "wer" in results_df.columns else "mfcc_distance"
    sns.barplot(data=results_df, x="method", y=metric_col, hue="attack", ax=ax)
    ax.set_xlabel("Jamming Method")
    ax.set_ylabel(metric_col.upper() if metric_col == "wer" else "MFCC Distance")
    ax.set_title(f"{metric_col.upper() if metric_col == 'wer' else 'MFCC Distance'} by Method and Attack")
    ax.tick_params(axis="x", rotation=15)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_heatmap(matrix, distances, angles, title="SNR Heatmap",
                 output_path=None):
    """Heatmap over distance-angle grid.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_distances, n_angles)
        Metric values.
    distances : list of float
        Y-axis labels.
    angles : list of float
        X-axis labels.
    title : str
    output_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=[f"{a}°" for a in angles],
                yticklabels=[f"{d}m" for d in distances],
                annot=True, fmt=".1f", cmap="RdYlBu_r", ax=ax,
                cbar_kws={"label": "SNR (dB)"})
    ax.set_xlabel("Angle")
    ax.set_ylabel("Distance")
    ax.set_title(title)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_heatmap_grid(matrices, method_names, distances, angles,
                      output_path=None):
    """Multiple heatmaps in a row, one per jamming method.

    Parameters
    ----------
    matrices : dict[str, np.ndarray]
        Mapping of method_name -> (n_distances, n_angles) array.
    method_names : list of str
        Ordered method names.
    distances, angles : lists
        Axis labels.
    output_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(method_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    vmin = min(m.min() for m in matrices.values())
    vmax = max(m.max() for m in matrices.values())

    for ax, name in zip(axes, method_names):
        sns.heatmap(matrices[name],
                    xticklabels=[f"{a}°" for a in angles],
                    yticklabels=[f"{d}m" for d in distances],
                    annot=True, fmt=".1f", cmap="RdYlBu_r",
                    vmin=vmin, vmax=vmax, ax=ax,
                    cbar_kws={"label": "SNR (dB)"})
        ax.set_xlabel("Angle")
        ax.set_ylabel("Distance")
        ax.set_title(name)

    fig.suptitle("SNR Heatmaps by Jamming Method")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
