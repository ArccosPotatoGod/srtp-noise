"""ExperimentRunner — batch parameter-sweep execution over a config grid."""

import copy
import csv
import hashlib
import itertools
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.config import ScenarioConfig, load_config, load_grid_config
from src.speaker import SpeakerModule
from src.channel import ChannelModule
from src.jammer import JammerModule
from src.spy_mic import SpyMicrophoneModule, build_sniffer_reference
from src.attacker import AttackerModule
from src.evaluator import Evaluator, save_text_report
from strategies.canceling import create_canceling
from strategies.coherent import create_coherent
from strategies.denoiser import create_denoiser
from strategies.asr import create_asr
from strategies.nonlinearity import create_nonlinearity


class ExperimentRunner:
    """Iterates over parameter grid, runs simulations, collects metrics."""

    def __init__(self, base_config_path: str, grid_config_path: str = None):
        self.base_config = load_config(base_config_path)
        self.grid = {}
        if grid_config_path:
            self.grid = load_grid_config(grid_config_path)
        self.results: List[Dict] = []
        self.rng = np.random.default_rng()

    def _expand_grid(self) -> List[Dict[str, Any]]:
        if not self.grid:
            return [{}]
        keys = list(self.grid.keys())
        values = [self.grid[k] for k in keys]
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations

    def _apply_overrides(self, override: Dict) -> ScenarioConfig:
        config = copy.deepcopy(self.base_config)
        for key_path, val in override.items():
            if key_path == "spy_mic_distance":
                # Set primary spy mic at specified distance along x-axis.
                # Add a second mic offset in y (15 cm) for multi-channel ICA/BF.
                src = self.base_config.source.pos
                primary = (src[0] + float(val), src[1], src[2])
                secondary = (primary[0], primary[1] + 0.15, primary[2])
                config.spy_mic.positions = [primary, secondary]
            elif key_path == "spy_mic_angle":
                # Rotate spy mic around source at fixed distance
                import math
                src = self.base_config.source.pos
                rad = math.radians(val)
                base_pos = self.base_config.spy_mic.positions[0]
                dist = math.sqrt(
                    (base_pos[0] - src[0]) ** 2 + (base_pos[1] - src[1]) ** 2
                )
                new_x = src[0] + dist * math.cos(rad)
                new_y = src[1] + dist * math.sin(rad)
                config.spy_mic.positions = [
                    (new_x, new_y, base_pos[2])
                    for base_pos in self.base_config.spy_mic.positions
                ]
            elif key_path == "jammer.coherent_strategy":
                config.jammer.coherent_strategy = val
                if val == "off":
                    config.jammer.canceling_strategy = "off"
            elif key_path == "jammer.canceling_strategy":
                # Only honor when coherent_strategy is active; otherwise
                # the coherent_strategy handler forces canceling off.
                if config.jammer.coherent_strategy != "off":
                    config.jammer.canceling_strategy = val
            else:
                parts = key_path.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], val)
        return config

    def _seed_from_override(self, override: Dict) -> int:
        """Deterministic seed from signal-generation params only.

        Excludes post-processing keys (attacker.*) so that the same physical
        scenario always produces the same spy recording regardless of which
        denoiser/ASR is tested later.
        """
        signal_keys = {k: v for k, v in override.items()
                       if not k.startswith("attacker.")}
        seed_str = str(sorted(signal_keys.items()))
        return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2 ** 31)

    def run(self, n_runs: int = 1) -> None:
        combinations = self._expand_grid()
        for override in tqdm(combinations, desc="Running experiments"):
            config = self._apply_overrides(override)
            seed = self._seed_from_override(override)
            if n_runs <= 1:
                metrics = self._run_single(config, seed)
                metrics.update(override)
                self.results.append(metrics)
            else:
                # Average metrics over multiple ASR corruption seeds
                all_metrics = []
                for run_idx in range(n_runs):
                    m = self._run_single(config, seed, run_index=run_idx)
                    all_metrics.append(m)
                avg = {k: float(np.mean([m[k] for m in all_metrics]))
                       for k in all_metrics[0]}
                avg.update(override)
                self.results.append(avg)

    def _run_single(self, config: ScenarioConfig, seed: int = 0,
                    run_index: int = 0) -> Dict:
        from scipy.signal import fftconvolve
        rng = np.random.default_rng(seed)

        audio_path = config.source.audio_file or "data/sample.wav"
        spk = SpeakerModule(audio_path, fs=config.sim.fs)
        s_src = spk.get_signal()
        ref_text = spk.get_reference_text()

        channel = ChannelModule(config)
        audible_rirs, ultrasonic_rirs = channel.compute_rir()

        # Speech component at spy mic
        speech_at_spy = fftconvolve(s_src, audible_rirs["src_to_spy"][0])[:len(s_src)]

        cancel = create_canceling(config.jammer.canceling_strategy,
                                  config.jammer.canceling_params)
        coherent = create_coherent(config.jammer.coherent_strategy,
                                   config.jammer.coherent_params, rng)
        jammer = JammerModule(config, cancel, coherent)
        s_cancel, n_coherent = jammer.generate(s_src, audible_rirs["src_to_ref"])

        nonlinearity = create_nonlinearity(config.spy_mic.nonlinearity,
                                           config.spy_mic.nonlinearity_params)
        spy_mic = SpyMicrophoneModule(audible_rirs, ultrasonic_rirs, nonlinearity)
        spy_rec = spy_mic.capture(s_src, s_cancel, n_coherent)
        # Use first channel for single-channel SNR/CWER metrics
        spy_1d = spy_rec[0] if spy_rec.ndim > 1 else spy_rec

        denoiser = create_denoiser(config.attacker.denoiser,
                                   config.attacker.denoiser_params)
        asr = create_asr(config.attacker.asr, {"ref_text": ref_text})
        attacker = AttackerModule(denoiser, asr)
        noise_ref = build_sniffer_reference(s_cancel, n_coherent,
                                           ultrasonic_rirs["jammer_to_spy"][0],
                                           len(s_src), nonlinearity)
        enhanced, hyp_text = attacker.attack(spy_rec, noise_ref=noise_ref,
                                              seed_offset=run_index)
        enh_1d = enhanced[0] if enhanced.ndim > 1 else enhanced
        raw_hyp = asr.transcribe(spy_1d, seed_offset=run_index)

        evaluator = Evaluator(fs=config.sim.fs)
        return evaluator.evaluate(speech_at_spy, spy_1d, enh_1d,
                                  ref_text, raw_hyp, hyp_text)

    def export_results(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            return
        keys = self.results[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)

    def export_text_report(self, path: str) -> None:
        """Save a comprehensive text report of all results."""
        save_text_report(self.results, path)

    def _group_by(self, key_x, key_y, key_group, key_style=None):
        """Group results by strategy & optional denoiser for per-series plotting."""
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            if key_style is not None:
                label = f"{r.get(key_group, '?')} | {r.get(key_style, '?')}"
            else:
                label = str(r.get(key_group, '?'))
            groups[label].append((r.get(key_x, 0), r.get(key_y, 0)))
        return groups

    def _build_strategy_cancel_label(self, r: Dict) -> str:
        """Composite label: coherent_strategy + optional cancel marker."""
        strat = r.get("jammer.coherent_strategy", "?")
        cancel = r.get("jammer.canceling_strategy", "phase_inversion")
        if cancel == "off":
            return f"{strat} (no cancel)"
        return strat

    def plot_snr_vs_distance(self, path: str) -> None:
        """Plot raw SNR vs distance grouped by jamming strategy + canceling.

        Raw SNR is measured before any denoiser is applied, so denoiser
        choice does not affect these values.  Canceling strategy does
        affect the raw spy recording, so both are shown.
        """
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            label = self._build_strategy_cancel_label(r)
            groups[label].append((r.get("spy_mic_distance", 0), r.get("snr_raw", 0)))
        # Aggregate: average snr_raw per (label, distance) since multiple
        # denoiser rows share the same raw value
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

    def plot_cwer_vs_distance(self, path: str) -> None:
        """Plot raw CWER vs distance grouped by jamming strategy + canceling.

        Raw CWER is measured before any denoiser is applied, so denoiser
        choice does not affect these values.  Canceling strategy does
        affect the raw spy recording, so both are shown.
        """
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            label = self._build_strategy_cancel_label(r)
            groups[label].append((r.get("spy_mic_distance", 0), r.get("cwer_raw", 0)))
        # Aggregate: average cwer_raw per (label, distance)
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

    # ------------------------------------------------------------------
    # Enhanced-metric plots (post-denoising, grouped by denoiser)
    # ------------------------------------------------------------------

    def plot_snr_enhanced_vs_distance(self, path: str, strategy: str = "fixed_weight") -> None:
        """Plot enhanced SNR vs distance for one jamming strategy, grouped by denoiser+canceling."""
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        subset = [r for r in self.results if r.get("jammer.coherent_strategy") == strategy]
        from collections import defaultdict
        groups = defaultdict(list)
        for r in subset:
            den = r.get("attacker.denoiser", "?")
            cancel = r.get("jammer.canceling_strategy", "?")
            label = f"{den}" if cancel == "phase_inversion" else f"{den} (no cancel)"
            groups[label].append(
                (r.get("spy_mic_distance", 0), r.get("snr_enhanced", 0)))
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

    def plot_cwer_enhanced_vs_distance(self, path: str, strategy: str = "fixed_weight") -> None:
        """Plot enhanced CWER vs distance for one jamming strategy, grouped by denoiser+canceling."""
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        subset = [r for r in self.results if r.get("jammer.coherent_strategy") == strategy]
        from collections import defaultdict
        groups = defaultdict(list)
        for r in subset:
            den = r.get("attacker.denoiser", "?")
            cancel = r.get("jammer.canceling_strategy", "?")
            label = f"{den}" if cancel == "phase_inversion" else f"{den} (no cancel)"
            groups[label].append(
                (r.get("spy_mic_distance", 0), r.get("cwer_enhanced", 0)))
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

    # ------------------------------------------------------------------
    # Denoiser comparison bar chart
    # ------------------------------------------------------------------

    def plot_denoiser_comparison(self, path: str, distance: float = 1.0) -> None:
        """Grouped bar chart: cwer_enhanced per denoiser, grouped by strategy+canceling."""
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        subset = [r for r in self.results
                  if abs(float(r.get("spy_mic_distance", 0)) - distance) < 0.01]
        if not subset:
            return

        # Build combined strategy+canceling labels
        strat_labels = sorted(set(
            f"{r['jammer.coherent_strategy']}"
            + ("" if r.get("jammer.canceling_strategy") == "phase_inversion"
               else " (no cancel)")
            for r in subset
        ))
        denoisers = sorted(set(r["attacker.denoiser"] for r in subset))

        x = np.arange(len(denoisers))
        width = 0.8 / len(strat_labels)

        fig, ax = plt.subplots(figsize=(14, 6))
        for i, sl in enumerate(strat_labels):
            sl_parts = sl.split(" (no cancel)")
            strat = sl_parts[0]
            no_cancel = len(sl_parts) > 1
            strat_rows = [r for r in subset
                          if r["jammer.coherent_strategy"] == strat
                          and (no_cancel == (r.get("jammer.canceling_strategy") != "phase_inversion"))]
            den_map = {r["attacker.denoiser"]: r.get("cwer_enhanced", 0) for r in strat_rows}
            y = [den_map.get(d, 0) for d in denoisers]
            bars = ax.bar(x + i * width, y, width, label=sl)
            for bar, val in zip(bars, y):
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

    # ------------------------------------------------------------------
    # CWER scatter: raw vs enhanced
    # ------------------------------------------------------------------

    def plot_cwer_scatter(self, path: str) -> None:
        """Scatter plot: cwer_raw vs cwer_enhanced colored by strategy, styled by denoiser."""
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        strategies = sorted(set(self._build_strategy_cancel_label(r) for r in self.results))
        denoisers = sorted(set(r["attacker.denoiser"] for r in self.results))
        colors = plt.cm.tab20(np.linspace(0, 1, len(strategies)))
        markers = ["o", "s", "D", "^", "v", "p"]

        fig, ax = plt.subplots(figsize=(12, 7))
        for si, strat in enumerate(strategies):
            for di, den in enumerate(denoisers):
                pts = [(r["cwer_raw"], r["cwer_enhanced"])
                       for r in self.results
                       if self._build_strategy_cancel_label(r) == strat
                       and r.get("attacker.denoiser") == den]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                label = f"{strat} | {den}" if si == 0 or di == len(denoisers) - 1 else None
                ax.scatter(xs, ys, c=[colors[si]], marker=markers[di % len(markers)],
                          alpha=0.7, s=40, label=label if (di == 0) else None)

        ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y = x (no change)")
        ax.set_xlabel("CWER raw (%)")
        ax.set_ylabel("CWER enhanced (%)")
        ax.set_title("CWER: Raw vs Enhanced")
        ax.set_xlim(-2, 105)
        ax.set_ylim(-2, 105)
        # Strategy legend
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

    # ------------------------------------------------------------------
    # Scenario comparison: no-jammer vs noise-only vs noise+cancel
    # ------------------------------------------------------------------

    def plot_scenario_comparison(self, path: str) -> None:
        """Compare the three core jamming scenarios across all distances.

        Scenarios:
          1. off (no noise, no cancel)           — baseline
          2. active + cancel=off (noise only)    — no anti-speech cancel
          3. active + cancel=phase_inversion     — full MicFrozen
        """
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        from collections import defaultdict

        # Classify each result into one of three scenarios
        def _scenario(r):
            if r.get("jammer.coherent_strategy") == "off":
                return "1. jammer off"
            if r.get("jammer.canceling_strategy") == "off":
                return "2. noise only (no cancel)"
            return "3. noise + cancel"

        scenarios = ["1. jammer off", "2. noise only (no cancel)", "3. noise + cancel"]
        denoisers = sorted(set(r["attacker.denoiser"] for r in self.results))

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        colors = {"1. jammer off": "tab:green",
                  "2. noise only (no cancel)": "tab:orange",
                  "3. noise + cancel": "tab:red"}

        # --- Panel 1: Raw SNR vs distance (per scenario, averaged over denoisers) ---
        ax = axes[0][0]
        for sc in scenarios:
            by_dist = defaultdict(list)
            for r in self.results:
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

        # --- Panel 2: Raw CWER vs distance ---
        ax = axes[0][1]
        for sc in scenarios:
            by_dist = defaultdict(list)
            for r in self.results:
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

        # --- Panel 3: Enhanced CWER for spectral_subtraction only ---
        ax = axes[1][0]
        for sc in scenarios:
            by_dist = defaultdict(list)
            for r in self.results:
                if _scenario(r) == sc and r.get("attacker.denoiser") == "spectral_subtraction":
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

        # --- Panel 4: CWER raw vs enhanced for all scenarios ---
        ax = axes[1][1]
        for sc in scenarios:
            xs, ys = [], []
            for r in self.results:
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

    # ------------------------------------------------------------------
    # Angle sweep
    # ------------------------------------------------------------------

    def run_angle_sweep(self, path: str = "results/angle_sweep.png",
                        distance: float = 2.0, angles: list = None,
                        strategy: str = "fixed_weight",
                        denoiser: str = "none") -> None:
        """CWER vs angle sweep at a fixed distance (paper Fig.11 angle dimension)."""
        import matplotlib.pyplot as plt
        if angles is None:
            angles = [0, 10, 20, 30, 40, 50, 60]
        config = copy.deepcopy(self.base_config)
        config.jammer.coherent_strategy = strategy
        config.attacker.denoiser = denoiser

        cwer_raw_vals = []
        cwer_enh_vals = []
        snr_raw_vals = []

        for angle in tqdm(angles, desc="Angle sweep"):
            src = self.base_config.source.pos
            base_pos = self.base_config.spy_mic.positions[0]
            rad = np.radians(angle)
            new_x = src[0] + distance * np.cos(rad)
            new_y = src[1] + distance * np.sin(rad)
            config.spy_mic.positions = [
                (new_x, new_y, base_pos[2]),
                (new_x, new_y + 0.15, base_pos[2]),
            ]
            seed = hash((distance, angle, strategy, denoiser)) & 0x7FFFFFFF
            metrics = self._run_single(config, seed)
            cwer_raw_vals.append(metrics["cwer_raw"])
            cwer_enh_vals.append(metrics["cwer_enhanced"])
            snr_raw_vals.append(metrics["snr_raw"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(angles, cwer_raw_vals, "o-", label="CWER raw", color="tab:red")
        ax1.plot(angles, cwer_enh_vals, "s-", label="CWER enhanced", color="tab:blue")
        ax1.set_xlabel("Angle (degrees)")
        ax1.set_ylabel("CWER (%)")
        ax1.set_title(f"Angle Sweep — {strategy} | {denoiser} @ {distance:.0f} m")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(angles, snr_raw_vals, "D-", color="tab:orange")
        ax2.set_xlabel("Angle (degrees)")
        ax2.set_ylabel("SNR (dB)")
        ax2.set_title(f"SNR vs Angle @ {distance:.0f} m")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Pipeline audio-stage visualization
    # ------------------------------------------------------------------

    def save_pipeline_audio_plots(self, output_dir: str = "results/audio_stages",
                                  config: ScenarioConfig = None,
                                  seed: int = 42) -> None:
        """Run one simulation and save waveform/spectrogram plots for every stage."""
        import matplotlib.pyplot as plt
        from scipy.signal import fftconvolve, spectrogram

        if config is None:
            config = copy.deepcopy(self.base_config)
        rng = np.random.default_rng(seed)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        audio_path = config.source.audio_file or "data/sample.wav"
        spk = SpeakerModule(audio_path, fs=config.sim.fs)
        s_src = spk.get_signal()
        fs = config.sim.fs
        t = np.arange(len(s_src)) / fs

        channel = ChannelModule(config)
        audible_rirs, ultrasonic_rirs = channel.compute_rir()
        speech_at_spy = fftconvolve(s_src, audible_rirs["src_to_spy"][0])[:len(s_src)]

        cancel = create_canceling(config.jammer.canceling_strategy, config.jammer.canceling_params)
        coherent = create_coherent(config.jammer.coherent_strategy, config.jammer.coherent_params, rng)
        jammer = JammerModule(config, cancel, coherent)
        ref_rir = audible_rirs["src_to_ref"]
        ref_sig = fftconvolve(s_src, ref_rir)[:len(s_src)]
        s_cancel, n_coherent = jammer.generate(s_src, ref_rir)

        nonlinearity = create_nonlinearity(config.spy_mic.nonlinearity, config.spy_mic.nonlinearity_params)
        spy_mic = SpyMicrophoneModule(audible_rirs, ultrasonic_rirs, nonlinearity)
        spy_rec = spy_mic.capture(s_src, s_cancel, n_coherent)
        spy_1d = spy_rec[0] if spy_rec.ndim > 1 else spy_rec

        # Audible & ultrasonic arrivals at spy
        audible_arrival = fftconvolve(s_src, audible_rirs["src_to_spy"][0])[:len(s_src)]
        jammer_baseband = s_cancel + n_coherent
        ultrasonic_arrival = fftconvolve(jammer_baseband, ultrasonic_rirs["jammer_to_spy"][0])[:len(s_src)]
        ultrasonic_demod = nonlinearity.apply(ultrasonic_arrival)

        denoiser = create_denoiser(config.attacker.denoiser, config.attacker.denoiser_params)
        asr = create_asr(config.attacker.asr, {"ref_text": spk.get_reference_text()})
        attacker = AttackerModule(denoiser, asr)
        noise_ref = build_sniffer_reference(s_cancel, n_coherent,
                                           ultrasonic_rirs["jammer_to_spy"][0],
                                           len(s_src), nonlinearity)
        enhanced, hyp_text = attacker.attack(spy_rec, noise_ref=noise_ref)
        enh_1d = enhanced[0] if enhanced.ndim > 1 else enhanced

        stages = [
            ("01_source_speech", s_src, "Source speech s(t)"),
            ("02_reference_mic", ref_sig, "Reference mic (RIR convolved)"),
            ("03_cancel_signal", s_cancel, "Cancel signal s_cancel(t)"),
            ("04_coherent_noise", n_coherent, "Coherent noise n_coherent(t)"),
            ("05_audible_arrival", audible_arrival, "Audible arrival at spy (linear)"),
            ("06_ultrasonic_arrival", ultrasonic_arrival, "Ultrasonic arrival at spy"),
            ("07_ultrasonic_demod", ultrasonic_demod, "Ultrasonic arrival after nonlinear demod"),
            ("08_spy_recording", spy_1d, "Spy recording (audible + demod ultrasonic)"),
            ("09_enhanced", enh_1d, f"After denoising ({config.attacker.denoiser})"),
            ("10_jammer_baseband", jammer_baseband, "Jammer baseband (cancel + noise)"),
            ("11_noise_reference", noise_ref, "Sniffer noise reference"),
        ]

        for fname, sig, title in stages:
            self._save_signal_plot(out, fname, sig, fs, title)

        # Combined overview plot
        self._save_overview_plot(out, s_src, s_cancel, n_coherent, spy_1d, enh_1d, fs)
        # Spectrogram comparison
        self._save_spectrogram_plot(out, s_src, spy_1d, enh_1d, fs)

        print(f"Pipeline audio plots saved to {out}/")

    def _save_signal_plot(self, out_dir, fname, signal, fs, title):
        """Save a single waveform plot."""
        import matplotlib.pyplot as plt
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

    def _save_overview_plot(self, out_dir, s_src, s_cancel, n_coherent, spy_rec, enhanced, fs):
        """Save a 3-row overview: source, jammer signals, spy+enhanced."""
        import matplotlib.pyplot as plt
        t = np.arange(len(s_src)) / fs
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(t, s_src, linewidth=0.5, color="tab:green", label="Source speech")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Source Speech s(t)")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, s_cancel, linewidth=0.5, color="tab:red", alpha=0.7, label="s_cancel")
        axes[1].plot(t, n_coherent, linewidth=0.5, color="tab:purple", alpha=0.7, label="n_coherent")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title("Jammer Signals")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, spy_rec, linewidth=0.5, color="tab:orange", alpha=0.7, label="Spy recording")
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

    def _save_spectrogram_plot(self, out_dir, s_src, spy_rec, enhanced, fs):
        """Save spectrogram comparison: source, jammed, enhanced."""
        import matplotlib.pyplot as plt
        from scipy.signal import spectrogram
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        for ax, sig, title in [
            (axes[0], s_src, "Source Speech"),
            (axes[1], spy_rec, "Spy Recording (jammed)"),
            (axes[2], enhanced, f"After Denoising"),
        ]:
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

    # ------------------------------------------------------------------
    # Heatmap (existing)
    # ------------------------------------------------------------------

    def run_coverage_heatmap(self, resolution: int = 10,
                             z: float = 1.5,
                             strategy: str = "fixed_weight",
                             denoiser: str = "none",
                             path: str = "results/heatmap.png") -> None:
        """2D spatial sweep: scan spy mic across the room x-y plane.

        Corresponds to the coverage heatmap in paper Fig.11.
        """
        import matplotlib.pyplot as plt

        room = self.base_config.room
        src = self.base_config.source.pos
        xs = np.linspace(0.5, room.dim[0] - 0.5, resolution)
        ys = np.linspace(0.5, room.dim[1] - 0.5, resolution)
        snr_grid = np.zeros((resolution, resolution))

        config = copy.deepcopy(self.base_config)
        config.jammer.coherent_strategy = strategy
        config.attacker.denoiser = denoiser

        total = resolution * resolution
        for i, spy_x in enumerate(tqdm(xs, desc="Heatmap X")):
            for j, spy_y in enumerate(ys):
                config.spy_mic.positions = [
                    (float(spy_x), float(spy_y), z),
                    (float(spy_x), float(spy_y) + 0.15, z),
                ]
                seed = hash((spy_x, spy_y, strategy, denoiser)) & 0x7FFFFFFF
                metrics = self._run_single(config, seed)
                snr_grid[j, i] = metrics["snr_raw"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(xs, ys, snr_grid, shading="auto", cmap="RdYlGn")
        ax.scatter(src[0], src[1], marker="*", s=200, color="black",
                   label="Source")
        ax.scatter(self.base_config.jammer.pos_spk[0],
                   self.base_config.jammer.pos_spk[1],
                   marker="s", s=100, color="blue", label="Jammer")
        cbar = fig.colorbar(im, ax=ax, label="SNR (dB)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Coverage Heatmap — {strategy} | {denoiser}\n"
                     f"SNR at z={z} m, res={resolution}x{resolution}")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
