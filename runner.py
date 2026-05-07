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
from src.spy_mic import SpyMicrophoneModule
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

    def run(self) -> None:
        combinations = self._expand_grid()
        for override in tqdm(combinations, desc="Running experiments"):
            config = self._apply_overrides(override)
            seed = self._seed_from_override(override)
            metrics = self._run_single(config, seed)
            metrics.update(override)
            self.results.append(metrics)

    def _run_single(self, config: ScenarioConfig, seed: int = 0) -> Dict:
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
        # Sniffer reference: jammer baseband through ultrasonic RIR to spy position.
        # Simulates an ultrasonic sniffer co-located with the spy mic that captures
        # the jammer signal through the same acoustic channel (RIR + nonlinearity).
        jammer_baseband = s_cancel + n_coherent
        jammer_rir = ultrasonic_rirs["jammer_to_spy"][0]
        noise_ref = fftconvolve(jammer_baseband, jammer_rir)[:len(s_src)]
        noise_ref = nonlinearity.apply(noise_ref)
        enhanced, hyp_text = attacker.attack(spy_rec, noise_ref=noise_ref)
        enh_1d = enhanced[0] if enhanced.ndim > 1 else enhanced
        raw_hyp = asr.transcribe(spy_1d)

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

    def _group_by(self, key_x, key_y, key_group, key_style):
        """Group results by strategy & denoiser for per-series plotting."""
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            label = f"{r.get(key_group, '?')} | {r.get(key_style, '?')}"
            groups[label].append((r.get(key_x, 0), r.get(key_y, 0)))
        return groups

    def plot_snr_vs_distance(self, path: str) -> None:
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        groups = self._group_by("spy_mic_distance", "snr_raw",
                                "jammer.coherent_strategy", "attacker.denoiser")
        fig, ax = plt.subplots(figsize=(8, 5))
        markers = ["o", "s", "D", "^", "v"]
        for i, (label, pts) in enumerate(sorted(groups.items())):
            pts_sorted = sorted(pts, key=lambda x: x[0])
            x = [p[0] for p in pts_sorted]
            y = [p[1] for p in pts_sorted]
            ax.plot(x, y, marker=markers[i % len(markers)], linestyle="-", label=label)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("SNR (dB)")
        ax.set_title("SNR vs Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def plot_cwer_vs_distance(self, path: str) -> None:
        import matplotlib.pyplot as plt
        if not self.results:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        groups = self._group_by("spy_mic_distance", "cwer_raw",
                                "jammer.coherent_strategy", "attacker.denoiser")
        fig, ax = plt.subplots(figsize=(8, 5))
        markers = ["o", "s", "D", "^", "v"]
        for i, (label, pts) in enumerate(sorted(groups.items())):
            pts_sorted = sorted(pts, key=lambda x: x[0])
            x = [p[0] for p in pts_sorted]
            y = [p[1] for p in pts_sorted]
            ax.plot(x, y, marker=markers[i % len(markers)], linestyle="-", label=label)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("CWER (%)")
        ax.set_title("CWER vs Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

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
