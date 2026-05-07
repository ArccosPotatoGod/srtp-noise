"""ExperimentRunner — batch parameter-sweep execution over a config grid."""

import copy
import csv
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
from src.evaluator import Evaluator
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
                # Set absolute distance from source along x-axis
                src = self.base_config.source.pos
                config.spy_mic.positions = [
                    (src[0] + float(val), src[1], src[2])
                ]
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

    def run(self) -> None:
        combinations = self._expand_grid()
        for override in tqdm(combinations, desc="Running experiments"):
            config = self._apply_overrides(override)
            metrics = self._run_single(config)
            metrics.update(override)
            self.results.append(metrics)

    def _run_single(self, config: ScenarioConfig) -> Dict:
        from scipy.signal import fftconvolve
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
                                   config.jammer.coherent_params, self.rng)
        jammer = JammerModule(config, cancel, coherent)
        s_cancel, n_coherent = jammer.generate(s_src, audible_rirs["src_to_ref"])

        nonlinearity = create_nonlinearity(config.spy_mic.nonlinearity,
                                           config.spy_mic.nonlinearity_params)
        spy_mic = SpyMicrophoneModule(audible_rirs, ultrasonic_rirs, nonlinearity)
        spy_rec = spy_mic.capture(s_src, s_cancel, n_coherent)

        denoiser = create_denoiser(config.attacker.denoiser,
                                   config.attacker.denoiser_params)
        asr = create_asr(config.attacker.asr, {"ref_text": ref_text})
        attacker = AttackerModule(denoiser, asr)
        enhanced, hyp_text = attacker.attack(spy_rec)

        # SNR = speech power / jamming power
        min_len = min(len(speech_at_spy), len(spy_rec))
        jamming_residual = spy_rec[:min_len] - speech_at_spy[:min_len]
        p_speech = float(np.sum(speech_at_spy[:min_len] ** 2))
        p_jam = float(np.sum(jamming_residual ** 2))
        snr_raw = 10.0 * np.log10(p_speech / max(p_jam, 1e-12))

        min_len_e = min(len(speech_at_spy), len(enhanced))
        jamming_after = enhanced[:min_len_e] - speech_at_spy[:min_len_e]
        p_jam_enh = float(np.sum(jamming_after ** 2))
        snr_enhanced = 10.0 * np.log10(p_speech / max(p_jam_enh, 1e-12))

        raw_hyp = asr.transcribe(spy_rec)
        from src.evaluator import _compute_cwer
        cwer_raw = _compute_cwer(ref_text, raw_hyp)
        cwer_enhanced = _compute_cwer(ref_text, hyp_text)

        return {
            "snr_raw": snr_raw,
            "snr_enhanced": snr_enhanced,
            "cwer_raw": cwer_raw,
            "cwer_enhanced": cwer_enhanced,
        }

    def export_results(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            return
        keys = self.results[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)

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
