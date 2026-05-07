#!/usr/bin/env python3
"""Single-run demo script — compares jammer-on vs jammer-off scenarios."""

import argparse
import numpy as np
from scipy.signal import fftconvolve

from src.config import load_config
from src.speaker import SpeakerModule
from src.channel import ChannelModule
from src.jammer import JammerModule
from src.spy_mic import SpyMicrophoneModule
from src.attacker import AttackerModule
from src.evaluator import _compute_snr, _compute_cwer
from src.report import format_metrics_table
from strategies.canceling import create_canceling
from strategies.coherent import create_coherent
from strategies.denoiser import create_denoiser
from strategies.asr import create_asr
from strategies.nonlinearity import create_nonlinearity


def compute_speech_component(s_src, rir_src_to_spy):
    """Compute the speech-only component arriving at spy mic."""
    return fftconvolve(s_src, rir_src_to_spy)[:len(s_src)]


def run_scenario(config, rng, jammer_on=True):
    """Run one complete simulation. Returns (metrics_dict, diagnostics_dict)."""
    audio_path = config.source.audio_file or "data/sample.wav"
    spk = SpeakerModule(audio_path, fs=config.sim.fs)
    s_src = spk.get_signal()
    ref_text = spk.get_reference_text()

    channel = ChannelModule(config)
    audible_rirs, ultrasonic_rirs = channel.compute_rir()

    speech_at_spy = compute_speech_component(s_src, audible_rirs["src_to_spy"][0])

    if jammer_on:
        cancel = create_canceling(config.jammer.canceling_strategy,
                                  config.jammer.canceling_params)
        coherent = create_coherent(config.jammer.coherent_strategy,
                                   config.jammer.coherent_params, rng)
        jammer = JammerModule(config, cancel, coherent)
        s_cancel, n_coherent = jammer.generate(s_src, audible_rirs["src_to_ref"])
    else:
        s_cancel = np.zeros_like(s_src)
        n_coherent = np.zeros_like(s_src)

    nonlinearity = create_nonlinearity(config.spy_mic.nonlinearity,
                                       config.spy_mic.nonlinearity_params)
    spy_mic = SpyMicrophoneModule(audible_rirs, ultrasonic_rirs, nonlinearity)
    spy_rec = spy_mic.capture(s_src, s_cancel, n_coherent)

    denoiser = create_denoiser(config.attacker.denoiser,
                               config.attacker.denoiser_params)
    asr = create_asr(config.attacker.asr, {"ref_text": ref_text})
    attacker = AttackerModule(denoiser, asr)
    enhanced, hyp_text = attacker.attack(spy_rec)

    # SNR: speech power / jamming residual power
    min_len = min(len(speech_at_spy), len(spy_rec))
    jamming_residual = spy_rec[:min_len] - speech_at_spy[:min_len]
    p_speech = float(np.sum(speech_at_spy[:min_len] ** 2))
    p_jam = float(np.sum(jamming_residual ** 2))
    snr_raw = 10.0 * np.log10(p_speech / max(p_jam, 1e-12))

    min_len_e = min(len(speech_at_spy), len(enhanced))
    jamming_after_denoise = enhanced[:min_len_e] - speech_at_spy[:min_len_e]
    p_jam_enh = float(np.sum(jamming_after_denoise ** 2))
    snr_enhanced = 10.0 * np.log10(p_speech / max(p_jam_enh, 1e-12))

    cwer_enhanced = _compute_cwer(ref_text, hyp_text)
    raw_hyp = asr.transcribe(spy_rec)
    cwer_raw = _compute_cwer(ref_text, raw_hyp)

    metrics = {
        "snr_raw_db": snr_raw,
        "snr_enhanced_db": snr_enhanced,
        "cwer_raw_pct": cwer_raw,
        "cwer_enhanced_pct": cwer_enhanced,
    }
    diagnostics = {
        "speech_rms": float(np.sqrt(np.mean(speech_at_spy ** 2))),
        "spy_rms": float(np.sqrt(np.mean(spy_rec ** 2))),
        "jamming_rms": float(np.sqrt(np.mean(jamming_residual ** 2))),
        "cancel_rms": float(np.sqrt(np.mean(s_cancel ** 2))),
        "noise_rms": float(np.sqrt(np.mean(n_coherent ** 2))),
        "hyp_text": hyp_text[:120],
        "ref_text": ref_text[:120],
    }
    return metrics, diagnostics


def main():
    parser = argparse.ArgumentParser(description="MicFrozen single-run simulation")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--audio", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.audio:
        config.source.audio_file = args.audio
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("MicFrozen Simulation — Jammer ON vs OFF Comparison")
    print("=" * 60)

    print("\n>>> Running with Jammer ON ...")
    met_on, diag_on = run_scenario(config, rng, jammer_on=True)

    print(">>> Running with Jammer OFF ...")
    met_off, diag_off = run_scenario(config, rng, jammer_on=False)

    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Jammer OFF':>12} {'Jammer ON':>12}")
    print("=" * 60)
    for key in ["snr_raw_db", "snr_enhanced_db", "cwer_raw_pct", "cwer_enhanced_pct"]:
        print(f"{key:<30} {met_off[key]:12.2f} {met_on[key]:12.2f}")

    print("\n--- Signal Power Diagnostics ---")
    print(f"{'Component':<30} {'Jammer OFF':>12} {'Jammer ON':>12}")
    print("-" * 60)
    for key in ["speech_rms", "spy_rms", "jamming_rms", "cancel_rms", "noise_rms"]:
        off_val = diag_off.get(key, 0)
        on_val = diag_on.get(key, 0)
        print(f"{key:<30} {off_val:12.6f} {on_val:12.6f}")

    print(f"\nReference text:  '{diag_on['ref_text']}'")
    print(f"Jammer ON  hyp:  '{diag_on['hyp_text']}'")
    print(f"Jammer OFF hyp:  '{diag_off['hyp_text']}'")

    snr_drop = met_off["snr_raw_db"] - met_on["snr_raw_db"]
    print(f"\n>>> SNR reduction from jamming: {snr_drop:.1f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
