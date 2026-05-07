#!/usr/bin/env python3
"""Single-run demo script — compares jammer-on vs jammer-off scenarios."""

import argparse
import numpy as np
from scipy.signal import fftconvolve

from src.config import load_config
from src.speaker import SpeakerModule
from src.channel import ChannelModule
from src.jammer import JammerModule
from src.spy_mic import SpyMicrophoneModule, build_sniffer_reference
from src.attacker import AttackerModule
from src.evaluator import Evaluator
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
    # Sniffer reference: jammer baseband through ultrasonic RIR to spy position
    noise_ref = build_sniffer_reference(s_cancel, n_coherent,
                                        ultrasonic_rirs["jammer_to_spy"][0],
                                        len(s_src), nonlinearity)
    enhanced, hyp_text = attacker.attack(spy_rec, noise_ref=noise_ref)

    # SNR: speech power / jamming residual power (use first channel)
    spy_1d = spy_rec[0] if spy_rec.ndim > 1 else spy_rec
    enh_1d = enhanced[0] if enhanced.ndim > 1 else enhanced
    min_len = min(len(speech_at_spy), len(spy_1d))
    jamming_residual = spy_1d[:min_len] - speech_at_spy[:min_len]

    evaluator = Evaluator(fs=config.sim.fs)
    raw_hyp = asr.transcribe(spy_1d)
    eval_metrics = evaluator.evaluate(speech_at_spy, spy_1d, enh_1d,
                                      ref_text, raw_hyp, hyp_text)

    metrics = {
        "snr_raw_db": eval_metrics["snr_raw"],
        "snr_enhanced_db": eval_metrics["snr_enhanced"],
        "cwer_raw_pct": eval_metrics["cwer_raw"],
        "cwer_enhanced_pct": eval_metrics["cwer_enhanced"],
    }
    diagnostics = {
        "speech_rms": float(np.sqrt(np.mean(speech_at_spy ** 2))),
        "spy_rms": float(np.sqrt(np.mean(spy_1d ** 2))),
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
    parser.add_argument("--export-audio", action="store_true",
                        help="Export waveform & spectrogram plots for all pipeline stages")
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

    # Export text report
    from src.evaluator import save_text_report
    results = [
        {**met_on, "scenario": "Jammer ON"},
        {**met_off, "scenario": "Jammer OFF"},
    ]
    save_text_report(results, "results/single_report.txt",
                     title="MicFrozen Single-Run Report")
    print("\nText report saved to results/single_report.txt")

    if args.export_audio:
        from runner import ExperimentRunner
        runner = ExperimentRunner(args.config)
        runner.save_pipeline_audio_plots(output_dir="results/audio_stages",
                                         config=config, seed=42)
        print("Pipeline audio plots saved to results/audio_stages/")


if __name__ == "__main__":
    main()
