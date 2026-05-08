#!/usr/bin/env python3
"""Single-run demo — compares jammer-on vs jammer-off."""

import argparse

from src.config import load_config, apply_overrides
from src.evaluator import save_text_report
from runner import run_single
from src.exporter import export_wav_stages, save_pipeline_audio_plots


def main():
    parser = argparse.ArgumentParser(description="MicFrozen single-run simulation")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--export-audio", action="store_true",
                        help="Export waveform & spectrogram PNGs for all pipeline stages")
    parser.add_argument("--export-wav", action="store_true",
                        help="Export WAV audio files for listening comparison "
                        "(source / noise-only / noise+cancel / enhanced)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.audio:
        config.source.audio_file = args.audio

    seed = 42

    # --- Jammer ON ---
    print(">>> Running with Jammer ON ...")
    met_on = run_single(config, seed=seed)

    # --- Jammer OFF ---
    print(">>> Running with Jammer OFF ...")
    config_off = apply_overrides(config, {"jammer.coherent_strategy": "off"})
    met_off = run_single(config_off, seed=seed)

    # --- Print comparison ---
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Jammer OFF':>12} {'Jammer ON':>12}")
    print("=" * 60)
    for key_disp, key in [("snr_raw_db", "snr_raw"), ("snr_enhanced_db", "snr_enhanced"),
                           ("cwer_raw_pct", "cwer_raw"), ("cwer_enhanced_pct", "cwer_enhanced")]:
        print(f"{key_disp:<30} {met_off[key]:12.2f} {met_on[key]:12.2f}")

    print("\n--- Signal Power Diagnostics ---")
    print(f"{'Component':<30} {'Jammer OFF':>12} {'Jammer ON':>12}")
    print("-" * 60)
    for key in ["speech_rms", "spy_rms", "jamming_rms", "cancel_rms", "noise_rms"]:
        print(f"{key:<30} {met_off[key]:12.6f} {met_on[key]:12.6f}")

    print(f"\nReference text:  '{met_on['ref_text']}'")
    print(f"Jammer ON  hyp:  '{met_on['hyp_text_enhanced']}'")
    print(f"Jammer OFF hyp:  '{met_off['hyp_text_enhanced']}'")

    snr_drop = met_off["snr_raw"] - met_on["snr_raw"]
    print(f"\n>>> SNR reduction from jamming: {snr_drop:.1f} dB")
    print("=" * 60)

    # --- Text report ---
    results = [
        {**met_on, "scenario": "Jammer ON"},
        {**met_off, "scenario": "Jammer OFF"},
    ]
    save_text_report(results, "results/single_report.txt",
                     title="MicFrozen Single-Run Report")
    print("\nText report saved to results/single_report.txt")

    # --- WAV export ---
    if args.export_wav:
        print("\n>>> Exporting WAV audio stages for listening comparison ...")
        # Run with return_signals to extract intermediate audio
        sig_on = run_single(config, seed=seed, return_signals=True)
        config_noise_only = apply_overrides(config, {"jammer.canceling_strategy": "off"})
        sig_noise_only = run_single(config_noise_only, seed=seed, return_signals=True)

        wav_signals = {
            "speech_at_spy": sig_on["speech_at_spy"],
            "spy_noise_only": sig_noise_only["spy_rec"],
            "spy_full": sig_on["spy_rec"],
            "enhanced": sig_on["enhanced"],
        }
        export_wav_stages(wav_signals, "results/audio", config.sim.fs)
        print("WAV files saved to results/audio/")

    # --- Pipeline audio plots ---
    if args.export_audio:
        print("\n>>> Exporting pipeline audio plots ...")
        sig = run_single(config, seed=seed, return_signals=True)
        from scipy.signal import fftconvolve
        from src.channel import ChannelModule
        from strategies.nonlinearity import create_nonlinearity

        channel = ChannelModule(config)
        audible_rirs, ultrasonic_rirs = channel.compute_rir()
        ref_sig = fftconvolve(sig["s_src"], audible_rirs["src_to_ref"])[:len(sig["s_src"])]

        nonlinearity = create_nonlinearity(config.spy_mic.nonlinearity,
                                           config.spy_mic.nonlinearity_params)
        audible_arrival = fftconvolve(sig["s_src"], audible_rirs["src_to_spy"][0])[:len(sig["s_src"])]
        jammer_baseband = sig["s_cancel"] + sig["n_coherent"]
        ultrasonic_arrival = fftconvolve(jammer_baseband,
                                         ultrasonic_rirs["jammer_to_spy"][0])[:len(sig["s_src"])]
        ultrasonic_demod = nonlinearity.apply(ultrasonic_arrival)

        plot_signals = {
            "s_src": sig["s_src"],
            "ref_sig": ref_sig,
            "s_cancel": sig["s_cancel"],
            "n_coherent": sig["n_coherent"],
            "audible_arrival": audible_arrival,
            "ultrasonic_arrival": ultrasonic_arrival,
            "ultrasonic_demod": ultrasonic_demod,
            "spy_rec": sig["spy_rec"],
            "enhanced": sig["enhanced"],
            "jammer_baseband": jammer_baseband,
            "noise_ref": sig["noise_ref"],
        }
        save_pipeline_audio_plots(plot_signals, "results/audio_stages",
                                  config.sim.fs, denoiser_name=config.attacker.denoiser)
        print("Pipeline audio plots saved to results/audio_stages/")


if __name__ == "__main__":
    main()
