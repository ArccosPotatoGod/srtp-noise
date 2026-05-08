"""Runner — single-experiment simulation entry point."""

from typing import Dict

import numpy as np
from scipy.signal import fftconvolve

from src.config import ScenarioConfig
from src.speaker import SpeakerModule
from src.channel import ChannelModule
from src.jammer import JammerModule
from src.spy_mic import SpyMicrophoneModule, build_sniffer_reference
from src.attacker import AttackerModule
from src.evaluator import Evaluator
from strategies.canceling import create_canceling
from strategies.coherent import create_coherent
from strategies.denoiser import create_denoiser
from strategies.asr import create_asr
from strategies.nonlinearity import create_nonlinearity


def run_single(config: ScenarioConfig, seed: int = 0, run_index: int = 0,
               return_signals: bool = False) -> Dict:
    """Run one complete MicFrozen simulation.

    Args:
        config: fully-resolved ScenarioConfig
        seed: RNG seed for reproducibility
        run_index: offset for ASR corruption seeds (used when averaging over multiple runs)
        return_signals: if True, include intermediate signal arrays in the returned dict

    Returns dict with keys:
        snr_raw, snr_enhanced, cwer_raw, cwer_enhanced,
        speech_rms, spy_rms, jamming_rms, cancel_rms, noise_rms,
        hyp_text_raw, hyp_text_enhanced, ref_text
        (+ s_src, speech_at_spy, spy_rec, enhanced, s_cancel, n_coherent, noise_ref
           when return_signals=True)
    """
    rng = np.random.default_rng(seed)

    audio_path = config.source.audio_file or "data/sample.wav"
    spk = SpeakerModule(audio_path, fs=config.sim.fs)
    s_src = spk.get_signal()
    ref_text = spk.get_reference_text()

    channel = ChannelModule(config)
    audible_rirs, ultrasonic_rirs = channel.compute_rir()

    # Speech component at spy position
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
    metrics = evaluator.evaluate(speech_at_spy, spy_1d, enh_1d,
                                 ref_text, raw_hyp, hyp_text)

    min_len = min(len(speech_at_spy), len(spy_1d))
    jamming_residual = spy_1d[:min_len] - speech_at_spy[:min_len]

    result = {
        "snr_raw": metrics["snr_raw"],
        "snr_enhanced": metrics["snr_enhanced"],
        "cwer_raw": metrics["cwer_raw"],
        "cwer_enhanced": metrics["cwer_enhanced"],
        "speech_rms": float(np.sqrt(np.mean(speech_at_spy ** 2))),
        "spy_rms": float(np.sqrt(np.mean(spy_1d ** 2))),
        "jamming_rms": float(np.sqrt(np.mean(jamming_residual ** 2))),
        "cancel_rms": float(np.sqrt(np.mean(s_cancel ** 2))),
        "noise_rms": float(np.sqrt(np.mean(n_coherent ** 2))),
        "hyp_text_raw": raw_hyp[:120],
        "hyp_text_enhanced": hyp_text[:120],
        "ref_text": ref_text[:120],
    }

    if return_signals:
        result.update({
            "s_src": s_src,
            "speech_at_spy": speech_at_spy,
            "spy_rec": spy_1d,
            "enhanced": enh_1d,
            "s_cancel": s_cancel,
            "n_coherent": n_coherent,
            "noise_ref": noise_ref,
        })

    return result
