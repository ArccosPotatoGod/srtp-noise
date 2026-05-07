"""Tests for core module integration and strategy correctness."""

import numpy as np
import pytest


class TestConfig:
    def test_load_base_yaml(self):
        from src.config import load_config
        config = load_config("configs/base.yaml")
        assert config.sim.fs == 16000
        assert config.room.rt60 == 0.3
        assert len(config.spy_mic.positions) == 1

    def test_config_to_dict_roundtrip(self):
        from src.config import load_config
        config = load_config("configs/base.yaml")
        d = config.to_dict()
        assert d["sim"]["fs"] == 16000


class TestSpeaker:
    def test_load_generates_signal(self):
        from src.speaker import SpeakerModule
        import soundfile as sf
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(16000, dtype=np.float32), 16000)
            spk = SpeakerModule(f.name, fs=16000)
            sig = spk.get_signal()
            assert sig.shape == (16000,)
            assert sig.dtype == np.float32


class TestChannel:
    def test_compute_rir_shapes(self):
        from src.config import load_config
        from src.channel import ChannelModule
        config = load_config("configs/base.yaml")
        channel = ChannelModule(config)
        audible, ultrasonic = channel.compute_rir()
        assert "src_to_ref" in audible
        assert "jammer_to_spy" in ultrasonic
        assert len(ultrasonic["jammer_to_spy"]) == 1


class TestJammer:
    def test_phase_inversion(self):
        from strategies.canceling import PhaseInversionCanceling
        ref = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        cancel = PhaseInversionCanceling(gain=1.0)
        out = cancel.compute(ref)
        assert out.shape == ref.shape


class TestCoherentNoise:
    def test_fixed_weight_output_shape(self):
        from strategies.coherent import FixedWeightCoherentNoise
        speech = np.random.randn(16000).astype(np.float32)
        noise = FixedWeightCoherentNoise(rng=np.random.default_rng(42))
        out = noise.compute(speech)
        assert out.shape == speech.shape

    def test_baselines_output_shape(self):
        from strategies.coherent import (
            BaselineGaussianNoise, BaselineSweepingNoise, BaselineHoppingNoise,
        )
        speech = np.random.randn(16000).astype(np.float32)
        for cls in [BaselineGaussianNoise, BaselineSweepingNoise, BaselineHoppingNoise]:
            out = cls(rng=np.random.default_rng(42)).compute(speech)
            assert out.shape == speech.shape


class TestSpyMic:
    def test_capture_single_channel(self):
        from strategies.nonlinearity import PassThroughNonlinearity
        from src.spy_mic import SpyMicrophoneModule
        T = 8000
        rir_len = 100
        audible = {
            "src_to_ref": np.ones(rir_len, dtype=np.float32),
            "src_to_spy": [np.ones(rir_len, dtype=np.float32) * 0.5],
        }
        ultrasonic = {
            "jammer_to_spy": [np.ones(rir_len, dtype=np.float32) * 0.3],
        }
        nl = PassThroughNonlinearity()
        spymic = SpyMicrophoneModule(audible, ultrasonic, nl)
        s = np.ones(T, dtype=np.float32) * 0.1
        cancel = np.zeros(T, dtype=np.float32)
        noise = np.zeros(T, dtype=np.float32)
        rec = spymic.capture(s, cancel, noise)
        assert rec.ndim == 1


class TestEvaluator:
    def test_snr_perfect_match(self):
        from src.evaluator import _compute_snr
        sig = np.ones(1000, dtype=np.float32)
        assert _compute_snr(sig, sig) > 50.0

    def test_snr_noise(self):
        from src.evaluator import _compute_snr
        sig = np.ones(1000, dtype=np.float32)
        noisy = sig + np.random.randn(1000).astype(np.float32) * 0.1
        snr = _compute_snr(sig, noisy)
        assert snr < 30.0

    def test_cwer_perfect(self):
        from src.evaluator import _compute_cwer
        assert _compute_cwer("hello world", "hello world") == 0.0

    def test_cwer_all_wrong(self):
        from src.evaluator import _compute_cwer
        assert _compute_cwer("hello world", "foo bar") == 100.0
