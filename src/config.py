"""Scenario configuration via YAML or dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class RoomConfig:
    dim: Tuple[float, float, float] = (6.0, 5.0, 3.0)
    rt60: float = 0.3
    temperature: float = 20.0
    humidity: float = 50.0


@dataclass
class SourceConfig:
    pos: Tuple[float, float, float] = (1.0, 2.5, 1.5)
    audio_dataset: str = "librispeech"
    audio_file: Optional[str] = None


@dataclass
class JammerConfig:
    pos_spk: Tuple[float, float, float] = (1.2, 2.5, 1.5)
    ref_mic_pos: Tuple[float, float, float] = (1.1, 2.5, 1.5)
    canceling_strategy: str = "phase_inversion"
    coherent_strategy: str = "fixed_weight"
    system_gain_db: float = 36.0
    canceling_params: Dict[str, Any] = field(default_factory=lambda: {
        "gain": 1.0,
        "precompensate": True,
    })
    coherent_params: Dict[str, Any] = field(default_factory=lambda: {
        "time_coupling": True,
        "freq_coupling": True,
        "mixing_dim": 3,
    })


@dataclass
class SpyMicConfig:
    positions: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(4.0, 2.5, 1.5)]
    )
    nonlinearity: str = "polynomial"
    nonlinearity_params: Dict[str, Any] = field(default_factory=lambda: {
        "coeff": [1.0, 0.1, 0.0],
    })


@dataclass
class AttackerConfig:
    denoiser: str = "ica"
    asr: str = "whisper_tiny"
    denoiser_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UltrasonicConfig:
    carrier_freq: float = 39000.0
    attenuation_db_per_m: float = 1.5
    apply_to_rir: bool = True


@dataclass
class SimConfig:
    fs: int = 16000
    duration: Optional[float] = None


@dataclass
class ScenarioConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    jammer: JammerConfig = field(default_factory=JammerConfig)
    spy_mic: SpyMicConfig = field(default_factory=SpyMicConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    ultrasonic: UltrasonicConfig = field(default_factory=UltrasonicConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict) -> "ScenarioConfig":
        return cls(
            sim=SimConfig(**d.get("sim", {})),
            room=RoomConfig(**d.get("room", {})),
            source=SourceConfig(**d.get("source", {})),
            jammer=JammerConfig(**d.get("jammer", {})),
            spy_mic=SpyMicConfig(**d.get("spy_mic", {})),
            attacker=AttackerConfig(**d.get("attacker", {})),
            ultrasonic=UltrasonicConfig(**d.get("ultrasonic", {})),
        )

    def to_dict(self) -> Dict:
        """Recursively convert to plain dict for serialization."""
        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return tuple(_convert(x) for x in obj)
            elif isinstance(obj, list):
                return [_convert(x) for x in obj]
            elif hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in obj.__dict__.items()}
            return obj
        return _convert(self.__dict__)


def load_config(path: str) -> ScenarioConfig:
    return ScenarioConfig.from_yaml(path)

def load_grid_config(path: str) -> Dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw.get("grid", {})
