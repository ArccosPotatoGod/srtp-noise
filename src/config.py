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


def apply_overrides(base_config: ScenarioConfig, override: Dict) -> ScenarioConfig:
    """Deep-copy base_config and apply key-value overrides.

    Special keys (handled positionally, not via setattr):
      - spy_mic_distance: place spy mics at given distance along x-axis
      - spy_mic_angle: rotate spy mic around source at given angle
      - jammer.coherent_strategy: forces canceling_strategy="off" when coherent="off"
      - jammer.canceling_strategy: only honored when coherent_strategy is active
    """
    import copy
    import math

    config = copy.deepcopy(base_config)
    for key_path, val in override.items():
        if key_path == "spy_mic_distance":
            src = base_config.source.pos
            primary = (src[0] + float(val), src[1], src[2])
            secondary = (primary[0], primary[1] + 0.15, primary[2])
            config.spy_mic.positions = [primary, secondary]
        elif key_path == "spy_mic_angle":
            src = base_config.source.pos
            rad = math.radians(val)
            base_pos = base_config.spy_mic.positions[0]
            dist = math.sqrt(
                (base_pos[0] - src[0]) ** 2 + (base_pos[1] - src[1]) ** 2
            )
            new_x = src[0] + dist * math.cos(rad)
            new_y = src[1] + dist * math.sin(rad)
            config.spy_mic.positions = [
                (new_x, new_y, base_pos[2])
                for base_pos in base_config.spy_mic.positions
            ]
        elif key_path == "jammer.coherent_strategy":
            config.jammer.coherent_strategy = val
            if val == "off":
                config.jammer.canceling_strategy = "off"
        elif key_path == "jammer.canceling_strategy":
            if config.jammer.coherent_strategy != "off":
                config.jammer.canceling_strategy = val
        else:
            parts = key_path.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], val)
    return config
