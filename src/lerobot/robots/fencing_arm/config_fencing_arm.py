# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("fencing_arm")
@dataclass
class FencingArmConfig(RobotConfig):
    """Configuration for the FencingArm robot.

    The first hardware revision uses the SO follower connection, camera, and
    safety settings. Fencing-specific settings can be added here without
    changing the SO100 or SO101 configurations.
    """

    # Bus 1 powers motor IDs 1-2; bus 2 powers motor IDs 3-5.
    port_1: str
    port_2: str
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True

    # Mechanical reduction between the shoulder-pitch motor and joint output.
    # Joint commands currently remain in the Feetech motor coordinate space;
    # this value is available for future output-space kinematics/conversion.
    shoulder_pitch_gear_ratio: float = 2.0

    # Position-loop tuning for the weapon-bearing arm. The generic SO follower
    # deliberately halves the Feetech default P gain to reduce tabletop-arm
    # shake; that makes this heavier arm feel soft and unable to hold a saber.
    position_p_coefficient: int = 32
    position_i_coefficient: int = 2
    position_d_coefficient: int = 32
    torque_limit: int = 1000
    acceleration: int = 254

    # Optional override for named-pose files. By default poses are stored next
    # to this robot's calibration data in a dedicated `poses` directory.
    pose_dir: Path | None = None

    # Hold the last safe goal if a 12-bit encoder wraps between the high and
    # low ends of its 0..4095 range.
    wraparound_protection: bool = True
    wrap_low_threshold: int = 256
    wrap_high_threshold: int = 3839

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.port_1 == self.port_2:
            raise ValueError("port_1 and port_2 must identify different motor buses")
        if self.shoulder_pitch_gear_ratio <= 0:
            raise ValueError("shoulder_pitch_gear_ratio must be greater than zero")
        for name, value in (
            ("position_p_coefficient", self.position_p_coefficient),
            ("position_i_coefficient", self.position_i_coefficient),
            ("position_d_coefficient", self.position_d_coefficient),
            ("acceleration", self.acceleration),
        ):
            if not 0 <= value <= 254:
                raise ValueError(f"{name} must be between 0 and 254")
        if not 0 <= self.torque_limit <= 1000:
            raise ValueError("torque_limit must be between 0 and 1000")
        if not 0 <= self.wrap_low_threshold < self.wrap_high_threshold <= 4095:
            raise ValueError("wrap thresholds must satisfy 0 <= low < high <= 4095")
