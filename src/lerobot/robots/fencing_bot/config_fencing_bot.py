#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class FencingBotArmConfig:
    """Configuration for the single fencing-arm follower (Feetech bus + optional cameras)."""

    port: str

    disable_torque_on_disconnect: bool = True

    # Limits the magnitude of the relative positional target vector for safety.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    use_degrees: bool = True

    # If set, JSON file for named poses (see lerobot-fencing-positions). Otherwise HF_LEROBOT_HOME/fencing_bot_positions/<id>.json
    saved_positions_path: Path | None = None


@RobotConfig.register_subclass("fencing_bot")
@dataclass
class FencingBotConfig(RobotConfig, FencingBotArmConfig):
    pass
