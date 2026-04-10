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

import json
import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_fencing_bot import FencingBotConfig

logger = logging.getLogger(__name__)


class FencingBot(Robot):
    """
    Single-arm fencing robot follower: Feetech STS3215 chain (same layout as SO-100 style arms).
    Calibration follows the SO-100 style: move joints manually through ranges when prompted.

    Use `lerobot-fencing-positions` to capture named poses and recall them by index from the terminal.
    """

    config_class = FencingBotConfig
    name = "fencing_bot"

    def __init__(self, config: FencingBotConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def saved_positions_fpath(self) -> Path:
        if self.config.saved_positions_path is not None:
            return Path(self.config.saved_positions_path)
        rid = self.id if self.id else "default"
        return HF_LEROBOT_HOME / "fencing_bot_positions" / f"{rid}.json"

    def joint_positions_from_observation(self, observation: RobotObservation) -> RobotAction:
        """Joint goal keys (`motor.pos`) only; excludes camera frames and other keys."""
        return {k: float(v) for k, v in observation.items() if k.endswith(".pos")}

    def load_saved_poses(self) -> list[dict[str, Any]]:
        path = self.saved_positions_fpath
        if not path.is_file():
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        poses = data.get("poses", [])
        if not isinstance(poses, list):
            raise ValueError(f"Invalid poses file {path}: 'poses' must be a list")
        return poses

    def write_saved_poses(self, poses: list[dict[str, Any]]) -> None:
        path = self.saved_positions_fpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"poses": poses}, f, indent=2)
        logger.info("Saved %d pose(s) to %s", len(poses), path)

    def validate_pose_joints(self, joints: dict[str, Any]) -> None:
        expected = set(self.action_features)
        got = set(joints)
        if got != expected:
            missing = expected - got
            extra = got - expected
            raise ValueError(f"Pose joints must match robot motors. Missing: {missing}, extra: {extra}")

    def capture_pose(self, name: str, poses: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Append current joint pose as `{"name": ..., "joints": {...}}` and persist. Pass `poses` to update a working list."""
        obs = self.get_observation()
        joints = self.joint_positions_from_observation(obs)
        self.validate_pose_joints(joints)
        updated = list(poses) if poses is not None else self.load_saved_poses()
        entry = {"name": name.strip() or f"pose_{len(updated) + 1}", "joints": joints}
        updated.append(entry)
        self.write_saved_poses(updated)
        return updated

    def go_to_saved_pose(self, pose_index_one_based: int, poses: list[dict[str, Any]] | None = None) -> RobotAction:
        """Move arm to pose #1, #2, ... (1-based). Uses `send_action` (position hold)."""
        plist = poses if poses is not None else self.load_saved_poses()
        if not plist:
            raise ValueError("No saved poses. Use capture <name> first (or check your poses file).")
        if pose_index_one_based < 1 or pose_index_one_based > len(plist):
            raise IndexError(f"Pose index must be 1..{len(plist)}, got {pose_index_one_based}")
        raw = plist[pose_index_one_based - 1]["joints"]
        joints = {k: float(v) for k, v in raw.items()}
        self.validate_pose_joints(joints)
        return self.send_action(joints)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
