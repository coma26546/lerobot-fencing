# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import math
import time
from bisect import bisect_right

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import OperatingMode
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..so_follower import SOFollower
from .config_fencing_arm import FencingArmConfig
from .poses import FENCING_ARM_JOINT_KEYS, MOTION_EASINGS, FencingArmPoseStore
from .split_bus import SplitFeetechMotorsBus
from .wraparound_safety import WraparoundSafety

logger = logging.getLogger(__name__)


class FencingArm(SOFollower):
    """Fencing robot arm built on the common SO follower architecture.

    Keeping this as a distinct class gives the arm its own robot identity and
    calibration directory while allowing its hardware behavior to diverge from
    the SO100/SO101 implementation as the design develops.
    """

    config_class = FencingArmConfig
    name = "fencing_arm"

    def __init__(self, config: FencingArmConfig):
        # Initialize the shared Robot lifecycle without constructing the
        # SO100/SO101-specific six-motor bus from SOFollower.__init__.
        Robot.__init__(self, config)
        self.config = config
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = SplitFeetechMotorsBus(
            port_1=config.port_1,
            port_2=config.port_2,
            bus_1_motors={
                "shoulder_yaw": Motor(1, "sts3250", norm_mode),
                "shoulder_pitch": Motor(2, "sts3095", norm_mode),
            },
            bus_2_motors={
                "elbow_pitch": Motor(3, "sts3250", norm_mode),
                "elbow_roll": Motor(4, "sts3250", norm_mode),
                "wrist_pitch": Motor(5, "sts3215", norm_mode),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        pose_dir = config.pose_dir if config.pose_dir is not None else self.calibration_dir / "poses"
        coordinate_system = "motor_degrees" if config.use_degrees else "normalized_range_m100_100"
        self.pose_store = FencingArmPoseStore(
            pose_dir / f"{self.id}.json",
            coordinate_system=coordinate_system,
            shoulder_pitch_gear_ratio=config.shoulder_pitch_gear_ratio,
        )
        self.wraparound_safety = WraparoundSafety(
            low_threshold=config.wrap_low_threshold,
            high_threshold=config.wrap_high_threshold,
        )
        self._last_safe_motor_observation: dict[str, float] = {}
        self._last_sent_action: RobotAction = {}
        self.telemetry_recorder = None

    @check_if_already_connected
    def connect_passive(self, calibrate: bool = True) -> None:
        """Connect and configure the arm while leaving motor torque disabled."""
        self.bus.connect()
        if calibrate and not self.is_calibrated:
            self.calibrate()
        for camera in self.cameras.values():
            camera.connect()

        self.bus.disable_torque()
        self.bus.configure_motors(
            maximum_acceleration=self.config.acceleration,
            acceleration=self.config.acceleration,
        )
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", motor, self.config.position_p_coefficient)
            self.bus.write("I_Coefficient", motor, self.config.position_i_coefficient)
            self.bus.write("D_Coefficient", motor, self.config.position_d_coefficient)
            self.bus.write("Torque_Limit", motor, self.config.torque_limit)
        logger.info(f"{self} connected with torque disabled.")

    def prepare_for_motion(self) -> dict[str, float]:
        """Set the current position as the goal before enabling torque."""
        observation = self.get_observation()
        current_pose = {key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS}
        self.send_action(current_pose)
        self.bus.enable_torque()
        return current_pose

    def move_to_rest(self) -> None:
        """Lower to the taught, mechanically supported rest pose before normal shutdown."""
        self.move_to_pose(
            "rest",
            max_step=1.0,
            fps=30.0,
            timeout_s=120.0,
            tolerance=5.0,
            duration_s=3.0,
            easing="smootherstep",
        )

    def _update_wraparound_safety(self, raw_positions: dict[str, int | float]) -> None:
        if not self.config.wraparound_protection:
            return

        positions = {motor: int(position) for motor, position in raw_positions.items()}
        triggered, released = self.wraparound_safety.update(positions)
        if triggered:
            logger.warning(
                "Encoder wraparound detected on %s. Holding the last safe motor goal and suppressing "
                "new actions until feedback returns across the boundary and moves inward.",
                ", ".join(sorted(triggered)),
            )
        if released:
            logger.info("Encoder wraparound hold released for: %s", ", ".join(sorted(released)))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation = super().get_observation()
        raw_positions = self.bus.sync_read("Present_Position", normalize=False)
        self._update_wraparound_safety(raw_positions)
        if self.wraparound_safety.is_latched:
            for motor in self.wraparound_safety.latched:
                key = f"{motor}.pos"
                if key in self._last_safe_motor_observation:
                    observation[key] = self._last_safe_motor_observation[key]
        else:
            self._last_safe_motor_observation = {
                key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS
            }
        if self.telemetry_recorder is not None:
            positions = {key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS}
            self.telemetry_recorder.capture(self.bus, positions)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.config.wraparound_protection:
            return super().send_action(action)

        raw_positions = self.bus.sync_read("Present_Position", normalize=False)
        self._update_wraparound_safety(raw_positions)
        if self.wraparound_safety.is_latched:
            # Do not write a new goal. The servo retains and actively holds the
            # last goal that was sent before the wrap was detected.
            return dict(self._last_sent_action or self._last_safe_motor_observation)

        sent_action = super().send_action(action)
        self._last_sent_action = dict(sent_action)
        return sent_action

    def record_pose(self, name: str) -> dict[str, float]:
        """Read and save the arm's current position under a stable name."""
        observation = self.get_observation()
        pose = {key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS}
        self.pose_store.save(name, pose)
        return pose

    def move_to_pose(
        self,
        name: str,
        max_step: float = 2.0,
        fps: float = 30.0,
        timeout_s: float = 30.0,
        tolerance: float = 0.5,
        duration_s: float | None = None,
        easing: str = "smoothstep",
        joint_timing: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Move smoothly from the current position to a recorded pose."""
        if max_step <= 0:
            raise ValueError("max_step must be greater than zero")
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if tolerance <= 0:
            raise ValueError("tolerance must be greater than zero")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be greater than zero")
        if easing not in MOTION_EASINGS:
            raise ValueError(f"easing must be one of {MOTION_EASINGS}")
        joint_timing = joint_timing or {}

        target = self.pose_store.load(name)
        deadline = time.monotonic() + timeout_s
        observation = self.get_observation()
        start = {key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS}
        trajectory_deltas = {key: target[key] - start[key] for key in FENCING_ARM_JOINT_KEYS}
        max_easing_slope = {"linear": 1.0, "smoothstep": 1.5, "smootherstep": 1.875}[easing]
        required_steps = []
        for key, delta in trajectory_deltas.items():
            timing = joint_timing.get(key, {})
            active_fraction = timing.get("end_fraction", 1.0) - timing.get("start_fraction", 0.0)
            required_steps.append(math.ceil(abs(delta) * max_easing_slope / (max_step * active_fraction)))
        duration_steps = math.ceil(duration_s * fps) if duration_s is not None else 0
        trajectory_steps = max(1, duration_steps, *required_steps)
        trajectory_step = 0
        logger.info(
            "Moving FencingArm to pose %r in %.2fs with %s easing",
            name,
            trajectory_steps / fps,
            easing,
        )
        next_progress_log = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            observation = self.get_observation()
            current = {key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS}
            deltas = {key: target[key] - current[key] for key in FENCING_ARM_JOINT_KEYS}
            if max(abs(delta) for delta in deltas.values()) <= tolerance:
                logger.info("FencingArm reached pose %r", name)
                return target
            if time.monotonic() >= next_progress_log:
                worst_joint = max(deltas, key=lambda key: abs(deltas[key]))
                logger.info(
                    "Moving to %r: waiting on %s (error %.2f, tolerance %.2f)",
                    name,
                    worst_joint,
                    deltas[worst_joint],
                    tolerance,
                )
                next_progress_log = time.monotonic() + 2.0
            if trajectory_step < trajectory_steps:
                fraction = (trajectory_step + 1) / trajectory_steps
                action = {}
                for key in FENCING_ARM_JOINT_KEYS:
                    timing = joint_timing.get(key, {})
                    start_fraction = timing.get("start_fraction", 0.0)
                    end_fraction = timing.get("end_fraction", 1.0)
                    local_fraction = (fraction - start_fraction) / (end_fraction - start_fraction)
                    local_fraction = max(0.0, min(1.0, local_fraction))
                    eased_fraction = self._ease(local_fraction, easing)
                    action[key] = start[key] + trajectory_deltas[key] * eased_fraction
                sent_action = self.send_action(action)
                accepted = all(
                    abs(float(sent_action[key]) - action[key]) < 1e-6
                    for key in FENCING_ARM_JOINT_KEYS
                )
                if accepted:
                    trajectory_step += 1
            time.sleep(1 / fps)

        remaining_error = max(abs(target[key] - current[key]) for key in FENCING_ARM_JOINT_KEYS)
        raise TimeoutError(
            f"FencingArm did not reach pose {name!r} within {timeout_s} seconds "
            f"(largest remaining error: {remaining_error:.2f})"
        )

    def play_recording(self, name: str, fps: float = 60.0, max_step: float = 6.0) -> None:
        """Replay a trimmed, time-scaled demonstration using interpolated position commands."""
        if fps <= 0 or max_step <= 0:
            raise ValueError("fps and max_step must be greater than zero")
        recording = self.pose_store.load_recording(name)
        samples = self._smooth_recording_samples(recording.samples, recording.smoothing_window)
        start_time = recording.trim_start_s
        end_time = samples[-1]["time_s"] - recording.trim_end_s
        maximum_joint_delay = max(recording.joint_delays_s.values(), default=0.0)
        duration = (end_time - start_time + maximum_joint_delay) / recording.playback_speed
        output_steps = max(1, math.ceil(duration * fps))
        observation = self.get_observation()
        previous = {key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS}
        sample_times = [sample["time_s"] for sample in samples]
        logger.info(
            "Playing recording %r in %.2fs at %.2fx speed with smoothing window %d",
            name,
            duration,
            recording.playback_speed,
            recording.smoothing_window,
        )
        for step in range(output_steps + 1):
            output_source_time = start_time + (step / output_steps) * (
                end_time - start_time + maximum_joint_delay
            )
            target = {}
            for key in FENCING_ARM_JOINT_KEYS:
                source_time = max(
                    start_time,
                    min(end_time, output_source_time - recording.joint_delays_s.get(key, 0.0)),
                )
                right = min(len(samples) - 1, bisect_right(sample_times, source_time))
                left = max(0, right - 1)
                left_sample = samples[left]
                right_sample = samples[right]
                span = right_sample["time_s"] - left_sample["time_s"]
                fraction = 0.0 if span == 0 else (source_time - left_sample["time_s"]) / span
                target[key] = left_sample["positions"][key] + (
                    right_sample["positions"][key] - left_sample["positions"][key]
                ) * fraction
            action = {
                key: previous[key] + max(-max_step, min(max_step, target[key] - previous[key]))
                for key in FENCING_ARM_JOINT_KEYS
            }
            previous = {key: float(value) for key, value in self.send_action(action).items()}
            time.sleep(1 / fps)

    @staticmethod
    def _smooth_recording_samples(samples: list[dict], window: int) -> list[dict]:
        if window == 1:
            return samples
        radius = window // 2
        smoothed = []
        for index, sample in enumerate(samples):
            neighbors = samples[max(0, index - radius) : min(len(samples), index + radius + 1)]
            smoothed.append(
                {
                    "time_s": sample["time_s"],
                    "positions": {
                        key: sum(neighbor["positions"][key] for neighbor in neighbors) / len(neighbors)
                        for key in FENCING_ARM_JOINT_KEYS
                    },
                }
            )
        return smoothed

    @staticmethod
    def _ease(fraction: float, easing: str) -> float:
        if easing == "linear":
            return fraction
        if easing == "smoothstep":
            return fraction * fraction * (3 - 2 * fraction)
        if easing == "smootherstep":
            return fraction**3 * (fraction * (fraction * 6 - 15) + 10)
        raise ValueError(f"Unsupported easing: {easing}")

    def calibrate(self) -> None:
        """Calibrate all five joints without SOFollower's wrist-roll assumption."""
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
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
        print(
            "Move each joint sequentially through its entire safe range of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {
            motor: MotorCalibration(
                id=definition.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
            for motor, definition in self.bus.motors.items()
        }
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)
