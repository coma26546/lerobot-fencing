"""Compose joint-position and wheel-velocity hardware into one LeRobot robot."""

import logging
import math
from numbers import Real

from lerobot.types import RobotAction, RobotObservation

from ..fencing_arm import FencingArm
from ..roboclaw import RoboClawDrive
from ..robot import Robot
from .config_fencing_robot import FencingRobotConfig

logger = logging.getLogger(__name__)
BASE_ACTION_KEYS = ("base.left_speed", "base.right_speed")


class FencingRobot(Robot):
    config_class = FencingRobotConfig
    name = "fencing_robot"

    def __init__(self, config: FencingRobotConfig):
        super().__init__(config)
        self.config = config
        # Use the arm's own ID and calibration path without migrating any files.
        self.arm = FencingArm(config.arm)
        self.drive = RoboClawDrive(config.drive)

    @property
    def action_features(self) -> dict:
        return {**self.arm.action_features, **dict.fromkeys(BASE_ACTION_KEYS, float)}

    @property
    def observation_features(self) -> dict:
        return {
            **self.arm.observation_features,
            **dict.fromkeys(BASE_ACTION_KEYS, float),
            "base.left_encoder": float,
            "base.right_encoder": float,
            "base.battery_voltage": float,
        }

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and self.drive.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.arm.is_connected or self.drive.is_connected:
            raise RuntimeError("FencingRobot is already or partially connected; disconnect first")
        try:
            self.drive.connect()  # Wheels are stopped before arm setup/calibration.
            self.arm.connect_passive(calibrate=calibrate)
            if calibrate:
                self.arm.prepare_for_motion()
        except BaseException:
            self._cleanup_after_error()
            raise

    def calibrate(self) -> None:
        try:
            self.drive.stop()
            self.arm.calibrate()
        except BaseException:
            self._cleanup_after_error()
            raise

    def configure(self) -> None:
        # Arm setup is performed by connect_passive; avoid SOFollower defaults.
        self._require_connected()
        self.stop()

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Both arm and drive must be connected")

    def _cleanup_after_error(self) -> None:
        try:
            self.disconnect()
        except BaseException:
            logger.exception("FencingRobot cleanup failed while handling another exception")

    def stop(self) -> None:
        """Stop the chassis. The arm retains its last position goal."""
        try:
            self.drive.stop()
        except BaseException:
            self._cleanup_after_error()
            raise

    def disconnect(self) -> None:
        """Stop/close the drive, then release arm buses and cameras even after failures."""
        errors = []
        try:
            self.drive.disconnect()
        except BaseException as exc:
            errors.append(exc)
        # Bypass the arm's connected decorator so partial connection is cleaned up.
        try:
            self.arm.bus.disconnect(self.config.arm.disable_torque_on_disconnect)
        except BaseException as exc:
            errors.append(exc)
        for camera in self.arm.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise BaseExceptionGroup("Failed to disconnect fencing hardware", errors)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.disconnect()
        else:
            self._cleanup_after_error()
        return False

    def validate_action(self, action: RobotAction) -> RobotAction:
        """Validate before writes; round finite wheel velocities to integer counts/s."""
        unknown = set(action) - set(self.action_features)
        if unknown:
            raise ValueError(f"Unknown action keys: {sorted(unknown)}")
        result = {}
        for key, value in action.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite numeric value")
            result[key] = float(value)
        for key in BASE_ACTION_KEYS:
            value = result.get(key, 0.0)
            if abs(value) > self.config.drive.max_speed:
                raise ValueError(f"{key} exceeds the wheel speed limit {self.config.drive.max_speed}")
            result[key] = round(value)
        return result

    def send_action(self, action: RobotAction) -> RobotAction:
        try:
            self._require_connected()
            validated = self.validate_action(action)
            arm_action = {key: value for key, value in validated.items() if key not in BASE_ACTION_KEYS}
            sent = self.arm.send_action(arm_action) if arm_action else {}
            self.drive.set_wheel_speeds(*(validated[key] for key in BASE_ACTION_KEYS))
            return {**sent, **{key: float(validated[key]) for key in BASE_ACTION_KEYS}}
        except BaseException:
            self._cleanup_after_error()
            raise

    def get_observation(self) -> RobotObservation:
        try:
            self._require_connected()
            observation = self.arm.get_observation()
            encoders = self.drive.get_encoders()
            speeds = self.drive.get_speeds()
            return {
                **observation,
                "base.left_encoder": float(encoders["left"]),
                "base.right_encoder": float(encoders["right"]),
                "base.left_speed": float(speeds["left"]),
                "base.right_speed": float(speeds["right"]),
                "base.battery_voltage": self.drive.get_battery_voltage(),
            }
        except BaseException:
            self._cleanup_after_error()
            raise
