"""Independent RoboClaw wheel backend; never configures velocity PID settings."""

import logging
import threading
import time
from contextlib import contextmanager

from .config_roboclaw import RoboClawDriveConfig

logger = logging.getLogger(__name__)


class RoboClawDrive:
    """Nonblocking wheel commands suitable for a shared arm/base control loop.

    Refresh commands faster than command_timeout_s. A software watchdog stops
    stale motion. It cannot stop hardware after power loss or a broken USB link.
    Use this object as a context manager to stop and close on exit.
    """

    def __init__(self, config: RoboClawDriveConfig | None = None):
        self.config = config or RoboClawDriveConfig()
        self._controller = None
        self._lock = threading.RLock()
        self._quit = threading.Event()
        self._thread = None
        self._moving = False
        self._last_command = 0.0
        self._fault: BaseException | None = None

    @property
    def is_connected(self) -> bool:
        return self._controller is not None

    def connect(self) -> None:
        with self._lock:
            if self.is_connected:
                raise RuntimeError("RoboClaw is already connected")
            try:
                from basicmicro import Basicmicro
            except ImportError as exc:
                raise ImportError("Install RoboClaw support with: uv sync --extra roboclaw") from exc
            self._controller = Basicmicro(self.config.port, self.config.baud)
            self._fault = None
            try:
                if not self._controller.Open():
                    raise ConnectionError(f"Could not open {self.config.port}")
                self.stop()
                self.get_version()
                self._quit.clear()
                self._thread = threading.Thread(target=self._watchdog, name="roboclaw-watchdog", daemon=True)
                self._thread.start()
            except BaseException:
                self._best_effort_stop()
                try:
                    self._controller.close()
                finally:
                    self._controller = None
                raise

    def disconnect(self) -> None:
        self._quit.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            if not self.is_connected:
                return
            try:
                self.stop()
            finally:
                try:
                    self._controller.close()
                finally:
                    self._controller = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.disconnect()
        else:
            try:
                self.disconnect()
            except BaseException:
                logger.exception("RoboClaw cleanup failed while handling another exception")
        return False

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("RoboClaw is not connected")

    @staticmethod
    def _check_ack(result, operation: str) -> None:
        if not result:
            raise ConnectionError(f"RoboClaw did not acknowledge {operation}")

    def _best_effort_stop(self) -> None:
        if not self.is_connected:
            return
        try:
            self.stop()
        except BaseException:
            logger.exception("Failed to stop RoboClaw motors")

    @contextmanager
    def _operation(self):
        with self._lock:
            self._require_connected()
            try:
                if self._fault is not None:
                    raise RuntimeError("RoboClaw watchdog failed; disconnect and reconnect") from self._fault
                yield
            except BaseException:
                self._best_effort_stop()
                raise

    def stop(self) -> None:
        """Command both wheels to zero; use zero duty as fallback on speed failure."""
        with self._lock:
            self._require_connected()
            try:
                self._check_ack(self._controller.SpeedM1M2(self.config.address, 0, 0), "stop")
            except BaseException:
                self._check_ack(self._controller.DutyM1M2(self.config.address, 0, 0), "fallback stop")
            finally:
                self._moving = False

    def set_wheel_speeds(self, left: int, right: int) -> None:
        with self._operation():
            for value in (left, right):
                if type(value) is not int or abs(value) > self.config.max_speed:
                    raise ValueError(f"Wheel speeds must be integers within +/-{self.config.max_speed}")
            m1, m2 = (left, right) if self.config.left_motor == 1 else (right, left)
            self._check_ack(
                self._controller.SpeedM1M2(
                    self.config.address, m1 * self.config.m1_sign, m2 * self.config.m2_sign
                ),
                "wheel speeds",
            )
            self._last_command = time.monotonic()
            self._moving = bool(left or right)

    def set_wheel_duties(self, left: int, right: int) -> None:
        """Set raw motor power (-32767..32767), bypassing velocity PID control."""
        with self._operation():
            for value in (left, right):
                if type(value) is not int or abs(value) > 32767:
                    raise ValueError("Wheel duties must be integers within +/-32767")
            m1, m2 = (left, right) if self.config.left_motor == 1 else (right, left)
            self._check_ack(
                self._controller.DutyM1M2(
                    self.config.address, m1 * self.config.m1_sign, m2 * self.config.m2_sign
                ),
                "wheel duties",
            )
            self._last_command = time.monotonic()
            self._moving = bool(left or right)

    def forward(self, speed: int) -> None:
        self.set_wheel_speeds(speed, speed)

    def backward(self, speed: int) -> None:
        self.set_wheel_speeds(-speed, -speed)

    def rotate_left(self, speed: int) -> None:
        self.set_wheel_speeds(-speed, speed)

    def rotate_right(self, speed: int) -> None:
        self.set_wheel_speeds(speed, -speed)

    def _read(self, method: str):
        result = getattr(self._controller, method)(self.config.address)
        if not result[0]:
            raise ConnectionError(f"RoboClaw {method} failed")
        return result

    def get_version(self) -> str:
        with self._operation():
            return self._read("ReadVersion")[1].strip()

    def get_battery_voltage(self) -> float:
        with self._operation():
            return self._read("ReadMainBatteryVoltage")[1] / 10.0

    def _wheel_feedback(self, prefix: str, speed: bool) -> dict[str, int]:
        with self._operation():
            values = []
            for motor, sign in ((1, self.config.m1_sign), (2, self.config.m2_sign)):
                _, value, status = self._read(f"{prefix}{motor}")
                if speed:
                    # Speed is an unsigned magnitude; bit 0 indicates reverse.
                    value = -value if status & 1 else value
                elif value >= 2**31:
                    # Encoder counts are an unsigned 32-bit representation.
                    value -= 2**32
                values.append(value * sign)
            left, right = values if self.config.left_motor == 1 else values[::-1]
            return {"left": left, "right": right}

    def get_encoders(self) -> dict[str, int]:
        return self._wheel_feedback("ReadEncM", speed=False)

    def get_speeds(self) -> dict[str, int]:
        return self._wheel_feedback("ReadSpeedM", speed=True)

    def reset_encoders(self) -> None:
        with self._operation():
            self.stop()
            self._check_ack(self._controller.ResetEncoders(self.config.address), "encoder reset")

    def _watchdog(self) -> None:
        interval = min(0.05, self.config.command_timeout_s / 4)
        while not self._quit.wait(interval):
            with self._lock:
                if self._moving and time.monotonic() - self._last_command >= self.config.command_timeout_s:
                    try:
                        self.stop()
                        logger.warning("RoboClaw command timeout: wheels stopped")
                    except BaseException as exc:
                        self._fault = exc
                        logger.exception("RoboClaw watchdog stop failed")
                        return
