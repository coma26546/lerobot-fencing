import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RoboClawDriveConfig:
    """Wheel speeds are integer encoder counts/second, positive toward the front."""

    port: str = "/dev/ttyACM0"
    baud: int = 38400
    address: int = 0x80
    left_motor: int = 1
    m1_sign: int = 1
    m2_sign: int = 1
    max_speed: int = 500
    command_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if not self.port or self.baud <= 0:
            raise ValueError("A port and positive baud rate are required")
        if not 0x80 <= self.address <= 0x87:
            raise ValueError("address must be between 0x80 and 0x87")
        if self.left_motor not in (1, 2):
            raise ValueError("left_motor must be 1 or 2")
        if self.m1_sign not in (-1, 1) or self.m2_sign not in (-1, 1):
            raise ValueError("Motor signs must be -1 or 1")
        if type(self.max_speed) is not int or not 0 < self.max_speed <= 2**31 - 1:
            raise ValueError("max_speed must be a positive signed 32-bit integer")
        if not math.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be finite and positive")
