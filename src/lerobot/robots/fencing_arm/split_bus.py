# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from collections.abc import Sequence
from contextlib import contextmanager
from pprint import pformat

from lerobot.motors import Motor, MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.utils import enter_pressed, move_cursor_up


class SplitFeetechMotorsBus:
    """Expose two independently powered Feetech buses as one logical bus."""

    def __init__(
        self,
        port_1: str,
        port_2: str,
        bus_1_motors: dict[str, Motor],
        bus_2_motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        calibration = calibration or {}
        self.bus_1 = FeetechMotorsBus(
            port=port_1,
            motors=bus_1_motors,
            calibration={name: calibration[name] for name in bus_1_motors if name in calibration},
        )
        self.bus_2 = FeetechMotorsBus(
            port=port_2,
            motors=bus_2_motors,
            calibration={name: calibration[name] for name in bus_2_motors if name in calibration},
        )
        self.motors = {**bus_1_motors, **bus_2_motors}

    @property
    def is_connected(self) -> bool:
        return self.bus_1.is_connected and self.bus_2.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus_1.is_calibrated and self.bus_2.is_calibrated

    def connect(self) -> None:
        self.bus_1.connect()
        try:
            self.bus_2.connect()
        except Exception:
            self.bus_1.disconnect(disable_torque=False)
            raise

    def disconnect(self, disable_torque: bool = True) -> None:
        errors = []
        for bus in (self.bus_1, self.bus_2):
            if not bus.is_connected:
                continue
            try:
                bus.disconnect(disable_torque)
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(f"Failed to disconnect one or more FencingArm buses: {errors}")

    def configure_motors(self, **kwargs) -> None:
        self.bus_1.configure_motors(**kwargs)
        self.bus_2.configure_motors(**kwargs)

    def write(self, data_name: str, motor: str, value, **kwargs) -> None:
        self._bus_for_motor(motor).write(data_name, motor, value, **kwargs)

    def sync_read(
        self,
        data_name: str,
        motors: str | Sequence[str] | None = None,
        **kwargs,
    ) -> dict[str, int | float]:
        selected = self._motor_names(motors)
        result = {}
        for bus in (self.bus_1, self.bus_2):
            names = [name for name in selected if name in bus.motors]
            if names:
                result.update(bus.sync_read(data_name, names, **kwargs))
        return result

    def sync_write(self, data_name: str, values: dict[str, int | float], **kwargs) -> None:
        for bus in (self.bus_1, self.bus_2):
            bus_values = {name: value for name, value in values.items() if name in bus.motors}
            if bus_values:
                bus.sync_write(data_name, bus_values, **kwargs)

    def disable_torque(self, motors: str | Sequence[str] | None = None, **kwargs) -> None:
        self._apply_to_selected("disable_torque", motors, **kwargs)

    def enable_torque(self, motors: str | Sequence[str] | None = None, **kwargs) -> None:
        self._apply_to_selected("enable_torque", motors, **kwargs)

    @contextmanager
    def torque_disabled(self, motors: str | Sequence[str] | None = None):
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def setup_motor(self, motor: str) -> None:
        self._bus_for_motor(motor).setup_motor(motor)

    def set_half_turn_homings(self) -> dict[str, int | float]:
        return {**self.bus_1.set_half_turn_homings(), **self.bus_2.set_half_turn_homings()}

    def write_calibration(self, calibration: dict[str, MotorCalibration]) -> None:
        for bus in (self.bus_1, self.bus_2):
            bus.write_calibration({name: calibration[name] for name in bus.motors})

    def record_ranges_of_motion(
        self, motors: str | Sequence[str] | None = None, display_values: bool = True
    ) -> tuple[dict[str, int | float], dict[str, int | float]]:
        motor_names = self._motor_names(motors)
        positions = self.sync_read("Present_Position", motor_names, normalize=False)
        mins = positions.copy()
        maxes = positions.copy()

        while True:
            positions = self.sync_read("Present_Position", motor_names, normalize=False)
            mins = {motor: min(positions[motor], mins[motor]) for motor in motor_names}
            maxes = {motor: max(positions[motor], maxes[motor]) for motor in motor_names}
            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motor_names:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")
            if enter_pressed():
                break
            if display_values:
                move_cursor_up(len(motor_names) + 3)

        unchanged = [motor for motor in motor_names if mins[motor] == maxes[motor]]
        if unchanged:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(unchanged)}")
        return mins, maxes

    def _bus_for_motor(self, motor: str) -> FeetechMotorsBus:
        if motor in self.bus_1.motors:
            return self.bus_1
        if motor in self.bus_2.motors:
            return self.bus_2
        raise KeyError(f"Unknown FencingArm motor: {motor}")

    def _motor_names(self, motors: str | Sequence[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        if isinstance(motors, str):
            return [motors]
        return list(motors)

    def _apply_to_selected(self, method: str, motors: str | Sequence[str] | None, **kwargs) -> None:
        selected = self._motor_names(motors)
        for bus in (self.bus_1, self.bus_2):
            names = [name for name in selected if name in bus.motors]
            if names:
                getattr(bus, method)(names, **kwargs)
