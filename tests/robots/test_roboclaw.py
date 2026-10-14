import sys
import time
from unittest.mock import MagicMock, call

import pytest

from lerobot.robots.roboclaw import RoboClawDrive, RoboClawDriveConfig


@pytest.fixture
def controller(monkeypatch):
    rc = MagicMock()
    rc.Open.return_value = True
    rc.SpeedM1M2.return_value = True
    rc.DutyM1M2.return_value = True
    rc.ReadVersion.return_value = (True, "test firmware\n")
    module = MagicMock()
    module.Basicmicro.return_value = rc
    monkeypatch.setitem(sys.modules, "basicmicro", module)
    return rc


def test_mapping_and_signed_feedback(controller):
    controller.ReadEncM1.return_value = (1, 2**32 - 10, 128)
    controller.ReadEncM2.return_value = (1, 20, 128)
    controller.ReadSpeedM1.return_value = (1, 30, 1)
    controller.ReadSpeedM2.return_value = (1, 40, 0)
    with RoboClawDrive(RoboClawDriveConfig(left_motor=2, m1_sign=-1)) as drive:
        drive.set_wheel_speeds(100, -200)
        controller.SpeedM1M2.assert_called_with(0x80, 200, 100)
        assert drive.get_encoders() == {"left": 20, "right": 10}
        assert drive.get_speeds() == {"left": 40, "right": 30}
    assert controller.method_calls[-2:] == [call.SpeedM1M2(0x80, 0, 0), call.close()]


@pytest.mark.parametrize("left,right", [(501, 0), (0, -501), (1.0, 0), (True, 0)])
def test_invalid_speed_stops(controller, left, right):
    with RoboClawDrive() as drive:
        drive.forward(100)
        with pytest.raises(ValueError):
            drive.set_wheel_speeds(left, right)
        controller.SpeedM1M2.assert_called_with(0x80, 0, 0)


def test_full_duty_mapping_and_watchdog(controller):
    with RoboClawDrive(
        RoboClawDriveConfig(left_motor=2, m1_sign=-1, command_timeout_s=0.02)
    ) as drive:
        drive.set_wheel_duties(32767, -32767)
        controller.DutyM1M2.assert_called_with(0x80, 32767, 32767)
        deadline = time.monotonic() + 1
        while drive._moving and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not drive._moving
        controller.SpeedM1M2.assert_called_with(0x80, 0, 0)


@pytest.mark.parametrize("value", [32768, -32768, True, 1.0])
def test_invalid_duty_stops(controller, value):
    with RoboClawDrive() as drive:
        drive.set_wheel_duties(32767, 32767)
        with pytest.raises(ValueError):
            drive.set_wheel_duties(value, 0)
        controller.SpeedM1M2.assert_called_with(0x80, 0, 0)


def test_failed_duty_stops(controller):
    with RoboClawDrive() as drive:
        controller.DutyM1M2.return_value = False
        with pytest.raises(ConnectionError):
            drive.set_wheel_duties(32767, 32767)
        controller.SpeedM1M2.assert_called_with(0x80, 0, 0)


def test_failed_command_and_read_stop(controller):
    with RoboClawDrive() as drive:
        controller.SpeedM1M2.side_effect = [False, True]
        with pytest.raises(ConnectionError):
            drive.forward(100)
        controller.SpeedM1M2.assert_called_with(0x80, 0, 0)
        controller.SpeedM1M2.side_effect = None
        controller.ReadEncM1.return_value = (0, 0, 0)
        with pytest.raises(ConnectionError):
            drive.get_encoders()
        controller.SpeedM1M2.assert_called_with(0x80, 0, 0)


def test_stop_fallback(controller):
    with RoboClawDrive() as drive:
        controller.SpeedM1M2.return_value = False
        drive.stop()
        controller.DutyM1M2.assert_called_with(0x80, 0, 0)


def test_disconnect_closes_even_when_both_stops_fail(controller):
    drive = RoboClawDrive()
    drive.connect()
    controller.SpeedM1M2.return_value = False
    controller.DutyM1M2.return_value = False
    with pytest.raises(ConnectionError):
        drive.disconnect()
    controller.close.assert_called_once()
    assert not drive.is_connected


def test_failed_connection_closes(controller):
    controller.ReadVersion.return_value = (False, "")
    drive = RoboClawDrive()
    with pytest.raises(ConnectionError):
        drive.connect()
    controller.close.assert_called_once()
    assert not drive.is_connected


def test_context_stops_on_interrupt(controller):
    with pytest.raises(KeyboardInterrupt), RoboClawDrive() as drive:
        drive.forward(100)
        raise KeyboardInterrupt
    assert controller.method_calls[-2:] == [call.SpeedM1M2(0x80, 0, 0), call.close()]


def test_watchdog_stops_stale_motion(controller):
    with RoboClawDrive(RoboClawDriveConfig(command_timeout_s=0.02)) as drive:
        drive.forward(100)
        deadline = time.monotonic() + 1
        while drive._moving and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not drive._moving
        controller.SpeedM1M2.assert_called_with(0x80, 0, 0)


def test_watchdog_failure_requires_reconnect(controller):
    drive = RoboClawDrive(RoboClawDriveConfig(command_timeout_s=0.02))
    drive.connect()
    try:
        drive.forward(100)
        controller.SpeedM1M2.return_value = False
        controller.DutyM1M2.return_value = False
        deadline = time.monotonic() + 1
        while drive._fault is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert drive._fault is not None
        with pytest.raises(RuntimeError, match="watchdog failed"):
            drive.forward(100)
        controller.SpeedM1M2.return_value = True
        drive.disconnect()
        drive.connect()
        drive.forward(100)
    finally:
        controller.SpeedM1M2.return_value = True
        drive.disconnect()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m1_sign": 0},
        {"left_motor": 3},
        {"address": 0x79},
        {"max_speed": 0},
        {"command_timeout_s": float("nan")},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        RoboClawDriveConfig(**kwargs)
