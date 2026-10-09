import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location(
    "wheel_distance_test",
    Path(__file__).resolve().parents[2] / "examples/fencing_robot/wheel_distance_test.py",
)
wheel_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_test)


@pytest.mark.parametrize("direction", [1, -1])
def test_distance_feedback_and_stop(monkeypatch, direction):
    drive = Mock()
    positions = {"left": 0, "right": 0}
    drive.get_encoders.side_effect = lambda: positions.copy()
    clock = [0.0]

    def advance(seconds):
        clock[0] += seconds
        left, right = drive.set_wheel_speeds.call_args.args
        positions["left"] += round(left * seconds)
        positions["right"] += round(right * seconds * 0.8)

    monkeypatch.setattr(wheel_test.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(wheel_test.time, "sleep", advance)
    wheel_test.move_to(drive, dict.fromkeys(positions, direction * 1998), direction, 400)
    assert all(abs(direction * 1998 - value) <= 10 for value in positions.values())
    drive.stop.assert_called_once()


def test_stalled_encoder_stops(monkeypatch):
    drive = Mock()
    drive.get_encoders.return_value = {"left": 0, "right": 0}
    clock = [0.0]
    monkeypatch.setattr(wheel_test.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(wheel_test.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(RuntimeError, match="stalled"):
        wheel_test.move_to(drive, {"left": 1998, "right": 1998}, 1, 400)
    drive.stop.assert_called_once()
