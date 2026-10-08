from unittest.mock import MagicMock, call, patch

import draccus
import pytest

from lerobot.robots.fencing_arm import FencingArmConfig
from lerobot.robots.fencing_robot import FencingRobot, FencingRobotConfig
from lerobot.robots.fencing_robot.drills import DrillStep, prepare_drill, run_drill
from lerobot.robots.roboclaw import RoboClawDriveConfig
from lerobot.robots.utils import make_robot_from_config


@pytest.fixture
def rig(tmp_path):
    config = FencingRobotConfig(
        id="whole_robot",
        calibration_dir=tmp_path / "robot",
        arm=FencingArmConfig(
            port_1="arm1", port_2="arm2", id="existing_arm", calibration_dir=tmp_path / "arm"
        ),
    )
    with (
        patch("lerobot.robots.fencing_robot.fencing_robot.FencingArm") as arm_class,
        patch("lerobot.robots.fencing_robot.fencing_robot.RoboClawDrive") as drive_class,
    ):
        arm, drive = arm_class.return_value, drive_class.return_value
        arm.is_connected = False
        drive.is_connected = False
        arm.is_calibrated = True
        arm.cameras = {}
        arm.action_features = {"shoulder_yaw.pos": float, "elbow_pitch.pos": float}
        arm.observation_features = dict(arm.action_features)
        arm.get_observation.return_value = {"shoulder_yaw.pos": 0.0, "elbow_pitch.pos": 0.0}
        arm.send_action.side_effect = lambda action: action
        arm.connect_passive.side_effect = lambda **kwargs: setattr(arm, "is_connected", True)
        arm.bus.disconnect.side_effect = lambda *args: setattr(arm, "is_connected", False)
        drive.connect.side_effect = lambda: setattr(drive, "is_connected", True)
        drive.disconnect.side_effect = lambda: setattr(drive, "is_connected", False)
        drive.get_encoders.return_value = {"left": 12, "right": -13}
        drive.get_speeds.return_value = {"left": 100, "right": -100}
        drive.get_battery_voltage.return_value = 12.6
        robot = make_robot_from_config(config)
        assert arm_class.call_args == call(config.arm)
        yield robot, arm, drive


def test_registration_config_and_calibration_reuse(rig):
    robot, arm, drive = rig
    assert isinstance(robot, FencingRobot)
    assert robot.config.type == "fencing_robot"
    assert robot.arm is arm
    assert robot.is_calibrated
    assert set(robot.action_features) == {
        "shoulder_yaw.pos",
        "elbow_pitch.pos",
        "base.left_speed",
        "base.right_speed",
    }
    decoded = draccus.decode(
        FencingRobotConfig,
        {
            "arm": {"port_1": "arm1", "port_2": "arm2", "id": "existing_arm"},
            "drive": {"m2_sign": -1},
        },
    )
    assert decoded.arm.id == "existing_arm"
    assert decoded.drive.m2_sign == -1


def test_connect_and_action_routing(rig):
    robot, arm, drive = rig
    with robot:
        arm.connect_passive.assert_called_once_with(calibrate=True)
        arm.prepare_for_motion.assert_called_once()
        result = robot.send_action(
            {"shoulder_yaw.pos": 10, "base.left_speed": 100.4, "base.right_speed": -200.2}
        )
        arm.send_action.assert_called_once_with({"shoulder_yaw.pos": 10.0})
        drive.set_wheel_speeds.assert_called_once_with(100, -200)
        assert result["base.right_speed"] == -200.0
    assert not robot.is_connected
    arm.bus.disconnect.assert_called_once_with(True)


def test_missing_wheels_stop(rig):
    robot, arm, drive = rig
    with robot:
        robot.send_action({"shoulder_yaw.pos": 10})
        drive.set_wheel_speeds.assert_called_once_with(0, 0)


def test_observation_schema(rig):
    robot, arm, drive = rig
    with robot:
        obs = robot.get_observation()
        assert set(obs) == set(robot.observation_features)
        assert obs["base.right_encoder"] == -13
        assert obs["base.battery_voltage"] == 12.6


@pytest.mark.parametrize(
    "action",
    [
        {"unknown": 1},
        {"base.left_speed": 501},
        {"base.left_speed": float("nan")},
        {"shoulder_yaw.pos": float("inf")},
        {"base.right_speed": True},
    ],
)
def test_invalid_actions_shutdown_before_writes(rig, action):
    robot, arm, drive = rig
    robot.connect()
    with pytest.raises(ValueError):
        robot.send_action(action)
    arm.send_action.assert_not_called()
    drive.set_wheel_speeds.assert_not_called()
    drive.disconnect.assert_called_once()
    arm.bus.disconnect.assert_called_once()


@pytest.mark.parametrize("failure", ["arm_action", "drive_action", "observation"])
def test_subsystem_failures_cleanup_both(rig, failure):
    robot, arm, drive = rig
    robot.connect()
    if failure == "arm_action":
        arm.send_action.side_effect = OSError("arm failed")
    elif failure == "drive_action":
        drive.set_wheel_speeds.side_effect = OSError("drive failed")
    else:
        arm.get_observation.side_effect = OSError("read failed")
    with pytest.raises(OSError):
        if failure == "observation":
            robot.get_observation()
        else:
            robot.send_action({"shoulder_yaw.pos": 10, "base.left_speed": 100})
    drive.disconnect.assert_called_once()
    arm.bus.disconnect.assert_called_once()


def test_partial_connection_cleanup_and_preserve_original_exception(rig):
    robot, arm, drive = rig
    camera = MagicMock(is_connected=True)
    arm.cameras = {"front": camera}
    arm.connect_passive.side_effect = OSError("camera failed")
    drive.disconnect.side_effect = OSError("stop failed")
    with pytest.raises(OSError, match="camera failed"):
        robot.connect()
    arm.bus.disconnect.assert_called_once()
    camera.disconnect.assert_called_once()


def test_calibration_connection_is_passive(rig):
    robot, arm, drive = rig
    robot.connect(calibrate=False)
    arm.prepare_for_motion.assert_not_called()
    robot.calibrate()
    drive.stop.assert_called_once()
    arm.calibrate.assert_called_once()
    robot.disconnect()


def test_disconnect_continues_after_stop_failure(rig):
    robot, arm, drive = rig
    robot.connect()
    drive.disconnect.side_effect = OSError("stop failed")
    with pytest.raises(ExceptionGroup):
        robot.disconnect()
    arm.bus.disconnect.assert_called_once()
    drive.is_connected = False


def test_drill_runs_arm_and_wheels_on_same_ticks(rig, monkeypatch):
    robot, arm, drive = rig
    clock = [0.0]
    monkeypatch.setattr("lerobot.robots.fencing_robot.drills.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "lerobot.robots.fencing_robot.drills.time.sleep",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    arm.pose_store.load.return_value = {"shoulder_yaw.pos": 20.0, "elbow_pitch.pos": 10.0}
    with robot:
        run_drill(robot, [DrillStep(0, pose="parry", left_speed=100, right_speed=100, transition_s=0.1)], 0.2)
        assert any(0 < item.args[0]["shoulder_yaw.pos"] < 20 for item in arm.send_action.call_args_list)
        assert call(100, 100) in drive.set_wheel_speeds.call_args_list
        assert drive.set_wheel_speeds.call_args_list[-1] == call(0, 0)
        assert arm.send_action.call_args_list[-1].args[0]["shoulder_yaw.pos"] == 20
        drive.stop.assert_called_once()


def test_drill_preflight_rejects_later_invalid_speed_before_motion(rig):
    robot, arm, drive = rig
    with pytest.raises(ValueError):
        prepare_drill(robot, [DrillStep(0), DrillStep(0.1, left_speed=501)], 0.2, 40)
    arm.send_action.assert_not_called()
    drive.connect.assert_not_called()


def test_port_collision():
    with pytest.raises(ValueError):
        FencingRobotConfig(
            arm=FencingArmConfig(port_1="arm1", port_2="arm2"), drive=RoboClawDriveConfig(port="arm1")
        )


def test_cli_nested_config_does_not_recurse():
    from lerobot.scripts.lerobot_fencing_robot import FencingDrillConfig

    config = draccus.parse(
        FencingDrillConfig,
        args=[
            "--robot.arm.port_1=arm1",
            "--robot.arm.port_2=arm2",
            "--robot.arm.id=existing_arm",
            "--drill=example.json",
            "--dry_run=true",
        ],
    )
    assert config.robot.type == "fencing_robot"
    assert config.robot.arm.type == "fencing_arm"
    assert config.robot.arm.id == "existing_arm"
    assert config.dry_run


def test_generic_robot_cli_can_select_fencing_robot():
    from dataclasses import dataclass

    from lerobot.robots import RobotConfig

    @dataclass
    class GenericConfig:
        robot: RobotConfig

    config = draccus.parse(
        GenericConfig,
        args=[
            "--robot.type=fencing_robot",
            "--robot.arm.port_1=arm1",
            "--robot.arm.port_2=arm2",
        ],
    )
    assert isinstance(config.robot, FencingRobotConfig)


def test_drill_cli_dry_run_never_connects(rig, tmp_path, monkeypatch, capsys):
    import json

    from lerobot.scripts.lerobot_fencing_robot import FencingDrillConfig, run

    robot, arm, drive = rig
    drill = tmp_path / "drill.json"
    drill.write_text(json.dumps({"duration_s": 1, "steps": [{"time_s": 0}]}), encoding="utf-8")
    monkeypatch.setattr("lerobot.scripts.lerobot_fencing_robot.FencingRobot", lambda config: robot)
    run(FencingDrillConfig(robot=robot.config, drill=drill, dry_run=True))
    assert "Validated" in capsys.readouterr().out
    drive.connect.assert_not_called()
    arm.connect_passive.assert_not_called()


def test_failed_stop_also_releases_arm(rig):
    robot, arm, drive = rig
    robot.connect()
    drive.stop.side_effect = OSError("stop failed")
    with pytest.raises(OSError, match="stop failed"):
        robot.stop()
    arm.bus.disconnect.assert_called_once()
