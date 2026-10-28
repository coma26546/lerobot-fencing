# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from unittest.mock import MagicMock, call, patch

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.fencing_arm import FencingArm, FencingArmConfig
from lerobot.robots.fencing_arm.poses import FENCING_ARM_JOINT_KEYS
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.utils import make_robot_from_config


def test_fencing_arm_has_distinct_registered_type():
    config = FencingArmConfig(port_1="/dev/bus1", port_2="/dev/bus2")

    assert config.type == "fencing_arm"
    assert FencingArm.config_class is FencingArmConfig
    assert FencingArm.name == "fencing_arm"
    assert issubclass(FencingArm, SOFollower)
    assert config.shoulder_pitch_gear_ratio == 2.0
    assert config.position_p_coefficient == 32
    assert config.position_i_coefficient == 2
    assert config.torque_limit == 1000
    assert config.acceleration == 254


def test_factory_creates_fencing_arm(tmp_path):
    config = FencingArmConfig(port_1="/dev/bus1", port_2="/dev/bus2", calibration_dir=tmp_path)

    with patch("lerobot.robots.fencing_arm.split_bus.FeetechMotorsBus"):
        robot = make_robot_from_config(config)

    assert isinstance(robot, FencingArm)
    assert robot.robot_type == "fencing_arm"
    assert robot.calibration_dir == tmp_path


def test_fencing_arm_motor_layout(tmp_path):
    config = FencingArmConfig(port_1="/dev/bus1", port_2="/dev/bus2", calibration_dir=tmp_path)

    with patch("lerobot.robots.fencing_arm.split_bus.FeetechMotorsBus") as bus_class:
        FencingArm(config)

    assert bus_class.call_count == 2
    bus_1_call, bus_2_call = bus_class.call_args_list
    assert bus_1_call.kwargs["port"] == "/dev/bus1"
    assert bus_2_call.kwargs["port"] == "/dev/bus2"
    bus_1_motors = bus_1_call.kwargs["motors"]
    bus_2_motors = bus_2_call.kwargs["motors"]
    assert {name: (motor.id, motor.model) for name, motor in bus_1_motors.items()} == {
        "shoulder_yaw": (1, "sts3250"),
        "shoulder_pitch": (2, "sts3095"),
    }
    assert {name: (motor.id, motor.model) for name, motor in bus_2_motors.items()} == {
        "elbow_pitch": (3, "sts3250"),
        "elbow_roll": (4, "sts3250"),
        "wrist_pitch": (5, "sts3215"),
    }


def test_shoulder_pitch_driver_supports_mixed_bus_and_motor_setup():
    motors = {
        "shoulder_yaw": Motor(1, "sts3250", MotorNormMode.DEGREES),
        "shoulder_pitch": Motor(2, "sts3095", MotorNormMode.DEGREES),
    }
    bus = FeetechMotorsBus("unused", motors)
    with patch.object(bus, "ping", side_effect=[2825, 2569]):
        bus._assert_motors_exist()

    with (
        patch.object(bus, "set_baudrate") as set_baudrate,
        patch.object(bus, "broadcast_ping", return_value={1: 2569}),
    ):
        assert bus._find_single_motor("shoulder_pitch") == (1_000_000, 1)
        set_baudrate.assert_called_once_with(1_000_000)

    assert bus.model_resolution_table["sts3095"] == 4096
    assert bus.model_ctrl_table["sts3095"]["Present_Position"] == (56, 2)
    assert bus.model_ctrl_table["sts3095"]["Goal_Position"] == (42, 2)
    assert bus.model_encoding_table["sts3095"]["Homing_Offset"] == 11


def test_connect_passive_uses_full_strength_fencing_tuning(tmp_path):
    config = FencingArmConfig(port_1="/dev/bus1", port_2="/dev/bus2", calibration_dir=tmp_path)

    with patch("lerobot.robots.fencing_arm.split_bus.FeetechMotorsBus") as bus_class:
        bus_1 = MagicMock()
        bus_1.motors = {"shoulder_yaw": MagicMock(), "shoulder_pitch": MagicMock()}
        bus_1.is_connected = False
        bus_1.connect.side_effect = lambda: setattr(bus_1, "is_connected", True)
        bus_2 = MagicMock()
        bus_2.motors = {
            "elbow_pitch": MagicMock(),
            "elbow_roll": MagicMock(),
            "wrist_pitch": MagicMock(),
        }
        bus_2.is_connected = False
        bus_2.connect.side_effect = lambda: setattr(bus_2, "is_connected", True)
        bus_class.side_effect = [bus_1, bus_2]
        arm = FencingArm(config)
        arm.connect_passive(calibrate=False)

    for physical_bus in (bus_1, bus_2):
        physical_bus.configure_motors.assert_called_once_with(
            maximum_acceleration=254, acceleration=254
        )
        for motor in physical_bus.motors:
            assert call("P_Coefficient", motor, 32) in physical_bus.write.call_args_list
            assert call("I_Coefficient", motor, 2) in physical_bus.write.call_args_list
            assert call("Torque_Limit", motor, 1000) in physical_bus.write.call_args_list


def test_move_to_pose_synchronizes_joints_with_different_travel():
    arm = FencingArm.__new__(FencingArm)
    start = dict.fromkeys(FENCING_ARM_JOINT_KEYS, 0.0)
    target = dict.fromkeys(FENCING_ARM_JOINT_KEYS, 2.0)
    target["shoulder_yaw.pos"] = 4.0
    arm.pose_store = MagicMock()
    arm.pose_store.load.return_value = target
    arm.get_observation = MagicMock(side_effect=[start, start, start, start, start, target])
    arm.send_action = MagicMock(side_effect=lambda action: action)

    with patch("lerobot.robots.fencing_arm.fencing_arm.time.sleep"):
        arm.move_to_pose("test", max_step=1.0, fps=30, timeout_s=5, easing="linear")

    expected_actions = []
    for step in range(1, 5):
        action = dict.fromkeys(FENCING_ARM_JOINT_KEYS, step * 0.5)
        action["shoulder_yaw.pos"] = float(step)
        expected_actions.append(call(action))
    assert arm.send_action.call_args_list == expected_actions
