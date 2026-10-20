"""Hardware-free checks for resting before releasing the fencing arm."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from lerobot.scripts import lerobot_fencing_arm_poses as scripts


class RestShutdownTests(unittest.TestCase):
    def setUp(self):
        self.arm = MagicMock()
        self.arm.pose_store.has_recording.return_value = True
        self.config = scripts.RunPoseConfig(
            robot=scripts.FencingArmConfig(port_1="bus1", port_2="bus2", id="test"),
            pose="cut5",
            confirm_each_move=False,
            hold_at_target=False,
        )

    def run_pose(self):
        with patch.object(scripts, "FencingArm", return_value=self.arm):
            scripts._run_pose(self.config)

    def test_normal_completion_lowers_before_disconnect(self):
        self.run_pose()
        calls = [entry[0] for entry in self.arm.mock_calls]
        self.assertLess(calls.index("play_recording"), calls.index("move_to_rest"))
        self.assertLess(calls.index("move_to_rest"), calls.index("disconnect"))

    def test_missing_rest_prevents_connection(self):
        self.arm.pose_store.load.side_effect = ValueError("Missing rest")
        with self.assertRaises(ValueError):
            self.run_pose()
        self.arm.connect_passive.assert_not_called()

    def test_interrupt_does_not_command_rest(self):
        self.arm.play_recording.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_pose()
        self.arm.move_to_rest.assert_not_called()
        self.arm.disconnect.assert_called_once()

    def test_rest_failure_still_disconnects(self):
        self.arm.move_to_rest.side_effect = TimeoutError("Rest unreachable")
        with self.assertRaises(TimeoutError):
            self.run_pose()
        self.arm.disconnect.assert_called_once()

    def test_record_rest_is_single_pose(self):
        config = scripts.RecordPoseConfig(robot=self.config.robot, pose="rest")
        with patch.object(scripts, "FencingArm", return_value=self.arm), patch("builtins.input"):
            inspect.unwrap(scripts.record_pose)(config)
        self.arm.record_pose.assert_called_once_with("rest")
        self.arm.pose_store.save_motion.assert_not_called()

    def test_console_quit_lowers_before_disconnect(self):
        config = scripts.MoveConsoleConfig(robot=self.config.robot)
        with patch.object(scripts, "FencingArm", return_value=self.arm), patch(
            "builtins.input", side_effect=["", "q"]
        ):
            inspect.unwrap(scripts.move_console)(config)
        calls = [entry[0] for entry in self.arm.mock_calls]
        self.assertLess(calls.index("move_to_rest"), calls.index("disconnect"))
