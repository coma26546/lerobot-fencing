"""Hardware-free checks for resting before releasing the fencing arm."""

import inspect
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from lerobot.scripts import lerobot_fencing_arm_poses as scripts
from lerobot.robots.fencing_arm.telemetry import FencingArmTelemetryRecorder


class RestShutdownTests(unittest.TestCase):
    def test_diagnostic_summary_uses_magnitudes_and_reports_missing_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FencingArmTelemetryRecorder(Path(directory))
            for current, load in [(-10, -20), (30, 40)]:
                recorder.writer.writerow({
                    "motor": "shoulder_yaw", "current_raw": current,
                    "torque_load_percent": load,
                })
            with patch("builtins.print") as output:
                recorder.print_summary()
            lines = [str(call.args[0]) for call in output.call_args_list]
            row = next(line for line in lines if line.startswith("shoulder_yaw"))
            self.assertEqual(row.split()[1:], ["20.00", "30.00", "0.130", "0.195", "30.00", "40.00"])
            self.assertFalse(any("Samples" in line for line in lines))
            self.assertTrue(any("No samples" in line for line in lines))

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
