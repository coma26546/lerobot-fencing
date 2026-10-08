import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import draccus
import pytest

from lerobot.robots.fencing_arm import FencingArm
from lerobot.robots.fencing_arm.poses import MotionKeyframe, RecordedMotion
from lerobot.robots.fencing_robot import FencingRobot
from lerobot.robots.fencing_robot.session import MoveSettings, ParryDrill, ParryDrillSession, prepare_move
from lerobot.robots.roboclaw import RoboClawDriveConfig


class Commands:
    def __init__(self):
        self.pending = deque()

    def poll(self):
        return self.pending.popleft() if self.pending else None


@pytest.fixture
def hardware(monkeypatch):
    clock = [0.0]
    encoders = {"left": 23.0, "right": -47.0}
    position = {"shoulder_yaw.pos": 0.0}
    speeds = {"left": 0, "right": 0}
    actions = []
    robot = MagicMock()
    robot.config = SimpleNamespace(drive=RoboClawDriveConfig())
    robot.action_features = {"shoulder_yaw.pos": float, "base.left_speed": float, "base.right_speed": float}
    robot.arm.action_features = {"shoulder_yaw.pos": float}
    robot.arm.get_observation.side_effect = lambda: dict(position)
    robot.arm._ease.side_effect = FencingArm._ease
    robot.arm._smooth_recording_samples.side_effect = FencingArm._smooth_recording_samples
    robot.arm.pose_store.has_recording.return_value = False
    robot.arm.pose_store.has_motion.return_value = False
    poses = {"enguard": 0, "parry3": 10, "parry4": 20, "cut5": 30}
    robot.arm.pose_store.load.side_effect = lambda name: {"shoulder_yaw.pos": poses[name]}
    robot.validate_action.side_effect = lambda action: FencingRobot.validate_action(robot, action)
    robot.drive.is_connected = True
    robot.drive.get_encoders.side_effect = lambda: {wheel: round(value) for wheel, value in encoders.items()}

    def send(action):
        normalized = robot.validate_action(action)
        position.update({key: value for key, value in normalized.items() if key in position})
        speeds.update(left=normalized["base.left_speed"], right=normalized["base.right_speed"])
        actions.append((clock[0], dict(normalized)))
        return normalized

    def stop():
        speeds.update(left=0, right=0)

    def sleep(seconds):
        assert seconds >= 0
        clock[0] += seconds
        for wheel in encoders:
            encoders[wheel] += speeds[wheel] * seconds

    robot.send_action.side_effect = send
    robot.stop.side_effect = stop
    monkeypatch.setattr("lerobot.robots.fencing_robot.session.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("lerobot.robots.fencing_robot.session.time.sleep", sleep)
    return robot, Commands(), clock, encoders, position, actions


def test_random_branches_reset_then_await_separate_start(hardware):
    robot, commands, clock, encoders, position, actions = hardware
    rng = MagicMock()
    rng.random.side_effect = [0.75, 0.25]
    prompts, snapshots = [], []
    ready_count, result_count = 0, 0

    def announce(message):
        nonlocal ready_count, result_count
        prompts.append(message)
        if message.startswith("Ready:"):
            ready_count += 1
            snapshots.append((dict(encoders), dict(position), len(actions)))
            commands.pending.append("s")
        elif message.startswith("r +"):
            result_count += 1
            commands.pending.append("r" if result_count == 1 else "q")

    session = ParryDrillSession(robot, ParryDrill(), commands, rng, announce)
    session.run()
    assert rng.random.call_count == 2
    assert ready_count == 2
    assert any("held parry 3" in message for message in prompts)
    assert any("parry 4 / cut 5" in message for message in prompts)
    assert abs(round(snapshots[1][0]["left"]) - 23) <= 10
    assert abs(round(snapshots[1][0]["right"]) + 47) <= 10
    assert snapshots[1][1]["shoulder_yaw.pos"] == pytest.approx(0)
    first_rep = actions[: snapshots[1][2]]
    assert max(item[1].get("shoulder_yaw.pos", 0) for item in first_rep) <= 10
    assert any(item[1].get("shoulder_yaw.pos") == 30 for item in actions)
    assert any(
        item[1]["base.left_speed"] < 0 and 0 < item[1].get("shoulder_yaw.pos", 0) < 10 for item in first_rep
    )
    assert robot.stop.call_count > 0


def test_q_ends_without_returning_home(hardware):
    robot, commands, clock, encoders, position, actions = hardware
    rng = MagicMock()
    rng.random.return_value = 1.0

    def announce(message):
        if message.startswith("Ready:"):
            commands.pending.append("s")
        elif message.startswith("r +"):
            commands.pending.append("q")

    ParryDrillSession(robot, ParryDrill(), commands, rng, announce).run()
    assert encoders["left"] < 23 - 50
    assert not any(action["base.left_speed"] > 0 for _, action in actions)


def test_end_during_motion_stops(hardware):
    robot, commands, clock, encoders, position, actions = hardware

    class EndDuringRetreat:
        def poll(self):
            return "q" if clock[0] >= 0.8 else commands.poll()

    def announce(message):
        if message.startswith("Ready:"):
            commands.pending.append("s")

    ParryDrillSession(robot, ParryDrill(), EndDuringRetreat(), announce=announce).run()
    assert any(action["base.left_speed"] < 0 for _, action in actions)
    robot.stop.assert_called()
    assert clock[0] < 1.0


def test_return_stall_and_timeout_stop(hardware):
    robot, commands, clock, encoders, position, actions = hardware
    session = ParryDrillSession(robot, ParryDrill(return_stall_timeout_s=0.1), commands)
    session.prepare()
    session.home = {"left": 1000, "right": 1000}
    robot.drive.get_encoders.return_value = {"left": 0, "right": 0}
    robot.drive.get_encoders.side_effect = None
    with pytest.raises(TimeoutError, match="No encoder progress"):
        session._return_home()
    robot.stop.assert_called()


def test_arm_never_reaches_target_stops(hardware):
    robot, commands, clock, encoders, position, actions = hardware
    session = ParryDrillSession(robot, ParryDrill(arm_settle_timeout_s=0.1), commands)
    session.prepare()
    robot.arm.get_observation.side_effect = None
    robot.arm.get_observation.return_value = {"shoulder_yaw.pos": 0}
    with pytest.raises(TimeoutError, match="Arm did not complete"):
        session._move("parry3", retreat=True)
    assert actions[-1][1]["base.left_speed"] == 0
    robot.stop.assert_called()


def test_full_preflight_rejects_missing_move_and_short_window(hardware):
    robot, commands, *_ = hardware
    session = ParryDrillSession(robot, ParryDrill(exchange_window_s=0.1), commands)
    with pytest.raises(ValueError, match="exchange_window_s"):
        session.prepare()
    robot.send_action.assert_not_called()
    robot.arm.pose_store.load.side_effect = ValueError("missing cut5")
    with pytest.raises(ValueError, match="missing cut5"):
        ParryDrillSession(robot, ParryDrill(), commands).prepare()


def test_preflight_rejects_invalid_saved_recording_timing(hardware):
    robot, commands, *_ = hardware
    robot.arm.pose_store.has_recording.return_value = True
    robot.arm.pose_store.load_recording.return_value = RecordedMotion(
        sample_fps=10,
        trim_start_s=0,
        trim_end_s=0,
        playback_speed=float("nan"),
        smoothing_window=1,
        joint_delays_s={},
        samples=[
            {"time_s": 0, "positions": {"shoulder_yaw.pos": 0}},
            {"time_s": 1, "positions": {"shoulder_yaw.pos": 10}},
        ],
    )
    with pytest.raises(ValueError, match="invalid duration"):
        ParryDrillSession(robot, ParryDrill(), commands).prepare()


def test_recording_trim_speed_smoothing_and_delay(hardware):
    robot, commands, *_ = hardware
    store = robot.arm.pose_store
    store.has_recording.return_value = True
    store.load_recording.return_value = RecordedMotion(
        sample_fps=10,
        trim_start_s=0.2,
        trim_end_s=0.2,
        playback_speed=2,
        smoothing_window=1,
        joint_delays_s={"shoulder_yaw.pos": 0.2},
        samples=[
            {"time_s": 0.0, "positions": {"shoulder_yaw.pos": 0.0}},
            {"time_s": 1.0, "positions": {"shoulder_yaw.pos": 100.0}},
        ],
    )
    plan = prepare_move(robot, MoveSettings("cut5"), 0.5)
    assert plan.duration_s == pytest.approx(0.4)
    assert plan.sample({}, 0)["shoulder_yaw.pos"] == pytest.approx(20)
    assert plan.sample({}, 0.4)["shoulder_yaw.pos"] == pytest.approx(80)
    stretched = prepare_move(robot, MoveSettings("cut5", duration_s=2), 0.5)
    assert stretched.duration_s == 2


def test_keyframe_motion_and_joint_timing(hardware):
    robot, commands, *_ = hardware
    robot.arm.pose_store.has_motion.return_value = True
    robot.arm.pose_store.load_motion.return_value = [
        MotionKeyframe(
            "parry3",
            duration_s=1,
            easing="linear",
            joint_timing={"shoulder_yaw.pos": {"start_fraction": 0.5, "end_fraction": 1}},
        ),
        MotionKeyframe("cut5", duration_s=1),
    ]
    plan = prepare_move(robot, MoveSettings("sequence"), 0.5)
    assert plan.duration_s == 2
    assert plan.sample({"shoulder_yaw.pos": 0}, 0.5)["shoulder_yaw.pos"] == 0
    assert plan.sample({"shoulder_yaw.pos": 0}, 0.75)["shoulder_yaw.pos"] == 5
    assert plan.sample({"shoulder_yaw.pos": 0}, 2)["shoulder_yaw.pos"] == 30


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retreat_speed": 501},
        {"riposte_probability": -1},
        {"return_timeout_s": 0},
        {"hz": 1},
        {"exchange_window_s": float("inf")},
        {"arm_max_step": 0},
    ],
)
def test_invalid_parameters(hardware, kwargs):
    robot, commands, *_ = hardware
    with pytest.raises(ValueError):
        ParryDrill(**kwargs).validate(robot)


def test_example_and_cli_dry_run(hardware, tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from lerobot.scripts.lerobot_fencing_session import FencingSessionConfig, run

    robot, *_ = hardware
    data = json.loads(Path("examples/fencing_robot/retreat_parry3.json").read_text())
    assert draccus.decode(ParryDrill, data).riposte_probability == 0.5
    path = tmp_path / "drill.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("lerobot.scripts.lerobot_fencing_session.FencingRobot", lambda config: robot)
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_fencing_session.KeyboardCommands", lambda: pytest.fail("keyboard started")
    )
    run(FencingSessionConfig(robot=robot.config, drill=path, dry_run=True))
    assert "validated" in capsys.readouterr().out
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()
