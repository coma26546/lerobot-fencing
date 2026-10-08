"""Scripted pose transitions and wheel velocities on one monotonic timeline."""

import math
import time
from dataclasses import dataclass

from lerobot.types import RobotAction

from .fencing_robot import BASE_ACTION_KEYS, FencingRobot


@dataclass
class DrillStep:
    time_s: float
    pose: str | None = None
    left_speed: int = 0
    right_speed: int = 0
    transition_s: float = 0.5


def prepare_drill(robot: FencingRobot, steps: list[DrillStep], duration_s: float, hz: float) -> list:
    """Resolve every saved pose and validate the entire schedule before moving."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not math.isfinite(hz) or not 10 <= hz <= 100 or 1 / hz >= robot.config.drive.command_timeout_s:
        raise ValueError("hz must be 10..100 and refresh faster than the wheel command timeout")
    if not steps or steps[0].time_s != 0:
        raise ValueError("A drill must start with a step at time_s=0")
    resolved = []
    previous = -1.0
    for index, step in enumerate(steps):
        if not math.isfinite(step.time_s) or not previous < step.time_s < duration_s:
            raise ValueError("Step times must increase strictly and be before duration_s")
        if not math.isfinite(step.transition_s) or step.transition_s < 0:
            raise ValueError("transition_s must be finite and nonnegative")
        next_pose_time = next(
            (later.time_s for later in steps[index + 1 :] if later.pose is not None), duration_s
        )
        if step.pose is not None and step.time_s + step.transition_s > next_pose_time:
            raise ValueError("Pose transitions must finish before the next pose or drill end")
        pose = robot.arm.pose_store.load(step.pose) if step.pose is not None else None
        action = robot.validate_action(
            {**(pose or {}), BASE_ACTION_KEYS[0]: step.left_speed, BASE_ACTION_KEYS[1]: step.right_speed}
        )
        resolved.append((step, {key: action[key] for key in pose} if pose is not None else None, action))
        previous = step.time_s
    return resolved


def run_drill(robot: FencingRobot, steps: list[DrillStep], duration_s: float, hz: float = 40) -> None:
    """Run on already-connected hardware, retaining the final arm goal and stopping wheels."""
    try:
        resolved = prepare_drill(robot, steps, duration_s, hz)
        initial = robot.arm.get_observation()
        current = {key: float(initial[key]) for key in robot.arm.action_features}
        source = dict(current)
        target = dict(current)
        transition_start = 0.0
        transition_duration = 0.0
        wheels = dict.fromkeys(BASE_ACTION_KEYS, 0)
        index = 0
        start = time.monotonic()
        tick = 0
        while (elapsed := time.monotonic() - start) < duration_s:
            while index < len(resolved) and resolved[index][0].time_s <= elapsed:
                step, pose, action = resolved[index]
                if pose is not None:
                    # Start at the last commanded position, avoiding a target jump.
                    source = dict(current)
                    target = pose
                    transition_start = step.time_s
                    transition_duration = step.transition_s
                wheels = {key: action[key] for key in BASE_ACTION_KEYS}
                index += 1
            fraction = (
                1.0
                if transition_duration == 0
                else min(1.0, (elapsed - transition_start) / transition_duration)
            )
            fraction = fraction * fraction * (3 - 2 * fraction)
            current = {key: source[key] + (target[key] - source[key]) * fraction for key in target}
            sent: RobotAction = robot.send_action({**current, **wheels})
            current = {key: float(sent[key]) for key in current}
            tick += 1
            # Skip missed ticks rather than bursting commands to catch up.
            tick = max(tick, math.floor((time.monotonic() - start) * hz) + 1)
            time.sleep(max(0.0, min(start + tick / hz, start + duration_s) - time.monotonic()))
        # Ensure the final target is sent even if no tick lands on its endpoint.
        if index < len(resolved):
            raise TimeoutError("Drill ended before all scheduled steps could be commanded")
        robot.send_action({**target, **dict.fromkeys(BASE_ACTION_KEYS, 0)})
    except BaseException:
        robot._cleanup_after_error()
        raise
    finally:
        if robot.drive.is_connected:
            robot.stop()
