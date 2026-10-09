"""Interactive retreat/parry drill with replaceable command input."""

import math
import queue
import random
import sys
import threading
import time
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .fencing_robot import FencingRobot


class CommandSource(Protocol):
    """A remote adapter can provide the same nonblocking interface as the keyboard."""

    def poll(self) -> str | None: ...


class KeyboardCommands:
    """Read newline-terminated commands without blocking the motion loop."""

    def __init__(self):
        self._queue = queue.SimpleQueue()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._read, name="fencing-keyboard", daemon=True)
        self._thread.start()

    def _read(self):
        while not self._closed.is_set():
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                self._queue.put("q")
                return
            if self._closed.is_set():
                return
            self._queue.put(line.strip().lower() if line else "q")
            if not line:
                return

    def poll(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed.set()


class EndSessionError(Exception):
    """Operator requested a normal session exit."""


@dataclass
class MoveSettings:
    name: str
    duration_s: float | None = None


@dataclass
class ParryDrill:
    standby: MoveSettings = field(default_factory=lambda: MoveSettings("enguard"))
    parry3: MoveSettings = field(default_factory=lambda: MoveSettings("parry3"))
    parry4: MoveSettings = field(default_factory=lambda: MoveSettings("parry4"))
    cut5: MoveSettings = field(default_factory=lambda: MoveSettings("cut5"))
    retreat_speed: int = 100
    retreat_duration_s: float = 1.0
    retreat_distance_counts: int | None = None
    retreat_timeout_s: float = 30.0
    retreat_stall_timeout_s: float = 2.0
    attack_delay_s: float = 0.5
    exchange_window_s: float = 4.0
    riposte_pause_s: float = 0.1
    riposte_probability: float = 0.5
    hz: float = 40.0
    pose_transition_s: float = 0.5
    arm_max_step: float = 6.0
    arm_tolerance: float = 2.0
    arm_settle_timeout_s: float = 1.0
    return_speed: int = 100
    return_tolerance_counts: int = 10
    return_timeout_s: float = 15.0
    return_stall_timeout_s: float = 2.0
    max_return_counts: int = 10000

    def validate(self, robot: FencingRobot) -> None:
        for name in (
            "retreat_duration_s",
            "retreat_timeout_s",
            "retreat_stall_timeout_s",
            "exchange_window_s",
            "pose_transition_s",
            "arm_max_step",
            "arm_tolerance",
            "arm_settle_timeout_s",
            "return_timeout_s",
            "return_stall_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("attack_delay_s", "riposte_pause_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.riposte_probability) or not 0 <= self.riposte_probability <= 1:
            raise ValueError("riposte_probability must be between zero and one")
        if (
            not math.isfinite(self.hz)
            or not 10 <= self.hz <= 100
            or 1 / self.hz >= robot.config.drive.command_timeout_s
        ):
            raise ValueError("hz must be 10..100 and refresh faster than the wheel watchdog")
        for name in ("retreat_speed", "return_speed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= robot.config.drive.max_speed:
                raise ValueError(f"{name} must be a positive integer within the wheel speed limit")
        for name in ("return_tolerance_counts", "max_return_counts"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for move in (self.standby, self.parry3, self.parry4, self.cut5):
            if move.duration_s is not None and (not math.isfinite(move.duration_s) or move.duration_s <= 0):
                raise ValueError("Move duration_s must be finite and positive")
        if self.retreat_distance_counts is not None and (
            type(self.retreat_distance_counts) is not int
            or not self.return_tolerance_counts < self.retreat_distance_counts <= self.max_return_counts
        ):
            raise ValueError("retreat_distance_counts must exceed tolerance and stay within max_return_counts")


@dataclass
class PreparedMove:
    duration_s: float
    sample: Callable[[dict, float], dict]


def prepare_move(robot: FencingRobot, settings: MoveSettings, default_duration: float) -> PreparedMove:
    """Resolve recordings, keyframe motions or poses without connecting hardware."""
    arm, store = robot.arm, robot.arm.pose_store
    if store.has_recording(settings.name):
        recording = store.load_recording(settings.name)
        samples = arm._smooth_recording_samples(recording.samples, recording.smoothing_window)
        for sample in samples:
            robot.validate_action(sample["positions"])
        begin = recording.trim_start_s
        end = samples[-1]["time_s"] - recording.trim_end_s
        span = end - begin + max(recording.joint_delays_s.values(), default=0)
        duration = settings.duration_s or span / recording.playback_speed
        times = [sample["time_s"] for sample in samples]

        def sample_recording(source, elapsed):
            progress = max(0.0, min(1.0, elapsed / duration))
            result = {}
            for key in samples[0]["positions"]:
                stamp = max(begin, min(end, begin + progress * span - recording.joint_delays_s.get(key, 0)))
                right = min(len(times) - 1, bisect_right(times, stamp))
                left = max(0, right - 1)
                fraction = (
                    0 if times[right] == times[left] else (stamp - times[left]) / (times[right] - times[left])
                )
                a, b = samples[left]["positions"][key], samples[right]["positions"][key]
                result[key] = a + (b - a) * fraction
            return result

        return PreparedMove(duration, sample_recording)

    if store.has_motion(settings.name):
        frames = store.load_motion(settings.name)
        parts = [
            (store.load(frame.pose), frame.duration_s or default_duration, frame.easing, frame.joint_timing)
            for frame in frames
        ]
    else:
        parts = [(store.load(settings.name), default_duration, "smoothstep", {})]
    for pose, *_ in parts:
        robot.validate_action(pose)
    original_duration = sum(part[1] for part in parts)
    duration = settings.duration_s or original_duration

    def sample_poses(source, elapsed):
        remaining = max(0.0, min(original_duration, elapsed / duration * original_duration))
        for target, seconds, easing, timings in parts:
            if remaining <= seconds:
                result = {}
                for key in target:
                    timing = timings.get(key, {})
                    begin, end = timing.get("start_fraction", 0), timing.get("end_fraction", 1)
                    fraction = max(0.0, min(1.0, (remaining / seconds - begin) / (end - begin)))
                    result[key] = source[key] + (target[key] - source[key]) * arm._ease(fraction, easing)
                return result
            remaining -= seconds
            source = target
        return dict(parts[-1][0])

    return PreparedMove(duration, sample_poses)


class ParryDrillSession:
    def __init__(
        self,
        robot: FencingRobot,
        drill: ParryDrill,
        commands: CommandSource | None,
        rng: random.Random | None = None,
        announce: Callable[[str], None] = print,
    ):
        self.robot, self.drill, self.commands = robot, drill, commands
        self.rng = rng or random.Random()
        self.announce = announce
        self.moves = {}
        self.home = None

    def prepare(self) -> None:
        self.drill.validate(self.robot)
        self.moves = {
            name: prepare_move(self.robot, getattr(self.drill, name), self.drill.pose_transition_s)
            for name in ("standby", "parry3", "parry4", "cut5")
        }
        for name, move in self.moves.items():
            if not math.isfinite(move.duration_s) or move.duration_s <= 0:
                raise ValueError(f"Saved move {name} has an invalid duration")
        # Reserve settling time for both moves so an overlong riposte is rejected before connecting.
        riposte_duration = (
            self.moves["parry4"].duration_s
            + self.moves["cut5"].duration_s
            + self.drill.riposte_pause_s
            + 2 * self.drill.arm_settle_timeout_s
        )
        if riposte_duration > self.drill.exchange_window_s:
            raise ValueError(
                "exchange_window_s must accommodate parry4, cut5, pause and both arm settling limits"
            )

    def _abort_if_requested(self) -> None:
        if self.commands is None:
            raise RuntimeError("A command source is required to run the session")
        while (command := self.commands.poll()) is not None:
            if command in ("q", "quit", "end", "stop"):
                raise EndSessionError
            # Start/reset commands received during motion are deliberately discarded.

    def _wait(self, prompt: str, allowed: set[str]) -> str:
        self.robot.stop()
        self._abort_if_requested()  # Discard queued start/reset commands from a prior phase.
        self.announce(prompt)
        while True:
            command = self.commands.poll()
            if command in ("q", "quit", "end", "stop"):
                raise EndSessionError
            if command in allowed:
                return command
            if command is not None:
                self.announce("Enter one of: " + ", ".join(sorted(allowed | {"q"})))
            time.sleep(1 / self.drill.hz)

    def _hold_until(self, deadline: float) -> None:
        self.robot.stop()
        while time.monotonic() < deadline:
            self._abort_if_requested()
            time.sleep(max(0.0, min(1 / self.drill.hz, deadline - time.monotonic())))

    def _move(self, name: str, retreat: bool = False, deadline: float | None = None) -> None:
        move = self.moves[name]
        observation = self.robot.arm.get_observation()
        source = {key: float(observation[key]) for key in self.robot.arm.action_features}
        previous = dict(source)
        start = time.monotonic()
        motion_end = max(move.duration_s, self.drill.retreat_duration_s if retreat else 0)
        distance_retreat = retreat and self.drill.retreat_distance_counts is not None
        if distance_retreat:
            encoders = self.robot.drive.get_encoders()
            wheel_targets = {
                wheel: value - self.drill.retreat_distance_counts for wheel, value in encoders.items()
            }
            best = dict.fromkeys(wheel_targets, math.inf)
            progress_at = dict.fromkeys(wheel_targets, start)
            wheel_done = dict.fromkeys(wheel_targets, False)
            motion_end = move.duration_s
        limit = start + motion_end + self.drill.arm_settle_timeout_s
        if distance_retreat:
            limit = max(limit, start + self.drill.retreat_timeout_s)
        if deadline is not None:
            limit = min(limit, deadline)
        try:
            while time.monotonic() < limit:
                self._abort_if_requested()
                elapsed = time.monotonic() - start
                target = move.sample(source, elapsed)
                action = {
                    key: previous[key]
                    + max(-self.drill.arm_max_step, min(self.drill.arm_max_step, target[key] - previous[key]))
                    for key in target
                }
                speed = (
                    -self.drill.retreat_speed if retreat and elapsed < self.drill.retreat_duration_s else 0
                )
                speeds = dict.fromkeys(("left", "right"), speed)
                wheels_complete = True
                if distance_retreat:
                    encoders = self.robot.drive.get_encoders()
                    now = time.monotonic()
                    for wheel, target_count in wheel_targets.items():
                        remaining = encoders[wheel] - target_count
                        if remaining <= self.drill.return_tolerance_counts:
                            wheel_done[wheel] = True
                        if wheel_done[wheel]:
                            speeds[wheel] = 0
                            continue
                        if now - start >= self.drill.retreat_timeout_s:
                            raise TimeoutError("One-metre retreat exceeded its time limit")
                        if remaining > self.drill.retreat_distance_counts + 20:
                            raise RuntimeError(f"Wrong encoder direction while retreating {wheel} wheel")
                        if remaining < best[wheel] - 1:
                            best[wheel], progress_at[wheel] = remaining, now
                        elif now - progress_at[wheel] >= self.drill.retreat_stall_timeout_s:
                            raise TimeoutError(f"No encoder progress while retreating {wheel} wheel")
                        speeds[wheel] = -min(self.drill.retreat_speed, max(1, round(remaining * 2)))
                    wheels_complete = all(wheel_done.values())
                sent = self.robot.send_action(
                    {**action, "base.left_speed": speeds["left"], "base.right_speed": speeds["right"]}
                )
                previous = {key: float(sent[key]) for key in target}
                if elapsed >= motion_end and wheels_complete:
                    observation = self.robot.arm.get_observation()
                    if all(
                        abs(float(observation[key]) - target[key]) <= self.drill.arm_tolerance
                        for key in target
                    ):
                        return
                time.sleep(1 / self.drill.hz)
            raise TimeoutError(f"Arm did not complete {name} within its drill deadline")
        finally:
            if self.robot.drive.is_connected:
                self.robot.stop()

    def _return_home(self) -> None:
        self._move("standby")
        start = time.monotonic()
        last_progress = dict.fromkeys(("left", "right"), start)
        best = dict.fromkeys(("left", "right"), math.inf)
        try:
            while time.monotonic() - start < self.drill.return_timeout_s:
                self._abort_if_requested()
                encoders = self.robot.drive.get_encoders()
                errors = {wheel: self.home[wheel] - encoders[wheel] for wheel in self.home}
                if all(abs(error) <= self.drill.return_tolerance_counts for error in errors.values()):
                    return
                speeds = {}
                now = time.monotonic()
                for wheel, error in errors.items():
                    if abs(error) > self.drill.max_return_counts:
                        raise ValueError(
                            "Return distance exceeds max_return_counts; check encoder direction and configuration"
                        )
                    if abs(error) <= self.drill.return_tolerance_counts:
                        speeds[wheel] = 0
                        continue
                    if abs(error) < best[wheel] - 1:
                        best[wheel], last_progress[wheel] = abs(error), now
                    elif now - last_progress[wheel] >= self.drill.return_stall_timeout_s:
                        raise TimeoutError(f"No encoder progress while returning {wheel} wheel")
                    magnitude = min(self.drill.return_speed, max(1, round(abs(error) * 2)))
                    speeds[wheel] = magnitude if error > 0 else -magnitude
                self.robot.send_action(
                    {"base.left_speed": speeds["left"], "base.right_speed": speeds["right"]}
                )
                time.sleep(1 / self.drill.hz)
            raise TimeoutError("Returning to the starting encoder position timed out")
        finally:
            if self.robot.drive.is_connected:
                self.robot.stop()

    def run(self) -> None:
        """Run on connected hardware. Caller owns connection and final disconnection."""
        try:
            if not self.moves:
                self.prepare()
            self.home = dict(self.robot.drive.get_encoders())
            rep = 0
            ready = False
            while True:
                self._wait("Ready: s + Enter starts a rep; q + Enter ends the session.", {"s", "start"})
                if not ready:
                    self._move("standby")
                    ready = True
                rep += 1
                riposte = self.rng.random() < self.drill.riposte_probability
                self.announce(f"Rep {rep}: retreat and parry 3. Attack the open 4 line.")
                self._move("parry3", retreat=True)
                self._hold_until(time.monotonic() + self.drill.attack_delay_s)
                exchange_end = time.monotonic() + self.drill.exchange_window_s
                if riposte:
                    self._move("parry4", deadline=exchange_end)
                    self._hold_until(min(exchange_end, time.monotonic() + self.drill.riposte_pause_s))
                    self._move("cut5", deadline=exchange_end)
                # No motion on the hold branch. Both branches leave time for the fencer's finish.
                self._hold_until(exchange_end)
                self.announce(f"Rep {rep} complete ({'parry 4 / cut 5' if riposte else 'held parry 3'}).")
                self._wait("r + Enter returns to the start; q + Enter ends the session.", {"r", "reset"})
                self._return_home()
                self.announce(
                    "Returned to the starting wheel encoder counts and standby move. Awaiting start."
                )
        except EndSessionError:
            self.announce("Drill session ended.")
        except BaseException:
            self.robot._cleanup_after_error()
            raise
        finally:
            if self.robot.drive.is_connected:
                self.robot.stop()
