# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import draccus

from lerobot.robots.fencing_arm import (
    FENCING_ARM_JOINT_KEYS,
    FencingArm,
    FencingArmConfig,
    MotionKeyframe,
    FencingArmTelemetryRecorder,
)
from lerobot.utils.utils import init_logging


def _require_robot_id(config: FencingArmConfig) -> None:
    if not config.id:
        raise ValueError("--robot.id is required so poses are associated with one physical arm")


def _prompt_keyframe_count(name: str) -> int:
    while True:
        response = input(f"How many keyframes should motion {name!r} contain (not counting enguard)? ")
        try:
            count = int(response)
        except ValueError:
            print("Enter a whole number greater than zero.")
            continue
        if count < 1:
            print("Enter a whole number greater than zero.")
            continue
        return count


@dataclass
class RecordPoseConfig:
    robot: FencingArmConfig
    pose: str = "enguard"


@draccus.wrap()
def record_pose(config: RecordPoseConfig) -> None:
    """Disable torque and record the manually positioned arm."""
    init_logging()
    _require_robot_id(config.robot)
    arm = FencingArm(config.robot)
    arm.connect_passive()
    try:
        # Manual keyframe teaching may legitimately cross an encoder boundary.
        # Torque is already disabled, so record the actual position instead of
        # applying the runtime command-hold behavior.
        arm.config.wraparound_protection = False
        if config.pose == "enguard":
            input(
                "Torque is OFF. Manually move the arm to pose 'enguard', support it securely, "
                "then press ENTER to record."
            )
            arm.record_pose("enguard")
            print(f"Saved pose 'enguard' to {arm.pose_store.path}")
        else:
            count = _prompt_keyframe_count(config.pose)
            keyframes = []
            for index in range(1, count + 1):
                keyframe = f"{config.pose}__keyframe_{index:02d}"
                input(
                    f"Torque is OFF. Move the arm to keyframe {index}/{count} for motion "
                    f"{config.pose!r}, then press ENTER to record."
                )
                arm.record_pose(keyframe)
                keyframes.append(keyframe)
                print(f"Saved keyframe {index}/{count}.")
            arm.pose_store.save_motion(config.pose, keyframes)
            print(f"Saved motion {config.pose!r} with {count} keyframes to {arm.pose_store.path}")
    finally:
        arm.disconnect()


@dataclass
class RecordMotionConfig:
    robot: FencingArmConfig
    motion: str
    standby_pose: str = "enguard"
    standby_max_step: float = 7.50
    standby_fps: float = 30.0
    standby_tolerance: float = 2.0
    sample_fps: float = 30.0
    max_duration_s: float = 60.0


@draccus.wrap()
def record_motion(config: RecordMotionConfig) -> None:
    """Move to en garde, release torque, and record a hand-guided demonstration."""
    init_logging()
    _require_robot_id(config.robot)
    if config.sample_fps <= 0 or config.max_duration_s <= 0:
        raise ValueError("sample_fps and max_duration_s must be greater than zero")
    arm = FencingArm(config.robot)
    arm.pose_store.has_recording(config.motion)
    arm.pose_store.load(config.standby_pose)
    arm.connect_passive()
    try:
        input(
            f"Workspace clear? Press ENTER to move to {config.standby_pose!r}, or Ctrl+C to abort."
        )
        arm.prepare_for_motion()
        arm.move_to_pose(
            config.standby_pose,
            max_step=config.standby_max_step,
            fps=config.standby_fps,
            timeout_s=config.max_duration_s,
            tolerance=config.standby_tolerance,
        )
        input(
            "Arm is holding en garde. Support the arm, then press ENTER to release torque and start "
            "recording."
        )
        arm.bus.disable_torque()
        arm.config.wraparound_protection = False
        stop_event = threading.Event()

        def wait_for_stop() -> None:
            input("Recording now. Perform the motion, then press ENTER to stop.\n")
            stop_event.set()

        threading.Thread(target=wait_for_stop, daemon=True).start()
        samples = []
        started_at = time.monotonic()
        next_sample_at = started_at
        while not stop_event.is_set():
            now = time.monotonic()
            if now - started_at >= config.max_duration_s:
                print(f"Maximum recording duration of {config.max_duration_s:.1f}s reached.")
                break
            observation = arm.get_observation()
            samples.append(
                {
                    "time_s": now - started_at,
                    "positions": {
                        key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS
                    },
                }
            )
            next_sample_at += 1 / config.sample_fps
            time.sleep(max(0.0, next_sample_at - time.monotonic()))
        arm.pose_store.save_recording(config.motion, config.sample_fps, samples)
        print(
            f"Saved recording {config.motion!r} with {len(samples)} samples to "
            f"{arm.pose_store.path}"
        )
    finally:
        arm.disconnect()


@dataclass
class RunPoseConfig:
    robot: FencingArmConfig
    pose: str
    standby_pose: str = "enguard"
    standby_max_step: float = 7.50
    standby_fps: float = 30.0
    standby_tolerance: float = 2.0
    max_step: float = 6.0
    fps: float = 30.0
    timeout_s: float = 120.0
    keyframe_tolerance: float = 2.0
    tolerance: float = 2.0
    confirm_each_move: bool = True
    hold_at_target: bool = True


@draccus.wrap()
def run_pose(config: RunPoseConfig) -> None:
    """Move to en garde, wait in standby, then move to a named preset."""
    _run_pose(config)


def _run_pose(
    config: RunPoseConfig,
    telemetry_output_dir: Path | None = None,
    telemetry_fps: float = 5.0,
) -> None:
    init_logging()
    _require_robot_id(config.robot)
    arm = FencingArm(config.robot)
    telemetry = None
    if telemetry_output_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = telemetry_output_dir / f"{config.pose}_{timestamp}"
        telemetry = FencingArmTelemetryRecorder(run_dir, telemetry_fps)
        arm.telemetry_recorder = telemetry
        print(f"Telemetry will be saved incrementally to {telemetry.csv_path}")

    # Fail before enabling motion if the standby pose or any requested
    # keyframe is unavailable.
    arm.pose_store.load(config.standby_pose)
    is_recording = arm.pose_store.has_recording(config.pose)
    target_keyframes = []
    if not is_recording:
        target_keyframes = (
            arm.pose_store.load_motion(config.pose)
            if arm.pose_store.has_motion(config.pose)
            else [MotionKeyframe(pose=config.pose)]
        )
        for keyframe in target_keyframes:
            arm.pose_store.load(keyframe.pose)
    else:
        arm.pose_store.load_recording(config.pose)

    arm.connect_passive()
    try:
        if config.confirm_each_move:
            input(
                f"Workspace clear? Press ENTER to move to standby pose {config.standby_pose!r}, "
                "or Ctrl+C to abort."
            )
        arm.prepare_for_motion()
        arm.move_to_pose(
            config.standby_pose,
            max_step=config.standby_max_step,
            fps=config.standby_fps,
            timeout_s=config.timeout_s,
            tolerance=config.standby_tolerance,
        )
        print(f"FencingArm is holding standby pose {config.standby_pose!r}.")

        if config.pose != config.standby_pose:
            if config.confirm_each_move:
                input(f"Press ENTER to move to preset {config.pose!r}, or Ctrl+C to abort.")
            if is_recording:
                arm.play_recording(config.pose, fps=config.fps, max_step=config.max_step)
            for index, keyframe in enumerate(target_keyframes, start=1):
                print(
                    f"Executing keyframe {index}/{len(target_keyframes)}: {keyframe.pose} "
                    f"({keyframe.easing}, duration={keyframe.duration_s or 'automatic'})"
                )
                is_final_keyframe = index == len(target_keyframes)
                arm.move_to_pose(
                    keyframe.pose,
                    max_step=config.max_step,
                    fps=config.fps,
                    timeout_s=config.timeout_s,
                    tolerance=(
                        config.tolerance if is_final_keyframe else config.keyframe_tolerance
                    ),
                    duration_s=keyframe.duration_s,
                    easing=keyframe.easing,
                    joint_timing=keyframe.joint_timing,
                )
            print(f"FencingArm completed preset {config.pose!r}.")

        if config.hold_at_target:
            input("Holding position. Press ENTER to disable torque and exit.")
    finally:
        arm.disconnect()
        if telemetry is not None:
            telemetry.close()

    if telemetry is not None:
        graphs = telemetry.render_graphs()
        print(f"Saved raw telemetry to {telemetry.csv_path}")
        print(f"Saved {len(graphs)} graphs to {telemetry.output_dir}")


@dataclass
class RunTelemetryConfig(RunPoseConfig):
    output_dir: Path = Path("outputs/fencing_arm_telemetry")
    telemetry_fps: float = 5.0


@draccus.wrap()
def run_pose_telemetry(config: RunTelemetryConfig) -> None:
    """Run a pose or recording while saving servo telemetry and graphs."""
    _run_pose(config, telemetry_output_dir=config.output_dir, telemetry_fps=config.telemetry_fps)


@dataclass
class EnguardCheckConfig:
    robot: FencingArmConfig
    pose: str = "enguard"
    max_step: float = 1.0
    fps: float = 30.0
    telemetry_fps: float = 5.0
    timeout_s: float = 15.0
    tolerance: float = 2.0
    output_dir: Path = Path("outputs/fencing_arm_diagnostics")


@draccus.wrap()
def enguard_check(config: EnguardCheckConfig) -> None:
    """Attempt en garde and always save goal/error and servo diagnostics."""
    init_logging()
    _require_robot_id(config.robot)
    arm = FencingArm(config.robot)
    target = arm.pose_store.load(config.pose)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / f"{config.pose}_check_{timestamp}"
    telemetry = FencingArmTelemetryRecorder(run_dir, config.telemetry_fps)
    arm.telemetry_recorder = telemetry
    success = False
    failure = None
    final_positions = {}
    registers = {}
    started_at = time.monotonic()
    try:
        arm.connect_passive()
        input(
            f"Workspace clear? Press ENTER to test movement to {config.pose!r}, or Ctrl+C to abort."
        )
        arm.prepare_for_motion()
        try:
            arm.move_to_pose(
                config.pose,
                max_step=config.max_step,
                fps=config.fps,
                timeout_s=config.timeout_s,
                tolerance=config.tolerance,
            )
            success = True
        except TimeoutError as error:
            failure = str(error)
            print(f"Diagnostic move timed out: {error}")
        observation = arm.get_observation()
        final_positions = {key: float(observation[key]) for key in FENCING_ARM_JOINT_KEYS}
        for register in (
            "Goal_Position",
            "Present_Position",
            "Present_Velocity",
            "Present_Load",
            "Present_Voltage",
            "Present_Temperature",
            "Present_Current",
            "Torque_Enable",
        ):
            registers[register] = arm.bus.sync_read(register, normalize=False)
    finally:
        if arm.is_connected:
            arm.disconnect()
        telemetry.close()

    errors = {
        key: target[key] - final_positions[key]
        for key in FENCING_ARM_JOINT_KEYS
        if key in final_positions
    }
    report = {
        "success": success,
        "failure": failure,
        "pose": config.pose,
        "elapsed_s": time.monotonic() - started_at,
        "tolerance_deg": config.tolerance,
        "target_positions_deg": target,
        "final_positions_deg": final_positions,
        "final_errors_deg": errors,
        "worst_joint": max(errors, key=lambda key: abs(errors[key])) if errors else None,
        "registers_raw": registers,
        "motor_settings": {
            "position_p_coefficient": config.robot.position_p_coefficient,
            "position_i_coefficient": config.robot.position_i_coefficient,
            "position_d_coefficient": config.robot.position_d_coefficient,
            "torque_limit": config.robot.torque_limit,
            "acceleration": config.robot.acceleration,
        },
        "telemetry_csv": str(telemetry.csv_path),
    }
    report_path = run_dir / "diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    graphs = telemetry.render_graphs()
    print(f"Diagnostic result: {'PASS' if success else 'TIMEOUT'}")
    print(f"Saved diagnostic report to {report_path}")
    print(f"Saved telemetry CSV and {len(graphs)} graphs to {run_dir}")


@dataclass
class MoveConsoleConfig:
    robot: FencingArmConfig
    standby_pose: str = "enguard"
    standby_max_step: float = 7.50
    standby_fps: float = 30.0
    standby_tolerance: float = 2.0
    max_step: float = 6.0
    fps: float = 60.0
    timeout_s: float = 120.0
    keyframe_tolerance: float = 2.0
    tolerance: float = 2.0
    action_delay_s: float = 0.0


MOVE_CONSOLE_SHORTCUTS = {
    "1": "cut5",
    "2": "cut4",
    "3": "cut3",
    "4": "parry4",
    "5": "parry5",
    "6": "parry3",
}


def _execute_named_move(arm: FencingArm, name: str, config: MoveConsoleConfig) -> None:
    if arm.pose_store.has_recording(name):
        arm.pose_store.load_recording(name)
        arm.play_recording(name, fps=config.fps, max_step=config.max_step)
        return
    if arm.pose_store.has_motion(name):
        keyframes = arm.pose_store.load_motion(name)
    else:
        arm.pose_store.load(name)
        keyframes = [MotionKeyframe(pose=name)]
    for index, keyframe in enumerate(keyframes, start=1):
        arm.move_to_pose(
            keyframe.pose,
            max_step=config.max_step,
            fps=config.fps,
            timeout_s=config.timeout_s,
            tolerance=(config.tolerance if index == len(keyframes) else config.keyframe_tolerance),
            duration_s=keyframe.duration_s,
            easing=keyframe.easing,
            joint_timing=keyframe.joint_timing,
        )


@draccus.wrap()
def move_console(config: MoveConsoleConfig) -> None:
    """Interactively execute named moves and return to en garde between them."""
    init_logging()
    _require_robot_id(config.robot)
    if config.action_delay_s < 0:
        raise ValueError("--action_delay_s must be non-negative")
    arm = FencingArm(config.robot)
    arm.pose_store.load(config.standby_pose)
    arm.connect_passive()
    try:
        input(
            f"Workspace clear? Press ENTER to move to {config.standby_pose!r}, or Ctrl+C to abort."
        )
        arm.prepare_for_motion()
        arm.move_to_pose(
            config.standby_pose,
            max_step=config.standby_max_step,
            fps=config.standby_fps,
            timeout_s=config.timeout_s,
            tolerance=config.standby_tolerance,
        )
        while True:
            recordings = arm.pose_store.recording_names()
            motions = arm.pose_store.motion_names()
            poses = [
                name
                for name in arm.pose_store.names()
                if "__keyframe_" not in name and name != config.standby_pose
            ]
            print(f"\nRecordings: {', '.join(recordings) or 'none'}")
            print(f"Keyframe motions: {', '.join(motions) or 'none'}")
            print(f"Single poses: {', '.join(poses) or 'none'}")
            print("Shortcuts: 1=cut5, 2=cut4, 3=cut3, 4=parry4, 5=parry5, 6=parry3")
            name = input("Enter a move number/name, or q to disable torque and quit: ").strip()
            if name.lower() in {"q", "quit", "exit"}:
                break
            name = MOVE_CONSOLE_SHORTCUTS.get(name, name)
            try:
                if config.action_delay_s > 0:
                    print(f"Moving to {name!r} in {config.action_delay_s:g} seconds...")
                    time.sleep(config.action_delay_s)
                _execute_named_move(arm, name, config)
            except (ValueError, TimeoutError) as error:
                print(f"Move {name!r} failed: {error}")
            input(
                f"Holding current position. Press ENTER to return to {config.standby_pose!r}, "
                "or Ctrl+C to abort."
            )
            arm.move_to_pose(
                config.standby_pose,
                max_step=config.standby_max_step,
                fps=config.standby_fps,
                timeout_s=config.timeout_s,
                tolerance=config.standby_tolerance,
            )
            print(f"FencingArm is holding {config.standby_pose!r}.")
    finally:
        arm.disconnect()


def record_pose_main() -> None:
    record_pose()


def record_motion_main() -> None:
    record_motion()


def run_pose_main() -> None:
    run_pose()


def run_pose_telemetry_main() -> None:
    run_pose_telemetry()


def enguard_check_main() -> None:
    enguard_check()


def move_console_main() -> None:
    move_console()
