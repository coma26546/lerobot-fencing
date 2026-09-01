# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

FENCING_ARM_JOINT_KEYS = (
    "shoulder_yaw.pos",
    "shoulder_pitch.pos",
    "elbow_pitch.pos",
    "elbow_roll.pos",
    "wrist_pitch.pos",
)

_POSE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
MOTION_EASINGS = ("linear", "smoothstep", "smootherstep")


@dataclass(frozen=True)
class MotionKeyframe:
    pose: str
    duration_s: float | None = None
    easing: str = "smoothstep"
    joint_timing: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordedMotion:
    sample_fps: float
    trim_start_s: float
    trim_end_s: float
    playback_speed: float
    smoothing_window: int
    joint_delays_s: dict[str, float]
    samples: list[dict]


class FencingArmPoseStore:
    """Persistent named poses for one physical FencingArm."""

    def __init__(self, path: Path, coordinate_system: str, shoulder_pitch_gear_ratio: float):
        self.path = path
        self.coordinate_system = coordinate_system
        self.shoulder_pitch_gear_ratio = shoulder_pitch_gear_ratio

    def _read(self) -> dict:
        if not self.path.is_file():
            return {
                "version": 1,
                "coordinate_system": self.coordinate_system,
                "shoulder_pitch_gear_ratio": self.shoulder_pitch_gear_ratio,
                "poses": {},
                "motions": {},
                "recordings": {},
            }

        with self.path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("poses"), dict):
            raise ValueError(f"Unsupported or malformed FencingArm pose file: {self.path}")
        data.setdefault("motions", {})
        data.setdefault("recordings", {})
        if not isinstance(data["motions"], dict):
            raise ValueError(f"Malformed FencingArm motions in pose file: {self.path}")
        if not isinstance(data["recordings"], dict):
            raise ValueError(f"Malformed FencingArm recordings in pose file: {self.path}")
        if data.get("coordinate_system") != self.coordinate_system:
            raise ValueError(
                f"Pose file uses {data.get('coordinate_system')!r}, but the robot uses "
                f"{self.coordinate_system!r}. Record new poses or restore the matching use_degrees setting."
            )
        stored_ratio = data.get("shoulder_pitch_gear_ratio")
        if not isinstance(stored_ratio, (int, float)) or not math.isclose(
            stored_ratio, self.shoulder_pitch_gear_ratio
        ):
            raise ValueError(
                "Pose file shoulder-pitch gear ratio does not match the robot configuration. "
                "Record new poses or restore the matching ratio."
            )
        return data

    def save(self, name: str, pose: dict[str, float]) -> None:
        self._validate_name(name)
        self._validate_pose(pose)
        data = self._read()
        data["poses"][name] = {key: float(pose[key]) for key in FENCING_ARM_JOINT_KEYS}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        temporary_path.replace(self.path)

    def load(self, name: str) -> dict[str, float]:
        self._validate_name(name)
        data = self._read()
        try:
            pose = data["poses"][name]
        except KeyError as error:
            available = ", ".join(sorted(data["poses"])) or "none"
            raise ValueError(f"Pose {name!r} is not recorded. Available poses: {available}") from error
        self._validate_pose(pose)
        return {key: float(pose[key]) for key in FENCING_ARM_JOINT_KEYS}

    def names(self) -> list[str]:
        return sorted(self._read()["poses"])

    def save_motion(self, name: str, keyframes: list[str]) -> None:
        self._validate_name(name)
        if not keyframes:
            raise ValueError("A motion must contain at least one keyframe")
        data = self._read()
        for keyframe in keyframes:
            self._validate_name(keyframe)
            if keyframe not in data["poses"]:
                raise ValueError(f"Cannot save motion {name!r}: pose {keyframe!r} does not exist")
        data["motions"][name] = {
            "keyframes": [
                {
                    "pose": keyframe,
                    "duration_s": None,
                    "easing": "smoothstep",
                    "joint_timing": {},
                }
                for keyframe in keyframes
            ]
        }
        self._write(data)

    def load_motion(self, name: str) -> list[MotionKeyframe]:
        self._validate_name(name)
        data = self._read()
        try:
            motion = data["motions"][name]
        except KeyError as error:
            available = ", ".join(sorted(data["motions"])) or "none"
            raise ValueError(f"Motion {name!r} is not recorded. Available motions: {available}") from error
        # Backward compatibility with the original list-of-pose-names format.
        if isinstance(motion, list):
            motion = {"keyframes": [{"pose": pose} for pose in motion]}
        if not isinstance(motion, dict) or not isinstance(motion.get("keyframes"), list):
            raise ValueError(f"Motion {name!r} has malformed keyframes")
        keyframes = [self._parse_motion_keyframe(name, item) for item in motion["keyframes"]]
        if not keyframes:
            raise ValueError(f"Motion {name!r} must contain at least one keyframe")
        return keyframes

    def has_motion(self, name: str) -> bool:
        self._validate_name(name)
        return name in self._read()["motions"]

    def motion_names(self) -> list[str]:
        return sorted(self._read()["motions"])

    def save_recording(self, name: str, sample_fps: float, samples: list[dict]) -> None:
        self._validate_name(name)
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be greater than zero")
        if len(samples) < 2:
            raise ValueError("A recorded motion must contain at least two samples")
        parsed_samples = self._validate_recorded_samples(name, samples)
        data = self._read()
        data["recordings"][name] = {
            "sample_fps": float(sample_fps),
            "trim_start_s": 0.0,
            "trim_end_s": 0.0,
            "playback_speed": 1.0,
            "smoothing_window": 1,
            "joint_delays_s": {},
            "samples": parsed_samples,
        }
        self._write(data)

    def has_recording(self, name: str) -> bool:
        self._validate_name(name)
        return name in self._read()["recordings"]

    def recording_names(self) -> list[str]:
        return sorted(self._read()["recordings"])

    def load_recording(self, name: str) -> RecordedMotion:
        self._validate_name(name)
        data = self._read()
        return self._load_recording(name, data, set())

    def _load_recording(self, name: str, data: dict, resolving: set[str]) -> RecordedMotion:
        try:
            recording = data["recordings"][name]
        except KeyError as error:
            available = ", ".join(sorted(data["recordings"])) or "none"
            raise ValueError(f"Recording {name!r} is not saved. Available recordings: {available}") from error
        if not isinstance(recording, dict):
            raise ValueError(f"Recording {name!r} is malformed")
        if "derived_from" in recording:
            return self._load_derived_recording(name, recording, data, resolving)
        sample_fps = recording.get("sample_fps")
        trim_start_s = recording.get("trim_start_s", 0.0)
        trim_end_s = recording.get("trim_end_s", 0.0)
        playback_speed = recording.get("playback_speed", 1.0)
        smoothing_window = recording.get("smoothing_window", 1)
        joint_delays_s = self._parse_joint_delays(name, recording.get("joint_delays_s", {}))
        numeric_positive = (("sample_fps", sample_fps), ("playback_speed", playback_speed))
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for _, value in numeric_positive):
            raise ValueError(f"Recording {name!r} sample_fps and playback_speed must be positive")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in (trim_start_s, trim_end_s)
        ):
            raise ValueError(f"Recording {name!r} trim values must be non-negative")
        if not isinstance(smoothing_window, int) or smoothing_window < 1 or smoothing_window % 2 == 0:
            raise ValueError(f"Recording {name!r} smoothing_window must be a positive odd integer")
        samples = self._validate_recorded_samples(name, recording.get("samples"))
        duration = samples[-1]["time_s"]
        if trim_start_s + trim_end_s >= duration:
            raise ValueError(f"Recording {name!r} trims remove the entire recording")
        return RecordedMotion(
            sample_fps=float(sample_fps),
            trim_start_s=float(trim_start_s),
            trim_end_s=float(trim_end_s),
            playback_speed=float(playback_speed),
            smoothing_window=smoothing_window,
            joint_delays_s=joint_delays_s,
            samples=samples,
        )

    def _load_derived_recording(
        self,
        name: str,
        recording: dict,
        data: dict,
        resolving: set[str],
    ) -> RecordedMotion:
        if name in resolving:
            raise ValueError(f"Recording {name!r} has a circular derived_from reference")
        base_name = recording.get("derived_from")
        if not isinstance(base_name, str):
            raise ValueError(f"Derived recording {name!r} must specify derived_from")
        self._validate_name(base_name)
        base = self._load_recording(base_name, data, resolving | {name})
        overlays = recording.get("joint_overlays", {})
        if not isinstance(overlays, dict) or not overlays:
            raise ValueError(f"Derived recording {name!r} must contain joint_overlays")
        useful_start = base.trim_start_s
        useful_end = base.samples[-1]["time_s"] - base.trim_end_s
        useful_duration = useful_end - useful_start
        samples = [
            {"time_s": sample["time_s"], "positions": dict(sample["positions"])}
            for sample in base.samples
        ]
        for joint, overlay in overlays.items():
            if joint not in FENCING_ARM_JOINT_KEYS or not isinstance(overlay, dict):
                raise ValueError(f"Derived recording {name!r} has invalid overlay for {joint!r}")
            offset = overlay.get("offset_deg")
            start = overlay.get("start_fraction", 0.0)
            end = overlay.get("end_fraction", 1.0)
            easing = overlay.get("easing", "smoothstep")
            if not isinstance(offset, (int, float)) or not math.isfinite(offset):
                raise ValueError(f"Derived recording {name!r} overlay offset_deg must be finite")
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end)):
                raise ValueError(f"Derived recording {name!r} overlay timing must be finite")
            if not 0 <= start < end <= 1 or easing not in MOTION_EASINGS:
                raise ValueError(
                    f"Derived recording {name!r} overlay requires 0 <= start_fraction < "
                    f"end_fraction <= 1 and easing in {MOTION_EASINGS}"
                )
            for sample in samples:
                fraction = (sample["time_s"] - useful_start) / useful_duration
                local = max(0.0, min(1.0, (fraction - start) / (end - start)))
                sample["positions"][joint] += float(offset) * self._ease(local, easing)
        playback_speed = recording.get("playback_speed", base.playback_speed)
        smoothing_window = recording.get("smoothing_window", base.smoothing_window)
        joint_delays_s = dict(base.joint_delays_s)
        joint_delays_s.update(
            self._parse_joint_delays(name, recording.get("joint_delays_s", {}))
        )
        if not isinstance(playback_speed, (int, float)) or playback_speed <= 0:
            raise ValueError(f"Derived recording {name!r} playback_speed must be positive")
        if not isinstance(smoothing_window, int) or smoothing_window < 1 or smoothing_window % 2 == 0:
            raise ValueError(f"Derived recording {name!r} smoothing_window must be a positive odd integer")
        return RecordedMotion(
            sample_fps=base.sample_fps,
            trim_start_s=base.trim_start_s,
            trim_end_s=base.trim_end_s,
            playback_speed=float(playback_speed),
            smoothing_window=smoothing_window,
            joint_delays_s=joint_delays_s,
            samples=samples,
        )

    @staticmethod
    def _parse_joint_delays(name: str, delays: object) -> dict[str, float]:
        if not isinstance(delays, dict):
            raise ValueError(f"Recording {name!r} joint_delays_s must be an object")
        parsed = {}
        for joint, delay in delays.items():
            if joint not in FENCING_ARM_JOINT_KEYS:
                raise ValueError(f"Recording {name!r} has a delay for unknown joint {joint!r}")
            if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
                raise ValueError(f"Recording {name!r} joint delays must be finite and non-negative")
            parsed[joint] = float(delay)
        return parsed

    @staticmethod
    def _ease(fraction: float, easing: str) -> float:
        if easing == "linear":
            return fraction
        if easing == "smoothstep":
            return fraction * fraction * (3 - 2 * fraction)
        if easing == "smootherstep":
            return fraction**3 * (fraction * (fraction * 6 - 15) + 10)
        raise ValueError(f"Unsupported easing: {easing}")

    def _validate_recorded_samples(self, name: str, samples: object) -> list[dict]:
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError(f"Recording {name!r} must contain at least two samples")
        parsed = []
        previous_time = -1.0
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"Recording {name!r} contains a malformed sample")
            time_s = sample.get("time_s")
            positions = sample.get("positions")
            if (
                not isinstance(time_s, (int, float))
                or not math.isfinite(time_s)
                or time_s < 0
                or time_s <= previous_time
            ):
                raise ValueError(f"Recording {name!r} sample times must be finite and increasing")
            self._validate_pose(positions)
            parsed.append(
                {
                    "time_s": float(time_s),
                    "positions": {key: float(positions[key]) for key in FENCING_ARM_JOINT_KEYS},
                }
            )
            previous_time = time_s
        return parsed

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        temporary_path.replace(self.path)

    def _parse_motion_keyframe(self, motion_name: str, item: object) -> MotionKeyframe:
        if not isinstance(item, dict) or not isinstance(item.get("pose"), str):
            raise ValueError(f"Motion {motion_name!r} contains a malformed keyframe")
        pose = item["pose"]
        self._validate_name(pose)
        duration_s = item.get("duration_s")
        if duration_s is not None and (
            not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or duration_s <= 0
        ):
            raise ValueError(f"Motion {motion_name!r} keyframe {pose!r} has invalid duration_s")
        easing = item.get("easing", "smoothstep")
        if easing not in MOTION_EASINGS:
            raise ValueError(
                f"Motion {motion_name!r} keyframe {pose!r} easing must be one of {MOTION_EASINGS}"
            )
        joint_timing = item.get("joint_timing", {})
        if not isinstance(joint_timing, dict):
            raise ValueError(f"Motion {motion_name!r} keyframe {pose!r} has invalid joint_timing")
        parsed_timing = {}
        for joint, timing in joint_timing.items():
            if joint not in FENCING_ARM_JOINT_KEYS or not isinstance(timing, dict):
                raise ValueError(f"Motion {motion_name!r} keyframe {pose!r} has invalid timing for {joint!r}")
            start = timing.get("start_fraction", 0.0)
            end = timing.get("end_fraction", 1.0)
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end)):
                raise ValueError(f"Motion {motion_name!r} keyframe {pose!r} timing must be finite")
            if not 0 <= start < end <= 1:
                raise ValueError(
                    f"Motion {motion_name!r} keyframe {pose!r} timing must satisfy "
                    "0 <= start_fraction < end_fraction <= 1"
                )
            parsed_timing[joint] = {"start_fraction": float(start), "end_fraction": float(end)}
        return MotionKeyframe(
            pose=pose,
            duration_s=float(duration_s) if duration_s is not None else None,
            easing=easing,
            joint_timing=parsed_timing,
        )

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _POSE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "Pose names must start with a lowercase letter and contain only lowercase letters, "
                "numbers, underscores, or hyphens."
            )

    @staticmethod
    def _validate_pose(pose: dict[str, float]) -> None:
        if set(pose) != set(FENCING_ARM_JOINT_KEYS):
            raise ValueError(f"Pose joints must be exactly: {', '.join(FENCING_ARM_JOINT_KEYS)}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in pose.values()):
            raise ValueError("Every pose value must be a finite number")
