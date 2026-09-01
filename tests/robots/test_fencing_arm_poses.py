# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import pytest

from lerobot.robots.fencing_arm import FENCING_ARM_JOINT_KEYS, FencingArmPoseStore, MotionKeyframe


def _pose(offset: float = 0.0) -> dict[str, float]:
    return {key: index + offset for index, key in enumerate(FENCING_ARM_JOINT_KEYS)}


def test_save_and_load_named_poses(tmp_path):
    store = FencingArmPoseStore(tmp_path / "arm.json", "motor_degrees", 2.0)

    store.save("enguard", _pose())
    store.save("lunge", _pose(10.0))

    assert store.names() == ["enguard", "lunge"]
    assert store.load("enguard") == _pose()
    assert store.load("lunge") == _pose(10.0)


def test_save_and_load_motion_keyframes(tmp_path):
    store = FencingArmPoseStore(tmp_path / "arm.json", "motor_degrees", 2.0)
    store.save("cut5__keyframe_01", _pose())
    store.save("cut5__keyframe_02", _pose(10.0))

    store.save_motion("cut5", ["cut5__keyframe_01", "cut5__keyframe_02"])

    assert store.has_motion("cut5")
    assert store.motion_names() == ["cut5"]
    assert store.load_motion("cut5") == [
        MotionKeyframe(pose="cut5__keyframe_01"),
        MotionKeyframe(pose="cut5__keyframe_02"),
    ]


def test_old_pose_file_without_motions_remains_compatible(tmp_path):
    store = FencingArmPoseStore(tmp_path / "arm.json", "motor_degrees", 2.0)
    store.save("enguard", _pose())

    assert not store.has_motion("cut5")


def test_save_and_load_recorded_motion(tmp_path):
    store = FencingArmPoseStore(tmp_path / "arm.json", "motor_degrees", 2.0)
    samples = [
        {"time_s": 0.0, "positions": _pose()},
        {"time_s": 0.1, "positions": _pose(1.0)},
    ]

    store.save_recording("cut5_demo", 30.0, samples)
    recording = store.load_recording("cut5_demo")

    assert recording.sample_fps == 30.0
    assert recording.playback_speed == 1.0
    assert recording.smoothing_window == 1
    assert recording.joint_delays_s == {}
    assert recording.samples == samples
    assert store.recording_names() == ["cut5_demo"]


def test_derived_recording_applies_timed_joint_offset(tmp_path):
    path = tmp_path / "arm.json"
    store = FencingArmPoseStore(path, "motor_degrees", 2.0)
    samples = [
        {"time_s": 0.0, "positions": _pose()},
        {"time_s": 0.5, "positions": _pose()},
        {"time_s": 1.0, "positions": _pose()},
    ]
    store.save_recording("cut5_demo", 30.0, samples)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["recordings"]["cut4_demo"] = {
        "derived_from": "cut5_demo",
        "joint_overlays": {
            "elbow_roll.pos": {
                "offset_deg": 90.0,
                "start_fraction": 0.25,
                "end_fraction": 0.75,
                "easing": "smoothstep",
            }
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    derived = store.load_recording("cut4_demo")

    base_elbow_roll = _pose()["elbow_roll.pos"]
    assert derived.samples[0]["positions"]["elbow_roll.pos"] == base_elbow_roll
    assert derived.samples[1]["positions"]["elbow_roll.pos"] == base_elbow_roll + 45.0
    assert derived.samples[2]["positions"]["elbow_roll.pos"] == base_elbow_roll + 90.0


def test_recording_joint_delays_are_loaded_and_inherited(tmp_path):
    path = tmp_path / "arm.json"
    store = FencingArmPoseStore(path, "motor_degrees", 2.0)
    samples = [
        {"time_s": 0.0, "positions": _pose()},
        {"time_s": 1.0, "positions": _pose(1.0)},
    ]
    store.save_recording("cut5", 30.0, samples)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["recordings"]["cut5"]["joint_delays_s"] = {"wrist_pitch.pos": 0.2}
    data["recordings"]["cut4"] = {
        "derived_from": "cut5",
        "joint_overlays": {
            "elbow_roll.pos": {"offset_deg": 90.0},
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    assert store.load_recording("cut5").joint_delays_s == {"wrist_pitch.pos": 0.2}
    assert store.load_recording("cut4").joint_delays_s == {"wrist_pitch.pos": 0.2}


def test_recording_rejects_negative_joint_delay(tmp_path):
    path = tmp_path / "arm.json"
    store = FencingArmPoseStore(path, "motor_degrees", 2.0)
    store.save_recording(
        "cut5",
        30.0,
        [
            {"time_s": 0.0, "positions": _pose()},
            {"time_s": 1.0, "positions": _pose(1.0)},
        ],
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["recordings"]["cut5"]["joint_delays_s"] = {"wrist_pitch.pos": -0.1}
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative"):
        store.load_recording("cut5")


def test_missing_pose_lists_available_names(tmp_path):
    store = FencingArmPoseStore(tmp_path / "arm.json", "motor_degrees", 2.0)
    store.save("enguard", _pose())

    with pytest.raises(ValueError, match="Available poses: enguard"):
        store.load("lunge")


@pytest.mark.parametrize("name", ["En Garde", "../enguard", "1enguard", ""])
def test_pose_name_must_be_safe(name, tmp_path):
    store = FencingArmPoseStore(tmp_path / "arm.json", "motor_degrees", 2.0)

    with pytest.raises(ValueError, match="Pose names"):
        store.save(name, _pose())


def test_pose_file_rejects_coordinate_changes(tmp_path):
    path = tmp_path / "arm.json"
    FencingArmPoseStore(path, "motor_degrees", 2.0).save("enguard", _pose())

    with pytest.raises(ValueError, match="coordinate"):
        FencingArmPoseStore(path, "normalized_range_m100_100", 2.0).load("enguard")
