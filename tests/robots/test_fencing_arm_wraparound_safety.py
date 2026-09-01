# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from lerobot.robots.fencing_arm.wraparound_safety import WraparoundSafety


def test_high_to_low_wrap_latches_until_joint_returns_and_descends():
    safety = WraparoundSafety()

    assert safety.update({"joint": 4000}) == (set(), set())
    triggered, released = safety.update({"joint": 50})
    assert triggered == {"joint"}
    assert not released
    assert safety.is_latched

    # Returning across zero is not enough; feedback must begin moving down
    # from the high endpoint before new goals are allowed again.
    assert safety.update({"joint": 4095}) == (set(), set())
    triggered, released = safety.update({"joint": 4094})
    assert not triggered
    assert released == {"joint"}
    assert not safety.is_latched


def test_low_to_high_wrap_uses_symmetric_return_behavior():
    safety = WraparoundSafety()

    safety.update({"joint": 50})
    triggered, _ = safety.update({"joint": 4000})
    assert triggered == {"joint"}

    assert safety.update({"joint": 0}) == (set(), set())
    _, released = safety.update({"joint": 1})
    assert released == {"joint"}
    assert not safety.is_latched


def test_normal_motion_near_but_not_across_boundary_does_not_latch():
    safety = WraparoundSafety()

    for position in (3800, 3900, 4000, 3900, 3800):
        assert safety.update({"joint": position}) == (set(), set())

    assert not safety.is_latched
