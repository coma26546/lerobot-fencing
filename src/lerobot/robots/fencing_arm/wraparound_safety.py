# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass, field
from enum import Enum


class WrapDirection(str, Enum):
    HIGH_TO_LOW = "high_to_low"
    LOW_TO_HIGH = "low_to_high"


@dataclass
class WraparoundSafety:
    """Latch encoder wrap events until the joint is manually returned."""

    low_threshold: int = 256
    high_threshold: int = 3839
    previous: dict[str, int] = field(default_factory=dict)
    latched: dict[str, WrapDirection] = field(default_factory=dict)
    crossed_back: set[str] = field(default_factory=set)

    def update(self, positions: dict[str, int]) -> tuple[set[str], set[str]]:
        triggered: set[str] = set()
        released: set[str] = set()

        for motor, current in positions.items():
            previous = self.previous.get(motor)
            if previous is None:
                self.previous[motor] = current
                continue

            direction = self.latched.get(motor)
            if direction is None:
                if previous >= self.high_threshold and current <= self.low_threshold:
                    self.latched[motor] = WrapDirection.HIGH_TO_LOW
                    triggered.add(motor)
                elif previous <= self.low_threshold and current >= self.high_threshold:
                    self.latched[motor] = WrapDirection.LOW_TO_HIGH
                    triggered.add(motor)
            elif direction is WrapDirection.HIGH_TO_LOW:
                returned_across_boundary = previous <= self.low_threshold and current >= self.high_threshold
                if motor not in self.crossed_back and returned_across_boundary:
                    self.crossed_back.add(motor)
                elif motor in self.crossed_back and current < previous and current >= self.high_threshold:
                    self._release(motor)
                    released.add(motor)
            elif direction is WrapDirection.LOW_TO_HIGH:
                returned_across_boundary = previous >= self.high_threshold and current <= self.low_threshold
                if motor not in self.crossed_back and returned_across_boundary:
                    self.crossed_back.add(motor)
                elif motor in self.crossed_back and current > previous and current <= self.low_threshold:
                    self._release(motor)
                    released.add(motor)

            self.previous[motor] = current

        return triggered, released

    @property
    def is_latched(self) -> bool:
        return bool(self.latched)

    def _release(self, motor: str) -> None:
        del self.latched[motor]
        self.crossed_back.discard(motor)
