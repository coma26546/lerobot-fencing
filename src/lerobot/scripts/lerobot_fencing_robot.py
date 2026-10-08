"""Run a JSON fencing drill with one arm/base timing loop."""

import json
import signal
from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.robots.fencing_robot import FencingRobot, FencingRobotConfig
from lerobot.robots.fencing_robot.drills import DrillStep, prepare_drill, run_drill


@dataclass
class FencingDrillConfig:
    robot: FencingRobotConfig
    drill: Path
    hz: float = 40.0
    dry_run: bool = False


@draccus.wrap()
def run(cfg: FencingDrillConfig) -> None:
    with cfg.drill.open(encoding="utf-8") as file:
        data = json.load(file)
    steps = [DrillStep(**step) for step in data["steps"]]
    duration_s = data["duration_s"]
    robot = FencingRobot(cfg.robot)
    prepare_drill(robot, steps, duration_s, cfg.hz)
    if cfg.dry_run:
        print(f"Validated {len(steps)} steps over {duration_s:g} seconds; hardware not connected.")
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        with robot:
            run_drill(robot, steps, duration_s, cfg.hz)
    except KeyboardInterrupt:
        print("Drill interrupted; hardware cleanup attempted.")
    finally:
        signal.signal(signal.SIGTERM, previous)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
