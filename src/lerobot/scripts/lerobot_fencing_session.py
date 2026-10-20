"""Interactive retreat/parry3, hold-or-parry4/cut5 drill session."""

import json
import signal
from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.robots.fencing_robot import FencingRobot, FencingRobotConfig
from lerobot.robots.fencing_robot.session import KeyboardCommands, ParryDrill, ParryDrillSession


@dataclass
class FencingSessionConfig:
    robot: FencingRobotConfig
    drill: Path
    dry_run: bool = False


@draccus.wrap()
def run(cfg: FencingSessionConfig) -> None:
    with cfg.drill.open(encoding="utf-8") as file:
        drill = draccus.decode(ParryDrill, json.load(file))
    robot = FencingRobot(cfg.robot)
    robot.arm.pose_store.load("rest")
    # Preparation needs no keyboard thread and validates moves before any connection.
    session = ParryDrillSession(robot, drill, commands=None)
    session.prepare()
    if cfg.dry_run:
        print("Drill configuration and all saved moves validated; hardware not connected.")
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    commands = None
    try:
        with robot:
            commands = KeyboardCommands()
            session.commands = commands
            session.run()
            robot.stop()
            robot.arm.move_to_rest()
    except KeyboardInterrupt:
        print("Interrupted; hardware cleanup attempted.")
    finally:
        if commands is not None:
            commands.close()
        signal.signal(signal.SIGTERM, previous)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
