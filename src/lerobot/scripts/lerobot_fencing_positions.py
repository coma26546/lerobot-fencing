# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hold and recall fixed arm poses for FencingBot (no teleop required).

Capture the current joint pose with a name, then type a 1-based index to move there.

Example:

```shell
lerobot-fencing-positions \\
    --robot.type=fencing_bot \\
    --robot.port=COM3 \\
    --robot.id=my_arm
```

Commands at the `fencing>` prompt:
  1, 2, 3, ...  Move to saved pose by number (poses stay until you send another command).
  capture NAME  Save current joint positions as a named pose (appended to the list).
  list          Show saved poses and numbers.
  delete N      Remove pose number N (1-based).
  reload        Re-read poses from disk.
  help          Show this summary.
  quit          Exit (arm stays at last commanded pose).
"""

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import RobotConfig, fencing_bot, make_robot_from_config  # noqa: F401
from lerobot.robots.fencing_bot.fencing_bot import FencingBot
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def _print_help() -> None:
    print(__doc__)


@dataclass
class FencingPositionsCLIConfig:
    robot: RobotConfig
    # If false, skip automatic calibration on connect (use when already calibrated).
    connect_calibrate: bool = True


def _interactive_loop(bot: FencingBot) -> None:
    poses = bot.load_saved_poses()
    print(f"Pose file: {bot.saved_positions_fpath}")
    print("Type 'help' for commands. 1-based pose numbers.\n")

    while True:
        try:
            line = input("fencing> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            break

        if cmd in ("h", "help", "?"):
            _print_help()
            continue

        if cmd == "list" or cmd == "l":
            if not poses:
                print("No saved poses. Use: capture <name>")
            else:
                for i, p in enumerate(poses, start=1):
                    print(f"  {i}. {p.get('name', '?')}")
            continue

        if cmd == "reload":
            poses = bot.load_saved_poses()
            print(f"Loaded {len(poses)} pose(s).")
            continue

        if cmd == "capture" or cmd == "c":
            name = parts[1].strip() if len(parts) > 1 else ""
            if not name:
                print("Usage: capture <name>")
                continue
            try:
                poses = bot.capture_pose(name, poses=poses)
                print(f"Saved as #{len(poses)}: {poses[-1]['name']}")
            except Exception as e:
                logger.exception("Capture failed")
                print(f"Error: {e}")
            continue

        if cmd == "delete" or cmd == "del":
            if len(parts) < 2 or not parts[1].strip().isdigit():
                print("Usage: delete <N>  (1-based index)")
                continue
            n = int(parts[1].strip())
            if n < 1 or n > len(poses):
                print(f"No pose #{n} (have {len(poses)}).")
                continue
            removed = poses.pop(n - 1)
            bot.write_saved_poses(poses)
            print(f"Removed #{n}: {removed.get('name', '?')}")
            continue

        if cmd.isdigit():
            n = int(cmd)
            try:
                bot.go_to_saved_pose(n, poses=poses)
                label = poses[n - 1].get("name", "?")
                print(f"Moving to #{n} ({label}).")
            except Exception as e:
                logger.exception("Go-to failed")
                print(f"Error: {e}")
            continue

        print("Unknown command. Try 'help'.")


@draccus.wrap()
def fencing_positions_main(cfg: FencingPositionsCLIConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    device = make_robot_from_config(cfg.robot)
    if not isinstance(device, FencingBot):
        raise ValueError("This script only supports --robot.type=fencing_bot")

    device.connect(calibrate=cfg.connect_calibrate)
    try:
        _interactive_loop(device)
    finally:
        device.disconnect()


def main() -> None:
    register_third_party_plugins()
    fencing_positions_main()


if __name__ == "__main__":
    main()
