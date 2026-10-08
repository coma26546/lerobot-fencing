"""Bounded, low-speed wheel tests: lerobot-roboclaw --help."""

import argparse
import math
import signal
import time

from lerobot.robots.roboclaw import RoboClawDrive, RoboClawDriveConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("status", "stop", "forward", "backward", "left", "right", "wheels")
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=38400)
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x80)
    parser.add_argument("--left-motor", type=int, choices=(1, 2), default=1)
    parser.add_argument("--m1-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--m2-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--max-speed", type=int, default=500, help="Absolute limit in encoder counts/s")
    parser.add_argument("--speed", type=int, default=100, help="Magnitude for directional commands")
    parser.add_argument("--left-speed", type=int, default=0)
    parser.add_argument("--right-speed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=1.0, help="Motion duration in seconds")
    parser.add_argument("--hz", type=float, default=40.0, help="Command refresh rate")
    args = parser.parse_args()
    config = RoboClawDriveConfig(
        port=args.port,
        baud=args.baud,
        address=args.address,
        left_motor=args.left_motor,
        m1_sign=args.m1_sign,
        m2_sign=args.m2_sign,
        max_speed=args.max_speed,
    )
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not math.isfinite(args.hz) or not 10 <= args.hz <= 100:
        parser.error("--hz must be between 10 and 100")
    if not 0 <= args.speed <= config.max_speed:
        parser.error("--speed must be between zero and --max-speed")
    if any(abs(value) > config.max_speed for value in (args.left_speed, args.right_speed)):
        parser.error("Wheel speeds must be within --max-speed")
    speeds = {
        "forward": (args.speed, args.speed),
        "backward": (-args.speed, -args.speed),
        "left": (-args.speed, args.speed),
        "right": (args.speed, -args.speed),
        "wheels": (args.left_speed, args.right_speed),
    }

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        with RoboClawDrive(config) as drive:
            if args.command == "stop":
                drive.stop()
            elif args.command == "status":
                print(f"Version: {drive.get_version()}")
                print(f"Battery: {drive.get_battery_voltage():.1f} V")
                print(f"Encoders: {drive.get_encoders()}")
                print(f"Speeds: {drive.get_speeds()}")
            else:
                left, right = speeds[args.command]
                print(f"Driving left={left}, right={right} counts/s for {args.duration:g} s")
                deadline = time.monotonic() + args.duration
                try:
                    while time.monotonic() < deadline:
                        drive.set_wheel_speeds(left, right)
                        time.sleep(max(0, min(1 / args.hz, deadline - time.monotonic())))
                finally:
                    drive.stop()
    except KeyboardInterrupt:
        print("Interrupted; wheel stop attempted.")
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
