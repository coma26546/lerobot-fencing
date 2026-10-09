"""Drive one metre forward, pause, and return using wheel encoder feedback."""

import argparse
import math
import signal
import time

from lerobot.robots.roboclaw import RoboClawDrive, RoboClawDriveConfig


COUNTS_PER_METRE = 753.2 / (math.pi * 0.120)
PORT = "/dev/serial/by-id/usb-Basicmicro_Inc._USB_Roboclaw_2x15A-if00"


def move_to(drive, targets, direction, speed, timeout_s=30.0):
    """Approach both targets without reversing to correct overshoot."""
    started = time.monotonic()
    previous = drive.get_encoders()
    progress_at = dict.fromkeys(targets, started)
    try:
        while True:
            now = time.monotonic()
            positions = drive.get_encoders()
            commands = {}
            for wheel, target in targets.items():
                remaining = direction * (target - positions[wheel])
                if remaining <= 10:
                    commands[wheel] = 0
                    continue
                if direction * (positions[wheel] - previous[wheel]) >= 2:
                    previous[wheel] = positions[wheel]
                    progress_at[wheel] = now
                if direction * (positions[wheel] - previous[wheel]) < -20:
                    raise RuntimeError(f"{wheel} encoder moved in the wrong direction")
                if now - progress_at[wheel] > 2.0:
                    raise RuntimeError(f"{wheel} wheel stalled or its encoder did not advance")
                commands[wheel] = direction * min(speed, max(30, round(remaining * 2)))
            if not any(commands.values()):
                return
            if now - started > timeout_s:
                raise TimeoutError("Wheel movement exceeded its time limit")
            drive.set_wheel_speeds(commands["left"], commands["right"])
            time.sleep(1 / 40)
    finally:
        drive.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--speed", type=int, default=400, help="Encoder counts/s; default about 0.20 m/s")
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--left-motor", type=int, choices=(1, 2), default=1)
    parser.add_argument("--m1-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--m2-sign", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    if not 30 <= args.speed <= 500:
        parser.error("--speed must be between 30 and 500 counts/s")
    if not math.isfinite(args.pause) or args.pause < 0:
        parser.error("--pause must be finite and nonnegative")
    config = RoboClawDriveConfig(
        port=args.port, left_motor=args.left_motor, m1_sign=args.m1_sign, m2_sign=args.m2_sign
    )
    distance = round(COUNTS_PER_METRE)
    print(f"1 metre = {distance} counts per wheel; speed limit {args.speed / COUNTS_PER_METRE:.3f} m/s")
    input("Clear the forward/return path and have the power stop ready. Press Enter to begin: ")

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        with RoboClawDrive(config) as drive:
            home = drive.get_encoders()
            print("Forward 1 metre", flush=True)
            move_to(drive, {wheel: value + distance for wheel, value in home.items()}, 1, args.speed)
            forward = drive.get_encoders()
            print(f"Forward encoder travel: { {w: forward[w] - home[w] for w in home} }", flush=True)
            print(f"Stopped; pausing {args.pause:g} seconds", flush=True)
            time.sleep(args.pause)
            print("Reversing to starting encoder positions", flush=True)
            move_to(drive, home, -1, args.speed)
            end = drive.get_encoders()
            print(f"Finished. Encoder error at start: { {w: end[w] - home[w] for w in home} }")
    except KeyboardInterrupt:
        print("Interrupted; wheel stop attempted.")
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
