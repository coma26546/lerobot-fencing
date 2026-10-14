"""Terminal wheel control: WASD velocity, arrows full power, Space stop, Q quit."""

import argparse
import math
import os
import signal
import sys
import time
from contextlib import contextmanager

from lerobot.robots.roboclaw import RoboClawDrive, RoboClawDriveConfig

PORT = "/dev/serial/by-id/usb-Basicmicro_Inc._USB_Roboclaw_2x15A-if00"
DIRECTIONS = {"w": (1, 1), "s": (-1, -1), "a": (-1, 1), "d": (1, -1)}
ARROWS = {"A": "w", "B": "s", "D": "a", "C": "d"}


@contextmanager
def terminal_keys():
    """Read terminal events without Enter; supports SSH and Windows consoles."""
    if not sys.stdin.isatty():
        raise RuntimeError("Run in an interactive terminal (SSH with a TTY is supported)")
    if os.name == "nt":
        import msvcrt

        def poll():
            events = []
            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    key = {"H": "A", "P": "B", "K": "D", "M": "C"}.get(msvcrt.getwch(), "")
                    events.append((ARROWS.get(key, ""), True))
                else:
                    events.append((key.lower(), False))
            return events

        yield poll
    else:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        pending = ""
        pending_at = 0.0

        def poll():
            nonlocal pending, pending_at
            events = []
            while select.select([fd], [], [], 0)[0]:
                data = os.read(fd, 1024)
                if not data:
                    raise EOFError("Keyboard input closed")
                pending += data.decode("ascii", errors="ignore")
                pending_at = time.monotonic()
            while pending:
                if pending[0] != "\x1b":
                    events.append((pending[0].lower(), False))
                    pending = pending[1:]
                elif len(pending) >= 3 and pending[1] in ("[", "O"):
                    events.append((ARROWS.get(pending[2], ""), True))
                    pending = pending[3:]
                elif time.monotonic() - pending_at > 0.05:
                    events.append(("\x1b", False))
                    pending = ""
                else:
                    break
            return events

        try:
            tty.setcbreak(fd)
            yield poll
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def run(drive, poll, speed, fast_duty, hold_timeout):
    target = None
    expires = 0.0
    while True:
        events = poll()
        # A stop in a batch takes priority over queued movement repeats.
        if any(key in ("q", "\x1b", "\x03") for key, _ in events):
            drive.stop()
            return
        if any(key == " " for key, _ in events):
            target = None
            drive.stop()
        else:
            for key, fast in events:
                if key in DIRECTIONS:
                    target = (key, fast)
                    expires = time.monotonic() + hold_timeout
        if target is not None:
            if time.monotonic() >= expires:
                drive.stop()
                target = None
            else:
                key, fast = target
                left, right = DIRECTIONS[key]
                magnitude = fast_duty if fast else speed
                command = drive.set_wheel_duties if fast else drive.set_wheel_speeds
                command(left * magnitude, right * magnitude)
        time.sleep(1 / 40)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--baud", type=int, default=38400)
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x80)
    parser.add_argument("--speed", type=int, default=400, help="WASD encoder counts/s (1..500)")
    parser.add_argument("--fast-duty", type=int, default=32767, help="Arrow power (1..32767; default 100%%)")
    parser.add_argument("--hold-timeout", type=float, default=0.2, help="Stop after this many seconds without a repeat")
    parser.add_argument("--left-motor", type=int, choices=(1, 2), default=1)
    parser.add_argument("--m1-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--m2-sign", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    if not 1 <= args.speed <= 500 or not 1 <= args.fast_duty <= 32767:
        parser.error("speed must be 1..500 and fast-duty must be 1..32767")
    if not math.isfinite(args.hold_timeout) or not 0.05 <= args.hold_timeout <= 0.5:
        parser.error("hold-timeout must be 0.05..0.5 seconds")
    config = RoboClawDriveConfig(
        port=args.port, baud=args.baud, address=args.address,
        left_motor=args.left_motor, m1_sign=args.m1_sign, m2_sign=args.m2_sign,
    )
    print(f"WASD: {args.speed} counts/s. Arrows: {args.fast_duty / 32767:.0%} motor power.")
    print("W/Up forward; S/Down reverse; A/Left rotate left; D/Right rotate right.")
    print("Space stops; Q/Esc/Ctrl+C quits. Keep the physical power stop ready.")
    print("Hold keys to repeat. Terminal repeat delay may cause a pause after the first tap.")

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        with terminal_keys() as poll, RoboClawDrive(config) as drive:
            run(drive, poll, args.speed, args.fast_duty, args.hold_timeout)
    except KeyboardInterrupt:
        print("Interrupted; wheel stop attempted.")
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
