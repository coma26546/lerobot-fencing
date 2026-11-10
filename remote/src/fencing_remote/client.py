"""Send button presses to the fencing drill over a private TCP connection."""

import argparse
import queue
import socket
import sys
import threading
from collections.abc import Callable

BUTTONS = {
    "enguard": ("g", 17),
    "start": ("s", 24),
    "replay": ("r", 27),
    "end": ("q", 22),
}
RECONNECT_DELAY_S = 1.0


def _connect(host: str, port: int, token: str) -> socket.socket:
    client = socket.create_connection((host, port), timeout=3.0)
    client.settimeout(3.0)
    client.sendall(f"AUTH {token}\n".encode())
    with client.makefile("rb") as response:
        result = response.readline(8)
    if result != b"OK\n":
        client.close()
        raise PermissionError(
            f"Robot rejected the remote connection: {result.decode(errors='replace').strip()}"
        )
    return client


def _clear_events(events: queue.SimpleQueue[str]) -> None:
    while True:
        try:
            events.get_nowait()
        except queue.Empty:
            return


def _send_loop(
    host: str,
    port: int,
    token: str,
    events: queue.SimpleQueue[str],
    stopping: threading.Event,
) -> None:
    while not stopping.is_set():
        try:
            with _connect(host, port, token) as client:
                _clear_events(events)
                print(f"Connected to fencing robot at {host}:{port}.")
                while not stopping.is_set():
                    try:
                        command = events.get_nowait()
                    except queue.Empty:
                        stopping.wait(0.02)
                        continue
                    client.sendall(command.encode("ascii") + b"\n")
                    print(f"Sent {command}.")
        except PermissionError:
            raise
        except OSError as exc:
            if stopping.is_set():
                return
            print(f"Remote disconnected ({exc}); retrying.", file=sys.stderr)
            stopping.wait(RECONNECT_DELAY_S)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi GPIO buttons for the fencing drill")
    parser.add_argument("--host", required=True, help="Robot Pi 5 IP address or hostname")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", required=True, help="Shared token configured on the robot")
    parser.add_argument("--enguard-pin", type=int, default=BUTTONS["enguard"][1], help="BCM pin")
    parser.add_argument("--start-pin", type=int, default=BUTTONS["start"][1], help="BCM pin")
    parser.add_argument("--replay-pin", type=int, default=BUTTONS["replay"][1], help="BCM pin")
    parser.add_argument("--end-pin", type=int, default=BUTTONS["end"][1], help="BCM pin")
    return parser


def run(
    args: argparse.Namespace,
    button_factory: Callable[..., object] | None = None,
) -> None:
    if len(args.token) < 16:
        raise ValueError("token must contain at least 16 characters")
    if not 0 < args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    pins = {
        "enguard": args.enguard_pin,
        "start": args.start_pin,
        "replay": args.replay_pin,
        "end": args.end_pin,
    }
    if any(pin < 0 for pin in pins.values()) or len(set(pins.values())) != len(pins):
        raise ValueError("Use a different nonnegative BCM GPIO pin for each button")

    if button_factory is None:
        from gpiozero import Button

        button_factory = Button

    events: queue.SimpleQueue[str] = queue.SimpleQueue()
    stopping = threading.Event()
    buttons = []
    try:
        for pin in pins.values():
            buttons.append(button_factory(pin, pull_up=True, bounce_time=0.08))
        for name, button in zip(pins, buttons, strict=True):
            command = BUTTONS[name][0]
            button.when_pressed = lambda command=command: events.put(command)
        print("Buttons: enguard moves to standby; start runs a rep; replay resets; end exits to rest.")
        _send_loop(args.host, args.port, args.token, events, stopping)
    except KeyboardInterrupt:
        print("Remote stopped.")
    finally:
        stopping.set()
        for button in buttons:
            button.close()


def main() -> None:
    run(_make_parser().parse_args())


if __name__ == "__main__":
    main()
