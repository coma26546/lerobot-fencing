import json
import socket
import time

from lerobot.robots.fencing_robot.remote import TcpCommandSource


def _connect(source, token):
    client = socket.create_connection(source.address)
    client.sendall(f"AUTH {token}\n".encode())
    response = client.makefile("rb").readline(8)
    return client, response


def _poll(source, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if command := source.poll():
            return command
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for a remote command")


def test_authenticated_remote_sends_keyboard_commands():
    token = "test-remote-token-1234"
    with TcpCommandSource("127.0.0.1", 0, token) as source, _connect(source, token)[0] as client:
        for command in ("s", "r", "q"):
            client.sendall(command.encode() + b"\n")
            assert _poll(source) == command


def test_rejects_remote_with_wrong_token():
    with TcpCommandSource("127.0.0.1", 0, "test-remote-token-1234") as source:
        client, response = _connect(source, "incorrect-remote-token")
        with client:
            assert response == b"NO\n"
        assert source.poll() is None


def test_rejects_commands_outside_keyboard_mapping():
    token = "test-remote-token-1234"
    with TcpCommandSource("127.0.0.1", 0, token) as source:
        client, response = _connect(source, token)
        with client:
            assert response == b"OK\n"
            client.sendall(json.dumps({"event": "move_arm"}).encode() + b"\n")
        time.sleep(0.02)
        assert source.poll() is None


def test_rejects_short_token():
    try:
        TcpCommandSource("127.0.0.1", 0, "short")
    except ValueError as exc:
        assert "at least 16" in str(exc)
    else:
        raise AssertionError("Expected short token to be rejected")
