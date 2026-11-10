"""TCP command input for the fencing drill remote."""

import hmac
import logging
import queue
import socket
import socketserver
import threading
from contextlib import suppress

logger = logging.getLogger(__name__)
VALID_COMMANDS = {"g", "s", "r", "q"}


class _RemoteServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], source: "TcpCommandSource"):
        self.source = source
        super().__init__(address, _RemoteHandler)


class _RemoteHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        source = self.server.source
        registered = False
        try:
            self.request.settimeout(5.0)
            auth = self.rfile.readline(130)
            expected = f"AUTH {source.token}\n".encode()
            if len(auth) > 129 or not hmac.compare_digest(auth, expected):
                self.wfile.write(b"NO\n")
                return
            if not source._register(self.request):
                self.wfile.write(b"BUSY\n")
                return
            registered = True
            self.request.settimeout(None)
            self.wfile.write(b"OK\n")
            self.wfile.flush()
            while line := self.rfile.readline(4):
                if len(line) > 3 or not line.endswith(b"\n"):
                    logger.warning("Rejected oversized fencing remote command")
                    return
                try:
                    command = line.strip().decode("ascii")
                except UnicodeDecodeError:
                    logger.warning("Rejected non-ASCII fencing remote command")
                    return
                if command not in VALID_COMMANDS:
                    logger.warning("Rejected unknown fencing remote command: %r", command)
                    return
                source._commands.put(command)
        except OSError as exc:
            logger.info("Fencing remote disconnected: %s", exc)
        finally:
            if registered:
                source._unregister(self.request)


class TcpCommandSource:
    """Receive keyboard-equivalent commands from one authenticated TCP client."""

    def __init__(self, host: str, port: int, token: str):
        if len(token) < 16:
            raise ValueError("remote token must contain at least 16 characters")
        if not 0 <= port <= 65535:
            raise ValueError("remote port must be between 0 and 65535")
        self.token = token
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._client: socket.socket | None = None
        self._closed = False
        self._server = _RemoteServer((host, port), self)
        self._thread = threading.Thread(target=self._server.serve_forever, name="fencing-remote-listener")
        self._thread.start()

    @property
    def address(self) -> tuple[str, int]:
        """Return the listener host and port."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def _register(self, client: socket.socket) -> bool:
        with self._lock:
            if self._closed or self._client is not None:
                return False
            self._client = client
            return True

    def _unregister(self, client: socket.socket) -> None:
        with self._lock:
            if self._client is client:
                self._client = None

    def poll(self) -> str | None:
        """Return the next command without blocking the drill loop."""
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        """Close the listener and any remote connection."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
        if client is not None:
            with suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
        return False
