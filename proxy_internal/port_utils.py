"""Safe port management utilities with process verification."""

import logging
import os
import signal
import socket
import subprocess
import sys
import time

logger = logging.getLogger("port_utils")

_PROXY_PATTERNS = ("proxy.py", "mcp_server.py", "uvicorn")


def _is_proxy_process(pid: int) -> bool:
    """Check if a PID belongs to a proxy-related process via ps."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
        cmd = result.stdout.strip().lower()
        return any(pattern in cmd for pattern in _PROXY_PATTERNS)
    except Exception:
        return False


def _wait_for_exit(pid: int, timeout: float = 3.0, poll_interval: float = 0.5) -> bool:
    """Poll until a process exits. Returns True if the process is gone."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(poll_interval)
    return False


def kill_stale_port(port: int, force: bool = False) -> bool:
    """Kill stale proxy processes on a port. Returns True if any were killed."""
    if sys.platform == "win32":
        logger.warning(f"Automatic port freeing not supported on Windows (port {port})")
        return False

    force = force or os.environ.get("PROXY_FORCE_PORT_KILL") == "1"

    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p for p in result.stdout.strip().split() if p and p.isdigit()]
        if not pids:
            return False
    except Exception as e:
        logger.debug(f"Could not query port {port}: {e}")
        return False

    my_pid = str(os.getpid())
    killed = False

    for pid_str in pids:
        if pid_str == my_pid:
            continue
        pid = int(pid_str)

        if not force and not _is_proxy_process(pid):
            logger.info(f"Skipping non-proxy process {pid} on port {port}")
            continue

        logger.warning(f"Sending SIGTERM to process {pid} on port {port}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            killed = True
            continue
        except PermissionError:
            logger.warning(f"No permission to kill process {pid}")
            continue

        if _wait_for_exit(pid):
            killed = True
            continue

        logger.warning(f"Process {pid} did not exit, sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.3)
            killed = True
        except (ProcessLookupError, PermissionError):
            killed = True

    return killed


def ensure_port_free(port: int, host: str = "0.0.0.0", max_attempts: int = 3):
    """Ensure a port is free by test-binding, killing stale processes if needed."""
    for attempt in range(1, max_attempts + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            sock.close()
            return
        except OSError:
            sock.close()
            if attempt == max_attempts:
                logger.error(f"Port {port} still in use after {max_attempts} attempts")
                raise
            logger.warning(f"Port {port} in use (attempt {attempt}/{max_attempts}), killing stale process...")
            kill_stale_port(port)
            time.sleep(1)
