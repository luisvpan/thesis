"""Start the IDE relay (FastAPI) then cv-system in one process tree."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _wait_for_relay(health_url: str, timeout_s: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(health_url, timeout=1.0)
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.12)
    return False


def main() -> None:
    host = os.environ.get("IDE_RELAY_HOST", "127.0.0.1")
    port = os.environ.get("IDE_RELAY_PORT", "8765")
    health = os.environ.get(
        "CV_STACK_HEALTH_URL",
        f"http://{host}:{port}/api/v1/health",
    )

    relay = subprocess.Popen(
        [sys.executable, "-m", "cv_system.bridge.ide_relay_server"],
        stdout=None,
        stderr=None,
    )
    rc = 1
    try:
        if not _wait_for_relay(health):
            print(
                "cv-stack: IDE relay did not respond at "
                f"{health} (timeout). Is the port free?",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("cv-stack: IDE relay OK; starting cv-system...")
        rc = subprocess.call(
            [sys.executable, "-c", "from cv_system.main import main; main()"],
        )
    finally:
        relay.terminate()
        try:
            relay.wait(timeout=8)
        except subprocess.TimeoutExpired:
            relay.kill()
            relay.wait(timeout=3)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
