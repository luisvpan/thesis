"""WebSocket Bridge client for Language Runtime communication."""

import asyncio
import json
import logging
import threading
from enum import Enum
from typing import Optional

import websockets
from websockets import ClientConnection

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"


class WebSocketConnectionRefusedError(Exception):
    """Raised when WebSocket server refuses the connection."""
    pass


class WebSocketConnectionClosedError(Exception):
    """Raised when WebSocket server closes connection unexpectedly."""
    pass


class WebSocketBridge:
    """
    WebSocket client for communicating touch events to Language Runtime.

    Handles connection lifecycle with exponential backoff retry on
    temporary disconnects, and sends JSON-serialized touch events.

    Attributes:
        url: WebSocket server URL (e.g., ws://localhost:3000/live).
        ws: WebSocket connection (optional, None when disconnected).
        state: Current connection state (DISCONNECTED, CONNECTING, CONNECTED).
        reconnect_enabled: Whether automatic reconnection is enabled.
        max_reconnect_delay: Maximum backoff delay in seconds (default: 60).
        base_reconnect_delay: Initial reconnect delay in seconds (default: 1).
        backoff_factor: Exponential backoff multiplier (default: 2).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        max_reconnect_delay: float = 60.0,
        base_reconnect_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """
        Initialize WebSocket Bridge.

        Args:
            url: WebSocket server URL. If None, uses LANGUAGE_RUNTIME_WS_URL env var.
            max_reconnect_delay: Maximum backoff delay in seconds (default: 60).
            base_reconnect_delay: Initial reconnect delay in seconds (default: 1).
            backoff_factor: Exponential backoff multiplier (default: 2).

        Raises:
            ValueError: If max_reconnect_delay < base_reconnect_delay.
        """
        import os

        if url is None:
            url = os.getenv("LANGUAGE_RUNTIME_WS_URL", "ws://localhost:3000/live")

        self.url = url
        self.ws: Optional[ClientConnection] = None
        self.state = ConnectionState.DISCONNECTED
        self.reconnect_enabled = True
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        if max_reconnect_delay < base_reconnect_delay:
            raise ValueError(
                f"max_reconnect_delay ({max_reconnect_delay}) must be greater than "
                f"base_reconnect_delay ({base_reconnect_delay})"
            )

        self.max_reconnect_delay = max_reconnect_delay
        self.base_reconnect_delay = base_reconnect_delay
        self.backoff_factor = backoff_factor
        self._reconnect_attempts = 0
        self._shutdown = False
        self._loop_thread_ident: Optional[int] = None

        logger.info(
            f"WebSocketBridge initialized with URL: {self.url}, "
            f"max_reconnect_delay={max_reconnect_delay}s, "
            f"base_reconnect_delay={base_reconnect_delay}s, "
            f"backoff_factor={backoff_factor}"
        )

    async def connect(self) -> None:
        """
        Establish WebSocket connection with exponential backoff retry.

        Raises:
            WebSocketConnectionRefusedError: If server rejects the connection.
        """
        if self.state != ConnectionState.DISCONNECTED:
            logger.warning(
                f"Already connecting or connected (state={self.state.value}), "
                "ignoring duplicate connect() call"
            )
            return

        if self._shutdown or not self.reconnect_enabled:
            return

        self.state = ConnectionState.CONNECTING
        self._reconnect_attempts = 0
        self.loop = asyncio.get_running_loop()

        while self.reconnect_enabled and not self._shutdown:
            try:
                logger.info(
                    f"Connecting to {self.url} (attempt {self._reconnect_attempts + 1})"
                )

                self.ws = await websockets.connect(self.url)
                self.state = ConnectionState.CONNECTED
                self._reconnect_attempts = 0
                logger.info(f"Connected successfully to {self.url}")

                asyncio.create_task(self._listen_for_messages())
                return

            except OSError as e:
                if e.errno == 111:
                    logger.error(f"Connection refused to {self.url}")
                    self.state = ConnectionState.DISCONNECTED
                    raise WebSocketConnectionRefusedError(
                        f"WebSocket connection refused to {self.url}"
                    ) from e

                backoff = self._calculate_backoff(self._reconnect_attempts)
                self.state = ConnectionState.DISCONNECTED
                self._reconnect_attempts += 1

                logger.warning(
                    f"Connection failed: {e}. "
                    f"Retrying in {backoff:.1f}s "
                    f"(attempt {self._reconnect_attempts + 1})"
                )

                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break

        self.state = ConnectionState.DISCONNECTED

    async def _listen_for_messages(self) -> None:
        """Listen for incoming WebSocket messages (pings, pongs, errors)."""
        if self.ws is None:
            return

        try:
            async for message in self.ws:
                logger.debug(f"Received message: {message[:100]}")

                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                assert isinstance(message, str)  # narrow the type for Pylance

                if message == "ping":
                    await self.ws.send("pong")
                    logger.debug("Sent pong")
                elif message.startswith("error:"):
                    logger.error(f"Error from server: {message}")
                else:
                    logger.debug(f"Unknown message type: {message[:50]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Server closed connection")
            self.state = ConnectionState.DISCONNECTED
            if self.reconnect_enabled and not self._shutdown:
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        asyncio.create_task(self.connect())
                except RuntimeError:
                    pass

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay capped at max_reconnect_delay.

        Args:
            attempt: Current reconnect attempt number (0-indexed).

        Returns:
            Delay in seconds with exponential backoff.
        """
        delay = self.base_reconnect_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_reconnect_delay)

    async def send_touch_event(self, touch: dict) -> None:
        """
        Send a touch event to the Language Runtime via WebSocket.

        Args:
            touch: Touch event dict with 'position' (x, y) and 'timestamp' fields.

        Raises:
            RuntimeError: If WebSocket is not connected.
        """
        if self.ws is None or self.state != ConnectionState.CONNECTED:
            raise RuntimeError("WebSocket is not connected. Call connect() first.")

        if "position" not in touch:
            raise ValueError("Touch event must have 'position' field")
        if "x" not in touch["position"]:
            raise ValueError("Position must have 'x' coordinate")
        if "y" not in touch["position"]:
            raise ValueError("Position must have 'y' coordinate")

        message = json.dumps(touch)

        logger.debug(
            f"Sending touch event: x={touch['position']['x']:.1f}, "
            f"y={touch['position']['y']:.1f}, "
            f"timestamp={touch['timestamp']}"
        )

        await self.ws.send(message)

    async def _graceful_shutdown(self) -> None:
        try:
            if self.ws is not None:
                await self.ws.close()
        except Exception as e:
            logger.debug("WebSocket close during shutdown: %s", e)
        finally:
            self.ws = None
            self.state = ConnectionState.DISCONNECTED
        loop = asyncio.get_running_loop()
        loop.call_soon(loop.stop)

    def disconnect(self) -> None:
        logger.info("Disconnecting from WebSocket...")
        self._shutdown = True
        self.reconnect_enabled = False

        loop = self.loop
        if loop is None or not loop.is_running():
            self.ws = None
            self.state = ConnectionState.DISCONNECTED
            logger.info("WebSocket disconnected")
            return

        loop_tid = self._loop_thread_ident
        same_thread = loop_tid is not None and threading.get_ident() == loop_tid
        if same_thread:
            asyncio.create_task(self._graceful_shutdown())
            logger.info("WebSocket disconnected (shutdown scheduled on loop thread)")
            return

        fut = asyncio.run_coroutine_threadsafe(self._graceful_shutdown(), loop)
        try:
            fut.result(timeout=15.0)
        except Exception as e:
            logger.warning("Graceful WebSocket shutdown failed: %s", e)
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            self.ws = None
            self.state = ConnectionState.DISCONNECTED

        logger.info("WebSocket disconnected")

    async def disconnect_async(self) -> None:
        """
        Graceful async disconnect with proper closing handshake.

        Use this when calling from an async context to ensure the server
        receives a proper WebSocket close frame.
        """
        logger.info("Disconnecting from WebSocket (async)...")
        self._shutdown = True
        self.reconnect_enabled = False

        if self.ws is not None:
            try:
                await self.ws.close()
                logger.info(f"WebSocket connection to {self.url} closed gracefully")
            except Exception as e:
                logger.warning(f"Error closing WebSocket connection: {e}")

        self.state = ConnectionState.DISCONNECTED
        self.ws = None