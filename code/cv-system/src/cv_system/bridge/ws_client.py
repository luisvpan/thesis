"""WebSocket Bridge client for Language Runtime communication."""

import asyncio
import json
import logging
from enum import Enum
from typing import Optional

import websockets

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """Connection states for WebSocket lifecycle tracking."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"


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
        # Get URL from environment variable if not provided
        import os
        if url is None:
            url = os.getenv("LANGUAGE_RUNTIME_WS_URL", "ws://localhost:3000/live")

        self.url = url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.state = ConnectionState.DISCONNECTED
        self.reconnect_enabled = True

        # Validate retry parameters
        if max_reconnect_delay < base_reconnect_delay:
            raise ValueError(
                f"max_reconnect_delay ({max_reconnect_delay}) must be greater than "
                f"base_reconnect_delay ({base_reconnect_delay})"
            )

        self.max_reconnect_delay = max_reconnect_delay
        self.base_reconnect_delay = base_reconnect_delay
        self.backoff_factor = backoff_factor
        self._reconnect_attempts = 0

        logger.info(
            f"WebSocketBridge initialized with URL: {self.url}, "
            f"max_reconnect_delay={max_reconnect_delay}s, "
            f"base_reconnect_delay={base_reconnect_delay}s, "
            f"backoff_factor={backoff_factor}"
        )

    async def connect(self) -> None:
        """
        Establish WebSocket connection with exponential backoff retry on temporary disconnects.

        Raises:
            ConnectionRefusedError: If WebSocket server rejects the connection.
            ConnectionClosedError: If server closes connection unexpectedly.
        """
        if self.state != ConnectionState.DISCONNECTED:
            logger.warning(
                f"Already connecting or connected (state={self.state.value}), "
                "ignoring duplicate connect() call"
            )
            return

        self.state = ConnectionState.CONNECTING
        self._reconnect_attempts = 0

        while self.reconnect_enabled:
            try:
                logger.info(
                    f"Connecting to {self.url} "
                    f"(attempt {self._reconnect_attempts + 1})"
                )

                self.ws = await websockets.connect(self.url)
                self.state = ConnectionState.CONNECTED
                self._reconnect_attempts = 0
                logger.info(f"Connected successfully to {self.url}")

                # Listen for incoming messages (for pings/pongs)
                asyncio.create_task(self._listen_for_messages())

                return  # Connection successful

            except OSError as e:
                if e.errno == 111:  # Connection refused
                    logger.error(f"Connection refused to {self.url}")
                    self.state = ConnectionState.DISCONNECTED
                    raise ConnectionRefusedError(
                        f"WebSocket connection refused to {self.url}"
                    ) from e

                # Other network errors are considered temporary
                backoff = self._calculate_backoff(self._reconnect_attempts)
                self.state = ConnectionState.DISCONNECTED
                self._reconnect_attempts += 1

                logger.warning(
                    f"Connection failed: {e}. "
                    f"Retrying in {backoff:.1f}s "
                    f"(attempt {self._reconnect_attempts + 1})"
                )

                await asyncio.sleep(backoff)

    async def _listen_for_messages(self) -> None:
        """Listen for incoming WebSocket messages (pings, pongs, errors)."""
        if self.ws is None:
            return

        try:
            async for message in self.ws:
                logger.debug(f"Received message: {message[:100]}")  # Truncate for logging

                # Handle different message types
                if message == "ping":
                    await self.ws.send("pong")
                    logger.debug("Sent pong")
                elif message.startswith("error:"):
                    logger.error(f"Error from server: {message}")
                else:
                    logger.debug(f"Unknown message type: {message[:50]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Server closed connection")
            if self.state == ConnectionState.CONNECTED and self.reconnect_enabled:
                # Attempt reconnection with exponential backoff
                asyncio.create_task(self.connect())

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay capped at max_reconnect_delay.

        Args:
            attempt: Current reconnect attempt number (0-indexed).

        Returns:
            Delay in seconds with exponential backoff.
        """
        # Exponential backoff: delay = base * (factor ^ attempt)
        delay = self.base_reconnect_delay * (self.backoff_factor**attempt)

        # Cap at maximum
        return min(delay, self.max_reconnect_delay)

    async def send_touch_event(self, touch: dict) -> None:
        """
        Send a touch event to the Language Runtime via WebSocket.

        Args:
            touch: Touch event dict with 'position' (x, y in projector space) and
                   'timestamp' (ISO 8601 format) fields.

        Raises:
            RuntimeError: If WebSocket is not connected.
        """
        if self.ws is None or self.state != ConnectionState.CONNECTED:
            raise RuntimeError(
                "WebSocket is not connected. Call connect() first."
            )

        # Validate touch event structure
        if "position" not in touch:
            raise ValueError("Touch event must have 'position' field")
        if "x" not in touch["position"]:
            raise ValueError("Position must have 'x' coordinate")
        if "y" not in touch["position"]:
            raise ValueError("Position must have 'y' coordinate")

        # Construct JSON message
        message = json.dumps(touch)

        logger.debug(
            f"Sending touch event: x={touch['position']['x']:.1f}, "
            f"y={touch['position']['y']:.1f}, "
            f"timestamp={touch['timestamp']}"
        )

        await self.ws.send(message)

    async def disconnect(self) -> None:
        """
        Gracefully disconnect from WebSocket server.

        Disables automatic reconnection and closes the connection if open.
        """
        logger.info("Disconnecting from WebSocket...")
        self.reconnect_enabled = False

        if self.ws is not None:
            try:
                await self.ws.close()
                logger.info(f"WebSocket connection to {self.url} closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket connection: {e}")

        self.state = ConnectionState.DISCONNECTED
        self.ws = None


class ConnectionRefusedError(Exception):
    """Raised when WebSocket server refuses the connection."""

    pass


class ConnectionClosedError(Exception):
    """Raised when WebSocket server closes connection unexpectedly."""

    pass
