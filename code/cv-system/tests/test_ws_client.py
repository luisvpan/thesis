"""Unit tests for WebSocket Bridge client."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cv_system.bridge.ws_client import (
    WebSocketBridge,
    ConnectionState,
    ConnectionRefusedError,
    ConnectionClosedError,
)


@pytest.fixture
def mock_websocket():
    """Create a mock for websockets.connect."""
    with patch("cv_system.bridge.ws_client.websockets") as mock:
        mock.connect = AsyncMock()
        yield mock


@pytest.fixture
def event_loop():
    """Create and close an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestWebSocketBridgeInit:
    """Tests for WebSocketBridge initialization."""

    def test_init_default_url_from_env(self, monkeypatch):
        """Test default URL from LANGUAGE_RUNTIME_WS_URL env var."""
        monkeypatch.setenv("LANGUAGE_RUNTIME_WS_URL", "ws://custom:4000/live")

        bridge = WebSocketBridge()

        assert bridge.url == "ws://custom:4000/live"
        assert bridge.state == ConnectionState.DISCONNECTED
        assert bridge.reconnect_enabled is True

    def test_init_explicit_url(self):
        """Test explicit URL parameter."""
        bridge = WebSocketBridge(url="ws://explicit:5000/live")

        assert bridge.url == "ws://explicit:5000/live"
        assert bridge.state == ConnectionState.DISCONNECTED

    def test_init_invalid_retry_params(self):
        """Test invalid retry parameters raise ValueError."""
        with pytest.raises(ValueError, match="max_reconnect_delay.*must be greater than"):
            WebSocketBridge(max_reconnect_delay=0.5, base_reconnect_delay=1.0)


class TestWebSocketBridgeConnect:
    """Tests for WebSocketBridge.connect() method."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_websocket, event_loop):
        """Test successful connection."""
        mock_ws = AsyncMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock()
        mock_websocket.connect = AsyncMock(return_value=mock_ws)

        bridge = WebSocketBridge(url="ws://localhost:3000/live")

        # Mock _listen_for_messages to avoid hanging
        with patch.object(bridge, "_listen_for_messages") as mock_listen:
            mock_listen = AsyncMock()
            mock_listen.return_value = None

            await bridge.connect()

            assert bridge.state == ConnectionState.CONNECTED
            assert bridge._reconnect_attempts == 0
            mock_websocket.connect.assert_called_once_with("ws://localhost:3000/live")

    @pytest.mark.asyncio
    async def test_connect_connection_refused(self, mock_websocket, event_loop):
        """Test connection refused handling."""
        # Simulate OSError with errno 111 (connection refused)
        mock_websocket.connect = AsyncMock(
            side_effect=OSError(111, "Connection refused")
        )

        bridge = WebSocketBridge(url="ws://localhost:3000/live")

        with pytest.raises(ConnectionRefusedError):
            await bridge.connect()

        assert bridge.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_temporary_network_error(self, mock_websocket, event_loop):
        """Test exponential backoff retry on temporary network errors."""
        # Simulate first connection failure, second success
        call_count = [0]

        async def failing_then_success(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError(111, "Temporary network error")
            # Create mock WebSocket for successful connection
            mock_ws = AsyncMock()
            mock_ws.send = AsyncMock()
            return mock_ws

        mock_websocket.connect = AsyncMock(side_effect=failing_then_success)

        # Mock _listen_for_messages
        with patch.object(WebSocketBridge, "_listen_for_messages") as mock_listen:
            mock_listen = AsyncMock(return_value=None)

            bridge = WebSocketBridge(
                base_reconnect_delay=0.1, max_reconnect_delay=10.0
            )

            # Give it time to complete retries
            await asyncio.sleep(0.5)

            assert call_count[0] == 2  # Failed once, succeeded on retry
            assert bridge._reconnect_attempts == 1
            assert bridge.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_websocket, event_loop):
        """Test duplicate connect() calls are ignored."""
        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.state = ConnectionState.CONNECTED
        bridge.ws = AsyncMock()

        await bridge.connect()

        mock_websocket.connect.assert_not_called()
        assert bridge.state == ConnectionState.CONNECTED


class TestSendTouchEvent:
    """Tests for send_touch_event() method."""

    @pytest.mark.asyncio
    async def test_send_touch_event_success(self, mock_websocket, event_loop):
        """Test sending valid touch event."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.ws = mock_ws
        bridge.state = ConnectionState.CONNECTED

        touch_event = {
            "position": {"x": 100.5, "y": 200.75},
            "timestamp": "2026-03-25T00:00:00Z",
        }

        await bridge.send_touch_event(touch_event)

        mock_ws.send.assert_called_once()
        sent_message = mock_ws.send.call_args[0][0][0]
        import json
        assert json.loads(sent_message) == touch_event

    @pytest.mark.asyncio
    async def test_send_touch_event_not_connected(self, event_loop):
        """Test sending when not connected raises RuntimeError."""
        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.ws = None

        touch_event = {
            "position": {"x": 100.0, "y": 200.0},
            "timestamp": "2026-03-25T00:00:00Z",
        }

        with pytest.raises(RuntimeError, match="WebSocket is not connected"):
            await bridge.send_touch_event(touch_event)

    @pytest.mark.asyncio
    async def test_send_touch_event_missing_position(self, mock_websocket, event_loop):
        """Test touch event without position field raises ValueError."""
        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.ws = AsyncMock()
        bridge.state = ConnectionState.CONNECTED

        touch_event_invalid = {
            "timestamp": "2026-03-25T00:00:00Z",
        }

        with pytest.raises(ValueError, match="must have 'position' field"):
            await bridge.send_touch_event(touch_event_invalid)

    @pytest.mark.asyncio
    async def test_send_touch_event_missing_x(self, mock_websocket, event_loop):
        """Test touch event without x coordinate raises ValueError."""
        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.ws = AsyncMock()
        bridge.state = ConnectionState.CONNECTED

        touch_event_invalid = {
            "position": {"y": 200.0},
            "timestamp": "2026-03-25T00:00:00Z",
        }

        with pytest.raises(ValueError, match="Position must have 'x' coordinate"):
            await bridge.send_touch_event(touch_event_invalid)

    @pytest.mark.asyncio
    async def test_send_touch_event_missing_y(self, mock_websocket, event_loop):
        """Test touch event without y coordinate raises ValueError."""
        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.ws = AsyncMock()
        bridge.state = ConnectionState.CONNECTED

        touch_event_invalid = {
            "position": {"x": 100.0},
            "timestamp": "2026-03-25T00:00:00Z",
        }

        with pytest.raises(ValueError, match="Position must have 'y' coordinate"):
            await bridge.send_touch_event(touch_event_invalid)


class TestDisconnect:
    """Tests for disconnect() method."""

    @pytest.mark.asyncio
    async def test_disconnect_connected(self, mock_websocket, event_loop):
        """Test disconnecting when connected."""
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()

        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.ws = mock_ws
        bridge.state = ConnectionState.CONNECTED

        await bridge.disconnect()

        mock_ws.close.assert_called_once()
        assert bridge.ws is None
        assert bridge.state == ConnectionState.DISCONNECTED
        assert bridge.reconnect_enabled is False

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, event_loop):
        """Test disconnecting when not connected is safe."""
        bridge = WebSocketBridge(url="ws://localhost:3000/live")
        bridge.ws = None

        await bridge.disconnect()

        assert bridge.ws is None
        assert bridge.state == ConnectionState.DISCONNECTED
        assert bridge.reconnect_enabled is False


class TestBackoffCalculation:
    """Tests for exponential backoff calculation."""

    def test_backoff_base_case(self):
        """Test backoff calculation with no previous attempts."""
        bridge = WebSocketBridge(
            base_reconnect_delay=1.0,
            backoff_factor=2.0,
            max_reconnect_delay=60.0,
        )

        # Attempt 0: base delay
        assert bridge._calculate_backoff(0) == 1.0

    def test_backoff_exponential(self):
        """Test exponential backoff growth."""
        bridge = WebSocketBridge(
            base_reconnect_delay=1.0,
            backoff_factor=2.0,
            max_reconnect_delay=60.0,
        )

        # Attempt 0: 1.0s
        assert bridge._calculate_backoff(0) == 1.0
        # Attempt 1: 2.0s
        assert bridge._calculate_backoff(1) == 2.0
        # Attempt 2: 4.0s
        assert bridge._calculate_backoff(2) == 4.0
        # Attempt 3: 8.0s
        assert bridge._calculate_backoff(3) == 8.0

    def test_backoff_capped_at_max(self):
        """Test backoff is capped at max_reconnect_delay."""
        bridge = WebSocketBridge(
            base_reconnect_delay=1.0,
            backoff_factor=2.0,
            max_reconnect_delay=10.0,
        )

        # Attempt 3 would be 8.0s, but should cap at 10.0s
        assert bridge._calculate_backoff(3) == 10.0


class TestConnectionStateEnum:
    """Tests for ConnectionState enum."""

    def test_connection_state_values(self):
        """Test all state values are defined."""
        assert ConnectionState.DISCONNECTED == "DISCONNECTED"
        assert ConnectionState.CONNECTING == "CONNECTING"
        assert ConnectionState.CONNECTED == "CONNECTED"
