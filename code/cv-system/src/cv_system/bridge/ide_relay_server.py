"""
HTTP + WebSocket relay for the React IDE (vision ingest, card batches, touch relay).

Replaces the former Bun/Elysia endpoints used by the CV pipeline and browser.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

START_TIME = time.time()

DEFAULT_RGB: dict[str, int] = {"width": 1920, "height": 1080}

vision_sockets: set[WebSocket] = set()
touch_sockets: set[WebSocket] = set()


def resolve_session_json_path() -> str:
    env = os.environ.get("CV_SESSION_JSON_PATH") or os.environ.get("SESSION_JSON_PATH")
    if env:
        return env
    root = Path(__file__).resolve().parents[3]
    return str(root / "config" / "session.json")


def read_rgb_resolution_from_session() -> tuple[dict[str, int], bool]:
    path = resolve_session_json_path()
    p = Path(path)
    if not p.is_file():
        return {**DEFAULT_RGB}, False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        cam = raw.get("camera") or {}
        rgb = cam.get("rgb_resolution")
        if not isinstance(rgb, list) or len(rgb) < 2:
            return {**DEFAULT_RGB}, False
        h, w = int(rgb[0]), int(rgb[1])
        if h <= 0 or w <= 0:
            return {**DEFAULT_RGB}, False
        return {"width": w, "height": h}, True
    except (OSError, ValueError, TypeError, KeyError):
        return {**DEFAULT_RGB}, False


async def broadcast_to_vision(msg: str) -> None:
    dead: list[WebSocket] = []
    for ws in vision_sockets:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        vision_sockets.discard(ws)


async def broadcast_to_touch(msg: str) -> None:
    dead: list[WebSocket] = []
    for ws in touch_sockets:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        touch_sockets.discard(ws)


class IngestBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    classId: int
    label: str
    number: int | float | None = None
    confidence: float | None = None
    position: dict[str, float] | None = None


class VisionCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    classId: int
    label: str
    confidence: float
    trackId: int | None = None
    position: dict[str, float]
    bbox: dict[str, float] | None = None


class CardsBody(BaseModel):
    cards: list[VisionCard]
    t: int | None = None


app = FastAPI(title="CV IDE relay", version="1.0.0")


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    uptime = int(time.time() - START_TIME)
    return {"status": "healthy", "version": "1.0.0", "uptime": uptime}


@app.get("/api/v1/vision/projector-resolution")
def projector_resolution() -> dict[str, Any]:
    resolution, ok = read_rgb_resolution_from_session()
    return {
        "rgbResolution": resolution,
        "sessionPath": resolve_session_json_path(),
        "fromSessionFile": ok,
    }


@app.post("/api/v1/vision/ingest")
async def vision_ingest(body: IngestBody) -> dict[str, Any]:
    payload = {
        "type": "detectedNumber",
        "classId": body.classId,
        "label": body.label,
        "number": body.number,
        "confidence": body.confidence,
        "position": body.position,
        "t": int(time.time() * 1000),
    }
    msg = json.dumps(payload)
    await broadcast_to_vision(msg)
    await broadcast_to_touch(msg)
    return {"ok": True, "forwarded": len(vision_sockets)}


@app.post("/api/v1/vision/cards")
async def vision_cards(body: CardsBody) -> dict[str, Any]:
    ts = body.t if body.t is not None else int(time.time() * 1000)
    payload = {
        "type": "cardDetections",
        "cards": [c.model_dump(exclude_none=True) for c in body.cards],
        "t": ts,
    }
    msg = json.dumps(payload)
    await broadcast_to_vision(msg)
    return {"ok": True, "forwarded": len(vision_sockets), "count": len(body.cards)}


@app.websocket("/ws/vision")
async def ws_vision(websocket: WebSocket) -> None:
    await websocket.accept()
    vision_sockets.add(websocket)
    logger.info("[vision] browser ws connected, clients: %s", len(vision_sockets))
    try:
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    finally:
        vision_sockets.discard(websocket)
        logger.info("[vision] browser ws closed, clients: %s", len(vision_sockets))


@app.websocket("/ws/touch")
async def ws_touch(websocket: WebSocket) -> None:
    await websocket.accept()
    touch_sockets.add(websocket)
    logger.info("[touch] browser connected: %s", len(touch_sockets))
    try:
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    finally:
        touch_sockets.discard(websocket)
        logger.info("[touch] browser disconnected: %s", len(touch_sockets))


@app.websocket("/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("[touch] CV system connected to /live")
    try:
        while True:
            raw = await websocket.receive_text()
            logger.info(f"[touch] received raw: {raw[:100]}...")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[touch] JSON decode error")
                continue
            event_type = data.get("type")
            logger.info(f"[touch] event_type={event_type}")
            if event_type not in ("touch", "touch_down", "touch_move", "touch_up"):
                logger.info(f"[touch] filtered out type={event_type}")
                continue
            pos = data.get("position") or {}
            if not isinstance(pos, dict):
                continue
            payload = {
                "type": event_type,  # Preserve original event type
                "touch_id": data.get("touch_id", 0),
                "position": {
                    "x": float(pos.get("x", 0)),
                    "y": float(pos.get("y", 0)),
                },
                "timestamp": str(data.get("timestamp", "")),
                "t": int(time.time() * 1000),
            }
            await broadcast_to_touch(json.dumps(payload))
            logger.info(
                "[touch] relay %s id=%s (%s, %s)",
                event_type,
                payload["touch_id"],
                payload["position"]["x"],
                payload["position"]["y"],
            )
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("[touch] CV system disconnected from /live")


def run() -> None:
    """CLI entry: ``uv run cv-ide-relay`` or ``python -m cv_system.bridge.ide_relay_server``."""
    import uvicorn

    host = os.environ.get("IDE_RELAY_HOST", "127.0.0.1")
    port = int(os.environ.get("IDE_RELAY_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
