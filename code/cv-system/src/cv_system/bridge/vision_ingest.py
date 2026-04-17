"""HTTP ingest hacia Elysia: reenvío de detecciones de cartas al frontend vía POST + WS."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from cv_system.detection.card_detector import CardDetection

logger = logging.getLogger(__name__)

_last_cards_post_monotonic: float = 0.0


def _post_json(url: str, payload: dict[str, Any], timeout: float = 2.0) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def post_card_batch_async(
    ingest_url: str,
    detections: list[CardDetection],
    proj_w: int,
    proj_h: int,
    *,
    min_interval_s: float = 0.15,
) -> None:
    """
    Envía un lote de cartas al API de visión sin bloquear el bucle principal.

    Se limita la frecuencia con min_interval_s para no saturar la red.
    """
    global _last_cards_post_monotonic
    now = time.monotonic()
    if now - _last_cards_post_monotonic < min_interval_s:
        return
    _last_cards_post_monotonic = now

    cards: list[dict[str, Any]] = []
    for d in detections:
        cx = (d.x1 + d.x2) / 2.0
        cy = (d.y1 + d.y2) / 2.0
        cards.append(
            {
                "classId": d.class_id,
                "label": d.label,
                "confidence": d.confidence,
                "position": {
                    "x": max(0.0, min(1.0, cx / float(proj_w))),
                    "y": max(0.0, min(1.0, cy / float(proj_h))),
                },
                "bbox": {
                    "x1": max(0.0, min(1.0, d.x1 / float(proj_w))),
                    "y1": max(0.0, min(1.0, d.y1 / float(proj_h))),
                    "x2": max(0.0, min(1.0, d.x2 / float(proj_w))),
                    "y2": max(0.0, min(1.0, d.y2 / float(proj_h))),
                },
            }
        )

    payload = {"cards": cards, "t": int(time.time() * 1000)}

    def _run() -> None:
        try:
            _post_json(ingest_url, payload)
        except urllib.error.URLError as e:
            logger.debug("Vision cards ingest failed: %s", e)
        except TimeoutError:
            logger.debug("Vision cards ingest timed out")
        except OSError as e:
            logger.debug("Vision cards ingest I/O error: %s", e)

    threading.Thread(target=_run, name="vision-ingest", daemon=True).start()
