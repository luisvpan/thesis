# Visión: YOLO (`plswork2.pt`) → Elysia → Frontend

## Flujo

1. **Python** (`vision_bridge.py`): cámara + modelo `models/plswork2.pt` + `config/dataflow_augmented.yaml` (índice de clase → nombre). Se envía `classId`, `label`, `confidence` y **`number` solo si la clase es un dígito** (`one`→1 … `nine`→9), no el índice de clase.
2. **Elysia** (`packages/http-api/src/vision.ts`): recibe el JSON y reenvía por **WebSocket** `ws://…/ws/vision` a todos los navegadores conectados (`type: "detectedNumber"`).
3. **React** (`VisionProvider` + `VisionDetectedBadge`): se conecta a `/ws/vision` (en dev, Vite hace proxy de `/ws` al puerto 3000).

4. **Canvas React Flow** (`DataflowPage`): cada mensaje nuevo (`last.t`) añade un **nodo número** (`type: 'number'`, `data.value` = clase YOLO) en una rejilla bajo el canvas (ids `vision-<timestamp>`).

## Arranque

Terminal 1 — API + frontend (monorepo Bun):

```bash
cd code/dataflow-execution-environment
bun run dev
```

Terminal 2 — puente Python (desde `code/`):

```bash
cd code
uv sync
uv run python vision_bridge.py
```

Variables útiles:

| Variable | Default |
|----------|---------|
| `VISION_INGEST_URL` | `http://127.0.0.1:3000/api/v1/vision/ingest` |
| `VISION_CAMERA_ID` | `0` |
| `VISION_CONF` | `0.45` |
| `VISION_SEND_MS` | `250` |

El valor mostrado como “número” es el **class id** de YOLO (índice de clase del modelo). Si tu modelo usa clases 0–9 para dígitos, coincidirá con el dígito.
