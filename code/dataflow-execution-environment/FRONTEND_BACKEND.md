# Frontend (Vite) + relé FastAPI (CV)

## Resumen

- **Ejecución del lenguaje dataflow**: solo en el navegador (`@dataflow/interpreter`), sin backend Node/Bun.
- **Visión y táctil**: el pipeline Python publica al **relé FastAPI** (`cv-ide-relay`); el IDE en desarrollo usa el **proxy de Vite** para `/api` y `/ws`.

## Requisitos

- [Bun](https://bun.sh) para el monorepo JS.
- Python 3.12+ con `uv` para `cv-system` (relé opcional si solo probás el grafo sin cámara).

## Instalación (monorepo JS)

```bash
cd code/dataflow-execution-environment
bun install
```

## Desarrollo

1. **Relé IDE (recomendado con visión/táctil)** — otro terminal:

   ```bash
   cd code/cv-system
   uv sync
   uv run cv-ide-relay
   ```

   Por defecto escucha en `http://127.0.0.1:8765` (`IDE_RELAY_PORT` / `IDE_RELAY_HOST`).

2. **Frontend Vite** (puerto 5173):

   ```bash
   cd code/dataflow-execution-environment
   bun run dev
   ```

El proxy en [`packages/frontend/vite.config.ts`](packages/frontend/vite.config.ts) reenvía `/api` y `/ws` a `http://127.0.0.1:8765`.

## Variables útiles

| Variable | Uso |
|----------|-----|
| `VITE_API_URL` | Base del relé si no usás mismo origen (opcional). |
| `VITE_VISION_WS_URL` | WebSocket visión explícito (por defecto `ws(s)://host/ws/vision`). |
| `VISION_CARDS_INGEST_URL` / `VISION_INGEST_URL` / `LANGUAGE_RUNTIME_WS_URL` | En Python; por defecto apuntan a `:8765`. |

## Producción del frontend

```bash
cd code/dataflow-execution-environment
bun run build
```

Sirve `packages/frontend/dist` con cualquier hosting estático; el relé FastAPI puede ir en la misma máquina o detrás de un reverse proxy. Ajustá `VITE_API_URL` / WebSockets según el despliegue.

## Paquetes relevantes

| Paquete | Rol |
|---------|-----|
| `packages/frontend` | Vite + React; intérprete en cliente; proxy a FastAPI en dev. |
| `packages/interpreter` | Motor dataflow en TypeScript. |
| `cv-system` (`cv-ide-relay`) | FastAPI: `/api/v1/vision/*`, `/ws/vision`, `/ws/touch`, `/live`. |
