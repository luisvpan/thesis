# Integración React (Vite) + Elysia (`@dataflow/http-api`)

## Requisitos

- [Bun](https://bun.sh) instalado.
- Una sola instalación de dependencias en la raíz de este monorepo (`node_modules` hoisted).

## Instalación

```bash
cd code/dataflow-execution-environment
bun install
```

## Desarrollo (API Elysia + Vite con proxy)

Levanta **API en el puerto 3000** y **Vite en 5173**; el proxy reenvía `/api/*` al API.

```bash
bun run dev
```

## Eden (tipado front ↔ back)

- Archivo: `packages/frontend/src/lib/eden.ts`.
- **`treaty<App>(baseUrl)`** de `@elysiajs/eden`, donde **`App`** es `typeof app` del servidor (`@dataflow/http-api/server`).
- Exporta **`edenClient`** (y alias `apiClient`): las rutas del cliente coinciden con Elysia (`edenClient.api.v1.health.get()`, etc.).
- Tipos inferidos de ejemplo: **`HealthEdenResult`**, **`HealthData`** para `GET /api/v1/health`.
- En el navegador, la base URL es por defecto el **mismo origen** que Vite; las peticiones van a `/api/v1/...` y el proxy las envía a Elysia (puerto 3000).

Documentación: [Eden (Elysia)](https://elysiajs.com/eden/overview).

## Producción (un solo proceso)

1. Build del frontend y del bundle del servidor:

   ```bash
   bun run build
   ```

2. Arranque: Elysia sirve `/api/v1/*` y los estáticos de `packages/frontend/dist`.

   ```bash
   bun run start
   ```

Abre `http://localhost:3000` (o el `PORT` que definas).

## Paquetes implicados

| Paquete | Rol |
|---------|-----|
| `packages/frontend` | Vite + React; depende de `@dataflow/http-api` solo para **tipos** + Eden. |
| `packages/http-api` | Elysia; `src/main.ts` escucha y en prod sirve el build de Vite. |
