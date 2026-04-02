/**
 * Eden Treaty: cliente HTTP tipado end-to-end con el servidor Elysia.
 * El tipo `App` proviene de `@dataflow/http-api`; las rutas del cliente coinciden con el backend.
 *
 * @see https://elysiajs.com/eden/overview
 */
import { treaty } from "@elysiajs/eden";
import type { App } from "@dataflow/http-api/server";

/** Reexport: mismo `App` que el backend (`export type App = typeof app`). */
export type { App };

function getApiBaseUrl(): string {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  if (typeof window !== "undefined") {
    return window.location.origin;
  }
  return "http://127.0.0.1:3000";
}

/**
 * Cliente Eden (Treaty) ligado a `App`.
 * Ejemplo: `await edenClient.api.v1.health.get()` — tipado según `server.ts`.
 */
export const edenClient = treaty<App>(getApiBaseUrl());

/** @deprecated Usa `edenClient` */
export const apiClient = edenClient;

/** Tipo del cliente Treaty (útil para genéricos y tests). */
export type EdenClient = typeof edenClient;

/** Resultado de una llamada Eden a `GET /api/v1/health` (`{ data, error, ... }`). */
export type HealthEdenResult = Awaited<
  ReturnType<typeof edenClient.api.v1.health.get>
>;

/** Payload JSON de `GET /api/v1/health` cuando `data` está definido. */
export type HealthData = NonNullable<HealthEdenResult["data"]>;
