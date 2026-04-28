/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_SOCKET_URL?: string;
  /** Override WebSocket visión (`/ws/vision` por defecto, proxy Vite `/ws`) */
  readonly VITE_VISION_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/** Depuración en dev: última ejecución local (`executeProgram`). */
interface Window {
  __DATAFLOW_LAST_DSL__?: string;
  __DATAFLOW_LAST_PROGRAM__?: unknown;
}
