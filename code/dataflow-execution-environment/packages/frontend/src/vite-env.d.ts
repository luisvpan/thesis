/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  /** WebSocket de visión (por defecto `ws(s)://host/ws/vision`) */
  readonly VITE_VISION_WS_URL?: string;
  /** WebSocket de toque (por defecto `ws(s)://host/ws/touch`) */
  readonly VITE_TOUCH_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
