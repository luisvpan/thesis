import {
  createContext,
  useContext,
  useEffect,
  useState,
  useMemo,
  useRef,
  type ReactNode,
} from "react";

/** Payload emitido por Elysia cuando Python envía `POST /api/v1/vision/ingest`. */
export type DetectedNumberPayload = {
  type: "detectedNumber";
  classId: number;
  label: string;
  number?: number;
  confidence?: number;
  position?: { x: number; y: number };
  t: number;
};

/** Una carta en vista de proyector (`POST /api/v1/vision/cards` → WS `cardDetections`). */
export type VisionCardItem = {
  classId: number;
  label: string;
  confidence: number;
  trackId?: number;
  position: { x: number; y: number };
  bbox?: { x1: number; y1: number; x2: number; y2: number };
};

export type CardDetectionsPayload = {
  type: "cardDetections";
  cards: VisionCardItem[];
  t: number;
};

type VisionState = {
  last: DetectedNumberPayload | null;
  lastCardFrame: CardDetectionsPayload | null;
  connected: boolean;
  error: string | null;
};

const VisionContext = createContext<VisionState | null>(null);

const WS_INITIAL_BACKOFF_MS = 800;
const WS_MAX_BACKOFF_MS = 15000;

/** WebSocket dedicado (`/ws/vision`): lotes YOLO desde el API tras `POST /api/v1/vision/cards`. */
export function getVisionWebSocketUrl(): string {
  if (import.meta.env.VITE_VISION_WS_URL) {
    return import.meta.env.VITE_VISION_WS_URL;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/vision`;
}

export function VisionProvider({ children }: { children: ReactNode }) {
  const [last, setLast] = useState<DetectedNumberPayload | null>(null);
  const [lastCardFrame, setLastCardFrame] = useState<CardDetectionsPayload | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let attempt = 0;

    const clearReconnect = () => {
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const attachHandlers = (ws: WebSocket, url: string) => {
      ws.onopen = () => {
        attempt = 0;
        setConnected(true);
        setError(null);
        console.log("[ide:vision] WebSocket conectado", url);
      };
      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        if (cancelled) return;
        const delay = Math.min(
          WS_MAX_BACKOFF_MS,
          WS_INITIAL_BACKOFF_MS * Math.pow(2, attempt),
        );
        attempt += 1;
        clearReconnect();
        reconnectTimerRef.current = setTimeout(() => connect(), delay);
      };
      ws.onerror = () => {
        setError("WebSocket visión: error de conexión");
      };

      ws.onmessage = (ev) => {
        console.log("[ide:vision] mensaje crudo:", ev.data);
        try {
          const data = JSON.parse(ev.data as string) as unknown;
          if (typeof data !== "object" || data === null) return;

          const typ = (data as { type?: string }).type;

          if (typ === "detectedNumber") {
            const det = data as DetectedNumberPayload;
            setLast(det);
            setError(null);
            const pos = det.position;
            const posStr =
              pos != null
                ? ` pos=(${Number(pos.x).toFixed(3)}, ${Number(pos.y).toFixed(3)})`
                : "";
            console.log(
              "[ide:vision] detección:",
              det.number != null ? `#${det.number}` : "(sin dígito)",
              det.label,
              det.confidence != null ? `${(det.confidence * 100).toFixed(0)}%` : "",
              posStr.trim() || "(sin posición)",
            );
            return;
          }

          if (typ === "cardDetections") {
            const frame = data as CardDetectionsPayload;
            setLastCardFrame(frame);
            setError(null);
            console.log(
              "[ide:vision] cartas:",
              frame.cards.length,
              frame.t,
              frame.cards.map(
                (c) =>
                  `${c.label}[${c.trackId ?? "?"}]@${c.position.x.toFixed(2)},${c.position.y.toFixed(2)}`,
              ),
            );
          }
        } catch (e) {
          console.warn("[ide:vision] JSON inválido", e);
          setError("Mensaje WS inválido");
        }
      };
    };

    function connect() {
      if (cancelled) return;
      clearReconnect();
      const url = getVisionWebSocketUrl();
      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;
        attachHandlers(ws, url);
      } catch (e) {
        console.warn("[ide:vision] fallo al crear WebSocket:", e);
        setError("No se pudo abrir WebSocket de visión");
        const delay = Math.min(
          WS_MAX_BACKOFF_MS,
          WS_INITIAL_BACKOFF_MS * Math.pow(2, attempt),
        );
        attempt += 1;
        reconnectTimerRef.current = setTimeout(() => connect(), delay);
      }
    }

    connect();

    return () => {
      cancelled = true;
      clearReconnect();
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  const value = useMemo(
    (): VisionState => ({ last, lastCardFrame, connected, error }),
    [last, lastCardFrame, connected, error],
  );

  return (
    <VisionContext.Provider value={value}>{children}</VisionContext.Provider>
  );
}

export function useVision(): VisionState {
  const ctx = useContext(VisionContext);
  if (!ctx) {
    throw new Error("useVision debe usarse dentro de VisionProvider");
  }
  return ctx;
}
