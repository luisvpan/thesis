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
  /** 1..9 solo para clases one..nine en data.yaml; operadores/formas no lo llevan */
  number?: number;
  confidence?: number;
  /** Centro YOLO normalizado al frame (0..1), para mapear al canvas de React Flow */
  position?: { x: number; y: number };
  t: number;
};

/** Una carta en vista de proyector (POST `/api/v1/vision/cards` → WS `cardDetections`). */
export type VisionCardItem = {
  classId: number;
  label: string;
  confidence: number;
  trackId?: number;  // Persistent tracking ID from YOLO tracker
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
  /** Último lote de cartas (tablero físico); posiciones normalizadas 0-1 */
  lastCardFrame: CardDetectionsPayload | null;
  connected: boolean;
  error: string | null;
};

const VisionContext = createContext<VisionState | null>(null);

/** URL efectiva del WebSocket de visión (misma lógica que la conexión real). */
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
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const WS_INITIAL_BACKOFF_MS = 300;
    const WS_MAX_BACKOFF_MS = 5000;
    let attempt = 0;
    let cancelled = false;

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
          WS_INITIAL_BACKOFF_MS * Math.pow(2, attempt)
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
              posStr.trim() || "(sin posición)"
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
              frame.cards.map((c) => {
                const x = c.position?.x;
                const y = c.position?.y;
                const xy =
                  typeof x === "number" &&
                  typeof y === "number" &&
                  Number.isFinite(x) &&
                  Number.isFinite(y)
                    ? `${x.toFixed(2)},${y.toFixed(2)}`
                    : "?";
                return `${c.label}[${c.trackId ?? "?"}]@${xy}`;
              })
            );
          }
        } catch (e) {
          console.warn("[ide:vision] JSON inválido", e);
          setError("Mensaje WS inválido");
        }
      };
    };

    const connect = () => {
      if (cancelled) return;
      const url = getVisionWebSocketUrl();
      const ws = new WebSocket(url);
      wsRef.current = ws;
      attachHandlers(ws, url);
    };

    connect();

    return () => {
      cancelled = true;
      clearReconnect();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
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