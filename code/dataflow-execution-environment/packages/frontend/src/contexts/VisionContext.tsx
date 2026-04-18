import {
  createContext,
  useContext,
  useEffect,
  useState,
  useMemo,
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

function getVisionWsUrl(): string {
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

  useEffect(() => {
    const url = getVisionWsUrl();
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      console.log("[ide:vision]", "WebSocket conectado", url);
    };
    ws.onclose = () => {
      setConnected(false);
    };
    ws.onerror = () => {
      setError("WebSocket visión: error de conexión");
    };
    ws.onmessage = (ev) => {
      console.log("[ide:vision]", "mensaje crudo:", ev.data);
      try {
        const data = JSON.parse(ev.data as string) as unknown;
        console.log("[ide:vision]", "mensaje parseado:", data);
        if (typeof data !== "object" || data === null) return;

        const typ = (data as { type?: string }).type;
        if (typ === "detectedNumber") {
          const det = data as DetectedNumberPayload;
          setLast(det);
          const pos = det.position;
          const posStr =
            pos != null
              ? ` pos=(${Number(pos.x).toFixed(3)}, ${Number(pos.y).toFixed(3)})`
              : "";
          console.log(
            "[ide:vision] detección:",
            det.number != null ? `#${det.number}` : "(sin dígito)",
            det.label,
            det.confidence != null
              ? `${(det.confidence * 100).toFixed(0)}%`
              : "",
            posStr.trim() || "(sin posición)",
          );
          return;
        }

        if (typ === "cardDetections") {
          const frame = data as CardDetectionsPayload;
          setLastCardFrame(frame);
          console.log(
            "[ide:vision] cartas:",
            frame.cards.length,
            frame.t,
            frame.cards.map((c) => `${c.label}@${c.position.x.toFixed(2)},${c.position.y.toFixed(2)}`),
          );
        }
      } catch (e) {
        console.warn("[ide:vision]", "JSON inválido", e);
        setError("Mensaje WS inválido");
      }
    };

    return () => {
      ws.close();
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
