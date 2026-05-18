import {
  createContext,
  useContext,
  useEffect,
  useState,
  useMemo,
  type ReactNode,
} from "react";
import { logger } from "@/lib/logger";

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
  /** "active" = detectada este frame; "lost" = ByteTrack la retiene aunque no esté visible */
  status?: "active" | "lost";
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

  useEffect(() => {
    const url = getVisionWebSocketUrl();
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      logger.vision.info("WebSocket connected", { url });
    };
    ws.onclose = () => {
      setConnected(false);
    };
    ws.onerror = () => {
      setError("WebSocket visión: error de conexión");
    };
    ws.onmessage = (ev) => {
      logger.vision.debug("Raw message", { data: ev.data });
      try {
        const data = JSON.parse(ev.data as string) as unknown;
        logger.vision.debug("Parsed message", { data });
        if (typeof data !== "object" || data === null) return;

        const typ = (data as { type?: string }).type;
        if (typ === "detectedNumber") {
          const det = data as DetectedNumberPayload;
          setLast(det);
          logger.vision.debug("Detection", {
            number: det.number,
            label: det.label,
            confidence: det.confidence,
            position: det.position,
          });
          return;
        }

        if (typ === "cardDetections") {
          const frame = data as CardDetectionsPayload;
          setLastCardFrame(frame);
          logger.vision.debug("Card detections", {
            count: frame.cards.length,
            t: frame.t,
            cards: frame.cards.map((c) => ({
              label: c.label,
              trackId: c.trackId,
              x: c.position.x,
              y: c.position.y,
            })),
          });
        }
      } catch (e) {
        logger.vision.warn("Invalid JSON", {
          error: e instanceof Error ? e.message : String(e),
        });
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
