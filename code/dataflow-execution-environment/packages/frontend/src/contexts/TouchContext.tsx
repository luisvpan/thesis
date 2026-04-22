import {
  createContext,
  useContext,
  useEffect,
  useState,
  useMemo,
  useRef,
  type ReactNode,
} from "react";

export type TouchEventPayload = {
  type: "touch";
  position: { x: number; y: number };
  timestamp: string;
  t: number;
};

type TouchState = {
  lastTouch: TouchEventPayload | null;
  connected: boolean;
  error: string | null;
};

const TouchContext = createContext<TouchState | null>(null);

/** URL del WebSocket de toques (CV → `/live`, navegador → `/ws/touch`). */
export function getTouchWebSocketUrl(): string {
  if (import.meta.env.VITE_TOUCH_WS_URL) {
    return import.meta.env.VITE_TOUCH_WS_URL;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/touch`;
}

function getTouchWsUrl(): string {
  return getTouchWebSocketUrl();
}

function dispatchSyntheticClick(x: number, y: number): void {
  const element = document.elementFromPoint(x, y);
  if (!element) {
    console.warn(`[touch] No element at (${x}, ${y})`);
    return;
  }
  (element as HTMLElement).click();
  console.log(`[touch] Click on <${element.tagName.toLowerCase()}> at (${x}, ${y})`);
}

type TouchIndicator = {
  id: number;
  x: number;
  y: number;
};

function TouchIndicatorOverlay({ indicators }: { indicators: TouchIndicator[] }) {
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        pointerEvents: "none",
        zIndex: 9999,
      }}
    >
      {indicators.map((indicator) => (
        <div
          key={indicator.id}
          style={{
            position: "absolute",
            left: indicator.x - 20,
            top: indicator.y - 20,
            width: 40,
            height: 40,
            borderRadius: "50%",
            backgroundColor: "rgba(0, 255, 0, 0.5)",
            border: "3px solid rgba(0, 255, 0, 0.8)",
            animation: "touchPulse 0.5s ease-out forwards",
          }}
        />
      ))}
      <style>
        {`
          @keyframes touchPulse {
            0% {
              transform: scale(0.5);
              opacity: 1;
            }
            100% {
              transform: scale(1.5);
              opacity: 0;
            }
          }
        `}
      </style>
    </div>
  );
}

export function TouchProvider({ children }: { children: ReactNode }) {
  const [lastTouch, setLastTouch] = useState<TouchEventPayload | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [indicators, setIndicators] = useState<TouchIndicator[]>([]);
  const indicatorIdRef = useRef(0);

  console.log("[touch] TouchProvider mounted");

  useEffect(() => {
    const url = getTouchWsUrl();
    console.log("[touch] useEffect running, connecting to:", url);
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      console.log("[touch] WebSocket connected", url);
    };

    ws.onclose = (ev) => {
      setConnected(false);
      console.log("[touch] WebSocket disconnected, code:", ev.code, "reason:", ev.reason);
    };

    ws.onerror = (ev) => {
      setError("Touch WebSocket error");
      console.error("[touch] WebSocket error:", ev);
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data as string) as Record<string, unknown> & {
          type?: string;
        };
        const typ = data.type;
        if (typ !== "touch") {
          return;
        }
        const t = data as unknown as TouchEventPayload;
        setLastTouch(t);

        const id = indicatorIdRef.current++;
        setIndicators((prev) => [...prev, { id, x: t.position.x, y: t.position.y }]);

        setTimeout(() => {
          setIndicators((prev) => prev.filter((i) => i.id !== id));
        }, 500);

        dispatchSyntheticClick(t.position.x, t.position.y);
      } catch (e) {
        console.warn("[touch] Invalid message", e);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  const value = useMemo(
    () => ({ lastTouch, connected, error }),
    [lastTouch, connected, error]
  );

  return (
    <TouchContext.Provider value={value}>
      {children}
      <TouchIndicatorOverlay indicators={indicators} />
    </TouchContext.Provider>
  );
}

export function useTouch(): TouchState {
  const ctx = useContext(TouchContext);
  if (!ctx) {
    throw new Error("useTouch must be used within TouchProvider");
  }
  return ctx;
}
