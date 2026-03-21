import { useVision } from "@/contexts/VisionContext";

/** Muestra el último número detectado por YOLO (Python → Elysia → WebSocket). */
export function VisionDetectedBadge() {
  const { last, connected, error } = useVision();

  return (
    <div className="rounded-xl border border-indigo-200/80 bg-white/90 px-3 py-2 text-left shadow-md backdrop-blur-sm max-w-[min(100vw-2rem,20rem)]">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-indigo-600">
        Visión (YOLO → API → WS)
      </p>
      <p className="text-xs text-slate-500">
        WS: {connected ? "conectado" : "desconectado"}
        {error ? ` · ${error}` : null}
      </p>
      {last ? (
        <>
          <p className="mt-0.5 text-[11px] text-slate-600 truncate" title={last.label}>
            {last.label} · class {last.classId}
          </p>
          <p className="mt-1 text-2xl font-black tabular-nums text-indigo-900">
            {last.number != null ? last.number : "—"}
            {last.confidence != null && (
              <span className="ml-2 text-sm font-normal text-slate-500">
                ({(last.confidence * 100).toFixed(0)}%)
              </span>
            )}
          </p>
        </>
      ) : (
        <p className="mt-1 text-sm text-slate-500">
          Esperando detecciones… Ejecuta{" "}
          <code className="rounded bg-slate-100 px-1 text-[10px]">uv run python vision_bridge.py</code>
        </p>
      )}
    </div>
  );
}
