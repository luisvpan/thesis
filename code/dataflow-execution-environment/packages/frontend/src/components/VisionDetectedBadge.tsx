import { useVision } from "@/contexts/VisionContext";

/** Muestra el último número detectado por YOLO (Python → Elysia → WebSocket). */
export function VisionDetectedBadge() {
  const { last, lastCardFrame, connected, error } = useVision();

  return (
    <div className="rounded-xl border border-indigo-200/80 bg-white/90 px-3 py-2 text-left shadow-md backdrop-blur-sm max-w-[min(100vw-2rem,20rem)]">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-indigo-600">
        Visión (YOLO → API → WS)
      </p>
      <p className="text-xs text-slate-500">
        WS: {connected ? "conectado" : "desconectado"}
        {error ? ` · ${error}` : null}
      </p>
      {lastCardFrame != null && lastCardFrame.cards.length > 0 && (
        <div className="mt-1 space-y-0.5">
          <p className="text-[11px] text-slate-600">
            Cartas en mesa:{" "}
            <span className="font-semibold text-slate-800">{lastCardFrame.cards.length}</span>
          </p>
          <ul className="max-h-24 overflow-y-auto font-mono text-[10px] text-slate-600">
            {lastCardFrame.cards.map((c, i) => (
              <li key={`${c.trackId ?? "x"}-${c.label}-${i}`} className="truncate" title={c.label}>
                {c.trackId != null && c.trackId >= 0 ? (
                  <span className="font-semibold text-indigo-700">#{c.trackId}</span>
                ) : (
                  <span className="text-amber-700">ID ?</span>
                )}{" "}
                · {c.label}
                {c.status === "lost" ? (
                  <span className="text-amber-600"> (perdida)</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      )}
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
          {last.position != null && (
            <p
              className="mt-0.5 font-mono text-[11px] text-slate-600"
              title="Centro del bbox normalizado al frame (0–1), enviado al API"
            >
              pos ({last.position.x.toFixed(3)}, {last.position.y.toFixed(3)})
            </p>
          )}
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
