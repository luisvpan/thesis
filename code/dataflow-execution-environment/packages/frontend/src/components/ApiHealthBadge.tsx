import { useCallback, useEffect, useState } from "react";
import { ChevronUp, RefreshCw } from "lucide-react";
import { fetchHealth, type HealthData } from "@/lib/apiHealth";

type LoadState =
  | { kind: "loading" }
  | { kind: "ok"; data: HealthData; fetchedAt: string }
  | { kind: "error"; message: string };

/**
 * Badge "API OK" con prueba de integración: respuesta de `GET /api/v1/health` (relé FastAPI).
 */
export function ApiHealthBadge() {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      const data = await fetchHealth();
      console.log("[backend:relay]", "GET /api/v1/health", data);
      setState({
        kind: "ok",
        data: {
          status: data.status,
          version: data.version,
          uptime: data.uptime,
        },
        fetchedAt: new Date().toISOString(),
      });
    } catch (e) {
      setState({
        kind: "error",
        message: e instanceof Error ? e.message : "Error de red",
      });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const label =
    state.kind === "loading"
      ? "API…"
      : state.kind === "ok"
        ? "API OK"
        : "API off";

  const dotClass =
    state.kind === "loading"
      ? "bg-amber-400 animate-pulse"
      : state.kind === "ok"
        ? "bg-emerald-400"
        : "bg-red-400";

  return (
    <div className="relative z-20">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-full bg-slate-900/85 px-3 py-1.5 text-xs font-medium text-white shadow-md backdrop-blur-sm border border-white/10 hover:bg-slate-900 transition-colors"
        aria-expanded={open}
        aria-controls="api-integration-panel"
        title="Pulsa para ver la prueba de integración con el relé CV"
      >
        <span className={`h-2 w-2 rounded-full shrink-0 ${dotClass}`} aria-hidden />
        <span>{label}</span>
        <ChevronUp
          className={`h-3.5 w-3.5 opacity-70 transition-transform ${open ? "" : "rotate-180"}`}
          aria-hidden
        />
      </button>

      {open && (
        <div
          id="api-integration-panel"
          role="region"
          aria-label="Respuesta del API /api/v1/health"
          className="absolute bottom-full right-0 mb-2 w-[min(100vw-2rem,22rem)] rounded-xl border border-slate-200 bg-white text-slate-900 shadow-xl overflow-hidden"
        >
          <div className="flex items-center justify-between gap-2 border-b border-slate-100 bg-slate-50 px-3 py-2">
            <p className="text-xs font-semibold text-slate-800">
              Integración <code className="font-mono text-[10px]">/api/v1/health</code>
            </p>
            <button
              type="button"
              onClick={() => void load()}
              disabled={state.kind === "loading"}
              className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-[10px] font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              <RefreshCw
                className={`h-3 w-3 ${state.kind === "loading" ? "animate-spin" : ""}`}
              />
              Reintentar
            </button>
          </div>

          <div className="max-h-[min(50vh,20rem)] overflow-auto p-3 text-left text-xs">
            {state.kind === "loading" && (
              <p className="text-slate-600">Cargando respuesta del relé…</p>
            )}
            {state.kind === "error" && (
              <div className="rounded-lg bg-red-50 px-2 py-1.5 text-red-800">
                <strong className="font-semibold">Error:</strong> {state.message}
                <p className="mt-1 text-[10px] text-red-600">
                  ¿Relé FastAPI en :8765 y proxy Vite <code>/api</code>?
                </p>
              </div>
            )}
            {state.kind === "ok" && (
              <div className="space-y-2">
                <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5">
                  <dt className="text-slate-500">status</dt>
                  <dd className="font-mono text-slate-900">{state.data.status}</dd>
                  <dt className="text-slate-500">version</dt>
                  <dd className="font-mono text-slate-900">{state.data.version}</dd>
                  <dt className="text-slate-500">uptime</dt>
                  <dd className="font-mono text-slate-900">{state.data.uptime}s</dd>
                </dl>
                <pre className="rounded-lg bg-slate-900 p-2 text-[10px] leading-relaxed text-emerald-300 overflow-x-auto">
                  {JSON.stringify(
                    { ...state.data, _fetchedAt: state.fetchedAt },
                    null,
                    2,
                  )}
                </pre>
                <p className="text-[10px] text-slate-400">
                  {new Date(state.fetchedAt).toLocaleString()}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
