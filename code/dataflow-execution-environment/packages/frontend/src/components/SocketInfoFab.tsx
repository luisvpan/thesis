import { useState } from 'react';
import { Radio } from 'lucide-react';
import { getVisionWebSocketUrl } from '@/contexts/VisionContext';
import { useTouch } from '@/contexts/TouchContext';
import { VisionDetectedBadge } from '@/components/VisionDetectedBadge';

function getTouchWebSocketUrl(): string {
  if (import.meta.env.VITE_TOUCH_WS_URL) {
    return import.meta.env.VITE_TOUCH_WS_URL;
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/touch`;
}

/**
 * FAB inferior derecha: estado de WebSockets (visión + toque) en el IDE.
 */
export function SocketInfoFab() {
  const [open, setOpen] = useState(false);
  const { connected: touchConnected } = useTouch();
  const visionWsUrl = getVisionWebSocketUrl();
  const touchWsUrl = getTouchWebSocketUrl();

  return (
    <div className="fixed bottom-6 right-6 z-[300] flex flex-col items-end gap-2 pointer-events-auto">
      {open && (
        <div
          role="dialog"
          aria-label="Estado de WebSockets"
          className="mb-2 w-[min(100vw-3rem,22rem)] max-h-[min(70vh,28rem)] overflow-y-auto rounded-2xl border border-slate-600 bg-slate-900/98 shadow-2xl shadow-black/60 backdrop-blur-md"
        >
          <div className="border-b border-slate-700 px-4 py-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              WebSocket visión (YOLO → relay)
            </h2>
            <p className="mt-1 break-all font-mono text-[11px] text-slate-500">{visionWsUrl}</p>
          </div>

          <div className="border-b border-slate-700 px-4 py-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              WebSocket toque (proyector)
            </h2>
            <p className="mt-1 break-all font-mono text-[11px] text-slate-500">{touchWsUrl}</p>
            <p className="mt-2 text-sm text-slate-200">
              Estado:{' '}
              <span className={touchConnected ? 'font-semibold text-emerald-400' : 'font-semibold text-rose-400'}>
                {touchConnected ? 'conectado' : 'desconectado'}
              </span>
            </p>
          </div>

          <div className="p-3">
            <VisionDetectedBadge />
          </div>
        </div>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex h-16 w-16 items-center justify-center rounded-full border-2 border-indigo-400/80 bg-indigo-600 text-white shadow-xl shadow-black/40 transition-colors hover:bg-indigo-500 focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-400/50"
        aria-expanded={open}
        aria-haspopup="dialog"
        title={open ? 'Cerrar información de WebSockets' : 'WebSockets: visión y toque'}
      >
        <Radio className="h-8 w-8" strokeWidth={2} aria-hidden />
      </button>
    </div>
  );
}
