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
 * Botón de la cabecera del IDE: estado de WebSockets (visión + toque).
 * El punto indica si el WebSocket de toque está conectado.
 */
export function SocketStatusButton({ className = '' }: { className?: string }) {
  const [open, setOpen] = useState(false);
  const { connected: touchConnected } = useTouch();
  const visionWsUrl = getVisionWebSocketUrl();
  const touchWsUrl = getTouchWebSocketUrl();

  return (
    <div className="relative">
      {open && (
        <div
          role="dialog"
          aria-label="Estado de WebSockets"
          className="absolute right-0 top-full z-[300] mt-4 w-[min(100vw-3rem,22rem)] max-h-[min(70vh,28rem)] overflow-y-auto rounded-2xl border border-slate-600 bg-slate-900/98 shadow-2xl shadow-black/60 backdrop-blur-md"
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
        className={`relative flex items-center justify-center border-indigo-400/50 bg-indigo-950 text-indigo-200 transition-colors hover:bg-indigo-900 hover:text-white ${className}`}
        aria-expanded={open}
        aria-haspopup="dialog"
        title={open ? 'Cerrar información de WebSockets' : 'WebSockets: visión y toque'}
      >
        <Radio className="h-12 w-12 pointer-events-none" strokeWidth={2} aria-hidden />
        <span
          aria-hidden
          className={`absolute right-3 top-3 h-4 w-4 rounded-full ring-2 ring-slate-900 ${
            touchConnected ? 'bg-emerald-400' : 'bg-rose-500'
          }`}
        />
      </button>
    </div>
  );
}
