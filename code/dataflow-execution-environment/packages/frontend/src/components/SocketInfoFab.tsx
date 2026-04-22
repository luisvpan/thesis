import { useEffect, useState } from 'react';
import { Radio } from 'lucide-react';
import { useSocket } from '@/contexts/SocketContext';
import { getVisionWebSocketUrl } from '@/contexts/VisionContext';
import { VisionDetectedBadge } from '@/components/VisionDetectedBadge';

function getSocketIoServerUrl(): string {
  return typeof import.meta.env.VITE_SOCKET_URL === 'string' && import.meta.env.VITE_SOCKET_URL
    ? import.meta.env.VITE_SOCKET_URL
    : window.location.origin;
}

/**
 * FAB inferior derecha: estado Socket.IO + contenido de visión/WebSocket (antes junto a Ejecutar).
 */
export function SocketInfoFab() {
  const [open, setOpen] = useState(false);
  const socket = useSocket();
  const [ioConnected, setIoConnected] = useState(false);
  const [ioId, setIoId] = useState<string | null>(null);

  useEffect(() => {
    if (!socket) {
      setIoConnected(false);
      setIoId(null);
      return;
    }
    const sync = () => {
      setIoConnected(socket.connected);
      setIoId(socket.id ?? null);
    };
    socket.on('connect', sync);
    socket.on('disconnect', sync);
    sync();
    return () => {
      socket.off('connect', sync);
      socket.off('disconnect', sync);
    };
  }, [socket]);

  const visionWsUrl = getVisionWebSocketUrl();
  const socketIoUrl = getSocketIoServerUrl();

  return (
    <div className="fixed bottom-6 right-6 z-[300] flex flex-col items-end gap-2 pointer-events-auto">
      {open && (
        <div
          role="dialog"
          aria-label="Estado de sockets"
          className="mb-2 w-[min(100vw-3rem,22rem)] max-h-[min(70vh,28rem)] overflow-y-auto rounded-2xl border border-slate-600 bg-slate-900/98 shadow-2xl shadow-black/60 backdrop-blur-md"
        >
          <div className="border-b border-slate-700 px-4 py-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              Socket.IO (control / navegación)
            </h2>
            <p className="mt-1 break-all font-mono text-[11px] text-slate-500" title="Origen del cliente">
              {socketIoUrl}
            </p>
            <p className="mt-2 text-sm text-slate-200">
              Estado:{' '}
              <span className={ioConnected ? 'font-semibold text-emerald-400' : 'font-semibold text-rose-400'}>
                {ioConnected ? 'conectado' : 'desconectado'}
              </span>
            </p>
            {ioConnected && ioId != null && (
              <p className="mt-1 font-mono text-xs text-slate-400">
                id: <span className="text-slate-300">{ioId}</span>
              </p>
            )}
          </div>

          <div className="border-b border-slate-700 px-4 py-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              WebSocket visión (`/ws/vision`, cartas YOLO)
            </h2>
            <p className="mt-1 break-all font-mono text-[11px] text-slate-500">{visionWsUrl}</p>
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
        title={open ? 'Cerrar información de sockets' : 'Sockets: Socket.IO y visión'}
      >
        <Radio className="h-8 w-8" strokeWidth={2} aria-hidden />
      </button>
    </div>
  );
}
