import { useState } from 'react';
import { CircleHelp } from 'lucide-react';
import type { NodeErrorMark } from '@/contexts/node/errorMarks';
import { FLOW_NODE_INTERACTIVE_CLASS } from './flowNodeChrome';

/**
 * El distintivo de error de una carta: siempre visible mientras el error dure, y
 * al pulsarlo cuenta qué pasó. Nada aparece al pasar por encima, y la carta no
 * se vuelve pulsable: ya tiene sus propios gestos.
 */
export function NodeErrorBadge({ mark }: { mark?: NodeErrorMark }) {
  const [open, setOpen] = useState(false);

  // El resto del camino solo se atenúa; no tiene nada que contar.
  if (!mark || mark.role === 'flow' || !mark.text) return null;

  const strong = mark.role === 'cause';

  return (
    <div className="relative">
      <button
        type="button"
        // `nodrag` para que pulsar el icono no arrastre la carta, y
        // `pointer-events-auto` porque algunas cartas (la salida) apagan los
        // eventos de su contenedor.
        className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} absolute -right-4 -top-4 z-30 flex h-14 w-14 items-center justify-center rounded-full border-4 shadow-lg transition-transform hover:scale-110 ${
          strong
            ? 'border-red-300 bg-red-500 text-white'
            : 'border-amber-300 bg-amber-400 text-slate-900'
        }`}
        aria-label={mark.text}
        aria-expanded={open}
        onClick={(event) => {
          event.stopPropagation();
          setOpen((value) => !value);
        }}
      >
        <CircleHelp className="h-9 w-9" strokeWidth={2.5} aria-hidden />
      </button>

      {open ? (
        // El aviso sale encima de la carta, para no tapar lo que tiene debajo.
        <div
          role="status"
          className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} absolute bottom-full right-0 z-40 mb-12 w-80 rounded-xl border-2 p-4 shadow-xl ${
            strong ? 'border-red-400 bg-slate-900/95' : 'border-amber-400 bg-slate-900/95'
          }`}
        >
          <p className={`text-base font-bold leading-snug ${strong ? 'text-red-300' : 'text-amber-300'}`}>
            {mark.text}
          </p>
          {(mark.details ?? []).map((line) => (
            <p key={line} className="mt-2 text-sm leading-snug text-slate-100">
              {line}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
