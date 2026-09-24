import { useState } from 'react';
import { CircleHelp } from 'lucide-react';
import type { NodeErrorMark } from '@/contexts/node/errorMarks';

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
        // `nodrag` para que pulsar el icono no arrastre la carta.
        className={`nodrag absolute -right-2 -top-2 z-10 flex h-7 w-7 items-center justify-center rounded-full border-2 shadow-lg transition-transform hover:scale-110 ${
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
        <CircleHelp className="h-4 w-4" strokeWidth={2.5} aria-hidden />
      </button>

      {open ? (
        <div className="nodrag absolute left-full top-0 z-20 ml-3 w-64 rounded-xl border border-slate-600 bg-slate-900/95 p-3 shadow-xl">
          <p className="text-sm font-bold leading-snug text-red-300">{mark.text}</p>
          {(mark.details ?? []).map((line) => (
            <p key={line} className="mt-1 text-xs leading-snug text-slate-200">
              {line}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
