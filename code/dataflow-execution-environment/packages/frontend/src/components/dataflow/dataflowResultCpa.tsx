import type { ReactNode } from 'react';

/**
 * Modo de PRESENTACIÓN visual del resultado.
 * Solo afecta cómo se renderiza un valor, no la semántica.
 * La semántica CPA está en el `category` del CPAObject (interpreter).
 */
export type ResultViewMode = 'pictorico' | 'concreto' | 'abstracto';

const numberNames: Record<number, string> = {
  0: 'Cero',
  1: 'Uno',
  2: 'Dos',
  3: 'Tres',
  4: 'Cuatro',
  5: 'Cinco',
  6: 'Seis',
  7: 'Siete',
  8: 'Ocho',
  9: 'Nueve',
};

export function formatResultCpa(value: number, mode: ResultViewMode): ReactNode {
  if (!Number.isFinite(value)) {
    return String(value);
  }

  const isInt = Number.isInteger(value);

  switch (mode) {
    case 'abstracto':
      return <span className="tabular-nums">{value}</span>;

    case 'concreto': {
      if (isInt && value >= 0 && value <= 9) {
        return <span>{numberNames[value]}</span>;
      }
      return <span className="tabular-nums">{value}</span>;
    }

    case 'pictorico': {
      if (isInt && value >= 0 && value <= 24) {
        return (
          <span className="flex flex-wrap gap-1 max-w-[min(280px,85vw)] justify-center items-center leading-none">
            {Array.from({ length: value }, (_, i) => (
              <span key={i} className="text-teal-400 text-3xl select-none" aria-hidden>
                ●
              </span>
            ))}
            {value === 0 ? (
              <span className="text-slate-500 text-lg italic">vacío</span>
            ) : null}
          </span>
        );
      }
      return <span className="tabular-nums text-5xl font-black text-teal-300">{value}</span>;
    }

    default:
      return String(value);
  }
}

/**
 * Formats a fraction as "numerator/denominator" for abstract mode display.
 * Returns just the numerator if denominator is 1 (integer result).
 */
export function formatFraction(numerator: number, denominator: number): ReactNode {
  if (denominator === 1) {
    return <span className="tabular-nums">{numerator}</span>;
  }
  return (
    <span className="tabular-nums">
      {numerator}/{denominator}
    </span>
  );
}
