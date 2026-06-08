/** Patrones de puntos para caras 1–6 de un dado. */
const DOT_PATTERNS: Record<number, Array<[number, number]>> = {
  1: [[50, 50]],
  2: [
    [28, 28],
    [72, 72],
  ],
  3: [
    [28, 28],
    [50, 50],
    [72, 72],
  ],
  4: [
    [28, 28],
    [72, 28],
    [28, 72],
    [72, 72],
  ],
  5: [
    [28, 28],
    [72, 28],
    [50, 50],
    [28, 72],
    [72, 72],
  ],
  6: [
    [28, 28],
    [72, 28],
    [28, 50],
    [72, 50],
    [28, 72],
    [72, 72],
  ],
};

type DiceFaceProps = {
  value?: number;
  spinning?: boolean;
  className?: string;
};

export function DiceFace({ value, spinning = false, className = '' }: DiceFaceProps) {
  const face = value && value >= 1 && value <= 6 ? value : undefined;
  const dots = face ? DOT_PATTERNS[face] : [];

  return (
    <div
      className={`relative flex h-16 w-16 items-center justify-center ${className}`}
      aria-label={face ? `Dado: ${face}` : 'Dado sin lanzar'}
    >
      <div
        className={`relative h-14 w-14 rounded-xl border-2 border-slate-300 bg-white shadow-lg ${
          spinning ? 'animate-dice-roll' : ''
        }`}
      >
        {face ? (
          dots.map(([x, y], i) => (
            <span
              key={`dot-${i}`}
              className="absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-800"
              style={{ left: `${x}%`, top: `${y}%` }}
            />
          ))
        ) : (
          <span className="absolute inset-0 flex items-center justify-center text-2xl font-black text-slate-400">
            ?
          </span>
        )}
      </div>
    </div>
  );
}
