import type { ResultVisualItem } from '@/services/executeProgram';

const MONTESSORI_CUBE: Record<string, string> = {
  verde: 'bg-emerald-500 shadow-emerald-900/50',
  naranja: 'bg-orange-500 shadow-orange-900/50',
  morado: 'bg-violet-600 shadow-violet-900/50',
  rojo: 'bg-red-500 shadow-red-900/50',
  azul: 'bg-blue-600 shadow-blue-900/50',
  amarillo: 'bg-amber-300 shadow-amber-900/40',
};

const CAP_COLORS: Record<string, string> = {
  azul: 'bg-blue-500 shadow-blue-900/50',
  blanco: 'bg-white shadow-slate-400/50',
};

const STICK_COLORS: Record<string, string> = {
  cian: 'bg-cyan-400 shadow-cyan-900/50',
  naranja: 'bg-orange-500 shadow-orange-900/50',
  rojo: 'bg-red-500 shadow-red-900/50',
};

const FOOD_DOT: Record<string, string> = {
  verde: 'bg-emerald-400',
  morado: 'bg-violet-400',
  rojo: 'bg-red-400',
  azul: 'bg-sky-400',
  amarillo: 'bg-amber-200',
};

function MontessoriCube({ color }: { color: string }) {
  const palette = MONTESSORI_CUBE[color] ?? 'bg-slate-500 shadow-slate-900/50';
  return (
    <span
      title={color}
      className={`inline-block h-7 w-7 shrink-0 rounded-md shadow-lg ring-1 ring-white/25 ${palette}`}
      style={{ transform: 'perspective(120px) rotateX(12deg) rotateY(-18deg)' }}
    />
  );
}

function FormaGlyph({ subtype }: { subtype: string }) {
  return (
    <span
      title={subtype}
      className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-slate-600 text-[10px] font-bold uppercase text-white ring-1 ring-white/20"
    >
      {subtype.slice(0, 2)}
    </span>
  );
}

function ComidaGlyph({ subtype, color }: { subtype: string; color: string }) {
  const dot = FOOD_DOT[color] ?? 'bg-slate-400';
  return (
    <span
      title={`${subtype} ${color}`}
      className="relative inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-700 ring-1 ring-white/20"
    >
      <span className={`absolute bottom-1 h-2 w-2 rounded-full ${dot}`} />
      <span className="text-[9px] font-semibold text-white/90">{subtype.slice(0, 1)}</span>
    </span>
  );
}

function CapGlyph({ color }: { color: string }) {
  const palette = CAP_COLORS[color] ?? 'bg-slate-500 shadow-slate-900/50';
  return (
    <span
      title={`Tapa ${color}`}
      className={`inline-block h-7 w-7 shrink-0 rounded-full shadow-lg ring-1 ring-white/25 ${palette}`}
    />
  );
}

function StickGlyph({ color }: { color: string }) {
  const palette = STICK_COLORS[color] ?? 'bg-slate-500 shadow-slate-900/50';
  return (
    <span
      title={`Palito ${color}`}
      className={`inline-block h-7 w-2 shrink-0 rounded-sm shadow-lg ring-1 ring-white/25 ${palette}`}
    />
  );
}

type ResultArrayVisualProps = {
  items: ResultVisualItem[];
};

const MAX_SHOW = 36;

export function ResultArrayVisual({ items }: ResultArrayVisualProps) {
  if (items.length === 0) return null;

  const shown = items.slice(0, MAX_SHOW);
  const overflow = items.length - shown.length;

  return (
    <div className="mt-2 flex w-full max-w-44 flex-col items-center gap-1">
      <div className="flex flex-wrap justify-center gap-1.5">
        {shown.map((item, i) => {
          if (item.kind === 'montessori') {
            return <MontessoriCube key={`v-${i}`} color={item.color} />;
          }
          if (item.kind === 'forma') {
            return <FormaGlyph key={`v-${i}`} subtype={item.subtype} />;
          }
          if (item.kind === 'cap') {
            return <CapGlyph key={`v-${i}`} color={item.color} />;
          }
          if (item.kind === 'stick') {
            return <StickGlyph key={`v-${i}`} color={item.color} />;
          }
          return (
            <ComidaGlyph
              key={`v-${i}`}
              subtype={item.subtype}
              color={item.color}
            />
          );
        })}
      </div>
      {overflow > 0 ? (
        <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
      ) : null}
    </div>
  );
}
