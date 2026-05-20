/**
 * Glyph components for CPA object visualization.
 * Used by ResultArrayVisual and ProgramOutputFlowNode.
 */

// =============================================================================
// Color Palettes
// =============================================================================

export const MONTESSORI_CUBE_COLORS: Record<string, string> = {
  verde: 'bg-emerald-500 shadow-emerald-900/50',
  naranja: 'bg-orange-500 shadow-orange-900/50',
  morado: 'bg-violet-600 shadow-violet-900/50',
  rojo: 'bg-red-500 shadow-red-900/50',
  azul: 'bg-blue-600 shadow-blue-900/50',
  amarillo: 'bg-amber-300 shadow-amber-900/40',
};

export const CAP_COLORS: Record<string, string> = {
  azul: 'bg-blue-500 shadow-blue-900/50',
  blanco: 'bg-white shadow-slate-400/50',
};

export const STICK_COLORS: Record<string, string> = {
  cian: 'bg-cyan-400 shadow-cyan-900/50',
  naranja: 'bg-orange-500 shadow-orange-900/50',
  rojo: 'bg-red-500 shadow-red-900/50',
};

export const FOOD_DOT_COLORS: Record<string, string> = {
  verde: 'bg-emerald-400',
  morado: 'bg-violet-400',
  rojo: 'bg-red-400',
  azul: 'bg-sky-400',
  amarillo: 'bg-amber-200',
};

// Generic (teal) palette for pictorico mode
const GENERIC_PALETTE = 'bg-teal-400 shadow-teal-900/50';
const GENERIC_DOT = 'bg-teal-400';
const FALLBACK_PALETTE = 'bg-slate-500 shadow-slate-900/50';

// =============================================================================
// Glyph Components
// =============================================================================

type GlyphProps = {
  generic?: boolean;
};

export function MontessoriCubeGlyph({ color, generic = false }: { color: string } & GlyphProps) {
  const palette = generic
    ? GENERIC_PALETTE
    : MONTESSORI_CUBE_COLORS[color] ?? FALLBACK_PALETTE;
  return (
    <span
      title={generic ? 'cubo' : color}
      className={`inline-block h-7 w-7 shrink-0 rounded-md shadow-lg ring-1 ring-white/25 ${palette}`}
      style={{ transform: 'perspective(120px) rotateX(12deg) rotateY(-18deg)' }}
    />
  );
}

export function CapGlyph({ color, generic = false }: { color: string } & GlyphProps) {
  const palette = generic
    ? GENERIC_PALETTE
    : CAP_COLORS[color] ?? FALLBACK_PALETTE;
  return (
    <span
      title={generic ? 'tapa' : `Tapa ${color}`}
      className={`inline-block h-7 w-7 shrink-0 rounded-full shadow-lg ring-1 ring-white/25 ${palette}`}
    />
  );
}

export function StickGlyph({ color, generic = false }: { color: string } & GlyphProps) {
  const palette = generic
    ? GENERIC_PALETTE
    : STICK_COLORS[color] ?? FALLBACK_PALETTE;
  return (
    <span
      title={generic ? 'palito' : `Palito ${color}`}
      className={`inline-block h-7 w-2 shrink-0 rounded-sm shadow-lg ring-1 ring-white/25 ${palette}`}
    />
  );
}

export function FormaGlyph({ subtype, generic = false }: { subtype: string } & GlyphProps) {
  const bgClass = generic ? 'bg-teal-600' : 'bg-slate-600';
  return (
    <span
      title={subtype}
      className={`inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${bgClass} text-[10px] font-bold uppercase text-white ring-1 ring-white/20`}
    >
      {subtype.slice(0, 2)}
    </span>
  );
}

export function ComidaGlyph({ subtype, color, generic = false }: { subtype: string; color: string } & GlyphProps) {
  const dot = generic ? GENERIC_DOT : FOOD_DOT_COLORS[color] ?? 'bg-slate-400';
  const bgClass = generic ? 'bg-teal-700' : 'bg-slate-700';
  return (
    <span
      title={generic ? subtype : `${subtype} ${color}`}
      className={`relative inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${bgClass} ring-1 ring-white/20`}
    >
      <span className={`absolute bottom-1 h-2 w-2 rounded-full ${dot}`} />
      <span className="text-[9px] font-semibold text-white/90">{subtype.slice(0, 1)}</span>
    </span>
  );
}
