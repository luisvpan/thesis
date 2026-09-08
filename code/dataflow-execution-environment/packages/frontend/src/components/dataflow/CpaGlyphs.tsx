/**
 * Glyph components for CPA object visualization.
 * Used by ResultArrayVisual and ProgramOutputFlowNode.
 */

import { foodEmoji } from '@/data/foodEmoji';
import type { FoodType, ShapeColor, ShapeSize, ShapeType } from '@/types/card-types';
import { MiniShapeGlyph } from './MiniShapeGlyph';

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
  azul: 'bg-blue-500 shadow-blue-900/50',
  naranja: 'bg-orange-500 shadow-orange-900/50',
  rojo: 'bg-red-500 shadow-red-900/50',
  madera: 'bg-amber-200 shadow-amber-900/40',
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
      title={generic ? 'paleta' : `Paleta ${color}`}
      className={`inline-block h-7 w-2 shrink-0 rounded-sm shadow-lg ring-1 ring-white/25 ${palette}`}
    />
  );
}

const SHAPE_SIZE_MAP: Record<string, ShapeSize> = {
  pequeño: 'pequeño',
  pequeña: 'pequeño',
  mediano: 'mediano',
  mediana: 'mediano',
  grande: 'grande',
};

function normalizeShapeSize(size?: string): ShapeSize {
  if (!size) return 'mediano';
  return SHAPE_SIZE_MAP[size] ?? 'mediano';
}

export function FormaGlyph({
  subtype,
  color,
  size,
  generic = false,
}: { subtype: string; color?: string; size?: string } & GlyphProps) {
  if (!generic) {
    const shape = subtype as ShapeType;
    const shapeSize = normalizeShapeSize(size);
    const shapeColor = (color as ShapeColor | undefined) ?? 'amarillo';
    return (
      <span className="inline-flex shrink-0 items-center justify-center" title={`${subtype} ${shapeSize}`}>
        <MiniShapeGlyph shape={shape} size={shapeSize} color={shapeColor} generic={false} />
      </span>
    );
  }

  return (
    <span className="inline-flex shrink-0 items-center justify-center" title={subtype}>
      <MiniShapeGlyph
        shape={subtype as ShapeType}
        size={normalizeShapeSize(size)}
        color="amarillo"
        generic
      />
    </span>
  );
}

export function ComidaGlyph({
  subtype,
  color,
  generic = false,
}: { subtype: string; color: string } & GlyphProps) {
  if (!generic) {
    return (
      <span
        className="inline-flex h-7 w-7 shrink-0 items-center justify-center text-xl leading-none"
        role="img"
        aria-label={subtype}
        title={`${subtype} ${color}`}
      >
        {foodEmoji(subtype as FoodType)}
      </span>
    );
  }

  const dot = GENERIC_DOT;
  const bgClass = 'bg-teal-700';
  return (
    <span
      title={subtype}
      className={`relative inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${bgClass} ring-1 ring-white/20`}
    >
      <span className={`absolute bottom-1 h-2 w-2 rounded-full ${dot}`} />
      <span className="text-[9px] font-semibold text-white/90">{subtype.slice(0, 1)}</span>
    </span>
  );
}
