import { useId } from 'react';
import type { ShapeColor, ShapeSize, ShapeType } from '@/types/card-types';

const SHAPE_COLORS: Record<ShapeColor, string> = {
  rojo: '#ef4444',
  azul: '#3b82f6',
  amarillo: '#eab308',
  verde: '#10b981',
  morado: '#7c3aed',
  naranja: '#f97316',
};

const GENERIC_COLOR = '#2dd4bf';

/** Tamaños del token caminante (px) por variante pequeño / mediano / grande. */
const MINI_SHAPE_PX: Record<ShapeSize, number> = {
  pequeño: 14,
  mediano: 22,
  grande: 30,
};

type MiniShapeGlyphProps = {
  shape: ShapeType;
  size: ShapeSize;
  color: ShapeColor;
  /** Pictórico: forma teal genérica manteniendo silueta y escala. */
  generic?: boolean;
};

export function MiniShapeGlyph({
  shape,
  size,
  color,
  generic = false,
}: MiniShapeGlyphProps) {
  const uid = useId().replace(/:/g, '');
  const hex = generic ? GENERIC_COLOR : (SHAPE_COLORS[color] ?? '#94a3b8');
  const px = MINI_SHAPE_PX[size];
  const gradId = `mini-${shape}-${uid}`;

  const gradient = (
    <defs>
      <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor={hex} stopOpacity={1} />
        <stop offset="100%" stopColor={hex} stopOpacity={0.7} />
      </linearGradient>
    </defs>
  );

  const fill = `url(#${gradId})`;
  const stroke = hex;
  const sw = 2;

  switch (shape) {
    case 'triangulo':
      return (
        <svg width={px} height={px} viewBox="0 0 100 100" aria-hidden>
          {gradient}
          <polygon
            points="50,10 90,90 10,90"
            fill={fill}
            stroke={stroke}
            strokeWidth={sw}
          />
        </svg>
      );
    case 'cuadrado':
      return (
        <svg width={px} height={px} viewBox="0 0 100 100" aria-hidden>
          {gradient}
          <rect x="10" y="10" width="80" height="80" fill={fill} stroke={stroke} strokeWidth={sw} />
        </svg>
      );
    case 'circulo':
      return (
        <svg width={px} height={px} viewBox="0 0 100 100" aria-hidden>
          {gradient}
          <circle cx="50" cy="50" r="40" fill={fill} stroke={stroke} strokeWidth={sw} />
        </svg>
      );
    case 'rectangulo':
      return (
        <svg width={px * 1.35} height={px} viewBox="0 0 150 100" aria-hidden>
          {gradient}
          <rect x="10" y="25" width="130" height="50" fill={fill} stroke={stroke} strokeWidth={sw} />
        </svg>
      );
    case 'rombo':
      return (
        <svg width={px} height={px} viewBox="0 0 100 100" aria-hidden>
          {gradient}
          <polygon
            points="50,10 90,50 50,90 10,50"
            fill={fill}
            stroke={stroke}
            strokeWidth={sw}
          />
        </svg>
      );
    case 'estrella':
      return (
        <svg width={px} height={px} viewBox="0 0 100 100" aria-hidden>
          {gradient}
          <polygon
            points="50,10 61,40 92,40 67,59 78,90 50,70 22,90 33,59 8,40 39,40"
            fill={fill}
            stroke={stroke}
            strokeWidth={sw}
          />
        </svg>
      );
    case 'trapecio':
      return (
        <svg width={px} height={px} viewBox="0 0 100 100" aria-hidden>
          {gradient}
          <polygon
            points="30,20 70,20 90,80 10,80"
            fill={fill}
            stroke={stroke}
            strokeWidth={sw}
          />
        </svg>
      );
    default:
      return null;
  }
}
