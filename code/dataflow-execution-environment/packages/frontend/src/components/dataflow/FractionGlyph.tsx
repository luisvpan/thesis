import type { ReactNode } from 'react';
import { divisionLines, wedgeClipPath } from './fractionGeometry';

/**
 * Un objeto incompleto: el mismo glifo dos veces —el fantasma entero debajo y
 * la parte presente recortada encima— con las líneas de división por delante.
 *
 * El recorte va por `clip-path` sobre la caja ya dibujada, así que le da igual
 * lo que haya dentro: un `<span>` con color de fondo, un SVG o un emoji. Un
 * solo componente sirve para los cinco glifos.
 */

/** Lo ausente no desaparece: se queda de fondo para que se vea qué falta. */
const GHOST_OPACITY = 0.28;

type FractionGlyphProps = {
  numerator: number;
  denominator: number;
  children: ReactNode;
};

export function FractionGlyph({ numerator, denominator, children }: FractionGlyphProps) {
  return (
    <span
      className="relative inline-flex shrink-0 items-center justify-center"
      title={`${numerator}/${denominator}`}
    >
      <span className="inline-flex" style={{ opacity: GHOST_OPACITY }}>
        {children}
      </span>
      <span
        className="absolute inset-0 inline-flex items-center justify-center"
        style={{ clipPath: wedgeClipPath(numerator, denominator) }}
        aria-hidden
      >
        {children}
      </span>
      <svg
        className="pointer-events-none absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden
      >
        {divisionLines(numerator, denominator).map((line, index) => (
          <line
            key={`cut-${index}`}
            x1="50"
            y1="50"
            x2={line.x}
            y2={line.y}
            stroke="#fff"
            strokeOpacity={line.solid ? 0.8 : 0.4}
            strokeWidth={line.solid ? 3 : 2.5}
            strokeDasharray={line.solid ? undefined : '7 5'}
            strokeLinecap="round"
          />
        ))}
      </svg>
    </span>
  );
}
