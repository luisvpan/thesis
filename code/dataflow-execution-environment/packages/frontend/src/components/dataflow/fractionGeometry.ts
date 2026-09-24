/**
 * Geometría del objeto incompleto.
 *
 * Una cantidad fraccionaria no se dibuja cortando el objeto, sino pintándolo
 * entero y dividido en `d` regiones radiales, de las cuales solo `n` están
 * presentes: las demás quedan como fantasma, con la línea de división punteada.
 * Así se ve a la vez en cuántos pedazos se repartió y cuántos hay.
 *
 * Por ahora las regiones son de **ángulo** igual, no de área igual, así que
 * sobre una silueta que no sea redonda 4/7 no es exactamente cuatro séptimos de
 * la figura. Es deliberado: primero la idea, luego el refinamiento.
 */

/** Hasta cuántas regiones se dibujan; por encima, la fracción solo se lee en el texto. */
export const MAX_FRACTION_PARTS = 12;

const TAU = Math.PI * 2;

/** Arranca a las 12 y avanza en sentido horario, como cualquier reloj o pastel. */
const START = -Math.PI / 2;

/**
 * Los rayos salen más allá de la caja para que la cuña cubra también las
 * esquinas: recortar fuera del borde no tiene efecto, así que sobra con pasarse.
 */
const RAY = 100;

/** Muestreo del arco. A 30° el polígono aún cubre la esquina (cos 15° > √2/2). */
const STEP = Math.PI / 6;

/** Media diagonal de la caja: la línea de división siempre alcanza el borde. */
const LINE_RADIUS = 71;

/** Si la fracción se puede dibujar; si no, el texto ya la lleva. */
export function isDrawableFraction(numerator: number, denominator: number): boolean {
  return (
    Number.isInteger(numerator) &&
    Number.isInteger(denominator) &&
    numerator > 0 &&
    numerator < denominator &&
    denominator <= MAX_FRACTION_PARTS
  );
}

/** `clip-path` que deja ver solo las `n` de `d` regiones presentes. */
export function wedgeClipPath(numerator: number, denominator: number): string {
  const end = (numerator / denominator) * TAU;
  const angles: number[] = [0];

  for (let angle = STEP; angle < end; angle += STEP) angles.push(angle);
  angles.push(end);

  const rim = angles.map((angle) => {
    const t = START + angle;
    return `${(50 + RAY * Math.cos(t)).toFixed(2)}% ${(50 + RAY * Math.sin(t)).toFixed(2)}%`;
  });

  return `polygon(50% 50%, ${rim.join(", ")})`;
}

/** Un corte, del centro al borde, en el sistema de coordenadas 0..100 del SVG. */
export type DivisionLine = {
  x: number;
  y: number;
  /** Borde de una región presente: se traza entero. Las demás, punteadas. */
  solid: boolean;
};

/** Las `d` líneas que dividen la figura, con el trazo que le toca a cada una. */
export function divisionLines(numerator: number, denominator: number): DivisionLine[] {
  return Array.from({ length: denominator }, (_, index) => {
    const t = START + (index / denominator) * TAU;
    return {
      x: Number((50 + LINE_RADIUS * Math.cos(t)).toFixed(2)),
      y: Number((50 + LINE_RADIUS * Math.sin(t)).toFixed(2)),
      // La que abre la primera región y la que cierra la última son bordes de
      // lo presente; de ahí en adelante, todo es fantasma.
      solid: index <= numerator,
    };
  });
}
