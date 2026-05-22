/** Progreso 0–1 en bucle para animación sobre un conector. */
export function loopEdgeProgress(nowMs: number, durationMs: number): number {
  if (durationMs <= 0) return 0;
  const t = nowMs / durationMs;
  return t - Math.floor(t);
}

/** Desplaza un punto perpendicular a la tangente hacia arriba en pantalla (y hacia abajo). */
export function offsetPointAbove(
  x: number,
  y: number,
  tangentX: number,
  tangentY: number,
  offsetPx: number
): { x: number; y: number } {
  const mag = Math.hypot(tangentX, tangentY) || 1;
  let nx = -tangentY / mag;
  let ny = tangentX / mag;
  if (ny > 0) {
    nx = -nx;
    ny = -ny;
  }
  return { x: x + nx * offsetPx, y: y + ny * offsetPx };
}

/**
 * Punto sobre un path SVG en t ∈ [0, 1], con offset perpendicular hacia arriba.
 */
export function samplePathPoint(
  path: SVGPathElement,
  t: number,
  offsetAbovePx = 12
): { x: number; y: number } {
  const clamped = Math.max(0, Math.min(1, t));
  const len = path.getTotalLength();
  const at = clamped * len;
  const pt = path.getPointAtLength(at);
  const ahead = Math.min(at + 1, len);
  const pt2 = path.getPointAtLength(ahead);
  return offsetPointAbove(pt.x, pt.y, pt2.x - pt.x, pt2.y - pt.y, offsetAbovePx);
}

export const DEFAULT_EDGE_WALK_DURATION_MS = 3500;
