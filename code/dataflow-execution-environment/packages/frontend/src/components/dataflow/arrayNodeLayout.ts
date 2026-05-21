/**
 * Layout de ArrayOpenNode / ArrayCloseNode.
 * Mantener alineado con arrayZoneGeometry.ts y ClickableHandle (h-20 w-20).
 */

/** Tailwind h-52 → 13rem */
export const ARRAY_NODE_HEIGHT = 208;

/** Tailwind w-26 → 6.5rem */
export const ARRAY_NODE_WIDTH = 104;

/** ClickableHandle zone-out / zone-in: translateX(±100px) */
export const ARRAY_ZONE_HANDLE_OFFSET_X = 100;

/** zone-in en ArrayCloseNode: top 28% */
export const ARRAY_CLOSE_ZONE_IN_TOP_FRAC = 0.28;

/** ClickableHandle: h-20 w-20 */
export const FLOW_HANDLE_SIZE = 80;

export const ARRAY_ZONE_OUT_HANDLE_STYLE = {
  transform: `translateX(${ARRAY_ZONE_HANDLE_OFFSET_X}px)`,
} as const;

export const ARRAY_ZONE_IN_HANDLE_STYLE = {
  top: `${ARRAY_CLOSE_ZONE_IN_TOP_FRAC * 100}%`,
  transform: `translateX(-${ARRAY_ZONE_HANDLE_OFFSET_X}px)`,
} as const;
