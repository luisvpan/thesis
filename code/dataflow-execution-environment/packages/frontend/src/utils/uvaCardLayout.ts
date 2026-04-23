/**
 * Carta «uva»: dos nodos en el grafo (marcador + resultado), un solo aspecto visual.
 * Ancho total = 2.5 × una carta normal (NumberFlowNode usa w-60 ≈ 240px).
 */
export const UVA_NORMAL_CARD_PX = 240;
export const UVA_MERGED_TOTAL_PX = Math.round(UVA_NORMAL_CARD_PX * 2.5);
/** Panel izquierdo: posición física del marcador (igual ancho que una carta estándar). */
export const UVA_LEFT_PANEL_PX = UVA_NORMAL_CARD_PX;
/** Panel derecho: resultado virtual. */
export const UVA_RIGHT_PANEL_PX = UVA_MERGED_TOTAL_PX - UVA_LEFT_PANEL_PX;

/** Misma traslación que NumberFlowNode (-translate-x-[30%] / -translate-y-[25%] sobre w-60 h-60). */
export const UVA_PAIR_TRANSFORM_CLASS =
  "-translate-x-[4.5rem] -translate-y-[3.75rem]";
