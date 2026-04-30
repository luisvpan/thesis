import type { PortDefinition } from "./types";

export const SOURCE_PORTS: PortDefinition[] = [
  { handleId: "out", handleType: "source", position: "right" },
];

export const OPERATOR_PORTS: PortDefinition[] = [
  { handleId: "a", handleType: "target", position: "left", offsetY: "25%" },
  { handleId: "b", handleType: "target", position: "left", offsetY: "75%" },
  { handleId: "out", handleType: "source", position: "right" },
];

export const VISION_FLOW_MIN_SIZE = 64;
export const VISION_NODE_HALF_W = 48;
export const VISION_NODE_HALF_H = 40;

/** Coincide con el ancho visual del nodo en el lienzo + margen hasta la carta de resultado */
export const VISION_CARD_BOX = 240;
export const VISION_RESULT_GAP = 24;
