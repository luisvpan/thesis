import type { ProgramOutputFlowNodeData } from "@/components/dataflow";
import { parseVisionLabel, visionOperatorToMathOperator } from "@/types/vision-card";
import { VISION_PROGRAM_OUTPUT_ID } from "@/utils/frontendFlowConstants";
import type { CardDetectionsPayload } from "../VisionContext";
import {
  VISION_CARD_BOX,
  VISION_RESULT_GAP,
  VISION_FLOW_MIN_SIZE,
} from "./constants";
import { toValidSlug, visionToFlowPosition } from "./helpers";
import type { DataflowNode } from "./types";

/**
 * Reemplaza nodos `card_*` por el estado derivado del último frame de visión.
 * Conserva nodos que no son `card_*` y el valor del nodo de salida fijo si existe.
 */
export function mergeVisionFrameIntoNodes(
  prev: DataflowNode[],
  lastCardFrame: CardDetectionsPayload,
  rect: DOMRectReadOnly
): DataflowNode[] {
  if (
    rect.width < VISION_FLOW_MIN_SIZE ||
    rect.height < VISION_FLOW_MIN_SIZE
  ) {
    return prev;
  }

  const prevOut = prev.find(
    (n) => n.id === VISION_PROGRAM_OUTPUT_ID && n.type === "programOutput"
  );
  const preservedValue =
    prevOut?.type === "programOutput"
      ? (prevOut.data as ProgramOutputFlowNodeData).value
      : undefined;

  const withoutLive = prev.filter((n) => !n.id.startsWith("card_"));

  let grapesFlowPos: { x: number; y: number } | null = null;
  const additions: DataflowNode[] = [];
  let idx = 0;

  for (const c of lastCardFrame.cards) {
    const parsed = parseVisionLabel(c.label);

    if (parsed.type === "resultAnchor") {
      grapesFlowPos = visionToFlowPosition(c.position, rect);
      continue;
    }

    const position = visionToFlowPosition(c.position, rect);
    const nodeId = toValidSlug(c.trackId, idx);

    if (parsed.type === "number") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "number",
          value: parsed.value,
          trackId: c.trackId,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "operator") {
      additions.push({
        id: nodeId,
        type: "operator" as const,
        position,
        data: {
          operator: visionOperatorToMathOperator(parsed.operator),
          trackId: c.trackId,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "operatorCanvas") {
      additions.push({
        id: nodeId,
        type: "operator" as const,
        position,
        data: {
          operator: parsed.operator,
          trackId: c.trackId,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "programResultCard") {
      additions.push({
        id: nodeId,
        type: "programOutput" as const,
        position,
        data: {},
      });
      idx++;
      continue;
    }

    if (parsed.type === "visionArrayOpen") {
      additions.push({
        id: nodeId,
        type: "arrayOpen" as const,
        position,
        data: {},
      });
      idx++;
      continue;
    }

    if (parsed.type === "visionArrayClose") {
      additions.push({
        id: nodeId,
        type: "arrayClose" as const,
        position,
        data: {},
      });
      idx++;
      continue;
    }

    if (parsed.type === "deckShape") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "shape",
          yoloClass: parsed.yoloClass,
          shape: parsed.shape,
          size: parsed.size,
          color: parsed.color,
          trackId: c.trackId,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "deckFood") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "food",
          yoloClass: parsed.yoloClass,
          food: parsed.food,
          trackId: c.trackId,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "deckMontessori") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "montessori",
          yoloClass: parsed.yoloClass,
          color: parsed.color,
          trackId: c.trackId,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "unknown") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "number",
          value: 0,
          visionSubtitle: parsed.label,
          trackId: c.trackId,
        },
      });
      idx++;
      continue;
    }
  }

  if (grapesFlowPos) {
    additions.push({
      id: VISION_PROGRAM_OUTPUT_ID,
      type: "programOutput" as const,
      position: {
        x: grapesFlowPos.x + VISION_CARD_BOX + VISION_RESULT_GAP,
        y: grapesFlowPos.y,
      },
      data: { value: preservedValue },
    });
  }

  return [...withoutLive, ...additions];
}
