import type { ProgramOutputFlowNodeData } from "@/components/dataflow";
import { parseVisionLabel, visionOperatorToMathOperator } from "@/types/vision-card";
import { VISION_PROGRAM_OUTPUT_ID } from "@/utils/frontendFlowConstants";
import type { CardDetectionsPayload } from "../VisionContext";
import {
  VISION_CARD_BOX,
  VISION_CARD_NODE_TTL_MS,
  VISION_RESULT_GAP,
  VISION_FLOW_MIN_SIZE,
} from "./constants";
import { toValidSlug, visionToFlowPosition } from "./helpers";
import { resolveVisionNodePosition } from "./resolveVisionNodePosition";
import type { DataflowNode } from "./types";
import {
  readVisionMeta,
  visionMetaFromCard,
  type VisionNodeMeta,
} from "./visionNodeMeta";

/**
 * Fusiona el frame de visión con nodos previos:
 * - Sigue la visión hasta {@link VISION_POSITION_LOCK_MS}; luego fija posición en el lienzo.
 * - Conserva nodos no vistos hasta {@link VISION_CARD_NODE_TTL_MS} para no romper aristas.
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

  const frameTimeMs = lastCardFrame.t ?? Date.now();

  const prevOut = prev.find(
    (n) => n.id === VISION_PROGRAM_OUTPUT_ID && n.type === "programOutput"
  );
  const preservedValue =
    prevOut?.type === "programOutput"
      ? (prevOut.data as ProgramOutputFlowNodeData).value
      : undefined;

  const withoutLive = prev.filter((n) => !n.id.startsWith("card_"));
  const prevCards = prev.filter((n) => n.id.startsWith("card_"));

  let grapesFlowPos: { x: number; y: number } | null = null;
  const additions: DataflowNode[] = [];
  const frameNodeIds = new Set<string>();
  let idx = 0;

  for (const c of lastCardFrame.cards) {
    const parsed = parseVisionLabel(c.label);

    if (parsed.type === "resultAnchor") {
      grapesFlowPos = visionToFlowPosition(c.position, rect);
      continue;
    }

    const nodeId = toValidSlug(c.trackId, idx);
    frameNodeIds.add(nodeId);
    const prevNode = prevCards.find((n) => n.id === nodeId);
    const meta = visionMetaFromCard(
      c,
      frameTimeMs,
      prevNode ? readVisionMeta(prevNode.data) : undefined
    );
    const position = resolveVisionNodePosition(
      nodeId,
      c,
      rect,
      frameTimeMs,
      prevCards
    );

    if (parsed.type === "number") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "number",
          value: parsed.value,
          ...meta,
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
          ...meta,
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
          ...meta,
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
        data: { ...meta },
      });
      idx++;
      continue;
    }

    if (parsed.type === "visionArrayOpen") {
      additions.push({
        id: nodeId,
        type: "arrayOpen" as const,
        position,
        data: { ...meta },
      });
      idx++;
      continue;
    }

    if (parsed.type === "visionArrayClose") {
      additions.push({
        id: nodeId,
        type: "arrayClose" as const,
        position,
        data: { ...meta },
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
          ...meta,
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
          ...meta,
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
          ...meta,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "deckCap") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "cap",
          yoloClass: parsed.yoloClass,
          color: parsed.color,
          ...meta,
        },
      });
      idx++;
      continue;
    }

    if (parsed.type === "deckStick") {
      additions.push({
        id: nodeId,
        type: "source" as const,
        position,
        data: {
          variant: "stick",
          yoloClass: parsed.yoloClass,
          color: parsed.color,
          ...meta,
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
          ...meta,
        },
      });
      idx++;
      continue;
    }
  }

  const retainedStale = retainStaleCardNodes(
    prevCards,
    frameNodeIds,
    frameTimeMs
  );

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

  return [...withoutLive, ...additions, ...retainedStale];
}

/** Nodos `card_*` ausentes del frame pero dentro del TTL — mantienen aristas del usuario. */
function retainStaleCardNodes(
  prevCards: DataflowNode[],
  frameNodeIds: Set<string>,
  frameTimeMs: number
): DataflowNode[] {
  const retained: DataflowNode[] = [];

  for (const node of prevCards) {
    if (frameNodeIds.has(node.id)) continue;

    const { lastSeenAt } = readVisionMeta(node.data);
    const lastSeen = lastSeenAt ?? frameTimeMs;
    if (frameTimeMs - lastSeen > VISION_CARD_NODE_TTL_MS) continue;

    const staleMeta: VisionNodeMeta = {
      ...readVisionMeta(node.data),
      visionStatus: "stale",
      lastSeenAt: lastSeen,
    };

    retained.push({
      ...node,
      data: {
        ...(node.data as Record<string, unknown>),
        ...staleMeta,
      },
    });
  }

  return retained;
}
