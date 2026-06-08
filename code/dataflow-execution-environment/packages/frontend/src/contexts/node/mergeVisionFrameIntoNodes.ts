import type {
  OperatorFlowNodeData,
  ProgramOutputFlowNodeData,
} from "@/components/dataflow";
import { parseVisionLabel, visionOperatorToMathOperator } from "@/types/vision-card";
import type { CardDetectionsPayload } from "../VisionContext";
import { VISION_CARD_NODE_TTL_MS, VISION_FLOW_MIN_SIZE } from "./constants";
import { toValidSlug } from "./helpers";
import { resolveVisionNodePosition } from "./resolveVisionNodePosition";
import type { DataflowNode } from "./types";
import {
  readVisionMeta,
  visionMetaFromCard,
  type VisionNodeMeta,
} from "./visionNodeMeta";
import { stabilizeVisionNodeList } from "./stabilizeVisionNodes";
import { withVisionNodeChrome } from "./visionNodePresentation";

/**
 * Fusiona el frame de visión con nodos previos:
 * - Sigue la visión hasta {@link VISION_POSITION_LOCK_MS}; luego fija posición en el lienzo.
 * - Conserva nodos no vistos hasta {@link VISION_CARD_NODE_TTL_MS} para no romper aristas.
 */
export function mergeVisionFrameIntoNodes(
  prev: DataflowNode[],
  lastCardFrame: CardDetectionsPayload,
  rect: DOMRectReadOnly,
  nodesDraggable = false
): DataflowNode[] {
  if (
    rect.width < VISION_FLOW_MIN_SIZE ||
    rect.height < VISION_FLOW_MIN_SIZE
  ) {
    return prev;
  }

  const frameTimeMs = lastCardFrame.t ?? Date.now();

  const withoutLive = prev.filter((n) => !n.id.startsWith("card_"));
  const prevCards = prev.filter((n) => n.id.startsWith("card_"));

  const additions: DataflowNode[] = [];
  const frameNodeIds = new Set<string>();
  let idx = 0;

  for (const c of lastCardFrame.cards) {
    const parsed = parseVisionLabel(c.label);

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
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "number",
              value: parsed.value,
              digitValue: parsed.value,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "operator") {
      const prevOp = prevNode?.data as OperatorFlowNodeData | undefined;
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "operator" as const,
            position,
            data: prevOp
              ? {
                  ...prevOp,
                  operator: visionOperatorToMathOperator(parsed.operator),
                  ...meta,
                }
              : {
                  operator: visionOperatorToMathOperator(parsed.operator),
                  ...meta,
                },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "operatorCanvas") {
      const prevOp = prevNode?.data as OperatorFlowNodeData | undefined;
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "operator" as const,
            position,
            data: prevOp
              ? { ...prevOp, operator: parsed.operator, criterio: parsed.criterio, ...meta }
              : { operator: parsed.operator, criterio: parsed.criterio, ...meta },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "programResultCard") {
      const prevData = prevNode?.data as ProgramOutputFlowNodeData | undefined;
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "programOutput" as const,
            position,
            data: prevData ? { ...prevData, ...meta } : { ...meta },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "visionArrayOpen") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "arrayOpen" as const,
            position,
            data: { ...meta },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "visionArrayClose") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "arrayClose" as const,
            position,
            data: { ...meta },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "deckShape") {
      additions.push(
        withVisionNodeChrome(
          {
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
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "deckFood") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "food",
              yoloClass: parsed.yoloClass,
              food: parsed.food,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "deckMontessori") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "montessori",
              yoloClass: parsed.yoloClass,
              color: parsed.color,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "deckCap") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "cap",
              yoloClass: parsed.yoloClass,
              color: parsed.color,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "deckStick") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "stick",
              yoloClass: parsed.yoloClass,
              color: parsed.color,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "deckCriteria") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "criteria",
              yoloClass: parsed.yoloClass,
              properties: parsed.properties,
              values: parsed.values,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "deckDice") {
      const prevData = prevNode?.data as { value?: number; previewFace?: number } | undefined;
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "dice",
              value: prevData?.value,
              previewFace: prevData?.previewFace,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }

    if (parsed.type === "unknown") {
      additions.push(
        withVisionNodeChrome(
          {
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "number",
              value: 0,
              visionSubtitle: parsed.label,
              ...meta,
            },
          },
          meta,
          nodesDraggable
        )
      );
      idx++;
      continue;
    }
  }

  const retainedStale = retainStaleCardNodes(
    prevCards,
    frameNodeIds,
    frameTimeMs,
    nodesDraggable
  );

  const merged = [...withoutLive, ...additions, ...retainedStale];
  return stabilizeVisionNodeList(prev, merged);
}

/** Nodos `card_*` ausentes del frame pero dentro del TTL — mantienen aristas del usuario. */
function retainStaleCardNodes(
  prevCards: DataflowNode[],
  frameNodeIds: Set<string>,
  frameTimeMs: number,
  nodesDraggable: boolean
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

    retained.push(
      withVisionNodeChrome(
        {
          ...node,
          data: {
            ...(node.data as Record<string, unknown>),
            ...staleMeta,
          },
        } as DataflowNode,
        staleMeta,
        nodesDraggable
      )
    );
  }

  return retained;
}
