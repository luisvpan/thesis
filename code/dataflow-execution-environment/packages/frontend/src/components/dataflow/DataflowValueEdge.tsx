import type { Edge, EdgeProps } from "@xyflow/react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useEdges,
  useNodes,
} from "@xyflow/react";
import type { DataflowNode } from "@/contexts/NodeContext";
import { getOutboundFlowValue } from "@/contexts/NodeContext";
import { useResultCardUi } from "@/contexts/ResultCardUiContext";

function formatEdgeValue(n: number): string {
  if (!Number.isFinite(n)) return "—";
  if (Number.isInteger(n)) return String(n);
  const s = n.toFixed(6).replace(/\.?0+$/, "");
  return s || "0";
}

/** Arista por defecto: muestra el valor que sale del nodo origen, centrado y por encima del cable. */
export function DataflowValueEdge(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    source,
    markerEnd,
    markerStart,
    style,
    interactionWidth,
  } = props;

  const nodes = useNodes();
  const edges = useEdges();
  const { showOperatorResults } = useResultCardUi();

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const hidden =
    style &&
    typeof style === "object" &&
    "opacity" in style &&
    (style as { opacity?: number }).opacity === 0;

  const raw = getOutboundFlowValue(
    source,
    nodes as DataflowNode[],
    edges as Edge[]
  );
  const labelText =
    !showOperatorResults ||
    hidden ||
    raw === undefined ||
    !Number.isFinite(raw)
      ? null
      : formatEdgeValue(raw);

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        markerStart={markerStart}
        style={style}
        interactionWidth={interactionWidth}
      />
      {labelText !== null ? (
        <EdgeLabelRenderer>
          <div
            className="nodrag nopan pointer-events-none"
            style={{
              position: "absolute",
              transform: `translate(-50%, -135%) translate(${labelX}px,${labelY}px)`,
              fontSize: "15px",
              fontWeight: 700,
              color: "#f1f5f9",
              textShadow:
                "0 0 8px rgb(0 0 0 / 0.95), 0 1px 3px rgb(0 0 0 / 0.9)",
              zIndex: 10,
            }}
          >
            {labelText}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}
