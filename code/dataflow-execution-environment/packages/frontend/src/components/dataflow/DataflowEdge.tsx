import { getBezierPath, type Edge, type EdgeProps } from '@xyflow/react';

type DataflowEdgeData = {
  tolerated?: boolean;
};

export function DataflowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps<Edge<DataflowEdgeData>>) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const tolerated = data?.tolerated ?? false;
  const dashArray = tolerated ? '6 4' : undefined;

  return (
    <g>
      {/* Glow outer */}
      <path
        d={edgePath}
        stroke="#4ade80"
        strokeWidth={10}
        fill="none"
        strokeOpacity={0.08}
        strokeLinecap="round"
        strokeDasharray={dashArray}
      />
      {/* Glow inner */}
      <path
        d={edgePath}
        stroke="#4ade80"
        strokeWidth={5}
        fill="none"
        strokeOpacity={0.18}
        strokeLinecap="round"
        strokeDasharray={dashArray}
      />
      {/* Main line */}
      <path
        id={id}
        d={edgePath}
        stroke="#4ade80"
        strokeWidth={2.5}
        fill="none"
        strokeLinecap="round"
        strokeDasharray={dashArray}
      />
    </g>
  );
}
