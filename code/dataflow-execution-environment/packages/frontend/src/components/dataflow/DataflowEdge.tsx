import { getBezierPath, type EdgeProps } from '@xyflow/react';

export function DataflowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

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
      />
      {/* Glow inner */}
      <path
        d={edgePath}
        stroke="#4ade80"
        strokeWidth={5}
        fill="none"
        strokeOpacity={0.18}
        strokeLinecap="round"
      />
      {/* Main line */}
      <path
        id={id}
        d={edgePath}
        stroke="#4ade80"
        strokeWidth={2.5}
        fill="none"
        strokeLinecap="round"
      />
    </g>
  );
}
