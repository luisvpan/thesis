import { getBezierPath, type EdgeProps } from '@xyflow/react';

export function ArrayZoneEdge({
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
      {/* Glow */}
      <path
        d={edgePath}
        stroke="#2dd4bf"
        strokeWidth={8}
        fill="none"
        strokeOpacity={0.12}
        strokeLinecap="round"
      />
      {/* Dashed zone link */}
      <path
        id={id}
        d={edgePath}
        stroke="#2dd4bf"
        strokeWidth={2}
        fill="none"
        strokeDasharray="8 5"
        strokeLinecap="round"
      />
    </g>
  );
}
