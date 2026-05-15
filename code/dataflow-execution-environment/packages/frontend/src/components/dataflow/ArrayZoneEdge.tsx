import { getBezierPath, type EdgeProps } from '@xyflow/react';

const MIN_RECT_DIM = 8;

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

  const rawW = Math.abs(targetX - sourceX);
  const rawH = Math.abs(targetY - sourceY);
  const rectX = Math.min(sourceX, targetX);
  const rectY = Math.min(sourceY, targetY);
  const rectW = Math.max(rawW, MIN_RECT_DIM);
  const rectH = Math.max(rawH, MIN_RECT_DIM);
  const shiftX = rawW < MIN_RECT_DIM ? (MIN_RECT_DIM - rawW) / 2 : 0;
  const shiftY = rawH < MIN_RECT_DIM ? (MIN_RECT_DIM - rawH) / 2 : 0;

  return (
    <g>
      <rect
        x={rectX - shiftX}
        y={rectY - shiftY}
        width={rectW}
        height={rectH}
        rx={6}
        ry={6}
        
        fillOpacity={0.08}
        stroke="#2dd4bf"
        strokeWidth={1.5}
        strokeOpacity={0.45}
      />
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
    
    </g>
  );
}
