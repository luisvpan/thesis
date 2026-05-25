import { getBezierPath, type EdgeProps } from '@xyflow/react';
import { FLOW_HANDLE_SIZE } from '@/utils/arrayZoneGeometry';

const MIN_RECT_DIM = 8;
const HANDLE_HALF = FLOW_HANDLE_SIZE / 2;

export function ArrayZoneEdge({
  id: _id,
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

  const rectX = Math.min(sourceX, targetX) - HANDLE_HALF;
  const rectY = Math.min(sourceY, targetY) - HANDLE_HALF;
  const rawW = Math.abs(targetX - sourceX) + FLOW_HANDLE_SIZE;
  const rawH = Math.abs(targetY - sourceY) + FLOW_HANDLE_SIZE;
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
      <path
        d={edgePath}
        stroke="#2dd4bf"
        strokeWidth={8}
        fill="none"
        strokeOpacity={0.12}
        strokeLinecap="round"
      />
    </g>
  );
}
