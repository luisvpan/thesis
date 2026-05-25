import { useMemo, useRef } from 'react';
import { getBezierPath, type EdgeProps } from '@xyflow/react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { EdgeFlowWalker } from './EdgeFlowWalker';
import { hasEdgeSourceMiniToken } from './EdgeSourceMiniToken';

function isArrayZoneEdge(
  sourceHandle: string | null | undefined,
  targetHandle: string | null | undefined
): boolean {
  return sourceHandle === 'zone-out' || targetHandle === 'zone-in';
}

export function DataflowEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  sourceHandle,
  targetHandle,
}: EdgeProps) {
  const pathRef = useRef<SVGPathElement>(null);
  const { showFlowResults, viewMode } = useResultCardUi();
  const { nodes, evalResults } = useNode();

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const { sourceNode, showWalker } = useMemo(() => {
    const src = nodes.find((n) => n.id === source);
    const tgt = nodes.find((n) => n.id === target);
    if (!showFlowResults || !src || !tgt || isArrayZoneEdge(sourceHandle, targetHandle)) {
      return { sourceNode: src, showWalker: false };
    }
    const dataNodeTypes = new Set([
      'source',
      'operator',
      'arrayClose',
      'programOutput',
    ]);
    const walker =
      dataNodeTypes.has(src.type) &&
      hasEdgeSourceMiniToken(src, evalResults) &&
      (tgt.type === 'operator' ||
        tgt.type === 'programOutput' ||
        src.type === 'operator' ||
        src.type === 'programOutput');
    return { sourceNode: src, showWalker: walker };
  }, [nodes, source, target, showFlowResults, sourceHandle, targetHandle, evalResults]);

  return (
    <>
      <g>
        <path
          d={edgePath}
          stroke="#4ade80"
          strokeWidth={10}
          fill="none"
          strokeOpacity={0.08}
          strokeLinecap="round"
        />
        <path
          d={edgePath}
          stroke="#4ade80"
          strokeWidth={5}
          fill="none"
          strokeOpacity={0.18}
          strokeLinecap="round"
        />
        <path
          ref={pathRef}
          id={id}
          d={edgePath}
          stroke="#4ade80"
          strokeWidth={2.5}
          fill="none"
          strokeLinecap="round"
        />
      </g>
      {showWalker && sourceNode ? (
        <EdgeFlowWalker pathRef={pathRef} sourceNode={sourceNode} viewMode={viewMode} />
      ) : null}
    </>
  );
}
