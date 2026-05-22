import { useMemo, useRef } from 'react';
import { getBezierPath, type EdgeProps } from '@xyflow/react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { EdgeFlowWalker } from './EdgeFlowWalker';

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
}: EdgeProps) {
  const pathRef = useRef<SVGPathElement>(null);
  const { showFlowResults, viewMode } = useResultCardUi();
  const { nodes } = useNode();

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
    const walker =
      showFlowResults &&
      src != null &&
      tgt != null &&
      (tgt.type === 'operator' || tgt.type === 'programOutput');
    return { sourceNode: src, showWalker: walker };
  }, [nodes, source, target, showFlowResults]);

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
