import { useEffect, useState, type RefObject } from 'react';
import { EdgeLabelRenderer } from '@xyflow/react';
import type { DataflowNode } from '@/contexts/node/types';
import type { ResultViewMode } from './dataflowResultCpa';
import { EdgeSourceMiniToken } from './EdgeSourceMiniToken';
import {
  DEFAULT_EDGE_WALK_DURATION_MS,
  loopEdgeProgress,
  samplePathPoint,
} from '@/utils/edgePathWalker';

type EdgeFlowWalkerProps = {
  pathRef: RefObject<SVGPathElement | null>;
  sourceNode: DataflowNode;
  viewMode: ResultViewMode;
};

export function EdgeFlowWalker({ pathRef, sourceNode, viewMode }: EdgeFlowWalkerProps) {
  const [pos, setPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const path = pathRef.current;
      if (path) {
        const t = loopEdgeProgress(performance.now(), DEFAULT_EDGE_WALK_DURATION_MS);
        setPos(samplePathPoint(path, t));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pathRef]);

  return (
    <EdgeLabelRenderer>
      <div
        className="nodrag nopan pointer-events-none"
        style={{
          position: 'absolute',
          transform: `translate(${pos.x}px, ${pos.y}px) translate(-50%, -50%)`,
        }}
      >
        <EdgeSourceMiniToken node={sourceNode} viewMode={viewMode} />
      </div>
    </EdgeLabelRenderer>
  );
}
