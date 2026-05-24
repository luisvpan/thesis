import { useEffect, useRef, useState, type RefObject } from 'react';
import { EdgeLabelRenderer } from '@xyflow/react';
import type { DataflowNode } from '@/contexts/node/types';
import type { ResultViewMode } from './dataflowResultCpa';
import { EdgeSourceMiniToken } from './EdgeSourceMiniToken';
import {
  DEFAULT_EDGE_WALK_DURATION_MS,
  isWalkerPathReady,
  loopEdgeProgress,
  samplePathPoint,
} from '@/utils/edgePathWalker';

type EdgeFlowWalkerProps = {
  pathRef: RefObject<SVGPathElement | null>;
  sourceNode: DataflowNode;
  viewMode: ResultViewMode;
};

/**
 * Token que recorre el conector. No usa (0,0) como posición visible: el path debe
 * estar listo antes de mostrar el contenido, y la posición se actualiza por ref (sin
 * setState por frame) para evitar parpadeos.
 */
export function EdgeFlowWalker({ pathRef, sourceNode, viewMode }: EdgeFlowWalkerProps) {
  const shellRef = useRef<HTMLDivElement>(null);
  const [showToken, setShowToken] = useState(false);

  useEffect(() => {
    let raf = 0;
    let revealed = false;

    const tick = () => {
      const path = pathRef.current;
      const shell = shellRef.current;
      if (path && shell && isWalkerPathReady(path)) {
        const t = loopEdgeProgress(performance.now(), DEFAULT_EDGE_WALK_DURATION_MS);
        const { x, y } = samplePathPoint(path, t);
        shell.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
        if (!revealed) {
          revealed = true;
          setShowToken(true);
        }
      }
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pathRef]);

  return (
    <EdgeLabelRenderer>
      <div
        ref={shellRef}
        className="nodrag nopan pointer-events-none"
        style={{
          position: 'absolute',
          visibility: showToken ? 'visible' : 'hidden',
          transform: 'translate(-9999px, -9999px) translate(-50%, -50%)',
        }}
      >
        {showToken ? (
          <EdgeSourceMiniToken node={sourceNode} viewMode={viewMode} />
        ) : null}
      </div>
    </EdgeLabelRenderer>
  );
}
