import { useEffect, useRef, useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import { FlowNodeCard } from '../FlowNodeCard';
import type { SourceFlowNodeData } from '../SourceFlowNode';
import { DiceFace } from './DiceFace';
import { useNode } from '@/contexts/NodeContext';

type DiceSourceData = Extract<SourceFlowNodeData, { variant: 'dice' }>;

const ROLL_DURATION_MS = 1000;

function randomDiceFace(): number {
  return Math.floor(Math.random() * 6) + 1;
}

export function DiceSourceFlowNode({
  data,
  nodeId,
}: {
  data: DiceSourceData;
  nodeId: string;
}) {
  const { setNodes } = useReactFlow();
  const { isPortOccupied } = useNode();
  const [spinning, setSpinning] = useState(false);
  const rollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isConnected = isPortOccupied(nodeId, 'out', 'source');

  const prevIsConnectedRef = useRef(false);
  const prevVisionStatusRef = useRef<string | undefined>(undefined);

  useEffect(
    () => () => {
      if (rollTimerRef.current) clearTimeout(rollTimerRef.current);
    },
    []
  );

  useEffect(() => {
    const prevConnected = prevIsConnectedRef.current;
    const curConnected = isConnected;
    const prevStatus = prevVisionStatusRef.current;
    const curStatus = data.visionStatus;

    let shouldRoll = false;

    if (curConnected && !prevConnected) {
      shouldRoll = true;
    }

    // Card re-entered view (stale/lost → active) while connected
    if (
      curConnected &&
      curStatus === 'active' &&
      prevStatus !== undefined &&
      prevStatus !== 'active'
    ) {
      shouldRoll = true;
    }

    if (shouldRoll) {
      if (rollTimerRef.current) clearTimeout(rollTimerRef.current);
      const finalValue = randomDiceFace();
      setSpinning(true);
      rollTimerRef.current = setTimeout(() => {
        rollTimerRef.current = null;
        setSpinning(false);
        setNodes((nds) =>
          nds.map((n) =>
            n.id === nodeId
              ? { ...n, data: { ...n.data, value: finalValue } }
              : n
          )
        );
      }, ROLL_DURATION_MS);
    }

    prevIsConnectedRef.current = curConnected;
    prevVisionStatusRef.current = curStatus;
  }, [isConnected, data.visionStatus, nodeId, setNodes]);

  return (
    <>
      {isConnected && (
        <div
          className="pointer-events-none absolute rounded-full border-4 border-amber-400/60 bg-amber-500/10"
          style={{
            width: '520px',
            height: '520px',
            left: '30%',
            top: '55%',
            transform: 'translate(-50%, -50%)',
          }}
        />
      )}
      <div className="flex flex-col items-center gap-2 translate-y-10">
        <DiceFace value={data.value} spinning={spinning} />
        <FlowNodeCard
          family="input"
          cardCategory="dice"
          title="Dado"
          content={
            <span className="text-3xl font-black tabular-nums text-slate-100">
              {data.value ?? '—'}
            </span>
          }
        />
      </div>
    </>
  );
}
