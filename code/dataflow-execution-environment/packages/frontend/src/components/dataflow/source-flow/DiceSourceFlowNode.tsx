import { useCallback, useEffect, useRef, useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import { FlowNodeCard } from '../FlowNodeCard';
import { FLOW_NODE_INTERACTIVE_CLASS } from '../flowNodeChrome';
import type { SourceFlowNodeData } from '../SourceFlowNode';
import { DiceFace } from './DiceFace';

type DiceSourceData = Extract<SourceFlowNodeData, { variant: 'dice' }>;

const ROLL_DURATION_MS = 1500;

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
  const isRollingRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [spinning, setSpinning] = useState(false);

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      isRollingRef.current = false;
    },
    []
  );

  const rollDice = useCallback(() => {
    if (isRollingRef.current) return;
    isRollingRef.current = true;

    if (timerRef.current) clearTimeout(timerRef.current);
    const finalValue = randomDiceFace();

    setSpinning(true);

    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      isRollingRef.current = false;
      setSpinning(false);
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, value: finalValue } }
            : n
        )
      );
    }, ROLL_DURATION_MS);
  }, [nodeId, setNodes]);

  return (
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
      <button
        type="button"
        onClick={rollDice}
        disabled={spinning}
        className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} mt-44 flex h-20 w-20 opacity-70 items-center justify-center rounded-lg border-2 border-amber-500 bg-amber-600 text-base font-black uppercase tracking-wide text-white shadow transition-colors hover:bg-amber-500 disabled:cursor-wait disabled:opacity-70`}
      >
        Lanzar
      </button>
    </div>
  );
}
