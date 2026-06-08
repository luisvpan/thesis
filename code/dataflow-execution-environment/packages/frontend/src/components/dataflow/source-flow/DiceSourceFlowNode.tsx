import { useCallback, useEffect, useRef, useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import { FlowNodeCard } from '../FlowNodeCard';
import { FLOW_NODE_INTERACTIVE_CLASS } from '../flowNodeChrome';
import type { SourceFlowNodeData } from '../SourceFlowNode';
import { DiceFace } from './DiceFace';

type DiceSourceData = Extract<SourceFlowNodeData, { variant: 'dice' }>;

const ROLL_DURATION_MS = 900;
const ROLL_TICK_MS = 70;

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
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const finishRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Local rolling state: not stored in node data so nothing external can interfere.
  const isRollingRef = useRef(false);
  const [animating, setAnimating] = useState(false);

  const clearTimers = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    if (finishRef.current) {
      clearTimeout(finishRef.current);
      finishRef.current = null;
    }
  }, []);

  useEffect(() => () => {
    clearTimers();
    isRollingRef.current = false;
  }, [clearTimers]);

  // Safety net: if animating is still true after the expected roll duration, force-reset.
  // Guards against edge cases where the finish timeout is killed (e.g. HMR effect
  // cleanup firing mid-roll), which would leave the spinner running forever.
  useEffect(() => {
    if (!animating) return;
    const safety = setTimeout(() => {
      isRollingRef.current = false;
      setAnimating(false);
      clearTimers();
    }, ROLL_DURATION_MS + 200);
    return () => clearTimeout(safety);
  }, [animating, clearTimers]);

  const rollDice = useCallback(() => {
    if (isRollingRef.current) return;
    isRollingRef.current = true;
    setAnimating(true);

    clearTimers();
    const finalValue = randomDiceFace();

    setNodes((nds) =>
      nds.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, previewFace: randomDiceFace() } }
          : n
      )
    );

    tickRef.current = setInterval(() => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, previewFace: randomDiceFace() } }
            : n
        )
      );
    }, ROLL_TICK_MS);

    finishRef.current = setTimeout(() => {
      clearTimers();
      isRollingRef.current = false;
      setAnimating(false);
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? {
                ...n,
                data: {
                  ...n.data,
                  value: finalValue,
                  previewFace: finalValue,
                },
              }
            : n
        )
      );
    }, ROLL_DURATION_MS);
  }, [clearTimers, nodeId, setNodes]);

  const displayFace = animating
    ? data.previewFace
    : data.value ?? data.previewFace;

  return (
    <div className="flex flex-col items-center gap-2 translate-y-10">
      <DiceFace value={displayFace} spinning={animating} />
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
        disabled={animating}
        className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} mt-44 flex h-20 w-20 opacity-70 items-center justify-center rounded-lg border-2 border-amber-500 bg-amber-600 text-base font-black uppercase tracking-wide text-white shadow transition-colors hover:bg-amber-500 disabled:cursor-wait disabled:opacity-70`}
      >
        Lanzar
      </button>
    </div>
  );
}
