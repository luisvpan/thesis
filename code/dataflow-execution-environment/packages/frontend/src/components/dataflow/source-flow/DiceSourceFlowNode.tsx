import { useState, useCallback, useEffect, useRef } from 'react';
import { useReactFlow } from '@xyflow/react';
import type { SourceFlowNodeData } from '../SourceFlowNode';

type DiceSourceData = Extract<SourceFlowNodeData, { variant: 'dice' }>;

/** Dot positions [cx%, cy%] for each face value (1–6). Standard dice layout. */
const DICE_DOTS: Record<number, [number, number][]> = {
  1: [[50, 50]],
  2: [[28, 28], [72, 72]],
  3: [[28, 28], [50, 50], [72, 72]],
  4: [[28, 28], [72, 28], [28, 72], [72, 72]],
  5: [[28, 28], [72, 28], [50, 50], [28, 72], [72, 72]],
  6: [[28, 28], [72, 28], [28, 50], [72, 50], [28, 72], [72, 72]],
};

function DiceFace({ value, spinning }: { value: number; spinning: boolean }) {
  const dots = DICE_DOTS[value] ?? DICE_DOTS[1];
  return (
    <div
      className={`relative w-16 h-16 rounded-xl bg-white border-2 border-slate-300 shadow-md flex items-center justify-center ${
        spinning ? 'animate-spin' : ''
      }`}
      style={{ animationDuration: spinning ? '0.15s' : undefined }}
    >
      <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full p-2">
        {dots.map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r={9} fill="#1e293b" />
        ))}
      </svg>
    </div>
  );
}

export function DiceSourceFlowNode({
  data,
  nodeId,
}: {
  data: DiceSourceData;
  nodeId: string;
}) {
  const { setNodes } = useReactFlow();
  const [displayValue, setDisplayValue] = useState(data.diceValue);
  const [isRolling, setIsRolling] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Keep display in sync if node data changes externally
  useEffect(() => {
    if (!isRolling) setDisplayValue(data.diceValue);
  }, [data.diceValue, isRolling]);

  const roll = useCallback(() => {
    if (isRolling) return;
    const newValue = Math.floor(Math.random() * 6) + 1;
    setIsRolling(true);

    let ticks = 0;
    intervalRef.current = setInterval(() => {
      setDisplayValue(Math.floor(Math.random() * 6) + 1);
      ticks++;
      if (ticks >= 12) {
        clearInterval(intervalRef.current!);
        intervalRef.current = null;
        setDisplayValue(newValue);
        setIsRolling(false);
        setNodes((nds) =>
          nds.map((n) =>
            n.id === nodeId ? { ...n, data: { ...n.data, diceValue: newValue } } : n
          )
        );
      }
    }, 80);
  }, [isRolling, nodeId, setNodes]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-3 p-3 text-white">
      <span className="text-xs font-semibold uppercase tracking-wide text-slate-300">
        dado
      </span>
      <DiceFace value={displayValue} spinning={isRolling} />
      <span className="text-2xl font-black text-white">{displayValue}</span>
      <button
        type="button"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={roll}
        disabled={isRolling}
        className="mt-1 px-3 py-1 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors shadow"
      >
        {isRolling ? 'Lanzando…' : 'Lanzar dado'}
      </button>
    </div>
  );
}
