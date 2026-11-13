import { useCallback, useEffect, useRef, useState } from 'react';
import { useReactFlow, Position } from '@xyflow/react';
import type { Node, NodeProps } from '@xyflow/react';
import type { VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { DiceFace } from './source-flow/DiceFace';
import { ClickableHandle } from './ClickableHandle';
import { useNode } from '@/contexts/NodeContext';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';
import { FLOW_NODE_INTERACTIVE_CLASS } from './flowNodeChrome';

export type DiceZoneFlowNodeData = VisionNodeMeta & {
  /** Discriminante del nodo dentro de la unión de tipos del lienzo. */
  readonly nodekind: 'diceZone';
  value?: number;
};

export type DiceZoneFlowNode = Node<DiceZoneFlowNodeData, 'diceZone'>;

/** Cuánto gira el dado antes de quedarse en una cara. */
const ROLL_MS = 700;
/** Cada cuánto cambia la cara mientras gira. */
const ROLL_TICK_MS = 90;

function randomDiceFace(): number {
  return Math.floor(Math.random() * 6) + 1;
}

/**
 * El dado es una carta más: el nodo ocupa lo mismo que la carta física
 * (VISION_NODE_HALF_W/H) y el dado, con su botón para tirarlo, flota encima.
 */
export function DiceZoneFlowNodeComponent({ id, data }: NodeProps<DiceZoneFlowNode>) {
  const { setNodes } = useReactFlow();
  const { registerPortKind, unregisterPortKinds } = useNode();
  const shellClass = useFlowNodeShellClass();
  const prevVisionStatusRef = useRef<string | undefined>(undefined);
  const timersRef = useRef<{ tick?: number; stop?: number }>({});
  // Cara que se ve mientras gira; solo el valor final llega a los datos del nodo,
  // para no reejecutar el programa en cada vuelta.
  const [rollingFace, setRollingFace] = useState<number | null>(null);

  useEffect(() => {
    registerPortKind(id, 'out', { produces: 'rational' });
    return () => unregisterPortKinds(id);
  }, [id, registerPortKind, unregisterPortKinds]);

  const clearTimers = useCallback(() => {
    window.clearInterval(timersRef.current.tick);
    window.clearTimeout(timersRef.current.stop);
    timersRef.current = {};
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  const roll = useCallback(() => {
    clearTimers();
    setRollingFace(randomDiceFace());
    timersRef.current.tick = window.setInterval(
      () => setRollingFace(randomDiceFace()),
      ROLL_TICK_MS
    );
    timersRef.current.stop = window.setTimeout(() => {
      clearTimers();
      setRollingFace(null);
      const value = randomDiceFace();
      setNodes((nds) =>
        nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, value } } : n))
      );
    }, ROLL_MS);
  }, [clearTimers, id, setNodes]);

  // Se tira solo en cuanto la carta aparece (o vuelve a aparecer).
  useEffect(() => {
    const prevStatus = prevVisionStatusRef.current;
    const curStatus = data.visionStatus;
    if (curStatus === 'active' && prevStatus !== 'active') roll();
    prevVisionStatusRef.current = curStatus;
  }, [data.visionStatus, roll]);

  const rolling = rollingFace !== null;

  return (
    <div className={`relative h-20 w-24 ${shellClass}`}>
      <div className="absolute bottom-full left-1/2 mb-10 flex -translate-x-1/2 flex-col items-center gap-4 opacity-40">
        {/* scale no ocupa espacio en el layout: la caja fija reserva lo que mide el dado escalado */}
        <div className="flex h-36 w-36 items-center justify-center">
          <div key={rolling ? 'rolling' : data.value} className="scale-[2.2] animate-dice-pop">
            <DiceFace value={rollingFace ?? data.value} spinning={rolling} />
          </div>
        </div>
        <button
          type="button"
          disabled={rolling}
          onClick={roll}
          className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} rounded-lg border-2 border-amber-600 bg-amber-500 px-5 py-2 text-lg font-bold text-white shadow transition-colors hover:bg-amber-600 disabled:cursor-not-allowed`}
        >
          Tirar
        </button>
      </div>

      <ClickableHandle
        type="source"
        position={Position.Right}
        id="out"
        nodeId={id}
        handleVariant="input-out"
        produces="rational"
      />
    </div>
  );
}
