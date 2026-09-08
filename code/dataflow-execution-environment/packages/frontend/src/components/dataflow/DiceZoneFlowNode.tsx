import { useEffect, useRef } from 'react';
import { useReactFlow, Position } from '@xyflow/react';
import type { Node, NodeProps } from '@xyflow/react';
import type { VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { DiceFace } from './source-flow/DiceFace';
import { ClickableHandle } from './ClickableHandle';
import { useNode } from '@/contexts/NodeContext';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';

export type DiceZoneFlowNodeData = VisionNodeMeta & {
  /** Discriminator so dataForProgramHash can keep the rolled value in the program hash. */
  readonly nodekind: 'diceZone';
  value?: number;
};

export type DiceZoneFlowNode = Node<DiceZoneFlowNodeData, 'diceZone'>;

function randomDiceFace(): number {
  return Math.floor(Math.random() * 6) + 1;
}

// The circle is 520px diameter. The vision position offset is VISION_NODE_HALF_W=48, VISION_NODE_HALF_H=40.
// To center the circle on the physical card: translate = -(260 - 48, 260 - 40) = (-212, -220).
const ZONE_SIZE = 520;

/** The dice area — a large circle that persists while connected or while the physical card is present. */
export function DiceZoneFlowNodeComponent({ id, data }: NodeProps<DiceZoneFlowNode>) {
  const { setNodes } = useReactFlow();
  const { registerPortKind, unregisterPortKinds } = useNode();
  const shellClass = useFlowNodeShellClass();
  const prevVisionStatusRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    registerPortKind(id, 'out', { produces: 'rational' });
    return () => unregisterPortKinds(id);
  }, [id, registerPortKind, unregisterPortKinds]);

  // Assign a new value the moment the physical dice card enters (or re-enters) the area.
  useEffect(() => {
    const prevStatus = prevVisionStatusRef.current;
    const curStatus = data.visionStatus;
    const entered = curStatus === 'active' && prevStatus !== 'active';

    if (entered) {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, value: randomDiceFace() } } : n
        )
      );
    }

    prevVisionStatusRef.current = curStatus;
  }, [data.visionStatus, id, setNodes]);

  const dicePresent =
    data.visionStatus === 'active' || data.visionStatus === undefined;

  return (
    <div
      className={`relative -translate-x-53 -translate-y-55 ${shellClass}`}
      style={{ width: ZONE_SIZE, height: ZONE_SIZE }}
    >
      {/* Circle border */}
      <div className="absolute inset-0 rounded-full border-4 border-amber-400/60 bg-amber-500/10" />

      {/* Dice card — visible when physical card is inside the area */}
      {dicePresent ? (
        <DiceCard value={data.value} />
      ) : (
        <EmptyHint />
      )}

      {/* Output handle — always visible, at the right edge of the circle */}
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

function DiceCard({ value }: { value?: number }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 pointer-events-none">
      {/* key=value remounts DiceFace on each new value, triggering the pop animation */}
      <div key={value} className="scale-[1.8] origin-center animate-dice-pop">
        <DiceFace value={value} />
      </div>
      <div className="rounded-xl border-2 border-amber-500/40 bg-slate-800/90 px-5 py-2 text-center shadow-xl mt-10">
        <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
          Dado
        </p>
        <p className="text-4xl font-black tabular-nums text-slate-100">
          {value ?? '—'}
        </p>
      </div>
    </div>
  );
}

function EmptyHint() {
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      <div className="rounded-2xl border-2 border-dashed border-amber-500/25 px-8 py-6 text-center">
        <p className="text-5xl opacity-20">🎲</p>
        <p className="mt-2 text-xs font-semibold uppercase tracking-wider text-amber-500/35">
          Coloca el dado
        </p>
      </div>
    </div>
  );
}
