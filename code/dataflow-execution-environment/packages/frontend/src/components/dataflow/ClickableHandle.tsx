import { useEffect, useRef, useState } from "react";
import { Handle, Position } from "@xyflow/react";
import { useNode } from "@/contexts/NodeContext";

const HANDLE_COOLDOWN_MS = 1000;

type ClickableHandleProps = {
  type: "source" | "target";
  position: Position;
  id: string;
  nodeId: string;
  className?: string;
  style?: React.CSSProperties;
};

export function ClickableHandle({
  type,
  position,
  id,
  nodeId,
  className = "",
  style,
}: ClickableHandleProps) {
  const { isPortSelected, handlePortClick } = useNode();

  const selected = isPortSelected(nodeId, id, type);

  const [coolingDown, setCoolingDown] = useState(false);
  const cooldownTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (cooldownTimerRef.current !== null) {
        clearTimeout(cooldownTimerRef.current);
      }
    };
  }, []);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (coolingDown) return;
    handlePortClick(nodeId, id, type);
    setCoolingDown(true);
    if (cooldownTimerRef.current !== null) {
      clearTimeout(cooldownTimerRef.current);
    }
    cooldownTimerRef.current = setTimeout(() => {
      setCoolingDown(false);
      cooldownTimerRef.current = null;
    }, HANDLE_COOLDOWN_MS);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (coolingDown) {
      e.preventDefault();
    }
  };

  const colorClass = selected ? "!bg-green-500 !border-green-300" : "!bg-white !border-slate-400";
  const cooldownClass = coolingDown
    ? "!opacity-40 pointer-events-none cursor-not-allowed"
    : "cursor-pointer";

  return (
    <Handle
      type={type}
      position={position}
      id={id}
      className={`!w-20 !h-20 !border-2 ${colorClass} ${cooldownClass} ${className}`}
      style={style}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
    />
  );
}
