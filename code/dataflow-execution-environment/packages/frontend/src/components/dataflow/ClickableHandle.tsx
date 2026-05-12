import { useState } from "react";
import { Handle, Position } from "@xyflow/react";
import { useNode } from "@/contexts/NodeContext";

type ClickableHandleProps = {
  type: "source" | "target";
  position: Position;
  id: string;
  nodeId: string;
  className?: string;
  style?: React.CSSProperties;
  /** Sin conexiones ni selección de puerto (p. ej. carta incompatible con el modo CPA). */
  disabled?: boolean;
};

export function ClickableHandle({
  type,
  position,
  id,
  nodeId,
  className = "",
  style,
  disabled = false,
}: ClickableHandleProps) {
  const { isPortSelected, handlePortClick, isNodeInsideArrayZone } = useNode();
  const [isCooldown, setIsCooldown] = useState(false);

  const selected = !disabled && isPortSelected(nodeId, id, type);
  const hideInArrayZone = isNodeInsideArrayZone(nodeId);

  const handleClick = (e: React.MouseEvent) => {
    if (disabled) return;
    if (isCooldown) return; // Ignorar durante cooldown

    e.stopPropagation();
    e.preventDefault();
    handlePortClick(nodeId, id, type);

    // Activar cooldown para evitar toques accidentales
    setIsCooldown(true);
    setTimeout(() => setIsCooldown(false), 500);
  };

  // Colores más oscuros para no afectar detección de CV
  const colorClass = selected
    ? "!bg-green-700 !border-green-500"
    : "!bg-slate-600 !border-slate-500";

  // Reducir opacidad durante cooldown
  const cooldownClass = isCooldown ? "opacity-50" : "";

  return (
    <Handle
      type={type}
      position={position}
      id={id}
      className={`nodrag nopan !h-20 !w-20 !border-2 ${colorClass} ${cooldownClass} ${
        hideInArrayZone ? "!invisible !pointer-events-none" : ""
      } ${
        disabled ? "pointer-events-none cursor-not-allowed opacity-35" : "cursor-pointer"
      } ${className}`}
      style={style}
      onClick={handleClick}
    />
  );
}
