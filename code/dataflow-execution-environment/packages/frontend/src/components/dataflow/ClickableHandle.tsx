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
  const [isCooldown, setIsCooldown] = useState(false);

  const selected = isPortSelected(nodeId, id, type);

  const handleClick = (e: React.MouseEvent) => {
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
      className={`nodrag nopan !h-20 !w-20 !border-2 ${colorClass} ${cooldownClass} cursor-pointer ${className}`}
      style={style}
      onClick={handleClick}
    />
  );
}
