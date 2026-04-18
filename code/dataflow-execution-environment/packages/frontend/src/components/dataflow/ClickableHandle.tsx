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

  const selected = isPortSelected(nodeId, id, type);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    handlePortClick(nodeId, id, type);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const colorClass = selected ? "!bg-green-500 !border-green-300" : "!bg-white !border-slate-400";

  return (
    <Handle
      type={type}
      position={position}
      id={id}
      className={`!w-15 !h-15 !border-2 ${colorClass} cursor-pointer ${className}`}
      style={style}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
    />
  );
}
