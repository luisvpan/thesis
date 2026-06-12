import { useState } from "react";
import { Handle, Position } from "@xyflow/react";
import { useNode } from "@/contexts/NodeContext";
import { type HandleKind } from "./handle-kinds";
import { FLOW_NODE_INTERACTIVE_CLASS } from "./flowNodeChrome";
import {
  type FlowHandleVariant,
  getHandleVariantShapeClass,
} from "./flowHandleVariants";
import type { PortHighlightState } from "./connectionRules";

type ClickableHandleProps = {
  type: "source" | "target";
  position: Position;
  id: string;
  nodeId: string;
  className?: string;
  style?: React.CSSProperties;
  disabled?: boolean;
  handleVariant?: FlowHandleVariant;
  produces?: HandleKind;
  accepts?: HandleKind[];
};

function getShapeClassFromKind(kind: HandleKind | undefined): string {
  switch (kind) {
    case "rational":
      return "!rounded-full !rounded-md";
    case "cpa":
      return "!rounded-md";
    case "keyword":
      return "!rounded-full !rounded-md !w-26 translate-x-[-20%]";
    case "group":
      return "!rounded-sm !border-2 !border-dashed";
    case "any":
    default:
      return "!rounded-full !rounded-md";
  }
}

function getEffectiveKind(
  type: "source" | "target",
  produces?: HandleKind,
  accepts?: HandleKind[]
): HandleKind {
  if (type === "source") {
    return produces ?? "any";
  }
  if (accepts && accepts.length === 1 && accepts[0] !== "any") {
    return accepts[0];
  }
  return "any";
}

function highlightColorClass(state: PortHighlightState): string {
  switch (state) {
    case "connected":
      return "!bg-blue-600/50 !border-blue-400/50";
    case "selected":
      return "!bg-green-700/50 !border-green-500/50";
    case "compatible":
      return "!bg-green-500/50 !border-green-400/50 ring-2 ring-green-400/50";
    case "incompatible":
      return "!bg-red-500/80 !border-red-400/80";
    case "idle":
    default:
      return "!bg-slate-600/50 !border-slate-500/50";
  }
}

export function ClickableHandle({
  type,
  position,
  id,
  nodeId,
  className = "",
  style,
  disabled = false,
  handleVariant,
  produces,
  accepts,
}: ClickableHandleProps) {
  const {
    isPortSelected,
    handlePortClick,
    isNodeInsideArrayZone,
    shakingPort,
    getPortHighlightState,
    isPortOccupied,
    disconnectPort,
  } = useNode();
  const [isCooldown, setIsCooldown] = useState(false);

  const occupied = isPortOccupied(nodeId, id, type);
  const selected = !disabled && !occupied && isPortSelected(nodeId, id, type);
  const hideInArrayZone = isNodeInsideArrayZone(nodeId);

  const variantShapeClass = getHandleVariantShapeClass(handleVariant);
  const useKindShape = !handleVariant || handleVariant === "input-out";
  const effectiveKind = getEffectiveKind(type, produces, accepts);
  const kindShapeClass = useKindShape ? getShapeClassFromKind(effectiveKind) : "";
  const shapeClass = variantShapeClass || kindShapeClass;

  const isShaking = shakingPort?.nodeId === nodeId && shakingPort?.handleId === id;

  const highlightState = getPortHighlightState(nodeId, id, type);
  const colorClass = highlightColorClass(
    selected ? "selected" : highlightState
  );

  const handleClick = (e: React.MouseEvent) => {
    if (disabled) return;
    if (isCooldown) return;

    e.stopPropagation();
    e.preventDefault();

    if (occupied) {
      disconnectPort(nodeId, id, type);
    } else {
      handlePortClick(nodeId, id, type);
    }

    setIsCooldown(true);
    setTimeout(() => setIsCooldown(false), 500);
  };

  const cooldownClass = isCooldown ? "opacity-50" : "";
  const shakeClass = isShaking ? "animate-shake" : "";

  return (
    <Handle
      type={type}
      position={position}
      id={id}
      className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} !h-20 !w-20 !border-2 ${shapeClass} ${colorClass} ${cooldownClass} ${shakeClass} ${
        hideInArrayZone ? "invisible! pointer-events-none!" : ""
      } ${
        disabled
          ? "pointer-events-none! cursor-not-allowed opacity-35"
          : "cursor-pointer"
      } ${className}`}
      style={style}
      onClick={handleClick}
    />
  );
}
