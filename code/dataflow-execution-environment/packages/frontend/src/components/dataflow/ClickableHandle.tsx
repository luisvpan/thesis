import { useState } from "react";
import { Handle, Position } from "@xyflow/react";
import { useNode } from "@/contexts/NodeContext";
import { type HandleKind, acceptsConnection } from "./handle-kinds";

type ClickableHandleProps = {
  type: "source" | "target";
  position: Position;
  id: string;
  nodeId: string;
  className?: string;
  style?: React.CSSProperties;
  /** Sin conexiones ni selección de puerto (reservado para uso interno o futuro). */
  disabled?: boolean;
  /** For source handles: what kind of data this handle produces */
  produces?: HandleKind;
  /** For target handles: what kinds of data this handle accepts */
  accepts?: HandleKind[];
};

/**
 * Get the shape class based on the handle kind.
 * - "rational" → circle (rounded-full)
 * - "cpa" → square with soft corners (rounded-md)
 * - "keyword" → horizontal pill (wider than tall)
 * - "any" or multiple → circle (default)
 */
function getShapeClass(kind: HandleKind | undefined): string {
  switch (kind) {
    case "rational":
      return "!rounded-full";
    case "cpa":
      return "!rounded-md";
    case "keyword":
      return "!rounded-full !w-26 translate-x-[-20%]"; // pill: wider than the default square
    case "any":
    default:
      return "!rounded-full";
  }
}

/**
 * Determine the effective kind for shape purposes.
 * For targets with multiple accepts, use the first specific one or "any".
 */
function getEffectiveKind(
  type: "source" | "target",
  produces?: HandleKind,
  accepts?: HandleKind[]
): HandleKind {
  if (type === "source") {
    return produces ?? "any";
  }
  // For targets, if accepts a single specific kind, use that for shape
  if (accepts && accepts.length === 1 && accepts[0] !== "any") {
    return accepts[0];
  }
  return "any";
}

export function ClickableHandle({
  type,
  position,
  id,
  nodeId,
  className = "",
  style,
  disabled = false,
  produces,
  accepts,
}: ClickableHandleProps) {
  const {
    isPortSelected,
    handlePortClick,
    isNodeInsideArrayZone,
    selectedPort,
    getPortKindInfo,
    shakingPort,
  } = useNode();
  const [isCooldown, setIsCooldown] = useState(false);

  const selected = !disabled && isPortSelected(nodeId, id, type);
  const hideInArrayZone = isNodeInsideArrayZone(nodeId);

  // Determine shape based on kind
  const effectiveKind = getEffectiveKind(type, produces, accepts);
  const shapeClass = getShapeClass(effectiveKind);

  // Check if this handle is shaking (incompatible connection attempted)
  const isShaking = shakingPort?.nodeId === nodeId && shakingPort?.handleId === id;

  // Calculate compatibility feedback when there's a selected port
  let compatibilityClass = "";
  if (selectedPort && !disabled && !selected) {
    const isOppositeType = selectedPort.handleType !== type;

    if (isOppositeType) {
      // Get the selected port's kind info
      const selectedInfo = getPortKindInfo(selectedPort.nodeId, selectedPort.handleId);

      let isCompatible = false;
      if (selectedPort.handleType === "source") {
        // Selected is source, this is target
        const sourceKind = selectedInfo?.produces ?? "any";
        const targetAccepts = accepts ?? ["any"];
        isCompatible = acceptsConnection(sourceKind, targetAccepts);
      } else {
        // Selected is target, this is source
        const sourceKind = produces ?? "any";
        const targetAccepts = selectedInfo?.accepts ?? ["any"];
        isCompatible = acceptsConnection(sourceKind, targetAccepts);
      }

      if (isCompatible) {
        compatibilityClass = "ring-2 ring-green-400 ring-opacity-75";
      } else {
        compatibilityClass = "opacity-30";
      }
    }
  }

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

  // Shake animation class
  const shakeClass = isShaking ? "animate-shake" : "";

  return (
    <Handle
      type={type}
      position={position}
      id={id}
      className={`nodrag nopan !h-20 !w-20 !border-2 ${shapeClass} ${colorClass} ${cooldownClass} ${compatibilityClass} ${shakeClass} ${
        hideInArrayZone ? "!invisible !pointer-events-none" : ""
      } ${
        disabled ? "pointer-events-none cursor-not-allowed opacity-35" : "cursor-pointer"
      } ${className}`}
      style={style}
      onClick={handleClick}
    />
  );
}
