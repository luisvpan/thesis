import { useState } from "react";
import { Handle, Position } from "@xyflow/react";
import { useNode } from "@/contexts/NodeContext";
import { type HandleKind, type HandleAcceptance, checkConnection, handleShape } from "./handle-kinds";

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
  acceptance?: HandleAcceptance;
};

/**
 * Get the shape class based on the handle shape.
 */
function getShapeClass(shape: "circle" | "square" | "rounded-square" | "pill"): string {
  switch (shape) {
    case "circle":
      return "!rounded-full";
    case "square":
      return "!rounded-md";
    case "rounded-square":
      return "!rounded-2xl";
    case "pill":
      return "!rounded-full !w-28"; // pill: wider than the default square
  }
}

/**
 * Determine the shape for this handle based on its type and kind info.
 */
function getHandleShape(
  type: "source" | "target",
  produces?: HandleKind,
  acceptance?: HandleAcceptance
): "circle" | "square" | "rounded-square" | "pill" {
  if (type === "source") {
    // Source handles: shape based on what they produce
    if (!produces || produces === "rational") return "circle";
    if (produces === "cpa") return "square";
    if (produces === "keyword") return "pill";
    return "circle";
  }
  // Target handles: use handleShape with primary kinds
  if (acceptance) {
    return handleShape(acceptance.primary);
  }
  // Default: circle
  return "circle";
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
  acceptance,
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
  const shape = getHandleShape(type, produces, acceptance);
  const shapeClass = getShapeClass(shape);

  // Check if this handle is shaking (incompatible connection attempted)
  const isShaking = shakingPort?.nodeId === nodeId && shakingPort?.handleId === id;

  // Calculate compatibility feedback when there's a selected port
  let compatibilityClass = "";
  if (selectedPort && !disabled && !selected) {
    const isOppositeType = selectedPort.handleType !== type;

    if (isOppositeType) {
      // Get the selected port's kind info
      const selectedInfo = getPortKindInfo(selectedPort.nodeId, selectedPort.handleId);

      // Default acceptance for handles without explicit acceptance
      const defaultAcceptance: HandleAcceptance = { primary: ["rational", "cpa"] };

      let match: "compatible" | "tolerated" | "incompatible";
      if (selectedPort.handleType === "source") {
        // Selected is source, this is target
        const sourceKind = selectedInfo?.produces ?? "rational";
        const targetAcceptance = acceptance ?? defaultAcceptance;
        match = checkConnection(sourceKind, targetAcceptance);
      } else {
        // Selected is target, this is source
        const sourceKind = produces ?? "rational";
        const targetAcceptance = selectedInfo?.acceptance ?? defaultAcceptance;
        match = checkConnection(sourceKind, targetAcceptance);
      }

      if (match === "compatible") {
        compatibilityClass = "ring-2 ring-green-400 ring-opacity-75";
      } else if (match === "tolerated") {
        compatibilityClass = "ring-2 ring-amber-400 ring-opacity-40";
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
