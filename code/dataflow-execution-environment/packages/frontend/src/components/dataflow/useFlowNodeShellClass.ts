import { useNode } from "@/contexts/NodeContext";
import { FLOW_NODE_SHELL_CLASS } from "./flowNodeChrome";

/** En juego: la carta no captura clics (solo handles). En dev: permite arrastrar la carta. */
export function useFlowNodeShellClass(): string {
  const { nodesDraggable } = useNode();
  return nodesDraggable ? "nodrag nopan" : FLOW_NODE_SHELL_CLASS;
}
