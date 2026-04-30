import { OPERATOR_PORTS, SOURCE_PORTS } from "./constants";
import type { PortDefinition } from "./types";

export function getNodePortsForType(
  nodeType: "source" | "operator"
): PortDefinition[] {
  return nodeType === "source" ? SOURCE_PORTS : OPERATOR_PORTS;
}
