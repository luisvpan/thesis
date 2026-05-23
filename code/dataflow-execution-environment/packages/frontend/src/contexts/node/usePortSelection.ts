import { useCallback, type Dispatch, type SetStateAction } from "react";
import { addEdge, type Connection, type Edge } from "@xyflow/react";
import type { DataflowNode, PortIdentifier, PortKindInfo } from "./types";
import { canConnectPorts, isPortOccupied } from "../../components/dataflow/connectionRules";

type SetEdges = Dispatch<SetStateAction<Edge[]>>;
type SetSelectedPort = Dispatch<SetStateAction<PortIdentifier | null>>;
type GetPortKindInfo = (nodeId: string, handleId: string) => PortKindInfo | undefined;
type TriggerIncompatibleFeedback = (nodeId: string, handleId: string) => void;

export function usePortSelection(
  selectedPort: PortIdentifier | null,
  setSelectedPort: SetSelectedPort,
  setEdges: SetEdges,
  nodes: DataflowNode[],
  edges: Edge[],
  getPortKindInfo: GetPortKindInfo,
  triggerIncompatibleFeedback: TriggerIncompatibleFeedback
) {
  const isPortSelected = useCallback(
    (nodeId: string, handleId: string, handleType: "source" | "target") =>
      selectedPort?.nodeId === nodeId &&
      selectedPort?.handleId === handleId &&
      selectedPort?.handleType === handleType,
    [selectedPort]
  );

  const clearSelection = useCallback(() => {
    setSelectedPort(null);
  }, [setSelectedPort]);

  const handlePortClick = useCallback(
    (nodeId: string, handleId: string, handleType: "source" | "target") => {
      const port = { nodeId, handleId, handleType };

      if (isPortOccupied(edges, port)) {
        return;
      }

      if (!selectedPort) {
        setSelectedPort(port);
        return;
      }

      if (
        selectedPort.nodeId === nodeId &&
        selectedPort.handleId === handleId &&
        selectedPort.handleType === handleType
      ) {
        setSelectedPort(null);
        return;
      }

      const first = selectedPort;
      const second = port;

      if (first.handleType === second.handleType) {
        if (isPortOccupied(edges, second)) {
          return;
        }
        setSelectedPort(second);
        return;
      }

      if (first.nodeId === second.nodeId) {
        if (isPortOccupied(edges, second)) {
          return;
        }
        setSelectedPort(second);
        return;
      }

      const source = first.handleType === "source" ? first : second;
      const target = first.handleType === "target" ? first : second;

      const ctx = { nodes, edges, getPortKindInfo };
      const result = canConnectPorts(source, target, ctx);

      if (!result.ok) {
        triggerIncompatibleFeedback(target.nodeId, target.handleId);
        setSelectedPort(null);
        return;
      }

      const connection: Connection = {
        source: source.nodeId,
        sourceHandle: source.handleId,
        target: target.nodeId,
        targetHandle: target.handleId,
      };

      const sourceNodeType = nodes.find((n) => n.id === source.nodeId)?.type;
      const targetNodeType = nodes.find((n) => n.id === target.nodeId)?.type;
      const edgeType =
        sourceNodeType === "arrayOpen" && targetNodeType === "arrayClose"
          ? "arrayZoneEdge"
          : undefined;

      setEdges((eds) => addEdge({ ...connection, type: edgeType }, eds));
      setSelectedPort(null);
    },
    [
      selectedPort,
      setEdges,
      setSelectedPort,
      nodes,
      edges,
      getPortKindInfo,
      triggerIncompatibleFeedback,
    ]
  );

  return { isPortSelected, clearSelection, handlePortClick };
}
