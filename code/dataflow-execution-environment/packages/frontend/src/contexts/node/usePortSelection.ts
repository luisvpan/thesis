import { useCallback, type Dispatch, type SetStateAction } from "react";
import { addEdge, type Connection, type Edge } from "@xyflow/react";
import type { DataflowNode, PortIdentifier } from "./types";

type SetEdges = Dispatch<SetStateAction<Edge[]>>;
type SetSelectedPort = Dispatch<SetStateAction<PortIdentifier | null>>;

export function usePortSelection(
  selectedPort: PortIdentifier | null,
  setSelectedPort: SetSelectedPort,
  setEdges: SetEdges,
  nodes: DataflowNode[]
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
      if (!selectedPort) {
        setSelectedPort({ nodeId, handleId, handleType });
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
      const second = { nodeId, handleId, handleType };

      if (first.handleType === second.handleType) {
        setSelectedPort(second);
        return;
      }

      if (first.nodeId === second.nodeId) {
        setSelectedPort(second);
        return;
      }

      const source = first.handleType === "source" ? first : second;
      const target = first.handleType === "target" ? first : second;

      const connection: Connection = {
        source: source.nodeId,
        sourceHandle: source.handleId,
        target: target.nodeId,
        targetHandle: target.handleId,
      };

      // Use zone edge type when connecting [ → ]
      const sourceNodeType = nodes.find((n) => n.id === source.nodeId)?.type;
      const targetNodeType = nodes.find((n) => n.id === target.nodeId)?.type;
      const edgeType =
        sourceNodeType === "arrayOpen" && targetNodeType === "arrayClose"
          ? "arrayZoneEdge"
          : undefined;

      setEdges((eds) => addEdge({ ...connection, type: edgeType }, eds));
      setSelectedPort(null);
    },
    [selectedPort, setEdges, setSelectedPort, nodes]
  );

  return { isPortSelected, clearSelection, handlePortClick };
}
