import { useCallback, type Dispatch, type SetStateAction } from "react";
import { addEdge, type Connection, type Edge } from "@xyflow/react";
import type { PortIdentifier } from "./types";

type SetEdges = Dispatch<SetStateAction<Edge[]>>;
type SetSelectedPort = Dispatch<SetStateAction<PortIdentifier | null>>;

export function usePortSelection(
  selectedPort: PortIdentifier | null,
  setSelectedPort: SetSelectedPort,
  setEdges: SetEdges
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

      setEdges((eds) => addEdge(connection, eds));
      setSelectedPort(null);
    },
    [selectedPort, setEdges, setSelectedPort]
  );

  return { isPortSelected, clearSelection, handlePortClick };
}
