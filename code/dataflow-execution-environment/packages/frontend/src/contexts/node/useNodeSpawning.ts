import { useCallback, type Dispatch, type SetStateAction } from "react";
import type { OperatorType } from "@/types/card-types";
import { spawnActionForYoloClass } from "@/data/yoloDeckCatalog";
import type { DataflowNode } from "./types";
import { getNodePortsForType } from "./getNodePortsForType";

type SetNodes = Dispatch<SetStateAction<DataflowNode[]>>;

export function useNodeSpawning(setNodes: SetNodes) {
  const getNodePorts = useCallback(
    (nodeType: "source" | "operator") => getNodePortsForType(nodeType),
    []
  );

  const addNumberNode = useCallback(
    (value: number, position?: { x: number; y: number }) => {
      const id = `num${value}_${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: "source" as const,
          position: position ?? {
            x: 100 + (nds.length % 3) * 60,
            y: 80 + Math.floor(nds.length / 3) * 100,
          },
          data: { variant: "number", value },
        },
      ]);
    },
    [setNodes]
  );

  const addOperatorNode = useCallback(
    (operator: OperatorType, position?: { x: number; y: number }) => {
      const id = `op${operator}_${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: "operator" as const,
          position: position ?? {
            x: 320 + (nds.filter((n) => n.type === "operator").length % 2) * 200,
            y: 120,
          },
          data: { operator },
        },
      ]);
    },
    [setNodes]
  );

  const addResultAnchorPair = useCallback(() => {
    setNodes((nds) => {
      const pairId = `manual_out_${Date.now()}`;
      return [
        ...nds,
        {
          id: `${pairId}`,
          type: "programOutput" as const,
          position: { x: 304, y: 120 },
          data: {},
        },
      ];
    });
  }, [setNodes]);

  const addResultCard = useCallback(() => {
    setNodes((nds) => [
      ...nds,
      {
        id: `result_${Date.now()}`,
        type: "programOutput" as const,
        position: { x: 380 + (nds.length % 4) * 220, y: 140 },
        data: {},
      },
    ]);
  }, [setNodes]);

  const spawnDeckYoloClass = useCallback(
    (yoloClass: string) => {
      const spawn = spawnActionForYoloClass(yoloClass);
      if (!spawn) return;
      if (spawn.kind === "number") return addNumberNode(spawn.value);
      if (spawn.kind === "operator") return addOperatorNode(spawn.operator);
      if (spawn.kind === "resultCard") return addResultCard();
      if (spawn.kind === "shape") {
        setNodes((nds) => [
          ...nds,
          {
            id: `deck_${spawn.yoloClass}_${Date.now()}`,
            type: "source" as const,
            position: { x: 120, y: 200 + (nds.length % 6) * 28 },
            data: {
              variant: "shape",
              yoloClass: spawn.yoloClass,
              shape: spawn.shape,
              size: spawn.size,
              color: spawn.color,
            },
          },
        ]);
        return;
      }
      if (spawn.kind === "food") {
        setNodes((nds) => [
          ...nds,
          {
            id: `deck_${spawn.yoloClass}_${Date.now()}`,
            type: "source" as const,
            position: { x: 120, y: 200 + (nds.length % 6) * 28 },
            data: { variant: "food", yoloClass: spawn.yoloClass, food: spawn.food },
          },
        ]);
      }
    },
    [addNumberNode, addOperatorNode, addResultCard, setNodes]
  );

  return {
    getNodePorts,
    addNumberNode,
    addOperatorNode,
    addResultAnchorPair,
    addResultCard,
    spawnDeckYoloClass,
  };
}
