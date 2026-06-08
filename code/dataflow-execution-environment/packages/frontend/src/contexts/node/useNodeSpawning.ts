import { useCallback, type Dispatch, type SetStateAction } from "react";
import type { OperatorType } from "@/types/card-types";
import { spawnActionForYoloClass } from "@/data/yoloDeckCatalog";
import type { DataflowNode } from "./types";
import { getNodePortsForType } from "./getNodePortsForType";
import { withVisionNodeChrome } from "./visionNodePresentation";

type SetNodes = Dispatch<SetStateAction<DataflowNode[]>>;

function devSpawnNode(
  node: DataflowNode,
  nodesDraggable: boolean
): DataflowNode {
  return withVisionNodeChrome(node, {}, nodesDraggable);
}

export function useNodeSpawning(setNodes: SetNodes, nodesDraggable = false) {
  const getNodePorts = useCallback(
    (nodeType: "source" | "operator") => getNodePortsForType(nodeType),
    []
  );

  const addNumberNode = useCallback(
    (value: number, position?: { x: number; y: number }) => {
      const id = `num${value}_${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        devSpawnNode(
          {
            id,
            type: "source" as const,
            position: position ?? {
              x: 100 + (nds.length % 3) * 60,
              y: 80 + Math.floor(nds.length / 3) * 100,
            },
            data: { variant: "number", value, digitValue: value },
          },
          nodesDraggable
        ),
      ]);
    },
    [setNodes, nodesDraggable]
  );

  const addOperatorNode = useCallback(
    (operator: OperatorType, position?: { x: number; y: number }) => {
      const id = `op${operator}_${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        devSpawnNode(
          {
            id,
            type: "operator" as const,
            position: position ?? {
              x: 320 + (nds.filter((n) => n.type === "operator").length % 2) * 200,
              y: 120,
            },
            data: { operator },
          },
          nodesDraggable
        ),
      ]);
    },
    [setNodes, nodesDraggable]
  );

  const addResultAnchorPair = useCallback(() => {
    setNodes((nds) => {
      const pairId = `manual_out_${Date.now()}`;
      return [
        ...nds,
        devSpawnNode(
          {
            id: `${pairId}`,
            type: "programOutput" as const,
            position: { x: 304, y: 120 },
            data: {},
          },
          nodesDraggable
        ),
      ];
    });
  }, [setNodes, nodesDraggable]);

  const addResultCard = useCallback(() => {
    setNodes((nds) => [
      ...nds,
      devSpawnNode(
        {
          id: `result_${Date.now()}`,
          type: "programOutput" as const,
          position: { x: 380 + (nds.length % 4) * 220, y: 140 },
          data: {},
        },
        nodesDraggable
      ),
    ]);
  }, [setNodes, nodesDraggable]);

  const addArrayOpenNode = useCallback(() => {
    setNodes((nds) => [
      ...nds,
      devSpawnNode(
        {
          id: `arr_open_${Date.now()}`,
          type: "arrayOpen" as const,
          position: { x: 80, y: 100 + (nds.length % 4) * 120 },
          data: {},
        },
        nodesDraggable
      ),
    ]);
  }, [setNodes, nodesDraggable]);

  const addArrayCloseNode = useCallback(() => {
    setNodes((nds) => [
      ...nds,
      devSpawnNode(
        {
          id: `arr_close_${Date.now()}`,
          type: "arrayClose" as const,
          position: { x: 500, y: 100 + (nds.length % 4) * 120 },
          data: {},
        },
        nodesDraggable
      ),
    ]);
  }, [setNodes, nodesDraggable]);

  const spawnDeckYoloClass = useCallback(
    (yoloClass: string) => {
      const spawn = spawnActionForYoloClass(yoloClass);
      if (!spawn) return;
      if (spawn.kind === "number") return addNumberNode(spawn.value);
      if (spawn.kind === "operator") return addOperatorNode(spawn.operator);
      if (spawn.kind === "resultCard") return addResultCard();
      if (spawn.kind === "arrayOpen") {
        setNodes((nds) => [
          ...nds,
          devSpawnNode(
            {
              id: `deck_open_${Date.now()}`,
              type: "arrayOpen" as const,
              position: { x: 120, y: 200 + (nds.length % 6) * 28 },
              data: {},
            },
            nodesDraggable
          ),
        ]);
        return;
      }
      if (spawn.kind === "arrayClose") {
        setNodes((nds) => [
          ...nds,
          devSpawnNode(
            {
              id: `deck_close_${Date.now()}`,
              type: "arrayClose" as const,
              position: { x: 120, y: 200 + (nds.length % 6) * 28 },
              data: {},
            },
            nodesDraggable
          ),
        ]);
        return;
      }
      if (spawn.kind === "shape") {
        setNodes((nds) => [
          ...nds,
          devSpawnNode(
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
            nodesDraggable
          ),
        ]);
        return;
      }
      if (spawn.kind === "food") {
        setNodes((nds) => [
          ...nds,
          devSpawnNode(
            {
              id: `deck_${spawn.yoloClass}_${Date.now()}`,
              type: "source" as const,
              position: { x: 120, y: 200 + (nds.length % 6) * 28 },
              data: { variant: "food", yoloClass: spawn.yoloClass, food: spawn.food },
            },
            nodesDraggable
          ),
        ]);
        return;
      }
      if (spawn.kind === "montessori") {
        setNodes((nds) => [
          ...nds,
          devSpawnNode(
            {
              id: `deck_${spawn.yoloClass}_${Date.now()}`,
              type: "source" as const,
              position: { x: 120, y: 200 + (nds.length % 6) * 28 },
              data: {
                variant: "montessori",
                yoloClass: spawn.yoloClass,
                color: spawn.color,
              },
            },
            nodesDraggable
          ),
        ]);
        return;
      }
      if (spawn.kind === "dice") {
        const diceValue = Math.floor(Math.random() * 6) + 1;
        setNodes((nds) => [
          ...nds,
          devSpawnNode(
            {
              id: `deck_dice_${Date.now()}`,
              type: "source" as const,
              position: { x: 120, y: 200 + (nds.length % 6) * 28 },
              data: { variant: "dice", diceValue },
            },
            nodesDraggable
          ),
        ]);
      }
    },
    [addNumberNode, addOperatorNode, addResultCard, setNodes, nodesDraggable]
  );

  return {
    getNodePorts,
    addNumberNode,
    addOperatorNode,
    addResultAnchorPair,
    addResultCard,
    spawnDeckYoloClass,
    addArrayOpenNode,
    addArrayCloseNode,
  };
}
