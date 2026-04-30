/**
 * Converts ReactFlow nodes/edges to the interpreter's Program format.
 */

import Fraction from "fraction.js";
import type {
  Program,
  SourceStatement,
  TransformStatement,
  SinkStatement,
  Operation,
} from "@dataflow/interpreter";
import type { DataflowNode } from "@/contexts/NodeContext";
import type { Edge } from "@xyflow/react";
import type { SourceFlowNodeData, OperatorFlowNodeData } from "@/components/dataflow";

const OPERATOR_MAP: Record<string, Operation> = {
  adicion: "sum",
  sustraccion: "substract",
  multiplicacion: "multiply",
  division: "divide",
};

// Normalizar tamaños a formas masculinas (el intérprete solo entiende masculino)
const SIZE_MAP: Record<string, string> = {
  pequeño: "pequeño",
  pequeña: "pequeño",
  mediano: "mediano",
  mediana: "mediano",
  grande: "grande",
};

function normalizeSize(size: string | undefined): string {
  if (!size) return "mediano";
  return SIZE_MAP[size] ?? size;
}

/**
 * Converts ReactFlow nodes and edges to a Program object for the interpreter.
 */
export function flowToProgram(nodes: DataflowNode[], edges: Edge[]): Program {
  console.log("[flowToProgram] Input nodes:", nodes.length, "edges:", edges.length);

  const statements: (SourceStatement | TransformStatement | SinkStatement)[] = [];

  // 1. Sources: all "source" nodes (numbers, shapes, food)
  for (const node of nodes) {
    if (node.type === "source") {
      const data = node.data as SourceFlowNodeData;

      if (data.variant === "number") {
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: { type: "NumberLiteral", value: new Fraction(data.value ?? 0) },
        });
      } else if (data.variant === "shape") {
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: {
            type: "ObjectLiteral",
            category: "pictorico",
            objectType: "forma",
            subtype: data.shape,
            size: normalizeSize(data.size),
            amount: new Fraction(1),
          },
        });
      } else if (data.variant === "food") {
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: {
            type: "ObjectLiteral",
            category: "concreto",
            objectType: "comida",
            subtype: data.food,
            color: "verde",
            amount: new Fraction(1),
          },
        });
      }
    }
  }

  // 2. Transforms: "operator" nodes
  for (const node of nodes) {
    if (node.type === "operator") {
      const data = node.data as OperatorFlowNodeData;
      const operator = data.operator ?? "adicion";
      const inputEdges = edges.filter((e) => e.target === node.id);

      // Sort by targetHandle: "a" first, "b" second
      const sortedEdges = inputEdges.sort((a, b) => {
        if (a.targetHandle === "a") return -1;
        if (b.targetHandle === "a") return 1;
        return 0;
      });

      const args = sortedEdges.map((e) => ({
        type: "Identifier" as const,
        name: e.source,
      }));

      statements.push({
        type: "TransformStatement",
        identifier: node.id,
        operation: OPERATOR_MAP[operator] ?? "sum",
        arguments: args,
      });
    }
  }

  // 3. Sinks: one per programOutput connected to an evaluable node
  for (const node of nodes) {
    if (node.type !== "programOutput") continue;

    // Find what node is connected to this programOutput's input
    const inputEdge = edges.find(
      (e) => e.target === node.id && e.targetHandle === "in"
    );

    if (!inputEdge) continue;

    // Verify the source is an evaluable node (source or operator)
    const sourceNode = nodes.find((n) => n.id === inputEdge.source);
    if (
      !sourceNode ||
      (sourceNode.type !== "source" && sourceNode.type !== "operator")
    ) {
      continue;
    }

    statements.push({
      type: "SinkStatement",
      identifier: `output_${node.id}`,
      sourceIdentifier: inputEdge.source,
    });
  }

  console.log("[flowToProgram] Generated", statements.length, "statements");
  return { type: "Program", statements };
}
