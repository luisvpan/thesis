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
  ArrayLiteral,
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

// Approximate rendered size of the bracket nodes (px, in XYFlow coordinates)
const BRACKET_NODE_W = 64;
const BRACKET_NODE_H = 80;

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

  // 2a. Array pairs: each edge from arrayOpen → arrayClose (zone-in) defines an array.
  //     The bounding box of the two bracket nodes determines which source/operator nodes
  //     are elements of the array.
  for (const edge of edges) {
    if (edge.targetHandle !== "zone-in") continue;

    const openNode = nodes.find(
      (n) => n.id === edge.source && n.type === "arrayOpen"
    );
    const closeNode = nodes.find(
      (n) => n.id === edge.target && n.type === "arrayClose"
    );
    if (!openNode || !closeNode) continue;

    const left   = Math.min(openNode.position.x, closeNode.position.x);
    const right  = Math.max(openNode.position.x, closeNode.position.x) + BRACKET_NODE_W;
    const top    = Math.min(openNode.position.y, closeNode.position.y);
    const bottom = Math.max(openNode.position.y, closeNode.position.y) + BRACKET_NODE_H;

    const inner = nodes
      .filter((n) => n.type === "source" || n.type === "operator")
      .filter((n) => {
        const nx = n.position.x;
        const ny = n.position.y;
        return nx >= left && nx <= right && ny >= top && ny <= bottom;
      })
      .sort((a, b) =>
        a.position.x !== b.position.x
          ? a.position.x - b.position.x
          : a.position.y - b.position.y
      );

    if (inner.length === 0) continue;

    const elements = inner.map((n) => ({ type: "Identifier" as const, name: n.id }));
    const arrayLiteral: ArrayLiteral = { type: "ArrayLiteral", elements };

    statements.push({
      type: "SourceStatement",
      identifier: closeNode.id,
      value: arrayLiteral,
    });
  }

  // 2b. Transforms: "operator" nodes
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

    // Verify the source is an evaluable node (source, operator, or arrayClose)
    const sourceNode = nodes.find((n) => n.id === inputEdge.source);
    if (
      !sourceNode ||
      (sourceNode.type !== "source" &&
        sourceNode.type !== "operator" &&
        sourceNode.type !== "arrayClose")
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
