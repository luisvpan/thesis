/**
 * Heuristics for determining how to render results based on the dataflow graph structure.
 * These functions analyze the graph to extract presentation hints without modifying RuntimeValue.
 */

import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/node/types";
import type { SourceFlowNodeData, OperatorFlowNodeData } from "./index";

export type MultiplicationGrouping =
  | { kind: "none" }
  | { kind: "grouped"; groupSize: number; groupCount: number };

/**
 * Determines if an operator node is a CPA × scalar multiplication.
 * Returns grouping info if applicable, "none" otherwise.
 *
 * @param operatorNode - The operator node to analyze
 * @param edges - All edges in the graph
 * @param nodes - All nodes in the graph
 * @param finalQuantity - Total quantity of the result (CPA quantity or array length)
 */
export function computeMultiplicationGrouping(
  operatorNode: DataflowNode,
  edges: Edge[],
  nodes: DataflowNode[],
  finalQuantity: number
): MultiplicationGrouping {
  // 1. Validate it's a multiplication operator
  if (operatorNode.type !== "operator") {
    return { kind: "none" };
  }

  const data = operatorNode.data as OperatorFlowNodeData;
  if (data.operator !== "multiplicacion") {
    return { kind: "none" };
  }

  // 2. Get incoming edges and source nodes
  const inputEdges = edges.filter((e) => e.target === operatorNode.id);
  if (inputEdges.length !== 2) {
    return { kind: "none" };
  }

  const sourceNodes = inputEdges
    .map((e) => nodes.find((n) => n.id === e.source))
    .filter((n): n is DataflowNode => n !== undefined);

  if (sourceNodes.length !== 2) {
    return { kind: "none" };
  }

  // 3. Find the scalar (source with variant="number") and check for CPA/array
  let scalarValue: number | null = null;
  let hasCpaOrArray = false;

  for (const node of sourceNodes) {
    if (node.type === "source") {
      const d = node.data as SourceFlowNodeData;
      if (d.variant === "number") {
        scalarValue = d.value ?? 0;
      } else {
        // CPA source variants: cap, stick, montessori, shape, food
        hasCpaOrArray = true;
      }
    } else if (node.type === "operator" || node.type === "arrayClose") {
      // Result from another operation or array
      hasCpaOrArray = true;
    }
  }

  // 4. Validate: exactly 1 positive integer scalar + 1 CPA/array
  if (scalarValue === null || !hasCpaOrArray) {
    return { kind: "none" };
  }

  if (!Number.isInteger(scalarValue) || scalarValue <= 0) {
    return { kind: "none" };
  }

  // 5. Calculate grouping
  const groupSize = finalQuantity / scalarValue;

  if (!Number.isInteger(groupSize) || groupSize <= 0) {
    return { kind: "none" };
  }

  return {
    kind: "grouped",
    groupSize,
    groupCount: scalarValue,
  };
}
