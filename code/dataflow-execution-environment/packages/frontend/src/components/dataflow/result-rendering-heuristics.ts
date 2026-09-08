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

export type DivisionGrouping =
  | { kind: "none" }
  | { kind: "grouped"; groupSize: number; groupCount: number; mode: "partitivo" | "cuotativo" }
  | { kind: "grouped-with-remainder"; groupSize: number; groupCount: number; remainderQty: number; mode: "partitivo" | "cuotativo" };

/**
 * Determines if an operator node is a CPA ÷ scalar division.
 * Returns grouping info based on the division mode (partitivo or cuotativo).
 *
 * @param operatorNode - The operator node to analyze
 * @param edges - All edges in the graph
 * @param nodes - All nodes in the graph
 * @param finalQuantity - The quotient (CPA quantity after division)
 * @param numerator - Numerator of the result fraction (from Fraction.js)
 * @param denominator - Denominator of the result fraction (from Fraction.js)
 */
export function computeDivisionGrouping(
  operatorNode: DataflowNode,
  edges: Edge[],
  nodes: DataflowNode[],
  finalQuantity: number,
  numerator?: number,
  denominator?: number
): DivisionGrouping {
  // 1. Validate it's a division operator
  if (operatorNode.type !== "operator") {
    return { kind: "none" };
  }

  const data = operatorNode.data as OperatorFlowNodeData;
  if (data.operator !== "division") {
    return { kind: "none" };
  }

  // 2. Get the divisor from input "b"
  const inputEdges = edges.filter((e) => e.target === operatorNode.id);
  const bEdge = inputEdges.find((e) => e.targetHandle === "b");
  if (!bEdge) return { kind: "none" };

  const bSource = nodes.find((n) => n.id === bEdge.source);
  if (!bSource || bSource.type !== "source") return { kind: "none" };

  const bData = bSource.data as SourceFlowNodeData;
  if (bData.variant !== "number") return { kind: "none" };

  const divisor = bData.value ?? 0;
  if (!Number.isInteger(divisor) || divisor <= 0) return { kind: "none" };

  // 3. Verify that input "a" is CPA (not a number)
  const aEdge = inputEdges.find((e) => e.targetHandle === "a");
  if (!aEdge) return { kind: "none" };

  const aSource = nodes.find((n) => n.id === aEdge.source);
  if (!aSource) return { kind: "none" };

  let isCpaInput = false;
  if (aSource.type === "source") {
    const aData = aSource.data as SourceFlowNodeData;
    isCpaInput = aData.variant !== "number";
  } else if (aSource.type === "operator" || aSource.type === "arrayClose") {
    isCpaInput = true;
  }
  if (!isCpaInput) return { kind: "none" };

  // 4. Calculate grouping based on numerator/denominator
  const mode = data.divisionMode ?? "partitivo";

  // If we don't have numerator/denominator, fallback to legacy behavior (exact division only)
  if (numerator === undefined || denominator === undefined) {
    const originalQuantity = finalQuantity * divisor;
    if (!Number.isInteger(originalQuantity)) return { kind: "none" };

    if (mode === "partitivo") {
      return { kind: "grouped", groupSize: finalQuantity, groupCount: divisor, mode };
    } else {
      return { kind: "grouped", groupSize: divisor, groupCount: finalQuantity, mode };
    }
  }

  // Exact division: denominator === 1 means result is an integer
  if (denominator === 1) {
    if (mode === "partitivo") {
      // Partitivo: divide into N groups (N = divisor)
      // 12 ÷ 3 partitivo = 3 groups of 4
      return { kind: "grouped", groupSize: numerator, groupCount: divisor, mode };
    } else {
      // Cuotativo: groups of size N (N = divisor)
      // 12 ÷ 3 cuotativo = 4 groups of 3
      return { kind: "grouped", groupSize: divisor, groupCount: numerator, mode };
    }
  }

  // Division with remainder: denominator > 1
  // Reconstruct original dividend: D = n * (divisor / d)
  // This works because result = D/divisor, and after simplification: n/d = D/divisor
  // So: D = n * divisor / d
  const originalDividend = numerator * (divisor / denominator);
  const quotient = Math.floor(originalDividend / divisor);
  const remainder = originalDividend % divisor;

  if (mode === "partitivo") {
    // Partitivo: N groups of quotient items + remainder items
    // 13 ÷ 4 partitivo = 4 groups of 3 + 1 remainder
    return {
      kind: "grouped-with-remainder",
      groupSize: quotient,
      groupCount: divisor,
      remainderQty: remainder,
      mode,
    };
  } else {
    // Cuotativo: quotient groups of N items + remainder items
    // 13 ÷ 4 cuotativo = 3 groups of 4 + 1 remainder
    return {
      kind: "grouped-with-remainder",
      groupSize: divisor,
      groupCount: quotient,
      remainderQty: remainder,
      mode,
    };
  }
}
