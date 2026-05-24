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
  ObjectLiteral,
  ObjectProperty,
} from "@dataflow/interpreter";
import type { Edge } from "@xyflow/react";
import type { SourceFlowNodeData, OperatorFlowNodeData } from "../components/dataflow";
import { isPictorialColorYoloClass } from "../data/pictorialColors";
import type { DataflowNode } from "../contexts/node/types";
import { getOrderedArrayZoneMembers } from "./arrayZoneGeometry";
import {
  resolveNumberSourceId,
  shouldEmitNumberSource,
} from "./numberTouchMerge";
import { logger } from "@/lib/logger";

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
 * Creates a CPA ObjectLiteral with the new properties-based format.
 */
function createCPAObject(
  category: string,
  type: string,
  subtype: string,
  quantity: number,
  attributes: Record<string, string> = {}
): ObjectLiteral {
  const properties: ObjectProperty[] = [
    { key: "category", value: category },
    { key: "type", value: type },
    { key: "subtype", value: subtype },
    { key: "quantity", value: new Fraction(quantity) },
  ];
  for (const [k, v] of Object.entries(attributes)) {
    properties.push({ key: k, value: v });
  }
  return { type: "ObjectLiteral", properties };
}

/**
 * Converts ReactFlow nodes and edges to a Program object for the interpreter.
 */
export function flowToProgram(nodes: DataflowNode[], edges: Edge[]): Program {
  logger.flow.debug("Input", { nodes: nodes.length, edges: edges.length });

  const statements: (SourceStatement | TransformStatement | SinkStatement)[] = [];

  // 1. Sources: all "source" nodes (numbers, shapes, food)
  for (const node of nodes) {
    if (node.type === "source") {
      const data = node.data as SourceFlowNodeData;

      if (data.variant === "number") {
        if (!shouldEmitNumberSource(node.id, nodes)) continue;
        // Use new grammar v3.1.0: numbers are CPA abstractos
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: createCPAObject("abstracto", "numero", "racional", data.value ?? 0, {}),
        });
      } else if (data.variant === "shape") {
        const shapeAttrs: Record<string, string> = {
          size: normalizeSize(data.size),
        };
        if (isPictorialColorYoloClass(data.yoloClass)) {
          shapeAttrs.color = data.color;
        }
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: createCPAObject(
            "pictorico",
            "forma",
            data.shape ?? "circulo",
            1,
            shapeAttrs
          ),
        });
      } else if (data.variant === "food") {
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: createCPAObject("concreto", "comida", data.food ?? "manzana", 1, {
            color: "verde",
          }),
        });
      } else if (data.variant === "montessori") {
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: createCPAObject("concreto", "montessori", data.color ?? "azul", 1, {
            color: data.color ?? "azul",
          }),
        });
      } else if (data.variant === "cap") {
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: createCPAObject("concreto", "cap", data.color ?? "azul", 1, {
            color: data.color ?? "azul",
          }),
        });
      } else if (data.variant === "stick") {
        statements.push({
          type: "SourceStatement",
          identifier: node.id,
          value: createCPAObject("concreto", "stick", data.color ?? "rojo", 1, {
            color: data.color ?? "rojo",
          }),
        });
      }
    }
  }

  // 2a. Array pairs: each edge from arrayOpen → arrayClose (zone-in) defines an array.
  //     AABB entre handlers zone-out (abrir) y zone-in (cerrar); miembros = fuentes/operadores
  //     cuya carta solapa la zona (inclusive en el borde).
  for (const edge of edges) {
    if (edge.targetHandle !== "zone-in") continue;

    const closeNode = nodes.find(
      (n) => n.id === edge.target && n.type === "arrayClose"
    );
    if (!closeNode) continue;

    const inner = getOrderedArrayZoneMembers(closeNode.id, nodes, edges);

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
        name: resolveNumberSourceId(e.source, nodes),
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
      sourceIdentifier: resolveNumberSourceId(inputEdge.source, nodes),
    });
  }

  logger.flow.debug("Generated statements", { count: statements.length });
  return { type: "Program", statements };
}
