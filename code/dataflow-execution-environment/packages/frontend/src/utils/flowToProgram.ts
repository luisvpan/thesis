/**
 * Converts ReactFlow nodes/edges to the interpreter's Program format.
 */

import type { EntrySpec, Operation, Program } from "@dataflow/interpreter";
import {
  createBag,
  createFilterCriterion,
  createOrderCriterion,
  createProgram,
} from "@dataflow/interpreter";
import type { Edge } from "@xyflow/react";
import type { SourceFlowNodeData, OperatorFlowNodeData } from "../components/dataflow";
import { isOrderOperatorType, isSingleInputOperatorType, resolveStickColor } from "../types/card-types";
import { isPictorialColorYoloClass } from "../data/pictorialColors";
import type { DataflowNode } from "../contexts/node/types";
import type { OrderCriterio } from "../data/yoloDeckCatalog";
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
  // Una sola operación de orden: la dirección viaja en el criterio.
  "orden-menor-mayor": "order",
  "orden-mayor-menor": "order",
  comparar: "compare",
  primero: "first",
  ultimo: "last",
  contar: "count",
  "filtrar-general": "filter",
  "filtrar-figuras": "filter",
  "filtrar-carros": "filter",
  "filtrar-comidas": "filter",
  "filtrar-animales": "filter",
  "filtrar-personas": "filter",
};

function resolveOperation(operator: string): Operation {
  return OPERATOR_MAP[operator] ?? "sum";
}

/** Sentido del orden según la carta de operador. */
function orderDirection(operator: string): "asc" | "desc" {
  return operator === "orden-mayor-menor" ? "desc" : "asc";
}

/** Identificador del criterio implícito de un operador de orden. */
function orderCriterionId(operatorNodeId: string): string {
  return `${operatorNodeId}__criterio`;
}

/**
 * El criterio de orden que declara una carta, en sus dos formas: por secuencia
 * de valores (el tamaño) o por orden natural de la propiedad (la cantidad).
 *
 * Una secuencia *es* el orden, así que para el sentido inverso se invierte la
 * secuencia en vez de pedirle una dirección. Una carta sin criterio —hecha a
 * mano, o de una sesión anterior— ordena por cantidad, que es lo que hacía
 * antes de que las cartas lo declararan.
 */
function orderCriterionOf(criterio: OrderCriterio | undefined, direction: "asc" | "desc") {
  const property = criterio?.property ?? "quantity";
  const sequence = criterio?.sequence;

  return createOrderCriterion({
    properties: [property],
    values: {
      [property]: sequence
        ? direction === "asc"
          ? [...sequence]
          : [...sequence].reverse()
        : direction,
    },
  });
}

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

// Mapeo de tipo de comida a su color natural
const FOOD_COLOR_MAP: Record<string, string> = {
  manzana: "rojo",
  pera: "verde",
  uva: "morado",
  hamburguesa: "naranja",
};

/**
 * La entrada de bolsa que declara una carta de datos, o `null` si la carta no
 * declara datos (un criterio, por ejemplo).
 */
function entryOf(node: DataflowNode): EntrySpec | null {
  if (node.type === "diceZone") {
    const value = (node.data as { value?: number }).value;
    if (value === undefined) return null;
    return { category: "abstracto", type: "numero", subtype: "racional", quantity: value };
  }

  if (node.type !== "source") return null;
  const data = node.data as SourceFlowNodeData;

  switch (data.variant) {
    case "number":
      return {
        category: "abstracto",
        type: "numero",
        subtype: "racional",
        quantity: data.value ?? 0,
      };

    case "shape": {
      const attributes: Record<string, string> = { size: normalizeSize(data.size) };
      if (isPictorialColorYoloClass(data.yoloClass)) attributes.color = data.color;
      return {
        category: "pictorico",
        type: "forma",
        subtype: data.shape ?? "circulo",
        quantity: 1,
        attributes,
      };
    }

    case "food": {
      const food = data.food ?? "manzana";
      return {
        category: "concreto",
        type: "comida",
        subtype: food,
        quantity: 1,
        attributes: { color: FOOD_COLOR_MAP[food] ?? "verde" },
      };
    }

    case "montessori":
      return {
        category: "concreto",
        type: "montessori",
        subtype: data.color ?? "azul",
        quantity: 1,
        attributes: { color: data.color ?? "azul" },
      };

    case "cap":
      return {
        category: "concreto",
        type: "cap",
        subtype: data.color ?? "azul",
        quantity: 1,
        attributes: { color: data.color ?? "azul" },
      };

    case "stick": {
      const color = resolveStickColor(data.color, data.yoloClass);
      return {
        category: "concreto",
        type: "stick",
        subtype: color,
        quantity: 1,
        attributes: { color },
      };
    }

    default:
      return null;
  }
}

/** Identificador en el programa para un nodo origen de arista (fuente, operador o salida). */
export function resolveFlowSourceId(
  nodeId: string,
  nodes: DataflowNode[]
): string {
  const node = nodes.find((n) => n.id === nodeId);
  if (node?.type === "programOutput") {
    return `output_${nodeId}`;
  }
  return resolveNumberSourceId(nodeId, nodes);
}

/**
 * Converts ReactFlow nodes and edges to a Program object for the interpreter.
 */
export function flowToProgram(nodes: DataflowNode[], edges: Edge[]): Program {
  logger.flow.debug("Input", { nodes: nodes.length, edges: edges.length });

  let program = createProgram();

  // 1a. diceZone nodes
  for (const node of nodes) {
    if (node.type !== "diceZone") continue;
    const entry = entryOf(node);
    if (entry) program = program.source(node.id, createBag().add(entry));
  }

  // 1b. Sources: all "source" nodes (numbers, shapes, food, criteria)
  for (const node of nodes) {
    if (node.type !== "source") continue;
    const data = node.data as SourceFlowNodeData;

    if (data.variant === "criteria") {
      // Un criterio por `source`: los criterios no se agrupan.
      program = program.source(
        node.id,
        createFilterCriterion({ properties: data.properties, values: data.values ?? {} })
      );
      continue;
    }

    if (data.variant === "number" && !shouldEmitNumberSource(node.id, nodes)) continue;

    const entry = entryOf(node);
    if (entry) program = program.source(node.id, createBag().add(entry));
  }

  // 2a. Array zones: una zona es UNA bolsa con las entradas de las cartas que
  //     contiene. Un `source` es entrada pura, así que las inlinea en vez de
  //     referenciarlas (AABB entre los handlers zone-out y zone-in).
  for (const node of nodes) {
    if (node.type !== "arrayClose") continue;

    let bag = createBag();

    for (const member of getOrderedArrayZoneMembers(node.id, nodes, edges)) {
      const entry = entryOf(member);
      if (entry) {
        bag = bag.add(entry);
      } else {
        // Un operador dentro de la zona no declara datos y no se puede inlinear.
        logger.flow.warn("Array zone member without data", { id: member.id, type: member.type });
      }
    }

    // Una zona vacía es `nulo`: un arreglo a medio armar no rompe el programa.
    program = program.source(node.id, bag);
  }

  // 2b. Transforms: "operator" nodes
  for (const node of nodes) {
    if (node.type !== "operator") continue;

    const data = node.data as OperatorFlowNodeData;
    const operator = data.operator ?? "adicion";
    const inputEdges = edges.filter((e) => e.target === node.id);

    const sortedEdges = isSingleInputOperatorType(operator)
      ? inputEdges.filter((e) => e.targetHandle === "a" || e.targetHandle == null)
      : inputEdges.sort((a, b) => {
          if (a.targetHandle === "a") return -1;
          if (b.targetHandle === "a") return 1;
          return 0;
        });

    const args = sortedEdges.map((e) => resolveFlowSourceId(e.source, nodes));

    if (isOrderOperatorType(operator)) {
      // El criterio va en su propio `source` y se referencia por nombre: los
      // argumentos de un transform son solo identificadores.
      const criterionId = orderCriterionId(node.id);

      program = program.source(
        criterionId,
        orderCriterionOf(data.criterio, orderDirection(operator))
      );

      args.push(criterionId);
    }

    // La operación es un dato de la carta y los argumentos salen de las
    // aristas, así que aquí manda el método genérico: los atajos por operación
    // no tendrían nada que comprobar.
    program = program.transform(node.id, resolveOperation(operator), args);
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

    program = program.sink(`output_${node.id}`, resolveFlowSourceId(inputEdge.source, nodes));
  }

  const built = program.build();
  logger.flow.debug("Generated statements", { count: built.statements.length });
  return built;
}
