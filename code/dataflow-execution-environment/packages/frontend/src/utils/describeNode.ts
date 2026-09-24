/**
 * El nombre de un nodo tal como lo diría alguien mirando la mesa: "el número 7",
 * "la manzana roja", "el grupo de 2 estrellas", "la suma".
 *
 * El intérprete solo conoce identificadores; quien sabe que `card_4110` es una
 * manzana es el lienzo. Se apoya en `spanishGrammar` para que el género y el
 * número concuerden.
 */

import type { Edge } from "@xyflow/react";
import type { OperatorType } from "@/types/card-types";
import type { SourceFlowNodeData } from "@/components/dataflow/SourceFlowNode";
import type { DataflowNode } from "@/contexts/node/types";
import { getOrderedArrayZoneMembers } from "./arrayZoneGeometry";
import { describeCountedNoun, nounForm, nounGender } from "./spanishGrammar";
import { normalizeSize } from "./flowToProgram";

/** Sufijo del `source` sintético que declara el criterio de una carta de orden. */
const CRITERION_SUFFIX = "__criterio";
/** Prefijo del `sink` sintético de una carta de salida. */
const OUTPUT_PREFIX = "output_";

const OPERATOR_NAME: Record<string, string> = {
  adicion: "la suma",
  sustraccion: "la resta",
  multiplicacion: "la multiplicación",
  division: "la división",
  comparar: "la comparación",
  primero: "el primero",
  ultimo: "el último",
  contar: "la cuenta",
};

const ORDER_PROPERTY_NAME: Record<string, string> = {
  quantity: "cantidad",
  size: "tamaño",
  color: "color",
  subtype: "forma",
};

/** Para montessori, cap y stick el sustantivo es el tipo, no el color. */
const NOUN_BY_VARIANT: Record<string, string> = {
  montessori: "montessori",
  cap: "cap",
  stick: "stick",
};

/**
 * Del id que usa el intérprete al nodo del lienzo. Dos ids son sintéticos: el
 * `sink` de una salida y el `source` del criterio de una carta de orden, que no
 * es una carta — quien lo declara es el operador.
 */
export function canvasNodeIdOf(interpreterNodeId: string): string {
  if (interpreterNodeId.endsWith(CRITERION_SUFFIX)) {
    return interpreterNodeId.slice(0, -CRITERION_SUFFIX.length);
  }
  if (interpreterNodeId.startsWith(OUTPUT_PREFIX)) {
    return interpreterNodeId.slice(OUTPUT_PREFIX.length);
  }
  return interpreterNodeId;
}

/** El sustantivo de una carta de datos, con el que concuerdan artículo y adjetivos. */
function nounKeyOf(data: SourceFlowNodeData): string | null {
  switch (data.variant) {
    case "shape":
      return data.shape ?? "circulo";
    case "food":
      return data.food ?? "manzana";
    case "montessori":
    case "cap":
    case "stick":
      return NOUN_BY_VARIANT[data.variant];
    default:
      return null;
  }
}

function withArticle(nounKey: string, phrase: string): string {
  return `${nounGender(nounKey) === "f" ? "la" : "el"} ${phrase}`;
}

function describeSource(data: SourceFlowNodeData): string {
  if (data.variant === "number") return `el número ${data.value ?? 0}`;

  if (data.variant === "criteria") {
    const [property] = data.properties ?? [];
    const name = property ? (ORDER_PROPERTY_NAME[property] ?? property) : null;
    return name ? `el criterio de ${name}` : "el criterio";
  }

  const noun = nounKeyOf(data);
  if (!noun) return "la carta";

  return withArticle(
    noun,
    describeCountedNoun(noun, 1, {
      size: data.variant === "shape" ? normalizeSize(data.size) : undefined,
      color: "color" in data ? data.color : undefined,
    })
  );
}

function describeOperator(data: { operator?: OperatorType; criterio?: { property: string } }): string {
  const operator = data.operator ?? "adicion";

  if (operator.startsWith("filtrar")) return "el filtro";

  if (operator.startsWith("orden")) {
    const sentido = operator === "orden-mayor-menor" ? "de mayor a menor" : "de menor a mayor";
    const property = data.criterio?.property;
    const por = property ? ` por ${ORDER_PROPERTY_NAME[property] ?? property}` : "";
    return `el orden${por} ${sentido}`;
  }

  return OPERATOR_NAME[operator] ?? `la operación ${operator}`;
}

/** Cuántas cartas hay en una zona y, si todas son lo mismo, de qué. */
function describeGroup(nodeId: string, nodes: DataflowNode[], edges: Edge[]): string {
  const members = getOrderedArrayZoneMembers(nodeId, nodes, edges);
  if (members.length === 0) return "el grupo vacío";

  const nouns = new Set(members.map((member) => nounKeyOf(member.data as SourceFlowNodeData)));
  const [noun] = [...nouns];
  const key = nouns.size === 1 && noun ? noun : "carta";

  return `el grupo de ${members.length} ${nounForm(key, members.length)}`;
}

/**
 * El nombre del nodo para enseñárselo a alguien, o `null` si ese id no
 * corresponde a ninguna carta de la mesa (pasa con los fallos internos, donde el
 * id ni siquiera existe).
 */
export function describeNode(
  interpreterNodeId: string | undefined,
  nodes: DataflowNode[],
  edges: Edge[]
): string | null {
  if (!interpreterNodeId) return null;

  const nodeId = canvasNodeIdOf(interpreterNodeId);
  const node = nodes.find((candidate) => candidate.id === nodeId);
  if (!node) return null;

  switch (node.type) {
    case "source":
      return describeSource(node.data as SourceFlowNodeData);
    case "operator":
      return describeOperator(node.data as { operator?: OperatorType });
    case "arrayClose":
      return describeGroup(node.id, nodes, edges);
    case "diceZone":
      return "el dado";
    case "programOutput":
      return "la salida";
    default:
      return null;
  }
}
