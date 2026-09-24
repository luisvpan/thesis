/**
 * Service for executing dataflow programs client-side using the interpreter.
 */

import {
  Interpreter,
  isBag,
  isBoolean,
  type DataflowError,
  type Entry,
  type ErrorCode,
  type RuntimeValue,
} from "@dataflow/interpreter";
import { flowToProgram } from "@/utils/flowToProgram";
import type { DataflowNode } from "@/contexts/NodeContext";
import { dataForProgramHash } from "@/contexts/node/visionNodeMeta";
import type { Edge } from "@xyflow/react";
import { logger } from "@/lib/logger";
import { jsonReplacer, toJsonSafe } from "@/utils/jsonReplacer";
import { describeCountedNoun, nounForm } from "@/utils/spanishGrammar";
import { describeNode } from "@/utils/describeNode";
import { isDrawableFraction } from "@/components/dataflow/fractionGeometry";

// ============================================================================
// Tipos para agrupación jerárquica de resultados
// ============================================================================

interface SubtypeGroup {
  subtype: string;
  items: Array<{
    size?: string;
    color?: string;
    amount: number;
    fractionStr: string;
  }>;
  totalAmount: number;
}

interface TypeGroup {
  type: string;
  subtypes: SubtypeGroup[];
  totalAmount: number;
  // Para números abstractos, guardamos el valor directo
  rationalValue?: number;
  rationalFractionStr?: string;
}

interface CategoryGroup {
  category: "abstracto" | "pictorico" | "concreto";
  types: TypeGroup[];
  totalAmount: number;
}

interface SemanticResult {
  categories: CategoryGroup[];
  totalAmount: number;
  description: string;
  /** Orden de aparición para la tira gráfica bajo el texto (cubos / iconos). */
  visualStrip: ResultVisualItem[];
  /** Elementos originales sin expandir, para re-ordenamiento en frontend. */
  originalElements: unknown[];
}

/** Unidad visual en la carta de salida (orden = orden del arreglo aplanado). */
export type ResultVisualMontessori = { kind: "montessori"; color: string };
export type ResultVisualForma = {
  kind: "forma";
  subtype: string;
  size: string;
  color?: string;
};
export type ResultVisualComida = { kind: "comida"; subtype: string; color: string };
export type ResultVisualCap = { kind: "cap"; color: string };
export type ResultVisualStick = { kind: "stick"; color: string };
/** La porción presente de un objeto incompleto: `n` de `d` regiones. */
export type VisualFraction = { numerator: number; denominator: number };

export type ResultVisualItem = (
  | ResultVisualMontessori
  | ResultVisualForma
  | ResultVisualComida
  | ResultVisualCap
  | ResultVisualStick
) & {
  /** Si está, el objeto se pinta incompleto en vez de entero. */
  fraction?: VisualFraction;
};

/** Metadata for single CPA object rendering */
export type SingleCpaObjectMeta = {
  type: string;      // "cap", "stick", "montessori", "forma", "comida"
  subtype: string;
  color: string;
  size?: string;
  quantity: number;
  // For exact fraction display (e.g., "13/4" instead of 3.25)
  numerator: string;
  denominator: string;
};

/** Single number value in an ordered array */
export type NumberArrayItem = {
  value: number;
  numerator: string;
  denominator: string;
};

export type ResultValue =
  | { kind: "number"; value: number; numerator?: string; denominator?: string }
  | { kind: "numberArray"; values: NumberArrayItem[] }
  | { kind: "boolean"; value: boolean }
  | {
      kind: "semantic";
      result: SemanticResult;
      isSingleCpaObject?: boolean;
      singleCpaObjectMeta?: SingleCpaObjectMeta;
    };

/**
 * Qué pasó y qué hacer, por código de error. El intérprete ya trae un detalle
 * exacto; esto lo traduce a algo que un niño pueda leer y arreglar.
 *
 * Son los once códigos, con el compilador comprobando que no falte ninguno.
 */
const ERROR_BY_CODE: Record<ErrorCode, { message: string; hint: string }> = {
  SYNTAX_ERROR: {
    message: "Hay algo mal escrito en el programa.",
    hint: "Esto no debería pasar con las cartas: avisa a quien cuide el programa.",
  },
  DUPLICATE_IDENTIFIER: {
    message: "Hay dos cartas con el mismo nombre.",
    hint: "Quita una de las dos de la mesa.",
  },
  UNDEFINED_REFERENCE: {
    message: "Falta conectar una carta.",
    hint: "Esto no debería pasar con las cartas: avisa a quien cuide el programa.",
  },
  CIRCULAR_DEPENDENCY: {
    message: "Las cartas se apuntan en círculo y no se puede empezar.",
    hint: "Quita una de las flechas del círculo para que haya un principio.",
  },
  UNKNOWN_OPERATION: {
    message: "Esa operación no existe.",
    hint: "Cambia la carta de operación por una de las del mazo.",
  },
  ARITY_ERROR: {
    message: "A esta operación le faltan o le sobran cartas.",
    hint: "Mira cuántas cartas necesita y conéctale las que le falten.",
  },
  TYPE_ERROR: {
    message: "Esa carta no va en ese lugar.",
    hint: "Prueba con otra carta: ahí no encaja la que hay puesta.",
  },
  INVALID_CRITERION: {
    message: "Ese criterio no sirve para esta operación.",
    hint: "Para filtrar hacen falta cartas de filtro, y para ordenar, de orden.",
  },
  INVALID_OBJECT: {
    message: "A esta carta le falta decir qué es.",
    hint: "Ponla de nuevo sobre la mesa para que la cámara la lea bien.",
  },
  EXPECTED_NUMBER: {
    message: "Aquí hace falta un número.",
    hint: "Cambia esa carta por una de número.",
  },
  DIVISION_BY_ZERO: {
    message: "No se puede dividir entre cero.",
    hint: "Cambia el cero por otro número.",
  },
};

/** Nombra un nodo como se vería en la mesa; sin lienzo delante, no nombra nada. */
export type NodeNamer = (nodeId: string | undefined) => string | null;

const NO_NAMES: NodeNamer = () => null;

/**
 * Lo que se pinta en la carta de salida: la frase de aula y, si se sabe, la
 * carta culpable por su nombre. El detalle exacto del intérprete se queda en el
 * log, que es donde sirve.
 */
function describeError(error: DataflowError, nameOf: NodeNamer): string {
  const { message } = ERROR_BY_CODE[error.code];
  const culprit = nameOf(error.causeNodeId ?? error.nodeId);
  return culprit ? `${message} (${culprit})` : message;
}

function formatInterpreterErrors(errors: DataflowError[]): string {
  return errors
    .map(
      (error, index) =>
        `${errors.length > 1 ? `${index + 1}. ` : ""}${describeError(error, NO_NAMES)}`
    )
    .join("\n");
}

function logInterpreterErrors(errors: DataflowError[]): void {
  for (const error of errors) {
    logger.execute.error(`Interpreter ${error.phase} error`, {
      code: error.code,
      nodeId: error.nodeId,
      causeNodeId: error.causeNodeId,
      sinkIds: error.sinkIds,
      line: error.line,
      column: error.column,
      message: error.message,
    });
  }
}

/** Un error ya traducido, en forma plana para poder viajar en `node.data`. */
export type OutputErrorInfo = {
  code: ErrorCode;
  /** Lo que se pinta en la carta. */
  text: string;
  /** La solución probable, para el popup de la salida. */
  hint: string;
  /** El nodo donde ocurrió y el que lo causó, para señalarlos en el lienzo. */
  nodeId?: string;
  causeNodeId?: string;
  /** Sus nombres en la mesa, ya redactados. */
  nodeName?: string;
  causeName?: string;
};

export type ExecuteResult = {
  /** No falló ninguna salida ni hubo errores de programa. */
  success: boolean;
  results: Map<string, ResultValue>;
  /** Errores por carta de salida (id del nodo `programOutput` del lienzo). */
  errorsByOutput: Map<string, OutputErrorInfo[]>;
  /**
   * Lo que no pertenece a ninguna salida: un error de sintaxis o un fallo
   * interno. En el lienzo no debería ocurrir nunca.
   */
  programError: string | null;
};

const EMPTY_RESULT: Omit<ExecuteResult, "programError" | "success"> = {
  results: new Map(),
  errorsByOutput: new Map(),
};

/** Del id del sink del intérprete (`output_x`) al id del nodo en el lienzo. */
function outputNodeIdOf(sinkId: string): string {
  return sinkId.startsWith("output_") ? sinkId.slice("output_".length) : sinkId;
}

/**
 * Reparte cada error entre las salidas a las que apaga. Un error sin salidas
 * —sintaxis, o un fallo nuestro— no es de nadie y sube como error de programa.
 */
function routeErrors(
  errors: DataflowError[],
  nameOf: NodeNamer
): {
  byOutput: Map<string, OutputErrorInfo[]>;
  orphans: DataflowError[];
} {
  const byOutput = new Map<string, OutputErrorInfo[]>();
  const orphans: DataflowError[] = [];

  for (const error of errors) {
    if (error.sinkIds.length === 0) {
      orphans.push(error);
      continue;
    }

    const info: OutputErrorInfo = {
      code: error.code,
      text: describeError(error, nameOf),
      hint: ERROR_BY_CODE[error.code].hint,
      nodeId: error.nodeId,
      causeNodeId: error.causeNodeId,
      nodeName: nameOf(error.nodeId) ?? undefined,
      causeName: nameOf(error.causeNodeId) ?? undefined,
    };

    for (const sinkId of error.sinkIds) {
      const nodeId = outputNodeIdOf(sinkId);
      const list = byOutput.get(nodeId);
      if (list) list.push(info);
      else byOutput.set(nodeId, [info]);
    }
  }

  return { byOutput, orphans };
}

/**
 * Extracts numerator and denominator from a Fraction.js object.
 * Handles BigInt properties (.n, .d, .s) and falls back to toFraction() or decimal.
 */
function extractFraction(
  qty: { valueOf(): number | bigint; n?: unknown; d?: unknown; s?: unknown; toFraction?: () => string }
): { numerator: string; denominator: string } {
  // Fraction.js stores n, d, s as bigints
  if (typeof qty.n === "bigint" && typeof qty.d === "bigint") {
    const sign = (qty.s as bigint) === -1n ? -1n : 1n;
    return {
      numerator: String((qty.n as bigint) * sign),
      denominator: String(qty.d),
    };
  }

  // Fallback: try toFraction() method if available
  if (typeof qty.toFraction === "function") {
    const frac = qty.toFraction();
    const parts = frac.split("/");
    return {
      numerator: parts[0],
      denominator: parts[1] ?? "1",
    };
  }

  // Last resort: use decimal value
  const value = Number(qty.valueOf());
  return {
    numerator: String(value),
    denominator: "1",
  };
}

function isAbstractNumber(entry: Entry): boolean {
  return entry.category === "abstracto" && entry.type === "numero";
}

/**
 * Traduce un valor del intérprete a lo que sabe pintar la interfaz.
 *
 * Solo hay tres formas de valor: la bolsa (los datos), el criterio y el
 * booleano. La bolsa vacía es `nulo`.
 */
function runtimeOutputToResultValue(output: RuntimeValue): ResultValue | undefined {
  if (isBoolean(output)) {
    return { kind: "boolean", value: output.value };
  }

  if (!isBag(output)) return undefined; // un criterio no es un resultado que mostrar

  const entries = output.entries;

  // Una sola entrada numérica: el número pelado.
  if (entries.length === 1 && isAbstractNumber(entries[0])) {
    const quantity = entries[0].quantity;
    return {
      kind: "number",
      value: Number(quantity.valueOf()),
      ...extractFraction(quantity),
    };
  }

  // Varias entradas numéricas: se conserva el orden, no se agregan.
  if (entries.length > 0 && entries.every(isAbstractNumber)) {
    const values: NumberArrayItem[] = entries.map((entry) => ({
      value: Number(entry.quantity.valueOf()),
      ...extractFraction(entry.quantity),
    }));
    return { kind: "numberArray", values };
  }

  const semantic = groupEntries(entries);

  // Un solo objeto: la interfaz lo pinta como carta, no como grupo.
  if (entries.length === 1) {
    const entry = entries[0];
    return {
      kind: "semantic",
      result: semantic,
      isSingleCpaObject: true,
      singleCpaObjectMeta: {
        type: entry.type,
        subtype: entry.subtype,
        color: entry.attributes.color ?? "",
        // El tamaño lo usa el encabezado del resultado único para concordar el
        // adjetivo ("1 estrella grande").
        size: entry.attributes.size,
        quantity: Number(entry.quantity.valueOf()),
        ...extractFraction(entry.quantity),
      },
    };
  }

  return { kind: "semantic", result: semantic };
}

// ============================================================================
// Helpers para leer una entrada de la bolsa
// ============================================================================

function getAmount(entry: Entry): number {
  return Number(entry.quantity.valueOf());
}

function getFractionString(entry: Entry): string {
  const qty = entry.quantity as unknown as { n?: unknown; d?: unknown; s?: unknown; valueOf(): number };

  // Fraction.js BigInt properties
  if (typeof qty.n === "bigint" && typeof qty.d === "bigint") {
    const sign = (qty.s as bigint) === -1n ? "-" : "";
    return qty.d === 1n ? `${sign}${qty.n}` : `${sign}${qty.n}/${qty.d}`;
  }

  return String(Number(qty.valueOf()));
}

// ============================================================================
// Sustantivos y concordancia de género/número para español
// ============================================================================
// (diccionario centralizado en @/utils/spanishGrammar, compartido con el
// encabezado del resultado único y el texto para TTS)

// Tipos cuyo sustantivo real no es el `subtype` que entrega el intérprete,
// sino el propio `type`: para montessori/cap/stick el "subtipo" agrupado es
// en realidad el color (no hay un subtipo semántico distinto del color).
const NOUN_BY_TYPE: Record<string, string> = {
  montessori: "montessori",
  cap: "cap",
  stick: "stick",
};

/** Clave de sustantivo real para un grupo de subtipo dado su tipo padre. */
function subtypeNounKey(typeKey: string, subtype: string): string {
  return NOUN_BY_TYPE[typeKey] ?? subtype;
}

/** Para montessori/cap/stick, la clave de agrupación ES el color. */
function impliedColor(typeKey: string, subtype: string): string | undefined {
  return NOUN_BY_TYPE[typeKey] ? subtype : undefined;
}

// ============================================================================
// Tira visual (cubos Montessori, etc.) — orden del arreglo en runtime
// ============================================================================

const MAX_VISUAL_UNITS = 48;

/** El glifo que le corresponde a una entrada, sin cantidad todavía. */
function visualItemOf(entry: Entry): ResultVisualItem | null {
  const color = entry.attributes.color ?? "verde";
  const size = entry.attributes.size ?? "mediano";

  switch (entry.type) {
    case "montessori":
      return { kind: "montessori", color };
    case "forma":
      return { kind: "forma", subtype: entry.subtype, size, color };
    case "comida":
      return { kind: "comida", subtype: entry.subtype, color };
    case "cap":
      return { kind: "cap", color };
    case "stick":
      return { kind: "stick", color };
    default:
      return null;
  }
}

/**
 * Lo que sobra tras los objetos enteros, como `n` de `d` regiones. `null` si la
 * cantidad es exacta, si es negativa —no hay objeto incompleto que pintar— o si
 * el denominador se dispara: ahí la fracción solo se lee en el texto.
 */
function fractionalPart(entry: Entry): VisualFraction | null {
  const qty = entry.quantity as unknown as { n?: unknown; d?: unknown; s?: unknown };
  if (typeof qty.n !== "bigint" || typeof qty.d !== "bigint") return null;
  if (qty.s === -1n) return null;

  const numerator = Number(qty.n % qty.d);
  const denominator = Number(qty.d);

  return isDrawableFraction(numerator, denominator) ? { numerator, denominator } : null;
}

function buildVisualStrip(entries: readonly Entry[]): ResultVisualItem[] {
  const strip: ResultVisualItem[] = [];

  for (const entry of entries) {
    const item = visualItemOf(entry);
    if (!item) continue;

    // Los enteros primero y el resto al final, como un número mixto: 3/2
    // manzanas son una manzana y otra a la que le falta la mitad.
    const whole = Math.max(0, Math.min(24, Math.floor(getAmount(entry) || 0)));

    for (let i = 0; i < whole; i++) {
      if (strip.length >= MAX_VISUAL_UNITS) return strip;
      strip.push({ ...item });
    }

    const fraction = fractionalPart(entry);
    if (fraction && strip.length < MAX_VISUAL_UNITS) strip.push({ ...item, fraction });
  }

  return strip;
}

// ============================================================================
// Agrupación jerárquica de elementos
// ============================================================================

function groupEntries(entries: readonly Entry[]): SemanticResult {
  const categoryMap = new Map<string, CategoryGroup>();
  let totalAmount = 0;

  for (const entry of entries) {
    const amount = getAmount(entry);
    totalAmount += amount;

    const { category, type, subtype, attributes } = entry;

    // Crear o actualizar categoría
    if (!categoryMap.has(category)) {
      categoryMap.set(category, {
        category,
        types: [],
        totalAmount: 0,
      });
    }
    const catGroup = categoryMap.get(category)!;
    catGroup.totalAmount += amount;

    // Crear o actualizar tipo
    let typeGroup = catGroup.types.find((t) => t.type === type);
    if (!typeGroup) {
      typeGroup = { type, subtypes: [], totalAmount: 0 };
      catGroup.types.push(typeGroup);
    }
    typeGroup.totalAmount += amount;

    // Para números abstractos, acumulamos el valor
    if (type === "numero") {
      typeGroup.rationalValue = (typeGroup.rationalValue ?? 0) + amount;
      // Para un solo número, guardamos su fracción exacta
      // Usamos "" como marcador de "múltiples números"
      const fractionStr = getFractionString(entry);
      if (typeGroup.rationalFractionStr === undefined) {
        typeGroup.rationalFractionStr = fractionStr;
      } else if (typeGroup.rationalFractionStr !== "") {
        // Segundo número: marcamos como múltiples (usaremos decimal)
        typeGroup.rationalFractionStr = "";
      }
    }

    // Montessori, cap y stick usan color como subtype efectivo
    const effectiveSubtype = subtype || attributes.color || null;

    if (effectiveSubtype) {
      let subtypeGroup = typeGroup.subtypes.find((s) => s.subtype === effectiveSubtype);
      if (!subtypeGroup) {
        subtypeGroup = { subtype: effectiveSubtype, items: [], totalAmount: 0 };
        typeGroup.subtypes.push(subtypeGroup);
      }
      subtypeGroup.totalAmount += amount;

      subtypeGroup.items.push({
        size: attributes.size,
        color: attributes.color,
        amount,
        fractionStr: getFractionString(entry),
      });
    }
  }

  const categories = Array.from(categoryMap.values());

  return {
    categories,
    totalAmount,
    description: generateDescription(categories, totalAmount),
    visualStrip: buildVisualStrip(entries),
    originalElements: toJsonSafe([...entries]),
  };
}

// ============================================================================
// Generación de descripción textual
// ============================================================================

function generateDescription(
  categories: CategoryGroup[],
  total: number
): string {
  if (categories.length === 0) return "vacío";

  // Si solo hay una categoría con un tipo, simplificar
  if (categories.length === 1) {
    const cat = categories[0];
    if (cat.types.length === 1) {
      return describeType(cat.types[0]);
    }
    // Una categoría con múltiples tipos
    const typeDescs = cat.types.map((t) => describeType(t));
    return `${total} objetos: ${typeDescs.join(", ")}`;
  }

  // Múltiples categorías: "X objetos: ..."
  const parts = categories.map((cat) => {
    const typeDescs = cat.types.map((t) => describeType(t));
    return typeDescs.join(", ");
  });

  return `${total} objetos: ${parts.join("; ")}`;
}

function describeType(type: TypeGroup): string {
  // Número abstracto: mostrar el valor
  if (type.type === "numero") {
    const typeName = nounForm("numero", type.totalAmount);
    // Usar fracción exacta si tenemos un solo número, sino usar decimal
    const val = type.rationalFractionStr || String(type.rationalValue ?? type.totalAmount);
    // Un solo número: "el número X"
    if (type.rationalFractionStr && type.rationalFractionStr !== "") {
      return `el número ${val}`;
    }
    // Múltiples números: mostrar suma
    return `${type.totalAmount} ${typeName} (suma: ${val})`;
  }

  const typeName = nounForm(type.type, type.totalAmount);

  // Sin subtipos (no debería pasar para forma/comida, pero por seguridad)
  if (type.subtypes.length === 0) {
    return `${type.totalAmount} ${typeName}`;
  }

  // Un solo subtipo
  if (type.subtypes.length === 1) {
    return describeSubtype(type.type, type.subtypes[0]);
  }

  // Múltiples subtipos
  const subDescs = type.subtypes.map((s) => describeSubtype(type.type, s));
  return `${type.totalAmount} ${typeName}: ${subDescs.join(", ")}`;
}

function describeSubtype(typeKey: string, sub: SubtypeGroup): string {
  const nounKey = subtypeNounKey(typeKey, sub.subtype);
  const groupColor = impliedColor(typeKey, sub.subtype);

  // Un solo item: incluir tamaño/color, usar fracción exacta
  if (sub.items.length === 1) {
    const item = sub.items[0];
    const phrase = describeCountedNoun(nounKey, sub.totalAmount, {
      size: item.size,
      color: item.color ?? groupColor,
    });
    return `${item.fractionStr} ${phrase}`;
  }

  // Múltiples items del mismo subtipo: si comparten tamaño/color, resumir;
  // si no, detallar cada uno por separado.
  const sizes = new Set(sub.items.map((i) => i.size ?? ""));
  const colors = new Set(sub.items.map((i) => i.color ?? groupColor ?? ""));
  const uniform = sizes.size <= 1 && colors.size <= 1;

  if (uniform) {
    const [item] = sub.items;
    return describeCountedNoun(nounKey, sub.totalAmount, {
      size: item?.size,
      color: item?.color ?? groupColor,
    });
  }

  const itemDescs = sub.items.map((item) => {
    const phrase = describeCountedNoun(nounKey, item.amount, {
      size: item.size,
      color: item.color ?? groupColor,
    });
    return `${item.fractionStr} ${phrase}`;
  });
  return `${sub.totalAmount} ${nounForm(nounKey, sub.totalAmount)} (${itemDescs.join(", ")})`;
}

// ============================================================================
// Executor principal
// ============================================================================

/**
 * Hash del programa (nodos + aristas) para detectar cambios semánticos.
 * Excluye posición en lienzo y metadatos de visión que cambian cada frame WS.
 */
export function computeProgramHash(nodes: DataflowNode[], edges: Edge[]): string {
  const nodesKey = nodes
    .filter((n) => n.type === "source" || n.type === "operator" || n.type === "programOutput")
    .map(
      (n) =>
        `${n.id}:${n.type}:${JSON.stringify(dataForProgramHash(n.data), jsonReplacer)}`
    )
    .sort()
    .join("|");
  const edgesKey = edges
    .map((e) => `${e.source}->${e.target}:${e.sourceHandle ?? ""}-${e.targetHandle ?? ""}`)
    .sort()
    .join("|");
  return `${nodesKey}::${edgesKey}`;
}

/** @deprecated internal alias */
function hashProgram(nodes: DataflowNode[], edges: Edge[]): string {
  return computeProgramHash(nodes, edges);
}

function resultValueFingerprint(v: ResultValue): string {
  switch (v.kind) {
    case "number":
      return `num:${v.value}`;
    case "numberArray":
      return `arr:${v.values.map((item) => item.value).join(",")}`;
    case "boolean":
      return `bool:${v.value}`;
    case "semantic":
      return `sem:${v.result.description}`;
  }
}

/**
 * Helper to create a hash of results for change detection.
 */
function hashResults(results: Map<string, ResultValue>): string {
  return Array.from(results.entries())
    .map(([k, v]) => `${k}:${resultValueFingerprint(v)}`)
    .sort()
    .join("|");
}

/**
 * Creates a program executor with its own Interpreter instance.
 * The interpreter maintains cache between executions within the same session.
 *
 * Call `reset()` when leaving the page or when you want to clear the cache.
 */
export function createProgramExecutor() {
  const interpreter = new Interpreter();

  // Cache for change detection - only log when something actually changes
  let lastProgramHash: string | null = null;
  let lastResultsHash: string | null = null;

  return {
    /**
     * Executes a dataflow program built from ReactFlow nodes/edges.
     */
    async execute(
      nodes: DataflowNode[],
      edges: Edge[]
    ): Promise<ExecuteResult> {
      if (nodes.length === 0) {
        return { ...EMPTY_RESULT, success: false, programError: "No hay nodos para ejecutar" };
      }

      // Check if program changed
      const currentProgramHash = hashProgram(nodes, edges);
      const programChanged = currentProgramHash !== lastProgramHash;

      const program = flowToProgram(nodes, edges);

      // Only log program details when it actually changed
      if (programChanged) {
        logger.execute.info("Program changed", {
          statements: program.statements.length,
        });
        lastProgramHash = currentProgramHash;
      }

      try {
        const { results, errors } = await interpreter.execute(program);

        // Los errores ya no cortan: una salida rota no impide que las demás
        // muestren su valor, así que se reparten y se sigue. Los nodos se
        // nombran aquí, que es donde se tiene el lienzo delante.
        const { byOutput, orphans } = routeErrors(errors, (nodeId) =>
          describeNode(nodeId, nodes, edges)
        );
        if (errors.length > 0) {
          logInterpreterErrors(errors);
        }
        const programError = orphans.length > 0 ? formatInterpreterErrors(orphans) : null;

        const resultsMap = new Map<string, ResultValue>();

        for (const [resultId, output] of results) {
          const nodeId = outputNodeIdOf(resultId);
          const converted = runtimeOutputToResultValue(output);
          if (converted) {
            resultsMap.set(nodeId, converted);
          } else {
            logger.execute.warn("Unknown output type", { nodeId, kind: output.kind });
          }
        }

        if (resultsMap.size === 0 && errors.length === 0) {
          logger.execute.warn("No results found");
          return { ...EMPTY_RESULT, success: false, programError: "Sin resultados" };
        }

        // Check if results changed
        const currentResultsHash = hashResults(resultsMap);
        const resultsChanged = currentResultsHash !== lastResultsHash;

        if (resultsChanged) {
          // Log summary of what changed
          const summary = Array.from(resultsMap.entries()).map(([id, val]) => ({
            id,
            value: resultValueFingerprint(val),
          }));
          logger.execute.info("Results updated", { results: summary });
          lastResultsHash = currentResultsHash;
        }

        return {
          success: errors.length === 0,
          results: resultsMap,
          errorsByOutput: byOutput,
          programError,
        };
      } catch (err) {
        logger.execute.error("Exception during execution", {
          error: err instanceof Error ? err.message : String(err),
          stack: err instanceof Error ? err.stack : undefined,
        });
        return {
          ...EMPTY_RESULT,
          success: false,
          programError: err instanceof Error ? err.message : "Error de ejecución",
        };
      }
    },

    /**
     * Clears the interpreter cache and resets change detection.
     * Call this when leaving the page or when you need a fresh state.
     */
    reset() {
      interpreter.reset();
      lastProgramHash = null;
      lastResultsHash = null;
      logger.execute.info("Interpreter reset");
    },

    /**
     * Get the interpreter's evaluation stats (for debugging).
     */
    getStats() {
      return interpreter.getEvaluationStats();
    },
  };
}

export type ProgramExecutor = ReturnType<typeof createProgramExecutor>;

/**
 * @deprecated Use createProgramExecutor() instead for proper lifecycle management.
 * This function is kept for backwards compatibility.
 */
export async function executeProgram(
  nodes: DataflowNode[],
  edges: Edge[]
): Promise<ExecuteResult> {
  const executor = createProgramExecutor();
  return executor.execute(nodes, edges);
}
