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
export type ResultVisualItem =
  | ResultVisualMontessori
  | ResultVisualForma
  | ResultVisualComida
  | ResultVisualCap
  | ResultVisualStick;

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
 * Mensaje para el aula, por código de error. El intérprete ya trae un detalle
 * exacto; esto lo traduce a algo que un niño pueda leer.
 */
const MESSAGE_BY_CODE: Record<ErrorCode, string> = {
  SYNTAX_ERROR: "Hay algo mal escrito en el programa.",
  DUPLICATE_IDENTIFIER: "Hay dos cartas con el mismo nombre.",
  UNDEFINED_REFERENCE: "Falta conectar una carta.",
  CIRCULAR_DEPENDENCY: "Las cartas se apuntan en círculo y no se puede empezar.",
  UNKNOWN_OPERATION: "Esa operación no existe.",
  ARITY_ERROR: "A esta operación le faltan o le sobran cartas.",
  TYPE_ERROR: "Esa carta no va en ese lugar.",
  INVALID_CRITERION: "Ese criterio no sirve para esta operación.",
  INVALID_OBJECT: "A esta carta le falta decir qué es.",
  EXPECTED_NUMBER: "Aquí hace falta un número.",
  DIVISION_BY_ZERO: "No se puede repartir entre cero.",
};

function describeError(error: DataflowError): string {
  const friendly = MESSAGE_BY_CODE[error.code];
  const where = error.nodeId ? ` (${error.nodeId})` : "";
  return `${friendly}${where} — ${error.detail}`;
}

function formatInterpreterErrors(errors: DataflowError[]): string {
  return errors
    .map((error, index) => `${errors.length > 1 ? `${index + 1}. ` : ""}${describeError(error)}`)
    .join("\n");
}

function logInterpreterErrors(errors: DataflowError[]): void {
  for (const error of errors) {
    logger.execute.error(`Interpreter ${error.phase} error`, {
      code: error.code,
      nodeId: error.nodeId,
      causeNodeId: error.causeNodeId,
      sinkId: error.sinkId,
      line: error.line,
      column: error.column,
      message: error.message,
    });
  }
}

export type ExecuteResult = {
  success: boolean;
  results?: Map<string, ResultValue>;
  error?: string;
};

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

function buildVisualStrip(entries: readonly Entry[]): ResultVisualItem[] {
  const strip: ResultVisualItem[] = [];

  for (const entry of entries) {
    const n = Math.max(0, Math.min(24, Math.round(getAmount(entry) || 0)));
    if (n === 0) continue;

    const color = entry.attributes.color ?? "verde";
    const size = entry.attributes.size ?? "mediano";

    for (let i = 0; i < n; i++) {
      if (strip.length >= MAX_VISUAL_UNITS) return strip;

      switch (entry.type) {
        case "montessori":
          strip.push({ kind: "montessori", color });
          break;
        case "forma":
          strip.push({ kind: "forma", subtype: entry.subtype, size, color });
          break;
        case "comida":
          strip.push({ kind: "comida", subtype: entry.subtype, color });
          break;
        case "cap":
          strip.push({ kind: "cap", color });
          break;
        case "stick":
          strip.push({ kind: "stick", color });
          break;
      }
    }
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
        return { success: false, error: "No hay nodos para ejecutar" };
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

        if (errors.length > 0) {
          logInterpreterErrors(errors);
          const errorMsg = formatInterpreterErrors(errors);
          logger.execute.error("Interpreter errors", { errorMsg });
          return { success: false, error: errorMsg };
        }

        const resultsMap = new Map<string, ResultValue>();

        for (const [resultId, output] of results) {
          const nodeId = resultId.startsWith("output_")
            ? resultId.replace("output_", "")
            : resultId;
          const converted = runtimeOutputToResultValue(output);
          if (converted) {
            resultsMap.set(nodeId, converted);
          } else {
            logger.execute.warn("Unknown output type", { nodeId, kind: output.kind });
          }
        }

        if (resultsMap.size === 0) {
          logger.execute.warn("No results found");
          return { success: false, error: "Sin resultados" };
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

        return { success: true, results: resultsMap };
      } catch (err) {
        logger.execute.error("Exception during execution", {
          error: err instanceof Error ? err.message : String(err),
          stack: err instanceof Error ? err.stack : undefined,
        });
        return {
          success: false,
          error: err instanceof Error ? err.message : "Error de ejecución",
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
