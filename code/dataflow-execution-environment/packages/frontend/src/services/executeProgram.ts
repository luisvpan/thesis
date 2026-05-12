/**
 * Service for executing dataflow programs client-side using the interpreter.
 */

import {
  Interpreter,
  RuntimeError,
  type ParseError,
} from "@dataflow/interpreter";
import { flowToProgram } from "@/utils/flowToProgram";
import type { DataflowNode } from "@/contexts/NodeContext";
import type { Edge } from "@xyflow/react";

// ============================================================================
// Tipos para agrupación jerárquica de resultados
// ============================================================================

interface SubtypeGroup {
  subtype: string;
  items: Array<{
    size?: string;
    color?: string;
    amount: number;
  }>;
  totalAmount: number;
}

interface TypeGroup {
  type: string;
  subtypes: SubtypeGroup[];
  totalAmount: number;
  // Para racionales, guardamos el valor directo
  rationalValue?: number;
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
}

/** Unidad visual en la carta de salida (orden = orden del arreglo aplanado). */
export type ResultVisualMontessori = { kind: "montessori"; color: string };
export type ResultVisualForma = { kind: "forma"; subtype: string; size: string };
export type ResultVisualComida = { kind: "comida"; subtype: string; color: string };
export type ResultVisualItem =
  | ResultVisualMontessori
  | ResultVisualForma
  | ResultVisualComida;

export type ResultValue =
  | { kind: "number"; value: number }
  | { kind: "semantic"; result: SemanticResult };

function formatInterpreterErrors(
  errors: Array<ParseError | RuntimeError>
): string {
  return errors
    .map((e, i) => {
      const prefix = errors.length > 1 ? `${i + 1}. ` : "";
      if (e instanceof RuntimeError) {
        return `${prefix}${e.message}`;
      }
      let s = `${prefix}${e.message}`;
      if (e.line !== undefined) {
        s += ` (línea ${e.line}`;
        if (e.column !== undefined) s += `, columna ${e.column}`;
        s += ")";
      }
      return s;
    })
    .join("\n");
}

function logInterpreterErrors(errors: Array<ParseError | RuntimeError>): void {
  for (const e of errors) {
    if (e instanceof RuntimeError) {
      console.error("[execute] Interpreter RuntimeError:", {
        code: e.code,
        nodeId: e.nodeId,
        message: e.message,
        stack: e.stack,
      });
    } else {
      console.error("[execute] Interpreter ParseError:", {
        message: e.message,
        line: e.line,
        column: e.column,
      });
    }
  }
}

export type ExecuteResult = {
  success: boolean;
  results?: Map<string, ResultValue>;
  error?: string;
};

// ============================================================================
// Helpers para extraer información de RuntimeValue
// ============================================================================

function getAmount(elem: unknown): number {
  if (!elem || typeof elem !== "object") return 1;
  const obj = elem as Record<string, unknown>;

  if (obj.kind === "racional" && obj.value) {
    return Number((obj.value as { valueOf(): number }).valueOf());
  }
  if (obj.kind === "abstracto" && obj.value) {
    return Number((obj.value as { valueOf(): number }).valueOf());
  }
  if ((obj.kind === "forma" || obj.kind === "comida" || obj.kind === "montessori") && obj.amount) {
    return Number((obj.amount as { valueOf(): number }).valueOf());
  }
  return 1;
}

function getCategory(elem: unknown): "abstracto" | "pictorico" | "concreto" {
  if (!elem || typeof elem !== "object") return "abstracto";
  const obj = elem as Record<string, unknown>;

  if (obj.kind === "racional" || obj.kind === "abstracto") return "abstracto";
  if (obj.kind === "forma") return "pictorico";
  if (obj.kind === "comida" || obj.kind === "montessori") return "concreto";
  return "abstracto";
}

function getType(elem: unknown): string {
  if (!elem || typeof elem !== "object") return "racional";
  const obj = elem as Record<string, unknown>;

  if (obj.kind === "racional" || obj.kind === "abstracto") return "racional";
  if (obj.kind === "forma") return "forma";
  if (obj.kind === "comida") return "comida";
  if (obj.kind === "montessori") return "montessori";
  return "racional";
}

function getSubtype(elem: unknown): string | null {
  if (!elem || typeof elem !== "object") return null;
  const obj = elem as Record<string, unknown>;

  if (obj.kind === "forma" || obj.kind === "comida") {
    return (obj.subtype as string) ?? null;
  }
  return null;
}

// ============================================================================
// Pluralización simple para español
// ============================================================================

const PLURALS: Record<string, string> = {
  forma: "formas",
  comida: "comidas",
  montessori: "montessoris",
  racional: "racionales",
  cuadrado: "cuadrados",
  circulo: "círculos",
  triangulo: "triángulos",
  rectangulo: "rectángulos",
  rombo: "rombos",
  estrella: "estrellas",
  trapecio: "trapecios",
  uva: "uvas",
  pera: "peras",
  manzana: "manzanas",
  hamburguesa: "hamburguesas",
  pasta: "pastas",
};

function pluralize(word: string, count: number): string {
  if (count === 1) return word;
  return PLURALS[word] ?? word + "s";
}

// ============================================================================
// Tira visual (cubos Montessori, etc.) — orden del arreglo en runtime
// ============================================================================

const MAX_VISUAL_UNITS = 48;

function flattenRuntimeElements(elements: unknown[]): unknown[] {
  const out: unknown[] = [];
  for (const el of elements) {
    if (!el || typeof el !== "object") continue;
    const o = el as Record<string, unknown>;
    if (o.kind === "arreglo" && Array.isArray(o.elements)) {
      out.push(...flattenRuntimeElements(o.elements));
    } else {
      out.push(el);
    }
  }
  return out;
}

function buildVisualStrip(elements: unknown[]): ResultVisualItem[] {
  const flat = flattenRuntimeElements(elements);
  const strip: ResultVisualItem[] = [];

  for (const elem of flat) {
    if (!elem || typeof elem !== "object") continue;
    const o = elem as Record<string, unknown>;
    const rawAmt = getAmount(elem);
    const n = Math.max(0, Math.min(24, Math.round(Number(rawAmt) || 0)));
    if (n === 0) continue;

    if (o.kind === "montessori") {
      const color = String(o.color ?? "verde");
      for (let i = 0; i < n; i++) {
        if (strip.length >= MAX_VISUAL_UNITS) return strip;
        strip.push({ kind: "montessori", color });
      }
    } else if (o.kind === "forma") {
      const subtype = String(o.subtype ?? "forma");
      const size = String(o.size ?? "mediano");
      for (let i = 0; i < n; i++) {
        if (strip.length >= MAX_VISUAL_UNITS) return strip;
        strip.push({ kind: "forma", subtype, size });
      }
    } else if (o.kind === "comida") {
      const subtype = String(o.subtype ?? "comida");
      const color = String(o.color ?? "verde");
      for (let i = 0; i < n; i++) {
        if (strip.length >= MAX_VISUAL_UNITS) return strip;
        strip.push({ kind: "comida", subtype, color });
      }
    }
  }

  return strip;
}

// ============================================================================
// Agrupación jerárquica de elementos
// ============================================================================

function groupElements(elements: unknown[]): SemanticResult {
  const categoryMap = new Map<string, CategoryGroup>();
  let totalAmount = 0;

  for (const elem of elements) {
    const amount = getAmount(elem);
    totalAmount += amount;

    const category = getCategory(elem);
    const type = getType(elem);
    const subtype = getSubtype(elem);

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

    // Para racionales, acumulamos el valor
    if (type === "racional") {
      typeGroup.rationalValue = (typeGroup.rationalValue ?? 0) + amount;
    }

    // Crear o actualizar subtipo (para forma/comida/montessori)
    const obj = elem as Record<string, unknown>;

    // Montessori no tiene subtype, pero agrupamos por color
    const effectiveSubtype = subtype ?? (obj.kind === "montessori" ? (obj.color as string) : null);

    if (effectiveSubtype) {
      let subtypeGroup = typeGroup.subtypes.find((s) => s.subtype === effectiveSubtype);
      if (!subtypeGroup) {
        subtypeGroup = { subtype: effectiveSubtype, items: [], totalAmount: 0 };
        typeGroup.subtypes.push(subtypeGroup);
      }
      subtypeGroup.totalAmount += amount;

      subtypeGroup.items.push({
        size: obj.kind === "forma" ? (obj.size as string) : undefined,
        color: (obj.kind === "comida" || obj.kind === "montessori") ? (obj.color as string) : undefined,
        amount,
      });
    }
  }

  const categories = Array.from(categoryMap.values());
  const description = generateDescription(categories, totalAmount);
  const visualStrip = buildVisualStrip(elements);

  return { categories, totalAmount, description, visualStrip };
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
  const typeName = pluralize(type.type, type.totalAmount);

  // Racional: mostrar el valor
  if (type.type === "racional") {
    const val = type.rationalValue ?? type.totalAmount;
    if (type.totalAmount === 1) {
      return `el número ${val}`;
    }
    return `${type.totalAmount} ${typeName} (suma: ${val})`;
  }

  // Sin subtipos (no debería pasar para forma/comida, pero por seguridad)
  if (type.subtypes.length === 0) {
    return `${type.totalAmount} ${typeName}`;
  }

  // Un solo subtipo
  if (type.subtypes.length === 1) {
    return describeSubtype(type.subtypes[0]);
  }

  // Múltiples subtipos
  const subDescs = type.subtypes.map((s) => describeSubtype(s));
  return `${type.totalAmount} ${typeName}: ${subDescs.join(", ")}`;
}

function describeSubtype(sub: SubtypeGroup): string {
  const name = pluralize(sub.subtype, sub.totalAmount);

  // Un solo item: incluir tamaño/color
  if (sub.items.length === 1) {
    const item = sub.items[0];
    if (item.size) return `${sub.totalAmount} ${name} ${item.size}`;
    if (item.color) return `${sub.totalAmount} ${name} ${item.color}`;
    return `${sub.totalAmount} ${name}`;
  }

  // Múltiples items del mismo subtipo con diferentes atributos
  if (sub.items.some((i) => i.size)) {
    const sizeDescs = sub.items.map((i) => `${i.amount} ${i.size}`);
    return `${sub.totalAmount} ${name} (${sizeDescs.join(", ")})`;
  }

  if (sub.items.some((i) => i.color)) {
    const colorDescs = sub.items.map((i) => `${i.amount} ${i.color}`);
    return `${sub.totalAmount} ${name} (${colorDescs.join(", ")})`;
  }

  return `${sub.totalAmount} ${name}`;
}

// ============================================================================
// Executor principal
// ============================================================================

/**
 * Creates a program executor with its own Interpreter instance.
 * The interpreter maintains cache between executions within the same session.
 *
 * Call `reset()` when leaving the page or when you want to clear the cache.
 */
export function createProgramExecutor() {
  const interpreter = new Interpreter();

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
      
      const program = flowToProgram(nodes, edges);
      console.log("[frontend → interpreter] Program:", program);
      console.log(
        "[execute] Program with",
        program.statements.length,
        "statements"
      );

      try {
        const { results, errors } = await interpreter.execute(program);

        console.log(
          "[execute] Interpreter results:",
          results.size,
          "errors:",
          errors.length
        );

        if (errors.length > 0) {
          logInterpreterErrors(errors);
          const errorMsg = formatInterpreterErrors(errors);
          console.error("[execute] Interpreter errors (texto):", errorMsg);
          return { success: false, error: errorMsg };
        }

        // Collect all sink results
        const resultsMap = new Map<string, ResultValue>();

        console.log("[execute] Processing results, size:", results.size);
        for (const [sinkId, output] of results) {
          console.log(`[execute] Result: sinkId="${sinkId}", kind="${output.kind}"`);
          if (!sinkId.startsWith("output_")) {
            console.log(
              `[execute] Skipping sinkId="${sinkId}" (no output_ prefix)`
            );
            continue;
          }

          // Extract the programOutput node ID: "output_card_76" → "card_76"
          const outputNodeId = sinkId.replace("output_", "");

          if (output.kind === "racional") {
            const value = Number(output.value.valueOf());
            console.log(`[execute] Setting ${outputNodeId} = ${value} (racional)`);
            resultsMap.set(outputNodeId, { kind: "number", value });
          } else if (output.kind === "arreglo") {
            // Agrupar elementos jerárquicamente
            const semantic = groupElements(output.elements);
            console.log(
              `[execute] Setting ${outputNodeId} = "${semantic.description}" (arreglo)`
            );
            resultsMap.set(outputNodeId, { kind: "semantic", result: semantic });
          } else if (output.kind === "forma" || output.kind === "comida") {
            // Objeto individual
            const semantic = groupElements([output]);
            console.log(
              `[execute] Setting ${outputNodeId} = "${semantic.description}" (${output.kind})`
            );
            resultsMap.set(outputNodeId, { kind: "semantic", result: semantic });
          } else if (output.kind === "abstracto") {
            const value = Number(
              (output as { value: { valueOf(): number } }).value.valueOf()
            );
            console.log(
              `[execute] Setting ${outputNodeId} = ${value} (abstracto)`
            );
            resultsMap.set(outputNodeId, { kind: "number", value });
          } else {
            console.log(
              `[execute] Unknown output type for ${outputNodeId}: kind="${output.kind}"`
            );
          }
        }

        console.log("[execute] Final resultsMap size:", resultsMap.size);

        if (resultsMap.size === 0) {
          console.warn("[execute] No results found, returning error");
          return { success: false, error: "Sin resultados" };
        }

        return { success: true, results: resultsMap };
      } catch (err) {
        console.error("[execute] Excepción al ejecutar:", err);
        if (err instanceof Error && err.stack) {
          console.error("[execute] Stack:", err.stack);
        }
        return {
          success: false,
          error: err instanceof Error ? err.message : "Error de ejecución",
        };
      }
    },

    /**
     * Clears the interpreter cache.
     * Call this when leaving the page or when you need a fresh state.
     */
    reset() {
      interpreter.reset();
      console.log("[execute] Interpreter reset");
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
