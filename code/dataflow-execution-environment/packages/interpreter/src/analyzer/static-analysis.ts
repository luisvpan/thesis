// Errores estáticos — LANGUAGE_SPEC.md §4.2
//
// Se detectan sobre la estructura ya construida, sin evaluar. Un error estático
// invalida el programa completo: no llega a evaluarse ningún nodo.
//
// Dos reglas acotan qué se mira, para que un programa a medio construir no se
// invalide a sí mismo:
//
//  - Solo se validan los nodos que **alcanzan alguna salida**. Los que ningún
//    `sink` alcanza no participan en la evaluación, así que tampoco se validan:
//    una carta todavía sin cablear no rompe nada.
//  - Los nodos incompletos no cuentan como error (§2.5): su categoría es
//    desconocida y los chequeos que dependen de ella se saltan.

import { describeArity, isOperation, parameterAt, SIGNATURES } from "../operations/signatures";
import { DataflowError } from "../runtime/errors";
import type { DependencyGraph } from "../runtime/graph";
import { QUANTITY_PROPERTY } from "../runtime/criteria";
import type { CriterionSubtype, ValueCategory } from "../runtime/types";
import type { CriterionLiteral, DataLiteral, Literal, Statement } from "./ast";

interface InferredCategory {
  category: ValueCategory;
  /** Solo para criterios: el subtipo declarado (§1.3). */
  criterion?: CriterionSubtype;
}

export function analyze(graph: DependencyGraph): DataflowError[] {
  const reached = reachableFromSinks(graph);

  // La bien-formación va primero (§2.2): sin ella, el resto de los chequeos no
  // tiene sobre qué razonar.
  const structural = [
    ...duplicateIdentifiers(graph, reached),
    ...unresolvedReferences(graph, reached),
    ...cycles(graph),
  ];
  if (structural.length > 0) return structural;

  const errors: DataflowError[] = [];
  const categories = new Map<string, InferredCategory | undefined>();

  /**
   * La categoría de salida de un nodo, fijada por su sentencia: la de un
   * `source` por su literal, la de un `transform` por su operación y la de un
   * `sink` por su fuente. `undefined` = nodo incompleto o indeterminable.
   */
  const categoryOf = (nodeId: string): InferredCategory | undefined => {
    if (categories.has(nodeId)) return categories.get(nodeId);
    categories.set(nodeId, undefined); // corta la recursión

    const node = graph.nodes.get(nodeId);
    if (!node) return undefined;

    const inferred = categoryOfStatement(node.statement, categoryOf);
    categories.set(nodeId, inferred);
    return inferred;
  };

  for (const node of graph.nodes.values()) {
    if (!reached.has(node.id)) continue;
    const stmt = node.statement;

    if (stmt.type === "SourceStatement" && stmt.value) {
      errors.push(...invalidObjects(stmt.identifier, stmt.value));
      errors.push(...malformedCriteria(stmt.identifier, stmt.value));
      continue;
    }

    if (stmt.type !== "TransformStatement" || !stmt.operation) continue;

    // §4.2.4 Operación desconocida
    if (!isOperation(stmt.operation)) {
      errors.push(
        new DataflowError(
          "UNKNOWN_OPERATION",
          `La operación '${stmt.operation}' no existe`,
          { nodeId: stmt.identifier }
        )
      );
      continue;
    }

    const signature = SIGNATURES[stmt.operation];
    const args = stmt.arguments;

    // §4.2.5 Error de aridad
    if (args.length < signature.minArity || (signature.maxArity !== null && args.length > signature.maxArity)) {
      errors.push(
        new DataflowError(
          "ARITY_ERROR",
          `${stmt.operation} admite ${describeArity(signature)} argumentos, y recibió ${args.length}`,
          { nodeId: stmt.identifier }
        )
      );
      continue;
    }

    for (const [index, argument] of args.entries()) {
      const parameter = parameterAt(signature, index);
      if (!parameter) continue;

      const actual = categoryOf(argument.name);
      if (!actual) continue; // nodo incompleto: no invalida el programa (§2.5)

      // §4.2.6 Categoría de valor equivocada
      if (actual.category !== parameter.category) {
        errors.push(
          new DataflowError(
            "TYPE_ERROR",
            `${stmt.operation} espera ${article(parameter.category)} en la posición ${index + 1}, y '${argument.name}' es ${article(actual.category)}`,
            { nodeId: stmt.identifier, causeNodeId: argument.name }
          )
        );
        continue;
      }

      // §4.2.7 Criterio inadecuado (subtipo equivocado)
      if (parameter.criterion && actual.criterion && actual.criterion !== parameter.criterion) {
        errors.push(
          new DataflowError(
            "INVALID_CRITERION",
            `${stmt.operation} espera un criterio de ${name(parameter.criterion)} en la posición ${index + 1}, y '${argument.name}' es de ${name(actual.criterion)}`,
            { nodeId: stmt.identifier, causeNodeId: argument.name }
          )
        );
      }
    }
  }

  return errors;
}

// =============================================================================
// Alcance: qué nodos participan
// =============================================================================

/**
 * Los nodos que alcanza alguna salida, siguiendo las dependencias hacia atrás.
 * La marca de visitados hace que un ciclo no cuelgue el recorrido: el ciclo lo
 * reporta después su propio chequeo.
 */
function reachableFromSinks(graph: DependencyGraph): Set<string> {
  const reached = new Set<string>();
  const pending = [...graph.sinkIds];

  while (pending.length > 0) {
    const id = pending.pop()!;
    if (reached.has(id)) continue;
    reached.add(id);
    pending.push(...(graph.nodes.get(id)?.dependencies ?? []));
  }

  return reached;
}

// =============================================================================
// Bien-formación (§2.2)
// =============================================================================

/**
 * §4.2.1 Nombre duplicado.
 *
 * Como el grafo indexa por nombre, el alcance se decide por nombre: si alguna
 * salida depende de él, el programa es ambiguo; si no lo alcanza ninguna, da
 * igual cuál de las dos declaraciones habría ganado.
 */
function duplicateIdentifiers(graph: DependencyGraph, reached: Set<string>): DataflowError[] {
  return graph.duplicateIds
    .filter((id) => reached.has(id))
    .map(
      (id) =>
        new DataflowError("DUPLICATE_IDENTIFIER", `El nombre '${id}' está declarado más de una vez`, {
          nodeId: id,
        })
    );
}

/** §4.2.2 Referencia sin resolver */
function unresolvedReferences(graph: DependencyGraph, reached: Set<string>): DataflowError[] {
  const errors: DataflowError[] = [];

  for (const node of graph.nodes.values()) {
    if (!reached.has(node.id)) continue;
    for (const dependency of node.dependencies) {
      if (!graph.nodes.has(dependency)) {
        errors.push(
          new DataflowError("UNDEFINED_REFERENCE", `No existe ningún nodo llamado '${dependency}'`, {
            nodeId: node.id,
            causeNodeId: dependency,
          })
        );
      }
    }
  }

  return errors;
}

/** §4.2.3 Ciclo. Recorre desde las salidas, que es justo lo alcanzable. */
function cycles(graph: DependencyGraph): DataflowError[] {
  const errors: DataflowError[] = [];
  const visited = new Set<string>();
  const stack = new Set<string>();

  const walk = (nodeId: string, path: string[]): void => {
    if (stack.has(nodeId)) {
      const cycle = [...path.slice(path.indexOf(nodeId)), nodeId];
      errors.push(
        new DataflowError("CIRCULAR_DEPENDENCY", `Ciclo de dependencias: ${cycle.join(" → ")}`, {
          nodeId,
        })
      );
      return;
    }
    if (visited.has(nodeId)) return;

    visited.add(nodeId);
    stack.add(nodeId);

    for (const dependency of graph.nodes.get(nodeId)?.dependencies ?? []) {
      walk(dependency, [...path, nodeId]);
    }

    stack.delete(nodeId);
  };

  for (const nodeId of graph.sinkIds) {
    if (!visited.has(nodeId)) walk(nodeId, []);
  }

  return errors;
}

// =============================================================================
// Categoría de los valores (§4.2.6)
// =============================================================================

function categoryOfLiteral(literal: Literal): InferredCategory {
  return literal.type === "CriterionLiteral"
    ? { category: "criterio", criterion: literal.sourceType }
    : { category: "bolsa" };
}

function categoryOfStatement(
  stmt: Statement,
  categoryOf: (nodeId: string) => InferredCategory | undefined
): InferredCategory | undefined {
  switch (stmt.type) {
    case "SourceStatement":
      return stmt.value ? categoryOfLiteral(stmt.value) : undefined;

    case "TransformStatement":
      if (!stmt.operation || !isOperation(stmt.operation)) return undefined;
      return { category: SIGNATURES[stmt.operation].result };

    case "SinkStatement":
      return stmt.sourceIdentifier ? categoryOf(stmt.sourceIdentifier) : undefined;
  }
}

// =============================================================================
// Literales de un `source`
// =============================================================================

function dataLiteralsOf(literal: Literal): DataLiteral[] {
  if (literal.type === "DataLiteral") return [literal];
  if (literal.type === "GroupLiteral") return literal.elements;
  return [];
}

/** §4.2.8 Objeto inválido: un componente de identidad CPA en blanco. */
function invalidObjects(nodeId: string, literal: Literal): DataflowError[] {
  const errors: DataflowError[] = [];

  for (const data of dataLiteralsOf(literal)) {
    const missing = (
      [
        ["category", data.category],
        ["type", data.objType],
        ["subtype", data.subtype],
      ] as const
    )
      .filter(([, value]) => value === "")
      .map(([key]) => key);

    if (missing.length > 0) {
      errors.push(
        new DataflowError(
          "INVALID_OBJECT",
          `Un objeto de datos necesita ${missing.map((key) => `"${key}"`).join(", ")}: sin eso no denota una identidad real`,
          { nodeId }
        )
      );
    }
  }

  return errors;
}

/**
 * §4.2.7 Criterio inadecuado, en la forma de sus valores: un criterio de filtro
 * solo admite un valor único por propiedad, y solo sobre propiedades de
 * identidad — nunca sobre la cantidad (§1.3).
 */
function malformedCriteria(nodeId: string, literal: Literal): DataflowError[] {
  if (literal.type !== "CriterionLiteral" || literal.sourceType !== "filter") return [];

  const errors: DataflowError[] = [];
  const criterion: CriterionLiteral = literal;

  if (criterion.properties.includes(QUANTITY_PROPERTY)) {
    errors.push(
      new DataflowError(
        "INVALID_CRITERION",
        "Un criterio de filtro prueba la identidad, no la cantidad",
        { nodeId }
      )
    );
  }

  for (const value of criterion.values) {
    if (Array.isArray(value.value) && criterion.properties.includes(value.key)) {
      errors.push(
        new DataflowError(
          "INVALID_CRITERION",
          `Un criterio de filtro fija un valor único por propiedad, y "${value.key}" tiene varios`,
          { nodeId }
        )
      );
    }
  }

  return errors;
}

// =============================================================================
// Redacción
// =============================================================================

function article(category: ValueCategory): string {
  return category === "bolsa" ? "una bolsa" : category === "criterio" ? "un criterio" : "un booleano";
}

function name(subtype: CriterionSubtype): string {
  return subtype === "filter" ? "filtro" : "orden";
}
