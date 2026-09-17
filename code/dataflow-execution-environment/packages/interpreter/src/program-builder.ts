// Construcción de programas — LANGUAGE_SPEC.md §2.1
//
// Un programa es una secuencia de sentencias, cada una con un nombre único. El
// builder es inmutable, como la bolsa: cada método devuelve uno nuevo, de modo
// que se puede partir de una base común y ramificar sin sorpresas.

import type {
  Expression,
  Literal,
  Program,
  Statement,
} from "./program";
import type { Operation } from "./operations/signatures";

/** Una o más referencias a nodos, por nombre. */
type Refs = [string, ...string[]];

function identifiers(names: readonly string[]): Expression[] {
  return names.map((name) => ({ type: "Identifier", name }));
}

/**
 * Programa en construcción.
 *
 * @example
 * const programa = createProgram()
 *   .source("frutas", createBag().add({ … }))
 *   .source("porCantidad", createOrderCriterion({ properties: ["quantity"], values: { quantity: "asc" } }))
 *   .order("ordenadas", "frutas", "porCantidad")
 *   .first("primera", "ordenadas")
 *   .sink("salida", "primera")
 *   .build();
 */
export class ProgramBuilder {
  private readonly statements: readonly Statement[];

  private constructor(statements: readonly Statement[]) {
    this.statements = Object.freeze(statements);
    Object.freeze(this);
  }

  /** @internal */
  static empty(): ProgramBuilder {
    return new ProgramBuilder([]);
  }

  private with(statement: Statement): ProgramBuilder {
    return new ProgramBuilder([...this.statements, statement]);
  }

  // ===========================================================================
  // Sentencias (§2.1)
  // ===========================================================================

  /**
   * Un nodo de entrada: aporta datos (una bolsa) o un criterio. Sin valor
   * declara un nodo incompleto, que evalúa a `nulo` (§2.5).
   */
  source(identifier: string, value?: Literal): ProgramBuilder {
    return this.with({ type: "SourceStatement", identifier, value });
  }

  /**
   * Un nodo de proceso: aplica una operación a otros nodos, que se nombran.
   * Sin operación declara un nodo incompleto.
   *
   * Es la puerta para construir desde datos, cuando la operación se decide en
   * ejecución; si se conoce al escribir el código, los atajos de abajo
   * comprueban además la aridad.
   */
  transform(
    identifier: string,
    operation?: Operation | (string & {}),
    args: readonly string[] = []
  ): ProgramBuilder {
    return this.with({
      type: "TransformStatement",
      identifier,
      operation,
      arguments: identifiers(args),
    });
  }

  /** Un nodo de salida: expone el valor de otro nodo como resultado. */
  sink(identifier: string, sourceIdentifier?: string): ProgramBuilder {
    return this.with({ type: "SinkStatement", identifier, sourceIdentifier });
  }

  // ===========================================================================
  // Atajos, uno por operación (§3). La aridad de la firma va en el tipo.
  // ===========================================================================

  /** `sum(bolsa, …) → bolsa` — variádica (§3.1.1). */
  sum(identifier: string, ...bags: Refs): ProgramBuilder {
    return this.transform(identifier, "sum", bags);
  }

  /** `substract(bolsa, bolsa) → bolsa` (§3.1.2). */
  substract(identifier: string, minuend: string, subtrahend: string): ProgramBuilder {
    return this.transform(identifier, "substract", [minuend, subtrahend]);
  }

  /** `multiply(bolsa, número) → bolsa` (§3.1.3). */
  multiply(identifier: string, bag: string, scalar: string): ProgramBuilder {
    return this.transform(identifier, "multiply", [bag, scalar]);
  }

  /** `divide(bolsa, número) → bolsa` (§3.1.4). */
  divide(identifier: string, bag: string, divisor: string): ProgramBuilder {
    return this.transform(identifier, "divide", [bag, divisor]);
  }

  /** `less_than(bolsa, número) → bolsa` (§3.2.1). */
  lessThan(identifier: string, bag: string, threshold: string): ProgramBuilder {
    return this.transform(identifier, "less_than", [bag, threshold]);
  }

  /** `greater_than(bolsa, número) → bolsa` (§3.2.2). */
  greaterThan(identifier: string, bag: string, threshold: string): ProgramBuilder {
    return this.transform(identifier, "greater_than", [bag, threshold]);
  }

  /** `compare(bolsa, bolsa) → booleano` (§3.2.3). */
  compare(identifier: string, left: string, right: string): ProgramBuilder {
    return this.transform(identifier, "compare", [left, right]);
  }

  /** `order(bolsa, criterio, …) → bolsa` (§3.3.1). */
  order(identifier: string, bag: string, ...criteria: Refs): ProgramBuilder {
    return this.transform(identifier, "order", [bag, ...criteria]);
  }

  /** `filter(bolsa, criterio, …) → bolsa` (§3.4.1). */
  filter(identifier: string, bag: string, ...criteria: Refs): ProgramBuilder {
    return this.transform(identifier, "filter", [bag, ...criteria]);
  }

  /** `first(bolsa) → bolsa` (§3.5.1). */
  first(identifier: string, bag: string): ProgramBuilder {
    return this.transform(identifier, "first", [bag]);
  }

  /** `last(bolsa) → bolsa` (§3.5.2). */
  last(identifier: string, bag: string): ProgramBuilder {
    return this.transform(identifier, "last", [bag]);
  }

  /** `count(bolsa) → número` (§3.6.1). */
  count(identifier: string, bag: string): ProgramBuilder {
    return this.transform(identifier, "count", [bag]);
  }

  // ===========================================================================

  /**
   * El programa construido. No valida nada: los errores los da `execute`,
   * cuando se decide ejecutar (§4).
   */
  build(): Program {
    return { type: "Program", statements: [...this.statements] };
  }
}

/** El único factory: un programa vacío al que se le van añadiendo sentencias. */
export function createProgram(): ProgramBuilder {
  return ProgramBuilder.empty();
}
