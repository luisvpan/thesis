// Errores — LANGUAGE_SPEC.md §4
//
// Todo error informa su naturaleza, el nodo donde ocurrió, el nodo que lo causó
// (si es otro) y, si surgió al evaluar, la salida en cuyo cálculo apareció.

/** El momento en que se detecta el error (§4). */
export type ErrorPhase = "syntax" | "static" | "runtime";

/** Errores estáticos (§4.2): invalidan el programa completo. */
export type StaticErrorCode =
  | "DUPLICATE_IDENTIFIER"
  | "UNDEFINED_REFERENCE"
  | "CIRCULAR_DEPENDENCY"
  | "UNKNOWN_OPERATION"
  | "ARITY_ERROR"
  | "TYPE_ERROR"
  | "INVALID_CRITERION"
  | "INVALID_OBJECT";

/** Errores de ejecución (§4.3): dependen del valor, aislados por salida. */
export type RuntimeErrorCode = "EXPECTED_NUMBER" | "DIVISION_BY_ZERO";

export type ErrorCode = StaticErrorCode | RuntimeErrorCode;

const STATIC_CODES = new Set<ErrorCode>([
  "DUPLICATE_IDENTIFIER",
  "UNDEFINED_REFERENCE",
  "CIRCULAR_DEPENDENCY",
  "UNKNOWN_OPERATION",
  "ARITY_ERROR",
  "TYPE_ERROR",
  "INVALID_CRITERION",
  "INVALID_OBJECT",
]);

/** Dónde queda situado el error (§4). */
export interface ErrorSite {
  /** El nodo que se estaba procesando. */
  nodeId?: string;
  /** El nodo que causó la falla; coincide con `nodeId` si la falla es local. */
  causeNodeId?: string;
  /** La salida en cuyo cálculo bajo demanda apareció. */
  sinkId?: string;
  /**
   * Posición del argumento culpable, cuando una operación falla por el valor de
   * uno de ellos. El evaluador la traduce a `causeNodeId`, que es quien conoce
   * los nombres de los nodos.
   */
  argumentIndex?: number;
}

export class RuntimeError extends Error {
  readonly code: ErrorCode;
  readonly phase: ErrorPhase;
  /** El mensaje sin el prefijo de situación. */
  readonly detail: string;
  nodeId?: string;
  causeNodeId?: string;
  sinkId?: string;
  readonly argumentIndex?: number;

  constructor(code: ErrorCode, detail: string, site: ErrorSite = {}) {
    super(detail);
    this.name = "RuntimeError";
    this.code = code;
    this.detail = detail;
    this.phase = STATIC_CODES.has(code) ? "static" : "runtime";
    this.nodeId = site.nodeId;
    this.causeNodeId = site.causeNodeId ?? site.nodeId;
    this.sinkId = site.sinkId;
    this.argumentIndex = site.argumentIndex;
    this.message = describe(this);
  }

  /**
   * Completa la situación del error con lo que sepa quien lo propaga, sin pisar
   * lo que ya estuviera fijado (el nodo donde ocurrió manda sobre el de quien
   * lo recibe).
   */
  situate(site: ErrorSite): this {
    this.nodeId ??= site.nodeId;
    this.causeNodeId ??= site.causeNodeId ?? this.nodeId;
    this.sinkId ??= site.sinkId;
    this.message = describe(this);
    return this;
  }
}

function describe(error: RuntimeError): string {
  const parts: string[] = [];

  if (error.nodeId) parts.push(`en '${error.nodeId}'`);
  if (error.causeNodeId && error.causeNodeId !== error.nodeId) {
    parts.push(`por '${error.causeNodeId}'`);
  }
  if (error.sinkId) parts.push(`salida '${error.sinkId}'`);

  const where = parts.length > 0 ? ` ${parts.join(", ")}` : "";
  return `[${error.code}]${where}: ${error.detail}`;
}
